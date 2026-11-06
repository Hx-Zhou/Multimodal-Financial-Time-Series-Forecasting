"""Train E3c with a daily pooled news branch and evaluate holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.gru_e2.model import build_optimizer, compute_loss
from experiments.gru_e2.train import set_seed
from experiments.gru_e3.train import numeric_batch as market_batch
from experiments.gru_e7.model import build_model
from experiments.validation_split import make_validation_split
from hs300_dataset import HS300Dataset


class DailyNewsIndex:
    """Compact per-stock daily means and counts for fast batch gathering."""

    def __init__(self, dataset: HS300Dataset):
        self.by_stock: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for code in dataset.stock_codes:
            times, ids = dataset.news_by_stock.get(
                code, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))
            )
            days = (times // 1_000_000).astype(np.int32)
            if len(days) == 0:
                self.by_stock[code] = (
                    np.empty(0, dtype=np.int32),
                    np.empty((0, 512), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                )
                continue
            unique_days, inverse = np.unique(days, return_inverse=True)
            sums = np.zeros((len(unique_days), dataset.news_vectors.shape[1]), dtype=np.float32)
            np.add.at(sums, inverse, dataset.news_vectors[ids])
            counts = np.bincount(inverse).astype(np.float32)
            means = sums / np.maximum(counts[:, None], 1.0)
            self.by_stock[code] = (unique_days, means, counts)


def add_news_batch(dataset, indices, batch, news_index: DailyNewsIndex) -> dict:
    indices = np.asarray(indices, dtype=np.int64)
    vectors = np.zeros((len(indices), 96, 512), dtype=np.float32)
    counts = np.zeros((len(indices), 96), dtype=np.float32)
    offsets = np.arange(96, dtype=np.int64)
    stock_indices = dataset.stock_index[indices]
    for stock in np.unique(stock_indices):
        mask = stock_indices == stock
        code = dataset.stock_codes[int(stock)]
        days = dataset.price_days[code][
            dataset.input_start[indices[mask]].astype(np.int64)[:, None] + offsets
        ]
        available_days, means, available_counts = news_index.by_stock[code]
        if len(available_days) == 0:
            continue
        positions = np.searchsorted(available_days, days)
        valid = (positions < len(available_days)) & (
            available_days[np.minimum(positions, len(available_days) - 1)] == days
        )
        row_ids, day_ids = np.nonzero(valid)
        if len(row_ids):
            selected_positions = positions[row_ids, day_ids]
            vectors[np.flatnonzero(mask)[row_ids], day_ids] = means[selected_positions]
            counts[np.flatnonzero(mask)[row_ids], day_ids] = available_counts[selected_positions]
    batch = dict(batch)
    batch["news"] = torch.as_tensor(vectors, dtype=torch.float32, device=batch["numerical"].device)
    batch["news_count"] = torch.as_tensor(counts, dtype=torch.float32, device=batch["numerical"].device)
    batch["has_news"] = (batch["news_count"] > 0).to(dtype=torch.float32)
    return batch


def make_batch(dataset, indices, model, news_index, device):
    market = market_batch(dataset, indices, device, 4, "relative", "standardized_close")
    return add_news_batch(dataset, indices, market, news_index)


def evaluate(dataset, indices, model, news_index, batch_size, device):
    model.eval()
    absolute = squared = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = make_batch(
                dataset, indices[start : start + batch_size], model, news_index, device
            )
            error = model(batch) - batch["y"]
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
    if count == 0:
        raise ValueError("evaluation set is empty")
    return {"mae": absolute / count, "mse": squared / count}


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    train = None if args.evaluate_holdout else HS300Dataset(args.package_dir, "train")
    val = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(val, args.holdout_fraction)
        val_news = DailyNewsIndex(val)
        if args.evaluate_holdout:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            if checkpoint.get("holdout_fraction") != args.holdout_fraction:
                raise ValueError("holdout fraction does not match the checkpoint")
            if int(checkpoint.get("epoch", 0)) < 1:
                raise ValueError("holdout evaluation requires a trained checkpoint")
            model = build_model(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            result = {
                "experiment": checkpoint.get("experiment", "E3c_E7_news_GRU"),
                "checkpoint_epoch": checkpoint["epoch"],
                "config": checkpoint["config"],
                "holdout": evaluate(val, split.holdout_indices, model, val_news, args.batch_size, device),
            }
        else:
            train_news = DailyNewsIndex(train)
            config = {
                "hidden_size": args.hidden_size,
                "horizon": 24,
                "news_input_size": 512,
                "news_dim": args.news_dim,
                "dropout": args.dropout,
                "residual_scale": 1.0,
            }
            model = build_model(config).to(device)
            optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
            indices = np.arange(len(train), dtype=np.int64)
            initial = evaluate(val, split.dev_indices, model, val_news, args.batch_size, device)
            history = [{"epoch": 0, **initial}]
            best_mae, best_epoch, wait = float("inf"), None, 0
            progress = args.output.with_suffix(".progress.json")
            progress.write_text(json.dumps({"experiment": "E3c_E7_news_GRU", "history": history}, indent=2) + "\n")
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(f"epoch 0 (E0 reference): dev_MAE={initial['mae']:.6f} dev_MSE={initial['mse']:.6f}", flush=True)
            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(indices)
                model.train()
                weighted_loss = 0.0
                for step, start in enumerate(range(0, len(indices), args.batch_size), 1):
                    chosen = indices[start : start + args.batch_size]
                    batch = make_batch(train, chosen, model, train_news, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = compute_loss(model(batch), batch["y"])
                    if not torch.isfinite(loss):
                        raise ValueError(f"non-finite loss at epoch {epoch}, batch {step}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    weighted_loss += loss.item() * len(chosen)
                    total_batches = (len(indices) + args.batch_size - 1) // args.batch_size
                    if step == 1 or step % 100 == 0 or step == total_batches:
                        print(f"epoch {epoch} batch {step}/{total_batches} loss={loss.item():.6f}", flush=True)
                dev = evaluate(val, split.dev_indices, model, val_news, args.batch_size, device)
                record = {"epoch": epoch, "train_loss": weighted_loss / len(indices), **dev}
                history.append(record)
                progress.write_text(json.dumps({"experiment": "E3c_E7_news_GRU", "history": history}, indent=2) + "\n")
                print(f"epoch {epoch}: dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f}", flush=True)
                if dev["mae"] < best_mae:
                    best_mae, best_epoch, wait = dev["mae"], epoch, 0
                    torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch, "experiment": "E3c_E7_news_GRU", "holdout_fraction": args.holdout_fraction}, args.checkpoint)
                else:
                    wait += 1
                    if wait >= args.patience:
                        break
            if best_epoch is None:
                raise RuntimeError("no trained epoch completed")
            selected = torch.load(args.checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(selected["model_state"])
            holdout = evaluate(val, split.holdout_indices, model, val_news, args.batch_size, device)
            progress.unlink(missing_ok=True)
            best_record = next(item for item in history if item["epoch"] == best_epoch)
            result = {
                "experiment": "E3c_E7_news_GRU",
                "seed": args.seed,
                "config": config,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "holdout_fraction": args.holdout_fraction,
                "e0_development": initial,
                "best_epoch": best_epoch,
                "best_development": best_record,
                "holdout": {"checkpoint_epoch": best_epoch, **holdout},
                "mae_delta_vs_e0": best_record["mae"] - initial["mae"],
                "history": history[1:],
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        if train is not None:
            train.close()
        val.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/gru_e7/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gru_e7/result.json"))
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--news-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
