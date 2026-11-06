"""Train the E9-1 frozen-E3c news adapter and evaluate holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gru_e2.model import build_optimizer, compute_loss
from experiments.gru_e2.train import set_seed
from experiments.gru_e3.train import numeric_batch as market_batch
from experiments.gru_e9_1.model import build_model
from experiments.validation_split import make_validation_split
from hs300_dataset import HS300Dataset


class DailyNewsIndex:
    """Daily mean and max pooling, retaining both common and salient signals."""

    def __init__(self, dataset: HS300Dataset):
        self.by_stock = {}
        for code in dataset.stock_codes:
            times, ids = dataset.news_by_stock.get(
                code, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))
            )
            days = (times // 1_000_000).astype(np.int32)
            if len(days) == 0:
                self.by_stock[code] = (
                    np.empty(0, dtype=np.int32),
                    np.empty((0, 1024), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                )
                continue
            unique_days, inverse = np.unique(days, return_inverse=True)
            vectors = dataset.news_vectors[ids]
            sums = np.zeros((len(unique_days), 512), dtype=np.float32)
            np.add.at(sums, inverse, vectors)
            counts = np.bincount(inverse).astype(np.float32)
            means = sums / np.maximum(counts[:, None], 1.0)
            maxima = np.full((len(unique_days), 512), -np.inf, dtype=np.float32)
            np.maximum.at(maxima, inverse, vectors)
            self.by_stock[code] = (
                unique_days,
                np.concatenate((means, maxima), axis=1),
                counts,
            )


def add_news_batch(dataset, indices, batch, news_index: DailyNewsIndex):
    indices = np.asarray(indices, dtype=np.int64)
    vectors = np.zeros((len(indices), 96, 1024), dtype=np.float32)
    counts = np.zeros((len(indices), 96), dtype=np.float32)
    offsets = np.arange(96, dtype=np.int64)
    stock_indices = dataset.stock_index[indices]
    for stock in np.unique(stock_indices):
        row_ids = np.flatnonzero(stock_indices == stock)
        code = dataset.stock_codes[int(stock)]
        starts = dataset.input_start[indices[row_ids]].astype(np.int64)
        days = dataset.price_days[code][starts[:, None] + offsets]
        available_days, pooled, available_counts = news_index.by_stock[code]
        if len(available_days) == 0:
            continue
        positions = np.searchsorted(available_days, days)
        safe = np.minimum(positions, len(available_days) - 1)
        valid = (positions < len(available_days)) & (available_days[safe] == days)
        local_rows, day_ids = np.nonzero(valid)
        if len(local_rows):
            selected = positions[local_rows, day_ids]
            vectors[row_ids[local_rows], day_ids] = pooled[selected]
            counts[row_ids[local_rows], day_ids] = available_counts[selected]
    result = dict(batch)
    result["news"] = torch.as_tensor(vectors, dtype=torch.float32, device=batch["numerical"].device)
    result["news_count"] = torch.as_tensor(counts, dtype=torch.float32, device=batch["numerical"].device)
    result["has_news"] = (result["news_count"] > 0).to(dtype=torch.float32)
    return result


def make_batch(dataset, indices, news_index, device):
    market = market_batch(dataset, indices, device, 4, "relative", "standardized_close")
    return add_news_batch(dataset, indices, market, news_index)


def evaluate(dataset, indices, model, news_index, batch_size, device):
    model.eval(); absolute = squared = 0.0; count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = make_batch(dataset, indices[start:start + batch_size], news_index, device)
            error = model(batch) - batch["y"]
            absolute += error.abs().sum().item(); squared += error.square().sum().item(); count += error.numel()
    if not count:
        raise ValueError("evaluation set is empty")
    return {"mae": absolute / count, "mse": squared / count}


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    set_seed(args.seed); device = torch.device(args.device)
    train = None if args.evaluate_holdout else HS300Dataset(args.package_dir, "train")
    val = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(val, args.holdout_fraction)
        val_news = DailyNewsIndex(val)
        if args.evaluate_holdout:
            checkpoint = load_checkpoint(args.checkpoint, device)
            if checkpoint.get("holdout_fraction") != args.holdout_fraction:
                raise ValueError("holdout fraction does not match checkpoint")
            if int(checkpoint.get("epoch", 0)) < 1:
                raise ValueError("holdout requires an E9-1 trained checkpoint")
            model = build_model(checkpoint["config"]).to(device); model.load_state_dict(checkpoint["model_state"])
            result = {"experiment": "E9_1_frozen_E3c_news", "checkpoint_epoch": checkpoint["epoch"], "config": checkpoint["config"], "holdout": evaluate(val, split.holdout_indices, model, val_news, args.batch_size, device)}
        else:
            e3c = load_checkpoint(args.market_checkpoint, device)
            config = {"hidden_size": int(e3c["config"].get("hidden_size", 64)), "horizon": 24, "news_dim": args.news_dim, "news_hidden": args.news_hidden, "dropout": args.dropout}
            model = build_model(config).to(device); model.load_e3c_state(e3c["model_state"])
            optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
            train_news = DailyNewsIndex(train); indices = np.arange(len(train), dtype=np.int64)
            initial = evaluate(val, split.dev_indices, model, val_news, args.batch_size, device)
            history = [{"epoch": 0, **initial}]; best_mae = float("inf"); best_epoch = None; wait = 0
            progress = args.output.with_suffix(".progress.json"); progress.write_text(json.dumps({"experiment": "E9_1_frozen_E3c_news", "history": history}, indent=2) + "\n")
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(f"epoch 0 (E3c reference): dev_MAE={initial['mae']:.6f} dev_MSE={initial['mse']:.6f}", flush=True)
            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(indices); model.train(); weighted = 0.0; total = (len(indices) + args.batch_size - 1) // args.batch_size
                for step, start in enumerate(range(0, len(indices), args.batch_size), 1):
                    batch = make_batch(train, indices[start:start + args.batch_size], train_news, device)
                    optimizer.zero_grad(set_to_none=True); loss = compute_loss(model(batch), batch["y"])
                    if not torch.isfinite(loss): raise ValueError("non-finite loss")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); weighted += loss.item() * len(batch["y"])
                    if step == 1 or step % 100 == 0 or step == total: print(f"epoch {epoch} batch {step}/{total} loss={loss.item():.6f}", flush=True)
                dev = evaluate(val, split.dev_indices, model, val_news, args.batch_size, device); record = {"epoch": epoch, "train_loss": weighted / len(indices), **dev}; history.append(record)
                progress.write_text(json.dumps({"experiment": "E9_1_frozen_E3c_news", "history": history}, indent=2) + "\n"); print(f"epoch {epoch}: dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f}", flush=True)
                if dev["mae"] < best_mae:
                    best_mae, best_epoch, wait = dev["mae"], epoch, 0; torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch, "experiment": "E9_1_frozen_E3c_news", "holdout_fraction": args.holdout_fraction}, args.checkpoint)
                else:
                    wait += 1
                    if wait >= args.patience: break
            if best_epoch is None: raise RuntimeError("no trained epoch completed")
            selected = load_checkpoint(args.checkpoint, device); model.load_state_dict(selected["model_state"]); best_record = next(item for item in history if item["epoch"] == best_epoch)
            holdout = evaluate(val, split.holdout_indices, model, val_news, args.batch_size, device); progress.unlink(missing_ok=True)
            result = {"experiment": "E9_1_frozen_E3c_news", "seed": args.seed, "config": config, "market_checkpoint": str(args.market_checkpoint), "learning_rate": args.learning_rate, "batch_size": args.batch_size, "holdout_fraction": args.holdout_fraction, "e3c_development": initial, "best_epoch": best_epoch, "best_development": best_record, "holdout": {"checkpoint_epoch": best_epoch, **holdout}, "mae_delta_vs_e3c": best_record["mae"] - initial["mae"], "history": history[1:]}
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2), flush=True); return result
    finally:
        if train is not None: train.close()
        val.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--package-dir", type=Path, default=ROOT); parser.add_argument("--market-checkpoint", type=Path, default=Path("experiments/gru_e3/checkpoints/best.pt")); parser.add_argument("--checkpoint", type=Path, default=Path("experiments/gru_e9_1/checkpoints/best.pt")); parser.add_argument("--output", type=Path, default=Path("experiments/gru_e9_1/result.json")); parser.add_argument("--evaluate-holdout", action="store_true"); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--patience", type=int, default=6); parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--news-dim", type=int, default=16); parser.add_argument("--news-hidden", type=int, default=32); parser.add_argument("--dropout", type=float, default=0.0); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-5); parser.add_argument("--holdout-fraction", type=float, default=0.2); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cpu"); run(parser.parse_args())


if __name__ == "__main__": main()
