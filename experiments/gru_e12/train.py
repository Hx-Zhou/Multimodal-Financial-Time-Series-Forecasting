"""Train a news adapter incrementally on frozen E10-B."""

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

from experiments.gru_e10_b.train import FinancialIndex, add_financial_batch
from experiments.gru_e2.model import build_optimizer, compute_loss
from experiments.gru_e2.train import set_seed
from experiments.gru_e3.train import numeric_batch as market_batch
from experiments.gru_e9_1.train import DailyNewsIndex, add_news_batch
from experiments.gru_e12.model import build_model
from experiments.validation_split import make_validation_split
from hs300_dataset import HS300Dataset


def make_batch(dataset, indices, news_index, financial_index, device):
    market = market_batch(dataset, indices, device, 4, "relative", "standardized_close")
    with_news = add_news_batch(dataset, indices, market, news_index)
    return add_financial_batch(dataset, indices, with_news, financial_index)


def evaluate(dataset, indices, model, news_index, financial_index, batch_size, device):
    model.eval()
    absolute = squared = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = make_batch(
                dataset, indices[start : start + batch_size],
                news_index, financial_index, device,
            )
            error = model(batch) - batch["y"]
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
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
    set_seed(args.seed)
    device = torch.device(args.device)
    train = None if args.evaluate_holdout else HS300Dataset(args.package_dir, "train")
    val = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(val, args.holdout_fraction)
        e10b = load_checkpoint(args.base_checkpoint, device)
        news_checkpoint = load_checkpoint(args.news_checkpoint, device)
        if int(e10b.get("report_lag_days", -1)) != args.report_lag_days:
            raise ValueError("report lag does not match E10-B checkpoint")
        stats = e10b.get("feature_stats")
        if not stats:
            raise ValueError("E10-B checkpoint has no financial feature statistics")
        feature_stats = (
            np.asarray(stats["mean"], dtype=np.float32),
            np.asarray(stats["std"], dtype=np.float32),
        )
        val_news = DailyNewsIndex(val)
        val_financial = FinancialIndex(val, args.report_lag_days, feature_stats)

        if args.evaluate_holdout:
            checkpoint = load_checkpoint(args.checkpoint, device)
            if checkpoint.get("holdout_fraction") != args.holdout_fraction:
                raise ValueError("holdout fraction does not match checkpoint")
            if int(checkpoint.get("epoch", 0)) < 1:
                raise ValueError("holdout requires a trained E12 checkpoint")
            model = build_model(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            result = {
                "experiment": "E12_E10B_incremental_news",
                "checkpoint_epoch": checkpoint["epoch"],
                "config": checkpoint["config"],
                "holdout": evaluate(
                    val, split.holdout_indices, model, val_news,
                    val_financial, args.batch_size, device,
                ),
            }
        else:
            config = {
                "hidden_size": int(e10b["config"].get("hidden_size", 64)),
                "horizon": int(e10b["config"].get("horizon", 24)),
                "financial_dim": int(e10b["config"].get("financial_dim", 16)),
                "financial_hidden": int(e10b["config"].get("financial_hidden", 32)),
                "news_dim": args.news_dim,
                "news_hidden": args.news_hidden,
                "dropout": args.dropout,
            }
            model = build_model(config).to(device)
            model.load_e10b_state(e10b["model_state"])
            if args.initialize_news:
                model.load_news_state(news_checkpoint["model_state"])
            model.zero_news_gate()
            optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
            train_news = DailyNewsIndex(train)
            train_financial = FinancialIndex(train, args.report_lag_days, feature_stats)
            indices = np.arange(len(train), dtype=np.int64)
            base = evaluate(val, split.dev_indices, model, val_news, val_financial, args.batch_size, device)
            history = [{"epoch": 0, **base}]
            best_mae, best_epoch, wait = float("inf"), None, 0
            progress = args.output.with_suffix(".progress.json")
            progress.write_text(json.dumps({"experiment": "E12_E10B_incremental_news", "history": history}, indent=2) + "\n", encoding="utf-8")
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(f"epoch 0 (E10-B reference): dev_MAE={base['mae']:.6f} dev_MSE={base['mse']:.6f}", flush=True)
            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(indices)
                model.train(); weighted = 0.0
                total = (len(indices) + args.batch_size - 1) // args.batch_size
                for step, start in enumerate(range(0, len(indices), args.batch_size), 1):
                    batch = make_batch(train, indices[start : start + args.batch_size], train_news, train_financial, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = compute_loss(model(batch), batch["y"])
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite loss")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                    weighted += loss.item() * len(batch["y"])
                    if step == 1 or step % 100 == 0 or step == total:
                        print(f"epoch {epoch} batch {step}/{total} loss={loss.item():.6f}", flush=True)
                dev = evaluate(val, split.dev_indices, model, val_news, val_financial, args.batch_size, device)
                record = {"epoch": epoch, "train_loss": weighted / len(indices), **dev}; history.append(record)
                progress.write_text(json.dumps({"experiment": "E12_E10B_incremental_news", "history": history}, indent=2) + "\n", encoding="utf-8")
                print(f"epoch {epoch}: dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f}", flush=True)
                if dev["mae"] < best_mae:
                    best_mae, best_epoch, wait = dev["mae"], epoch, 0
                    torch.save({
                        "model_state": model.state_dict(), "config": config, "epoch": epoch,
                        "experiment": "E12_E10B_incremental_news", "holdout_fraction": args.holdout_fraction,
                        "report_lag_days": args.report_lag_days, "base_checkpoint": str(args.base_checkpoint),
                        "news_checkpoint": str(args.news_checkpoint), "feature_stats": {"mean": feature_stats[0].tolist(), "std": feature_stats[1].tolist()},
                    }, args.checkpoint)
                else:
                    wait += 1
                    if wait >= args.patience:
                        break
            if best_epoch is None:
                raise RuntimeError("no trained epoch completed")
            selected = load_checkpoint(args.checkpoint, device); model.load_state_dict(selected["model_state"])
            best_record = next(item for item in history if item["epoch"] == best_epoch)
            holdout = evaluate(val, split.holdout_indices, model, val_news, val_financial, args.batch_size, device)
            progress.unlink(missing_ok=True)
            result = {
                "experiment": "E12_E10B_incremental_news", "seed": args.seed,
                "config": config, "base_checkpoint": str(args.base_checkpoint),
                "news_checkpoint": str(args.news_checkpoint), "initialized_news": bool(args.initialize_news),
                "report_lag_days": args.report_lag_days, "learning_rate": args.learning_rate,
                "batch_size": args.batch_size, "holdout_fraction": args.holdout_fraction,
                "e10b_development": base, "best_epoch": best_epoch,
                "best_development": best_record, "holdout": {"checkpoint_epoch": best_epoch, **holdout},
                "mae_delta_vs_e10b": best_record["mae"] - base["mae"], "history": history[1:],
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
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("experiments/gru_e10_b/checkpoints/best.pt"))
    parser.add_argument("--news-checkpoint", type=Path, default=Path("experiments/gru_e9_1/checkpoints/best.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/gru_e12/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gru_e12/result.json"))
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--initialize-news", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--news-dim", type=int, default=16)
    parser.add_argument("--news-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--report-lag-days", type=int, default=30)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__": main()
