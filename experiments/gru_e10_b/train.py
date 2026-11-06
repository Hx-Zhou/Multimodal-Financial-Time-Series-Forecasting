"""Train E10-B with lagged, train-only standardized financial features."""

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

from experiments.gru_e10_a.train import (
    REPORT_FIELDS,
    REPORT_KINDS,
    _date,
    _signed_log,
)
from experiments.gru_e2.model import build_optimizer, compute_loss
from experiments.gru_e2.train import set_seed
from experiments.gru_e3.train import numeric_batch as market_batch
from experiments.gru_e10_b.model import build_model
from experiments.validation_split import make_validation_split
from hs300_dataset import HS300Dataset


class FinancialIndex:
    """Lagged report snapshots with train-only normalization for 12 values."""

    def __init__(
        self,
        dataset: HS300Dataset,
        report_lag_days: int = 30,
        feature_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        if report_lag_days < 0:
            raise ValueError("report_lag_days must be non-negative")
        self.report_lag_days = int(report_lag_days)
        self.by_stock = {}
        for code in dataset.stock_codes:
            records = dataset._reports(code)
            dates = sorted({
                int(row["end_date"])
                for kind in REPORT_KINDS
                for row in records.get(kind, [])
            })
            snapshots = []
            for date in dates:
                values = []
                for kind, field in REPORT_FIELDS:
                    rows = [
                        row for row in records.get(kind, [])
                        if int(row["end_date"]) == date
                    ]
                    values.append(_signed_log(rows[-1].get(field)) if rows else 0.0)
                present = [
                    float(any(int(row["end_date"]) == date for row in records.get(kind, [])))
                    for kind in REPORT_KINDS
                ]
                snapshots.append(values + present)
            values = np.asarray(snapshots, dtype=np.float32).reshape(-1, 15)
            if feature_stats is not None and len(values):
                mean, std = feature_stats
                values[:, :12] = (values[:, :12] - mean) / std
            self.by_stock[code] = (
                np.asarray(dates, dtype=np.int32),
                np.asarray([_date(date).toordinal() for date in dates], dtype=np.int64),
                values,
            )


def fit_feature_stats(dataset: HS300Dataset, report_lag_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit value statistics using reports available by training inputs only."""
    raw = FinancialIndex(dataset, report_lag_days)
    pieces = []
    for stock_index, code in enumerate(dataset.stock_codes):
        report_dates, report_ordinals, report_values = raw.by_stock[code]
        if not len(report_values):
            continue
        sample_mask = dataset.stock_index == stock_index
        starts = dataset.input_start[sample_mask].astype(np.int64)
        if not len(starts):
            continue
        input_days = dataset.price_days[code][starts + 95]
        max_available = max(_date(day).toordinal() for day in input_days) - report_lag_days
        usable = report_ordinals <= max_available
        if np.any(usable):
            pieces.append(report_values[usable, :12])
    if not pieces:
        raise ValueError("no training financial reports available for normalization")
    values = np.concatenate(pieces, axis=0).astype(np.float64)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def add_financial_batch(dataset, indices, batch, financial_index: FinancialIndex):
    indices = np.asarray(indices, dtype=np.int64)
    values = np.zeros((len(indices), 96, 16), dtype=np.float32)
    stock_indices = dataset.stock_index[indices]
    for stock in np.unique(stock_indices):
        rows = np.flatnonzero(stock_indices == stock)
        code = dataset.stock_codes[int(stock)]
        report_dates, report_ordinals, report_values = financial_index.by_stock[code]
        if len(report_dates) == 0:
            continue
        starts = dataset.input_start[indices[rows]].astype(np.int64)
        input_days = dataset.price_days[code][starts + 95]
        input_ordinals = np.asarray([_date(day).toordinal() for day in input_days], dtype=np.int64)
        positions = np.searchsorted(
            report_ordinals,
            input_ordinals - financial_index.report_lag_days,
            side="right",
        ) - 1
        valid = positions >= 0
        for row, position, input_ordinal in zip(
            rows[valid], positions[valid], input_ordinals[valid]
        ):
            age = max(int(input_ordinal) - int(report_ordinals[position]), 0)
            values[row, :, :15] = report_values[position]
            values[row, :, 15] = np.log1p(float(age))
    result = dict(batch)
    result["financial"] = torch.as_tensor(
        values, dtype=torch.float32, device=batch["numerical"].device
    )
    return result


def make_batch(dataset, indices, financial_index, device):
    market = market_batch(dataset, indices, device, 4, "relative", "standardized_close")
    return add_financial_batch(dataset, indices, market, financial_index)


def evaluate(dataset, indices, model, financial_index, batch_size, device):
    model.eval()
    absolute = squared = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = make_batch(
                dataset, indices[start : start + batch_size], financial_index, device
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
    if args.report_lag_days < 0:
        raise ValueError("report-lag-days must be non-negative")
    set_seed(args.seed)
    device = torch.device(args.device)
    train = None if args.evaluate_holdout else HS300Dataset(args.package_dir, "train")
    val = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(val, args.holdout_fraction)
        if args.evaluate_holdout:
            checkpoint = load_checkpoint(args.checkpoint, device)
            if checkpoint.get("holdout_fraction") != args.holdout_fraction:
                raise ValueError("holdout fraction does not match checkpoint")
            if int(checkpoint.get("epoch", 0)) < 1:
                raise ValueError("holdout requires an E10-B trained checkpoint")
            if int(checkpoint.get("report_lag_days", -1)) != args.report_lag_days:
                raise ValueError("report lag does not match checkpoint")
            stats = checkpoint.get("feature_stats")
            if not stats:
                raise ValueError("checkpoint has no financial feature statistics")
            feature_stats = (
                np.asarray(stats["mean"], dtype=np.float32),
                np.asarray(stats["std"], dtype=np.float32),
            )
            val_financial = FinancialIndex(val, args.report_lag_days, feature_stats)
            model = build_model(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            result = {
                "experiment": "E10_B_frozen_E3c_normalized_financial",
                "checkpoint_epoch": checkpoint["epoch"],
                "report_lag_days": args.report_lag_days,
                "config": checkpoint["config"],
                "holdout": evaluate(
                    val, split.holdout_indices, model, val_financial,
                    args.batch_size, device,
                ),
            }
        else:
            e3c = load_checkpoint(args.market_checkpoint, device)
            feature_mean, feature_std = fit_feature_stats(train, args.report_lag_days)
            train_financial = FinancialIndex(
                train, args.report_lag_days, (feature_mean, feature_std)
            )
            val_financial = FinancialIndex(
                val, args.report_lag_days, (feature_mean, feature_std)
            )
            config = {
                "hidden_size": int(e3c["config"].get("hidden_size", 64)),
                "horizon": int(e3c["config"].get("horizon", 24)),
                "financial_dim": args.financial_dim,
                "financial_hidden": args.financial_hidden,
                "dropout": args.dropout,
            }
            model = build_model(config).to(device)
            model.load_e3c_state(e3c["model_state"])
            optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
            indices = np.arange(len(train), dtype=np.int64)
            initial = evaluate(
                val, split.dev_indices, model, val_financial, args.batch_size, device
            )
            history = [{"epoch": 0, **initial}]
            best_mae = float("inf")
            best_epoch = None
            wait = 0
            progress = args.output.with_suffix(".progress.json")
            progress.write_text(
                json.dumps({
                    "experiment": "E10_B_frozen_E3c_normalized_financial",
                    "report_lag_days": args.report_lag_days,
                    "history": history,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"epoch 0 (E3c reference): dev_MAE={initial['mae']:.6f} "
                f"dev_MSE={initial['mse']:.6f} report_lag_days={args.report_lag_days}",
                flush=True,
            )
            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(indices)
                model.train()
                weighted = 0.0
                total = (len(indices) + args.batch_size - 1) // args.batch_size
                for step, start in enumerate(range(0, len(indices), args.batch_size), 1):
                    batch = make_batch(
                        train, indices[start : start + args.batch_size],
                        train_financial, device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = compute_loss(model(batch), batch["y"])
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    weighted += loss.item() * len(batch["y"])
                    if step == 1 or step % 100 == 0 or step == total:
                        print(
                            f"epoch {epoch} batch {step}/{total} loss={loss.item():.6f}",
                            flush=True,
                        )
                dev = evaluate(
                    val, split.dev_indices, model, val_financial,
                    args.batch_size, device,
                )
                record = {"epoch": epoch, "train_loss": weighted / len(indices), **dev}
                history.append(record)
                progress.write_text(
                    json.dumps({
                        "experiment": "E10_B_frozen_E3c_normalized_financial",
                        "report_lag_days": args.report_lag_days,
                        "history": history,
                    }, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"epoch {epoch}: dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f}",
                    flush=True,
                )
                if dev["mae"] < best_mae:
                    best_mae, best_epoch, wait = dev["mae"], epoch, 0
                    torch.save({
                        "model_state": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "experiment": "E10_B_frozen_E3c_normalized_financial",
                        "report_lag_days": args.report_lag_days,
                        "holdout_fraction": args.holdout_fraction,
                        "market_checkpoint": str(args.market_checkpoint),
                        "feature_stats": {
                            "mean": feature_mean.tolist(),
                            "std": feature_std.tolist(),
                        },
                    }, args.checkpoint)
                else:
                    wait += 1
                    if wait >= args.patience:
                        break
            if best_epoch is None:
                raise RuntimeError("no trained epoch completed")
            selected = load_checkpoint(args.checkpoint, device)
            model.load_state_dict(selected["model_state"])
            best_record = next(item for item in history if item["epoch"] == best_epoch)
            holdout = evaluate(
                val, split.holdout_indices, model, val_financial,
                args.batch_size, device,
            )
            progress.unlink(missing_ok=True)
            result = {
                "experiment": "E10_B_frozen_E3c_normalized_financial",
                "seed": args.seed,
                "report_lag_days": args.report_lag_days,
                "config": config,
                "market_checkpoint": str(args.market_checkpoint),
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "holdout_fraction": args.holdout_fraction,
                "feature_normalization": "train-only mean/std for 12 signed-log value fields; presence and age unchanged",
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                "e3c_development": initial,
                "best_epoch": best_epoch,
                "best_development": best_record,
                "holdout": {"checkpoint_epoch": best_epoch, **holdout},
                "mae_delta_vs_e3c": best_record["mae"] - initial["mae"],
                "history": history[1:],
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        if train is not None:
            train.close()
        val.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--market-checkpoint", type=Path, default=Path("experiments/gru_e3/checkpoints/best.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/gru_e10_b/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gru_e10_b/result.json"))
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--financial-dim", type=int, default=16)
    parser.add_argument("--financial-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--report-lag-days", type=int, default=30)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
