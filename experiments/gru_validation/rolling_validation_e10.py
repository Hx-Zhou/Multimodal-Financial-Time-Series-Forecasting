"""Rolling development evaluation for E3c, E10-B and E10-C checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset
from experiments.gru_e3.model import build_model as build_market_model
from experiments.gru_e3.train import numeric_batch
from experiments.gru_e10_b.model import build_model as build_e10b_model
from experiments.gru_e10_b.train import FinancialIndex as E10FinancialIndex
from experiments.gru_e10_b.train import add_financial_batch as add_e10b_batch
from experiments.gru_e10_c.model import build_model as build_e10c_model
from experiments.gru_e10_c.train import add_financial_batch as add_e10c_batch
from experiments.validation_split import make_validation_split

ROOT = Path(__file__).resolve().parents[2]


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def make_date_folds(dataset: HS300Dataset, indices: np.ndarray, folds: int) -> list[np.ndarray]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    dates = np.asarray(dataset.target_start_date, dtype=np.int32)
    unique_dates = np.unique(dates[indices])
    if len(unique_dates) < folds:
        raise ValueError(f"only {len(unique_dates)} development dates available for {folds} folds")
    return [
        indices[np.isin(dates[indices], date_group)]
        for date_group in np.array_split(unique_dates, folds)
    ]


def add_metrics(total: dict, error: torch.Tensor) -> None:
    total["absolute"] += error.abs().sum().item()
    total["squared"] += error.square().sum().item()
    total["count"] += error.numel()


def finish_metrics(total: dict) -> dict[str, float]:
    return {
        "mae": total["absolute"] / total["count"],
        "mse": total["squared"] / total["count"],
    }


def evaluate_window(
    dataset: HS300Dataset,
    indices: np.ndarray,
    market_model: torch.nn.Module,
    e10b_model: torch.nn.Module,
    e10c_model: torch.nn.Module,
    financial_index: E10FinancialIndex,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    market_model.eval()
    e10b_model.eval()
    e10c_model.eval()
    totals = {
        name: {"absolute": 0.0, "squared": 0.0, "count": 0}
        for name in ("e0_last_value", "e3c", "e10_b", "e10_c")
    }
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            market = numeric_batch(
                dataset, selected, device, 4, "relative", "standardized_close"
            )
            target = market["y"]
            e0 = market["last_close"][:, None].expand_as(target)
            add_metrics(totals["e0_last_value"], e0 - target)
            e3c = market_model(market)
            add_metrics(totals["e3c"], e3c - target)

            e10b_batch = add_e10b_batch(dataset, selected, market, financial_index)
            add_metrics(totals["e10_b"], e10b_model(e10b_batch) - target)
            e10c_batch = add_e10c_batch(dataset, selected, market, financial_index)
            add_metrics(totals["e10_c"], e10c_model(e10c_batch) - target)
    return {name: finish_metrics(total) for name, total in totals.items()}


def run(args) -> dict:
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    device = torch.device(args.device)
    dataset = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(dataset, args.holdout_fraction)
        folds = make_date_folds(dataset, split.dev_indices, args.folds)
        e3_checkpoint = load_checkpoint(args.market_checkpoint, device)
        e10b_checkpoint = load_checkpoint(args.e10b_checkpoint, device)
        e10c_checkpoint = load_checkpoint(args.e10c_checkpoint, device)

        e3c_model = build_market_model(e3_checkpoint["config"]).to(device)
        e3c_model.load_state_dict(e3_checkpoint["model_state"])
        e10b_model = build_e10b_model(e10b_checkpoint["config"]).to(device)
        e10b_model.load_state_dict(e10b_checkpoint["model_state"])
        e10c_model = build_e10c_model(e10c_checkpoint["config"]).to(device)
        e10c_model.load_state_dict(e10c_checkpoint["model_state"])

        if int(e10b_checkpoint.get("report_lag_days", -1)) != int(e10c_checkpoint.get("report_lag_days", -2)):
            raise ValueError("E10-B and E10-C report lags differ")
        report_lag_days = int(e10b_checkpoint["report_lag_days"])
        b_stats = e10b_checkpoint.get("feature_stats")
        c_stats = e10c_checkpoint.get("feature_stats")
        if not b_stats or not c_stats:
            raise ValueError("E10 checkpoint is missing feature statistics")
        b_mean = np.asarray(b_stats["mean"], dtype=np.float32)
        b_std = np.asarray(b_stats["std"], dtype=np.float32)
        c_mean = np.asarray(c_stats["mean"], dtype=np.float32)
        c_std = np.asarray(c_stats["std"], dtype=np.float32)
        if not np.allclose(b_mean, c_mean) or not np.allclose(b_std, c_std):
            raise ValueError("E10-B and E10-C use different feature statistics")
        financial_index = E10FinancialIndex(
            dataset, report_lag_days, (b_mean, b_std)
        )

        windows = []
        for fold_number, indices in enumerate(folds, 1):
            dates = dataset.target_start_date[indices]
            print(
                f"fold {fold_number}/{len(folds)}: {int(dates.min())}-{int(dates.max())} "
                f"samples={len(indices)}",
                flush=True,
            )
            windows.append({
                "fold": fold_number,
                "target_start_min": int(dates.min()),
                "target_start_max": int(dates.max()),
                "samples": int(len(indices)),
                "metrics": evaluate_window(
                    dataset, indices, e3c_model, e10b_model, e10c_model,
                    financial_index, args.batch_size, device,
                ),
            })

        result = {
            "experiment": "rolling_validation_e10_fixed_checkpoints",
            "folds": args.folds,
            "source_split": "val_development_only",
            "holdout_fraction": args.holdout_fraction,
            "holdout_start_date_excluded": split.holdout_start_date,
            "report_lag_days": report_lag_days,
            "market_checkpoint": str(args.market_checkpoint),
            "e10b_checkpoint": str(args.e10b_checkpoint),
            "e10c_checkpoint": str(args.e10c_checkpoint),
            "windows": windows,
            "development_aggregate": evaluate_window(
                dataset, split.dev_indices, e3c_model, e10b_model, e10c_model,
                financial_index, args.batch_size, device,
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--market-checkpoint", type=Path, default=Path("experiments/gru_e3/checkpoints/best.pt"))
    parser.add_argument("--e10b-checkpoint", type=Path, default=Path("experiments/gru_e10_b/checkpoints/best.pt"))
    parser.add_argument("--e10c-checkpoint", type=Path, default=Path("experiments/gru_e10_c/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gru_validation/rolling_validation_e10.json"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
