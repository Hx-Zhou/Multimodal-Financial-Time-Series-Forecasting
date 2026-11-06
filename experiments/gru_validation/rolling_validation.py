"""Evaluate fixed checkpoints on chronological development windows.

This is a diagnostic rolling-time evaluation.  It does not retrain models and
never uses the official holdout portion for model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset
from experiments.gru_e3.model import build_model as build_market_model
from experiments.gru_e3.train import numeric_batch
from experiments.gru_e8.train import FinancialIndex, add_financial_batch
from experiments.gru_e9_2.model import build_model as build_financial_model
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
    date_groups = np.array_split(unique_dates, folds)
    return [
        indices[np.isin(dates[indices], group)]
        for group in date_groups
    ]


def update_metrics(total: dict, error: torch.Tensor) -> None:
    total["absolute"] += error.abs().sum().item()
    total["squared"] += error.square().sum().item()
    total["count"] += error.numel()


def finish_metrics(total: dict) -> dict[str, float]:
    if total["count"] == 0:
        raise ValueError("evaluation window is empty")
    return {
        "mae": total["absolute"] / total["count"],
        "mse": total["squared"] / total["count"],
    }


def evaluate_window(
    dataset: HS300Dataset,
    indices: np.ndarray,
    market_model: torch.nn.Module,
    financial_model: torch.nn.Module,
    financial_index: FinancialIndex,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    market_model.eval()
    financial_model.eval()
    metrics = {
        "e0_last_value": {"absolute": 0.0, "squared": 0.0, "count": 0},
        "e3c": {"absolute": 0.0, "squared": 0.0, "count": 0},
        "e9_2": {"absolute": 0.0, "squared": 0.0, "count": 0},
    }
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = numeric_batch(
                dataset, selected, device, 4, "relative", "standardized_close"
            )
            target = batch["y"]
            e0_prediction = batch["last_close"][:, None].expand_as(target)
            update_metrics(metrics["e0_last_value"], e0_prediction - target)
            e3c_prediction = market_model(batch)
            update_metrics(metrics["e3c"], e3c_prediction - target)
            financial_batch = add_financial_batch(dataset, selected, batch, financial_index)
            e9_2_prediction = financial_model(financial_batch)
            update_metrics(metrics["e9_2"], e9_2_prediction - target)
    return {name: finish_metrics(value) for name, value in metrics.items()}


def run(args) -> dict:
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    device = torch.device(args.device)
    dataset = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(dataset, args.holdout_fraction)
        folds = make_date_folds(dataset, split.dev_indices, args.folds)

        e3_checkpoint = load_checkpoint(args.market_checkpoint, device)
        e9_checkpoint = load_checkpoint(args.financial_checkpoint, device)
        market_model = build_market_model(e3_checkpoint["config"]).to(device)
        market_model.load_state_dict(e3_checkpoint["model_state"])
        financial_model = build_financial_model(e9_checkpoint["config"]).to(device)
        financial_model.load_state_dict(e9_checkpoint["model_state"])
        financial_index = FinancialIndex(dataset)

        results = []
        for fold_number, indices in enumerate(folds, 1):
            dates = dataset.target_start_date[indices]
            print(
                f"fold {fold_number}/{len(folds)}: {int(dates.min())}-{int(dates.max())} "
                f"samples={len(indices)}",
                flush=True,
            )
            results.append(
                {
                    "fold": fold_number,
                    "target_start_min": int(dates.min()),
                    "target_start_max": int(dates.max()),
                    "samples": int(len(indices)),
                    "metrics": evaluate_window(
                        dataset, indices, market_model, financial_model,
                        financial_index, args.batch_size, device,
                    ),
                }
            )

        # Aggregate over all development windows, retaining the same date-only
        # split used by the fold reports.
        aggregate = evaluate_window(
            dataset, split.dev_indices, market_model, financial_model,
            financial_index, args.batch_size, device,
        )
        result = {
            "experiment": "rolling_validation_fixed_checkpoints",
            "folds": args.folds,
            "source_split": "val_development_only",
            "holdout_fraction": args.holdout_fraction,
            "holdout_start_date_excluded": split.holdout_start_date,
            "market_checkpoint": str(args.market_checkpoint),
            "financial_checkpoint": str(args.financial_checkpoint),
            "windows": results,
            "development_aggregate": aggregate,
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
    parser.add_argument("--financial-checkpoint", type=Path, default=Path("experiments/gru_e9_2/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gru_validation/rolling_validation.json"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
