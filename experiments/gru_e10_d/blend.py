"""Select and evaluate a fixed E3c/E10-B prediction blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset
from experiments.gru_e3.model import build_model as build_market_model
from experiments.gru_e3.train import numeric_batch
from experiments.gru_e10_b.model import build_model as build_financial_model
from experiments.gru_e10_b.train import FinancialIndex, add_financial_batch
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
        indices[np.isin(dates[indices], group)]
        for group in np.array_split(unique_dates, folds)
    ]


def empty_totals(weights: list[float]) -> dict[str, dict[str, float]]:
    return {
        str(weight): {"absolute": 0.0, "squared": 0.0, "count": 0}
        for weight in weights
    }


def add_batch_metrics(
    totals: dict[str, dict[str, float]],
    weights: list[float],
    market_prediction: torch.Tensor,
    financial_prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    for weight in weights:
        prediction = (1.0 - weight) * market_prediction + weight * financial_prediction
        error = prediction - target
        total = totals[str(weight)]
        total["absolute"] += error.abs().sum().item()
        total["squared"] += error.square().sum().item()
        total["count"] += error.numel()


def finish_totals(totals: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        weight: {
            "mae": values["absolute"] / values["count"],
            "mse": values["squared"] / values["count"],
        }
        for weight, values in totals.items()
    }


def evaluate_indices(
    dataset: HS300Dataset,
    indices: np.ndarray,
    market_model: torch.nn.Module,
    financial_model: torch.nn.Module,
    financial_index: FinancialIndex,
    weights: list[float],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    market_model.eval()
    financial_model.eval()
    totals = empty_totals(weights)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            market = numeric_batch(
                dataset, selected, device, 4, "relative", "standardized_close"
            )
            financial = add_financial_batch(dataset, selected, market, financial_index)
            market_prediction = market_model(market)
            financial_prediction = financial_model(financial)
            add_batch_metrics(
                totals, weights, market_prediction, financial_prediction, market["y"]
            )
    return finish_totals(totals)


def choose_weight(metrics: dict[str, dict[str, float]]) -> float:
    """Choose by development MAE, with MSE as a deterministic tie-breaker."""
    return float(min(metrics, key=lambda weight: (metrics[weight]["mae"], metrics[weight]["mse"])))


def run(args) -> dict:
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    weights = sorted({float(weight) for weight in args.weights})
    if not weights or any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("weights must be between 0 and 1")
    device = torch.device(args.device)
    dataset = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(dataset, args.holdout_fraction)
        folds = make_date_folds(dataset, split.dev_indices, args.folds)
        market_checkpoint = load_checkpoint(args.market_checkpoint, device)
        financial_checkpoint = load_checkpoint(args.financial_checkpoint, device)
        market_model = build_market_model(market_checkpoint["config"]).to(device)
        market_model.load_state_dict(market_checkpoint["model_state"])
        financial_model = build_financial_model(financial_checkpoint["config"]).to(device)
        financial_model.load_state_dict(financial_checkpoint["model_state"])
        report_lag_days = int(financial_checkpoint.get("report_lag_days", -1))
        stats = financial_checkpoint.get("feature_stats")
        if report_lag_days < 0 or not stats:
            raise ValueError("E10-B checkpoint is missing report lag or feature statistics")
        financial_index = FinancialIndex(
            dataset,
            report_lag_days,
            (
                np.asarray(stats["mean"], dtype=np.float32),
                np.asarray(stats["std"], dtype=np.float32),
            ),
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
                "metrics": evaluate_indices(
                    dataset, indices, market_model, financial_model,
                    financial_index, weights, args.batch_size, device,
                ),
            })

        development = evaluate_indices(
            dataset, split.dev_indices, market_model, financial_model,
            financial_index, weights, args.batch_size, device,
        )
        selected_weight = choose_weight(development)
        holdout_all = evaluate_indices(
            dataset, split.holdout_indices, market_model, financial_model,
            financial_index, [selected_weight], args.batch_size, device,
        )
        result = {
            "experiment": "E10_D_fixed_e3c_e10b_blend",
            "weights_definition": "prediction=(1-w)*E3c+w*E10-B",
            "weights_tested": weights,
            "selected_weight": selected_weight,
            "selection_metric": "development aggregate MAE, MSE tie-breaker",
            "folds": args.folds,
            "source_split": "val_development_only_for_selection",
            "holdout_fraction": args.holdout_fraction,
            "holdout_start_date_excluded": split.holdout_start_date,
            "report_lag_days": report_lag_days,
            "market_checkpoint": str(args.market_checkpoint),
            "financial_checkpoint": str(args.financial_checkpoint),
            "windows": windows,
            "development_aggregate": development,
            "selected_holdout": holdout_all[str(selected_weight)],
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
    parser.add_argument("--financial-checkpoint", type=Path, default=Path("experiments/gru_e10_b/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gru_e10_d/result.json"))
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
