"""E0 baselines for the HS300 forecasting task.

The baselines predict all 24 future standardized closes with either the last
standardized close in the input window or a recent moving average.  When
requested, the official validation split is divided chronologically into an
earlier tuning portion and a later pseudo-test portion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split


HORIZONS = (1, 6, 12, 24)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Return aggregate and selected-horizon MAE/MSE for [N, 24] arrays."""
    difference = prediction - target
    result = {
        "samples": int(target.shape[0]),
        "mae": float(np.abs(difference).mean()),
        "mse": float(np.square(difference).mean()),
    }
    for horizon in HORIZONS:
        error = difference[:, horizon - 1]
        result[f"mae_h{horizon}"] = float(np.abs(error).mean())
        result[f"mse_h{horizon}"] = float(np.square(error).mean())
    return result


def _collect(dataset: HS300Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build last-value, moving-average inputs and targets from a split."""
    n = len(dataset)
    last = np.empty(n, dtype=np.float32)
    close_means = np.empty((n, 20), dtype=np.float32)
    targets = np.empty((n, 24), dtype=np.float32)

    for index in range(n):
        sample = dataset[index]
        close = sample["numerical"][:, 3].astype(np.float32)
        mean = np.float32(sample["close_mean"])
        std = max(np.float32(sample["close_std"]), np.float32(1e-6))
        standardized_close = (close - mean) / std
        last[index] = standardized_close[-1]
        for column, window in enumerate(range(1, 21)):
            close_means[index, column] = standardized_close[-window:].mean()
        targets[index] = np.asarray(sample["y"], dtype=np.float32)
    return last, close_means, targets


def _evaluate(
    last: np.ndarray,
    recent_means: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Evaluate last-value and moving-average baselines on a boolean mask."""
    target = target[mask]
    result = {"last_value": _metrics(np.repeat(last[mask, None], 24, axis=1), target)}
    for window in (5, 10, 20):
        baseline = recent_means[mask, window - 1]
        prediction = np.repeat(baseline[:, None], 24, axis=1)
        result[f"moving_average_{window}"] = _metrics(prediction, target)
    return result


def run(package_dir: Path, split: str, pseudo_test_fraction: float) -> dict:
    if not 0.0 <= pseudo_test_fraction < 1.0:
        raise ValueError("pseudo-test fraction must be in [0, 1)")

    dataset = HS300Dataset(package_dir, split)
    try:
        last, recent_means, target = _collect(dataset)
        split_info = (
            make_validation_split(dataset, pseudo_test_fraction)
            if pseudo_test_fraction > 0.0
            else None
        )
    finally:
        dataset.close()

    if split_info is None:
        tuning_mask = np.ones(len(target), dtype=bool)
        pseudo_test_mask = np.zeros(len(target), dtype=bool)
        dates = np.asarray(dataset.target_start_date, dtype=np.int32)
        development_start = int(dates.min())
        development_end = int(dates.max())
        development_target_end = None
        pseudo_test_start = None
        pseudo_test_end = None
        purged_samples = 0
    else:
        tuning_mask = np.zeros(len(target), dtype=bool)
        pseudo_test_mask = np.zeros(len(target), dtype=bool)
        tuning_mask[split_info.dev_indices] = True
        pseudo_test_mask[split_info.holdout_indices] = True
        development_start = split_info.dev_start_date
        development_end = split_info.dev_end_date
        development_target_end = split_info.dev_target_end_date
        pseudo_test_start = split_info.holdout_start_date
        pseudo_test_end = split_info.holdout_end_date
        purged_samples = len(split_info.purged_indices)

    result = {
        "split": split,
        "pseudo_test_fraction": pseudo_test_fraction,
        "development_samples": int(tuning_mask.sum()),
        "purged_overlap_samples": int(purged_samples),
        "pseudo_test_samples": int(pseudo_test_mask.sum()),
        "development_target_start": development_start,
        "development_target_start_max": development_end,
        "development_target_end": development_target_end,
        "pseudo_test_target_start": pseudo_test_start,
        "pseudo_test_target_end": pseudo_test_end,
        "development": _compact_metrics(_evaluate(last, recent_means, target, tuning_mask)),
        "pseudo_test": (
            _compact_metrics(_evaluate(last, recent_means, target, pseudo_test_mask))
            if pseudo_test_mask.any()
            else {}
        ),
    }
    return result


def _compact_metrics(metrics: dict) -> dict:
    """Keep the result file focused on the aggregate comparison."""
    return {
        name: {"mae": values["mae"], "mse": values["mse"]}
        for name, values in metrics.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root containing data/ (defaults to the repository root)",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--pseudo-test-fraction",
        type=float,
        default=0.2,
        help="fraction of the latest dates reserved as chronological pseudo-test data",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run(args.package_dir, args.split, args.pseudo_test_fraction)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
