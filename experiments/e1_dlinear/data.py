"""Materialize Close windows from the unchanged official reader."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split
from .model import HORIZON, LOOKBACK, prepare_batch


@dataclass
class CloseData:
    x: torch.Tensor
    y: torch.Tensor
    stocks: np.ndarray
    codes: list[str]
    news_counts: np.ndarray | None = None
    target_dates: np.ndarray | None = None


def materialize(dataset: HS300Dataset, indices: np.ndarray, *, news_counts=False) -> CloseData:
    """Vectorized access avoids parsing reports for every pure-price example."""
    if dataset.y is None:
        raise ValueError("training/evaluation requires labels")
    stocks = dataset.stock_index[indices]
    starts = dataset.input_start[indices]
    x = np.empty((len(indices), LOOKBACK), dtype=np.float32)
    counts = np.zeros(len(indices), dtype=np.int64) if news_counts else None
    for stock, code in enumerate(dataset.stock_codes):
        positions = np.flatnonzero(stocks == stock)
        if not len(positions):
            continue
        raw = dataset.price_values[code][:, 3]
        mean = np.float32(dataset.close_mean[stock])
        std = max(np.float32(dataset.close_std[stock]), np.float32(1e-6))
        windows = np.lib.stride_tricks.sliding_window_view(raw, LOOKBACK)
        x[positions] = (windows[starts[positions]] - mean) / std
        if counts is not None:
            times = dataset.news_by_stock.get(code, (np.empty(0, dtype=np.int64), None))[0]
            days = dataset.price_days[code]
            left = days[starts[positions]].astype(np.int64) * 1_000_000
            right = days[starts[positions] + LOOKBACK - 1].astype(np.int64) * 1_000_000 + 235959
            counts[positions] = np.searchsorted(times, right, side="right") - np.searchsorted(times, left)
    y = np.asarray(dataset.y[indices], dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite numerical data")
    # Verify this faster path against the public sample interface and raw future prices.
    for position in np.unique(np.linspace(0, len(indices) - 1, min(20, len(indices)), dtype=int)):
        sample = dataset[int(indices[position])]
        batch = prepare_batch([sample])
        np.testing.assert_array_equal(x[position], batch["close"][0].numpy())
        np.testing.assert_array_equal(y[position], batch["y"][0].numpy())
        stock = int(stocks[position])
        raw = dataset.price_values[dataset.stock_codes[stock]][:, 3]
        start = int(starts[position]) + LOOKBACK
        expected = (raw[start:start + HORIZON] - np.float32(sample["close_mean"])) / max(
            np.float32(sample["close_std"]), np.float32(1e-6)
        )
        np.testing.assert_allclose(y[position], expected, rtol=2e-5, atol=2e-5)
    return CloseData(torch.from_numpy(x), torch.from_numpy(y), stocks, dataset.stock_codes, counts,
                     np.asarray(dataset.target_start_date[indices], dtype=np.int32))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(package_dir: Path) -> tuple[CloseData, CloseData, dict]:
    print("Loading fixed reader and checking the shared chronological split...", flush=True)
    train = HS300Dataset(package_dir, "train")
    val = HS300Dataset(package_dir, "val")
    try:
        if train.stock_codes != val.stock_codes:
            raise ValueError("train and val stock-code order differs")
        np.testing.assert_array_equal(train.close_mean, val.close_mean)
        np.testing.assert_array_equal(train.close_std, val.close_std)
        split = make_validation_split(val, holdout_fraction=0.2)
        metadata = {
            "development_samples": len(split.dev_indices),
            "purged_overlap_samples": len(split.purged_indices),
            "holdout_samples": len(split.holdout_indices),
            "development_target_start_date_min": split.dev_start_date,
            "development_target_start_date_max": split.dev_end_date,
            "development_target_end_date_max": split.dev_target_end_date,
            "holdout_target_start_date_min": split.holdout_start_date,
            "holdout_target_start_date_max": split.holdout_end_date,
        }
        reference = json.loads((Path(__file__).resolve().parents[1] / "validation_split.json").read_text())
        for key, value in metadata.items():
            if value != reference[key]:
                raise ValueError(f"shared split mismatch: {key}: {value} != {reference[key]}")
        train_end = max(
            int(train.price_days[code][train.input_start[train.stock_index == i] + LOOKBACK + HORIZON - 1].max())
            for i, code in enumerate(train.stock_codes) if np.any(train.stock_index == i)
        )
        if train_end >= split.dev_start_date:
            raise ValueError("training labels overlap development target dates")
        metadata.update({
            "train_samples": len(train), "train_target_end_date_max": train_end,
            "holdout_evaluated": False,
            "train_npz_sha256": sha256_file(package_dir / "data/train.npz"),
            "val_npz_sha256": sha256_file(package_dir / "data/val.npz"),
        })
        training = materialize(train, np.arange(len(train)))
        development = materialize(val, split.dev_indices, news_counts=True)
        metadata["train_close_sha256"] = hashlib.sha256(training.x.numpy().tobytes()).hexdigest()
        metadata["dev_close_sha256"] = hashlib.sha256(development.x.numpy().tobytes()).hexdigest()
        return training, development, metadata
    finally:
        train.close()
        val.close()
