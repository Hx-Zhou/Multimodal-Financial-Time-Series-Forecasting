"""Fixed-reader data, training-only transforms, and compact daily news pooling."""
from dataclasses import dataclass
import gc
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset, _load_embeddings
from experiments.validation_split import make_validation_split
from ..data import sha256_file
from ..run import ROOT, write_json


@dataclass
class Samples:
    starts: np.ndarray
    y: np.ndarray
    stocks: np.ndarray
    dates: np.ndarray


def stationary_fields(prices, mean, std):
    close = prices[:, 3]
    safe_close = np.maximum(close, 1e-6)
    previous = np.r_[close[:1], close[:-1]]
    return np.column_stack((
        np.log(safe_close / np.maximum(previous, 1e-6)),
        (prices[:, 1] - prices[:, 2]) / safe_close,
        (close - prices[:, 0]) / np.maximum(prices[:, 0], 1e-6),
        np.log1p(np.maximum(prices[:, 4], 0)),
        (prices[:, 5] != 0).astype(np.float32),
        (prices[:, 6] != 0).astype(np.float32),
        (close - mean) / std,
    )).astype(np.float32)


def standardize(values, train_mask, columns):
    result = values.copy()
    selected = values[train_mask][:, columns].astype(np.float64)
    mean = selected.mean(axis=0)
    std = np.maximum(selected.std(axis=0), 1e-6)
    result[:, columns] = ((values[:, columns].astype(np.float64) - mean) / std).astype(np.float32)
    return result, {"columns": columns, "mean": mean.tolist(), "std": std.tolist(),
                    "unique_training_input_days": int(train_mask.sum())}


class Store:
    def batch(self, samples, rows, representation, news=False, zero_semantics=False, news_rows=None):
        days = samples.starts[rows, None] + np.arange(96, dtype=np.int64)
        features = self.features[representation][days].copy()
        anchor = self.close[days[:, -1]][:, None]
        if representation == "raw":
            features[:, :, :4] -= anchor[:, None, :]
        else:
            features[:, :, 6:7] -= anchor[:, None, :]
            # The public input window does not supply a preceding day.
            features[:, 0, 0] = self.zero_return
        batch = {"market": torch.from_numpy(features), "anchor": torch.from_numpy(anchor)}
        if news:
            news_days = days if news_rows is None else samples.starts[news_rows, None] + np.arange(96)
            ids = self.news_ids[news_days]
            unique, inverse = np.unique(ids, return_inverse=True)
            batch.update(
                news_unique=torch.from_numpy(np.asarray(self.pooled[unique]).copy()),
                news_inverse=torch.from_numpy(inverse),
                has_news=torch.from_numpy((ids != 0).astype(np.float32)),
                news_count=torch.from_numpy(np.log1p(self.news_counts[news_days]).astype(np.float32)),
                zero_semantics=zero_semantics,
            )
        return batch


def load_store(package_dir, cache_dir, with_news):
    cache_dir.mkdir(parents=True, exist_ok=False)
    reference = json.loads((ROOT / "experiments/e1_dlinear/results/calibration/summary.json").read_text())
    for split_name in ("train", "val"):
        actual = sha256_file(package_dir / f"data/{split_name}.npz")
        if actual != reference["split"][f"{split_name}_npz_sha256"]:
            raise ValueError(f"{split_name} data fingerprint changed")
    train = HS300Dataset(package_dir, "train")
    val = HS300Dataset(package_dir, "val")
    store = Store()
    try:
        np.testing.assert_array_equal(train.close_mean, val.close_mean)
        np.testing.assert_array_equal(train.close_std, val.close_std)
        if train.stock_codes != val.stock_codes:
            raise ValueError("stock ordering changed")
        split = make_validation_split(val)
        actual_split = {
            "development_samples": len(split.dev_indices),
            "purged_overlap_samples": len(split.purged_indices),
            "holdout_samples": len(split.holdout_indices),
            "development_target_start_date_min": split.dev_start_date,
            "development_target_start_date_max": split.dev_end_date,
            "development_target_end_date_max": split.dev_target_end_date,
            "holdout_target_start_date_min": split.holdout_start_date,
            "holdout_target_start_date_max": split.holdout_end_date,
        }
        for key, value in actual_split.items():
            if value != reference["split"][key]:
                raise ValueError(f"split mismatch: {key}")
        if split.dev_target_end_date >= split.holdout_start_date:
            raise ValueError("development labels overlap holdout")
        store.codes = list(train.stock_codes)
        lengths = [len(train.price_values[code]) for code in store.codes]
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        total = int(offsets[-1])
        raw, stationary = np.empty((total, 7), np.float32), np.empty((total, 7), np.float32)
        training_days = np.zeros(total, dtype=bool)
        for stock, code in enumerate(store.codes):
            p = train.price_values[code]
            section = slice(offsets[stock], offsets[stock + 1])
            mean = np.float32(train.close_mean[stock])
            std = max(np.float32(train.close_std[stock]), np.float32(1e-6))
            raw[section] = p
            raw[section, :4] = (p[:, :4] - mean) / std
            stationary[section] = stationary_fields(p, mean, std)
            starts = train.input_start[train.stock_index == stock].astype(np.int64)
            diff = np.zeros(len(p) + 1, dtype=np.int32)
            np.add.at(diff, starts, 1)
            np.add.at(diff, starts + 96, -1)
            training_days[section] = np.cumsum(diff[:-1]) > 0
        raw, raw_stats = standardize(raw, training_days, [4, 5, 6])
        stationary, stationary_stats = standardize(stationary, training_days, [0, 1, 2, 3])
        store.features = {"raw": raw, "stationary": stationary}
        store.close = stationary[:, 6].copy()
        store.zero_return = np.float32(-stationary_stats["mean"][0] / stationary_stats["std"][0])
        store.train = Samples(offsets[train.stock_index] + train.input_start,
                              train.y.astype(np.float32), train.stock_index.copy(), train.target_start_date.copy())
        idx = split.dev_indices
        store.dev = Samples(offsets[val.stock_index[idx]] + val.input_start[idx],
                            val.y[idx].astype(np.float32), val.stock_index[idx].copy(), val.target_start_date[idx].copy())
        # Check numerical inputs/labels against the unchanged official reader.
        for ds, samples, indices in ((train, store.train, np.arange(len(train))), (val, store.dev, idx)):
            for position in np.linspace(0, len(indices)-1, 12, dtype=int):
                sample = ds[int(indices[position])]
                expected = (sample["numerical"][:, 3] - np.float32(sample["close_mean"])) / max(np.float32(sample["close_std"]), np.float32(1e-6))
                np.testing.assert_array_equal(store.close[samples.starts[position]:samples.starts[position]+96], expected)
                np.testing.assert_array_equal(samples.y[position], sample["y"])
        metadata = {"split": actual_split, "train_samples": len(store.train.y),
                    "raw_columns": ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
                    "stationary_columns": ["log_return", "(High-Low)/Close", "(Close-Open)/Open",
                                           "log1p_volume", "has_dividend", "has_split", "standardized_Close"],
                    "raw_statistics": raw_stats, "stationary_statistics": stationary_stats,
                    "normalization": "OHLC use supplied per-stock train Close mean/std; other fitted stats use unique training input days only",
                    "holdout_evaluated": False}
        if with_news:
            # Exact calendar trading-day buckets, matching NewsWindow.__getitem__.
            # Non-trading-day articles are excluded; duplicate vector IDs count once per stock/day.
            buckets = []
            store.news_ids = np.zeros(total, dtype=np.int64)
            store.news_counts = np.zeros(total, dtype=np.int32)
            for stock, code in enumerate(store.codes):
                times, ids = train.news_by_stock.get(code, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                days = train.price_days[code]
                left = np.searchsorted(times, days.astype(np.int64) * 1000000)
                right = np.searchsorted(times, days.astype(np.int64) * 1000000 + 235959, side="right")
                for day_index in np.flatnonzero(right > left):
                    unique_ids = np.unique(ids[left[day_index]:right[day_index]])
                    row = len(buckets) + 1
                    global_day = int(offsets[stock] + day_index)
                    store.news_ids[global_day] = row
                    store.news_counts[global_day] = len(unique_ids)
                    buckets.append(unique_ids)
            pooled_path = cache_dir / "pooled_news.npy"
            pooled = np.lib.format.open_memmap(pooled_path, mode="w+", dtype=np.float32, shape=(len(buckets)+1, 512))
            pooled[0] = 0
            for row, ids in enumerate(buckets, 1):
                pooled[row] = train.news_vectors[ids].mean(axis=0, dtype=np.float64).astype(np.float32)
            pooled.flush()
            del pooled
            store.pooled = np.load(pooled_path, mmap_mode="r")
            metadata["news"] = {"daily_pool": "mean of unique vector_ids on the same trading calendar date",
                                "non_trading_day_articles": "excluded",
                                "days_with_news": len(buckets),
                                "embedding_sha256": sha256_file(package_dir / "data/text/embeddings.npz")}
        write_json(cache_dir / "data_protocol.json", metadata)
        store.metadata = metadata
        return store
    finally:
        train.close()
        val.close()
        # The reader's cached bound method retains dataset instances, including
        # their full news matrices, even after local train/val names are deleted.
        HS300Dataset._reports.cache_clear()
        del train, val
        _load_embeddings.cache_clear()
        gc.collect()
