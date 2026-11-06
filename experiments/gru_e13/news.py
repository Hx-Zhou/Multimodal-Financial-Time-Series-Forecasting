"""Causal trading-interval news aggregation and training-only PCA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NewsEvents:
    times: np.ndarray
    ids: np.ndarray
    bins: np.ndarray


class AlignedNewsIndex:
    """Assign each article to the first trading cutoff at/after publication.

    Repeated vector IDs are treated as repeated articles within a stock. Keep
    only the earliest occurrence, never a later duplicate's timestamp. Different
    stocks retain their own association with the same article. This is exact ID
    deduplication, not semantic or near-duplicate detection.
    """

    def __init__(self, dataset, cutoff_time: int = 235959):
        hour, minute, second = cutoff_time // 10000, cutoff_time // 100 % 100, cutoff_time % 100
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            raise ValueError("cutoff-time must be a valid HHMMSS integer")
        self.cutoff_time = cutoff_time
        self.by_stock = {}
        self.audit = dict(raw_records=0, duplicate_records_removed=0, retained_records=0,
                          outside_price_calendar=0, assigned_records=0,
                          assigned_to_later_calendar_date=0)
        for code in dataset.stock_codes:
            times, ids = dataset.news_by_stock.get(code, (np.empty(0, np.int64), np.empty(0, np.int32)))
            order = np.argsort(times, kind="stable")
            times, ids = times[order], ids[order]
            _, first = np.unique(ids, return_index=True)
            first.sort()
            self.audit["raw_records"] += len(ids)
            self.audit["duplicate_records_removed"] += len(ids) - len(first)
            times, ids = times[first], ids[first]
            self.audit["retained_records"] += len(ids)
            days = dataset.price_days[code].astype(np.int64)
            if np.any(np.diff(days) <= 0):
                raise ValueError(f"price dates must be strictly increasing: {code}")
            cutoffs = days * 1_000_000 + cutoff_time
            bins = np.searchsorted(cutoffs, times, side="left")
            # Do not import articles before the available price calendar starts.
            valid = (times >= days[0] * 1_000_000) & (bins < len(days))
            self.audit["outside_price_calendar"] += int((~valid).sum())
            times, ids, bins = times[valid], ids[valid], bins[valid]
            self.audit["assigned_records"] += len(ids)
            self.audit["assigned_to_later_calendar_date"] += int((times // 1_000_000 != days[bins]).sum())
            self.by_stock[code] = NewsEvents(times, ids, bins)

    def training_vector_ids(self, dataset, indices, lookback: int = 20) -> np.ndarray:
        """Only vectors actually falling in selected training feature windows."""
        pieces = []
        indices = np.asarray(indices, dtype=np.int64)
        stocks = dataset.stock_index[indices]
        for stock in np.unique(stocks):
            code = dataset.stock_codes[int(stock)]
            ends = dataset.input_start[indices[stocks == stock]].astype(np.int64) + 95
            starts = np.maximum(ends - lookback + 1, 0)
            changes = np.zeros(len(dataset.price_days[code]) + 1, dtype=np.int64)
            np.add.at(changes, starts, 1)
            np.add.at(changes, ends + 1, -1)
            covered = np.cumsum(changes[:-1]) > 0
            events = self.by_stock[code]
            pieces.append(events.ids[covered[events.bins]])
        ids = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, np.int32)
        if len(ids) < 2:
            raise ValueError("not enough available training news for PCA")
        return ids

    def features(self, dataset, indices, projected, windows=(1, 5, 20)):
        """For each window: article-weighted mean PCA, count, coverage, recency.

        No future bin contributes to a prefix difference. Metadata has three
        entries per window: log1p(count), fraction of bins with news, and
        log1p(trading intervals since latest news), zero when absent.
        """
        indices = np.asarray(indices, dtype=np.int64)
        dim = projected.shape[1]
        content = np.zeros((len(indices), len(windows) * dim), np.float32)
        metadata = np.zeros((len(indices), len(windows) * 3), np.float32)
        active = np.zeros(len(indices), dtype=bool)
        stocks = dataset.stock_index[indices]
        for stock in np.unique(stocks):
            rows = np.flatnonzero(stocks == stock)
            code = dataset.stock_codes[int(stock)]
            ends = dataset.input_start[indices[rows]].astype(np.int64) + 95
            events = self.by_stock[code]
            n = len(dataset.price_days[code])
            counts = np.bincount(events.bins, minlength=n)
            sums = np.zeros((n, dim), dtype=np.float64)
            np.add.at(sums, events.bins, projected[events.ids])
            count_prefix = np.r_[0, np.cumsum(counts)]
            coverage_prefix = np.r_[0, np.cumsum(counts > 0)]
            sum_prefix = np.vstack((np.zeros((1, dim)), np.cumsum(sums, axis=0)))
            last = np.maximum.accumulate(np.where(counts > 0, np.arange(n), -n))
            for i, window in enumerate(windows):
                starts = np.maximum(ends - window + 1, 0)
                count = count_prefix[ends + 1] - count_prefix[starts]
                present = count > 0
                mean = (sum_prefix[ends + 1] - sum_prefix[starts]) / np.maximum(count[:, None], 1)
                content[rows, i * dim:(i + 1) * dim] = mean
                metadata[rows, i * 3] = np.log1p(count)
                metadata[rows, i * 3 + 1] = (coverage_prefix[ends + 1] - coverage_prefix[starts]) / window
                metadata[rows, i * 3 + 2] = np.where(present, np.log1p(ends - last[ends]), 0)
                active[rows] |= present
        return content, metadata, active


def fit_pca(vectors, ids, components=16, max_vectors=50000, seed=42):
    """Fit a centered (not whitened) PCA using training article IDs only."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=min(len(ids), max_vectors), replace=False)
    if not 1 <= components <= min(len(chosen) - 1, vectors.shape[1]):
        raise ValueError("PCA components exceed the available training matrix rank")
    x = torch.as_tensor(vectors[chosen], dtype=torch.float64)
    mean = x.mean(dim=0)
    x -= mean
    _, eigenvectors = torch.linalg.eigh(x.T @ x / len(x))
    basis = eigenvectors[:, -components:].flip(1)
    return mean.float().numpy(), basis.float().numpy(), len(chosen)


def project_vectors(vectors, mean, basis):
    output = np.empty((len(vectors), basis.shape[1]), np.float32)
    for start in range(0, len(vectors), 8192):
        output[start:start + 8192] = (vectors[start:start + 8192] - mean) @ basis
    return output


def shuffled_projection(projected, training_ids, seed=42):
    """Reassign article content from a training-only bank; preserve event times.

    The mapping is fixed per article ID across stocks and splits. Training IDs
    get a permutation; unseen IDs get random training-bank vectors. No future
    validation article is used to supply content for a training article.
    """
    rng = np.random.default_rng(seed)
    sources = rng.choice(training_ids, size=len(projected), replace=True)
    sources[training_ids] = rng.permutation(training_ids)
    return projected[sources]
