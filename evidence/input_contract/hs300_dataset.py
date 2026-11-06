"""Fixed reader for the FinMultiTime HS300 course packages.

The reader retains each trading-day price row, each news article embedding,
and the original financial report records. It does not average news or impose
a report publication lag. Stock suffix .SS in price filenames is matched with
.SH in the news and financial records.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")
CODE = re.compile(r"\d{6}\.(?:SZ|SH|SS)")


def _code(member: str) -> str:
    match = CODE.search(member)
    if not match:
        raise ValueError(f"no stock code in {member}")
    code = match.group(0)
    return code[:-2] + "SH" if code.endswith("SS") else code


@lru_cache(maxsize=2)
def _load_embeddings(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Share the large read-only vector matrix across train and val readers."""
    with np.load(path, allow_pickle=False) as news:
        return (
            news["vectors"],
            news["codes"].tolist(),
            news["offsets"],
            news["times"],
            news["vector_ids"],
        )


class NewsWindow:
    """A 96-day view of separate article embeddings and original timestamps."""

    def __init__(self, days: np.ndarray, times: np.ndarray, ids: np.ndarray, vectors: np.ndarray):
        self.days = days
        self._times = times
        self._ids = ids
        self._vectors = vectors

    def __len__(self) -> int:
        return len(self.days)

    def _range(self, day: int) -> tuple[int, int]:
        left = int(np.searchsorted(self._times, day * 1_000_000, side="left"))
        right = int(np.searchsorted(self._times, day * 1_000_000 + 235959, side="right"))
        return left, right

    def __getitem__(self, index: int) -> np.ndarray:
        left, right = self._range(int(self.days[index]))
        if left == right:
            return np.empty((0, self._vectors.shape[1]), dtype=np.float32)
        return self._vectors[self._ids[left:right]]

    def times(self, index: int) -> np.ndarray:
        left, right = self._range(int(self.days[index]))
        return self._times[left:right]

    def events(self) -> tuple[np.ndarray, np.ndarray]:
        """Return every article within the input date range, including non-trading days."""
        left = int(np.searchsorted(self._times, int(self.days[0]) * 1_000_000, side="left"))
        right = int(np.searchsorted(self._times, int(self.days[-1]) * 1_000_000 + 235959, side="right"))
        return self._vectors[self._ids[left:right]], self._times[left:right]

    def mean_recent(self, days: int = 20) -> np.ndarray:
        """A convenience aggregation used only by the supplied baseline model."""
        start = int(self.days[max(0, len(self.days) - days)])
        left = int(np.searchsorted(self._times, start * 1_000_000, side="left"))
        right = int(np.searchsorted(self._times, int(self.days[-1]) * 1_000_000 + 235959, side="right"))
        if left == right:
            return np.zeros(self._vectors.shape[1], dtype=np.float32)
        return self._vectors[self._ids[left:right]].mean(axis=0, dtype=np.float32)


class NewsTimes:
    def __init__(self, news: NewsWindow):
        self.news = news

    def __len__(self) -> int:
        return len(self.news)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.news.times(index)


class ReportWindow:
    """Original financial records whose report period ends by input_end."""

    def __init__(self, records: dict[str, list[dict]], input_end: int):
        self._records = records
        self.input_end = input_end

    def records(self, kind: str) -> list[dict]:
        return [row for row in self._records[kind] if int(row["end_date"]) <= self.input_end]

    def latest(self, kind: str) -> dict | None:
        available = self.records(kind)
        return max(available, key=lambda x: int(x["end_date"])) if available else None

    def __getitem__(self, kind: str) -> list[dict]:
        return self.records(kind)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._records)


class HS300Dataset:
    """Dataset for a split stored inside one extracted package directory."""

    def __init__(self, package_dir: str | Path, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.package_dir = Path(package_dir)
        self.data_dir = self.package_dir / "data"
        with np.load(self.data_dir / f"{split}.npz", allow_pickle=False) as values:
            self.stock_codes = values["stock_codes"].tolist()
            self.stock_index = values["stock_index"]
            self.input_start = values["input_start"]
            self.target_start_date = values["target_start_date"]
            self.close_mean = values["close_mean"]
            self.close_std = values["close_std"]
            self.y = values["y"] if "y" in values.files else None
        self.split = split
        self.price_days: dict[str, np.ndarray] = {}
        self.price_values: dict[str, np.ndarray] = {}
        price_root = self.data_dir / "numerical/HS300_time_series"
        for source_file in price_root.rglob("*.csv"):
            code = _code(str(source_file))
            with source_file.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                days = []
                numbers = []
                for row in reader:
                    day = int(row["Date"][:10].replace("-", ""))
                    if day < 20200331 or day > 20250331:
                        continue
                    days.append(day)
                    numbers.append([float(row[col]) for col in PRICE_COLUMNS])
            self.price_days[code] = np.asarray(days, dtype=np.int32)
            self.price_values[code] = np.asarray(numbers, dtype=np.float32)
        missing = set(self.stock_codes) - self.price_days.keys()
        if missing:
            raise ValueError(f"price series missing for {sorted(missing)[:3]}")
        self.news_vectors, news_codes, offsets, times, ids = _load_embeddings(
            (self.data_dir / "text/embeddings.npz").resolve()
        )
        if self.news_vectors.dtype != np.float32 or self.news_vectors.shape[1] != 512:
            raise ValueError("news embeddings must be float32 with 512 columns")
        self.news_by_stock: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, code in enumerate(news_codes):
            left, right = int(offsets[i]), int(offsets[i + 1])
            order = np.argsort(times[left:right], kind="stable")
            self.news_by_stock[code] = (times[left:right][order], ids[left:right][order])
        self.report_members: dict[str, dict[str, Path]] = {}
        report_root = self.data_dir / "tabular/hs300_tabular"
        for member in report_root.rglob("*.jsonl"):
            self.report_members.setdefault(_code(str(member)), {})[member.stem] = member
        if not self.report_members:
            raise FileNotFoundError(f"no financial reports under {report_root}")

    def __len__(self) -> int:
        return len(self.stock_index)

    @lru_cache(maxsize=32)
    def _reports(self, code: str) -> dict[str, list[dict]]:
        result = {}
        for kind, member in self.report_members.get(code, {}).items():
            records = []
            with member.open("r", encoding="utf-8") as stream:
                for line in stream:
                    value = json.loads(line)
                    records.extend(value if isinstance(value, list) else [value])
            result[kind] = records
        return result

    def __getitem__(self, index: int) -> dict:
        stock_index = int(self.stock_index[index])
        code = self.stock_codes[stock_index]
        start = int(self.input_start[index])
        days = self.price_days[code][start : start + 96]
        numerical = self.price_values[code][start : start + 96]
        if days.shape != (96,) or numerical.shape != (96, len(PRICE_COLUMNS)):
            raise IndexError(f"invalid price window {index} for {code}")
        empty_times = np.empty(0, dtype=np.int64)
        empty_ids = np.empty(0, dtype=np.int32)
        times, ids = self.news_by_stock.get(code, (empty_times, empty_ids))
        news = NewsWindow(days, times, ids, self.news_vectors)
        sample = {
            "stock_code": code,
            "dates": days,
            "numerical": numerical,
            "news": news,
            "news_time": NewsTimes(news),
            "financial": ReportWindow(self._reports(code), int(days[-1])),
            "close_mean": float(self.close_mean[stock_index]),
            "close_std": float(self.close_std[stock_index]),
            "target_start_date": int(self.target_start_date[index]),
        }
        if self.y is not None:
            sample["y"] = self.y[index]
        return sample

    def close(self) -> None:
        """Kept as a stable dataset interface; directory files are closed after reading."""
