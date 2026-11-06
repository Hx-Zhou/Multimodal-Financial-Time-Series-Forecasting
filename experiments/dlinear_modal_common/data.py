"""Availability-aware financial features; original readers/splits stay unchanged."""
import csv
from datetime import date, timedelta
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import _code
from experiments.e1_dlinear.data import sha256_file
from experiments.e1_dlinear.e6_plus.data import load_store, Samples
from experiments.e1_dlinear.run import write_json
from experiments.validation_split import make_validation_split

FIELDS = {
    "income": ("total_revenue", "n_income_attr_p", "basic_eps", "operate_profit"),
    "balance_sheet": ("total_assets", "total_liab", "total_hldr_eqy_exc_min_int", "money_cap"),
    "cashflow": ("n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act", "c_cash_equ_end_period"),
}
REPORT_DIM = 54  # 12 levels + 12 YoY changes + 24 masks + 3 ages + 3 presence masks
PRIMARY_LAG = 180
SENSITIVITY_LAG = 270


def ordinal(value):
    text = str(value).replace("-", "")[:8]
    return date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()


def finite(value):
    return value is not None and isinstance(value, (int, float)) and np.isfinite(value)


def consolidate(rows, fields, lag):
    """Ambiguous revised fields are masked; never choose the last JSON row."""
    groups = {}
    for row in rows:
        end = int(row["end_date"])
        ordinal(end)  # fail on corrupt dates rather than quietly realign
        groups.setdefault(end, []).append(row)
    result, conflicts, announced = [], 0, 0
    for end, group in sorted(groups.items()):
        available = []
        for row in group:
            actual = [ordinal(row[k]) for k in ("ann_date", "f_ann_date") if row.get(k)]
            announced += bool(actual)
            # Sensitivity adds 90 days even when genuine announcement dates exist.
            available.append(max(actual) + lag - PRIMARY_LAG if actual else ordinal(end) + lag)
        values = {}
        for field in fields:
            candidates = {float(row[field]) for row in group if finite(row.get(field))}
            if len(candidates) == 1 and all(finite(row.get(field)) for row in group):
                values[field] = candidates.pop()
            else:
                values[field] = None
                conflicts += len(candidates) > 1
        result.append({"end": end, "end_ordinal": ordinal(end),
                       "available": max(max(available), ordinal(end)), "values": values})
    return result, {"rows": len(rows), "periods": len(groups), "duplicate_rows": len(rows)-len(groups),
                    "conflicting_selected_fields_masked": conflicts, "rows_with_announcement_date": announced}


def financial_block(days, records, fields):
    """Latest available period and same-quarter prior-year change at each cutoff."""
    n = len(days)
    values = np.zeros((n, 8), np.float32)
    valid = np.zeros((n, 8), bool)
    ages = np.zeros(n, np.float32)
    present = np.zeros(n, np.float32)
    by_end = {r["end"]: r for r in records}
    # Sorting by period means later periods overwrite older ones only when available.
    for current in records:
        mask = days >= current["available"]
        if not mask.any():
            continue
        values[mask] = 0
        valid[mask] = False
        present[mask] = 1
        ages[mask] = np.minimum((days[mask] - current["end_ordinal"]) / 365.25, 5)
        previous = by_end.get(current["end"] - 10000)
        for j, field in enumerate(fields):
            now = current["values"][field]
            if now is None:
                continue
            unit = 1.0 if field == "basic_eps" else 1e8
            values[mask, j] = np.sign(now) * np.log1p(abs(now) / unit)
            valid[mask, j] = True
            if previous is None or previous["values"][field] is None:
                continue
            before = previous["values"][field]
            both = mask & (days >= previous["available"])
            values[both, j+4] = 2 * (now-before) / max(abs(now)+abs(before), unit*1e-6)
            valid[both, j+4] = True
    return values, valid, ages, present


def normalize_reports(values, valid, training_endpoints, fitted=None):
    """Fit observed values at training input cutoffs only; zero-fill after scaling."""
    if fitted is None:
        selected = values[training_endpoints].astype(np.float64)
        mask = valid[training_endpoints]
        counts = mask.sum(axis=0)
        mean = (selected * mask).sum(axis=0) / np.maximum(counts, 1)
        variance = (np.square(selected-mean) * mask).sum(axis=0) / np.maximum(counts, 1)
        std = np.maximum(np.sqrt(variance), 1e-6)
        fitted = {"mean": mean.tolist(), "std": std.tolist(), "observed_training_counts": counts.tolist()}
    mean, std = np.array(fitted["mean"]), np.array(fitted["std"])
    transformed = np.clip((values.astype(np.float64)-mean)/std, -10, 10)
    transformed[~valid] = 0
    return transformed.astype(np.float32), fitted


def add_reports(store, package_dir, cache_dir):
    price_root = package_dir / "data/numerical/HS300_time_series"
    calendars = {}
    for p in price_root.rglob("*.csv"):
        with p.open(encoding="utf-8-sig", newline="") as stream:
            days = [r["Date"][:10] for r in csv.DictReader(stream)]
        calendars[_code(str(p))] = np.array([ordinal(d) for d in days if "2020-03-31" <= d <= "2025-03-31"], np.int32)
    lengths = [len(calendars[c]) for c in store.codes]
    store.offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    assert store.offsets[-1] == len(store.close)
    store.input_dates_ordinal = np.concatenate([calendars[c] for c in store.codes])
    sources = {}
    rows_by_stock = {}
    for p in sorted((package_dir / "data/tabular/hs300_tabular").rglob("*.jsonl")):
        if p.stem not in FIELDS:
            continue
        sources[p.relative_to(package_dir).as_posix()] = sha256_file(p)
        rows = []
        with p.open(encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                for row in value if isinstance(value, list) else [value]:
                    # Keep only selected fields; full financial tables have >100 columns.
                    keys = (*FIELDS[p.stem], "end_date", "ann_date", "f_ann_date")
                    rows.append({k: row[k] for k in keys if k in row})
        rows_by_stock.setdefault(_code(str(p)), {})[p.stem] = rows
    total = len(store.close)
    store.reports = {}
    audit, stats = {}, None
    for lag in (PRIMARY_LAG, SENSITIVITY_LAG):
        values = np.zeros((total, 24), np.float32)
        valid = np.zeros((total, 24), bool)
        age_present = np.zeros((total, 6), np.float32)
        for stock, code in enumerate(store.codes):
            section = slice(store.offsets[stock], store.offsets[stock+1])
            for k, (kind, fields) in enumerate(FIELDS.items()):
                records, counts = consolidate(rows_by_stock.get(code, {}).get(kind, []), fields, lag)
                if lag == PRIMARY_LAG:
                    for key, value in counts.items():
                        audit[f"{kind}/{key}"] = audit.get(f"{kind}/{key}", 0) + value
                block, mask, age, present = financial_block(calendars[code], records, fields)
                values[section, k*8:(k+1)*8] = block
                valid[section, k*8:(k+1)*8] = mask
                age_present[section, k] = age
                age_present[section, k+3] = present
        normalized, stats = normalize_reports(values, valid, store.train.starts+95, stats)
        path = cache_dir / f"reports_lag{lag}.npy"
        output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(total, REPORT_DIM))
        output[:, :24], output[:, 24:48], output[:, 48:] = normalized, valid, age_present
        output.flush()
        del output, values, valid, age_present, normalized
        store.reports[lag] = np.load(path, mmap_mode="r")
    del rows_by_stock
    gc.collect()
    store.report_metadata = {
        "fields": {k: list(v) for k, v in FIELDS.items()}, "dimension": REPORT_DIM,
        "primary_lag_calendar_days": PRIMARY_LAG, "sensitivity_lag_calendar_days": SENSITIVITY_LAG,
        "availability": "max available date of duplicate records; actual announcement if present, else end_date + lag; available <= input last day",
        "vintage_limit": "No original release vintages supplied; conservative lag cannot rule out retrospective revisions. Conflicting duplicate fields are masked.",
        "changes": "same-quarter prior-year symmetric change 2*(current-prior)/(abs(current)+abs(prior)); both must be available",
        "normalization": "observed training input endpoints only; missing=0 with explicit masks; clip z scores to [-10,10]; delayed sensitivity reuses primary training statistics",
        "statistics": stats, "audit": audit,
        "files": len(sources), "file_manifest_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "sensitivity": "Frozen primary weights, switch report availability to 270 days; no retraining or delay selection",
    }
    write_json(cache_dir / "report_manifest.json", sources)
    write_json(cache_dir / "report_protocol.json", store.report_metadata)


def batch(store, samples, rows, news, reports, lag=PRIMARY_LAG, news_rows=None, report_rows=None, **flags):
    item = store.batch(samples, rows, "stationary", news=news, news_rows=news_rows)
    if reports:
        selected = rows if report_rows is None else report_rows
        features = np.asarray(store.reports[lag][samples.starts[selected]+95]).copy()
        item["reports"] = torch.from_numpy(features)
        item["has_report"] = torch.from_numpy(features[:, -3:].any(axis=1, keepdims=True).astype(np.float32))
    item.update(flags)
    return item


def load_holdout(store, package_dir):
    """Use unchanged split implementation, without reloading the large news matrix."""
    from types import SimpleNamespace
    class SplitView(SimpleNamespace):
        def __len__(self):
            return len(self.stock_index)
    with np.load(package_dir / "data/val.npz", allow_pickle=False) as z:
        ds = SplitView(**{key: z[key] for key in z.files})
    ds.stock_codes = ds.stock_codes.tolist()
    ds.price_days = {}
    for i, code in enumerate(store.codes):
        days = store.input_dates_ordinal[store.offsets[i]:store.offsets[i+1]]
        ds.price_days[code] = np.array([int(date.fromordinal(int(d)).strftime("%Y%m%d")) for d in days])
    split = make_validation_split(ds)
    idx = split.holdout_indices
    assert ds.stock_codes == store.codes
    assert len(idx) == store.metadata["split"]["holdout_samples"]
    assert split.dev_target_end_date < split.holdout_start_date
    holdout = Samples(store.offsets[ds.stock_index[idx]]+ds.input_start[idx], ds.y[idx].astype(np.float32),
                      ds.stock_index[idx].copy(), ds.target_start_date[idx].copy())
    return holdout
