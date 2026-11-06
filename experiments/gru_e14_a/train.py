"""Train E14-A: E10-B with field-level financial missingness masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.gru_e10_a.train import REPORT_FIELDS, REPORT_KINDS, _date, _signed_log
from experiments.gru_e10_b import train as base_train
from experiments.gru_e14_a.model import build_model

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class FinancialIndex:
    """Lagged snapshots: 12 normalized values, 12 field masks, 3 kind masks, age."""

    def __init__(self, dataset, report_lag_days=30, feature_stats=None):
        if report_lag_days < 0:
            raise ValueError("report_lag_days must be non-negative")
        self.report_lag_days = int(report_lag_days)
        self.by_stock = {}
        for code in dataset.stock_codes:
            records = dataset._reports(code)
            dates = sorted({int(row["end_date"]) for kind in REPORT_KINDS
                            for row in records.get(kind, [])})
            snapshots = []
            for date in dates:
                values, field_present = [], []
                for kind, field in REPORT_FIELDS:
                    rows = [row for row in records.get(kind, [])
                            if int(row["end_date"]) == date]
                    field_present.append(float(bool(rows and rows[-1].get(field) is not None)) )
                    values.append(_signed_log(rows[-1].get(field)) if rows else 0.0)
                kind_present = [float(any(int(row["end_date"]) == date
                                         for row in records.get(kind, [])))
                                for kind in REPORT_KINDS]
                snapshots.append(values + field_present + kind_present)
            values = np.asarray(snapshots, dtype=np.float32).reshape(-1, 27)
            if feature_stats is not None and len(values):
                mean, std = feature_stats
                values[:, :12] = (values[:, :12] - mean) / std
            self.by_stock[code] = (
                np.asarray(dates, dtype=np.int32),
                np.asarray([_date(date).toordinal() for date in dates], dtype=np.int64),
                values,
            )


def fit_feature_stats(dataset, report_lag_days):
    raw = FinancialIndex(dataset, report_lag_days)
    pieces = []
    for stock_index, code in enumerate(dataset.stock_codes):
        _, report_ordinals, report_values = raw.by_stock[code]
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
    std = np.maximum(values.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


def add_financial_batch(dataset, indices, batch, financial_index):
    indices = np.asarray(indices, dtype=np.int64)
    values = np.zeros((len(indices), 96, 28), dtype=np.float32)
    stock_indices = dataset.stock_index[indices]
    for stock in np.unique(stock_indices):
        rows = np.flatnonzero(stock_indices == stock)
        code = dataset.stock_codes[int(stock)]
        report_dates, report_ordinals, report_values = financial_index.by_stock[code]
        if not len(report_dates):
            continue
        starts = dataset.input_start[indices[rows]].astype(np.int64)
        input_days = dataset.price_days[code][starts + 95]
        input_ordinals = np.asarray([_date(day).toordinal() for day in input_days], dtype=np.int64)
        positions = np.searchsorted(report_ordinals,
                                    input_ordinals - financial_index.report_lag_days,
                                    side="right") - 1
        valid = positions >= 0
        for row, position, input_ordinal in zip(rows[valid], positions[valid], input_ordinals[valid]):
            values[row, :, :27] = report_values[position]
            age = max(int(input_ordinal) - int(report_ordinals[position]), 0)
            values[row, :, 27] = np.log1p(float(age))
    result = dict(batch)
    import torch
    result["financial"] = torch.as_tensor(values, dtype=torch.float32,
                                           device=batch["numerical"].device)
    return result


def _rewrite_outputs(args, result):
    checkpoint = base_train.load_checkpoint(args.checkpoint, "cpu")
    checkpoint["experiment"] = "E14_A_financial_field_masks"
    checkpoint["financial_input_size"] = 28
    checkpoint["feature_schema"] = {
        "normalized_signed_log_values": 12,
        "field_presence_masks": 12,
        "report_kind_presence_masks": 3,
        "log_calendar_age": 1,
    }
    import torch
    torch.save(checkpoint, args.checkpoint)
    result["experiment"] = "E14_A_financial_field_masks"
    result["financial_input_size"] = 28
    result["feature_schema"] = checkpoint["feature_schema"]
    result["feature_normalization"] = (
        "train-only mean/std for 12 signed-log value fields; "
        "12 field masks, 3 report-kind masks and age unchanged"
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run(args):
    # The established E10-B loop is reused with this module's feature builder
    # and model. Its checkpoint/result paths are redirected to E14-A.
    replacements = {
        "FinancialIndex": FinancialIndex,
        "fit_feature_stats": fit_feature_stats,
        "add_financial_batch": add_financial_batch,
        "build_model": build_model,
    }
    originals = {name: getattr(base_train, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base_train, name, value)
        result = base_train.run(args)
    finally:
        for name, value in originals.items():
            setattr(base_train, name, value)
    return _rewrite_outputs(args, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--market-checkpoint", type=Path, default=ROOT / "experiments/gru_e3/checkpoints/best.pt")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints/best.pt")
    parser.add_argument("--output", type=Path, default=HERE / "result.json")
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
