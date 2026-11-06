"""Train E3c with latest available financial reports and evaluate holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gru_e2.model import build_optimizer, compute_loss
from experiments.gru_e2.train import set_seed
from experiments.gru_e3.train import numeric_batch as market_batch
from experiments.gru_e8.model import build_model
from experiments.validation_split import make_validation_split
from hs300_dataset import HS300Dataset

REPORT_KINDS = ("income", "balance_sheet", "cashflow")
REPORT_FIELDS = (
    ("income", "total_revenue"), ("income", "operate_profit"),
    ("income", "n_income"), ("income", "basic_eps"),
    ("balance_sheet", "total_assets"), ("balance_sheet", "total_liab"),
    ("balance_sheet", "total_hldr_eqy_inc_min_int"), ("balance_sheet", "money_cap"),
    ("cashflow", "n_cashflow_act"), ("cashflow", "n_cashflow_inv_act"),
    ("cashflow", "n_cash_flows_fnc_act"), ("cashflow", "free_cashflow"),
)


def _signed_log(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(np.sign(value) * np.log1p(abs(value)))


class FinancialIndex:
    """Latest report snapshots indexed by stock and report end date."""

    def __init__(self, dataset: HS300Dataset):
        self.by_stock = {}
        for code in dataset.stock_codes:
            records = dataset._reports(code)
            dates = sorted({int(row["end_date"]) for kind in REPORT_KINDS for row in records.get(kind, [])})
            snapshots = []
            for date in dates:
                vector = []
                for kind, field in REPORT_FIELDS:
                    rows = [row for row in records.get(kind, []) if int(row["end_date"]) == date]
                    vector.append(_signed_log(rows[-1].get(field)) if rows else 0.0)
                present = [float(any(int(row["end_date"]) == date for row in records.get(kind, []))) for kind in REPORT_KINDS]
                snapshots.append(vector + present)
            self.by_stock[code] = (
                np.asarray(dates, dtype=np.int32),
                np.asarray(snapshots, dtype=np.float32).reshape(-1, 15),
            )


def add_financial_batch(dataset, indices, batch, financial_index: FinancialIndex):
    indices = np.asarray(indices, dtype=np.int64)
    values = np.zeros((len(indices), 96, 16), dtype=np.float32)
    stock_indices = dataset.stock_index[indices]
    offsets = np.arange(96, dtype=np.int64)
    for stock in np.unique(stock_indices):
        rows = np.flatnonzero(stock_indices == stock)
        code = dataset.stock_codes[int(stock)]
        report_dates, report_values = financial_index.by_stock[code]
        if len(report_dates) == 0:
            continue
        starts = dataset.input_start[indices[rows]].astype(np.int64)
        input_days = dataset.price_days[code][starts + 95]
        positions = np.searchsorted(report_dates, input_days, side="right") - 1
        valid = positions >= 0
        valid_rows = rows[valid]
        valid_days = input_days[valid]
        for row, position, input_day in zip(valid_rows, positions[valid], valid_days):
            age = max(int(input_day) - int(report_dates[position]), 0)
            snapshot = report_values[position]
            values[row, :, :15] = snapshot
            values[row, :, 15] = np.log1p(float(age))
    result = dict(batch)
    result["financial"] = torch.as_tensor(values, dtype=torch.float32, device=batch["numerical"].device)
    return result


def make_batch(dataset, indices, financial_index, device):
    market = market_batch(dataset, indices, device, 4, "relative", "standardized_close")
    return add_financial_batch(dataset, indices, market, financial_index)


def evaluate(dataset, indices, model, financial_index, batch_size, device):
    model.eval(); absolute = squared = 0.0; count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = make_batch(dataset, indices[start:start + batch_size], financial_index, device)
            error = model(batch) - batch["y"]
            absolute += error.abs().sum().item(); squared += error.square().sum().item(); count += error.numel()
    return {"mae": absolute / count, "mse": squared / count}


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    set_seed(args.seed); device = torch.device(args.device)
    train = None if args.evaluate_holdout else HS300Dataset(args.package_dir, "train")
    val = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(val, args.holdout_fraction)
        val_financial = FinancialIndex(val)
        if args.evaluate_holdout:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            if int(checkpoint.get("epoch", 0)) < 1: raise ValueError("holdout requires trained checkpoint")
            model = build_model(checkpoint["config"]).to(device); model.load_state_dict(checkpoint["model_state"])
            result = {"experiment": "E3c_E8_financial_GRU", "checkpoint_epoch": checkpoint["epoch"], "config": checkpoint["config"], "holdout": evaluate(val, split.holdout_indices, model, val_financial, args.batch_size, device)}
        else:
            train_financial = FinancialIndex(train)
            config = {"hidden_size": args.hidden_size, "horizon": 24, "financial_input_size": 16, "financial_dim": args.financial_dim, "dropout": args.dropout}
            model = build_model(config).to(device); optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
            indices = np.arange(len(train), dtype=np.int64); initial = evaluate(val, split.dev_indices, model, val_financial, args.batch_size, device)
            history = [{"epoch": 0, **initial}]; best_mae = float("inf"); best_epoch = None; wait = 0
            progress = args.output.with_suffix(".progress.json"); progress.write_text(json.dumps({"experiment": "E3c_E8_financial_GRU", "history": history}, indent=2) + "\n")
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(f"epoch 0 (E0 reference): dev_MAE={initial['mae']:.6f} dev_MSE={initial['mse']:.6f}", flush=True)
            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(indices); model.train(); weighted = 0.0
                total = (len(indices) + args.batch_size - 1) // args.batch_size
                for step, start in enumerate(range(0, len(indices), args.batch_size), 1):
                    batch = make_batch(train, indices[start:start + args.batch_size], train_financial, device)
                    optimizer.zero_grad(set_to_none=True); loss = compute_loss(model(batch), batch["y"])
                    if not torch.isfinite(loss): raise ValueError("non-finite loss")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); weighted += loss.item() * len(batch["y"])
                    if step == 1 or step % 100 == 0 or step == total: print(f"epoch {epoch} batch {step}/{total} loss={loss.item():.6f}", flush=True)
                dev = evaluate(val, split.dev_indices, model, val_financial, args.batch_size, device); record = {"epoch": epoch, "train_loss": weighted / len(indices), **dev}; history.append(record)
                progress.write_text(json.dumps({"experiment": "E3c_E8_financial_GRU", "history": history}, indent=2) + "\n"); print(f"epoch {epoch}: dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f}", flush=True)
                if dev["mae"] < best_mae:
                    best_mae, best_epoch, wait = dev["mae"], epoch, 0; torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch, "experiment": "E3c_E8_financial_GRU", "holdout_fraction": args.holdout_fraction}, args.checkpoint)
                else:
                    wait += 1
                    if wait >= args.patience: break
            if best_epoch is None: raise RuntimeError("no trained epoch completed")
            selected = torch.load(args.checkpoint, map_location=device, weights_only=True); model.load_state_dict(selected["model_state"])
            best_record = next(item for item in history if item["epoch"] == best_epoch); holdout = evaluate(val, split.holdout_indices, model, val_financial, args.batch_size, device); progress.unlink(missing_ok=True)
            result = {"experiment": "E3c_E8_financial_GRU", "seed": args.seed, "config": config, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "holdout_fraction": args.holdout_fraction, "e0_development": initial, "best_epoch": best_epoch, "best_development": best_record, "holdout": {"checkpoint_epoch": best_epoch, **holdout}, "mae_delta_vs_e0": best_record["mae"] - initial["mae"], "history": history[1:]}
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2), flush=True); return result
    finally:
        if train is not None: train.close()
        val.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--package-dir", type=Path, default=ROOT); parser.add_argument("--checkpoint", type=Path, default=Path("experiments/gru_e8/checkpoints/best.pt")); parser.add_argument("--output", type=Path, default=Path("experiments/gru_e8/result.json")); parser.add_argument("--evaluate-holdout", action="store_true"); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--patience", type=int, default=6); parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--hidden-size", type=int, default=64); parser.add_argument("--financial-dim", type=int, default=16); parser.add_argument("--dropout", type=float, default=0.0); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-5); parser.add_argument("--holdout-fraction", type=float, default=0.2); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cpu"); run(parser.parse_args())


if __name__ == "__main__": main()
