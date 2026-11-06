"""E8: minimal market backbone + financial-report branch (no news).

Financial reports have no announcement-date field (only report period
`end_date`), so a conservative availability lag is applied: a report is only
usable once `end_date + report_lag_days <= input_end`.  Default lag is 120 days
(annual-report worst case); tune `--report-lag-days` for a sensitivity check.

Feature design (docs/experiment_route.md section 6, E8): key fields + change
rates + missing masks + report age.  All value features are scale-invariant
(growth rates and financial ratios) so raw levels do not leak cross-stock
magnitude (a bank vs. a small firm).  Change rates are quarter-over-quarter
(vs. the immediately preceding report); YoY is a later refinement.

Feature vector (FIN_DIM = 22):
  0  rev_qoq        revenue growth QoQ
  1  ni_qoq         net income growth QoQ
  2  op_qoq         operating profit growth QoQ
  3  ni_margin      n_income / total_revenue
  4  op_margin      operate_profit / total_revenue
  5  gross_margin   (total_revenue - total_cogs) / total_revenue
  6  asset_qoq      total_assets growth QoQ
  7  debt_ratio     total_liab / total_assets
  8  eq_ratio       total_hldr_eqy_exc_min_int / total_assets
  9  current_ratio  total_cur_assets / total_cur_liab
  10 roa            n_income / total_assets
  11 ocf_qoq        operating cash flow growth QoQ
  12 ocf_margin     n_cashflow_act / total_revenue
  13 ocf_to_ni      n_cashflow_act / n_income
  14 fcf_qoq        free cash flow growth QoQ
  15 fcf_margin     free_cashflow / total_revenue
  16 age_income     (input_end - latest income end_date) / 365
  17 age_balance    (input_end - latest balance end_date) / 365
  18 age_cashflow   (input_end - latest cashflow end_date) / 365
  19 has_income     0/1
  20 has_balance    0/1
  21 has_cashflow   0/1

Model (residual prediction, mirrors E7):
    delta_market = Linear(z_96)                       [B, 24]   (tiny)
    fin_z        = MLP(feat) * has_fin                [B, 16]
    delta_fin    = Fusion([fin_z; has_fin]) * has_fin [B, 24]
    y_hat        = z_96 + delta_market + delta_fin

`use_fin=False` in forward() ablates the whole financial branch at eval time
(R3 financial-zeroing check in docs/experiment_route.md section 8).

Run from the repository root:

    python -m experiments.e8.e8_financials --epochs 20 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split

# Unified settings (docs/experiment_route.md sections 2 and 4).
SEQ_LEN = 96
PRED_LEN = 24
FIN_DIM = 22
FIN_HIDDEN = 64
FIN_OUT = 16
FUSION_HIDDEN = 32
DROPOUT = 0.1
HORIZONS = (1, 6, 12, 24)
CLOSE_COLUMN = 3  # PRICE_COLUMNS: Open, High, Low, Close, Volume, Dividends, Stock Splits
REPORT_LAG_DAYS = 120  # conservative disclosure lag; tune via --report-lag-days


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Feature helpers
# --------------------------------------------------------------------------- #
def _parse_date(value) -> object:
    if isinstance(value, str):
        return datetime.strptime(value, "%Y%m%d").date()
    return datetime.strptime(str(int(value)), "%Y%m%d").date()


def _f(record: dict | None, field: str) -> float | None:
    if record is None:
        return None
    value = record.get(field)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _clip(value: float, low: float = -10.0, high: float = 10.0) -> float:
    if value != value or value == float("inf") or value == float("-inf"):
        return 0.0
    return float(max(low, min(high, value)))


def _ratio(num: float | None, den: float | None) -> float:
    if num is None or den is None or den == 0:
        return 0.0
    return _clip(num / den)


def _growth(cur: float | None, prev: float | None) -> float:
    if cur is None or prev is None or prev == 0:
        return 0.0
    return _clip((cur - prev) / abs(prev))


def _age(record: dict | None, input_end: object) -> float:
    if record is None:
        return 0.0
    days = (input_end - _parse_date(record.get("end_date"))).days
    return _clip(days / 365.0, 0.0, 3.0)


def _latest_available(records: list[dict], input_end: object, lag_days: int) -> tuple[dict | None, dict | None]:
    """Return (latest, previous) reports whose end_date + lag <= input_end.

    `records` is sorted newest-first by end_date.  The previous report is the
    one immediately after `latest` in that order; because it is older it is
    automatically available whenever `latest` is.
    """
    for i, record in enumerate(records):
        end_date = _parse_date(record.get("end_date"))
        if end_date + timedelta(days=lag_days) <= input_end:
            previous = records[i + 1] if i + 1 < len(records) else None
            return record, previous
    return None, None


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def _load_reports(dataset: HS300Dataset, cache: dict, code: str) -> dict[str, list[dict]]:
    if code in cache:
        return cache[code]
    result: dict[str, list[dict]] = {}
    for kind, member in dataset.report_members.get(code, {}).items():
        records: list[dict] = []
        with member.open("r", encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                records.extend(value if isinstance(value, list) else [value])
        records.sort(key=lambda r: str(r.get("end_date", "")), reverse=True)
        result[kind] = records
    cache[code] = result
    return result


def _build_sample_feat(reports: dict[str, list[dict]], input_end: object, lag_days: int) -> tuple[np.ndarray, float]:
    feat = np.zeros(FIN_DIM, dtype=np.float32)
    inc_latest, inc_prev = _latest_available(reports.get("income", []), input_end, lag_days)
    bal_latest, bal_prev = _latest_available(reports.get("balance_sheet", []), input_end, lag_days)
    csh_latest, csh_prev = _latest_available(reports.get("cashflow", []), input_end, lag_days)

    has_income = 1.0 if inc_latest is not None else 0.0
    has_balance = 1.0 if bal_latest is not None else 0.0
    has_cashflow = 1.0 if csh_latest is not None else 0.0

    if inc_latest is not None:
        feat[0] = _growth(_f(inc_latest, "total_revenue"), _f(inc_prev, "total_revenue"))
        feat[1] = _growth(_f(inc_latest, "n_income"), _f(inc_prev, "n_income"))
        feat[2] = _growth(_f(inc_latest, "operate_profit"), _f(inc_prev, "operate_profit"))
        feat[3] = _ratio(_f(inc_latest, "n_income"), _f(inc_latest, "total_revenue"))
        feat[4] = _ratio(_f(inc_latest, "operate_profit"), _f(inc_latest, "total_revenue"))
        revenue = _f(inc_latest, "total_revenue")
        cogs = _f(inc_latest, "total_cogs")
        feat[5] = _ratio(revenue - cogs if revenue is not None and cogs is not None else None, revenue)
        feat[16] = _age(inc_latest, input_end)

    if bal_latest is not None:
        feat[6] = _growth(_f(bal_latest, "total_assets"), _f(bal_prev, "total_assets"))
        feat[7] = _ratio(_f(bal_latest, "total_liab"), _f(bal_latest, "total_assets"))
        feat[8] = _ratio(_f(bal_latest, "total_hldr_eqy_exc_min_int"), _f(bal_latest, "total_assets"))
        feat[9] = _ratio(_f(bal_latest, "total_cur_assets"), _f(bal_latest, "total_cur_liab"))
        feat[10] = _ratio(_f(inc_latest, "n_income"), _f(bal_latest, "total_assets"))
        feat[17] = _age(bal_latest, input_end)

    if csh_latest is not None:
        feat[11] = _growth(_f(csh_latest, "n_cashflow_act"), _f(csh_prev, "n_cashflow_act"))
        feat[12] = _ratio(_f(csh_latest, "n_cashflow_act"), _f(inc_latest, "total_revenue"))
        feat[13] = _ratio(_f(csh_latest, "n_cashflow_act"), _f(inc_latest, "n_income"))
        feat[14] = _growth(_f(csh_latest, "free_cashflow"), _f(csh_prev, "free_cashflow"))
        feat[15] = _ratio(_f(csh_latest, "free_cashflow"), _f(inc_latest, "total_revenue"))
        feat[18] = _age(csh_latest, input_end)

    feat[19] = has_income
    feat[20] = has_balance
    feat[21] = has_cashflow
    has_fin = 1.0 if (has_income or has_balance or has_cashflow) else 0.0
    return feat, has_fin


def extract_features(dataset: HS300Dataset, indices: np.ndarray, lag_days: int) -> dict[str, np.ndarray]:
    n = len(indices)
    z96 = np.empty((n, 1), dtype=np.float32)
    feat = np.empty((n, FIN_DIM), dtype=np.float32)
    has_fin = np.empty((n, 1), dtype=np.float32)
    y = np.empty((n, PRED_LEN), dtype=np.float32)
    report_cache: dict[str, dict[str, list[dict]]] = {}

    for k, idx in enumerate(indices):
        si = int(dataset.stock_index[idx])
        code = dataset.stock_codes[si]
        start = int(dataset.input_start[idx])
        days = dataset.price_days[code][start : start + SEQ_LEN]
        close = dataset.price_values[code][start : start + SEQ_LEN, CLOSE_COLUMN]
        mean = float(dataset.close_mean[si])
        std = max(float(dataset.close_std[si]), 1e-6)
        z96[k, 0] = (close[-1] - mean) / std

        input_end = _parse_date(int(days[-1]))
        reports = _load_reports(dataset, report_cache, code)
        feat[k], has_fin[k, 0] = _build_sample_feat(reports, input_end, lag_days)

        y[k] = dataset.y[idx]

        if (k + 1) % 50000 == 0:
            print(f"    extracted {k + 1}/{n}", flush=True)

    return {"z96": z96, "feat": feat, "has_fin": has_fin, "y": y}


def _cache_path(package_dir: Path, lag_days: int, split: str) -> Path:
    return package_dir / "experiments" / "e8" / "cache" / f"{split}_lag{lag_days}.npz"


def load_or_extract(
    package_dir: Path, split: str, no_cache: bool, lag_days: int, max_samples: int | None = None
) -> dict[str, np.ndarray]:
    dataset = HS300Dataset(package_dir, split)
    if max_samples is not None:
        try:
            print(f"[e8] extracting first {max_samples} {split} samples ...", flush=True)
            return extract_features(dataset, np.arange(min(max_samples, len(dataset))), lag_days)
        finally:
            dataset.close()

    path = _cache_path(package_dir, lag_days, split)
    if path.exists() and not no_cache:
        dataset.close()
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    print(f"[e8] extracting {split} features (lag {lag_days}d) ...", flush=True)
    try:
        features = extract_features(dataset, np.arange(len(dataset)), lag_days)
    finally:
        dataset.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **features)
    print(f"[e8] cached {split} features -> {path}", flush=True)
    return features


def _stack(features: dict[str, np.ndarray], indices: np.ndarray) -> tuple[torch.Tensor, ...]:
    z96 = torch.from_numpy(features["z96"][indices])
    feat = torch.from_numpy(features["feat"][indices])
    has_fin = torch.from_numpy(features["has_fin"][indices])
    y = torch.from_numpy(features["y"][indices])
    return z96, feat, has_fin, y


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class E8Financials(nn.Module):
    """Last-value residual head plus a gated financial-report branch."""

    def __init__(self) -> None:
        super().__init__()
        self.market_head = nn.Linear(1, PRED_LEN)
        self.fin_mlp = nn.Sequential(
            nn.Linear(FIN_DIM, FIN_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FIN_HIDDEN, FIN_OUT),
        )
        self.fusion = nn.Sequential(
            nn.Linear(FIN_OUT + 1, FUSION_HIDDEN),
            nn.GELU(),
            nn.Linear(FUSION_HIDDEN, PRED_LEN),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.market_head.weight)
        nn.init.zeros_(self.market_head.bias)
        for linear in (self.fin_mlp[0], self.fin_mlp[3], self.fusion[0], self.fusion[2]):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        z96: torch.Tensor,
        feat: torch.Tensor,
        has_fin: torch.Tensor,
        use_fin: bool = True,
    ) -> torch.Tensor:
        """z96 [B,1], feat [B,FIN_DIM], has_fin [B,1] -> [B,24]."""
        delta_market = self.market_head(z96)  # [B, 24]
        if not use_fin:
            return z96 + delta_market
        fin_z = self.fin_mlp(feat) * has_fin  # [B, FIN_OUT], zeroed when no reports
        fused = torch.cat([fin_z, has_fin], dim=-1)  # [B, FIN_OUT + 1]
        delta_fin = self.fusion(fused) * has_fin  # [B, 24], zeroed when no reports
        return z96 + delta_market + delta_fin


def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unified training objective: MAE + 0.2 * MSE."""
    difference = prediction - target
    return difference.abs().mean() + 0.2 * difference.square().mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    z96: torch.Tensor,
    feat: torch.Tensor,
    has_fin: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: str,
    use_fin: bool,
) -> dict:
    model.eval()
    predictions = []
    for start in range(0, z96.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        predictions.append(
            model(z96[sl].to(device), feat[sl].to(device), has_fin[sl].to(device), use_fin=use_fin).cpu()
        )
    prediction = torch.cat(predictions, dim=0)
    difference = prediction - y
    result = {
        "samples": int(z96.shape[0]),
        "mae": float(difference.abs().mean()),
        "mse": float(difference.square().mean()),
    }
    for horizon in HORIZONS:
        error = difference[:, horizon - 1]
        result[f"mae_h{horizon}"] = float(error.abs().mean())
        result[f"mse_h{horizon}"] = float(error.square().mean())
    return result


def train_and_evaluate(
    package_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    patience: int,
    device: str,
    limit_train: int | None,
    weight_decay: float = 0.0,
    no_cache: bool = False,
    limit_val: int | None = None,
    lag_days: int = REPORT_LAG_DAYS,
    checkpoint: Path | None = None,
) -> dict:
    set_seed(seed)

    train_feats = load_or_extract(package_dir, "train", no_cache, lag_days, limit_train)
    val_feats = load_or_extract(package_dir, "val", no_cache, lag_days, limit_val)

    if limit_val is None:
        val_dataset = HS300Dataset(package_dir, "val")
        split = make_validation_split(val_dataset, holdout_fraction=0.2)
        val_dataset.close()
        dev_indices = split.dev_indices
        holdout_indices = split.holdout_indices
    else:
        dev_indices = np.arange(val_feats["z96"].shape[0])
        holdout_indices = np.empty(0, dtype=np.int64)

    z_train, f_train, hf_train, y_train = _stack(train_feats, np.arange(train_feats["z96"].shape[0]))
    z_dev, f_dev, hf_dev, y_dev = _stack(val_feats, dev_indices)
    z_holdout, f_holdout, hf_holdout, y_holdout = _stack(val_feats, holdout_indices)

    model = E8Financials().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(z_train, f_train, hf_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )

    best_dev_mae = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for zb, fb, hfb, yb in loader:
            zb, fb, hfb, yb = zb.to(device), fb.to(device), hfb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(zb, fb, hfb, use_fin=True), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * zb.shape[0]

        dev = evaluate(model, z_dev, f_dev, hf_dev, y_dev, batch_size, device, use_fin=True)
        train_loss = running_loss / len(loader.dataset)
        print(
            f"epoch {epoch:3d}: train_loss={train_loss:.6f} "
            f"dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f} "
            f"({time.time() - started:.1f}s)"
        )
        if dev["mae"] < best_dev_mae:
            best_dev_mae = dev["mae"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"early stop after {epoch} epochs")
            break

    model.load_state_dict(best_state)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint)

    dev = evaluate(model, z_dev, f_dev, hf_dev, y_dev, batch_size, device, use_fin=True)
    dev_fin_off = evaluate(model, z_dev, f_dev, hf_dev, y_dev, batch_size, device, use_fin=False)
    holdout = (
        evaluate(model, z_holdout, f_holdout, hf_holdout, y_holdout, batch_size, device, use_fin=True)
        if z_holdout.shape[0] > 0
        else {}
    )

    result = {
        "experiment": "e8",
        "model": "E8Financials",
        "input": "last-value Close + 22-dim financial ratios/growth/age/masks",
        "device": device,
        "seed": seed,
        "config": {
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "fin_dim": FIN_DIM,
            "fin_hidden": FIN_HIDDEN,
            "fin_out": FIN_OUT,
            "fusion_hidden": FUSION_HIDDEN,
            "dropout": DROPOUT,
            "report_lag_days": lag_days,
            "change_rate": "QoQ (vs immediately preceding report)",
            "loss": "MAE + 0.2 * MSE",
            "optimizer": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs_run": best_epoch,
            "patience": patience,
            "params": n_params,
        },
        "train_samples": int(z_train.shape[0]),
        "dev_samples": int(z_dev.shape[0]),
        "holdout_samples": int(z_holdout.shape[0]),
        "best_epoch": best_epoch,
        "dev": dev,
        "dev_fin_off": dev_fin_off,
        "holdout": holdout,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root containing data/ (defaults to the repository root)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int, default=None, help="cap train samples for quick tests")
    parser.add_argument("--limit-val", type=int, default=None, help="cap val samples for quick tests (disables holdout)")
    parser.add_argument("--no-cache", action="store_true", help="re-extract features, ignore cache")
    parser.add_argument("--report-lag-days", type=int, default=REPORT_LAG_DAYS, help="conservative disclosure lag")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    result = train_and_evaluate(
        package_dir=args.package_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        device=args.device,
        limit_train=args.limit_train,
        weight_decay=args.weight_decay,
        no_cache=args.no_cache,
        limit_val=args.limit_val,
        lag_days=args.report_lag_days,
        checkpoint=args.checkpoint,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
