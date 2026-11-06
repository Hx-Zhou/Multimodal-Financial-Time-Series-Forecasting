"""Submission model using the E10-B market and financial-report architecture."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hs300_dataset import HS300Dataset

LOOKBACK = 96
HORIZON = 24
REPORT_LAG_DAYS = 30
REPORT_KINDS = ("income", "balance_sheet", "cashflow")
REPORT_FIELDS = (
    ("income", "total_revenue"), ("income", "operate_profit"),
    ("income", "n_income"), ("income", "basic_eps"),
    ("balance_sheet", "total_assets"), ("balance_sheet", "total_liab"),
    ("balance_sheet", "total_hldr_eqy_inc_min_int"),
    ("balance_sheet", "money_cap"),
    ("cashflow", "n_cashflow_act"), ("cashflow", "n_cashflow_inv_act"),
    ("cashflow", "n_cash_flows_fnc_act"), ("cashflow", "free_cashflow"),
)

# E10-B training-only statistics, embedded so inference does not need an
# experiment directory or an external preprocessing file.
FINANCIAL_MEAN = np.asarray(
    [21.9248638153, 15.3756666183, 15.8249187469, 0.2505151033,
     20.8675289154, 20.1657047272, 19.9151191711, 18.2669467926,
     7.8987622261, -13.1604185104, 0.7698169947, -3.5457904339],
    dtype=np.float32,
)
FINANCIAL_STD = np.asarray(
    [2.2384519577, 12.3434896467, 11.5325136187, 0.3488742709,
     7.7519626617, 7.6015686989, 7.8581881523, 7.7525219917,
     18.0909996033, 14.2250499725, 19.2387294760, 18.7582721710],
    dtype=np.float32,
)


def _date(day: int) -> dt.date:
    text = str(int(day))
    return dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _signed_log(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return float(np.sign(number) * np.log1p(abs(number)))


def _market_features(sample: dict) -> tuple[np.ndarray, float]:
    numerical = np.asarray(sample["numerical"], dtype=np.float32)
    if numerical.shape != (LOOKBACK, 7):
        raise ValueError(f"numerical must have shape [96, 7], got {numerical.shape}")
    last_raw = float(numerical[-1, 3])
    market = numerical[:, :4] / max(abs(last_raw), 1e-6) - 1.0
    close_std = max(float(sample["close_std"]), 1e-6)
    last_close = (last_raw - float(sample["close_mean"])) / close_std
    if not np.isfinite(market).all() or not np.isfinite(last_close):
        raise ValueError("non-finite market input")
    return market.astype(np.float32), float(last_close)


def _financial_features(sample: dict, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Build E10-B's delayed, latest-report snapshot and repeat it over time."""
    input_end = int(np.asarray(sample["dates"])[-1])
    available_day = _date(input_end) - dt.timedelta(days=REPORT_LAG_DAYS)
    available_code = int(available_day.strftime("%Y%m%d"))
    report_window = sample["financial"]
    dates = set()
    for kind in REPORT_KINDS:
        for row in report_window[kind]:
            try:
                end_date = int(row["end_date"])
            except (KeyError, TypeError, ValueError):
                continue
            if end_date <= available_code:
                dates.add(end_date)
    output = np.zeros((LOOKBACK, 16), dtype=np.float32)
    if not dates:
        return output
    report_date = max(dates)
    values = []
    for kind, field in REPORT_FIELDS:
        rows = [row for row in report_window[kind]
                if int(row.get("end_date", -1)) == report_date]
        values.append(_signed_log(rows[-1].get(field)) if rows else 0.0)
    present = [float(any(int(row.get("end_date", -1)) == report_date
                         for row in report_window[kind]))
               for kind in REPORT_KINDS]
    values = (np.asarray(values, dtype=np.float32) - mean) / std
    age = max((_date(input_end) - _date(report_date)).days, 0)
    snapshot = np.concatenate((values, np.asarray(present, dtype=np.float32),
                               np.asarray([np.log1p(age)], dtype=np.float32)))
    output[:] = snapshot
    if not np.isfinite(output).all():
        raise ValueError("non-finite financial input")
    return output


def prepare_batch(samples: list[dict], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Convert reader samples into E10-B market and financial tensors."""
    if not samples:
        raise ValueError("samples must not be empty")
    labels = ["y" in sample for sample in samples]
    if any(labels) and not all(labels):
        raise ValueError("a batch cannot mix labeled and unlabeled samples")
    market, last_close, financial, targets = [], [], [], []
    for sample in samples:
        market_i, last_i = _market_features(sample)
        market.append(market_i)
        last_close.append(last_i)
        financial.append(_financial_features(sample, FINANCIAL_MEAN, FINANCIAL_STD))
        if "y" in sample:
            targets.append(np.asarray(sample["y"], dtype=np.float32))
    result = {
        "numerical": torch.as_tensor(np.stack(market), dtype=torch.float32, device=device),
        "last_close": torch.as_tensor(np.asarray(last_close), dtype=torch.float32, device=device),
        "financial": torch.as_tensor(np.stack(financial), dtype=torch.float32, device=device),
    }
    if targets:
        result["y"] = torch.as_tensor(np.stack(targets), dtype=torch.float32, device=device)
    return result


class SubmissionGRU(nn.Module):
    def __init__(self, hidden_size=64, horizon=24, financial_dim=16,
                 financial_hidden=32, dropout=0.0):
        super().__init__()
        self.input_size = 4
        self.representation = "relative"
        self.prediction_mode = "standardized_close"
        self.register_buffer("financial_mean", torch.as_tensor(FINANCIAL_MEAN))
        self.register_buffer("financial_std", torch.as_tensor(FINANCIAL_STD))
        self.market_gru = nn.GRU(4, hidden_size, batch_first=True)
        self.market_head = nn.Linear(hidden_size, horizon)
        self.financial_projection = nn.Sequential(
            nn.Linear(16, financial_dim), nn.LayerNorm(financial_dim), nn.GELU())
        self.financial_gru = nn.GRU(financial_dim, financial_hidden, batch_first=True)
        self.financial_head = nn.Sequential(
            nn.Linear(hidden_size + financial_hidden, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, horizon))
        self.financial_gate = nn.Parameter(torch.zeros(horizon))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        market, financial = batch["numerical"], batch["financial"]
        if market.ndim != 3 or market.shape[1:] != (96, 4):
            raise ValueError(f"numerical must have shape [B, 96, 4], got {tuple(market.shape)}")
        if financial.ndim != 3 or financial.shape[1:] != (96, 16):
            raise ValueError(f"financial must have shape [B, 96, 16], got {tuple(financial.shape)}")
        _, market_hidden = self.market_gru(market)
        market_state = market_hidden[-1]
        base = batch["last_close"][:, None] + self.market_head(market_state)
        financial_latent = self.financial_projection(financial)
        _, financial_hidden = self.financial_gru(financial_latent)
        financial_delta = self.financial_head(
            torch.cat((market_state.detach(), financial_hidden[-1]), dim=-1))
        return base + financial_delta * self.financial_gate


def build_model(config: dict | None = None) -> nn.Module:
    config = config or {}
    return SubmissionGRU(
        hidden_size=int(config.get("hidden_size", 64)),
        horizon=int(config.get("horizon", 24)),
        financial_dim=int(config.get("financial_dim", 16)),
        financial_hidden=int(config.get("financial_hidden", 32)),
        dropout=float(config.get("dropout", 0.0)),
    )


def build_optimizer(model: nn.Module, learning_rate=3e-4, weight_decay=1e-5):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = prediction - target
    return error.abs().mean() + 0.2 * error.square().mean()


def batches(dataset: HS300Dataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        yield [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]


def train_and_validate(package_dir: Path, checkpoint: Path, epochs: int,
                       batch_size: int, device="cpu") -> None:
    train, val = HS300Dataset(package_dir, "train"), HS300Dataset(package_dir, "val")
    model, optimizer = build_model().to(device), None
    optimizer = build_optimizer(model)
    best_mae, best_state = float("inf"), None
    try:
        for epoch in range(epochs):
            model.train()
            for samples in batches(train, batch_size):
                batch = prepare_batch(samples, device)
                optimizer.zero_grad(set_to_none=True)
                compute_loss(model(batch), batch["y"]).backward()
                optimizer.step()
            model.eval(); absolute = squared = count = 0
            with torch.inference_mode():
                for samples in batches(val, batch_size):
                    batch = prepare_batch(samples, device); error = model(batch) - batch["y"]
                    absolute += float(error.abs().sum()); squared += float(error.square().sum()); count += error.numel()
            mae, mse = absolute / count, squared / count
            print(f"epoch {epoch + 1}: val_MAE={mae:.6f}, val_MSE={mse:.6f}")
            if mae < best_mae:
                best_mae = mae
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("no epoch completed")
        torch.save(best_state, checkpoint)
    finally:
        train.close(); val.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pt"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_and_validate(args.package_dir, args.checkpoint, args.epochs, args.batch_size, args.device)
