"""Single-variable GRU model for the E2 pure-price experiment.

The model consumes only the 96 standardized closing prices.  It predicts a
24-step residual from the last observed standardized close, which gives the
network a strong persistence starting point while still allowing it to learn
future changes.
"""

from __future__ import annotations

import torch
import numpy as np
from torch import nn


class SingleVariableGRU(nn.Module):
    """One-layer GRU encoder followed by a direct 24-step prediction head."""

    def __init__(self, hidden_size: int = 64, horizon: int = 24, dropout: float = 0.0):
        super().__init__()
        if hidden_size <= 0 or horizon <= 0:
            raise ValueError("hidden_size and horizon must be positive")
        self.horizon = horizon
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)
        # Start exactly at the last-value baseline. The GRU must earn any
        # residual correction instead of adding random offsets at epoch zero.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        close = batch["close"]
        if close.ndim != 3 or close.shape[-1] != 1:
            raise ValueError(f"close must have shape [B, 96, 1], got {tuple(close.shape)}")
        _, hidden = self.gru(close)
        residual = self.head(self.dropout(hidden[-1]))
        return batch["last_close"].unsqueeze(1) + residual


def prepare_batch(
    samples: list[dict], device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """Convert reader samples to the E2 tensors.

    Only Close is used.  The reader stores raw prices, so Close is converted
    with the per-stock training-period ``close_mean`` and ``close_std`` used
    to construct ``y``.
    """
    close_values = []
    last_values = []
    targets = []
    has_target = all("y" in sample for sample in samples)
    for sample in samples:
        close = sample["numerical"][:, 3].astype("float32")
        std = max(float(sample["close_std"]), 1e-6)
        standardized = (close - float(sample["close_mean"])) / std
        close_values.append(standardized[:, None])
        last_values.append(standardized[-1])
        if has_target:
            targets.append(sample["y"])

    if not samples:
        raise ValueError("samples must not be empty")
    batch = {
        "close": torch.as_tensor(np.stack(close_values), dtype=torch.float32, device=device),
        "last_close": torch.as_tensor(np.asarray(last_values), dtype=torch.float32, device=device),
    }
    if has_target:
        batch["y"] = torch.as_tensor(np.stack(targets), dtype=torch.float32, device=device)
    return batch


def build_model(config: dict | None = None) -> nn.Module:
    config = config or {}
    return SingleVariableGRU(
        hidden_size=int(config.get("hidden_size", 64)),
        horizon=int(config.get("horizon", 24)),
        dropout=float(config.get("dropout", 0.0)),
    )


def build_optimizer(model: nn.Module, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Use the route's initial MAE + 0.2*MSE objective."""
    error = prediction - target
    return error.abs().mean() + 0.2 * error.square().mean()
