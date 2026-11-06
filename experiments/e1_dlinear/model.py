"""Close-only DLinear with the repository's 96 -> 24 residual interface."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LOOKBACK = 96
HORIZON = 24


def prepare_batch(samples: list[dict], device="cpu") -> dict[str, torch.Tensor]:
    """Use supplied training-period statistics; inference samples need no y."""
    if not samples:
        raise ValueError("samples must not be empty")
    x = np.stack([np.asarray(s["numerical"], dtype=np.float32)[:, 3] for s in samples])
    mean = np.asarray([s["close_mean"] for s in samples], dtype=np.float32)[:, None]
    std = np.asarray([s["close_std"] for s in samples], dtype=np.float32)[:, None]
    x = (x - mean) / np.maximum(std, np.float32(1e-6))
    if x.shape[1:] != (LOOKBACK,) or not np.isfinite(x).all():
        raise ValueError("expected finite Close inputs of shape [B, 96]")
    batch = {"close": torch.as_tensor(x, device=device)}
    has_y = ["y" in s for s in samples]
    if any(has_y) and not all(has_y):
        raise ValueError("a batch cannot mix labeled and unlabeled samples")
    if all(has_y):
        y = np.stack([s["y"] for s in samples]).astype(np.float32)
        if y.shape != (len(samples), HORIZON) or not np.isfinite(y).all():
            raise ValueError("expected finite targets of shape [B, 24]")
        batch["y"] = torch.as_tensor(y, device=device)
    return batch


class DLinear(nn.Module):
    def __init__(self, moving_average: int = 25, center_last: bool = False):
        super().__init__()
        if moving_average <= 0 or moving_average % 2 == 0 or moving_average > LOOKBACK:
            raise ValueError("moving_average must be an odd integer in [1, 96]")
        self.moving_average = moving_average
        self.center_last = center_last
        self.trend_head = nn.Linear(LOOKBACK, HORIZON)
        self.remainder_head = nn.Linear(LOOKBACK, HORIZON)
        # Begin at the E0 last-value baseline, then learn the residual correction.
        for layer in (self.trend_head, self.remainder_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = (self.moving_average - 1) // 2
        padded = F.pad(x.unsqueeze(1), (padding, padding), mode="replicate")
        trend = F.avg_pool1d(padded, self.moving_average, stride=1).squeeze(1)
        return trend, x - trend

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["close"]
        if x.ndim != 2 or x.shape[1] != LOOKBACK:
            raise ValueError("close must have shape [B, 96]")
        last = x[:, -1:]
        # Optional E1 variant: learn changes relative to the observed endpoint.
        # Both subtraction and restoration use only the historical input.
        inputs = x - last if self.center_last else x
        trend, remainder = self.decompose(inputs)
        delta = self.trend_head(trend) + self.remainder_head(remainder)
        return last + delta


def build_model(config: dict | None = None) -> DLinear:
    return DLinear(**(config or {}))


def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = prediction - target
    return error.abs().mean() + 0.2 * error.square().mean()
