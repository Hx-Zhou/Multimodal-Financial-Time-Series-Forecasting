"""OHLC or seven-variable GRU with a last-close residual forecast."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def transform_window(
    numerical: np.ndarray, close_mean: float, close_std: float, input_size: int = 7,
    representation: str = "global",
) -> np.ndarray:
    """Transform [..., 96, 7] windows using only available input information."""
    if numerical.shape[-2:] != (96, 7):
        raise ValueError(f"expected [..., 96, 7], got {numerical.shape}")
    if input_size not in (4, 7):
        raise ValueError("input_size must be 4 (OHLC) or 7 (all fields)")
    if representation not in ("global", "relative"):
        raise ValueError("representation must be global or relative")
    x = np.array(numerical[..., :input_size], dtype=np.float32, copy=True)
    std = max(float(close_std), 1e-6)
    if representation == "relative":
        last_close = np.maximum(np.abs(x[..., -1:, 3:4]), 1e-6)
        x[..., :4] = x[..., :4] / last_close - 1.0
    else:
        x[..., :4] = (x[..., :4] - float(close_mean)) / std
    if input_size == 7 and representation == "global":
        volume = np.log1p(np.maximum(x[..., 4], 0))
        volume_mean = volume.mean(axis=-1, keepdims=True)
        volume_std = volume.std(axis=-1, keepdims=True)
        x[..., 4] = (volume - volume_mean) / np.maximum(volume_std, 1e-6)
        x[..., 5] /= std
    if not np.isfinite(x).all():
        raise ValueError("non-finite numerical input")
    return x


class MultivariateGRU(nn.Module):
    def __init__(
        self, hidden_size: int = 64, horizon: int = 24, dropout: float = 0.0,
        input_size: int = 7, residual_scale: float = 1.0,
        representation: str = "global",
        prediction_mode: str = "standardized_close",
    ):
        super().__init__()
        if input_size not in (4, 7):
            raise ValueError("input_size must be 4 or 7")
        self.input_size = input_size
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = float(residual_scale)
        if representation not in ("global", "relative"):
            raise ValueError("representation must be global or relative")
        if prediction_mode not in ("standardized_close", "relative_return"):
            raise ValueError("prediction_mode must be standardized_close or relative_return")
        self.representation = representation
        self.prediction_mode = prediction_mode
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["numerical"]
        if x.ndim != 3 or x.shape[1:] != (96, self.input_size):
            raise ValueError(
                f"numerical must have shape [B, 96, {self.input_size}], got {tuple(x.shape)}"
            )
        _, hidden = self.gru(x)
        residual = self.head(self.dropout(hidden[-1])) * self.residual_scale
        if self.prediction_mode == "relative_return":
            return residual
        baseline = batch.get("last_close", x[:, -1, 3])
        return baseline[:, None] + residual


def build_model(config: dict | None = None) -> nn.Module:
    config = config or {}
    return MultivariateGRU(
        int(config.get("hidden_size", 64)), int(config.get("horizon", 24)),
        float(config.get("dropout", 0.0)), int(config.get("input_size", 7)),
        float(config.get("residual_scale", 1.0)),
        str(config.get("representation", "global")),
        str(config.get("prediction_mode", "standardized_close")),
    )


def prepare_batch(
    samples: list[dict], device: str | torch.device = "cpu", input_size: int = 7,
    representation: str = "global",
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("samples must not be empty")
    has_targets = ["y" in sample for sample in samples]
    if any(has_targets) and not all(has_targets):
        raise ValueError("batch contains a mixture of labeled and unlabeled samples")
    x = np.stack([
        transform_window(
            s["numerical"], s["close_mean"], s["close_std"], input_size, representation
        )
        for s in samples
    ])
    batch = {"numerical": torch.as_tensor(x, dtype=torch.float32, device=device)}
    if representation == "relative":
        last_close = [
            (float(s["numerical"][-1, 3]) - float(s["close_mean"]))
            / max(float(s["close_std"]), 1e-6)
            for s in samples
        ]
        batch["last_close"] = torch.as_tensor(last_close, dtype=torch.float32, device=device)
    if all(has_targets):
        batch["y"] = torch.as_tensor(np.stack([s["y"] for s in samples]), dtype=torch.float32, device=device)
    return batch
