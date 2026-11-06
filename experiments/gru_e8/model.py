"""E3c market GRU with a compact latest-financial-report branch."""

from __future__ import annotations

import torch
from torch import nn


class E3cFinancialGRU(nn.Module):
    def __init__(
        self, hidden_size: int = 64, horizon: int = 24, financial_input_size: int = 16,
        financial_dim: int = 16, dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = 4
        self.representation = "relative"
        self.prediction_mode = "standardized_close"
        self.financial_input_size = financial_input_size
        self.financial_dim = financial_dim
        self.financial_projection = nn.Sequential(
            nn.Linear(financial_input_size, financial_dim),
            nn.LayerNorm(financial_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(4 + financial_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        market = batch["numerical"]
        financial = batch["financial"]
        if market.ndim != 3 or market.shape[1:] != (96, 4):
            raise ValueError(f"numerical must have shape [B, 96, 4], got {tuple(market.shape)}")
        if financial.ndim != 3 or financial.shape[:2] != market.shape[:2] or financial.shape[2] != self.financial_input_size:
            raise ValueError(
                f"financial must have shape [B, 96, {self.financial_input_size}], got {tuple(financial.shape)}"
            )
        financial_latent = self.financial_projection(financial)
        _, hidden = self.gru(torch.cat((market, financial_latent), dim=-1))
        residual = self.head(self.dropout(hidden[-1]))
        return batch["last_close"][:, None] + residual


def build_model(config: dict | None = None) -> nn.Module:
    config = config or {}
    return E3cFinancialGRU(
        hidden_size=int(config.get("hidden_size", 64)),
        horizon=int(config.get("horizon", 24)),
        financial_input_size=int(config.get("financial_input_size", 16)),
        financial_dim=int(config.get("financial_dim", 16)),
        dropout=float(config.get("dropout", 0.0)),
    )
