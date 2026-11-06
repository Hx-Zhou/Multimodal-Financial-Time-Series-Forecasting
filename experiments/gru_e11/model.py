"""Frozen E3c market model with an expanded financial GRU input."""

from __future__ import annotations

import torch
from torch import nn


class E3cFrozenExpandedFinancialGRU(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        horizon: int = 24,
        financial_input_size: int = 35,
        financial_dim: int = 16,
        financial_hidden: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = 4
        self.representation = "relative"
        self.prediction_mode = "standardized_close"
        self.financial_input_size = financial_input_size
        self.financial_dim = financial_dim
        self.financial_hidden = financial_hidden
        self.market_gru = nn.GRU(4, hidden_size, batch_first=True)
        self.market_head = nn.Linear(hidden_size, horizon)
        self.financial_projection = nn.Sequential(
            nn.Linear(financial_input_size, financial_dim),
            nn.LayerNorm(financial_dim),
            nn.GELU(),
        )
        self.financial_gru = nn.GRU(financial_dim, financial_hidden, batch_first=True)
        self.financial_head = nn.Sequential(
            nn.Linear(hidden_size + financial_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )
        self.financial_gate = nn.Parameter(torch.zeros(horizon))

    @staticmethod
    def _select(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not selected:
            raise ValueError(f"checkpoint has no {prefix!r} parameters")
        return selected

    def load_e3c_state(self, state: dict[str, torch.Tensor]) -> None:
        self.market_gru.load_state_dict(self._select(state, "gru."))
        self.market_head.load_state_dict(self._select(state, "head."))
        for parameter in (*self.market_gru.parameters(), *self.market_head.parameters()):
            parameter.requires_grad_(False)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        market = batch["numerical"]
        financial = batch["financial"]
        if market.ndim != 3 or market.shape[1:] != (96, 4):
            raise ValueError(f"numerical must have shape [B, 96, 4], got {tuple(market.shape)}")
        if financial.ndim != 3 or financial.shape[1:] != (96, self.financial_input_size):
            raise ValueError(
                f"financial must have shape [B, 96, {self.financial_input_size}], "
                f"got {tuple(financial.shape)}"
            )
        _, market_hidden = self.market_gru(market)
        market_state = market_hidden[-1]
        base = batch["last_close"][:, None] + self.market_head(market_state)
        financial_latent = self.financial_projection(financial)
        _, financial_hidden = self.financial_gru(financial_latent)
        delta = self.financial_head(
            torch.cat((market_state.detach(), financial_hidden[-1]), dim=-1)
        )
        return base + delta * self.financial_gate


def build_model(config: dict | None = None) -> nn.Module:
    config = config or {}
    return E3cFrozenExpandedFinancialGRU(
        hidden_size=int(config.get("hidden_size", 64)),
        horizon=int(config.get("horizon", 24)),
        financial_input_size=int(config.get("financial_input_size", 35)),
        financial_dim=int(config.get("financial_dim", 16)),
        financial_hidden=int(config.get("financial_hidden", 32)),
        dropout=float(config.get("dropout", 0.0)),
    )
