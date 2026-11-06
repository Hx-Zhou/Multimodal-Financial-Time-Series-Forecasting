"""News adapter on top of a frozen E3c relative-OHLC GRU."""

from __future__ import annotations

import torch
from torch import nn


class E3cFrozenNewsGRU(nn.Module):
    def __init__(self, hidden_size: int = 64, horizon: int = 24, news_dim: int = 16, news_hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        self.input_size = 4
        self.representation = "relative"
        self.prediction_mode = "standardized_close"
        self.news_input_size = 1024
        self.news_dim = news_dim
        self.news_hidden = news_hidden
        self.market_gru = nn.GRU(4, hidden_size, batch_first=True)
        self.market_head = nn.Linear(hidden_size, horizon)
        self.news_projection = nn.Sequential(
            nn.Linear(1024, news_dim), nn.LayerNorm(news_dim), nn.GELU()
        )
        self.news_gru = nn.GRU(news_dim + 2, news_hidden, batch_first=True)
        self.news_head = nn.Sequential(
            nn.Linear(hidden_size + news_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )
        # A zero gate makes the initial E9-1 output exactly the E3c output.
        self.news_gate = nn.Parameter(torch.zeros(horizon))

    def load_e3c_state(self, state: dict[str, torch.Tensor]) -> None:
        self.market_gru.load_state_dict({key[4:]: value for key, value in state.items() if key.startswith("gru.")})
        self.market_head.load_state_dict({key[5:]: value for key, value in state.items() if key.startswith("head.")})
        for parameter in list(self.market_gru.parameters()) + list(self.market_head.parameters()):
            parameter.requires_grad_(False)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        market = batch["numerical"]
        news = batch["news"]
        if market.ndim != 3 or market.shape[1:] != (96, 4):
            raise ValueError(f"numerical must have shape [B, 96, 4], got {tuple(market.shape)}")
        if news.ndim != 3 or news.shape[1:] != (96, 1024):
            raise ValueError(f"news must have shape [B, 96, 1024], got {tuple(news.shape)}")
        _, market_hidden = self.market_gru(market)
        base = batch["last_close"][:, None] + self.market_head(market_hidden[-1])
        news_latent = self.news_projection(news)
        mask = batch["has_news"].to(dtype=market.dtype).unsqueeze(-1)
        count = torch.log1p(batch["news_count"].to(dtype=market.dtype)).unsqueeze(-1)
        _, news_hidden = self.news_gru(torch.cat((news_latent * mask, mask, count), dim=-1))
        delta = self.news_head(torch.cat((market_hidden[-1].detach(), news_hidden[-1]), dim=-1))
        return base + delta * self.news_gate


def build_model(config: dict | None = None) -> nn.Module:
    config = config or {}
    return E3cFrozenNewsGRU(
        hidden_size=int(config.get("hidden_size", 64)),
        horizon=int(config.get("horizon", 24)),
        news_dim=int(config.get("news_dim", 16)),
        news_hidden=int(config.get("news_hidden", 32)),
        dropout=float(config.get("dropout", 0.0)),
    )
