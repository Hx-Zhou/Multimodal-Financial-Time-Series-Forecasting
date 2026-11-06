"""Joint news and financial adapters on top of a frozen E3c market GRU."""

from __future__ import annotations

import torch
from torch import nn


class E3cFrozenNewsFinancialGRU(nn.Module):
    """Fuse two temporal modalities as gated residual corrections to E3c.

    The market encoder and market head are frozen.  The two adapter branches
    are independent, and each forecast step has its own scalar gate:

        prediction = E3c + news_gate * news_delta
                           + financial_gate * financial_delta

    Gates can be initialized from the separately trained E9-1/E9-2 adapters
    or reset to zero for an exact E3c epoch-0 reference.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        horizon: int = 24,
        news_dim: int = 16,
        news_hidden: int = 32,
        financial_dim: int = 16,
        financial_hidden: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = 4
        self.representation = "relative"
        self.prediction_mode = "standardized_close"
        self.news_input_size = 1024
        self.financial_input_size = 16
        self.news_dim = news_dim
        self.news_hidden = news_hidden
        self.financial_dim = financial_dim
        self.financial_hidden = financial_hidden

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

        self.financial_projection = nn.Sequential(
            nn.Linear(16, financial_dim), nn.LayerNorm(financial_dim), nn.GELU()
        )
        self.financial_gru = nn.GRU(financial_dim, financial_hidden, batch_first=True)
        self.financial_head = nn.Sequential(
            nn.Linear(hidden_size + financial_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

        self.news_gate = nn.Parameter(torch.zeros(horizon))
        self.financial_gate = nn.Parameter(torch.zeros(horizon))

    @staticmethod
    def _select(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not selected:
            raise ValueError(f"checkpoint has no {prefix!r} parameters")
        return selected

    def load_e3c_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load the root E3c checkpoint and freeze its market path."""
        self.market_gru.load_state_dict(self._select(state, "gru."))
        self.market_head.load_state_dict(self._select(state, "head."))
        for parameter in (*self.market_gru.parameters(), *self.market_head.parameters()):
            parameter.requires_grad_(False)

    def load_news_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load only the adapter portion of an E9-1 checkpoint."""
        for module_name in ("news_projection", "news_gru", "news_head"):
            module = getattr(self, module_name)
            module.load_state_dict(self._select(state, module_name + "."))
        if "news_gate" not in state:
            raise ValueError("news checkpoint has no news_gate")
        self.news_gate.data.copy_(state["news_gate"])

    def load_financial_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load only the adapter portion of an E9-2 checkpoint."""
        for module_name in ("financial_projection", "financial_gru", "financial_head"):
            module = getattr(self, module_name)
            module.load_state_dict(self._select(state, module_name + "."))
        if "financial_gate" not in state:
            raise ValueError("financial checkpoint has no financial_gate")
        self.financial_gate.data.copy_(state["financial_gate"])

    def zero_gates(self) -> None:
        """Reset both residual paths to the exact frozen E3c forecast."""
        with torch.no_grad():
            self.news_gate.zero_()
            self.financial_gate.zero_()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        market = batch["numerical"]
        news = batch["news"]
        financial = batch["financial"]
        if market.ndim != 3 or market.shape[1:] != (96, 4):
            raise ValueError(f"numerical must have shape [B, 96, 4], got {tuple(market.shape)}")
        if news.ndim != 3 or news.shape[1:] != (96, 1024):
            raise ValueError(f"news must have shape [B, 96, 1024], got {tuple(news.shape)}")
        if financial.ndim != 3 or financial.shape[1:] != (96, 16):
            raise ValueError(f"financial must have shape [B, 96, 16], got {tuple(financial.shape)}")

        _, market_hidden = self.market_gru(market)
        market_state = market_hidden[-1]
        base = batch["last_close"][:, None] + self.market_head(market_state)

        news_latent = self.news_projection(news)
        mask = batch["has_news"].to(dtype=market.dtype).unsqueeze(-1)
        count = torch.log1p(batch["news_count"].to(dtype=market.dtype)).unsqueeze(-1)
        _, news_hidden = self.news_gru(torch.cat((news_latent * mask, mask, count), dim=-1))
        news_delta = self.news_head(torch.cat((market_state.detach(), news_hidden[-1]), dim=-1))

        financial_latent = self.financial_projection(financial)
        _, financial_hidden = self.financial_gru(financial_latent)
        financial_delta = self.financial_head(
            torch.cat((market_state.detach(), financial_hidden[-1]), dim=-1)
        )
        return base + news_delta * self.news_gate + financial_delta * self.financial_gate


def build_model(config: dict | None = None) -> nn.Module:
    config = config or {}
    return E3cFrozenNewsFinancialGRU(
        hidden_size=int(config.get("hidden_size", 64)),
        horizon=int(config.get("horizon", 24)),
        news_dim=int(config.get("news_dim", 16)),
        news_hidden=int(config.get("news_hidden", 32)),
        financial_dim=int(config.get("financial_dim", 16)),
        financial_hidden=int(config.get("financial_hidden", 32)),
        dropout=float(config.get("dropout", 0.0)),
    )
