"""E3c + daily news fusion model.

The market branch uses E3c relative OHLC features. News embeddings are mean
pooled per trading day, projected from 512 to a small learned representation,
and accompanied by presence and count features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from experiments.gru_e3.model import transform_window


class E3cNewsGRU(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        horizon: int = 24,
        news_input_size: int = 512,
        news_dim: int = 16,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if hidden_size <= 0 or horizon <= 0 or news_input_size <= 0 or news_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.input_size = 4
        self.news_input_size = news_input_size
        self.news_dim = news_dim
        self.prediction_mode = "standardized_close"
        self.representation = "relative"
        self.residual_scale = float(residual_scale)
        self.news_projection = nn.Sequential(
            nn.Linear(news_input_size, news_dim),
            nn.LayerNorm(news_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(4 + news_dim + 2, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        market = batch["numerical"]
        news = batch["news"]
        has_news = batch["has_news"]
        news_count = batch["news_count"]
        expected = (market.shape[0], 96)
        if market.ndim != 3 or market.shape[1:] != (96, 4):
            raise ValueError(f"numerical must have shape [B, 96, 4], got {tuple(market.shape)}")
        if news.ndim != 3 or news.shape[:2] != expected or news.shape[2] != self.news_input_size:
            raise ValueError(
                f"news must have shape [B, 96, {self.news_input_size}], got {tuple(news.shape)}"
            )
        if has_news.shape != expected or news_count.shape != expected:
            raise ValueError("has_news and news_count must have shape [B, 96]")
        news_latent = self.news_projection(news)
        mask = has_news.to(dtype=market.dtype).unsqueeze(-1)
        news_latent = news_latent * mask
        count = torch.log1p(news_count.to(dtype=market.dtype)).unsqueeze(-1)
        features = torch.cat((market, news_latent, mask, count), dim=-1)
        _, hidden = self.gru(features)
        residual = self.head(self.dropout(hidden[-1])) * self.residual_scale
        baseline = batch["last_close"]
        return baseline[:, None] + residual


def build_model(config: dict | None = None) -> nn.Module:
    config = config or {}
    return E3cNewsGRU(
        hidden_size=int(config.get("hidden_size", 64)),
        horizon=int(config.get("horizon", 24)),
        news_input_size=int(config.get("news_input_size", 512)),
        news_dim=int(config.get("news_dim", 16)),
        dropout=float(config.get("dropout", 0.0)),
        residual_scale=float(config.get("residual_scale", 1.0)),
    )


def prepare_batch(samples: list[dict], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("samples must not be empty")
    market = []
    last_close = []
    news = []
    has_news = []
    news_count = []
    targets = []
    for sample in samples:
        numerical = np.asarray(sample["numerical"], dtype=np.float32)
        market.append(
            transform_window(
                numerical, sample["close_mean"], sample["close_std"], 4, "relative"
            )
        )
        last_close.append(
            (float(numerical[-1, 3]) - float(sample["close_mean"]))
            / max(float(sample["close_std"]), 1e-6)
        )
        daily = np.zeros((96, 512), dtype=np.float32)
        counts = np.zeros(96, dtype=np.float32)
        for day_index in range(96):
            articles = np.asarray(sample["news"][day_index], dtype=np.float32)
            if len(articles):
                daily[day_index] = articles.mean(axis=0)
                counts[day_index] = len(articles)
        news.append(daily)
        news_count.append(counts)
        has_news.append((counts > 0).astype(np.float32))
        if "y" in sample:
            targets.append(sample["y"])
    batch = {
        "numerical": torch.as_tensor(np.stack(market), dtype=torch.float32, device=device),
        "last_close": torch.as_tensor(last_close, dtype=torch.float32, device=device),
        "news": torch.as_tensor(np.stack(news), dtype=torch.float32, device=device),
        "has_news": torch.as_tensor(np.stack(has_news), dtype=torch.float32, device=device),
        "news_count": torch.as_tensor(np.stack(news_count), dtype=torch.float32, device=device),
    }
    if targets:
        if len(targets) != len(samples):
            raise ValueError("batch contains a mixture of labeled and unlabeled samples")
        batch["y"] = torch.as_tensor(np.stack(targets), dtype=torch.float32, device=device)
    return batch
