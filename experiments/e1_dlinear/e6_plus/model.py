"""Multivariate DLinear and a masked, daily TaTS-style news extension."""
import torch
from torch import nn
from torch.nn import functional as F

LOOKBACK, HORIZON = 96, 24


class Heads(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.trend = nn.Linear(LOOKBACK * channels, HORIZON)
        self.remainder = nn.Linear(LOOKBACK * channels, HORIZON)
        for layer in (self.trend, self.remainder):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        z = x.transpose(1, 2)
        trend = F.avg_pool1d(F.pad(z, (12, 12), mode="replicate"), 25, stride=1).transpose(1, 2)
        return self.trend(trend.flatten(1)) + self.remainder((x - trend).flatten(1))


class Model(nn.Module):
    def __init__(self, news=False):
        super().__init__()
        self.with_news = news
        self.market = Heads(7)
        self.register_buffer("scale", torch.arange(1, 25).float()[None] / 24)
        if news:
            self.news_mlp = nn.Sequential(nn.Linear(512, 8), nn.SiLU(), nn.Linear(8, 8))
            self.news_heads = Heads(10)

    def forward(self, batch):
        delta = self.market(batch["market"])
        if self.with_news:
            # Project only unique pooled days within this batch, including row zero.
            z = self.news_mlp(batch["news_unique"])[batch["news_inverse"]]
            z = z.reshape(-1, LOOKBACK, 8) * batch["has_news"][..., None]
            if batch.get("zero_semantics", False):
                z = torch.zeros_like(z)
            x = torch.cat((z, batch["has_news"][..., None], batch["news_count"][..., None]), dim=-1)
            has_window = batch["has_news"].any(dim=1, keepdim=True)
            news_delta = self.news_heads(x) * has_window
            delta = delta + news_delta
        return batch["anchor"] + self.scale * delta

    def freeze_market(self):
        self.market.requires_grad_(False)
