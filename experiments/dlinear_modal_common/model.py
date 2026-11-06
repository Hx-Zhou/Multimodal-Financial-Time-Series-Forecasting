"""Fixed E6 numerical backbone and explicit small auxiliary gates."""
import torch
from torch import nn
from experiments.e1_dlinear.e6_plus.model import Heads


class FrozenMarket(nn.Module):
    with_news = False
    with_reports = False

    def __init__(self):
        super().__init__()
        self.market = Heads(7)
        self.market.requires_grad_(False)
        self.register_buffer("scale", torch.arange(1, 25).float()[None] / 24)

    def load_market(self, saved):
        self.market.load_state_dict({k.removeprefix("market."): v for k, v in saved["state_dict"].items() if k.startswith("market.")})

    def forward(self, batch):
        delta = self.market(batch["market"]) + self.auxiliary(batch)
        return batch["anchor"] + self.scale * delta

    def dropout_modality(self, delta):
        # One Bernoulli per sample and modality, not independent horizon dropout.
        if self.training:
            keep = (torch.rand((len(delta), 1), device=delta.device) >= 0.1).to(delta.dtype)
            delta = delta * keep / 0.9
        return delta


class NewsBranch(nn.Module):
    """Same daily 512->8->8 representation and DLinear heads as E7."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(512, 8), nn.SiLU(), nn.Linear(8, 8))
        self.heads = Heads(10)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, batch):
        z = self.mlp(batch["news_unique"])[batch["news_inverse"]].reshape(-1, 96, 8)
        z = z * batch["has_news"][..., None]
        if batch.get("zero_semantics", False):
            z = torch.zeros_like(z)
        x = torch.cat((z, batch["has_news"][..., None], batch["news_count"][..., None]), dim=-1)
        return torch.sigmoid(self.gate) * self.heads(x) * batch["has_news"].any(dim=1, keepdim=True)
