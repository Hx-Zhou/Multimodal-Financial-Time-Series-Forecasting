"""A bounded SpecTF-style adaptation, not a full reproduction of the paper."""
import math
import torch
from torch import nn
from experiments.dlinear_modal_common.model import FrozenMarket
from experiments.e1_dlinear.e6_plus.model import Heads


class Model(FrozenMarket):
    with_news = True

    def __init__(self):
        super().__init__()
        self.modes = 13  # DC plus the first 12 frequencies; fixed before evaluation
        self.query = nn.Linear(14, 16)
        self.news_mlp = nn.Sequential(nn.Linear(512, 8), nn.SiLU(), nn.Linear(8, 16))
        self.coverage = nn.Linear(2, 16, bias=False)
        self.value = nn.Linear(16, 16, bias=False)
        self.heads = Heads(8)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        phase = 2 * math.pi * torch.arange(96).float() / 96
        self.register_buffer("cos_phase", phase.cos()[None, :, None])
        self.register_buffer("sin_phase", phase.sin()[None, :, None])

    def auxiliary(self, batch):
        if batch.get("disable_news", False):
            return torch.zeros_like(batch["anchor"])
        market = torch.fft.rfft(batch["market"], dim=1, norm="ortho")[:, :self.modes]
        query = self.query(torch.cat((market.real, market.imag), dim=-1))
        z = self.news_mlp(batch["news_unique"])[batch["news_inverse"]].reshape(-1, 96, 16)
        if batch.get("zero_semantics", False):
            z = torch.zeros_like(z)
        mask = batch["has_news"].bool()
        coverage = torch.stack((batch["has_news"], batch["news_count"]), dim=-1)
        z = (z + self.coverage(coverage)) * mask[..., None]
        # Complex time-phase encoding; missing dates stay zero after all projections.
        real, imag = z[..., :8], z[..., 8:]
        key = torch.cat((real*self.cos_phase-imag*self.sin_phase,
                         real*self.sin_phase+imag*self.cos_phase), dim=-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(16)
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        weights = torch.softmax(scores, dim=-1) * mask[:, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        fused = torch.bmm(weights, self.value(key))
        spectral = torch.complex(fused[..., :8], fused[..., 8:])
        spectral = torch.nn.functional.pad(spectral, (0, 0, 0, 49-self.modes))
        temporal = torch.fft.irfft(spectral, n=96, dim=1, norm="ortho")
        delta = self.heads(temporal) * torch.sigmoid(self.gate) * mask.any(dim=1, keepdim=True)
        return self.dropout_modality(delta)
