"""Small univariate PatchTST-style encoder and matched DLinear control."""
import torch
from torch import nn
from torch.nn import functional as F

LOOKBACK, HORIZON = 96, 24


def history_transform(close):
    if close.ndim != 2 or close.shape[1] != LOOKBACK or not torch.isfinite(close).all():
        raise ValueError("expected finite standardized Close [B,96]")
    anchor = close[:, -1:]
    # Input-window-only scale; no fitted validation/test statistics or future y.
    sigma = (close.var(dim=1, keepdim=True, unbiased=False)+1e-5).sqrt().detach()
    return (close-anchor)/sigma, anchor, sigma


class ResidualForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("horizon_scale", torch.arange(1, 25).float()[None]/24)

    def forward(self, batch):
        x, anchor, sigma = history_transform(batch["close"])
        return anchor+sigma*self.horizon_scale*self.residual(x)


class PatchTST(ResidualForecast):
    def __init__(self, patch_len=16, stride=8, d_model=128, nhead=4, layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        if not 0 < patch_len <= LOOKBACK or stride <= 0 or d_model % nhead:
            raise ValueError("invalid patch/attention configuration")
        self.patch_len, self.stride = patch_len, stride
        self.patches = (LOOKBACK-patch_len)//stride+1
        self.projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(1, self.patches, d_model))
        nn.init.normal_(self.position, std=0.01)
        block = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.patches*d_model, HORIZON)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, x):
        tokens = self.projection(x.unfold(1, self.patch_len, self.stride))+self.position
        return self.head(self.dropout(self.encoder(tokens).flatten(1)))


class DLinearControl(ResidualForecast):
    """Same normalization, anchor, h/24, loss/optimizer/batches as PatchTST."""
    def __init__(self):
        super().__init__()
        self.trend = nn.Linear(LOOKBACK, HORIZON)
        self.remainder = nn.Linear(LOOKBACK, HORIZON)
        for layer in (self.trend, self.remainder):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def residual(self, x):
        trend = F.avg_pool1d(F.pad(x[:, None], (12, 12), mode="replicate"), 25, stride=1).squeeze(1)
        return self.trend(trend)+self.remainder(x-trend)


def build_model(kind, config):
    if kind == "patchtst":
        return PatchTST(**config)
    if kind == "dlinear_control":
        return DLinearControl()
    raise ValueError(kind)
