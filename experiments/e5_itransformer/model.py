"""Each standardized market variable is a token; Close token predicts 24 days."""
import torch
from torch import nn
from experiments.e1_dlinear.e6_plus.model import Model as MarketLinear


class ITransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(96, d_model)
        self.variable = nn.Parameter(torch.empty(1, 7, d_model))
        nn.init.normal_(self.variable, std=0.01)
        block = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        for layer in self.encoder.layers:
            for p in layer.parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)
        self.head = nn.Linear(d_model, 24)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("scale", torch.arange(1, 25).float()[None]/24)

    def forward(self, batch):
        tokens = self.embedding(batch["market"].transpose(1, 2))+self.variable
        # stationary feature index6 is standardized Close; inputs already centered.
        delta = self.head(self.encoder(tokens)[:, 6])*batch.get("alpha", 1.0)
        return batch["anchor"]+self.scale*delta


class ScaleAwareMarketLinear(MarketLinear):
    def __init__(self):
        super().__init__(news=False)

    def forward(self, batch):
        return batch["anchor"]+self.scale*(self.market(batch["market"])*batch.get("alpha", 1.0))
