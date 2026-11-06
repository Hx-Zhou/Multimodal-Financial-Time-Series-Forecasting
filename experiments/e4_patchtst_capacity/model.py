"""Original E4 architecture; alpha changes validation/inference, never training loss."""
from experiments.e4_patchtst.model import PatchTST, DLinearControl, history_transform


class ScaleAwarePatchTST(PatchTST):
    def forward(self, batch):
        x, anchor, sigma = history_transform(batch["close"])
        return anchor+sigma*self.horizon_scale*(self.residual(x)*batch.get("alpha", 1.0))


class ScaleAwareLinear(DLinearControl):
    def forward(self, batch):
        x, anchor, sigma = history_transform(batch["close"])
        return anchor+sigma*self.horizon_scale*(self.residual(x)*batch.get("alpha", 1.0))
