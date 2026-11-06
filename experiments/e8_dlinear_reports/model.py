"""Small gated financial residual on the unchanged frozen E6 backbone."""
import torch
from torch import nn
from experiments.dlinear_modal_common.data import REPORT_DIM
from experiments.dlinear_modal_common.model import FrozenMarket


class ReportBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(REPORT_DIM, 16), nn.SiLU(), nn.Linear(16, 8), nn.SiLU())
        self.head = nn.Linear(8, 24)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, batch):
        x = batch["reports"]
        if batch.get("zero_report_values", False):
            x = x.clone()
            x[:, :24] = 0  # retain missing masks and ages for a content diagnostic
        return self.head(self.encoder(x)) * torch.sigmoid(self.gate) * batch["has_report"]


class Model(FrozenMarket):
    with_reports = True

    def __init__(self):
        super().__init__()
        self.reports = ReportBranch()

    def auxiliary(self, batch):
        if batch.get("disable_reports", False):
            return torch.zeros_like(batch["anchor"])
        return self.dropout_modality(self.reports(batch))
