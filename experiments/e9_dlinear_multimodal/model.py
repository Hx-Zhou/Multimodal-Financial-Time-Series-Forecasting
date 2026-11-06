"""Train both modalities together; frozen numerical weights match E6/E7/E8."""
import torch
from experiments.dlinear_modal_common.model import FrozenMarket, NewsBranch
from experiments.e8_dlinear_reports.model import ReportBranch


class Model(FrozenMarket):
    with_news = True
    with_reports = True

    def __init__(self):
        super().__init__()
        self.news = NewsBranch()
        self.reports = ReportBranch()

    def auxiliary(self, batch):
        delta = torch.zeros_like(batch["anchor"])
        if not batch.get("disable_news", False):
            delta = delta + self.dropout_modality(self.news(batch))
        if not batch.get("disable_reports", False):
            delta = delta + self.dropout_modality(self.reports(batch))
        return delta
