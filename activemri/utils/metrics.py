"""Custom torchmetrics metrics for MRI reconstruction evaluation."""

import torch
from torchmetrics import Metric


class NMSE(Metric):
    """Normalized Mean Squared Error (NMSE) metric."""

    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_nmse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_size = target.shape[0]
        for i in range(batch_size):
            gt = target[i].flatten()
            pred = preds[i].flatten()
            self.sum_nmse += torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2
            self.count += 1

    def compute(self):
        return self.sum_nmse / self.count
