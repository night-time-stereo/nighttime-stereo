import numpy as np
import torch
from torchmetrics import Metric

class BadX(Metric):
    def __init__(self, threshold):
        super().__init__()
        self.add_state("errors", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        mask = target > 0
        self.errors += torch.sum(((preds - target).abs() > self.threshold) * mask)
        self.total += mask.sum()

    def compute(self):
        return self.errors.float() / self.total

class Rmse(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("errors", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        mask = (target > 0)
        self.errors += torch.sum(((preds - target) ** 2) * mask)
        self.total += mask.sum()

    def compute(self):
        return torch.sqrt(self.errors / self.total)
