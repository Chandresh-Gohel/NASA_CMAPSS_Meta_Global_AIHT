"""Deep ensemble of Gaussian-head CNNs: the backup uncertainty model (same outputs as DKL)."""
from __future__ import annotations

import torch
from torch import nn

from .cnn import CNNGaussianRegressor


class GaussianEnsemble(nn.Module):
    def __init__(self, members: list[CNNGaussianRegressor]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means, variances = zip(*(m(x, task) for m in self.members))
        return torch.stack(means), torch.stack(variances)   # (K, B) each
