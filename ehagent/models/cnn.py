"""Residual 1D-CNN: the feature extractor plus point and Gaussian regression heads."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class _Scaled(nn.Module):
    """A spectrally-normalised layer with its output rescaled by ``coeff``."""

    def __init__(self, layer: nn.Module, coeff: float):
        super().__init__()
        self.layer = spectral_norm(layer)
        self.coeff = float(coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff * self.layer(x)


def _sn(layer: nn.Module, use_sn: bool, coeff: float) -> nn.Module:
    return _Scaled(layer, coeff) if use_sn else layer


class ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float, use_sn: bool, coeff: float):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = _sn(nn.Conv1d(channels, channels, kernel_size, padding=pad), use_sn, coeff)
        self.conv2 = _sn(nn.Conv1d(channels, channels, kernel_size, padding=pad), use_sn, coeff)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.conv1(x)))
        h = self.conv2(h)
        return self.act(x + h)


class FeatureExtractor(nn.Module):
    """A (B, L, F) window in, a (B, D) embedding out."""

    def __init__(self, n_features: int, channels: int = 64, n_blocks: int = 3, kernel_size: int = 5,
                 embed_dim: int = 16, dropout: float = 0.1, use_sn: bool = True, sn_coeff: float = 3.0,
                 n_tasks: int = 1, use_task_embedding: bool = False, task_embed_dim: int = 4):
        super().__init__()
        pad = kernel_size // 2
        self.inp = _sn(nn.Conv1d(n_features, channels, kernel_size, padding=pad), use_sn, sn_coeff)
        self.blocks = nn.Sequential(*[ResBlock1d(channels, kernel_size, dropout, use_sn, sn_coeff) for _ in range(n_blocks)])
        self.use_task_embedding = bool(use_task_embedding and n_tasks > 1)
        self.task_emb = nn.Embedding(n_tasks, task_embed_dim) if self.use_task_embedding else None
        head_in = 2 * channels + (task_embed_dim if self.use_task_embedding else 0)
        self.head = nn.Sequential(
            _sn(nn.Linear(head_in, 64), use_sn, sn_coeff),
            nn.GELU(),
            _sn(nn.Linear(64, embed_dim), use_sn, sn_coeff),
        )
        self.embed_dim = int(embed_dim)

    def forward(self, x: torch.Tensor, task: torch.Tensor | None = None) -> torch.Tensor:
        h = x.transpose(1, 2)                      # (B, F, L) for Conv1d
        h = nn.functional.gelu(self.inp(h))
        h = self.blocks(h)
        feats = torch.cat([h.mean(dim=-1), h[..., -1]], dim=-1)
        if self.use_task_embedding:
            if task is None:
                raise ValueError("task indices are required when use_task_embedding=True")
            feats = torch.cat([feats, self.task_emb(task)], dim=-1)
        return self.head(feats)


def extractor_kwargs_from_cfg(cfg: Any, n_features: int, n_tasks: int, use_task_embedding: bool) -> dict:
    m = cfg.model
    return dict(
        n_features=int(n_features), channels=int(m.channels), n_blocks=int(m.n_blocks),
        kernel_size=int(m.kernel_size), embed_dim=int(m.embed_dim), dropout=float(m.dropout),
        use_sn=bool(m.spectral_norm), sn_coeff=float(m.sn_coeff), n_tasks=int(n_tasks),
        use_task_embedding=bool(use_task_embedding), task_embed_dim=int(m.task_embed_dim),
    )


class CNNRegressor(nn.Module):
    """Point regressor for pretraining and the baseline: one linear output per task, RUL / cap."""

    def __init__(self, extractor: FeatureExtractor, n_tasks: int = 1):
        super().__init__()
        self.extractor = extractor
        self.out = nn.Linear(extractor.embed_dim, n_tasks)

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        z = self.extractor(x, task)
        y = self.out(z)
        return y.gather(1, task.view(-1, 1)).squeeze(1)


class CNNGaussianRegressor(nn.Module):
    """Mean + variance head for the deep-ensemble backup. Outputs are in RUL / cap units."""

    def __init__(self, extractor: FeatureExtractor, n_tasks: int = 1):
        super().__init__()
        self.extractor = extractor
        self.n_tasks = int(n_tasks)
        self.out = nn.Linear(extractor.embed_dim, 2 * self.n_tasks)

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.extractor(x, task)
        o = self.out(z).view(-1, self.n_tasks, 2)
        o = o[torch.arange(o.shape[0], device=o.device), task]
        mean = o[:, 0]
        var = nn.functional.softplus(o[:, 1]) + 1e-4
        return mean, var
