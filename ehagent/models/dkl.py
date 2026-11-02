"""Deep kernel learning: a sparse variational GP (GPyTorch) on top of the CNN embedding.

* ``SingleTaskGPHead``   - one GP per subset (or the backup "embedding" multitask mode).
* ``LMCMultiTaskGPHead`` - coregionalisation: the subsets share Q latent GPs, and each
  window is scored for its own subset through ``task_indices``.
"""
from __future__ import annotations

from typing import Any

import gpytorch
import numpy as np
import torch
from torch import nn

from .cnn import FeatureExtractor


class SingleTaskGPHead(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        num_inducing, dim = inducing_points.shape
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=dim))

    def forward(self, z: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(z), self.covar_module(z))


class LMCMultiTaskGPHead(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, num_tasks: int):
        num_latents, num_inducing, dim = inducing_points.shape
        batch = torch.Size([num_latents])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing, batch_shape=batch)
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=dim, batch_shape=batch), batch_shape=batch
        )
        self.num_tasks = int(num_tasks)

    def forward(self, z: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(z), self.covar_module(z))

    def task_correlation(self) -> np.ndarray:
        """Subset-to-subset correlation implied by the LMC mixing coefficients."""
        a = self.variational_strategy.lmc_coefficients.detach().cpu().numpy()  # (Q, T)
        cov = a.T @ a
        d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        return cov / np.outer(d, d)


class DKLRegressor(nn.Module):
    def __init__(self, extractor: FeatureExtractor, gp: gpytorch.models.ApproximateGP,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood, multitask_mode: str):
        super().__init__()
        self.extractor = extractor
        self.gp = gp
        self.likelihood = likelihood
        self.multitask_mode = str(multitask_mode)   # "none" | "lmc" | "embedding"

    def embed(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        return self.extractor(x, task if self.extractor.use_task_embedding else None)

    def latent(self, x: torch.Tensor, task: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Latent RUL/cap distribution per window (epistemic uncertainty only)."""
        z = self.embed(x, task)
        if self.multitask_mode == "lmc":
            return self.gp(z, task_indices=task)
        return self.gp(z)


def init_inducing_points(embeddings: np.ndarray, num_inducing: int, seed: int, num_latents: int | None = None) -> torch.Tensor:
    """k-means centres of pretrained embeddings (replicated with jitter per LMC latent)."""
    from sklearn.cluster import MiniBatchKMeans

    m = int(min(num_inducing, len(embeddings)))
    km = MiniBatchKMeans(n_clusters=m, random_state=seed, batch_size=max(1024, 2 * m), n_init=3)
    centres = torch.from_numpy(km.fit(embeddings).cluster_centers_.astype(np.float32))
    if not num_latents:
        return centres
    gen = torch.Generator().manual_seed(seed)
    jitter = 0.01 * centres.std(dim=0, keepdim=True) * torch.randn(num_latents, *centres.shape, generator=gen)
    return centres.unsqueeze(0).repeat(num_latents, 1, 1) + jitter


def init_gp_hyperparameters(gp: gpytorch.models.ApproximateGP, likelihood: gpytorch.likelihoods.GaussianLikelihood,
                            embeddings: np.ndarray, target_mean: float, noise: float = 0.01) -> None:
    """Sane starting point: lengthscales from the embedding spread, mean from the targets."""
    std = torch.from_numpy(np.clip(embeddings.std(axis=0), 1e-3, None).astype(np.float32))
    base = gp.covar_module.base_kernel
    base.lengthscale = std.expand_as(base.lengthscale).clone()
    with torch.no_grad():
        for name, param in gp.mean_module.named_parameters():
            if "constant" in name:
                param.fill_(float(target_mean))
    likelihood.noise = torch.tensor(float(noise))


def build_dkl(meta: dict[str, Any]) -> DKLRegressor:
    """Rebuild an empty DKL with the right shapes so a saved state_dict fits."""
    extractor = FeatureExtractor(**meta["extractor"])
    dim = int(meta["extractor"]["embed_dim"])
    m = int(meta["num_inducing"])
    if meta["multitask_mode"] == "lmc":
        gp: gpytorch.models.ApproximateGP = LMCMultiTaskGPHead(torch.zeros(int(meta["num_latents"]), m, dim), int(meta["num_tasks"]))
    else:
        gp = SingleTaskGPHead(torch.zeros(m, dim))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    return DKLRegressor(extractor, gp, likelihood, meta["multitask_mode"])
