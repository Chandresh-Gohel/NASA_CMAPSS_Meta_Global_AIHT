"""Model loading plus one prediction interface for the DKL, ensemble and CNN families.

Every family answers with the same outputs in cycles: ``mu``, ``sigma_epi``,
``sigma_alea`` and ``sigma``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, get_device, path_for


def model_path(cfg: Config, family: str, tag: str) -> Path:
    return path_for(cfg, "artifacts_dir", "models", f"{family}_{tag}.pt")


def predictor_tag(family: str, multitask: bool) -> str:
    return f"{family}_{'multitask' if multitask else 'single'}"


def _load_blob(path: Path, device: torch.device) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}. Train it first with `python -m ehagent.train ...`.")
    return torch.load(path, map_location=device, weights_only=False)


def build_module(family: str, meta: dict[str, Any]) -> torch.nn.Module:
    from .models.cnn import CNNGaussianRegressor, CNNRegressor, FeatureExtractor

    ek = dict(meta["extractor"])
    n_tasks = int(meta["num_tasks"])
    if family == "cnn":
        return CNNRegressor(FeatureExtractor(**ek), n_tasks)
    if family == "ensemble":
        from .models.ensemble import GaussianEnsemble

        return GaussianEnsemble([CNNGaussianRegressor(FeatureExtractor(**ek), n_tasks) for _ in range(int(meta["ensemble_size"]))])
    if family == "dkl":
        from .models.dkl import build_dkl

        return build_dkl(meta)
    raise ValueError(f"Unknown model family: {family}")


def load_module(path: Path, device: torch.device, train_mode: bool = False):
    """Load a checkpoint into a fresh module. Returns (module, meta, family)."""
    blob = _load_blob(path, device)
    module = build_module(blob["family"], blob["meta"])
    module.load_state_dict(blob["state"])
    module.to(device)
    module.train(train_mode)
    return module, blob["meta"], blob["family"]


class LoadedModel:
    def __init__(self, family: str, meta: dict, module: torch.nn.Module, device: torch.device):
        self.family = family
        self.meta = meta
        self.module = module
        self.device = device

    @torch.no_grad()
    def predict_scaled(self, x: torch.Tensor, task: torch.Tensor):
        """Mean, epistemic variance, aleatoric variance - all in RUL/cap units."""
        m = self.module
        if self.family == "dkl":
            import gpytorch

            with gpytorch.settings.cholesky_jitter(1e-4):
                f = m.latent(x, task)
                mean = f.mean
                var_epi = f.variance.clamp_min(0.0)
            var_alea = m.likelihood.noise.reshape(-1)[0].expand_as(mean)
        elif self.family == "ensemble":
            means, variances = m(x, task)
            mean = means.mean(dim=0)
            var_epi = means.var(dim=0, unbiased=False)
            var_alea = variances.mean(dim=0)
        else:
            mean = m(x, task)
            var_epi = torch.zeros_like(mean)
            var_alea = torch.zeros_like(mean)
        return mean, var_epi, var_alea

    def mean_scaled(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """Differentiable predictive mean, for Integrated Gradients."""
        m = self.module
        if self.family == "dkl":
            import gpytorch

            with gpytorch.settings.cholesky_jitter(1e-4):
                return m.latent(x, task).mean
        if self.family == "ensemble":
            means, _ = m(x, task)
            return means.mean(dim=0)
        return m(x, task)


def load_model(cfg: Config, family: str, tag: str, device: torch.device) -> LoadedModel:
    module, meta, fam = load_module(model_path(cfg, family, tag), device, train_mode=False)
    return LoadedModel(fam, meta, module, device)


class Predictor:
    """Routes each subset to its model: one multitask model, or one model per subset."""

    def __init__(self, cfg: Config, family: str | None = None, multitask: bool | None = None,
                 subsets: list[str] | None = None, device: torch.device | None = None):
        self.cfg = cfg
        self.family = str(family or cfg.model.family)
        self.multitask = bool(cfg.features.multitask_model if multitask is None else multitask)
        self.subsets = list(subsets or cfg.data.subsets)
        self.cap = float(cfg.data.rul_cap)
        self.window = int(cfg.data.window)
        self.device = device or get_device(cfg)
        self._models: dict[str, LoadedModel] = {}
        self._task_index: dict[str, int] = {}
        if self.multitask:
            lm = load_model(cfg, self.family, "multitask", self.device)
            order = list(lm.meta["subsets"])
            for s in self.subsets:
                if s not in order:
                    raise ValueError(f"Subset {s} is not part of the multitask model (trained on {order})")
                self._models[s] = lm
                self._task_index[s] = order.index(s)
        else:
            for s in self.subsets:
                self._models[s] = load_model(cfg, self.family, s, self.device)
                self._task_index[s] = 0

    @property
    def tag(self) -> str:
        return predictor_tag(self.family, self.multitask)

    def model_file(self, subset: str) -> Path:
        return model_path(self.cfg, self.family, "multitask" if self.multitask else subset)

    def loaded(self, subset: str) -> LoadedModel:
        return self._models[subset]

    def predict(self, subset: str, X: np.ndarray, batch_size: int = 1024) -> dict[str, np.ndarray]:
        lm = self._models[subset]
        task_id = self._task_index[subset]
        n = len(X)
        if n == 0:
            empty = np.zeros(0)
            return {"mu": empty, "sigma": empty, "sigma_epi": empty, "sigma_alea": empty}
        means, epis, aleas = [], [], []
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size], dtype=np.float32)).to(self.device)
            tb = torch.full((len(xb),), task_id, dtype=torch.long, device=self.device)
            mean, ve, va = lm.predict_scaled(xb, tb)
            means.append(mean.float().cpu().numpy())
            epis.append(ve.float().cpu().numpy())
            aleas.append(va.float().cpu().numpy())
        mean = np.concatenate(means)
        var_epi = np.concatenate(epis)
        var_alea = np.concatenate(aleas)
        return {
            "mu": np.maximum(mean * self.cap, 0.0),
            "sigma": np.sqrt(np.maximum(var_epi + var_alea, 1e-12)) * self.cap,
            "sigma_epi": np.sqrt(np.maximum(var_epi, 0.0)) * self.cap,
            "sigma_alea": np.sqrt(np.maximum(var_alea, 0.0)) * self.cap,
        }

    def mean_torch(self, subset: str, x: torch.Tensor) -> torch.Tensor:
        lm = self._models[subset]
        task = torch.full((x.shape[0],), self._task_index[subset], dtype=torch.long, device=x.device)
        return lm.mean_scaled(x, task) * self.cap

    def task_correlation(self) -> dict | None:
        """Subset correlation learned by the LMC head, if this is an LMC multitask DKL."""
        if not (self.multitask and self.family == "dkl"):
            return None
        lm = next(iter(self._models.values()))
        gp = lm.module.gp
        if not hasattr(gp, "task_correlation"):
            return None
        return {"subsets": list(lm.meta["subsets"]), "correlation": np.round(gp.task_correlation(), 3).tolist()}
