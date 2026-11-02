"""Step 5 exit check: tiny single-task DKL smoke on FD001 CPU (<2 min, no API).

Full run: python -m ehagent.train dkl --mode single --subsets FD001
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ehagent.config import load_config, set_seed
from ehagent.data import load_subset
from ehagent.predictor import load_module, model_path
from ehagent.train import arrays_for, train_cnn, train_dkl


def main() -> None:
    cfg = load_config()
    set_seed(int(cfg.project.seed))
    cfg.train.dkl_epochs = 2
    cfg.train.pretrain_epochs = 2
    cfg.train.dkl_batch_size = 128
    cfg.train.batch_size = 128
    cfg.model.inducing_points = 64
    device = torch.device("cpu")
    print(f"device={device}")

    datasets = {"FD001": load_subset(cfg, "FD001")}
    tr_full = arrays_for(cfg, datasets, ["FD001"], "train")
    va_full = arrays_for(cfg, datasets, ["FD001"], "val")
    ntr, nva = min(512, len(tr_full[1])), min(256, len(va_full[1]))
    tr = (tr_full[0][:ntr], tr_full[1][:ntr], tr_full[2][:ntr])
    va = (va_full[0][:nva], va_full[1][:nva], va_full[2][:nva])
    print(f"smoke windows: train={len(tr[1])} val={len(va[1])}")

    cnn, _, cnn_info = train_cnn(cfg, tr, va, 1, False, device)
    print(f"cnn smoke val_rmse={cnn_info['val_rmse']:.2f}")

    dkl, dinfo = train_dkl(cfg, cnn, tr, va, 1, "none", device)
    print(f"dkl smoke val={dinfo['val']} inducing={dinfo['num_inducing']}")

    # Uncertainty must be positive on held-out windows.
    dkl.eval()
    import gpytorch

    with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4):
        xb = torch.from_numpy(va[0][:32])
        tb = torch.from_numpy(va[2][:32])
        f = dkl.latent(xb, tb)
        var = f.variance.clamp_min(0.0).numpy()
        noise = float(dkl.likelihood.noise.reshape(-1)[0].detach())
    sigma = np.sqrt(np.maximum(var + noise, 1e-12)) * float(cfg.data.rul_cap)
    print(f"sigma mean={sigma.mean():.2f} min={sigma.min():.4f}")
    assert np.all(sigma > 0), "sigma must be > 0"
    print("Step 5 OK (smoke)")


if __name__ == "__main__":
    main()
