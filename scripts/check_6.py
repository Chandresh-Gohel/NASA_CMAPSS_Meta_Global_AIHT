"""Step 6 exit check: tiny multitask LMC smoke on CPU (<2 min, no API).

Full run: python -m ehagent.train dkl --mode multitask
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ehagent.config import load_config, set_seed
from ehagent.data import load_subset
from ehagent.train import arrays_for, train_cnn, train_dkl


def main() -> None:
    cfg = load_config()
    set_seed(int(cfg.project.seed))
    cfg.train.dkl_epochs = 1
    cfg.train.pretrain_epochs = 1
    cfg.train.dkl_batch_size = 128
    cfg.train.batch_size = 128
    cfg.model.inducing_points = 32
    cfg.model.num_latents = 2
    device = torch.device("cpu")
    subsets = ["FD001", "FD002"]
    datasets = {s: load_subset(cfg, s) for s in subsets}
    tr_full = arrays_for(cfg, datasets, subsets, "train")
    va_full = arrays_for(cfg, datasets, subsets, "val")
    ntr, nva = min(512, len(tr_full[1])), min(256, len(va_full[1]))
    tr = (tr_full[0][:ntr], tr_full[1][:ntr], tr_full[2][:ntr])
    va = (va_full[0][:nva], va_full[1][:nva], va_full[2][:nva])
    print(f"smoke multitask windows: train={len(tr[1])} val={len(va[1])}")

    cnn, ek, _ = train_cnn(cfg, tr, va, 2, False, device)
    dkl, dinfo = train_dkl(cfg, cnn, tr, va, 2, "lmc", device)
    print(f"dkl smoke val={dinfo['val']}")
    import numpy as np

    corr = dkl.gp.task_correlation()
    print("task-correlation:\n", np.round(corr, 3))
    assert corr.shape == (2, 2)
    print("Step 6 OK (smoke)")


if __name__ == "__main__":
    main()
