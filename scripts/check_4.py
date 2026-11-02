"""Step 4 exit check: tiny CNN smoke on CPU (<2 min, no API key).

Two epochs on a few hundred FD001 windows, just to catch shape/wiring errors.
Full baseline: python -m ehagent.train baseline --mode multitask
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ehagent.config import get_device, load_config, set_seed
from ehagent.data import load_subset, training_windows
from ehagent.train import arrays_for, train_cnn


def main() -> None:
    cfg = load_config()
    set_seed(int(cfg.project.seed))
    cfg.train.pretrain_epochs = 2
    cfg.train.num_workers = 0
    device = torch.device("cpu")
    print(f"device={device}")

    subsets = ["FD001"]
    datasets = {s: load_subset(cfg, s) for s in subsets}
    tr_full = arrays_for(cfg, datasets, subsets, "train")
    va_full = arrays_for(cfg, datasets, subsets, "val")
    ntr, nva = min(512, len(tr_full[1])), min(256, len(va_full[1]))
    tr = (tr_full[0][:ntr], tr_full[1][:ntr], tr_full[2][:ntr])
    va = (va_full[0][:nva], va_full[1][:nva], va_full[2][:nva])
    print(f"smoke windows: train={len(tr[1])} val={len(va[1])} feat={tr[0].shape[1:]}")
    assert tr[0].shape[1:] == (30, 14), tr[0].shape

    model, ek, info = train_cnn(cfg, tr, va, 1, False, device)
    print(f"smoke val_rmse={info['val_rmse']:.2f}")
    assert np.isfinite(info["val_rmse"])
    print("Step 4 OK (smoke)")


if __name__ == "__main__":
    main()
