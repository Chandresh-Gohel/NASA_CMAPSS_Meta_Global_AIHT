"""Step 7 diagnostic: per-subset/per-bin calib q against val residuals (no config change).

Run: python scripts/diag_7.py --mode multitask
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np

from ehagent.calibrate import collect_cut_predictions, calibration_path, MondrianConformal
from ehagent.config import load_config
from ehagent.data import load_all_subsets
from ehagent.predictor import Predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["multitask", "single"], default="multitask")
    args = ap.parse_args()
    cfg = load_config()
    datasets = load_all_subsets(cfg)
    predictor = Predictor(cfg, family="dkl", multitask=(args.mode == "multitask"))
    conf = MondrianConformal.load(calibration_path(cfg, predictor.tag))
    for split, seed in (("calib", 0), ("val", 1)):
        subs, mu, sig, epi, y = collect_cut_predictions(cfg, predictor, datasets, split, seed_offset=seed)
        print(f"--- {split} ---")
        for s in sorted(set(subs.tolist())):
            m = subs == s
            scores = np.abs(y[m] - mu[m]) / np.maximum(sig[m], 1e-6)
            lo, hi, _ = conf.interval(str(s), mu[m], sig[m])
            cov = float(np.mean((y[m] >= lo) & (y[m] <= hi)))
            b = conf._bin(mu[m])
            nb = [(int((b == k).sum())) for k in range(len(cfg.calibration.bins) - 1)]
            print(f"{s}: n={int(m.sum())} cov={cov:.3f} score_p50={float(np.median(scores)):.3f} "
                  f"score_p90={float(np.quantile(scores, 0.9)):.3f} mu_med={float(np.median(mu[m])):.1f} "
                  f"sig_med={float(np.median(sig[m])):.1f} bins_n={nb} q={conf.tables['0.1000']['subsets'][str(s)]}")


if __name__ == "__main__":
    main()
