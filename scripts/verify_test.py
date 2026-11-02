"""Verify test-set generalization without the service layer.

Uses test-file truth (never shown to agents).
Run: python scripts/verify_test.py --mode both
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
from scipy.stats import norm

from ehagent.calibrate import MondrianConformal, calibration_path
from ehagent.config import load_config
from ehagent.data import last_windows, load_all_subsets
from ehagent.metrics import coverage, mean_width, rmse
from ehagent.predictor import Predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["multitask", "single", "both"], default="both")
    args = ap.parse_args()
    cfg = load_config()
    datasets = load_all_subsets(cfg)
    modes = [True, False] if args.mode == "both" else [args.mode == "multitask"]
    cap = float(cfg.data.rul_cap)
    for multitask in modes:
        try:
            predictor = Predictor(cfg, family="dkl", multitask=multitask)
        except FileNotFoundError as exc:
            print(f"skip multitask={multitask}: {exc}")
            continue
        conf = MondrianConformal.load(calibration_path(cfg, predictor.tag))
        alpha = float(cfg.calibration.alpha)
        lt = float(conf.lower_tail(alpha)) if hasattr(conf, "lower_tail") else alpha / 2.0
        z = float(norm.ppf(1 - lt))
        print(f"model={predictor.tag} z={z:.3f} (lower_tail={lt:.4f})")
        for s in predictor.subsets:
            X, y_true, _ = last_windows(datasets[s], "test")
            pred = predictor.predict(s, X)
            mu, sig = pred["mu"], pred["sigma"]
            lo, hi, half = conf.interval(s, mu, sig, alpha)
            y_cap = np.minimum(y_true, cap)
            print(f"  {s}: RMSEcapped {rmse(y_cap, mu):.2f} RMSEuncapped {rmse(y_true, mu):.2f} "
                  f"coverage {coverage(y_cap, lo, hi):.3f} width {mean_width(lo, hi):.1f}")


if __name__ == "__main__":
    main()
