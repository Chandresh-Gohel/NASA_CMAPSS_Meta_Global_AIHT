"""Step 9 exit check: IG adds up within ~10%; median faithfulness above 1 on 30 engines.

Run: python scripts/check_9.py [--mode multitask|single|both] (default: both)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from ehagent.attribution import deletion_test, integrated_gradients
from ehagent.config import load_config
from ehagent.data import last_windows, load_all_subsets
from ehagent.predictor import Predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["multitask", "single", "both"], default="both")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config()
    modes = [True, False] if args.mode == "both" else [args.mode == "multitask"]
    for multitask in modes:
        print(f"--- multitask={multitask} ---")
        run_mode(cfg, multitask, args.n)
    print("Step 9 OK")


def run_mode(cfg, multitask: bool, n: int) -> None:
    datasets = load_all_subsets(cfg)
    predictor = Predictor(cfg, family="dkl", multitask=multitask)
    print(f"model={predictor.tag}")

    steps = int(cfg.attribution.ig_steps)
    top_k = int(cfg.attribution.deletion_top_k)
    trials = int(cfg.attribution.deletion_random_trials)
    ratios, sum_errs, tested = [], [], 0
    for subset in predictor.subsets:
        sd = datasets[subset]
        _, _, eids = last_windows(sd, "test")
        for eid in eids:
            if tested >= n:
                break
            rec = sd.engines[eid]
            x = rec.norm[-int(cfg.data.window):]
            if len(x) < int(cfg.data.window):
                pad = np.repeat(x[:1], int(cfg.data.window) - len(x), axis=0)
                x = np.concatenate([pad, x])
            base = sd.healthy_baseline
            attr, fx, fb = integrated_gradients(predictor, subset, x.astype(np.float32), base.astype(np.float32), steps=steps)
            denom = max(abs(fx - fb), 1e-6)
            sum_err = abs(attr.sum() - (fx - fb)) / denom
            sum_errs.append(sum_err)
            per_sensor = attr.sum(axis=0)
            dt = deletion_test(predictor, subset, x.astype(np.float32), base.astype(np.float32), per_sensor, top_k=top_k, trials=trials)
            ratios.append(dt["faithfulness_ratio"])
            tested += 1
            if tested <= 3:
                print(f"  {eid}: f(x)={fx:.1f} f(base)={fb:.1f} sum_err={sum_err:.3f} ratio={dt['faithfulness_ratio']}")
        if tested >= n:
            break
    ratios = np.asarray(ratios, dtype=float)
    sum_errs = np.asarray(sum_errs, dtype=float)
    print(f"engines={tested} median_sum_err={float(np.median(sum_errs)):.3f} max_sum_err={float(sum_errs.max()):.3f}")
    print(f"median_faithfulness={float(np.median(ratios)):.2f} share>1={float(np.mean(ratios > 1.0)):.2f}")
    assert float(np.median(sum_errs)) <= 0.10, f"median IG sum error {np.median(sum_errs):.3f} > 0.10"
    assert float(np.median(ratios)) > 1.0, f"median faithfulness {np.median(ratios):.2f} <= 1"


if __name__ == "__main__":
    main()
