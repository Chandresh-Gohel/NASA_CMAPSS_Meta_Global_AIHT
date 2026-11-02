"""Calibration check with honest error bars: validation coverage, bootstrapped over engines.

    python scripts/check_calibration.py --mode multitask
    python scripts/check_calibration.py --mode single

Cuts the validation engines exactly like `python -m ehagent.calibrate` does (same
seed, same subset order), so the point estimates match its printout. Engines, not
windows, are resampled: one engine's cuts correlate strongly, so a window-level
interval would be far too narrow. Validation engines only; test files untouched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ehagent.calibrate import MondrianConformal, calibration_path  # noqa: E402
from ehagent.config import load_config  # noqa: E402
from ehagent.data import cut_windows, load_all_subsets  # noqa: E402
from ehagent.predictor import Predictor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["multitask", "single"], default="multitask")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    cfg = load_config()
    datasets = load_all_subsets(cfg)
    predictor = Predictor(cfg, multitask=args.mode == "multitask")
    conf = MondrianConformal.load(calibration_path(cfg, predictor.tag))
    target = 1 - float(cfg.calibration.alpha)
    cap = float(cfg.data.rul_cap)
    rng = np.random.default_rng(int(cfg.project.seed) + 1)   # same cuts as calibrate's validation check
    boot_rng = np.random.default_rng(0)

    print(f"model {predictor.tag} | target coverage {target:.2f} | 90% bootstrap interval over engines\n")
    for s in predictor.subsets:
        X, y, index = cut_windows(datasets[s], "val", int(cfg.calibration.cuts_per_engine), int(cfg.data.window),
                                  int(cfg.data.min_history), cap, rng)
        if len(y) == 0:
            print(f"{s}: no validation windows")
            continue
        pred = predictor.predict(s, X)
        lo, hi, _ = conf.interval(s, pred["mu"], pred["sigma"])
        inside = (y >= lo) & (y <= hi)

        engines = np.array([e for e, _ in index])
        ids = np.unique(engines)
        per_engine = {e: inside[engines == e] for e in ids}
        boots = [np.concatenate([per_engine[e] for e in boot_rng.choice(ids, size=len(ids))]).mean()
                 for _ in range(args.boot)]
        b_lo, b_hi = np.quantile(boots, [0.05, 0.95])
        if b_lo <= target <= b_hi:
            verdict = "ok"
        elif b_lo > target:
            verdict = "wider than needed (safe side)"
        else:
            verdict = "TOO NARROW"
        below = y < lo                     # over-predicted remaining life: the unsafe miss
        rmse = float(np.sqrt(np.mean((pred["mu"] - y) ** 2)))
        print(f"{s}: coverage {inside.mean():.3f}  interval [{b_lo:.3f}, {b_hi:.3f}]  below {below.mean():.3f}  "
              f"RMSE {rmse:.2f}  mean width {np.mean(hi - lo):.1f}  ({len(ids)} engines, {len(y)} cuts) -> {verdict}")

        for a, b in ((0, 30), (30, 60), (60, 90), (90, cap + 1)):
            m = (y >= a) & (y < b)
            if m.any():
                n_eng = len(np.unique(engines[m]))
                print(f"    true RUL {a:>3.0f}-{min(b, cap):<3.0f}  coverage {inside[m].mean():.3f}  "
                      f"below {below[m].mean():.3f}  mean width {np.mean(hi[m] - lo[m]):5.1f} cycles  "
                      f"({int(m.sum())} cuts, {n_eng} engines)")
        print()

    print("Pass: the target sits inside each subset's interval.")
    print(f"'below' = true RUL under the band (the model over-predicted). Aim for at most {conf.lower_tail():.3f};")
    print("misses there are the unsafe kind, and the 0-30 and 30-60 rows matter most.")


if __name__ == "__main__":
    main()
