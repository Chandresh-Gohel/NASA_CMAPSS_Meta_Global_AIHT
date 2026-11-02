"""Step 7 exit check: calibration coverage on validation cuts (<2 min, no API).

Needs trained models. Full run: python -m ehagent.calibrate --mode both
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ehagent.calibrate import calibrate, collect_cut_predictions
from ehagent.config import load_config
from ehagent.data import load_all_subsets
from ehagent.predictor import Predictor


def main() -> None:
    cfg = load_config()
    datasets = load_all_subsets(cfg)
    for multitask in (True, False):
        try:
            predictor = Predictor(cfg, family="dkl", multitask=multitask)
        except FileNotFoundError as exc:
            print(f"skip ({'multitask' if multitask else 'single'}): {exc}")
            continue
        conf = calibrate(cfg, predictor, datasets)
        subs, mu, sig, _, y = collect_cut_predictions(cfg, predictor, datasets, "val", seed_offset=1)
        print(f"mode={'multitask' if multitask else 'single'}:")
        for s in sorted(set(subs.tolist())):
            m = subs == s
            lo, hi, _ = conf.interval(str(s), mu[m], sig[m])
            cov = float(np.mean((y[m] >= lo) & (y[m] <= hi)))
            print(f"  {s}: val coverage {cov:.3f} (target 0.90)")
            assert 0.87 <= cov <= 0.93, f"{s} coverage {cov:.3f} outside 0.87-0.93"
    print("Step 7 OK")


if __name__ == "__main__":
    main()
