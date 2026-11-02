"""Cross-fitted calibration check: every engine outside the training split takes a turn as validation.

    python scripts/crossfit_calibration.py --mode multitask          # or: single | average
    python scripts/crossfit_calibration.py --mode single --write     # also save the final calibration file

Why: the 10 validation engines of FD001 and FD003 are too few to judge a band. This pools the calibration split
(15%) and the validation split (10%) of the training-file engines, cut exactly as `ehagent.calibrate` and
`scripts/check_calibration.py` cut them, and splits the pooled engines into K folds. Each fold's bands come from
quantiles fitted on the other folds, so every number printed here is out of sample. The test files and the replay
engines are never touched.

--mode average averages the multitask and single-task models of this repo (both must exist). A two-model mixture:
mean = (mu1 + mu2) / 2, variance = (s1^2 + s2^2) / 2 + (mu1 - mu2)^2 / 4.

--write fits the final quantiles on all pooled engines and saves them as the model's calibration file, keeping the
old one as <tag>.bak.json. Re-run precompute afterwards. Caveat to state in the README: the validation engines were
also used for early stopping, so these bands can be slightly optimistic.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ehagent.calibrate import MondrianConformal, calibration_path  # noqa: E402
from ehagent.config import load_config  # noqa: E402
from ehagent.data import cut_windows, load_all_subsets  # noqa: E402
from ehagent.predictor import Predictor  # noqa: E402

ROWS = ((0, 30), (30, 60), (60, 90), (90, None))


class AveragePredictor:
    """Mean of the multitask and single-task models (prediction only)."""

    def __init__(self, cfg):
        self.parts = [Predictor(cfg, multitask=True), Predictor(cfg, multitask=False)]
        self.subsets = self.parts[0].subsets
        self.tag = f"{self.parts[0].family}_average"

    def predict(self, subset: str, X: np.ndarray) -> dict:
        a, b = (p.predict(subset, X) for p in self.parts)
        spread = ((a["mu"] - b["mu"]) / 2.0) ** 2
        epi2 = (a["sigma_epi"] ** 2 + b["sigma_epi"] ** 2) / 2.0 + spread
        alea2 = (a["sigma_alea"] ** 2 + b["sigma_alea"] ** 2) / 2.0
        return {"mu": (a["mu"] + b["mu"]) / 2.0, "sigma_epi": np.sqrt(epi2), "sigma_alea": np.sqrt(alea2),
                "sigma": np.sqrt(epi2 + alea2)}


def collect(cfg, predictor, datasets) -> dict:
    """Cuts of the calibration split (seed) and the validation split (seed + 1), as calibrate/check_calibration do."""
    cap, window, mh = float(cfg.data.rul_cap), int(cfg.data.window), int(cfg.data.min_history)
    cuts = int(cfg.calibration.cuts_per_engine)
    out = {k: [] for k in ("subset", "engine", "mu", "sigma", "epi", "y")}
    for split, offset in (("calib", 0), ("val", 1)):
        rng = np.random.default_rng(int(cfg.project.seed) + offset)
        for s in predictor.subsets:
            X, y, index = cut_windows(datasets[s], split, cuts, window, mh, cap, rng)
            if len(y) == 0:
                continue
            p = predictor.predict(s, X)
            out["subset"].append(np.full(len(y), s))
            out["engine"].append(np.array([e for e, _ in index]))
            out["mu"].append(p["mu"]); out["sigma"].append(p["sigma"]); out["epi"].append(p["sigma_epi"])
            out["y"].append(y)
    return {k: np.concatenate(v) for k, v in out.items()}


def new_conformal(cfg) -> MondrianConformal:
    cc = cfg.calibration
    return MondrianConformal([float(cc.alpha), float(cc.alpha_conservative)], list(cc.bins), int(cc.min_group_size),
                             float(cc.get("lower_tail_share", 0.5)))


def crossfit(cfg, d: dict, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold lower edge, upper edge, for every cut. Folds are drawn per subset, by engine."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(d["y"]), dtype=int)
    for s in np.unique(d["subset"]):
        engines = np.unique(d["engine"][d["subset"] == s])
        assign = {e: i % k for i, e in enumerate(rng.permutation(engines))}
        m = d["subset"] == s
        fold[m] = [assign[e] for e in d["engine"][m]]
    lo, hi = np.empty(len(d["y"])), np.empty(len(d["y"]))
    for f in range(k):
        tr, te = fold != f, fold == f
        conf = new_conformal(cfg).fit(d["subset"][tr], d["mu"][tr], d["sigma"][tr], d["y"][tr], d["epi"][tr])
        for s in np.unique(d["subset"][te]):
            m = te & (d["subset"] == s)
            lo[m], hi[m], _ = conf.interval(str(s), d["mu"][m], d["sigma"][m])
    return lo, hi, fold


def report(cfg, d: dict, lo: np.ndarray, hi: np.ndarray, tag: str, k: int, n_boot: int) -> None:
    target = 1 - float(cfg.calibration.alpha)
    budget = float(cfg.calibration.alpha) * float(cfg.calibration.get("lower_tail_share", 0.5))
    cap = float(cfg.data.rul_cap)
    inside = (d["y"] >= lo) & (d["y"] <= hi)
    below = d["y"] < lo
    boot = np.random.default_rng(0)
    print(f"model {tag} | {k}-fold cross-fit over calibration + validation engines | target coverage {target:.2f} | "
          f"below-band budget {budget:.3f}\n")
    pooled = []
    for s in np.unique(d["subset"]):
        m = d["subset"] == s
        eng = d["engine"][m]
        ids = np.unique(eng)
        groups = [np.flatnonzero(eng == e) for e in ids]
        ins = inside[m]
        boots = [ins[np.concatenate([groups[i] for i in boot.integers(0, len(ids), len(ids))])].mean() for _ in range(n_boot)]
        b_lo, b_hi = np.quantile(boots, [0.05, 0.95])
        verdict = "ok" if b_lo <= target <= b_hi else ("wider than needed (safe side)" if b_lo > target else "TOO NARROW")
        err = d["mu"][m] - d["y"][m]
        pooled.append(err)
        print(f"{s}: coverage {ins.mean():.3f}  interval [{b_lo:.3f}, {b_hi:.3f}]  below {below[m].mean():.3f}  "
              f"RMSE {np.sqrt(np.mean(err ** 2)):.2f}  mean width {np.mean(hi[m] - lo[m]):.1f}  "
              f"({len(ids)} engines, {int(m.sum())} cuts) -> {verdict}")
        for a, b in ROWS:
            r = m & (d["y"] >= a) & (d["y"] < (b if b is not None else cap + 1))
            if r.any():
                print(f"    true RUL {a:>3}-{b if b is not None else int(cap):<3}  coverage {inside[r].mean():.3f}  "
                      f"below {below[r].mean():.3f}  mean width {np.mean(hi[r] - lo[r]):5.1f} cycles  "
                      f"({int(r.sum())} cuts, {len(np.unique(d['engine'][r]))} engines)")
        print()
    allerr = np.concatenate(pooled)
    print(f"pooled RMSE over all cuts: {np.sqrt(np.mean(allerr ** 2)):.2f} ({len(allerr)} cuts)")
    print(f"'below' = true RUL under the band (the unsafe miss); the budget is {budget:.3f}. "
          "The 0-30 and 30-60 rows matter most.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["multitask", "single", "average"], default="multitask")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--write", action="store_true", help="fit on all pooled engines and save the calibration file")
    args = ap.parse_args()

    cfg = load_config()
    datasets = load_all_subsets(cfg)
    predictor = AveragePredictor(cfg) if args.mode == "average" else Predictor(cfg, multitask=args.mode == "multitask")
    d = collect(cfg, predictor, datasets)
    lo, hi, _ = crossfit(cfg, d, args.folds, int(cfg.project.seed) + 7)
    report(cfg, d, lo, hi, predictor.tag, args.folds, args.boot)

    if args.write:
        conf = new_conformal(cfg).fit(d["subset"], d["mu"], d["sigma"], d["y"], d["epi"])
        out = calibration_path(cfg, predictor.tag)
        if out.exists():
            backup = out.with_name(out.stem + ".bak.json")
            shutil.copy2(out, backup)
            print(f"kept the previous calibration as {backup}")
        conf.save(out)
        print(f"saved {out} (fitted on calibration + validation engines). Re-run precompute next.")
        if args.mode == "average":
            print("Note: the app and the service don't load the averaged model yet; this file is for that integration.")


if __name__ == "__main__":
    main()
