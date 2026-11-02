"""Mondrian split-conformal calibration: separate lower and upper band edges.

Held-out calibration engines are cut at random points (like the test set). Each
edge gets its own signed score:

    lower edge: (mu - y) / sigma   (large when the model claims too much life: unsafe)
    upper edge: (y - mu) / sigma

The miss budget alpha splits by ``lower_share`` (config ``lower_tail_share``):
0.5 is symmetric (5%/5% for a 90% band); 0.25 allows 2.5% below and 7.5% above,
so misses land mostly on the safe side. Either way it stays a 90% band.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .config import Config, ensure_dir, load_config, path_for
from .data import cut_windows, load_all_subsets

FORMAT = "asymmetric-v1"


def calibration_path(cfg: Config, tag: str) -> Path:
    return path_for(cfg, "artifacts_dir", "calibration", f"{tag}.json")


def _akey(alpha: float) -> str:
    return f"{float(alpha):.4f}"


class MondrianConformal:
    def __init__(self, alphas: list[float], bins: list[float], min_group_size: int = 30, lower_share: float = 0.5):
        self.alphas = [float(a) for a in alphas]
        self.bins = [float(b) for b in bins]
        self.min_group = int(min_group_size)
        self.lower_share = float(lower_share)
        if not 0.0 < self.lower_share < 1.0:
            raise ValueError("lower_share must be between 0 and 1")
        self.tables: dict[str, dict] = {}
        self.epi_quantiles: dict[str, list[float]] = {}
        self.n_scores: dict[str, int] = {}

    # ------------------------------------------------------------------ fitting
    def _bin(self, mu) -> np.ndarray:
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        return np.clip(np.digitize(mu, self.bins[1:-1]), 0, len(self.bins) - 2)

    @staticmethod
    def _quantile(scores: np.ndarray, level: float) -> float:
        """Finite-sample conformal quantile at ``level``; floored at 0 so mu stays inside."""
        n = len(scores)
        lv = min(1.0, math.ceil((n + 1) * level) / n)
        return max(0.0, float(np.quantile(scores, lv, method="higher")))

    def _pair(self, s_lo: np.ndarray, s_hi: np.ndarray, levels: tuple[float, float]) -> list[float] | None:
        if len(s_lo) == 0:
            return None
        return [self._quantile(s_lo, levels[0]), self._quantile(s_hi, levels[1])]

    def lower_tail(self, alpha: float | None = None) -> float:
        """Share of engines allowed below the band (the unsafe miss) at this alpha."""
        alpha = self.alphas[0] if alpha is None else float(alpha)
        return alpha * self.lower_share

    def fit(self, subsets: np.ndarray, mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
            sigma_epi: np.ndarray) -> "MondrianConformal":
        subsets = np.asarray(subsets)
        mu = np.asarray(mu, dtype=float)
        y = np.asarray(y, dtype=float)
        sig = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
        s_lo = (mu - y) / sig
        s_hi = (y - mu) / sig
        for alpha in self.alphas:
            levels = (1.0 - alpha * self.lower_share, 1.0 - alpha * (1.0 - self.lower_share))
            table: dict = {"global": self._pair(s_lo, s_hi, levels), "subsets": {}}
            for s in np.unique(subsets):
                m = subsets == s
                entry = {"subset": self._pair(s_lo[m], s_hi[m], levels), "bins": []}
                b = self._bin(mu[m])
                for k in range(len(self.bins) - 1):
                    mk = b == k
                    entry["bins"].append(self._pair(s_lo[m][mk], s_hi[m][mk], levels) if mk.sum() >= self.min_group else None)
                table["subsets"][str(s)] = entry
            self.tables[_akey(alpha)] = table
        for s in np.unique(subsets):
            m = subsets == s
            self.epi_quantiles[str(s)] = np.quantile(np.asarray(sigma_epi, float)[m], np.linspace(0, 1, 101)).tolist()
            self.n_scores[str(s)] = int(m.sum())
        return self

    # --------------------------------------------------------------------- use
    def q_for(self, subset: str, mu, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper multipliers (in sigma units) for each prediction."""
        alpha = self.alphas[0] if alpha is None else float(alpha)
        table = self.tables[_akey(alpha)]
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        g_lo, g_hi = table["global"]
        q_lo = np.full(mu.shape, float(g_lo))
        q_hi = np.full(mu.shape, float(g_hi))
        entry = table["subsets"].get(subset)
        if entry is None:
            return q_lo, q_hi
        b = self._bin(mu)
        for k, pair in enumerate(entry["bins"]):
            pair = pair if pair is not None else entry["subset"]
            if pair is not None:
                q_lo[b == k] = float(pair[0])
                q_hi[b == k] = float(pair[1])
        return q_lo, q_hi

    def interval(self, subset: str, mu, sigma, alpha: float | None = None):
        """(lo, hi, lower half-width). ``lo`` clips at 0 cycles.

        The lower half-width feeds the failure probability: the decision engine
        divides it by z so the Gaussian's lower tail matches the calibrated edge.
        """
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
        q_lo, q_hi = self.q_for(subset, mu, alpha)
        half_lo = q_lo * sigma
        return np.maximum(mu - half_lo, 0.0), mu + q_hi * sigma, half_lo

    def epistemic_percentile(self, subset: str, sigma_epi) -> np.ndarray:
        qs = self.epi_quantiles.get(subset)
        sigma_epi = np.atleast_1d(np.asarray(sigma_epi, dtype=float))
        if not qs:
            return np.full(sigma_epi.shape, np.nan)
        qs = np.maximum.accumulate(np.asarray(qs, dtype=float))   # guard against tiny non-monotonicity
        return np.interp(sigma_epi, qs, np.linspace(0, 100, len(qs)))

    # ---------------------------------------------------------------------- io
    def to_json(self) -> dict:
        return {"format": FORMAT, "alphas": self.alphas, "bins": self.bins, "min_group_size": self.min_group,
                "lower_share": self.lower_share, "tables": self.tables, "epi_quantiles": self.epi_quantiles,
                "n_scores": self.n_scores}

    def save(self, path: Path) -> None:
        ensure_dir(Path(path).parent)
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "MondrianConformal":
        if not Path(path).exists():
            raise FileNotFoundError(f"Calibration file not found: {path}. Run `python -m ehagent.calibrate` first.")
        d = json.loads(Path(path).read_text())
        if d.get("format") != FORMAT:
            raise ValueError(f"{path} uses an older calibration format. Re-run `python -m ehagent.calibrate --mode both`.")
        obj = cls(d["alphas"], d["bins"], d.get("min_group_size", 30), d.get("lower_share", 0.5))
        obj.tables = d["tables"]
        obj.epi_quantiles = d.get("epi_quantiles", {})
        obj.n_scores = d.get("n_scores", {})
        return obj


def collect_cut_predictions(cfg: Config, predictor, datasets: dict, split: str, seed_offset: int = 0):
    rng = np.random.default_rng(int(cfg.project.seed) + seed_offset)
    subs, mus, sigs, epis, ys = [], [], [], [], []
    for s in predictor.subsets:
        X, y, _ = cut_windows(datasets[s], split, int(cfg.calibration.cuts_per_engine), int(cfg.data.window),
                              int(cfg.data.min_history), float(cfg.data.rul_cap), rng)
        if len(y) == 0:
            continue
        pred = predictor.predict(s, X)
        subs.append(np.full(len(y), s))
        mus.append(pred["mu"])
        sigs.append(pred["sigma"])
        epis.append(pred["sigma_epi"])
        ys.append(y)
    return (np.concatenate(subs), np.concatenate(mus), np.concatenate(sigs), np.concatenate(epis), np.concatenate(ys))


def calibrate(cfg: Config, predictor, datasets: dict) -> MondrianConformal:
    subs, mu, sig, epi, y = collect_cut_predictions(cfg, predictor, datasets, "calib")
    cc = cfg.calibration
    conf = MondrianConformal([float(cc.alpha), float(cc.alpha_conservative)], list(cc.bins), int(cc.min_group_size),
                             float(cc.get("lower_tail_share", 0.5)))
    return conf.fit(subs, mu, sig, y, epi)


def main(argv: list[str] | None = None) -> None:
    from .predictor import Predictor

    ap = argparse.ArgumentParser(description="Conformal calibration of RUL bands")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["multitask", "single", "both"], default="both")
    ap.add_argument("--family", default=None, help="dkl | ensemble (default: config)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    datasets = load_all_subsets(cfg)
    modes = [True, False] if args.mode == "both" else [args.mode == "multitask"]
    for multitask in modes:
        try:
            predictor = Predictor(cfg, family=args.family, multitask=multitask)
        except FileNotFoundError as exc:
            print(f"skip ({'multitask' if multitask else 'single'}): {exc}")
            continue
        conf = calibrate(cfg, predictor, datasets)
        out = calibration_path(cfg, predictor.tag)
        conf.save(out)
        # Sanity check on validation engines (never used for fitting q).
        subs, mu, sig, _, y = collect_cut_predictions(cfg, predictor, datasets, "val", seed_offset=1)
        target = 1 - float(cfg.calibration.alpha)
        report = []
        for s in np.unique(subs):
            m = subs == s
            lo, hi, _ = conf.interval(str(s), mu[m], sig[m])
            report.append(f"{s}: val coverage {np.mean((y[m] >= lo) & (y[m] <= hi)):.3f} (target {target:.2f}), "
                          f"below band {np.mean(y[m] < lo):.3f} (target <= {conf.lower_tail():.3f})")
        print(f"saved {out}\n  " + "\n  ".join(report))


if __name__ == "__main__":
    main()
