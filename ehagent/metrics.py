"""Evaluation maths: point accuracy, band calibration and selective prediction."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def rmse(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else float("nan")


def mae(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(np.mean(np.abs(p - y))) if len(y) else float("nan")


def nasa_score(y_true, y_pred) -> float:
    """PHM08 score: over-estimating remaining life (late) costs more than under-estimating."""
    d = _arr(y_pred) - _arr(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def coverage(y, lo, hi) -> float:
    y = _arr(y)
    return float(np.mean((y >= _arr(lo)) & (y <= _arr(hi)))) if len(y) else float("nan")


def mean_width(lo, hi) -> float:
    return float(np.mean(_arr(hi) - _arr(lo))) if len(_arr(lo)) else float("nan")


def gaussian_nll(y, mu, sigma) -> float:
    s = np.maximum(_arr(sigma), 1e-6)
    z = (_arr(y) - _arr(mu)) / s
    return float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * z ** 2))


def gaussian_crps(y, mu, sigma) -> float:
    s = np.maximum(_arr(sigma), 1e-6)
    z = (_arr(y) - _arr(mu)) / s
    crps = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def binned_coverage(y, lo, hi, edges) -> list[dict]:
    y, lo, hi = _arr(y), _arr(lo), _arr(hi)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b)
        out.append({"rul_from": float(a), "rul_to": float(b), "n": int(m.sum()),
                    "coverage": float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) if m.any() else None})
    return out


def risk_coverage_curve(y, mu, uncertainty, n_points: int = 20) -> list[dict]:
    """Error on the most-confident slice of engines, widening to the full set."""
    y, mu, u = _arr(y), _arr(mu), _arr(uncertainty)
    order = np.argsort(u)
    out = []
    for frac in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(1, int(round(frac * len(y))))
        idx = order[:k]
        out.append({"coverage": float(frac), "rmse": rmse(y[idx], mu[idx])})
    return out


def aurc(curve: list[dict]) -> float:
    xs = [c["coverage"] for c in curve]
    ys = [c["rmse"] for c in curve]
    return float(sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)))
