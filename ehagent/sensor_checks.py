"""Sensor integrity checks: out-of-range values, sudden jumps and stuck sensors.

The baselines come from the training fleet, so a check only fires on behaviour
rarer than anything seen between healthy and failed training engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .constants import sensor_name


@dataclass
class SensorCheckStats:
    sensors: list[str]
    z_abs_max: np.ndarray   # (F,) largest |z| seen in training (condition-normalised)
    dz_p999: np.ndarray     # (F,) 99.9th percentile of |z_t - z_{t-1}|
    flat_rate: np.ndarray   # (F,) share of training windows whose raw value never moved
    stuck_window: int


def fit_check_stats(records: Iterable[Any], sensors: list[str], check_cfg: Any) -> SensorCheckStats:
    """Learn check baselines from records exposing ``raw`` and ``norm`` arrays (T, F)."""
    k = int(check_cfg.get("stuck_min_cycles", 10))
    n_feat = len(sensors)
    z_abs_max = np.zeros(n_feat)
    dz_chunks: list[np.ndarray] = []
    flat_counts = np.zeros(n_feat)
    total_windows = 0
    for rec in records:
        z = np.asarray(rec.norm, dtype=float)
        raw = np.asarray(rec.raw, dtype=float)
        if len(z) == 0:
            continue
        z_abs_max = np.maximum(z_abs_max, np.abs(z).max(axis=0))
        if len(z) > 1:
            dz_chunks.append(np.abs(np.diff(z, axis=0)))
        if len(raw) >= k:
            win = sliding_window_view(raw, k, axis=0)          # (T-k+1, F, k)
            ptp = win.max(axis=-1) - win.min(axis=-1)
            flat_counts += (ptp == 0).sum(axis=0)
            total_windows += ptp.shape[0]
    dz = np.concatenate(dz_chunks, axis=0) if dz_chunks else np.ones((1, n_feat))
    dz_p999 = np.quantile(dz, 0.999, axis=0)
    dz_p999 = np.where(dz_p999 < 1e-6, 1e-6, dz_p999)
    flat_rate = flat_counts / max(total_windows, 1)
    return SensorCheckStats(list(sensors), z_abs_max, dz_p999, flat_rate, k)


def _flag(sensor: str, check: str, detail: str) -> dict:
    return {"sensor": sensor, "name": sensor_name(sensor), "check": check, "detail": detail, "severity": "high"}


def run_sensor_checks(raw_hist: np.ndarray, norm_hist: np.ndarray, stats: SensorCheckStats, check_cfg: Any) -> list[dict]:
    """Check one engine's recent cycles. An empty list means the data looks trustworthy."""
    n_cycles = len(raw_hist)
    if n_cycles == 0:
        return []
    lookback = min(int(check_cfg.get("lookback", 15)), n_cycles)
    margin = float(check_cfg.get("z_abs_max_margin", 1.5))
    jump_mult = float(check_cfg.get("jump_multiplier", 2.0))
    flat_max = float(check_cfg.get("flat_rate_max", 0.01))
    z = np.asarray(norm_hist[-lookback:], dtype=float)
    flags: list[dict] = []
    for j, sensor in enumerate(stats.sensors):
        zj = z[:, j]
        limit = max(float(stats.z_abs_max[j]) * margin, 4.0)
        peak = float(np.abs(zj).max())
        if peak > limit:
            flags.append(_flag(sensor, "out_of_range", f"reading is {peak:.1f} standard deviations from normal; the training fleet never exceeded {limit:.1f}"))
        if len(zj) >= 2:
            jump = float(np.abs(np.diff(zj)).max())
            jump_limit = float(stats.dz_p999[j]) * jump_mult
            if jump > jump_limit:
                flags.append(_flag(sensor, "jump", f"cycle-to-cycle change of {jump:.1f} standard deviations; normal changes stay below {jump_limit:.1f}"))
        k = int(stats.stuck_window)
        if n_cycles >= k:
            seg = np.asarray(raw_hist[-k:, j], dtype=float)
            if float(seg.max() - seg.min()) == 0.0 and float(stats.flat_rate[j]) < flat_max:
                flags.append(_flag(sensor, "stuck", f"value has not changed for the last {k} cycles"))
    return flags
