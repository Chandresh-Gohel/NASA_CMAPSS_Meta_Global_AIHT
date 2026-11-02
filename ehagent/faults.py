"""Sensor fault injection for demos and robustness tests (stuck, drift, spike, noise)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import EngineRecord, SubsetData

FAULT_KINDS = ("stuck", "drift", "spike", "noise")


def inject_fault(rec: EngineRecord, sensor: str, kind: str, start_index: int, magnitude: float, sd: SubsetData,
                 seed: int = 0) -> EngineRecord:
    """Copy ``rec`` with a fault on ``sensor`` from ``start_index`` on (magnitude in training std)."""
    if kind not in FAULT_KINDS:
        raise ValueError(f"Unknown fault kind '{kind}'. Use one of {FAULT_KINDS}.")
    if sensor not in sd.sensors:
        raise ValueError(f"Sensor {sensor} is not among the model inputs {sd.sensors}")
    j = sd.sensors.index(sensor)
    new = rec.copy()
    raw = new.raw.copy()
    n = len(raw)
    s = int(np.clip(start_index, 0, n - 1))
    std_t = sd.normalizer.std[new.cond, j]
    if kind == "stuck":
        raw[s:, j] = raw[s, j]
    elif kind == "drift":
        steps = np.arange(1, n - s + 1)
        raw[s:, j] = raw[s:, j] + float(magnitude) * std_t[s:] * steps / 10.0
    elif kind == "spike":
        raw[s, j] = raw[s, j] + float(magnitude) * std_t[s]
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        raw[s:, j] = raw[s:, j] + rng.normal(0.0, float(magnitude) * std_t[s:])
    new.raw = raw
    new.norm = sd.normalizer.transform(raw, new.cond)
    new.fault = {"sensor": sensor, "kind": kind, "start_cycle": int(new.cycles[s]), "magnitude": float(magnitude)}
    return new


def apply_fault_spec(rec: EngineRecord, spec: dict[str, Any], sd: SubsetData) -> EngineRecord:
    """Spec from the app/tools: {"sensor": "s11", "kind": "stuck", "start_offset": 20, "magnitude": 3.0}."""
    kind = str(spec.get("kind", "stuck"))
    offset = int(spec.get("start_offset", 20))
    if kind == "spike":
        offset = min(offset, 3)
    start = max(0, rec.length - max(offset, 1))
    return inject_fault(rec, str(spec["sensor"]), kind, start, float(spec.get("magnitude", 3.0)), sd, seed=int(spec.get("seed", 0)))


def fault_key(spec: dict | None) -> str:
    if not spec:
        return "clean"
    return "|".join(f"{k}={spec[k]}" for k in sorted(spec))
