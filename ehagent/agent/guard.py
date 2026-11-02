"""Number guard: a briefing may only repeat the decision engine's action and numbers
found in tool outputs. Anything else forces a rewrite, then a deterministic template."""
from __future__ import annotations

import math
import re
from typing import Any

NUM_RE = re.compile(r"(?<![A-Za-z0-9_.\-])[-+]?\d+(?:\.\d+)?")
SMALL_INTS = {float(i) for i in range(0, 11)}


def _walk(obj: Any, pool: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        if math.isfinite(v):
            pool.add(v)
            pool.add(abs(v))
            if 0.0 <= v <= 1.0:
                pool.add(round(v * 100, 1))
                pool.add(float(round(v * 100)))
        return
    if isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            pool.add(float(m))
            pool.add(abs(float(m)))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk(v, pool)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, pool)


def evidence_numbers(evidence: Any, cfg: Any = None) -> set[float]:
    pool: set[float] = set()
    _walk(evidence, pool)
    if cfg is not None:
        pool.update({float(cfg.data.rul_cap), round(float(cfg.decision.p_inspect) * 100, 1),
                     round(float(cfg.decision.p_clear) * 100, 1), round((1 - float(cfg.calibration.alpha)) * 100, 1),
                     float(cfg.decision.epistemic_percentile_abstain), float(cfg.decision.horizon_cycles)})
    return pool


def _matches(value: float, pool: set[float], tol_abs: float = 0.6, tol_rel: float = 0.01) -> bool:
    return any(abs(value - p) <= max(tol_abs, tol_rel * abs(p)) for p in pool)


def check_numbers_in_text(text: str, pool: set[float]) -> list[str]:
    bad = []
    for m in NUM_RE.findall(text or ""):
        v = float(m)
        if v in SMALL_INTS or _matches(v, pool):
            continue
        bad.append(m)
    return bad


def check_report(report: dict, authoritative_action: str, evidence: dict, cfg: Any) -> list[str]:
    violations: list[str] = []
    if report.get("recommendation") != authoritative_action:
        violations.append(f"recommendation must be exactly {authoritative_action} (got {report.get('recommendation')})")
    pool = evidence_numbers(evidence, cfg)
    for kn in report.get("key_numbers", []) or []:
        try:
            v = float(kn.get("value"))
        except (TypeError, ValueError):
            violations.append(f"key number '{kn.get('label')}' is not numeric")
            continue
        if not _matches(v, pool):
            violations.append(f"key number '{kn.get('label')}' = {v} does not appear in the tool outputs")
    for sd in report.get("sensor_drivers", []) or []:
        try:
            v = float(sd.get("contribution_cycles"))
        except (TypeError, ValueError):
            violations.append(f"sensor driver {sd.get('sensor')} has no numeric contribution")
            continue
        if not _matches(v, pool):
            violations.append(f"sensor driver {sd.get('sensor')} contribution {v} does not appear in explain_prediction")
    text = " ".join([report.get("headline", ""), report.get("summary", ""), report.get("abstain_reason", "")]
                    + list(report.get("next_steps", []) or []))
    for m in check_numbers_in_text(text, pool):
        violations.append(f"the number {m} in the text does not appear in the tool outputs (do not compute new numbers)")
    if authoritative_action == "ABSTAIN" and not str(report.get("abstain_reason", "")).strip():
        violations.append("abstain_reason must explain why the system is not making the call")
    return violations
