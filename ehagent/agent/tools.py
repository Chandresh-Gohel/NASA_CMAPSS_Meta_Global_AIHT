"""Agent tools: deterministic Python over the EngineService, so every number an agent
can quote originates here (and the guard verifies that it did)."""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np

ANALYST_TOOL_NAMES = ["get_engine_status", "explain_prediction", "check_sensors", "what_if_horizon"]
PLANNER_TOOL_NAMES = ["fleet_triage", "plan_inspections", "get_engine_status"]


def _specs(subsets: list[str]) -> dict[str, dict]:
    engine = {"type": "string", "description": "Engine id, e.g. FD001-TE-012"}
    subset = {"type": "string", "enum": ["ALL"] + list(subsets), "description": "Fleet subset, or ALL"}
    return {
        "get_engine_status": {
            "name": "get_engine_status",
            "description": ("Latest RUL estimate, calibrated 90% band, probability of failure within the planning horizon, "
                            "sensor-integrity flags and the decision engine's action (INSPECT / NO_INSPECTION / ABSTAIN). "
                            "The action is authoritative."),
            "parameters": {"type": "object", "properties": {"engine_id": engine}, "required": ["engine_id"], "additionalProperties": False},
        },
        "explain_prediction": {
            "name": "explain_prediction",
            "description": ("Which sensors drive the RUL estimate: Integrated Gradients contribution of each sensor in cycles, "
                            "relative to an average healthy engine, plus a faithfulness check."),
            "parameters": {"type": "object", "properties": {
                "engine_id": engine,
                "top_k": {"type": "integer", "minimum": 1, "maximum": 14, "description": "How many sensors to return"},
            }, "required": ["engine_id"], "additionalProperties": False},
        },
        "check_sensors": {
            "name": "check_sensors",
            "description": "Sensor integrity checks (stuck, sudden jump, out of range) over the recent cycles.",
            "parameters": {"type": "object", "properties": {"engine_id": engine}, "required": ["engine_id"], "additionalProperties": False},
        },
        "what_if_horizon": {
            "name": "what_if_horizon",
            "description": "Failure probability and decision if the next maintenance opportunity is a different number of cycles away.",
            "parameters": {"type": "object", "properties": {
                "engine_id": engine,
                "horizon_cycles": {"type": "integer", "minimum": 5, "maximum": 200},
            }, "required": ["engine_id", "horizon_cycles"], "additionalProperties": False},
        },
        "fleet_triage": {
            "name": "fleet_triage",
            "description": "Fleet ranked by risk with each engine's action, band and failure probability, plus counts per action.",
            "parameters": {"type": "object", "properties": {
                "subset": subset,
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
            }, "required": [], "additionalProperties": False},
        },
        "plan_inspections": {
            "name": "plan_inspections",
            "description": ("Deterministic schedule: INSPECT engines into hangar slots before their deadline, ABSTAIN engines "
                            "into engineering-review slots. This schedule is authoritative."),
            "parameters": {"type": "object", "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 30},
                "slots_per_day": {"type": "integer", "minimum": 1, "maximum": 20},
                "review_slots_per_day": {"type": "integer", "minimum": 1, "maximum": 20},
                "subset": subset,
            }, "required": [], "additionalProperties": False},
        },
    }


def tool_defs(names: list[str], subsets: list[str]) -> list[dict]:
    """Chat Completions tool definitions (function metadata nested under ``function``)."""
    specs = _specs(subsets)
    return [{"type": "function", "function": specs[n]} for n in names]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _compact(row: dict) -> dict:
    keys = ("engine_id", "action", "urgency", "rul_mean", "rul_lo", "rul_hi", "p_fail_within_horizon", "reasons",
            "recheck_in_cycles", "horizon_cycles")
    return {k: row.get(k) for k in keys}


class ToolExecutor:
    """Runs tool calls against the service, keeping a trace for the app and the guard."""

    def __init__(self, service, faults: dict | None = None, horizon: int | None = None):
        self.service = service
        self.faults = faults or {}
        self.horizon = horizon
        self.trace: list[dict] = []

    def execute(self, name: str, arguments: Any) -> dict:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            result = {"error": f"invalid arguments: {exc}"}
            self.trace.append({"tool": name, "arguments": arguments, "result": result})
            return result
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            result = {"error": f"unknown tool '{name}'"}
        else:
            try:
                result = fn(**args)
            except TypeError as exc:
                result = {"error": f"bad arguments for {name}: {exc}"}
            except (KeyError, ValueError) as exc:
                result = {"error": str(exc)}
        result = _json_safe(result)
        self.trace.append({"tool": name, "arguments": args, "result": result})
        return result

    def _t_get_engine_status(self, engine_id: str) -> dict:
        return self.service.analyze(engine_id, horizon=self.horizon, fault=self.faults.get(engine_id))

    def _t_explain_prediction(self, engine_id: str, top_k: int = 3) -> dict:
        return self.service.explain(engine_id, top_k=int(max(1, min(int(top_k), 14))), fault=self.faults.get(engine_id))

    def _t_check_sensors(self, engine_id: str) -> dict:
        return self.service.check_sensors(engine_id, fault=self.faults.get(engine_id))

    def _t_what_if_horizon(self, engine_id: str, horizon_cycles: int) -> dict:
        return self.service.what_if(engine_id, int(horizon_cycles), fault=self.faults.get(engine_id))

    def _t_fleet_triage(self, subset: str = "ALL", top_n: int = 10) -> dict:
        subsets = None if subset in (None, "ALL") else [subset]
        rows = self.service.fleet(subsets, horizon=self.horizon, faults=self.faults)
        counts = Counter(r["action"] for r in rows)
        return {"total_engines": len(rows), "counts": dict(counts), "engines": [_compact(r) for r in rows[: int(top_n)]]}

    def _t_plan_inspections(self, days: int | None = None, slots_per_day: int | None = None,
                            review_slots_per_day: int | None = None, subset: str = "ALL") -> dict:
        subsets = None if subset in (None, "ALL") else [subset]
        return self.service.plan(subsets, horizon=self.horizon, faults=self.faults, days=days,
                                 slots_per_day=slots_per_day, review_slots_per_day=review_slots_per_day)
