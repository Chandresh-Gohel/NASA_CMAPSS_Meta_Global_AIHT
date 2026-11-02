"""Deterministic fleet planner: INSPECT engines into hangar slots before their deadline,
ABSTAIN engines into engineering-review slots. The planner agent explains this
schedule; it cannot change it."""
from __future__ import annotations

import math
from typing import Any


def _deadline_day(d: dict, cycles_per_day: float) -> int:
    """Last day an inspection may happen: the conservative RUL bound in days (day 1 = today)."""
    return max(1, 1 + int(math.floor(max(0.0, float(d.get("rul_lo_conservative") or 0.0)) / cycles_per_day)))


def _priority(d: dict) -> str:
    if d["action"] == "INSPECT" and d.get("urgency") == "immediate":
        return "P1"
    if d["action"] == "ABSTAIN" and "sensor_integrity" in d.get("reasons", []):
        return "P1"
    if d["action"] == "INSPECT":
        return "P2"
    return "P3"


def _review_rank(d: dict) -> int:
    reasons = d.get("reasons", [])
    if "sensor_integrity" in reasons:
        return 0
    if "ambiguous_band" in reasons:
        return 1
    return 2


def _work_order(d: dict, day: int, kind: str, deadline: int) -> dict:
    return {
        "engine_id": d["engine_id"],
        "day": int(day),
        "action": kind,
        "deadline_day": int(deadline),
        "priority": _priority(d),
        "reason": d.get("resolution", ""),
        "p_fail_within_horizon": d.get("p_fail_within_horizon"),
        "rul_lo_conservative": d.get("rul_lo_conservative"),
    }


def plan_inspections(decisions: list[dict], cfg: Any, days: int | None = None, slots_per_day: int | None = None,
                     review_slots_per_day: int | None = None) -> dict:
    pc = cfg.planner
    days = int(days or pc.horizon_days)
    slots = int(slots_per_day or pc.slots_per_day)
    review_slots = int(review_slots_per_day or pc.review_slots_per_day)
    cpd = float(pc.cycles_per_day)

    schedule = {day: {"inspections": [], "reviews": []} for day in range(1, days + 1)}
    conflicts: list[str] = []
    unplaced: list[str] = []

    inspections = [d for d in decisions if d["action"] == "INSPECT"]
    inspections.sort(key=lambda d: (0 if d.get("urgency") == "immediate" else 1, _deadline_day(d, cpd),
                                    -float(d.get("p_fail_within_horizon") or 0.0)))
    for d in inspections:
        deadline = _deadline_day(d, cpd)
        placed = False
        for day in range(1, days + 1):
            if len(schedule[day]["inspections"]) < slots:
                schedule[day]["inspections"].append(_work_order(d, day, "INSPECT", deadline))
                if day > deadline:
                    conflicts.append(f"{d['engine_id']} is scheduled on day {day}, after its deadline (day {deadline}).")
                placed = True
                break
        if not placed:
            unplaced.append(d["engine_id"])
            conflicts.append(f"No inspection slot within {days} days for {d['engine_id']} (deadline day {deadline}).")

    reviews = [d for d in decisions if d["action"] == "ABSTAIN"]
    reviews.sort(key=lambda d: (_review_rank(d), -float(d.get("p_fail_within_horizon") or 0.0)))
    for d in reviews:
        deadline = 1 if "sensor_integrity" in d.get("reasons", []) else _deadline_day(d, cpd)
        placed = False
        for day in range(1, days + 1):
            if len(schedule[day]["reviews"]) < review_slots:
                schedule[day]["reviews"].append(_work_order(d, day, "ENGINEERING_REVIEW", deadline))
                if day > deadline:
                    conflicts.append(f"Review of {d['engine_id']} lands on day {day}, after day {deadline}.")
                placed = True
                break
        if not placed:
            unplaced.append(d["engine_id"])
            conflicts.append(f"No engineering-review slot within {days} days for {d['engine_id']}.")

    watch = sorted([d for d in decisions if d["action"] == "NO_INSPECTION"],
                   key=lambda d: (d.get("recheck_in_cycles") or 10_000, -float(d.get("p_fail_within_horizon") or 0.0)))[:10]
    n_insp = sum(len(v["inspections"]) for v in schedule.values())
    n_rev = sum(len(v["reviews"]) for v in schedule.values())
    return {
        "days": [{"day": day, **schedule[day]} for day in range(1, days + 1)],
        "conflicts": conflicts,
        "unplaced": unplaced,
        "watch_list": [{"engine_id": d["engine_id"], "recheck_in_cycles": d.get("recheck_in_cycles"),
                        "p_fail_within_horizon": d.get("p_fail_within_horizon")} for d in watch],
        "summary": {
            "engines_considered": len(decisions),
            "inspections_needed": len(inspections),
            "reviews_needed": len(reviews),
            "inspections_planned": n_insp,
            "reviews_planned": n_rev,
            "days": days,
            "inspection_slots_per_day": slots,
            "review_slots_per_day": review_slots,
            "cycles_per_day": cpd,
        },
    }
