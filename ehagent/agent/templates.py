"""Deterministic briefings and plans for when the API is off or a draft fails the guard.
Same shape as the Muse Spark outputs, so the app renders both identically."""
from __future__ import annotations

from typing import Any

CAVEAT = ("Decision support on a simulated benchmark (NASA C-MAPSS); not an airworthiness determination. "
          "A qualified engineer makes the final call.")

HEADLINES = {
    "INSPECT": "Inspect: failure risk before the next opportunity is elevated",
    "NO_INSPECTION": "No inspection needed now",
    "ABSTAIN": "No safe call: escalated to an engineer",
}


def _pct(p: float | None) -> str:
    return "n/a" if p is None else f"{100 * float(p):.1f}%"


def template_report(evidence: dict, authoritative_action: str, cfg: Any) -> dict:
    st = evidence.get("get_engine_status") or {}
    ex = evidence.get("explain_prediction") or {}
    eid = st.get("engine_id", "?")
    horizon = st.get("horizon_cycles")
    mu, lo, hi = st.get("rul_mean"), st.get("rul_lo"), st.get("rul_hi")
    p = st.get("p_fail_within_horizon")
    resolution = st.get("resolution", "")

    parts = [f"Estimated remaining life is {mu} cycles (90% band {lo} to {hi} cycles).",
             f"Probability of failure within {horizon} cycles: {_pct(p)}."]
    if st.get("healthy_region"):
        parts.append("The upper end of the band reaches the healthy region, so the exact remaining life above the cap cannot be estimated.")
    parts.append(resolution)

    drivers = []
    for c in (ex.get("contributions") or [])[:3]:
        drivers.append({"sensor": c["sensor"], "name": c["name"], "contribution_cycles": c["contribution_cycles"],
                        "plain_language": f"{c['description']} {c['effect']} by about {abs(c['contribution_cycles'])} cycles."})

    key_numbers = [
        {"label": "Estimated RUL", "value": mu, "unit": "cycles", "source": "get_engine_status"},
        {"label": "Band lower bound", "value": lo, "unit": "cycles", "source": "get_engine_status"},
        {"label": "Band upper bound", "value": hi, "unit": "cycles", "source": "get_engine_status"},
        {"label": f"P(failure within {horizon} cycles)", "value": p, "unit": "probability", "source": "get_engine_status"},
    ]
    key_numbers = [k for k in key_numbers if k["value"] is not None]
    next_steps = [resolution] if resolution else []
    if st.get("sensor_flags"):
        next_steps.append("Check the flagged sensors before trusting any estimate for this engine.")
    return {
        "engine_id": eid,
        "recommendation": authoritative_action,
        "headline": HEADLINES.get(authoritative_action, authoritative_action),
        "summary": " ".join(x for x in parts if x),
        "key_numbers": key_numbers,
        "sensor_drivers": drivers,
        "abstain_reason": resolution if authoritative_action == "ABSTAIN" else "",
        "next_steps": next_steps,
        "caveats": [CAVEAT],
    }


def template_plan(plan: dict) -> dict:
    orders = []
    for day in plan.get("days", []):
        for wo in day.get("inspections", []) + day.get("reviews", []):
            orders.append({"engine_id": wo["engine_id"], "day": wo["day"], "action": wo["action"],
                           "priority": wo["priority"], "reason": wo.get("reason", "")})
    s = plan.get("summary", {})
    summary = (f"{s.get('inspections_planned', 0)} inspections and {s.get('reviews_planned', 0)} engineering reviews "
               f"scheduled over {s.get('days', 0)} days ({s.get('inspection_slots_per_day', 0)} hangar slots and "
               f"{s.get('review_slots_per_day', 0)} review slots per day).")
    return {"summary": summary, "work_orders": orders, "capacity_warnings": list(plan.get("conflicts", []))}
