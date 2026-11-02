"""Agent orchestration on Muse Spark.

ask():  fixed status call -> analyst tool loop -> structured briefing
        -> number guard (one rewrite, then template) -> safety reviewer (escalate-only)
plan(): fixed schedule call -> planner tool loop -> structured work orders,
        reconciled against the deterministic schedule

The decision engine's action rules everywhere: agents explain, gather evidence
and escalate, but can never soften a call.
"""
from __future__ import annotations

import json
from typing import Any

from ..config import Config, path_for
from ..constants import CLEARING_ACTIONS
from .client import ApiError, MetaClient, OfflineError, json_schema_format, parse_json_object
from .guard import check_numbers_in_text, check_report, evidence_numbers
from .schemas import ENGINE_BRIEFING_SCHEMA, MAINTENANCE_PLAN_SCHEMA, SAFETY_REVIEW_SCHEMA
from .templates import CAVEAT, template_plan, template_report
from .tools import ANALYST_TOOL_NAMES, PLANNER_TOOL_NAMES, ToolExecutor, tool_defs

PROMPT_NAMES = ("analyst_system", "report_writer_system", "reviewer_system", "planner_system")
AGENT_ERRORS = (OfflineError, ApiError, ValueError, KeyError, TypeError)


def load_prompt(cfg: Config, name: str) -> str:
    return path_for(cfg, "prompts_dir", f"{name}.md").read_text(encoding="utf-8")


def _normalize_report(r: dict, engine_id: str) -> dict:
    out = {
        "engine_id": str(r.get("engine_id") or engine_id),
        "recommendation": str(r.get("recommendation", "")),
        "headline": str(r.get("headline", "")),
        "summary": str(r.get("summary", "")),
        "key_numbers": [k for k in (r.get("key_numbers") or []) if isinstance(k, dict)],
        "sensor_drivers": [s for s in (r.get("sensor_drivers") or []) if isinstance(s, dict)],
        "abstain_reason": str(r.get("abstain_reason") or ""),
        "next_steps": [str(x) for x in (r.get("next_steps") or [])],
        "caveats": [str(x) for x in (r.get("caveats") or [])],
    }
    if not any("simulat" in c.lower() for c in out["caveats"]):
        out["caveats"].append(CAVEAT)
    return out


class AgentOrchestrator:
    def __init__(self, cfg: Config, service, client: MetaClient | None = None):
        self.cfg = cfg
        self.service = service
        self.client = client or MetaClient(cfg)
        self.prompts = {name: load_prompt(cfg, name) for name in PROMPT_NAMES}
        self.subsets = list(service.subsets)

    def _evidence(self, trace: list[dict], authoritative: str) -> dict:
        ev: dict[str, Any] = {}
        for t in trace:
            res = t.get("result") or {}
            if "error" in res:
                continue
            if t["tool"] == "what_if_horizon":
                ev.setdefault("what_if_horizon", []).append(res)
            else:
                ev[t["tool"]] = res
        ev["authoritative_action"] = authoritative
        ev["thresholds"] = {
            "p_inspect": float(self.cfg.decision.p_inspect),
            "p_clear": float(self.cfg.decision.p_clear),
            "rul_cap": float(self.cfg.data.rul_cap),
            "interval_coverage": round(1 - float(self.cfg.calibration.alpha), 2),
        }
        return ev

    def ask(self, question: str, engine_id: str, faults: dict | None = None,
            horizon: int | None = None, use_agents: bool | None = None, use_reviewer: bool | None = None) -> dict:
        use_agents = bool(self.cfg.features.agents if use_agents is None else use_agents)
        use_reviewer = bool(self.cfg.features.reviewer if use_reviewer is None else use_reviewer)
        ex = ToolExecutor(self.service, faults, horizon)
        result: dict[str, Any] = {"engine_id": engine_id, "question": question, "used_fallback": False,
                                  "violations": [], "errors": [], "review": None}

        status = ex.execute("get_engine_status", {"engine_id": engine_id})
        if "error" in status:
            result.update(error=status["error"], report=None, trace=ex.trace)
            return result
        authoritative = status["action"]

        report = None
        evidence: dict = {}
        if use_agents:
            try:
                self._analyst_loop(question, engine_id, ex)
                evidence = self._evidence(ex.trace, authoritative)
                report = self._write_report(question, evidence, engine_id)
                violations = check_report(report, authoritative, evidence, self.cfg)
                if violations:
                    result["violations"] = list(violations)
                    report = self._write_report(question, evidence, engine_id, feedback=violations)
                    violations = check_report(report, authoritative, evidence, self.cfg)
                    result["violations_after_retry"] = list(violations)
                if violations:
                    report = None
            except AGENT_ERRORS as exc:
                result["errors"].append(f"analyst: {exc}")
                report = None
        if report is None:
            if not any(t["tool"] == "explain_prediction" for t in ex.trace):
                ex.execute("explain_prediction", {"engine_id": engine_id, "top_k": 3})
            evidence = self._evidence(ex.trace, authoritative)
            report = template_report(evidence, authoritative, self.cfg)
            result["used_fallback"] = True

        final_action = authoritative
        if use_agents and use_reviewer:
            try:
                review = self._review(evidence, report)
                result["review"] = review
                final_action, report = self._apply_review(review, authoritative, report)
            except AGENT_ERRORS as exc:
                result["errors"].append(f"reviewer: {exc}")
        result.update(report=report, final_action=final_action, authoritative_action=authoritative,
                      trace=ex.trace, usage=dict(self.client.usage))
        return result

    def _analyst_loop(self, question: str, engine_id: str, ex: ToolExecutor) -> None:
        tools = tool_defs(ANALYST_TOOL_NAMES, self.subsets)
        status = ex.trace[0]["result"]
        messages: list[dict] = [
            {"role": "system", "content": self.prompts["analyst_system"]},
            {"role": "user", "content": (f"Engine: {engine_id}\nQuestion: {question}\n\n"
                                         f"get_engine_status was already called. Result:\n{json.dumps(status)}")},
        ]
        for _ in range(int(self.cfg.agent.max_tool_rounds)):
            msg = self.client.chat(messages, tools=tools)
            calls = msg.get("tool_calls") or []
            if not calls:
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                res = ex.execute(call["function"]["name"], call["function"].get("arguments") or "{}")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(res)})

    def _write_report(self, question: str, evidence: dict, engine_id: str, feedback: list[str] | None = None) -> dict:
        user = f"Engineer's question: {question}\n\nEvidence (tool outputs, JSON):\n{json.dumps(evidence, indent=1)}"
        if feedback:
            user += "\n\nYour previous draft broke these rules. Fix every one:\n- " + "\n- ".join(feedback)
        msg = self.client.chat(
            [{"role": "system", "content": self.prompts["report_writer_system"]}, {"role": "user", "content": user}],
            response_format=json_schema_format("engine_briefing", ENGINE_BRIEFING_SCHEMA),
        )
        return _normalize_report(parse_json_object(msg.get("content")), engine_id)

    def _review(self, evidence: dict, report: dict) -> dict:
        user = ("Evidence (tool outputs, JSON):\n" + json.dumps(evidence, indent=1)
                + "\n\nBriefing to review:\n" + json.dumps(report, indent=1))
        msg = self.client.chat(
            [{"role": "system", "content": self.prompts["reviewer_system"]}, {"role": "user", "content": user}],
            response_format=json_schema_format("safety_review", SAFETY_REVIEW_SCHEMA),
        )
        r = parse_json_object(msg.get("content"))
        verdict = str(r.get("verdict", "revise"))
        if verdict not in ("approve", "revise", "force_abstain"):
            verdict = "revise"
        return {"verdict": verdict, "issues": [str(x) for x in (r.get("issues") or [])],
                "safety_concerns": [str(x) for x in (r.get("safety_concerns") or [])], "rationale": str(r.get("rationale", ""))}

    @staticmethod
    def _apply_review(review: dict, authoritative: str, report: dict) -> tuple[str, dict]:
        """The reviewer may escalate a clearing call to ABSTAIN or add notes; never relax anything."""
        report = dict(report)
        notes = (review.get("safety_concerns") or []) + (review.get("issues") or [])
        notes = [str(n) for n in notes][:3]
        if review.get("verdict") == "force_abstain" and authoritative in CLEARING_ACTIONS:
            report["recommendation"] = "ABSTAIN"
            report["headline"] = "Escalated by the safety reviewer: engineer review required"
            report["abstain_reason"] = "The safety reviewer did not accept a clearing decision: " + ("; ".join(notes) or review.get("rationale", ""))
            report["next_steps"] = ["Engineer review required before this engine keeps flying."] + list(report.get("next_steps", []))
            return "ABSTAIN", report
        if review.get("verdict") in ("revise", "force_abstain") and notes:
            report["caveats"] = list(report.get("caveats", [])) + [f"Safety reviewer: {n}" for n in notes]
        return authoritative, report

    def plan(self, faults: dict | None = None, horizon: int | None = None, days: int | None = None,
             slots_per_day: int | None = None, review_slots_per_day: int | None = None, subset: str = "ALL",
             use_agents: bool | None = None) -> dict:
        if use_agents is None:
            use_agents = bool(self.cfg.features.agents and self.cfg.features.planner_agent)
        ex = ToolExecutor(self.service, faults, horizon)
        args = {"days": days, "slots_per_day": slots_per_day, "review_slots_per_day": review_slots_per_day, "subset": subset}
        det = ex.execute("plan_inspections", {k: v for k, v in args.items() if v is not None})
        result: dict[str, Any] = {"deterministic_plan": det, "used_fallback": False, "errors": [], "violations": []}
        if "error" in det:
            result.update(plan=None, error=det["error"], trace=ex.trace)
            return result
        plan = None
        if use_agents:
            try:
                self._planner_loop(ex, subset)
                evidence = {t["tool"]: t["result"] for t in ex.trace if "error" not in (t.get("result") or {})}
                draft = self._write_plan(evidence)
                plan, result["violations"] = self._reconcile_plan(draft, det)
            except AGENT_ERRORS as exc:
                result["errors"].append(f"planner: {exc}")
                plan = None
        if plan is None:
            plan = template_plan(det)
            result["used_fallback"] = True
        result.update(plan=plan, trace=ex.trace, usage=dict(self.client.usage))
        return result

    def _planner_loop(self, ex: ToolExecutor, subset: str) -> None:
        tools = tool_defs(PLANNER_TOOL_NAMES, self.subsets)
        det = ex.trace[0]["result"]
        messages: list[dict] = [
            {"role": "system", "content": self.prompts["planner_system"]},
            {"role": "user", "content": (f"Build the maintenance plan for subset {subset}. plan_inspections was already called. "
                                         f"Result:\n{json.dumps(det)}\nGather any extra evidence you need, then stop calling tools.")},
        ]
        for _ in range(int(self.cfg.agent.max_tool_rounds)):
            msg = self.client.chat(messages, tools=tools)
            calls = msg.get("tool_calls") or []
            if not calls:
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                res = ex.execute(call["function"]["name"], call["function"].get("arguments") or "{}")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(res)})

    def _write_plan(self, evidence: dict) -> dict:
        user = ("Write the work orders and summary for this schedule.\n\nEvidence (tool outputs, JSON):\n"
                + json.dumps(evidence, indent=1))
        msg = self.client.chat(
            [{"role": "system", "content": self.prompts["planner_system"]}, {"role": "user", "content": user}],
            response_format=json_schema_format("maintenance_plan", MAINTENANCE_PLAN_SCHEMA),
        )
        return parse_json_object(msg.get("content"))

    def _reconcile_plan(self, draft: dict, det: dict) -> tuple[dict, list[str]]:
        """Keep the deterministic schedule; borrow the planner's wording only when it adds no numbers."""
        base = template_plan(det)
        orders = base["work_orders"]
        by_id = {wo["engine_id"]: wo for wo in orders}
        pool = evidence_numbers(det, self.cfg)
        violations: list[str] = []
        reasons: dict[str, str] = {}
        for wo in draft.get("work_orders") or []:
            if not isinstance(wo, dict):
                continue
            eid = wo.get("engine_id")
            if eid not in by_id:
                violations.append(f"planner proposed {eid}, which is not in the schedule (ignored)")
                continue
            if wo.get("day") != by_id[eid]["day"] or wo.get("action") != by_id[eid]["action"]:
                violations.append(f"planner changed the day/action for {eid} (schedule kept)")
            reasons[eid] = str(wo.get("reason", "")).strip()
        final_orders = []
        for wo in orders:
            r = reasons.get(wo["engine_id"], "")
            final_orders.append({**wo, "reason": r} if r and not check_numbers_in_text(r, pool) else wo)
        summary = str(draft.get("summary", "")).strip()
        if not summary or check_numbers_in_text(summary, pool):
            if summary:
                violations.append("planner summary contained numbers not in the schedule (replaced)")
            summary = base["summary"]
        warnings = [str(w) for w in (draft.get("capacity_warnings") or []) if not check_numbers_in_text(str(w), pool)]
        for c in det.get("conflicts", []):
            if c not in warnings:
                warnings.append(c)
        return {"summary": summary, "work_orders": final_orders, "capacity_warnings": warnings}, violations
