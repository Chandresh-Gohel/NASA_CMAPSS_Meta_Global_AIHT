"""Step 12 exit check (both modes, offline-first): the templates for one INSPECT,
one ABSTAIN and one NO_INSPECTION engine all pass the number guard.

Live API briefings (needs a key): unset EHA_OFFLINE and pass --live
(single-engine calls only, cached).
Run: python scripts/check_12.py [--live]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from ehagent.agent.guard import check_report
from ehagent.agent.templates import template_report
from ehagent.agent.tools import ToolExecutor
from ehagent.config import load_config
from ehagent.service import EngineService


def pick_examples(service: EngineService) -> dict[str, str]:
    fleet = service.fleet()
    out: dict[str, str] = {}
    for row in fleet:
        out.setdefault(row["action"], row["engine_id"])
        if len(out) == 3:
            break
    assert set(out) == {"INSPECT", "ABSTAIN", "NO_INSPECTION"}, f"need all 3 actions, got {out}"
    return out


def offline_check(service: EngineService, cfg, examples: dict[str, str]) -> None:
    for action, eid in examples.items():
        ex = ToolExecutor(service)
        status = ex.execute("get_engine_status", {"engine_id": eid})
        assert status.get("action") == action, f"{eid}: {status.get('action')} != {action}"
        ex.execute("explain_prediction", {"engine_id": eid, "top_k": 3})
        if action == "ABSTAIN" or status.get("sensor_flags"):
            ex.execute("check_sensors", {"engine_id": eid})
        ev = {}
        for t in ex.trace:
            if "error" not in (t.get("result") or {}):
                ev[t["tool"]] = t["result"]
        ev["authoritative_action"] = action
        ev["thresholds"] = {"p_inspect": float(cfg.decision.p_inspect), "p_clear": float(cfg.decision.p_clear),
                            "rul_cap": float(cfg.data.rul_cap),
                            "interval_coverage": round(1 - float(cfg.calibration.alpha), 2)}
        report = template_report(ev, action, cfg)
        violations = check_report(report, action, ev, cfg)
        print(f"  {action} {eid}: violations={violations}")
        assert not violations, f"template {action} failed guard: {violations}"
        assert report["recommendation"] == action
        if action == "ABSTAIN":
            assert report["abstain_reason"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also run 3 live API briefings (needs key, cached)")
    args = ap.parse_args()
    cfg = load_config()
    for multitask in (True, False):
        print(f"--- multitask={multitask} ---")
        try:
            service = EngineService(cfg, multitask=multitask, use_precomputed=True)
        except FileNotFoundError as exc:
            print(f"skip multitask={multitask}: {exc}")
            continue
        examples = pick_examples(service)
        print(f"examples: {examples}")
        offline_check(service, cfg, examples)
        if args.live:
            if os.environ.get("EHA_OFFLINE", "0") == "1":
                print("EHA_OFFLINE=1, skipping live calls")
                continue
            from ehagent.agent.orchestrator import AgentOrchestrator

            orch = AgentOrchestrator(cfg, service)
            if not orch.client.available:
                print("no API key, skipping live calls")
                continue
            for action, eid in examples.items():
                res = orch.ask("Should this engine keep flying?", eid)
                v = res.get("violations") or res.get("violations_after_retry") or []
                print(f"  live {action} {eid}: fallback={res['used_fallback']} final={res['final_action']} violations={v}")
                assert res["final_action"] in ("INSPECT", "NO_INSPECTION", "ABSTAIN")
    print("Step 12 OK (offline templates)")


if __name__ == "__main__":
    main()
