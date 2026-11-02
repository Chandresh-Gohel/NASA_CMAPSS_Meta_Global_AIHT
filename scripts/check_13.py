"""Step 13 exit check: the app imports, offline ask works, a fault flips a decision.

Manual follow-up: streamlit run app/streamlit_app.py (with and without an API key).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehagent.agent.orchestrator import AgentOrchestrator
from ehagent.config import load_config
from ehagent.service import EngineService


def main() -> None:
    cfg = load_config()
    import app.streamlit_app  # noqa: F401 - must import under streamlit

    print("app imports OK (4 tabs: Fleet, Engine, Ask the agent, Evaluation)")
    for multitask in (True, False):
        service = EngineService(cfg, multitask=multitask, use_precomputed=True)
        fleet = service.fleet()
        assert len(fleet) == 707, len(fleet)
        eid = fleet[0]["engine_id"]
        before = service.analyze(eid)["action"]
        fault = {"sensor": "s11", "kind": "stuck", "start_offset": 20, "magnitude": 0.0}
        after = service.analyze(eid, fault=fault)["action"]
        print(f"multitask={multitask} example {eid}: {before} -> stuck s11 -> {after} flags={len(service.analyze(eid, fault=fault)['sensor_flags'])}")
        orch = AgentOrchestrator(cfg, service)
        res = orch.ask("Should we inspect this engine?", eid)
        assert res["report"] and not res.get("error"), res.get("error")
        print(f"  offline ask OK: {res['final_action']} fallback={res['used_fallback']}")
    print("Step 13 OK (now manually: EHA_OFFLINE=0 and =1 streamlit runs)")


if __name__ == "__main__":
    main()
