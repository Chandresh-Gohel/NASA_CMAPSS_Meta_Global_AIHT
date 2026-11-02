"""Step 10 exit check (both modes): ranked fleet, unique plan engines, stable fingerprint.

Run: python scripts/check_10.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehagent.config import load_config
from ehagent.service import EngineService, risk_key


def check_mode(cfg, multitask: bool) -> None:
    service = EngineService(cfg, multitask=multitask, use_precomputed=False)
    fleet = service.fleet()
    total_test = sum(len(service.datasets[s].splits["test"]) for s in service.subsets)
    print(f"model={service.predictor.tag} fleet={len(fleet)} expected={total_test}")
    assert len(fleet) == total_test, f"fleet {len(fleet)} != test {total_test}"
    keys = [risk_key(r) for r in fleet]
    assert keys == sorted(keys), "fleet not riskiest-first"
    plan = service.plan()
    seen: dict[str, str] = {}
    for day in plan["days"]:
        for wo in day["inspections"] + day["reviews"]:
            assert wo["engine_id"] not in seen, f"{wo['engine_id']} twice in plan"
            seen[wo["engine_id"]] = f"day {day['day']}"
    print(f"  plan inspections={plan['summary']['inspections_planned']} reviews={plan['summary']['reviews_planned']} "
          f"conflicts={len(plan['conflicts'])} unplaced={len(plan['unplaced'])} watch={len(plan['watch_list'])}")
    fp1 = service._fingerprint()
    service2 = EngineService(cfg, multitask=multitask, use_precomputed=True)
    fp2 = service2._fingerprint()
    assert fp1 == fp2, "fingerprint unstable"
    print("  fingerprint stable; stale files ignored by design")


def main() -> None:
    cfg = load_config()
    for multitask in (True, False):
        print(f"--- multitask={multitask} ---")
        check_mode(cfg, multitask)
    print("Step 10 OK (precompute run: python -m ehagent.precompute --mode both [--no-explain])")


if __name__ == "__main__":
    main()
