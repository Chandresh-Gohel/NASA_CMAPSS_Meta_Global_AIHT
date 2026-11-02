"""Step 11 exit check: reports/metrics.json plus charts exist; the replay table prints.

Run after: python -m ehagent.evaluate [--mode multitask] [--skip-replay]
Then: python scripts/check_11.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehagent.config import load_config, path_for


def main() -> None:
    cfg = load_config()
    reports = path_for(cfg, "reports_dir")
    metrics_p = reports / "metrics.json"
    assert metrics_p.exists(), f"{metrics_p} not found. Run `python -m ehagent.evaluate` first."
    m = json.loads(metrics_p.read_text())
    assert "test" in m and len(m["test"]) == 4, "metrics.json must have 4 subsets"
    charts = [reports / "risk_coverage.png", reports / "coverage_by_rul.png", reports / "policy_comparison.png"]
    for c in charts:
        assert c.exists(), f"{c} not found"
        print(f"found {c}")
    pc = m.get("policy_comparison", {}).get("summary", {})
    assert pc, "policy_comparison missing (remove --skip-replay)"
    print("| policy | missed | unnecessary | median lead |")
    for name, v in pc.items():
        print(f"| {name} | {v['failures_missed']}/{v['engines']} | {v['unnecessary_inspections']} | {v['median_lead_time_cycles']} |")
    print("Step 11 OK")


if __name__ == "__main__":
    main()
