"""Which held-out replay engine did the uncertainty-aware policy miss, and why?

    python scripts/find_missed.py

Runs the same replay as `ehagent.evaluate` (``ehagent.simulate.simulate_policy``) on every replay engine, computed
live so the current config is used. For each engine the uncertainty-aware policy missed, it prints whether the point
estimate caught it, the calls in the engine's last 40 cycles, and the reasons in its last 30.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ehagent.config import load_config  # noqa: E402
from ehagent.service import EngineService  # noqa: E402
from ehagent.simulate import simulate_policy  # noqa: E402


def main() -> None:
    cfg = load_config()
    service = EngineService(cfg, use_precomputed=False)
    useful = float(cfg.simulation.useful_inspection_rul)
    missed, total = [], 0
    for s in service.subsets:
        for eid in service.datasets[s].splits["sim"]:
            total += 1
            tr = service.trajectory(eid, include_truth=True)
            ours = simulate_policy(tr, "uncertainty_aware", cfg)
            if not ours["caught"]:
                point = simulate_policy(tr, "point_estimate", cfg)
                missed.append((eid, point, tr))
    print(f"model {service.predictor.tag}: the uncertainty-aware policy missed {len(missed)} of {total} replay engines")
    for eid, point, tr in missed:
        how = f"caught it with {point['lead_time']:.0f} cycles left" if point["caught"] else "missed it too"
        print(f"\n{eid}: the point estimate {how}. Calls in the last 40 cycles:")
        tail = tr[tr["rul_true"] <= 40][["cycle", "rul_true", "mu", "lo", "hi", "p_fail", "action", "reasons"]]
        print(tail.round(2).to_string(index=False))
        window = tail[(tail["rul_true"] <= useful) & (tail["rul_true"] >= float(cfg.simulation.min_lead_cycles))]
        print(f"reasons while a catch was possible (true RUL {cfg.simulation.min_lead_cycles}-{useful:.0f}):",
              window["reasons"].astype(str).value_counts().to_dict())


if __name__ == "__main__":
    main()
