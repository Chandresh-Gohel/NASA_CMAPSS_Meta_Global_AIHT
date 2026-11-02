"""Print README-ready markdown tables from reports/metrics.json (JSON only, no models).

    python scripts/results_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ehagent.config import load_config, path_for  # noqa: E402
from ehagent.constants import sensor_label  # noqa: E402


def fmt(v, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> None:
    cfg = load_config()
    p = path_for(cfg, "reports_dir", "metrics.json")
    if not p.exists():
        sys.exit(f"{p} not found. Run `python -m ehagent.evaluate` first.")
    m = json.loads(p.read_text())
    print(f"Model `{m.get('model')}`, generated {m.get('generated_at')}\n")

    print("| Subset | RMSE | NASA score | 90% band coverage | Mean band width (cycles) | RMSE when true RUL <= 30 |")
    print("|---|---|---|---|---|---|")
    for s, t in m.get("test", {}).items():
        print(f"| {s} | {fmt(t['rmse_capped_truth'])} | {fmt(t['nasa_score_capped_truth'], 0)} | {fmt(t['coverage'])} | "
              f"{fmt(t['mean_band_width'], 1)} | {fmt(t['rmse_true_rul_le_30'])} |")

    pc = m.get("policy_comparison")
    if pc:
        print("\n| Policy (held-out engines replayed to failure) | Missed failures | Unnecessary inspections | "
              "Engineer reviews | Median warning lead (cycles) |")
        print("|---|---|---|---|---|")
        for name, v in pc["summary"].items():
            print(f"| {name.replace('_', ' ')} | {v['failures_missed']}/{v['engines']} | {v['unnecessary_inspections']} | "
                  f"{v.get('engineering_reviews', 0)} | {fmt(v['median_lead_time_cycles'], 0)} |")

    ab = m.get("ablation")
    if ab:
        tags = sorted({tag for rows in ab.values() for tag in rows})
        print("\n| Subset | " + " | ".join(f"RMSE {t}" for t in tags) + " | " + " | ".join(f"coverage {t}" for t in tags) + " |")
        print("|---|" + "---|" * (2 * len(tags)))
        for s, rows in ab.items():
            rm = [fmt(rows.get(t, {}).get("rmse_capped_truth")) for t in tags]
            cv = [fmt(rows.get(t, {}).get("coverage")) for t in tags]
            print(f"| {s} | " + " | ".join(rm + cv) + " |")

    fi = m.get("fault_injection")
    if fi:
        print(f"\nSensor faults on {sensor_label(fi['sensor'])}, {fi['engines']} {fi['subset']} test engines "
              f"(abstain rate without a fault: {100 * fi['clean_abstain_rate']:.0f}%):\n")
        print("| Fault | Abstained | Sensor flagged | Decision changed | Mean RUL shift (cycles) |")
        print("|---|---|---|---|---|")
        for kind, v in fi["faults"].items():
            print(f"| {kind} | {100 * v['abstain_rate']:.0f}% | {100 * v['sensor_flag_rate']:.0f}% | "
                  f"{100 * v['action_changed_rate']:.0f}% | {v['mean_rul_shift_cycles']:+.1f} |")

    at = m.get("attribution")
    if at:
        print(f"\nExplanation faithfulness ({at['engines']} {at['subset']} test engines): median ratio "
              f"{at['median_faithfulness_ratio']:.2f}, above 1 for {100 * at['share_ratio_above_1']:.0f}% of engines.")

    ood = m.get("ood")
    if ood:
        i, o = ood["in_distribution"], ood["out_of_distribution"]
        print(f"\nUnfamiliar engines: {ood['target']} engines fed to the {ood['source']} model were abstained on "
              f"{100 * o['abstain_rate']:.0f}% of the time, vs {100 * i['abstain_rate']:.0f}% for real {ood['source']} engines.")


if __name__ == "__main__":
    main()
