"""Replay held-out run-to-failure engines cycle by cycle and compare maintenance policies.

Policies
* fixed_interval     - inspect every N cycles (today's practice)
* point_estimate     - inspect when the mean RUL drops below the horizon (no uncertainty)
* uncertainty_aware  - our decision engine (inspect / no inspection / abstain)

An inspection "catches" a failure when the true RUL at that moment is within the
actionable window (and still leaves some warning); an earlier inspection finds
nothing and counts as unnecessary. A failure is missed if the engine dies first.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POLICIES = ("fixed_interval", "point_estimate", "uncertainty_aware")


def simulate_policy(traj: pd.DataFrame, policy: str, cfg: Any, horizon: int | None = None) -> dict:
    sc = cfg.simulation
    horizon = int(horizon if horizon is not None else cfg.decision.horizon_cycles)
    useful = float(sc.useful_inspection_rul)
    min_lead = float(sc.min_lead_cycles)
    cooldown = int(sc.cooldown_cycles)
    fixed = int(sc.fixed_interval_cycles)
    abstain_action = str(sc.abstain_action)

    last_event = last_review = -10_000
    unnecessary = reviews = abstain_cycles = 0
    for row in traj.itertuples(index=False):
        cycle = int(row.cycle)
        rul_true = float(row.rul_true)
        if rul_true < min_lead:
            break
        if policy == "uncertainty_aware" and row.action == "ABSTAIN":
            abstain_cycles += 1
            if abstain_action == "review":     # engineer review: counted, never credited as a catch
                if cycle - last_review >= cooldown:
                    reviews += 1
                    last_review = cycle
                continue
        if cycle - last_event < cooldown:
            continue
        trigger = None
        if policy == "fixed_interval":
            trigger = "inspect" if cycle % fixed == 0 else None
        elif policy == "point_estimate":
            trigger = "inspect" if float(row.mu) <= horizon else None
        elif policy == "uncertainty_aware":
            if row.action == "INSPECT":
                trigger = "inspect"
            elif row.action == "ABSTAIN":
                trigger = {"inspect": "inspect", "review_oracle": "review"}.get(abstain_action)
        if trigger is None:
            continue
        last_event = cycle
        if rul_true <= useful:
            return {"caught": True, "lead_time": rul_true, "unnecessary": unnecessary, "reviews": reviews + (trigger == "review"),
                    "abstain_cycles": abstain_cycles, "caught_by": trigger}
        if trigger == "inspect":
            unnecessary += 1
        else:
            reviews += 1
    return {"caught": False, "lead_time": None, "unnecessary": unnecessary, "reviews": reviews,
            "abstain_cycles": abstain_cycles, "caught_by": None}


def replay(service, subsets: list[str] | None = None, horizon: int | None = None) -> dict:
    per_engine: dict[str, dict] = {}
    for s in subsets or service.subsets:
        for eid in service.datasets[s].splits["sim"]:
            traj = service.trajectory(eid, horizon=horizon, include_truth=True)
            per_engine[eid] = {p: simulate_policy(traj, p, service.cfg, horizon) for p in POLICIES}
    summary = {}
    for p in POLICIES:
        res = [v[p] for v in per_engine.values()]
        leads = [r["lead_time"] for r in res if r["caught"]]
        summary[p] = {
            "engines": len(res),
            "failures_caught": int(sum(r["caught"] for r in res)),
            "failures_missed": int(sum(not r["caught"] for r in res)),
            "unnecessary_inspections": int(sum(r["unnecessary"] for r in res)),
            "engineering_reviews": int(sum(r["reviews"] for r in res)),
            "abstain_cycles": int(sum(r["abstain_cycles"] for r in res)),
            "mean_lead_time_cycles": float(np.mean(leads)) if leads else None,
            "median_lead_time_cycles": float(np.median(leads)) if leads else None,
        }
    return {"summary": summary, "per_engine": per_engine,
            "settings": {k: service.cfg.simulation[k] for k in service.cfg.simulation},
            "horizon_cycles": int(horizon if horizon is not None else service.cfg.decision.horizon_cycles)}


def plot_policy_comparison(summary: dict, out_path, mode: str = "light") -> None:
    """Three small panels (one measure each, never a dual axis). Ours is blue, baselines grey."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import CATEGORICAL, CHROME, DE_EMPHASIS, style_mpl_axes

    labels = {"fixed_interval": "Fixed\ninterval", "point_estimate": "Point\nestimate", "uncertainty_aware": "Uncertainty-\naware (ours)"}
    names = [labels[p] for p in POLICIES]
    colors = [DE_EMPHASIS[mode], DE_EMPHASIS[mode], CATEGORICAL[mode][0]]
    panels = (
        ([summary[p]["failures_missed"] for p in POLICIES], "Missed failures (lower is better)"),
        ([summary[p]["unnecessary_inspections"] for p in POLICIES], "Unnecessary inspections (lower is better)"),
        ([summary[p]["median_lead_time_cycles"] or 0 for p in POLICIES], "Median warning lead time, cycles"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=CHROME[mode]["page"])
    for ax, (vals, title) in zip(axes, panels):
        style_mpl_axes(ax, mode)
        ax.bar(range(len(vals)), vals, width=0.45, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        top = max(vals) if max(vals) > 0 else 1
        ax.set_ylim(0, top * 1.2)
        for i, v in enumerate(vals):
            ax.text(i, v + top * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=CHROME[mode]["text"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
