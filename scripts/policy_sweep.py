"""Is the uncertainty worth it? Trade-off curves over inspection rules.

    python scripts/policy_sweep.py                  # model from features.multitask_model
    python scripts/policy_sweep.py --mode single

Held-out engines are replayed to failure under the same rules as `ehagent.simulate`
(actionable window, cooldown between events). Rule families compared:

  point estimate          inspect when the mean RUL <= T                          (T swept)
  risk within L cycles    inspect when P(RUL <= L) >= eps, from the calibrated     (eps swept,
                          Gaussian the decision engine uses (mu, sigma columns)     L = 10, 20, 30)

"P(RUL <= 10) >= 0.05" reads: inspect while there is still at least a 95% chance
of 10 or more cycles of warning. L = 30 with eps = p_inspect is the shipped
decision rule (horizon 30). Sensor-integrity and unfamiliar-engine abstentions go
to an engineer review, counted separately; reviews never catch a failure.

Each family's threshold is picked on validation engines (cheapest rule with no
failure missed or caught late and a median warning of at least --min-median-lead
cycles), then scored on the held-out replay engines, so nothing is tuned on the
engines it is scored on. eps above 0.5 is not swept: it would inspect uncertain
engines later than confident ones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ehagent.config import ensure_dir, load_config, path_for  # noqa: E402

POINT_T = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60)
WINDOWS = (10, 20, 30)
EPS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
GATES = "sensor_integrity|unfamiliar_engine"
POINT = "point estimate"


def family(L: int) -> str:
    return f"risk within {L} cycles"


def prob_within(mu: np.ndarray, sigma: np.ndarray, L: float) -> np.ndarray:
    """P(RUL <= L) for a Gaussian truncated to RUL >= 0 (same formula as ehagent.decision)."""
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    mu = np.asarray(mu, dtype=float)
    a, b = norm.cdf((L - mu) / s), norm.cdf(-mu / s)
    return np.clip((a - b) / np.maximum(1.0 - b, 1e-12), 0.0, 1.0)


def replay_one(traj: pd.DataFrame, inspect: np.ndarray, review: np.ndarray, sc) -> dict:
    """One engine under one rule. Mirrors ehagent.simulate.simulate_policy for inspections."""
    useful, min_lead, cooldown = float(sc.useful_inspection_rul), float(sc.min_lead_cycles), int(sc.cooldown_cycles)
    cycles = traj["cycle"].to_numpy(dtype=int)
    rul = traj["rul_true"].to_numpy(dtype=float)
    last_insp = last_rev = -10_000
    unnecessary = reviews = 0
    for i in range(len(traj)):
        if rul[i] < min_lead:
            break
        if review[i] and cycles[i] - last_rev >= cooldown:
            reviews += 1
            last_rev = cycles[i]
        if inspect[i] and cycles[i] - last_insp >= cooldown:
            last_insp = cycles[i]
            if rul[i] <= useful:
                return {"caught": True, "lead": float(rul[i]), "unnecessary": unnecessary, "reviews": reviews}
            unnecessary += 1
    return {"caught": False, "lead": None, "unnecessary": unnecessary, "reviews": reviews}


def summarise(results: list[dict], late_lead: float) -> dict:
    leads = [r["lead"] for r in results if r["caught"]]
    return {
        "engines": len(results),
        "missed": int(sum(not r["caught"] for r in results)),
        "late": int(sum(r["caught"] and r["lead"] < late_lead for r in results)),
        "unnecessary": int(sum(r["unnecessary"] for r in results)),
        "reviews": int(sum(r["reviews"] for r in results)),
        "median_lead": float(np.median(leads)) if leads else float("nan"),
    }


def load_trajectories(service, split: str) -> list[pd.DataFrame]:
    return [service.trajectory(eid, include_truth=True) for s in service.subsets for eid in service.datasets[s].splits[split]]


def sweep(trajs: list[pd.DataFrame], sc, late_lead: float) -> pd.DataFrame:
    rows = []
    no_review = [np.zeros(len(tr), dtype=bool) for tr in trajs]
    for t in POINT_T:
        res = [replay_one(tr, tr["mu"].to_numpy() <= t, nr, sc) for tr, nr in zip(trajs, no_review)]
        rows.append({"family": POINT, "rule": f"RUL <= {t}", "window": np.nan, "value": float(t), **summarise(res, late_lead)})
    gates = [tr["reasons"].astype(str).str.contains(GATES).to_numpy() for tr in trajs]
    for L in WINDOWS:
        probs = [prob_within(tr["mu"].to_numpy(), tr["sigma"].to_numpy(), L) for tr in trajs]
        for eps in EPS:
            res = [replay_one(tr, p >= eps, g, sc) for tr, p, g in zip(trajs, probs, gates)]
            rows.append({"family": family(L), "rule": f"P(RUL <= {L}) >= {eps:.2f}", "window": float(L), "value": float(eps),
                         **summarise(res, late_lead)})
    return pd.DataFrame(rows)


def check_formula(trajs: list[pd.DataFrame], horizon: float) -> float:
    """Largest gap between our P(RUL <= H) and the decision engine's p_fail column (want ~0)."""
    gaps = [np.max(np.abs(prob_within(tr["mu"].to_numpy(), tr["sigma"].to_numpy(), horizon) - tr["p_fail"].to_numpy()))
            for tr in trajs if len(tr)]
    return float(max(gaps)) if gaps else float("nan")


def choose(table: pd.DataFrame, fam: str, min_median_lead: float) -> pd.Series:
    """Cheapest rule with no failure missed or late and enough typical warning."""
    t = table[table["family"] == fam]
    ok = t[(t["missed"] == 0) & (t["late"] == 0) & (t["median_lead"] >= min_median_lead)]
    if len(ok):
        return ok.sort_values(["unnecessary", "median_lead"], ascending=[True, False]).iloc[0]
    return t.assign(bad=t["missed"] + t["late"]).sort_values(["bad", "unnecessary"]).iloc[0]


def plot(sim: pd.DataFrame, chosen: dict, out: Path, late_lead: float, mode: str = "light", y_cap: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ehagent.viz import CATEGORICAL, CHROME, DE_EMPHASIS, style_mpl_axes

    ink = CHROME[mode]
    colors = {POINT: DE_EMPHASIS[mode], **{family(L): CATEGORICAL[mode][i] for i, L in enumerate(WINDOWS)}}
    fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor=ink["page"])
    style_mpl_axes(ax, mode)
    hidden, top = 0, 0
    for fam, color in colors.items():
        t = sim[sim["family"] == fam].sort_values(["unnecessary", "value"])
        y = t["missed"] + t["late"]
        keep = y <= y_cap
        hidden += int((~keep).sum())
        top = max(top, int(y[keep].max()) if keep.any() else 0)
        ax.plot(t["unnecessary"][keep], y[keep], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=ink["surface"], markeredgewidth=1.5, solid_capstyle="round", label=fam)
        c = chosen[fam]
        ax.plot([c["unnecessary"]], [c["missed"] + c["late"]], "o", color=color, markersize=12,
                markeredgecolor=ink["surface"], markeredgewidth=2)
    from matplotlib.ticker import MaxNLocator

    ax.set_ylim(-0.6, max(4, top) + 0.6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Unnecessary inspections (held-out engines)")
    ax.set_ylabel(f"Failures missed or caught < {late_lead:g} cycles ahead")
    note = f"; {hidden} rule(s) above {y_cap} not shown" if hidden else ""
    ax.set_title(f"Cost of safety: large dots = rules chosen on validation engines{note}", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=ink["text2"])
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    from ehagent.service import EngineService

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["multitask", "single"], default=None)
    ap.add_argument("--late-lead", type=float, default=10.0, help="a catch with fewer cycles of warning counts as late")
    ap.add_argument("--min-median-lead", type=float, default=20.0, help="required typical warning on validation")
    args = ap.parse_args()
    cfg = load_config()
    service = EngineService(cfg, multitask=None if args.mode is None else args.mode == "multitask")
    sc, H = cfg.simulation, float(cfg.decision.horizon_cycles)
    print(f"model {service.predictor.tag} | catch window <= {sc.useful_inspection_rul} cycles | late < {args.late_lead:g} "
          f"| required median warning {args.min_median_lead:g} | current rule P(RUL <= {H:g}) >= {cfg.decision.p_inspect}")
    print("loading trajectories (validation, then held-out replay engines) ...")
    val_trajs, sim_trajs = load_trajectories(service, "val"), load_trajectories(service, "sim")
    print(f"formula check vs the decision engine's p_fail: largest gap {check_formula(val_trajs + sim_trajs, H):.2e} (should be ~0)")
    val, sim = sweep(val_trajs, sc, args.late_lead), sweep(sim_trajs, sc, args.late_lead)

    families = [POINT] + [family(L) for L in WINDOWS]
    chosen = {}
    for fam in families:
        c = choose(val, fam, args.min_median_lead)
        chosen[fam] = sim[(sim["family"] == fam) & (sim["value"] == c["value"])].iloc[0]

    cols = ["rule", "missed", "late", "unnecessary", "reviews", "median_lead"]
    with pd.option_context("display.width", 140):
        print("\nValidation engines (used to choose each family's rule):")
        print(val[cols].to_string(index=False))
        print("\nHeld-out replay engines:")
        print(sim[cols].to_string(index=False))
    print("\nAt the rules chosen on validation, held-out engines:\n")
    print("| Rule | Missed | Caught late | Unnecessary inspections | Engineer reviews | Median warning (cycles) |")
    print("|---|---|---|---|---|---|")
    for fam in families:
        r = chosen[fam]
        lead = "n/a" if pd.isna(r["median_lead"]) else f"{r['median_lead']:.0f}"
        print(f"| {r['rule']} | {r['missed']}/{r['engines']} | {r['late']} | {r['unnecessary']} | {r['reviews']} | {lead} |")
    out = ensure_dir(path_for(cfg, "reports_dir")) / "policy_tradeoff.png"
    plot(sim, chosen, out, args.late_lead)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
