"""Accuracy and band width by stage of life (an evaluation script: true RUL is used here only).

    python scripts/rul_stage_report.py                 # model from features.multitask_model
    python scripts/rul_stage_report.py --mode single

1. Held-out replay engines (`sim` split), every cycle, by true RUL left: median band
   width, coverage against the capped truth, and the share of each call.
2. Official test engines, last cycle, by true RUL left: RMSE, mean error (positive =
   predicted too much life) and the share of total squared error. Cross-checks the
   per-subset RMSE against reports/metrics.json.
3. When the prediction leaves the RUL cap for good (true RUL left at that cycle).

Writes reports/rul_stage_report_<model>.json and prints README-ready tables.
Nothing here reaches an agent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ehagent.config import ensure_dir, load_config, path_for  # noqa: E402

BINS = [0, 30, 60, 90, 125, float("inf")]
LABELS = ["0-30", "30-60", "60-90", "90-125", "above 125"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rmse(e: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def replay_tables(service, cap: float):
    frames, leaves = [], []
    for s in service.subsets:
        for eid in service.datasets[s].splits["sim"]:
            tr = service.trajectory(eid, include_truth=True)
            tr["subset"] = s
            frames.append(tr)
            still = np.flatnonzero(tr["mu"].to_numpy() >= cap - 10)    # within 10 cycles of the cap
            if len(still) and still[-1] + 1 < len(tr):
                leaves.append({"subset": s, "rul_left": float(tr["rul_true"].iloc[still[-1] + 1])})
    rp = pd.concat(frames, ignore_index=True)
    truth = np.minimum(rp["rul_true"], cap)
    rp["width"] = rp["hi"] - rp["lo"]
    rp["inside"] = (truth >= rp["lo"]) & (truth <= rp["hi"])
    rp["stage"] = pd.cut(rp["rul_true"], BINS, labels=LABELS, right=False)
    g = rp.groupby("stage", observed=True)
    table = pd.DataFrame({
        "cycles": g.size(),
        "median_band_width": g["width"].median(),
        "coverage": g["inside"].mean(),
        "share_abstain": g["action"].apply(lambda a: float((a == "ABSTAIN").mean())),
        "share_inspect": g["action"].apply(lambda a: float((a == "INSPECT").mean())),
    })
    log(f"replay: {rp['subset'].size} cycles from {sum(len(service.datasets[s].splits['sim']) for s in service.subsets)} engines")
    return table, pd.DataFrame(leaves), frames


def heads_up(frames: list[pd.DataFrame], horizon: float) -> dict:
    """Relate the calls to the point estimate on the replay engines.

    Agreement: on cycles with no sensor/unfamiliar gate, how often INSPECT is
    exactly "mean RUL <= horizon". Heads-up: per engine, the cycles between the
    first ambiguous-band review flag and the first INSPECT.
    """
    agree, leads, engines_with_inspect = [], [], 0
    for tr in frames:
        reasons = tr["reasons"].astype(str)
        ungated = ~reasons.str.contains("sensor_integrity|unfamiliar_engine")
        agree.append(((tr["action"] == "INSPECT") == (tr["mu"] <= horizon))[ungated].to_numpy())
        inspect = np.flatnonzero((tr["action"] == "INSPECT").to_numpy())
        if not len(inspect):
            continue
        engines_with_inspect += 1
        review = np.flatnonzero(reasons.str.contains("ambiguous_band").to_numpy())
        review = review[review < inspect[0]]
        if len(review):
            leads.append(float(tr["cycle"].iloc[inspect[0]] - tr["cycle"].iloc[review[0]]))
    a = np.concatenate(agree) if agree else np.zeros(0, bool)
    out = {"ungated_cycles": int(a.size), "inspect_equals_mean_below_horizon": float(a.mean()) if a.size else float("nan"),
           "engines_with_inspect": engines_with_inspect, "engines_flagged_before_inspect": len(leads)}
    if leads:
        out.update({"median_heads_up_cycles": float(np.median(leads)),
                    "q25": float(np.percentile(leads, 25)), "q75": float(np.percentile(leads, 75))})
    return out


def test_tables(service, cap: float) -> tuple[pd.DataFrame, pd.Series]:
    rows = [{"subset": service.subset_of(r["engine_id"]), "mu": float(r["rul_mean"]),
             "true": service.answer_key(r["engine_id"])} for r in service.fleet()]
    te = pd.DataFrame(rows)
    te["err"] = te["mu"] - np.minimum(te["true"], cap)
    te["stage"] = pd.cut(te["true"], BINS, labels=LABELS, right=False)
    total = float(np.square(te["err"]).sum())
    g = te.groupby("stage", observed=True)
    table = pd.DataFrame({
        "engines": g.size(),
        "rmse": g["err"].apply(rmse),
        "mean_error": g["err"].mean(),
        "share_of_squared_error": g["err"].apply(lambda e: float(np.square(e).sum()) / total),
    })
    log(f"test: {len(te)} engines, last cycle")
    return table, te.groupby("subset")["err"].apply(rmse)


def main() -> None:
    from ehagent.service import EngineService

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["multitask", "single"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    cap = float(cfg.data.rul_cap)
    service = EngineService(cfg, multitask=None if args.mode is None else args.mode == "multitask",
                            use_precomputed=False)   # live, so the calls follow the current config
    tag = service.predictor.tag
    log(f"model {tag}")

    replay, leaves, frames = replay_tables(service, cap)
    hu = heads_up(frames, float(cfg.decision.horizon_cycles))
    test, by_subset = test_tables(service, cap)

    mpath = path_for(cfg, "reports_dir", "metrics.json")
    reported = {}
    if mpath.exists():
        m = json.loads(mpath.read_text())
        if m.get("model") == tag:
            reported = {s: float(v["rmse_capped_truth"]) for s, v in m.get("test", {}).items()}
    for s, v in by_subset.items():
        ref = reported.get(s)
        note = "" if ref is None else f" (metrics.json {ref:.2f}, difference {abs(v - ref):.3f})"
        log(f"test RMSE capped {s}: {v:.2f}{note}")

    print("\n**Band width and calls by stage of life** (held-out replay engines, every cycle)\n")
    print("| True RUL left | Cycles | Median band width (cycles) | Coverage | Abstained | Inspected |")
    print("|---|---|---|---|---|---|")
    for stage, r in replay.iterrows():
        print(f"| {stage} | {int(r['cycles'])} | {r['median_band_width']:.0f} | {r['coverage']:.2f} | "
              f"{100 * r['share_abstain']:.0f}% | {100 * r['share_inspect']:.0f}% |")

    print("\n**Where the test RMSE comes from** (official test engines, last cycle, truth capped at "
          f"{cap:.0f})\n")
    print("| True RUL left | Engines | RMSE (cycles) | Mean error (cycles) | Share of squared error |")
    print("|---|---|---|---|---|")
    for stage, r in test.iterrows():
        print(f"| {stage} | {int(r['engines'])} | {r['rmse']:.1f} | {r['mean_error']:+.1f} | "
              f"{100 * r['share_of_squared_error']:.0f}% |")
    print("\nMean error above 0 means the model predicted more life than was left (the unsafe direction).")

    leaves_out = {}
    if len(leaves):
        q1, q3 = leaves["rul_left"].quantile([0.25, 0.75])
        leaves_out = {"median": float(leaves["rul_left"].median()), "q25": float(q1), "q75": float(q3),
                      "engines": int(len(leaves)),
                      "median_by_subset": leaves.groupby("subset")["rul_left"].median().to_dict()}
        print(f"\nThe prediction leaves the cap for good with a median of {leaves_out['median']:.0f} cycles left "
              f"(half the replay engines between {q1:.0f} and {q3:.0f}); the target starts falling at {cap:.0f}.")

    h = float(cfg.decision.horizon_cycles)
    print(f"\nINSPECT equals 'predicted RUL <= {h:.0f}' on {100 * hu['inspect_equals_mean_below_horizon']:.2f}% of "
          f"{hu['ungated_cycles']} replay cycles without a sensor or unfamiliar-engine gate.")
    if "median_heads_up_cycles" in hu:
        print(f"{hu['engines_flagged_before_inspect']} of {hu['engines_with_inspect']} replay engines got an engineer-review "
              f"flag before their first INSPECT, a median {hu['median_heads_up_cycles']:.0f} cycles earlier "
              f"(half between {hu['q25']:.0f} and {hu['q75']:.0f}).")

    out = {"model": tag, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "heads_up": hu,
           "replay_by_stage": replay.reset_index().astype({"stage": str}).to_dict(orient="records"),
           "test_by_stage": test.reset_index().astype({"stage": str}).to_dict(orient="records"),
           "test_rmse_capped_by_subset": by_subset.to_dict(),
           "prediction_leaves_cap": leaves_out}
    p = ensure_dir(path_for(cfg, "reports_dir")) / f"rul_stage_report_{tag}.json"
    p.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"saved {p}")


if __name__ == "__main__":
    main()
