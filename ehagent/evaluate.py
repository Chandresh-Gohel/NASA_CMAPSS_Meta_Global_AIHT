"""Full evaluation: accuracy, calibration, abstention, robustness and policy impact.

    python -m ehagent.evaluate                 # model from features.multitask_model
    python -m ehagent.evaluate --mode single
    python -m ehagent.evaluate --plots-only    # redraw charts from the existing metrics.json

Writes reports/metrics.json plus charts (policy comparison, risk-coverage, coverage by RUL).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm

from .calibrate import MondrianConformal, calibrate, calibration_path
from .config import Config, ensure_dir, load_config, path_for
from .data import last_windows, make_windows
from .decision import decide
from .metrics import (aurc, binned_coverage, coverage, gaussian_crps, gaussian_nll, mae, mean_width, nasa_score,
                      risk_coverage_curve, rmse)
from .predictor import Predictor
from .sensor_checks import run_sensor_checks
from .simulate import plot_policy_comparison, replay


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _lower_z(conformal, alpha: float) -> float:
    lt = float(conformal.lower_tail(alpha)) if hasattr(conformal, "lower_tail") else alpha / 2.0
    return float(norm.ppf(1 - lt))


def evaluate_test(service) -> dict:
    cfg = service.cfg
    alpha = float(cfg.calibration.alpha)
    z = _lower_z(service.conformal, alpha)
    cap = float(cfg.data.rul_cap)
    out = {}
    for s in service.subsets:
        sd = service.datasets[s]
        X, y_true, eids = last_windows(sd, "test")
        pred = service.predictor.predict(s, X)
        mu, sig = pred["mu"], pred["sigma"]
        lo, hi, half = service.conformal.interval(s, mu, sig, alpha)
        sig_cal = np.maximum(half / z, 1e-3)
        y_cap = np.minimum(y_true, cap)
        near = y_cap <= 30
        curve = risk_coverage_curve(y_cap, mu, sig_cal)
        out[s] = {
            "n_engines": len(eids),
            "rmse_capped_truth": rmse(y_cap, mu),
            "rmse_uncapped_truth": rmse(y_true, mu),
            "nasa_score_capped_truth": nasa_score(y_cap, mu),
            "nasa_score_uncapped_truth": nasa_score(y_true, mu),
            "mae_capped_truth": mae(y_cap, mu),
            "rmse_true_rul_le_30": rmse(y_cap[near], mu[near]) if near.any() else None,
            "coverage": coverage(y_cap, lo, hi),
            "target_coverage": 1 - alpha,
            "mean_band_width": mean_width(lo, hi),
            "nll_calibrated": gaussian_nll(y_cap, mu, sig_cal),
            "crps_calibrated": gaussian_crps(y_cap, mu, sig_cal),
            "coverage_by_true_rul": binned_coverage(y_cap, lo, hi, [0, 30, 60, 90, cap + 1]),
            "risk_coverage": curve,
            "aurc": aurc(curve),
            "rmse_after_abstaining_on_widest_20pct": curve[int(0.8 * len(curve)) - 1]["rmse"],
        }
    return out


def _decision_for(service, subset: str, rec_raw, rec_norm, cycle: int, pred_row: dict, check_stats) -> dict:
    flags = run_sensor_checks(rec_raw, rec_norm, check_stats, service.cfg.sensor_checks)
    d = decide(engine_id="ood", subset=subset, cycle=cycle, mu=pred_row["mu"], sigma=pred_row["sigma"],
               sigma_epi=pred_row["sigma_epi"], sigma_alea=pred_row["sigma_alea"], conformal=service.conformal,
               sensor_flags=flags, cfg=service.cfg)
    return d.to_dict()


def ood_cross_subset(service, source: str, target: str, max_engines: int = 100) -> dict | None:
    """Score engines from another subset (different conditions/faults) as if they were `source`."""
    if source not in service.datasets or target not in service.datasets:
        return None
    cfg = service.cfg
    src, tgt = service.datasets[source], service.datasets[target]
    thr = float(cfg.decision.epistemic_percentile_abstain)

    def run(records, normalise_with_source: bool) -> dict:
        X, raws, norms, cycles = [], [], [], []
        for rec in records:
            if normalise_with_source:
                norm_vals = src.normalizer.transform(rec.raw, src.clusterer.predict(rec.ops))
            else:
                norm_vals = rec.norm
            X.append(make_windows(norm_vals, np.array([rec.length - 1]), service.window)[0])
            raws.append(rec.raw)
            norms.append(norm_vals)
            cycles.append(int(rec.cycles[-1]))
        pred = service.predictor.predict(source, np.stack(X))
        pct = service.conformal.epistemic_percentile(source, pred["sigma_epi"])
        actions, reasons = [], []
        for i in range(len(X)):
            row = {k: float(v[i]) for k, v in pred.items()}
            d = _decision_for(service, source, raws[i], norms[i], cycles[i], row, src.check_stats)
            actions.append(d["action"])
            reasons.extend(d["reasons"])
        return {
            "engines": len(X),
            "epistemic_gate_rate": float(np.mean(pct >= thr)),
            "median_epistemic_percentile": float(np.median(pct)),
            "abstain_rate": float(np.mean([a == "ABSTAIN" for a in actions])),
            "sensor_integrity_flags": int(sum(r == "sensor_integrity" for r in reasons)),
        }

    in_dist = [src.engines[e] for e in src.splits["test"][:max_engines]]
    ood = [tgt.engines[e] for e in tgt.splits["test"][:max_engines]]
    return {"source": source, "target": target, "in_distribution": run(in_dist, False), "out_of_distribution": run(ood, True)}


def fault_injection_test(service, subset: str, n_engines: int) -> dict:
    cfg = service.cfg
    eids = service.datasets[subset].splits["test"][:n_engines]
    clean = [service.analyze(e) for e in eids]
    out = {"subset": subset, "engines": len(eids), "sensor": str(cfg.faults.default_sensor),
           "clean_abstain_rate": float(np.mean([a["action"] == "ABSTAIN" for a in clean])), "faults": {}}
    for kind, magnitude in cfg.faults.magnitudes.items():
        spec = {"sensor": str(cfg.faults.default_sensor), "kind": kind, "magnitude": float(magnitude),
                "start_offset": int(cfg.faults.default_start_offset)}
        res = [service.analyze(e, fault=spec) for e in eids]
        out["faults"][kind] = {
            "abstain_rate": float(np.mean([a["action"] == "ABSTAIN" for a in res])),
            "sensor_flag_rate": float(np.mean([bool(a["sensor_flags"]) for a in res])),
            "action_changed_rate": float(np.mean([a["action"] != c["action"] for a, c in zip(res, clean)])),
            "mean_rul_shift_cycles": float(np.mean([a["rul_mean"] - c["rul_mean"] for a, c in zip(res, clean)])),
        }
    return out


def attribution_test(service, subset: str, n_engines: int) -> dict:
    eids = service.datasets[subset].splits["test"][:n_engines]
    ratios, tops = [], {}
    for e in eids:
        ex = service.explain(e, top_k=3)
        ratios.append(ex["faithfulness"]["faithfulness_ratio"])
        for c in ex["contributions"][:1]:
            tops[c["name"]] = tops.get(c["name"], 0) + 1
    ratios = np.asarray(ratios, dtype=float)
    return {"subset": subset, "engines": len(eids), "median_faithfulness_ratio": float(np.median(ratios)),
            "share_ratio_above_1": float(np.mean(ratios > 1.0)), "most_common_top_sensor": dict(sorted(tops.items(), key=lambda kv: -kv[1]))}


def ablation(service, cfg: Config) -> dict | None:
    """Single-task vs multitask DKL on the same test engines (needs both models on disk)."""
    other = not service.predictor.multitask
    try:
        pred_other = Predictor(cfg, family=service.predictor.family, multitask=other)
    except (FileNotFoundError, ValueError):
        return None
    cal_file = calibration_path(cfg, pred_other.tag)
    conf_other = MondrianConformal.load(cal_file) if cal_file.exists() else calibrate(cfg, pred_other, service.datasets)
    rows = {}
    for label, predictor, conf in (("current", service.predictor, service.conformal), ("other", pred_other, conf_other)):
        for s in service.subsets:
            X, y, _ = last_windows(service.datasets[s], "test")
            p = predictor.predict(s, X)
            lo, hi, _ = conf.interval(s, p["mu"], p["sigma"])
            y_cap = np.minimum(y, float(cfg.data.rul_cap))
            rows.setdefault(s, {})[predictor.tag] = {"rmse_capped_truth": rmse(y_cap, p["mu"]),
                                                     "coverage": coverage(y_cap, lo, hi), "mean_band_width": mean_width(lo, hi)}
    return rows


def plot_reports(metrics: dict, reports_dir, mode: str = "light") -> list[str]:
    """Risk-coverage lines and coverage-by-RUL bars. Each subset keeps its fixed colour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import CHROME, style_mpl_axes, subset_color

    ink = CHROME[mode]
    files = []
    test = metrics["test"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=ink["page"])
    style_mpl_axes(ax, mode)
    for s, m in test.items():
        xs = [c["coverage"] for c in m["risk_coverage"]]
        ys = [c["rmse"] for c in m["risk_coverage"]]
        color = subset_color(s, mode)
        # The final RMSE rides in the legend: end-point labels collided (some finals differ by <0.1).
        ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", label=f"{s} ({ys[-1]:.1f})")
        ax.plot(xs[-1], ys[-1], "o", color=color, markersize=5, markeredgecolor=ink["surface"], markeredgewidth=1.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Share of engines the system makes a call on")
    ax.set_ylabel("RMSE on those engines (cycles)")
    ax.set_title("Abstaining on the most uncertain engines lowers the error", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=ink["text2"], loc="upper left", title="Final RMSE in brackets",
              title_fontsize=8)
    fig.tight_layout()
    p = reports_dir / "risk_coverage.png"
    fig.savefig(p, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    files.append(str(p))

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=ink["page"])
    style_mpl_axes(ax, mode)
    n = max(1, len(test))
    slot = 0.8 / n
    bar_w = slot * 0.85
    first = next(iter(test.values()))
    for i, (s, m) in enumerate(test.items()):
        bins = m["coverage_by_true_rul"]
        xs = np.arange(len(bins)) + i * slot
        ax.bar(xs, [b["coverage"] or 0 for b in bins], width=bar_w, color=subset_color(s, mode), label=s)
    labels = [f"{int(b['rul_from'])}-{int(b['rul_to'])}" for b in first["coverage_by_true_rul"]]
    ax.set_xticks(np.arange(len(labels)) + 0.4 - slot / 2)
    ax.set_xticklabels(labels)
    target = first["target_coverage"]
    ax.axhline(target, color=ink["text2"], ls="--", lw=1, label=f"target {target:.0%}")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("True RUL range (cycles, capped at the RUL cap)")
    ax.set_ylabel("Share of engines inside the 90% band")
    ax.set_title("Band coverage by remaining-life range (test engines)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=ink["text2"], ncol=n + 1, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    p = reports_dir / "coverage_by_rul.png"
    fig.savefig(p, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    files.append(str(p))

    if metrics.get("policy_comparison"):
        p = reports_dir / "policy_comparison.png"
        plot_policy_comparison(metrics["policy_comparison"]["summary"], p)
        files.append(str(p))
    return files


def main(argv: list[str] | None = None) -> None:
    from .service import EngineService

    ap = argparse.ArgumentParser(description="Evaluate the Engine Health Agent")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["multitask", "single"], default=None)
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--plots-only", action="store_true",
                    help="only redraw the PNG charts from the existing reports/metrics.json (no models, no API)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    reports_dir = ensure_dir(path_for(cfg, "reports_dir"))
    if args.plots_only:
        metrics_path = reports_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"{metrics_path} not found. Run a full evaluation first.")
        metrics = json.loads(metrics_path.read_text())
        metrics["charts"] = plot_reports(metrics, reports_dir)
        metrics_path.write_text(json.dumps(_jsonable(metrics), indent=1))
        print(f"redrew {metrics['charts']}")
        return
    multitask = None if args.mode is None else args.mode == "multitask"
    service = EngineService(cfg, multitask=multitask, use_precomputed=False)
    ec = cfg.evaluation

    print("test-set metrics ...")
    metrics: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "model": service.predictor.tag, "test": evaluate_test(service)}
    for s, m in metrics["test"].items():
        print(f"  {s}: RMSE {m['rmse_capped_truth']:.2f} (capped truth) / {m['rmse_uncapped_truth']:.2f} (uncapped) | "
              f"coverage {m['coverage']:.2f} | width {m['mean_band_width']:.1f}")
    print("out-of-distribution test ...")
    metrics["ood"] = ood_cross_subset(service, str(ec.ood_source), str(ec.ood_target), int(ec.ood_max_engines))
    first = service.subsets[0]
    print("fault-injection test ...")
    metrics["fault_injection"] = fault_injection_test(service, first, int(ec.fault_test_engines))
    print("attribution faithfulness ...")
    metrics["attribution"] = attribution_test(service, first, int(ec.attribution_test_engines))
    print("single vs multitask ablation ...")
    metrics["ablation"] = ablation(service, cfg)
    metrics["task_correlation"] = service.predictor.task_correlation()
    if not args.skip_replay:
        print("policy replay on held-out engines ...")
        rep = replay(service)
        metrics["policy_comparison"] = {"summary": rep["summary"], "settings": rep["settings"], "horizon_cycles": rep["horizon_cycles"]}
        for p, v in rep["summary"].items():
            print(f"  {p}: missed {v['failures_missed']}/{v['engines']}, unnecessary {v['unnecessary_inspections']}, "
                  f"median lead {v['median_lead_time_cycles']}")
    metrics["charts"] = plot_reports(metrics, reports_dir)
    out = reports_dir / "metrics.json"
    out.write_text(json.dumps(_jsonable(metrics), indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
