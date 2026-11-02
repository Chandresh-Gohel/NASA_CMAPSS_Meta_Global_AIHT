"""The decision engine: inspect / no inspection / abstain, in plain auditable code.

No language model is involved in this call. The gates run in order:

1. Sensor-integrity gate  -> ABSTAIN when a sensor looks stuck, jumps or is out of range.
2. Novelty gate            -> ABSTAIN when epistemic uncertainty ranks in the top percentile
                             of the calibration fleet (an engine unlike anything in training).
3. Failure probability     -> P(RUL <= H) from the *calibrated* Gaussian, truncated at 0:
        p >= p_inspect  -> INSPECT
        p <= p_clear    -> NO_INSPECTION
        otherwise       -> ABSTAIN: the band straddles the horizon, so neither call is safe.

A wide band far from the horizon still clears confidently: uncertainty only
blocks a decision when it changes the decision.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm


@dataclass
class Decision:
    engine_id: str
    subset: str
    cycle: int
    action: str
    reasons: list = field(default_factory=list)
    rul_mean: float = 0.0
    rul_lo: float = 0.0
    rul_hi: float = 0.0
    rul_lo_conservative: float = 0.0
    interval_coverage: float = 0.9
    healthy_region: bool = False
    rul_cap: float = 125.0
    p_fail_within_horizon: float = 0.0
    horizon_cycles: int = 30
    sigma_cycles: float = 0.0
    sigma_epistemic: float = 0.0
    sigma_aleatoric: float = 0.0
    epistemic_percentile: float | None = None
    sensor_flags: list = field(default_factory=list)
    recheck_in_cycles: int | None = None
    urgency: str = "routine"
    resolution: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("rul_mean", "rul_lo", "rul_hi", "rul_lo_conservative", "sigma_cycles", "sigma_epistemic", "sigma_aleatoric", "rul_cap"):
            d[k] = round(float(d[k]), 1)
        d["p_fail_within_horizon"] = round(float(d["p_fail_within_horizon"]), 3)
        ep = d["epistemic_percentile"]
        d["epistemic_percentile"] = None if ep is None or not math.isfinite(float(ep)) else round(float(ep), 1)
        d["interval_coverage"] = round(float(d["interval_coverage"]), 2)
        d["interval_width"] = round(float(self.rul_hi - self.rul_lo), 1)
        d["mean_minus_horizon"] = round(float(self.rul_mean - self.horizon_cycles), 1)
        return d


def failure_probability(mu: float, sigma: float, horizon: float) -> float:
    """P(RUL <= horizon) under a Gaussian truncated to RUL >= 0."""
    sigma = max(float(sigma), 1e-6)
    a = norm.cdf((float(horizon) - mu) / sigma)
    b = norm.cdf((0.0 - mu) / sigma)
    denom = max(1.0 - b, 1e-12)
    return float(np.clip((a - b) / denom, 0.0, 1.0))


def decide(*, engine_id: str, subset: str, cycle: int, mu: float, sigma: float, sigma_epi: float,
           sigma_alea: float, conformal, sensor_flags: list[dict], cfg: Any, horizon: int | None = None) -> Decision:
    dc, cc = cfg.decision, cfg.calibration
    horizon = int(horizon if horizon is not None else dc.horizon_cycles)
    alpha, alpha_c = float(cc.alpha), float(cc.alpha_conservative)
    cap = float(cfg.data.rul_cap)
    max_recheck = int(dc.get("max_recheck_cycles", 50))

    lo, hi, half = (float(v[0]) for v in conformal.interval(subset, [mu], [sigma], alpha))
    lo_c = float(conformal.interval(subset, [mu], [sigma], alpha_c)[0][0])
    lower_tail = float(conformal.lower_tail(alpha)) if hasattr(conformal, "lower_tail") else alpha / 2.0
    sigma_cal = max(half / float(norm.ppf(1 - lower_tail)), 1e-3)
    p = failure_probability(mu, sigma_cal, horizon)
    p_urgent = failure_probability(mu, sigma_cal, float(dc.urgent_cycles))
    epi_pct = float(conformal.epistemic_percentile(subset, [sigma_epi])[0])
    healthy = hi >= cap

    reasons: list[str] = []
    recheck: int | None = None
    high_flags = [f for f in sensor_flags if f.get("severity") == "high"]

    if high_flags:
        action, urgency = "ABSTAIN", "soon"
        reasons.append("sensor_integrity")
        names = ", ".join(sorted({f"{f['name']} ({f['sensor']})" for f in high_flags}))
        checks = ", ".join(sorted({f["check"].replace("_", " ") for f in high_flags}))
        resolution = (f"Verify {names}: possible {checks}. The prediction is withheld until the sensor data "
                      "can be trusted; an engineer should review this engine.")
    elif math.isfinite(epi_pct) and epi_pct >= float(dc.epistemic_percentile_abstain):
        action, urgency = "ABSTAIN", "soon"
        reasons.append("unfamiliar_engine")
        resolution = ("This engine's sensor pattern is unlike the engines the model learned from, so its estimate "
                      "is not reliable. Route it to an engineer for a manual assessment.")
    elif p >= float(dc.p_inspect):
        action = "INSPECT"
        reasons.append("failure_probability_high")
        urgent = p_urgent >= float(dc.p_inspect) or lo_c <= float(dc.urgent_cycles)
        urgency = "immediate" if urgent else "soon"
        resolution = f"Schedule an inspection within {max(0, int(lo_c))} cycles (conservative lower bound of remaining life)."
    elif p <= float(dc.p_clear):
        action, urgency = "NO_INSPECTION", "routine"
        reasons.append("failure_probability_low")
        if healthy:
            reasons.append("healthy_region")
        recheck = int(np.clip(lo - horizon, 1, max_recheck))
        resolution = f"No inspection needed now. Re-evaluate in {recheck} cycles."
    else:
        action, urgency = "ABSTAIN", "soon"
        reasons.append("ambiguous_band")
        recheck = int(np.clip(lo_c / 2.0, 1, max_recheck))
        resolution = (f"The uncertainty band straddles the {horizon}-cycle planning horizon, so neither 'inspect' "
                      f"nor 'no inspection' is a safe call. Re-assess in {recheck} cycles or escalate for a manual inspection.")

    return Decision(
        engine_id=engine_id, subset=subset, cycle=int(cycle), action=action, reasons=reasons,
        rul_mean=float(mu), rul_lo=lo, rul_hi=hi, rul_lo_conservative=lo_c, interval_coverage=1 - alpha,
        healthy_region=bool(healthy), rul_cap=cap, p_fail_within_horizon=p, horizon_cycles=horizon,
        sigma_cycles=sigma_cal, sigma_epistemic=float(sigma_epi), sigma_aleatoric=float(sigma_alea),
        epistemic_percentile=epi_pct if math.isfinite(epi_pct) else None, sensor_flags=list(sensor_flags),
        recheck_in_cycles=recheck, urgency=urgency, resolution=resolution,
    )
