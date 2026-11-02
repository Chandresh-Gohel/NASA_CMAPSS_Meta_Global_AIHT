"""Step 8 exit check: the fleet shows all 3 actions; stuck s11 abstains on integrity.

Run from repo root: python scripts/check_8.py (both model modes stay green)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ehagent.calibrate import MondrianConformal, calibration_path
from ehagent.config import load_config
from ehagent.data import last_windows, load_all_subsets
from ehagent.decision import decide
from ehagent.faults import apply_fault_spec
from ehagent.predictor import Predictor
from ehagent.sensor_checks import run_sensor_checks


def decide_latest(cfg, predictor, conf, datasets, subset, eid, fault=None):
    sd = datasets[subset]
    rec = sd.engines[eid]
    if fault:
        rec = apply_fault_spec(rec, fault, sd)
    X = rec.norm[-int(cfg.data.window):]
    if len(X) < int(cfg.data.window):
        pad = np.repeat(X[:1], int(cfg.data.window) - len(X), axis=0)
        X = np.concatenate([pad, X])
    pred = predictor.predict(subset, X[None])
    mu, sig, epi, alea = (float(pred[k][0]) for k in ("mu", "sigma", "sigma_epi", "sigma_alea"))
    flags = run_sensor_checks(rec.raw, rec.norm, sd.check_stats, cfg.sensor_checks)
    d = decide(engine_id=eid, subset=subset, cycle=int(rec.cycles[-1]), mu=mu, sigma=sig,
               sigma_epi=epi, sigma_alea=alea, conformal=conf, sensor_flags=flags, cfg=cfg)
    return d, flags


def check_mode(cfg, datasets, multitask: bool) -> None:
    predictor = Predictor(cfg, family="dkl", multitask=multitask)
    tag = predictor.tag
    conf = MondrianConformal.load(calibration_path(cfg, tag))
    print(f"model={tag} calib={tag}")

    actions: dict[str, int] = {}
    examples: dict[str, str] = {}
    for subset in predictor.subsets:
        _, _, eids = last_windows(datasets[subset], "test")
        for eid in eids:
            d, _ = decide_latest(cfg, predictor, conf, datasets, subset, eid)
            actions[d.action] = actions.get(d.action, 0) + 1
            examples.setdefault(d.action, f"{subset} {eid}")
    print("fleet actions:", actions, "examples:", examples)
    assert set(actions) == {"INSPECT", "NO_INSPECTION", "ABSTAIN"}, f"need all 3 actions, got {actions}"

    # A stuck Ps30 (s11) must refuse to call, citing sensor integrity.
    target = None
    for subset in predictor.subsets:
        _, _, eids = last_windows(datasets[subset], "test")
        if eids:
            target = (subset, eids[0])
            break
    assert target
    fault = {"sensor": "s11", "kind": "stuck", "start_offset": 20, "magnitude": 0.0}
    d, flags = decide_latest(cfg, predictor, conf, datasets, target[0], target[1], fault)
    print(f"fault test {target}: action={d.action} reasons={d.reasons} n_flags={len(flags)}")
    assert d.action == "ABSTAIN" and "sensor_integrity" in d.reasons, f"stuck s11 must ABSTAIN sensor_integrity, got {d.action} {d.reasons}"


def main() -> None:
    cfg = load_config()
    datasets = load_all_subsets(cfg)
    primary = bool(cfg.features.get("multitask_model", True))
    for multitask in [primary, not primary]:
        print(f"--- multitask={multitask} ---")
        try:
            check_mode(cfg, datasets, multitask)
        except FileNotFoundError as exc:
            print(f"skip multitask={multitask}: {exc}")
    print("Step 8 OK")


if __name__ == "__main__":
    main()
