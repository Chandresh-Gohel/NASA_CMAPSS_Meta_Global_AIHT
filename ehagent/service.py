"""EngineService: the single object answering every question from the app and the tools.

Test engines' true RUL (the answer key) never leaks out, except through
``trajectory(..., include_truth=True)``, which the app only shows behind the
answer-key switch.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import deletion_test, integrated_gradients, sensor_contributions
from .calibrate import MondrianConformal, calibration_path
from .config import Config, path_for, resolve
from .data import EngineRecord, _make_record, _read_table, load_all_subsets, make_windows, trajectory_ends
from .decision import Decision, decide
from .faults import apply_fault_spec, fault_key
from .predictor import Predictor
from .sensor_checks import run_sensor_checks

_ACTION_RANK = {"INSPECT": 0, "ABSTAIN": 1, "NO_INSPECTION": 2}


def risk_key(d: dict) -> tuple:
    urgent = 0 if d.get("urgency") == "immediate" else 1
    sensor = 0 if "sensor_integrity" in d.get("reasons", []) else 1
    return (_ACTION_RANK.get(d.get("action"), 3), urgent, sensor, -float(d.get("p_fail_within_horizon") or 0.0))


def precomputed_path(cfg: Config, tag: str) -> Path:
    return path_for(cfg, "artifacts_dir", "precomputed", f"{tag}.pkl")


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0

def _file_hash(p: Path) -> str | None:
    """Content hash, so copies, clones and deploys keep the precomputed results valid."""
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _LRU(OrderedDict):
    def __init__(self, maxsize: int = 512):
        super().__init__()
        self.maxsize = maxsize
        self._lock = threading.Lock()

    def get_or(self, key, fn):
        with self._lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
        # Computed outside the lock: a duplicate computation is harmless, a stuck lock is not.
        value = fn()
        with self._lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
            self[key] = value
            if len(self) > self.maxsize:
                self.popitem(last=False)
            return value


class EngineService:
    def __init__(self, cfg: Config, family: str | None = None, multitask: bool | None = None,
                 use_precomputed: bool = True, datasets: dict | None = None,
                 predictor: Predictor | None = None, conformal: MondrianConformal | None = None):
        self.cfg = cfg
        self.subsets = list(cfg.data.subsets)
        self.datasets = datasets or load_all_subsets(cfg, self.subsets)
        self.predictor = predictor or Predictor(cfg, family=family, multitask=multitask, subsets=self.subsets)
        self.calibration_file = calibration_path(cfg, self.predictor.tag)
        self.conformal = conformal or MondrianConformal.load(self.calibration_file)
        self.window = int(cfg.data.window)
        self.min_history = int(cfg.data.min_history)
        self.cap = float(cfg.data.rul_cap)
        self._cache = _LRU(1024)
        self.precomputed = self._load_precomputed() if use_precomputed else {}
        self.uploads: OrderedDict[str, EngineRecord] = OrderedDict()
        self._uploads_lock = threading.Lock()
        try:
            self.keep_last = int(cfg.upload.keep_last)
        except AttributeError:
            self.keep_last = 50
        self._raw_cache: dict[str, pd.DataFrame] = {}

    # def _fingerprint(self) -> dict:
    #     files = sorted({str(self.predictor.model_file(s)) for s in self.subsets})
    #     return {"models": {f: _mtime(Path(f)) for f in files}, "calibration": _mtime(self.calibration_file)}
    def _fingerprint(self) -> dict:
            """Changes when the model, the calibration or any setting that shapes the results changes."""
            files = sorted({str(self.predictor.model_file(s)) for s in self.subsets})
            sections = {k: self.cfg.get(k) for k in ("data", "decision", "calibration", "sensor_checks", "attribution")}
            return {"models": {Path(f).name: _file_hash(Path(f)) for f in files},
                    "calibration": _file_hash(self.calibration_file),
                    "config": hashlib.sha256(json.dumps(sections, sort_keys=True, default=str).encode()).hexdigest()}

    def _load_precomputed(self) -> dict:
        p = precomputed_path(self.cfg, self.predictor.tag)
        if not p.exists():
            return {}
        try:
            with open(p, "rb") as fh:
                blob = pickle.load(fh)
        except Exception:
            return {}
        if blob.get("meta", {}).get("fingerprint") != self._fingerprint():
            return {}
        return blob

    @staticmethod
    def subset_of(engine_id: str) -> str:
        return engine_id.split("-")[0]

    def has_engine(self, engine_id: str) -> bool:
        s = self.subset_of(engine_id)
        if s in self.datasets and engine_id in self.datasets[s].engines:
            return True
        with self._uploads_lock:
            return engine_id in self.uploads

    def engine_ids(self, subset: str | None = None, split: str = "test") -> list[str]:
        subsets = [subset] if subset else self.subsets
        return [e for s in subsets for e in self.datasets[s].splits[split]]

    def is_upload(self, engine_id: str) -> bool:
        with self._uploads_lock:
            return engine_id in self.uploads

    def record(self, engine_id: str, fault: dict | None = None) -> EngineRecord:
        s = self.subset_of(engine_id)
        if s in self.datasets and engine_id in self.datasets[s].engines:
            rec = self.datasets[s].engines[engine_id]
            sd = self.datasets[s]
        else:
            with self._uploads_lock:
                rec = self.uploads.get(engine_id)
            if rec is None:
                raise KeyError(f"Unknown engine '{engine_id}'")
            sd = self.datasets[rec.subset]
        return apply_fault_spec(rec, fault, sd) if fault else rec

    def add_upload(self, df: pd.DataFrame, subset: str) -> list[str]:
        """Store uploaded engines, normalised with the train-fitted stats (never fitted)."""
        if subset not in self.datasets:
            raise KeyError(f"Unknown subset '{subset}'")
        from dataclasses import replace

        sd = self.datasets[subset]
        h = hashlib.sha256()
        h.update(subset.encode("utf-8"))
        h.update(np.ascontiguousarray(df.to_numpy(dtype=float)).tobytes())
        digest = h.hexdigest()[:6]
        units = sorted(int(u) for u in df["unit"].unique().tolist())
        ids: list[str] = []
        new_recs: list[EngineRecord] = []
        for u in units:
            g = df[df["unit"] == u]
            rec = _make_record(subset, int(u), g, sd.sensors, sd.clusterer, sd.normalizer,
                               "upload", float("nan"), False)
            rec = replace(rec, engine_id=f"{subset}-UP-{digest}-{int(u):03d}")
            ids.append(rec.engine_id)
            new_recs.append(rec)
        with self._uploads_lock:
            if all(i in self.uploads for i in ids):
                return list(ids)
            for i, rec in zip(ids, new_recs):
                self.uploads[i] = rec
                self.uploads.move_to_end(i)
            while len(self.uploads) > self.keep_last:
                self.uploads.popitem(last=False)
        return list(ids)

    def _decide_at(self, rec: EngineRecord, end: int, pred: dict, horizon: int | None) -> Decision:
        sd = self.datasets[rec.subset]
        flags = run_sensor_checks(rec.raw[: end + 1], rec.norm[: end + 1], sd.check_stats, self.cfg.sensor_checks)
        return decide(engine_id=rec.engine_id, subset=rec.subset, cycle=int(rec.cycles[end]), mu=float(pred["mu"]),
                      sigma=float(pred["sigma"]), sigma_epi=float(pred["sigma_epi"]), sigma_alea=float(pred["sigma_alea"]),
                      conformal=self.conformal, sensor_flags=flags, cfg=self.cfg, horizon=horizon)

    def _analysis_dict(self, rec: EngineRecord, d: Decision) -> dict:
        out = d.to_dict()
        out["history_cycles"] = rec.length
        out["fault_injected"] = rec.fault
        return out

    def analyze(self, engine_id: str, horizon: int | None = None, fault: dict | None = None) -> dict:
        """Prediction + calibrated band + decision at the engine's latest cycle."""
        if horizon is None and fault is None and self.precomputed.get("analysis", {}).get(engine_id):
            return dict(self.precomputed["analysis"][engine_id])

        def compute() -> dict:
            rec = self.record(engine_id, fault)
            end = rec.length - 1
            pred = self.predictor.predict(rec.subset, make_windows(rec.norm, np.array([end]), self.window))
            return self._analysis_dict(rec, self._decide_at(rec, end, {k: v[0] for k, v in pred.items()}, horizon))

        out = dict(self._cache.get_or(("analyze", engine_id, horizon, fault_key(fault)), compute))
        if self.is_upload(engine_id):
            out["source"] = "uploaded"
        return out

    def trajectory(self, engine_id: str, fault: dict | None = None, horizon: int | None = None,
                   include_truth: bool | None = None) -> pd.DataFrame:
        """Prediction, band and decision at every cycle of an engine's history (the fan chart)."""
        rec = self.record(engine_id, fault)
        if include_truth is None:
            include_truth = rec.full_life
        if include_truth and self.is_upload(engine_id):
            include_truth = False
        pre = self.precomputed.get("trajectory", {}).get(engine_id)
        if horizon is None and fault is None and pre is not None:
            df = pd.DataFrame(pre)
        else:
            def compute() -> pd.DataFrame:
                ends = trajectory_ends(rec.length, self.min_history)
                pred = self.predictor.predict(rec.subset, make_windows(rec.norm, ends, self.window))
                rows = []
                for i, e in enumerate(ends):
                    d = self._decide_at(rec, int(e), {k: v[i] for k, v in pred.items()}, horizon)
                    rows.append({"cycle": d.cycle, "mu": d.rul_mean, "lo": d.rul_lo, "hi": d.rul_hi,
                                 "lo_c": d.rul_lo_conservative, "sigma": d.sigma_cycles, "sigma_epi": d.sigma_epistemic,
                                 "p_fail": d.p_fail_within_horizon, "action": d.action, "reasons": ",".join(d.reasons)})
                return pd.DataFrame(rows)

            df = self._cache.get_or(("trajectory", engine_id, horizon, fault_key(fault)), compute).copy()
        if include_truth:
            truth = {int(c): float(r) for c, r in zip(rec.cycles, rec.rul)}
            df["rul_true"] = [truth.get(int(c), np.nan) for c in df["cycle"]]
        return df

    def explain(self, engine_id: str, top_k: int | None = None, fault: dict | None = None) -> dict:
        top_k = int(top_k or self.cfg.attribution.top_k)
        pre = self.precomputed.get("explain", {}).get(engine_id)
        if fault is None and pre is not None and len(pre.get("contributions", [])) >= top_k:
            out = dict(pre)
            out["contributions"] = pre["contributions"][:top_k]
            return out

        def compute() -> dict:
            rec = self.record(engine_id, fault)
            sd = self.datasets[rec.subset]
            x = make_windows(rec.norm, np.array([rec.length - 1]), self.window)[0]
            ac = self.cfg.attribution
            attr, f_x, f_b = integrated_gradients(self.predictor, rec.subset, x, sd.healthy_baseline, int(ac.ig_steps))
            per = attr.sum(axis=0)
            faith = deletion_test(self.predictor, rec.subset, x, sd.healthy_baseline, per,
                                  int(ac.deletion_top_k), int(ac.deletion_random_trials), int(self.cfg.project.seed))
            return {
                "engine_id": engine_id,
                "method": "Integrated Gradients against an average healthy-engine window",
                "rul_mean": round(f_x, 1),
                "healthy_reference_rul": round(f_b, 1),
                "total_attributed_cycles": round(float(per.sum()), 1),
                "contributions": sensor_contributions(attr, sd.sensors),
                "faithfulness": faith,
            }

        full = self._cache.get_or(("explain", engine_id, fault_key(fault)), compute)
        out = dict(full)
        out["contributions"] = full["contributions"][:top_k]
        return out

    def check_sensors(self, engine_id: str, fault: dict | None = None) -> dict:
        rec = self.record(engine_id, fault)
        sd = self.datasets[rec.subset]
        flags = run_sensor_checks(rec.raw, rec.norm, sd.check_stats, self.cfg.sensor_checks)
        return {"engine_id": engine_id, "status": "suspect" if flags else "ok", "flags": flags,
                "lookback_cycles": int(self.cfg.sensor_checks.lookback)}

    def what_if(self, engine_id: str, horizon: int, fault: dict | None = None) -> dict:
        a = self.analyze(engine_id, horizon=int(horizon), fault=fault)
        keys = ("engine_id", "horizon_cycles", "p_fail_within_horizon", "action", "reasons", "resolution",
                "rul_mean", "rul_lo", "rul_hi", "recheck_in_cycles")
        return {k: a[k] for k in keys}

    def fleet(self, subsets: list[str] | None = None, horizon: int | None = None, faults: dict | None = None) -> list[dict]:
        """Decisions for every test-file engine (the 'current fleet'), riskiest first."""
        faults = faults or {}
        rows: list[dict] = []
        for s in subsets or self.subsets:
            sd = self.datasets[s]
            eids = sd.splits["test"]
            pre = self.precomputed.get("fleet", {}).get(s)
            if horizon is None and pre is not None and not any(e in faults for e in eids):
                rows.extend(dict(r) for r in pre)
                continue
            recs = [self.record(e, faults.get(e)) for e in eids]
            X = np.stack([make_windows(r.norm, np.array([r.length - 1]), self.window)[0] for r in recs])
            pred = self.predictor.predict(s, X)
            for i, r in enumerate(recs):
                d = self._decide_at(r, r.length - 1, {k: v[i] for k, v in pred.items()}, horizon)
                rows.append(self._analysis_dict(r, d))
        return sorted(rows, key=risk_key)

    def plan(self, subsets: list[str] | None = None, horizon: int | None = None, faults: dict | None = None,
             days: int | None = None, slots_per_day: int | None = None, review_slots_per_day: int | None = None) -> dict:
        from .planner import plan_inspections

        return plan_inspections(self.fleet(subsets, horizon, faults), self.cfg, days, slots_per_day, review_slots_per_day)

    def answer_key(self, engine_id: str) -> float:
        """True RUL at the latest cycle (evaluation / demo answer key only - never given to the agent)."""
        if self.is_upload(engine_id):
            raise KeyError("uploaded engines have no answer key")
        return float(self.record(engine_id).rul[-1])

    def export_raw(self, engine_id: str) -> str:
        """This engine's rows from NASA's original file (space-separated, 26 columns, no header)."""
        if self.is_upload(engine_id):
            raise KeyError(f"export_raw is not available for uploads: '{engine_id}'")
        subset = self.subset_of(engine_id)
        rec = self.record(engine_id)
        fname = f"test_{subset}.txt" if "-TE-" in engine_id else f"train_{subset}.txt"
        with self._uploads_lock:
            raw = self._raw_cache.get(fname)
        if raw is None:
            from .constants import RAW_COLUMNS

            raw = _read_table(resolve(self.cfg, self.cfg.paths.raw_dir) / fname)
            with self._uploads_lock:
                self._raw_cache[fname] = raw
        from .constants import RAW_COLUMNS

        g = raw[raw["unit"] == int(rec.unit)].sort_values("cycle")
        lines = []
        for _, row in g.iterrows():
            parts = []
            for col in RAW_COLUMNS:
                if col in ("unit", "cycle"):
                    parts.append(str(int(row[col])))
                else:
                    parts.append(repr(float(row[col])))
            lines.append(" ".join(parts))
        return "\n".join(lines) + "\n"
