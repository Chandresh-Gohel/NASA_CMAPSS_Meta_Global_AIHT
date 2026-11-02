"""C-MAPSS loading, per-condition normalisation, engine-ID splits and windowing.

Engine ids read ``FD001-TR-012`` for run-to-failure training engines and
``FD001-TE-012`` for truncated test engines (truth only in RUL_FD001.txt).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ensure_dir, path_for, resolve
from .constants import OP_COLUMNS, RAW_COLUMNS
from .sensor_checks import SensorCheckStats, fit_check_stats

SPLITS = ("train", "val", "calib", "sim", "test")


@dataclass
class EngineRecord:
    engine_id: str
    subset: str
    unit: int
    split: str
    cycles: np.ndarray      # (T,)
    raw: np.ndarray         # (T, F) selected sensors, raw units
    ops: np.ndarray         # (T, 3) operating settings
    cond: np.ndarray        # (T,) operating-condition id per cycle
    norm: np.ndarray        # (T, F) condition-normalised sensors (float32)
    rul: np.ndarray         # (T,) true RUL per cycle (uncapped)
    full_life: bool         # run-to-failure training engines only
    fault: dict | None = None

    def copy(self) -> "EngineRecord":
        return replace(self, raw=self.raw.copy(), norm=self.norm.copy(), fault=dict(self.fault) if self.fault else None)

    @property
    def length(self) -> int:
        return int(len(self.cycles))


@dataclass
class ConditionClusterer:
    """Maps each cycle to a discrete operating condition (k-means over op settings)."""

    n_conditions: int
    centers: np.ndarray | None = None
    offset: np.ndarray | None = None
    scale: np.ndarray | None = None

    @classmethod
    def fit(cls, ops: np.ndarray, n_conditions: int, seed: int = 0) -> "ConditionClusterer":
        if n_conditions <= 1:
            return cls(1)
        from sklearn.cluster import KMeans

        offset = ops.mean(axis=0)
        scale = ops.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        km = KMeans(n_clusters=n_conditions, n_init=10, random_state=seed).fit((ops - offset) / scale)
        centers = km.cluster_centers_
        order = np.lexsort((centers[:, 1], centers[:, 0]))  # stable, readable condition ids
        return cls(n_conditions, centers[order], offset, scale)

    def predict(self, ops: np.ndarray) -> np.ndarray:
        if self.n_conditions <= 1 or self.centers is None:
            return np.zeros(len(ops), dtype=np.int64)
        z = (np.asarray(ops, dtype=float) - self.offset) / self.scale
        dist = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=-1)
        return dist.argmin(axis=1).astype(np.int64)


@dataclass
class ConditionNormalizer:
    """Per-condition z-scoring of the sensors (fitted on the train split only)."""

    mean: np.ndarray  # (C, F)
    std: np.ndarray   # (C, F)

    @classmethod
    def fit(cls, raw: np.ndarray, cond: np.ndarray, n_conditions: int) -> "ConditionNormalizer":
        n_feat = raw.shape[1]
        mean = np.zeros((n_conditions, n_feat))
        std = np.ones((n_conditions, n_feat))
        for c in range(n_conditions):
            mask = cond == c
            if mask.sum() < 2:
                continue
            mean[c] = raw[mask].mean(axis=0)
            s = raw[mask].std(axis=0)
            std[c] = np.where(s < 1e-6, 1.0, s)
        return cls(mean, std)

    def transform(self, raw: np.ndarray, cond: np.ndarray) -> np.ndarray:
        return ((raw - self.mean[cond]) / self.std[cond]).astype(np.float32)


@dataclass
class SubsetData:
    name: str
    sensors: list[str]
    n_conditions: int
    clusterer: ConditionClusterer
    normalizer: ConditionNormalizer
    engines: dict[str, EngineRecord]
    splits: dict[str, list[str]]
    check_stats: SensorCheckStats
    healthy_baseline: np.ndarray   # (L, F) mean normalised window of healthy engines
    window: int
    rul_cap: float


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download CMAPSSData.zip and unzip it so "
            f"train_/test_/RUL_FD00x.txt exist under the raw_dir in configs/default.yaml."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(RAW_COLUMNS)]
    df.columns = RAW_COLUMNS
    return df


def load_raw(raw_dir: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = _read_table(raw_dir / f"train_{subset}.txt")
    test = _read_table(raw_dir / f"test_{subset}.txt")
    rul_path = raw_dir / f"RUL_{subset}.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"{rul_path} not found")
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)
    return train, test, rul


def split_units(units: list[int], fractions: Any, rng: np.random.Generator) -> dict[str, list[int]]:
    """Split whole engines (never windows) so nothing leaks across train/val/calib/sim."""
    shuffled = [int(u) for u in rng.permutation(units)]
    n = len(shuffled)
    other = ["val", "calib", "sim"]
    counts = [max(1, int(round(float(fractions[k]) * n))) for k in other]
    n_train = n - sum(counts)
    if n_train < 1:
        raise ValueError("Split fractions leave no training engines")
    out = {"train": sorted(shuffled[:n_train])}
    i = n_train
    for name, count in zip(other, counts):
        out[name] = sorted(shuffled[i : i + count])
        i += count
    return out


def _make_record(subset: str, unit: int, g: pd.DataFrame, sensors: list[str], clusterer: ConditionClusterer,
                 normalizer: ConditionNormalizer, split: str, rul_last: float, full_life: bool) -> EngineRecord:
    g = g.sort_values("cycle")
    cycles = g["cycle"].to_numpy(dtype=int)
    raw = g[sensors].to_numpy(dtype=float)
    ops = g[OP_COLUMNS].to_numpy(dtype=float)
    cond = clusterer.predict(ops)
    norm = normalizer.transform(raw, cond)
    rul = (cycles[-1] - cycles).astype(float) + float(rul_last)
    tag = "TR" if full_life else "TE"
    return EngineRecord(
        engine_id=f"{subset}-{tag}-{unit:03d}", subset=subset, unit=unit, split=split, cycles=cycles,
        raw=raw, ops=ops, cond=cond, norm=norm, rul=rul, full_life=full_life,
    )


def healthy_baseline(records: list[EngineRecord], window: int, cap: float) -> np.ndarray:
    """Mean normalised window over healthy (capped-RUL) training windows: the IG reference."""
    acc = []
    n_feat = records[0].norm.shape[1] if records else 0
    for rec in records:
        ends = np.where(rec.rul >= cap)[0]
        ends = ends[ends >= window - 1]
        if len(ends):
            acc.append(make_windows(rec.norm, ends, window).mean(axis=0))
    if not acc:
        return np.zeros((window, n_feat), dtype=np.float32)
    return np.mean(acc, axis=0).astype(np.float32)


def prepare_subset(cfg: Config, subset: str) -> SubsetData:
    raw_dir = resolve(cfg, cfg.paths.raw_dir)
    train_df, test_df, rul_test = load_raw(raw_dir, subset)
    sensors = list(cfg.data.sensors)
    n_cond = int(cfg.data.n_conditions[subset])
    seed = int(cfg.project.seed)
    window = int(cfg.data.window)
    cap = float(cfg.data.rul_cap)

    clusterer = ConditionClusterer.fit(train_df[OP_COLUMNS].to_numpy(dtype=float), n_cond, seed)
    units = sorted(int(u) for u in train_df["unit"].unique())
    rng = np.random.default_rng(seed + sum(ord(c) for c in subset))
    unit_splits = split_units(units, cfg.data.split, rng)
    split_of_unit = {u: name for name, us in unit_splits.items() for u in us}

    tr_mask = train_df["unit"].isin(unit_splits["train"]).to_numpy()
    tr_cond = clusterer.predict(train_df.loc[tr_mask, OP_COLUMNS].to_numpy(dtype=float))
    normalizer = ConditionNormalizer.fit(train_df.loc[tr_mask, sensors].to_numpy(dtype=float), tr_cond, n_cond)

    engines: dict[str, EngineRecord] = {}
    splits: dict[str, list[str]] = {name: [] for name in SPLITS}
    for unit, g in train_df.groupby("unit", sort=True):
        rec = _make_record(subset, int(unit), g, sensors, clusterer, normalizer, split_of_unit[int(unit)], 0.0, True)
        engines[rec.engine_id] = rec
        splits[rec.split].append(rec.engine_id)
    for unit, g in test_df.groupby("unit", sort=True):
        rec = _make_record(subset, int(unit), g, sensors, clusterer, normalizer, "test", float(rul_test[int(unit) - 1]), False)
        engines[rec.engine_id] = rec
        splits["test"].append(rec.engine_id)

    train_recs = [engines[e] for e in splits["train"]]
    check_stats = fit_check_stats(train_recs, sensors, cfg.sensor_checks)
    baseline = healthy_baseline(train_recs, window, cap)
    return SubsetData(subset, sensors, n_cond, clusterer, normalizer, engines, splits, check_stats, baseline, window, cap)


def subset_path(cfg: Config, subset: str) -> Path:
    return path_for(cfg, "artifacts_dir", "data", f"{subset}.pkl")


def save_subset(cfg: Config, sd: SubsetData) -> Path:
    p = subset_path(cfg, sd.name)
    ensure_dir(p.parent)
    with open(p, "wb") as fh:
        import pickle

        pickle.dump(sd, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_subset(cfg: Config, subset: str) -> SubsetData:
    p = subset_path(cfg, subset)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `python -m ehagent.train prepare` first.")
    with open(p, "rb") as fh:
        import pickle

        return pickle.load(fh)


def load_all_subsets(cfg: Config, subsets: list[str] | None = None) -> dict[str, SubsetData]:
    return {s: load_subset(cfg, s) for s in (subsets or list(cfg.data.subsets))}


def make_windows(values: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    """Windows ending at each index in ``ends`` (inclusive); short histories pad with the first row."""
    ends = np.asarray(ends, dtype=int)
    n_rows = values.shape[0]
    offsets = np.arange(-window + 1, 1)
    idx = np.clip(ends[:, None] + offsets[None, :], 0, n_rows - 1)
    return np.ascontiguousarray(values[idx], dtype=np.float32)


def trajectory_ends(n_cycles: int, min_history: int) -> np.ndarray:
    start = min(max(min_history - 1, 0), n_cycles - 1)
    return np.arange(start, n_cycles)


def training_windows(sd: SubsetData, split: str, window: int, min_history: int, cap: float):
    """Every window of the engines in ``split``: X (N, L, F), capped y (N,), [(engine_id, cycle)]."""
    xs, ys, index = [], [], []
    for eid in sd.splits[split]:
        rec = sd.engines[eid]
        ends = np.arange(min(max(min_history - 1, 0), rec.length - 1), rec.length)
        xs.append(make_windows(rec.norm, ends, window))
        ys.append(np.minimum(rec.rul[ends], cap))
        index.extend((eid, int(rec.cycles[e])) for e in ends)
    if not xs:
        n_feat = len(sd.sensors)
        return np.zeros((0, window, n_feat), np.float32), np.zeros(0, np.float32), []
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), index


def cut_windows(sd: SubsetData, split: str, cuts_per_engine: int, window: int, min_history: int, cap: float,
                rng: np.random.Generator):
    """Random truncation points per engine, mimicking how the official test set was cut."""
    xs, ys, index = [], [], []
    for eid in sd.splits[split]:
        rec = sd.engines[eid]
        candidates = np.arange(min(max(min_history - 1, 0), rec.length - 1), rec.length)
        k = min(int(cuts_per_engine), len(candidates))
        ends = np.sort(rng.choice(candidates, size=k, replace=False))
        xs.append(make_windows(rec.norm, ends, window))
        ys.append(np.minimum(rec.rul[ends], cap))
        index.extend((eid, int(rec.cycles[e])) for e in ends)
    if not xs:
        n_feat = len(sd.sensors)
        return np.zeros((0, window, n_feat), np.float32), np.zeros(0, np.float32), []
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), index


def last_windows(sd: SubsetData, split: str = "test"):
    """The latest window of every engine in ``split``: X, uncapped true RUL, engine ids."""
    eids = list(sd.splits[split])
    xs = [make_windows(sd.engines[e].norm, np.array([sd.engines[e].length - 1]), sd.window)[0] for e in eids]
    y = np.array([sd.engines[e].rul[-1] for e in eids], dtype=float)
    if not xs:
        return np.zeros((0, sd.window, len(sd.sensors)), np.float32), y, eids
    return np.stack(xs).astype(np.float32), y, eids
