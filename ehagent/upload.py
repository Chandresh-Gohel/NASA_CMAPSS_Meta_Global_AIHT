"""Uploaded engine files: parsing plus subset hints (nothing is ever fitted on uploads).

An upload follows NASA's file layout (``train_FD00x.txt`` / ``test_FD00x.txt``):
space-separated, no header, 26 columns in ``RAW_COLUMNS`` order. Commas and
semicolons are accepted too, and a non-numeric first line is skipped as a
header. Columns always map by position.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .constants import OP_COLUMNS, RAW_COLUMNS


class UploadError(ValueError):
    """A problem with the uploaded file. The message goes straight to the uploader."""


def _split_line(raw: str) -> list[str] | None:
    """Fields of one line, or None for a blank line. Empty values raise UploadError."""
    if not raw.strip():
        return None
    if "," in raw or ";" in raw:
        parts = [p.strip() for p in re.split(r"[;,]", raw)]
        if any(p == "" for p in parts):
            raise UploadError(f"missing value (line: {raw.strip()[:80]!r})")
        fields: list[str] = []
        for p in parts:
            fields.extend(p.split())
        if not fields:
            return None
        return fields
    return raw.split()


def _is_numeric_line(fields: list[str]) -> bool:
    try:
        for tok in fields:
            float(tok)
        return True
    except ValueError:
        return False


def parse_engine_file(data: bytes, cfg: Any) -> pd.DataFrame:
    """Parse an upload into a DataFrame with ``RAW_COLUMNS`` columns."""
    if not data or not data.strip():
        raise UploadError("The file is empty.")
    max_mb = float(cfg.upload.max_mb)
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(f"File too large ({len(data)} bytes); the limit is {max_mb:g} MB.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadError("The file must be text (UTF-8).") from exc
    lines = text.splitlines()
    rows: list[list[float]] = []
    skipped_header = False
    for lineno, raw in enumerate(lines, start=1):
        try:
            fields = _split_line(raw)
        except UploadError as exc:
            raise UploadError(f"row {lineno}: {exc}") from exc
        if fields is None:
            continue
        if not skipped_header and not rows:
            # First content line: a non-numeric one is a header, skip it.
            if not _is_numeric_line(fields):
                skipped_header = True
                continue
        if len(fields) < 26:
            raise UploadError(f"row {lineno}: expected 26 columns, got {len(fields)}.")
        fields = fields[:26]
        try:
            rows.append([float(t) for t in fields])
        except ValueError:
            raise UploadError(f"row {lineno}: non-numeric value (26 numeric columns expected).")
    if not rows:
        raise UploadError("The file is empty.")
    max_rows = int(cfg.upload.max_rows)
    if len(rows) > max_rows:
        raise UploadError(f"Too many rows ({len(rows)}); the limit is {max_rows}.")
    arr = np.array(rows, dtype=float)
    units = arr[:, 0]
    cycles = arr[:, 1]
    for i, (u, c) in enumerate(zip(units, cycles)):
        if float(u) != int(float(u)) or float(c) != int(float(c)):
            raise UploadError(f"row {i + 1}: unit and cycle must be whole numbers.")
    df = pd.DataFrame(arr, columns=RAW_COLUMNS)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    max_engines = int(cfg.upload.max_engines)
    unit_ids = list(dict.fromkeys(int(u) for u in df["unit"].tolist()))
    if len(unit_ids) > max_engines:
        raise UploadError(f"Too many engines ({len(unit_ids)}); the limit is {max_engines}.")
    min_history = int(cfg.data.min_history)
    for u in unit_ids:
        g = df[df["unit"] == u]
        cyc = g["cycle"].tolist()
        if len(cyc) < min_history:
            raise UploadError(f"engine {u}: only {len(cyc)} cycles, at least {min_history} needed.")
        for a, b in zip(cyc, cyc[1:]):
            if b <= a:
                raise UploadError(f"engine {u}: cycles must strictly increase (cycle {a} followed by {b}).")
    return df[[*RAW_COLUMNS]]


def count_conditions(df: pd.DataFrame, datasets: dict) -> int:
    """How many operating conditions the rows span (fitted clusterer only)."""
    clusterer = None
    for name, sd in datasets.items():
        if int(getattr(sd, "n_conditions", 1)) > 1:
            clusterer = sd.clusterer
            break
    if clusterer is None:
        return 1
    ops = df[OP_COLUMNS].to_numpy(dtype=float)
    return int(len(set(clusterer.predict(ops).tolist())))


def suggest_subset(df: pd.DataFrame, datasets: dict) -> tuple[str, str]:
    """The subset to pre-select in the UI, plus a one-line reason."""
    n = count_conditions(df, datasets)
    if n <= 1:
        return "FD001", "one operating condition; FD001 and FD003 can't be told apart from the operating settings"
    return "FD002", f"the rows span {n} operating conditions"


def mismatch_warning(df: pd.DataFrame, subset: str, datasets: dict) -> str | None:
    """Warn when the condition count disagrees with the chosen subset. Warn, don't block."""
    sd = datasets.get(subset)
    if sd is None:
        return f"Unknown engine type {subset}."
    n = count_conditions(df, datasets)
    expected = int(getattr(sd, "n_conditions", 1))
    if expected <= 1 and n > 1:
        return (f"Data spans {n} operating conditions but {subset} expects 1. "
                "The sensor-integrity and unfamiliar-engine gates are the safety net; expect ABSTAIN.")
    if expected > 1 and n <= 1:
        return (f"Data shows 1 operating condition but {subset} expects {expected}. "
                "The sensor-integrity and unfamiliar-engine gates are the safety net; expect ABSTAIN.")
    if expected > 1 and n != expected:
        return (f"Data spans {n} operating conditions but {subset} expects {expected}. "
                "The sensor-integrity and unfamiliar-engine gates are the safety net; expect ABSTAIN.")
    return None
