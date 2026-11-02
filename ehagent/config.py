"""Configuration: loading the YAML, resolving paths, seeding, picking a device."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class Config(dict):
    """A plain dict with attribute-style reads, so ``cfg.data.window`` works."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    # Turn nested dicts into Configs (lists pass through element-wise).
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def repo_root() -> Path:
    """The repository root: the folder that contains ``ehagent/``."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    """Read the YAML config. ``EHA_CONFIG`` swaps the whole file when set."""
    if path is None:
        path = os.environ.get("EHA_CONFIG", str(repo_root() / "configs" / "default.yaml"))
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = repo_root() / path
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        _deep_update(raw, overrides)
    cfg = _wrap(raw)
    cfg["root"] = str(repo_root())
    return cfg


def resolve(cfg: Config, rel: str | os.PathLike) -> Path:
    """Absolute path for a config-relative one (absolute inputs pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["root"]) / p


def path_for(cfg: Config, key: str, *parts: str) -> Path:
    """A path inside one of the configured folders, e.g. ``path_for(cfg, "artifacts_dir", "models")``."""
    base = resolve(cfg, cfg.paths[key])
    return base.joinpath(*parts) if parts else base


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # some entry points never touch torch
        pass


def get_device(cfg: Config):
    import torch

    want = str(cfg.project.get("device", "auto"))
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def load_dotenv_if_present() -> None:
    """Load ``.env`` from the repo root when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = repo_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
