"""Step 3 exit check: splits, windows, conditions. No API key needed.

Run from repo root:  python scripts/check_3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehagent.config import load_config
from ehagent.data import last_windows, prepare_subset, training_windows


def main() -> None:
    cfg = load_config()
    window = int(cfg.data.window)
    for subset in list(cfg.data.subsets):
        sd = prepare_subset(cfg, subset)
        counts = {k: len(v) for k, v in sd.splits.items()}
        print(f"{subset}: " + " / ".join(f"{v} {k}" for k, v in counts.items()))

        # No engine may sit in two splits.
        seen: dict[str, str] = {}
        for split, eids in sd.splits.items():
            for eid in eids:
                assert eid not in seen, f"{eid} in both {seen[eid]} and {split}"
                seen[eid] = split

        # Windows come out as (N, 30, 14).
        Xtr, _, _ = training_windows(sd, "train", window, int(cfg.data.min_history), float(cfg.data.rul_cap))
        assert Xtr.ndim == 3 and Xtr.shape[1] == 30 and Xtr.shape[2] == 14, f"{subset} train windows {Xtr.shape}"
        Xte, _, _ = last_windows(sd, "test")
        assert Xte.ndim == 3 and Xte.shape[1:] == (30, 14), f"{subset} test windows {Xte.shape}"
        print(f"  windows train {Xtr.shape} test-last {Xte.shape}")

        # The multi-condition subsets must show all 6.
        n_cond = len(set(int(v) for rec in sd.engines.values() for v in rec.cond.tolist()))
        print(f"  conditions found: {n_cond} (expected {cfg.data.n_conditions[subset]})")
        if subset in ("FD002", "FD004"):
            assert n_cond == 6, f"{subset}: expected 6 conditions, got {n_cond}"
    print("Step 3 OK")


if __name__ == "__main__":
    main()
