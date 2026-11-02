"""Precompute clean-data results so the app answers instantly.

    python -m ehagent.precompute --mode both

Blobs are fingerprinted with the model/calibration file times; the app drops
them silently when stale. Fault-injected or custom-horizon requests always run live.
"""
from __future__ import annotations

import argparse
import pickle

from .config import ensure_dir, load_config


def main(argv: list[str] | None = None) -> None:
    from .service import EngineService, precomputed_path

    ap = argparse.ArgumentParser(description="Precompute fleet results for the app")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["multitask", "single", "both"], default="both")
    ap.add_argument("--no-explain", action="store_true", help="skip Integrated Gradients (faster)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    modes = [True, False] if args.mode == "both" else [args.mode == "multitask"]
    for multitask in modes:
        service = EngineService(cfg, multitask=multitask, use_precomputed=False)

        blob = {"meta": {"fingerprint": service._fingerprint(), "tag": service.predictor.tag},
                "analysis": {}, "fleet": {}, "trajectory": {}, "explain": {}}
        for s in service.subsets:
            sd = service.datasets[s]
            rows = service.fleet([s])
            blob["fleet"][s] = rows
            for r in rows:
                blob["analysis"][r["engine_id"]] = r
            for eid in sd.splits["test"] + sd.splits["sim"]:
                blob["trajectory"][eid] = service.trajectory(eid, include_truth=False).to_dict(orient="list")
            if not args.no_explain:
                for eid in sd.splits["test"]:
                    blob["explain"][eid] = service.explain(eid, top_k=len(sd.sensors))
            print(f"{s}: {len(rows)} fleet engines, {len(sd.splits['sim'])} replay engines")
        out = precomputed_path(cfg, service.predictor.tag)
        ensure_dir(out.parent)
        with open(out, "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
