#!/usr/bin/env bash
# Full pipeline, same order as the README (Option B). Needs CMAPSSData/ at the root.
set -e
python -m ehagent.train prepare
python -m ehagent.train baseline --mode multitask
python -m ehagent.train dkl --mode multitask
python -m ehagent.train baseline --mode single
python -m ehagent.train dkl --mode single
python -m ehagent.calibrate --mode both
python -m ehagent.precompute --mode both
python -m ehagent.evaluate
python scripts/results_table.py
python scripts/rul_stage_report.py
