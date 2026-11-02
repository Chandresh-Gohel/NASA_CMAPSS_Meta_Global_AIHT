# Work done — continuous build log

One file for the whole project, across all sessions and chats. Newest entry at the bottom.
Per-step authority stays in `.agents/memory/progress.md`; this file is the narrative that
connects the sessions (including the archived `greeting.json` session export at repo root).

Rule going forward: append one dated entry per session/task. Never rewrite history.

---

## 2026-10-28 — Session 1 ("greeting.json" session): Steps 1–8 from an empty repo

- Started with only `AGENTS.md`, `docs/SPEC.md`, `reference/` (unexecuted). User ruled: follow
  AGENTS.md, don't run `.py` without asking. Data confirmed real: all 12 C-MAPSS files in
  `CMAPSSData/` at repo root (`paths.raw_dir: CMAPSSData` deviation recorded).
- Built, files created by agent, checks run by user: Step 1 skeleton (`data.window` = 30 OK),
  Step 2 smoke client, Step 3 data prep (engine-ID splits; windows `(N,30,14)`; 6 conditions in
  FD002/FD004; `train prepare` saved 4 `.pkl`), Step 4 CNN baseline (multitask best val RMSE
  15.51, epoch 1), Step 5 DKL single FD001 (CNN 15.80 → DKL 16.26, sigma > 0).
- Step 6 LMC multitask wired; Step 7 hit the wall: val coverage FD001 0.975 / FD002 0.946
  (over-cover, outside the 0.87–0.93 gate). User: skip to Step 8, come back to 7.
- Step 8 `decision.py` + `faults.py` + `check_8.py` created. Calibration rework followed:
  asymmetric bands, `diag_7.py`, retrain-vs-revert debate; user chose retrain with new
  hyperparams (dropout 0.3, sn_coeff 0.9, pretrain_lr 5e-4, weight_decay 1e-2,
  dkl_lr_features 1e-5, cuts 30, recheck 20). Session ended mid-saga (export outcome: failed).

## 2026-10-29/30 — Steps 8–12 + frozen results

- Retrain + Section 12 config lock: p_inspect 0.50 / p_clear 0.10 (sweep: 0/106 missed,
  35 unnecessary), ABSTAIN = engineer review, multitask default.
- Frozen `reports/metrics.json` (2026-10-30T12:30:36): RMSE 13.98 / 14.24 / 14.02 / 15.49;
  coverage 0.88 / 0.92 / 0.93 / 0.83; replay 0/106 missed, 35 unnecessary, 265 reviews,
  lead 25; faults 100% abstain; attribution median 1.59; OOD 100% vs 11%.
- Balanced-sampling experiment (NEW_REPO_TRAIN clone): rejected — only FD003 RMSE improved
  (14.02→12.84); FD001/FD002/FD004 RMSE, replay (35→36), calibration val, attribution all
  regressed. Root kept as the shippable canonical copy.

## 2026-10-31 — Step 13 addition: engine upload (DONE)

- Spec `.agents/tasks/engine_upload.md` (user-approved Oct 30). Upload stays in the Engine tab;
  no new deps; no `decision`/`calibration`/`sensor_checks` changes.
- New/changed: `ehagent/upload.py`; `ehagent/service.py` (uploads, `export_raw`, locks, truth
  guard); `ehagent/agent/client.py` (atomic cache write); `app/streamlit_app.py` (uploader,
  type select + warnings, session uploads, download button, captions, label fix);
  `scripts/check_upload.py`; `configs/default.yaml` (`device: auto`, `engine_upload`, `upload:`).
- Exit checks, all green: `check_upload.py` offline OK (exact round trips, IDs stable, fleet 707
  unchanged, wrong-type ABSTAIN `sensor_integrity`, 6 bad files rejected, offline template
  guard-clean); `check_13.py` still passes; `check_upload.py --live` OK (`used_fallback=False`,
  reviewer `approve`, no violations, cached rerun, no new calls). Two script bugs found and
  fixed along the way (env scoping, cache hermeticity, non-numeric test row).
- Remaining: manual Streamlit walkthrough (8 steps); README mention (step 14); fresh-clone dry
  run; offline demo record; video ≤3 min; submit by Wed Nov 4, 22:00 IST.
- Upload cap fix (same day): widget showed Streamlit's 200MB default while config rejected >2MB.
  Set both to 10MB (`upload.max_mb`, new `.streamlit/config.toml` `maxUploadSize`). Row/engine
  caps already bound real uploads (~5MB worst case), so no safety change.
- Whole-file upload (user has only official train/test files): `max_engines` 20 -> 300 (FD002
  holds 260), `keep_last` 50 -> 500 (must exceed max_engines or a full file evicts itself).
  `max_rows`/`max_mb` already cover the largest file (~65k rows, ~16MB). Check script's
  "too many engines" case now scales with config instead of hardcoding 21 units.

## 2026-11-01 — Fan-chart truth lines, chart cleanup, off-topic probe
- Fan chart: the uncapped `rul_true` line (a straight decline from 200-300+) dwarfed the capped
  prediction. Now plots `rul_target = min(rul_true, 125)` solid as "Answer key (capped at 125)",
  uncapped truth faint dotted, plus caption "Above 125 cycles healthy engines look alike, so the
  model reports 'healthy' instead of guessing." Computed app-side; `simulate.py` replay still
  uses uncapped `rul_true`; service contract unchanged.
- Overlap fixes: fan-chart hline labels split to opposite sides ("Horizon N" left, "RUL cap" right)
  with shorter text; legend top margin 36 -> 64 for the wrapped 2-row legend; contributions chart
  right margin 20 -> 48 (outside labels clipped); risk-coverage end labels staggered
  (-18/-6/+6/+18pt, FD001/FD003 final RMSEs differ by <0.1); coverage-by-RUL "target 90%"
  moved off the bars into the legend.
- `evaluate.py --plots-only`: redraws the 3 PNGs from existing `metrics.json` (no models/eval).
- New `scripts/probe_offtopic.py [--live]`: 5 fixed off-topic questions (incl. one instruction-
  override probe) on one INSPECT engine; asserts containment (recommendation == authoritative).

## 2026-11-02 — Chart fixes round 2 (from user screenshots) + probe results
- Probe `--live` (user run, 20 new calls, cached): all 5 contained (INSPECT kept, approve, no
  violations). Finding: the agent ignores off-topic content entirely — always returns the engine
  briefing, never weather/haiku; the Q3 override attempt changed nothing. Schema + evidence
  grounding leave no room for anything else.
- Fan chart: y-axis title was clipped (l=10) and "Cycle" title sat on the tick labels (b=10).
  Margins now l=64, b=48.
- risk_coverage: dropped end-point annotations (FD001/FD003 finals <0.1 apart, FD004 label shared
  the title row); final RMSE now in the legend labels ("FD001 (14.0)"), xlim back to 1.02, dots
  slightly smaller with white edges.
- coverage_by_rul: legend was drawn over the bars; moved below the x-axis (bbox_to_anchor,
  bbox_inches="tight" on save).
- Model radio reverted (user: not necessary): back to the original "Multitask GP across subsets"
  checkbox exactly as in Step 13. Correlation table ("What the multitask GP learned") removed
  from the Engine tab right column (user: not needed); README Engine-row updated to match.
  `predictor.task_correlation()` and the `metrics.json` entry stay (eval report, not app UI).

## 2026-11-02 — Submission folder `submission_meta_1`
- Clean-room rewrite of the whole repo for submission: identical logic and values, fresh comments.
  Excluded: `reference/`, experiment clones, `greeting.json`, data, artifacts, cache, logs,
  reports (empty start — user retrains), `.env`, `*_1.py` dupes (verified identical, except
  `policy_sweep_1.py`, which is the full trade-off version and ships as `policy_sweep.py`).
- Fixes folded in: `check_5.py` device `gpu` -> `cpu`; `run_all.sh` matches README Option B.
- Results (user runs): recalibrate + evaluate reproduced the canonical numbers
  (RMSE 13.98/14.24/14.02/15.49, coverage 0.88/0.92/0.93/0.83, replay 0/106 missed, 35
  unnecessary, 264 reviews, lead 25). FD003 validation stays TOO NARROW (0.843; 60-90 bin 0.724,
  below 0.276) - disclosed in README Limits; FD004 val ok (0.897) vs test 0.83 = shift.
  Decision: no retrain (recorded 2026-11-02).
- `rul_stage_report_1.py` (user's enhanced copy) adds a heads-up analysis
  (INSPECT == mu <= horizon agreement; review-flag-before-INSPECT lead). Merged it into
  `rul_stage_report.py` so the README command reproduces it; README reviews 265 -> 264.
- Full retrain experiment (2026-11-02, from-start, same config): test RMSE 13.87/14.74/14.62/15.91
  (champion 13.98/14.24/14.02/15.49), coverage 0.88/0.91/0.94/0.85 (champion FD004 0.83),
  replay 0/106 missed but 41 unnecessary + lead 24 (champion 35 + 25), NASA worse on 3 subsets,
  FD003 single-task (14.07) beats multitask (14.62) - breaks the wins-all-four claim.
  FD003 val still TOO NARROW (0.827, below 0.067). Verdict: restore final_1 champion.
- GitHub audit (2026-11-02): champion restore verified; README cleaned for publication
  (upload bullet confirmed merged, results-provenance note, checks section completed,
  layout updated, reviews 264). Placeholders left for the owner: LIVE_APP_URL, VIDEO_URL,
  RELEASE_URL. .gitignore now covers CMAPSSData/ and reports/*/ backup subfolders.
  reports/ backup subfolders (final_1, retrain, Rejected, New folder) to be removed by the
  owner before pushing - they hold competing metrics.json files.
- Decision (user, 2026-11-02): operational-safety story wins over point-accuracy (NASA) gains.
  Restore final_1 models+calibration+reports, regenerate precompute, verify with check_8 +
  results_table (expect champion numbers, reviews 264). Restore verified by user
  (results_table: champion numbers, generated 2026-11-02T11:22:16; stage-report JSON copied
  back from final_1; policy_tradeoff.png regenerated from the champion - user confirmed).
- Decision (2026-11-02, user): operational-safety over point accuracy. Champion wins decision
  cost (35 vs 41 unnecessary, lead 25 vs 24), RMSE/NASA and the multitask-wins-all-four claim;
  ties missed 0/106 and robustness. NASA score is point-only and credits no band/abstention, so
  it stays a secondary honesty metric. Restore final_1 models + calibration; do NOT recalibrate
  or re-evaluate after restore (would re-roll the numbers).
