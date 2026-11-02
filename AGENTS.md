# AGENTS.md: Engine Health Agent

Instructions for Muse Code in this repository.

**At the start of every session:**
1. Read this file.
2. Read `.agents/memory/progress.md` to see what's done.
3. Say which build step is next and what its exit check is.
4. Continue unless the user asks for something else.

---

## 1. The project

Engine Health Agent predicts the remaining useful life (RUL) of turbofan engines on NASA C-MAPSS,
with an uncertainty band calibrated to really cover 90%. C-MAPSS is simulated sensor data in text
files; there are no images.

Plain code turns the band into one of three calls:
- **INSPECT**;
- **NO_INSPECTION**;
- **ABSTAIN**, when the uncertainty is too wide to make a safe call, a sensor looks broken, or the
  engine is unlike the training data.

**Muse Spark agents** explain each call: which sensors drive it, what to do next, and why the
system abstains. A safety reviewer checks every briefing, and a planner explains the hangar
schedule.

- **Event:** Meta Global AI Developer Hackathon 2026, Indie track, solo builder.
- **Judging:** creativity, technical ambition, and whether it works.
- **Requirement:** built with Meta's frontier models, with Muse integrated.
- **Deadline:** Nov 4, 9:00 PM ET = Thu Nov 5, 07:30 IST. Target: submitted by Wed Nov 4,
  22:00 IST.

---

## 2. Scope

**In scope:**
- data prep;
- multitask deep kernel learning model;
- conformal calibration;
- decision engine, sensor checks and fault injection;
- Integrated Gradients explanations;
- service layer and deterministic planner;
- evaluation, including the maintenance-policy replay;
- Muse Spark agents: analyst, report writer, number guard, safety reviewer, planner;
- Streamlit app with 4 tabs (Fleet, Engine, Ask the agent, Evaluation);
- README and demo.

**Out of scope (decided Oct 27):** anything with images. That means no SAM 3.1, no vision calls,
no inspection photos and no Inspection tab. Concretely:
- The only actions are INSPECT, NO_INSPECTION and ABSTAIN. Drop REMOVE_FROM_SERVICE, MONITOR and
  RETURN_TO_SERVICE everywhere: the schema enum, labels, templates, chart colours.
- The only clearing action the reviewer may escalate is NO_INSPECTION.
- There is no `get_inspection_finding` tool and no `inspections` argument.
- When writing the prompts from the spec (Appendix A), remove analyst step 5, report-writer rule 7,
  and the reviewer's inspection bullet and its RETURN_TO_SERVICE example.
- Don't add image features even where the spec mentions them.

---

## 3. Sources of truth

| File | Role |
|---|---|
| `AGENTS.md` (this file) | how to work, scope, rules. **Wins over everything else.** |
| `docs/SPEC.md` | detailed spec: formulas, thresholds, shapes, API formats, prompts (Appendix A), schemas (Appendix B), config (Appendix C). **Ignore** section 13, prompt A.5, schema B.4, the `inspection:` config section, `features.inspection_loop`, `paths.inspection_images_dir`, and every mention of photos, SAM or inspection results. |
| `configs/default.yaml` | every tunable. Never hard-code a number that belongs in the config. |
| `reference/` (if present) | an earlier implementation that was **never executed**. Consult it when useful, treat it as untested, and prove any code you adopt with the step's exit check. |
| `.agents/memory/progress.md` | the build log you keep (section 6). |

Spec section map:

| Spec section | Topic |
|---|---|
| 5 | data |
| 6 | model |
| 7 | calibration |
| 8 | decision engine |
| 9 | sensor checks and faults |
| 10 | explanations |
| 11 | service and planner |
| 12 | agents |
| 14 | evaluation |
| 15 | app (without the Inspection tab) |
| 16 | config and run order |
| 18 | demo (without the inspection step) |

---

## 4. Rules that never bend

1. **The language model never makes the maintenance call.** `ehagent/decision.py` (plain code)
   decides; agents gather evidence and explain.
2. **Agents can only escalate.** The safety reviewer may turn NO_INSPECTION into ABSTAIN. Nothing
   may make INSPECT or ABSTAIN less cautious.
3. **Every number an agent writes comes from a tool output.** The number guard enforces this:
   one rewrite with the violations listed, then the deterministic template.
4. **The true RUL of test-file ("fleet") engines never reaches an agent or a tool.** Only the
   evaluation and the app's answer-key switch use it.
5. **The app runs with no API key.** Deterministic templates and the response cache
   (`EHA_OFFLINE=1`) cover every agent call.
6. **Split by engine ID, never by window.** Fit normalisers and sensor-check statistics on the
   train split only. Tune on validation. Use the test files only for the final evaluation.
7. **Thresholds and decision rules stay as specified.** Ask the user before changing anything in
   the `decision`, `calibration` or `sensor_checks` config sections.
8. **Never write a result you didn't produce.** Every metric in the README, the app or the demo
   script comes from `reports/metrics.json` produced by a real run.
9. **Label everything honestly:** simulated benchmark, decision support, not an airworthiness
   determination.

---

## 5. How to work

- **One build step at a time** (section 7). Before coding, state the step, its spec section and
  its exit check. After coding, run the exit check and show the output.
- **Smoke first, then full.** Run every training or evaluation command on a tiny setup first (one
  subset, 1-2 epochs, a few hundred windows) to catch shape and wiring errors. Then do the full
  run.
- **Long runs:** if a command will take more than a few minutes, run it in the background with
  output to `logs/<step>.log` and check the log, or give the user the exact command to run.
- **If an exit check fails twice,** stop, report what you saw, and propose the fallback listed for
  that step. Don't keep retrying.
- **Keep changes small and local.** Don't refactor modules outside the current step.
- **Data** lives in `data/raw/CMAPSSData/` (`train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`;
  26 space-separated columns). The user downloads it. If it's missing, stop and ask.
- **Ask the user before:**
  - deleting files;
  - changing thresholds or decision rules;
  - adding a dependency not listed in section 9;
  - running anything that calls the Meta Model API in a loop (for example agents over the whole
    fleet);
  - touching `.env`.
- **API use during development:** the smoke test and single-engine calls only, with the cache on.
  Automated checks never call the API; they run with `EHA_OFFLINE=1` and the templates.

---

## 6. Progress log

Keep `.agents/memory/progress.md` current, with a one-line entry for it in
`.agents/memory/MEMORY.md`. After each step, record:
- the step number and name, the date, and the status: done, partial or blocked;
- the exit-check output in one or two lines, with numbers copied exactly;
- decisions made and any deviation from the spec, with the reason;
- open issues and the next step.

---

## 7. Build order

| # | Step | Spec | Exit check | Fallback |
|---|---|---|---|---|
| 1 | Repo skeleton, config loader, `.env.example`, `requirements.txt` | 16, App. C | loading the config prints `data.window` = 30 | - |
| 2 | API smoke test: plain chat, one tool round trip, structured output (no image steps) | 12.8 | all three print sensible output with the user's key | fix the request format from the error text, then ask the user |
| 3 | Data prep: load, operating conditions, per-condition normalisation, engine-ID splits, windows, sensor-check stats, healthy baseline | 5, 9 | per subset: engine counts per split; no engine in two splits; windows of shape (N, 30, 14); 6 conditions found in FD002 and FD004 | - |
| 4 | CNN baseline with per-subset heads | 6.3, 6.4 | validation RMSE printed per epoch; final value logged | lower the learning rate; check the normalisation |
| 5 | DKL, single task, FD001 first | 6.1, 6.4 | validation RMSE close to the CNN baseline; sigma > 0, and larger on FD004 windows than on FD001 windows | check spectral norm; lower the GP learning rate |
| 6 | DKL multitask (LMC over the 4 subsets) | 6.2, 6.4 | validation RMSE per subset no worse than the CNN baseline by more than about 0.5 cycles; the task-correlation matrix prints | `multitask_mode: embedding`, then the deep ensemble (6.5) |
| 7 | Calibration (Mondrian split-conformal) | 7 | validation coverage between about 0.87 and 0.93 for every subset | check the score definition and the quantile level |
| 8 | Decision engine + sensor checks + fault injection | 8, 9 | the fleet shows all 3 actions; a stuck Ps30 (s11) makes an engine ABSTAIN with reason `sensor_integrity` | check that thresholds are read from the config |
| 9 | Explanations (Integrated Gradients + deletion test) | 10 | contributions sum to within about 10% of RUL minus the healthy reference RUL; median faithfulness ratio > 1 on 30 engines | check the baseline window and the step count |
| 10 | Service layer + planner + precompute | 11 | `fleet()` returns ranked rows; no engine appears twice in the plan; precomputed results load, and are ignored after retraining | - |
| 11 | Evaluation + policy replay + charts | 14 | `reports/metrics.json` and the charts exist; the replay table prints | `--skip-replay`, fix it later |
| 12 | Agents: client, tools, schemas, guard, templates, orchestrator (`ask` and `plan`) | 12, App. A (A.1-A.4), App. B (B.1-B.3) | with the API: briefings for one INSPECT, one ABSTAIN and one NO_INSPECTION engine pass the guard. Offline: the same three come from templates without errors | tighten the prompts using the guard's violation messages |
| 13 | App: Fleet, Engine, Ask the agent, Evaluation | 15 | starts with and without an API key; all 4 tabs render; injecting a fault changes the decision | hide the failing tab behind its switch |
| 14 | README, results table, demo prep | 18 | the README steps work from a fresh clone; the README tables come from `metrics.json` | - |


- If the build is behind, check `progress.md` and tell the user which steps are at risk.

---

## 8. Details that are easy to get wrong

**Data**
- RUL target capped at 125 and scaled to RUL/125 for training; predictions go back to cycles.
- 14 sensors: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.
- Operating conditions: k-means with k = 6 (FD002, FD004) or 1 (FD001, FD003), on standardised
  op settings.
- 30-cycle windows, edge-padded with the first row; training windows start at index 14.
- Splits by engine: train/val/calib/sim = 0.60/0.10/0.15/0.15.

**Model (GPyTorch)**
- Feature extractor: residual 1D-CNN with spectral norm (output x 3.0), pooling = concat(mean
  over time, last step), 16-dim embedding.
- LMC inducing points have shape (Q, M, D) = (3, 512, 16), initialised from k-means on 20,000
  pretrained embeddings plus small per-latent jitter.
- Use batch shape [Q] everywhere:
  - `CholeskyVariationalDistribution(M, batch_shape=[Q])`
  - `ConstantMean(batch_shape=[Q])`
  - `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=16, batch_shape=[Q]), batch_shape=[Q])`
- Wrap: `LMCVariationalStrategy(VariationalStrategy(self, inducing, var_dist,
  learn_inducing_locations=True), num_tasks=4, num_latents=3, latent_dim=-1)`.
- Call `gp(z, task_indices=subset_idx)`. It returns a MultivariateNormal for each window's own
  subset.
- Loss: `-VariationalELBO(likelihood, gp, num_data=N_train)(output, y)`.
- Two optimiser groups: extractor lr 1e-4, GP + likelihood lr 1e-2. Wrap forward passes in
  `gpytorch.settings.cholesky_jitter(1e-4)`.
- Outputs in cycles: `sigma_epi = sqrt(var_f) * 125`, `sigma_alea = sqrt(noise) * 125`,
  `sigma = sqrt(var_f + noise) * 125`.
- Keep a differentiable mean path for Integrated Gradients (no `no_grad` there).

**Calibration**
- Score = |y - mu| / sigma on calibration engines cut at 12 random points each.
- Quantile level `min(1, ceil((n+1)(1-alpha))/n)`, `method="higher"`. Not a plain 90th
  percentile.
- Groups: per subset x predicted-RUL bin, edges [0, 30, 60, 90, 1000]. Fewer than 30 scores in a
  bin: use the subset-level q.
- Band: `lo = max(mu - q*sigma, 0)`, `hi = mu + q*sigma`. Fit for alpha 0.10 and 0.02.
- Also store per-subset quantiles of `sigma_epi` for the unfamiliar-engine gate.

**Decision engine** (in this order)
1. Any sensor flag: ABSTAIN `sensor_integrity`.
2. Epistemic percentile >= 99: ABSTAIN `unfamiliar_engine`.
3. Otherwise compute:
   - `sigma_cal = half_width_90 / 1.645`;
   - `p = P(RUL <= H)` under N(mu, sigma_cal^2) truncated at 0, with H = 30.
4. Then:
   - p >= 0.10: INSPECT;
   - p <= 0.02: NO_INSPECTION;
   - otherwise: ABSTAIN `ambiguous_band`.

Section 8 of the spec has the urgency, recheck intervals and the resolution wording.

**Meta Model API (for the app's agents)**
- **Connection:** OpenAI-compatible SDK, `base_url = https://api.meta.ai/v1`. The key is in
  `MODEL_API_KEY` (fall back to `META_API_KEY`). The model is `muse-spark-1.3`, overridable with
  `MUSE_MODEL`.
- **Never send `tool_choice`.** Only "auto" is supported.
- **Structured output:** `response_format={"type": "json_schema", "json_schema": {"name": ...,
  "schema": ..., "strict": False}}`. Parse tolerantly (code fences, leading prose).
- **Reasoning:** `reasoning_effort` (low) goes in `extra_body`. If a 400 error mentions it, or
  the temperature, drop that parameter and retry.
- **Tool round trip:** append the assistant message with its `tool_calls`, then one
  `{"role": "tool", "tool_call_id": ..., "content": json}` per call.
- **Retries:** retry 429, 5xx and timeouts with exponential backoff, honouring `Retry-After`.
  Don't retry other 4xx errors.
- **Cache:** cache every response on disk, keyed by a hash of the full request.

**Number guard**
- The pool of allowed numbers: every number in the evidence (including numbers inside strings),
  their absolute values, 0-1 values x 100, and the config thresholds (125, 10, 2, 90, 99, 30).
- A number matches within max(0.6, 1%). Integers 0-10 are always allowed in free text.
- `recommendation` must equal the authoritative action. ABSTAIN needs an `abstain_reason`.

---

## 9. Conventions

- Python 3.10+, `from __future__ import annotations`, type hints, small modules under `ehagent/`.
- Dependencies: numpy, pandas, scipy, scikit-learn, torch, gpytorch, pyyaml, openai, streamlit,
  plotly, matplotlib, python-dotenv. Ask before adding anything else.
- Every pipeline step is a CLI, `python -m ehagent.<module> ...`, that prints timestamped
  progress.
- Seeds and paths come from the config.
- Where things go: `artifacts/` (prepared data, models, calibration, precomputed results),
  `reports/`, `.cache/`, `logs/`. Don't commit those, `data/raw/` or `.env`.
- Exit checks live in `scripts/check_<step>.py`. They run in under two minutes on CPU and need no
  API key.
- Charts:
  - decision colours (green / yellow / red) always come with an icon and a label;
  - each subset keeps one fixed colour;
  - one y axis per chart;
  - text in ink colours, never in a series colour;
  - every chart has a table view.

---

## 10. Make the Muse integration visible

Judges need to see Muse Spark doing real work:

- **Ask tab:** label the briefing as written by Muse Spark. Show the safety reviewer's verdict and
  an expandable tool trace with every call and result. If the template was used, say so.
- **Fleet tab:** say whether the planner agent or the template wrote the plan.
- **README:** a "Where Muse Spark is used" section listing the four roles: evidence gathering
  with tools, the structured briefing, the safety review, the hangar plan.
- **Demo:** record from cached Muse responses (`EHA_OFFLINE=1` after one live run). Never record
  with the agents switched off, because the template looks the same and hides the integration.

---
## 11. Definition of done
- `.agents/memory/progress.md` shows every step as done, or explains why not.


## 12. Decisions (these override the spec and the values in section 8)

- **Training:** dropout 0.3, sn_coeff 0.9, pretrain_lr 5e-4, weight_decay 1e-2, dkl_lr_features 1e-5.
  This fixed overfitting and the epoch-1 blow-up (FD002 validation RMSE 16.9 -> 14.7).
- **Calibration:** separate lower and upper band edges (calibrate.py, format "asymmetric-v1"),
  lower_tail_share 0.25 (2.5% of misses below the band, 7.5% above), cuts_per_engine 30.
  decision.py and evaluate.py use z = norm.ppf(1 - conformal.lower_tail(alpha)).
- **Decision:** p_inspect 0.50 (inspect when predicted RUL reaches the 30-cycle horizon), p_clear 0.10.
  Chosen with scripts/policy_sweep.py: 0/106 missed, 35 unnecessary inspections. Don't revert to 0.10 / 0.02.
- **Replay:** ABSTAIN counts as an engineer review (simulation.abstain_action: review). Reviews are
  counted separately and never credited as a catch.
- **Default model:** multitask (features.multitask_model: true).
