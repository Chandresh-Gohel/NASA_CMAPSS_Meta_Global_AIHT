# Engine Health Agent: spec

Everything decided and built so far, written so you or Muse can rebuild it piece by piece.
State as of Wed Oct 28, 2026.

**Scope (decided Oct 27):** no images. C-MAPSS is sensor data only, so there is no SAM 3.1, no
vision and no inspection photos. The only actions are **INSPECT**, **NO_INSPECTION** and **ABSTAIN**.

**Status:** the design is complete. A reference implementation (about 4,600 lines of Python,
`engine-health-agent.zip`) was written by Claude on Oct 24 and has **never been executed**. Only
static checks were run on it (syntax, cross-module imports, config keys, lint). It predates the
scope change and still contains the image parts; ignore those (section 20).

## Contents

0. Using this document with Muse
1. Context and constraints
2. What we're building and why it should score
3. Research summary
4. Architecture
5. Data
6. Model
7. Calibration
8. Decision engine
9. Sensor checks and fault injection
10. Explanations
11. Service layer and planner
12. Agents
13. (removed: inspection loop, out of scope)
14. Evaluation
15. App
16. Configuration
17. Build plan
18. Demo script
19. Cost and credits
20. Reference implementation status
21. Open questions
- Appendix A: agent prompts
- Appendix B: structured-output schemas
- Appendix C: configuration file

---

## 0. Using this document with Muse

- Give Muse sections 1-4 for context, then the section for the module you're building. The
  appendices hold the exact prompts, output schemas and config.
- Build in the order of section 17. The step-by-step version (14 steps, each with an exit check
  and a fallback) is in AGENTS.md section 7. Don't move on until a step's check passes.
- Repeat these invariants in every prompt. They are what makes the project safe and what the
  judges will probe:
  1. The language model never makes the maintenance call. Plain code does (section 8).
  2. Agents can only escalate (make an outcome more cautious), never relax a decision.
  3. Every number an agent states must come from a tool output (number guard, section 12.5).
  4. The true remaining life of fleet (test-file) engines is never given to an agent.
  5. Every external dependency has a fallback: deterministic template, response cache, UI switch.
- Where the reference implementation already has a module you like, paste that file and ask Muse
  to review and adapt it rather than regenerate it.

Prompt template for Muse Code:

```
You are implementing one part of the Engine Health Agent. Read the spec below.
Implement <module> exactly as specified in section <N>: same function names, inputs and outputs.
Do not change thresholds or decision rules. Read every tunable from configs/default.yaml.
Then write a short check script that runs it on the real data and prints <exit check>.

<paste sections 4, 5 and N, plus Appendix C>
```

---

## 1. Context and constraints

| Item | Detail |
|---|---|
| Event | Meta Global AI Developer Hackathon 2026, **Indie track** (individuals and students). Solo, from India. |
| Build window | Mon Oct 26 to Wed Nov 4, 2026 |
| Deadline | Nov 4, 9:00 PM ET = 02:00 UTC Thu Nov 5 = **07:30 IST Thu Nov 5**. Some Indian listings say 6:30 AM IST; they miss the US clock change on Nov 1. Plan to submit on the evening of Nov 4. |
| Judging | Nov 4-8, live finals Nov 9. Indie track is judged on "creativity, technical ambition, and whether it works" (no weights published). |
| Support | Live workshops and Discord office hours, Oct 27 to Nov 4. |
| Models | Muse Spark (Chat Completions) through the Meta Model API. No image models are used. The event pages also mention Muse Wonder (not evaluated yet). |
| Credits | $150 in Meta Model API credits per selected participant. A dev.meta.ai account exists with the application email; credits were not visible yet on Oct 26. How they are applied is not public. |
| Rules | The project must be built with Meta's frontier models and have Muse integrated. Remaining open questions are in section 21. |
| Time goal | Build starts Wed Oct 28 (Day 1). Feature-complete by Sat Oct 31, with Nov 3-4 kept as buffer. Nothing else is cut. |

---

## 2. What we're building and why it should score

**Pitch:** most remaining-useful-life (RUL) models give one number. This one gives a calibrated
range, turns it into a safe maintenance call, says which sensors drive it, and knows when not to
make a call.

The pieces:

- **Prognostics:** multitask deep kernel learning on NASA C-MAPSS. Output: mean RUL plus epistemic
  and aleatoric uncertainty.
- **Calibration:** Mondrian split-conformal, so the 90% band really covers 90%, per subset and per
  RUL range.
- **Decision engine (plain code):** INSPECT / NO_INSPECTION / ABSTAIN, with a sensor-integrity gate
  and an unfamiliar-engine gate.
- **Explanations:** Integrated Gradients per sensor, in cycles, with a deletion faithfulness test.
- **Muse Spark agents:** analyst (tool calls), then report writer (JSON), then number guard, then
  safety reviewer (escalate-only). A planner agent explains a deterministic hangar schedule.
- **Evaluation:** accuracy, calibration, abstention, out-of-distribution engines, sensor faults,
  explanation faithfulness, single- vs multitask, and a replay of maintenance policies.

How it maps to the judging criteria:

| Criterion | What answers it |
|---|---|
| Creativity | Abstention only when uncertainty changes the decision; a safety reviewer that can only escalate; impact measured as missed failures and unnecessary inspections, not just RMSE. |
| Technical ambition | Multitask DKL with an LMC head across 4 subsets; spectral-normalised features for distance-aware uncertainty; per-bin conformal calibration; IG with a deletion test; OOD and fault-injection tests. |
| Whether it works | Deterministic fallback for every agent; disk cache for an offline demo; precomputed results; sidebar switches to hide anything that breaks; API smoke test before building on it. |

---

## 3. Research summary

Best RMSE reported on the four subsets (from the papers, found on Oct 24):

| Method | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| RGPD (arXiv 2507.09766) | 10.77 | 11.58 | 10.78 | 12.38 |
| STAR (Sensors 2024) | 10.61 | 13.47 | 10.71 | 15.87 |

Also near the top: PSTFormer and PVA-FFG.

Caveats:
- Protocols differ (RUL cap of 125 or 130, window length, normalisation, which cycle of each test
  engine is scored), so these numbers are not strictly comparable.
- These are point predictions. The papers report no calibrated uncertainty, no abstention and no
  decision quality.

What this means for us: don't chase state-of-the-art RMSE. Get a competitive RMSE on the standard
protocol (cap 125, last cycle of each test engine) and win on calibrated bands, safe decisions and
explanations, which none of these offer. Report RMSE against capped truth (comparable to papers)
and against uncapped truth (honest).

---

## 4. Architecture

```
sensor history (30-cycle window, normalised per operating condition)
      |
      v
residual 1D-CNN (spectral norm) --> multitask sparse GP (LMC over FD001-FD004)   [deep kernel learning]
      |                                   |
      |                     mean RUL + epistemic sigma + aleatoric sigma
      v                                   v
Integrated Gradients          Mondrian split-conformal calibration (per subset x RUL range)
(sensor drivers, in cycles)               |
      |                                   v
      |           decision engine (plain code):
      |           sensor-integrity gate -> unfamiliar-engine gate -> P(fail within H)
      |           -> INSPECT / NO_INSPECTION / ABSTAIN
      |                                   |
      +-----------> Muse Spark agents <---+
                    analyst (tool calls) -> report writer (JSON schema) -> number guard -> safety reviewer
                    planner (tool calls) -> work orders, reconciled with the deterministic schedule
```

Code layout:

```
app/streamlit_app.py        the demo app (4 tabs: Fleet, Engine, Ask the agent, Evaluation)
configs/default.yaml        every tunable (Appendix C)
ehagent/
  config.py                 YAML loading with attribute access, paths, seeding, device
  constants.py              column names, sensor names, action vocabularies
  data.py                   loading, per-condition normalisation, engine splits, windows
  models/cnn.py             feature extractor, point head, Gaussian head
  models/dkl.py             GP heads (single, LMC multitask), DKL wrapper, inducing-point init
  models/ensemble.py        deep ensemble (backup)
  train.py                  CLI: prepare | baseline | dkl | ensemble | all
  predictor.py              one predict() for every model family
  calibrate.py              Mondrian split-conformal + epistemic percentiles
  sensor_checks.py          stuck / jump / out-of-range checks
  decision.py               the decision engine
  faults.py                 sensor fault injection
  attribution.py            Integrated Gradients + deletion test
  service.py                EngineService: the one API used by the app and the agent tools
  planner.py                deterministic hangar and review schedule
  simulate.py               maintenance-policy replay
  evaluate.py               full evaluation, metrics.json, charts
  precompute.py             cached fleet results for an instant app
  viz.py                    validated chart palette and styling
  agent/client.py           Meta Model API client: retries, disk cache, offline mode
  agent/tools.py            tool definitions + executor with a trace
  agent/schemas.py          JSON schemas for structured output (Appendix B)
  agent/guard.py            number guard
  agent/templates.py        deterministic briefings and plans
  agent/orchestrator.py     ask() and plan()
prompts/                    4 system prompts (Appendix A)
scripts/                    smoke_test_apis.py, run_all.sh, results_table.py
docs/                       build_plan.md, demo_script.md
```

---

## 5. Data

**Dataset.** Four subsets. Each row: unit, cycle, 3 operational settings, 21 sensors (26
space-separated columns). Training files run each engine to failure. Test files stop early; the
true RUL at the last cycle is in `RUL_FD00x.txt`. 100 to 260 engines per file.

| Subset | Operating conditions | Fault modes |
|---|---|---|
| FD001 | 1 | 1 (HPC degradation) |
| FD002 | 6 | 1 (HPC) |
| FD003 | 1 | 2 (HPC + fan) |
| FD004 | 6 | 2 (HPC + fan) |

**Sensors used (14).** The other 7 are near-constant.

| id | Symbol | Meaning |
|---|---|---|
| s2 | T24 | Total temperature at LPC outlet |
| s3 | T30 | Total temperature at HPC outlet |
| s4 | T50 | Total temperature at LPT outlet |
| s7 | P30 | Total pressure at HPC outlet |
| s8 | Nf | Physical fan speed |
| s9 | Nc | Physical core speed |
| s11 | Ps30 | Static pressure at HPC outlet |
| s12 | phi | Ratio of fuel flow to Ps30 |
| s13 | NRf | Corrected fan speed |
| s14 | NRc | Corrected core speed |
| s15 | BPR | Bypass ratio |
| s17 | htBleed | Bleed enthalpy |
| s20 | W31 | HPT coolant bleed |
| s21 | W32 | LPT coolant bleed |

**Processing.**
- **Operating conditions:** k-means on the 3 standardised op settings (k = 6 for FD002/FD004, 1
  otherwise), fit on the training file. Sort the cluster ids by centre (lexsort) so they are stable.
- **Normalisation:** z-score each sensor per operating condition, fit on the train split only. A
  std below 1e-6 becomes 1.
- **Target:** RUL in cycles. Training-file engines: last cycle minus cycle. Test-file engines:
  (last cycle minus cycle) plus the RUL file value. The model's target is capped at 125
  (piecewise-linear) and scaled to RUL/125.
- **Windows:** 30 cycles ending at each cycle. Histories shorter than 30 are edge-padded with the
  first row. Training windows start at cycle index 14 (`min_history` 15).
- **Splits:** by engine ID inside each training file: train 60%, val 10%, calib 15%, sim 15%
  (at least 1 engine each), seeded per subset. **No engine appears in two splits.** The test file
  is the "current fleet".
- **Engine IDs:** `FD001-TR-012` (training file), `FD001-TE-012` (test file).
- **Healthy baseline:** the average normalised window over training windows with RUL >= 125. This
  is the reference point for Integrated Gradients.
- Persist each prepared subset (engines, splits, clusterer, normaliser, sensor-check stats,
  baseline) to `artifacts/data/FD00x.pkl`.

---

## 6. Model

### 6.1 Why deep kernel learning

- A GP directly on raw windows (30 x 14 = 420 numbers) works poorly. A CNN learns features, and a
  GP on the 16-dimensional embedding gives principled uncertainty.
- Spectral normalisation (Lipschitz-bounded layers, coefficient 3.0) keeps the embedding
  distance-aware: unfamiliar engines land far from the inducing points, so epistemic variance grows
  (the idea behind DUE and SNGP). Without it, features can collapse and the GP stays confident on
  inputs it has never seen.
- A sparse variational GP (512 inducing points) scales to about 100k windows with minibatches.

### 6.2 Why multitask, and why LMC

- The four subsets share degradation physics (HPC degradation everywhere, fan degradation in
  FD003/FD004) but differ in operating conditions and fault mix. Sharing latent functions lets the
  subsets borrow strength from each other.
- Each window belongs to exactly one subset. A Kronecker-structured `MultitaskKernel` assumes every
  input has outputs for all tasks, which doesn't fit, and exact GPs cost O(N^3).
- So use `LMCVariationalStrategy` with `task_indices` (Hadamard style): Q = 3 shared latent GPs,
  and subset t's function is the sum over q of a[q,t] * f_q.
- The learned LMC coefficients give a subset-to-subset correlation matrix (A^T A, normalised),
  which the app shows.
- Backup "embedding" mode: a shared CNN plus a learned subset embedding (dim 4) feeding one GP.

### 6.3 Feature extractor (exact)

```
input (B, 30, 14) -> transpose -> (B, 14, 30)
Conv1d(14 -> 64, kernel 5, padding 2)            [spectral norm, output x 3.0] -> GELU
3 x ResBlock(64):  conv k5 [SN] -> GELU -> Dropout(0.1) -> conv k5 [SN] -> add input -> GELU
pool: concat(mean over time, last time step) -> 128
(embedding mode only: concat a 4-dim subset embedding -> 132)
Linear(128 -> 64) [SN] -> GELU -> Linear(64 -> 16) [SN]  -> 16-dim embedding
```

SN = `torch.nn.utils.parametrizations.spectral_norm`, with the layer output multiplied by 3.0.

### 6.4 Training (exact)

**CNN pretraining (also the point baseline).**
- Head: Linear(16 -> number of subsets); use the output for the window's own subset.
- Loss: MSE on RUL/125.
- AdamW, lr 1e-3, weight decay 1e-5, cosine schedule over 40 epochs, batch 256, grad-norm clip 5.
- Early stop on validation RMSE (in cycles), patience 8; keep the best weights.

**DKL.**
1. Embed 20,000 random training windows with the pretrained extractor. MiniBatchKMeans gives
   512 centres = the inducing points. For LMC, replicate them per latent (Q = 3) with jitter of
   0.01 x per-dimension std, shape (Q, M, D).
2. GP: ConstantMean; ScaleKernel(Matern nu = 2.5, ARD over 16 dims) with batch shape [Q] for LMC;
   CholeskyVariationalDistribution; learn inducing locations; GaussianLikelihood.
3. Init: lengthscales = per-dimension std of the embeddings (floor 1e-3); constant mean = mean of
   the scaled targets; noise = 0.01.
4. Loss: negative VariationalELBO(likelihood, gp, num_data = number of training windows).
5. Optimiser: Adam with two groups: extractor lr 1e-4 (weight decay 1e-5), GP + likelihood lr 1e-2.
   Cosine schedule over 30 epochs, batch 512, `cholesky_jitter(1e-4)`, grad clip 5. Early stop on
   validation RMSE (patience 8, or NLL via config).
6. Forward: z = extractor(x). LMC: `gp(z, task_indices=subset_index)` returns a MultivariateNormal
   for each window's own subset. Single-task: `gp(z)`.

Train one multitask model (all 4 subsets, the primary) and one single-task model per subset (the
ablation). Save checkpoints as `{family, meta, state_dict}` so the module can be rebuilt with the
right shapes before loading.

### 6.5 Backup: deep ensemble

- 5 members, each a CNN extractor plus a head giving (mean, softplus(var) + 1e-4) per subset.
- Train each with a different seed: 10 warm-up epochs of MSE on the mean, then beta-NLL with
  beta = 0.5:

  ```
  loss = mean( 0.5 * (log var + (y - mu)^2 / var) * stopgrad(var)^0.5 )
  ```

- Up to 40 epochs, AdamW lr 1e-3, early stop on validation NLL.
- Prediction: mean of member means. Epistemic = variance of member means; aleatoric = mean of
  member variances.

### 6.6 Prediction outputs (in cycles)

```
mu         = max(mean * 125, 0)
sigma_epi  = sqrt(var_f) * 125          # "the model hasn't seen engines like this"
sigma_alea = sqrt(noise) * 125          # irreducible noise
sigma      = sqrt(var_f + noise) * 125
```

Predict in batches of 1024 windows under `cholesky_jitter(1e-4)`. Integrated Gradients needs a
differentiable mean, so keep a separate `mean_torch()` path without `no_grad`.

---

## 7. Calibration (Mondrian split-conformal)

1. On the calibration engines, cut each engine at 12 random points (at or after `min_history`),
   mimicking how the test set was truncated. Use the window ending at each cut;
   y = min(true RUL, 125).
2. Score = |y - mu| / sigma.
3. For each alpha in {0.10, 0.02}, compute q(alpha) = the empirical quantile of the scores at level
   `min(1, ceil((n + 1) * (1 - alpha)) / n)`, method "higher". **Use this finite-sample level, not
   a plain 90th percentile.**
4. Mondrian groups: separately per subset and per predicted-RUL bin, bin edges [0, 30, 60, 90, 1000].
   A bin with fewer than 30 scores falls back to the subset-level q; an unknown subset falls back
   to the global q.
5. Band: `lo = max(mu - q * sigma, 0)`, `hi = mu + q * sigma`, `half = q * sigma`.
6. Epistemic reference: store 101 quantiles (0 to 100%) of sigma_epi per subset on the calibration
   cuts. A new engine's epistemic percentile is interpolated from them.
7. Check: coverage on the validation engines, cut with a different seed, per subset. Target 0.90.
8. Save as JSON (alphas, bins, min group size, tables, epistemic quantiles, score counts).

---

## 8. Decision engine (plain code, authoritative)

```
lo90, hi90, half90 = band(subset, mu, sigma, alpha=0.10)
lo98               = band(subset, mu, sigma, alpha=0.02).lo        # conservative lower bound
sigma_cal = max(half90 / 1.645, 1e-3)        # Gaussian whose 90% band equals the conformal band
p        = P(RUL <= H)  under N(mu, sigma_cal^2) truncated to RUL >= 0
         = (Phi((H - mu)/s) - Phi(-mu/s)) / (1 - Phi(-mu/s))
p_urgent = same with H = 10
epi_pct  = epistemic percentile vs calibration engines of the same subset
healthy  = hi90 >= 125

if any sensor flag:            ABSTAIN  [sensor_integrity], urgency soon
                               "Verify <sensors>: possible <checks>. Prediction withheld; engineer review."
elif epi_pct >= 99:            ABSTAIN  [unfamiliar_engine], urgency soon
                               "Unlike the engines the model learned from; manual assessment."
elif p >= 0.10:                INSPECT  [failure_probability_high]
                               urgency immediate if p_urgent >= 0.10 or lo98 <= 10, else soon
                               "Schedule an inspection within floor(lo98) cycles."
elif p <= 0.02:                NO_INSPECTION [failure_probability_low (+ healthy_region)], routine
                               recheck = clip(lo90 - H, 1, 50) -> "Re-evaluate in <recheck> cycles."
else:                          ABSTAIN  [ambiguous_band], urgency soon
                               recheck = clip(lo98 / 2, 1, 50)
                               "The band straddles the H-cycle horizon; neither call is safe."
```

- H = 30 cycles by default (the cycles until the next maintenance opportunity). The app and the
  `what_if_horizon` tool can change it.
- A wide band far from the horizon is still a confident NO_INSPECTION. Uncertainty only blocks a
  decision when it changes the decision.
- If `healthy` is true, say "healthy region: remaining life above the cap can't be estimated".
  Never present 125 as a prediction.
- The decision record includes: action, reasons, rul_mean, rul_lo, rul_hi, rul_lo_conservative,
  interval_coverage (0.9), healthy_region, p_fail_within_horizon, horizon_cycles, sigma_cycles,
  sigma_epistemic, sigma_aleatoric, epistemic_percentile, sensor_flags, recheck_in_cycles, urgency,
  resolution, interval_width, mean_minus_horizon.

---

## 9. Sensor checks and fault injection

**Checks.** Run over the last 15 cycles. Baselines are learned from the train-split engines of the
same subset.

| Check | Fires when |
|---|---|
| out_of_range | peak abs(z) > max(1.5 x the largest abs(z) seen in training, 4.0) |
| jump | largest abs(cycle-to-cycle change in z) > 2.0 x the 99.9th percentile of that change in training |
| stuck | the raw value hasn't changed in the last 10 cycles, **and** fewer than 1% of 10-cycle training windows were flat for that sensor (so low-resolution sensors aren't flagged) |

Each flag: `{sensor, name, check, detail (plain words with the numbers), severity: "high"}`.

**Fault injection** (for the demo and the robustness test). Applied to raw values, then
re-normalised. Magnitudes are in training std for the cycle's operating condition:

| Fault | Effect |
|---|---|
| stuck | hold the value from the start cycle onward |
| drift | add 3 std per 10 cycles from the start |
| spike | add 12 std at one cycle, placed within the last 3 cycles |
| noise | add N(0, 3 std) from the start |

Defaults: sensor s11 (Ps30), starting 20 cycles before the latest cycle. Fault spec:
`{"sensor", "kind", "magnitude", "start_offset", "seed"}`.

---

## 10. Explanations

- **Integrated Gradients** from the healthy baseline window to the engine's latest window. Use 32
  midpoint steps on the differentiable mean RUL in cycles. Attributions are in cycles and add up to
  roughly RUL(engine) minus RUL(healthy baseline).
- Per-sensor contribution = sum over time. Sort by absolute value. Each row: `{sensor, name,
  description, contribution_cycles (1 decimal), effect: "lowers/raises the RUL estimate"}`.
- **Deletion test:** reset the top 3 sensors to the baseline and measure the change in predicted
  RUL. Compare with 20 random 3-sensor resets. Report `{top_k, top_k_change_cycles,
  random_k_change_cycles, faithfulness_ratio}`. A ratio above 1 means the explanation is faithful.

---

## 11. Service layer and planner

**EngineService** is the one object the app and every agent tool call:

| Method | Returns |
|---|---|
| `analyze(engine_id, horizon=None, fault=None)` | the decision record at the latest cycle, plus history length and injected fault |
| `trajectory(engine_id, fault, horizon, include_truth)` | a table per cycle: mu, lo, hi, lo_c, sigma, sigma_epi, p_fail, action, reasons (+ rul_true only when asked) |
| `explain(engine_id, top_k, fault)` | IG contributions + faithfulness + healthy reference RUL |
| `check_sensors(engine_id, fault)` | status ok/suspect + flags |
| `what_if(engine_id, horizon, fault)` | decision fields at another horizon |
| `fleet(subsets, horizon, faults)` | decisions for all test-file engines, riskiest first: INSPECT before ABSTAIN before NO_INSPECTION, then immediate urgency, then sensor flags, then higher p |
| `plan(...)` | the deterministic schedule below |
| `answer_key(engine_id)` | true RUL at the latest cycle (demo and evaluation only, **never to agents**) |

- Precompute clean results (fleet, analyses, trajectories, explanations) and fingerprint them with
  the model and calibration file times. If either changed, ignore the stale results.
- Keep an LRU cache of computed results (key includes horizon and fault).

**Planner (deterministic).**
- Assumptions: 4 cycles per day; 2 hangar inspection slots and 4 engineering-review slots per day;
  7-day horizon.
- INSPECT engines are sorted by immediate first, then deadline day, then higher p. Deadline day =
  1 + floor(lo98 / cycles_per_day). Each goes on the earliest day with a free slot. A conflict is
  recorded if that day is after the deadline or no slot is free.
- ABSTAIN engines go into ENGINEERING_REVIEW slots: sensor-integrity cases first (deadline day 1),
  then ambiguous bands (deadline from lo98, as above).
- Priorities: P1 = immediate INSPECT or sensor-integrity ABSTAIN; P2 = other INSPECT; P3 = other
  reviews.
- Watch list: the 10 NO_INSPECTION engines with the soonest recheck.
- Output: days (inspections, reviews), conflicts, unplaced, watch list, summary counts.

---

## 12. Agents (Muse Spark)

### 12.1 Engine briefing flow: `ask(question, engine_id, faults, horizon)`

1. Call `get_engine_status` deterministically first; it is the anchor.
2. Authoritative action = the decision engine's action.
3. **Analyst loop** (system prompt A.1): up to 5 rounds of tool calls with the analyst tools. The
   analyst only gathers evidence.
4. Evidence = `{tool name: result, ..., what_if_horizon: [list], authoritative_action, thresholds:
   {p_inspect, p_clear, rul_cap, interval_coverage}}`.
5. **Report writer** (A.2) with structured output (schema B.1).
6. **Number guard** (12.5). On violations, rewrite once with the violation list appended. If it
   still fails, use the deterministic template.
7. **Safety reviewer** (A.3, schema B.2). `force_abstain` escalates NO_INSPECTION (the only
   clearing action) to ABSTAIN and rewrites the headline, abstain reason and next steps. `revise`
   adds up to 3 notes as caveats. The reviewer can never relax a decision.
8. Return: report, review, final_action, authoritative_action, used_fallback, violations, errors,
   tool trace, token usage.

Any API or parsing error falls back to the template. The app always gets a briefing.

### 12.2 Tools

All tools are deterministic Python over EngineService. Every call is recorded in a trace that the
app shows.

| Tool | Arguments | Returns |
|---|---|---|
| get_engine_status | engine_id | the full decision record (section 8) |
| explain_prediction | engine_id, top_k 1-14 (default 3) | IG contributions + faithfulness |
| check_sensors | engine_id | status, flags, lookback |
| what_if_horizon | engine_id, horizon_cycles 5-200 | p, action, reasons, resolution, band, recheck |
| fleet_triage (planner) | subset ALL/FD00x, top_n 1-50 | totals, counts per action, top engines |
| plan_inspections (planner) | days, slots_per_day, review_slots_per_day, subset | the deterministic schedule |

Analyst tools: the first four. Planner tools: fleet_triage, plan_inspections, get_engine_status.
Bad arguments or an unknown tool return `{"error": ...}`. Results are made JSON-safe (NaN becomes
null).

### 12.3 Briefing output (schema B.1)

`engine_id`, `recommendation` (INSPECT, NO_INSPECTION or ABSTAIN), `headline` (12 words max), `summary` (2-4
sentences), `key_numbers` [{label, value, unit, source}], `sensor_drivers` [{sensor, name,
contribution_cycles, plain_language}], `abstain_reason`, `next_steps`, `caveats`. If no caveat
mentions the simulated benchmark, append: "Decision support on a simulated benchmark (NASA C-MAPSS);
not an airworthiness determination. A qualified engineer makes the final call."

### 12.4 Safety reviewer (schema B.2)

The reviewer checks the action, the numbers and understated risk. It flags a briefing that presents
the engine as safe while any of these hold:
- p is above p_clear;
- the lower band edge is at or below the horizon;
- there are sensor flags;
- the epistemic percentile is 95 or higher.

It also checks that an abstention is explained, and the tone. Verdicts: `approve`, `revise`,
`force_abstain`.

### 12.5 Number guard

- The pool of allowed numbers is every number in the evidence (including numbers inside strings),
  their absolute values, values between 0 and 1 times 100 (percentages), and the config thresholds:
  cap 125, p_inspect 10%, p_clear 2%, coverage 90%, epistemic gate 99, default horizon 30.
- A number matches if it is within max(0.6, 1%) of a pool value. In the free text, integers 0-10
  are always allowed.
- Checked: `key_numbers` values, `sensor_drivers` contributions, and every number in the headline,
  summary, abstain reason and next steps.
- Also: `recommendation` must equal the authoritative action, and ABSTAIN needs a non-empty
  abstain_reason.

### 12.6 Planner agent: `plan(...)`

1. Call `plan_inspections` deterministically first.
2. Planner loop (A.4) with the planner tools.
3. The planner writes JSON (schema B.3).
4. Reconcile: keep the deterministic schedule (engines, days, actions, priorities). Take the
   planner's reason and summary text only if they add no new numbers. Ignore engines that aren't
   on the schedule, and append every schedule conflict to the warnings.

### 12.7 Templates (fallbacks)

Deterministic briefing and plan builders with the same structure as the agent outputs, so the app
renders both the same way.

### 12.8 Meta Model API: client and formats

Formats come from the dev.meta.ai docs (checked Oct 24). **Verify each one with the smoke test.**

- Use the OpenAI-compatible SDK. `base_url = https://api.meta.ai/v1`, key in `MODEL_API_KEY`.
  Model `muse-spark-1.3` (from the docs examples), overridable with `MUSE_MODEL`.
- **Tools:** `[{"type": "function", "function": {name, description, parameters}}]`. **Never send
  `tool_choice`**; only "auto" is supported. To round-trip, append the assistant message with its
  `tool_calls`, then one `{"role": "tool", "tool_call_id", "content": json}` per call.
- **Structured output:** `response_format = {"type": "json_schema", "json_schema": {"name",
  "schema", "strict": false}}`. Parse tolerantly (strip code fences, find the first JSON object).
- **Reasoning:** `reasoning_effort` goes in `extra_body` (minimal / low / medium / high / xhigh;
  we use low). Temperature 0.2. If a 400 error mentions either one, drop it and retry.
- **Retries:** 3 with exponential backoff (1, 2, 4 s) on rate-limit, connection, timeout and 5xx
  errors. No retry on other HTTP status errors.
- **Disk cache:** responses cached under `.cache/llm/`, keyed by SHA-256 of the full request.
  `EHA_OFFLINE=1` means never call the API: replay from the cache, otherwise the template.
- Count calls, cache hits and tokens for the app.

---

## 13. (Removed) Inspection loop

Out of scope since Oct 27: C-MAPSS has no images. The section number is kept so references stay
valid.

---

## 14. Evaluation

| Test | What |
|---|---|
| Test metrics, per subset | On the last window of each test engine: RMSE vs capped truth (primary, comparable to papers) and vs uncapped truth; NASA score; MAE; RMSE when true RUL <= 30; 90% coverage; mean band width; NLL and CRPS of the calibrated Gaussian; coverage by true-RUL range (0-30, 30-60, 60-90, 90-cap); risk-coverage curve (abstain on the most uncertain first) and its area; RMSE after abstaining on the widest 20%. |
| Out-of-distribution | Feed FD004 test engines to the FD001 model (normalised with FD001 statistics) and compare with real FD001 test engines: epistemic-gate rate, abstain rate, sensor-flag counts. |
| Sensor faults | On 30 FD001 test engines, per fault kind: abstain rate, flag rate, share of decisions changed, mean RUL shift. |
| Explanation faithfulness | On 30 engines: median faithfulness ratio, share above 1, most common top sensor. |
| Ablation | Single-task vs multitask on the same test engines: RMSE, coverage, width. |
| Policy replay | On held-out "sim" engines replayed cycle by cycle to failure. Policies: fixed interval (every 50 cycles), point estimate (inspect when mu <= H), and ours (ABSTAIN counted as an inspection by default). An inspection catches the failure if true RUL <= 30 at that moment (and >= 1); earlier inspections count as unnecessary; 10-cycle cooldown between events. Measures: missed failures, unnecessary inspections, median warning lead time. |

NASA score: d = predicted minus true; sum of exp(-d/13) - 1 for d < 0 and exp(d/10) - 1 otherwise.

Write `reports/metrics.json` and three charts:
- the policy comparison: 3 small panels (one measure each), baselines grey, ours blue;
- risk-coverage lines per subset;
- coverage-by-RUL columns per subset with a dashed 90% target line.

`scripts/results_table.py` prints README tables from the JSON.

---

## 15. App (Streamlit)

**Sidebar:**
- switches: multitask GP, Muse Spark agents, safety reviewer, planner agent, fault injection,
  show answer key;
- planning-horizon slider (10-100, step 5);
- subset filter;
- model tag and API status (connected, or offline with cache/templates);
- a "clear all faults" button.

**Tabs:**
- **Fleet:**
  - stat tiles: engines, and the count per decision;
  - ranked table: decision, urgency, RUL, band, P(fail), why;
  - plan builder (days, hangar slots, review slots) using the planner agent or the template.
- **Engine:** switch between fleet engines and replay engines.
  - Decision card: estimated RUL, 90% band, P(fail within H), unfamiliarity percentile,
    resolution and flags.
  - Fan chart: predicted RUL line, band, actual RUL (replay engines or answer key), dashed horizon
    line, cap line, and a decision strip underneath on the same x axis. Plus a table view.
  - Sensor-driver bars and the faithfulness note.
  - Fault-injection panel.
  - The LMC subset-correlation table.
- **Ask the agent:**
  - suggested questions and free text;
  - the briefing: headline with action icon and label, summary, abstain reason, key-numbers and
    drivers tables, next steps, caveats;
  - a "written by Muse Spark" label (or a note when the template was used), the reviewer
    verdict, the guard findings and the tool trace.
- **Evaluation:** the metric tables (the accessible twin of the charts), the charts, and expanders
  for OOD, faults, faithfulness and ablation.

**Chart rules:**
- Status colours are only for decisions and always come with an icon and a label:
  good #0ca30c (NO_INSPECTION), warning #fab219 (ABSTAIN), critical #d03b3b (INSPECT).
- Subset colours follow the entity: FD001 #2a78d6, FD002 #eb6834, FD003 #1baf7a, FD004 #eda100 in
  light mode (dark: #3987e5, #d95926, #199e70, #c98500). This palette passed a colour-vision check.
  Aqua and yellow have low contrast, so always add labels or a table.
- Signed contributions: red #e34948 for negative, blue #2a78d6 for positive, with the signed value
  printed.
- 2 px lines, the band at about 12% opacity, hairline solid grids, text in ink colours (never in a
  series colour), one y axis per chart.

---

## 16. Configuration

Every tunable lives in `configs/default.yaml` (Appendix C). The key values:

- **Data:** 14 sensors, cap 125, window 30, min history 15, splits 60/10/15/15.
- **Model:** dkl + lmc, 512 inducing points, 3 latents, embedding 16, 64 channels, 3 blocks,
  kernel 5, dropout 0.1, spectral norm 3.0.
- **Calibration:** alpha 0.10 and 0.02, bins [0, 30, 60, 90, 1000], 12 cuts per engine, minimum
  group 30.
- **Decision:** H 30, p_inspect 0.10, p_clear 0.02, epistemic gate 99th percentile, urgent 10 cycles,
  maximum recheck 50.
- **Sensor checks:** lookback 15, stuck after 10 cycles, range margin 1.5, jump x2.0, flat rate 1%.
- **Agent:** muse-spark-1.3, reasoning low, temperature 0.2, 5 tool rounds, cache on, 90 s timeout,
  3 retries.

Run order:

```
python -m ehagent.train prepare
python -m ehagent.train dkl --mode multitask        # pretrains the CNN if missing
python -m ehagent.train dkl --mode single           # ablation
python -m ehagent.calibrate --mode both
python -m ehagent.evaluate --mode multitask
python -m ehagent.precompute --mode multitask
streamlit run app/streamlit_app.py
```

---

## 17. Build plan

Day 1 = Wed Oct 28. The same plan as AGENTS.md section 7, grouped by day.

| Day | Date | Work | Exit check | If it fails |
|---|---|---|---|---|
| 1 | Wed Oct 28 | repo skeleton and config; API smoke test (chat, tools, JSON); data prep; CNN baseline; single-task DKL on FD001; multitask DKL (start it in the background); calibration | validation RMSE logged for the baseline and DKL; validation coverage between about 0.87 and 0.93 for every subset | embedding mode or the deep ensemble. If the credits haven't arrived, do the smoke test before Day 3 |
| night | Oct 28-29 | single-task DKL for FD002, FD003, FD004 (the ablation) | models saved | skip the ablation |
| 2 | Thu Oct 29 | decision engine, sensor checks and faults; explanations; service, planner and precompute; evaluation and policy replay | the fleet shows all 3 actions; a stuck Ps30 makes an engine abstain; `metrics.json` and the charts exist | `--skip-replay`, fix it later |
| 3 | Fri Oct 30 | agents end to end with the API: analyst, report writer, guard, reviewer, planner | briefings for an INSPECT, an ABSTAIN and a NO_INSPECTION engine pass the guard; offline, the same three come from templates | tighten the prompts using the guard's messages |
| 4 | Sat Oct 31 | app (Fleet, Engine, Ask the agent, Evaluation); record the demo engines into the cache; full run with `EHA_OFFLINE=1` | the whole demo works offline | hide the failing tab behind its switch |
| 5 | Sun Nov 1 | README results table and "Where Muse Spark is used", UI text, demo video | video recorded | re-record the next day |
| 6 | Mon Nov 2 | dry run from a fresh clone and venv; early submission if the platform allows edits | a stranger could run it from the README | fix the README, not the code |
| 7-8 | Tue Nov 3 - Wed Nov 4 | buffer only | submitted by **Wed Nov 4, 22:00 IST** at the latest | |

---

## 18. Demo script (about 3 minutes)

**Before recording:** click through the exact flow once with the API on, so every response is
cached. Then record with `EHA_OFFLINE=1`. Pick these engines in advance:
- an INSPECT engine;
- an ambiguous ABSTAIN engine;
- a NO_INSPECTION engine;
- a replay engine whose band tightens near failure;
- an engine whose call changes when the horizon moves from 30 to 60 cycles.

| Time | What to show |
|---|---|
| 0:00 | The hook: "Most models give one number; this one knows when that number isn't safe to act on." |
| 0:15 | Fleet tiles and the ranked table. The calls come from plain code, not the model. |
| 0:35 | A replay engine: the band and the decision strip. |
| 1:00 | Sensor drivers in cycles, and the faithfulness check. |
| 1:20 | Inject a stuck Ps30: the engine abstains on sensor integrity. |
| 1:40 | Ask the agent: the briefing, number checking, the escalate-only reviewer, the tool trace. |
| 2:10 | What-if on that last engine: "What if the next check is 60 cycles away?" The call changes, and the agent explains why. |
| 2:30 | The policy replay: missed failures and unnecessary inspections. Quote only what the numbers show. |
| 2:50 | Build the hangar plan, then close. |

If something fails live, untick the agents; the template briefing still shows. Say throughout that
the engines are simulated and that this is decision support, not an airworthiness call.

---

## 19. Cost and credits

- **Prices** (dev.meta.ai pricing page, Oct 26), per million tokens:
  - muse-spark-1.3: input $1.25, output $4.25, cached input $0.15.
  - Contributor mode (`muse-spark-1.3-contributor`): $0.10 / $0.20. In exchange, Meta may use your
    content to train future models. Limited to 150 requests per minute.
- **Rate limits** (per team): muse-spark 3,000 requests per minute and 4M tokens per minute.
- **Rough estimate:** a few cents per "Ask the agent" question, so $150 covers a couple of
  thousand runs. The cache makes replays free. Don't loop the agents over the whole fleet.
- Muse Code sessions probably spend the same credits (it runs on Muse Spark 1.2 by default), so
  check the dashboard after the first day.
- The Baseten, Modal and Fireworks partner credits are for Meta's open-weights models, not Muse
  Spark.

---

## 20. Reference implementation status

**Written** on Oct 24 by Claude: all modules in section 4, the prompts, the config, the README,
the build plan, the demo script, `run_all.sh`, the smoke test and the results table.

**Out of scope but still in the reference code:** `ehagent/inspection.py`,
`prompts/inspection_vision.md`, the Inspection tab, the post-inspection actions
(REMOVE_FROM_SERVICE, MONITOR, RETURN_TO_SERVICE), the `inspection` config section and the smoke
test's `--image` steps. Remove them, or skip them when borrowing code. Two of them break startup if
deleted on their own: the orchestrator loads every prompt listed in `PROMPT_NAMES`, and the smoke
test imports from `inspection.py`.

**Checked statically only, never run:**
- all 33 Python files parse;
- every cross-module import resolves;
- all 190 config reads exist in the YAML;
- pyflakes is clean;
- the shell script syntax is valid.

**Changed after review:**
- sensor jump multiplier 3.0 -> 2.0, and test spike 8 -> 12 std (so the fault test reliably
  catches spikes);
- the guard now allows the 99 and 30 thresholds.

**Unverified, check first:**
- GPyTorch LMC shapes and memory at runtime;
- training time on your hardware;
- the Meta API formats on your account (smoke test);
- Streamlit layout and charts.

---

## 21. Open questions

- Official Rules: is code written before Oct 26 allowed? Can code generated outside Meta's tools be
  included? (The reference implementation was written on Oct 24 with Claude.)
- What to submit: video length, public repo, write-up fields, whether judges must be able to run it.
- Who owns the submission and under what licence.
- How and when the $150 credits are applied.
- Whether `muse-spark-1.3-contributor` supports tools and JSON schema (smoke test).
- What Muse Wonder offers, and whether it would make a better safety reviewer.

---

## Appendix A: agent prompts

The system prompts for `prompts/`, with the image and inspection parts removed. Muse can tune the
wording, but keep the rules.

### A.1 Analyst (tool-calling evidence gatherer)

`prompts/analyst_system.md`

````markdown
You are the Prognostics Analyst in an engine health monitoring system for a fleet of turbofan engines (NASA C-MAPSS simulation). Your job in this step is to gather the evidence needed to answer the engineer's question by calling tools. A separate step writes the final briefing from the evidence you collect.

## What the system already decided
`get_engine_status` has been called for you. Its `action` field (INSPECT, NO_INSPECTION or ABSTAIN) comes from a deterministic decision engine that uses a calibrated uncertainty band. That action is final. You explain it; you never change it, argue against it, or suggest a different action.

## How to work
1. Read the status result you were given.
2. Call `explain_prediction` (top_k = 3) unless the question is only about scheduling or a different horizon.
3. If the action is ABSTAIN or `sensor_flags` is not empty, call `check_sensors`.
4. If the engineer asks about a different planning horizon ("what if we fly 50 more cycles", "next check is in 2 weeks"), call `what_if_horizon` with that horizon in cycles. If they give days, do not convert them yourself; use the default horizon and say a cycle count is needed.
5. Stop once you have what the question needs: usually two to four tool calls in total. Do not call the same tool twice with the same arguments.
6. When you are done, reply with one short sentence saying the evidence is ready. Do not write the briefing.

## Rules
- Never invent numbers. Every number used later must come from a tool result.
- The true remaining life of fleet engines is unknown by design. Never guess it.
- Engine ids look like FD001-TE-012. Use the id you were given; do not invent others.
- If a tool returns an error, do not retry it more than once.

````

### A.2 Report writer (structured briefing)

`prompts/report_writer_system.md`

````markdown
You write the maintenance briefing for one turbofan engine from the evidence JSON you are given. Your output must match the JSON schema exactly.

## Hard rules (a checker rejects drafts that break them)
1. `recommendation` must be exactly `evidence.authoritative_action`. Never change it, soften it or add a second recommendation.
2. Use only numbers that appear in the evidence. Copy them at the precision shown. Do not compute anything new: no subtractions, sums, averages, unit conversions or rounding to different precision. Probabilities may be written as percentages (0.043 -> 4.3%).
3. `key_numbers`: 3 to 6 items. Each `value` is copied from the evidence and `source` names the tool it came from (for example `get_engine_status`).
4. `sensor_drivers`: up to 3 sensors from `explain_prediction.contributions`, in the order given, with `contribution_cycles` copied exactly. A negative contribution means this sensor's deviation from a healthy engine pulls the remaining-life estimate down by that many cycles. Use the sensor's name (for example T50 or Ps30) and say in plain words what it measures.
5. If the action is ABSTAIN, `abstain_reason` says in plain words why the system will not make the call: a sensor-integrity problem, engine behaviour unlike the training fleet, or an uncertainty band that straddles the planning horizon. `next_steps` must include the `resolution` text from the evidence. For any other action, `abstain_reason` is an empty string.
6. If `healthy_region` is true, say the engine is in the healthy region and that the exact remaining life above the cap cannot be estimated. Never present the cap as a prediction.

## Style
- Audience: line maintenance engineers. Plain language, no machine-learning jargon. Say "uncertainty band", not "posterior variance"; "unfamiliar engine behaviour", not "epistemic percentile".
- `headline`: at most 12 words, leads with the action.
- `summary`: 2 to 4 sentences: the estimate with its band, the failure probability within the horizon, and why the action follows.
- `next_steps`: 1 to 3 concrete actions.
- `caveats`: always include that results come from a simulated benchmark (NASA C-MAPSS) and are decision support, not an airworthiness determination. Add a caveat if sensor flags exist or the band is very wide.

````

### A.3 Safety reviewer (escalate-only)

`prompts/reviewer_system.md`

````markdown
You are the Safety Reviewer. You independently check a maintenance briefing against the raw evidence before it reaches an engineer. You can only make outcomes more cautious, never less.

## Check, in order
1. Action: does `recommendation` equal `evidence.authoritative_action`?
2. Numbers: does every number in the briefing appear in the evidence (probabilities may appear as percentages)?
3. Understated risk: does the briefing present the engine as safe to keep flying while any of these hold?
   - `p_fail_within_horizon` is above `thresholds.p_clear`
   - `rul_lo` is at or below `horizon_cycles`
   - `sensor_flags` is not empty, or `check_sensors.status` is "suspect"
   - `epistemic_percentile` is 95 or higher
4. Abstentions: if the action is ABSTAIN, does the briefing explain why and give a concrete resolution?
5. Tone: does it avoid claiming airworthiness or certainty the evidence does not support?

## Verdicts
- `approve`: no issues.
- `revise`: wording or number problems that do not change safety. List each one in `issues`.
- `force_abstain`: the evidence does not support letting this engine keep flying (for example NO_INSPECTION with unresolved sensor flags). List the reasons in `safety_concerns`. The system will escalate the engine to an engineer.

Do not use `force_abstain` to disagree with an INSPECT or ABSTAIN action; those are already cautious. Be brief: each issue or concern is one sentence. `rationale` is at most two sentences.

````

### A.4 Fleet planner

`prompts/planner_system.md`

````markdown
You are the Fleet Maintenance Planner for a fleet of turbofan engines (NASA C-MAPSS simulation).

## Tool step
`plan_inspections` has already produced the schedule. It is authoritative: the same engines, on the same days, with the same actions and priorities. You may call `fleet_triage` or `get_engine_status` to understand why engines are on the list, then stop calling tools.

## Writing step (when asked to write the plan JSON)
- `work_orders`: one entry per scheduled item, copying `engine_id`, `day`, `action` and `priority` exactly from the schedule. Write `reason` in one plain sentence based on that engine's evidence (for example "failure risk within the horizon is elevated" or "Ps30 sensor looks stuck; verify before trusting any estimate"). Do not put numbers in `reason` unless they appear in the evidence.
- Priorities: P1 = act today (immediate urgency, or a sensor-integrity problem), P2 = inspect soon, P3 = routine review.
- `summary`: 2 to 3 sentences: how many inspections and reviews, over how many days, and the biggest risk. Use only counts and numbers from the evidence.
- `capacity_warnings`: copy every item from the schedule's `conflicts`, then add plain-language warnings if hangar or review capacity is the bottleneck.

## Rules
- Never add, drop or move an engine. Never change an action or a day.
- Never invent numbers. Never guess the true remaining life of an engine.
- This is decision support for maintenance planners, not an airworthiness determination.

````

---

## Appendix B: structured-output schemas

Adapted from `ehagent/agent/schemas.py`, with the image parts removed. B.1 = `ENGINE_BRIEFING_SCHEMA`, B.2 = `SAFETY_REVIEW_SCHEMA`, B.3 = `MAINTENANCE_PLAN_SCHEMA`. Send each as `response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": False}}`.

````python
"""JSON schemas for Muse Spark structured outputs (object roots, no recursion, all keys required)."""
from __future__ import annotations

ACTIONS = ["INSPECT", "NO_INSPECTION", "ABSTAIN"]

ENGINE_BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "engine_id": {"type": "string"},
        "recommendation": {"type": "string", "enum": ACTIONS},
        "headline": {"type": "string", "description": "At most 12 words."},
        "summary": {"type": "string", "description": "2-4 plain-language sentences for a maintenance engineer."},
        "key_numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "source": {"type": "string", "description": "Tool the number was copied from."},
                },
                "required": ["label", "value", "unit", "source"],
                "additionalProperties": False,
            },
        },
        "sensor_drivers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sensor": {"type": "string"},
                    "name": {"type": "string"},
                    "contribution_cycles": {"type": "number"},
                    "plain_language": {"type": "string"},
                },
                "required": ["sensor", "name", "contribution_cycles", "plain_language"],
                "additionalProperties": False,
            },
        },
        "abstain_reason": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["engine_id", "recommendation", "headline", "summary", "key_numbers", "sensor_drivers",
                 "abstain_reason", "next_steps", "caveats"],
    "additionalProperties": False,
}

SAFETY_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "revise", "force_abstain"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "safety_concerns": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "issues", "safety_concerns", "rationale"],
    "additionalProperties": False,
}

MAINTENANCE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "work_orders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "engine_id": {"type": "string"},
                    "day": {"type": "integer"},
                    "action": {"type": "string", "enum": ["INSPECT", "ENGINEERING_REVIEW"]},
                    "priority": {"type": "string", "enum": ["P1", "P2", "P3"]},
                    "reason": {"type": "string"},
                },
                "required": ["engine_id", "day", "action", "priority", "reason"],
                "additionalProperties": False,
            },
        },
        "capacity_warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "work_orders", "capacity_warnings"],
    "additionalProperties": False,
}
````

---

## Appendix C: configuration file

`configs/default.yaml`, with the image settings removed. Every value is read by the code; tune
values, not code.

````yaml
# Engine Health Agent - default configuration.
# Every value here is read by the code: tune values, not code.

project:
  seed: 42
  device: auto            # auto | cpu | cuda

paths:
  raw_dir: data/raw/CMAPSSData        # train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt
  artifacts_dir: artifacts            # prepared data, models, calibration, precomputed results
  reports_dir: reports                # evaluation JSON + charts
  cache_dir: .cache                   # cached Muse Spark responses (replayable offline demo)
  prompts_dir: prompts

data:
  subsets: [FD001, FD002, FD003, FD004]
  # The 14 informative sensors used by most C-MAPSS papers (the other 7 are ~constant).
  sensors: [s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21]
  n_conditions: {FD001: 1, FD002: 6, FD003: 1, FD004: 6}
  rul_cap: 125            # piecewise-linear target: early-life labels are capped here
  window: 30              # cycles per input window
  min_history: 15         # shortest history used for training windows (edge-padded)
  split: {train: 0.60, val: 0.10, calib: 0.15, sim: 0.15}   # by engine ID, training file only

model:
  family: dkl             # dkl (primary) | ensemble (backup) | cnn (point baseline)
  multitask_mode: lmc     # lmc = LMC variational GP over subsets | embedding = backup (shared CNN + subset embedding + one GP)
  embed_dim: 16
  channels: 64
  n_blocks: 3
  kernel_size: 5
  dropout: 0.1
  spectral_norm: true     # keeps the feature map distance-aware so the GP can notice unfamiliar engines
  sn_coeff: 3.0
  task_embed_dim: 4
  inducing_points: 512
  num_latents: 3          # LMC latent GPs
  ensemble_size: 5

train:
  batch_size: 256
  dkl_batch_size: 512
  pretrain_epochs: 40
  pretrain_lr: 1.0e-3
  dkl_epochs: 30
  dkl_lr_features: 1.0e-4
  dkl_lr_gp: 1.0e-2
  ensemble_epochs: 40
  ensemble_warmup_epochs: 10
  ensemble_lr: 1.0e-3
  beta_nll: 0.5
  weight_decay: 1.0e-5
  grad_clip: 5.0
  patience: 8
  early_stop_metric: rmse   # rmse | nll
  inducing_init_samples: 20000
  num_workers: 0

calibration:
  alpha: 0.10               # 90% bands
  alpha_conservative: 0.02  # 98% lower bound, used for deadlines and safe re-check intervals
  bins: [0, 30, 60, 90, 1000]   # Mondrian bins on predicted RUL
  cuts_per_engine: 12
  min_group_size: 30

decision:
  horizon_cycles: 30        # H = cycles until the next maintenance opportunity
  p_inspect: 0.10           # inspect if P(fail within H) >= this
  p_clear: 0.02             # no inspection if P(fail within H) <= this; in between = abstain
  epistemic_percentile_abstain: 99.0
  urgent_cycles: 10
  max_recheck_cycles: 50

sensor_checks:
  lookback: 15
  stuck_min_cycles: 10
  z_abs_max_margin: 1.5
  jump_multiplier: 2.0      # flag a cycle-to-cycle change above 2x the 99.9th percentile seen in training
  flat_rate_max: 0.01

faults:
  default_sensor: s11       # Ps30
  default_start_offset: 20  # cycles before the latest cycle
  magnitudes: {stuck: 0.0, drift: 3.0, spike: 12.0, noise: 3.0}   # in training std (drift: per 10 cycles)

simulation:
  useful_inspection_rul: 30 # an inspection finds actionable degradation if true RUL <= this
  min_lead_cycles: 1
  cooldown_cycles: 10
  fixed_interval_cycles: 50
  abstain_action: inspect   # inspect | review_oracle | skip

planner:
  cycles_per_day: 4
  slots_per_day: 2
  review_slots_per_day: 4
  horizon_days: 7

attribution:
  ig_steps: 32
  top_k: 5
  deletion_top_k: 3
  deletion_random_trials: 20

evaluation:
  ood_source: FD001
  ood_target: FD004
  ood_max_engines: 100
  fault_test_engines: 30
  attribution_test_engines: 30

agent:
  default_model: muse-spark-1.3   # check dev.meta.ai/docs/models for current IDs
  model_env: MUSE_MODEL
  default_base_url: https://api.meta.ai/v1
  base_url_env: MODEL_API_BASE_URL
  api_key_env: MODEL_API_KEY
  offline_env: EHA_OFFLINE
  reasoning_effort: low     # minimal | low | medium | high | xhigh ; null to omit
  temperature: 0.2          # null to omit
  max_tool_rounds: 5
  cache: true
  timeout_s: 90
  max_retries: 3

features:          # demo switches (the app sidebar can override them live)
  multitask_model: true
  agents: true
  reviewer: true
  planner_agent: true
  fault_injection: true
  show_answer_key: false
````
