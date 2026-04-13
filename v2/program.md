# ART2 v2 Program

This is the definitive operating protocol for the current exact-chain side-collapse reset.
Read this file first, then [COMMANDS.md](COMMANDS.md), then [current_state.md](docs/current_state.md).

If optimization pressure conflicts with project judgment, defer to [founder_intent.md](docs/founder_intent.md).

## Current Status

- **v4 exact-chain — profitable model with optimized execution policy (exp_144)**
- Active manifest: `v2/data.pt`
- Dataset version: `v4_exact_chain`
- Dataset fingerprint: `46f2d184e186496f`
- Unique days: 986
- Per-day sidecars: `v2/data_sidecars/`
- Official exact-chain scored runs: `exp_074`–`exp_078`, `exp_099`, `exp_104`, `exp_106`, `exp_122`, `exp_125`, `exp_133`, `exp_137`, `exp_139`, `exp_140`, `exp_142`, `exp_143`, `exp_144`
- Current official baseline artifact: **`exp_144`** (`score=0.547`, PF 1.356, DD 8.2%, Sortino 7.36, +$4,771)
- Current working code: exp_144 (exp_139 architecture + breakeven_trigger_pct 0.15 + cooldown_bars 3)
- Next experiment: `exp_145`
- Session status: **profitable; optimizing**

## Mission Boundary

- The current mission is not live trading.
- The current mission is a trustworthy exact-chain research system.

## Established Facts

- Oracle replay scores `6.0`, so the evaluation harness is achievable
- Logistic regression reaches `60.9%` test direction accuracy from current-bar features
- Longer lookback windows did not improve that diagnostic
- Contract features alone have **zero predictive power** for oracle contract selection (8.2% exact match vs 6.2% random chance)
- At `SOFT_TEMP=0.20`, the KL selection target is near-uniform (median max prob 0.23) — the selection gradient is negligible
- At `SOFT_TEMP=0.05`, targets become peaked (median max prob 0.51) — but exp_079 showed too few trades at this temp
- Gate class imbalance is 4:1 (80% trade / 20% no-trade), not 15:1 — learnable with balanced sampling
- 30.6% of bars have top margin < 0.01 (no meaningful "best" contract) — training on noise
- The gate threshold (0.04) never filters anything in the oracle — all tradeable bars exceed it
- The current bottleneck is conditional side calibration inside contract scoring, not raw side-label imbalance
- `exp_106` materially improved profit factor and positive-day rate versus `exp_104`, but its promote trace still collapsed fully to calls (`191C / 0P`)
- Kronos-inspired standalone screens `exp_107` through `exp_109` did not produce a promotable candidate; `exp_109` only partially improved direction balance and still failed the hard gate
- `exp_110` confirmed that cross-side calibration is the right failure surface, but the first margin-style implementation overcorrected into puts and remained economically weak
- `exp_119` is the strongest screening result so far: direction preserved (`119C / 34P`) with the best economics of the side-aware family
- `exp_121` ranking loss collapsed to `17C / 94P` with `100.2%` drawdown, proving the side-collapse problem survives a loss-family swap
- The executable snapshot labels and soft KL targets are roughly side-neutral to slightly call-favored; side collapse is a model-dynamics problem, not a raw label-majority fact

## Current Live Baseline

- Model: exact-chain contract scorer
- Inputs: 30-bar windows of 47 normalized context features plus the current executable contract snapshot
- Outputs: `NO_TRADE` plus one score per executable contract
- Loss: balanced gate BCE plus soft KL selection only
- `SOFT_TEMP=0.10`
- `NOISE_MARGIN=0.01`
- No direct PnL regression
- No auxiliary side head
- No gate reweighting
- No pairwise ranking selection loss
- Current policy window: bar `60` through `120`
- Risk remains policy-driven in `v2/core/policy.py`

## Compute Rules

- All model training runs on the Akash H100 GPU via `./v2/ops/deploy.sh`
- Never train locally on the MacBook
- Local work is for editing, replay/evaluation, plotting, data rebuilds, and documentation
- Every experiment trains from scratch

## Mutable Surface

Default experiment loop:

- `v2/train.py`
- `v2/core/policy.py`

Approved expanded surface for the current Kronos-inspired block:

- `v2/train.py`
- `v2/core/metrics.py`
- `v2/replay.py`
- `v2/core/data_integrity.py`
- `v2/ops/pre_run_gate.py`
- live docs that must stay consistent with the active protocol

Use the expanded surface only when the hypothesis cannot be tested honestly inside the default two-file loop. Any related live docs must be updated in the same change set so code and protocol stay aligned.

## Required Pre-GPU Gate

Before any GPU run, the local gate must pass:

- Command: `python3 -m v2.ops.pre_run_gate --data v2/data.pt`
- `./v2/ops/deploy.sh run_screen ...` and `run_one ...` run this automatically

The gate fails on:

- failing `harness_eval`
- log/doc/code disagreement in the live exact-chain surface
- legacy runner or old-head references in live files
- non-official rows in `v2/results.tsv`
- mismatch between `v2/train.py` and `v2/docs/how_training_works.md`
- data integrity errors (manifest, features, sidecars)

## Required Post-Run Trace

After every official run, a decision trace **must** be generated before the keep/revert decision:

- Command: `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`
- Output: `v2/artifacts/replay_traces.csv` + printed summary
- The trace captures every eligible bar: model scores, oracle answer, realized P&L, skip reasons
- The trace summary reports: gate accuracy, selection accuracy, P&L gap vs oracle, noisy bar %, exit reason breakdown

The keep/revert decision must be informed by trace analysis, not score alone. A model with a higher score but worse gate accuracy or selection accuracy than the baseline should be investigated before keeping.

### Trace-Informed Hypothesis Formation

When forming the next hypothesis after a keep or revert:

1. Load the traces: `v2/artifacts/replay_traces.csv`
2. Identify the top failure mode by category:
   - **Gate false positives** (traded when shouldn't have): filter `decision=trade` where `oracle_pnl < 0`
   - **Gate false negatives** (skipped a winner): filter `skip_reason=gate` where `oracle_pnl > 0.04`
   - **Selection misses** (traded but picked wrong contract): filter `decision=trade` where `delta_pnl < -0.05`
   - **Timing misses** (right idea, wrong bar): cluster by `bar_of_day` and `vix_regime`
3. The next hypothesis should target the largest failure cluster

## Experiment Workflow

### Screening Run

- Command: `./v2/ops/deploy.sh run_screen exp_NNN`
- Runs 1 fold only
- No artifacts saved
- No model downloaded
- No `results.tsv` entry
- Screening notes go to `v2/lab_notebook.md` only
- Screening is for hypothesis triage, not promotion

Reject a screening run as a promotion candidate on:

- hard gate failure
- zero trades
- score `<= 0`

Direction mix is a required diagnostic on every screening run, but it is not an automatic rejection rule by itself.

A screening run may still justify the next experiment in the same hypothesis family when it materially improves the targeted trace failure mode, even if it remains non-promotable on score or hard gates.

### One Hypothesis, Not One File

- Keep experiments scoped to one hypothesis.
- A hypothesis may span up to a few coordinated edits when training, replay, or audit wiring are inseparable.
- Do not bundle unrelated ideas into the same experiment just because the expanded mutable surface is available.

### Official Run

- Command: `./v2/ops/deploy.sh run_one exp_NNN`
- Runs all 5 folds
- Saves artifacts and downloads `v2/models/model_candidate.pt`
- Official runs are the only scored runs
- Official runs append to both `v2/results.tsv` and `v2/lab_notebook.md`

Keep only if the aggregate result:

- beats the current exact-chain best in `v2/results.tsv`
- beats all four baselines
- has no hard-gate failure

Call/put balance must still be reviewed on every official run, but it no longer overrides the economics as an automatic hard fail.

Do not promote an official run solely because traces improved. Promotion remains score- and baseline-gated.

### Post-Official Workflow

After an official run, complete ALL steps below before reporting to the user. Do not skip any.

1. **VALIDATE MODEL LOCALLY** (confirm numbers match remote):
   `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote`

2. **DECISION TRACE** (mandatory before keep/revert):
   `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`
   Compare trace diagnostics vs current best: gate accuracy, selection accuracy, delta gap, direction balance.

3. **KEEP or REVERT** (informed by trace analysis, not score alone):
   - Keep: `python3 -m v2.ops.model_manage keep`
   - Revert: `python3 -m v2.ops.model_manage revert` (optionally `git checkout HEAD~1 -- v2/train.py v2/core/policy.py`)

4. **REGENERATE VISUALIZATIONS:**
   `python3 -m v2.plot_trades` → trades.html, equity.html, trades.csv
   `python3 v2/plot_progress.py` → progress.png

5. **ANALYZE TRADES** from `v2/output/trades.csv`:
   Report exit breakdown, direction split, bar-of-day profitability, average P&L per trade.

6. **UPDATE LAB NOTEBOOK** (`v2/lab_notebook.md`):
   Full entry: hypothesis, results table, trace summary, keep/revert decision, next direction.

7. **IF PROMOTING, UPDATE LIVE DOCS:**
   - `v2/HANDOFF.md` (current live code, research position, key findings)
   - `v2/program.md` (current status section)
   - `v2/docs/current_state.md` (snapshot section)

8. **COMMIT EVERYTHING** in one clean commit: code, artifacts, models, docs, lab notebook.

9. **FORM NEXT HYPOTHESIS** from trace analysis (see "Trace-Informed Hypothesis Formation" above).

10. **CHECK SESSION LIMITS** before the next experiment.

**Present to user:** results table, trace comparison vs baseline, trade analysis summary, what docs were updated, proposed next hypothesis. One complete message.

If the official run is not promotable but traces show a targeted failure-mode improvement, it is valid to continue the same hypothesis family into the next experiment instead of abandoning it immediately.

## Dataset Escape Hatch

- The frozen `v4_exact_chain` dataset remains the live authority by default.
- A separate audit track may inspect raw inputs and sidecars for structural breaks, stagnant stretches, or label-path anomalies without rebuilding the dataset.
- Open a deliberate dataset-version decision only if the audit flags at least `5` sessions or materially overlaps the worst trace days from the current baseline/candidate artifact.
- Never silently mutate `v2/data.pt` or reinterpret old results under a new dataset contract.

## Abandoned Or Parked Approaches

- direct PnL regression in the live loss stack
- shared auxiliary side heads
- gate BCE reweighting
- score regularization as the live baseline
- treating `exp_088` or `exp_089` as scored evidence
- tanh-bounded score head (gradient saturation kills selection)
- LOOKBACK=1 current-bar-only (temporal context needed for direction balance)

## Hypothesis Queue

Current queue after exp_140 promotion (optimized exit policy):

1. Improve selection accuracy (7.4% → higher) — most direct lever for higher PF
2. Reduce fold variance — 3/5 folds still at -0.200; fold 0 improved to +0.076 with tighter trailing
3. Fix theta formula for puts in data pipeline (`v2/pipeline/compute_features.py`) — requires sidecar rebuild
4. Explore further trailing tier optimization — intermediate tiers between 15% and 50%
5. Audit decision: keep `v4_exact_chain` frozen unless the anomaly track reaches the explicit trigger

## Session Limits

- 20 experiments
- 10 hours
- 6 no-improve streak
- 4hr plateau
- 3 crashes

Stop when any limit fires.

## Score

```python
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```

Hard gates:

- Minimum 30 trades
- Minimum 15 traded days
- Maximum 20% account drawdown

Direction diagnostics:

- report call count, put count, minority side share, and direction balance on every screening and official run
- use traces plus the side-bias audit to investigate one-sided behavior; do not overwrite the score with a synthetic direction-collapse penalty

The model must also beat all four baselines:

1. Random
2. ATM-Always
3. Simple-Rules
4. ATM-Trailing

## Source Of Truth

See `v2/docs/README.md` for the live documentation index.

For founder voice, anti-goals, and standards, see `v2/docs/founder_intent.md`.

Do not read `archive/` unless the human asks for historical context.
