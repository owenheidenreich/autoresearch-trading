# ART2 v2 Program

This is the definitive operating protocol for the current exact-chain reset.
Read this file first, then [COMMANDS.md](COMMANDS.md), then [current_state.md](docs/current_state.md).

If optimization pressure conflicts with project judgment, defer to [founder_intent.md](docs/founder_intent.md).

## Current Status

- **v4 exact-chain reset and baseline re-establishment phase**
- Active manifest: `v2/data.pt`
- Dataset version: `v4_exact_chain`
- Dataset fingerprint: `46f2d184e186496f`
- Unique days: 986
- Per-day sidecars: `v2/data_sidecars/`
- Official exact-chain scored runs: `exp_074` through `exp_078`
- Screening history: `exp_079` through `exp_087` in `v2/lab_notebook.md`
- Unresolved code-only states: `exp_088`, `exp_089`
- Next experiment: `exp_096`
- Session status: **6 no-improve streak fired** (exp_090–095 all scored -0.2 or -0.3)

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
- The current bottleneck is the selection loss design (near-uniform KL targets) and task decomposition (38-way ranking vs binary direction)

## Current Live Baseline

- Model: exact-chain contract scorer
- Inputs: 30-bar windows of 47 normalized context features plus the current executable contract snapshot
- Outputs: `NO_TRADE` plus one score per executable contract
- Loss: gate BCE plus soft KL selection only
- `SOFT_TEMP=0.20`
- No direct PnL regression
- No auxiliary side head
- No gate reweighting
- No score regularization
- Risk remains policy-driven in `v2/core/policy.py`

## Compute Rules

- All model training runs on the Akash H100 GPU via `./v2/ops/deploy.sh`
- Never train locally on the MacBook
- Local work is for editing, replay/evaluation, plotting, data rebuilds, and documentation
- Every experiment trains from scratch

## Mutable Surface

During the normal experiment loop, only these two files are mutable:

- `v2/train.py`
- `v2/core/policy.py`

Everything else is frozen harness, data, ops, or documentation unless the work is explicitly infrastructure, archive, or docs cleanup.

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

Reject a screening run on:

- hard gate failure
- zero trades
- score `<= 0`
- minority direction balance `< 15%`
- fully one-sided behavior

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

### Post-Official Workflow

After an official run:

1. **DECISION TRACE** (mandatory before keep/revert):
   - `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`
   - Review trace summary: gate accuracy, selection accuracy, P&L gap, failure modes
   - Compare against baseline traces if available
2. KEEP or REVERT (informed by trace analysis, not score alone):
   - `python3 v2/ops/model_manage.py keep`
   - or: `git checkout HEAD~1 -- v2/train.py v2/core/policy.py` then `python3 v2/ops/model_manage.py revert`
3. Regenerate plots:
   - `python3 -m v2.plot_trades`
   - `python3 v2/plot_progress.py`
4. Run `python3 -m v2.analysis.analyze_losses`
5. Update `v2/lab_notebook.md` with trace insights
6. **FORM NEXT HYPOTHESIS from traces** (see "Trace-Informed Hypothesis Formation" above)
7. Check session limits before the next experiment

## Abandoned Or Parked Approaches

- direct PnL regression in the live loss stack
- shared auxiliary side heads
- gate BCE reweighting
- score regularization as the live baseline
- treating `exp_088` or `exp_089` as scored evidence
- tanh-bounded score head (gradient saturation kills selection)
- LOOKBACK=1 current-bar-only (temporal context needed for direction balance)

## Hypothesis Queue

Post-diagnostic hypothesis queue — informed by the five-part diagnostic (2026-04-10):

- `exp_096`: `SOFT_TEMP=0.05` + balanced gate sampling — combine peaked KL targets with balanced gate; the two most impactful individual findings
- `exp_097`: filter noisy bars (top margin < 0.01) from selection loss — remove the 30.6% of training signal that is pure noise
- `exp_098`: direction-first decomposition — learn call/put direction from context features, then select strike within the chosen direction; recovers the old binary task structure
- Defer broad architecture changes until the selection signal is fixed

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
- Minimum 15% minority direction balance
- Maximum 20% account drawdown

The model must also beat all four baselines:

1. Random
2. ATM-Always
3. Simple-Rules
4. ATM-Trailing

## Source Of Truth

See `v2/docs/README.md` for the live documentation index.

For founder voice, anti-goals, and standards, see `v2/docs/founder_intent.md`.

Do not read `archive/` unless the human asks for historical context.
