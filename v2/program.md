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
- Next experiment: `exp_093`

## Mission Boundary

- The current mission is not live trading.
- The current mission is a trustworthy exact-chain research system.

## Established Facts

- Oracle replay scores `6.0`, so the evaluation harness is achievable
- Logistic regression reaches `60.9%` test direction accuracy from current-bar features
- Longer lookback windows did not improve that diagnostic
- The current bottleneck is model architecture and training, not the frozen exact-chain dataset or replay harness

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

1. KEEP or REVERT:
   - `python3 v2/ops/model_manage.py keep`
   - or: `git checkout HEAD~1 -- v2/train.py v2/core/policy.py` then `python3 v2/ops/model_manage.py revert`
2. Regenerate plots:
   - `python3 -m v2.plot_trades`
   - `python3 v2/plot_progress.py`
3. Run `python3 -m v2.analysis.analyze_losses`
4. Update `v2/lab_notebook.md`
5. Check session limits before the next experiment

## Abandoned Or Parked Approaches

- direct PnL regression in the live loss stack
- shared auxiliary side heads
- gate BCE reweighting
- score regularization as the live baseline
- treating `exp_088` or `exp_089` as scored evidence
- tanh-bounded score head (gradient saturation kills selection)

## Hypothesis Queue

- `exp_093`: current-bar-only / `LOOKBACK=1` architecture test
- `exp_094`: consider a replay-compatible follow-up based on `exp_093` results
- Defer detached two-stage side models until simpler replay-compatible changes are exhausted

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
