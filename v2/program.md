# ART2 v2 Program

This is the definitive operating protocol for the current exact-chain recovery.
Read this file first, then [COMMANDS.md](COMMANDS.md), then [current_state.md](docs/current_state.md).

## Current Status

- **v4 exact-chain recovery phase.**
- The active manifest is `v2/data.pt`.
- Current dataset version: `v4_exact_chain`
- Dataset fingerprint: `46f2d184e186496f`
- Unique days: 986
- Per-day sidecars: `v2/data_sidecars/`
- All exact-chain experiments so far (exp_074 through exp_078) failed
- Next experiment: `exp_079`

## Compute Rules

- All model training runs on the Akash H100 GPU via `./v2/ops/deploy.sh`.
- Never train locally on the MacBook.
- Local work is for editing, committing, replay/evaluation, plotting, data rebuilds, and documentation.
- Every experiment trains from scratch. `deploy.sh start` clears the remote `v2/model.pt` before the next run.

## Current System Summary

- Model: exact-chain contract scorer
- Inputs: 30-bar windows of 47 normalized context features plus the current executable contract snapshot
- Output: one `NO_TRADE` score plus one score per executable contract
- Risk management: policy-driven fixed stop / target / max-hold in `v2/core/policy.py`
- Evaluation: 5-fold walk-forward CV over 300 test days plus 20 shadow days
- Artifact rule: only promoted artifacts are eligible for default replay and they must match the current dataset fingerprint

## Mutable Surface

During the normal experiment loop, only these two files are mutable:

- `v2/train.py`
- `v2/core/policy.py`

Everything else is the frozen evaluation harness unless you are doing explicit infrastructure, data, or documentation work.

## Two-Tier Experiment Workflow

### Screening Run (provisional)

- Command: `./v2/ops/deploy.sh run_screen exp_NNN`
- Runs 1 fold only
- No artifacts saved, no model downloaded, no `results.tsv` entry
- Log a short note in `lab_notebook.md` only
- Screening never drives keep/revert decisions

Screening reject rules:
- Gate failure
- Zero trades
- Score <= 0
- Minority direction balance < 15%
- Fully one-sided behavior

### Official Run (scored)

- Command: `./v2/ops/deploy.sh run_one exp_NNN`
- Runs all 5 folds
- Saves artifacts, downloads model_candidate.pt
- Entries go to both `results.tsv` and `lab_notebook.md`
- Official runs are the only scored runs

Official keep rules:
- Aggregate 5-fold score beats the current exact-chain best in `results.tsv`
- Beats all four baselines
- No hard-gate failure

### Post-Official Workflow

After an official run:
1. KEEP or REVERT:
   - `python3 v2/ops/model_manage.py keep`
   - or: `git checkout HEAD~1 -- v2/train.py v2/core/policy.py` then `python3 v2/ops/model_manage.py revert`
2. Regenerate plots:
   - `python3 -m v2.plot_trades --model v2/model.pt`
   - `python3 v2/plot_progress.py`
3. Run `python3 -m v2.analysis.analyze_losses`
4. Update `results.tsv` and `lab_notebook.md`
5. Check session limits before next experiment

## Setup

1. Work from a clean branch.
2. Verify the active dataset:
   - `python3 -m v2.analysis.harness_eval --data v2/data.pt`
3. Boot the GPU:
   - `./v2/ops/deploy.sh boot`
4. Upload code, `data.pt`, and chain sidecars:
   - `./v2/ops/deploy.sh start`

## Hard Protocol Rules

- Screening is provisional and never drives keep/revert
- Official 5-fold runs are the only scored runs
- Infrastructure/doc/archive changes are not experiments
- Screening notes go to `lab_notebook.md` only
- Official runs go to `results.tsv` and `lab_notebook.md`
- Plots and `analyze_losses` are required after official runs only
- After 3 consecutive official reverts, analyze trade-level failure before another structural model change

## Session Limits

- 20 experiments
- 10 hours
- 6 no-improve streak
- 4hr plateau
- 3 crashes

Stop when any limit fires.

## Harness Eval Suite

- `core_regression`: non-negotiable invariants, zero tolerated regressions
- `optimization`: cases used while repairing the harness
- `holdout`: unseen tagged cases used to verify harness generalization

If an experiment exposes a harness bug:

1. stop the loop
2. add a harness eval case
3. repair the harness on a dedicated branch
4. re-freeze before resuming experiments

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

See `v2/docs/README.md` for the full live documentation index.

Do not read `archive/` unless the human asks for historical context.
