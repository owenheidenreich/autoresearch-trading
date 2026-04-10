# ART2 v2 Program

This is the definitive operating protocol for the current exact-chain rebuild.
Read this file first, then [COMMANDS.md](COMMANDS.md), then [current_state.md](docs/current_state.md).

## Current Status

- **v4 exact-chain harness rebuild.**
- The active manifest is `v2/data.pt`.
- Current dataset version: `v4_exact_chain`
- Canonical raw option source: full same-day SPXW 0DTE chain cache
- `v2/data.pt` stores the market-context manifest; per-day chain snapshots live in `v2/data_sidecars/`
- Experiment numbering resumes at `exp_074`

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

## Setup

1. Work from a clean branch.
2. Verify the active dataset:
   - `python3 -m v2.analysis.harness_eval --data v2/data.pt`
3. Boot the GPU:
   - `./v2/ops/deploy.sh boot`
4. Upload code, `data.pt`, and chain sidecars:
   - `./v2/ops/deploy.sh start`

## Experiment Loop

1. Read the latest results, plots, and trade analysis.
2. Write one concrete hypothesis.
3. Edit only `v2/train.py` and/or `v2/core/policy.py`.
4. Run `python3 -m py_compile` on every changed Python file.
5. Commit the change.
6. Run one experiment:
   - `./v2/ops/deploy.sh run_one exp_NNN`
7. Read the aggregate walk-forward score and the per-fold scores from stdout.
8. Decide:
   - KEEP: `python3 v2/ops/model_manage.py keep`
   - REVERT: `git checkout HEAD~1 -- v2/train.py v2/core/policy.py`
   - Then: `python3 v2/ops/model_manage.py revert`
9. Append the result to `v2/results.tsv`.
10. Regenerate visual artifacts:
   - `python3 -m v2.plot_trades --model v2/model.pt`
   - `python3 v2/plot_progress.py`
11. Read the loss / trade diagnostics:
   - `python3 -m v2.analysis.analyze_losses`
12. Update `v2/lab_notebook.md`.
13. Check session limits before the next experiment.

## Harness Eval Suite

- `core_regression`: non-negotiable invariants, zero tolerated regressions
- `optimization`: cases used while repairing the harness
- `holdout`: unseen tagged cases used to verify harness generalization

If a nightly experiment exposes a harness bug:

1. stop the nightly loop
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

Use these docs for current behavior:

- [current_state.md](docs/current_state.md)
- [data_contract.md](docs/data_contract.md)
- [evaluator.md](docs/evaluator.md)
- [how_training_works.md](docs/how_training_works.md)
