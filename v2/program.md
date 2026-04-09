# ART2 v2 Program

This is the definitive operating protocol for the current repaired v2 harness.
Read this file first, then [COMMANDS.md](COMMANDS.md), then [current_state.md](docs/current_state.md).

## Current Status

- The harness repair is complete and re-frozen.
- The active dataset is `v2/data.pt`.
- Current dataset version: `v2_harness_repair`
- Current dataset fingerprint: `03566aeb8adf1040`
- `v2/data_harness_repair.pt` is currently a backup copy of the same repaired dataset.
- Pre-repair scores and legacy artifacts are historical only. They are not the baseline for current research.
- Live and paper trading are not implemented yet. Right now this repo is a research and replay harness.

## Compute Rules

- All model training runs on the Akash H100 GPU via `./v2/ops/deploy.sh`.
- Never train locally on the MacBook.
- Local work is for editing, committing, replay/evaluation, plotting, data rebuilds, and documentation.
- Every experiment trains from scratch. `deploy.sh start` clears the remote `v2/model.pt` before the next run.

## Current System Summary

- Input: 30-bar windows of 47 normalized features.
- Feature groups: 28 price/market-structure, 11 option/Greeks, 8 flow.
- Labels: dual-direction P&L with Tier 3 variable-risk labels.
- Active label grid: `stops=[0.15, 0.2, 0.25, 0.3, 0.4, 0.5]`, `targets=[0.2, 0.3, 0.5, 0.8, 1.2]`, `holds=[30, 60, 120, 240, 390]`
- Current dataset metadata: `236,641` signal bars, `157,232` trade bars, trade window `bar 30-270`
- Training runner: `v2/ops/run_experiment_wf.py`
- Evaluation: 5-fold walk-forward cross-validation over 300 total test days, plus a 20-day shadow holdout
- Artifact rule: only promoted artifacts are eligible for default replay, and they must match the current dataset fingerprint

## Mutable Surface

During the normal experiment loop, only these two files are mutable:

- `v2/train.py`
- `v2/core/policy.py`

Everything else is the frozen evaluation harness unless you are doing explicit infrastructure, data, or documentation work.

## Setup

1. Work from a clean branch.
2. Verify the active dataset:
   - `python -m v2.analysis.contract_drift_audit --data v2/data.pt`
3. Boot the GPU:
   - `./v2/ops/deploy.sh boot`
4. Upload code and the canonical dataset:
   - `./v2/ops/deploy.sh start`

## Experiment Loop

1. Read the latest results, plots, and trade analysis.
2. Write one concrete hypothesis.
3. Edit only `v2/train.py` and/or `v2/core/policy.py`.
4. Run `python -m py_compile` on every changed Python file.
5. Commit the change.
6. Run one experiment:
   - `./v2/ops/deploy.sh run_one exp_NNN`
7. Read the aggregate walk-forward score and the per-fold scores from stdout.
8. Decide:
   - KEEP: `python v2/ops/model_manage.py keep`
   - REVERT: `git checkout HEAD~1 -- v2/train.py v2/core/policy.py`
   - Then: `python v2/ops/model_manage.py revert`
9. Append the result to `v2/results.tsv`.
10. Regenerate visual artifacts:
   - `python -m v2.plot_trades --model v2/model.pt`
   - `python v2/plot_progress.py`
11. Read the loss/trade diagnostics:
   - `python -m v2.analysis.analyze_losses`
12. Update `v2/lab_notebook.md`.
13. Check session limits before the next experiment.

## Artifact Lifecycle

- `run_one` downloads the remote result to `v2/model_candidate.pt`.
- `run_one` also downloads the matching artifact bundle to `v2/artifacts/exp_NNN/`.
- `python v2/ops/model_manage.py keep` promotes the candidate to `v2/model.pt` and `v2/model_best.pt`, and marks the artifact `promoted=true`.
- `python v2/ops/model_manage.py revert` deletes `v2/model_candidate.pt` and marks the artifact `promoted=false`.
- Default replay loads the best promoted artifact that matches the current dataset fingerprint.
- If no compatible promoted artifact exists yet, replay falls back to the raw checkpoint on disk so the harness remains usable between runs.

## Score

The promotion score is:

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

## Walk-Forward Geometry

| Fold | Train Days | Val Days | Test Days |
|------|------------|----------|-----------|
| 0 | 0-673 | 634-673 | 674-733 |
| 1 | 0-733 | 694-733 | 734-793 |
| 2 | 0-793 | 754-793 | 794-853 |
| 3 | 0-853 | 814-853 | 854-913 |
| 4 | 0-913 | 874-913 | 914-973 |

- Total walk-forward test window: 300 days
- Shadow holdout: days 974-993

## Session Limits

| Limit | Threshold | Action |
|------|-----------|--------|
| Experiments | 20 | Stop the session |
| Time | 10 hours | Stop the session |
| No-improve streak | 6 reverts | Stop and rethink |
| Plateau | 4 hours | Stop the session |
| Crash storm | 3 crashes | Stop and fix infrastructure |

## If Stuck

If you hit 3 or more consecutive reverts:

1. Stop changing hyperparameters blindly.
2. Read the trade-level diagnostics:
   - `python -m v2.analysis.analyze_losses`
   - `python -m v2.analysis.analyze_whipsaw`
3. Regenerate plots.
4. Form the next hypothesis from the actual failure pattern.

## Source Of Truth

Use these docs for current behavior:

- [current_state.md](docs/current_state.md)
- [data_contract.md](docs/data_contract.md)
- [feature_schema.md](docs/feature_schema.md)
- [labeling.md](docs/labeling.md)
- [evaluator.md](docs/evaluator.md)
- [how_training_works.md](docs/how_training_works.md)

Treat `v2/docs/audit/` as historical analysis, not current operator protocol.
