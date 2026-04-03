# v2 Commands

When the human says one of these, do it. Read `v2/program.md` for full protocol.

## Research

- **begin experiment loop** -- Start autonomous ART² cycle. Setup, baseline, then loop until session limit. Keep/revert is automatic. Do not ask for permission between experiments.
- **fresh start** -- Reset all state (`rm -f v2/model.pt v2/.best_score v2/.inner_loop_state.json v2/results.tsv`). Use after structural changes.

## Evaluation

- **evaluate model** -- `python -m v2.replay --model v2/model.pt --mask promote`. Score on held-out days with baselines.
- **evaluate on shadow** -- Same but `--mask shadow`. Live-readiness check only.
- **analyze trades** -- Load replay trades, inspect which trades won/lost and why.

## Data

- **rebuild dataset** -- `python -m v2.pipeline.build_dataset --tier 3`. Tier 3 labels, 4-way split. 30-60 min.

## Monitoring

- **status** -- `python v2/ops/monitor.py`. Session state, scores, streaks.

## Live (not yet implemented)

- **begin shadow session** -- Run model on live data, no orders. Verify intent parity.
- **begin paper session** -- Real IBKR paper orders, full RTH session.
- **kill switch** -- Emergency stop, close all positions.
