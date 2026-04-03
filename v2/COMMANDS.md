# v2 Commands

When the human says one of these, do it. Read `v2/program.md` for full protocol.

All training runs on Akash H100 GPU, never locally. Local machine is for editing code, committing, reading results, and running replay/evaluation only.

## Research

- **begin experiment loop** -- Boot Akash GPU, setup, baseline, then loop until session limit. Keep/revert is automatic. Do not ask for permission between experiments.
- **fresh start** -- Reset all state (`rm -f v2/model.pt v2/.best_score v2/.inner_loop_state.json v2/results.tsv`). Use after structural changes.

## Evaluation (runs locally)

- **evaluate model** -- `python -m v2.replay --model v2/model.pt --mask promote`. Score on held-out days with baselines.
- **evaluate on shadow** -- Same but `--mask shadow`. Live-readiness check only.
- **analyze trades** -- Load replay trades, inspect which trades won/lost and why.

## Data (runs locally)

- **rebuild dataset** -- `python -m v2.pipeline.build_dataset --tier 3`. Tier 3 labels, 4-way split. 30-60 min.

## GPU

- **boot gpu** -- `./v2/ops/deploy.sh boot`. Start Akash H100 instance.
- **stop gpu** -- `./v2/ops/deploy.sh stop`. Tear down Akash deployment.

## Monitoring

- **status** -- `python v2/ops/monitor.py`. Session state, scores, streaks.

## Live (not yet implemented)

- **begin shadow session** -- Run model on live data, no orders. Verify intent parity.
- **begin paper session** -- Real IBKR paper orders, full RTH session.
- **kill switch** -- Emergency stop, close all positions.
