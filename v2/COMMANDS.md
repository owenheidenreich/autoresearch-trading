# v2 Commands

When the human says one of these, do it. Read `v2/program.md` for full protocol.

All training runs on Akash H100 GPU, never locally. Local machine is for editing code, committing, reading results, and running replay/evaluation only.

## Research

- **begin experiment loop** -- Boot Akash GPU, then Claude drives the loop: edit train.py, commit, `deploy.sh run_one exp_NNN`, read score, keep/revert, repeat. One command per experiment.
- **fresh start** -- `rm -f v2/model.pt v2/model.pt.best`. Use after architecture changes.

## Evaluation (runs locally)

- **evaluate model** -- `python -m v2.replay --mask promote`. Loads best model from artifact system. Score on held-out days with baselines.
- **evaluate on shadow** -- Same but `--mask shadow`. Live-readiness check only.
- **analyze trades** -- `python v2/analyze_losses.py`. Loads best model from artifact system, inspects which trades won/lost and why.

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
