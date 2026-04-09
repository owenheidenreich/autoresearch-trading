# v2 Commands

When the human says one of these, do it. Read [program.md](program.md) for the full protocol.

All training runs are remote on the Akash H100. Local commands are for replay, analysis, plotting, data rebuilds, commits, and docs.

## Research

- `begin experiment loop`
  Use the full loop from `v2/program.md`: hypothesis, edit `train.py` and/or `core/policy.py`, commit, `deploy.sh run_one exp_NNN`, then keep or revert.

## Evaluation

- `evaluate model`
  Run:
  `python -m v2.replay --model v2/model.pt --data v2/data.pt --mask promote`

- `evaluate promoted artifact`
  Run:
  `python -m v2.replay --data v2/data.pt --mask promote`

- `evaluate on shadow`
  Run:
  `python -m v2.replay --data v2/data.pt --mask shadow`

- `analyze trades`
  Run:
  `python -m v2.analysis.analyze_losses`

- `audit dataset`
  Run:
  `python -m v2.analysis.contract_drift_audit --data v2/data.pt`

## Data

- `rebuild dataset`
  Rebuild the canonical repaired dataset from raw caches:
  1. `python -m v2.pipeline.build_v2_dataset --output v2/data.pt`
  2. `python -m v2.pipeline.relabel_tier3 --data v2/data.pt --tier 3`
  3. `python -m v2.analysis.contract_drift_audit --data v2/data.pt`

- `relabel dataset`
  Run:
  `python -m v2.pipeline.relabel_tier3 --data v2/data.pt --tier 3`

## GPU

- `boot gpu`
  Run:
  `./v2/ops/deploy.sh boot`

- `start gpu`
  Run:
  `./v2/ops/deploy.sh start`

- `run experiment exp_NNN`
  Run:
  `./v2/ops/deploy.sh run_one exp_NNN`

- `stop gpu`
  Run:
  `./v2/ops/deploy.sh stop`

- `gpu status`
  Run:
  `./v2/ops/deploy.sh status`

## Monitoring

- `plot progress`
  Run:
  `python v2/plot_progress.py`

- `plot trades`
  Run:
  `python -m v2.plot_trades --model v2/model.pt`

## Live

These are not implemented yet:

- `begin shadow session`
- `begin paper session`
- `kill switch`
