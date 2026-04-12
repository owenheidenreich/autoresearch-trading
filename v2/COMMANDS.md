# v2 Commands

When the human says one of these, do it. Read [program.md](program.md) for the full protocol.

All training runs are remote on the Akash H100. Local commands are for replay, analysis, plotting, data rebuilds, commits, and docs.

## Research

- `run screening experiment exp_NNN`
  Screen first (1-fold, no artifacts):
  `./v2/ops/deploy.sh run_screen exp_NNN`

- `run official experiment exp_NNN`
  Official 5-fold scored run:
  `./v2/ops/deploy.sh run_one exp_NNN`

- `begin experiment loop`
  Use the full loop from `v2/program.md`: hypothesis, edit `train.py` and/or `core/policy.py`, commit, screen first, then if screening passes run official, then keep or revert.

## Evaluation

- `evaluate model`
  Run:
  `python3 -m v2.replay --data v2/data.pt --mask promote`

- `evaluate promoted artifact`
  Run:
  `python3 -m v2.replay --data v2/data.pt --mask promote`

- `evaluate on shadow`
  Run:
  `python3 -m v2.replay --data v2/data.pt --mask shadow`

- `analyze trades`
  Run:
  `python3 -m v2.analysis.analyze_losses`

- `audit dataset`
  Run:
  `python3 -m v2.analysis.harness_eval --data v2/data.pt`

## Decision Traces

- `trace model` or `decision trace`
  Generate per-bar decision trace for the candidate model (mandatory before keep/revert):
  `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`
  Output: `v2/artifacts/replay_traces.csv` + printed summary with gate accuracy, selection accuracy, P&L gap, failure modes.

- `trace artifact` or `trace exp_NNN`
  Trace a specific artifact:
  `python3 -m v2.replay --artifact v2/artifacts/exp_NNN --data v2/data.pt --mask promote --traces`

## Data Integrity

- `validate data` or `check data`
  Run the full data integrity pipeline (manifest, features, sidecars):
  `python3 -m v2.core.data_integrity --data v2/data.pt`

## Data

- `download full chain`
  Run:
  `python3 -m v2.pipeline.download_full_chain`

- `rebuild dataset`
  Rebuild the canonical exact-chain dataset from raw caches:
  1. `python3 -m v2.pipeline.build_v2_dataset --output v2/data.pt --sidecar-dir v2/data_sidecars`
  2. `python3 -m v2.analysis.harness_eval --data v2/data.pt --build`
  3. `python3 -m v2.analysis.harness_eval --data v2/data.pt`

## GPU

- `boot gpu`
  Run:
  `./v2/ops/deploy.sh boot`

- `start gpu`
  Run:
  `./v2/ops/deploy.sh start`

- `stop gpu`
  Run:
  `./v2/ops/deploy.sh stop`

- `gpu status`
  Run:
  `./v2/ops/deploy.sh status`

## Monitoring

- `project status`
  Run:
  `python3 -m v2.ops.status_report`

- `plot progress`
  Run:
  `python3 v2/plot_progress.py`

- `plot trades`
  Run:
  `python3 -m v2.plot_trades`
