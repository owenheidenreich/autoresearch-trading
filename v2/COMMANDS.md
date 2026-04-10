# v2 Commands

When the human says one of these, do it. Read [program.md](program.md) for the full protocol.

All training runs are remote on the Akash H100. Local commands are for replay, analysis, plotting, data rebuilds, commits, and docs.

## Research

- `begin experiment loop`
  Use the full loop from `v2/program.md`: hypothesis, edit `train.py` and/or `core/policy.py`, commit, `deploy.sh run_one exp_NNN`, then keep or revert.

## Evaluation

- `evaluate model`
  Run:
  `python3 -m v2.replay --model v2/model.pt --data v2/data.pt --mask promote`

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
  `python3 v2/plot_progress.py`

- `plot trades`
  Run:
  `python3 -m v2.plot_trades --model v2/model.pt`
