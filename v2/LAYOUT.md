# v2 Directory Map

## Research Surface

- `train.py` -- model architecture, loss, optimizer, and training loop
- `core/policy.py` -- trading policy and replay-time decision ranges

## Frozen Harness

- `replay.py` -- replay driver, baselines, artifact loading
- `core/features.py` -- feature constants and normalization contract
- `core/metrics.py` -- score formula and hard gates
- `core/schema.py` -- `TradeIntent` and `SimulatedTrade`
- `core/simulator.py` -- trade simulation rules
- `core/walkforward.py` -- 5-fold walk-forward geometry
- `ops/run_experiment_wf.py` -- canonical GPU experiment runner

## Data

- `data.pt` -- active repaired dataset used by training and replay
- `data_harness_repair.pt` -- backup copy of the same repaired dataset
- `pipeline/build_v2_dataset.py` -- rebuild features and fixed-risk labels from raw caches
- `pipeline/relabel_tier3.py` -- upgrade to Tier 3 variable-risk labels
- `pipeline/compute_features.py` -- raw feature computation

## Models And Artifacts

- `model.pt` -- current promoted local model
- `model_best.pt` -- best promoted checkpoint on disk
- `model_candidate.pt` -- latest downloaded experiment result awaiting keep/revert
- `artifacts/` -- experiment bundles with checkpoint, policy snapshot, and manifest

## Analysis And Output

- `analysis/contract_drift_audit.py` -- honesty audit for the dataset contract
- `analysis/analyze_losses.py` -- trade-level post-mortem
- `analysis/analyze_whipsaw.py` -- losing-day feature comparison
- `plot_trades.py` -- writes `output/trades.html` and `output/equity.html`
- `plot_progress.py` -- writes `output/progress.png`
- `results.tsv` -- experiment result log
- `lab_notebook.md` -- human-readable research notebook

## Operations

- `ops/deploy.sh` -- boot/start/run_one/status/stop for the Akash GPU workflow
- `ops/model_manage.py` -- keep/revert the downloaded candidate model
- `ops/artifact.py` -- artifact save/load/lineage helpers

## Documentation

- `program.md` -- definitive operator protocol
- `COMMANDS.md` -- command phrases the agent should honor
- `HANDOFF.md` -- current state snapshot
- `docs/current_state.md` -- detailed source-of-truth system overview
- `docs/` -- contracts, evaluation docs, historical audits, and domain notes
