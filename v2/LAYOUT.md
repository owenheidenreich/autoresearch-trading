# v2 Directory Map

## Live Research Surface

- `train.py` — model architecture, loss, optimizer, and training loop
- `core/policy.py` — trading policy and replay-time decision ranges

## Frozen Harness

- `replay.py` — replay driver, baselines, artifact loading, decision trace collection
- `core/features.py` — feature constants and normalization contract
- `core/metrics.py` — score formula and hard gates
- `core/schema.py` — `TradeIntent` and `SimulatedTrade`
- `core/simulator.py` — trade simulation rules
- `core/walkforward.py` — 5-fold walk-forward geometry
- `core/decision_trace.py` — per-bar decision trace dataclass, save/load, summary analysis
- `core/data_integrity.py` — data quality validation (manifest, features, sidecars)
- `ops/run_experiment_wf.py` — canonical GPU experiment runner

## Data

- `data.pt` — active dataset used by training and replay
- `data_sidecars/*.pt` — per-day exact contract identities and P&L labels
- `pipeline/build_v2_dataset.py` — rebuild features and labels from raw caches
- `pipeline/download_full_chain.py` — canonical raw full-chain downloader
- `pipeline/compute_features.py` — raw feature computation

## Models And Artifacts

- `models/model.pt` — current promoted local model
- `models/model_best.pt` — best promoted checkpoint on disk
- `models/model_candidate.pt` — latest downloaded experiment result awaiting keep/revert
- `artifacts/` — experiment bundles with checkpoint, policy snapshot, and manifest
  Treat this as operational output, not default reading context.

## Runtime State

- `state/baseline_cache.json` — cached replay baseline outputs
- `state/best_score.txt` — synced best score from remote loop state
- `state/inner_loop_state.json` — synced remote loop/session state

## Analysis And Output

- `analysis/harness_eval.py` — pre-deploy harness regression suite
- `analysis/analyze_losses.py` — trade-level post-mortem
- `plot_trades.py` — writes `output/trades.html` and `output/equity.html`
- `plot_progress.py` — writes `output/progress.png`
- `results.tsv` — experiment result log (exact-chain official runs)
- `lab_notebook.md` — research notebook

## Operations

- `ops/deploy.sh` — boot/start/run_one/run_screen/status/stop for Akash GPU
- `ops/pre_run_gate.py` — local integrity gate before GPU spend
- `ops/status_report.py` — one-command live health and project-truth summary
- `ops/model_manage.py` — keep/revert the downloaded candidate model
- `ops/artifact.py` — artifact save/load/lineage helpers

## Documentation

- `program.md` — definitive operator protocol
- `COMMANDS.md` — command phrases the agent should honor
- `HANDOFF.md` — current state snapshot
- `docs/README.md` — live documentation index
- `docs/founder_intent.md` — founder voice, standards, and anti-goals
- `docs/decision_log.md` — stable decisions
- `docs/open_questions.md` — current unresolved questions
- `docs/` — contracts, evaluation docs, system documentation

## Archived Surfaces

These were moved out of the live `v2/` path to reduce context drift:

- `archive/v2_historical/live/` — paper/live trading stubs
- `archive/v2_historical/analysis/` — one-off repair diagnostics and stale analysis scripts
- `archive/v2_historical/pipeline/` — superseded dataset builders and raw extraction helpers
- `archive/v2_historical/docs/` — redundant or historical docs
- `archive/v2_historical/data/` — backup manifests and pre-reset datasets

## Where To Look When X Fails

- **Dataset / label issues** — `pipeline/build_v2_dataset.py`, `core/chain_data.py`, `docs/data_contract.md`, `docs/labeling.md`
- **Data quality issues** — `core/data_integrity.py`, `python3 -m v2.core.data_integrity`
- **Training / model issues** — `train.py`, `docs/how_training_works.md`
- **Replay / score issues** — `replay.py`, `core/metrics.py`, `core/walkforward.py`, `docs/evaluator.md`
- **Decision trace / model behavior** — `core/decision_trace.py`, `replay.py --traces`
- **Protocol / operator issues** — `program.md`, `AGENTS.md`, `CLAUDE.md`, `HANDOFF.md`, `ops/pre_run_gate.py`
