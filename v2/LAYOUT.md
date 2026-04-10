# v2 Directory Map

## Research Surface

- `train.py` — model architecture, loss, optimizer, and training loop
- `core/policy.py` — trading policy and replay-time decision ranges

## Frozen Harness

- `replay.py` — replay driver, baselines, artifact loading
- `core/features.py` — feature constants and normalization contract
- `core/metrics.py` — score formula and hard gates
- `core/schema.py` — `TradeIntent` and `SimulatedTrade`
- `core/simulator.py` — trade simulation rules
- `core/walkforward.py` — 5-fold walk-forward geometry
- `ops/run_experiment_wf.py` — canonical GPU experiment runner

## Data

- `data.pt` — active dataset used by training and replay
- `data_sidecars/*.pt` — per-day exact contract identities and P&L labels
- `pipeline/build_v2_dataset.py` — rebuild features and labels from raw caches
- `pipeline/relabel_tier3.py` — upgrade to variable-risk labels
- `pipeline/compute_features.py` — raw feature computation

## Models And Artifacts

- `model.pt` — current promoted local model
- `model_best.pt` — best promoted checkpoint on disk
- `model_candidate.pt` — latest downloaded experiment result awaiting keep/revert
- `artifacts/` — experiment bundles with checkpoint, policy snapshot, and manifest

## Analysis And Output

- `analysis/harness_eval.py` — pre-deploy harness regression suite
- `analysis/analyze_losses.py` — trade-level post-mortem
- `analysis/analyze_whipsaw.py` — losing-day feature comparison
- `plot_trades.py` — writes `output/trades.html` and `output/equity.html`
- `plot_progress.py` — writes `output/progress.png`
- `results.tsv` — experiment result log (exact-chain official runs)
- `lab_notebook.md` — research notebook

## Operations

- `ops/deploy.sh` — boot/start/run_one/run_screen/status/stop for Akash GPU
- `ops/model_manage.py` — keep/revert the downloaded candidate model
- `ops/artifact.py` — artifact save/load/lineage helpers

## Documentation

- `program.md` — definitive operator protocol
- `COMMANDS.md` — command phrases the agent should honor
- `HANDOFF.md` — current state snapshot
- `FRESH_SESSION_HANDOFF.md` — pointer to read order for new sessions
- `docs/README.md` — live documentation index
- `docs/` — contracts, evaluation docs, system documentation

## Where To Look When X Fails

- **Dataset / label issues** — `pipeline/build_v2_dataset.py`, `core/chain_data.py`, `docs/data_contract.md`, `docs/labeling.md`
- **Training / model issues** — `train.py`, `docs/how_training_works.md`
- **Replay / score issues** — `replay.py`, `core/metrics.py`, `core/walkforward.py`, `docs/evaluator.md`, `docs/baselines.md`
- **Protocol / operator issues** — `program.md`, `AGENTS.md`, `CLAUDE.md`, `HANDOFF.md`
