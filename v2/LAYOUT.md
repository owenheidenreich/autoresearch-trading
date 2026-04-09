# v2/ Directory Map

## Training (mutable)
- `train.py` -- model architecture and training loop
- `core/policy.py` -- trading policy parameters (gate, stop, target, hold)
- `model.pt` -- current best model checkpoint

## Evaluation (immutable harness)
- `replay.py` -- validation/replay simulator
- `core/` -- schema, features, labels, metrics, simulator, candidates, walkforward

## Output (generated every experiment)
- `output/trades.html` -- interactive trade chart (Plotly)
- `output/equity.html` -- equity curve (Plotly)
- `output/trades.csv` -- trade log export
- `output/progress.png` -- experiment score history (Karpathy-style)
- `results.tsv` -- experiment results table

## Analysis
- `analysis/analyze_losses.py` -- losing trade post-mortem
- `analysis/analyze_whipsaw.py` -- whipsaw day deep-dive
- `plot_trades.py` -- generates output/trades.html + equity.html + trades.csv
- `plot_progress.py` -- generates output/progress.png from results.tsv

## Data
- `data.pt` -- active dataset (referenced everywhere, do not move)
- `data/` -- backup and alternate dataset versions
- `pipeline/` -- dataset build scripts (build_dataset, compute_features, extract_raw, download_wide_grid)

## Operations
- `ops/` -- deploy.sh, run_experiment, inner_loop, sweeps, preflight, monitor

## Live Trading
- `live/` -- IBKR integration (service, decision, execution, market)

## Research and Docs
- `research/` -- QuantConnect validation scripts
- `docs/` -- specifications and domain knowledge
- `artifacts/` -- experiment bundles (exp_001 through exp_060+)
- `lab_notebook.md` -- experiment log
- `program.md` -- master protocol
