# Handoff: Current v2 State

Read this file, then [docs/founder_intent.md](docs/founder_intent.md), then [program.md](program.md), then [docs/current_state.md](docs/current_state.md).

For domain knowledge: [docs/domain/](docs/domain/) contains 0DTE options knowledge and practitioner trading journals. Read these to understand the instrument before making architecture decisions.

## FRESH SLATE — Scoring & Data Reset (2026-04-13)

All prior scores in results.tsv are invalidated. The scoring formula, feature set, and simulation model have been overhauled. Comparisons to exp_074–exp_146 are not meaningful under the new system. The next experiment (exp_147) starts the new era.

### What Changed

1. **Scoring v3.0**: `(0.5*sortino + 0.5*PF) * PDR * dd_mult` — rewards profit factor alongside sortino. DD gate raised 20%→25%, penalty-free zone 8%→12%, sortino cap 6→10.
2. **Contract features 15→19**: Added vega, charm (dDelta/dTime), contract price momentum (5-bar, 10-bar mid change).
3. **Context features 47→49**: Added aggregate_charm (chain-wide dealer hedging signal), vwap_slope (VWAP direction).
4. **Simulation realism**: Spread widening on fast moves (bar_range > 2x avg → up to 2x spread cost).
5. **Data rebuilt**: Sidecars and data.pt rebuilt with all new features and stop_pct=0.35 in labels.

## Previous Best (old scoring, for reference only)

exp_146 under old scoring: 0.807 (old formula). Not comparable to new scores.

## Live Code

- `v2/train.py` — TradingModel: 49-feature encoder + 19-feature contract_proj + call/put score heads + put_bias + no_trade_head
- `v2/core/policy.py` — DecisionPolicy: stop=35%, target=50%, hold=120, trailing exit, breakeven=0.15, cooldown=3, window bars 60-105, extra_trailing_tiers=((0.25, 0.08),)
- `v2/core/simulator.py` — simulate_trade(), TRAILING_TIERS, _build_trailing_tiers(), spread widening
- `v2/core/metrics.py` — compute_score() v3.0 composite PF/sortino
- `v2/replay.py` — replay_validation(), baselines, traces

## Dataset

- `v2/data.pt` — v4_exact_chain, 986 days, 49 context features, 19 contract features
- `v2/data_sidecars/*.pt` — per-day contract snapshots and labels
- Labels computed with: stop=35%, target=50%, hold=120, TRAILING exit, breakeven=0.15, extra_tiers=((0.25, 0.08),)
- Trade window in labels: bars 60-105
- New contract features: vega, charm, mid_chg_5, mid_chg_10
- New context features: aggregate_charm, vwap_slope

## What To Trust

- `v2/data.pt` and `v2/data_sidecars/` as the canonical dataset (rebuilt 2026-04-13)
- `v2/results.tsv` — empty, fresh start
- `v2/lab_notebook.md` as the experiment log (historical entries are context, not baselines)
- `v2/models/model.pt` — stale from exp_146 era, will be replaced by exp_147

## What Not To Trust

- Any score from exp_074–exp_146 — computed under old scoring formula
- `v2/models/model.pt` — trained on old 15-feature contracts, will fail on 19-feature data
- `v2/state/` files — reset to clean slate
- Archive docs that reference specific feature priorities or model changes

## Abandoned Approaches (still valid under new system)

- Hierarchical direction head (exp_100-103): direction didn't decompose at 52% accuracy. **May revisit** — the new feature set (charm, momentum) may help.
- SOFT_TEMP=0.05 (exp_141): call selection collapsed. **Sweep SOFT_TEMP** is planned for Wave 2.
- SIDE_SEL_W=0.30 (exp_145): screening fooled by fold-4 bias. Not re-attempted yet.

## Next Steps (Wave 2 — GPU experiments)

1. **exp_147**: Train from scratch with enriched features (19 contract, 49 context). Establishes new baseline under scoring v3.0.
2. **Directional prediction head**: Auxiliary loss that teaches the model to predict SPX direction/magnitude. The core architecture change.
3. **SOFT_TEMP sweep**: 0.10 vs 0.25 vs 0.50 — stop hyper-focusing on exact oracle strike.
4. **Contract cross-attention**: Let contracts see each other (chain-level patterns).
5. **LOOKBACK 30→60**: See the full opening range.

## Data Limitations (permanent)

- **No open interest**: Polygon minute_aggs flat files don't include OI
- **No bid/ask**: Only OHLC + volume + transactions per contract per bar
