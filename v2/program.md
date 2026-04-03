# ART² v2 Program

Single source of truth for v2. If anything conflicts with this file, this file wins.

## Mission

Build a model that profitably trades SPX 0DTE long options.
Same mission as v1. v2 is a structural reset, not a strategy change.

## Central Contract: TradeIntent

Everything revolves around one frozen dataclass. See [docs/v2/contracts.md](../docs/v2/contracts.md).

- Replay scores it (v2/core/simulator.py + v2/core/metrics.py)
- Live executes it (v2/live/execution.py)
- Training learns to emit it (v2/train.py via v2/core/labels.py)

## What Is Fixed

- **Input contract:** 39 features per [docs/v2/feature_schema.md](../docs/v2/feature_schema.md)
- **Output contract:** TradeIntent per [docs/v2/contracts.md](../docs/v2/contracts.md)
- **Evaluator rules:** fill model, spreads, stops, scoring per [docs/v2/evaluator.md](../docs/v2/evaluator.md)
- **Baselines:** GPU gates per [docs/v2/baselines.md](../docs/v2/baselines.md)

## What Is NOT Fixed

- Model architecture (the thing being researched)
- Loss formulation (follows from label scheme)
- Hyperparameters (tuned by autoresearch loop)

## Module Map

| Module | Purpose | v1 Origin |
|--------|---------|-----------|
| v2/core/schema.py | TradeIntent contract | training/live/contracts.py |
| v2/core/features.py | 39-feature engineering | training/prepare.py |
| v2/core/labels.py | Oracle labeler | training/prepare.py |
| v2/core/candidates.py | Dynamic candidate generation | training/trading_rules.py |
| v2/core/simulator.py | Trade simulation | training/replay.py |
| v2/core/metrics.py | Replay scoring | training/replay.py + train.py |
| v2/live/market.py | IBKR market data | training/live/features.py + service.py |
| v2/live/decision.py | Model -> TradeIntent | training/live/decision.py |
| v2/live/execution.py | IBKR orders | training/live/execution.py + resolver.py |
| v2/live/service.py | Main trading loop | training/live/service.py |
| v2/ops/inner_loop.py | Experiment orchestrator | tools/inner_loop.py + run_loop.py |
| v2/pipeline/build_dataset.py | Dataset pipeline | training/prepare.py |

## Current Phase: 0 (Skeleton)

No v2 behavior is implemented. All modules are stubs with TODOs.
The v1 system at training/ remains the active system.
See [docs/v2/migration.md](../docs/v2/migration.md) for the cutover plan.

## v1 Reference

The v1 system is archived at `legacy_v1/` (frozen, never imported).
Active v1 code remains at `training/`, `tools/`, `infra/`.
v1 program: `training/program.md`. v1 governance: `training/principles.md`.

## Key Design Documents

| Document | Purpose |
|----------|---------|
| [contracts.md](../docs/v2/contracts.md) | TradeIntent specification |
| [evaluator.md](../docs/v2/evaluator.md) | Replay rules, promotion score formula |
| [baselines.md](../docs/v2/baselines.md) | GPU spend gates, baseline definitions |
| [labeling.md](../docs/v2/labeling.md) | Oracle label generation |
| [execution.md](../docs/v2/execution.md) | IBKR order state machine |
| [feature_schema.md](../docs/v2/feature_schema.md) | 39 features, normalization |
| [data_contract.md](../docs/v2/data_contract.md) | Dataset format, fingerprinting |
| [archive_map.md](../docs/v2/archive_map.md) | v1 -> v2 file disposition |
| [migration.md](../docs/v2/migration.md) | Cutover sequence, coexistence rules |
