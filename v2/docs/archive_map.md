# v2 Archive Map

## Purpose

Maps every v1 source file to its v2 disposition: ported, split, superseded, or dropped.

---

## Source File Disposition

### training/

| v1 File | v2 Destination | Disposition |
|---------|---------------|-------------|
| train.py (25KB) | v2/train.py | Rewritten. Mutable research file. New loss from trade-outcome labels. |
| best_train.py (25KB) | v2/best_train.py | Auto-synced copy of v2/train.py on KEEP. Same role. |
| prepare.py (178KB) | Split into 3 modules | See below. |
| replay.py (145KB) | Split into 2 modules | See below. |
| trading_rules.py (11KB) | v2/core/candidates.py | Replaced. Dynamic candidate scoring replaces hardcoded rules. |
| run_loop.py (34KB) | v2/ops/inner_loop.py | Merged with tools/inner_loop.py. Safety validation + experiment orchestration unified. |
| program.md (7.7KB) | v2/program.md | Rewritten for v2 contracts and module map. |
| principles.md (12KB) | docs/v2/goal.md + docs/v2/evaluator.md | Split. Mission/integrity -> goal.md. Migration roadmap -> evaluator.md. |
| lab_notebook.md (1.6KB) | v2/lab_notebook.md | Fresh start. v1 entries archived in legacy_v1/. |

### prepare.py Split

| v1 Section | v2 Module | What Moves |
|------------|-----------|------------|
| FEATURE_NAMES, compute_features(), _FEAT_IDX | v2/core/features.py | Feature computation logic |
| normalize_features(), _rolling_zscore, _per_day_zscore | v2/core/features.py | Normalization |
| compute_v18_labels(), compute_prediction_labels_from_prices() | v2/core/labels.py | Label generation |
| compute_dynamic_pnl(), stopped P&L simulation | v2/core/labels.py | Oracle P&L computation |
| download_*, cache management, make_dataloader | v2/pipeline/build_dataset.py | Data I/O |
| Constants (BARS_PER_DAY, NUM_FEATURES, etc.) | v2/core/features.py | Shared constants |

### replay.py Split

| v1 Section | v2 Module | What Moves |
|------------|-----------|------------|
| Trade simulation loop, position management | v2/core/simulator.py | Execution engine |
| compute_adaptive_spread_bps() | v2/core/simulator.py | Cost model |
| Score computation, metrics aggregation | v2/core/metrics.py | Scoring |
| Ledger writing, CSV/JSON output | v2/core/metrics.py | Trade logging |
| load_model(), CLI entry point | v2/replay.py (root) | Entry point |

### training/live/

| v1 File | v2 Destination | Disposition |
|---------|---------------|-------------|
| service.py | v2/live/service.py | Ported. Same bar-by-bar loop, emits TradeIntent instead of DecisionIntent. |
| decision.py | v2/live/decision.py | Ported. Model -> TradeIntent (replaces InferenceResult -> DecisionIntent). |
| execution.py | v2/live/execution.py | Ported. OCOExecutionEngine receives TradeIntent, resolves to IBKR contract. |
| features.py | v2/live/market.py | Ported. FiveSecondMinuteAggregator + LiveFeatureEngine unified. |
| resolver.py | v2/live/execution.py | Merged into execution. Contract resolution is part of order placement. |
| context.py | v2/live/market.py | Merged. Context bootstrap is part of market data setup. |
| contracts.py | v2/core/schema.py | Replaced. DecisionIntent/ExecutionState -> TradeIntent. FeatureContractVersion preserved. |
| entitlements.py | v2/live/service.py | Merged. Entitlement check is part of service startup. |

### tools/

| v1 File | v2 Destination | Disposition |
|---------|---------------|-------------|
| inner_loop.py | v2/ops/inner_loop.py | Ported + merged with run_loop.py. |
| daily_pipeline.py | v2/ops/ (TBD) | Ported after core is stable. |
| paper_live.py | v2/live/service.py CLI | Merged. Paper trading CLI becomes service.py entry point. |
| monitor.py | v2/ops/ (TBD) | Ported after core is stable. |
| export_trades.py | Dropped | Replaced by standardized trade log format. |
| export_trades_csv.py | Dropped | Replaced by standardized trade log format. |
| ib_account_snapshot.py | Dropped | Diagnostic only, not part of core loop. |
| ib_entitlements.py | Dropped | Diagnostic only. |
| ibkr_analyze.py | Dropped | Replaced by v2/core/metrics.py analysis. |
| run_tournament.sh | Dropped | Feature tournament was v1-specific. |

### infra/

| v1 File | v2 Destination | Disposition |
|---------|---------------|-------------|
| deploy.sh | v2/ops/deploy.sh | Ported. |
| preflight.py | v2/ops/inner_loop.py | Merged into pre-GPU checks. |
| start_loop.sh | v2/ops/deploy.sh | Merged. |
| watchdog.sh | v2/ops/ (TBD) | Ported after deployment is stable. |
| deploy-autoresearch.yaml | v2/ops/ | Ported. |
| *.plist (LaunchD) | v2/ops/ | Ported after live trading is stable. |

### tests/

| v1 File | v2 Destination | Disposition |
|---------|---------------|-------------|
| test_version_consistency.py | tests/v2/ (rewritten) | v2 version consistency against new contracts. |
| test_training_v5.py | tests/v2/ (rewritten) | v2 training tests against new loss/score. |
| test_live_decision.py | tests/v2/ (ported) | TradeIntent output tests. |
| test_live_execution.py | tests/v2/ (ported) | IBKR execution tests. |
| test_live_contracts.py | tests/v2/ (ported) | FeatureContractVersion tests (preserved). |
| test_live_resolver.py | tests/v2/ (merged) | Resolver tests merged into execution tests. |
| test_replay_battery.py | tests/v2/ (rewritten) | Simulator battery tests. |
| test_replay_ledger_qa.py | tests/v2/ (rewritten) | Trade log QA tests. |
| test_replay_loader.py | tests/v2/ (rewritten) | Model loading tests. |
| test_incremental_cache.py | tests/v2/ (ported) | Cache tests still relevant. |
| test_observability_upgrade.py | tests/v2/ (rewritten) | Anomaly detection tests. |
| test_live_feature_parity_report.py | tests/v2/ (ported) | Feature parity still critical. |
| test_live_order_parity_report.py | tests/v2/ (ported) | Order parity still critical. |
| test_ibkr_mock_training.py | tests/v2/ (ported) | Mock training tests. |

---

## DecisionIntent -> TradeIntent Field Map

| v1 DecisionIntent | v2 TradeIntent | Notes |
|-------------------|----------------|-------|
| action: int | trade: bool + strike/right | Explicit identity replaces enum |
| contract: Any | expiry/strike/right | Serializable, broker-agnostic |
| qty: int | qty: int | Unchanged |
| entry_order: str | order_style: str | Renamed |
| stop_price: float | stop_price: float | Unchanged |
| take_profit_price: float | take_profit_price: float | Unchanged |
| confidence: float | confidence: float | Unchanged |
| reason_codes: list | reason_codes: tuple | Immutable |
| entry_limit_price: float | limit_price: float | Renamed |
| reference_price: float | entry_ref_price: float | Renamed |
| intent_id: str | intent_id: str | Unchanged |
| decision_id: str | (dropped) | Redundant with intent_id |
| metadata: dict | (dropped) | Structured fields replace opaque dict |
| (missing) | tif: str | New: time-in-force |
| (missing) | max_hold_bars: int | New: explicit hold limit |
| (missing) | exit_policy: str | New: exit strategy |
| (missing) | bid/ask_at_decision | New: quote provenance |
| (missing) | underlying_price | New: SPX reference |
| (missing) | policy_version: str | New: versioning |
| (missing) | bar_index: int | New: time context |
| (missing) | timestamp: str | New: ISO 8601 |

---

## Feature Disposition (All 39)

All 39 features are preserved as-is in v2. Feature changes are a future
research question, not a structural decision.

See feature_schema.md for the complete table.

---

## Binary Artifacts

| Artifact | Location | v2 Disposition |
|----------|----------|---------------|
| data.pt (352MB) | training/data.pt | v1 artifact. v2 writes to separate path. Never overwritten. |
| best_model.pt | training/best_model.pt | v1 artifact. v2 writes to v2/best_model.pt. |
| results/ (8.9GB) | results/ | v1 run outputs. v2 writes to results/v2/. |
| backtest_output/ (3.5GB) | backtest_output/ | v1 backtests. v2 uses results/v2/ for replay output. |
| archive/ (867MB) | archive/ | Pre-v1 legacy. Untouched. |
