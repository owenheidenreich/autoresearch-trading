# Legacy v1 Snapshot

Read-only archive of the v1 autoresearch-trading system (v18 5-head TradingModel).

## Rules

- **DO NOT import from this directory.** New code must never depend on legacy_v1/.
- **DO NOT modify files here.** This is a frozen reference.
- This snapshot contains code only. Binary artifacts (data.pt, model weights,
  results/, backtest_output/) are not included.

## What's Here

- training/ -- train.py (v18 5-head), prepare.py (39 features), replay.py, trading_rules.py,
  run_loop.py, best_train.py, program.md, principles.md, lab_notebook.md
- training/live/ -- IBKR live trading: service.py, decision.py, execution.py,
  features.py, resolver.py, context.py, contracts.py, entitlements.py
- tools/ -- inner_loop.py (experiment orchestrator), daily_pipeline.py, paper_live.py,
  monitor.py, export tools, IBKR diagnostic tools
- infra/ -- deploy.sh (Akash GPU), preflight.py, start_loop.sh, watchdog.sh,
  deploy-autoresearch.yaml, LaunchD plists
- tests/ -- 14 test files covering version consistency, replay, live execution,
  live contracts, live decisions, observability
- docs/domain/ -- 5 trading domain knowledge documents
- docs/journal/ -- v17/v18 handoff documents

## Key Concepts in v1

- 5-head model: market prediction, entry gate, risk params, exit signal, 6-class direction
- 39 features: price returns, volume, gamma, volatility, VWAP, session, trend, VIX, Greeks, etc.
- Fixed 6-class action space: call/put x ATM/OTM5/OTM10
- Score formula: direction_accuracy * (1 + max(0, rank_correlation))
- Proxy labels: MFE/MAE entry gate, ATR risk targets, SPX-based exit
- Only direction labels use real stopped option P&L
- DecisionIntent -> ExecutionState lifecycle for IBKR orders
- Autoresearch loop: mutate train.py -> train -> score -> keep/revert
