# Agent Directives

> **Protocol101 Path-D status (2026-08-01):** Read
> [`v4/docs/protocol101/PATH_D_CURRENT_STATE.md`](v4/docs/protocol101/PATH_D_CURRENT_STATE.md)
> before Protocol101 model, data, training, validation, or paper work. Path D is
> in design repair; Phase-1 training/backtest and learned-trader paper readiness
> are not authorized.

## Current Bootstrap (2026-05-26)

This file is the agent front door. Older v2-first instructions were stale and are superseded by this section.

1. Read [PROJECT_SECTION_AND_FEATURE_MAP.md](PROJECT_SECTION_AND_FEATURE_MAP.md) first for the current repo map.
2. Read [docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md](docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md) for the current trading-bot truth, while checking runtime evidence when claims conflict.
3. Read [research_ops/AI_AGENT_OPERATING_CONTRACT.md](research_ops/AI_AGENT_OPERATING_CONTRACT.md) for practical operator workflow. Its formal binding status is still unresolved in the map.
4. Read [v4/README.md](v4/README.md), [v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md](v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md), and [v4/docs/DATA_CONTRACT.md](v4/docs/DATA_CONTRACT.md) before changing or validating the v4 system.
5. Treat `v2/`, `v3/`, `archive/`, and `archive_quarantine/` as protected history unless there is current import, runtime, test, log, registry, or owner evidence that a specific file is active.

## Current Operating Truth

- Current controlled paper spine: `PAPER_DEFAULT_PROTOCOL101`.
- Current intended path: daily autopilot -> Protocol160 persistent paper trader -> Protocol101 entry -> Protocol051/054 surface scorer -> Protocol081/066 lifecycle/exit -> IBKR paper guard/executor -> paper logs and monitoring.
- Paper trading is guarded paper-submit infrastructure only. It is separate from real-money trading.
- A paper fill proves execution capability only. It does not prove alpha, profitability, approval, or normal-threshold readiness.
- "Active" means currently used in the pipeline or owner workflow. It does not mean profitable, fully validated, promotion-approved, or real-money ready.
- If evidence is missing, mark the answer `UNKNOWN`. If docs conflict, report the conflict instead of resolving it silently.

## Document Precedence

When docs disagree, use this order:

1. Runtime evidence: selected registry entry, configs, manifests, logs, tests, and actual imports.
2. [PROJECT_SECTION_AND_FEATURE_MAP.md](PROJECT_SECTION_AND_FEATURE_MAP.md).
3. [docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md](docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md).
4. `research_ops/` operating docs, subject to unresolved binding status.
5. Current `v4/docs/` documents.
6. Root README and agent docs after 2026-05-26 cleanup.
7. Older v2/v3/archive docs, old promotion packets, readiness claims, and stale runbooks as historical evidence only.

## Hard Safety Rules

Do not casually run commands that can trade, contact a broker, download paid data, mutate runtime posture, install launchd jobs, train models, tune thresholds, or promote challengers.

Commands involving the following require explicit owner authorization and a fresh safety read:

- IBKR, broker, order, live, paper-submit, no-order-trading, or market-data scripts.
- Paid Databento/Polygon downloads or broad backfills.
- Model training, threshold tuning, challenger promotion, or artifact promotion.
- Runtime flag edits, launchd install/uninstall, `bootstrap`, `bootout`, `enable`, `disable`, or plist mutation.
- Cleanup moves outside a reviewed, manifest-backed quarantine batch.

Safe read/validation examples include registry print-selection, targeted pytest suites that do not contact broker/data endpoints, static file inspection, and chart export from local artifacts.

## Current Project Structure

| Path | Meaning |
|------|---------|
| `PROJECT_SECTION_AND_FEATURE_MAP.md` | Current high-level cartography and active/stale split. |
| `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` | Current trading-bot truth doc, subject to runtime evidence. |
| `research_ops/` | Practical current operator/agent front door. |
| `v4/` | Current research, validation, paper runtime, guard, monitoring, and artifact surface. |
| `v4/artifacts/`, `v4/audit/`, `v4/logs/`, `v4/runtime/` | Generated evidence, reports, logs, charts, flags, and runtime state. Inspect before trusting. |
| `data/`, `raw/`, `cache/`, `vendor/`, `processed/`, `v4/raw/`, `v4/normalized/`, `v4/feature/`, `v4/label/` | Protected data areas. Do not delete or reorganize casually. |
| `v2/`, `v3/`, `archive/`, `archive_quarantine/` | Protected history unless proven active per-file. |
| `_cleanup_quarantine/` | Manifest-backed quarantine moves. Files here were moved, not deleted. |

## Current Paper Spine

The current controlled paper path should be understood in plain English as:

1. A daily autopilot selects the paper default.
2. The selected Protocol101 runner starts the persistent paper trader.
3. Protocol101 decides whether a candidate trade should exist.
4. Protocol051/054 provide the option-surface scoring dependency.
5. Protocol081/066 handle lifecycle, exit, and inference lineage.
6. Guard/executor code controls whether a paper order can be submitted.
7. Logs, audit files, chart exports, and monitor reports provide reconstruction evidence.

This path is active infrastructure, not an approval to trade real money.

## Safe Validation Commands

These are examples of read/local validation commands that should not call the broker or train models:

```bash
PYTHONPATH=. python -m v4.scripts.run_daily_paper_autopilot --session-date 2026-05-26 --print-selection

PYTHONPATH=. python -m pytest \
  v4/tests/test_daily_paper_autopilot.py \
  v4/tests/test_protocol101_entry.py \
  v4/tests/test_protocol051_surface_edge.py \
  v4/tests/test_protocol066_inference.py \
  v4/tests/test_protocol113_trade_charts.py \
  v4/tests/test_paid_data_guard.py

PYTHONPATH=. python -m v4.scripts.export_protocol101_trade_charts \
  --out-dir /tmp/protocol101_trade_charts \
  --skip-train-validation
```

Before running broader tests, inspect them for broker, paid-data, launchd, training, and runtime-mutation behavior.

## Cleanup Rules

- Cleanup should happen one small reviewed batch at a time.
- Use a dated quarantine directory with a manifest, rationale, original paths, dependency notes, and rollback notes.
- Move files only after checking for current imports, runtime references, tests, logs, owner workflow, and docs that still depend on them.
- Do not delete in the first pass. Do not move active data, current model artifacts, runtime flags, current logs, launchd files, or broker-related configs casually.
- Tests follow the feature they cover: active-feature tests are protected; research-history tests move only with their family after review.

## Code Quality

- Read files before editing them, especially in this dirty worktree.
- Preserve user changes and unrelated dirty files.
- Use targeted tests proportional to the risk of the change.
- For documentation-only cleanup, prefer clear status labels over silent rewrites of project history.
