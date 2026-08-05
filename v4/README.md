# SPX 0DTE v4 - Frozen Legacy Research And Paper-Trading Stack

> **V4 FROZEN 2026-08-05.** New research, plans, status updates, and agent instructions live in
> [`v5/`](../v5/README.md). Use [`v5/TOOLBOX.md`](../v5/TOOLBOX.md) to find the few vetted v4 dependencies.
> This tree preserves history, generated evidence, and dangerous paper/broker capabilities; it is not the
> active project boundary.

> **NOT A STATUS DOCUMENT (2026-08-05).** Current status, gates, and open jobs are in [`STATUS.md`](../STATUS.md). The Protocol101 paper-default description below is the LEGACY spine, not the current Path-D research path.


v4 is the current home of the SPX 0DTE research, validation, Protocol101 paper-default, paper-runtime, guard, monitoring, and artifact surface. The original clean-slate protocol language remains useful lineage, but older Phase 0 claims no longer describe the whole current state.

Current repo map: [../PROJECT_SECTION_AND_FEATURE_MAP.md](../PROJECT_SECTION_AND_FEATURE_MAP.md).

Current trading-bot truth doc: [../docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md](../docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md).

Historical research protocol: `/Users/gduby/.claude/plans/ok-well-this-just-declarative-puppy.md` (filename is historical; canonical title is `SPX_0DTE_v4_RESEARCH_PROTOCOL.md`).

## Governing principle

> v4 is not a neural-network rebuild. v4 is a causal execution-research system for SPX 0DTE long options. The model is subordinate to the simulator; the simulator is subordinate to point-in-time data; every result is subordinate to out-of-sample falsification. **The hero is the audit trail, not the model.**

## Current status

As of the 2026-05-26 cleanup/cartography pass, the controlled paper spine is `PAPER_DEFAULT_PROTOCOL101` when selected by current registry/runtime evidence:

daily autopilot -> Protocol160 persistent paper trader -> Protocol101 entry -> Protocol051/054 surface scorer -> Protocol081/066 lifecycle/exit -> IBKR paper guard/executor -> paper logs and monitoring.

This is guarded paper-submit infrastructure only. It is separate from real-money trading, and a paper fill is capability evidence, not alpha proof or normal-threshold proof.

## Directory layout

### Code packages (Python)
- [schema/](schema/) — schema dataclasses for raw / normalized / feature / label / audit rows
- [parser/](parser/) — canonical SPX/SPXW contract ID parser
- [ingest/](ingest/) — deterministic ingest (OptionsDX scaffold first); fingerprinting + hashing
- [dataset/](dataset/) — SPXW 0DTE neural decision-row builders
- [checks/](checks/) — bid/ask sanity, timestamp monotonicity, duplicate-key, deterministic-rebuild
- [greeks/](greeks/) — Black-Scholes IV/Greek calculation; reconciliation against vendor Greeks
- [leakage/](leakage/) — planted-leak test, shuffled-label test, leak-detection CI
- [sim/](sim/) — minute-grain simulator skeleton + order-state-machine skeleton
- [tests/](tests/) — pytest suite
- [scripts/](scripts/) — research scripts, validation/export tools, paper-runtime entry points, broker/paper tools, and experimental protocol scripts. Inspect before running; some scripts are dangerous outside explicit owner authorization.

Key current paper/validation scripts include:

- [scripts/run_daily_paper_autopilot.py](scripts/run_daily_paper_autopilot.py) — selects the current paper default; `--print-selection` is the safe selection check.
- [scripts/run_protocol160_protocol101_persistent_paper_trader.py](scripts/run_protocol160_protocol101_persistent_paper_trader.py) — Protocol101 persistent paper trader entry point.
- [scripts/export_protocol101_trade_charts.py](scripts/export_protocol101_trade_charts.py) — local chart export for `equity.html`, `trades.html`, CSV, and summary artifacts.
- [scripts/run_protocol141_ibkr_paper_order_guard.py](scripts/run_protocol141_ibkr_paper_order_guard.py), [scripts/run_protocol142_ibkr_paper_executor_smoke.py](scripts/run_protocol142_ibkr_paper_executor_smoke.py), and related paper/broker scripts — dangerous unless explicitly authorized.

### Data directories (Parquet content gitignored)
- [raw/](raw/) — vendor-original messages, immutable, hash-stamped
- [normalized/](normalized/) — unified contract symbology; deterministic from raw
- [feature/](feature/) — causal features only (`is_live_reproducible=True`)
- [label/](label/) — future outcomes; never joined into features

### Markdown artifacts
- [docs/](docs/) — `DATA_CONTRACT.md`, `SIMULATOR_CARD.md`, etc.
- [audit/](audit/) — pipeline integrity reports, leakage audit results
- [ledger/](ledger/) — `RESEARCH_LEDGER.md` (append-only) + per-experiment files
- [promotion/](promotion/) — promotion-packet templates + instances

Key operating docs:

- [docs/protocol101/training/README.md](docs/protocol101/training/README.md) — canonical Protocol101 training state, contracts, gate map, alpha-preservation ledger, and next permitted phase.
- [docs/MODEL_IMPROVEMENT_GUIDELINES.md](docs/MODEL_IMPROVEMENT_GUIDELINES.md) — required rulebook for deciding whether a challenger is actually better than the paper default.
- [docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md](docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md) — required stage-gate process from hypothesis to experiment to validation to research freeze to runtime parity to paper promotion.
- [docs/NAMING_GUIDE.md](docs/NAMING_GUIDE.md) — role labels for paper defaults, challengers, experiments, audits, runtime harnesses, and decisions.
- [docs/PROMOTION_SEQUENCE.md](docs/PROMOTION_SEQUENCE.md) — promotion ladder from research candidate to paper/live consideration.

## Standing ground rules

1. **v4 is intended to be independent of v2/v3 imports.** Verify with tests or static checks; do not assume because this README says so.
2. **Every feature row has `is_live_reproducible`.** If a feature could not exist live at decision_time, it does not belong in the feature layer.
3. **Deterministic rebuild.** Re-running ingest produces byte-identical normalized output.
4. **Leak-detection CI on every commit.** Planted-leak test must catch the planted leak; shuffled-label test must produce no edge.
5. **No broad paid-data backfill, model training, threshold tuning, promotion, broker call, or paper-submit run without explicit owner authorization.**

## Historical Phase 0 ticket list

This section is historical context from the original clean-slate build. Do not treat it as the current status without checking the root cartography report, current docs, runtime registry, tests, and generated evidence.

See protocol Section 9.4. Phase 0 was originally defined as complete when all 15 tickets were merged, `audit/PIPELINE_INTEGRITY_REPORT.md` was green, a research-ledger entry existed, and the promotion-packet template was ready for Phase 0 -> Phase 0.5 review.

## SPXW 0DTE Pilot

Pilot and data-download instructions are in [docs/SPXW_0DTE_DATA_DOWNLOADS.md](docs/SPXW_0DTE_DATA_DOWNLOADS.md). Treat data downloads as protected operations: inspect the docs and local cache, but do not run paid downloads or broad backfills without explicit owner authorization.
