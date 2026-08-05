# Autoresearch Trading — SPX 0DTE Research And Paper-Trading System

> **Current status, gates, and open jobs live in [`STATUS.md`](STATUS.md).** This README describes the repository; it is not a status document. The 'current front door' line below is stale — the current research path is Path-D, not the legacy Protocol101 paper spine.


Research system for building, auditing, falsifying, and guarded-paper-testing strategies for same-day SPX options. The current front door is the v4 Protocol101 paper-trading spine, with older v2/v3/archive material preserved as project history unless a current import, registry, runtime pointer, or owner workflow proves otherwise.

For the current end-to-end map, start with [PROJECT_SECTION_AND_FEATURE_MAP.md](PROJECT_SECTION_AND_FEATURE_MAP.md).

## Reviewer Summary

- **Domain:** SPX/SPXW 0DTE options research
- **Focus:** data integrity, leakage prevention, replay validation, paper-trading safety controls, and experiment governance
- **Stack:** Python, PyTorch, pandas, NumPy, pytest, Databento/Polygon data adapters, Akash GPU workflows
- **Status:** v4 research and guarded paper-trading workbench. Real-money trading is not the default and is not declared approved here.

## What this repo is

A research and operations pipeline for learning when and what to trade in same-day SPX options, then testing the chosen behavior through a controlled paper-trading bridge. The system acquires market data, builds point-in-time datasets, trains models, validates them by replaying historical days, exports trade/equity evidence, and runs a guarded paper-submit path for Protocol101.

This is intentionally presented as an engineering and research-governance project, not as a financial claim. "Active" means currently used in the repo or owner workflow; it does not mean profitable, promotion-approved, or real-money ready.

## Where everything lives

| Path | Current meaning |
|------|-----------------|
| [PROJECT_SECTION_AND_FEATURE_MAP.md](PROJECT_SECTION_AND_FEATURE_MAP.md) | Current repo cartography: active core, dependencies, stale/conflict areas, and workflow map. |
| [docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md](docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md) | Current trading-bot truth document when it agrees with runtime evidence. |
| [research_ops/](research_ops/) | Practical operator/agent front door for current workflows, with binding status still requiring explicit review. |
| [v4/](v4/) | Current research, validation, Protocol101, paper-runtime, guard, monitoring, and artifact surface. |
| [v2/](v2/) and [v3/](v3/) | Protected history and research lineage unless specifically proven active by imports, runtime references, or owner instruction. |
| [archive/](archive/) and [archive_quarantine/](archive_quarantine/) | Historical/quarantined material. Do not treat as active by default. |
| [CLAUDE.md](CLAUDE.md) | Agent bootstrap. Current instructions supersede older v2-first guidance. |
| [ARCHIVE_POLICY.md](ARCHIVE_POLICY.md) | Explains archive policy and historical quarantine context. |

The version numbers are historical project phases, not production releases. Do not infer current status from a folder existing; use the current docs, registry/runtime evidence, tests, logs, and owner-confirmed workflow.

## Security and Data Notes

- API keys belong only in local `.env` files.
- Paid market data, broker credentials, and vendor secrets must never be committed.
- Model checkpoints and generated artifacts are research outputs, not required to understand the code. They should be pruned from future public branches where practical.
- This repository documents research and tooling only; it does not provide financial advice.
- Public reviewers should focus on source code, docs, tests, and audit methodology. Any local broker/API credentials, purchased market data, and private account configuration are intentionally absent.

## The pipeline

```
1. Data Acquisition     vendor APIs/files -> raw market data
2. Normalization        raw data -> canonical contracts, timestamps, quotes, trades
3. Feature Build        point-in-time data -> live-reproducible features
4. Training             feature/label layers -> model candidates
5. Validation           model candidates -> replay metrics, trade plots, audit reports
6. Promotion            pass gates -> promotion packet + artifact bundle
7. Paper Trading        Protocol101 guarded paper-submit path, separated from real-money trading
```

## Quick start

```bash
# Current cartography and front-door docs
less PROJECT_SECTION_AND_FEATURE_MAP.md
less docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md

# Safe registry selection check for the current paper default.
# This prints the selected Protocol101 paper runner without placing orders.
PYTHONPATH=. python -m v4.scripts.run_daily_paper_autopilot --session-date 2026-05-26 --print-selection

# Generate Protocol101 validation charts without training or broker calls.
PYTHONPATH=. python -m v4.scripts.export_protocol101_trade_charts --out-dir /tmp/protocol101_trade_charts --skip-train-validation
```

## Key docs

| Doc | Purpose |
|-----|---------|
| [PROJECT_SECTION_AND_FEATURE_MAP.md](PROJECT_SECTION_AND_FEATURE_MAP.md) | Current project and feature map |
| [docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md](docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md) | Current trading-bot truth document |
| [research_ops/AI_AGENT_OPERATING_CONTRACT.md](research_ops/AI_AGENT_OPERATING_CONTRACT.md) | Agent/operator contract for current workflows |
| [v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md](v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md) | Protocol101 paper-trading workflow |
| [v4/docs/DATA_CONTRACT.md](v4/docs/DATA_CONTRACT.md) | v4 data contract |
| [v2/ART2_LOOP.md](v2/ART2_LOOP.md) | Historical v2 hill-climbing protocol |
| [v2/docs/current_state.md](v2/docs/current_state.md) | Historical v2 system state snapshot |
| [v2/docs/founder_intent.md](v2/docs/founder_intent.md) | Non-negotiable project standards |
| [v2/docs/README.md](v2/docs/README.md) | Historical v2 documentation index |

## Current status

Current review should treat the repo as a v4-centered research and guarded paper-trading system with substantial protected history. Protocol101 is the current controlled paper spine when selected by registry/runtime evidence. Older promotion packets, readiness claims, v2/v3 instructions, and archive material may be useful lineage but are not current truth by default.

## Reviewer Notes

- Start with [PROJECT_SECTION_AND_FEATURE_MAP.md](PROJECT_SECTION_AND_FEATURE_MAP.md), [docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md](docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md), and [v4/README.md](v4/README.md).
- Look for engineering discipline: immutable raw data, deterministic normalized layers, explicit label separation, replay-oriented validation, and audit artifacts.
- Do not expect live credentials, private paid datasets, or real-money approval in the public repo.
