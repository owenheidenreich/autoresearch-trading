# Autoresearch Trading — SPX 0DTE Research System

Research system for building, auditing, and falsifying strategies for same-day SPX options. The project is not presented as a live trading product; its core value is the research process, audit trail, and insistence on out-of-sample validation before any paper-trading or live-trading claim.

## Reviewer Summary

- **Domain:** SPX/SPXW 0DTE options research
- **Focus:** data integrity, leakage prevention, replay validation, and experiment governance
- **Stack:** Python, PyTorch, pandas, NumPy, pytest, Databento/Polygon data adapters, Akash GPU workflows
- **Status:** research system and protocol workbench; live execution is intentionally deferred

## What this repo is

A research pipeline for learning when and what to trade in same-day SPX options. The system acquires market data, builds point-in-time datasets, trains models, validates them by replaying historical days, and documents why a result should or should not be trusted. The current focus is making the research pipeline trustworthy before any live execution.

This is intentionally presented as an engineering and research-governance project, not as a trading product. The interesting work is the machinery around data contracts, leakage prevention, deterministic rebuilds, promotion packets, and audit trails.

## Where everything lives

```
v2/                        The active system. All code, data, docs.
archive/                   Historical v1/v3 artifacts. Read-only reference.
archive_quarantine/        Files removed during 2026-04-15 cleanup. See ARCHIVE_POLICY.md.
CLAUDE.md                  Agent bootstrap (AI sessions start here).
ARCHIVE_POLICY.md          Explains the two archive directories.
```

`v2/` is the current canonical package on the public main branch. The version names are historical project phases, not production releases.

## Security and Data Notes

- API keys belong only in local `.env` files.
- Paid market data, broker credentials, and vendor secrets must never be committed.
- Model checkpoints and generated artifacts are research outputs, not required to understand the code.
- Public reviewers should focus on source code, docs, tests, and audit methodology. Any local broker/API credentials, purchased market data, and private account configuration are intentionally absent.
- This repository documents research and tooling only; it does not provide financial advice.

## The pipeline

```
1. Data Acquisition     vendor APIs/files -> raw market data
2. Dataset Build        raw data -> normalized dataset + sidecars
3. Training             datasets -> model candidates
4. Validation           model candidates -> replay metrics, trade plots, eval reports
5. Promotion            pass gates -> promoted checkpoint + artifact bundle
6. Paper Trading        deferred until the pipeline clears trust gates
```

## Quick start

```bash
# Health check — does the pipeline hang together?
python -m v2.ops.health quick

# Replay the current model on out-of-sample days
python -m v2.replay --model v2/models/model.pt --mask promote

# Generate trade visualizations
python -m v2.plot_trades
```

## Key docs

| Doc | Purpose |
|-----|---------|
| [v2/HANDOFF.md](v2/HANDOFF.md) | Current state, session bootstrap |
| [v2/COMMANDS.md](v2/COMMANDS.md) | Command reference |
| [v2/program.md](v2/program.md) | Operating protocol |

| [v2/PIPELINE.md](v2/PIPELINE.md) | System overview for humans |
| [v2/docs/founder_intent.md](v2/docs/founder_intent.md) | Non-negotiable project standards |
| [v2/docs/README.md](v2/docs/README.md) | Full documentation index |

## Current status

The pipeline integrity fix (2026-04-15) corrected three compounding metric bugs that masked real losses as near-breakeven. All prior experiment scores were stale and have been replaced. A fresh supervised model, BC agent, and AWAC agent have been trained on the corrected pipeline. All health checks pass.

## Reviewer Notes

- Start with [v2/docs/README.md](v2/docs/README.md), [v2/COMMANDS.md](v2/COMMANDS.md), and [v2/PIPELINE.md](v2/PIPELINE.md).
- Look for engineering discipline: deterministic data builds, replay-oriented validation, promotion gates, and explicit audit artifacts.
- Do not expect live credentials, private paid datasets, or an executable trading bot in the public repo.
