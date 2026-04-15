# ART2 — SPX 0DTE Options Research System

Autoresearch system for training and evaluating models that trade SPX 0DTE long options.

## What this repo is

A research pipeline for learning when and what to trade in same-day SPX options. The system acquires market data, trains models, validates them by replaying historical days, and (eventually) tests them in IBKR paper trading. The current focus is making the research pipeline trustworthy before any live execution.

## Where everything lives

```
v2/                        The active system. All code, data, docs.
archive/                   Historical v1/v3 artifacts. Read-only reference.
archive_quarantine/        Files removed during 2026-04-15 cleanup. See ARCHIVE_POLICY.md.
CLAUDE.md                  Agent bootstrap (AI sessions start here).
ARCHIVE_POLICY.md          Explains the two archive directories.
```

`v2/` is the current canonical package. The name is historical. The GitHub repo references "v4 exact chain" which describes the data schema version, not a separate system.

## The pipeline

```
1. Data Acquisition     polygon API -> raw market data
2. Dataset Build        raw data -> v2/data.pt + v2/data_sidecars/
3. Training             data.pt -> v2/models/model.pt        (Akash H100 GPU)
4. Validation           model.pt + data.pt -> replay metrics, trade plots, eval reports
5. Promotion            pass gates -> promoted checkpoint + artifact bundle
6. Paper Trading        (deferred until pipeline is trustworthy)
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
| [v2/ART2_LOOP.md](v2/ART2_LOOP.md) | Canonical hill-climbing protocol |
| [v2/docs/current_state.md](v2/docs/current_state.md) | Current system state snapshot |
| [v2/COMMANDS.md](v2/COMMANDS.md) | Command reference |
| [v2/PIPELINE.md](v2/PIPELINE.md) | System overview |
| [v2/docs/founder_intent.md](v2/docs/founder_intent.md) | Non-negotiable project standards |
| [v2/docs/README.md](v2/docs/README.md) | Full documentation index |

## Current status

The pipeline integrity fix (2026-04-15) corrected three compounding metric bugs that masked real losses as near-breakeven. All prior experiment scores were stale and have been replaced. A fresh supervised model, BC agent, and AWAC agent have been trained on the corrected pipeline. All health checks pass.
