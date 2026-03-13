# Autoresearch-Trading Handoff (Intraday Edition)

> **Date**: March 13, 2026
> **Status**: Ready for deployment — intraday rewrite complete, needs data prep

## What This Is

Karpathy's autoresearch framework adapted for **intraday SPY options trading**.
The model predicts next-hour SPY direction every hour during RTH, outputting a
position signal in [-1, 1]. End goal: give $500 to this model and let it trade
0DTE/1DTE SPY options autonomously.

**Read `program.md` first.** It has the full instruction set.

## What Changed (v0.1 daily → v0.2 intraday)

| Component | Before (v0.1) | After (v0.2) |
|-----------|--------------|--------------|
| Data source | yfinance (daily bars) | **Polygon.io** (5-min bars) |
| Features | 39 (daily) | **71** (intraday + session + internals + multi-instrument + daily context) |
| Target | Next-day return | **Next-hour return** |
| Decision frequency | Once per day | **6 times per day** (10:30-15:30 hourly) |
| Lookback | 60 daily bars (3 months) | **36 hourly bars (6 trading days)** |
| Transaction costs | 5 bps | **8 bps** (options bid-ask) |
| Validation bars | ~250 (daily) | **~1500** (hourly, much more statistical power) |
| New features | n/a | VWAP, IB, session structure, TICK/TRIN, A/D, NQ/ES, VIX term structure |

## Infrastructure

| Component | Details |
|-----------|---------|
| **Local repo** | `/Users/gduby/Documents/Trinity/Trinity/autoresearch-trading` |
| **Branch** | `autoresearch/mar13` |
| **GitHub** | `https://github.com/owenheidenreich/autoresearch-trading` |
| **H100 Container** | Akash DSEQ `25914541` |
| **SSH** | `sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 root@provider.h100.wdc.hh.akash.pub` |
| **Container repo** | `/workspace/autoresearch-trading` |
| **GPU** | NVIDIA H100 80GB HBM3 |
| **Polygon.io** | API key needed: set `POLYGON_API_KEY` env var |
| **Data** | Needs re-download (old daily cache must be cleared) |

## Before Running Experiments

1. **Set Polygon API key on container** (user provides the key):
   ```
   $SSH 'export POLYGON_API_KEY=THE_KEY && cd /workspace/autoresearch-trading && git pull'
   ```

2. **Clear old data cache**:
   ```
   $SSH 'rm -rf /root/.cache/autoresearch-trading/'
   ```

3. **Install new dependencies and run data prep**:
   ```
   $SSH 'cd /workspace/autoresearch-trading && PATH="$HOME/.local/bin:$PATH" POLYGON_API_KEY=KEY uv run prepare.py'
   ```
   This downloads SPY/VIX/QQQ/SPX 5-min bars from Polygon (takes 5-15 min).

4. **Run baseline and start experiment loop** per program.md.

## Key Files

| File | Role | Modify? |
|------|------|---------|
| `prepare.py` | Polygon data download, 71 feature computation, hourly evaluation harness | **NO** |
| `train.py` | IntradayTradingModel, optimizer, loss, hyperparameters | **YES** |
| `program.md` | Instructions, domain knowledge, experiment ideas | Read-only |
| `pyproject.toml` | Dependencies (`polygon-api-client` replaces `yfinance`) | NO |
| `prepare_daily_backup.py` | Backup of old daily prepare.py | Archive |
| `train_daily_backup.py` | Backup of old daily train.py | Archive |

## Lessons from v0.1

- Sharpe loss alone went strongly negative — use `combined` loss (directional + sharpe)
- H100 is massively underutilized (220MB / 80GB) — room for much larger models
- `set -e` in Akash startup kills container on non-fatal errors
- Docker needs `tzdata` package