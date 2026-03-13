# Autoresearch-Trading Handoff v0.3 (Options-Native)

> **Date**: March 13, 2026
> **Status**: Ready for deployment — options-native rewrite complete, needs data prep

## What This Is

Karpathy's autoresearch framework adapted for **options-native intraday SPY trading**.
The model predicts next-hour **options-adjusted P&L** (not raw equity return) every
hour during RTH. End goal: give $500 to this model and let it trade 0DTE/1DTE SPY
options autonomously.

**Read `program.md` first.** It has the full instruction set.

## What Changed (v0.2 → v0.3)

| Component | v0.2 | v0.3 |
|-----------|------|------|
| Features | 71 (equity-only) | **105** (+Greeks, IV, flow, risk, options context) |
| Target | Raw next-hour equity return | **Options-adjusted return** (delta + gamma - theta) |
| Options data | None (placeholder) | **Real Greeks, IV, OI, volume** from Polygon |
| Model | Simple transformer | **Feature-group-gated transformer** with risk head |
| Parameters | ~130K | **~450K** (d_model=192, depth=6, 6 heads) |
| Loss | directional | **combined** (directional + sharpe) |
| Transaction cost | 8 bps | **12 bps** (more realistic for options) |
| Confidence | 0 (always trade) | **0.05** (forced flat below threshold) |
| put_call_ratio | Hardcoded 0.0 | **Real** from Polygon options chain |
| New feature groups | n/a | Options Greeks (16), Flow (6), Risk State (6), Target Context (6) |
| Loss option | n/a | **`options_aware`** asymmetric loss (1.5x penalty for losses) |

## Key Architectural Changes

### Feature Group Gating
The model has a gating layer that learns which feature groups matter per timestep.
For example, it might learn that Greeks features matter more near expiry, momentum
features matter more in the morning, and risk features matter after a drawdown.

### Risk-Aware Output Head
The final decision layer receives both the transformer output AND the raw risk state
features (session P&L, drawdown, consecutive losses, time to close). This gives the
model direct access to "how am I doing today" for the final trade/no-trade decision.

### Options-Adjusted Target
Instead of predicting equity return, the model predicts:
```
P&L = delta × return × price + 0.5 × gamma × return² - theta/6.5
```
This means the Sharpe ratio directly reflects **what a $500 options account would make**.

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
| **Polygon.io** | Options Developer plan. API key set in env. |
| **Data** | Needs full re-download (v0.3 cache has new format) |

## Before Running Experiments

1. **Set Polygon API key on container**:
   ```
   $SSH 'export POLYGON_API_KEY=YOUR_KEY && cd /workspace/autoresearch-trading && git pull'
   ```

2. **Clear ALL old data caches** (v0.3 uses different cache files):
   ```
   $SSH 'rm -rf /root/.cache/autoresearch-trading/'
   ```

3. **Install new dependencies and run data prep**:
   ```
   $SSH 'cd /workspace/autoresearch-trading && PATH="$HOME/.local/bin:$PATH" POLYGON_API_KEY=YOUR_KEY uv run prepare.py'
   ```
   This downloads equity bars (5-15 min) + options chain snapshots (15-30 min).

4. **Run baseline and start experiment loop** per program.md.

## Key Files

| File | Role | Modify? |
|------|------|---------|
| `prepare.py` | Polygon data + options chain download, 105 features, evaluation | **NO** |
| `train.py` | OptionsAwareTradingModel, feature gating, optimizer, loss | **YES** |
| `program.md` | Instructions, domain knowledge, experiment ideas | Read-only |
| `pyproject.toml` | Dependencies | NO |
| `prepare_v02_backup.py` | Backup of v0.2 prepare.py | Archive |
| `train_v02_backup.py` | Backup of v0.2 train.py | Archive |
| `prepare_daily_backup.py` | Backup of v0.1 prepare.py | Archive |
| `train_daily_backup.py` | Backup of v0.1 train.py | Archive |

## Lessons from v0.1 and v0.2

- Sharpe loss alone went strongly negative — use `combined` or `options_aware` loss
- H100 massively underutilized — v0.3 model is bigger but still only ~2GB / 80GB
- `set -e` in Akash startup kills container on non-fatal errors
- Docker needs `tzdata` package
- Raw equity Sharpe doesn't translate to options profit — theta eats edges < 30bps
