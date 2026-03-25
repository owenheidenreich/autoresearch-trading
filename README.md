# Auto Research Trader (ART)

Autonomous SPX 0DTE long options trading bot. An AI agent (Claude Sonnet) iteratively evolves a neural network trading model through hundreds of experiments on GPU, then deploys the best model to live paper trading on IBKR.

**ART²** (ART squared) is the meta-loop: an outer loop (Claude Opus) makes strategic decisions — architecture changes, loss function design, feature engineering — while the inner loop (Claude Sonnet) optimizes hyperparameters and training dynamics within those constraints.

**Model:** Three-head transformer (v5) — gate head (enter/exit), direction head (strike + call/put), value head (predicted remaining P&L). 37 features, 7-dim position state, dynamic stop-loss. Exit priority: stop > gate > value > max_hold > EOD.

## Core Design Principles

- **The model IS a trader.** Training simulates minute-by-minute 0DTE options trading — the same game as live IBKR paper trading.
- **No hardcoded exits.** The model's gate head learns when to enter and exit. A value head predicts remaining P&L and exits when upside is gone. Dynamic stop-loss (15%-60%) is the emergency backstop.
- **Data integrity first.** `data.pt` must always include option chain sidecar data.
- **Train on GPU, not locally.** All training runs deploy to Akash H100s via `deploy.sh`.

## Quick Start

```bash
# Environment
cp .env.example .env  # Add POLYGON_API_KEY, POLYGON_S3_KEY_ID, POLYGON_S3_SECRET
set -a && source .env && set +a

# Training
./infra/deploy.sh boot                                    # Spin up H100 on Akash
./infra/deploy.sh start --hours 8 --max-experiments 200   # Upload + start
python3 tools/monitor.py                                  # Dashboard → localhost:8420
./infra/deploy.sh stop                                    # Download + close

# Paper trading
python tools/paper_live.py --context-only                 # Refresh context
python tools/paper_live.py --paper-auto --max-minutes 390 # Full session

# Replay
python3 training/replay.py --date 2026-03-17 --output replay-trades.csv
```

## Documentation

- [docs/reference.md](docs/reference.md) — Full project reference (key files, subcommands, constants, pipeline)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture diagrams
- [.claude/rules/art2-operating-manual.md](.claude/rules/art2-operating-manual.md) — ART² lifecycle and process authority

## Safety

- Real-money trading is not enabled
- Paper trading verified end-to-end (2026-03-24): all 37 features (v3), LMT fills, OCO brackets
- All positions are 1 SPX contract, long calls/puts only
- Dynamic stop loss (15%-60%) on all positions
