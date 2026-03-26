# Auto Research Trader (ART²)

Autonomous SPX 0DTE long options trading bot. An AI agent (Claude Sonnet) iteratively evolves a neural network trading model through hundreds of experiments on GPU, then deploys the best model to live paper trading on IBKR.

**ART²** (ART squared) is the meta-loop: an outer loop (Claude Opus) makes strategic decisions — architecture changes, loss function design, feature engineering — while the inner loop (Claude Sonnet) optimizes hyperparameters and training dynamics within those constraints.

## Model Architecture (v6 Four-Head)

Four-head transformer for SPX 0DTE options trading:

| Head | Output | Purpose |
|------|--------|---------|
| **Gate** | (batch, 2) → [NO_TRADE, TRADE] | Entry/exit signal |
| **Direction** | (batch, 6) → 3 call + 3 put strikes | Strike + direction selection |
| **Value** | (batch, 1) → remaining P&L prediction | Exit intelligence (MSE) |
| **Risk** | (batch, 3) → [stop_pct, size_frac, conviction] | Account-aware risk management |

- **37 features** (v3): price/volume, session/time, options/Greeks, market structure
- **7-dim position state**: in_trade, bars_held, unrealized P&L, account health, loss streak, best P&L, bars since high
- **4-dim account state** (risk head only): growth ratio, log size, daily P&L fraction, win rate
- **Dynamic stop-loss**: 15%-60% range, learned by risk head (formula fallback)
- **Exit priority**: stop_loss > model_exit > value_exit > max_hold > EOD

## Training Pipeline

The inner loop runs on Akash H100 GPUs. Each experiment (~6 min):
1. Sonnet agent proposes targeted edits to `train.py`
2. `inner_loop.py` validates, uploads, trains, scores
3. If score improves: promote model + code. Otherwise: revert.

**PBT mode** (Population-Based Training): N members compete per generation with evolutionary selection — elite carry-forward, exploit top-25%, explore top-50%.

## Core Design Principles

- **The model IS a trader.** Training simulates minute-by-minute 0DTE options trading — the same game as live IBKR paper trading.
- **No hardcoded exits.** The gate head learns when to enter and exit. The value head predicts remaining P&L and exits when upside is gone. Dynamic stop-loss is the emergency backstop.
- **Data integrity first.** `data.pt` must always include option chain sidecar data.
- **Train on GPU, not locally.** All training runs deploy to Akash H100s via `deploy.sh`.
- **Feature parity enforced.** Training, replay, and live all use the same `compute_features()` pipeline.

## Quick Start

```bash
# Environment
cp .env.example .env  # Add POLYGON_API_KEY, POLYGON_S3_KEY_ID, POLYGON_S3_SECRET
set -a && source .env && set +a

# Training (inner loop)
./infra/deploy.sh boot                                    # Spin up H100 on Akash
./infra/deploy.sh start --hours 8 --max-experiments 200   # Upload + start
python3 tools/monitor.py                                  # Dashboard → localhost:8420
./infra/deploy.sh stop                                    # Download + close

# ART² (outer loop — autonomous cycles)
python3 tools/art2.py daemon --max-cycles 3 --minutes 90  # 3 autonomous cycles

# Paper trading (IBKR)
python tools/paper_live.py --context-only                 # Refresh context
python tools/paper_live.py --paper-auto --max-minutes 390 # Full session

# Replay / backtest
python3 training/replay.py --date 2026-03-17 --output replay-trades.csv
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/](docs/README.md) | Documentation index — see `docs/README.md` for full map |
| [docs/operations/reference.md](docs/operations/reference.md) | Full project reference: key files, constants, subcommands, IBKR ops, daily pipeline |
| [docs/architecture/](docs/architecture/) | System architecture diagrams (data flow, model heads, live stack, inner loop) |
| [.claude/rules/art2-operating-manual.md](.claude/rules/art2-operating-manual.md) | ART² lifecycle, roles, policies (auto-loaded) |
| [docs/domain/](docs/domain/) | 0DTE options + practical trading domain knowledge |
| [docs/operations/ibkr-trade-analysis-guide.md](docs/operations/ibkr-trade-analysis-guide.md) | Guide for analyzing IBKR paper trading sessions |

## Safety

- Real-money trading is not enabled
- Paper trading verified end-to-end: all 37 features (v3), four-head inference, LMT fills, OCO brackets
- All positions are 1 SPX contract, long calls/puts only
- Dynamic stop loss (15%-60%) on all positions
- 5-bar cooldown after stop-loss exits
- Human review gate (REVIEW phase) before every ART² training cycle
