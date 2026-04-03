# Auto Research Trader (ART^2)

Autonomous SPX 0DTE options trading system. A neural network learns complete trading decisions (entry, direction, strike, risk, exit) from minute-bar market data, then executes on IBKR paper trading.

**ART^2** is the meta-loop: Claude Opus makes strategic decisions (architecture, features, loss design) while Claude Sonnet optimizes hyperparameters within those constraints. Training runs on Akash H100 GPUs in 5-minute experiments. The best model promotes automatically and trades the next session.

## Architecture (v18 Five-Head Trader)

Transformer backbone (d=64, 4 attention heads, 3 layers, causal masking) feeds five task-specific heads:

| Head | Output | Purpose |
|------|--------|---------|
| **Market** | (batch, 3) | SPX % change at 15/30/60 bar horizons |
| **Entry Gate** | (batch,) sigmoid | P(should enter trade now) |
| **Risk** | (batch, 3) | stop distance, target distance, conviction |
| **Exit** | (batch,) sigmoid | P(should exit current position) |
| **Direction** | (batch, 6) softmax | CALL/PUT at ATM, OTM+5, OTM+10 strikes |

**39 features** from SPX price, SPY volume, VIX, and SPXW option chains (IV, Greeks, gamma pressure, skew). Labels use path-quality metrics (MFE/MAE) from forward price action. No hindsight P&L in training.

## How The Loop Works

```
Sunday 8 PM PT: Weekly Retrain
  1. prepare.py rebuilds data.pt (4-year SPX history + option chains)
  2. deploy.sh boots Akash H100
  3. Opus proposes mutation -> inner_loop.py runs experiment (5 min)
  4. Score improves? KEEP (promote model + code). Otherwise REVERT.
  5. Repeat until budget exhausted. Shut down GPU.

Weekday 2:30 AM PT: Daily Pipeline
  1. Incremental data update (append-only cache)
  2. paper_live.py runs full session (9:30-16:00 ET)
     - 5-second bars aggregate into 1-min candles
     - 39 features computed (identical to training)
     - Model inference -> DecisionIntent -> IBKR orders
  3. All trades logged to trades.jsonl
```

## Project Structure

```
training/
  train.py           # v18 model definition + training loop
  best_train.py      # Promoted production code (auto-synced)
  best_model.pt      # Production weights
  prepare.py         # Data pipeline: 39 features, path-quality labels
  replay.py          # Backtest simulator (forward-test on unseen days)
  program.md         # Architecture contract (single source of truth)
  lab_notebook.md    # What works and what doesn't (37 failed patterns)
  trading_rules.py   # Hardcoded safety envelope (stop limits, cooldowns)
  live/
    service.py       # IBKR paper trading session manager
    decision.py      # Model output -> trading intent
    execution.py     # OCO bracket order placement
    features.py      # Live feature computation (parity with training)
    resolver.py      # SPXW contract resolution
    context.py       # Normalization context bundles

tools/
  inner_loop.py      # Experiment orchestrator (validate -> upload -> train -> score -> keep/revert)
  daily_pipeline.py  # Scheduled data rebuild + paper trading
  paper_live.py      # Paper trading CLI
  monitor.py         # Training dashboard (localhost:8420)

infra/
  deploy.sh          # Akash GPU lifecycle (boot/start/stop/sync)
  daily_pipeline.plist   # launchd: weekday 2:30 AM PT
  weekly_retrain.plist   # launchd: Sunday 8 PM PT
```

## Quick Start

```bash
# Environment
cp .env.example .env  # Add POLYGON_API_KEY, POLYGON_S3_KEY_ID, POLYGON_S3_SECRET
set -a && source .env && set +a

# Training (inner loop on Akash H100)
./infra/deploy.sh boot
./infra/deploy.sh start --hours 8 --max-experiments 200
python3 tools/monitor.py
./infra/deploy.sh stop

# Paper trading (IBKR)
python tools/paper_live.py --context-only
python tools/paper_live.py --paper-auto --max-minutes 390

# Replay / backtest
python3 training/replay.py --backtest --model training/best_model.pt
```

## Scoring

Experiments are scored on a held-out validation set:

```
score = direction_accuracy * (1 + max(0, rank_correlation))
```

Direction accuracy measures how often the model predicts the correct sign of SPX returns. Rank correlation measures whether predicted magnitudes rank correctly against actual returns. Both must improve together.

## Design Principles

- **The model is a trader.** It learns entry timing, strike selection, risk parameters, and exit signals. Not just direction prediction.
- **No hindsight.** Labels come from forward price paths (MFE/MAE), not realized option P&L.
- **Feature parity enforced.** Training, replay, and live all run the same `compute_features()` pipeline. 39 features, same normalization.
- **Train on GPU, not locally.** All training deploys to Akash H100 via deploy.sh. 5-minute budget per experiment.
- **Score is the arbiter.** Inner loop promotes or reverts based on score alone. No manual cherry-picking.

## Safety

- Real-money trading is not enabled. Paper trading only.
- 1 SPX contract maximum. Long calls/puts only. 0DTE only.
- Dynamic stop-loss (15%-60% range), learned by risk head.
- 5% daily loss cap. 5-bar cooldown after stops.
- Lunch suppression (10:30-13:30 ET). No trades in first 30 minutes.
- Exit priority: stop_loss > model_exit > max_hold > EOD.

## Documentation

| Doc | Purpose |
|-----|---------|
| [training/program.md](training/program.md) | Architecture contract (locked elements, agent constraints) |
| [training/lab_notebook.md](training/lab_notebook.md) | Anti-patterns and experimental results |
| [docs/domain/](docs/domain/) | 0DTE options, volatility, trading psychology |
