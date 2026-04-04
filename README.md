# ART² -- Autonomous Research Trader

SPX 0DTE options trading system. A neural network learns complete trading decisions (entry, direction, strike, risk management) from minute-bar market data.

Built on Karpathy's autoresearch pattern: the AI proposes hypotheses, trains models on GPU, evaluates against held-out data, keeps what improves, reverts what doesn't. Humans program the research organization, not the research itself.

## How It Works

```
1. AI edits v2/train.py (model architecture, loss, hyperparams)
2. Trains on 859 days of historical SPX data (55 features, honest labels)
3. Replays on 60 held-out days the model never saw
4. Scores the equity curve: Sortino * consistency * drawdown guard
5. Score improved? KEEP. Otherwise REVERT. Repeat.
```

## Data Pipeline

- **Wide-grid option data**: 82 contracts per day (ATM +/- 100pt) from Polygon flat files
- **55 features**: 39 market (SPX, VIX, volume, Greeks) + 16 option-enriched (moneyness, per-strike volume, flow ratio, spread proxy)
- **Honest labels**: Risk-grid search (64 stop/target/hold combos), gate=True only when profitable, 28% of signal bars are no-trade
- **Real-time SPX**: estimated via call-put parity (not fixed opening ATM)
- **Forward-filled**: matches live IBKR behavior (stale quotes, not missing data)

## Project Structure

```
v2/                     # The active system
  program.md            # Protocol (read this first)
  COMMANDS.md           # What you can tell the AI to do
  train.py              # Model + training loop (MUTABLE)
  core/
    policy.py           # Trading parameters (MUTABLE)
    schema.py           # TradeIntent contract
    simulator.py        # Trade simulation engine
    metrics.py          # Score formula
    features.py         # 55-feature spec (39 market + 16 enriched)
    labels.py           # Label generation
  replay.py             # Evaluation harness
  ops/
    deploy.sh           # Akash H100 GPU lifecycle
    run_experiment.py   # Train + replay + score (one command)
    inner_loop.py       # Keep/revert + session limits
    monitor.py          # Human oversight dashboard
  pipeline/
    download_wide_grid.py  # Download 82 contracts/day from Polygon
    build_v2_dataset.py    # Build data with enriched features + honest labels
    extract_raw.py         # Extract OHLCV from raw Polygon cache
  live/                 # IBKR execution (stubs)
  docs/                 # Design specs + domain knowledge

archive/                # Frozen v1 system (reference only)
```

## Scoring

Models are evaluated on a $10,000 equity curve with SPX $100 multiplier:

```
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```

Hard gates: 30+ trades, 15+ traded days, both directions, max 20% drawdown. Must beat random, ATM-always, and simple-rules baselines.

## Quick Start

```bash
# Download wide-grid option data from Polygon (first time, ~10 min)
python -m v2.pipeline.download_wide_grid

# Build dataset with enriched features + honest labels (~30 sec)
python -m v2.pipeline.build_v2_dataset

# Boot Akash H100 and upload
./v2/ops/deploy.sh boot
./v2/ops/deploy.sh start

# Run experiment on GPU
ssh root@<gpu> "python v2/ops/run_experiment.py --id baseline"

# Monitor
python v2/ops/monitor.py
```

## Safety

- Paper trading only. No real money.
- 1 SPX contract max. Long calls/puts only. 0DTE only.
- Dynamic stop-loss (30-60% range), learned by model risk head.
- 5% daily loss cap. 5-bar cooldown after stops.
- EOD flatten at 15:59 ET.
