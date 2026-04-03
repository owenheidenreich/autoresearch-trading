# ART² -- Autonomous Research Trader

SPX 0DTE options trading system. A neural network learns complete trading decisions (entry, direction, strike, risk management) from minute-bar market data.

Built on Karpathy's autoresearch pattern: the AI proposes hypotheses, trains models on GPU, evaluates against held-out data, keeps what improves, reverts what doesn't. Humans program the research organization, not the research itself.

## How It Works

```
1. AI edits v2/train.py (model architecture, loss, hyperparams)
2. Trains on 859 days of historical SPX data (Tier 3 oracle labels)
3. Replays on 60 held-out days the model never saw
4. Scores the equity curve: Sortino * consistency * drawdown guard
5. Score improved? KEEP. Otherwise REVERT. Repeat.
```

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
    features.py         # 39-feature spec
    labels.py           # Oracle label generation
  replay.py             # Evaluation harness
  ops/
    run_experiment.py   # Train + replay + score (one command)
    inner_loop.py       # Keep/revert + session limits
    monitor.py          # Human oversight dashboard
    artifact.py         # Self-describing model packages
  pipeline/
    build_dataset.py    # Data construction (Tier 1/2/3 labels, 4-way split)
  live/                 # IBKR execution (stubs, Phase 5)
  docs/                 # Design specs + domain knowledge

archive/                # Frozen v1 system (reference only)
```

## Scoring

Models are evaluated on a $50,000 equity curve with SPX $100 multiplier:

```
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```

Hard gates: 30+ trades, 15+ traded days, both directions, max 20% drawdown. Must beat random, ATM-always, and simple-rules baselines.

## Quick Start

```bash
# Rebuild dataset (Tier 3 oracle labels, 30-60 min)
python -m v2.pipeline.build_dataset --tier 3

# Run one experiment
python v2/ops/run_experiment.py --id baseline

# Evaluate
python -m v2.replay --model v2/model.pt --mask promote

# Monitor
python v2/ops/monitor.py
```

## Safety

- Paper trading only. No real money.
- 1 SPX contract max. Long calls/puts only. 0DTE only.
- Dynamic stop-loss (10-65% range), learned by model.
- 5% daily loss cap. 5-bar cooldown after stops.
- EOD flatten at 15:59 ET.
