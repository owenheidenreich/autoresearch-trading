# autoresearch-trading

Autonomous SPX 0DTE options trader built with an autoresearch loop: an AI agent iteratively evolves a neural network trading model through hundreds of experiments on GPU, then deploys the best model to live paper trading on IBKR.

## How It Works

```
data (prepare.py) → training (run_loop.py on Akash GPU) → replay validation → IBKR paper trading
```

1. **Data pipeline** builds a feature tensor from SPY/SPX/VIX bars + SPXW option chains
2. **Autoresearch loop** runs on an Akash H100 — Claude Sonnet proposes code mutations to `train.py`, each is trained and scored, improvements are kept
3. **Replay engine** validates the trained model against historical data
4. **Live paper trading** runs the model against real-time IBKR market data

## Core Design Principles

- **The model IS a trader.** Training simulates minute-by-minute 0DTE options trading — the same game as live IBKR paper trading. No peeking at future prices, no batch classification.
- **No hardcoded exits.** The model's gate head learns when to enter and exit. The 30% stop loss is an emergency backstop only.
- **Data integrity first.** `data.pt` must always include option chain sidecar data. Building without it silently breaks all experiments.
- **Train on GPU, not locally.** All training runs deploy to Akash H100s via `deploy.sh`.

## Quick Start

### Environment setup
```bash
cp .env.example .env  # Add ANTHROPIC_API_KEY, POLYGON_API_KEY, POLYGON_S3_KEY_ID, POLYGON_S3_SECRET
set -a && source .env && set +a
```

### Deploy a training run
```bash
./infra/deploy.sh boot                              # Spin up H100 on Akash (~2 min)
./infra/deploy.sh start --hours 8 --max-experiments 200  # Upload code + data, start loop
./infra/deploy.sh logs                              # Tail experiment output
./infra/deploy.sh status                            # GPU + experiment progress
python3 tools/monitor.py                            # Rich terminal dashboard
./infra/deploy.sh stop                              # Kill loop → download → close
```

### Run IBKR paper trading
```bash
python tools/paper_live.py --context-only           # Refresh market context bundle
python tools/paper_live.py --dry-run --max-minutes 60  # Dry-run (no orders)
python tools/paper_live.py --max-minutes 390        # Live paper session
```

### Replay validation
```bash
cd training
python3 replay.py --date 2026-03-17 --output replay-trades.csv
```

## Project Map

```text
training/
  prepare.py            # Data pipeline: SPY/SPX/VIX bars + SPXW options → data.pt
  train.py              # Model architecture + training loop (agent modifies THIS)
  run_loop.py           # Autoresearch orchestration: Claude → mutate → train → score → keep/revert
  replay.py             # Replay simulation engine + model loading
  program.md            # Strict contract for the autonomous loop (injected into system prompt)
  lab_notebook.md       # Persistent experiment context: improvements + dead ends
  live/
    service.py          # Live trading service: IBKR connection, order management
    decision.py         # Model inference engine for live decisions
    context.py          # Real-time feature construction from IBKR + Polygon data

tools/
  monitor.py            # Rich terminal dashboard (local + remote GPU monitoring)
  paper_live.py         # IBKR paper trading CLI entry point
  replay_battery.py     # Multi-day replay validation suite
  live_order_parity_report.py   # Verify live orders match model signals
  live_feature_parity_report.py # Verify live features match training features
  ib_account_snapshot.py        # IBKR account status check

infra/
  deploy.sh             # Akash GPU lifecycle: boot/start/sync/stop/ssh/logs/status
  deploy-autoresearch.yaml  # Akash SDL (H100/A100, 64GB RAM, PyTorch 2.5.1)
  start_loop.sh         # Remote loop launcher
  watchdog.sh           # GPU health + process monitor

docs/
  CLAUDE.md             # Project guide for Claude (codebase navigation)
  0dte-domain-knowledge.md  # SPX 0DTE options trading domain primer
```

## Architecture

### Model
- Two-head output: **Gate** `[NO_TRADE, TRADE]` + **Direction** `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]`
- 8 effective actions: `DO_NOTHING`, 6 entry types, `EXIT`
- 32 features (v2): SPY/SPX/VIX technicals + options greeks + time features (reduced from 70 for signal density)
- Position-aware inference: model sees current P&L, hold time, and bars held

### Autoresearch Loop
- Claude Sonnet proposes mutations to `train.py` (architecture, loss functions, hyperparameters)
- Each experiment trains for ~4 min on H100, scored by profit factor + trades/day + secondary metrics
- Multi-objective scoring with 9 tunable knobs (SCORE_* env vars)
- Prefetch pipeline: speculative Claude API calls during training for throughput
- Auto-sync downloads results to local machine every 30s

### Scoring
The score formula balances profitability and trading realism:
- Primary: log profit factor × trade frequency bonus
- Secondary gates: drawdown, consecutive losses, stop rate, hold time
- All secondary knobs default neutral — the agent activates them as needed

## Canonical Runtime Layout

```text
results/
  current_run.txt                    # Active run pointer
  promoted/                          # Cross-run best models
  run-YYYY-MM-DD-HHMMSS/
    experiments.v2.jsonl             # Structured experiment log
    results.tsv                      # Human-readable experiment summary
    status.json                      # Current loop state
    run_metadata.json                # Run config snapshot
    artifacts/exp-<id>/              # Per-experiment: reasoning, prompts, code, logs
```

## Prompt Control Plane

- `training/program.md` — strict autonomous contract (what the agent can/cannot modify)
- `training/lab_notebook.md` — persistent cross-run context (improvements + dead ends)
- Both injected into Claude's system prompt with Anthropic prompt caching

## Known Issues

- `deploy.sh stop` has reliability issues with SSH exit codes and interactive prompts
- Context bundle must be refreshed before each live session (features evolve with training)
- VIX data has occasional gaps from IBKR historical data API

## Safety

- Real-money trading is not enabled
- Live paper-trading is under active validation
- All positions are 1 SPX contract, long calls/puts only
- 30% hard stop loss on all positions
