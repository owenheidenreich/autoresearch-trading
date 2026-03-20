# CLAUDE.md — Project Guide

Quick reference for navigating the autoresearch-trading codebase.

## What This Project Does

Autonomous SPX 0DTE options trading system. An AI agent (Claude Sonnet) iteratively evolves a PyTorch trading model through hundreds of experiments on Akash H100 GPUs, then the best model runs live paper trading on IBKR.

## Key Files by Category

### Data Pipeline
- [training/prepare.py](../training/prepare.py) — Builds `data.pt` from SPY/SPX/VIX bars + SPXW option chains. Defines feature contract (32 features, v2 reduced from 70), action space (8 actions), and scoring formula. **Read-only by the autoresearch agent.**
- `~/.cache/autoresearch-trading/features/data.pt` — The training tensor (~201MB, ~383k bars)

### Autoresearch Loop
- [training/run_loop.py](../training/run_loop.py) — Main orchestration: calls Claude → validates mutation → trains → scores → keeps or reverts. Contains prefetch pipeline, retry/repair logic, anomaly detection, and auto-sync.
- [training/program.md](../training/program.md) — The strict contract injected into Claude's system prompt. Defines what the agent can/cannot modify.
- [training/lab_notebook.md](../training/lab_notebook.md) — Persistent cross-run memory: improvements log + dead ends. Also injected into system prompt.
- [training/train.py](../training/train.py) — The file the autoresearch agent mutates. Contains model architecture, loss functions, training loop.

### Model & Replay
- [training/replay.py](../training/replay.py) — Replay simulation engine + model loading. The `evaluate_trades()` function IS the trading simulation. Also contains `load_model()` and `_load_model_class_from_train_py()` for loading evolved architectures.

### Live Paper Trading
- [tools/paper_live.py](../tools/paper_live.py) — CLI entry point for IBKR paper trading
- [training/live/service.py](../training/live/service.py) — Trading service: IBKR connection, session management, order lifecycle
- [training/live/decision.py](../training/live/decision.py) — Model inference engine: loads checkpoint, runs forward pass, produces trading decisions
- [training/live/context.py](../training/live/context.py) — Real-time feature construction from IBKR + Polygon data → context bundle

### Infrastructure
- [infra/deploy.sh](../infra/deploy.sh) — Akash GPU deployment lifecycle (boot/start/sync/stop/ssh/logs/status). Main entry point for all remote operations.
- [infra/deploy-autoresearch.yaml](../infra/deploy-autoresearch.yaml) — Akash SDL manifest (H100/A100, 64GB RAM, PyTorch 2.5.1)
- [infra/start_loop.sh](../infra/start_loop.sh) — Remote loop launcher (run by deploy.sh start)
- [infra/watchdog.sh](../infra/watchdog.sh) — GPU health + process monitoring daemon

### Monitoring & Tools
- [tools/monitor.py](../tools/monitor.py) — Web dashboard (http://localhost:8420): GPU gauges, experiment table, metric charts, Claude reasoning stream, live train.py viewer, log tail. Auto-detects remote from `.deploy-state`.
- [tools/replay_battery.py](../tools/replay_battery.py) — Multi-day replay validation suite
- [tools/live_order_parity_report.py](../tools/live_order_parity_report.py) — Verify live IBKR orders match model signals
- [tools/live_feature_parity_report.py](../tools/live_feature_parity_report.py) — Verify live features match training features

## Critical Design Rules

1. **Training = live trading.** The `evaluate_trades()` simulation must be the exact same game as IBKR paper trading. No future peeking, no batch classification.
2. **No hardcoded take-profit.** The model's gate head learns exits. 30% SL is emergency backstop only.
3. **data.pt must include options.** Never build without option chain sidecar data — causes silent total failure.
4. **Train on Akash, not locally.** User's laptop can't handle training. Always use `deploy.sh`.
5. **Model architecture evolves.** `train.py` may define custom modules (PositionStateGenerator, etc.) that differ from the default TradingModel in replay.py. The `load_model()` function handles this via dynamic class loading.

## Environment Variables

```
ANTHROPIC_API_KEY     — Claude API key (required for autoresearch loop)
POLYGON_API_KEY       — Market data API key (required for context refresh)
POLYGON_S3_KEY_ID     — S3 access key for flat files
POLYGON_S3_SECRET     — S3 secret key for flat files
TRAIN_*               — Hyperparameter overrides (TRAIN_LOOKBACK, TRAIN_LR, etc.)
SCORE_*               — Score formula tuning knobs (all default neutral)
DEPOSIT_AKT           — Akash deployment deposit (default 15, use 50 for 8h runs)
```

## Results Layout

```
results/current_run.txt              → active run name
results/run-YYYY-MM-DD-HHMMSS/
  experiments.v2.jsonl               → structured experiment log
  results.tsv                        → human-readable summary
  status.json                        → current loop state
  artifacts/exp-<id>/                → per-experiment: reasoning, prompts, code, logs
results/promoted/                    → cross-run best models
```
