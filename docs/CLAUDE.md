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

**Verification status (2026-03-20):** End-to-end feature parity confirmed — all three stages (training via `prepare.py`, replay via `replay.py`, live via `live/features.py`) call the same `compute_features()` and `normalize_features_with_context()` from `prepare.py`, use the same 32 `FEATURE_NAMES`, and the same data sources. IBKR paper trading confirmed working: LMT order fills (OCO bracket), live type-1 market data, option Greeks <1s staleness, and correct model inference.

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
- [tools/ib_probe.py](../tools/ib_probe.py) — IBKR connectivity probe: tests SPX/SPY/VIX bars + SPXW option chain availability
- [tools/ib_entitlements.py](../tools/ib_entitlements.py) — IBKR data entitlement checker: verifies live (non-delayed) market data per instrument

## Critical Design Rules

1. **Training = live trading.** The `evaluate_trades()` simulation must be the exact same game as IBKR paper trading. No future peeking, no batch classification.
2. **No hardcoded take-profit.** The model's gate head learns exits. 30% SL is emergency backstop only.
3. **data.pt must include options.** Never build without option chain sidecar data — causes silent total failure.
4. **Train on Akash, not locally.** User's laptop can't handle training. Always use `deploy.sh`.
5. **Model architecture evolves.** `train.py` may define custom modules (PositionStateGenerator, etc.) that differ from the default TradingModel in replay.py. The `load_model()` function handles this via dynamic class loading.
6. **Feature parity is verified.** Training, replay, and live all call `compute_features()` + `normalize_features_with_context()` from `prepare.py`. Confirmed end-to-end on 2026-03-20 with IBKR paper trading. Do not introduce separate feature computation paths.

## Warm-Start vs Fresh-Start Rules

When deploying to Akash, `best_model.pt` can be uploaded as a warm-start checkpoint. **Not all changes are warm-start compatible.** Use this decision matrix:

### MUST fresh-start (tensor shape incompatibility — model will crash or silently corrupt)
- **Feature count change** (e.g., 70→32) — `input_proj` weight dimensions change
- **Position state dimension change** (e.g., 3→5) — `position_proj` weight dimensions change
- **D_MODEL, DEPTH, or N_HEADS change** — all transformer layer weights change shape
- **New module added to model class** (e.g., BalancedStrikeGate) — state_dict keys mismatch
- **Action space change** (gate/direction head output sizes)

### SHOULD fresh-start (objective landscape changed — warm weights optimized for wrong goal)
- **Score formula changes** (SCORE_* env vars, penalty thresholds) — model was rewarded for different behavior
- **Position sizing changes** (RISK_PER_TRADE → POSITION_RISK_TARGET) — dollar P&L scale changes
- **Evaluation mechanics changes** (stop-loss %, max hold, cooldown, cost model) — what counts as "good" changes
- **Multiple simultaneous infrastructure changes** — compounding effects make warm-start unreliable

### Safe to warm-start (training dynamics only)
- Hyperparameter tuning (LR, batch size, dropout, weight decay)
- Loss weight adjustments (GATE_LOSS_WEIGHT, EXIT_LOSS_WEIGHT, etc.)
- Bias tuning on existing heads
- Label smoothing changes
- run_loop.py / program.md / lab_notebook.md changes (prompt-only, no model impact)

### How to fresh-start
Move `training/best_model.pt` aside before deploying:
```bash
mv training/best_model.pt training/best_model.pt.bak
```
The deploy script skips warm-start upload when the file doesn't exist.

## Project Knowledge Base

### Lessons Learned
- **The autoresearch agent fixates.** If lab_notebook.md says "reduce X", the agent will run 20+ experiments all targeting X. Keep "Next Priorities" open-ended ("beat the best score") rather than prescriptive ("reduce stop-loss rate").
- **New nn.Module subclasses almost never work.** 0/16+ attempts. The 4-minute training budget isn't enough for fresh parameters to converge. Bias tuning on existing heads is consistently more effective.
- **Prompt caching saves ~50% API cost.** See [API Cache Architecture](#api-cache-architecture) below for details.
- **The model overfit to 2022-2024.** Full backtest shows PF degrades from 7.5 → 2.3 → 0.78 across three time phases. Winners shrink from 121% to 27%. The model compensates by trading MORE, which makes it worse. This is a training signal problem, not an architecture problem.
- **Position sizing matters for backtest realism.** Fixed 1-contract sizing caused P&L deceleration as the account grew. Scaled sizing (`n_contracts = max(1, floor(balance * 0.05 / cost))`) produces more realistic equity curves.
- **The agent will game its own scoring metric.** Score_config knobs (SCORE_DRAWDOWN_PENALTY, SCORE_HOLD_BONUS, etc.) only affect post-training evaluation, NOT model training. The agent tuned these to inflate scores 6x without improving PF or TPD. Score config is now locked in train.py with a validation guard in run_loop.py.

### Common Issues
- **SSH "Permission denied" during deploy** — usually transient. Retry. If persistent, check that `.deploy-state` has correct `SSH_PORT` and that sshpass is installed.
- **"Credit balance too low" API error** — check which workspace the API key belongs to at console.anthropic.com. Credits are per-workspace.
- **deploy.sh stop hangs** — the interactive confirmation prompt blocks in non-TTY. Known issue.
- **Lab notebook gets overwritten by sync** — the auto-sync process copies lab_notebook.md from the remote. Kill the sync (`pkill -f sync`) before editing locally.
- **Equity curve HTML has gaps** — for multi-day charts, use sequential x-axis mapping (already implemented in replay.py). Plotly rangebreaks can't handle 800+ non-trading dates.
- **chunk_details empty in training output** — if `evaluate_trades` returns fewer than 5 trades, chunk_details is empty. This is expected for very selective models.

### Key Constants
| Constant | Value | Location | Notes |
|----------|-------|----------|-------|
| STARTING_CAPITAL | $10,000 | prepare.py | Account sim starting balance |
| SPX_MULTIPLIER | $100 | prepare.py | SPX option contract multiplier |
| POSITION_RISK_TARGET | 0.05 (5%) | prepare.py | Max account fraction per trade |
| STOP_LOSS_PCT | -0.30 (-30%) | prepare.py | Emergency stop-loss on premium |
| MAX_HOLD_BARS | 60 | prepare.py | Max 60 minutes per position |
| COOLDOWN_BARS | 5 | prepare.py | Bars between trades after stop-loss |
| NO_TRADE_BEFORE_BAR | 30 | prepare.py | Skip first 30 min of session |
| Train/val split | 70/30 | prepare.py | Temporal split, no stratification |

## API Cache Architecture

The autoresearch loop uses Anthropic's prompt caching to reduce API costs. Each Claude call includes a system prompt with three cached blocks. **Block ordering is critical** — caching uses prefix matching, so if Block N changes, all subsequent blocks are invalidated.

### System Prompt Block Layout

```
Block 1: Instructions + program.md     (~3k tokens)   — never changes
Block 2: train.py                      (~8.5k tokens)  — changes only on kept experiments
Block 3: Lab notebook                  (~1.5k tokens)  — changes every experiment (dead-end entries)
```

Built in `build_system_prompt()` in [run_loop.py](../training/run_loop.py). Each block has `cache_control: {"type": "ephemeral", "ttl": "1h"}`.

### Why Order Matters

Prompt caching matches from the **prefix**. When a block changes, it and everything after it must be rewritten (cache write). Putting volatile content last minimizes invalidation:

- **Lab notebook changes every experiment** — `update_lab_notebook_dead_end()` appends after each rejected experiment. If this were Block 2, it would invalidate train.py (~8.5k tokens) on every call.
- **train.py changes only on kept experiments** (~1 in 5). As Block 2, it stays cached for 4-5 consecutive calls.
- With this ordering, a typical experiment cache-reads ~11.5k tokens and only rewrites ~1.5k (the notebook).

**Before this fix (2026-03-21):** Lab notebook was Block 2, train.py Block 3. Cache hit rate: ~28%. After reorder: ~77%.

### Pricing (Claude Sonnet 4)

| Token Type | Cost per MTok | Notes |
|------------|---------------|-------|
| Uncached input | $3.00 | First call or cache miss |
| Cache write (1h TTL) | $6.00 | 2x base — written on first use or invalidation |
| Cache read | $0.30 | 0.1x base — the goal |
| Output | $15.00 | Dominant cost (~76%), inherent to full-file code generation |

Minimum cacheable block: 1,024 tokens (Sonnet 4). 1h TTL is GA (no beta header needed).

### Cost Tracking

`run_loop.py` tracks cumulative token usage in `_cumulative_tokens` and writes cost data to `status.json` under the `api_cost` key after each experiment. The monitor dashboard ([tools/monitor.py](../tools/monitor.py)) reads this to display:

- **API Cost**: cumulative dollar spend for the run
- **Cache Hit %**: fraction of input tokens served from cache
- **$/Experiment**: average cost per experiment

Check the experiment log for `[cost]` lines showing per-experiment breakdowns and cache hit rates. A healthy run should show cache hit >70% after the first experiment.

### Typical Run Costs

| Run Duration | Experiments | Approx Cost | Notes |
|-------------|-------------|-------------|-------|
| 4 hours | ~30 | ~$20-25 | With cache optimization |
| 8 hours | ~60 | ~$40-50 | Standard overnight run |

Output tokens are the dominant cost. Cache optimization primarily saves on input tokens (~$4/4h run).

## Environment Variables

```
ANTHROPIC_API_KEY     — Claude API key (required for autoresearch loop)
POLYGON_API_KEY       — Market data API key (required for context refresh)
POLYGON_S3_KEY_ID     — S3 access key for flat files
POLYGON_S3_SECRET     — S3 secret key for flat files
TRAIN_*               — Hyperparameter overrides (TRAIN_LOOKBACK, TRAIN_LR, etc.)
SCORE_*               — Score formula (LOCKED — agent cannot modify, see program.md)
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
