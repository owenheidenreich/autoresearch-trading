# Project Reference

Single reference for navigating the autoresearch-trading codebase. For process/lifecycle details, see the [operating manual](../../.claude/rules/art2-operating-manual.md) (auto-loaded every session).

## What This Project Does

Autonomous SPX 0DTE options trading system. Opus (strategist) directs Sonnet agents (code writers) to iteratively evolve a PyTorch trading model on Akash H100 GPUs, then the best model runs live paper trading on IBKR.

## Key Files

### Data Pipeline
- [training/prepare.py](../../training/prepare.py) — Builds `data.pt` from SPY/SPX/VIX bars + SPXW option chains. Defines feature contract (37 features, v3), action space (8 actions), and scoring formula. Uses incremental cache updates (`_incremental_update()`).
- `~/.cache/autoresearch-trading/features/data.pt` — Training tensor (~201MB, ~383k bars)
- `~/.cache/autoresearch-trading/data/` — Raw caches: `spy_1min.pkl`, `spx_1min.pkl`, `vix_1min.pkl` (append-only), `spxw/` and `spxw_chain/` (per-day incremental)

### Training Loop
- [tools/inner_loop.py](../../tools/inner_loop.py) — Agent mode: `experiment --mutation FILE --summary "hypothesis"`. Atomic (validate → backup → upload → train → score → keep/revert). Also provides PBT mode (`pbt-init` / `pbt-run` / `pbt-status`).
- [training/run_loop.py](../../training/run_loop.py) — Utility library: validation, anomaly detection, scoring helpers. Imported by inner_loop.py.
- [training/train.py](../../training/train.py) — v7 four-head model architecture + training loop. The file Sonnet agents mutate.
- [training/program.md](../../training/program.md) — Agent contract. Injected into Sonnet prompts (sequential mode only; PBT does not use it).
- [training/lab_notebook.md](../../training/lab_notebook.md) — Cross-run memory: dead ends + priorities. Read by Opus (strategist) and injected into Sonnet prompts (sequential mode only). PBT mode is purely mechanical and does not read this file.
- [training/exit_policy.py](../../training/exit_policy.py) — RL exit agent (standalone, 11-dim obs → 3-action MLP). Not yet integrated into live decision.py.

### Replay & Live Trading
- [training/replay.py](../../training/replay.py) — `evaluate_trades()` IS the trading simulation. Also loads evolved architectures.
- [tools/paper_live.py](../../tools/paper_live.py) — IBKR paper trading CLI entry point
- [training/live/service.py](../../training/live/service.py) — `PaperTradingService`: session orchestrator, bar loop, position tracking, cooldown enforcement
- [training/live/decision.py](../../training/live/decision.py) — `ModelDecisionEngine`: four-head inference, position/account state building, entry intent, exit policy
- [training/live/execution.py](../../training/live/execution.py) — `OCOExecutionEngine`: LMT orders, dynamic stop, OCO brackets, IBKR order lifecycle
- [training/live/features.py](../../training/live/features.py) — `LiveFeatureEngine`: 5s→1m bar aggregation, 37-feature computation, normalization
- [training/live/context.py](../../training/live/context.py) — Historical context bundle (30-day rolling)
- [training/live/resolver.py](../../training/live/resolver.py) — SPXW 0DTE contract resolution (ATM/OTM strikes)
- [training/live/contracts.py](../../training/live/contracts.py) — Typed data contracts between live components

**Feature parity:** All 37 features (v3) confirmed across training/replay/live — shared `compute_features()` + `normalize_features_with_context()`.

**Architecture parity:** Four-head model (v7: gate + direction + value + risk) with 7-dim position state + 4-dim account state. Risk head provides learned stop distance, position sizing, and conviction. Value head (v7) is a binary exit classifier: logit → sigmoid → exit probability, compared against conviction-adjusted threshold (0.5 + conviction × 0.2). Exit priority: stop_loss > model_exit > value_exit > max_hold > EOD.

### Infrastructure & Tools
- [infra/deploy.sh](../../infra/deploy.sh) — Akash GPU lifecycle (boot/start/sync/stop/ssh/logs/status)
- [tools/art2.py](../../tools/art2.py) — ART² orchestrator (9-phase meta-loop)
- [tools/monitor.py](../../tools/monitor.py) — Web dashboard (localhost:8420)
- [tools/daily_pipeline.py](../../tools/daily_pipeline.py) — Pre-market data refresh + weekly retrain
- [tools/ibkr_analyze.py](../../tools/ibkr_analyze.py) — IBKR session analyzer
- [tools/live_dash.py](../../tools/live_dash.py) — Live trading dashboard
- [tools/live_feature_parity_report.py](../../tools/live_feature_parity_report.py) — Feature parity verification
- [tools/verify_context_parity.py](../../tools/verify_context_parity.py) — Context bundle parity check

## Critical Design Rules

1. **Training = live trading.** `evaluate_trades()` must be the exact same game as IBKR paper trading.
2. **No hardcoded take-profit.** Gate head learns exits. Dynamic stop-loss (15%-60%) adapts per-trade.
3. **data.pt must include options.** Never build without option chain sidecar — causes silent total failure.
4. **Train on Akash, not locally.** Always use `deploy.sh`.
5. **No custom modules.** New nn.Module subclasses: 0/16+ failure rate.
6. **Feature parity is verified.** Do not introduce separate feature computation paths.
7. **Incremental cache updates.** Never delete SPY/SPX/VIX caches to force re-download.
8. **ART² is mechanical, Opus is strategic.** art2.py manages subprocesses; Opus makes decisions.
9. **No API cost.** Inner loop uses Max subscription Sonnet agents. Only Akash GPU compute costs.
10. **Human review before every training run.** REVIEW phase pauses daemon for approval.

## Model Architecture (v7 Four-Head)

| Head | Output | Activation | Range | Purpose |
|------|--------|-----------|-------|---------|
| Gate | (batch, 2) | softmax | [0,1] probability | Enter/exit decision |
| Direction | (batch, 6) | softmax | 6 actions | Strike + direction selection |
| Value | (batch, 1) | raw logit | scalar | Binary exit classifier (v7: BCE on sparse exit labels) |
| Risk | (batch, 3) | sigmoid/tanh | see below | Account-aware risk management |

**Risk head outputs:**
- `stop_pct`: sigmoid → [0.15, 0.60] — dynamic stop distance
- `size_frac`: sigmoid → [0, 1] — position size fraction
- `conviction`: tanh → [-1, +1] — trade conviction signal

**Position state (7 dims, fed to gate + value heads):**

| Dim | Field | Normalization |
|-----|-------|--------------|
| 0 | in_trade | 0 or 1 |
| 1 | bars_held | / BARS_PER_DAY |
| 2 | unrealized_pnl | tanh(× PNL_TANH_SCALE) |
| 3 | account_health | balance / starting_capital |
| 4 | loss_streak | consecutive_losses / 3.0 |
| 5 | best_pnl_since_entry | tanh(× BEST_PNL_TANH_SCALE) |
| 6 | bars_since_pnl_high | / BARS_PER_DAY |

**Account state (4 dims, fed to risk head only):**

| Dim | Field | Normalization |
|-----|-------|--------------|
| 0 | account_growth_ratio | balance / starting_capital |
| 1 | log_account_size | log10(balance / 1000) / 3.0 |
| 2 | daily_pnl_fraction | daily_pnl / balance |
| 3 | win_rate_20 | wins / last 20 trades |

**Transformer backbone:**
- Input: (batch, lookback=120, 37 features)
- d_model=64, n_heads=4, depth=3, ff_mult=3, dropout=0.3
- Pre-LN, causal masking, GELU activation
- Learned positional embedding

**Direction actions:** CALL_ATM (0), CALL_OTM5 (1), CALL_OTM10 (2), PUT_ATM (3), PUT_OTM5 (4), PUT_OTM10 (5)

**Bias initialization (domain knowledge):**
- Gate: pro-trade (+0.3 TRADE, -0.3 NO_TRADE)
- Direction: ATM favored (+0.20), OTM penalized (-0.15 to -0.20)
- Risk: mid-range stop (sigmoid(0)=0.5), conservative sizing (sigmoid(-1)≈0.27), neutral conviction

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| NUM_FEATURES | 37 (v3) | prepare.py |
| STARTING_CAPITAL | $10,000 | prepare.py |
| SPX_MULTIPLIER | $100 | prepare.py |
| POSITION_RISK_TARGET | 0.05 (5%) | prepare.py |
| DYNAMIC_STOP_BASE | 0.35 | prepare.py |
| DYNAMIC_STOP_MIN / MAX | 0.15 / 0.60 | prepare.py |
| MAX_HOLD_BARS | 390 | prepare.py |
| STOP_COOLDOWN_BARS | 5 | prepare.py (enforced in live too) |
| NO_TRADE_BEFORE_BAR | 30 | prepare.py |
| BARS_PER_DAY | 390 | prepare.py |
| FORWARD_BARS | 30 | prepare.py |
| Train/val split | 70/30 | prepare.py |
| POSITION_STATE_DIM | 7 | train.py |
| ACCOUNT_STATE_DIM | 4 | train.py |
| LOOKBACK | 120 | train.py (env: TRAIN_LOOKBACK) |
| D_MODEL | 64 | train.py |
| N_HEADS | 4 | train.py |
| DEPTH | 3 | train.py |
| BATCH_SIZE | 512 | train.py |
| LR | 2.5e-4 | train.py |

### Loss Weights (env-var tunable, score formula LOCKED)

| Weight | Default | Env Var |
|--------|---------|---------|
| Gate loss | 0.5 | TRAIN_GATE_W |
| Direction loss | 2.5 | TRAIN_DIR_W |
| PnL alignment | 0.5 | TRAIN_PNL_W |
| Exit loss | 0.40 | TRAIN_EXIT_W |
| Confidence loss | 0.10 | TRAIN_CONF_W |
| Value loss | 0.5 | TRAIN_VALUE_W |
| Risk loss | 0.2 | TRAIN_RISK_W |

## Warm-Start vs Fresh-Start

### MUST fresh-start (tensor shape incompatibility)
- Feature count change — `input_proj` dimensions change
- Position state dimension change — `position_proj` dimensions change
- D_MODEL, DEPTH, or N_HEADS change — transformer weights change shape
- New module added — state_dict keys mismatch (e.g. adding value head)
- Action space change
- **Version gate enforced:** train.py and replay.py reject checkpoints missing `has_value_head`, `has_risk_head`, or with `position_state_dim < 7`. Incompatible checkpoints cause training from scratch automatically.

### SHOULD fresh-start (objective landscape changed)
- Score formula changes, position sizing changes, evaluation mechanics changes, multiple simultaneous infrastructure changes

### Safe to warm-start
- Hyperparameter tuning, loss weight adjustments, bias tuning, label smoothing, prompt-only changes

```bash
# Fresh-start: move checkpoint aside
mv training/best_model.pt training/best_model.pt.bak
```

## ART² Subcommands

| Command | What it does |
|---------|-------------|
| `time` | Check current time context (PT/ET, market status) |
| `train --minutes N` | Full training cycle on Akash (preflight → boot → upload → train → download) |
| `analyze` | Collect metrics from latest run into `analysis.json` |
| `replay` | Backtest `best_model.pt` — PF, win rate, time-of-day, hold times |
| `diagnose` | Compare training vs replay metrics, detect tunnel vision |
| `research` | Deep trade analysis vs domain knowledge → hypotheses |
| `viability` | Profitability assessment → VIABLE / PROMISING / INCONCLUSIVE / NOT VIABLE |
| `report` | Generate briefing for Opus (metrics + research + lab notebook) |
| `verify --dates N` | OOS validation + IBKR probe + gaming detection |
| `cycle --minutes N` | Full pipeline: train → analyze → replay → diagnose → research → viability → report |
| `daemon --minutes N` | 9-phase lifecycle with REVIEW gate. `--max-cycles N` to limit. |
| `autonomous --minutes N` | Market-aware: train when closed, monitor when open |
| `market` | Check market status + IBKR compatibility gate |
| `status` | Current ART² state + cycle history |
| `init` | Initialize `results/art2/` structure |
| `ibkr-analyze` | Parse IBKR paper trading audit, produce metrics |

**Control files:** `results/art2/REVIEW` (remove to approve), `results/art2/STOP` (create to exit), `results/art2/PAUSED` (remove to resume).

## Training Modes

Two distinct modes for running experiments. Choose based on model maturity:

**Sequential Warm-Start** — `inner_loop.py experiment --summary "hypothesis"`
- Each experiment builds on the last kept model. One at a time, score-gated.
- Use for: fresh starts, architecture changes, early convergence, debugging training signals.
- Opus or Sonnet proposes targeted train.py edits per experiment.

**PBT (Population-Based Training)** — `inner_loop.py pbt-init && inner_loop.py pbt-run`
- Population of N members competing per generation with env-var overrides.
- Evolutionary: elite carry-forward, exploit top-25%, explore top-50%.
- Use for: multi-parameter optimization once model has a stable baseline.
- Don't use PBT on a fresh start — the model needs to learn basic behavior first.
- Anti-stagnation: injects random members after 2 stalled generations.

**Decision guide:**
- Model just had architecture/label changes → Sequential (let it converge)
- Score stuck after 5+ sequential experiments → Switch to PBT
- PBT stagnation (all members converge) → Back to sequential with structural changes

## Inner Loop Subcommands

| Command | What it does |
|---------|-------------|
| `init` | Create run directory, backup train.py, write status.json |
| `experiment --summary "hypothesis"` | Atomic experiment (validate → upload → train → score → keep/revert). Optional: `--mutation FILE` |
| `status` | Show current run state |
| `pbt-init --population N` | Initialize PBT population (generalists + specialists). Optional: `--generations N`, `--focus`, `--specialists` |
| `pbt-run` | Run PBT: iterate through members and generations. Optional: `--time-budget N` |
| `pbt-status` | Print PBT state as JSON |

## IBKR Live Trading Reference

### Session Lifecycle

1. `paper_live.py` loads `.env`, connects to IB Gateway (port 4001 live / 4002 paper)
2. `service.py:run_session()` runs preflight: IBKR probe, entitlement check, context bundle load
3. **Startup cleanup**: `cancel_all_open_orders()` cancels all orphaned orders from previous sessions
4. Stream subscribes to SPX 5-second bars via `FiveSecondMinuteAggregator`

### Bar Loop (1-minute cycle)

Each minute:
1. Aggregate 5-sec bars → 1-min bar
2. `LiveFeatureEngine.compute_snapshot()` → 37 normalized features + completeness score
3. Skip if completeness < 0.65
4. Update position state (unrealized P&L via `resolver.quote_mid()`)
5. Feed account state to risk head (balance, daily P&L, win rate)
6. `decision.infer()` → gate + direction + value + risk heads
7. If no position: check cooldown → `build_entry_intent()` → `place_entry()`
8. If holding: value exit check → model exit check → risk update

### OCO Bracket Structure

Every entry places 3 linked orders:
- **Parent:** LMT BUY at `1.05 × mid` (5% above mid, aggressive fill)
- **Stop:** STP SELL at `entry_mid × (1 - stop_pct)` (dynamic stop from model risk head)
- **Take Profit:** LMT SELL at `entry_mid × 6.0` (effectively unreachable — exits come from model/stop)

Orders linked via `ocaGroup` (One-Cancels-All). When stop fills, TP should auto-cancel (but IBKR isn't reliable here — we cancel orphans explicitly).

### SPXW Tick Rules (`_round_spxw_price()` in execution.py)
- Premium < $3.00: tick size $0.05
- Premium ≥ $3.00: tick size $0.10

### Bracket Lifecycle (IBKR-specific)
Normal bracket sequence: `Cancelled → PreSubmitted → Filled`
- The initial `Cancelled` status is NORMAL for child orders — do NOT treat as rejection
- Error code 201 = real rejection (PDT rule, permissions, etc.)

### Cooldown Enforcement

`STOP_COOLDOWN_BARS = 5` (5 minutes after stop-loss before re-entry allowed).

| Context | Where Enforced |
|---------|---------------|
| Training evaluation | `prepare.py` — `if (k - last_stop_bar) < STOP_COOLDOWN_BARS: continue` |
| Replay/backtest | `replay.py` — same logic, tagged `blocked_cooldown` |
| Live trading | `service.py:617` — `if (processed - _last_stop_bar) < STOP_COOLDOWN_BARS` |

Only triggers after **stop-loss** exits. Profit exits and model exits do NOT trigger cooldown.

### Exit Hierarchy (in-position logic)

Checked every bar, in this order:

1. **Value exit** — value head predicts remaining P&L < threshold AND bars_held ≥ 2
2. **Model exit** — gate head says NO_TRADE while holding
3. **Risk update** — if gate says TRADE while holding, update stop based on trailing tiers:
   - +120% unrealized → lock +80%
   - +80% → lock +50%
   - +50% → lock +25%
   - +30% → lock breakeven
4. **Stop/TP** — IBKR-side OCO bracket handles stop loss and take profit fills asynchronously
5. **EOD flatten** — at `end_time_et`, flatten any open position at market

### Audit Events

All events written to `results/live/*.jsonl` as `{ts, event, payload}`.

| Event | Source | Key Fields |
|-------|--------|-----------|
| `session_start` | service.py | session_id, seed_spx, dry_run |
| `session_end` | service.py | processed_minutes, counters (entries, exits, cooldown_blocked) |
| `model_inference` | service.py | action, gate_trade_prob, confidence, direction_probs |
| `entry_intent` | service.py | contract, stop_price, take_profit_price, action, qty |
| `entry_intent_applied` | service.py | position_id, intent_id |
| `entry_blocked_cooldown` | service.py | bars_since_stop, cooldown_bars |
| `position_closed` | service.py | pnl_pct, pnl_dollar, exit_reason, bars_held |
| `entry_live` / `entry_dry_run` | execution.py | fill_price, contract, order details |
| `ib_order_status` | execution.py | status (PreSubmitted, Submitted, Filled, Cancelled) |
| `pnl_update` | execution.py | trade_pnl_pct, trade_pnl_dollars, session_pnl_dollars |
| `value_exit` | service.py | value_pred, threshold, bars_held |
| `model_exit` | service.py | gate_trade_prob |
| `bar_snapshot` | service.py | completeness, missing_feature_names, staleness_seconds |

### Key Metrics to Monitor

| Metric | Healthy Range | How to Check |
|--------|--------------|-------------|
| Gate selectivity (% bars with gate_prob < 0.5) | > 70% | grep `model_inference` → `gate_trade_prob` |
| Trades per session | 0-3 | count `entry_intent_applied` per session |
| Cooldown blocks per session | 0-5 | count `entry_blocked_cooldown` events |
| Close-to-reentry gap | > 5 minutes | time between `position_closed` and next `entry_intent` |
| Stop-out rate | < 50% | `position_closed` with stop exit / total closes |
| Model exit rate | > 30% | `model_exit` or `value_exit` / total closes |
| Orphan positions | 0 | entries without matching `position_closed` |

### Known Constraints

| Constraint | Detail |
|-----------|--------|
| **PDT rule** | Paper account needs ≥$25k equity to avoid Pattern Day Trader restrictions |
| **SELL permissions** | Paper account must have SPX index options selling enabled in Account Management |
| **Async cancellation** | IBKR order cancels are async — `ib.sleep(0.5)` after cancel to let propagate |
| **OCO unreliable** | IBKR doesn't always cancel the other leg of OCO — explicit `cancel_orphaned_orders()` required |
| **Quote staleness** | `quote_mid()` can return None if ticker hasn't updated — use cached fallback |

## IBKR Compatibility Gate

| Check | Validates | How |
|-------|-----------|-----|
| model_exists | `best_model.pt` present | File exists + SHA256 |
| model_checkpoint | Gate=2, Dir=6, Value=1, Risk=3 outputs | Load state_dict, check shapes |
| feature_parity | 37 features, `val_start_idx` | Load data.pt, verify dimensions |
| ibkr_probe | TWS connectivity + data | `ib_probe.py` (market hours only) |

## Daily Pipeline

Two modes: **daily** (data rebuild + trade) and **weekly** (data rebuild + retrain + trade).

| Time (ET) | Daily Mode | Weekly Mode (Sunday) |
|-----------|-----------|---------------------|
| 5:30 AM | Data rebuild (incremental, ~2-3 min) | Data rebuild |
| 5:45 AM | Paper trading (background, until 4 PM) | — |
| 8:00 PM PT | — | Akash training (~2h, ~20-30 experiments) |
| 4:00 PM | CSV export + IBKR session analysis | — |

**Why not retrain daily?** Z-score normalization makes features regime-invariant. Daily sessions are too short (7-10 experiments). Weekly gives room to explore.

```bash
python3 tools/daily_pipeline.py              # Daily: data + trade
python3 tools/daily_pipeline.py --retrain    # Weekly: data + retrain + trade
python3 tools/daily_pipeline.py --dry-run    # Preview
```

**Schedules:** `infra/daily_pipeline.plist` (weekdays 2:30 AM PT), `infra/weekly_retrain.plist` (Sunday 8 PM PT). Requires IB Gateway running (2FA, can't auto-start).

## State Management

State at `results/art2/state.json`: `{cycle, phase, run_name, best_score, last_action, timestamp}`

Per-cycle artifacts in `results/art2/cycle-NNN/`: `analysis.json`, `research.json`, `research.md`, `briefing.md`, `decision.md`, `review.md`, `replay/`, `logs/`

## Environment Variables

```
POLYGON_API_KEY       — Market data API (required for context refresh)
POLYGON_S3_KEY_ID     — S3 access key for flat files
POLYGON_S3_SECRET     — S3 secret key for flat files
TRAIN_*               — Hyperparameter overrides (TRAIN_LOOKBACK, TRAIN_LR, etc.)
SCORE_*               — Score formula (LOCKED — agent cannot modify)
DEPOSIT_ACT           — Akash deposit in ACT tokens (default 5, ~$5 USD)
```

## Results Layout

```
results/art2/
  state.json                         → ART² state
  cycle-NNN/                         → per-cycle artifacts
  REVIEW / STOP / PAUSED             → control files

results/live/
  audit-YYYY-MM-DD.jsonl             → paper trading events
  trades-YYYY-MM-DD.csv              → trade CSV
  summary-YYYY-MM-DD.txt             → daily summary

results/ibkr_sessions/
  YYYY-MM-DD.json                    → session metrics
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| SSH "Permission denied" during deploy | Transient — retry. Check `.deploy-state` SSH_PORT. |
| deploy.sh stop hangs | Interactive prompt blocks in non-TTY. Known issue. |
| Akash "Deposit invalid" | BME migration: mint ACT first (`provider-services tx bme mint-act <amount>uakt`). |
| Training stuck at experiment N | API credit exhaustion. Auto-shuts down after 3 failures. |
| Replay diverges from training | Run `live_feature_parity_report.py`. |
| Score gaming detected | Score config locked. Check keep/reject gates. |
