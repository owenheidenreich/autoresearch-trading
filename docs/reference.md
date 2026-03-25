# Project Reference

Single reference for navigating the autoresearch-trading codebase. For process/lifecycle details, see the [operating manual](../.claude/rules/art2-operating-manual.md) (auto-loaded every session).

## What This Project Does

Autonomous SPX 0DTE options trading system. Opus (strategist) directs Sonnet agents (code writers) to iteratively evolve a PyTorch trading model on Akash H100 GPUs, then the best model runs live paper trading on IBKR.

## Key Files

### Data Pipeline
- [training/prepare.py](../training/prepare.py) — Builds `data.pt` from SPY/SPX/VIX bars + SPXW option chains. Defines feature contract (37 features, v3), action space (8 actions), and scoring formula. Uses incremental cache updates (`_incremental_update()`).
- `~/.cache/autoresearch-trading/features/data.pt` — Training tensor (~201MB, ~383k bars)
- `~/.cache/autoresearch-trading/data/` — Raw caches: `spy_1min.pkl`, `spx_1min.pkl`, `vix_1min.pkl` (append-only), `spxw/` and `spxw_chain/` (per-day incremental)

### Training Loop
- [tools/inner_loop.py](../tools/inner_loop.py) — Agent mode: `experiment --mutation FILE --summary "hypothesis"`. Atomic (validate → backup → upload → train → score → keep/revert).
- [training/run_loop.py](../training/run_loop.py) — Utility library: validation, anomaly detection, scoring helpers. Imported by inner_loop.py.
- [training/train.py](../training/train.py) — Model architecture + training loop. The file Sonnet agents mutate.
- [training/program.md](../training/program.md) — Agent contract. Injected into Sonnet prompts.
- [training/lab_notebook.md](../training/lab_notebook.md) — Cross-run memory: dead ends + priorities. Injected into Sonnet prompts.

### Replay & Live Trading
- [training/replay.py](../training/replay.py) — `evaluate_trades()` IS the trading simulation. Also loads evolved architectures.
- [tools/paper_live.py](../tools/paper_live.py) — IBKR paper trading CLI
- [training/live/service.py](../training/live/service.py) — IBKR connection, order lifecycle
- [training/live/decision.py](../training/live/decision.py) — Model inference engine
- [training/live/context.py](../training/live/context.py) — Real-time feature construction

**Feature parity (2026-03-24):** All 37 features (v3) confirmed across training/replay/live — shared `compute_features()` + `normalize_features_with_context()`.

**Architecture parity (2026-03-24):** Three-head model (v5: gate + direction + value) with 7-dim position state. Value head exit logic consistent across prepare.py `evaluate_trades()`, replay.py `run_replay()`, and decision.py `build_risk_update_intent()`. Exit priority: stop_loss > model_exit > value_exit > max_hold > EOD.

### Infrastructure & Tools
- [infra/deploy.sh](../infra/deploy.sh) — Akash GPU lifecycle (boot/start/sync/stop/ssh/logs/status)
- [tools/art2.py](../tools/art2.py) — ART² orchestrator (9-phase meta-loop)
- [tools/monitor.py](../tools/monitor.py) — Web dashboard (localhost:8420)
- [tools/daily_pipeline.py](../tools/daily_pipeline.py) — Pre-market data refresh + weekly retrain
- [tools/ibkr_analyze.py](../tools/ibkr_analyze.py) — IBKR session analyzer

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

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| STARTING_CAPITAL | $10,000 | prepare.py |
| SPX_MULTIPLIER | $100 | prepare.py |
| POSITION_RISK_TARGET | 0.05 (5%) | prepare.py |
| STOP_LOSS_PCT | -0.30 (-30%) | prepare.py |
| MAX_HOLD_BARS | 60 | prepare.py |
| COOLDOWN_BARS | 5 | prepare.py |
| NO_TRADE_BEFORE_BAR | 30 | prepare.py |
| Train/val split | 70/30 | prepare.py |
| POSITION_STATE_DIM | 7 | train.py |
| VALUE_LOSS_WEIGHT | 0.3 | train.py (TRAIN_VALUE_W) |
| VALUE_EXIT_THRESHOLD | 0.02 | train.py (TRAIN_VALUE_EXIT_THRESH) |

## Warm-Start vs Fresh-Start

### MUST fresh-start (tensor shape incompatibility)
- Feature count change — `input_proj` dimensions change
- Position state dimension change — `position_proj` dimensions change
- D_MODEL, DEPTH, or N_HEADS change — transformer weights change shape
- New module added — state_dict keys mismatch (e.g. adding value head)
- Action space change
- **Version gate enforced:** train.py and replay.py reject checkpoints missing `has_value_head` or with `position_state_dim < 7`. Incompatible checkpoints cause training from scratch automatically.

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

**Control files:** `results/art2/REVIEW` (remove to approve), `results/art2/STOP` (create to exit), `results/art2/PAUSED` (remove to resume).

## IBKR Compatibility Gate

| Check | Validates | How |
|-------|-----------|-----|
| model_exists | `best_model.pt` present | File exists + SHA256 |
| model_checkpoint | Gate=2, Dir=6 outputs | Load state_dict, check shapes |
| feature_parity | 37 features, `val_start_idx` | Load data.pt, verify dimensions |
| ibkr_probe | TWS connectivity + data | `ib_probe.py` (market hours only) |

## State Management

State at `results/art2/state.json`: `{cycle, phase, run_name, best_score, last_action, timestamp}`

Per-cycle artifacts in `results/art2/cycle-NNN/`: `analysis.json`, `research.json`, `research.md`, `briefing.md`, `decision.md`, `review.md`, `replay/`, `logs/`

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
