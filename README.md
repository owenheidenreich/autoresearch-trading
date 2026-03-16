# autoresearch-trading

A recursive self-improving AI that learns to trade SPX 0DTE options intraday.

An LLM (Claude Sonnet 4) iteratively modifies a PyTorch training script, runs
5-minute GPU experiments on a rented H100, and keeps only improvements — evolving
the model autonomously over 6-8 hour runs. Adapted from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## What This Actually Does

The model uses a **two-head architecture**:
- **Gate head**: `[NO_TRADE, TRADE]` — decides if a trade opportunity exists
- **Direction head**: `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]` — decides which strike and direction

This produces 8 effective actions: **DO_NOTHING**, **BUY_CALL_ATM**, **BUY_CALL_OTM5**,
**BUY_CALL_OTM10**, **BUY_PUT_ATM**, **BUY_PUT_OTM5**, **BUY_PUT_OTM10**, **EXIT**.
The model is a **sniper** — it waits for A+ setups, takes few high-conviction trades
(3-5 per day), and can actively exit positions when the gate head says "no longer a
good time to trade." OTM options are cheaper (lower premium), which is valuable for
live testing with real capital.

### The Feedback Loop

```
Claude reads train.py + history → proposes one change → train 5 min on H100
  → evaluate on held-out data → score improved? → keep : revert → repeat
```

Over 6 hours (~50-80 experiments), the model architecture, loss function, and
hyperparameters evolve without human intervention.

## Cost Rules

> **The H100 costs several dollars per hour. Never waste it on non-GPU work.**

1. **`prepare.py` runs LOCALLY** — downloads data, computes features, saves tensors
2. **Only `train.py` and `run_loop.py` run on the H100**
3. **Do NOT run data downloads or feature computation on the H100**

## Quick Start

```bash
# 1. Prepare data locally (1-min bars, requires IBKR Gateway running)
python3 training/prepare.py --use-spx --ib-port 4002   # full rebuild: IBKR + Polygon flat files

# 2. Source API key
set -a && source .env && set +a

# 3. Deploy H100 + start loop (two commands)
./infra/deploy.sh boot                                    # deploy H100 on Akash (~2 min)
./infra/deploy.sh start --hours 6 --max-experiments 100   # upload + pre-flight + start loop

# 4. Monitor (sync in a second terminal)
./infra/deploy.sh sync           # auto-download improvements as they happen
./infra/deploy.sh status         # GPU, loop PID, last scores
./infra/deploy.sh logs           # tail loop.log
./infra/deploy.sh ssh            # shell into H100

# 5. Stop (downloads results + copies model weights back, then closes deployment)
./infra/deploy.sh stop

# 6. Paper trading validation (after training)
python3 training/replay.py --date 2026-03-17                    # instant replay on unseen day
python3 training/replay.py --date 2026-03-17 --output trades.csv # save full trade journal
```

## Project Structure

```
training/
  prepare.py                 — data pipeline + 60 features + ATM/OTM P&L targets (runs locally)
  train.py                   — two-head sniper model + 6-class direction + option-P&L loss (Claude modifies)
  replay.py                  — paper trading replay: model vs unseen day + trade journal (runs locally)
  run_loop.py                — autoresearch orchestrator (Claude API + train + model weight preservation)
  program.md                 — domain knowledge + instructions for Claude
  best_model.pt              — trained model weights (persisted across deployments via warm-start)
  ib_probe.py                — IB Gateway connectivity test (ES, SPX, SPXW availability)
infra/
  deploy.sh                  — Akash deployment CLI (boot/start/sync/status/logs/download/stop)
  deploy-autoresearch.yaml   — Akash SDL (container spec: H100 + SSH + tini + PyTorch)
  start_loop.sh              — Remote loop launcher (called by deploy.sh start)
  watchdog.sh                — Container health monitor (PID 1, loop, GPU mem, OOM events)
results/
  run-YYYY-MM-DD/            — Downloaded artifacts from each training run
docs/                        — operational docs, domain knowledge, historical plans
```

## How It Works

1. **prepare.py** (LOCAL) downloads ~986 days of 1-min bars from IBKR (SPY volume,
   real SPX prices, real VIX), plus SPXW 0DTE option bars from Polygon flat files (S3)
   at ATM and OTM strikes (ATM±5, ATM±10). Covers March 14, 2022 → March 14, 2026
   (MWF only before May 2022, daily after). Computes 60 features across 13 groups
   (returns, volume, volatility, VWAP, session, key levels, trend, microstructure,
   time, options, VIX/regime, OTM/skew, Greeks). Computes actual option P&L targets
   for all 6 strike/direction combinations plus EXIT labels. Saves to
   `~/.cache/autoresearch-trading/features/data.pt`.

2. **run_loop.py** (H100) orchestrates the autoresearch loop: reads `program.md`
   (domain knowledge) + current `train.py` + experiment history, calls Claude for
   one code modification, validates syntax + structural safety, runs training,
   evaluates, keeps or reverts. Auto-strips `torch.compile` and rejects attempts
   to merge the two-head architecture back into a single head. Preserves model
   weights across experiments (backup/restore pattern).

3. **train.py** (H100) loads `data.pt`, trains a two-head transformer (~250K params):
   - Input: `(batch, 120, 60)` — 2 hours of 1-min bars × 60 features
   - Gate head: `(batch, 2)` → `[NO_TRADE, TRADE]`
   - Direction head: `(batch, 6)` → `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]`
   - Loss: cross-entropy on actual option P&L across all 6 strikes,
     time-weighted (afternoon 1.5x), EXIT loss (EXIT_LOSS_WEIGHT=0.3),
     P&L alignment bonus
   - Evaluates via discrete trade simulation with model-driven EXIT across ATM and OTM

4. **Metric**: `score = profit_factor × trade_sharpe × freq_mult`, where
   `freq_mult = min(1, tpd/2)` below 6 tpd, and `max(0.1, (6/tpd)²)` above
   (quadratic over-trading penalty). Sweet spot: 3-5 trades/day.

5. **Claude** modifies only `train.py` — architecture, loss function, optimizer,
   hyperparameters, lookback window. Everything is fair game except merging the
   two heads, reducing direction outputs below 6, or reverting to percentile-based labels.

6. **replay.py** (LOCAL) runs the trained model against unseen market days using
   IBKR as the only data source for options — matching the live trading environment.
   Produces a full trade journal with entry/exit prices, P&L, model confidence,
   MFE/MAE, and session statistics.

## Model Architecture

```
Input: (batch, 120, 60) — 2 hours of 1-min bars × 60 features
  → FeatureGroupGating (13 groups, auto-adapts to feature count)
  → TransformerEncoder (depth=4-6, d_model=64-112, causal masking)
  → Gate head: LayerNorm → Linear → GELU → Dropout → Linear(2)  → [NO_TRADE, TRADE]
  → Dir head:  LayerNorm → Linear → GELU → Dropout → Linear(6)  → [CALL_ATM, ..., PUT_OTM10]

Effective actions (8):
  gate=NO_TRADE, not in trade → DO_NOTHING (0)
  gate=TRADE,    dir=0-2      → BUY_CALL_ATM/OTM5/OTM10 (1-3)
  gate=TRADE,    dir=3-5      → BUY_PUT_ATM/OTM5/OTM10 (4-6)
  gate=NO_TRADE, in trade     → EXIT (7) — model-driven profit-taking
```

## Training Targets

**Primary**: Actual SPXW option P&L for ATM and OTM strikes (in data.pt)
- ATM: `call_pnl`, `put_pnl` — from real ATM SPXW prices
- OTM+5: `otm5_call_pnl`, `otm5_put_pnl` — from ATM±5 strikes
- OTM+10: `otm10_call_pnl`, `otm10_put_pnl` — from ATM±10 strikes
- Formula: `(exit_price - entry_price) / entry_price - spread_cost`
- Hold horizon: 30 bars (30 min), same day only
- Gate target: TRADE when any option P&L > 0 across all 6 strikes
- Direction target: argmax of 6 P&L values (best strike/direction wins)

**EXIT labels**: `exit_call_label`, `exit_put_label` — 1.0 when unrealized P&L > 20%.
Now actively used in the loss function (EXIT_LOSS_WEIGHT = 0.3).

**Secondary**: Forward 6-bar return (kept for fallback/compatibility)

## Features (60)

| Group | Features | Count |
|-------|----------|-------|
| Returns | 5-bar, 15-bar, 30-bar, 60-bar, 120-bar | 5 |
| Volume | ratio, z-score, at-price percentile | 3 |
| Volatility | bar range, realized vol, range ratio | 3 |
| VWAP | distance, slope, upper/lower ±1σ, upper/lower ±2σ | 6 |
| Session | IB high/low/width, AM range %, session range % | 5 |
| Key Levels | ONH/ONL, prev high/low/close/VWAP distance | 6 |
| Trend | HH/HL, EMA cross, close position | 3 |
| Microstructure | gap, inside bar | 2 |
| Time | minutes to close, sin/cos, day of week, half-hour proximity, IB complete | 6 |
| Options | ATM IV, IV skew, ATM premium %, put/call vol ratio, option volume, theta rate | 6 |
| VIX/Regime | VIX level, VIX change, VIX regime, variance risk premium | 4 |
| OTM/Skew | OTM call IV, OTM put IV, IV skew (near), call wing, put wing, OTM vol ratio | 6 |
| Greeks | delta, gamma, theta, vega, gamma/theta ratio | 5 |

All features (except time encodings) are rolling z-score normalized, clipped ±5.

## Data

- **Equity prices**: IBKR (real SPX cash index + SPY volume via IB Gateway, 1-min bars)
- **VIX**: IBKR (real CBOE VIX index, 1-min bars)
- **Options (training)**: Polygon flat files / S3 (SPXW 0DTE bars — ATM + OTM chain at ±5/±10 strikes)
- **Options (replay/live)**: IBKR only — matches live trading environment
- **Dataset**: ~382,920 bars × 60 features, ~986 trading days (March 2022–March 2026)
- **0DTE filtering**: MWF only before May 11, 2022 (before daily 0DTE launched)
- **Train/Val split**: ~70/30 by day index
- **Option coverage**: ~90%+ ATM, lower for OTM (NaN gracefully handled)

## Infrastructure

- **GPU**: Akash Network, H100 80GB HBM3, `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- **Resources**: 8 CPU, 64 GB RAM, 50 GB storage
- **PID 1**: `tini` (proper init process — prevents zombie accumulation)
- **SSH**: Password auth, port mapped to 30000–33000 range
- **Watchdog**: `watchdog.sh` monitors container health every 5s (PID 1, loop status, GPU memory, cgroup memory, OOM events)

### Deploy lifecycle

```
boot  → deploy H100, wait for SSH, verify GPU
start → upload code + data + best_model.pt (warm-start from previous run)
        → pre-flight on H100 (validates data.pt, GPU, Claude API, disk space)
        → launch loop + watchdog
sync  → poll every 60s, auto-download improvements, heartbeat logging
stop  → SIGTERM → wait 15s → SIGKILL → download all results
        → copy best_model.pt to training/ (warm-start for next run)
        → close Akash deployment
```

### Known operational notes

- `torch.compile` is **auto-stripped** by `run_loop.py` — it OOM-kills the 64GB container.
- Set `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` to prevent thread over-subscription.
- Container files are **ephemeral** — pod restarts wipe `/root/`. Always re-upload after restart.
- `tini` as PID 1 prevents zombie process accumulation that previously caused silent restarts.
- Structural safety patterns reject single-head merges, percentile label reversion, and `DataParallel`.
- **Docker builds MUST use `--platform linux/amd64`** — dev machine is Apple Silicon but Akash is amd64.
- Model weights are preserved across deployments: `stop` copies `best_model.pt` to `training/`, `start` uploads it for warm-start.

## Cost Estimate (6-hour run)

| Item | Cost |
|------|------|
| H100 rental | ~$2–3/hr × 6 hrs = ~$15 |
| Claude API | ~$0.02/experiment × 60 experiments = ~$1.20 |
| **Total** | **~$16** |

## Docs

- [docs/AKASH-SSH.md](docs/AKASH-SSH.md) — How to find your SSH port when Akash Console doesn't show it
- [docs/0dte-domain-knowledge.md](docs/0dte-domain-knowledge.md) — 0DTE Greeks, GEX mechanics, vol surface (from 150+ trading books)
- [docs/pickles-trading-knowledge.md](docs/pickles-trading-knowledge.md) — Expert trader journal extraction (218 files, entry/exit rules)
- [docs/FUTURE-FEATURES.md](docs/FUTURE-FEATURES.md) — Future features and training run findings
