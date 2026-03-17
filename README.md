# autoresearch-trading

A recursive self-improving AI that learns to trade SPX 0DTE options intraday.

An LLM (Claude Sonnet 4) iteratively modifies a PyTorch training script, runs
training experiments on a rented H100, and keeps only improvements — evolving
the model autonomously. Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

---

## How It Works

```
Claude reads train.py + history → proposes one change → train on H100
  → evaluate on held-out data → score improved? → keep : revert → repeat
```

Over a 5-hour run (~24 experiments), the model architecture, loss function, and
hyperparameters evolve without human intervention.

### The Model

Two-head transformer sniper (~800K params):
- **Gate head**: `[NO_TRADE, TRADE]` — should I trade right now?
- **Direction head**: `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]` — which strike?

This produces 8 effective actions: DO_NOTHING, BUY_CALL_ATM/OTM5/OTM10,
BUY_PUT_ATM/OTM5/OTM10, EXIT. The model waits for A+ setups, takes 3-5
trades per day, and can actively exit positions.

### Scoring

```
score = profit_factor × trade_sharpe × freq_mult

freq_mult:
  tpd 2-6   → SWEET SPOT (full score)
  tpd < 0.5 → -10.0 (DO_NOTHING trap)
  tpd > 6   → quadratic decay (10→0.36x, 15→0.16x)

Additional penalties: consecutive losses, 1-bar holds, stop-loss rate,
direction collapse (>80% same direction)
```

---

## Quick Start

```bash
# 1. Prepare data locally (requires IBKR Gateway running)
python3 training/prepare.py --use-spx --ib-port 4002

# 2. Source API key
set -a && source .env && set +a

# 3. Deploy + train (two commands, everything else is automatic)
DEPOSIT_AKT=30 ./infra/deploy.sh boot
./infra/deploy.sh start --hours 5 --max-experiments 24

# That's it. Auto-sync runs in background, continuously pulling results.
# Monitor with:
./infra/deploy.sh status         # GPU, loop PID, last scores
./infra/deploy.sh logs           # tail loop.log
tail -f results/run-$(date +%Y-%m-%d)/sync.log  # sync progress

# Stop (downloads results, closes deployment, refunds remaining AKT)
./infra/deploy.sh stop

# Paper trading validation
python3 training/replay.py --date 2026-03-17
```

---

## Project Structure

```
training/
  train.py              — model + loss + training loop (Claude modifies this)
  best_train.py          — snapshot of train.py that produced best score (architecture lock)
  best_model.pt          — trained model weights (persisted across deployments via warm-start)
  prepare.py             — data pipeline: 60 features + option P&L targets (runs locally)
  replay.py              — paper trading: model vs unseen market day (runs locally)
  run_loop.py            — autoresearch orchestrator (runs on H100)
  program.md             — domain knowledge + rules for Claude
  ib_probe.py            — IBKR connectivity test
infra/
  deploy.sh              — Akash deployment CLI (boot/start/stop/status/logs/ssh)
  deploy-autoresearch.yaml — Akash SDL (H100 container spec)
  start_loop.sh          — remote loop launcher
  watchdog.sh            — container health monitor
results/
  run-YYYY-MM-DD/        — artifacts from each training run
docs/
  0dte-domain-knowledge.md — Greeks, GEX, vol surface (from 130+ trading books)
  pickles-trading-knowledge.md — expert trader journal extraction
```

### File Lifecycle

```
                    LOCAL                           H100 (ephemeral)
                    ─────                           ────────────────
train.py       ──upload──>   train.py (Claude modifies each experiment)
best_train.py  ──upload──>   best_train.py (architecture lock reference)
best_model.pt  ──upload──>   best_model.pt (warm-start weights)
                              │
                              ▼  (after each improvement)
                              best_train.py + best_model.pt updated
                              │
train.py       <──sync────   best_train.py (becomes new train.py)
best_train.py  <──sync────   best_train.py
best_model.pt  <──sync────   best_model.pt
```

**Key rule**: After a run, `best_train.py` IS the new `train.py`. The sync
handles this automatically — no manual copying needed.

---

## Deploy Lifecycle

```
boot  → deploy H100 on Akash, wait for SSH, verify GPU
start → upload code + data + best_model.pt (warm-start)
        → pre-flight checks (data, GPU, API)
        → launch training loop + watchdog
        → auto-start background sync (polls every 30s)
        → on each improvement: sync model + code back to training/
stop  → kill background sync → kill training loop
        → download all results → copy best model to training/
        → close Akash deployment (refunds remaining AKT)
```

### Commands

| Command | Purpose |
|---------|---------|
| `./infra/deploy.sh boot` | Deploy H100 container (~2 min) |
| `./infra/deploy.sh start --hours H --max-experiments N` | Upload code, start loop + auto-sync |
| `./infra/deploy.sh status` | GPU utilization, loop PID, last scores |
| `./infra/deploy.sh logs` | Tail the training loop log |
| `./infra/deploy.sh ssh` | Shell into the H100 |
| `./infra/deploy.sh sync` | Manual foreground sync (auto-sync runs with start) |
| `./infra/deploy.sh download` | One-time full download of results |
| `./infra/deploy.sh stop` | Kill loop → download → close deployment |

### Deposit Guide

| AKT | Duration | Use Case |
|-----|----------|----------|
| 5 | ~1 hour | Quick test (2-3 experiments) |
| 15 | ~3 hours | Short run (~12 experiments) |
| 30 | ~5 hours | Full run (~24 experiments) |
| 50 | ~8 hours | Extended run (~40 experiments) |

---

## Data Pipeline

`prepare.py` runs locally (never on H100 — that wastes GPU money).

### Sources
- **SPX prices**: IBKR (real cash index, 1-min bars)
- **SPY volume**: IBKR (1-min bars, proxy for SPX volume)
- **VIX**: IBKR (real CBOE VIX, 1-min bars)
- **SPXW options (training)**: Polygon flat files / S3 (ATM + OTM at ±5/±10 strikes)
- **SPXW options (replay/live)**: IBKR only (matches live trading environment)

### Dataset
- ~382,920 bars × 60 features, ~986 trading days
- March 2022 – March 2026 (MWF only before May 11, 2022; daily after)
- Train/Val split: ~70/30 by chronological day index
- Output: `~/.cache/autoresearch-trading/features/data.pt`

### Features (60)

| Group | Features | Count |
|-------|----------|-------|
| Returns | 5-bar, 15-bar, 30-bar, 60-bar, 120-bar | 5 |
| Volume | ratio, z-score, at-price percentile | 3 |
| Volatility | bar range, realized vol, range ratio | 3 |
| VWAP | distance, slope, upper/lower ±1σ, ±2σ | 6 |
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
Normalization context buffer (500 bars) saved in data.pt for OOS/live continuity.

### Training Targets
- **Option P&L**: Actual SPXW P&L for all 6 strike/direction combos (30-bar hold, same day)
- **Gate target**: TRADE when any option P&L > 0 across all strikes
- **Direction target**: argmax of 6 P&L values (best strike/direction wins)
- **EXIT labels**: 1.0 when unrealized P&L > 20% (EXIT_LOSS_WEIGHT = 0.3)

---

## Autoresearch Loop Details

### Safety Guards (run_loop.py)
- **Architecture lock**: D_MODEL, DEPTH, N_HEADS must match best_train.py
- **Structural safety**: Rejects single-head merges, `torch.compile`, `DataParallel`, batch > 256
- **Backup/restore**: Before each experiment, backs up model weights + code. Reverts on failure.
- **5 consecutive failures**: Auto-restores best_train.py as starting point
- **Warm-start**: Loads best_model.pt weights at start of each experiment (LR reduced to 0.3x)

### What Claude Can Modify
- Loss function, optimizer, hyperparameters, lookback window
- Layer structure within the two-head architecture
- Feature gating, dropout, normalization
- Everything EXCEPT: merging heads, reducing direction outputs below 6, architecture constants

### Cost Per Run
| Item | Cost |
|------|------|
| H100 rental | ~$2–3/hr |
| Claude API | ~$0.43/experiment |
| **5-hour run (24 experiments)** | **~$25** |

---

## Replay Mode (Paper Trading)

```bash
python3 training/replay.py --date 2026-03-17                     # instant replay
python3 training/replay.py --date 2026-03-17 --speed 10          # 10x speed
python3 training/replay.py --date 2026-03-17 --verbose           # every bar decision
python3 training/replay.py --date 2026-03-17 --output trades.csv # save trade journal
```

Uses IBKR as the only data source for options — matching the live trading environment.
Produces trade journal with entry/exit prices, P&L, model confidence, MFE/MAE.

---

## Real-Time Paper Trader (IBKR + Polygon Context)

New live subsystem (paper only) with:
- Daily pre-open `context_refresh` using Polygon historical context (plus IBKR SPX/VIX fill-ins)
- IBKR entitlement probe (`marketDataType=1`) before order enablement
- 5-second intraminute ingestion -> minute-close decisions
- Mandatory bracket/OCO orders on entry
- Monotonic ratchet rules (`stop` and `take_profit` can only move up)
- Kill-switch support

Requirements:
- `POLYGON_API_KEY` for context refresh (historical bootstrap)
- IB Gateway/TWS paper session running with market data entitlements for SPX/SPXW/VIX/SPY

### Commands

```bash
# 1) Refresh context bundle only (recommended pre-open, ~9:10 ET)
python3 tools/paper_live.py --context-only --port 4002 --context-days 30

# 2) Run session in dry-run mode (no orders, full decisions + audits)
python3 tools/paper_live.py --dry-run --port 4002 --model training/best_model.pt

# 3) Run fully automatic paper execution (places paper orders)
python3 tools/paper_live.py --paper-auto --port 4002 --model training/best_model.pt

# Optional: explicit entitlement probe only
python3 tools/ib_entitlements.py --port 4002

# 4) Enable kill switch (content: stop/on/true/1)
echo stop > /tmp/trading-kill-switch
python3 tools/paper_live.py --paper-auto --kill-switch /tmp/trading-kill-switch
```

Audit trail is written to `results/live/audit.jsonl` by default.
See `docs/IBKR-LIVE-CHECKLIST.md` for required IBKR data/API settings.

---

## Infrastructure

- **GPU**: Akash Network, NVIDIA H100 80GB HBM3
- **Container**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- **Resources**: 8 CPU, 64 GB RAM, 50 GB storage
- **PID 1**: `tini` (prevents zombie accumulation)
- **SSH**: Password auth (`root`/`autoresearch2026`), port in 30000-33000 range
- **Watchdog**: Monitors PID 1, loop status, GPU memory, cgroup, OOM events every 5s

### Operational Notes
- `torch.compile` is auto-stripped — it OOM-kills the 64GB container
- Container files are ephemeral — pod restarts wipe `/root/`
- Docker builds MUST use `--platform linux/amd64` (dev machine is arm64, Akash is amd64)
- `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` prevent thread over-subscription

### Troubleshooting: Finding SSH Port

If Akash Console doesn't show the forwarded port, scan for it:

```bash
# Scan ports 30000-33000 for SSH
for p in $(seq 30000 33000); do
  (echo "" | nc -w 0.5 $HOST $p 2>/dev/null | grep -q SSH && echo "SSH on $p") &
done; wait

# Try your password on each SSH port found
SSHPASS='autoresearch2026' sshpass -e ssh -p $PORT root@$HOST "nvidia-smi"
```

---

## Project History

| Phase | What | When |
|-------|------|------|
| Data Pipeline | 60 features, 1-min bars, option P&L targets | March 2026 |
| Training Script v3.1 | Two-head, 6-class direction, option-P&L loss, EXIT | March 2026 |
| Autoresearch Loop | Claude → modify → eval → keep/revert | March 2026 |
| Deploy Automation | deploy.sh + tini + watchdog + auto-sync | March 2026 |
| Extended Data | 986 days, 382K bars, March 2022–March 2026 | March 15, 2026 |
| Operational Hardening | Model weight backup/restore, pre-flight, warm-start | March 16, 2026 |
| Normalization Fix | Rolling z-score context buffer for OOS continuity | March 16, 2026 |
| Auto-Sync | Background sync on start, warm-start copy on improvement | March 17, 2026 |

### Training Run Results

| Run | Experiments | Best Score | Key Finding |
|-----|------------|------------|-------------|
| v2 (March 15) | 86 | 155.9 | Low dropout, D_MODEL 80-112, DEPTH 5-6 optimal |
| v3 test (March 15) | 3 | 15.72 | EXIT working, safety guards 0% failure rate |
| v3.1 prod (March 16) | 2 | 0.91 | Extended data, 60 features |
| v3.1 prod (March 17) | 8+ | 4.21 | Warm-start, 10-min budget |

---

## Future Work

### Near Term
- [ ] Paper trade 20+ days, validate metrics match backtest
- [ ] Live trading (1 contract, OTM, risk limits, kill switch)

### Feature Candidates
- **GEX / Dealer Positioning**: 4 features from Polygon OI data (requires API tier testing)
- **Market Internals (TICK, Breadth)**: Leading indicators, blocked on IBKR data availability
- **Walk-forward validation**: Per-chunk eval to detect temporal instability

---

## Reference Docs

- [docs/0dte-domain-knowledge.md](docs/0dte-domain-knowledge.md) — 0DTE Greeks, GEX mechanics, vol surface (from 130+ trading books)
- [docs/pickles-trading-knowledge.md](docs/pickles-trading-knowledge.md) — Expert trader journal extraction (218 files, entry/exit rules)
- [docs/IBKR-LIVE-CHECKLIST.md](docs/IBKR-LIVE-CHECKLIST.md) — Required IBKR entitlements/settings for non-delayed paper automation
