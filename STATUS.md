# Autoresearch-Trading: Project Status

> **Last Updated:** March 16, 2026

---

## Phase Overview

| # | Phase | Status | Notes |
|---|-------|--------|-------|
| 1 | Data Pipeline | ✅ DONE | 60 features, 1-min bars, SPXW ATM+OTM option P&L targets, EXIT labels |
| 2 | Training Script (train.py) | ✅ DONE | v3.1 — two-head, 6-class direction, option-P&L loss, EXIT loss |
| 3 | Autoresearch Loop (run_loop.py) | ✅ DONE | Claude → modify → eval → keep/revert, structural safety guards |
| 4 | DO_NOTHING Fix | ✅ DONE | CE loss + curriculum scoring + two-head architecture |
| 5 | Domain Knowledge | ✅ DONE | Pickles journal + 0DTE books → program.md |
| 6 | Deployment Automation | ✅ DONE | deploy.sh + tini PID 1 + torch.compile auto-strip |
| 7 | First Working Training Run | ✅ DONE | 86 experiments, score ceiling 155.9, 0% failure rate on v3 |
| 8 | v3 Rewrite (Fundamental Gaps) | ✅ DONE | Two-head, option P&L loss, EXIT action, time-weighted loss |
| 9 | IBKR Integration | ✅ DONE | Real SPX + VIX via IB Gateway, monthly batch downloads |
| 10 | v3.1 EXIT + OTM Expansion | ✅ DONE | EXIT loss fix, 6-class direction, OTM features + tradeable actions |
| 11 | Extended Data (4-year, 1-min) | ✅ DONE | 382,920 bars, ~986 0DTE days, 60 features, March 2022–March 2026 |
| 12 | Operational Reliability | ✅ DONE | Model weight preservation, pre-flight checks, deploy pipeline hardening |
| 13 | Production Runs on H100 | 🔄 IN PROGRESS | Multiple runs completed, best score 0.91+, actively training |
| 14 | Paper Trading (Replay Mode) | 🔄 BUILT | replay.py built, IBKR-only data path (matches live trading) |
| 15 | Live Trading | ⬜ TODO | Real money execution |

---

## Phase 11: Extended Data (4-year, 1-min bars) ✅ (March 15, 2026)

Major data pipeline overhaul — switched to 1-minute bars and maximized history:
- **Bar size**: 5-min → 1-min (390 bars/day, lookback=120 = 2 hours)
- **Data range**: March 14, 2022 → March 14, 2026 (4 years)
- **0DTE day filtering**: Mon/Wed/Fri before May 11, 2022 (before daily 0DTE launched), daily after
- **~986 trading days**, 382,920 bars
- **60 features** across 13 groups (added Greeks group: delta, gamma, theta, vega, gamma_theta_ratio)
- **Data sources**:
  - SPY (volume) → IBKR (full history, weekly batches)
  - SPX (price) → IBKR
  - VIX → IBKR
  - SPXW options → Polygon flat files (S3 bulk, for training data preparation)
  - SPXW options (replay/live) → IBKR only (matches live trading environment)

---

## Phase 12: Operational Reliability ✅ (March 16, 2026)

Closed the loop on deploy → train → save → shutdown pipeline:

### Model Weight Preservation (CRITICAL fix)
- **Problem**: `train.py` unconditionally overwrites `best_model.pt` every run. `run_loop.py` only reverted `train.py` on bad experiments, not model weights — good weights were permanently lost.
- **Fix**: Backup/restore pattern in `run_loop.py`. Before each experiment, backs up `best_model.pt`. On improvement: removes backup (keeps new weights). On failure/no improvement: restores from backup.

### Pre-Flight Validation
- Runs before consuming GPU time: data.pt validation (keys, shape, NaN%), GPU check, disk space, Claude API test
- `./deploy.sh start` runs `--dry-run` on the H100 before launching the loop
- Catches bad data, missing API keys, and environment issues before wasting hours

### Deploy Pipeline Hardening
- **API key injection**: Passed via env var (was fragile `sed` rewrite of start_loop.sh)
- **Warm-start**: `best_model.pt` uploaded on `start`, preserved on `download`/`stop` — weights persist across deployments
- **Sync reliability**: Variable defaults prevent silent crash on parse failure; heartbeat logging shows sync is alive
- **Stop race condition**: Waits for loop to die (up to 15s, then SIGKILL) before downloading
- **Training timeout**: 420s (300s budget + 120s buffer) instead of 600s — catches hung trains faster

### Cleanup
- Deleted stale `launch_loop.sh` (hardcoded TIME_BUDGET=30 test config) and `startup.sh`
- Added `watchdog.sh` — monitors container health every 5s (PID 1, loop, GPU memory, cgroup, OOM events)

---

## Phase 13: Production Runs on H100 🔄

### 13A: 3-Loop Validation Test ✅ (March 15, 2026)

Validated full pipeline end-to-end on Akash H100:

| Exp | Score | PF | TPD | Win Rate | Status |
|-----|-------|----|-----|----------|--------|
| #1  | 4.70  | 2.74 | 4.2 | 40.9% | KEPT |
| #2  | -10.0 | 0.46 | 0.04 | 41.7% | reverted |
| #3  | 15.72 | 3.18 | 22.6 | 34.4% | KEPT (best) |

Issues found & fixed: over-trading penalty (quadratic), gate collapse prevention, theta bug.

### 13B: Production Run ✅ (March 16, 2026 — first attempt)

- 2 experiments completed, best score 0.2231 (PF=1.08, 2.1 tpd)
- Training ran 17+ hours but only completed 2 experiments (long Claude response times)
- Sync captured improvement #1 and downloaded all artifacts

### 13C: Production Run 🔄 (March 16, 2026 — current)

- DSEQ 25967176, Amsterdam H100 provider
- 6-hour budget, 100 experiments max, 15 AKT deposit
- Warm-started from previous run's best_model.pt
- After 2 experiments: **score 0.9078** (4x improvement over 13B)
- Actively training with ~5.8 hours remaining

---

## Phase 14: Paper Trading (Replay Mode) 🔄

`replay.py` built — runs trained model against unseen market days:
- [x] Bar-by-bar inference with trade simulation (same rules as evaluate_trades)
- [x] Comprehensive trade journal: SPX context, VIX, volume, MFE/MAE, model confidence
- [x] Session activity stats: near-misses, direction preference, gate probability distribution
- [x] CSV + JSON output with full trade documentation
- [x] Speed control (instant to real-time), verbose mode
- [x] **IBKR-only option data** — matches live trading environment (no Polygon fallback)
- [x] IBKR option download function (`download_spxw_ibkr`) with same cache format
- [ ] Validate on unseen market day after training completes
- [ ] Paper trade 20+ days, validate metrics match backtest

## Phase 15: Live Trading ⬜

- [ ] Graduate to live (1 contract, OTM for lower capital requirement, risk limits, kill switch)

---

## Quick Reference

**Commands:**
```bash
cd autoresearch-trading

# Rebuild data (local) — 1-min bars, March 2022–March 2026
python3 training/prepare.py --use-spx --ib-port 4002      # full rebuild with IBKR + flat files
python3 training/prepare.py --skip-download --use-spx      # recompute features from cached bars

# Replay mode (paper trading validation — uses IBKR for option data)
python3 training/replay.py --date 2026-03-17               # instant replay on unseen day
python3 training/replay.py --date 2026-03-17 --speed 10    # 10x speed
python3 training/replay.py --date 2026-03-17 --verbose     # every bar decision
python3 training/replay.py --date 2026-03-17 --output trades.csv  # save trade journal

# Local smoke test
TIME_BUDGET=30 python3 training/train.py

# Source API key, then deploy
set -a && source .env && set +a
./infra/deploy.sh boot                               # deploy H100 (~2 min)
./infra/deploy.sh start --hours 6 --max-experiments 100  # upload + pre-flight + start loop

# Monitor
./infra/deploy.sh status        # GPU, loop PID, last scores
./infra/deploy.sh logs          # tail loop.log
./infra/deploy.sh sync          # auto-download improvements (run in 2nd terminal)
./infra/deploy.sh ssh           # shell into H100

# Download & Stop
./infra/deploy.sh download      # download results + copy best_model.pt to training/
./infra/deploy.sh stop          # kill loop → wait → download → close deployment

# Deposit control (default 15 AKT)
DEPOSIT_AKT=5 ./infra/deploy.sh boot    # 5 AKT for short test (~1 hour)
DEPOSIT_AKT=25 ./infra/deploy.sh boot   # 25 AKT for full run (~8 hours)
```

**Key Files:**
| File | Purpose | Runs on |
|------|---------|---------|
| prepare.py | Data pipeline + 60 features (1-min bars) + ATM/OTM option P&L targets | Local Mac |
| train.py | Two-head sniper model + 6-class direction + option-P&L loss | H100 GPU |
| replay.py | Paper trading replay — model vs unseen day + trade journal (IBKR-only data) | Local Mac |
| run_loop.py | Autoresearch orchestrator (Claude API + train + model weight preservation) | H100 GPU |
| program.md | Instructions + domain knowledge for Claude | H100 (read) |
| deploy.sh | Akash H100 deployment CLI (boot/start/sync/status/logs/download/stop) | Local Mac |
| start_loop.sh | Remote loop launcher (called by deploy.sh start) | H100 GPU |
| watchdog.sh | Container health monitor (PID 1, loop, GPU mem, cgroup, OOM events) | H100 GPU |

**Scoring:**
```
if tpd <= 6: freq_mult = min(1, tpd/2)
else:        freq_mult = max(0.1, (6/tpd)²)   ← QUADRATIC over-trading penalty

score = profit_factor × trade_sharpe × freq_mult

Penalties:
  tpd < 0.5  → score = -10.0  (DO_NOTHING trap)
  tpd 0.5–1.5 → ramped from -5.0 to raw score
  tpd 2–6    → SWEET SPOT (full score)
  tpd > 6    → QUADRATIC decay (10→0.36x, 12→0.25x, 15→0.16x, 22→0.07x)

Additional penalties (post-scoring):
  consecutive losses >3 → score *= max(0.5, 1.0 - 0.05*(consec-3))
  1-bar holds >30%      → score *= max(0.7, 1.0 - (pct - 0.30))
  stop-loss rate >30%   → score *= max(0.5, 1.0 - (rate - 0.30))

Baseline: -5.0 (model must beat this to be "kept")
```

**Deploy Lifecycle:**
```
boot  → deploy H100, wait for SSH, verify GPU
start → upload code + data + best_model.pt (warm-start)
        → pre-flight on H100 (data, GPU, API)
        → launch loop + watchdog
sync  → poll every 60s, auto-download improvements, heartbeat
stop  → SIGTERM → wait 15s → SIGKILL → download all → copy weights → close
```
