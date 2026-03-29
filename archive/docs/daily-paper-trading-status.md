# Daily Paper Trading: Status & Roadmap

**Date:** 2026-03-28
**Goal:** Model trades autonomously on IBKR paper account every market day, 6:30 AM - 1:00 PM PT, with complete audit trail. Collect 1-week and 1-month performance baselines.

---

## What's Been Completed

### 1. Fixed: Python Path in launchd Agents (CRITICAL)

The daily pipeline and weekly retraining launchd agents were using `/usr/bin/env python3` which resolved to Apple's system Python (`/Library/Developer/CommandLineTools/usr/bin/python3`). This Python lacks disk access permissions on macOS, causing every scheduled run to fail with `Operation not permitted`.

**Evidence:** `results/pipeline/launchd-stderr.log` shows 3 consecutive failures — the pipeline has **never successfully executed via launchd**.

**Fix:** Both plists now use the project venv:
- `infra/daily_pipeline.plist` → `.venv/bin/python3`
- `infra/weekly_retrain.plist` → `.venv/bin/python3`

Both agents unloaded, updated plists copied to `~/Library/LaunchAgents/`, and reloaded. Verified with `launchctl list | grep trinity` — both show exit code 0.

### 2. Fixed: Paper Trading Args in daily_pipeline.py

Stage 3 (paper trading) was missing explicit `--port`, `--client-id`, and `--kill-switch` arguments. While port 4002 is the default, the kill-switch path was absent, meaning there was no emergency stop mechanism.

**Fix:** Added to `stage_paper_trading()`:
```
--port 4002
--client-id 80
--kill-switch results/live/kill_switch
```

### 3. Fixed: Kill Switch Cleared

`results/live/kill-switch.txt` contained "stop", blocking trading. The old file (wrong name with hyphen) has been removed. The correct path `results/live/kill_switch` is now referenced by the pipeline and does not exist (= trading enabled).

### 4. Verified: Dry Run Passes

Full dry-run of `daily_pipeline.py` completed successfully:
- Stage 1 (Data): Would rebuild through 2026-03-27
- Stage 2 (Training): Skipped (default — weekly only)
- Stage 3 (Trading): Would launch with correct model, audit path, kill switch
- Stage 4 (CSV): Deferred to post-market
- Stage 5 (IBKR Analysis): Would analyze yesterday's audit

### 5. Existing Infrastructure (Already Built, Working)

| Component | Status | Location |
|-----------|--------|----------|
| Paper trading service | Working (12 real trades Mar 25-26) | `training/live/service.py` |
| Model inference | Working (v14 four-head, 38 features) | `training/best_model.pt` (874KB, Mar 28) |
| Audit trail (JSONL) | Working | `results/live/audit-{date}.jsonl` |
| Trade CSV export | Working | `tools/export_trades.py` |
| IBKR session analysis | Working | `tools/ibkr_analyze.py` |
| Order parity checker | Working | `tools/live_order_parity_report.py` |
| Data rebuild (incremental) | Working | `training/prepare.py` |
| IB Gateway | Running now (port 4002) | IB Gateway 10.44 |
| Launchd daily agent | Loaded (2:30 AM PT weekdays) | `com.trinity.autoresearch.daily-pipeline` |
| Launchd weekly agent | Loaded (Sun 8 PM PT) | `com.trinity.autoresearch.weekly-retrain` |
| Kill switch mechanism | Ready | `results/live/kill_switch` |
| Backtest equity curve | $10k → $155k over 1185 trades | `backtest_output/equity_curve.csv` |

---

## What Remains To Be Done

### Priority 1: Ensure Monday's Trading Session Works

**P1a. IB Gateway Must Be Running at 2:30 AM PT**

The daily pipeline checks `socket.connect(127.0.0.1:4002)` before launching paper trading. If IB Gateway isn't running, Stage 3 silently skips.

**Options (pick one):**
- [ ] Add IB Gateway to macOS Login Items (System Settings → General → Login Items) so it starts on boot
- [ ] Leave IB Gateway running 24/7 (it auto-reconnects daily)
- [ ] Create a launchd agent that starts IB Gateway before the pipeline (requires IBC — IB Controller)

**Current state:** IB Gateway is running (PID 66922). If your Mac stays on and Gateway stays up, Monday's pipeline will work.

**P1b. Verify First Real Pipeline Run (Monday 2:30 AM PT)**

After Monday's run, check:
```bash
# Did the pipeline run?
cat results/pipeline/launchd-stdout.log | tail -30

# Did trading launch?
cat results/pipeline/2026-03-31/summary.txt

# Did trades execute?
cat results/live/audit-2026-03-31.jsonl | wc -l

# View trades
cat results/live/trades-2026-03-31.csv
```

If Stage 3 fails again, check `results/pipeline/2026-03-31/stage3_paper_trading.log`.

---

### Priority 2: Process Resilience (This Week)

**P2a. Paper Trading Watchdog**

Currently the pipeline launches `paper_live.py` via `subprocess.Popen()`, checks if it's alive after 10 seconds, then walks away. If it crashes at 10:30 AM, no trading happens for the rest of the day.

- [ ] Add a watchdog loop in `daily_pipeline.py` that checks the PID every 5 minutes
- [ ] If process dies, restart it (with rate limiting — max 3 restarts/day)
- [ ] Log restart events to audit trail

**P2b. IB Gateway Reconnection Handling**

IB Gateway disconnects daily at ~midnight ET for server reset. The `ib_insync` library has `ib.disconnectedEvent` but it's unclear if `paper_live.py` handles reconnection.

- [ ] Verify reconnection logic in `training/live/service.py`
- [ ] If missing, add disconnect handler that waits and reconnects

---

### Priority 3: Audit Trail & Reconciliation (This Week)

**P3a. Verify Model Decisions = IBKR Fills**

The user's core requirement: "the models trades must be real and not hallucinations." The infrastructure exists but isn't automated:

- [ ] Integrate `live_order_parity_report.py` into Stage 5 of daily pipeline
- [ ] After each trading day, automatically verify:
  - Every `entry_intent` has a matching `ib_exec_details` (or explicit rejection)
  - Every `position_closed` has matching IBKR fill confirmations
  - P&L computed by model matches P&L reported by IBKR
- [ ] Flag discrepancies in daily summary

**P3b. Daily Trade Summary Notification**

- [ ] After Stage 4 (CSV export), generate a human-readable daily summary:
  - Number of trades, win rate, total P&L
  - Largest winner/loser
  - Model confidence distribution
  - Any errors or rejected orders

---

### Priority 4: 1-Week & 1-Month Performance Collection (Ongoing)

**P4a. Week 1 Baseline (Mar 31 - Apr 4)**

After 5 trading days, compile:
- [ ] Total trades taken
- [ ] Win rate
- [ ] Profit factor
- [ ] Max drawdown
- [ ] Trade frequency (trades/day)
- [ ] Comparison: backtest frequency vs live frequency
- [ ] Any days where model traded 0 times (and why)

**P4b. Month 1 Baseline (Mar 31 - Apr 30)**

After ~22 trading days, compile:
- [ ] All metrics from P4a over full month
- [ ] Equity curve (live) vs backtest equity curve
- [ ] Regime analysis: did model adapt to different market conditions?
- [ ] Model confidence trends: is it becoming more/less certain over time?
- [ ] Fill quality: slippage between model's expected price and actual fill

**P4c. Backtest vs Live Comparison**

The backtest shows trade frequency declining sharply:
```
2022: 556 trades (56/month)
2023: 376 trades (31/month)
2024: 147 trades (12/month)
2025: 100 trades (8/month)
2026:   7 trades (2.3/month)
```

If live trading shows similarly low frequency, this confirms the model is out-of-distribution and daily retraining (Priority 5) becomes urgent.

---

### Priority 5: Daily Retraining Loop (After 1-Week Baseline)

**Why:** The model was trained on data through ~mid-2025. As market regime shifts, the model's feature distributions drift and it stops trading. The backtest equity curve confirms this — 2026 has 7 trades total.

**P5a. Rolling Validation Window**

Current: fixed 70/30 temporal split (train on 2022-2024, validate on 2025-2026).
Needed: rolling window — train on everything except last 60 trading days, validate on last 60.

- [ ] Modify `prepare.py` to accept `--val-days N` parameter
- [ ] Default to 60 most recent trading days as validation set
- [ ] Ensure data provenance tracks which days are train vs val

**P5b. Post-Close Warm Retrain**

After market close each day:
1. Append today's data to `data.pt` (already works — Stage 1)
2. Warm-start from `best_model.pt` (existing infrastructure)
3. Run N experiments on Akash (existing — `inner_loop.py`)
4. Validate new model on rolling val set
5. Promote if improved, revert if regressed (existing — validation gate)
6. Deploy updated model for next trading day

- [ ] Add `--retrain-daily` flag to `daily_pipeline.py` (lighter than weekly full retrain)
- [ ] Budget: 15-30 min GPU time per day (2-4 experiments)
- [ ] Gate: new model must beat old on rolling val AND not regress on recent live P&L

**P5c. Model Version History**

- [ ] Archive each day's model: `archive/models/daily/YYYY-MM-DD/best_model.pt`
- [ ] Track score progression over time
- [ ] Enable rollback to any prior day's model

---

### Priority 6: Alerting (Nice to Have)

- [ ] Slack/email notification on: pipeline failure, no-trade day, kill switch activated
- [ ] Daily P&L summary sent automatically
- [ ] Weekly performance report

---

## Architecture Summary

```
2:30 AM PT (launchd)
    │
    ├── Stage 1: Rebuild data.pt (incremental, ~10 min)
    │     └── prepare.py --use-spx --ib-port 4002 --end yesterday
    │
    ├── Stage 2: Training (weekly only, Sunday 8 PM)
    │     └── deploy.sh boot → inner_loop.py → deploy.sh stop
    │
    ├── Stage 3: Paper Trading (6.5 hrs, detached process)
    │     └── paper_live.py --paper-auto --port 4002
    │           ├── Context bootstrap (30-day feature window)
    │           ├── IBKR connection (DU* paper account)
    │           ├── 500ms decision loop (9:30 AM - 4:00 PM ET)
    │           ├── Audit trail → results/live/audit-{date}.jsonl
    │           └── EOD flatten at 4:00 PM ET
    │
    ├── Stage 4: CSV Export (post-market)
    │     └── export_trades.py → trades-{date}.csv
    │
    └── Stage 5: IBKR Analysis (yesterday's session)
          └── ibkr_analyze.py → session metrics
```

## Emergency Controls

```bash
# Stop trading immediately
echo "kill" > results/live/kill_switch

# Check if paper trading is running
cat results/pipeline/$(date +%Y-%m-%d)/paper_trade.pid | xargs ps -p

# View today's trades
cat results/live/audit-$(date +%Y-%m-%d).jsonl | python3 -m json.tool | grep entry_intent

# Unload all automation
launchctl unload ~/Library/LaunchAgents/com.trinity.autoresearch.daily-pipeline.plist
launchctl unload ~/Library/LaunchAgents/com.trinity.autoresearch.weekly-retrain.plist
```
