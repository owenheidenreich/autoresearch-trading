# Daily Automation Pipeline

Two-mode automation: **daily** (data rebuild + trade) and **weekly** (data rebuild + retrain + trade).

## Why Two Modes?

The model uses rolling z-score normalization — features like "VIX is 2 std above recent mean" are regime-invariant by design. Daily data rebuilds keep the normalization fresh. But daily **retraining** causes problems:

- **Architecture instability**: The autoresearch agent proposes random mutations. Daily retraining means model behavior can change unpredictably day-to-day.
- **45 min is too short**: Breakthrough runs historically took 50+ experiments. Short daily sessions risk marginal or harmful changes.
- **Compounding drift**: Small random-walk changes accumulate over weeks into large, uncontrolled drift.

The solution: **trade daily with a stable model** (fresh data/normalization), **retrain weekly** with enough time for the autoresearch agent to explore meaningfully.

## What It Does

### Daily Mode (weekdays, default)
```
5:30 AM ET   Stage 1: Data rebuild (incremental)
             - SPY/SPX/VIX: appends new bars to existing caches (1 week of IBKR data)
             - SPXW: downloads yesterday's option chain (Polygon S3, per-day cache)
             - Rebuilds aggregate option caches from per-day files
             - Full recompute of features + rolling z-score normalization
             - Refreshes norm_raw_buffer (500 bars) for live inference
             - ~2-3 minutes (vs ~30-45 min for cold start)

5:45 AM ET   Stage 3: Paper trading
             - Launches paper_live.py as background process
             - Waits internally for 9:30 AM market open
             - Trades 9:30 AM - 4:00 PM ET using existing best model
             - Audit trail → results/live/audit-YYYY-MM-DD.jsonl

4:00 PM ET   Stage 4: CSV export
             - Converts audit JSONL to results/live/trades-YYYY-MM-DD.csv
             - Writes summary with win rate, P&L, trade count
```

### Weekly Mode (--retrain, Sunday evening)
```
8:00 PM PT   Stage 1: Data rebuild (same as daily)

8:15 PM PT   Stage 2: Akash training
             - Backs up current best_model.pt → best_model.pt.prev
             - Boots H100 GPU on Akash (~2 min)
             - Runs autoresearch loop for ~2 hours (~20-30 experiments)
             - Downloads best model + train.py, closes deployment
             - Validation gate: only promotes new model if score improves
             - If score didn't improve, restores previous model
             - ~2.5 hours total
```

## Files

| File | Purpose |
|------|---------|
| `tools/daily_pipeline.py` | Main orchestrator — sequences all stages |
| `tools/export_trades.py` | JSONL audit → CSV converter + daily summary |
| `infra/daily_pipeline.plist` | macOS launchd: daily data+trade (2:30 AM PT weekdays) |
| `infra/weekly_retrain.plist` | macOS launchd: weekly retrain (Sunday 8 PM PT) |

## Usage

```bash
# Daily mode (default) — rebuild data, trade with existing model
python3 tools/daily_pipeline.py

# Weekly retrain — rebuild data, retrain on Akash, validate, then trade
python3 tools/daily_pipeline.py --retrain

# Longer training session (2 hours instead of default 45 min)
python3 tools/daily_pipeline.py --retrain --training-minutes 120

# Dry run — see what would happen
python3 tools/daily_pipeline.py --dry-run

# Skip data rebuild
python3 tools/daily_pipeline.py --skip-data

# Just export trades for a specific date
python3 tools/daily_pipeline.py --export-only 2026-03-20

# Export CSV directly
python3 tools/export_trades.py results/live/audit-2026-03-20.jsonl
```

## Installing the Schedules

```bash
# Daily (weekday data rebuild + trade)
cp infra/daily_pipeline.plist ~/Library/LaunchAgents/com.trinity.autoresearch.daily-pipeline.plist
launchctl load ~/Library/LaunchAgents/com.trinity.autoresearch.daily-pipeline.plist

# Weekly (Sunday retrain)
cp infra/weekly_retrain.plist ~/Library/LaunchAgents/com.trinity.autoresearch.weekly-retrain.plist
launchctl load ~/Library/LaunchAgents/com.trinity.autoresearch.weekly-retrain.plist

# Verify
launchctl list | grep trinity

# Uninstall
launchctl unload ~/Library/LaunchAgents/com.trinity.autoresearch.daily-pipeline.plist
launchctl unload ~/Library/LaunchAgents/com.trinity.autoresearch.weekly-retrain.plist
```

## Manual Dependency

IB Gateway must be running before the pipeline starts. It requires 2FA login so it cannot be auto-started. Start it once and leave it running — it has an auto-restart setting in its config. The pipeline checks IB Gateway connectivity at startup and fails fast if unavailable.

## Design Decisions

**Why full feature recompute instead of append?** `prepare.py` uses rolling z-score normalization across the entire dataset. Appending new bars would corrupt the normalization of earlier bars. Full recompute from cached raw data takes only ~60 seconds — the download used to be the slow part.

**Why incremental downloads?** SPY/SPX/VIX caches are monolithic pkl files. Previously, the pipeline deleted them to force re-download including yesterday's data — triggering a 30-45 minute full re-download from IBKR. Now `_incremental_update()` in prepare.py loads the existing cache, finds the last date, downloads only new bars (1 week at most), appends, and saves. Per-day SPXW option caches (`~/.cache/.../spxw/YYYY-MM-DD.pkl`) were always incremental.

**Why not retrain daily?** Z-score normalization makes features regime-invariant — model weights don't need daily updates to handle market changes. Daily autoresearch sessions are too short (7-10 experiments) for meaningful improvement and risk architectural instability. Weekly sessions (20-30+ experiments) give the agent room to explore properly.

**Why a validation gate?** The autoresearch agent doesn't guarantee improvement. The validation gate compares the new model's score against the previous best and only promotes if improved. If the weekly retrain doesn't help, the proven model continues trading — no wasted trading days.

**Why training failure is non-fatal.** If Akash is unavailable, the pipeline logs the error and paper trades with the previous best model.

## Output Locations

```
results/pipeline/YYYY-MM-DD/          Per-day pipeline logs
  summary.txt                         Pipeline stage results
  stage1_data_rebuild.log             prepare.py output
  stage2_boot.log                     Akash boot output (retrain only)
  stage2_start.log                    Training start output (retrain only)
  stage2_stop.log                     Training stop + download output (retrain only)
  stage3_paper_trading.log            Paper trading stdout/stderr
  paper_trade.pid                     Paper trading process ID

training/.best_score                  Last promoted model's score (for validation gate)
training/best_model.pt.prev           Previous model backup (for rollback)

results/live/audit-YYYY-MM-DD.jsonl   Raw audit trail (all events)
results/live/trades-YYYY-MM-DD.csv    Trade-level CSV
results/live/summary-YYYY-MM-DD.txt   Daily trade summary
```

## Created

2026-03-20, as part of the IBKR paper trading automation milestone.
Updated 2026-03-20: split into daily (data+trade) and weekly (retrain) modes.
