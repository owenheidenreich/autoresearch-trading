# ART² v2 Program

You are an autonomous researcher improving a trading model that trades SPX 0DTE long options. The model learns to emit TradeIntents (trade/no-trade, direction, strike, stop, target, hold). Replay simulates those intents against historical data. Score measures the equity curve. Read this file, then start experimenting.

For command syntax, see `v2/COMMANDS.md`. For detailed specs, see `v2/docs/`.

## Compute

**All training runs on Akash H100 GPU.** Never run training locally -- the dev machine is a MacBook. Local use is limited to: editing code, committing, reading results, and running replay/evaluation (which is CPU-friendly).

Akash workflow:
1. Boot GPU: `./v2/ops/deploy.sh boot`
2. Upload code + data: `./v2/ops/deploy.sh start`
3. Training runs on the remote GPU
4. Download results: `./v2/ops/deploy.sh sync`
5. Evaluate locally: `python -m v2.replay --model v2/model.pt --mask promote`
6. Shut down GPU: `./v2/ops/deploy.sh stop`

The experiment runner (`v2/ops/run_experiment.py`) is designed to run ON the GPU machine, not locally.

## Data

**47 features** (28 price/market structure + 11 option/Greeks + 8 volume/flow).
All features computed from raw data in `v2/pipeline/compute_features.py`.
**Wide-grid option data**: ATM +/- 100pt (82 contracts per day), full OHLCV per bar per strike.
**Labels**: Dual-direction P&L (both call AND put simulated per bar). Model learns which direction from features.

Key design: all features are RELATIVE (moneyness %, normalized prices) so patterns
learned at SPX 4300 transfer to SPX 6500. Rolling z-score normalization (60-day window)
preserves inter-day regime information (unlike per-day z-score which destroyed it).
Option features use dynamic ATM tracking from the wide grid.

Label statistics:
- ~237K signal bars, label_call_pnl and label_put_pnl per bar
- Fixed risk: stop=0.30, target=0.50, hold=30 bars (no grid search)
- Direction = whichever side had higher P&L (model learns to predict)
- Costs: adaptive spread (by time/VIX) + $1.30 commission per round trip

## Setup

1. **Create a branch**: `git checkout -b autoresearch/v2-<tag>` from current main.
2. **Read the in-scope files**:
   - This file (`v2/program.md`) -- your instructions.
   - `v2/train.py` -- the model and training loop. You modify this.
   - `v2/core/policy.py` -- the trading policy. You can modify this too.
   - `v2/lab_notebook.md` -- experiment log.
3. **Verify data**: `v2/data.pt` must exist (check metadata version = `v2_wide_grid_risk_search`).
4. **Boot Akash GPU** and establish baseline.
5. **Record baseline** in `v2/results.tsv`.

## What You CAN Modify

Two files only:

- **`v2/train.py`** -- Model architecture, loss function, hyperparameters, optimizer, batch construction, head design. Everything about how the model learns.
- **`v2/core/policy.py`** -- Gate threshold, risk output ranges, cooldown bars, time blocks, order style, exit policy. Everything about how model outputs become trading decisions.

## What You CANNOT Modify

Everything else. These are the immutable evaluation harness:

- `v2/core/simulator.py` -- how trades play out
- `v2/core/metrics.py` -- how score is computed
- `v2/core/walkforward.py` -- walk-forward CV fold generation and execution
- `v2/replay.py` -- how model outputs become trades and get evaluated
- `v2/core/labels.py` -- how labels are generated
- `v2/core/schema.py` -- TradeIntent and SimulatedTrade contracts
- `v2/data.pt` -- the dataset
- `v2/ops/run_experiment_wf.py` -- the walk-forward experiment runner
- `v2/ops/deploy.sh` -- GPU deployment and `run_one` command

## The Goal

**Get the highest score.** The score is:

```
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```

Where:
- `daily_sortino` = Sortino ratio from daily dollar returns on a $10K equity curve
- `positive_day_rate` = fraction of traded days that were profitable
- `dd_mult` = 1.0 when max drawdown <= 8%, linear decay to 0.0 at 20%

Hard gates (score goes negative if any fail):
- Minimum 30 trades
- Minimum 15 traded days
- At least 15% minority direction (must trade both calls and puts)
- Max account drawdown <= 20%

Model must also beat all four baselines:
1. **Random**: 2% entry probability, random contract, STOP_TP_TIME exits, 5 seeds averaged
2. **ATM-Always**: buy ATM call at bar 30 every day, STOP_TP_TIME exits
3. **Simple-Rules**: 5-bar momentum heuristic with 10-bar cooldown, STOP_TP_TIME exits
4. **ATM-Trailing**: buy ATM call at bar 30 every day, TRAILING exits with model's risk params (the honest test -- isolates neural net value from exit strategy)

## Key Architecture Facts

- **Input**: (batch, 30, 47) -- 30 bars of 47 features (28 price/market + 11 option/Greeks + 8 volume/flow)
- **Output**: P&L predictions (call_pnl, put_pnl) + risk params. Gate/direction derived from P&L.
- **Labels**: Dual-direction P&L -- both call and put simulated per bar with fixed risk params
- **Evaluation**: Walk-forward CV (5 folds, 300 test days across diverse market regimes)
- **Score**: Account curve health (Sortino * consistency * drawdown guard), averaged across folds
- **Equity**: $10K starting, $100 SPX multiplier, 1 contract max
- **Training**: Akash H100 GPU. ~25 minutes per experiment (5 folds x 5 min). Never local.
- **Costs**: adaptive spread (by time-of-day, VIX, moneyness) + $1.30 commission per round trip

## Walk-Forward Evaluation

Experiments use walk-forward cross-validation: train on the past, test on the future, slide forward. This tests the model across diverse market regimes (bull, bear, chop) instead of a single 60-day window.

| Fold | Train Days | Val (last 40) | Test Days (60) |
|------|-----------|---------------|----------------|
| 0 | 0-673 | 634-673 | 674-733 |
| 1 | 0-733 | 694-733 | 734-793 |
| 2 | 0-793 | 754-793 | 794-853 |
| 3 | 0-853 | 814-853 | 854-913 |
| 4 | 0-913 | 874-913 | 914-973 |

- **300 total test days** across 5 folds (Dec 2024 -- Mar 2026)
- **Shadow**: last 20 days (974-993) reserved for final live-readiness check
- **Score**: mean of per-fold scores. Each fold scored independently with hard gates.
- **Seed**: each fold uses a different random seed (base_seed + fold_idx)
- **Production model**: last fold's model (most training data) is saved as model.pt

## The Experiment Loop

One-time setup: `./v2/ops/deploy.sh boot` then `./v2/ops/deploy.sh start`.

LOOP:

1. Look at last experiment results. Decide what to try. Write your hypothesis.
2. Edit `v2/train.py` and/or `v2/core/policy.py`.
3. `git commit` your changes.
4. `./v2/ops/deploy.sh run_one exp_NNN` -- uploads code, trains 5 walk-forward folds on GPU (~25 min), downloads model.pt.
5. Read the aggregate score from stdout (mean of 5 fold scores). Also check per-fold scores for consistency.
6. If crashed: read the log, try to fix. If unfixable, log as crash, move on.
7. If score improved AND beats all baselines: **KEEP**. Branch advances.
8. If score equal or worse: **REVERT**. `git checkout HEAD~1 -- v2/train.py v2/core/policy.py`.
9. Append result to `v2/results.tsv`.
10. Check session limits (see below). If any limit hit, stop.
11. Go to step 1.

## Session Limits

| Limit | Threshold | Action |
|-------|-----------|--------|
| Experiments | 20 max | Stop session (~25 min each = 8+ hours) |
| Time | 10 hours | Stop session |
| No-improve streak | 6 consecutive reverts | Stop, rethink approach |
| Plateau | 4 hours without improvement | Stop session |
| Crash storm | 3 consecutive crashes | Stop, fix infrastructure |

## When You're Stuck

If you hit 3+ consecutive reverts:

1. **Stop trying random things.**
2. Read the trade-level replay data. Look at which trades lost money and why.
3. Form a hypothesis about WHY the model is failing.
4. Try structural changes, not just hyperparameter tweaks.
