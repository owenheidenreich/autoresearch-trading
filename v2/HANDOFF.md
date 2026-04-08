# Handoff: Walk-Forward CV Session

Read this before running the experiment loop. It explains what changed and why.

## What Happened Last Session (2026-04-07)

### Experiment Loop (20 experiments, exp_032-051)

Ran the first experiments on a rebuilt 47-feature dataset (rolling z-score normalization). Score went from 4.92 to 5.67 through four changes:

1. **Lookback 60->30** (exp_034): shorter context reduced overfitting
2. **P&L sample weighting** (exp_036): `1 + |max_pnl|` in loss focuses on high-signal bars
3. **Seed 42->123** (exp_043): different random init matters a lot for this model
4. **Asymmetric loss 3x->5x** (exp_046): stronger penalty for predicting profit on actual losers

### The Problem That Was Identified

The 5.67 score was achieved by making the model extremely selective: 77 trades across 36 of 60 test days, sitting out 40% of days entirely. The user correctly identified this as **reward hacking** -- the model learned to avoid trading rather than to trade well.

The root cause: the score formula `min(sortino, 6.0) * positive_day_rate * dd_mult` makes PDR the binding constraint once sortino caps at 6.0. Optimizing solely for PDR leads to over-conservative models that cherry-pick easy days.

**Key feedback from the user:**
- Losing days are normal. Don't try to eliminate them all.
- Over-trading (trading too often, giving back gains) is the real risk. Low trade count is fine.
- But a model that hides from 40% of trading days is not a real edge.

### Walk-Forward CV (New)

The old evaluation used a single 60-day promote window. That's too thin -- a few lucky days swing the score. We replaced it with **walk-forward cross-validation**:

- 5 folds, each training on all prior data, testing on the next 60 days
- 300 total test days covering Dec 2024 through Mar 2026
- Final score = mean of 5 fold scores
- Each fold uses a different random seed (base_seed + fold_idx)
- ~25 min per experiment (5 x 5 min training)
- Last fold's model saved as model.pt (most training data)

This is now the default when you run `deploy.sh run_one`.

## Current State of train.py

```
Lookback: 30 bars
d_model: 64, depth: 3, n_heads: 4, dropout: 0.05
Batch: 2048, LR: 5e-4, weight_decay: 0.01
Seed: 123 (but walk-forward overrides per fold)
Asymmetric loss: 5x for optimistic errors (predicted profit, actual loss)
Sample weighting: 1 + |max_pnl| (up-weight high-signal bars)
Huber delta: 0.5
```

### What the Model Does

P&L prediction model. Predicts expected call_pnl and put_pnl for every bar. Trading decisions derived at inference:
- Gate: trade when max(call_pnl, put_pnl) > 0 (sigmoid threshold 0.50)
- Direction: argmax(call_pnl, put_pnl)
- Strike: always ATM
- Risk: model outputs stop/target/hold, squashed to policy ranges

Architecture: TransformerEncoder (3 layers, causal mask) + FiLM regime conditioning on last bar's raw features.

## What to Do Next

### First Experiment

The walk-forward setup has NOT been tested on GPU yet. The first experiment should be a baseline run with the current train.py to establish the walk-forward score. This score will be different from the old single-split score of 5.67.

```
1. deploy.sh boot
2. deploy.sh start
3. deploy.sh run_one exp_052
4. Read the aggregate score + per-fold breakdown
5. Log as the new baseline
```

Expect ~25 min for the run. If any fold crashes, check the error -- it might be a mask/data issue since this is the first walk-forward run on GPU.

### What to Optimize For

The score formula hasn't changed, but now it's averaged across 5 diverse market windows. A model that only works in one regime will score poorly. Focus on:

- **Consistency across folds**: low std_fold_score means the model generalizes
- **Trade quality**: win rate, profit factor, not just PDR
- **Trading in diverse conditions**: the model should trade across all 5 folds, not just the easy ones

Do NOT try to make every day profitable. Losses are part of trading.

### The 5x Asymmetric Loss Question

The 5x asymmetric loss was tuned to maximize PDR on a single 60-day window. Under walk-forward, it might be too conservative (the model may refuse to trade in some folds). The first thing to test after the baseline might be **dialing it back to 3x** and seeing if that produces a better walk-forward score with more consistent trading across folds.

### Known Issues

- `v2/analyze_losses.py` still uses the old fixed promote_mask. It works for quick trade inspection but doesn't cover all 5 fold windows.
- Local `python -m v2.replay --mask promote` only evaluates on the old 60-day window. The real walk-forward score comes from the experiment runner on GPU.
- The `.best_score` file still tracks single-split scores. Walk-forward scores will likely be lower (harder test). Need to reset it after the first walk-forward baseline.
