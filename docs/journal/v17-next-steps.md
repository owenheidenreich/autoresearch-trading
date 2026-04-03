# v17 Next Steps (Post-ART2 Loop)

**Date:** 2026-04-02
**Model:** v17 PredictionModel, score 0.959, dir accuracy 68.2%

## Problem Statement

The model predicts SPX direction well (71%) but the execution layer wastes the edge:
- MODEL_EXIT never fires (0 of 197 trades)
- 46% of trades hit stop loss at -33% avg
- Short holds (<30 bars) have 10% WR vs 66% for 30+ bars
- All profit comes from EOD exits (84% WR, PF 56.5)

## Track Results

### Track 1: Direction Reversal Exit -- KEPT
**Files:** training/replay.py
**Change:** Added direction reversal exit: if model predicted UP at entry but now predicts DOWN (or vice versa) with >0.05% magnitude, exit after MIN_HOLD_BARS.
**Result:** 72 DIRECTION_REVERSAL exits (27% of trades). Avg hold dropped 54 -> 35 bars. PF dropped 4.0 -> 3.21 (cuts some winners early). Max DD improved.

### Track 2: Wider Stops -- KEPT
**Files:** training/prepare.py
**Change:** DYNAMIC_STOP_BASE from 0.35 to 0.45.
**Result:** Stop rate dropped 46% -> 32%. WR improved 45% -> 49%. Total return improved. Avg loser deeper (-27.5% -> -32.5%) but fewer stops means more trades survive to profit.

### Track 3: Trading Rules Engine -- KEPT
**Files:** training/replay.py (wired to training/trading_rules.py)
**Change:** Replaced inline cooldown/pre-10am/lunch checks with trading_rules.should_enter(). Adds power hour blocking (bar 330+) and lunch chop suppression (2x confidence required).
**Result:** WR improved 49% -> 55%. PF improved 3.24 -> 4.24. Total return +13,000%. Same trade count but much higher quality from time-of-day filtering.

### Track 4: OTM Strike Selection -- REVERTED
**Files:** training/replay.py
**Change:** Routed high-confidence to ATM, moderate to OTM5, low to OTM10.
**Result:** PF dropped 4.24 -> 3.61. Max DD worsened -14.4% -> -17.1%. OTM options have wider spreads and deeper percentage losses on stops. ATM-only is correct for this model.

## Cumulative Progress

| Metric | Baseline | Final (T1+T2+T3) | Change |
|--------|----------|-------------------|--------|
| Trades | 197 | 197 | same |
| Win rate | 46% | 55% | +9pp |
| Profit factor | 4.00 | 4.24 | +6% |
| Total return | +9,900% | +13,000% | +31% |
| Max drawdown | -14.8% | -14.4% | better |
| Stop rate | 46% | 30% | -16pp |
| Avg hold | 54 bars | 46 bars | -15% |
| Direction reversal exits | 0 | 74 (36%) | new |

## Conclusion

The v17 redesign produced a model with genuine directional edge (68% direction accuracy, 71% on replay trades). Three execution-layer improvements (direction reversal exit, wider stops, trading rules engine) increased win rate from 46% to 55% and profit factor from 4.0 to 4.24 with zero GPU cost.

Key insight: the model's prediction quality is the bottleneck now, not the execution layer. The execution improvements squeezed out the easy gains. Further improvement requires either:
1. Better predictions (more GPU training, PBT hyperparameter tuning)
2. Confidence calibration (the model's trade_prob correlates weakly with outcome quality)

## Next Steps

| Priority | Action | Why | Effort |
|----------|--------|-----|--------|
| 1 | PBT sweep on stable GPU | Akash crashed twice. Need a stable provider or shorter sweep (pop 4, gen 2). The model has room to improve with hyperparameter tuning. | GPU, 1 hour |
| 2 | Train ACTION_W=0.0 | The action head contributes nothing (exit_signal never fires, trade_prob weakly predictive). Removing it lets the model focus entirely on return prediction. | GPU, 1 experiment |
| 3 | Trailing stop integration | The trading_rules.py trailing stops (Pickles' thirds) are defined but not wired into replay. Could lock profits earlier on big winners. | No GPU |
| 4 | Paper trading validation | The backtest shows +13,000% but this is hindsight-inflated. Paper trade for 1 full day on IBKR to validate execution realism. | IBKR, 1 day |
| 5 | Walk-forward cross-validation | Currently using fixed 70/30 split. Implement rolling 60-day train / 20-day validate windows to check if the edge is consistent across time. | GPU, 5 experiments |
