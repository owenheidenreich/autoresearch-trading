# v4 SPXW 0DTE Autoresearch Program

This program adapts the small-loop idea from `karpathy/autoresearch` to the
SPXW 0DTE neural prototype. The loop is intentionally conservative because the
target is real options trading risk, not a benchmark game.

## Research Objective

Train a neural action model that decides each minute among:

1. no trade
2. buy one nearest-ATM SPXW call
3. buy one nearest-ATM SPXW put

Every trade is one contract, long-only, ask-entry/bid-exit, commission-excluded, and
flat before close through the existing v4 label policies.

## Fixed Data Rules

- Use v4 Databento-derived SPXW CBBO/OHLCV/statistics data as label truth.
- Use v2/v3 only for feature ideas, priors, pretraining candidates, or regime
  diagnostics. Do not use v2/v3 proxy bid/ask labels as executable PnL truth.
- Do not download paid data inside an autoresearch loop.
- Do not change train/validation/test dates inside a loop.

## Split Discipline

- January 2026: neural training.
- First half of February 2026 sessions: early stopping/calibration.
- Second half of February 2026 sessions: research selection score.
- March 2026: protected audit holdout.

The loop may report March metrics, but it must not use March to choose a trial,
architecture, threshold, time filter, policy horizon, or next experiment.

## Anti-Reward-Hacking Contract

- All trials are pre-registered in code before running.
- Every trial is logged, including failures and no-trade outcomes.
- Selection score is computed only from the February selection split.
- Test metrics are audit-only and must not affect the selected champion.
- A trial with too few trades cannot win by avoiding risk.
- Top-day concentration is penalized so one lucky day cannot dominate.
- Drawdown, positive-day breadth, and profit factor are part of selection.
- Any future change to this contract requires a new loop ID and must cite why.

## First Loop

Loop `v4_autoresearch_001` tests whether the existing action neural network can
use its no-trade output directly instead of relying on a post-training threshold
grid. The trial surface is deliberately small:

- all regular-session minutes
- skip the first 30 minutes
- post-open morning plus late afternoon
- late afternoon only
- low and modest raw edge floors against the no-trade score
- simple daily trade limits and one daily loss stop

If this loop does not produce a validation-selected champion with credible March
audit behavior, the next work should improve the training objective itself, not
add more selection knobs.
