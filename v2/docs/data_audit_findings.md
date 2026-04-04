# Data Audit Findings (2026-04-03)

## Finding 1: Oracle labels are 100% winners (CRITICAL)

All 286,053 trade labels have P&L > 0. Zero losing trades in training.
The model learns to classify "oracle-profitable bars" not to trade.
PF=388, WR=84% are artifacts of this labeling, not real performance.

## Finding 2: Moneyness drift (CRITICAL)

Strikes are classified relative to OPENING ATM (fixed per day). By 11am,
53% of days have SPX drifted $10+ from open. By 2pm, 22% have drifted $20+.

The "ATM call" at 2pm might actually be 15 points ITM or OTM. The model
has no feature that tracks CURRENT moneyness -- only opening-day labels.

The model's strike selection head is choosing from stale classifications.

## Finding 3: No real bid/ask data

Spread costs are estimated from premium tier assumptions (30-150 bps by
time of day). No actual bid/ask quotes from the market. Real spreads
vary by strike, volume, and market conditions.

## Finding 4: Simple signals have no edge

Tested momentum (ret_6 + ema_cross) with honest triple-barrier labels across
all stop/target/hold combinations. PF=0.49-0.54. Even with zero spreads: PF=0.52.
The causal features as currently constructed don't predict option P&L.

## Finding 5: Missing trader-essential data

- No vega (IV sensitivity)
- No per-strike volume
- No open interest
- No bid/ask
- No real-time moneyness feature

## What's sound

- Option prices track the same contract all day (no contract switching)
- 39 features well-engineered for SPX price action
- 4 years of history (sufficient for deep learning)
- Simulator correctly models same-contract entry/exit with realistic spreads
- Infrastructure (deploy, train, replay, scoring) all working end-to-end

## What must be fixed before training

1. Add current-moneyness features (how far each strike is from current SPX)
2. Replace oracle labeler with triple-barrier (includes losing trades)
3. Get real bid/ask data from Polygon (or estimate from option price dynamics)
4. Add walk-forward validation (not just fixed 60-day holdout)
