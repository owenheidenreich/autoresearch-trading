# Lab Notebook

## System
SPX 0DTE | 2-head gate+dir | 60 features + position state | 8 actions | 1-min bars | learned exits (no hardcoded TP)

## Critical Change (2026-03-19)
Removed hardcoded 20% profit target. Model now learns its own exits via gate head (NO_TRADE while holding = exit).
Emergency 30% SL remains as backstop. Score formula expanded with 9 new tunable knobs.
EXIT labels now use hindsight-optimal timing instead of fixed threshold.
MAX_TRADE_RETURN raised from 2.0 to 5.0 to allow larger winners.
All prior scores are incomparable. Fresh baseline required.

## Backtest Analysis (old regime, 222 trades over 198 days)
- Win rate: 46%, Cumulative P&L: -650%
- Old R:R was inverted: 30% stop / 20% TP (needed 60% WR to break even)
- ATM calls: only profitable bucket
- OTM trades: -601% cumulative
- Direction accuracy: 51% (near random)
- Model exits were rarely used — hardcoded TP fired first

## Best Runs (old regime — not comparable)
| Run | Score | PF | TPD | Key Change |
|-----|-------|----|-----|------------|
| run-2026-03-18-012841#1 | 3.48 | 1.85 | 4.32 | Best under old contract |

## Dead Ends
| Change | Result | Why |
|--------|--------|-----|
| GREEKS_ADAPTATION_STRENGTH outside 0.5 | PF < 1.0 | 0.5 is the sweet spot |
| QUALITY_GATE_STRENGTH > 0.8 | PF 0.93 | Disrupts learned patterns |
| NO_TRADE bias beyond -0.2 | Timeouts | Gate head very sensitive |
| Aggressive tanh quality gate | 92% no-trade | Model became too cautious |
