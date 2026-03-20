# Lab Notebook

## System
SPX 0DTE | 2-head gate+dir | 32 features (v2 reduced from 70) | 5-dim position state | 8 actions | 1-min bars | learned exits (no hardcoded TP)

## CRITICAL: Causal Exit Labels (2026-03-20)
Exit labels no longer peek into future bars. The model must learn exits from backward-only signals:
- **Trailing stop**: exit when P&L drops >50% from high-water mark (HWM must be >5%)
- **Momentum stall**: exit when P&L is positive but hasn't improved in 10 bars
These are the ONLY exit label signals. The model cannot see the future to find optimal exits.
**Prior models overfit because exit labels used future price peaks — that's fixed now.**

## CRITICAL: Account-Aware Scoring Regime (2026-03-20)
This is a NEW scoring regime. Prior promoted scores are invalid — the baseline is -5.0.
The simulation now tracks a **real $10,000 account** with affordability checks and ruin detection.

### What Changed
- Inline equity tracking: account starts at $10,000, contract cost = premium * 100
- **Affordability check**: if contract_cost > account_balance, trade is BLOCKED
- **Ruin penalty**: equity below 25% of starting capital → severe score penalty → -5.0 floor
- **Risk fraction penalty**: avg trade cost > 30% of account → multiplicative penalty
- **Position state** expanded to 5 dims: model sees account_health and loss_streak_frac

### Key Output Metrics to Watch
- `hit_ruin`: True/False — did equity drop below 25% of starting capital?
- `min_equity_frac`: lowest equity as fraction of starting capital (1.0 = healthy, 0.05 = near-wipeout)
- `avg_risk_fraction`: average trade cost as fraction of account balance (lower is safer)
- `trades_blocked_by_balance`: how many trades were blocked due to insufficient funds
- `max_equity_dd`: maximum drawdown on the equity curve

### PROBLEM: All Prior Experiments Hit Ruin
Every experiment so far has hit ruin (min_equity_frac 3-6%). The model trades aggressively, blows through the account, then gets blocked by affordability checks (500-1800 trades blocked).

### PRIORITY: Capital Preservation
The #1 goal is to **avoid ruin**. A model that survives (hit_ruin=False) with even modest profit will score dramatically higher than an aggressive model that blows up.

### Strategies to Try
1. **Target cheaper contracts**: OTM5 and OTM10 options cost less per contract ($200-$800 vs $1000-$3000 for ATM). Lower cost = lower risk fraction = more trades before ruin.
2. **Tighter stop losses**: STOP_LOSS_PCT default is 0.30 (30%). Tighter stops (0.15-0.20) limit per-trade damage.
3. **Reduce trade frequency**: Fewer, higher-conviction trades preserve capital. The model sees account_health in position_state[3] — it can learn to skip marginal setups when account is low.
4. **Use position state dims 3-4**: account_health (dim 3) and loss_streak_frac (dim 4) are fed to the model. Wire these into gating logic so the model becomes selective when account is stressed.
5. **Dynamic stop tightening**: When account_health < 0.5, the model should be more conservative with exits.

### What NOT to Do
- Do NOT add complex charm/greeks modules — 5 prior experiments tried this and all hit ruin
- Do NOT ignore the account metrics — they are the primary signal now
- Do NOT increase trade frequency — more trades with bad sizing = faster blowup

## Score Formula Knobs (active by default)
- SCORE_DRAWDOWN_PENALTY = 0.5 (penalizes deep drawdowns)
- SCORE_RUIN_PENALTY = 1.0 (severe penalty for account blowup)
- SCORE_RUIN_THRESHOLD = 0.25 (triggers at 75% loss from peak)
- SCORE_RISK_FRACTION_PENALTY = 0.5 (penalizes oversized positions)
- SCORE_CONSEC_LOSS_THRESHOLD = 3 (15% penalty per consecutive loss beyond 3)

## Prior Knowledge
- ATM calls were the only profitable bucket in old regime
- OTM trades: -601% cumulative historically — but OTM is CHEAPER (lower risk fraction)
- Direction accuracy was ~51% (near random) — improving this is key
- GREEKS_ADAPTATION_STRENGTH sweet spot is 0.5
- NO_TRADE bias beyond -0.2 causes timeouts (gate head is sensitive)
- Aggressive quality gates (>0.8) disrupt learned patterns

## Best Runs
| Run | Score | PF | TPD | Key Change |
|-----|-------|----|-----|------------|
| (fresh start — no baseline yet) | — | — | — | Account-aware 5-dim position state |

## Dead Ends
| Change | Result | Why |
|--------|--------|-----|
| Charm in dynamic stops (5 variants) | All hit ruin, scores 0.007-0.029 | Added complexity without addressing capital preservation |
| Curriculum learning scheduler | Score 0.007 | Training instability, worst performer |
| GREEKS_ADAPTATION_STRENGTH outside 0.5 | PF < 1.0 | 0.5 is the sweet spot |
| QUALITY_GATE_STRENGTH > 0.8 | PF 0.93 | Disrupts learned patterns |
| NO_TRADE bias beyond -0.2 | Timeouts | Gate head very sensitive |
