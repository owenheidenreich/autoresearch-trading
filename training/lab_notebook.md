# Lab Notebook

## System
SPX 0DTE | 2-head gate+dir | 32 features (v2) | 5-dim position state | 8 actions | 1-min bars | learned exits (no hardcoded TP)

## Causal Exit Labels
Exit labels use backward-only signals only (no future peek):
- **Trailing stop**: exit when P&L drops >50% from high-water mark (HWM must be >5%)
- **Momentum stall**: exit when P&L is positive but hasn't improved in 10 bars

## Account-Aware Scoring
$10,000 starting account with affordability checks and ruin detection.
- Ruin = equity below 25% of starting capital → score floor -5.0
- Risk fraction penalty: avg trade cost > 30% of account → multiplicative penalty
- Position state dims 3-4: account_health, loss_streak_frac

## What Works (proven across 95 experiments)
1. **BalancedStrikeGate module** — best architecture. Modulates dir_logits based on account_health, biasing toward cheaper OTM when stressed. Score 1.769, PF 3.42.
2. **Symmetric ATM penalty** — penalize BOTH CALL_ATM and PUT_ATM equally (-0.25 each). Foundation since exp-3.
3. **OTM5 bias > OTM10** — OTM5 hits sweet spot of affordability and payoff. OTM10 too cheap (low delta).
4. **NO_TRADE gate bias +0.5** — selectivity is the foundation. Every kept model uses this.
5. **Lower learning rate (1.5e-4)** — halved from 3e-4, improved convergence. Score jumped 21%.
6. **CapitalPreservationGate** — account-health-aware gating. Score 0.802, breakthrough to capital survival.
7. **Bias tuning on existing heads** — consistently outperforms adding new modules.

## What Fails (do NOT retry)
| Pattern | Attempts | Result |
|---------|----------|--------|
| New nn.Module subclasses (VolatilityRegimeAdapter, FeatureGroupProcessor, etc.) | 0/8 | Not enough training time for new params |
| Dynamic gate scaling from position_state | 0/6 | Disrupts learned gate behavior |
| Curriculum learning / staged training | 0/4 | Training instability, always scores <0.25 |
| Hand-coded stress scaling formulas | 0/6 | Inverted logic or too aggressive |
| Third output head (size_logits) | 0/1 | Incompatible with eval contract (expects 2-tuple) |
| OTM10 over-emphasis (bias >0.4) | 0/2 | Encourages overtrading on cheap contracts |
| Asymmetric direction bias (puts only) | 0/1 | Loses directional flexibility |
| Attention-based feature reweighting | 0/2 | Too many new params for 4-min training budget |
| Contrastive / momentum loss terms | 0/3 | Added loss complexity without improving score |
| LR < 1e-4 or > 3e-4 | 0/2 | Too slow or too unstable |

## Current Best Model (exp-52)
- **Score**: 1.769 | **PF**: 3.42 | **TPD**: 1.3
- Architecture: BalancedStrikeGate + CapitalPreservationGate
- LR: 1.5e-4 | Dropout: 0.15 | d_model: 64 | depth: 3
- Stop-loss rate: ~20% | Win rate: ~56% (on full backtest)
- Full backtest: $10k → $37.6k (+276%) over 986 days, max DD -10.2%

## Best Runs
| Run | Score | PF | TPD | Key Change |
|-----|-------|----|-----|------------|
| #1 | 0.36 | 1.86 | 1.9 | NO_TRADE gate bias +0.5 |
| #3 | 0.42 | 2.40 | 1.9 | Symmetric ATM penalty -0.25 + OTM5 bias |
| #12 | 0.51 | 1.96 | 1.7 | Halved LR to 1.5e-4 |
| #16 | 0.80 | 2.28 | 1.4 | CapitalPreservationGate module |
| #27 | 1.47 | 2.97 | 1.4 | SelectiveStrikeGate module |
| #52 | 1.77 | 3.42 | 1.3 | BalancedStrikeGate module |

## Dead Ends
| Change | Result | Why |
|--------|--------|-----|

## Next Priorities
- Reduce stop-loss rate (currently ~20% of trades) — biggest drag on PF
- Improve worst_chunk_pf consistency (tail risk across market regimes)
- Explore warm-starting from best_model.pt weights instead of training from scratch
