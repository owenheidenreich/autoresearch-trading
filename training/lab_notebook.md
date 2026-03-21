# Lab Notebook

## System
SPX 0DTE | 3-output gate+dir+etv (v5) | 32 features (v2) | 5-dim position state | 8 actions | 1-min bars | learned exits (no hardcoded TP)

### v5 Architecture Changes (from v4)
- **EV-weighted loss**: Gate uses sigmoid(best_stopped_pnl * scale) instead of binary classification. Direction uses return-weighted soft targets instead of argmax.
- **ETV head**: Expected Trade Value regression head (1 linear layer, 65 params). Predicts P&L magnitude.
- **Stopped P&L labels**: Training labels now include dynamic stop (DYNAMIC_STOP_BASE=0.35) matching evaluation reality.
- **Day-sequential batching**: 70% sequential bars within days (carrying real position state), 30% random.
- **Score config LOCKED**: _score_config is hardcoded, mutation-guarded in run_loop.py.

## Causal Exit Labels
Exit labels use backward-only signals only (no future peek):
- **Trailing stop**: exit when P&L drops >50% from high-water mark (HWM must be >5%)
- **Momentum stall**: exit when P&L is positive but hasn't improved in 10 bars

## Account-Aware Scoring
$10,000 starting account with affordability checks and ruin detection.
- **Scaled position sizing**: n_contracts = max(1, floor(balance * 0.05 / contract_cost)). Always whole contracts, scales with account growth.
- Ruin = equity below 25% of starting capital → score floor -5.0
- Risk fraction penalty: avg trade cost > 30% of account → multiplicative penalty
- Position state dims 3-4: account_health, loss_streak_frac

## What Fails (do NOT retry)
| Pattern | Attempts | Result |
|---------|----------|--------|
| New nn.Module subclasses (adding brand new modules with fresh parameters) | 0/16+ | Not enough training time for new params to converge in 4-min budget |
| Dynamic gate scaling from position_state | 0/6 | Disrupts learned gate behavior |
| Curriculum learning / staged training | 0/4 | Training instability, always scores <0.25 |
| Hand-coded stress scaling formulas | 0/6 | Inverted logic or too aggressive |
| Attention-based feature reweighting | 0/2 | Too many new params for 4-min training budget |
| Contrastive / momentum loss terms | 0/3 | Added loss complexity without improving score |
| LR < 1e-4 or > 3e-4 | 0/2 | Too slow or too unstable |
| Fixating on a single metric (stop-loss rate, temporal consistency, etc.) | 0/25+ | Leads to tunnel vision; all experiments target same thing and none improve |
| Explicit temporal/recency weighting in loss | 0/10 | TEMPORAL_CONSISTENCY_WEIGHT, TEMPORAL_ROBUSTNESS_WEIGHT, TEMPORAL_RECENCY_WEIGHT all failed to beat baseline |
| Rewriting exit label computation | 0/3 | Changing trailing/stall thresholds destabilized training; scores went negative |
| Tuning _score_config values (drawdown_penalty, hold_bonus, rr_bonus, etc.) | 0/7 | Games the evaluation metric without improving model — score inflates while PF/TPD stay flat. Score config is now LOCKED. |

## Best Runs
| Run | Score | PF | TPD | Key Change |
|-----|-------|----|-----|------------|
| v4 #52 | 1.77 | 3.42 | 1.3 | BalancedStrikeGate module (v4 baseline) |

## Dead Ends
| Change | Result | Why |
|--------|--------|-----|

## Next Priorities
- Beat the current best score (1.77). The agent is free to explore any approach.
- v5 has EV-weighted loss, stopped P&L labels, ETV head, and day-sequential batching. Focus on tuning loss weights and training dynamics.
- Per-chunk PF and win rate are now visible in training output — use them to evaluate temporal consistency.
- The score formula is locked — improve TRADING BEHAVIOR (PF, win rate, drawdown), not the scorer.

## What Works (proven across 95+ experiments)
1. **BalancedStrikeGate module** — best architecture so far. Modulates dir_logits based on account_health, biasing toward cheaper OTM when stressed. Score 1.77, PF 3.42.
2. **Symmetric ATM penalty** — penalize BOTH CALL_ATM and PUT_ATM equally (-0.25 each). Foundation since exp-3.
3. **OTM5 bias > OTM10** — OTM5 hits sweet spot of affordability and payoff. OTM10 too cheap (low delta).
4. **NO_TRADE gate bias +0.5** — selectivity is the foundation. Every kept model uses this.
5. **Lower learning rate (1.5e-4)** — halved from 3e-4, improved convergence. Score jumped 21%.
6. **Bias tuning on existing heads** — consistently outperforms adding new modules.
