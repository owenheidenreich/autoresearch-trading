# Lab Notebook

## System
SPX 0DTE | 2-head gate+dir (v4) | 32 features (v2) | 5-dim position state | 8 actions | 1-min bars | learned exits (no hardcoded TP)
- **Score config LOCKED**: _score_config is hardcoded, mutation-guarded in run_loop.py.

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
| STOPPED_PNL_WEIGHT as additional loss term | 0/9 | Adding stopped P&L as extra loss creates conflicting gradients with raw P&L. All 9 attempts degraded PF from 3.42 to 1.2-1.8. The correct fix (now applied) is to REPLACE raw P&L with stopped P&L, not add it alongside. |

## Best Runs
| Run | Score | PF | TPD | Key Change |
|-----|-------|----|-----|------------|
| (fresh start — architecture cleaned: removed BalancedStrikeGate, ETV head, OTM bias. ATM-favoring init.) | — | — | — | — |

## Dead Ends
| Change | Result | Why |
|--------|--------|-----|
| WEIGHT_DECAY = _env_float("TRAIN_WEIGHT_DECAY", 0.08, lo=0.0 | score=6.09 | score_not_improved |
| print(f"Architecture: v4 simplified two-head (gate+dir) + ba | score=5.96 | score_not_improved |
| minor change | score=-1.56 | score_not_improved |
| PNL_ALIGNMENT_WEIGHT = _env_float("TRAIN_PNL_W", 0.4, lo=0.0 | score=-10.00 | score_not_improved;critical_anomalies:cost_realism_low_coverage,entry_quality_to |
| actionable_mask=y_dict['am'],; otm10_call_stopped_pnl=y_dict | score=-10.00 | score_not_improved;critical_anomalies:cost_realism_low_coverage,entry_quality_to |
| PNL_ALIGNMENT_WEIGHT = _env_float("TRAIN_PNL_W", 0.1, lo=0.0 | score=-10.00 | score_not_improved;critical_anomalies:cost_realism_low_coverage,entry_quality_to |
| PNL_ALIGNMENT_WEIGHT = _env_float("TRAIN_PNL_W", 0.3, lo=0.0 | score=-10.00 | score_not_improved;critical_anomalies:cost_realism_low_coverage,entry_quality_to |
| PNL_ALIGNMENT_WEIGHT = _env_float("TRAIN_PNL_W", 0.2, lo=0.0 | score=-2.32 | score_not_improved |
| self.gate_head[-1].bias[0] -= 0.2   # NO_TRADE (reduced from | score=-0.33 | score_not_improved |
| COOLDOWN_RATIO = _env_float("TRAIN_COOLDOWN_RATIO", 0.4, lo= | score=-10.00 | score_not_improved;critical_anomalies:cost_realism_low_coverage,entry_quality_to |

## Next Priorities
Beat the current best score by improving TRADING BEHAVIOR (PF, win rate, drawdown).
Explore these directions (pick ONE coherent hypothesis per experiment):
- Loss weight scheduling (ramp gate_w, dir_w, pnl_w across training epochs)
- Sample weighting (weight morning trades higher, or weight by VIX regime)
- Learning rate schedule (cosine decay, OneCycle, warmup/cooldown ratio changes)
- Gate head bias initialization (tune the NO_TRADE vs TRADE prior)
- Feature noise strategies (targeted noise on volatile features vs stable ones)
- Batch construction (change DAY_SEQ_RATIO, mine hard examples)
- Creative _env_float combinations not yet tried (check Dead Ends first)
Use the trade diagnostics printed at the end of training to identify the
WEAKEST area, then target that specific weakness.
Do NOT repeat approaches from Dead Ends or What Fails.

