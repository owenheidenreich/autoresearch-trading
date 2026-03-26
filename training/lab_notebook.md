# Lab Notebook

## System
SPX 0DTE | 4-head gate+dir+value+risk (v6) | 37 features (v3) | 7-dim position state | 4-dim account state | 8 actions | 1-min bars | learned exits (no hardcoded TP)
- **v3 features (2026-03-24):** 5 market structure features added: poc_dist, va_position, vwap_band_sigma, ib_break, theta_pressure
- **Fresh start:** All previous training sessions archived to `archive/pre-v3-2026-03-24/`. No best_model.pt. Clean slate.
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
| STOPPED_PNL_WEIGHT as additional loss term | 0/9 | Adding stopped P&L as extra loss creates conflicting gradients with raw P&L. The correct fix (now applied) is to REPLACE raw P&L with stopped P&L, not add it alongside. |
| EXIT_W ≥ 0.5 (exit override too aggressive) | 0/5 | 97.7% of profitable bars have exit labels. EXIT_W=0.5 flips half → only 12.3% TRADE targets remain. Model learns "never trade". Current default: 0.10. |
| GATE_W < 0.5 | 0/1 | Weakens gate learning signal, causes zero-trade collapse |
| Changing position state tanh scaling in only ONE file | 0/1 | Was hardcoded independently in 3 files → mismatch. Now shared via PNL_TANH_SCALE / BEST_PNL_TANH_SCALE from prepare.py. Agents can tune via env var. |
| Manual single-param EXIT_W tuning (0.20-0.35) | 0/10 | Score 41.92 depends on fragile penalty balance (consec loss, drawdown). Changing exit timing cascades through penalties. Best attempt: EXIT_W=0.25 DAY_SEQ=0.90 scored 26.15 (38% regression). Use PBT for multi-param exploration instead. |

## Best Runs
| Run | Score | PF (train) | PF (val) | Trades | Win% | Notes |
|-----|-------|------------|----------|--------|------|-------|
| cycle-002 exp2 | 41.92 | 6.78 | 3.28 | 128/208 | 47%/50% | v6, BATCH_SIZE=1024+DIR_W=3.0+DROPOUT=0.20, VIABLE (p=0.0020), max_consec=4, val 63 days |
| cycle-001 exp9 | 18.78 | 6.24 | 2.77 | 216 | 48.6% | v6 four-head, warm start from exp6, VIABLE (p=0.0035) |
| cycle-001 exp6 | 17.47 | 5.55 | — | — | — | v6 warm start from baseline |
| cycle-001 exp4 | 13.14 | 6.06 | — | — | — | v6 fresh start baseline |

## Dead Ends
| Pattern | Attempts | Result |
|---------|----------|--------|
| Position state tanh scaling *2.0 in training vs *5.0 in eval | 0/1 | Train/eval mismatch: model learned exit thresholds at wrong scale. Fixed: PNL_TANH_SCALE shared constant from prepare.py (default 5.0, tunable via env var). |

## Lessons From Pre-v3 (archived, for reference)
- **Spread proxy was broken:** `_spread_bps_proxy()` used bar range as bid-ask spread → killed 85-98% of bars on most dates. Fixed with premium-tier lookup.
- **Warm start was silently broken:** `model = TradingModel()` always random init, never loaded best_model.pt. Fixed.
- **Entry/exit gradient conflict:** Gate received contradictory signals (TRADE + NO_TRADE) on same bars. Fixed with position-conditional exit overrides.
- **Date specialization:** Model memorized specific dates (Sep 17, Mar 16) instead of generalizing. Fixed with per-day normalization + WEIGHT_DAY_DIVERSITY + single_date_specialist guardrail.
- **Pro-trade gate bias works:** [-0.3, +0.3] was a breakthrough (score -0.11→4.92).
- **Feature noise doesn't work on warm start:** Both σ=0.03 and σ=0.01 caused degradation.

## Next Priorities

**WARM START from v6 best_model.pt (score 41.92).** PNL_TANH_SCALE now shared constant from prepare.py (tunable via env var). IBKR live session confirmed pipeline works end-to-end. STOP_COOLDOWN_BARS now enforced in live (matching training). Model VIABLE (p=0.0020, val PF=3.28).

**BASELINE PARAMS (PROVEN):** BATCH_SIZE=1024, DIR_W=3.0, DROPOUT=0.20. Always use these unless explicitly testing alternatives. BATCH_SIZE=1024 gives 6.6x more gradient steps per 5-min experiment.

**Priority 1: IBKR paper trading validation.**
- 40 experiments (10 manual + 30 PBT) couldn't beat 41.92. Score optimization exhausted.
- Run 3-5 IBKR sessions. Compare live P&L to backtest (val PF=3.28, ~3.3 trades/day).
- Monitor: cooldown enforcement, model exits, value exits, gate behavior while holding.
- Divergence between live and backtest reveals what to fix next.

**Priority 2: Reduce EOD dependence.**
- 74% of val profit comes from EOD exits (49 trades held to close, avg +74.57%).
- Better model exits = less reliance on EOD. This is a natural consequence of PBT finding better loss weights.

**Priority 3: General warm-start compounding (after PBT).**
- If PBT finds better loss weights, use those as new defaults.
- Then explore: WEIGHT_DECAY, LR, DAY_SEQ_RATIO, label smoothing.
- Don't touch: gate bias, score config, EXIT_W ≥ 0.5, GATE_W < 0.5.

Current best defaults: BATCH_SIZE=1024, DROPOUT=0.20, DIR_W=3.0, WEIGHT_DECAY=0.08, DAY_SEQ_RATIO=0.85, GATE_W=0.5, EXIT_W=0.15, PNL_W=0.5, LR=2.5e-4, COOLDOWN=0.3, TRAIN_RISK_W=0.2.
**Normalization:** per-day mean + global std (anti-fingerprint fix in prepare.py).
Gate bias: pro-trade [-0.3, +0.3]. Position-conditional exit overrides.
Active anti-overfit: WEIGHT_RECENT_BOOST=0.3, WEIGHT_DAY_DIVERSITY=1.0, REG_GATE_ENTROPY=0.10.
FALSE_ENTRY_PENALTY=1.2.
**PNL_TANH_SCALE=5.0, BEST_PNL_TANH_SCALE=2.0** (shared constants from prepare.py, tunable via env var).

**Primary goal: GENERALIZATION.** Model must trade on 5+ distinct validation dates (`num_trade_dates ≥ 3` enforced via `single_date_specialist` anomaly flag). Train/val divergence must close below 50%.

**New features (v3):** poc_dist, va_position, vwap_band_sigma, ib_break, theta_pressure. These give the model volume profile context, VWAP band awareness, IB break detection, and explicit afternoon theta pressure signal.

**Available levers:**
- Existing _env_float values (DIR_W, GATE_W, PNL_W, EXIT_W, DROPOUT, WEIGHT_DECAY, LR, DAY_SEQ_RATIO, COOLDOWN)
- New SCHED_* _env_float: loss weight scheduling (ramp weights over training)
- New WEIGHT_* _env_float: sample weighting (hard examples, time-of-day, regime)
- New WARM_* _env_float: warm start controls (freeze layers, LR multiplier)
- New REG_* _env_float: regularization (L1 sparsity, gradient penalty)
- BATCH_SIZE (currently 512, up to 1024)
- ONE regularization function (must be REG_* controlled, default 0.0)
- LR scheduling, bias initialization, batch construction, gradient accumulation
- Layer freezing and differential LR for warm start
- Early stopping based on validation metrics

**AVOID:** EXIT_W ≥ 0.5, GATE_W < 0.5, new nn.Module subclasses, score config changes, DAY_SEQ_RATIO < 0.7, LR > 3e-4, feature noise on warm start.
Do NOT repeat approaches from What Fails.
