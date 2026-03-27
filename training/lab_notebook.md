# Lab Notebook

## System
SPX 0DTE | 4-head gate+dir+value+risk (v8) | 37 features (v3) | 7-dim position state | 4-dim account state | 8 actions | 1-min bars | learned exits (no hardcoded TP)
- **v8 (2026-03-26):** Reverted v7's counterproductive changes. VALUE_W=0.0 (disabled), EXIT_W=0.15, VALUE_LOSS_TYPE=mse (reverted from BCE), COOLDOWN_RATIO=0.4, BATCH_SIZE=1024. P2 fp32 precision casts in loss. Checkpoint saves full training_config with warm-start validation. Outer loop keep/revert with pipeline snapshots.
- **v7 (2026-03-26):** Value head changed from MSE regression → BCE binary exit classifier. Exit labels sparse (66.5% vs 98%). Take-profit signal at 20%. EXIT_W=0.40, VALUE_W=0.5, DAY_SEQ_RATIO=0.92. Sigmoid value exit with conviction-adjusted threshold. **Postmortem:** All v7 changes made things worse. PBT drove EXIT_W→0 and VALUE_W→0. Score dropped from 41.92 to 8.09. BCE value head and sparse exit labels proven counterproductive.
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
| EXIT_W ≥ 0.5 (exit override too aggressive) | 0/5 | With old 98% exit labels, EXIT_W=0.5 killed all TRADE targets. v7 sparse labels (66.5%) allow higher EXIT_W. Current default: 0.40. |
| v7.1 PBT: EXIT_W and VALUE_W near zero wins | 1/48 | PBT best (16.34) had EXIT_W=0.019, VALUE_W≈0. Exit labels and value head may be counterproductive — gate+PNL handles exits. BUT replay PF=0.99, train/replay divergence is severe. |
| PBT promotion re-trains from random init (BUG) | ALL PBT runs affected | `upload_model=False` on promotion re-training meant winner was re-trained from scratch, not warm-started. Saved model had score 0.90, but .best_score said 16.34. **Fixed:** promotion now uses `upload_model=True` + post-download validation. |
| GATE_W < 0.5 | 0/1 | Weakens gate learning signal, causes zero-trade collapse |
| Changing position state tanh scaling in only ONE file | 0/1 | Was hardcoded independently in 3 files → mismatch. Now shared via PNL_TANH_SCALE / BEST_PNL_TANH_SCALE from prepare.py. Agents can tune via env var. |
| Manual single-param EXIT_W tuning (0.20-0.35) | 0/10 | Score 41.92 depends on fragile penalty balance (consec loss, drawdown). Changing exit timing cascades through penalties. Best attempt: EXIT_W=0.25 DAY_SEQ=0.90 scored 26.15 (38% regression). Use PBT for multi-param exploration instead. |
| Hardcoding defaults in two places (train.py + inner_loop.py) | 5 params drifted | `_PARAM_SPACE` had stale defaults (EXIT_W=0.15 vs actual 0.40, VALUE_W=0.3 vs 0.5, etc). PBT baseline explored wrong center. **Fixed:** `_PARAM_SPACE` no longer stores defaults — `_parse_train_defaults()` reads them from train.py at PBT init time. Single source of truth. |

## Best Runs
| Run | Score | PF (train) | PF (val) | Trades | Win% | Notes |
|-----|-------|------------|----------|--------|------|-------|
| v8 cycle-006 exp5 | 13.78 | 5.45 | 1.43 (replay) | 133/334 | 44%/40% | v8 fresh start, VALUE_W=0 EXIT_W=0.15 COOLDOWN=0.4 BATCH=1024. Replay profitable (+120%). |
| v7.1 PBT gen3 | 16.34 | 3.62 | 0.99 (replay) | 100/227 | 46%/44% | v7, EXIT_W≈0 VALUE_W≈0 PNL_W=1.7 GATE_W=0.97 LR=0.0037. Replay NOT VIABLE. |
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
**WARM START from current best_model.pt (score 13.78).** v8 architecture, 37 features.

**v8 DEFAULTS (already applied):** BATCH_SIZE=1024, VALUE_W=0.0, EXIT_W=0.15, COOLDOWN_RATIO=0.4, VALUE_LOSS_TYPE=mse.

**CURRENT STATE:** Replay PF=1.43 (profitable!), training PF=5.45. The 3.8x gap = overfitting. 3 consecutive reverts at score 13.78. Model trades only 17% of days — very selective.

**PRIORITY 1: Reduce overfitting (training/replay PF gap)**
- Training PF 5.45 but replay PF only 1.43. Model memorizes training patterns.
- Try DROPOUT=0.25 (from 0.20). Higher dropout reduces overfitting.
- Try WEIGHT_DECAY=5e-4 (from 1e-4). Stronger L2 regularization.
- Try LABEL_SMOOTH=0.15 (from 0.10). Softer targets reduce overconfidence.
- Run one at a time. KEEP if score holds AND replay PF improves.

**PRIORITY 2: Break the plateau (score stuck at 13.78)**
- 4 experiments failed since best. Random re-init may be finding better local optima.
- Try LR=2.5e-4 (slight increase from 2e-4) to escape flat loss landscape.
- Try PNL_W=2.0 (from 1.5) to strengthen profit-factor signal.
- Try GATE_W=0.80 (from 0.95) to give other heads more gradient.

**PRIORITY 3: PBT sweep if sequential stalls**
- If 5 more sequential experiments all revert, switch to PBT.
- PBT population=6, generations=3, sweep: DROPOUT (0.15-0.30), WEIGHT_DECAY (1e-4 to 1e-3), LR (1.5e-4 to 3e-4), PNL_W (1.0-2.5).
- Center PBT around current v8 defaults.

**MONITORING: PUT vs CALL in replay**
- PUT trades: $9,647 total P&L (dominant). CALL trades: $2,870 total P&L.
- Track direction distribution. If call PF < 1.0, consider DIR_W increase.

**Available levers:** DROPOUT (0.15-0.30), WEIGHT_DECAY (1e-4 to 1e-3), LR (1.5e-4 to 3e-4), PNL_W (1.0-2.5), GATE_W (0.7-1.0), DIR_W (1.0-3.0), LABEL_SMOOTH (0.05-0.20), COOLDOWN (0.3-0.6).

**AVOID:** EXIT_W ≥ 0.5, GATE_W < 0.5, new nn.Module subclasses, score config changes, DAY_SEQ_RATIO < 0.7, LR > 3e-4, VALUE_W > 0 (proven counterproductive), feature noise on warm start.
Do NOT repeat approaches from What Fails.
