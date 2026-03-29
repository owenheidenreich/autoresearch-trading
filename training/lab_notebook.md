# Lab Notebook

## System
SPX 0DTE | 4-head gate+dir+value+risk (v14) | 38 features | 7-dim position state | 4-dim account state | 14 directions | 1-min bars | learned exits (no hardcoded TP)
- **v14 (2026-03-28):** Exact v10 restoration. Fixed hidden bug: v13 silently ran v11's failed gate labels (AND of regime+setup+pnl ≈ 1.6% TRADE) because data.pt had setup/regime masks. Reverted to pure `pnl_ok.long()` gate labels. Reverted to 38 features (removed 6 redundant raw candle features). Warm start from v10 weights.
- **v13 (2026-03-28, FAILED):** Score -0.27. Intended v10 restoration but ran v11 AND-gate labels (~1.6% TRADE). Direction collapsed to 100% PUT, 11.8% WR.
- **v12 (2026-03-27, FAILED):** Unified action head (15-class). Score 1.16 after 48 experiments. Gate-direction coupling collapsed to 1 strike. 10x weaker entropy than v10.
- **v11 (2026-03-27, FAILED):** Regime+setup gate labels. Score 0.48. Circular — relabeling with lagging features doesn't add information.
- **v10 (2026-03-27):** Replay PF 1.74, 38 features, 14-class direction head. Score 16.73 was on **70 val days** (old data.pt). On current 298-day val set, v10 model scores **0.064**. V14 (0.138) beats it. Archived at `archive/models/v10/`.
- **v8 (2026-03-26):** VALUE_W=0.0, EXIT_W=0.15, COOLDOWN_RATIO=0.4, BATCH_SIZE=1024. Replay PF 1.43.
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
| Regime+setup gate labels from lagging features (v11) | 0/1 | Score 0.48 (-97%). `detect_setups()` and `compute_regime_labels()` use same features already in input. Circular — doesn't add information, just removes positive labels. |
| Unified 15-class action head replacing gate+direction (v12) | 0/48 | Score 1.16. Gate-direction coupling collapses to 1 strike. ENTROPY_COEFF=0.02 is 10x weaker than DIRECTION_ENTROPY_BONUS=0.20. DO_NOTHING dominates. |
| Multiple simultaneous changes (v12: architecture + loss + features + lookback) | 0/1 | Can't attribute regression to any single change. Every version that changed 2+ things failed. |
| Silent data.pt gate label activation (v13 bug) | 0/4 | setup_mask/regime_mask in data.pt silently activates v11 AND-gate logic (~1.6% TRADE labels). Archive captured AFTER v11 code added. **Fixed:** removed setup_mask/regime_mask from sniper_loss. |
| Comparing scores across different data.pt / val sets | 3 days wasted | v10 scored 16.73 on 70 val days. v14 scored 0.14 on 298 val days. Spent 3 days + $50 GPU trying to "fix" v14 when it was actually the best model. **Root cause:** data.pt was rebuilt (expanding val set 4x) without version tracking. Scores are NOT comparable across different val sets. **Fixed:** data.pt now has `_provenance` metadata, SHA256 sidecar, fatal feature mismatch (no silent truncation). |

## Best Runs

**NOTE:** Scores before v14 were on a 70-day val set. Current 298-day val set produces lower but more honest scores. Do NOT compare across val sets.

| Run | Score | Val Days | PF | Trades | Win% | Notes |
|-----|-------|----------|-----|--------|------|-------|
| **v14 (current)** | **0.138** | **298** | **1.58** | **555** | **28.6%** | Best model on 298-day eval. All 5 chunks profitable (worst PF=1.34). +36.5% return. |
| v10 on 298 days | 0.064 | 298 | 2.02 | 180 | 33.3% | v10 model re-evaluated on current val set. WORSE than v14. Worst chunk PF=0.61. |
| v10 (original) | 16.73 | 70 | 5.81 | 115 | 43.5% | ⚠️ 70-day val set (no longer reproducible). Inflated by small sample. |
| cycle-002 exp2 | 41.92 | ~70 | 6.78 | 128 | 47% | ⚠️ 70-day val set. v6, BATCH_SIZE=1024+DIR_W=3.0+DROPOUT=0.20. |
| v8 cycle-006 | 13.78 | ~70 | 5.45 | 133 | 44% | ⚠️ 70-day val set. Replay PF 1.43 (+120% return). |

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
**v14 CONFIRMED AS BEST MODEL.** Replay PF 2.77 (+1,191% return, $10K→$129K). 537 trades, 30% WR, 91% model exits, 1% stop-loss exits. Score 0.138 on 298-day val set (score 0.064 for v10 on same set — v14 wins).

**CURRENT STATE:** Pipeline integrity overhaul complete. data.pt has provenance metadata + SHA256 sidecar. Feature mismatch is now FATAL. All experiments logged with data_fingerprint and num_val_days.

**Incremental improvements (one at a time, warm start):**
1. VWAP bands (price vs ±1σ/±2σ) — Pickles' #1 signal, highest-leverage feature addition
2. Lunch penalty in loss — time-of-day weighting to suppress low-quality lunch entries
3. Gate confidence threshold — minimum gate probability before entering (execution-layer filter)
4. EXIT_W tuning — model exits well (91% model_exit), but EXIT_W=0.15 is low vs proven range
5. Regularization — DROPOUT (0.30), WEIGHT_DECAY (0.08) may have room for adjustment

**Available levers:** GATE_W (0.95), DIR_W (1.5), PNL_W (1.5), CONF_W (0.05), EXIT_W (0.15), VALUE_W (0.0), RISK_W (0.2), DROPOUT (0.30), WEIGHT_DECAY (0.08), LR (2.5e-4), BATCH_SIZE (1024).

**AVOID:** Regime/setup gate labels, unified action head, multiple simultaneous changes, new nn.Module subclasses, score config changes, VALUE_W>0 (proven destructive).
Do NOT repeat approaches from What Fails.
