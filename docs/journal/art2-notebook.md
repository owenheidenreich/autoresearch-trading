# ART² Lab Notebook — Outer Loop Memory

## System
ART² meta-loop wrapping autoresearch inner loop.
Outer loop (Opus) makes strategic decisions; inner loop (Sonnet agents via inner_loop.py) optimizes train.py.
**v3 era (2026-03-24):** 37 features, fresh start, all pre-v3 cycles archived.

## Strategic Changes Tried
| Cycle | Change | OOS PF Before | OOS PF After | Verdict |
|-------|--------|---------------|--------------|---------|
| 003 | PBT sweep: Tier 1 loss weights, pop=6, gen=5 | 3.28 | 3.28 (unchanged) | FAILED — 30/30 reverted, best=25.99, stagnation=5 |
| 002 | EXIT_W tuning (0.20-0.35), VALUE_W boost (0.4-0.5), DAY_SEQ=0.90, LR/WD | 3.28 | 3.28 (unchanged) | FAILED — 10/10 reverted, tunnel vision on EXIT_W |
| 001 | v6 four-head + BATCH_SIZE=1024 breakthrough | N/A (fresh) | 3.28 | VIABLE (p=0.0020, 208 trades, 63 val days) |

## Paper Trading P&L Tracking
| Date | Trades | P&L % | Backtest Expected | Divergence |
|------|--------|-------|-------------------|------------|
*requires update with v6 paper trading sessions.*

## Dead Ends (Strategic Level)
| Change | Cycles | Result |
|--------|--------|--------|
| Manual EXIT_W tuning (0.20-0.35) | 1 (cycle 002) | 10/10 reverted. Score penalty terms (consec loss, drawdown) are fragile — changing exit timing cascades through penalties. |
| PBT sweep on Tier 1 loss weights | 1 (cycle 003) | 30/30 reverted (5 gens × 6 members). Best=25.99 vs 41.92 baseline. Stagnation=5. Multi-param exploration didn't help — the 41.92 is a stochastic outlier, not an achievable optimum. |

## Lessons From Pre-v3 (230 cycles archived)
- **Score gaming:** Agent tuned SCORE_DRAWDOWN_PENALTY to inflate scores 6x without PF improvement. Score config now LOCKED.
- **Rigid lab notebook = tunnel vision:** "EXACTLY 2 _env_float values per experiment" caused 100% tunnel vision, 0% accept rate. Keep priorities open-ended.
- **Regularization without fixing architecture = wasted cycles.** Fix structural bugs first, then tune.
- **More GPU steps ≠ better model.** 6.6x more steps (bf16 + batch=512) didn't help. Problem was training signal, not capacity.
- **Inner loop agent mode saves API costs.** Opus IS the loop, Sonnet agents write code. No Anthropic API calls.
- **Warm start compounding requires warm start to actually work.** Previous 230 cycles had silently broken warm start (always random init).
- **EXIT_W trap:** 97.7% of profitable bars have exit labels. EXIT_W=0.5 leaves only 12.3% TRADE targets. Keep EXIT_W ≤ 0.35.
- **Value exit is value-destructive at VALUE_W=0.3:** 35 value exits at avg -2.82% P&L. Value head needs stronger training signal (VALUE_W ≥ 0.4).
- **IBKR live pipeline worked but model didn't exit:** Position state was all zeros (unrealized P&L never fed). Fixed. Gate stays TRADE 90%+ even with fix → training issue (EXIT_W too low).
- **STOP_COOLDOWN_BARS not enforced live:** Training eval blocks entries for 5 bars after stop. Live had zero cooldown → 2.7s reentry. Fixed.
- **Spread proxy was miscalibrated:** Bar range ≠ bid-ask spread. Killed 85-98% of bars. Fixed with premium-tier lookup + hard-fail guardrails.

## Current Hypothesis
**Paper trading validation.** 40 experiments (10 manual + 30 PBT) couldn't improve the 41.92 score — it's a stochastic outlier, not an optimizable target. Model is VIABLE (val PF=3.28, p=0.0020). Per ground truth hierarchy (paper P&L > backtest > score), next step is IBKR paper trading validation: 3-5 sessions, measure live P&L vs backtest expectation. If they match, model is validated. If not, divergence reveals what to fix. Feature additions (GEX, market internals, walk-forward) are higher leverage than more hyperparameter tuning.

## What Works (Outer Loop)
- ART² correctly identified gamed baselines, structural bugs, and architecture dead weight across 230 cycles.
- Promoted history audit trail makes gaming visible and reversible.
- Research phase identifies model bias patterns (call-only, morning-only, ATM-only).
- Deploy.sh no longer overwrites train.py when local changes exist.
- Agent mode (Opus + Sonnet agents) eliminates API costs.
- REVIEW gate ensures human approval before every training run.

## Roadmap

### High Priority
- **GEX (Gamma Exposure):** Dealer positioning — positive GEX = mean-reverting, negative = trending. Would condition direction bias. Requires options flow data (Squeezemetrics/SpotGamma).
- **Market Internals (TICK, $ADD, Breadth):** Pickles' primary confirmation signal. Divergences between price and internals. Available via IBKR/Polygon.
- **Walk-Forward Validation:** Rolling N-day train, M-day validate. Catches regime-specific overfitting that static 70/30 split misses.

### Medium Priority
- **VIX Regime Stratification:** Ensure train/val splits match VIX regime distributions. Report per-regime PF.
- **Charm Flow Rate:** Rate of delta decay for predictable PM dealer unwind flows (1:30-3:30 PM).
- **IV Skew (25-delta put vs call):** Downside fear premium, early warning for directional bias changes.
- **Sub-Minute Bars (5s/15s):** Finer resolution for gamma dynamics. Major data pipeline rework (12-60x data volume).

### Lower Priority
- Credit spread / premium-selling strategy (different paradigm entirely)
- Ensemble models (morning vs afternoon vs power hour specialists)
- Event calendar integration (CPI/FOMC IV crush prediction)
- Order flow indicators (L2 book imbalance, block trades)
- DIX (dark pool index, leads by 1-3 days but less useful for intraday 0DTE)
