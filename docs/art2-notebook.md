# ART² Lab Notebook — Outer Loop Memory

## System
ART² meta-loop wrapping autoresearch inner loop.
Outer loop (Opus) makes strategic decisions; inner loop (Sonnet agents via inner_loop.py) optimizes train.py.
**v3 era (2026-03-24):** 37 features, fresh start, all pre-v3 cycles archived.

## Strategic Changes Tried
| Cycle | Change | OOS PF Before | OOS PF After | Verdict |
|-------|--------|---------------|--------------|---------|
*No v3 cycles yet — fresh start.*

## Paper Trading P&L Tracking
| Date | Trades | P&L % | Backtest Expected | Divergence |
|------|--------|-------|-------------------|------------|
*No v3 paper trading sessions yet.*

## Dead Ends (Strategic Level)
| Change | Cycles | Result |
|--------|--------|--------|
*No v3 dead ends yet — clean slate.*

## Lessons From Pre-v3 (230 cycles archived)
- **Score gaming:** Agent tuned SCORE_DRAWDOWN_PENALTY to inflate scores 6x without PF improvement. Score config now LOCKED.
- **Rigid lab notebook = tunnel vision:** "EXACTLY 2 _env_float values per experiment" caused 100% tunnel vision, 0% accept rate. Keep priorities open-ended.
- **Regularization without fixing architecture = wasted cycles.** Fix structural bugs first, then tune.
- **More GPU steps ≠ better model.** 6.6x more steps (bf16 + batch=512) didn't help. Problem was training signal, not capacity.
- **Inner loop agent mode saves API costs.** Opus IS the loop, Sonnet agents write code. No Anthropic API calls.
- **Warm start compounding requires warm start to actually work.** Previous 230 cycles had silently broken warm start (always random init).
- **EXIT_W trap:** 97.7% of profitable bars have exit labels. EXIT_W=0.5 leaves only 12.3% TRADE targets. Keep EXIT_W ≤ 0.15.
- **Spread proxy was miscalibrated:** Bar range ≠ bid-ask spread. Killed 85-98% of bars. Fixed with premium-tier lookup + hard-fail guardrails.

## Current Hypothesis
**Three-head architecture (v5) with value head for intelligent exits.** Phase D replaces the failed RL exit policy (REINFORCE collapsed to 100% EXIT) with a value head — a third output on the transformer that predicts remaining P&L via MSE regression. The value head shares the transformer backbone (rich 37-feature context), uses stable training (MSE, not policy gradient), and provides continuous exit signal. PBT evolves `TRAIN_VALUE_W` (loss weight) and `TRAIN_VALUE_EXIT_THRESH` (exit threshold). Position state expanded from 5→7 dims (`best_pnl_since_entry`, `bars_since_pnl_high`). Exit priority: stop_loss > model_exit > value_exit > max_hold > EOD — consistent across train/replay/IBKR.

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
