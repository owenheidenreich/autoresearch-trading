# ART² Lab Notebook — Outer Loop Memory

## System
ART² meta-loop wrapping autoresearch inner loop.
Outer loop (Opus) makes strategic decisions; inner loop (Sonnet agents via inner_loop.py) optimizes train.py.
**v14 (2026-03-28, CURRENT):** Exact v10 restoration. 38 features, 4-head (gate+dir+value+risk), sniper_loss. Replay PF 2.77 (+1,191% return, $10K→$129K). 537 trades, 30% WR, 91% model exits. Score 0.138 on 298-day val set. **VIABLE** (p=0.0001). Pipeline integrity overhaul: data provenance, SHA256 sidecar, fatal feature mismatch, enriched experiment logging.
**Pre-v14 (archived):** v10-v13 failed due to silent bugs (data.pt gate labels, feature truncation, unified action head). v8 had replay PF 1.43. v6 was first viable model. 230 pre-v3 cycles archived. See `archive/` for details.

## Strategic Changes Tried
| Cycle | Change | Replay PF Before | Replay PF After | Verdict |
|-------|--------|------------------|-----------------|---------|
| 009 | Win-rate-first restructure: gate margin, PnL clip, WR reg, score config | 2.77 | (pending) | Addresses HINDSIGHT_DEPENDENT verdict |
| 008 | v14 pipeline integrity + doc cleanup + Monte Carlo | 2.77 | (current) | Infrastructure — no model changes |
| 007 | v14 confirmation: re-eval v10 on 298-day set, proved v14 > v10 | N/A | 2.77 | v14 IS the best model. v10 scores 0.064 on same set. |
| 006 | v8: revert v7, VALUE_W=0, EXIT_W=0.15, fresh start | 0.99 (v7) | 1.43 | Profitable but on 70-day val set (inflated). |
| 005 | v7.1 PBT sweep (8 gen × 6 pop) | N/A | 0.99 | PARTIAL — score improved but replay NOT VIABLE |
| 001-004 | v6 breakthrough → v7 experiments → PBT sweeps | N/A | various | See archive for details |

## Current Hypothesis
**Win-rate-first restructure.** v14 is VIABLE (PF 2.77) but Monte Carlo verdict is HINDSIGHT_DEPENDENT — removing top 5% of winners kills profitability. The model chases lottery tickets (30% WR, +88.8% avg winner vs -13.5% avg loser).

Pickles (17yr trader, $100M+): "Stats are great but market don't give a hoot about stats. The stat is correct but the market goes into full fuckery mode to prove the stat right in the worst kind of way." Even statistically sound strategies fail because the PATH to profitability is psychologically unsurvivable.

**Three root causes addressed:**
1. Gate labels are pure hindsight (`best_pnl > 0.0`) → added `REG_GATE_MARGIN` (require meaningful profit)
2. PnL alignment rewards fat tails (uncapped) → added `REG_PNL_CLIP` (cap at ±50%)
3. Score blind to win rate (`win_rate_bonus: 0.0`) → enabled at 0.5, `rr_bonus` 0.3→0.1

**Target:** WR >= 40%, PF >= 1.3, Monte Carlo ROBUST. A 45% WR / PF 1.5 model is more robust and tradeable than 30% WR / PF 2.77.

## Dead Ends (Strategic Level)
| Change | Cycles | Result |
|--------|--------|--------|
| Manual EXIT_W tuning (0.20-0.35) | 1 | 10/10 reverted. Penalty cascade. |
| PBT sweep on loss weights | 1 | 30/30 reverted. Model at sharp optimum. |
| VALUE_W > 0 (train value head) | 2 | Destructive — 42 PBT experiments failed. |
| Unified action head (v12) | 1 | 48 experiments, collapsed to 1 strike. |
| Regime/setup gate labels (v11) | 1 | Circular — lagging features don't add info. |
| Comparing scores across different val sets | N/A | 3 days + $50 wasted on phantom problem. |

## Lessons
- **Score gaming:** Agent tuned SCORE_DRAWDOWN_PENALTY to inflate scores 6x. Score config now LOCKED.
- **Rigid lab notebook = tunnel vision.** Keep priorities open-ended.
- **Fix structural bugs first, then tune.** Regularization without architecture fixes = waste.
- **Warm start compounding requires warm start to actually work.** 230 cycles had broken warm start.
- **Backtest P&L ≠ real P&L.** Hindsight bias inflates returns. Win rate with margin of error is the true signal.

## What Works
- ART² correctly identifies gamed baselines, structural bugs, architecture dead weight.
- Pipeline integrity (v14): data provenance, SHA256 sidecar, fatal feature mismatch.
- REVIEW gate ensures human approval before GPU spend.
- Replay backtest as ground truth (not training score).
- Monte Carlo stress-testing for robustness beyond single-path backtesting.
