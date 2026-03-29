# ART² Lab Notebook — Outer Loop Memory

## System
ART² meta-loop wrapping autoresearch inner loop.
Outer loop (Opus) makes strategic decisions; inner loop (Sonnet agents via inner_loop.py) optimizes train.py.
**v14 (2026-03-28, CURRENT):** Exact v10 restoration. 38 features, 4-head (gate+dir+value+risk), sniper_loss. Replay PF 2.77 (+1,191% return, $10K→$129K). 537 trades, 30% WR, 91% model exits. Score 0.138 on 298-day val set. **VIABLE** (p=0.0001). Pipeline integrity overhaul: data provenance, SHA256 sidecar, fatal feature mismatch, enriched experiment logging.
**Pre-v14 (archived):** v10-v13 failed due to silent bugs (data.pt gate labels, feature truncation, unified action head). v8 had replay PF 1.43. v6 was first viable model. 230 pre-v3 cycles archived. See `archive/` for details.

## Strategic Changes Tried
| Cycle | Change | Replay PF Before | Replay PF After | Verdict |
|-------|--------|------------------|-----------------|---------|
| 008 | v14 pipeline integrity + doc cleanup + Monte Carlo | 2.77 | (current) | Infrastructure — no model changes |
| 007 | v14 confirmation: re-eval v10 on 298-day set, proved v14 > v10 | N/A | 2.77 | v14 IS the best model. v10 scores 0.064 on same set. |
| 006 | v8: revert v7, VALUE_W=0, EXIT_W=0.15, fresh start | 0.99 (v7) | 1.43 | Profitable but on 70-day val set (inflated). |
| 005 | v7.1 PBT sweep (8 gen × 6 pop) | N/A | 0.99 | PARTIAL — score improved but replay NOT VIABLE |
| 001-004 | v6 breakthrough → v7 experiments → PBT sweeps | N/A | various | See archive for details |

## Current Hypothesis
**v14 is VIABLE but backtest P&L is hindsight-inflated.** Pickles (the trader): "backtests have advantage of hindsight, where something you would have never taken in the moment actually worked out on paper." Focus on **win rate with margin of error** rather than raw P&L to emulate the human factor. Monte Carlo stress-testing now validates robustness beyond single-path backtesting.

**Key v14 weaknesses (from cycle-008 research):**
- Midday trades: PF 0.30, 64 trades avg -11% → suppress lunch entries
- Short holds: 419 trades held ≤3 bars avg -7.4% → gate too jittery
- EOD exits: 44 trades avg +264% → model should hold longer
- Win rate 30% — needs improvement for real trading confidence

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
