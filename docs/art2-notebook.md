# ART² Lab Notebook — Outer Loop Memory

## System
ART² meta-loop wrapping autoresearch inner loop.
Outer loop (Opus) makes strategic changes; inner loop (Sonnet) optimizes train.py.

## Strategic Changes Tried
| Cycle | Change | OOS PF Before | OOS PF After | Verdict |
|-------|--------|---------------|--------------|---------|
| 004 | Purged 7 gamed entries from promoted history (SCORE_DRAWDOWN_PENALTY exploit). Reset baseline 10.29→1.77 | N/A | N/A | CRITICAL FIX |
| 020 | Fresh start + relaxed inner loop constraints. program.md: scheduling/weighting/LR schedules unlocked. lab_notebook: rigid 2-param grid → open exploration. Model .bak'd (54.5% train/eval mismatch) | PF=0.65 | PF=1.11 | STRUCTURAL REFORM |
| 022 | Action G: Removed BalancedStrikeGate (dead weight, 0/16+ pattern), fixed direction bias to favor ATM (+0.15) over OTM (-0.10), removed stressed-account OTM push, removed ETV head. Fresh start. | PF=1.11 | TBD | ARCHITECTURE FIX |

## Paper Trading P&L Tracking
| Date | Trades | P&L % | Backtest Expected | Divergence |
|------|--------|-------|-------------------|------------|

## Dead Ends (Strategic Level)
| Change | Cycles | Result |
|--------|--------|--------|

## Dead Ends (Strategic Level)
| Change | Cycles | Result |
|--------|--------|--------|
| Rigid 2-param _env_float grid in lab_notebook | 017-020 | 100% tunnel vision, 0% accept rate. Inner loop has no creative freedom. |

## Current Hypothesis
- Cycle 022 achieved replay PF=1.11 (9 trades, 55.6% WR) — first profitable replay since fresh start.
- But the model is still one-dimensional: morning calls only, ATM only, no puts.
- Root cause identified: BalancedStrikeGate added random noise (0/16+ new module pattern), direction bias init pushed toward OTM (-601% cumulative), stressed-account bias pushed ATM → OTM during drawdowns.
- Fix: removed all three anti-patterns, reversed direction bias to favor ATM (+0.15), removed ETV head complexity.
- With a cleaner architecture and domain-knowledge-aligned initialization, the inner loop should find broader trading patterns (puts, afternoon trades) with higher PF.

## What Works (Outer Loop)
- ART² correctly identified the gamed baseline as the root cause of 0% accept rate (cycle 004).
- Promoted history audit trail makes gaming visible and reversible.
- Research phase successfully identified model bias patterns (call-only, morning-only, ATM-only).
- Deploy.sh no longer overwrites train.py when local changes exist.
