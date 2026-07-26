# Challengers And Failure Patterns

This document explains why later challengers have not replaced Protocol101, even when some showed higher exposed-split PnL.

## Higher PnL Was Not Enough

Protocol248 is the cleanest comparison between Protocol101 and a stronger gross-PnL challenger.

| Policy | Trades | PnL | Win rate | Avg PnL | PF | Median premium | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Challenger premium blend | 2,346 | $465,340 | 54.8% | $198 | 2.17 | $1,650 | -$9,605 |
| Protocol101 | 907 | $268,220 | 76.0% | $296 | 3.78 | $2,850 | -$4,450 |

The challenger made more money in replay by trading more and expanding the opportunity set, but it degraded the properties that matter for paper-default use:

- lower hit rate,
- lower average trade quality,
- lower profit factor,
- larger drawdown,
- more churn,
- lower premium/OTM exposure that was not consistently high quality.

This is the central reason "beat Protocol101 on PnL" is not enough.

## Timing Fragility

Protocol194 and Protocol215-style full-action challengers showed that broader candidate surfaces can find large opportunity. But Protocol195 showed Protocol194 was extremely timing fragile:

| Split | Original | Entry +1m | Exit +1m | Both +1m |
|---|---:|---:|---:|---:|
| March 2026 | $73,440 | -$27,980 | $73,540 | -$27,880 |
| Q1 2026 | $202,720 | -$40,135 | $197,110 | -$47,310 |
| Q3 2025 | $81,870 | -$35,095 | $82,050 | -$42,560 |
| Q4 2025 | $138,895 | -$39,320 | $141,085 | -$43,490 |
| Recent 2026 | $25,295 | -$12,490 | $24,325 | -$11,530 |

Entry timing was the primary damage. A model that wins only at exact historical quotes is not a paper-default candidate until live no-order parity and fill evidence prove it can be executed.

Protocol101 also has timing fragility. Protocol114 and Protocol126 show severe degradation under delayed entry. The difference is that Protocol101 has more operational plumbing and readiness work, not that timing risk is solved.

## Lifecycle Failure: Both Clipping And Overholding

The project found both lifecycle failure modes:

- Some models clipped winners too early.
- Other models held too long, gave back MFE, or blocked better later entries.

Protocol200 and Protocol276 are the key warnings. Protocol276's attribution found:

- 165 negative `A_enter` trades for `-$35,480`.
- 159 overhold/late-exit trades for `-$33,275`.
- Actual exits left `-$424,160` versus best observed path PnL.
- The lifecycle model was trained on a distribution that did not match the entry policy's deployed state distribution.

This means "hold longer" and "exit faster" are both wrong as generic instructions. The strategy needs a context-aware taxonomy: scalp, runner, failed continuation, giveback risk, and slot-cost risk.

## Slot Opportunity Cost

The single open-position slot is not a bookkeeping detail. It is a tradeable asset.

Protocol276 failed partly because challenger trades occupied the slot and blocked later Protocol101 opportunities. The learned-defer work corrected this directionally by charging expected blocked Protocol101 cost before allowing challenger overrides.

The learned-defer challenger produced positive diagnostic replay deltas, but the validation packet blocked holdout scoring because slot-cost calibration was unstable:

- Q3/Q4 AUC near `0.99`.
- Q1 AUC about `0.81`.
- Recent AUC about `0.74`.
- Formal strategy-matrix/PBO controls missing.
- Live no-order full-action parity and fill evidence missing.

Lesson: baseline-relative slot cost is the right concept, but the current estimator is not yet promotion-grade.

## Label And Objective Mismatch

Many ML protocols optimized labels that did not fully match the deployed trading problem.

Examples:

- Full-surface rowwise entry labels tried to learn sparse positive `A_enter` values even though the deployed action is listwise and serial.
- Lifecycle labels used future-best path information that may overstate real hold value.
- Premium/return labels increased exposure but did not preserve trader-quality constraints.
- Unified action-advantage labels gave a coherent game, but Protocol276 showed the entry model and lifecycle model did not align under deployed state distribution.

The next model needs a trader-level objective first. Only then should labels be built.

## What Not To Repeat

Do not run more experiments that are only:

- bigger/smaller MLP,
- more epochs,
- seed sweeps,
- threshold hunts,
- raw PnL optimization,
- pure return-on-premium optimization,
- rowwise candidate scoring without listwise/slot context,
- lifecycle "hold longer" rules,
- exposed-split tuning.

The next model should be a test of a named trading idea.

