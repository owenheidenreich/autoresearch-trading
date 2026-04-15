# Current State

## Project goal
Build a trustworthy exact-chain research system for SPX 0DTE long options, then a session-level agent that watches a day unfold, maintains a thesis, updates it, and acts coherently through time.

## Phase map
1. Substrate / truthfulness — completed
2. Static supervised redesign — completed, failed in a useful way
3. Sequential prototype — completed
4. BC + oracle refinement — completed
5. RL beyond BC — completed
6. Economic validity / side-bias phase — completed
7. Late-session zombie bleed phase — completed
8. Chop-side economics phase — completed
9. Fold-1 regime-generalization / Side13 refinement — completed
10. Current phase: **milestone freeze + strategic reassessment**

## Current best overall agent
Side13 structural feature agent (epoch 5) with:
- `best_call_score - best_put_score` as session state feature 12 (dim 12→13)
- bar89 late-entry block (`ENV_LATE_ENTRY_BAR=89`)
- bar95 underwater zombie-close (`ENV_LATE_EXIT_BAR=95`)
- side-balance penalty `RL_SIDE_IMBALANCE=0.03`
- decay penalty `ENV_DECAY_COEFF=0.001`
- escalating entry cost `RL_ENTRY_COST=0.015`, `RL_ENTRY_ESCALATION=0.015`
- KL anchor `RL_KL_COEFF=0.03`
- overall PF ≈ 0.963 across 300 out-of-sample days (5 folds)
- chop PF improved from 0.632 to 0.826
- call% 48.4% (learned, not forced)
- flips/day 0.36
- Fold 4 PF 1.190
- Fold 1 PF 0.830 (regressed from reference's 0.904)

## Most important completed findings
- Session state is load-bearing (zeroed-state ablation: total behavioral collapse).
- BC teaches structure (thesis persistence, phase timing, low flips) but not trade-quality judgment.
- RL can improve trade quality (PF 0.937→1.422 in first run) and participation selectivity (4/day→1.5/day).
- Wrong-side collapse was caused during RL by temporal overfit to validation window, not by data/oracle/encoder.
- Side-balance penalty (0.03) fixed the put bias without hard-forcing 50/50.
- Late-session zombie bleed was real: bar89 entry block + bar95 zombie close improved PF 0.855→0.920.
- Side13 feature materially improved chop economics and overall PF (0.920→0.963).
- Longer RL training (8 epochs vs 5) made things worse (PF 0.963→0.828). The policy over-specializes on recent data with more epochs. Epoch 5 is a sweet spot where BC structure is still partially intact.
- Remaining Fold 1 regression is structural, not undertraining.

## Current bottleneck
The Side13 epoch 5 agent at PF 0.963 is likely near the ceiling of what this architecture + training setup can achieve. The remaining -4.12 net PnL gap is spread across multiple folds and day types with no single dominant pattern left to target. Further RL tuning makes things worse, not better.

The next decision is whether:
- PF 0.963 on 300 out-of-sample days is a credible stopping point for this training approach
- Or whether the remaining gap justifies architectural changes (encoder unfreezing, richer memory, different RL method)

## What NOT to do next
- No more RL tuning with the current setup (diminishing returns, over-specialization risk)
- No encoder unfreezing without clear hypothesis
- No action-space expansion
- No broad reward redesign
- No oracle re-refinement
