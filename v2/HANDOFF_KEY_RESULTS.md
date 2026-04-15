# Key Results

## Major milestone progression

| Agent | Overall PF | Fold 1 PF | Fold 4 PF | Call % | Flips/day | Key change |
|---|---|---|---|---|---|---|
| BC (persistent oracle) | 0.937 | — | — | 53% | 0.75 | structure from BC |
| RL v1 (flat cost) | 0.860* | 0.420 | 1.422 | 35% | 0.79 | trade quality from RL |
| Stabilized (selectivity) | 0.633* | — | 0.895 | 0.9% | 0.01 | participation selectivity |
| Side-balanced | 0.791* | 0.377 | 0.983 | 49% | 0.23 | fixed RL-induced put bias |
| Grid-optimized (s03/d001) | 0.855* | 0.777 | 1.022 | 34% | 0.47 | best reward operating point |
| + bar89/bar95 constraints | 0.920 | 0.904 | 1.011 | 35% | 0.42 | late-session bleed fix |
| **Side13 (structural)** | **0.963** | 0.830 | **1.190** | **48%** | **0.36** | side-score feature |

*Evaluated on promote_mask (60 days) only; all others on full 5-fold (300 days)

## Important economic findings
- Calls beat puts in 4/5 folds. Call PF ≈ 1.0 across all agents; put PF ≈ 0.7-0.9.
- RL-induced put bias was temporal overfit to validation window, not present in data/oracle/BC/encoder.
- Chop-day puts are the single largest loss source. Chop calls are breakeven-profitable.
- Probe forcing chop puts to calls recovered 69% of remaining loss (PF 0.920→0.975).
- Side13 recovered 62% of chop loss through learned behavior (PF 0.920→0.963).
- Longer RL (8 epochs vs 5) degraded PF from 0.963 to 0.828 — over-specialization, not improvement.
- The remaining -4.12 net PnL gap has no single dominant pattern to target.

## Session state ablation
Zeroing session state produces total collapse: 13+ flips/day, 0 exits, 0.014 hold rate. Session state carries all structured behavior.

## Side-score feature (best_call_score - best_put_score)
- Added as session[12], increasing state dim from 12 to 13
- Agent naturally learned regime-conditional side selection: 82% calls in bull, 14% in bear
- Chop-day call preference emerged without hardcoded overrides
- Fold 1 regressed (grind puts became unprofitable) — structural tradeoff, not fixable with more RL

## Current interpretation
Project is no longer blocked by architecture or broad behavior. The session agent demonstrates genuine stateful trading: thesis persistence, regime-aware side selection, active exits, participation selectivity. The remaining question is whether PF 0.963 on 300 OOS days is sufficient, or whether a different training method (offline RL) could close the last ~4%.
