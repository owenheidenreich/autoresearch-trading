# G1 ES Direction Screen — Held Plan

**State: HOLD pending the independent measurement-capacity review. Do not implement or run this plan yet.**

## Purpose

Answer the project’s only economic question: can a simple, causal policy predict ES direction over 15,
30, or 60 minutes strongly enough to beat 0.358 ES points per completed round trip?

## Frozen family

- M1: first-five-minute opening-range acceptance or rejection.
- M3: non-roll overnight gap continuation or reversal.
- Their predeclared joint combinations, with no fitted model and no post-result threshold search.

The full definitions, folds, occupancy accounting, multiplicity correction, and pass conditions are in
[the gate-chain audit §3](../../research/findings/GATE_CHAIN_AUDIT_2026_08_05.md#3-g1--direction).

## Order of work after release

1. Freeze timestamps, horizons, side rules, abstentions, one-position accounting, and the complete family.
2. Run the matched-surrogate known-answer campaign without inspecting real-policy economics.
3. Repair the null at most once if the predeclared false-pass or injected-effect recovery gate fails.
4. Only after the null passes, run the raw M1/M3 economic screen once on owned ES data.
5. Recompute power from the frozen realized occupancy; report `UNDERPOWERED`, `NO LARGE EDGE`, or `PASS`
   honestly rather than treating an unresolved result as proof of no edge.

## Pass consequence

Only an exact 60-minute G1 pass may reopen one locked option-dollar replay. It does not authorize model
training, threshold tuning, data purchases, paper trading, or promotion.
