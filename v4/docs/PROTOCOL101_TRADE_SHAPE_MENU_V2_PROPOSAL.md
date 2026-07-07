# Protocol101 Trade-Shape Menu v2 — PROPOSAL (awaiting owner sign-off)

Status: DRAFT for owner review. Nothing in this document is implemented until
approved. Approval triggers: pinned-policy update, full-corpus label rebuild,
verifier version bump (v3.4), and re-acceptance — all mechanical under the
existing pipeline.

## Why the menu is changing

The owner's directive (2026-07-06): the trader should adapt hold time, exit
behavior, and trade frequency to the situation — 0 trades some days, 5-6 on
others; 2-minute holds or 2-hour holds — without any single hard-coded
formula. In a supervised stage-1 system, adaptivity lives in the breadth of
the menu times the intelligence of the chooser. The current 3-policy menu
only describes quick, tight-stopped trades (10/25/45-minute holds, 35-65%
stops), which structurally biases any learner toward scalping — the style the
owner explicitly does not want as the identity of this trader.

Evidence for patient shapes: this project's own v3-era study found fixed
-35%/+60% stops cost ~$120k versus naive holding across 986 days — premature
stop-outs destroy convex payoffs on long options.

## Proposed menu (7 shapes)

All shapes are long-option, single-contract, ask-entry/bid-exit, forced flat
by 15:55 ET. `stop` and `target` are fractions of entry premium. `stop=1.00`
means the premium itself is the stop (exit only if bid reaches zero);
`target=99.0` is a sentinel for "no practical target — hold to time exit."

| # | Name | Stop | Target | Max hold | Trade thesis it expresses |
|---|------|------|--------|----------|---------------------------|
| 0 | quick_scalp (existing) | 0.35 | 0.60 | 10m | fast momentum burst |
| 1 | short_swing (existing) | 0.50 | 1.00 | 25m | standard intraday move |
| 2 | extended_swing (existing) | 0.65 | 1.50 | 45m | trend continuation |
| 3 | patient_trend (new) | 0.50 | 2.00 | 90m | multi-leg intraday trend |
| 4 | thesis_hold (new) | 1.00 | 3.00 | 120m | conviction play, premium-at-risk, harvest at 4x |
| 5 | convexity_run (new) | 1.00 | 9.99 | to forced flat | V-recovery / regime-break; let the tail run |
| 6 | tail_lottery (new) | 1.00 | 99.0 | to forced flat | pure convexity; exit only at close |

Notes:
- Shapes 4-6 have no premature stop: for a long option the maximum loss is the
  premium, which the affordability check already prices. This matches how a
  discretionary convexity trader actually risks.
- The owner's V-shaped-recovery example maps to: far-OTM call candidate (the
  ladder spans ±$50) + shape 4 or 5.
- Frequency adaptivity is unchanged by this document: abstention remains a
  first-class action, and the serial simulator's one-position-at-a-time rule
  plus hold length naturally bounds trades/day.

## What implementation requires (for scoping, not action yet)

1. `LabelPolicy` list extended in `NeuralDatasetConfig` (7 entries) and
   mirrored in the verifier's `PINNED_LABEL_POLICIES`.
2. Full-corpus label rebuild (labels tensor 3 -> 7 policies; ~2.3x label
   storage; several hours machine time) and v3.4 re-acceptance. Sampler
   coverage cells expand automatically (policy x offset); forced-flat
   reachability windows are config-derived, so shapes 4-6 (reachable across
   most of the day) sharply increase forced-flat verification coverage.
3. No simulator change: cooldown == max hold per selected policy, as today.
   For shapes 5-6, cooldown runs to forced-flat, i.e. at most one such trade
   per day can be open at once — consistent with serial one-account replay.

## Open questions for the owner

1. Approve all four new shapes, or trim? (Each shape adds label cost and
   multiple-testing surface; 7 is a reasonable ceiling, more is not better.)
2. Shape 6 (exit only at close) — keep, or is shape 5's 10x harvest enough?
3. Any shape you feel is missing — e.g., a mid-morning-only variant — or is
   time-of-day conditioning better left to the model (recommended)?
