---
date: 2026-04-26
parent: h3c_tcn_5seed_2026_04_26.md
status: FALSIFIED — replacing binary peak target with continuous regret regression catastrophically degrades aggregate PF (-0.965 from baseline, p[d<=0]=1.000); confirms C-Pre's reading that magnitude is irreducibly noisy
---

# H3f: Continuous Regression Target — Falsified

## Setup

User pushed back on premature deployment ("if we deploy and it trades poorly
or well, how will that help?"). Six prior experiments tested architecture
and selection but always with the same target: `int(current_pnl >=
suffix_max[i])` (binary peak detection).

Hypothesis: the C-Pre "98% within-cell noise" was partly target-induced.
The eval scores `hybrid_live_utility($)` (continuous, magnitude-preserving)
but training discards magnitude. Replace with regression target =
`suffix_max[i] - current_pnl` (regret-from-holding) and use
`HistGradientBoostingRegressor(loss="absolute_error")`. Inference: exit
when `predicted_regret <= threshold` (default $50).

## Result on seed 42 (376 OOS trades)

```
   variant   agg_pf   spread    floor
  baseline    2.195    0.995    1.405
       h3a    2.308    1.596    1.606
    regret    1.230    0.883    0.851
```

Bootstrap delta-PF CI: **[-1.481, -0.554]**, p[d<=0] = **1.000**.
Catastrophic, statistically certain.

Per-seed delta: -0.965 (way past the -0.10 discipline floor).

Per-cell:
- bull × low (n=143): 2.381 → 1.110 (-1.271)
- chop × mid (n=44):  2.155 → 1.297 (-0.858)
- bull × mid (n=45):  1.830 → 0.851 (-0.979)
- chop × high (n=43): 2.345 → 1.620 (-0.725)
- bear × high (n=32): 2.401 → 1.734 (-0.667)
- chop × low (n=47):  1.405 → 1.540 (**+0.135**) — only cell that lifted

## Mechanism

The peak target benefits from binary discretization: the model learns
"given features, P(this is the peak?)." That's a robust signal because
many similar bars across many trades cluster around similar peak-or-not
distributions.

The regret target is regression on a continuous, hugely noisy variable
(future suffix_max ranges across $-5000 to +$5000). With noisy
supervision, the regressor's outputs cluster near the training mean
(low-variance prediction is the safe play under absolute_error loss).
Result: most predicted regrets fall in a narrow band around the mean,
and threshold-based exits fire too uniformly — no edge.

This is **exactly** what C-Pre found: continuous regression R² on
signed_log_hl was -0.015 (linear) and +0.06 (RF). Predicting magnitude
is hard. The binary peak target works because it doesn't *try* to
predict magnitude.

## What this rules out

The hypothesis "binary target throws away magnitude info" is **wrong**.
The opposite is true: **binary peak target is the right discretization
for this noisy signal.** Magnitude information isn't lost in supervision;
it never existed extractably from features.

H3f at a different angle confirms C-Pre's reading. The L3 oracle's
representation ceiling is in the underlying signal, not in the target.

## What this teaches us about the chop × low cell

chop × low has now been tested under 4 different interventions:
- H3a (3 trajectory features): +0.05
- H3e (target reshape + Greeks): +0.05 (in mixed result)
- TCN (sequence model): **+0.83** (consistent across all 5 seeds)
- regret (regression target): +0.14

**Every intervention improves chop × low.** It's the persistent
small-winner failure cell, and it responds to almost any change. But
no intervention generalizes to the other cells. This is consistent with
"chop × low has different distributional structure than other cells"
rather than "the model has a generic bug."

## Discipline action: NOT reverted

Per user instruction (preserve infra with independent value):
- `target_mode` plumbing in `_build_trade_data`, `train_models_by_window`,
  `replay_trade_set` retained
- `--target {peak, regret}` and `--regret-threshold` flags retained
- `HistGradientBoostingRegressor` import retained

This is reusable scaffolding for any future regression-target experiment
(quantile loss, different loss families, ensemble of regressor + classifier).

No code changes reverted. No promotion claim either.

## What's left (per user's A → C → B order)

**A (this plan, H3f) — FALSIFIED.**

**C (next): qualitative analysis of specific failure modes.** Pull the
worst-loss trades from chop × low and look at them bar by bar. What
does a bad chop × low trade look like that a good one doesn't? Generate
hypotheses for what's actually distinguishable.

**B (after C if C generates hypotheses): domain-specific features
(GEX, charm flows, gamma flip).** New features = new signal, by
construction. But expensive to implement.

## Cost

- ~1 hour dev (H3f-1 plumbing)
- ~42 min CPU (seed-42 build)
- ~30s eval
- $0 GPU
- Total: ~1.5 hours. Cheap conclusive falsification.

## Branch state

```
research/h3e-features:
  33f52350 H3f-1: implement --target {peak, regret}  (KEPT, no behavior on default)
  bbfd45c0 H3c-3 RESULT: 5-seed TCN partial
  e184114f H3c-2 seed-42 result: TCN lifts floor + aggregate
  acf9c6bc H3c-2 part 1: integrate TCN oracle class
  6c8097b1 H3c-1: causal TCN module + bit-exact causality tests
  ... (full ladder history preserved)
```

## Files

```
v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_regret.npz  (kept for reproducibility)
v3/artifacts/research/regret_seed42_eval.json
v3/reference/h3f_regret_falsified_2026_04_26.md  (this doc)
```
