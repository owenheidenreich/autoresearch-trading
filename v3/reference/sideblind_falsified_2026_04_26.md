---
date: 2026-04-26
parent: regime_diagnostic_2026_04_26.md
status: FALSIFIED but informative — sideblind oracle narrows spread to 0.710 (well below 1.00 target) BUT crashes aggregate PF (-0.705); per-seed delta -0.705 fails discipline gate; reverted
---

# Step A (Sideblind): Spread Narrows But Aggregate Crashes

## Setup

Plan ladder Step A. Question: does removing `direction_is_call` from
`trade_state` narrow the cross-cell PF spread by forcing a side-agnostic
exit policy?

Method: added `--side-blind` flag to L3 oracle build that zeros the
`direction_is_call` feature. Trees ignore constant features, so this is
functionally equivalent to dropping the column. Built seed-42 oracle
on top of the H3a 10-feature variant.

## Results (seed 42, 376 OOS trades, hybrid_live scoring)

```
   variant   agg_pf   spread    floor
  baseline    2.195    0.995    1.405
       h3a    2.308    1.596    1.606
 sideblind    1.490    0.710    1.311
```

Per-cell deltas (vs baseline):

```
trend  vol   n     base   sideblind   delta
chop   low   47    1.41    1.45      +0.05  ← floor LIFTED
chop   high  43    2.35    2.02      -0.32
chop   mid   44    2.16    1.36      -0.80
bull   high  90    1.55    (similar regression — see csv)
bear   high  32    2.40    1.46      -0.94
bull   mid   45    1.83    1.31      -0.52
bull   low  143    2.38    1.36      -1.02  ← biggest crash
```

Bootstrap delta-PF 95% CI: **[-1.14, -0.36]**, p[d ≤ 0] = 1.000. Aggregate
PF regression is statistically significant.

## Key finding: spread/aggregate tradeoff is real

Sideblind delivered exactly what its construction predicted:
- ✅ **Spread dramatically narrowed** (0.995 → 0.710, well below the user's
  1.00 target).
- ✅ **Floor lifted** in the worst-baseline cell (chop × low: 1.41 → 1.45).
- ❌ **Every strong cell crashed** (bull × low: -1.02, bear × high: -0.94,
  chop × mid: -0.80).
- ❌ **Aggregate PF lost 32%** (-0.705 PF).

The model lost the ability to differentiate calls vs puts and applies a
"compromise" exit policy that's worse than the directional-aware policy
in EVERY non-floor cell.

## Lesson updated

The plan's hypothesis: "if A fails, side-conditioning isn't the spread
driver."

**This is wrong.** Side-conditioning IS exactly the spread driver —
removing it does narrow the spread. But the cost is too high:
side-information is load-bearing for aggregate PF in every regime where
side direction matters (i.e., almost every cell).

Implication for Angle C: the right intervention is NOT to remove side info
but to **keep all information AND explicitly upweight rare regimes during
training**. That preserves the per-cell discrimination while reducing the
training-time bias toward bull regimes (54% of training data).

## Discipline action: revert per plan

`--side-blind` flag was reverted via `git revert dcffa3eb` →
`1b8c9a13`. Net code change: zero.

The eval-script parameterization commit (`e1346b25`) is kept as useful
infra — it makes the spread-eval reusable for future variants (DRO,
H3c, etc.).

## What this rules out and what's next

**Ruled out:**
- "Removing direction_is_call narrows spread without aggregate cost" —
  falsified. Aggregate cost is severe (-0.705 PF, 32% loss).

**Confirmed:**
- Side-conditioning is the spread driver. Removing it narrows spread by
  construction.
- The regime spread is genuinely an L3 problem (the L2 audit showed it's
  also partly L2 — both contribute).

**Next: Angle C — keep side info, target spread via training-time
re-weighting.** Two variants:

1. **One-shot inverse-frequency:** weight each training row by 1 /
   cell_count (bears get ~10x weight of bulls). Single training run.
2. **Iterative DRO:** start uniform, eval per-cell, upweight worst cells,
   retrain; converge in 2-3 rounds.

Plan said "use iterative DRO if A failed." But A's failure mode was very
specific (over-aggressive intervention crashed aggregate). One-shot
inverse-frequency is a milder, cheaper test that may already work. Will
likely run one-shot first; iterate if it doesn't narrow spread enough
without crashing aggregate.

## Files

```
v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_sideblind.npz  (kept)
v3/artifacts/research/sideblind_seed42_eval.json
scripts/research_eval_h3e_oracle.py  (parameterized in e1346b25, kept)
v3/reference/sideblind_falsified_2026_04_26.md  ← this file
```

## Branch state

```
research/h3e-features:
  1b8c9a13 Revert "Step A: add --side-blind flag" (this revert)
  e1346b25 research_eval_h3e_oracle: parameterize variant + gate thresholds (KEPT)
  dcffa3eb Step A: add --side-blind flag (REVERTED above)
  cf92e4a8 Step B: Layer-2 entry audit
  ce7bfe76 H3e RESULT: falsified
  ... (H3e + H3a chain)
```

## Cost

- 43 min CPU: seed-42 sideblind oracle build
- ~5 min: eval + revert + writeup
- $0 GPU
- Total: ~50 min CPU. Cheap falsification.
