# Side-Contrastive Rescue Falsified — 2026-04-22

## Question

The cycle-6 stacking result showed `CPU w=0.20 + L3 robust 0.90`
gave the best mean PF (`1.834`) but the worst min PF (`1.649`), with
seed 43 regressing from `1.725 → 1.649` and DD jumping from `21.7% →
32.9%`. Is this regression local (fixable with a smaller weight) or
broad (a feature of side-contrastive on this stack)?

## Diagnosis — Seed 43 Regression Localizes to Window 5

Diffed the per-window calibrated composed output between
`layer3_unified_cpu_w000_seed43/` and `layer3_unified_cpu_w020_seed43/`:

| W | thr (both) | n w=0.00 | n w=0.20 | Δn | Δpnl |
|---|---|---|---|---|---|
| 0 | 0.20 | 46 | 44 | -2 | -$4601 |
| 1 | 0.15 | 56 | 56 | 0 | -$2490 |
| 2 | 0.15 | 4 | 5 | +1 | -$49 |
| 3 | 0.15 | 28 | 41 | +13 | +$2521 |
| 4 | 0.15 | 3 | 3 | 0 | -$82 |
| **5** | **0.15** | **22** | **46** | **+24** | **-$11040** |
| 6 | 0.15 | 27 | 27 | 0 | +$3320 |
| 7 | 0.19 | 32 | 32 | 0 | +$682 |
| 8 | 0.15 | 17 | 17 | 0 | -$2655 |
| 9 | 0.15 | 41 | 9 | -32 | -$22 |
| 10 | 0.15 | 56 | 57 | +1 | +$1410 |
| 11 | 0.15/0.19 | 38 | 36 | -2 | +$1692 |
| 12 | 0.19/0.15 | 20 | 40 | +20 | +$6809 |

**W05 alone is -$11040**, driven by entry count doubling (22 → 46).
L3 cannot rescue these extra entries — PF at every threshold is
`≤ 0.96`:

| thr | w=0.00 W05 | w=0.20 W05 |
|---|---|---|
| 0.15 | PF 2.57, n=22, +$5198 | PF 0.47, n=46, -$5842 |
| 0.19 | 2.34, 22, +$5377 | 0.96, 46, -$466 |
| 0.20 | 2.46, 22, +$5537 | 0.90, 46, -$1339 |
| 0.25 | 2.41, 22, +$5811 | 0.87, 46, -$1925 |
| 0.30 | 2.45, 22, +$6913 | 0.96, 46, -$662 |

The damage is in the **entry set**, not the exit calibration. Side-
contrastive is shifting which days the policy trades, and for seed
43 the extra 24 days in W05 are predominantly losers.

Cross-seed W05 comparison (thr 0.15):

| seed | n w=0.00 | n w=0.20 | Δpnl |
|---|---|---|---|
| 42 | 19 | 36 | **+$4751** |
| 43 | 22 | 46 | **-$11040** |
| 44 | 42 | 42 | +$143 |

**Seed 42 W05 doubles under w=0.20 and profits; seed 43 W05
doubles under w=0.20 and loses heavily; seed 44 W05 doesn't double.**
The regression is seed-initialization specific, not a systematic
overtrading pathology.

## Rescue Attempt — w=0.10 on All Three Seeds

Smaller weight → smaller boundary shift → less day-selection
disturbance. Hypothesis: w=0.10 keeps enough side signal to lift
mean PF while avoiding seed 43's W05 overtrading.

Commands:

```bash
for s in 42 43 44; do
  .venv/bin/python -m v3.layer2.train_unified_policy --tier dev --device cpu \
    --seed $s --run-dir v3/artifacts/layer2_unified_policy_side_w010_seed$s \
    --w-side-contrastive 0.10
  .venv/bin/python -m v3.layer3.train_rolling --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_side_w010_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_w010_seed$s
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_w010_seed$s \
    --policy prior_window_robust --robust-slack 0.90 \
    --out-suffix robust_90
done
```

## Results

Calibrated L3 robust-90 PF / DD per seed:

| config | seed 42 | seed 43 | seed 44 | mean PF | min PF | mean DD |
|---|---|---|---|---|---|---|
| CPU w=0.00 | 1.711 (20.2%) | 1.725 (21.7%) | **1.913** (24.1%) | 1.783 | **1.711** | **22.0%** |
| CPU w=0.10 | **1.861** (25.7%) | 1.713 (27.4%) | 1.605 (22.0%) | 1.726 | 1.605 | 25.0% |
| CPU w=0.20 | 1.816 (19.1%) | 1.649 (32.9%) | 2.036 (30.9%) | **1.834** | 1.649 | 27.6% |

W05 entry counts under w=0.10:

| seed | W05 n (w=0.00) | W05 n (w=0.10) | W05 n (w=0.20) |
|---|---|---|---|
| 43 | 22 | **22** (rescued!) | 46 (broken) |

## Hypothesis Falsified

`w=0.10` **does rescue seed 43's W05** (back to 22 entries and
`+$3215` in that window), but **breaks seed 44** (PF `1.913 → 1.605`,
a `-0.308 PF` regression). Mean PF falls to `1.726` — worse than
`w=0.00`. Min PF is the worst of all three configs at `1.605`.

The pattern is now clear across three weights tested:

- `w=0.00`: cleanest per-seed spread, best min PF
- `w=0.10`: breaks seed 44
- `w=0.20`: breaks seed 43

Every non-zero weight tested degrades at least one seed's composed
output. The side-contrastive auxiliary is **not a stable de-biasing
tool on this stack** — it is a seed-dependent boundary perturbation
that shuffles which days the policy trades, and different seeds'
initializations land differently on the new boundary.

A third weight (e.g. `w=0.15`) might average the two damages but is
almost certainly not a rescue: it would sit between the two known
failure modes. Not worth the cycle.

## Decision — Lock Provisional Champion

**Champion**: `CPU w=0.00 + L3 robust 0.90`

| metric | value |
|---|---|
| mean PF | 1.783 |
| min PF | **1.711** |
| mean DD | **22.0%** |
| max DD | 24.1% |

This is the steadiest composed stack across seeds. Both the mean
and the floor are defensible, and DD is closest to Layer-2.5's
`21.4%` reference.

`w=0.20 + L3 robust` has a higher mean but its min PF and max DD
make it a worse promotion-grade choice. The "mean PF lift" of
`+0.051` is from seed 42 and seed 44 improving while seed 43
regresses, and that tradeoff is exactly what a min-PF-gated
promotion should reject.

## Belief Change

- **Side-contrastive is not a knob to tune on the composed stack.**
  Any non-zero weight tested broke a seed. The assumption that
  "lower weight = smaller perturbation = safer rescue" is not
  wrong, but it also made the policy miss the days that the larger
  weights were catching on other seeds. There is no monotone-safe
  direction in this loss dimension.
- **The 93% call share in w=0.00 is not an overrun bug; it is the
  training regime's genuine bias.** Trying to correct it with
  side-contrastive replaces one seed's bias with another seed's
  different bias.

## Next Workstream

Per the loop spec, champion lock transitions to the **first
outer-loop retrain hypothesis**: retrain the unified policy against
utilities that already include the composed exit from the locked
champion. That is the next cycle's work.

## Artifacts

- `v3/artifacts/layer2_unified_policy_side_w010_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_w010_seed{42,43,44}/rolling_layer3_calibrated_robust_90.json`
