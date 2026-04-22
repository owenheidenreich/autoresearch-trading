# Side-Contrastive Small-Weight Sweep — 2026-04-22

## Hypothesis

The `0.5` side-contrastive weight damaged PF (CPU baseline `1.066 →
1.002`, `-0.064`). Small weights (`0.10`, `0.20`) should reduce call
share gently without destroying margin quality. Find the sweet spot
where call share drops below `93%` and PF stays ≥ `1.066`.

## Mechanism Expected

`_side_contrastive_loss` pushes best_side_pred above other_side_pred
on bars where the sign of the utility gap is material (≥ $10). Small
weights should apply a modest de-biasing pressure that the rest of
the loss can absorb.

## Commands

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --tier dev --device cpu --seed 42 \
  --run-dir v3/artifacts/layer2_unified_policy_side_w010_seed42 \
  --w-side-contrastive 0.10

.venv/bin/python -m v3.layer2.train_unified_policy \
  --tier dev --device cpu --seed 42 \
  --run-dir v3/artifacts/layer2_unified_policy_side_w020_seed42 \
  --w-side-contrastive 0.20
```

## Results

CPU dev, seed 42, all 13 rolling windows:

| weight | trades | PF | DD | mean/trade | **call share** |
|---|---|---|---|---|---|
| 0.0 (baseline) | 410 | 1.066 | 48.8% | +$24.6 | 93.5% |
| **0.10** | 426 | **1.074** | **42.5%** | +$27.8 | **93.9%** |
| **0.20** | 438 | **1.091** | 44.1% | +$34.0 | **94.5%** |
| 0.5 (prior) | ~438 | 1.002 | 59.7% | +$0.8 | 90.9% |

## Interpretation — Hypothesis Falsified

Call share **did not drop**. It held at baseline at `0.10` (93.9% ≈
93.5%) and actually **rose** at `0.20` (94.5%). The only weight in
the sweep that reduced call share was `0.5`, and that reduction came
from the loss being large enough to degrade ranking across the board
rather than from systematic side discrimination.

### Why

The side-contrastive term points in the direction of the training
data's observed utility gap. In this dataset calls have higher mean
positive utility than puts on most bars (the training window covers
a trending-up market). So the loss mostly fires with sign "push best
call above best put". A **small** weight correctly fits this signal
and **reinforces** the call prior. A **large** weight still points
the same way, but is loud enough to interfere with the other loss
components and randomise the side selection.

In other words: the side-contrastive loss is not a de-biasing tool.
It is a **side-signal-amplification** tool. If the data favors one
side, small contrastive weights make the model more biased toward
that side.

### What it *does* do

A small weight produces a modest PF and DD improvement:

- `w=0.20` gives PF `1.091` vs baseline `1.066` (`+0.025`), DD 44.1%
  vs 48.8% (`-4.7pp`)
- `w=0.10` gives PF `1.074` vs baseline `1.066` (`+0.008`), DD 42.5%
  vs 48.8% (`-6.3pp`)

The PF improvement is small; the DD improvement is more material.
These are seed-42 CPU numbers; they do not change promotion status
on their own.

### Repo-belief change

Before: "93% call share is a model pathology that side-contrastive
training can fix."

After: **the 93% call share reflects real training-data regime bias,
not a model pathology.** To reduce call share systematically we would
need training-time sample reweighting (force balanced call-winning
and put-winning bars per batch) or an explicit side-uniformity
penalty, not a gradient that points along the true utility gap.

## Next-cycle implications

- Drop "tune the side-contrastive weight to fix the side prior" as a
  hypothesis. It is falsified.
- Keep `--w-side-contrastive` as a tunable; `0.10`–`0.20` produces
  modest PF/DD gains. If we ever GPU-promote again, these weights
  are a reasonable default candidate.
- The "reduce call share" problem itself may be the wrong frame. If
  puts did not earn PF on most OOS bars in this training window,
  trading more puts is **not** an improvement. Live monitoring for
  regime change (VIX, ES trend) is the honest response; forced side
  balance is not.
- Move back to the leading composition (unified + Layer-3 honest
  calibration, mean PF `1.670`) for the next cycle.

## Artifact Locations

- `v3/artifacts/layer2_unified_policy_side_w010_seed42/`
- `v3/artifacts/layer2_unified_policy_side_w020_seed42/`
- prior: `v3/artifacts/layer2_unified_policy_side_contrastive_seed42/` (`w=0.5`)
