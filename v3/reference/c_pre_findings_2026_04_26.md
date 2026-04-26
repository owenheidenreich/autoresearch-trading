---
date: 2026-04-26
parent: regime_diagnostic_2026_04_26.md
status: NO GO — continuous-magnitude predictability test shows per-trade outcome is ~98% within-cell noise; cell-conditional reweighting can't move the needle on spread; bottleneck is structural (representation level)
---

# C-Pre: Continuous-Magnitude Predictability Test — NO GO

## Setup

User flagged that the binary win/loss AUC test (used in the intraday-axis
search) conflates +$5000 and +$50 wins. Replaced with three magnitude-aware
analyses to better gate whether C3 (cell-conditional reweighting) is worth
the compute.

Method: 1664-trade sample (5 seeds, after dropping bar-120 boundary).

1. **Variance decomposition.** Of total trade-pnl variance, what fraction
   is between-cell vs within-cell? (Plan gate: ≥15% strong, ≥5% weak, <5% NO GO.)
2. **Continuous regression** on `signed_log_hl = sign(hl) × log10(|hl|+1)`.
   5-fold CV R² with linear, random forest, cell-only, and cell+features.
3. **Per-cell regression.** Same target, trained separately per cell.
   Compare R²s.

## Results

### Variance decomposition

```
Target = hl (raw $):
  Total variance:    965962
  Between-cell:       20012  (2.1% of total)
  Within-cell:       945950  (97.9% of total)

Target = signed_log_hl:
  Total variance:     6.65
  Between-cell:       0.05  (0.8% of total)
  Within-cell:        6.59  (99.2% of total)
```

**2.1% between-cell.** Way below the plan's 5% NO GO threshold.

### Continuous regression R² (5-fold CV)

```
Linear regression (entry features):     R² = -0.015 ± 0.014
Random forest (non-linear):             R² = +0.059 ± 0.017
Cell ID only (9 dummies):               R² = -0.008 ± 0.010
Cell ID + features:                     R² = -0.021 ± 0.018
```

Linear models worse than predicting the mean. RF catches some non-linearity
(+0.06). Cell membership alone is useless.

### Per-cell regression

```
cell  n    mean_hl   lin_R²
   0  126     153   -0.151
   1  207      50   +0.096
   2  222     113   -0.065
   3  205     120   -0.103
   4  169     179   -0.183
   5  180     509   -0.070
   6  224     158   -0.031
   7  178      90   -0.065
   8  153     394   -0.202
```

8 of 9 cells produce negative R² (worse than cell-mean predictor). The
one positive (cell 1, +0.096) is small-sample artifact.

## Interpretation

**Per-trade outcome is ~98% within-cell noise from a feature-prediction
perspective.** The 1.85 PF spread we identified in the intraday-axis search
is a difference of cell *means* with massive within-cell variance — real but
a tiny effect size relative to total trade-pnl variance.

Reweighting cells in HGB training won't change the fundamental
signal-to-noise ratio. It will marginally shift per-cell averages but
won't unlock new patterns the model can't already extract.

## Why earlier "spread" findings looked promising

The 1.85 PF spread across cells is a real difference in *cell-conditional
means*. But:
- A cell with PF 3.10 vs PF 1.25 doesn't mean "trades in the strong cell are
  predictably better" — it means "trades in the strong cell, on AVERAGE,
  realized higher pnl, with the same enormous within-cell variance."
- Reweighting can't move trade outcomes from the within-cell noise
  distribution; it can only emphasize/de-emphasize examples during HGB
  splitting.

## Bottleneck verdict

The HGB + scalar-feature architecture has hit a representation ceiling.
The feature signal is in the random-forest non-linearity (+0.06 R²) but
it's modest, and the cells aren't the right axis to exploit it.

## Options after NO GO

Per the plan's "NO GO" branch:

1. **H3c (sequence model / TCN).** Same features as sequences instead of
   scalars. Temporal trajectory shape might unlock pattern signal that
   scalar snapshots miss. Bigger swing, ~6-8h CPU.

2. **Accept the ceiling.** Baseline already at PF 1.881 / spread 1.30
   (close to the 1.00 target). 5 experiments since H3a haven't improved
   it. Maybe deploy and optimize Layer 2 instead.

3. **Layer 2 redesign.** L2 audit found real issues (chop×low calibration
   takes 245 trades at negative expected dollar score; 10x pick-rate
   variation across regimes). Bigger lift available there than continued
   L3 exit-policy work.

## Discipline

- Predictability test added: **scripts/research_predictability_continuous.py**
- C Round 1 (~50 min CPU) and Round 2 (~50 min CPU) NOT executed —
  saved by the C-Pre gate.
- No code changes to the v3 oracle pipeline.

## Cost

- Dev: 0 (script was already drafted)
- Compute: ~30 seconds (CV regression on 1664 trades)
- $0 GPU
- Total: ~30 seconds. Cheap NO GO is the best kind of NO GO.
