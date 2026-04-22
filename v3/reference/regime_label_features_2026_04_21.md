# Phase 3A — Regime Label + Per-Day Features — 2026-04-21

## TL;DR

**Dataset assembled, but severe class imbalance is a red flag for
Phase 3B.** 281 in-sample + 20 OOS day-rows. Label = 1 (V0-favorable)
on only 11% of in-sample days. Per-fold rate varies 1.7% to 26.7%.
Folds 2 and 3 have only 1 V0-favorable day each.

## Setup

Script: [v3/analysis/regime_label_features.py](../analysis/regime_label_features.py).
Artifacts:
- [v3/artifacts/regime_labels/regime_labels.csv](../artifacts/regime_labels/regime_labels.csv)
- [v3/artifacts/regime_labels/regime_labels_summary.json](../artifacts/regime_labels/regime_labels_summary.json)

Label: L1 with smoothing — `1 if V0_pnl > V1_pnl + $50 else 0`.
The $50 buffer prevents single-trade noise from flipping near-equal days.

Per-day features (extracted at bar 14 = end of first 15 minutes):
opening_gap_pct, session_open_dist, vwap_dist, first15_range_pct,
first15_close_position, vix_roc, atm_iv, iv_percentile, realized_vol,
vix_regime, sigma_pos, omar_range_pct, omar_mid_pos_units. 13 total.

## Distribution

| Slice | Days | V0-favorable | Rate |
|---|---:|---:|---:|
| In-sample total | 281 | 31 | 11.0% |
| In-sample, V0 chose to trade | 275 | 27 | 9.8% |
| OOS total | 20 | 3 | 15.0% |

**Per-fold (in-sample):**

| Fold | Days | V0 positives | Rate |
|---:|---:|---:|---:|
| 0 | 55 | 3 | 5.5% |
| 1 | 40 | 6 | 15.0% |
| **2** | 60 | **1** | **1.7%** |
| **3** | 60 | **1** | **1.7%** |
| 4 | 60 | 16 | 26.7% |

## Why this is concerning for Phase 3B

Phase 3B (classifier training with walk-forward) requires sufficient
positive examples in the prior folds to learn a decision boundary:

- Fold 0: fallback (no training). 3 positives in 55 days.
- Fold 1: trains on fold 0 only — 3 positives → likely overfits or guesses
- Fold 2: trains on folds 0+1 — 9 positives in 95 days → still very sparse
- Fold 3: trains on folds 0+1+2 — 10 positives in 155 days → marginal
- Fold 4: trains on folds 0+1+2+3 — 11 positives in 215 days → best chance

The best-case scenario is fold 4's classifier. Folds 1-3 will likely
produce near-random AUC.

**Plan's Phase 3B kill gate:** AUC < 0.55 → abandon Phase 3.

This isn't impossible to clear, but it requires:
1. The features (gap, vix, IV, etc.) to actually carry signal about
   "today is a directional regime"
2. The model to extract that signal from <10 positive examples per
   training fold

Both are possible but not likely.

## What this DOES tell us

1. **The "V0 vs V1" choice is NOT a 50/50 problem.** It's overwhelmingly
   V1-favorable on this sample (89% of days). V1 (always_put) was the
   right baseline.
2. **Fold 4 is the directional-up regime.** 26.7% of days are
   V0-favorable, the highest. This matches the Stage C1 finding that
   V1 sacrifices fold-4 in-sample PF (1.801 → 1.126) by removing call
   alpha.
3. **Folds 2 and 3 are nearly impossible to learn from.** With 1
   positive example each, no classifier can reliably learn the
   regime signature.

## What this does NOT tell us

1. **Whether the features carry meaningful signal.** Phase 3B will
   answer this empirically.
2. **Whether an alternative label definition** (e.g., L2: SPX session
   direction up > +0.5%) would be more balanced and learnable.
3. **Whether label smoothing threshold** ($50 vs $100 vs $200)
   meaningfully changes the signal.

## Recommendation

Proceed to Phase 3B (regime classifier training + walk-forward). The
plan's kill gate (AUC < 0.55) is appropriate for this label balance
— if the classifier can't beat random, we abandon Phase 3 cleanly
and proceed to Phase 4.

Cost: ~3-4 hours CPU per the plan. Cheap relative to the value of
either confirming or killing Phase 3.

## Verification

- [x] `python -m py_compile v3/analysis/regime_label_features.py` passes
- [x] 301 day-rows assembled (281 in-sample + 20 OOS)
- [x] 13 per-day features extracted at bar 14
- [x] Label distribution reported per fold
- [x] No missing values in feature columns
- [ ] Commit (deferred to Phase 3 wrap)
