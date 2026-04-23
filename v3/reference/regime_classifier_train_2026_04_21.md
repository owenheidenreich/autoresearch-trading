# Phase 3B — Morning-Snapshot Regime Classifier — ABANDON — 2026-04-21

## TL;DR

**ABANDON per plan gate.** The "is today V0-favorable" question is
NOT learnable from 13 morning-snapshot features (extracted at bar 14
= end of first 15 minutes):

- Aggregated walk-forward AUC: **0.536** (< 0.55 gate)
- OOS AUC: **0.196** (worse than random — anti-signal)
- Per-fold AUCs wildly unstable: 0.627, 0.475, 0.068, 0.455
- Mean per-fold AUC: 0.406
- Permutation importance mostly NEGATIVE (model fitting noise)
- OOS recall on the 3 V0-favorable days: 0/3 (0%)

## Setup

Script: [v3/analysis/regime_classifier_train.py](../analysis/regime_classifier_train.py).
Artifact: [v3/artifacts/regime_classifier_train/regime_classifier_train.json](../artifacts/regime_classifier_train/regime_classifier_train.json).

- 301 day-rows (281 in-sample + 20 OOS)
- Label: `V0_pnl > V1_pnl + $50` → binary
- 13 features extracted at bar 14: opening_gap_pct, session_open_dist,
  vwap_dist, first15_range_pct, first15_close_position, vix_roc,
  atm_iv, iv_percentile, realized_vol, vix_regime, sigma_pos,
  omar_range_pct, omar_mid_pos_units
- Walk-forward: fold k trains on days < k, tests on fold k
- HistGradientBoostingClassifier with class weights

## Per-fold table

| Fold | Train n | Train pos | Test n | Test pos | AUC | AP | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 59 | 6 | fallback (no training) | | |
| 1 | 59 | 6 | 42 | 7 | 0.627 | 0.223 | 0.643 |
| 2 | 101 | 13 | 60 | 1 | 0.475 | 0.031 | 0.883 |
| 3 | 161 | 14 | 60 | 1 | 0.068 | 0.018 | 0.933 |
| 4 | 221 | 15 | 60 | 16 | 0.455 | 0.263 | 0.733 |

Fold 3's AUC of 0.068 is particularly damning — it's ANTI-predictive.
Fold 2 with 1 positive example is effectively unlearnable.

## OOS evaluation

Using the fold-4 model (trained on all folds 0-3, 221 train days, 15
positives):

| Metric | Value |
|---|---:|
| OOS AUC | 0.196 (worse than random) |
| Average Precision | 0.123 |
| Accuracy | 0.800 (spurious — driven by 85% V1-favorable base rate) |
| Precision | 0.000 |
| Recall | 0.000 |
| Confusion matrix | [[TN=16, FP=1], [FN=3, TP=0]] |

The model flagged one OOS day as V0-favorable (March 27) — that day
was actually V1-favorable. It missed all 3 days where V0 actually
won (March 9, 10, 31).

Per-day OOS scores:

```
2026-03-05: label=0 pred=0.031 ✓
2026-03-06: label=0 pred=0.274 ✓
2026-03-09: label=1 pred=0.008 ✗ (big FN — model gave 0.008 to an actual positive)
2026-03-10: label=1 pred=0.003 ✗ (bigger FN)
2026-03-11: label=0 pred=0.004 ✓
...
2026-03-27: label=0 pred=0.765 ✗ (big FP — high confidence for a negative)
2026-03-31: label=1 pred=0.017 ✗ (FN)
```

Permutation importance shows NEGATIVE AUC drops on most features,
meaning shuffling improves predictions — the model is anti-fitting.

## Why this fails

1. **Severe class imbalance.** 89% V1-favorable globally; folds 2-3
   have only 1 positive each. Nothing to learn from.
2. **Morning-snapshot features are too early.** Bar 14 is 9:45am. A
   "directional up" day's signature emerges later in the session as
   trend develops; it's often not visible at 9:45.
3. **13 features may be too few or the wrong ones.** Opening gap,
   VIX, and IV don't carry enough directional signal about how the
   session will unfold.

## What this says about the broader question

The FAILURE of this specific approach (morning snapshots + 13 features)
does NOT prove "V0's directional signal can never be recovered". It
proves only that bar-14 snapshots don't predict V0-favorable days.

The methodology overhaul (plan R1-R6, dated 2026-04-22) attempts the
same question with:
- Rolling-window evaluation across 13 disjoint 60-day OOS windows (780
  OOS days vs 20) — stronger evidence base
- Features at the CHOSEN ENTRY BAR (not bar 14) — closer to the
  decision point
- Intraday-DEVELOPING features (vwap slope to entry, cumulative delta
  pressure, path drift) — capture session development, not early-open
  snapshots

The morning-snapshot approach is retired. The entry-bar + rolling-window
approach replaces it.

## Verification

- [x] Aggregated AUC 0.536 < 0.55 → ABANDON per plan gate
- [x] OOS AUC 0.196 confirms anti-signal
- [x] Per-fold table reported
- [x] Permutation importance reported (mostly negative)
- [x] Honest caveat: failure is method-specific, not strategy-terminal
- [ ] Commit
