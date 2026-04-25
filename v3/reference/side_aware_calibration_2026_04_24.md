---
date: 2026-04-24
parent: w5_day_diagnostic_2026_04_24.md
status: side-aware calibration alone insufficient; pivot to per-bar inverse-frequency sample weights
---

# Side-Aware Calibration on W5 — Insufficient

## Setup

After the W5 day-level diagnostic confirmed real put opportunities exist
(2,453 of 5,349 W5 OOS bars favor puts by ≥$50; counterfactual swap on 9
of 14 chosen days lifts W5 from −$1,480 to +$5,585), the next-plan called
for trying per-side calibration first as the cheapest valid intervention.

Hypothesis: per-side calibrated thresholds (decision_margin, min_win_prob,
max_stopout_prob) on the put slot might let put trades through that the
global threshold filters out.

Script: `v3/analysis/side_aware_calibration.py`. Postprocessing only — no
retrain. For each row computes (best_call_score/win/stopout/label) and
(best_put_score/win/stopout/label) at the model's argmax-per-side action.
Calibrates each side independently against val data (40 days), then
applies side-aware selection on OOS (60 days).

## Result

```
=== W5 comparison ===
variant                    trades  calls  puts   pf_label  mean$/trade     total$
production (orig calib)        14     13     1      0.703      -133.80   -1873.20
global recalibrated            26     25     1      0.970       -11.77    -306.01
side-aware (strict)            13     13     0      0.774       -99.41   -1292.37
side-aware (permissive)        42     13    29      0.758       -88.84   -3731.47
```

Two side-aware variants:

- **Strict** (fail-out if no pf-qualified config exists): the put-side
  calibration grid found NOTHING profitable on val data. Best
  unqualified put-side config had val PF 0.683 (losing). With fail-out,
  the put gate rejects everything; selection collapses to call-only.
  Result: 13 calls / 0 puts, PF 0.774. No improvement.
- **Permissive** (accept the best of bad configs): adds 29 put trades.
  All collectively losing — net delta vs production: −$1,858 from 29
  added puts. PF actually drops below permissive call-only.

In either variant, **per-side calibration cannot extract put winners
from this model.** Why:

1. The val set has 3,578 put-side rows. The grid sweep over (margin ∈
   {0.0..0.35}, min_win ∈ {0..0.5}, max_stopout ∈ {0.45..1.0}) found no
   threshold producing a put-only val PF ≥ 1.0.
2. This means the model's put scoring is non-discriminative on val: the
   bars the model scores highly as puts are not systematically
   profitable. The discrimination signal isn't there to be extracted.
3. Confirms the side_prior_audit's earlier reading: put scores are
   systematically lower than call scores regardless of truth side, and
   the per-cohort `frac_call_above_put` barely moves between truth=call
   and truth=put cohorts (4.8 pp on train, 0.2 pp on OOS).

## Decision

**Path 4 is the remaining live hypothesis: per-bar inverse-frequency
sample weights with a single-seed retrain.** The W5 diagnostic shows put
winners exist in the labels; the side-aware calibration shows the
trained model can't surface them with any threshold. The remaining
intervention is to fix the gradient asymmetry at training time so the
model learns put-discriminative features in the first place.

## Outputs

- `v3/analysis/side_aware_calibration.py` — postprocessing harness with
  strict fail-out by default.
- `v3/artifacts/side_aware_calibration/seed42_w5.json` — full
  calibration result + per-trade rows.
