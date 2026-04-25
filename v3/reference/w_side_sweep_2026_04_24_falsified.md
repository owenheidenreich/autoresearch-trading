---
date: 2026-04-24
parent: side_prior_audit_2026_04_24.md
exp_base: spx_w_side_sweep_001
status: H1 partially falsified — pivot to class-balance rebalance
---

# w_side_contrastive Sweep — Hypothesis H1 Falsified

## Setup

Hypothesis H1 (from `side_prior_audit_2026_04_24.md`): the call bias is a
learned global side prior caused by `w_side_contrastive=0.0` in production
training. Predicted that re-enabling it at non-trivial weight should:
1. shift `frac_model_call_above_put` on truth=put bars by ≥10 pp,
2. shift 2024-04-01 model side counts toward truth's 14/77 split,
3. lift W5 OOS PF without hurting earlier windows.

Experiment: H100 retrain of seed 42 only (single-seed, all 13 windows,
hybrid_live target, same dataset and seed-42 oracle as
`spx_live_hybrid_001`) with `--w-side-contrastive ∈ {0.5, 1.0, 2.0}`.
Run-dirs: `v3/artifacts/layer2_unified_policy_spx_w_side_sweep_001_w{0p5,1p0,2p0}_seed42`.

## Results

### Aggregate PF / DD / per-window (full 13-window OOS)

```
variant    agg_pf   agg_dd%   min_pf   per-window pf max   n_chosen_trades
baseline    1.486    17.644    0.000           inf                396
w=0.5       1.547     0.000    0.000           inf                393
w=1.0       1.449     0.000    0.000           inf                410
w=2.0       1.309     6.796    0.310         2.758                401
```

(Baseline = original `spx_live_hybrid_001` seed-42 quarantined run; agg_dd
shows 0.0% on w=0.5/1.0 because no rolled-equity drawdown across windows in
this slice — flattering not significant.)

Aggregate PF does not lift meaningfully and falls at w=2.0.

### Chosen-side share on selected trades

```
variant    trades  call  put  call_share
baseline      396   375   21       0.947
w=0.5         393   357   36       0.908
w=1.0         410   370   40       0.902
w=2.0         401   353   48       0.880
```

Modest shift in put count, 21 → 48. Call share moves 94.7% → 88.0%. Direction
matches H1's prediction, but magnitude is tiny — far from balancing.

### Underlying score discrimination (side_prior_audit, full dataset)

```
variant    cohort                  n_bars  mb_call   mb_put  call-put  frac_c>p
baseline   train/truth=put          11268  -1.3840  -1.4865    0.1024     0.902
w=0.5      train/truth=put          11268  -1.3409  -1.4434    0.1026     0.917
w=1.0      train/truth=put          11268  -1.3322  -1.4329    0.1007     0.925
w=2.0      train/truth=put          11268  -1.3301  -1.4248    0.0947     0.929

baseline   oos/truth=put             2629  -1.3243  -1.4172    0.0928     0.899
w=0.5      oos/truth=put             2629  -1.3022  -1.4025    0.1002     0.906
w=1.0      oos/truth=put             2629  -1.2904  -1.3893    0.0990     0.915
w=2.0      oos/truth=put             2629  -1.2842  -1.3778    0.0936     0.923
```

**The underlying score distribution moved in the wrong direction.** On
truth=put bars (where the side-contrastive loss is supposed to push puts
above calls), `frac_call_above_put` rises monotonically from 0.902 → 0.929
on train and 0.899 → 0.923 on OOS. The mean call-put margin shrinks (0.10 →
0.09) but stays positive — the model still scores best-call higher than
best-put on ~92% of put-truth bars even at w=2.0.

### 2024-04-01 spot

```
variant    truth_call  truth_put  model_call  model_put  frac_c>p  call-put_margin
baseline           14         77          90          1     0.988          0.127
w=0.5              14         77          89          2     0.977          0.121
w=1.0              14         77          90          1     0.988          0.125
w=2.0              14         77          90          1     0.988          0.125
```

On the day where the original failure was diagnosed, every variant still
picks 89–90 calls vs 1–2 puts. The side-contrastive loss did not move the
needle here.

### W5 specifically (per_window[5])

```
variant     pf      dd%     mean$    trades
baseline   0.703   21.6%   -134       14
w=0.5      1.964    5.8%   +272        8
w=1.0      3.155    2.8%   +496        6
w=2.0      0.310    3.8%   -317        3
```

W5 PF lifts dramatically at w=0.5/1.0, but the trade count collapses (14 → 6).
Combined with the audit showing the underlying score distribution did NOT
shift, the W5 PF improvement is most likely a small-sample-trade-count
artifact: the model gates more bars out, so a few residual winners dominate
the W5 sample.

## Why the term failed

`_side_contrastive_loss` (v3/layer2/unified_policy.py:423-483) symmetrically
pools call-better and put-better bars in the denominator:

```
total = call_loss * call_better.float() + put_loss * put_better.float()
denom = torch.clamp(call_better.sum() + put_better.sum(), min=1).float()
return total.sum() / denom
```

In the train set, call-better bars outnumber put-better bars by **2.83:1**
(31,872 vs 11,268, from the audit table). The gradient is dominated by the
call-better cohort, so the loss term *reinforces* "score call above put on
the average bar" — it doesn't enforce balanced discrimination, it amplifies
the existing prior.

## Updated hypothesis (H2)

The call bias is dominated by **per-bar truth-best-side class imbalance in
the training distribution** (~67% call-best, ~31% put-best, ~2% flat-best
across the dataset). No additional same-bar margin term will rebalance this;
the gradient is structurally biased.

## Recommended next experiment

Before any more GPU spend, decide between two paths:

1. **Per-bar inverse-frequency sample weights.** Compute, per bar, weight =
   1/freq(truth_best_side) so put-best and call-best contribute equal
   gradient mass. Apply across regression, ranking, win, stopout, clean,
   dollar, return heads — and the side-contrastive term too. Lowest-risk
   intervention; one new helper in `_slice_inputs` and a per-row weight
   tensor multiplied through the loss heads.
2. **Balanced batch sampling.** Oversample put-best bars at the data-loader
   level so each batch is ~50/50. Cleaner conceptually but interacts with
   batch-norm-style statistics and per-window epoch budget; probably worse
   first move.

Recommend path 1 first — instrument as a `--side-balance-weight` toggle
(default 0.0 = current behavior, 1.0 = full inverse-frequency rebalance),
smoke on CPU, then a fresh single-seed retrain.

Verification criteria (same as before, calibrated to revised hypothesis):
- Audit `frac_model_call_above_put` on OOS truth=put cohort drops by ≥10 pp
  from baseline 0.899 (target ≤0.78). This is the load-bearing falsification
  measure.
- 2024-04-01 model side counts shift toward truth (target ≥10 puts vs
  current 1).
- W5 PF lifts AND trade count holds (≥10 trades, vs current 14 baseline).
- Earlier-window PF does not collapse below baseline by >10%.

## What is committed

- 3 variant report.json files (`v3/artifacts/layer2_unified_policy_spx_w_side_sweep_001_w*p*_seed42/seed_42/report.json`)
- 3 audit JSONs (`v3/artifacts/side_prior_audit/sweep001_w*p*_seed42_window05.json`)
- This finding note (`v3/reference/w_side_sweep_2026_04_24_falsified.md`)
- Large reproducible artifacts (model.pkl, oos_predictions.pkl, oof_predictions.pkl)
  remain gitignored per the existing pattern.
