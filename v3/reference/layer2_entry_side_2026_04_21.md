# Layer-2 Entry/Side Branch — 2026-04-21

Status note:
- This doc records the first profitable Layer-2 CPU branch.
- It remains the stable CPU reference.
- It is no longer the frontier max-PF artifact; see [layer2_shared_encoder_diagnostics_2026_04_21.md](layer2_shared_encoder_diagnostics_2026_04_21.md), [layer2_diagnostic_full_2026_04_21.md](layer2_diagnostic_full_2026_04_21.md), and [layer2_route_aware_fallback_2026_04_21.md](layer2_route_aware_fallback_2026_04_21.md) for the later neural follow-up and its qualification results.

## Summary

`v3` now has a working Layer-2 export/train/replay path.

Initial CPU variants were tested:

1. **Pure learned-side, oracle-best targets**
   - `entry_target = entry_value_rank`
   - `side_target = side_margin_raw`
   - replay used learned side sign
   - Result: `PF 0.916`, `DD 58.3%`, `0.570` trades/day

2. **Fully time-stop-aligned targets**
   - `entry_target = time_stop_value_rank`
   - `side_target = time_stop_margin_raw`
   - replay used learned side sign
   - Result got worse; entry target absorbed exit noise

3. **Hybrid policy (first profitable branch)**
   - `entry_target = entry_value_rank`
   - `side_target = time_stop_margin_raw`
   - `direction_mode = teacher_if_triggered_else_put`
   - Result: `PF 0.986`, `DD 57.8%`, `0.623` trades/day

Then one more refinement changed the picture:

4. **Hybrid policy + fixed quantile calibration (current best)**
   - same targets and direction rule as above
   - `calibration_mode = fixed_quantiles`
   - `entry_quantile = 0.60`
   - `side_quantile = 0.10`
   - `score_mode = product`
   - Result: `PF 1.122`, `DD 56.2%`, `0.957` trades/day

Post-A1 teacher baseline:
- `PF 0.961`
- `DD 65.4%`
- `0.993` trades/day

The fixed-quantile hybrid branch is the current promoted CPU baseline. It beats the post-A1 teacher baseline on PF, drawdown, and coverage while staying close to the teacher trade count.

## What Was Implemented

New package:
- [v3/layer2/export_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/export_dataset.py)
- [v3/layer2/train_entry_side.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_entry_side.py)
- [v3/layer2/replay.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/replay.py)
- [v3/layer2/common.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/common.py)

Export bundle:
- one row per eligible bar
- joins the `W2a` regime block from `v2.2`
- includes:
  - bar context
  - teacher outputs
  - surface summary
  - oracle-best forward labels
  - directional time-stop labels on the mechanically selected contract for each side
  - fold ID for the 5 canonical walk-forward folds

Current export stats:
- `89,692` rows
- `986` days
- `42` model features

## What We Learned

### 1. Entry and side should not be treated the same way

The entry model is learning useful signal.

Evidence:
- top-decile entry-score bars beat random eligible bars on oracle-quality value
- top-1/day chosen bars beat teacher-triggered bars on oracle-quality value

The side model is not yet reliable as a free-form direction chooser.

Evidence:
- pure learned-side replay stayed below baseline PF
- even when side target was aligned to time-stop economics, raw learned sign remained weak

### 2. Full time-stop alignment on the entry target is too noisy

Switching `entry_target` from `entry_value_rank` to `time_stop_value_rank` made the branch worse.

Interpretation:
- time-stop exit noise should not define the bar-quality target
- entry should still learn “where the move is”
- realized exit discipline should be handled separately

### 3. Teacher-conditioned direction is the right current bridge

The best current rule is:
- apply moderate fixed entry gating and permissive fixed side gating
- choose the bar with the highest `entry_score * side_conf`
- if a teacher is active on the chosen bar, use the teacher direction
- otherwise default the abstention-style pick to `put`

This means the side model is currently more valuable as a **confidence / separability filter** than as a raw sign head, but it still helps rank bars inside the gated set.

### 4. Calibration stability was the missing ingredient

The first profitable hybrid branch was still too fragile:
- only `187` trades
- PF below `1.0`
- strong sensitivity to the fold-local validation argmax

The next question was whether the score surface was real but the calibration was too noisy.

That answer now looks like **yes**.

Evidence:
- the fixed-quantile branch improved mean selected time-stop value from `-7.76` to `+68.80`
- replay improved from `PF 0.986` to `PF 1.122`
- drawdown improved from `57.8%` to `56.2%`
- coverage moved from `0.623` to `0.957` trades/day, much closer to the teacher baseline

Interpretation:
- the entry/side model pair was already useful
- the 40-day fold-local threshold argmax was too noisy
- stable quantile gates are a better current bias than per-fold threshold chasing

## Current Best Run

Command:

```bash
.venv/bin/python -m v3.layer2.train_entry_side \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10 \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.10 \
  --score-mode product

.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10
```

Artifacts:
- [audit.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10/audit.json)
- [replay_report.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10/replay_report.json)

Headline metrics:

| system | trades | PF | DD | trades/day |
|---|---:|---:|---:|---:|
| Layer-2 hybrid fixed-q | 287 | **1.122** | **56.2%** | **0.957** |
| Post-A1 teacher baseline | 298 | 0.961 | 65.4% | 0.993 |

Slice detail:
- `abstention`: Layer-2 found `10` trades with mean PnL `+$1,553.40`; teacher baseline has no trades there by definition
- `side_error`: `3` Layer-2 trades landed in this slice with mean PnL `-$1,008.85`; side-error recovery is still not solved yet

## Current Hypothesis

The most grounded interpretation is:

- entry signal exists and is learnable at the bar-state level
- free-form learned side sign is still too noisy
- teacher structure plus a bearish abstention prior is more robust than unconstrained side prediction
- product ranking works better than entry-only ranking once the side gate is permissive
- calibration stability matters more right now than extra feature churn

So the next GPU run should **port the fixed-quantile hybrid policy shape**, not the old `v2` architecture:

- entry head learns `entry_value_rank`
- side head learns confidence / separation from `time_stop_margin_raw`
- direction policy is teacher-conditioned
- thresholding starts from the fixed quantile prior (`0.60 / 0.10`)
- within-day choice uses `entry_score * side_conf`

## Neural GPU Prep

Two neural checks were run after the tree calibration fix:

1. **Latest-fold CPU smoke**
   - command used the same targets and fixed quantile calibration
   - result on fold 4 only: `PF 0.808` vs teacher `0.773`
   - useful as a smoke and compatibility check, not promotion evidence

2. **Full 5-fold CPU neural run**
   - same policy shape, same fixed quantile calibration
   - result: `PF 0.899`, `DD 196.3%`, `0.993` trades/day
   - this failed against both the post-A1 teacher and the fixed-quantile tree baseline

Interpretation:
- the neural path is now implemented and runnable
- the fixed-quantile hybrid policy is the right thing to port
- but the current torch MLP is not yet an upgrade over the tree model
- GPU should be treated as a research run, not as a presumed promotion

## What Not To Do Next

- Do not go back to new late-session teachers. That branch is already falsified.
- Do not reopen the `v2` full-chain transformer as the mainline.
- Do not switch the entry target fully to time-stop rank again without a new reason.

## Next Step

Prepare the first GPU branch as a neural version of this exact fixed-quantile hybrid policy.

Success criterion:
- it must beat the CPU fixed-quantile baseline above, not just the post-A1 teacher baseline.
