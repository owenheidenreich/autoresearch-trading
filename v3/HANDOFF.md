# v3 Handoff — Post-A1, Layer-2 Mainline

**Date:** 2026-04-21  
**Project framing:** SPX 0DTE long premium, $25k account, 1 contract, Moderate-tier Layer 0 rails

## Purpose

This is the current handoff for `v3` after:
- Stage 1 research completed
- A1 ORC sigma gate shipped
- late-session teacher search falsified
- W2a soft features shipped into `v2.2`
- first `v3` Layer-2 supervised branch implemented and screened

The mainline is no longer “write more teachers” or “keep tuning `v2/train.py`.”
The mainline is now the `v3` Layer-2 bar-state branch in [v3/layer2](/Users/gduby/Documents/autoresearch-trading/v3/layer2).

## TL;DR

What is actually true now:

- **A1 is real and shipped.** ORC with the hard `sigma_pos` direction veto reduced `side_error 402 → 176` with exact dry-run/runtime agreement. See [orc_direction_fix_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/orc_direction_fix_2026_04_20.md).
- **Late-session teacher families did not survive.** The tournament went `0/8`; that regime is Layer-2-only. See [late_session_tournament_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/late_session_tournament_2026_04_20.md).
- **W2a features shipped, but `v2` still failed.** `exp_179` confirmed the old training loop is the bottleneck, not the feature surface. See [w2a_handoff_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/w2a_handoff_2026_04_21.md).
- **The new mainline is `v3` Layer-2.** Export/train/replay tooling now exists under [v3/layer2](/Users/gduby/Documents/autoresearch-trading/v3/layer2).
- **The best current Layer-2 policy is hybrid and calibration-stable, not pure learned-side.**
  - Entry model: learn `entry_value_rank`
  - Side model: learn `time_stop_margin_raw` as a confidence signal
  - Live direction rule: `teacher_if_triggered_else_put`
  - Calibration rule: fixed quantiles, `entry=0.60`, `side=0.10`
  - Within-day choice rule: `entry_score * side_conf`
  - Result: `287` trades, `PF 1.122`, `DD 56.2%`, `0.957` trades/day
  - Post-A1 teacher baseline: `298` trades, `PF 0.961`, `DD 65.4%`, `0.993` trades/day

Reference: [layer2_entry_side_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_entry_side_2026_04_21.md)

## What Changed

### Stage 1 / teacher layer

- ORC now carries the shipped `sigma_pos` hard veto.
- FailedBreak remains the only other baseline teacher.
- No new late-session teacher should be added on the current evidence.

### Feature / denominator layer

- `v2.core.market_structure` is the shared source of truth for:
  - `sigma_pos`
  - OMAR fields
  - last-10 structure
- The 8 W2a regime features are stable in `v2.2` and are now joined into `v3` Layer-2 export.

### Layer-2 tooling

New package:
- [v3/layer2/export_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/export_dataset.py)
- [v3/layer2/train_entry_side.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_entry_side.py)
- [v3/layer2/replay.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/replay.py)
- [v3/layer2/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer2/README.md)

What they do:
- export one row per eligible bar after both oracle passes
- join the W2a regime block from `v2.2`
- train simple tabular entry + side models
- calibrate thresholds on val folds only, now with explicit fixed-quantile support
- replay a one-trade-per-day policy against the post-A1 teacher baseline

New neural path:
- [v3/layer2/neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/neural.py)
- [v3/layer2/train_neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_neural.py)

What it does:
- keeps the same export bundle, replay interface, and artifact layout
- swaps the tree regressors for small torch MLP regressors
- supports `--latest-only` smoke tests and `--device cuda` for GPU training

## Current Best Hypothesis

The surviving signal is **bar-quality first, structural direction second**.

More concretely:

- The entry model is learning something real. Top-decile and top-1/day bar selection beat random eligible bars and beat teacher-triggered bars on oracle-quality metrics.
- The learned side sign is still too noisy to trust directly.
- The side model is more useful as a **confidence gate** than as a free-form direction chooser.
- The earlier fold-local threshold search was too noisy on 40-day validation windows.
- The best current policy is:
  - use fixed quantile gates with moderate entry selectivity and permissive side gating
  - choose the bar with the highest `entry_score * side_conf`
  - if a teacher already has a direction on that bar, trust the teacher direction
  - otherwise default the abstention-style pick to `put`

This means the next GPU run should **not** copy the old `v2` “global gate + global side sign + contract scorer” setup. It should port the fixed-quantile hybrid policy shape instead.

## Commands

Dataset export:

```bash
.venv/bin/python -m v3.layer2.export_dataset
```

Current best train run:

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
```

Replay against post-A1 teacher baseline:

```bash
.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10
```

Neural GPU-prep command:

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_neural_gpu_fixedq \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.10 \
  --score-mode product \
  --device cuda
```

## Immediate Next Work

### 1. Lock the fixed-quantile hybrid Layer-2 result as the new CPU baseline

- Treat `teacher_if_triggered_else_put` plus fixed quantile calibration as the current best policy shape.
- Do not regress to pure learned-side replay without a specific reason.

### 2. Prepare the GPU branch around the right problem

The GPU hypothesis should be:

- entry head learns `entry_value_rank`
- side head learns **confidence / separation**, not free-form sign across all bars
- direction logic is teacher-conditioned:
  - teacher side if present
  - abstention fallback handled separately
- thresholding is stabilized with fixed quantiles unless a cleaner walk-forward calibration beats it

That can be implemented as a small neural two-head model, but the target should be this policy shape, not the old `v2` full-chain objective.
Current warning: the full 5-fold CPU neural run still underperformed both the fixed-quantile tree baseline and the post-A1 teacher baseline, so GPU is prepared as an experiment, not a promoted replacement.

### 3. Keep exit modeling deferred

- Current replay still uses time-stop.
- Exit modeling remains a later Layer-2/3 addition after entry+direction are more stable.
- Do not let fixed `-35/+60` stop-target logic back into Layer 0.

## Files To Read First

1. [v3/reference/orc_direction_fix_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/orc_direction_fix_2026_04_20.md)
2. [v3/reference/late_session_tournament_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/late_session_tournament_2026_04_20.md)
3. [v3/reference/w2a_handoff_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/w2a_handoff_2026_04_21.md)
4. [v3/reference/layer2_entry_side_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_entry_side_2026_04_21.md)
5. [v3/layer2/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer2/README.md)

## Known Gotchas

- `v3/artifacts/layer2_dataset.pkl` is a derived artifact, not source of truth. Rebuild it if the logger/oracle/export logic changes.
- The current positive Layer-2 result is still **hybrid**. The learned side model alone has not beaten the teacher baseline.
- The key improvement after the initial hybrid branch was **stable calibration**, not a new feature set or a more complex model.
- `replay.py --latest-only` is now available for one-fold smoke validation of neural runs.
- `v2` remains useful for sidecars and shared market-structure features, but it is no longer the mainline training loop.
