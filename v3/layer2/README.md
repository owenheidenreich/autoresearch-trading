# v3 Layer-2

This package is the current `v3` mainline.

It does three things:

1. export one supervised row per eligible `v3` bar
2. train simple entry + side tabular models on walk-forward folds
3. replay a one-trade-per-day policy against the post-A1 teacher baseline

## Commands

Export:

```bash
.venv/bin/python -m v3.layer2.export_dataset
```

Train the current best CPU baseline:

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

Replay:

```bash
.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10
```

## Current Best Settings

- `entry_target = entry_value_rank`
- `side_target = time_stop_margin_raw`
- `direction_mode = teacher_if_triggered_else_put`
- `calibration_mode = fixed_quantiles`
- `entry_quantile = 0.60`
- `side_quantile = 0.10`
- `score_mode = product`

Interpretation:

- Entry is learned from oracle-quality ranking.
- Side magnitude is learned as a confidence / separation signal.
- Direction is structural:
  - teacher direction if a teacher is active on the chosen bar
  - otherwise default to `put`
- Fold-local threshold chasing was too noisy on 40-day validation windows.
- A fixed quantile gate is the better current bias:
  - moderately selective on entry
  - permissive on side confidence
- Within the gated set, ranking by `entry_score * side_conf` is still better than entry-only ranking.

This is the best current approximation of the evidence:

- A1 proved teacher-side structure matters.
- Pure learned-side replay is still weak.
- Late-session abstention bars are where the fallback direction matters.
- The main remaining problem was calibration variance, not missing features.

## Current Best Result

Artifact directory:
- [v3/artifacts/layer2_entry_side_fixedq_60_10](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10)

Headline:

- Layer-2 hybrid fixed-q: `287` trades, `PF 1.122`, `DD 56.2%`, `0.957` trades/day
- Post-A1 teacher baseline: `298` trades, `PF 0.961`, `DD 65.4%`, `0.993` trades/day

See:
- [audit.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10/audit.json)
- [replay_report.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10/replay_report.json)
- [layer2_entry_side_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_entry_side_2026_04_21.md)

## GPU Prep

The next GPU model should **not** resurrect the old `v2` full-chain objective.

The GPU hypothesis to port is:

- bar-state encoder
- entry head for `entry_value_rank`
- side-confidence head for `time_stop_margin_raw`
- teacher-conditioned direction policy, not unconstrained side sign
- fixed quantile calibration (`0.60 / 0.10`) unless a cleaner walk-forward replacement is proven
- `score_mode = product`

Current neural status:

- `train_neural.py` exists and reuses the same artifact layout as the tree trainer.
- `replay.py --latest-only` can validate one-fold neural smoke runs.
- A latest-fold CPU smoke slightly beat the teacher on fold 4.
- A full 5-fold CPU neural run did **not** beat either the teacher baseline or the fixed-quantile tree baseline.

So the GPU run is prepared, but the acceptance bar is now:
- first beat the post-A1 teacher baseline
- then beat the fixed-quantile tree baseline above

Suggested GPU launch command:

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
  --device cuda \
  --hidden-dim 128 \
  --depth 3 \
  --dropout 0.10 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --batch-size 4096 \
  --max-epochs 40 \
  --patience 6
```

Latest-fold smoke check:

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_neural_fixedq_smoke \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.10 \
  --score-mode product \
  --latest-only \
  --device cpu

.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_neural_fixedq_smoke \
  --latest-only
```
