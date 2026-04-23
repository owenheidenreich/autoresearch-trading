# Simulated-L3 Iteration-2 Margin-Offset Rescue — Falsified

## Question

Can the iteration-2 simulated-L3 branch recover its PF floor with a
simple stricter entry gate, without another retrain?

The candidate mechanism was:

- iteration-2 improved the mean PF but overtraded in a few bad windows
- a global positive offset on top of each window's calibrated
  `decision_margin` might cut the marginal trades while preserving most
  of the seed-43 mean lift

## Infrastructure

Added:

- [v3/layer2/reselect_unified_policy_trades.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/reselect_unified_policy_trades.py)

This helper:

- loads saved unified-policy `oof_predictions.pkl`
- loads each window's existing `calibration.json`
- applies a global additive margin offset
- reselects one trade per day from the saved predictions
- writes a new `chosen_trades.pkl` plus a small report

## Commands

Reselect iteration-2 entries with `+0.05`:

```bash
.venv/bin/python -m v3.layer2.reselect_unified_policy_trades \
  --seed-dir v3/artifacts/layer2_unified_policy_simL3_iter2_seed42/seed_42 \
  --output-dir v3/artifacts/layer2_unified_policy_simL3_iter2_margin005_seed42/seed_42 \
  --decision-margin-offset 0.05

.venv/bin/python -m v3.layer2.reselect_unified_policy_trades \
  --seed-dir v3/artifacts/layer2_unified_policy_simL3_iter2_seed43/seed_43 \
  --output-dir v3/artifacts/layer2_unified_policy_simL3_iter2_margin005_seed43/seed_43 \
  --decision-margin-offset 0.05

.venv/bin/python -m v3.layer2.reselect_unified_policy_trades \
  --seed-dir v3/artifacts/layer2_unified_policy_simL3_iter2_seed44/seed_44 \
  --output-dir v3/artifacts/layer2_unified_policy_simL3_iter2_margin005_seed44/seed_44 \
  --decision-margin-offset 0.05
```

Compose those entry sets through Layer-3:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/layer2_unified_policy_simL3_iter2_margin005_seed42/seed_42/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_margin005_seed42 \
  --seed 42

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/layer2_unified_policy_simL3_iter2_margin005_seed43/seed_43/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_margin005_seed43 \
  --seed 43

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/layer2_unified_policy_simL3_iter2_margin005_seed44/seed_44/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_margin005_seed44 \
  --seed 44
```

Calibrate honestly:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.calibrate_threshold \
  --run-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_margin005_seed42 \
  --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.calibrate_threshold \
  --run-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_margin005_seed43 \
  --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.calibrate_threshold \
  --run-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_margin005_seed44 \
  --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
```

## Results

Entry reselect alone:

- seed `42`: trades `302 -> 220`
- seed `43`: trades `334 -> 262`
- seed `44`: trades `294 -> 205`

Honest composed PF after Layer-3 robust `0.90`:

- seed `42`: PF `1.187`, DD `28.6%`, `220` trades
- seed `43`: PF `2.049`, DD `24.3%`, `262` trades
- seed `44`: PF `1.879`, DD `9.7%`, `205` trades

Cross-seed summary:

- mean PF: `1.705`
- min PF: `1.187`
- mean DD: `20.9%`
- total trades: `687`
- aggregated PF: `1.689`

Comparison to the current promoted champion:

- promoted champion: mean PF `1.950`, min PF `1.786`, mean DD `10.1%`
- iteration-2 challenger: mean PF `2.016`, min PF `1.733`, mean DD `11.8%`
- margin-offset rescue: mean PF `1.705`, min PF `1.187`, mean DD `20.9%`

## Diagnosis

This is not a clean rescue.

The stricter gate helped seeds `43` and `44`, but seed `42` still failed
badly. The main residual failure stayed concentrated in window `6`:

- seed `42`, window `6`: `29` trades, PnL `-$5.7k`, calibrated PF still
  below `1.0`

So the iteration-2 mean-vs-floor tradeoff is **not** caused by a simple
"too many marginal trades everywhere" problem.

## Conclusion

Global positive `decision_margin` offsets are **falsified** as a rescue
path for the iteration-2 simulated-L3 branch.

What this changes:

- do not spend more loops on broad entry-gate tightening
- keep the per-seed simulated-L3 composed stack as the provisional champion
- treat the next branch as structural, not scalar:
  - either diagnose why window `6` diverges so sharply across seeds
  - or build a more faithful composed target such as candidate-trained L3
