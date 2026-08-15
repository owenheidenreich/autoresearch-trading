# Candidate-Trained Layer-3 — First Pass, 2026-04-23

## Hypothesis

The simulated-L3 outer loop has been repeatedly training and simulating
exits from the chosen-trade distribution. The seed-42 W6 diagnostic showed
iteration-2's damage came from confidently scored extra days that the
chosen-trained Layer-3 could not exit profitably.

If that is a candidate-distribution mismatch, then Layer-3 should train on
a broader candidate-trade distribution from the action surface, while still
replaying a fixed entry policy honestly on held-out windows.

## Code changes

Added candidate-surface Layer-3 training support:

- [v3/layer3/common.py](/Users/gduby/Documents/autoresearch-trading/v3/layer3/common.py)
  - `load_action_surface_candidate_trades(...)`
  - side/time-stratified deterministic candidate sampler
  - optional `test_trade_data` in `train_models_by_window(...)`
- [v3/layer3/train_rolling.py](/Users/gduby/Documents/autoresearch-trading/v3/layer3/train_rolling.py)
  - `--l3-training-source entry_policy|candidate_surface`
  - `--candidate-dataset`
  - `--candidate-train-max-per-day`
- [v3/layer2/build_simulated_l3_oracle.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/build_simulated_l3_oracle.py)
  - same `--l3-training-source candidate_surface` hook for future oracle builds

Default candidate training sample:

- `max_per_day=4`
- `3,092` candidate trades
- `775` days
- call share `50.5%`
- put share `49.5%`
- sampled from current contract surface fields only, not future PnL

## Commands

Iteration-2 entries, candidate-trained Layer-3:

```bash
for s in 42 43 44; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.train_rolling \
    --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_simL3_iter2_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_seed${s}_candidate_l3_mpd4 \
    --seed $s \
    --l3-training-source candidate_surface \
    --candidate-train-max-per-day 4

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_seed${s}_candidate_l3_mpd4 \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done
```

Oracle-builder smoke:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m v3.layer2.build_simulated_l3_oracle \
  --dataset v3/artifacts/layer2_action_surface_dataset.pkl \
  --l3-training-source candidate_surface \
  --candidate-train-max-per-day 1 \
  --champion-calibration v3/artifacts/layer3_unified_cpu_simL3_champion_seed42_candidate_l3_mpd4/rolling_layer3_calibrated_robust_90.json \
  --output v3/artifacts/simulated_l3_oracle_candidate_l3_smoke.npz \
  --max-days 210 \
  --min-train-trades 1000 \
  --progress-every-days 50
```

Smoke result: `19,108` rows, `33,782` candidate simulations, `0` skips.

## Results

Iteration-2 entries composed with candidate-trained Layer-3 robust `0.90`:

| seed | chosen-trained L3 PF | candidate-trained L3 PF | candidate DD | trades |
|---|---:|---:|---:|---:|
| 42 | 1.733 | 1.746 | 12.2% | 302 |
| 43 | 2.384 | 2.650 | 7.2% | 334 |
| 44 | 1.932 | 1.691 | 7.3% | 294 |
| mean | 2.016 | **2.029** | **8.9%** | 930 |
| min | **1.733** | 1.691 | — | — |

Aggregate PF across all three candidate-trained runs: `1.993`.
Mean trade PnL: `+$181.1`.

Seed-42 W6 specifically:

- chosen-trained iteration-2 L3: `-$1.26k`
- candidate-trained iteration-2 L3: `+$3.19k`

This confirms the W6 diagnosis: candidate-trained exits can rescue the
extra-day distribution that chosen-trained L3 mishandled.

## Control: promoted entries

Candidate-trained L3 is not universally better on already selective
entries. On the current promoted seed-42 entry set:

- original promoted seed-42 L3: `PF 1.786`, DD `11.7%`
- candidate-trained L3: `PF 1.725`, DD `9.0%`

So this is not a drop-in exit replacement for the current champion. It is
most useful when the entry distribution expands, as iteration-2 did.

## Interpretation

Candidate-trained L3 is a real structural improvement, but not a clean
promotion yet:

- it fixes the known W6 failure mode
- it improves mean PF slightly on iteration-2 entries
- it materially reduces DD
- it loses the min-PF floor because seed 44 regresses

Current repo truth:

- current champion remains `simulated-L3 per-seed oracle + objective-consistent unified entry + chosen-trained Layer-3 robust 0.90`
- candidate-trained L3 is now the best-supported next outer-loop direction
- the next full branch should build a simulated-L3 oracle using candidate-trained L3 models, retrain unified entry against that oracle, then compose with candidate-trained Layer-3
