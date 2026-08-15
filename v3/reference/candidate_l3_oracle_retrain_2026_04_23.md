# Candidate-Trained-L3 Oracle Retrain — 2026-04-23

## Hypothesis

Candidate-trained Layer-3 fixed the known seed-42 W6 failure when replayed
on iteration-2 entries, but it was not a drop-in exit replacement for the
current promoted stack. The correct outer-loop test is therefore:

1. build a simulated-L3 oracle whose exit models are trained on the broader
   candidate-surface distribution
2. retrain unified entries against that oracle
3. compose those entries through candidate-trained Layer-3
4. compare honestly against the current promoted champion

## Commands

Built full candidate-trained-L3 oracle sidecars:

```bash
for s in 42 43 44; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer2.build_simulated_l3_oracle \
    --dataset v3/artifacts/layer2_action_surface_dataset.pkl \
    --l3-training-source candidate_surface \
    --candidate-train-max-per-day 4 \
    --champion-calibration v3/artifacts/layer3_unified_cpu_simL3_iter2_seed${s}_candidate_l3_mpd4/rolling_layer3_calibrated_robust_90.json \
    --output v3/artifacts/simulated_l3_oracle_seed${s}_candidate_l3_mpd4_fp.npz \
    --seed $s \
    --progress-every-days 25
done
```

Each oracle completed cleanly:

- rows/actions: `89,692 x 25`
- candidate simulations: `780,847`
- skipped simulations: `0`

Retrained unified entries:

```bash
for s in 42 43 44; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer2.train_unified_policy \
    --tier dev --device cpu --seed $s \
    --run-dir v3/artifacts/layer2_unified_policy_simL3_candidateL3_seed$s \
    --utility-target simulated_l3 \
    --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_seed${s}_candidate_l3_mpd4_fp.npz
done
```

Composed with candidate-trained Layer-3:

```bash
for s in 42 43 44; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.train_rolling \
    --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_simL3_candidateL3_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_simL3_candidateL3_seed$s \
    --seed $s \
    --l3-training-source candidate_surface \
    --candidate-train-max-per-day 4

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_simL3_candidateL3_seed$s \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done
```

## Entry behavior

The retrained entry policy is less call-concentrated than prior simulated-L3
branches:

| seed | trades | call share | time-stop PF | time-stop DD |
|---|---:|---:|---:|---:|
| 42 | 294 | 68.7% | 1.235 | 41.0% |
| 43 | 348 | 79.9% | 0.881 | 85.5% |
| 44 | 340 | 74.1% | 0.954 | 69.0% |

This is a real behavioral shift away from the old ~93-99% call prior.

## Composed Result

Candidate-trained-L3 oracle retrain + candidate-trained Layer-3 robust `0.90`:

| seed | PF | DD | mean/trade | trades |
|---|---:|---:|---:|---:|
| 42 | 2.854 | 8.7% | +$309.7 | 294 |
| 43 | 1.635 | 11.5% | +$125.7 | 348 |
| 44 | 1.691 | 18.3% | +$162.8 | 340 |
| mean | **2.060** | 12.8% | +$199.4 | 982 |
| min | 1.635 | — | — | — |

Aggregate across all calibrated trades:

- PF `1.960`
- DD `14.4%`
- mean/trade `+$193.6`
- trades `982`

## Comparison

| stack | mean PF | min PF | mean DD | aggregate PF | trades | mean/trade |
|---|---:|---:|---:|---:|---:|---:|
| current promoted champion | 1.950 | **1.786** | **10.1%** | **1.976** | 809 | +$161.4 |
| iteration-2, chosen-trained L3 | 2.016 | 1.733 | 11.8% | 2.000 | 930 | +$158.1 |
| iteration-2, candidate-trained L3 | 2.029 | 1.691 | 8.9% | **1.993** | 930 | +$181.1 |
| candidate-L3 oracle retrain + candidate-L3 exit | **2.060** | 1.635 | 12.8% | 1.960 | **982** | **+$193.6** |

## Interpretation

This is a productive branch, but not a promotion.

What improved:

- highest mean PF so far (`2.060`)
- highest trade count among the composed simulated-L3 branches (`982`)
- highest mean/trade (`+$193.6`)
- materially lower call concentration
- seed 42 becomes very strong (`PF 2.854`, DD `8.7%`)

What blocks promotion:

- min PF regresses to `1.635`, below the current champion's `1.786`
- mean DD worsens versus champion (`12.8%` vs `10.1%`)
- aggregate PF is slightly below champion (`1.960` vs `1.976`)
- seed 44 DD rises to `18.3%`

Current repo truth:

- keep the current promoted champion:
  `simulated-L3 per-seed oracle + objective-consistent unified entry + chosen-trained Layer-3 robust 0.90`
- candidate-trained-L3 oracle retrain is a high-mean challenger, not a floor-safe replacement
- the next question is not whether candidate-trained L3 has signal; it does
- the next bottleneck is controlling seed-43/44 floor while preserving the seed-42 lift and broader side coverage

Good next diagnostics:

- localize seed-43/44 losing windows under the candidate-L3 oracle retrain
- compare candidate-L3 oracle chosen actions against iteration-2 chosen actions on the same days
- try an ensemble/mixture target between chosen-trained-L3 oracle and candidate-trained-L3 oracle rather than fully replacing the exit target
