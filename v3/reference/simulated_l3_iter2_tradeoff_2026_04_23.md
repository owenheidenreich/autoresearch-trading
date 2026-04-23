# Simulated-L3 Iteration-2 Outer-Loop Tradeoff — 2026-04-23

## Hypothesis

After the per-seed simulated-L3 oracle branch became the new provisional
champion, the next honest question was whether one more outer-loop pass
would keep improving the stack if the oracle itself were rebuilt from the
**current promoted stack** instead of the prior one.

In practice:

1. build iteration-2 seed-specific oracles from the promoted stack
2. retrain the unified entry model against those oracles
3. compose with Layer-3 robust `0.90`
4. compare to the current per-seed-oracle champion

## Commands

Iteration-2 oracle build:

```bash
for s in 42 43 44; do
  # seed 42 uses the objfix run; seeds 43/44 use the per-seed promoted run
  # exact chosen-trades and calibration paths were used per seed
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer2.build_simulated_l3_oracle \
    --dataset v3/artifacts/layer2_action_surface_dataset.pkl \
    --champion-chosen-trades <current-promoted-seed-chosen-trades> \
    --champion-calibration <current-promoted-seed-robust90-json> \
    --output v3/artifacts/simulated_l3_oracle_seed${s}_iter2_fp.npz
done
```

Iteration-2 retrain + compose:

```bash
for s in 42 43 44; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer2.train_unified_policy \
    --tier dev --device cpu --seed $s \
    --run-dir v3/artifacts/layer2_unified_policy_simL3_iter2_seed$s \
    --utility-target simulated_l3 \
    --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_seed${s}_iter2_fp.npz

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.train_rolling \
    --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_simL3_iter2_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_seed$s \
    --seed $s

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_simL3_iter2_seed$s \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done
```

## Honest result

Compared to the current per-seed-oracle champion:

| seed | champion PF | iter-2 PF | Δ PF | champion DD | iter-2 DD | champion trades | iter-2 trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.786 | 1.733 | -0.052 | 11.7% | 16.8% | 235 | 302 |
| 43 | 1.802 | 2.384 | +0.582 | 9.1% | 9.1% | 241 | 334 |
| 44 | 2.264 | 1.932 | -0.332 | 9.6% | 9.6% | 333 | 294 |
| mean | 1.950 | **2.016** | **+0.066** | 10.1% | 11.8% | 809 | 930 |
| min | **1.786** | 1.733 | **-0.052** | — | — | — | — |

Aggregated PF:

- current champion: `1.976`
- iter-2: **`2.000`**

Mean trade PnL:

- current champion: `+$161.4`
- iter-2: `+$158.1`

## Interpretation

Iteration-2 is **not** a clean promotion. It improves the mean and the
aggregated PF, but it gives back floor and some DD:

- mean PF improves `1.950 → 2.016`
- aggregated PF improves `1.976 → 2.000`
- min PF regresses `1.786 → 1.733`
- mean DD worsens `10.1% → 11.8%`
- trade count rises `809 → 930`

This is the classic mean-vs-floor tradeoff, not a strict dominance move.

The repo should therefore keep the current per-seed simulated-L3 oracle
stack as the cleaner provisional champion, and treat iteration-2 as a
useful challenger that proves the branch is not exhausted but is now
entering a stability tradeoff regime.

## Best next step

Do **not** blindly keep iterating the same outer loop.

If we continue, the next hypothesis should be more surgical:

1. localize why seed `43` benefits so strongly from iter-2 while seed `44`
   regresses
2. or move to **candidate-trained L3** so the exit model itself matches the
   candidate trade distribution instead of repeatedly distilling through
   oracle reconstruction

For now, this is a clean stopping point:

- per-seed simulated-L3 oracle stack remains the provisional champion
- iteration-2 is a stronger mean-PF challenger, but not clearly more robust
