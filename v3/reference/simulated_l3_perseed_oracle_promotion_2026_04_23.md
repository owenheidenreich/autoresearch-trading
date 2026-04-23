# Simulated-L3 Per-Seed Oracle Promotion Run — 2026-04-23

## Hypothesis

The corrected simulated-L3 branch became a real challenger once
decision-margin calibration was made objective-consistent, but seeds
`43` and `44` were still trained against a **seed-42 oracle**. If the
remaining weakness is oracle distribution shift rather than lack of
signal, then rebuilding the simulated-L3 oracle from each seed's own
current champion stack should improve the composed result.

## Commands

Built per-seed fingerprinted oracles:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m v3.layer2.build_simulated_l3_oracle \
  --dataset v3/artifacts/layer2_action_surface_dataset.pkl \
  --champion-chosen-trades v3/artifacts/layer2_unified_policy_dev_3seed/seed_43/chosen_trades.pkl \
  --champion-calibration v3/artifacts/layer3_unified_cpu_w000_seed43/rolling_layer3_calibrated_robust_90.json \
  --output v3/artifacts/simulated_l3_oracle_seed43_fp.npz

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m v3.layer2.build_simulated_l3_oracle \
  --dataset v3/artifacts/layer2_action_surface_dataset.pkl \
  --champion-chosen-trades v3/artifacts/layer2_unified_policy_dev_3seed/seed_44/chosen_trades.pkl \
  --champion-calibration v3/artifacts/layer3_unified_cpu_w000_seed44/rolling_layer3_calibrated_robust_90.json \
  --output v3/artifacts/simulated_l3_oracle_seed44_fp.npz
```

Reran seeds `43` and `44` against their own oracle:

```bash
for s in 43 44; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer2.train_unified_policy \
    --tier dev --device cpu --seed $s \
    --run-dir v3/artifacts/layer2_unified_policy_simL3_perseed_seed$s \
    --utility-target simulated_l3 \
    --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_seed${s}_fp.npz

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.train_rolling \
    --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_simL3_perseed_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_simL3_perseed_seed$s \
    --seed $s

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_simL3_perseed_seed$s \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done
```

Seed `42` already used its own oracle in the objective-consistent rerun,
so the final 3-seed comparison is:

- seed `42`: [layer3_unified_cpu_simL3_objfix_seed42](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer3_unified_cpu_simL3_objfix_seed42)
- seed `43`: [layer3_unified_cpu_simL3_perseed_seed43](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer3_unified_cpu_simL3_perseed_seed43)
- seed `44`: [layer3_unified_cpu_simL3_perseed_seed44](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer3_unified_cpu_simL3_perseed_seed44)

## Entry-only change

Compared with the shared-oracle objective-consistent rerun:

| seed | shared-oracle time-stop PF | per-seed-oracle time-stop PF | Δ |
|---|---:|---:|---:|
| 43 | 1.016 | 0.966 | -0.051 |
| 44 | 1.034 | 1.199 | +0.165 |

Per-seed oracle hurts seed `43` slightly at the entry layer but helps
seed `44` materially and cuts its entry DD from `44.8%` to `19.9%`.

## Honest composed-stack result

Final comparison against the current provisional champion
`CPU w=0.00 + L3 robust 0.90`:

| seed | champion PF | simL3 per-seed PF | Δ PF | champion DD | simL3 per-seed DD | champion trades | simL3 trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.711 | 1.786 | +0.074 | 20.2% | 11.7% | 410 | 235 |
| 43 | 1.725 | 1.802 | +0.077 | 21.7% | 9.1% | 390 | 241 |
| 44 | 1.913 | 2.264 | +0.351 | 24.1% | 9.6% | 401 | 333 |
| mean | 1.783 | **1.950** | **+0.167** | 22.0% | **10.1%** | 1201 | 809 |
| min | 1.711 | **1.786** | **+0.074** | — | — | — | — |

Aggregated PF across all calibrated trades:

- champion: `1.7820`
- simL3 per-seed: **`1.9765`**

Mean trade PnL:

- champion: `+$141.0`
- simL3 per-seed: **`+$161.4`**

## Interpretation

This is the first outer-loop branch that clears the bar cleanly.

The shared-oracle simulated-L3 rerun already showed the target family
was real once calibration was fixed. The per-seed oracle result shows
the remaining shortfall was mostly a distribution-shift issue:

- all 3 seeds now beat the champion
- mean PF, min PF, DD, aggregated PF, and mean/trade all improve
- trade count remains lower (`809` vs `1201`), so this is a more
  selective stack, not just a higher-volume one

## New repo belief

The new provisional champion should be:

`simulated-L3 per-seed oracle + objective-consistent unified entry + Layer-3 robust 0.90`

That is the strongest honest composed-stack result in the branch so far.

## Best next step

The next loop should stop arguing about whether simulated-L3 is valid and
start exploiting it responsibly. Highest-value follow-up:

1. Treat this per-seed-oracle stack as the new provisional champion.
2. Write a focused promotion/diagnostic note comparing opportunity cost
   of lower trade count versus materially better PF/DD.
3. If further research is desired, the next technical branch is
   candidate-trained L3 rather than champion-trained L3, since that is
   now the clearest remaining source of target mismatch.
