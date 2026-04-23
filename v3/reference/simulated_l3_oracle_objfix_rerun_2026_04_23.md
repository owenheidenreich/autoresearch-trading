# Simulated-L3 Oracle Objective-Consistent Rerun — 2026-04-23

## Why this rerun happened

The first simulated-L3 outer-loop result was directionally interesting,
but it mixed objectives:

- training used `--utility-target simulated_l3`
- decision-margin calibration still optimized `chosen_time_stop_pnl`

That meant the model was trained on composed utility but admitted /
rejected on session-end PF. This rerun fixes that mismatch and hardens
oracle alignment:

- [v3/layer2/train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py)
  now calibrates on `chosen_objective_pnl` for non-time-stop targets
  and reports a separate `time_stop_reference`
- [v3/layer2/action_surface_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/action_surface_dataset.py)
  now defines a deterministic dataset fingerprint
- [v3/layer2/build_simulated_l3_oracle.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/build_simulated_l3_oracle.py)
  now persists that fingerprint into the oracle sidecar

## Commands run

```bash
.venv/bin/python -m v3.layer2.build_simulated_l3_oracle \
  --dataset v3/artifacts/layer2_action_surface_dataset.pkl \
  --champion-chosen-trades v3/artifacts/layer2_unified_policy_dev_3seed/seed_42/chosen_trades.pkl \
  --champion-calibration v3/artifacts/layer3_unified_cpu_w000_seed42/rolling_layer3_calibrated_robust_90.json \
  --output v3/artifacts/simulated_l3_oracle_seed42_fp.npz

for s in 42 43 44; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer2.train_unified_policy \
    --tier dev --device cpu --seed $s \
    --run-dir v3/artifacts/layer2_unified_policy_simL3_objfix_seed$s \
    --utility-target simulated_l3 \
    --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_seed42_fp.npz

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.train_rolling \
    --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_simL3_objfix_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_simL3_objfix_seed$s \
    --seed $s

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_simL3_objfix_seed$s \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done
```

## Entry-only rerun

Time-stop reference on the corrected entry sets:

| seed | PF | DD | mean/trade | trades |
|---|---:|---:|---:|---:|
| 42 | 0.961 | 50.8% | -$14.3 | 235 |
| 43 | 1.016 | 51.9% | +$6.0 | 288 |
| 44 | 1.034 | 44.8% | +$12.8 | 272 |

This is materially better than the old mixed-objective impression.
The corrected simulated-L3 entries are no longer clearly “bad unless
L3 rescues them”; they are roughly breakeven by time-stop on 2/3 seeds.

## Honest composed-stack result

Layer-3 robust `0.90` on top of the corrected entries:

| seed | champion PF | simL3 obj-fix PF | Δ PF | champion DD | simL3 obj-fix DD | champion trades | simL3 trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.711 | 1.786 | +0.074 | 20.2% | 11.7% | 410 | 235 |
| 43 | 1.725 | 1.815 | +0.090 | 21.7% | 9.1% | 390 | 288 |
| 44 | 1.913 | 1.724 | -0.189 | 24.1% | 10.4% | 401 | 272 |
| mean | 1.783 | 1.775 | -0.008 | 22.0% | 10.4% | 1201 | 795 |
| min | 1.711 | 1.724 | +0.013 | — | — | — | — |

Aggregated across all calibrated trades:

- champion aggregated PF: `1.7820`
- simL3 obj-fix aggregated PF: `1.7735`
- champion mean/trade: `+$141.0`
- simL3 obj-fix mean/trade: `+$139.6`

## Interpretation

The old “simulated-L3 fails the honest PF floor” verdict was too harsh.
Once calibration is made objective-consistent, simulated-L3 becomes a
real composed-stack challenger:

- mean PF is effectively tied with the provisional champion
- min PF is slightly better than the provisional champion
- DD is dramatically better (`10.4%` vs `22.0%`)
- trade count is materially lower (`795` vs `1201`)

So this branch did **not** win a clean promotion on mean PF, but it also
did **not** fail in the way the earlier mixed-objective result suggested.
The current honest statement is:

`CPU w=0.00 + L3 robust 0.90` remains the provisional champion on
slightly higher mean PF and larger opportunity set, while corrected
simulated-L3 is the strongest low-DD / higher-floor challenger so far.

## Next decision points

The next loop should not re-litigate whether simulated-L3 has signal.
It does. The open question is the promotion criterion:

1. Keep the current provisional champion if mean PF and trade count are
   the primary decision rule.
2. Promote simulated-L3 if min PF + DD carry more weight than a
   `0.008` mean-PF gap.
3. If more lift is needed before promotion, the next productive branch
   is reducing distribution shift inside the oracle target:
   per-seed oracles, or candidate-trained L3 instead of champion-trained L3.
