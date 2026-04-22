# Fold-0 Fallback: time_of_day_90 vs time_stop — 2026-04-22

## Hypothesis

Layer-3 honest calibration uses `time_stop` as the fallback exit when
the L3 model is unavailable (fold 0, and sometimes fold 1). An
earlier exit policy (`time_of_day_90` with `fallback_bars=90`) might
cut theta decay on days where no early-exit signal triggers, and
improve Fold-0 PF.

## Mechanism

`time_stop` holds until session end. For 0DTE premium that often
means eating late-session decay on trades where the model didn't
learn to exit early. A fixed-horizon exit at ~90 minutes after entry
might cap those losses while still capturing mid-day moves.

## Commands

```bash
.venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/v3_unified_promo_001/seed_42/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_seed42_tod90 \
  --fold0-fallback time_of_day_90 --fold0-fallback-bars 90

.venv/bin/python -m v3.layer3.calibrate_threshold \
  --run-dir v3/artifacts/layer3_unified_seed42_tod90
```

## Results (seed 42)

|  | time_stop (cycle 2) | **time_of_day_90** |
|---|---|---|
| Exploratory best | 2.033 (thr 0.19) | **1.969** (thr 0.19) |
| Calibrated PF | 1.889 | **1.826** |
| Calibrated DD | 22.3% | **33.8%** |
| Calibrated mean/trade | +$168.3 | +$152.7 |
| W0 PnL | +$4352 | **-$1118** |

## Interpretation — Hypothesis Falsified

Every metric is worse under `time_of_day_90` fallback. W0 itself
flips from `+$4352` to `-$1118` — the fallback is shorter-duration
and exits BEFORE the natural day's signal matures. Holding to
session end is strictly better on this dataset for fold-0 trades.

Plausible explanation: these unified-policy entries are selected on
mornings that end up "cleanly" winning or losing across the whole
session. Cutting at `+90` bars gives up captured MFE on days the
move comes later (`bars_since_break_*` features suggest many entries
trigger just before a sustained continuation), and also prematurely
locks in drawdowns on days where the trade reverses and then works.

## Belief change

Drop "Fold-0 time_of_day_90 fallback is a PF lift" as a hypothesis.
Keep `time_stop` as the default fallback for future L3 runs on this
entry regime.

## Open

This does not exhaust fold-0 improvements. Other candidates:
- learn a fold-0 L3 model on Layer-2 training-set trades (needs
  careful construction to avoid leakage into later windows)
- use a more conservative fallback such as `time_of_day_60` — but
  given `time_of_day_90` hurt, shorter horizons are unlikely to
  help
- accept that fold-0 is the noisiest window by construction and
  focus effort elsewhere

Current best result remains cycle-2 honest calibration: mean PF
`1.670` across seeds 42/43/44.

## Artifact

- `v3/artifacts/layer3_unified_seed42_tod90/`
