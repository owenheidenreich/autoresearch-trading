# Layer-3 Unified-Entry Honest Threshold Calibration — 2026-04-22

## Hypothesis

The cross-seed Layer-3 result from the same-day cycle (mean PF `1.789`
at `thr 0.19`, mean PF `1.825` at `thr 0.15`) used same-OOS threshold
picking. If we instead pick each window's threshold using only *prior*
rolling-window OOS replays (never the window being evaluated), most
of the PF lift should survive — turning the exploratory result into a
deployment-grade one.

## Mechanism

For each rolling window W:
- The Layer-3 classifier M_W is trained on trades from windows
  `0..W-1` (this was already the case in `train_rolling.py`).
- The **threshold** for window W is picked by sweeping the grid over
  rows from windows `0..W-1`'s own OOS replays (each from their own
  model M_w<W) and taking the threshold that maximized aggregated
  prior-window PF.
- That chosen threshold is then applied to window W's OOS rows.

Fold 0 has no prior data, so it uses the grid midpoint (`0.20`) as a
conservative default.

## Implementation

New script that post-processes an existing rolling L3 artifact dir:

- [v3/layer3/calibrate_threshold.py](/Users/gduby/Documents/autoresearch-trading/v3/layer3/calibrate_threshold.py)

Does not re-train any models; it only re-selects which threshold's
per-window rows to use, using per-threshold CSVs already written by
`train_rolling.py`.

## Commands

```bash
for seed in 42 43 44; do
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_seed${seed}
done
```

## Results

Per-seed (GPU promotion artifact entries, L3 exits):

| seed | entry+time-stop PF | exploratory best (same-OOS) | **calibrated (prior-window)** | DD | mean/trade |
|---|---|---|---|---|---|
| 42 | 1.151 | 2.033 (thr 0.19) | **1.889** | 22.3% | +$168.3 |
| 43 | 1.101 | 1.669 (thr 0.25) | **1.423** | 26.2% | +$100.4 |
| 44 | 1.097 | 1.812 (thr 0.15) | **1.698** | 22.7% | +$161.4 |

Cross-seed aggregates:

- mean calibrated PF: **`1.670`**
- min calibrated PF: **`1.423`** (seed 43)
- mean DD: `23.7%`

Calibrated loss vs exploratory best:

| seed | PF loss |
|---|---|
| 42 | -0.144 |
| 43 | -0.246 |
| 44 | -0.114 |
| mean | -0.168 |

Per-window chosen thresholds converged into a stable regime after
the first few windows:

- Seed 42: `0.19` for windows 6–12 (7 consecutive)
- Seed 43: mostly `0.20` and `0.30`, occasional `0.15`/`0.25`
- Seed 44: mostly `0.30` from window 6 onward

Seed 43 had the widest per-window threshold drift, which matches its
wider exploratory range and lower calibrated PF.

## Comparison to Baselines

| baseline | PF | DD |
|---|---|---|
| `V0 + time-stop` | 1.132 | 95.9% (cold-start artifact) |
| `Layer-2.5 + time-stop` | ~1.795 | 21.4% |
| Unified + time-stop (GPU, mean of 3 seeds) | 1.116 | 32% |
| **Unified + L3 honest calibration (mean of 3 seeds)** | **1.670** | **23.7%** |

Every seed's honestly-calibrated PF beats V0's `1.132` by `+0.29` or
more. Mean beats V0 by `+0.54` and is within `0.13` of
Layer-2.5+time-stop despite coming from a thinner (entry-only) stack.
DD is close to Layer-2.5 and far better than V0.

## Interpretation

### What this proves

- **The unified+Layer-3 composition is not a same-OOS artifact.**
  Removing that leak cost `~0.17` PF on average; the result still
  beats V0 by a wide margin across every seed.
- **Threshold choices stabilize.** Once a few windows of prior data
  exist, the calibrator converges on a consistent threshold per seed.
  Early windows are noisier, which is expected.
- **This is the first promotion-grade PF claim for the unified stack.**
  The prior GPU-promotion result was honest but too short on PF; the
  honest calibration sequel now clears the `V0 + time-stop` bar for
  every seed.

### What this does not yet prove

- DD is still anchored by a few bad days (the "L3 DD is identical
  across thresholds within a seed" observation from the prior note).
  Calibrated seed 43 is the weakest (`PF 1.423`, `DD 26.2%`) and its
  behavior on window 7 (2024 Q4) dragged `-$6181` on one window at
  `thr 0.30`.
- Fold 0 uses a grid-midpoint fallback rather than a learned default.
  That adds OOS noise on the first window per seed.
- Call share inside the entry stage is still `~93%`. Side
  discrimination remains open work.
- Only 3 seeds, 1 contract, Moderate rails. Paper trading still
  deferred per the prior memory.

### Repo-belief change

Unified policy (entry stage alone) is still `1.5%` short of V0 on PF.
Unified policy composed with Layer-3 under honest prior-window
threshold calibration beats V0 by `+0.54` PF on mean and by `+0.29`
on the weakest seed. **This is the leading promotion candidate** and
supersedes the CPU / GPU entry-only runs as the honest unified-stack
result.

### Next cycle candidates

1. **Fold 0 fallback threshold.** Rather than grid-midpoint, pick a
   conservative fallback based on the full training-set's predicted
   probability distribution (e.g. `p90` of exit-prob on training
   data). Would reduce the fold-0 noise without touching the honest
   calibration elsewhere.
2. **Why seed 43 W07 collapses.** One window dragged `-$6181` at the
   calibrator's chosen threshold. Diagnostic cycle: inspect what
   makes W07 different from W06 and W08 on seed 43 — if the
   calibrator had been told something useful, it might have picked a
   gentler threshold.
3. **Side-contrastive weight sweep**, small values (`0.05`, `0.10`,
   `0.20`). The `0.5` weight was too aggressive; a lighter hand may
   lift PF above `1.9` without destroying margin quality, and then
   composing on top of L3 would push further.

## Artifact Locations

- `v3/artifacts/layer3_unified_seed{42,43,44}/rolling_layer3_calibrated.json`
- `v3/artifacts/layer3_unified_seed{42,43,44}/layer3_trades_calibrated.csv`
