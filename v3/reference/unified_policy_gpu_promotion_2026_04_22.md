# Unified Policy GPU Promotion — 2026-04-22

## Purpose

Run the plan's section-1 promotion protocol (3 seeds, full 13-window
rolling, CUDA, up to 12 epochs) on the audit-fixed unified action
policy. The CPU 3-seed dev result was mean PF `1.112` (std `0.045`);
the open question was whether this was a compute ceiling or an
architecture ceiling.

## Setup

- GPU: Akash H100 80GB, provider `akash15pk...hr`, DSEQ `26503524`
- Deposit: `5 ACT` (`~$5`)
- Boot+start: `~9 min` total (workspace sync, dataset upload, pyarrow install)
- Training: `~20 min` (3 seeds × 13 windows)

Added `cmd_run_v3_promotion` to `v2/ops/deploy.sh` and added upload of
`v3/artifacts/layer2_action_surface_dataset.pkl` to `cmd_start`.

Command:

```bash
./v2/ops/deploy.sh boot
./v2/ops/deploy.sh start
./v2/ops/deploy.sh run_v3_promotion v3_unified_promo_001
./v2/ops/deploy.sh stop -y
```

## Results

| seed | trades | share | PF | DD | mean/trade | clean | fast | below-1 windows | min-window PF | slip $25 PF | runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 352 | 0.451 | **1.151** | 29.9% | +$54.3 | 0.37 | 0.20 | 4 | 0.491 | 1.079 | 346s |
| 43 | 432 | 0.554 | 1.101 | 38.5% | +$37.3 | 0.34 | 0.19 | 5 | 0.514 | 1.032 | 362s |
| 44 | 387 | 0.496 | 1.097 | 27.4% | +$35.3 | 0.38 | 0.21 | 6 | 0.159 | 1.027 | 452s |

- **Mean PF: `1.116`** (std `0.025`, range `1.097–1.151`)
- **Aggregated across `1171` trades: PF `1.115`**
- **Baseline: `1.132`**

## Comparison to CPU 3-seed V1

|  | CPU | GPU |
|---|---|---|
| Mean PF | 1.112 | **1.116** (+0.004) |
| Std | 0.045 | **0.025** (-0.020) |
| Agg PF | 1.111 | 1.115 |
| DD (seed 42) | 48.8% | **29.9%** (-18.9pp) |
| DD (seed 43) | 44.1% | 38.5% (-5.6pp) |
| DD (seed 44) | 45.3% | **27.4%** (-17.9pp) |
| Fast-loser rate | 0.20 avg | 0.20 avg |

GPU gave us:
- essentially the same mean PF (delta `+0.004` is within noise)
- half the cross-seed variance
- dramatically better drawdowns (mean `~13pp` lower)
- better slippage resilience (all seeds clear `PF > 1.0` at `$25` round-trip)

But **no lift over the baseline**. The architecture is signal-limited,
not compute-limited.

## Promotion Gate Status

| gate | threshold | result |
|---|---|---|
| W1 PF vs baseline | `PF > 1.132` | **FAIL** (mean `1.116`) |
| W1 DD vs Layer-2.5 | `DD ≤ 21.4%` OR `+0.15 PF` | FAIL (mean DD `31.9%`) |
| W1 trade share | `0.25 ≤ share ≤ 0.70` | PASS (`0.451–0.554`) |
| Per-seed PF ≥ 1.0 floor | all 3 seeds | **PASS** (`1.097–1.151`) |
| Patience fast-loser | `≤ 0.209` | borderline (`0.19–0.21`) |

All three seeds clear the per-seed `PF ≥ 1.0` floor from the plan. The
architecture is consistent and honest. It just does not clear the
aggregate PF gate.

## What GPU Proved

Two things that weren't clear from CPU alone:

1. **Sharpness is real.** DD dropped dramatically without losing PF.
   The GPU-trained model's contract picks are cleaner; the losses are
   shallower. This is a property of longer training + better
   convergence, not a fluke.
2. **The PF ceiling is architectural, not compute-driven.** A 1.5%
   gap to baseline with tight variance after the plan's prescribed
   promotion protocol means there is no cheap "more epochs" fix.

## What GPU Did Not Solve

The aggregate PF gate. Seeds cluster at `1.097–1.151`. To clear
`1.132` honestly, we need to change the signal surface or the
decision regime, not the optimizer.

## Recommendation

**Do not promote this build as the v3 champion.** The `V0 + time-stop`
baseline at `PF 1.132 / DD 21.4%` remains the honest champion. The
unified action policy is shelved as "working architecture, needs
different data regime to clear."

Next-step options per the plan, in order of cost/expected-value:

1. **Layer-3 outer loop composition (plan section 5).** The unified
   policy produces trades with lower DD than the baseline; feeding
   those trades into the rolling Layer-3 exit and retraining entry on
   composed utilities is the single move the plan identifies as most
   likely to lift aggregate PF. CPU-feasible. No new GPU needed.
2. **Bar-level decisions instead of daily picks.** The current rule
   picks one bar per day; ~85% of scored bars are discarded. A
   multi-entry policy over the day would extract more signal from the
   same model weights. Requires rule change, new harness metric.
3. **Longer sequence context (20→60 bars) with GPU.** Only worth
   running after the Layer-3 loop closes the PF gap; otherwise
   conflates lift sources.

The committed model weights for each seed stay in
`v3/artifacts/v3_unified_promo_001/seed_*/window_*/model.pkl` as
reference artifacts for the Layer-3 outer loop.

## Artifact Locations

- Per-seed reports: `v3/artifacts/v3_unified_promo_001/seed_{42,43,44}/report.json`
- Per-window models: `v3/artifacts/v3_unified_promo_001/seed_*/window_*/model.pkl`
- Chosen trades: `v3/artifacts/v3_unified_promo_001/seed_*/chosen_trades.pkl`
- Manifest: `v3/artifacts/v3_unified_promo_001/manifest.json`
