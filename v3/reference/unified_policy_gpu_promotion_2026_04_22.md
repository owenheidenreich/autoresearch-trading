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

## Baseline Sources (correcting an earlier conflation)

Two different baselines exist and an earlier draft of this doc mixed
them up. Keeping them straight:

| baseline | PF | DD | notes |
|---|---|---|---|
| `V0 + time-stop` (entry-only, same shape as unified policy) | **`1.132`** | `95.9%` (cold-start equity-curve artifact) | the PF gate reference |
| Layer-2.5 patience-gated (entry + patience, different stack) | `~1.795` | **`21.4%`** | the DD gate reference |

The unified policy is structurally comparable to `V0 + time-stop`
(both are entry-only with time-stop exit). On DD the unified policy
is **dramatically better** than V0 (`27–38%` vs `95.9%`). On PF it is
`1.5%` short of V0.

## Promotion Gate Status

| gate | threshold | source | result |
|---|---|---|---|
| W1 PF vs baseline | `PF > 1.132` | V0+time-stop | **FAIL** (mean `1.116`) |
| W1 DD vs Layer-2.5 | `DD ≤ 21.4%` OR `+0.15 PF` | Layer-2.5 | FAIL (mean DD `31.9%`) |
| W1 trade share | `0.25 ≤ share ≤ 0.70` | — | PASS (`0.451–0.554`) |
| Per-seed PF ≥ 1.0 floor | all 3 seeds | plan section 1 | **PASS** (`1.097–1.151`) |
| Patience fast-loser | `≤ 0.209` | — | borderline (`0.19–0.21`) |

All three seeds clear the per-seed `PF ≥ 1.0` floor from the plan.
The DD gate reference is Layer-2.5's `21.4%`, which is a stricter
bar than the unified policy can clear at this architecture. Against
the structurally-matched comparison (V0), DD is far better.

## Side Concentration — Not Yet Exhausted

Call share across seeds: `93.5% / 91.9% / 93.5%` (mean `~93%`). This
is extreme enough that "working architecture, signal-limited" is not
an honest conclusion yet. The current ranking loss has two terms:

- best-vs-rest: pushes argmax(utility) above all other tradeable actions
- flat-vs-contract: pushes winning contracts above flat, flat above losing contracts

Neither of these directly contrasts same-bar call-vs-put. A call
with utility `+$500` and a put with utility `+$100` both get pushed
up relative to flat by the second term; only the best-vs-rest term
separates them, and the best-vs-rest gradient on "right side" is
diluted across ~24 contracts. The model can coast on a side prior.

A same-bar contrastive term (e.g., `relu(margin - (best_side_pred -
other_side_pred))` with weighting by the sign of the true utility
gap) forces direct side discrimination. This is a cheap CPU test and
should happen before declaring the architecture signal-limited.

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
baseline at `PF 1.132` remains the honest PF champion. But do not
shelve the unified policy yet; the `93%` call share is not sufficiently
interrogated.

Revised next-step order (after Codex code review flagged the
baseline conflation and the side-concentration issue):

1. **Layer-3 outer loop composition (plan section 5).** Highest EV,
   CPU-feasible. The unified policy produces trades with DD much
   better than V0; composing the rolling Layer-3 exit on those
   entries and retraining entry on composed utilities is the
   single move most likely to lift aggregate PF.
2. **Same-bar call-vs-put contrastive loss ablation.** Cheap CPU
   smoke + one reduced rolling slice. Add a hinge term that forces
   the model to directly discriminate between the best call and the
   best put on each bar, weighted by the sign of the true utility
   gap. If this drops call share and lifts PF, the "signal-limited"
   conclusion was premature and the architecture has more to give.
3. **Hold off on bar-level multi-entry changes** until after (1) and
   (2). Changing entry regime and exit regime together would muddy
   attribution.
4. Longer sequence context (`20→60`) on GPU — only if (1) and (2)
   close the PF gap and we need more lift.

The committed model weights for each seed stay in
`v3/artifacts/v3_unified_promo_001/seed_*/window_*/model.pkl` as
reference artifacts for the Layer-3 outer loop.

## Artifact Locations

- Per-seed reports: `v3/artifacts/v3_unified_promo_001/seed_{42,43,44}/report.json`
- Per-window models: `v3/artifacts/v3_unified_promo_001/seed_*/window_*/model.pkl`
- Chosen trades: `v3/artifacts/v3_unified_promo_001/seed_*/chosen_trades.pkl`
- Manifest: `v3/artifacts/v3_unified_promo_001/manifest.json`
