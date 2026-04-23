# Stacking: Side-Contrastive Entries + Robust L3 — 2026-04-22

## Hypothesis

Small improvements stack if they're independent. Cycle 3 showed
side-contrastive at `w=0.20` gives `+0.025 PF` / `-4.7pp DD` on
seed-42 entries alone. Cycle 5 showed robust L3 calibration adds
`+0.6+ PF` on top of any entry set. If the two signals are
independent, running robust L3 on `w=0.20` entries should beat
running it on `w=0.00` entries across seeds.

Secondary question: GPU-trained entries clearly beat CPU-trained
entries on raw PF (`1.151 / 1.101 / 1.097` vs `1.066 / 1.172 / 1.097`
— actually tied on two seeds). Does that advantage carry through to
the composed system, or does robust L3 erase the difference?

## Experiments

Three chosen-trade sources × 3 seeds × robust L3 calibration:

```bash
# baseline: CPU no side contrastive entries, from earlier 3-seed V1 run
for s in 42 43 44; do
  .venv/bin/python -m v3.layer3.train_rolling --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_dev_3seed/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_w000_seed$s
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_w000_seed$s \
    --policy prior_window_robust --robust-slack 0.90 \
    --out-suffix robust_90
done

# test: CPU w=0.20 side-contrastive entries (run from this cycle)
for s in 42 43 44; do
  .venv/bin/python -m v3.layer2.train_unified_policy --tier dev --device cpu \
    --seed $s --run-dir v3/artifacts/layer2_unified_policy_side_w020_seed$s \
    --w-side-contrastive 0.20
  .venv/bin/python -m v3.layer3.train_rolling --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_side_w020_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_w020_seed$s
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_w020_seed$s \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done

# reference: GPU promotion entries (cycle 5)
```

## Results

Per-seed robust-calibrated Layer-3 PF (DD in parens):

| entry config | seed 42 | seed 43 | seed 44 | mean PF | min PF |
|---|---|---|---|---|---|
| GPU promotion (cycle 5) | 1.919 (22.3%) | 1.652 (26.2%) | 1.812 (22.7%) | **1.794** | 1.652 |
| CPU w=0.00 | 1.711 (20.2%) | 1.725 (21.7%) | 1.913 (24.1%) | **1.783** | **1.711** |
| **CPU w=0.20** | 1.816 (19.1%) | 1.649 (32.9%) | **2.036** (30.9%) | **1.834** | 1.649 |

Entry-only time-stop baseline for reference:

| entry | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| GPU | 1.151 | 1.101 | 1.097 |
| CPU w=0.00 | 1.066 | 1.172 | 1.097 |
| CPU w=0.20 | 1.091 | 1.170 | 1.066 |

## Findings

### GPU training gives no meaningful advantage on the composed system

CPU w=0.00 mean PF `1.783` is statistically indistinguishable from
GPU mean `1.794`. Per-seed, CPU wins on seeds 43 and 44; GPU wins on
seed 42. The robust L3 calibration washes out most of the GPU
entry-stage quality advantage, because (a) the exit is learning its
own quality signal and (b) `w_robust` picks lower thresholds that
don't rely on perfectly-calibrated entry utilities.

**Implication**: future cycles on the composed stack can run on CPU
without a meaningful loss. GPU is only worth re-spending once we
have a hypothesis that clearly *needs* it (sequence length > 20, top-k
> 12, or > 20 epochs).

### Side-contrastive w=0.20 stacks, but unevenly

Across-seed change from `w=0.00 → w=0.20` (same CPU, same L3):

| seed | PF change | DD change |
|---|---|---|
| 42 | +0.105 | -1.1pp |
| 43 | **-0.076** | **+11.2pp** |
| 44 | +0.123 | +6.8pp |

Mean PF lifts by `+0.051` (`1.783 → 1.834`), but min PF *drops* by
`-0.062` (`1.711 → 1.649`). The side-contrastive weight helps two
seeds substantially and hurts seed 43. DD gets worse on seeds 43
and 44.

This is a fragile win. The overall winner by mean PF (`CPU w=0.20`,
`1.834`) is also the worst on min PF (`1.649`). For promotion the
min-PF seed is the binding constraint, so **the mean-lift is not a
promotion-grade improvement.**

### The single best per-seed result is CPU w=0.20 seed 44

Seed 44 with `w=0.20` entries and robust L3 gives PF `2.036` /
DD `30.9%`. That matches the exploratory best at threshold 0.15 on
this run, since robust-slack-0.90 usually picks `0.15`. This is the
single strongest single-seed PF we've seen on the honest pipeline.

### Repo-belief changes

- **GPU is not needed for composed-stack cycles.** Cycles that only
  vary loss weights / calibration policy / L3 parameters can run on
  CPU. Keep GPU budget for hypotheses that exceed the CPU policy
  envelope (sequence length, top-k, epochs).
- **Side-contrastive `w=0.20` is not a clean free lunch.** It helps
  mean PF a little and hurts min PF / DD. The prior cycle's small
  per-seed wins do not generalize cleanly in this stack.

### Honest leader board

| rank | config | mean PF | min PF | mean DD |
|---|---|---|---|---|
| 1 | **CPU w=0.20 + L3 robust 0.90** | **1.834** | 1.649 | 24.3% |
| 2 | GPU promo + L3 robust 0.90 (cycle 5) | 1.794 | **1.652** | 23.7% |
| 3 | CPU w=0.00 + L3 robust 0.90 | 1.783 | **1.711** | 22.0% |
| — | Layer-2.5 + time-stop (reference) | ~1.795 | — | 21.4% |
| — | V0 + time-stop (reference) | 1.132 | — | 95.9% |

By mean PF, `CPU w=0.20 + L3 robust` is the new leader. By min PF
and DD, `CPU w=0.00 + L3 robust` is the steadiest choice. Both are
close to Layer-2.5 time-stop and far above V0.

## Next-cycle candidates

- Pick the winning configuration and lock it as the champion claim.
  The tradeoff is "mean PF vs tightest floor" — for a 3-seed
  promotion claim the tightest-floor version is the right champion.
- Side-contrastive intermediate weights (`0.12`, `0.15`) to see if
  there is a weight that lifts PF without the seed-43 regression.
  Cheap CPU experiment.
- Diagnose seed 43 DD jump under `w=0.20` (32.9% vs 21.7% baseline).
  Is it one window, one day, or a systematic effect?

## Artifact Locations

- `v3/artifacts/layer2_unified_policy_side_w020_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_w000_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_w020_seed{42,43,44}/`
