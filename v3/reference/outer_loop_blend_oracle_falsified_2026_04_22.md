# Outer-Loop Blend (Oracular Target) — Falsified — 2026-04-22

## Hypothesis

Cycle 7 locked `CPU w=0.00 + L3 robust 0.90` as the provisional
champion. The loop spec asks us to **begin** the first outer-loop
retrain hypothesis: retrain the unified entry policy against a
utility target that already includes information about the composed
exit, rather than time-stop PnL.

The simplest such target available in the current action-surface
dataset is `best_exit_pnl` — the oracle maximum PnL over each
contract's per-minute path from entry to session end. Blending
time_stop_pnl with best_exit_pnl by weight `alpha` gives a tunable
pull toward "pick contracts whose peak opportunity is better,"
which is closer in spirit to what L3 captures than the raw
session-end PnL.

`utility_blend = α → training target = (1-α) · time_stop_pnl + α · best_exit_pnl`

## Mechanism (pre-hypothesis)

- Mean `time_stop_pnl` across tradeable cells: `-$28` (theta decay
  dominates naive holds).
- Mean `best_exit_pnl` across same cells: `+$520` (oracle upper
  bound with 18% realistic L3 capture observed on chosen trades).
- A small α should shift the entry ranking toward contracts whose
  path has useful peaks; a large α should collapse the signal into
  "any contract with a non-trivial MFE is worth trading," which
  would over-trade and break abstention.

## Implementation

Added `--utility-blend α` to [v3/layer2/train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py)
. `_slice_inputs` now computes both the training target utility
(blended) and preserves the original `time_stop_pnl` as a separate
array `time_stop_raw` so downstream reporting (chosen_time_stop_pnl,
trade-share accounting) still reflects honest realized PnL rather
than the training-time blended target. Manifest records the
`utility_blend` value.

Ranges tested: `α ∈ {0.0 (champion), 0.10, 0.30}`, all on CPU,
3-seed dev harness (seeds 42/43/44), then composed with L3 robust
slack=0.90.

## Commands

```bash
for s in 42 43 44; do
  .venv/bin/python -m v3.layer2.train_unified_policy --tier dev --device cpu \
    --seed $s --run-dir v3/artifacts/layer2_unified_policy_blend030_seed$s \
    --utility-blend 0.30
  .venv/bin/python -m v3.layer3.train_rolling --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_blend030_seed$s/seed_$s/chosen_trades.pkl \
    --out-dir v3/artifacts/layer3_unified_cpu_blend030_seed$s
  .venv/bin/python -m v3.layer3.calibrate_threshold \
    --run-dir v3/artifacts/layer3_unified_cpu_blend030_seed$s \
    --policy prior_window_robust --robust-slack 0.90 --out-suffix robust_90
done
# repeat with --utility-blend 0.10 -> layer2_unified_policy_blend010_seedN
```

## Results — α = 0.30 (Cycle 8)

| seed | L3 PF (champion) | L3 PF (α=0.30) | Δ | DD (champion) | DD (α=0.30) |
|---|---|---|---|---|---|
| 42 | 1.711 | **1.742** | +0.031 | 20.2% | 20.5% |
| 43 | 1.725 | 1.445 | **-0.280** | 21.7% | **46.0%** |
| 44 | 1.913 | 1.375 | **-0.538** | 24.1% | **44.8%** |
| mean | **1.783** | 1.520 | **-0.263** | 22.0% | **37.1%** |
| min | 1.711 | 1.375 | -0.336 | — | — |

Entry-only time-stop baselines *did* improve at α=0.30 for seeds 42
(1.066→1.145) and regressed for seeds 43 (1.172→1.053) and 44
(1.097→0.964). Even where the entries improved by the time-stop
metric, the composed L3 PF did not follow.

## Results — α = 0.10 (Cycle 9)

| seed | L3 PF (champion) | L3 PF (α=0.10) | Δ | DD (champion) | DD (α=0.10) |
|---|---|---|---|---|---|
| 42 | 1.711 | **2.020** | **+0.309** | 20.2% | **13.2%** |
| 43 | 1.725 | 1.565 | -0.160 | 21.7% | 31.9% |
| 44 | 1.913 | 1.791 | -0.122 | 24.1% | **14.8%** |
| mean | 1.783 | **1.792** | +0.009 | 22.0% | **20.0%** |
| min | **1.711** | 1.565 | -0.146 | — | — |

`α=0.10` is almost exactly tied on mean PF and actually *improves*
mean DD by `-2.0pp`. But min PF drops by `-0.146` and max DD jumps
by `+7.8pp`. Same seed-dependent spread pattern: seed 42 benefits
dramatically, seeds 43 and 44 both regress.

## Interpretation

### Direction of failure is consistent across α

At both `α=0.10` and `α=0.30`, seed 42 is the one that benefits and
seeds 43/44 are the ones that regress. This is the same seed-
dependent noise pattern we saw with side-contrastive (seed 43 broke
at w=0.20, seed 44 broke at w=0.10). Different auxiliary signal,
same failure mode: the non-zero knob is a seed-specific boundary
perturbation, not a systematic improvement.

### Why the oracular target does not transfer to L3

The training target `best_exit_pnl` rewards contracts whose **peak
PnL at any point in the 330-bar session** is large. But L3 robust
0.90 exits at mean `73 bars` — it cannot capture peaks at bars
`200+`. The oracle-trained entry policy therefore biases toward
contracts whose peaks are late-session, which L3 systematically
misses.

Evidence: at α=0.10 the entry-only time-stop PF *did* improve on
all three seeds (1.066→1.135, 1.172→1.054, 1.097→1.139) — the
entries are objectively "better" by that naive metric. But composed
L3 PF is worse for 2/3 seeds. The entries that look better at
session-end are not the entries L3 can cash in on its typical hold
horizon.

### Repo-belief changes

- **Oracular utility blending with `best_exit_pnl` is not a
  productive training target on this stack.** At two magnitudes,
  the result is the same: one seed benefits, two regress, and the
  net is a worse promotion-grade stack.
- **"Entries look better" is not the right evaluation axis.** The
  only metric that matters is the composed-L3 PF, because that is
  what we would deploy. A blended target that improves time-stop
  PF while degrading composed PF is a negative result.
- **The productive outer-loop target is L3-hold-horizon-aware, not
  oracle-peak.** Any future composed-utility target needs to reflect
  PnL within L3's actual exit distribution (mean ~73 bars), not an
  unbounded session-wide peak. Options: (a) build a simulated-L3
  utility oracle that runs the champion L3 model over every
  candidate contract's path; (b) try `mfe_20` blended (early-window
  peak, better aligned to L3's hold), but the scale is smaller
  (mean +$28 vs +$520 for best_exit_pnl) so a different
  normalization is needed.

## Stop Decision

Stopping this workstream for now. Two cycles of blend-magnitude
exploration have shown the direction does not produce a clean
improvement at any α tested. A third cycle with a different target
(e.g. `mfe_20` blending) would be materially different hypothesis,
not grinding the same idea; if pursued, it should be its own
workstream.

The locked champion `CPU w=0.00 + L3 robust 0.90` (mean PF 1.783,
min 1.711, mean DD 22.0%) remains the provisional promotion target.

## Artifacts

- `v3/artifacts/layer2_unified_policy_blend010_seed{42,43,44}/`
- `v3/artifacts/layer2_unified_policy_blend030_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_blend010_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_blend030_seed{42,43,44}/`
- New flag: `--utility-blend` in [train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py)
