# Layer-2 Reality Checks Before Paper Trading — 2026-04-21

## Verdict

**Paper trading paused. The direction head is cosmetic.** Check 2 fired
a hard stop: the Layer-2 model's aggregate PF 1.455 survives — but at
random call/put direction on the same entry gate, PF is still **1.116**.
The model adds +0.34 PF over random direction (meaningful), but the
PRIMARY edge is the entry gate (fixed-quantile entry_score + product
ranking + teacher-conditioned direction fallback), NOT the side head.

The "Layer-2 shared encoder is a directional model" framing is
overstated. The more accurate description is: **entry-selection model
with a teacher-conditioned direction rule and a weak side lean from
the side head on top.**

Check 1 (per-fold) and Check 3 (slippage) surfaced additional facts
that qualify what's deployable:

- **No single-feature regime shift above the 1σ bar** (largest:
  `atm_iv` at 0.71σ), so there's no obvious regime gate to add.
- **BUT the model's side_score magnitude drifts ~10× across folds**
  (fold 0 median +0.10, fold 3 median +1.18). This is model-side
  instability masquerading as regime signal.
- **Slippage-robust.** Even at $50 additional cost per round trip,
  the Layer-2 PF stays above 1.0 and the edge over the teacher
  baseline holds at +0.45 PF. Costs don't kill it.
- **94% of chosen trades are on NON-oracle bars.** The entry model
  isn't finding oracle moments; it's finding a different kind of bar
  that happens to be profitable under the fixed-quantile gate +
  teacher-put fallback rule.

## Per-check summary

### Check 1 — Per-fold regime diagnostic

Per-fold metrics (from [layer2_per_fold_diagnostic.py](../analysis/layer2_per_fold_diagnostic.py)):

| Fold | Test window | Trades | PF | DD% | WR | Mean $ | Call% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2024-12-19 .. 2025-03-19 | 55 | **0.860** | 28.2 | 30.9% | −76 | 10.9% |
| 1 | 2025-03-20 .. 2025-06-13 | 40 | 1.439 | 22.5 | 32.5% | +223 | 22.5% |
| 2 | 2025-06-16 .. 2025-09-10 | 60 | 1.038 | 27.6 | 38.3% | +18 | 1.7% |
| 3 | 2025-09-11 .. 2025-12-04 | 60 | **2.095** | 21.4 | 35.0% | +614 | 1.7% |
| 4 | 2025-12-05 .. 2026-03-04 | 60 | **1.801** | 32.2 | 43.3% | +322 | 41.7% |

Feature distribution comparison (losing fold 0 vs winning folds 3+4):
no feature's median shifts by more than ~1σ. Largest drift is
`atm_iv`: fold 0 median 0.107 vs winning-fold median 0.079 (0.71σ).
Not a hard stop, but a candidate for regime gating if a future
iteration wants to carve out the fold-0-like regime.

**Score distribution — the actual red flag:**

```
                    median    mean     std
fold 0: side_conf   +0.112  +0.126   0.128
fold 1: side_conf   +0.049  +0.074   0.135
fold 2: side_conf   +1.094  +1.072   0.177
fold 3: side_conf   +1.183  +1.138   0.214
fold 4: side_conf   +0.948  +0.936   0.210
```

The model's confidence magnitude jumps ~10× between folds 0/1 and
folds 2/3/4. That's not a feature drift — that's a training
instability. Each fold trains a fresh model on increasingly more
data (666 → 906 days), and the learned confidence scale shifts
dramatically. Since `fixed_quantiles` calibrates per-fold, the gate
ADAPTS — but the fact that the model's internal scale isn't stable
suggests whatever directional signal exists is fragile.

**Oracle outcome mix:** 94% of chosen trades are in the `empty`
bucket — meaning they are NOT at the oracle bar. The model's entry
selection is finding something OTHER than oracle moments to trade on.
That can still be real edge, but it's not "Layer-2 learning the oracle
surface."

### Check 2 — Random-direction ablation

Reference: [layer2_random_direction_ablation.py](../analysis/layer2_random_direction_ablation.py).
10 random seeds, same entry gate, coin-flip direction:

```
PF:        mean=1.116  std=0.168  min=0.847  max=1.316
DD%:       mean=70.1   std=26.8   min=38.1   max=115.1
mean_pnl:  mean=+56.0  std=86.8
```

Model reference: PF 1.455, DD 36.9%, mean +$225.

**Per plan §2b this triggers the hard-stop criterion.** Random
direction on the same entry gate still produces PF > 1.0, so the
claim "direction head is load-bearing" is false. What IS load-bearing:

- The entry gate (fixed-quantile `entry_score >= 0.60`)
- The product-score ranking (entry_score × side_conf) within a day
- The `teacher_if_triggered_else_put` direction rule

The side head contributes a +0.34 PF lift over random (1.116 → 1.455)
and tightens DD from 70% → 37% via lower trade-selection variance, so
it's not useless. But the primary edge is architectural — the gate +
teacher-put fallback works. Any serious "this is a directional model"
paper trading narrative is mis-scoped.

### Check 3 — Slippage / spread stress

Reference: [layer2_slippage_stress.py](../analysis/layer2_slippage_stress.py).
Additional $ per round-trip applied uniformly:

| Slip $/RT | Layer-2 PF | L2 DD% | L2 mean $ | Teacher PF | Teacher mean $ | Edge PF |
|---:|---:|---:|---:|---:|---:|---:|
|  0 | 1.455 | 36.9 | +225 | 0.961 | −21 | **+0.494** |
| 10 | 1.429 | 38.9 | +215 | 0.943 | −31 | +0.486 |
| 25 | 1.392 | 43.4 | +200 | 0.917 | −46 | **+0.474** |
| 50 | 1.332 | 52.0 | +175 | 0.876 | −71 | +0.456 |

Per plan §2c this is a PASS. Layer-2 stays profitable at $50
additional RT slippage, and the edge over the teacher baseline is
remarkably stable at ~+0.47 PF. Teacher baseline goes negative at
$25+; Layer-2 stays robust. Whatever the Layer-2 selection is doing,
the margin it takes is wider than typical 0DTE execution costs.

## What's actually real here

Distilling across the three checks:

1. **The entry gate is real edge.** Random direction at the gate
   makes PF 1.116 across 10 seeds. That's not noise.
2. **The side head adds +0.34 PF** (1.116 → 1.455). It's useful,
   not cosmetic *in absolute terms*, but it's not the dominant
   lever — the plan's hard-stop criterion rightly called this out
   because the "directional model" framing was the whole rationale
   for the multitask design.
3. **The edge survives costs.** Execution-realism concerns are not
   the blocker.
4. **Fold-0-like regimes are the blocker.** Under a specific
   2024-12 to 2025-03 distribution, the model loses money. Paper
   trading now is betting that regime doesn't return.

## What paper trading would actually be betting on

If we paper-traded today using the `--detach-side` model exactly as
shipped, we would be staking capital on:

- The entry gate's +0.116 PF lift over random direction holding in
  out-of-sample. (Real signal in-sample. Robustness unknown.)
- The teacher-conditioned direction rule adding lift. (Real lift
  +0.34 PF, mechanism is the put-fallback doing the work on chopped
  days, not the side head being accurate.)
- Slippage staying under $50/RT. (Realistic for body-of-day SPX 0DTE.)
- The 2025-Q1-like regime NOT recurring. (This is the unhedged risk.)

## Revised next-step tree

Per plan §3:

- **Check 1**: no hard stop on feature distributions, BUT model-side
  drift is a concern. Flagged, not blocking.
- **Check 2**: HARD STOP. Direction head not primary.
- **Check 3**: PASS.

Net: paper trading NOT unblocked. Three viable next directions:

1. **Simplify the claim and the architecture.** Drop the side head;
   use entry_score + teacher-conditioned direction rule alone.
   If the pared-down version hits PF ~1.15 on the same folds with
   less model complexity and less training instability, it's more
   robust to paper trade. The "detach-side shared encoder" story
   becomes "the entry-selection model that actually matters."
2. **Diagnose the fold-0 regime.** The 2024-12-2025-03 window
   broke this model. Why? Higher atm_iv (0.107 vs 0.079) is the
   leading statistical clue; the mechanism is untested. Without
   knowing the failure mode, we can't paper-trade safely.
3. **Regime-conditional gating.** If the fold-0 regime is
   detectable in real time (via atm_iv or similar), add a live
   gate that flattens exposure when we detect we're in it.

These are all LESSER workstreams than the current plan's scope.
Picking one is the user's call.

## What this plan does NOT change

- The tree fixed-quantile hybrid (`layer2_entry_side_fixedq_60_10`,
  PF 1.122, DD 56.2%) is unaffected and still a viable CPU baseline.
  Note the tree version WAS NOT subjected to the three checks here;
  running them on the tree baseline would be a useful next step to
  see whether the random-direction finding is a shared-encoder
  artifact or a general Layer-2 finding.
- W2a features remain the frozen input set.
- A1 (ORC σ-position gate, PF/attribution win) is untouched and
  remains shipped.

## Artifacts

- `v3/analysis/layer2_per_fold_diagnostic.py` — Check 1 script
- `v3/analysis/layer2_random_direction_ablation.py` — Check 2 script
- `v3/analysis/layer2_slippage_stress.py` — Check 3 script
- This doc — verdict

All three check scripts are CPU-only, run in seconds to minutes
against the existing artifacts in
`v3/artifacts/layer2_shared_enc_fixedq_detach/`. No GPU, no
rebuild, no retrain.

## Verification checklist (plan §6)

- [x] `python -m py_compile v3/analysis/layer2_per_fold_diagnostic.py` passes
- [x] `python -m py_compile v3/analysis/layer2_random_direction_ablation.py` passes
- [x] `python -m py_compile v3/analysis/layer2_slippage_stress.py` passes
- [x] Each script runs end-to-end on the current artifacts
- [x] Per-fold report separates folds 3/4 from 0/2; largest feature
      shift identified (`atm_iv` at 0.71σ); side_score magnitude
      drift called out as the subtler finding
- [x] Random-direction ablation produces specific PF = 1.116 mean
      over 10 seeds
- [x] Slippage stress produces $0 / $10 / $25 / $50 grid for both
      layer2 and teacher baseline
- [x] This doc written with explicit one-line verdict at the top
- [ ] Commit
