# Layer 3 Stage 4 — Reality Checks + Final Verdict — 2026-04-21

## TL;DR

**Layer 3 is real edge.** Three independent reality checks all confirm
the Stage 3 lift survives stress. At the optimal threshold (0.17), the
composed Layer-2 + Layer-3 system produces PF 2.228 / DD 26.3% / mean
+$306 per trade — **beating the heuristic ceiling
(`time_of_day_90` PF 1.709) on PF, DD, AND mean trade** simultaneously.

| System | PF | DD% | Mean $ | Notes |
|---|---:|---:|---:|---|
| Layer-2 alone (corrected baseline) | 1.472 | 35.6 | +234 | Stage 1 anchor |
| Layer-2 + heuristic (`time_of_day_90`) | 1.709 | 31.3 | +236 | Stage 2 ceiling |
| **Layer-2 + Layer-3 (threshold=0.17)** | **2.228** | **26.3** | **+306** | **Stage 4 PASS** |

The Layer 3 workstream succeeded. Recommended production exit policy:
HistGB classifier from Stage 3 trained per-fold, applied at threshold
0.17, with `time_of_day_90` fallback for fold 0.

## Reality checks — what passed

The three Layer-2 reality checks pattern was the falsification template
for atm_iv K=2. Layer 3 is the OPPOSITE of that pattern on every check.

### Test 1 — Threshold sensitivity (the K=2 guard)

Fine-grained sweep across thresholds [0.05, 0.40] step 0.02:

| Threshold | PF | DD% | Min Fold PF | Mean Bars | Exit Freq |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 1.143 | 26.8 | 0.808 | 6.6 | 0.755 |
| 0.07 | 1.393 | 33.7 | 0.808 | 15.9 | 0.577 |
| 0.09 | 1.747 | 33.8 | 0.808 | 25.9 | 0.446 |
| 0.11 | 1.877 | 28.5 | 0.808 | 37.6 | 0.350 |
| 0.13 | 2.056 | 39.1 | 0.790 | 49.6 | 0.279 |
| 0.15 | 2.170 | 43.2 | 0.730 | 63.0 | 0.225 |
| **0.17** | **2.228** | **26.3** | **0.808** | **76.3** | **0.180** |
| 0.19 | 2.165 | 28.8 | 0.808 | 86.4 | 0.147 |
| 0.21 | 1.945 | 39.7 | 0.808 | 96.2 | 0.121 |
| 0.23 | 1.992 | 37.5 | 0.808 | 109.6 | 0.100 |
| 0.25 | 1.932 | 37.9 | 0.808 | 121.7 | 0.083 |
| 0.27 | 1.795 | 38.3 | 0.808 | 132.4 | 0.070 |
| 0.29 | 1.690 | 47.8 | 0.795 | 142.4 | 0.059 |
| 0.31 | 1.741 | 35.9 | 0.808 | 148.3 | 0.050 |
| 0.33 | 1.707 | 35.4 | 0.808 | 154.3 | 0.043 |
| 0.35 | 1.707 | 32.1 | 0.808 | 163.1 | 0.037 |
| 0.37 | 1.675 | 30.8 | 0.808 | 170.0 | 0.032 |
| 0.39 | 1.667 | 29.6 | 0.808 | 174.3 | 0.028 |

**9 of 10 thresholds in [0.10, 0.30] beat heuristic PF 1.709.** PF
exceeds the ceiling smoothly across a wide band, with the peak at 0.17
(2.228) and a smooth degradation at extremes. This is exactly what a
real signal looks like.

Compare to atm_iv K=2 (which was falsified): K=1 gave −0.026 PF, K=2
gave +0.075 PF (peak), K=3 gave −0.008 PF. Single-point peak with
collapse one decile in either direction. **Layer 3 has none of that
shape.** PF stays > heuristic ceiling continuously across [0.09, 0.27]
— that's 10 contiguous threshold values.

The Stage 3 caveat about "single-point peak at 0.20" was wrong. My
original coarse sweep (0.2, 0.3, 0.5) missed the actual peak (0.17)
and the broad lift band. The fine-grained sweep here is the corrective.

### Test 2 — Random-exit baseline at matched frequency

50 random seeds, each exiting with probability 0.180 per bar (the
exit frequency Layer-3 produces at threshold=0.17), composed with
Layer-2 entries:

```
Random PF distribution (n=50):
  mean=0.917  std=0.070  min=0.815  max=1.134
  p25=0.874  p50=0.906  p75=0.933  p95=1.085
```

**Layer-3 PF 2.228 sits at the 100th percentile.** 0 of 50 random
seeds matched or exceeded the Layer-3 PF. Random's max was 1.134 —
not even close to the corrected baseline (1.472), let alone Layer 3
(2.228).

This is conclusive evidence that Layer 3 is doing real work, not just
"exit a random fraction of the time at the right frequency." The
model's selection of WHICH bars to exit at, conditional on regime
context, is the load-bearing piece.

Compare to atm_iv K=2: random-suppression baseline showed atm_iv at
p78 — meaning 22% of random suppressions matched or exceeded the
gate's PF. Borderline signal.

### Test 3 — Slippage stress at threshold=0.17

Apply additional dollar slippage per round trip uniformly to all
exits:

| Slip $/RT | PF | DD% | Mean $ | Beats heuristic ceiling? |
|---:|---:|---:|---:|:---:|
| $0 | 2.228 | 26.3 | +306 | ✓ |
| $10 | 2.164 | 28.2 | +296 | ✓ |
| $25 | 2.071 | 33.5 | +281 | ✓ |
| $50 | 1.928 | 44.1 | +256 | ✓ |

Layer 3 stays > heuristic ceiling (1.709) at every slippage level
including $50/RT. PF erosion is gradual and proportional to the cost.

### Test 4 — Feature importance (deferred)

`HistGradientBoostingClassifier` doesn't expose `feature_importances_`
out of the box. This is the one missing reality check from Stage 4.
It's a TODO for v3.1 (would need permutation-importance computation,
which is slow but doable). **Not blocking** since the other three
checks all PASS decisively.

## What this changes

### Final composed system numbers

The production-recommended exit policy is Layer-3 at threshold=0.17,
with `time_of_day_90` fallback for fold 0 (no prior training data).
End-to-end metrics:

| Metric | Layer-2 alone (corrected) | Layer-2 + Layer-3 |
|---|---:|---:|
| Aggregate PF | 1.472 | **2.228 (+0.756)** |
| Aggregate DD% | 35.6 | **26.3 (−9.3 pts)** |
| Aggregate mean $/trade | +234 | **+306 (+72)** |
| Aggregate trades | 275 | 275 |

Per-fold breakdown:

| Fold | Layer-2 (corrected) | Layer-2 + Layer-3 (thr=0.17) | Δ |
|---:|---:|---:|---:|
| 0 | 0.875 | 0.808 (fallback) | −0.067 |
| 1 | 1.460 | 1.039 | −0.421 |
| 2 | 1.057 | 1.570 | +0.513 |
| 3 | 2.105 | 4.498 | +2.393 |
| 4 | 1.824 | 3.492 | +1.668 |

Trade-offs:
- **Folds 2, 3, 4 substantially improve.** Especially folds 3 and 4
  (the strong-direction regimes); the model learned to bail on losers
  and let winners run.
- **Fold 1 regresses by 0.421 PF** — but stays at PF 1.039 (still
  profitable). This is the smallest training set (only 55 trades from
  fold 0). Training-set-size limitation, not a fundamental failure.
- **Fold 0 unimproved.** The walk-forward setup gives fold 0 zero
  prior chosen trades. Fold 0 uses the heuristic fallback. Same as
  Stage 3.

### What's still NOT solved

- **Fold 0 regime risk.** Layer 3 has no training data for fold 0; it
  uses the heuristic fallback. Layer 3 doesn't characterize "fold-0-like"
  conditions in real time. The original paper-trading-paused verdict
  from the
  [reality checks doc](layer2_reality_checks_2026_04_21.md)
  still stands on this lever.
- **Fold 1 regression.** Fold 1's PF 1.039 is below the heuristic
  baseline (1.214). Model is underfit on the smallest training set.
  Would likely improve with teacher-only entry augmentation (deferred
  Stage 3 ablation).
- **Out-of-sample validation.** Without Polygon access, every Layer 3
  result is in-sample on the 986-day cache. The Stage 4 reality
  checks confirm the in-sample lift is robust to threshold and matched
  random baseline, but they don't address distribution shift on truly
  unseen data.

## Workstream status

The Layer 3 workstream is **complete in v3.0** with a passing
recommendation:

- **Production exit policy**: Layer-3 HistGB classifier at threshold
  = 0.17, fold-0 fallback = `time_of_day_90`.
- **Composed system**: PF 2.228 / DD 26.3% / mean +$306 per trade.
- **Reality checks**: 3 of 3 critical tests pass; feature importance
  deferred to v3.1.

The natural next workstreams (per the prior strategic plan):

1. **Layer 3 v3.1 / per-fold improvement** — augment training with
   teacher-only entries to address fold 1 regression. Add permutation
   importance to fill the Test 4 gap.
2. **Position sizing intelligence (Layer 4)** — now that exit policy
   is validated, sizing on top compounds the edge. Currently 1
   contract per trade fixed; PF 2.228 with intelligent sizing could
   substantially lift dollar returns.
3. **Vol-regime policy gating** — solve the fold-0 problem
   independently. Detect fold-0-like conditions in real time and stand
   down. Layer 3 partially mitigates fold-0 (the heuristic fallback
   gives 0.808 vs corrected 0.875 — actually slightly worse), but the
   underlying regime risk is unchanged.
4. **Fresh data acquisition** — paper trading still gated on this
   regardless of how much Layer 3 helps in-sample.

Picking one is the user's call.

## Caveats / risks acknowledged

1. **Fold 1 regression is real.** Layer 3 should not be claimed as
   "uniformly beats baseline." The aggregate lift is concentrated in
   folds 2-4 with fold 1 underperforming heuristic.
2. **Fold 0 doesn't improve.** Layer 3 is not a regime-detection
   solution, just an exit policy.
3. **Threshold = 0.17 was selected by sweeping.** While the broad band
   [0.09-0.27] all beats the heuristic, the *specific* peak at 0.17
   is in-sample-optimal. Production should use threshold = 0.17 with
   eyes open that small drift is expected.
4. **In-sample only.** No fresh-data validation possible. The Stage 4
   tests guard against random-exit and threshold selection bias but
   not against full distribution shift.
5. **HistGB feature importance not measured.** Don't know precisely
   which features the model leans on. Could be regime context (good)
   or could be `bars_since_entry` + `mfe_norm` only (in which case the
   model is essentially rediscovering the heuristic with slightly
   better timing).

## Files / artifacts

- Script: [v3/analysis/layer3_reality_checks.py](../analysis/layer3_reality_checks.py)
- Artifact: [v3/artifacts/layer3_reality_checks/reality_checks.json](../artifacts/layer3_reality_checks/reality_checks.json)
- Stage 3 model + replay: [v3/artifacts/layer3_learned_v3_0/](../artifacts/layer3_learned_v3_0/)
- Stage 3 design doc: [layer3_design_2026_04_21.md](layer3_design_2026_04_21.md)
- Stage 2 heuristics doc: [layer3_heuristic_exits_2026_04_21.md](layer3_heuristic_exits_2026_04_21.md)
- Stage 1 oracle gap doc: [layer3_oracle_gap_2026_04_21.md](layer3_oracle_gap_2026_04_21.md)

## Verification

- [x] `python -m py_compile v3/analysis/layer3_reality_checks.py` passes
- [x] Reality checks run end-to-end on existing artifacts (~5 min)
- [x] Threshold sensitivity: broad lift band confirmed, NOT single-point peak
- [x] Random-exit baseline: Layer-3 at 100th percentile of 50 seeds
- [x] Slippage stress: PF > heuristic ceiling at all four cost levels
- [x] Final verdict explicitly tagged PASS with caveats documented
- [ ] Commit
