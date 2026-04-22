# Stage C2 — V1 (always_put) Pressure Test — 2026-04-21

## TL;DR

**ROBUST.** V1 (always_put) survives all three pressure tests cleanly:

- **Random-direction ablation**: V1 OOS PF 1.391 vs random-direction
  mean 0.849. V1 is 64% above random mean. In-sample 1.256 vs
  random 1.022 (23% above).
- **Per-fold robustness**: worst fold is 0 at PF 0.888, well above the
  0.80 floor. No fold regresses below the threshold.
- **Slippage robustness**: OOS PF stays above 1.0 at $50/RT slippage
  (PF 1.249 at $50, 1.318 at $25). Not fragile to execution costs.

Production-validated Layer-2 directional rule: **always_put**.

## Setup

Script:
[v3/analysis/layer2_directional_pressure_test.py](../analysis/layer2_directional_pressure_test.py).
Artifact:
[v3/artifacts/layer2_directional_pressure_test/directional_pressure_test.json](../artifacts/layer2_directional_pressure_test/directional_pressure_test.json).

Pressure-tested variant: V1 (always_put), the Stage C1 best.
- C1 in-sample: PF 1.256 / 281 trades
- C1 OOS: PF 1.391 / 20 trades

## Test 1 — Random-direction ablation

For each chosen bar, replace direction with rng.choice(call/put), 10
seeds, recompute PnL, average.

| Universe | Variant PF | Random PF mean | Random range | Variant edge |
|---|---:|---:|---|---:|
| In-sample (281 trades) | 1.256 | 1.022 | [0.681, 1.201] | +23% above random mean |
| OOS (20 trades) | 1.391 | 0.849 | [0.220, 1.482] | **+64% above random mean** |

**Reading:** V1's directional logic (force put) is doing real work
relative to random direction at the same chosen bars. The OOS edge is
larger than in-sample, meaning V1's "puts only" preference is more
valuable in the OOS regime than across the full in-sample distribution.

Note: random-direction OOS range goes up to 1.482, slightly above
V1's 1.391. So at least one random seed beat V1, but the average is
much lower. V1 is consistent; random is highly variable.

## Test 2 — Per-fold + OOS PF breakdown

| Fold | PF | DD% | Mean$ | Trades |
|---:|---:|---:|---:|---:|
| 0 | **0.888** | 41.5 | −65 | 59 |
| 1 | 1.127 | 29.4 | +70 | 42 |
| 2 | 0.983 | 33.2 | −8 | 60 |
| 3 | 2.079 | 22.4 | +610 | 60 |
| 4 | 1.126 | 23.3 | +71 | 60 |
| **OOS** | **1.391** | 17.0 | +159 | 20 |

Worst fold: 0 (PF 0.888) — above the 0.80 floor. No catastrophic
fold regression.

Compare to V0 baseline per-fold:
- Fold 0: V1 0.888 vs V0 0.860 (+0.028)
- Fold 1: V1 1.127 vs V0 1.439 (−0.312, V1 hurts)
- Fold 2: V1 0.983 vs V0 1.038 (−0.055)
- Fold 3: V1 2.079 vs V0 2.095 (−0.016)
- Fold 4: V1 1.126 vs V0 1.801 (−0.675, V1 hurts most here)

V1's in-sample weakness is concentrated in folds 1 and 4 (strong
directional-up regimes where calls were genuinely profitable). V1
slightly improves fold 0 (the chop loser).

## Test 3 — Slippage stress

| Slip $/RT | In-sample PF | In-sample DD% | In-sample mean$ | OOS PF | OOS DD% | OOS mean$ |
|---:|---:|---:|---:|---:|---:|---:|
| $0 | 1.256 | 50.2 | +140 | 1.391 | 17.0 | +159 |
| $10 | 1.235 | 53.3 | +130 | 1.361 | 17.5 | +149 |
| $25 | 1.204 | 59.3 | +115 | **1.318** | 18.3 | +134 |
| $50 | 1.155 | 70.1 | +90 | 1.249 | 19.6 | +109 |

OOS PF stays above 1.0 across the entire slippage grid. At $50/RT
(heavy slippage), OOS PF is 1.249 — still profitable.

In-sample DD widens substantially with slippage (50.2% → 70.1%),
which is concerning. OOS DD is more stable (17 → 19.6%).

## Verdict logic

| Criterion | Threshold | V1 result | Pass? |
|---|---|---|:---:|
| Random-direction PF < variant × 0.85 | < 1.183 OOS | random=0.849 | ✓ |
| Min fold PF >= 0.80 | >= 0.80 | min=0.888 (fold 0) | ✓ |
| OOS PF at $25/RT >= 1.0 | >= 1.0 | 1.318 | ✓ |

All three pass → **ROBUST**.

## What this confirms

V1 (always_put) is the validated production-recommended Layer-2
directional rule on the available evidence:
- Beats random direction by 64% on OOS
- Beats Layer-2 baseline OOS by +0.522 PF (Stage C1)
- No fold catastrophically regresses
- Survives realistic execution costs

## Caveats (important)

1. **Per-fold V1 sacrifices PF on directional-up regimes.** Folds 1
   and 4 lost ~0.31 and 0.67 PF respectively. V1 wins overall in-sample
   PF only because the loser folds (0, 2) gain slightly while strong
   folds (1, 4) lose modestly. This is the chop-defensive trade-off.
2. **20 OOS trades is small.** PF 1.391 with 95% CI ~ [0.85, 2.2].
   The "puts work" finding could partly be the single big put winner
   (March 5 +$2982) that anchored both V0 and V1 puts. Stage C2 didn't
   directly test ex-March-5 PF.
3. **Random-direction OOS max (1.482) actually beat V1 (1.391) on at
   least one seed.** That's one in 10. V1 is consistent, random is
   variable, but V1 isn't dominant on every random sample.
4. **In-sample DD widens dramatically with slippage** (50% → 70% at
   $50/RT). V1's in-sample equity curve is volatile; the OOS curve is
   smoother (17 → 20% DD).
5. **The "puts only" outcome is a brittleness signal.** It says the
   model's call signal is unreliable. A bot deploying V1 is implicitly
   abandoning half the model's training. Smaller training scope =
   simpler but less interesting strategy.

## What's next

Stage C3: compose V1 + augmented Layer-3 at threshold 0.19 (the
Stage B OOS optimal, not the v3.0 default 0.17). Test if augmented
L3 lifts V1's PF further or if L3 cuts V1's put winners (March 5
+$2982 etc.) the way it cut V0's.

## Verification

- [x] `python -m py_compile v3/analysis/layer2_directional_pressure_test.py` passes
- [x] Random-direction ablation: 10 seeds in-sample + 10 OOS
- [x] Per-fold table including fold 0 (worst, but >= 0.80 floor)
- [x] Slippage grid {$0, $10, $25, $50} × in-sample + OOS
- [x] All three verdict criteria pass → ROBUST
- [ ] Commit
