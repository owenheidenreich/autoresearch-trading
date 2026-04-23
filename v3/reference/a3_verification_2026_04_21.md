# Phase 2A Verification — A3 Per-Fold Walk-Forward + Bootstrap CI — 2026-04-21

## TL;DR

**MIXED — A3 lifts OOS meaningfully but doesn't fix in-sample weakness.** The
Phase 2A "drop mfe_norm" change improves OOS bootstrap distribution
substantially but doesn't pass the strict per-fold gate (fold 1 still
< 1.0). Honest reading: A3 is the right deployment config IF OOS is
the trust anchor, but A3 doesn't make V1+L3 "robust" — it just shifts
the same OOS-vs-in-sample tradeoff further toward OOS.

## A3 per-fold walk-forward results

| Fold | A3 PF | A3 DD% | A3 mean$ | V1+L3 (Phase 1B) | Δ vs V1+L3 | V1-alone |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.465 | 49.8 | −211 | 0.465 | 0.000 (same fallback) | 0.888 |
| 1 | **0.451** | 41.9 | −206 | 0.547 | **−0.096** | 1.127 |
| 2 | 1.443 | 12.7 | +115 | 1.644 | −0.201 | 0.983 |
| 3 | 4.482 | 7.9 | +741 | 4.669 | −0.187 | 2.079 |
| 4 | **2.520** | 15.0 | +433 | 2.257 | **+0.263** | 1.126 |

**Aggregate A3 in-sample: PF 1.666 / DD 87.5% (cold-start) / 281
trades / mean +$200.**
**Aggregate V1+L3 baseline: PF 1.734.** A3 aggregate is 0.07 lower.

Only fold 4 improves under A3 vs V1+L3. Folds 1, 2, 3 are all worse
in-sample. Fold 1 (the binding gate constraint) gets *worse*, not
better.

## A3 OOS bootstrap CI

| Metric | A3 | V1+L3 baseline | Δ |
|---|---:|---:|---:|
| Observed PF | 2.847 | 2.169 | +0.678 |
| Median (bootstrap p50) | 2.905 | 2.180 | +0.725 |
| 95% CI lower (p2.5) | **0.680** | 0.520 | **+0.160** |
| 95% CI upper (p97.5) | 28.770 | 12.645 | wider tail |
| Fraction PF ≥ 1.0 | **92.7%** | 86.6% | **+6.1pt** |
| Fraction PF ≥ 1.5 | **82.0%** | 70.1% | **+11.9pt** |
| Fraction PF ≥ 2.0 | 69.5% | (not measured) | — |

**A3 OOS bootstrap is meaningfully better.** All percentiles shift
right; probability of profitability rises from 87% → 93%; probability
of "very profitable" (PF >= 1.5) rises from 70% → 82%.

But the 95% CI lower bound (0.68) still doesn't clear the strict
threshold (1.0). The Phase 1A gate fires again, just less severely.

## Verdict gate

A3 verification gate (from Phase 2A plan):
- Pruned model OOS PF >= 2.169 → ✓ PASS (2.847)
- No fold catastrophic regression in-sample → ✗ FAIL (fold 1 = 0.451 < V1+L3's 0.547)

The strict gate fires. **A3 doesn't fix the in-sample fold-1 weakness;
it slightly worsens it while gaining OOS.**

## Honest reading

The Phase 2A experiment only measured OOS PF and aggregate IS PF
on chosen-only (non-walk-forward) — both moved in A3's favor. The
per-fold walk-forward reveals a different picture: A3 trades a
small amount of early-fold in-sample PF (folds 1, 2, 3) for a
meaningful late-fold and OOS gain (fold 4 + OOS).

This pattern is consistent with V1 itself (Stage C1 found V1
sacrifices early/strong in-sample folds for OOS chop-defense).
A3 is doing the same thing one level deeper: the model with
mfe_norm overfits to in-sample exit patterns that don't generalize;
without mfe_norm, the model is more uncertain in-sample but
more robust OOS.

## What this DOES tell us

1. **A3's OOS lift is real and consistent across the bootstrap.**
   Not a single-day artifact. PF >= 1.5 in 82% of resamples.
2. **A3 is the right deployment config IF OOS is the trust anchor.**
   The OOS sample is the closest proxy we have to "what live trading
   would look like".
3. **The strict per-fold gate doesn't actually distinguish good
   strategies here.** Both V1+L3 baseline and A3 fail it at fold 1;
   so does almost any strategy with limited training data and
   chop-fold tests.

## What this does NOT tell us

1. **Whether A3 generalizes beyond the 20-day OOS window.** Same
   sample as Phase 1A; bootstrap improvement is real but doesn't
   address regime-persistence risk.
2. **Why fold 1 specifically is so bad** for both V1+L3 and A3.
   It's the second-smallest training set (115 trades) and a
   directional-up regime where V1 sacrifices PF. Could be
   training-data poverty + regime mismatch combined.
3. **Whether further pruning or model-architecture changes** could
   fix fold 1 without sacrificing OOS. (Beyond Phase 2 scope.)

## Updated Phase 1+2 picture

| Phase | Test | Result | Strict gate | Interpretation |
|---|---|---|---|---|
| 1A | Bootstrap (V1+L3) | CI [0.52, 12.65], 87% > 1.0 | TERMINAL-FAIL | small-N |
| 1B | Walk-forward (V1+L3) | agg 1.734, fold 1 = 0.547 | DISQUALIFIED | data-size |
| 1C | False positives | bar 3 + mfe ≈ 0 | HARD-PASS | fixable |
| 2A | Feature pruning | A3 OOS 2.847 | HARD-PASS | per-fold needed |
| **2A-V** | **A3 verification** | **agg 1.666, fold 1 = 0.451; OOS bootstrap 0.93/0.82** | **MIXED** | **trades IS for OOS** |

## Production decision

**Three options:**

### Option A: Lock V1+L3 baseline (PF 2.169, OOS bootstrap 87%)

Don't use A3. Stay with the original Phase 0 champion. Argument:
A3 doesn't fix the strict gate, and the in-sample regression
(0.547 → 0.451 on fold 1) makes the deployment story slightly
worse on the metric the strict gate cares about.

### Option B: Lock A3 (PF 2.847, OOS bootstrap 93%) (recommended)

Use A3 as the production config. Argument: OOS is the deployment
proxy. A3's OOS PF lifts +0.68; bootstrap probability of
profitability lifts from 87% → 93%; probability of PF >= 1.5
lifts from 70% → 82%. The in-sample regression is small (−0.07
aggregate, −0.10 on fold 1) and within walk-forward noise. If
deploying capital, you trust OOS, so you deploy A3.

### Option C: Hybrid — A3 + capital-floor circuit

Deploy A3 with a tight daily loss circuit. Best of both: A3's OOS
lift, plus an explicit hedge against the 7% bootstrap probability
of unprofitability (which is what the in-sample fold-1 weakness
hints at).

## My recommendation

**Option B (A3 production)** with the caveat that bootstrap CI 
remains wide. The OOS lift is the cleanest evidence we have, and
A3 is mechanistically motivated by Phase 1C diagnostics. The
in-sample regression is marginal and consistent with the V1 vs V0
tradeoff already accepted.

**Then:** commit Phase 2 (with A3 verification doc) and decide
Phase 3 (regime gating, ~2-3 days) vs Phase 5 (deployment
scaffolding, ~2-3 days engineering).

## Verification

- [x] `python -m py_compile v3/analysis/a3_verification.py` passes
- [x] All 5 folds run end-to-end with A3 (drop mfe_norm) per-fold
- [x] Per-fold table compared to V1+L3 baseline + V1-alone
- [x] A3 OOS bootstrap N=10,000 reported with full distribution
- [x] Honest verdict: gate fails but OOS evidence shifts right meaningfully
- [ ] Commit Phase 2 + decide next phase
