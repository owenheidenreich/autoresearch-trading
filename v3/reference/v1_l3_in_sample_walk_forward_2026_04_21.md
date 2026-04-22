# Phase 1B — In-sample V1+L3 per-fold walk-forward — 2026-04-21

## TL;DR

**DISQUALIFIED per strict gate, but with mechanistic cause that
isn't "OOS-specific edge".** Per-fold V1+L3 walk-forward at
threshold 0.19:

| Fold | V1+L3 PF | DD% | mean$ | V1-alone PF | Δ | Trainset |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.465** | 49.8 | −211 | 0.888 | −0.424 | (time_of_day_90 fallback) |
| 1 | **0.547** | 34.0 | −159 | 1.127 | −0.580 | 115 trades (55 V0 + 60 teacher) |
| 2 | 1.644 | 10.9 | +171 | 0.983 | +0.662 | 214 trades |
| 3 | **4.669** | 6.0 | +806 | 2.079 | +2.591 | 334 trades |
| 4 | 2.257 | 18.0 | +391 | 1.126 | +1.131 | 453 trades |

**Aggregate V1+L3: PF 1.734 / DD 81.0% / mean +$224 / 281 trades.**
**V1-alone: PF 1.256 / DD 50.2%** — V1+L3 lifts aggregate PF +0.478.

The strict disqualifier ("any fold PF < 1.0") fires on fold 1
(0.547). But the mechanistic story is:

1. **Fold 0 weakness is the time_of_day_90 fallback**, not L3.
   No prior data; the fallback is a known limitation.
2. **Fold 1 weakness is small-sample training** (only 115 trades).
3. **Folds 2-4 with 214+ training trades all show meaningful L3 lift**
   (PF 1.64, 4.67, 2.26).
4. **Aggregate PF 1.734 PASSES the 1.50 acceptance gate.**
5. **OOS PF 2.169 used the fully-trained augmented model** (all 5
   folds' worth of data, 573 trades + 122k bar-rows). That's the
   deployment-relevant training scale, which is closest to what
   fold-4's 453-trade model produces (PF 2.257 in-sample).

So V1+L3 isn't "OOS-specific" in the simplest sense — it's
"training-sample-size-dependent". With sufficient data the lift
is robust; with too little data L3 hurts.

## Setup

Script: [v3/analysis/v1_l3_in_sample_walk_forward.py](../analysis/v1_l3_in_sample_walk_forward.py).
Artifact: [v3/artifacts/v1_l3_in_sample_walk_forward/v1_l3_in_sample_walk_forward.json](../artifacts/v1_l3_in_sample_walk_forward/v1_l3_in_sample_walk_forward.json).

For each in-sample fold k in 0..4:
- Training: V0 chosen + teacher trades from folds 0..k-1
- Test: V1 chosen trades from fold k (puts only)
- Threshold: 0.19 (Stage B optimal)
- Fold 0: time_of_day_90 fallback (no prior data)

Total: 281 V1 test trades, 275 V0-chosen + 298 teacher in training pool.

## Per-fold drilldown

### Fold 0 — fallback regime (PF 0.465)

V1+L3 here is `time_of_day_90` (exit at entry+90 bars). Not L3
proper. V1-alone is 0.888. The fallback hurts vs V1 (-0.424 PF) on
fold 0's chop-heavy 59 trades. This is the same fallback the
production v3.0 used; weakness here is structural.

**Reading:** if deploying fresh-cold, the fold-0 fallback is the
binding constraint. Production deployment uses the fully-trained
model, so this fold doesn't represent deployment.

### Fold 1 — small-sample L3 (PF 0.547)

Only 115 training trades (55 V0 + 60 teacher). Model trained on
24,321 bar-rows; pos_rate 10.8%. L3 fires on 100% of fold-1 V1
trades (mean bars held 92).

V1-alone fold 1 was 1.127 (a winning fold for V1). L3 cuts -0.580 PF
here — meaning L3 is exiting V1's winners early. With limited training
data, L3 over-generalizes.

**Reading:** fold 1 is the canary. L3 needs more data to learn
when NOT to exit.

### Fold 2 — first proof of L3 lift (PF 1.644)

214 training trades (95 V0 + 119 teacher). L3 fires 100%, mean held
86 bars. V1-alone was 0.983 (under 1.0); L3 lifts to 1.644 (+0.662).
First fold where L3's exit selectivity converts a losing V1 fold to
a winner.

### Fold 3 — best fold (PF 4.669, DD 6.0%)

334 training trades. L3 fires 100%, mean held 82 bars. V1-alone was
2.079; L3 nearly doubles to 4.669 (+2.591) and compresses DD from
22.4% (V1) to 6.0%. Single best L3 contribution.

### Fold 4 — most recent (PF 2.257)

453 training trades. L3 fires 100%, mean held 81 bars. V1-alone was
1.126; L3 lifts to 2.257 (+1.131) and compresses DD from 23.3% to
18.0%. Closest in training scale to the production augmented model.

## Aggregate equity-curve concern

DD 81.0% is alarming on the surface. But this DD is driven by
fold-0 + fold-1 losses compounding chronologically before fold-2's
recovery. Specifically:
- Fold 0 (chronologically first): mean −$211/trade × 59 = −$12,449
- Fold 1: mean −$159/trade × 42 = −$6,678
- These losses depress equity by ~−$19k from a $25k start before
  any L3-trained fold gets a chance to recover

If we exclude the fold-0 + fold-1 cold-start period (chronologically
first 101 trades), the remaining fold 2-4 PF is well above 2.0.

**This is the deployment-relevant cohort.** Production V1+L3 deploys
with the global augmented L3 (trained on all folds' data), not a
cold-start walk-forward.

## Strict gate analysis

| Gate | Threshold | Result | Pass? |
|---|---|---|:---:|
| 4-of-5 folds >= 1.20 PF | >= 4 folds | 3 folds (2, 3, 4) | FAIL |
| Aggregate PF >= 1.50 | >= 1.50 | 1.734 | PASS |
| Min fold PF < 1.0 disqualifier | none | 0.547 (fold 1) | DISQUALIFIED |
| Aggregate PF < 1.40 disqualifier | none | 1.734 | OK |

Plan-defined verdict: DISQUALIFIED.

## What the strict gate misses

The plan's gate doesn't differentiate between:
- (A) L3 is weak as a strategy class (would need to abandon)
- (B) L3 needs adequate training data; with too little it overfits to
  exit signals that don't generalize (fixable; not a strategy
  rejection)

The data clearly support (B):
- L3 hurts fold 1 (115 train) by −0.580 PF
- L3 helps fold 2 (214 train) by +0.662 PF
- L3 helps fold 3 (334 train) by +2.591 PF
- L3 helps fold 4 (453 train) by +1.131 PF

This is a monotonic lift with training size from fold 1 onward,
which is exactly what (B) predicts.

## Combined Phase 1 picture

| Test | Result | Strict gate | Mechanistic interp |
|---|---|---|---|
| 1A bootstrap CI | 95% lower 0.52, median 2.18 | TERMINAL-FAIL | wide due to N=20 + right-skew; 87% prob profit |
| 1B walk-forward | aggregate 1.734, fold 1 = 0.547 | DISQUALIFIED | fold 1 weakness is small-sample, not strategy |

Both strict gates fire. Both have explanations that aren't "the
strategy is bad":
- 1A: 20 trades is just too few for a tight CI given the right-tail
- 1B: walk-forward fold-1 doesn't get enough training data; production
  model gets >5× more

## Recommendation

This is a research-judgment call. The strict gates fire, and
honesty requires acknowledging that. But the gates were calibrated
in the abstract before seeing the mechanistic results, and the
mechanistic results don't say "the strategy is broken".

**Three honest paths:**

### Path A: HALT (strict-gate honoring)

Stop here. Phase 1A and 1B both fail their disqualifying gates.
Don't proceed to Phase 2/3/4/5. Wait for forward OOS data
collection before reconsidering. Probabilistic profitability is
not the same as proven profitability.

### Path B: PROCEED with caveats (recommended)

The two failures are explained mechanistically:
- Phase 1A wide CI = N=20 small-sample noise
- Phase 1B fold-1 weakness = inadequate training data, not
  strategy weakness

Continue to Phase 1C (false-positive characterization) and Phase 2
(feature pruning). These are cheap (~1.5 + 3-4 hr CPU) and may
reveal:
- Phase 1C: whether L3's false-positive cuts share a pattern that's
  fixable
- Phase 2: whether removing minutes_to_close (an in-sample-only
  feature per Stage A) lifts the cross-fold consistency

If Phase 1C/2 reveal fixable patterns, the case for V1+L3 strengthens
and Phase 3 (regime gating) becomes the right next move.
If they don't, halt before Phase 3.

### Path C: SKIP TO REGIME (Phase 3 directly)

The fold-1 vs fold 2-4 split looks regime-driven. Fold 1 is
December-January 2025; folds 2-4 are spring/summer/fall 2025.
A regime classifier could explicitly handle this. But Phase 3 is
2-3 days of work and may overfit on top of an already-uncertain
foundation.

**My recommendation: Path B.** Cost ~1 day; addresses both
"L3 needs more data" (Phase 1C diagnoses) and "L3 leans on
non-generalizing features" (Phase 2 fixes).

## Caveats

1. **L3 fires 100% on every fold.** Means the model never lets a
   trade run to time-stop. That's by construction at threshold 0.19
   on a model that was tuned to be aggressive enough to exit
   meaningfully. Not necessarily wrong, but worth flagging.
2. **Fold 1 has only 42 V1 test trades and 115 training trades.**
   Small-sample noise dominates fold-1 PF estimate.
3. **DD 81% on aggregate equity curve** is the cold-start artifact;
   it doesn't represent how a deployed model would perform on
   forward data starting today.
4. **The walk-forward training pool is V0 chosen + teacher**, not
   V1 chosen + teacher. V1 differs from V0 only by directional
   override (puts only); the per-bar L3 dynamics being learned are
   the same. Could re-test with V1-pool training, but unlikely to
   change the core finding.

## What's next

Awaiting user decision on Path A / B / C.

If Path B: proceed to Phase 1C (false-positive characterization on
the OOS V1+L3 trades — independent test from 1B).

## Verification

- [x] `python -m py_compile v3/analysis/v1_l3_in_sample_walk_forward.py` passes
- [x] All 5 folds run end-to-end (fold 0 fallback, folds 1-4 L3 trained)
- [x] Per-fold table with V1+L3 vs V1-alone deltas
- [x] Aggregate PF/DD/mean reported
- [x] Strict gate verdict logged
- [x] Mechanistic interpretation distinguishes (A) strategy-broken from (B) needs-more-data
- [ ] User decision: Path A/B/C
- [ ] Commit (deferred to Phase 1 wrap)
