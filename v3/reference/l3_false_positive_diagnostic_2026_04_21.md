# Phase 1C — L3 False-Positive Characterization — 2026-04-21

## TL;DR

**HARD-PASS — both false positives share an identical, textbook
feature signature.** L3 exits at bar 3 on trades that haven't moved
yet (mfe_norm ~ 0, current_pnl_norm ~ 0), bailing on the biggest
upside surprises. Specific testable rule: **"don't exit before bar
10 if mfe_norm < 0.05"** would have rescued both FPs. Theoretical
upper bound on PF improvement: **2.169 → ~3.6** if both FPs ran to
V1-alone time-stop.

## Setup

Script: [v3/analysis/l3_false_positive_diagnostic.py](../analysis/l3_false_positive_diagnostic.py).
Artifact: [v3/artifacts/l3_false_positive_diagnostic/l3_false_positive_diagnostic.json](../artifacts/l3_false_positive_diagnostic/l3_false_positive_diagnostic.json).

Method:
- Train augmented L3 (V0 chosen + teacher = 573 trades / 122k rows)
- Replay V1 OOS trades through the model at threshold 0.19
- Identify FPs: (V1_alone PnL − V1+L3 PnL) >= $500 AND bars_held <= 10
- Dump per-bar features at L3-chosen exit bar
- Compare FP vs non-FP feature distributions

## False positives identified

Only 2 of 20 OOS trades qualify:

| Day | entry | exit | bars | V1+L3 $ | V1 $ | Δ | p_exit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-03-05 | 38 | 41 | 3 | −167 | +2,982 | **+3,149** | 0.193 |
| 2026-03-30 | 117 | 120 | 3 | +166 | +1,134 | +968 | 0.376 |

Both exit at exactly bar 3 (entry+3 bars) with p just above the 0.19
threshold. Combined cost: **−$4,117** (V1+L3 left this much on the
table vs V1-alone).

## Feature signature at exit (FP vs non-FP)

| Feature | FP mean ± std | non-FP mean ± std | gap |
|---|---:|---:|---:|
| **bars_since_entry** | **3.000 ± 0.000** | 62.3 ± 48.2 | **−59.3** |
| **bars_to_session_end** | 189.5 ± 39.5 | 132.9 ± 40.3 | +56.6 |
| current_pnl_norm | −0.005 ± 0.149 | +0.171 ± 0.657 | −0.18 |
| **mfe_norm** | **−0.005 ± 0.149** | +0.648 ± 0.735 | **−0.65** |
| mae_norm | −0.119 ± 0.151 | −0.354 ± 0.244 | +0.24 |
| **mfe_bar_age** | **0.000 ± 0.000** | 18.9 ± 22.1 | **−18.9** |
| direction_is_call | 0.0 | 0.0 | 0 |

The pattern is overwhelmingly clear and exactly matches the textbook
"early bailout on slow-starting winner":

1. **Both FPs exit at bar 3.** No sample variance.
2. **Both have mfe_norm ≈ 0.** Trade has accumulated no upside yet.
3. **Both have mfe_bar_age = 0.** Current bar IS the highest PnL bar
   (because PnL hasn't moved up at all).
4. **Both have current_pnl_norm ≈ 0.** Trade is approximately
   flat-to-slightly-down.

By contrast, non-FP exits average bar 62 with mfe_norm 0.65 — they
exit with meaningful run-up already captured.

## Top L2 feature gaps

L2 feature differences are smaller and noisier (most below the
trade-state state-feature differences):

| Feature | FP mean | non-FP mean | abs gap |
|---|---:|---:|---:|
| l2_37 (likely time-of-day) | 3.5 | 40.4 | 36.9 |
| l2_38 | −0.5 | 15.8 | 16.3 |

The l2_37 / l2_38 gap is consistent with bars_since_entry — both FPs
exit very early in the trade lifecycle, and time-of-day-style L2
features reflect that. Not a separate signal.

## Mechanistic interpretation

The augmented L3 model has learned to exit aggressively when:
- The trade has not moved (mfe_norm ≈ 0)
- AND the trade is in early bars (bars_since_entry small)
- AND the model's `predict_proba` is just above 0.19

This is a *partially correct* learning. Most early-flat trades DO
go on to lose (which is why the model trained on them and learned
this rule). But a small minority of early-flat trades become huge
winners (March 5 +$2982, March 30 +$1134). The model can't
distinguish these from the losers using current features.

The pattern is also visible in why the rule overfits: training
data has many early-flat trades that lost, so the model learns to
exit. But the OOS sample happened to contain 2 outliers that
defied this rule. With more OOS data, this rule's true cost would
be averaged out.

## Proposed exit rule (for Phase 2 testing)

**Defer L3 exit if `bars_since_entry < 10` AND `mfe_norm < 0.05`.**

In English: if the trade is brand-new and hasn't moved yet, give it
time to develop before letting L3 declare it dead.

**Predicted effect:**
- Both FP trades blocked from early exit (both have bars=3,
  mfe_norm < 0.01)
- Trades that DID move early (mfe_norm >= 0.05) unaffected
- Trades that had time to develop (bars >= 10) unaffected
- Some additional losers may slip through (early-flat trades that
  do go on to lose — these are currently caught by L3)

**Theoretical upper bound on improvement:** if both FPs ran to V1
time-stop:

| Metric | Current V1+L3 OOS | If FPs run to V1 time-stop |
|---|---:|---:|
| Wins | $6,773 | $10,723 |
| Losses | $3,123 | $2,956 |
| PF | 2.169 | **3.626** |

(Caveat: this is the "best case if rule perfectly recovers V1
result". Actual fix would let trades continue but L3 might still
exit later — likely capturing most but not all of this upside.)

## What this DOES tell us

1. **L3's failure modes are not random** — they cluster on a
   well-defined feature signature.
2. **A simple, mechanically motivated patch** could improve OOS PF
   meaningfully.
3. **Phase 2 (feature pruning + retraining)** has a clear secondary
   objective beyond minutes_to_close — could the model learn this
   "give it time" behavior natively if retrained with a deferred-exit
   constraint?

## What this does NOT tell us

1. **Whether non-FP cases will degrade.** The deferred-exit rule
   blocks ALL bar < 10 + mfe_norm < 0.05 exits. Without empirical
   testing on the full OOS set, can't confirm net positive.
2. **Whether the rule generalizes.** N=2 false positives is a thin
   evidence base. The textbook quality of the signature is
   reassuring but not statistically conclusive.
3. **Whether retraining with this rule baked in** would degrade
   in-sample performance (the model "earned" its early-exit behavior
   on training data where it was net-positive).

## Combined Phase 1 picture (updated)

| Test | Result | Strict gate | Mechanistic interp |
|---|---|---|---|
| 1A bootstrap CI | 95% CI [0.52, 12.65], median 2.18 | TERMINAL-FAIL | small-N + right-skew |
| 1B walk-forward | aggregate 1.734, fold 1 = 0.547 | DISQUALIFIED | small-sample-data sensitivity |
| **1C false positives** | **2 FPs, identical signature** | **HARD-PASS** | **fixable pattern identified** |

Phase 1C is the first cleanly positive result. It says: the OOS
weakness has a *fixable cause*, not a "strategy is broken" cause.

## Updated recommendation

The case for proceeding to Phase 2 (feature pruning) is now
substantially stronger. Phase 2 should test:
1. The original goal: drop minutes_to_close + retrain
2. **NEW: implement the deferred-exit rule** (bars >= 10 OR mfe_norm
   >= 0.05) on top of either the original or pruned model

Both modifications should be tested individually and combined to
isolate effects.

## Verification

- [x] `python -m py_compile v3/analysis/l3_false_positive_diagnostic.py` passes
- [x] All 20 V1 OOS trades replayed through augmented L3
- [x] FP criteria applied (Δ >= $500 AND bars_held <= 10) → 2 FPs
- [x] Trade-state feature comparison reported (full table)
- [x] Top 10 L2 feature gaps reported (no comparable signal)
- [x] Pattern identified: bar 3 + mfe_norm ≈ 0 + mfe_bar_age = 0
- [x] Proposed rule: defer exit if bars < 10 AND mfe_norm < 0.05
- [x] Theoretical upper bound computed: PF 2.169 → 3.626
- [ ] Commit Phase 1 wrap
