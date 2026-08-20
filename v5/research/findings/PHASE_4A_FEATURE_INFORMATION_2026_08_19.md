# Phase 4a — the feature-information preflight: raw result, referred for adjudication

**Date:** 2026-08-19 · **Job:** 46 lifecycle-training · **Declaration:**
[`PHASE_4A_DECLARATION_V1.json`](../../work/lifecycle-training/PHASE_4A_DECLARATION_V1.json)
(`f6894fb8…`) · **Receipts:** `phase_4a_receipt.json`, `phase_4a_ablation.json`,
`phase_4a_controls.json`, `phase_4a_null_distribution.json`

## 1. What this is, and what it is not

**The pre-declared verdict rule returns PROCEED, and that outcome is published as the rule
requires.** It is *not* being acted on, because the signed work-allocation memo §5 routes one
specific situation away from the session that produced it:

> Phase 4a returns **ambiguous** — near the +4.0pp boundary, **or plant recovered while real
> features behave oddly** — Do **not** interpret it. Report the raw numbers only. **Stop.**
> → **Fable**, adjudication.

The plant recovered cleanly and the real features behave oddly in three separate ways, all
measured below. So this document reports numbers and names the questions. It does not answer
them, and nothing here should be read as a finding about whether the features work.

**No economics were opened.** The probe reads member P's binary label and no dollar column. No
score block was touched; every fold lives inside the chronological training prefix.

## 2. The declared result

| | |
|---|---|
| Population | 405-session training prefix, 2022-06-01 → 2024-01-29 |
| Candidate actions | 1,301,541 (1,299,351 labelled, 2,190 unknown) |
| Scored out-of-fold | 1,043,873 actions over 324 sessions, 4 nested chronological folds |
| Probe | logistic, **25 parameters**, the declared ceiling |
| Operating rate | 2 selections/session → **648 selections** |
| Base rate | **30.70%** |
| Precision at the rate | **39.35%** |
| **Lift** | **+8.65pp** |
| Wilson upper on lift | +12.46pp |
| Cluster-bootstrap 95% interval | **+3.94pp to +13.33pp** |
| Verdict statistic (larger upper) | **+13.33pp** against the **+4.0pp** bar |
| Planted twin | **+19.98pp** recovered against **+20.0pp** planted (bar +10.0pp) |
| Mechanical verdict | **PROCEED** |

The plant recovery is essentially exact, so the probe is powered at the geometry that matters.

## 3. Oddity one — the null clears the declared bar 35% of the time

Twenty within-session label permutations. Shuffling **within** session preserves every feature,
every base rate and the session structure, and destroys exactly the within-session ranking the
operating rate consumes.

| | |
|---|---|
| Null lift | mean **−0.24pp**, sd **1.60pp**, max **+2.17pp** |
| Nulls reaching the observed +8.65pp | **0 of 20** |
| Observed lift vs null | **+5.6 null standard deviations** |
| Nulls whose **upper bound** clears the **+4.0pp** bar | **7 of 20 (35%)** |
| Largest null upper bound | **+5.96pp** |

Two facts, and they do not point the same way.

The **point estimate** is far outside the null: no permutation came close to +8.65pp. Whatever the
probe is ranking on, it is not an artefact of the harness.

The **declared verdict statistic** is a different matter. At 648 selections the standard error on
precision is about 1.8pp, so the upper bound sits roughly 3.5pp above the point estimate before any
signal exists. A +4.0pp bar applied to that upper bound is therefore cleared by a shuffled label in
about a third of draws. The rule as written is weak in a way that was not visible when it was
declared — and it was declared exactly as the design specified, so this is a property of the design,
not a deviation from it.

**This does not change the verdict.** The rule is applied as written. It is reported because a
PROCEED whose bar a null clears 35% of the time is a different object from a PROCEED whose bar it
never clears, and the adjudicator needs to know which one this is.

## 4. Oddity two — the lift does not come from the declared hypothesis

Per-group ablation, diagnostic only; the declaration bars any probe but the declared one from
changing the verdict.

| Feature group | Fields | Alone | Full set without it | Cost of removing |
|---|---|---|---|---|
| Tape (SPX close-series channels) | 4 | **−9.41pp** | **−0.76pp** | **+9.41pp** |
| Contract price (`ask`, `spread`, `self_iv`, `theta`) | 4 | +6.49pp | +6.95pp | +1.70pp |
| Clock | 5 | +3.40pp | +8.19pp | +0.46pp |
| **Chain internals — the hypothesis** | 7 | **+1.86pp** | +6.03pp | +2.62pp |
| **Per-contract chain (`smile_residual`, depth)** | 2 | **+0.16pp** | +6.18pp | +2.47pp |
| Ordering (`is_call`, moneyness) | 2 | +1.86pp | **+10.04pp** | **−1.39pp** |

Three things here are not what a clean result looks like:

- **The tape group is anti-predictive alone (−9.41pp) yet carries +9.41pp in combination.** A sign
  reversal of that size between a group's solo and marginal contribution is not the signature of a
  stable linear signal.
- **Removing the two ordering fields improves the lift**, from +8.65pp to +10.04pp.
- **The declared hypothesis contributes least.** The nine chain-internal fields — the entire reason
  this design was built, and the one information family this corpus has never fitted — are worth
  +1.86pp and +0.16pp alone, and +2.62pp and +2.47pp at the margin.

## 5. Oddity three — the one stable coefficient is a volatility channel

Tape coefficients from each fold's training fit (standardised units):

| Fold | `close_from_session_open_points` | `range_position` | `return_1m` | `realised_vol_15m` |
|---|---|---|---|---|
| 0 | −0.0471 | +0.0282 | −0.0108 | **+0.1295** |
| 1 | −0.0113 | −0.0096 | −0.0036 | **+0.1150** |
| 2 | −0.0098 | −0.0115 | −0.0039 | **+0.1527** |
| 3 | −0.0018 | −0.0211 | −0.0029 | **+0.1639** |

Three of the four tape channels sit near zero and wander in sign. **`realised_vol_15m` is stable in
sign, stable in magnitude, and an order of magnitude larger than its neighbours across all four
folds.**

The question this raises is for the adjudicator, not for this session: ledger row 332 measured
magnitude as *fully priced* — selecting the busiest third raised the realised move 51% and the
premium 42% and moved P&L by thirty cents — and the design's §5.3 names rediscovering magnitude as
the failure mode that loses before it starts. Whether a stable positive loading on realised
volatility, scored against a **percentage-move** bracket label, is that failure mode or is something
else is exactly the judgement §5 of the memo reserves. **This session does not make it.**

## 6. Fold heterogeneity, for completeness

Real features, per fold: **+1.70pp, +9.82pp, +9.84pp, +13.22pp** — monotonically increasing across
chronological folds. Planted twin, per fold: +25.88, +12.27, +14.69, +27.07pp.

## 7. What is certain, and what is not

**Certain.** The probe is powered — it recovers +19.98pp of a +20.0pp plant. The observed +8.65pp
is not a harness artefact — 0 of 20 within-session permutations came near it. The mechanical
verdict under the declaration as written is PROCEED. Nothing economic was read, no score block was
touched, and the declaration's implementation hashes verified before the run.

**Not certain, and deliberately left open.** Whether +8.65pp on a binary percentage-move label
constitutes evidence worth spending the 118-parameter fit on, given that the bar it cleared is
cleared by a third of nulls, that the declared hypothesis contributes least, and that the one stable
coefficient is a volatility channel in a project whose ledger says magnitude is fully priced.

**Handing to Fable under memo §5.** No fit, no spend, no vendor contact, no promotion.
