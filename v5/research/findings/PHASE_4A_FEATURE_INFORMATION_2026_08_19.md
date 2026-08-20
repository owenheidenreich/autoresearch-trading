# Phase 4a — the feature-information preflight: raw result, referred for adjudication

> **ADJUDICATED 2026-08-19 — outcome D, and §8.7 resolved as option 1.** The cold ruling, its
> reasoning, and the pre-declared Phase 4b discriminant are in
> [`PHASE_4A_ADJUDICATION_2026_08_19.md`](PHASE_4A_ADJUDICATION_2026_08_19.md). This document
> remains the raw referral and is unedited below this line.

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

The plant recovered, and this session judged the second clause of that routing criterion to
apply on the three diagnostics in §§3–5. So this document reports numbers and names the
questions. It does not answer them, and nothing here should be read as a finding about whether
the features work.

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

## 3. Diagnostic one — the null distribution and what the bar does on it

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

Two measurements, bearing on different quantities.

On the **point estimate**: no permutation reached +8.65pp, and the observed value sits about 5.6
null standard deviations from the null mean.

On the **declared verdict statistic**: at 648 selections the standard error on precision is about
1.8pp, so the upper bound sits roughly 3.5pp above the point estimate before any signal exists, and
the +4.0pp bar applied to that upper bound was cleared by a shuffled label in 7 of 20 draws. The bar
was declared exactly as the design specified and no threshold was moved at any point.

**This does not change the verdict.** The rule is applied as written. It is reported because the
bar's behaviour under the null is part of what a reader needs in order to weigh the verdict, and
§8.7 puts the governance choice it raises to the adjudicator.

## 4. Diagnostic two — per-family attribution

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

> **CORRECTION, 2026-08-19, after adjudication.** The "Alone" column below is **not a measurement of
> these families' contract-ranking information**, and should not be read as one. Tape, clock and
> chain-state are **minute-common**: all 16 of their features take a single value across every
> contract in a minute, so a probe restricted to them scores every contract in a minute identically
> and cannot rank contracts at all. Selection then falls to a stable-sort tie-break on stored row
> order, which is ascending `contract_id` — the deepest-OTM puts. Their "Alone" figures measure
> minute selection plus that tie-break. The "without" and "cost of removing" columns are unaffected.
> Phase 4b D3 replaces these numbers with randomized-tie-break versions. Left visible rather than
> deleted: the referral was built on these rows, and the adjudicator's correction of them is part of
> the record. See `PHASE_4A_ADJUDICATION_2026_08_19.md` §6 and §7.

Three observations, stated without characterisation:

- The tape group scores **−9.41pp alone** and **+9.41pp** as a marginal contribution to the full set.
- Removing the two ordering fields moves the result from **+8.65pp to +10.04pp**.
- The nine chain-internal fields — the families the design was built around, and the one information
  family this corpus has never fitted — score **+1.86pp** and **+0.16pp** alone, and **+2.62pp** and
  **+2.47pp** at the margin.

## 5. Diagnostic three — coefficient stability across folds

Tape coefficients from each fold's training fit (standardised units):

| Fold | `close_from_session_open_points` | `range_position` | `return_1m` | `realised_vol_15m` |
|---|---|---|---|---|
| 0 | −0.0471 | +0.0282 | −0.0108 | **+0.1295** |
| 1 | −0.0113 | −0.0096 | −0.0036 | **+0.1150** |
| 2 | −0.0098 | −0.0115 | −0.0039 | **+0.1527** |
| 3 | −0.0018 | −0.0211 | −0.0029 | **+0.1639** |

Three of the four tape channels sit near zero and change sign between folds.
**`realised_vol_15m` holds the same sign and a magnitude of +0.115 to +0.164 across all four folds**,
an order of magnitude above its neighbours.

Two pieces of context a reader needs, neither of them a conclusion: ledger **row 332** measured
magnitude selection as raising the realised move 51% and the premium 42% while moving P&L by thirty
cents, and the design's **§5.3** names magnitude rediscovery as a failure mode while nominating
cross-sectional path ordering as the hypothesis. Member P's label is defined on percentage moves of
the option mid. What relationship, if any, holds between those facts and this coefficient is the
judgement memo §5 reserves. **This session does not make it.**

## 6. Fold heterogeneity, for completeness

Real features, per fold: **+1.70pp, +9.82pp, +9.84pp, +13.22pp** — monotonically increasing across
chronological folds. Planted twin, per fold: +25.88, +12.27, +14.69, +27.07pp.

## 7. What is certain, and what is not

**Certain.** The probe is powered — it recovers +19.98pp of a +20.0pp plant. The observed +8.65pp
is not a harness artefact — 0 of 20 within-session permutations came near it. The mechanical
verdict under the declaration as written is PROCEED. Nothing economic was read, no score block was
touched, and the declaration's implementation hashes verified before the run.

**Not certain, and deliberately left open.** Whether +8.65pp on a binary percentage-move label
constitutes evidence worth spending the 118-parameter fit on, read against the bar's null behaviour,
the per-family attribution, and the coefficient audit. §8 puts that decision to Fable as four
mutually exclusive outcomes.

**Handing to Fable under memo §5.** No fit, no spend, no vendor contact, no promotion.

---

# 8. Adjudication brief for Fable

Written by the session that produced the result, under memo §5. It is built to be read cold. It
points at primary sources rather than restating them, argues both readings at equal weight, and
keeps the producing session's own lean quarantined in §8.9 where it can be discounted.

## 8.1 The decision

**The question is: what is the +8.65pp lift?**

Four mutually exclusive outcomes. Fable is asked to select one, or to say that none fits and name
what does.

| | Outcome | What it means for the job |
|---|---|---|
| **A** | The effect is tradeable signal at the operating rate | The full 118-parameter fit is warranted; measured per-field information sets the shrink order |
| **B** | The effect is a rediscovery of already-priced magnitude | The fit is not warranted on this feature set; the design's §5.3 failure mode has occurred and the job publishes that |
| **C** | The effect is an artifact — of the bar, of feature instability, or of the probe's construction | The measurement does not support either conclusion; what is wrong must be named before anything is re-run |
| **D** | Inconclusive; a different test is required | Neither A nor B can be reached on this evidence; the discriminating test is named and Phase 4a is re-run or replaced |

## 8.2 Primary sources

Fable has the repository. Nothing below is a substitute for these.

- This finding, §§1–7 — the raw result and the three diagnostics.
- [`PHASE_4A_DECLARATION_V1.json`](../../work/lifecycle-training/PHASE_4A_DECLARATION_V1.json)
  (`f6894fb8…`) — every threshold, pre-stated, with what was excluded from the probe and why.
- Receipts, all in `v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/`:
  `phase_4a_receipt.json`, `phase_4a_ablation.json`, `phase_4a_controls.json`,
  `phase_4a_null_distribution.json`.
- [`DO_NOT_RETEST.md`](../history/DO_NOT_RETEST.md) **row 332** (near-ATM 0DTE premium closed as
  efficiently priced; the magnitude-signal result) and **row 338** (the 375-cell conditional drift
  census, and its named reopening conditions).
- The V5 instability attribution: `v4/audit/autoresearch/causal_day_v5_instability_attribution_2026_08_14_attempt002/receipt.json`,
  and [`ATTRIBUTION_DECLARATION_V1.json`](../../work/entry-exit-attribution/ATTRIBUTION_DECLARATION_V1.json).
- The design this probe tests:
  [`LEARNING_CONTENT_DESIGN_2026_08_16.md`](../../work/lifecycle-training/LEARNING_CONTENT_DESIGN_2026_08_16.md),
  §5.2 for the preflight and §5.3 for what the model is asked to learn that is not already priced.

## 8.3 The raw result

Population, probe and outcome are tabulated in §2 above. In one line: **+8.65pp lift on a 30.70%
base rate, 648 selections over 324 sessions, 1,043,873 out-of-fold actions, 25 parameters, 4 nested
chronological folds inside the 405-session training prefix; verdict statistic +13.33pp against a
+4.0pp bar; plant recovered at +19.98pp of +20.0pp planted.** No economics were read; no score block
was touched; the declaration's implementation hashes verified before the run.

## 8.4 The three diagnostics

Reported in §§3–5 above with their numbers. Restated here as bare facts, without characterisation:

1. **Null distribution** (§3). Twenty within-session label permutations: mean −0.24pp, sd 1.60pp,
   max +2.17pp. Zero of twenty reach +8.65pp. Seven of twenty produce an upper bound at or above the
   +4.0pp bar; the largest null upper bound is +5.96pp.
2. **Per-family attribution** (§4). Chain internals alone +1.86pp, per-contract chain alone +0.16pp;
   these are the families the design was built around. Tape alone −9.41pp; the full set without tape
   −0.76pp. Removing the two ordering fields moves the result from +8.65pp to +10.04pp.
3. **Coefficient stability** (§5). `realised_vol_15m` holds the same sign and a magnitude of +0.115
   to +0.164 across all four folds; the other three tape channels sit near zero and change sign
   between folds.

Fold lifts, real: +1.70, +9.82, +9.84, +13.22pp. Planted: +25.88, +12.27, +14.69, +27.07pp.

## 8.5 The two readings, argued at equal weight

Presented in the order the outcomes were enumerated. Order carries no weight, and neither case is
the brief's position.

### 8.5.1 The strongest honest case that this is real signal (outcome A)

The probe is powered and the calibration is not marginal: it recovered +19.98pp of a +20.0pp plant,
which is close to exact, so an absence of signal would have been detectable and was not what
happened.

The effect is far outside its own null. Twenty within-session permutations — which preserve every
feature, every base rate and the full session structure, and destroy only the within-session ranking
the operating rate consumes — produced a maximum of +2.17pp against an observed +8.65pp. The point
estimate sits about 5.6 null standard deviations out. Whatever is being ranked on, the harness is not
generating it.

The probe is small by declaration, 25 parameters against 1.04 million out-of-fold actions, and every
fold trains strictly earlier than it scores with imputation and standardisation fitted on the
training part only. There is no threshold search, no architecture selection and no repeated looks;
the operating rate and the statistic were fixed before the run. That is a narrow surface for
overfitting.

The result is stable across folds in sign and grows in the later ones (+1.70, +9.82, +9.84, +13.22pp),
so it is not carried by a single window. The bootstrap lower bound is +3.94pp, i.e. the interval
excludes zero at the session level of clustering.

On the attribution: a family that contributes little *alone* is not thereby uninformative. Chain
internals at +1.86pp and per-contract chain at +0.16pp alone still cost +2.62pp and +2.47pp when
removed from the full set — a marginal contribution larger than their solo one is what conditional
information looks like when it needs other channels to be read against. The design's hypothesis is
explicitly *conditional*: chain state is claimed to matter through interaction with side and depth,
not as a standalone ranker.

On the label: member P is a **path-order** target — reaching +50% before −30% — not a magnitude
target. Row 332's result concerns expected gross P&L of buying premium; a bracket that pays for order
of arrival is not the same quantity, and the design (§5.3) selected it precisely because the premium
is one number per contract and cannot separately price cross-sectional ordering of path outcomes.
A loading on realised volatility could be the conditioning variable that makes that ordering legible
rather than the thing being traded.

### 8.5.2 The strongest honest case that this is an artifact or a rediscovery (outcomes B or C)

The declared verdict statistic does not discriminate at this sample size. Seven of twenty nulls
clear the +4.0pp bar on the upper bound. At 648 selections the standard error on precision is about
1.8pp, so the upper bound sits roughly 3.5pp above the point estimate before any signal exists; a bar
of +4.0pp on that quantity is approximately "point estimate above +0.5pp". A PROCEED obtained against
a bar a third of nulls clear carries correspondingly little information about whether the fit is
warranted, whatever the point estimate does.

The families the design was built to test contribute least. The nine chain-internal fields are the
entire reason this architecture exists and the one information family the corpus has never fitted;
they are worth +1.86pp and +0.16pp alone. If the fit is authorised on this result, it is authorised
on channels the design did not nominate.

The tape group is anti-predictive alone at −9.41pp and additive at +9.41pp in combination, and
removing two fields (`is_call`, `moneyness_itm_points`) raises the result from +8.65pp to +10.04pp.
Both are facts about the fitted solution's dependence on which other columns are present. V5's
autopsy is the local precedent for a fitted object whose behaviour was a property of its
parameterisation rather than of the market; the attribution receipt in §8.2 records how that was
established.

`realised_vol_15m` is the one coefficient stable in sign and magnitude across all four folds, and it
is a volatility channel. Row 332 measured magnitude selection as raising the realised move 51% and
the premium 42% while moving P&L by thirty cents, and row 338's 375-cell census found no observable
state in which buying near-ATM premium has positive expected gross. Member P's label is defined on
**percentage** moves of the option mid, and percentage thresholds are mechanically easier to reach on
cheaper and higher-volatility contracts. A ranking that loads on realised volatility can therefore
raise a percentage-bracket hit rate without implying anything about dollars — and no dollar column was
read in this run, so nothing here bears on P&L either way.

The fold pattern is monotonically increasing in chronology (+1.70 → +13.22pp), which is consistent
with a signal strengthening over time and equally consistent with a fold-dependent relationship;
this run does not separate those.

## 8.6 Discriminating evidence that was not gathered

Stated as open questions. This session did not run them and does not propose a plan; Fable may want
different ones, and naming these should not narrow that.

- Does the lift survive **within-premium-bucket** ranking — selecting the top action from within
  narrow bands of entry ask or moneyness, so that price level cannot be the ranker?
- Does the lift survive **removal of `realised_vol_15m` specifically**, as opposed to removal of the
  tape group? And what does a probe on that single channel produce alone?
- Does the same probe, against a **dollar-denominated target** rather than a percentage-move bracket,
  behave the same way? (Member P's executable dollar value exists in the adapter; reading it opens
  economics, which this phase was declared not to do.)
- Is the effect present at other **operating rates** — 1, 3, 5 selections per session — and does it
  behave as a genuine ranking should as the rate widens?
- Is the tape group's solo-versus-marginal sign reversal reproducible under a different **fold count
  or seed**, and does it survive regularisation of the probe?
- Does the chronological gradient in fold lifts track a **regime variable** already measured — the
  base rate's ρ=+0.355 correlation with realised volatility is recorded in the pre-fit review — rather
  than an increase in learnable structure?
- Would a **matched control** on session realised volatility, comparing selected against unselected
  actions inside the same volatility stratum, leave any lift?

## 8.7 The governance question, which must be ruled on explicitly

**This arose after the outcome, which is what makes it dangerous.**

The +4.0pp bar was declared in the design as a Wilson-*upper* threshold and was applied exactly as
written. The null measurement then established that this bar is cleared by a shuffled label in 7 of
20 draws. The bar's weakness is a property of applying an effect-size threshold to an upper bound at
648 selections; it was not visible when the bar was set, and no threshold was moved at any point.

**Changing a bar after seeing a result is precisely what this project forbids.** The honest options
are therefore two, and Fable is asked to choose between them explicitly rather than leave it implicit:

1. **Accept the result under the bar as declared.** PROCEED stands as the mechanical outcome, with
   the null behaviour recorded alongside it so that any downstream reader knows what the bar was
   worth. The verdict keeps its pre-registered meaning and nothing is rewritten.
2. **Declare the test inconclusive and re-run under a properly constructed bar** — one calibrated
   against the measured null distribution rather than against a historical effect size, declared
   before the re-run, with the current result treated as spent.

**What is not available** is retroactively re-setting the bar so that this specific result passes, or
so that it fails. Either would convert a pre-registered test into a post-hoc one, and the fact that
the producing session can see which way it would go is exactly why the choice is not the producing
session's to make.

## 8.8 Out of scope for this adjudication

Not reopenable here: the signed risk law ($2,000 dollar-stated ticket cap, 20% breaker, −40%
backstop), the chronology and its score blocks, the entry-then-frozen-exit ordering, the compounding
account law, the stated target (31.65% baseline, 45–50% survival, ≈+14 points), and the
**closed event-calendar question** (owner ruling 2026-08-19: no calendar data acquired, no calendar
feature admitted, FOMC days handled by not trading).

## 8.9 Phase 5 slice availability

Recorded so Fable knows what is and is not computable if the adjudication reaches Phase 5.

**The FOMC dates are no longer owed.** The owner supplied and verified **34** announcement dates on
2026-08-19, and they are recorded with their source in
[`session_calendar.py`](../session_calendar.py). Every Phase 5 day-type slice is therefore
computable now: event day, FOMC-excluded, OPEX, quarterly OPEX, month end, last Friday, month of
year, weekday.

One caveat travels with them: **2025-07-30 is an FOMC day absent from the corpus**, excluded by the
clock gate for three missing interior minutes. The all-sessions figures are already short one FOMC
day before any deliberate exclusion, and `coverage()` reports that rather than returning zero.

## 8.10 Disclosure — the producing session's prior

**Quarantined here deliberately, and placed last so it is separable from the evidence.** Fable can
discount a stated lean; it cannot discount one woven through a brief.

This session's prior, on reading the diagnostics, leaned toward outcomes B or C — that the stable
`realised_vol_15m` loading against a percentage-move bracket is closer to row 332's already-priced
magnitude than to the design's cross-sectional-ordering hypothesis. That lean is **not** a finding,
was not tested, and rests on an association between a volatility coefficient and a prior result about
dollars that this run did not measure and could not have measured, having read no economic column.

Two things follow that Fable should weigh against the lean rather than with it. The +8.65pp point
estimate is genuinely far outside its null and this session has no explanation for it under outcome
B — a magnitude rediscovery would still have to be a real ranking of something. And this session
chose which three diagnostics to run, which shapes what §8.4 contains; the ablation, the null and the
coefficient audit were selected because a large result warranted scrutiny, and a different set might
have surfaced different structure.
