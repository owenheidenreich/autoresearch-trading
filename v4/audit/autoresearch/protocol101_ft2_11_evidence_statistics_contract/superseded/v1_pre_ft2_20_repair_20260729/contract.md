# Protocol101 FT2-11 Evidence And Statistics Contract

Status: **FROZEN DESIGN**

Graph node: `FT2-11-EVIDENCE-STATISTICS-CONTRACT`

Outcome: `design_ready`

Sole next node: `FT2-20-PARALLEL-DESIGN-REVIEW`

Product contract SHA-256:
`893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832`

## 1. Purpose

This contract fixes the rules for deciding whether a learned Protocol101
entry policy has evidence of real skill. It is not a model, a training run,
or a claim that skill exists.

The entry policy's full action space remains:

```text
WAIT
or
BUY one exact eligible contract from the 42-slot SPXW 0DTE ladder
```

The entry claim is deliberately harder than "the backtest made money." A
candidate must improve whole-path entry quality over both:

1. P5's frozen VWAP-side nearest-ATM heuristic under identical safety; and
2. a feature-independent, exposure-matched random policy that randomizes
   timing, side, and exact contract from the same governed ladder.

Positive absolute PnL is never proof that the model learned anything.

## 2. Verified Authority

The following sources were rehashed before this contract was written:

| Authority | SHA-256 |
|---|---|
| Consolidated Full Trader authority | `893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832` |
| Graph V2 | `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09` |
| Signed G4 v2 | `ff554ef5dd34086cd09954b906bbb5ce8455fa467776ae83906b01f80d565da5` |
| Signed Trader Charter | `3dbc1cf45e2200b7fd789c92be7e927c714b476b4b45aec95b5e0d9686c1ed66` |
| Signed D1 V2 amendment | `fcaf69ae080418cf6146f08d90386665c3c9f576f7382f4de8d8c51c23cf8598` |
| FT2-08 receipt | `56885fad09ea7034ce112c356edcce92bad0d9c52e93fb8e32cf66934f5ce518` |
| FT2-10 receipt | `ec3eff0d42947717b00d2b2055aa20e80ccfb22459b1285923e2c2bd79c682f9` |
| FT2-05 census result | `8a8805b22ef04a290edf421d5229d75eebc2301bc5246bd3bc595ee616cae583` |

Every FT2-08 and FT2-10 deliverable also reproduced the hash in its receipt.
Graph V2 contains one outgoing FT2-11 edge:
`design_ready -> FT2-20-PARALLEL-DESIGN-REVIEW`.

The consolidated authority retains all seven forecast horizons
`{3,5,10,20,45,90,remaining_session}`. That agrees with FT2-10.

## 3. What Counts As Independent Evidence

The market session is the dependence cluster.

Minutes from one day share volatility, market phase, opportunities, account
state, and serial occupancy. Forty-two contracts at one minute share the same
market event. Neither minutes, contracts, nor trades may be counted as
independent market samples.

Every statistic starts from a hashed session ledger. For every candidate,
seed, fold, and comparator, the ledger contains one chronological row per
session. The required session statistic is recomputed from all governed
decision minutes, including WAIT.

### 3.1 Effective sample size

For each statistic and fold:

1. Order sessions chronologically.
2. Estimate autocorrelations through lag five.
3. Retain the initial strictly positive sequence, stopping at the first
   nonpositive autocorrelation.
4. Compute:

```text
tau = max(1, 1 + 2 * sum(retained autocorrelations))
ESS = raw session count / tau
```

Selection requires ESS at least 20 in each fold and at least 100 pooled.
Lower ESS means `insufficient_evidence`, never automatic rejection.

### 3.2 Session-block bootstrap

All final intervals and p-values use 20,000 circular moving-block bootstrap
replicates with five-session blocks.

Within each 45-session outer fold, block starts are sampled with replacement,
wraparound is allowed, and the concatenated sample is truncated back to 45.
For pooled evidence, each fold is resampled independently and the five
fold-sized samples are concatenated. Blocks never cross a fold boundary.

The same bootstrap indices are reused across every candidate, seed, metric,
and comparator. This synchronization preserves paired dependence and makes the
maxT correction more powerful than pretending the tests are unrelated.

Per-fold reports use ordinary two-sided 95% session-block intervals. Hard
pooled improvement uses the studentized maxT-adjusted one-sided lower bound
and adjusted one-sided p-value defined in `bootstrap_spec.json`.

## 4. Entry Acceptance Standard

### 4.1 Primary whole-path claim

For every governed decision minute:

```text
DeltaQ = Q(candidate action) - Q(comparator action)
```

`Q` is the equal-family whole-path quality score from FT2-10:

- early drawdown;
- time to first real profit;
- pre-profit adverse excursion;
- underwater burden; and
- profitable-window stability.

WAIT has utility zero and remains in the denominator. The selected BUY action
means one exact contract, not an ATM proxy. Upside `U` is secondary and must
also point positive.

A candidate must have, against both P5 and matched random:

- maxT-adjusted 95% lower improvement bound strictly above zero; and
- maxT-adjusted one-sided `p <= 0.05`.

Equality at zero fails. Fixed-exit profit cannot substitute for this result.

### 4.2 Chronological robustness

At least four of five outer folds must have positive DeltaQ point estimates
against both comparators.

If one fold is negative, all of the following must hold:

- its magnitude is no more than half the pooled positive effect;
- its ordinary 95% interval includes zero;
- it has a frozen mechanical explanation such as sampling uncertainty,
  preregistered regime/side/phase mix, or mask/coverage mix; and
- it does not violate G4, daily stop, or ruin protection.

An unknown or post-result market story is not an explanation.

### 4.3 Seed robustness

There are three independent training seeds. Seeds are replications, not three
hypotheses. Every seed must:

- have positive pooled DeltaQ against both comparators;
- be positive in at least four folds against both; and
- pass the signed safety rules.

The frozen ensemble or aggregate must independently pass adjusted pooled
inference. One lucky seed can never promote a candidate.

### 4.4 Metric and regime robustness

All five quality-family deltas must be shown. At least four must have
nonnegative pooled point estimates, and none may have a 95% upper confidence
bound below zero. DeltaU must be positive.

Call/put, market phase, month, and frozen volatility-regime results are also
required. A stratum with at least 20 sessions and ESS at least 10 cannot be
credibly negative. If removing one stratum erases the entire pooled result,
the candidate does not pass unless the remaining data independently meet ESS
and MDE.

### 4.5 Fees and signed survival rules

Primary labels use the frozen $3 round-trip fee. The established $2.60 view is
diagnostic. The $4 round-trip view is a hard stress:

- DeltaQ must remain positive against both comparators; and
- G4 and the daily/ruin rules must still pass.

Entry quality does not define a learned exit. To carry signed G4 without
letting a fixed exit select the entry model, every candidate and comparator is
also replayed through one common neutral lifecycle: hold the entry to the
frozen 15:55 ET forced-flat boundary, use executable bids, one contract, one
open position, simulator v5, and the same D48/D49 safety.

Under both $3 and $4:

- pooled fee-adjusted net PnL / pooled maximum drawdown must be at least 1.0;
- equity in every fold must remain at or above $5,000 from $10,000 start;
- realized session loss stops new entries at 5% of session-start equity; and
- no overlap, revenge trade, scaling, or second contract is allowed.

This safety replay cannot rescue failed DeltaQ. It only prevents freezing an
entry policy that violates the signed survival contract.

## 5. Retired Gate Roles In The Full-Ladder Game

| Old role | Full Trader operational replacement |
|---|---|
| G1 profitability | MaxT-adjusted whole-path DeltaQ above zero versus both P5 and matched random. Fixed-exit PnL is report-only. G4 separately retains serial economics. |
| G2 no-skill null | D1 V2 strong target shuffle plus feature-independent timing/side/exact-contract randomization with the signed exposure-matching table. |
| G3 heuristic | P5 is the mandatory benchmark under identical safety, never the learned model's opportunity filter, direction rule, or contract selector. |
| G5 seeds | Three seeds all point positive and meet four-of-five fold robustness; the frozen aggregate passes adjusted inference. |
| G6 eras | Four of five chronological folds positive, bounded explained fifth, and no credible dependence on one side, phase, month, or volatility regime. |
| G7 frequency | No hard trade count. High frequency must pay fees and survive stress; rare trading is `insufficient_evidence` when ESS/MDE are inadequate. |
| G8 calibration | Report-only under the signed G8 amendment. If confidence controls WAIT/ENTER, the separate FT2-10 action-conditioned gate must first be smoke-tested, independently audited, and owner-signed. |

## 6. D1 V2 Full-Ladder Control

The random control is outcome-blind and feature-independent. It randomizes
timing, call/put, and the exact contract from the same 42-slot governed risk
set. Contract-only randomization is diagnostic, not hard evidence.

After serial replay it must match:

| Dimension | Bound |
|---|---:|
| Opportunity set and intent budget | Same / exact |
| Entry-intent count | Exact |
| Executed-trade relative drift | <= 5% |
| Call/put total variation | <= 5% |
| ATM/near/wing total variation | <= 6% |
| Mean executed-premium relative drift | <= 10% |
| Mean realized holding-time relative drift | <= 10% |
| Total occupancy relative drift | <= 10% |

An incomplete match is `insufficient_evidence`. A strong shuffled model with a
significant adjusted positive increment over its valid random control
invalidates real-candidate selection evidence. The old $3-per-trade
equivalence is a precision diagnostic only.

## 7. Multiplicity

The ordered multiplicity ledger includes:

- up to three canonical-price admission attempts;
- up to three outer-shortlisted entry lineages;
- all 4 guardrail anchors x 4 uncertainty multipliers for each lineage;
- both hard entry comparisons, P5 and matched random;
- both later lifecycle model families and every selectable lifecycle variant.

The maximum entry family is therefore:

```text
3 candidate lineages x 16 composer variants x 2 comparisons = 96 statistics
```

Forecast heads are not hypotheses because they are shared outputs of one
jointly trained model. Seeds are robustness replications, not selectable
hypotheses. Diagnostics that cannot alter a candidate are not hypotheses.

Within each fixed stage, synchronized studentized maxT controls one-sided FWER
at 0.05. Stages follow a fixed hierarchy: admission, entry, lifecycle, complete
Full Trader. A later family cannot be opened unless its prerequisite family
passes and freezes. Failed and abandoned attempts remain in the ledger.

## 8. MDE Before GPU Spend

FT2-05 is design-only evidence and does not say a model works. It supplies the
only allowed variance inputs for this node.

The cited remaining-session oracle-minus-P5 census row has:

```text
46 sessions
mean session increment       $5,645.804347826087
session increment variance   23,285,825.227536228
session increment standard deviation $4,825.538853593061
```

The FT2-10 design band is 0.6630 to 5.3043 trades/session, centered at
2.6522. That projects:

| Scope | Sessions | Projected trades |
|---|---:|---:|
| One outer fold | 45 | 30 to 238 |
| Five pooled folds | 225 | 150 to 1,193 |

Using 95% confidence, 80% power, and the 96-statistic conservative
Bonferroni planning approximation:

| Scope | Unadjusted MDE/session | Multiplicity-planning MDE/session |
|---|---:|---:|
| 45 sessions | $2,015.32 | $3,101.42 |
| 225 sessions | $901.28 | $1,386.9975 |

The frozen maximum plausible economic edge is 25% of the hindsight ceiling:

```text
$5,645.804347826087 x 0.25 = $1,411.4510869565217/session
```

At the center trade rate that is about $532.19/trade. The conservative pooled
planning MDE is only $24.45 below the bound. That is deliberately reported as
a narrow margin, not a green GPU light.

Before tranche 1, FT2-30 must use training-side-only pilot evidence to replace
the planning count and variance with:

- the actual ordered-ledger count;
- DeltaQ session variance versus P5 and matched random;
- actual projected sessions/trades; and
- the <= $10 checkpoint/resume proof.

The first $20 GPU tranche is blocked unless:

- pooled economic MDE <= $1,411.4511/session; and
- pooled DeltaQ MDE <= 0.05 against both comparators.

If either bound fails, route `resource_owner_decision_required` before spend.
A fold need not be individually significant; the high 45-session MDE is why
fold consistency, not five separate significance tests, is the era rule.

## 9. Exact Terminal Boundaries

Precedence is:

```text
invalid_evidence
  -> insufficient_evidence
  -> no_genuine_entry_signal
  -> entry_evidence_pass
```

### 9.1 `invalid_evidence`

Use this when the question cannot be answered because evidence is wrong or
unverifiable: leakage, duplicate/missing identities, role contamination,
wrong masks/fees/action space/simulator, failed D1 exposure matching after
replay, a significant positive shuffled-control artifact, broken hashes, or a
producer metric the independent auditor cannot reproduce.

No model-quality conclusion is allowed.

### 9.2 `insufficient_evidence`

Use this when machinery is valid but precision is inadequate. Examples:

- pooled ESS below 100 or any fold ESS below 20;
- DeltaQ MDE above 0.05;
- a required adjusted interval includes zero and also includes +0.05;
- valid outcome-blind matching attempts cannot create a sufficiently matched
  random control; or
- a required seed/fold/stratum denominator is too small to settle the route.

A profitable point estimate may still be insufficient. It is neither a pass
nor proof of no signal.

### 9.3 `no_genuine_entry_signal`

Use this when evidence is valid and precise enough, but the balanced claim
fails. For a required contrast, an adjusted interval with lower bound at or
below zero and upper bound below 0.05 rules out the minimum meaningful effect.
The same route applies to a sufficiently evidenced fold, seed, metric, regime,
G4, daily-stop, ruin-floor, or $4 stress failure.

Absolute profit cannot override this result.

### 9.4 `entry_evidence_pass`

Every integrity, pooled, fold, seed, metric, stratum, D1, MDE, fee-stress, G4,
daily-stop, and ruin requirement must pass. This freezes an entry candidate
for later graph work. It does not create a complete trader and does not earn
paper readiness.

## 10. Owner-Only Tripwires

Tripwires are never optimized and never auto-pass or auto-fail a candidate.
They pause the graph for owner judgment.

### 10.1 Four-bucket profile

Using fee-adjusted return on premium:

| Bucket | Frozen boundary |
|---|---|
| Big win | return >= +40% |
| Small win / scratch | -5% <= return < +40% |
| Small loss | -30% < return < -5% |
| Big loss | return <= -30% |

Route `profile_tripwire_owner_decision` when:

- big-loss share is strictly above 4% and there are at least three such
  trades; or
- with at least 50 trades, big-win share is below 2%.

Lower counts are reported as insufficient profile evidence. The Charter's
18/73/8/about-2 shape remains a north star, never a loss or threshold target.

### 10.2 Harvest ratio

On identical frozen entries:

```text
harvest ratio =
  sum(realized fee-adjusted PnL)
  / sum(positive peak available fee-adjusted PnL)
```

Only trades with positive peak available PnL enter this ratio. Compare the
learned exit with its floor on and off.

With at least 50 eligible trades and 20 sessions, route
`harvest_tripwire_owner_decision` only when all hold:

- floor-on ratio falls at least 0.10;
- floor-on / floor-off is below 0.75; and
- the paired session-block 95% upper bound for floor-on minus floor-off is
  below zero.

This tripwire belongs to lifecycle freeze. It is defined now so the later loop
cannot choose a friendlier rule after seeing its exits.

## 11. Reproducibility Law

Every reported number must be recomputable from hashed inputs by a separate
auditor.

The producer packet must contain hashes for authority, graph, labels, census
citations, FT2-08/10 contracts, multiplicity ledger, decision/session/contract
ledgers, models, calibrators, composer, controls, simulator/fees, and bootstrap
seed material.

The independent auditor writes to a different immutable directory and:

1. rehashes every input;
2. rebuilds decision and session metrics from source ledgers;
3. regenerates all bootstrap indices from the frozen seed rule;
4. recomputes intervals, p-values, G4, profile, and terminal route; and
5. reports any mismatch.

A producer-derived metric that is merely copied is not evidence.

## 12. No-Order Shadow Sufficiency

Before FT2-95 may even ask for paper authorization, the frozen complete Full
Trader must accumulate at least:

- 15 complete no-order live sessions;
- 3,000 flat-state decisions;
- 30 model ENTER intents across at least five sessions;
- 100 open-state HOLD/EXIT decisions; and
- one session in the training-defined top volatility tercile.

Required paired agreement:

| Behavior | Point minimum | One-sided 95% Wilson lower |
|---|---:|---:|
| WAIT vs ENTER | 0.99 | 0.98 |
| Exact contract when both ENTER | 0.98 | 0.95 |
| HOLD vs EXIT | 0.99 | 0.98 |
| Action mask / block reason | 0.995 | reported |

Each complete session must have at least 0.98 WAIT/ENTER agreement. Floor,
forced-flat, and daily-stop actions must agree 100% whenever observed. Every
stale, delayed, reconnecting, missing-minute, or incomplete-ladder boundary
must take the frozen safe action. Every decision must finish before the next
minute boundary. Broker/order endpoints remain untouched.

Low counts mean `insufficient_shadow_evidence`. Sufficient counts with failed
agreement mean `shadow_transfer_fail`. Broken identity, causality, pairing,
feed, or reconstruction means `invalid_shadow_evidence`.

A shadow pass permits an owner authorization request. It does not authorize
paper orders.

## 13. Synthetic Worked Evaluation

All numbers below are invented fixtures. They are not market statistics and
cannot support a trading claim.

### 13.1 Synthetic candidate A: pass

Assume the independent auditor reproduces:

```text
Fold DeltaQ vs P5:     +0.070, +0.058, +0.064, +0.051, -0.018
Fold DeltaQ vs random: +0.095, +0.082, +0.088, +0.075, +0.011
Pooled adjusted interval vs P5:     [+0.009, +0.081], p=0.021
Pooled adjusted interval vs random: [+0.030, +0.111], p=0.004
Pooled DeltaU: +0.020
Quality-family points: +0.030, +0.020, +0.010, +0.040, -0.003
Per-fold ESS: 31, 29, 30, 27, 25; pooled ESS 142
DeltaQ MDE: 0.041
```

The fifth-fold loss is bounded: `0.018 <= 0.5 x pooled 0.045`, its interval
`[-0.061,+0.025]` includes zero, and its preregistered explanation is regime
mix. All three seeds are positive pooled and have at least four positive folds.
The slightly negative fifth family is not significantly harmful.

The strong shuffle does not beat its matched random control; D1 matching
passes. At $3, neutral-lifecycle PnL is $18,000 and drawdown is $12,000,
Calmar 1.50. Every fold stays above $6,200. The same safety gates pass at $4.
Big-loss share is 3% and big-win share is 12%, so no profile tripwire fires.

Result: `entry_evidence_pass`.

### 13.2 Synthetic candidate B: insufficient

Assume:

```text
Pooled adjusted DeltaQ interval vs P5: [-0.018, +0.062]
Pooled adjusted DeltaQ interval vs random: [-0.011, +0.071]
Per-fold ESS: 18, 21, 20, 17, 19; pooled ESS 88
DeltaQ MDE: 0.071
```

The intervals contain both zero and the meaningful +0.05 effect, ESS is below
the frozen minima, and MDE is too large. The candidate may even have positive
absolute PnL, but the experiment cannot tell whether the model helped.

Result: `insufficient_evidence`, not failure and not pass.

### 13.3 Synthetic candidate C: valid, precise fail

Assume:

```text
Fold DeltaQ vs P5: +0.010, +0.000, -0.010, +0.020, +0.000
Pooled adjusted interval vs P5: [-0.012, +0.028]
Pooled adjusted interval vs random: [-0.008, +0.031]
Per-fold ESS: 30, 28, 29, 31, 27; pooled ESS 145
DeltaQ MDE: 0.032
```

The valid, adequately powered experiment rules out the contract's +0.05
meaningful effect and fewer than four folds are positive. Safety passing would
not manufacture entry skill.

Result: `no_genuine_entry_signal`.

### 13.4 Synthetic invalid packet

Assume a positive candidate but its random control has moneyness total
variation `0.08`, above the signed `0.06` bound. The comparison is not
exposure-matched.

Result: `invalid_evidence`. No pass, insufficiency, or no-signal story may be
claimed until the control is repaired.

## 14. Scope Receipt

This node performed design work only:

- no model training or fitting;
- no market-session statistic beyond cited FT2-05 census rows;
- no outer validation or protected holdout access;
- no recorder, broker, paid-data, GPU, runtime, promotion, or paper activity;
- no modification of FT2-08, FT2-10, signed contracts, or Graph V2; and
- no start of FT2-20.

Highest allowed claim:

> The evidence-statistics contract is frozen; the three-contract set is complete and ready for parallel design review.
