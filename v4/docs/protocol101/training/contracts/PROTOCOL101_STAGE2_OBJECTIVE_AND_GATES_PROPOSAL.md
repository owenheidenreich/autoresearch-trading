# Protocol101 Full Trader Learned-Lifecycle Contract - OWNER DECISION DRAFT

Status: **UNSIGNED OWNER-DECISION DRAFT**

Reconciled: **2026-07-28**

This document amends the former Stage-2 proposal in place. It removes obsolete
corpus counts, simulator assumptions, inherited Stage-1 verdicts, and unresolved
July 7 questions.

It does not change any signed Stage-1 contract until the owner signs it. It
does not authorize lifecycle training, selection, holdout access, broker
contact, paper submission, or promotion.

## Owner-Authorized Exploratory Stage-0 Exception

On 2026-07-28, after policy-neutral selector Stage-0 attempt002 returned
`stop_no_preliminary_signal`, the owner directed the program to keep
deterministic P5 contract selection and proceed with the next decision.

That instruction authorizes one narrow offline exception:

- deterministic P5 VWAP-side nearest-ATM entries may be used solely as the
  standardized-entry control for
  `exploratory_non_promotable_stage0` HOLD/EXIT feasibility;
- the pilot must use training-side sessions only, nested chronological roles,
  the accepted simulator-v5 and identity machinery, a causal lifecycle
  firewall, and one-step executable-bid hold advantage;
- the pilot may not establish P5 entry alpha, select or promote a lifecycle
  model, run the full lifecycle campaign, integrate learned entries, access
  G9 or the protected holdout, inspect recorder or shadow outcomes, contact a
  broker, submit paper orders, or modify runtime state; and
- a positive result earns only a return to the owner to sign or reject the
  complete learned-lifecycle contract before any full campaign.

This exception records the owner's bounded instruction in the current task.
It does not sign the checklist at the end of this document or authorize any
other lifecycle training.

## Inherited Authority

Until this draft is signed, the following documents remain binding:

- [Full Trader program](../README.md)
- [Trader Charter](PROTOCOL101_TRADER_CHARTER.md)
- [Stage-1 objective and G1-G9](PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md)
- [G4 and holdout revision](PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md)
- [G8 calibration revision](PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md)
- [Stage-1 H0-H3 design](PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md)
- [Signed regimen repair amendment](PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md)
- [Scoped synchronization decision](../../synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md)

Signing this draft adds the learned-lifecycle requirements. It does not erase
the entry, synchronization, simulator, multiplicity, or integrity controls
inherited from those authorities.

## Owner Intent

Protocol101 is not finished when it learns only where to enter.

The intended product is one SPXW 0DTE trader that:

```text
WAITs when no entry is justified
  -> BUYs one call or one put
  -> HOLDs while continuing has positive expected value
  -> EXITs when selling now is preferable
```

Learned exits are mandatory before IBKR paper trading. Fixed stop, target, and
time policies are Stage-1 measurement scaffolds and safety fallbacks, not the
final lifecycle philosophy.

The charter remains binding: one contract, no martingale, no overnight
position, a 5% session-starting-equity daily stop, account survival, fee-aware
economics, many harmless scratches, rare large losses, and room for uncommon
large winners.

## Relationship To Stage 1

Stage 1 asks whether the model can identify worthwhile entry opportunities.
It scores each eligible contract against seven fixed exit-label shapes so that
entry hypotheses can be compared under frozen outcomes.

This lifecycle phase asks a different question:

> After a position has been opened, can a causal model decide each minute
> whether continuing to hold has more expected fee-adjusted value than selling
> now?

The final object is always the combined entry-plus-exit trader. Entry and
lifecycle components remain separately testable so improvements and failures
can be attributed honestly.

The existing 420 H0-H3 models and their old economic replay are historical
benchmarks only. The Full Trader campaign uses freshly fitted entry models
under independently accepted simulator-v5 machinery.

## Full-Campaign Preconditions

A full lifecycle dataset, campaign, or candidate-selection process may begin
only after:

1. Simulator v5 and the two-clock exit contract pass independent machinery
   acceptance.
2. The fresh entry campaign establishes either:
   - an independently accepted Stage-1 entry candidate; or
   - genuine entry signal with independent evidence that fixed exits are the
     binding loss-tail or drawdown limitation.
3. The lifecycle feature contract, label contract, standardized-entry control,
   folds, fees, stress grid, model variants, seeds, nulls, gates, and stopping
   rules are preregistered and hashed.
4. The lifecycle feature firewall and historical/live construction paths pass
   focused parity and leakage tests.
5. The protected holdout, recorder confirmation evidence, live shadow data,
   and paper results remain outside fitting and tuning.

Failure of a precondition stops the lifecycle run. It does not authorize an
agent to loosen the precondition.

The exploratory Stage-0 exception above is the sole permitted activity before
these full-campaign preconditions pass.

## Runtime Action Contract

When the account is flat, the entry component may:

- `WAIT`;
- `BUY_ONE_CALL`; or
- `BUY_ONE_PUT`.

When one position is open, entry scoring is suspended and the lifecycle
component may:

- `HOLD`; or
- `EXIT_NOW`.

`EXIT_NOW` means sell the one open contract using the current executable-bid
assumption in historical evaluation and the guarded sell-order path in paper
trading.

The trader cannot scale in, scale out, reverse, add a second contract, sell
premium, or hold overnight. Forced flat remains an unconditional safety rule.

## Simulator And Exit Clocks

All lifecycle evidence uses simulator v5 and the signed two-clock contract:

- `label_source_exit_quote_time_ns` identifies the causal quote that prices
  the exit.
- `label_realized_exit_time_ns` identifies when the serial position releases
  occupancy and capital.

Threshold and no-bid exits normally share one timestamp. Deadline and
forced-flat exits may use the latest causal quote at or before the deadline
while occupancy continues through the deadline.

The simulator must:

- preflight the complete stream before replay;
- use the realized occupancy exit rather than synthetic maximum hold;
- preserve one-account cash continuity and the session-starting-equity daily
  stop;
- apply the round-trip fee exactly once;
- enforce affordability and one open position;
- fail closed on duplicate or missing identities; and
- reject simulator-v4 or one-clock artifacts.

## Causal Lifecycle Features

The lifecycle feature set may contain only information observable for the open
position at the current decision minute.

Candidate families for preregistration are:

- elapsed minutes since entry and minutes to forced flat;
- entry ask, current executable bid/ask/mid, and current spread state;
- current unrealized dollar PnL and return on entry premium;
- running MFE and MAE through the current minute;
- giveback from MFE and time since MFE;
- one-, three-, and five-minute causal PnL or option-price velocity;
- the synchronized non-VIX market context authorized for entry;
- static contract geometry and moneyness;
- approved internally recomputed delta and gamma; and
- frozen entry-model score, threshold, and margin when their live
  construction is identical.

Current quote state is authorized here because it defines the executable exit
and open-position condition. This does not unquarantine raw quote
microstructure as entry alpha. Every quote-derived lifecycle feature must
receive source-drift, timing-jitter, and stale-quote stress.

Direct or derived features outside this list require a separately
preregistered parity and leakage amendment before results are inspected.

## Label Firewall

Future information may create supervised training answers but may never enter
the model-facing feature frame.

Forbidden model inputs include:

- future bid, ask, mid, return, or PnL;
- future best or worst PnL;
- future MFE or MAE;
- future exit timestamp or reason;
- oracle HOLD/EXIT action;
- `q_exit`, `q_hold`, `a_hold`, `a_exit`, or `a_switch`;
- future recovery, regret, save, runner, or decay labels; and
- any protected-holdout, recorder, shadow, or paper outcome.

The lifecycle loader must fail closed if a forbidden field, alias, duplicate
identity, role collision, or post-entry future timestamp appears in the
model-facing matrix.

## Supervised Learning Target

The primary lifecycle target is hold advantage:

```text
q_exit = fee-adjusted value of selling now at the executable bid
q_hold = fee-adjusted value of continuing for at least one more minute
a_hold = q_hold - q_exit
```

Future quote paths may be used to construct these labels only. They are not
present at inference.

The model predicts expected `a_hold` or an equivalent preregistered
HOLD-versus-EXIT action value:

```text
predicted hold advantage > frozen decision threshold -> HOLD
otherwise                                           -> EXIT_NOW
```

The decision threshold is selected inside training/validation roles only and
is frozen before outer-fold evaluation. It may not be tuned on outer test
folds, confirmation, holdout, recorder, shadow, or paper outcomes.

Win probability and win rate are diagnostics, never the optimized target.

## Experimental Sequence

### Phase A: Exit Isolation

Preregister one deterministic standardized-entry policy before lifecycle
results are inspected. The policy fixes entry timing, direction, contract
selection, affordability, and no-overlap behavior.

Run fixed exits and learned exits on exactly the same standardized entries.
This phase determines whether the lifecycle learner adds value independent of
entry-model skill.

The standardized-entry policy is an experimental control, not a candidate for
paper trading.

### Phase B: Learned Entries

Apply the same lifecycle design to the selected fresh entry model. Compare:

1. selected entries plus their best fixed exit;
2. the same selected entries plus the learned lifecycle; and
3. matched-rate random lifecycle decisions.

Only this combined entry-plus-exit replay can produce a Full Trader candidate.

### Phase C: Architecture Challenge

Train at least:

- a transparent HGB model over the current causal position-state snapshot; and
- a neural/sequence challenger that receives only the causal history actually
  available in live runtime.

Both candidates use identical session roles, folds, fees, stress, feature
families, and simulator semantics. A sequence model is inadmissible if
historical replay receives a full sequence while live inference receives only
one row.

Neither architecture wins by preference. Among hard-gate-eligible candidates,
the frozen selection metric is plain fee-adjusted strict-serial net PnL.
Learning curves and compute cost are reported, but no tunable utility may
replace the primary selection metric.

## Acceptance Requirements

A learned lifecycle can advance only if every applicable hard requirement
holds.

### Complete-System Economics

- The combined entry-plus-exit trader passes signed Stage-1 G1-G7 under
  simulator v5.
- G3 compares the complete trader with the frozen heuristic plus its best
  fixed exit on the same folds.
- G4 uses the signed Calmar floor and per-fold $5,000 equity floor.
- The 5% session-starting-equity daily stop, affordability, one-contract
  limit, and forced flat are active.

### Fixed-Exit Uplift

- Learned lifecycle fee-adjusted PnL exceeds the best preregistered fixed exit
  on identical entries when pooled.
- Learned lifecycle also exceeds that baseline on at least four of five outer
  folds.
- The comparison includes identical fees, stress, entry decisions, and serial
  opportunity cost.

### No-Skill And Multiplicity

- Uplift versus a matched-rate random-exit refit null has pooled z-score at
  least 3.0.
- The worst initial seed has z-score at least 2.0.
- All registered lifecycle variants participate in the signed campaign-level
  multiplicity control. Winner-only reporting is forbidden.

### Seed And Era Robustness

- At least three initial seeds are run.
- The worst seed passes G1, G2, G4, and the fixed-exit uplift requirement.
- No governed era is systematically negative without the signed
  `regime_bound_requires_owner_review` route.
- Trade frequency remains inside G7; an exit model cannot manufacture a pass
  by suppressing almost all completed trades.

### Calibration

G8 remains report-only when calibration controls no action.

If a probability or calibrated confidence controls HOLD/EXIT, threshold
abstention, routing, or another lifecycle behavior, an action-conditioned
calibration gate must be preregistered, smoke-tested, independently audited,
and owner-signed before that model is trained. The Stage-1 post-selection ECE
readout cannot be reused as a lifecycle gate without this amendment.

Direct action-value models must report predicted-versus-realized hold
advantage and threshold-margin stability even when they do not emit a
probability.

### Fresh Confirmation

After the complete candidate is selected and frozen, one never-used
confirmation seed must independently pass:

- G1, G2, and G4;
- fixed-exit uplift;
- the no-ruin backstop; and
- the feature, model, threshold, simulator, and source-hash checks.

Confirmation is spend-once for that candidate hash. It cannot tune or repair
the candidate.

## Mandatory Diagnostics

Every standardized-entry and learned-entry lifecycle report includes:

- four-bucket outcome distribution from the Trader Charter;
- fee-adjusted PnL and return on premium;
- max drawdown, Calmar, minimum equity, and worst day;
- mean and tail loss on losing trades;
- big-loss incidence;
- tail-win capture and PnL concentration;
- realized PnL divided by peak available PnL, or harvest ratio;
- MFE giveback and premature-exit opportunity cost;
- underwater duration;
- HOLD/EXIT frequency, threshold margins, and action churn;
- time-in-trade and time-of-day exposure;
- call/put, moneyness, premium, and regime exposure;
- skipped entries while capital is occupied;
- fee, spread, fill, timing-jitter, stale-quote, and feature-noise stress;
- per-fold, per-seed, per-era, and pooled results; and
- every failure, invalid row, and missing-path count.

These diagnostics explain the trader. They do not become post-hoc optimization
targets unless a future global amendment freezes them before a new campaign.

## Protected Holdout

The proposed Full Trader rule is:

- keep the protected holdout closed throughout entry and lifecycle
  development;
- open it once only after the complete entry-plus-exit system and every
  dependency hash are frozen;
- apply the signed G4/holdout economics under identical simulator-v5, fee,
  stress, and one-account semantics;
- treat an implausibly strong result as an audit trigger; and
- burn the candidate on failure.

This timing rule becomes binding only when this contract is owner-signed. It
does not silently rewrite the currently signed Stage-1 holdout language.

## Historical/Live Transfer And Paper Boundary

Passing offline gates earns at most:

> Full Trader offline candidate eligible for paper-readiness validation.

It does not earn paper readiness.

Before paper authorization, the frozen candidate must pass:

1. Candidate-specific historical and IBKR decision-shadow comparison.
2. Entry-feature, candidate, score, action, and lifecycle-state reconstruction.
3. HOLD/EXIT action and exit-timing comparison under the same live feature
   history interface used in training.
4. No-order live shadow with stale data, reconnect, missing-minute,
   incomplete-ladder, and forced-flat behavior.
5. Guard, observability, rollback, and owner-authorization review.

Only a separate owner decision may authorize guarded IBKR paper orders.
Real-money paths remain out of scope.

## Stopping And Routing

- If standardized entries show no learned-exit skill, stop and report
  `exit_skill_not_observed`; do not hide failure by combining with a strong
  entry model.
- If exit isolation passes but the learned-entry combination fails, report
  `entry_exit_integration_failed` and attribute the interaction before changing
  architecture.
- If only a few regimes or sides work, report
  `regime_bound_requires_owner_review`; do not silently narrow the product.
- If a neural challenger fails to beat the eligible HGB baseline, keep HGB.
- If both fail, redesign labels, features, or strategy assumptions before
  adding capacity.
- If any leakage, identity, simulator, role, or hash defect appears, mark all
  affected economics invalid and return to independent machinery acceptance.
- If the complete candidate passes, freeze it and proceed only to fresh
  confirmation.

## Owner Decision Checklist

- [ ] Learned HOLD/EXIT behavior is mandatory before paper trading.
- [ ] Standardized-entry exit isolation precedes learned-entry evaluation.
- [ ] The Full Trader campaign uses fresh Stage-1 entry fits; the old 420
      models remain benchmarks only.
- [ ] HGB and neural/sequence lifecycle candidates compete under identical
      evidence, with the best eligible model selected.
- [ ] Hold-advantage labels and the future-information firewall are approved.
- [ ] The complete system must satisfy the economics, fixed-exit uplift,
      random-exit null, robustness, confirmation, and diagnostic requirements.
- [ ] The protected holdout remains closed until the complete trader is frozen.
- [ ] Historical/IBKR decision shadow and no-order live shadow remain mandatory
      before a paper-order authorization decision.

Owner signature: ______________________________

Date: ______________________________
