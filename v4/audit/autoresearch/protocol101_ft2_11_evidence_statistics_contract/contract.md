# Protocol101 FT2-11 Evidence And Statistics Contract V3

Status: `producer_repaired`

Authority:
`2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`

This is the final coordinated repair-attempt-2 statistical design. It freezes
future evidence treatment only. It performs no model fitting, new session
analysis beyond the separately authorized census v3, protected-data access,
recorder action, broker action, paid compute, simulator edit, or re-review.

## 1. Shared causal and comparator authorities

Every FT2-11 statistic consumes:

```text
FT2-08 intent/fill law
  v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/intent_fill_recheck_law.json
  sha256 5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0

FT2-10 canonical matched-random generator
  v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/matched_random_generator_spec.json
  sha256 8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754
```

No FT2-11 candidate-specific seed namespace exists. All eight attempts, every
incomplete or invalid attempt, and the fixed aggregate remain in multiplicity.

The constants are integer cents, multiplier 100, floor quote 100 cents, fee
paths 300 and 400 cents, computed floor costs 10,300 and 10,400 cents, and
`floor(session_start_equity_cents*5/100)` daily budget. Soft close is the
current-ladder existence predicate under the active fee path, not a fixed
remaining-budget shortcut.

## 2. Population and claim layers

FT2-10 owns the component population. Candidate, P5, and controls use separate
causal ledgers and retain WAIT, failed-fill, pending, occupied, soft-closed,
daily-stopped, and zero-trade sessions. A successful component entry is held to
exact 15:55 ET, so the neutral component game permits no more than one
successful trade per session.

Entry-component acceptance can freeze a component for lifecycle work only. It
does not establish selection eligibility, incremental edge, live parity,
promotion, paper readiness, or profitability. Only the fully assembled
entry/lifecycle trader at FT2-80 can make the separately registered combined
incremental-dollar comparison.

The FT2-05 census v3 multi-trade oracle activity is report-only planning
context. It cannot select, reject, calibrate, or project the neutral component.

## 3. Session dependence and bootstrap

The market session is the dependence cluster. Noncircular moving blocks of
length 3, 5, and 8 are sampled independently within each fold, with synchronized
indices across every registered hypothesis. No block wraps or crosses a fold.
Hard evidence uses the conservative envelope.

### 3.1 Observed and replicate standard errors

For fold `f`, ordered series `x_f`, raw size `n_f`, block length `L`, and
`K_f=n_f-L+1`, delete each contiguous `L`-observation block. Let
`theta_f,-j` be the retained mean and let `theta_bar_f,delete` be their mean.

```text
V_f = (n_f-L)/(L*K_f)
      * sum_j (theta_f,-j - theta_bar_f,delete)^2

w_f = n_f / sum_g n_g
SE_h = sqrt(sum_f w_f^2 V_f)
```

For every centered-null outer bootstrap replicate, apply that same delete-`L`
formula directly to each emitted resampled fold sequence, with the same fixed
raw-fold weights:

```text
SE_h_star_b = sqrt(sum_f w_f^2 V_f_star_b)
t_h_star_b  = centered_resample_mean_h_b / SE_h_star_b
```

This is not a nested bootstrap. Reusing observed `SE_h`, adding an epsilon,
using an analytic replacement, crossing folds, or wrapping is forbidden.
Every required fold must have `n_f>L` and at least two delete blocks.

### 3.2 Numeric fixture

For `[1,2,3,4,5]` and `L=2`, delete-block means are
`[4,10/3,8/3,2]`. Their average is 3, their squared-deviation sum is `20/9`,
and the factor is `3/8`. Therefore:

```text
V = 5/6
SE_h = sqrt(5/6) = 0.9128709291752769
t_h = 3/SE_h = 3.2863353450309964
```

For the centered outer resample `[-2,-2,0,0,2]`, the delete-block means are
`[2/3,0,-2/3,-4/3]`, so `SE_h_star_b` is again
`0.9128709291752769` and:

```text
t_h_star_b = -0.4/SE_h_star_b = -0.4381780460041329
```

`bootstrap_spec.json` is the machine authority and the consistency checker
recomputes this fixture.

## 4. MDE and terminal decisions

MDE uses paired candidate-minus-comparator serial-dollar session deltas,
statistic-specific marginal SD, Geyer initial-positive-sequence ESS/LRV, the
actual ordered multiplicity family, and the same 3/5/8 block-power envelope
with replicate-specific `SE_h_star`.

Census v3 variance is planning context only. A future training-side pilot must
measure its own paired variance, LRV, ESS, activity, analytic MDE, and
block-power MDE before any GPU tranche. No census statistic authorizes spend.

Terminal precedence is:

```text
invalid_evidence
resource_owner_decision_required
owner_decision_required
insufficient_evidence
no_genuine_entry_signal
entry_component_freeze_pass
```

The three invented terminal fixtures are executed by the final consistency
checker. Adjusted lower and upper bounds use the mechanically defined
studentized max-T procedure.

## 5. Action-conditioned evidence

WAIT reliability, expected selected-contract regret calibration, q90 regret
coverage, and all registered denominators remain mandatory. In addition,
every ENTER must have a finite same-final-model conformalized q90 normalized
selected-contract-regret upper bound `<=0.10`. Missing, nonfinite, or larger
values emit WAIT. Calibration accuracy does not substitute for this magnitude
constraint.

## 6. MNAR alternate-label procedure

The full-loss no-bid view is primary. The no-bid-excluded view is sensitivity
only. The alternate view must independently refit every label-dependent item:

- all path heads;
- nested empirical CDFs;
- continuous, binary, survival, count, WAIT, and regret calibrators;
- guardrail and composer thresholds;
- WAIT-probability and expected/q90-regret action heads;
- inner selection and the frozen ensemble.

Features, masks, architectures, grids, folds, embargoes, roles, seeds, losses,
comparators, serial population, fee paths, multiplicity, tie-breaks, and the
terminal resolver stay identical. The label view is the only changed input.
Cross-view artifact reuse and tuning on the primary outcome are forbidden.

The selected identity, component terminal, every selected-lineage pairwise
order, and the 0.95 Spearman floor must remain stable. This goal freezes that
future procedure; it does not fit either view.

## 7. Source-transfer graph topology

Candidate-specific historical/IBKR transfer is not an entry-component
acceptance precondition. The graph order is:

```text
FT2-91-PROTECTED-HOLDOUT pass
  -> FT2-92-IBKR-DECISION-SHADOW
       executes the candidate-specific transfer measurement
       and emits pass / insufficient / fail / invalid
  -> FT2-92 pass
  -> FT2-93-NO-ORDER-LIVE-SHADOW
```

A component may reach FT2-92 with `transfer_not_yet_run`. A transfer pass is
mandatory to leave FT2-92 for FT2-93 and every later live activation. Nothing
in this packet authorizes new recorder collection or orders.

## 8. Phase-F hard-reporting floor

The synthetic AR coverage design covers 45-session fold shapes. Therefore,
hard Phase-F/no-order-shadow bounds and pass/fail gates require at least 45
complete sessions. With 1 through 44 complete sessions, only point estimates
and raw denominators may be reported; the terminal is
`insufficient_shadow_evidence`. No activation claim is permitted below the
covered floor.

## 9. Highest packet claim

The FT2-11 evidence design is mechanically complete for final re-review. This
is not empirical evidence that any model, component, or trader passes it.
