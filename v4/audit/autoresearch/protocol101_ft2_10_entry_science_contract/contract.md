# Protocol101 FT2-10 Entry Science Contract V4

Status: `scoped_final_round_repaired`

Authority:
`3c7a0aaf2334ae7f04090fb3e67eb2db16591c40545bcf3c09e78c04f0640033`

This packet is the scoped final-round entry-component design.
It freezes a research contract only. It performs no training, fitting,
threshold tuning, protected-data access, broker action, or re-review.

## 1. Causal entry law

The sole shared intent/fill authority is:

```text
v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/intent_fill_recheck_law.json
sha256 5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0
```

At completed decision minute `t`, candidate, P5, and every live-style control
assemble their legal action set from information through `t` only. The price is
the last known executable ask `A_t`. The decision-time cost is:

```text
intent_cost_cents = A_t_quote_cents * 100 + active_round_trip_fee_cents
daily_budget_cents = floor(session_start_equity_cents * 5 / 100)
```

D48 requires `intent_cost_cents <= daily_budget_cents`. D49 requires realized
session loss plus that intent cost to be within the same budget. The actual
`A_(t+1)`, its presence, freshness, fill outcome, and every future path or
label field are forbidden during composition.

After one exact identity is selected, only that identity is rechecked at
`t+1`. Identity/freshness, the 100-cent premium floor, affordability, D48, and
D49 must pass using the actual `A_(t+1)`. Failure opens no position, charges no
premium or fee, changes realized PnL by zero, permits no substitute, and allows
the next decision no earlier than `t+2`.

The two causal fee paths use 300 and 400 cents. Their computed one-dollar floor
costs are 10,300 and 10,400 cents. At every flat decision, soft close becomes
permanent for the active policy/fee-path session when the current exact ladder
contains no otherwise-eligible contract with `A_t >= 100` cents whose D48 and
D49 intent masks pass. A shared literal threshold is forbidden.

## 2. Model-free realized-label audit composer

The sole action-conditioned target producer is:

```text
v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/realized_label_audit_composer_spec.json
sha256 d833297dbaf2a547d8246cdf3cf404dea0d8ff4234026ceac37797e462a8422f
```

The RLAC executes once over the complete label-eligible intent population
before any model head is fit. It uses only frozen label families, fit-role
reference distributions, and deterministic anchor-specific thresholds. It
produces a WAIT target for every label-complete governed flat minute and a
nonnegative per-contract regret target for every label-complete intent-eligible
contract on every included minute, including RLAC WAIT minutes. A runtime
composer, fitted model output, composer-selected
contract, or model-selected action may neither create a target nor select its
membership. Calibrated heads freeze before the runtime composer runs, so there
is no target/composer fixed point and no iterative relabeling.

## 3. Targets and empirical CDFs

The target families and label firewall remain in `forecast_heads.json`.
Continuous, bounded, count, and empirical-CDF reference values for finite
`hN` require the complete uncensored `N`-mark window. A finite value from a
near-close shortened or censored window is excluded. Remaining-session values
require the complete grid through exact 15:55 ET.

Right-censored time-to-first-profit rows are retained only for the registered
discrete-survival likelihood with their censoring indicator. They are not
converted into finite event values for a continuous CDF.

Each empirical CDF remains nested by fold, fit block, metric, horizon, unit,
decision-time premium band, and phase. Sessions receive equal weight and rows
within a session share that session weight. Midrank ties, interpolation,
same-band fallback, minimum support, and fail-closed insufficiency are frozen
in `objective_spec.json`.

## 4. Serial component population

Every candidate and comparator traverses the same governed minute grid but
mutates only its own integer-cent causal ledger. WAIT, pending, failed-fill,
occupied, soft-closed, daily-stopped, and zero-trade states remain in their
denominators.

A successful component entry is held without a learned exit to the exact
15:55 ET bid-or-zero forced-flat boundary. It therefore permits at most one
successful trade per session. This neutral game evaluates the entry component;
it does not claim to reproduce a later learned lifecycle with earlier exits
and re-entry. The assembled learned entry/lifecycle trader must be evaluated
again on complete combined-system ledgers at FT2-80.

## 5. Composer

`composer_spec.json` is authoritative. Its order is:

1. Complete-ladder and time-`t` physical/safety intent mask.
2. Causal state and horizon checks.
3. Registered quality screen, including the mandatory alpha-zero arm.
4. Conservative upside rank.
5. Positive-after-fee check.
6. Directional substitute-cluster uncertainty check.
7. Selected exact-contract q90 regret magnitude check.
8. Mandatory action-conditioned calibration gate.
9. Deterministic exact-identity tie-break.

The substitute cluster is same expiry and right within plus or minus 10 strike
points, or 10,000 milli-points, of the selected contract. It prevents a smooth
three-or-more-strike directional surface from creating an artificial
near-duplicate WAIT comparison.

The same-final-model conformalized q90 normalized selected-contract regret
upper bound is a hard action constraint, not a report. ENTER requires a finite
bound `<= 0.10`; missing, nonfinite, or larger values emit WAIT. The regret
calibration and coverage requirements remain separately mandatory.

## 6. Canonical matched-random comparator

The sole generator source is:

```text
v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/matched_random_generator_spec.json
sha256 8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754
```

It freezes all eight seed IDs, canonical JSON key, label-blind
candidate-specific intent budget, time-`t` legal-pair population,
selected-only `t+1` recheck, no-redraw law, post-replay tolerances, incomplete
attempt treatment, and fixed equal-weight aggregate. No consumer may add a
candidate-specific seed namespace or a second random key.

## 7. Census v4 reconciliation

The current census receipt is:

```text
v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/receipt.json
sha256 ddc6167abdf763070416f5e729225ece97d594a9b21b64893b8d2bd1f25d6928
```

The census is a 45-session multi-trade hindsight/oracle design diagnostic. Its
guardrail activity and trade-rate tables are report-only. They cannot discard,
retain, rank, calibrate, select, or fail a component setting. They also cannot
project neutral-lifecycle trade count, because the component game has a hard
one-successful-trade-per-session maximum.

The v3-to-v4 impact note is:

```text
v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/v3_v4_impact.json
sha256 913f127563daf9e5dff25cb8da72b4c3cdefd7a06d7e5edf569ca3c736806c3e
```

## 8. Acceptance and graph boundary

Entry-component acceptance requires reproducible integrity, quality,
action-conditioned calibration, q90 regret action control, signed safety,
tail/MNAR review, and exact no-harm versus P5 on separate fee trajectories.
DeltaQ remains diagnostic and cannot substitute for serial-dollar evidence.

Candidate-specific historical/IBKR transfer is not an entry-component
acceptance precondition. A frozen component may carry
`transfer_not_yet_run`. Graph ordering is:

```text
FT2-91 pass
  -> FT2-92 executes candidate-specific historical/IBKR transfer measurement
  -> pass -> FT2-93 no-order live shadow
  -> fail -> STOP-CANDIDATE-REJECTED
  -> insufficient_evidence -> STOP-OWNER-DECISION
```

The exact FT2-92 emitted vocabulary is `pass`, `fail`, and
`insufficient_evidence`. Missing evidence and transfer failure both block every
later live activation but take their distinct legal graph edges. Neither
retroactively makes component acceptance depend on evidence produced later.

The highest FT2-10 claim is only that its repaired entry-science design is
complete for final re-review. It is not evidence of profitability, selection
eligibility, promotion, paper readiness, or real-money readiness.
