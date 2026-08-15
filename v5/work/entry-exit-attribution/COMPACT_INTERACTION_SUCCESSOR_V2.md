# Compact interaction successor V2 — signable scope-widening packet

**Status: preconditions pass; the owner signed the exact scoped re-ruling on 2026-08-14. No fit or
evaluation was run as part of signing.**

This packet supersedes `COMPACT_INTERACTION_SUCCESSOR_V1.md` for authorization purposes. V1 remains on
disk as history, but its asserted architecture-family membership and missing selector-attainability
guard make it non-signable.

## The two preconditions now pass

### 1. Canonical architecture and computed count

`compact_interaction_entry` is registered in both the governance architecture map and the canonical
model builder. The value **48** is no longer transcribed from the compact module or prose. Both V2
declarations carry the result of:

`computed_parameter_counts()['compact_interaction_entry'] == 48`

The computation builds the model at dimensions derived from the live tensorizer and counts its trainable
tensors. A test independently requires the registry count, the directly built compact count and 48 to be
identical. Forty-eight fits the conservative design-effect range of **29–50 parameters**; that range must
be reported beside any eventual result.

### 2. Selector attainability is enforced as a defect class

The policy gate now has a reusable `SelectorAttainability` contract. An absolute threshold at or above a
declared target or prediction clip ceiling is refused before fit authorization. Relative action-value
rules must carry a finite built-model witness that strictly fires the same comparison used in inference.
Both the trainer and evaluator pass that proof through `assert_fit_permitted`; the executable selector
also rechecks it.

For `serial_action_advantage_120m`:

- target transform: `Q_usd / 1000.0`, with **no clamp or clip**;
- model output: linear and **unclipped**;
- selector: no absolute threshold and no rank; ENTER must be strictly greater than
  `max($0, predicted WAIT)`;
- canonical built-model witness: predicted ENTER **$1,000**, predicted WAIT **$0**, so the selector fires;
  and
- a target-loss test proves values far outside the old 30-point magnitude ceiling remain unchanged.

The Job 39 defect is now refusable as a class: a threshold of 30 against a clipped ceiling of 30 fails in
the gate by test.

## Sealed V2 declarations

- Fit: `ACTION_VALUE_FIT_DECLARATION_V2.json`
  - self-hash: `86fb0ef30b251d298661b8a59448570c045e3e7a4d0466955fd866446bbb7ab3`
- Evaluation: `ACTION_VALUE_EVALUATION_DECLARATION_V2.json`
  - self-hash: `68b520ea0785ae5866843101d2f57c9d66f86f05ab1c0683d1cffb376108b3e2`

V2 preserves the already-reviewed downstream law: one trade per session; structural $0 WAIT floor;
mid-to-mid existence first; no executable economics if that mean is non-positive; identical shuffled
fit; outcome-blind composition control; corrected 649-family inference; 4/5 chronological signs; and
the $500 ticket/loss constraints for a $10,000 account. V1 declarations remain immutable history and are
superseded, not rewritten.

Both V2 entry points were invoked. They passed declaration integrity, canonical parameter count and
selector-attainability preflight, then stopped at the current signed-scope boundary. No fit, optimizer
output, prediction, economic result or evidence directory was created.

## Job 39 reopening disposition

The original V4 zero-trade result did **not** spend the reopening. Its threshold equaled its clipped-label
ceiling, so its immutable correction receipt correctly records `INCONCLUSIVE_SPECIFICATION_DEFECT` and
`reopening_spent: false`.

The corrected V5 rank policy did produce a measured primary result and failed a precommitted kill
condition. Its receipt records `NEGATIVE_CONTROL_OR_CHRONOLOGY_KILL_FAILED`. Therefore:

> **The Job 39 `itm_depth_magnitude` reopening is spent and closed by corrected V5. V4 did not spend it.**

The proposed action-value experiment is not a continuation of unspent Job 39 authority. It is a new,
explicit scope widening and must be signed as such. No re-labeling or neighboring Job 39 retry is implied.

## Exact scope proposed for owner signature

Permit exactly one real fit and its identically configured shuffled-label null for:

- architecture: `compact_interaction_entry`;
- canonical count: `computed_parameter_counts()['compact_interaction_entry'] == 48`;
- label: `serial_action_advantage_120m`;
- horizon: 120 minutes;
- corpus: `causal_day_quote_243`;
- fit declaration self-hash:
  `86fb0ef30b251d298661b8a59448570c045e3e7a4d0466955fd866446bbb7ab3`;
- evaluation declaration self-hash:
  `68b520ea0785ae5866843101d2f57c9d66f86f05ab1c0683d1cffb376108b3e2`; and
- the existing seven kill conditions plus the declared executable-profitability, 4/5 chronology and
  $10,000-account risk requirements.

All other architectures, labels, horizons, corpora, operating points and retries remain refused. A
non-positive measured midpoint mean spends this new scope immediately. Any later executable, control,
chronology, confidence or risk failure closes the branch without a neighboring retry. The conservative
29–50 parameter budget is a mandatory caveat even if the result is positive.

## Owner decision

Owen Heidenreich signed the exact scope above on 2026-08-14 in
[`CAUSAL_DAY_ACTION_VALUE_SCOPE_RERULING_2026_08_14.md`](../../governance/CAUSAL_DAY_ACTION_VALUE_SCOPE_RERULING_2026_08_14.md).
No fit or evaluation was run as part of signing.
