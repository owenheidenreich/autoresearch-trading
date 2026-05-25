# Required Logging Patch RFC

Iteration ID: `ITER-001_quote_age_truth`

## Research Need

Existing logs cannot prove live quote age truth. A future CEO-approved observability patch should log enough causal timestamp evidence to recompute quote age for every selected and rejected candidate.

## Required Fields

- `quote_timestamp`
- `quote_timestamp_source`
- `received_timestamp`
- `decision_timestamp`
- `quote_age_ms`
- `recomputed_quote_age_ms`
- `quote_age_delta_ms`
- `candidate_contract_id`
- `candidate_rank`
- `selected_contract_id`
- `guard_passed`
- `guard_reason`

## Constraints

- Logging-only change.
- No runtime flag mutation.
- No broker behavior change.
- No model loading beyond the existing runtime path.
- No threshold tuning.
- No paper-submit behavior change.

## Current Evidence

- Diagnostic verdict: `unknown`.
- Persisted quote-age rows found: `1`.
- Trustworthy quote-age rows found: `0`.

## Acceptance Criteria

A later verifier must be able to recompute quote age from raw timestamp fields and match persisted age within declared tolerance for selected and rejected candidates before paper-submit quote freshness can be trusted.
