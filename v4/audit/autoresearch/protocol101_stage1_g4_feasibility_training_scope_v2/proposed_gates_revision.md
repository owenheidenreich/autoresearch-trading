# Proposed Stage-1 G4 / Holdout Revision

Status: **DRAFT FOR OWNER SIGNATURE**.

This draft is generated from `Protocol101Stage1G4FeasibilityV2`. It does not change the signed
gate document by itself.

## Evidence

- Current G4 nominal cap on $10k starting cash: `$2,500`.
- Random-policy mean drawdown across folds: `$20,493`.
- Random-policy max drawdown across folds: `$38,245`.
- Synthetic oracle max drawdown: `$0`.
- Cheap synthetic oracle max drawdown: `$0`.

## Proposed Text Replacement

Replace the holdout sentence:

> fee-adjusted PnL > 0, max DD <= $1,500, and result within the 90% bootstrap CI implied by CV

with:

> fee-adjusted PnL > 0; max strict-serial drawdown must satisfy the owner-signed
> G4 rule active for this candidate generation, with the same fee/stress and
> one-account semantics used in CV; and the result must fall within the 90%
> bootstrap CI implied by CV. A holdout result wildly above expectations remains
> an audit trigger, not a celebration.

## Rationale

The `$1,500` absolute drawdown cap is stale. It conflicts with the already
signed relative G4 direction and would make the protected holdout auto-fail
otherwise valid candidates for a retired rule.
