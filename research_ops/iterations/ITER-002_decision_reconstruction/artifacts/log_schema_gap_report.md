# Log Schema Gap Report

## Verdict

`schema patch required`

## Primary Gaps

- Full candidate feature rows are not logged for every decision.
- Raw quote timestamps and quote age reconstruction fields are not logged for every decision.
- Artifact references sufficient to bind model/scaler/manifest state are not logged per decision.
- Protocol101 logits, wait logit, and rejected candidate logits are not consistently logged.
- Guard inputs are not logged in enough detail to independently rerun every guard decision.

## Evidence Counts

- Decision rows inspected: `3495`
- Insufficient decision rows: `3495`
- Broker endpoint rows: `0`

## Required Schema Patch

For every live/paper/no-order decision, log candidate set, feature vector or hash plus schema version, raw quote timestamp fields, model artifact references, model logits, selected action, risk-gate inputs/result, account state, and order/fill outcome where applicable.
