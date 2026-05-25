# Feature Parity Report

Decision: `replay metrics not yet usable`

Reason: logs do not contain paired replay/live feature, candidate, and logit evidence

## Counts

- Files read: `15`
- Feature comparison rows: `13592`
- Logit comparison rows: `1103`
- Candidate-set comparison rows: `906`
- JSON errors: `0`

## Interpretation

This diagnostic does not load models or rerun replay. It asks whether existing logs contain paired evidence needed to compare live Protocol051-to-Protocol101 features against replay for the same market state. When replay references, full live features, candidate tensors, or logits are absent, the result is not comparable.

## Required Evidence For Pass

- Live normalized market state and option ladder snapshot.
- Replay-built market state for the same timestamp/contracts.
- Candidate set identifiers before and after Protocol051 filtering.
- Protocol051 surface scores and Protocol101 feature rows or hashes.
- Protocol101 scaled tensor hash, wait logit, candidate logits, selected action, and threshold.
