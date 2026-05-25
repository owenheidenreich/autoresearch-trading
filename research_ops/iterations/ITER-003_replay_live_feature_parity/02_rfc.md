# Experiment RFC

Iteration ID: `ITER-003_replay_live_feature_parity`
Assumption ID: `A004`
Title: Replay/live feature parity

## Research Question

Does live Protocol051-to-Protocol101 feature construction match replay for the same market state?

## Null Hypothesis

Replay metrics are not usable for live trust unless paired replay/live candidate sets, feature rows, and logits can be compared for the same timestamp/contracts.

## Required Inputs

- Existing JSONL logs under `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading`.
- No model binaries.
- No paid data.
- No broker APIs.

## Required Outputs

- `feature_parity_report.md`
- `feature_diff.csv`
- `logit_diff.csv`
- `candidate_set_diff.csv`
- `feature_parity_summary.json`

## Formulas

This diagnostic is evidence-availability based:

- `status = comparable` only when both live and replay evidence exist for the same comparison scope.
- `status = not_comparable` when live value, replay reference, or diff is missing.

Comparison scopes:

- Protocol051 surface scores
- Protocol101 candidate features
- Quote/market state
- Model artifact references
- Protocol101 logits
- Candidate set membership

## Columns

- `feature_diff.csv`: scope-level live/replay/diff availability.
- `logit_diff.csv`: live score/threshold/wait-logit/candidate-logit/replay-logit availability.
- `candidate_set_diff.csv`: candidate count/sample/features/surface/replay availability.

## Pass/Fail Criteria

- `replay metrics usable`: all inspected rows contain comparable paired replay/live evidence.
- `replay metrics not yet usable`: any required comparison category is not comparable, or no paired evidence exists.

## Tests Required

- Candidate row without replay reference is not comparable.
- Logit row without replay logits is not comparable.
- CLI writes all expected artifacts.

## Implementation Plan

Create `research_ops/diagnostics/feature_parity.py`, add tests, run against existing logs, and write artifacts under this iteration.

## Interpretation Guide

This diagnostic does not claim feature values differ. It asks whether the existing evidence is strong enough to compare them. Absence of paired evidence is a governance failure, not a model result.

## Limitations

- It does not load Protocol051/101 models.
- It does not rebuild replay rows.
- It does not compare numeric tensors because logs do not contain paired tensors.
