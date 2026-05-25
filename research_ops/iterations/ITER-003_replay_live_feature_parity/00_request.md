# Iteration Request

Iteration ID: `ITER-003_replay_live_feature_parity`
Assumption ID: `A004`
Title: Replay/live feature parity

## Request

Determine whether live Protocol051-to-Protocol101 feature construction can be proven to match replay for the same market state.

## Required Outputs

- `feature_parity_report.md`
- `feature_diff.csv`
- `logit_diff.csv`
- `candidate_set_diff.csv`

## Decision Options

- `replay metrics usable`
- `replay metrics not yet usable`

## Hard Constraints

- Do not modify v4 trading logic.
- Do not mutate runtime flags, launchd, broker behavior, paid data, training, thresholds, or model artifacts.
- Do not call broker APIs.
- Do not load model binaries.
- Read existing logs only.
