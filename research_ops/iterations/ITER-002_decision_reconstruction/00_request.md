# Iteration Request

Iteration ID: `ITER-002_decision_reconstruction`
Assumption ID: `A006`
Title: Decision reconstruction

## Request

Determine whether every live/paper/no-order Protocol101 decision can be reconstructed from logs alone.

## Required Outputs

- `decision_reconstruction_matrix.csv`
- `missing_fields_report.md`
- `log_schema_gap_report.md`

## Decision Options

- `logs sufficient`
- `logs insufficient`
- `schema patch required`

## Hard Constraints

- Do not modify v4 trading logic.
- Do not mutate runtime flags, launchd, broker behavior, paid data, training, thresholds, or model artifacts.
- Do not call broker APIs.
- Do not load models or score protected holdouts.
- Read existing logs only.
