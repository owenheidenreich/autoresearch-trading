# FOUNDATION_UNTOUCHED_EVAL_BLOCK_RESERVATION_V1

What is this: foundation gate / untouched evaluation block reservation
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `untouched_holdout_reserved_pending_new_data_collection`

## Reserved Block

- Name: `UNTOUCHED_EVAL_BLOCK_V1`
- Status: `reserved_pending_collection`
- Reserved on: `2026-05-24`
- Start after: `2026-05-24`
- Data status: `pending_new_data_collection`
- Intended use: `final_evaluation_only`
- Split label: `future_unseen_block_after_2026_05_24`
- Final claim available now: `False`

## Exposed Diagnostics

These blocks are no longer sacred final holdouts for a new challenger:

- `q3_2025`
- `q4_2025`
- `q1_2026`
- `march_2026`
- `recent_2026`
- `q1_2025`
- `q1_2025_partial`
- `q2_2025`
- `q4_2024`

## Forbidden Uses

- `feature_selection`
- `architecture_selection`
- `threshold_selection`
- `objective_selection`
- `sizing_selection`
- `exit_rule_selection`
- `model_selection`
- `training`

## Validation

- Status: `pass`
- Data available: `False`
- Errors: `[]`
- Warnings: `[]`

## Next Required Evidence

1. Collect or purchase the named new block only after the policy, labels, metrics, and thresholds are frozen.
2. Freeze the raw/processed data manifest before scoring the challenger.
3. Score the final challenger once against Protocol101 under strict one-account replay and stress assumptions.

## Outputs

- Summary: `v4/audit/autoresearch/unified_untouched_holdout_reservation/summary.json`
- Report: `v4/audit/autoresearch/unified_untouched_holdout_reservation/report.md`
- Reservation: `v4/audit/autoresearch/unified_untouched_holdout_reservation/untouched_holdout_reservation_v1.json`
- Docs copy: `v4/docs/UNTOUCHED_HOLDOUT_RESERVATION.md`
