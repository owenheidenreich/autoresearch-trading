# DATASET_UNIFIED_SLOT_OPPORTUNITY_COST_LABELS_V1

What is this: dataset / candidate-level blocked-Protocol101 opportunity-cost labels
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `slot_opportunity_cost_labels_ready_for_causal_estimator`

## Label Semantics

- Blocked entries: Protocol101 entry events in the same session with current_decision_dt <= baseline_entry_dt < candidate_exit_dt.
- Blocked PnL: Sum of those baseline trade PnLs under deterministic slippage stress. This is a label, not a model input.
- These are labels, not runtime features.

## Split Summary

| split | rows | sessions | blocked_entry_rows | blocked_entry_row_fraction | mean_blocked_protocol101_pnl_0_00 | p95_blocked_protocol101_pnl_0_00 |
|---|---|---|---|---|---|---|
| q1_2026 | 551413 | 53 | 128190 | 0.2325 | 96.3593 | 870.0000 |
| q3_2025 | 529645 | 55 | 122343 | 0.2310 | 72.3156 | 560.0000 |
| q4_2025 | 534830 | 54 | 130530 | 0.2441 | 117.4806 | 870.0000 |
| recent_2026 | 303630 | 29 | 48656 | 0.1602 | 11.0947 | 320.0000 |

## Next Required Evidence

1. Train/calibrate a causal opportunity-cost estimator using current-state features only.
2. Use the estimator in strict replay as a defer overlay; do not use label columns as runtime inputs.
3. Require nonnegative Q1/Q3 same-scope deltas under $0.00/$0.10/$0.25 stress before broader claims.

## Outputs

- summary: `v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/summary.json`
- report: `v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/report.md`
- labels: `v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/slot_opportunity_cost_labels.parquet`
- split_summary: `v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/split_summary.csv`
- doc: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_COST_LABEL_DATASET.md`
