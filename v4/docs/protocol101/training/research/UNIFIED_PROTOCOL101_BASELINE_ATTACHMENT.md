# DATASET_PROTOCOL101_BASELINE_ACTION_ATTACHMENT_V1

What is this: dataset foundation / Protocol101 baseline action attachment
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `protocol101_baseline_attachment_ready_for_baseline_aligned_training_scope`

## Coverage

- Flat events: `123439`
- Event-seed rows: `443199`
- Baseline trades: `4906`
- Matched entry trades: `4365`
- Unmatched entry trades: `0`
- Baseline trades without event: `541`
- Splits with flat events: `['q1_2025', 'q1_2026', 'q2_2025', 'q3_2025', 'q4_2025', 'recent_2026']`
- Splits with baseline: `['march_2026', 'q1_2026', 'q3_2025', 'q4_2025', 'recent_2026']`
- Splits without baseline: `['q1_2025', 'q2_2025']`
- Baseline splits without flat events: `['march_2026']`

## Baseline-Aligned Training Scope

- Included splits: `['q1_2026', 'q3_2025', 'q4_2025', 'recent_2026']`
- Excluded flat splits: `['q1_2025', 'q2_2025']`
- Baseline alias splits excluded: `['march_2026']`
- Scope decision: `protocol101_baseline_attachment_ready_all_trajectory_splits`
- Scope flat events: `79940`
- Scope event-seed rows: `399700`
- Scope baseline trades: `4365`
- Scope matched entry trades: `4365`
- Scope unmatched entry trades: `0`
- Scope baseline trades without event: `0`

## Action Counts

| action | rows |
|---|---:|
| `baseline_not_available_for_split` | 43499 |
| `enter` | 4365 |
| `exit_then_wait` | 3998 |
| `holding` | 49095 |
| `wait` | 342242 |

## Next Required Evidence

1. Add Protocol101 baseline actions for trajectory splits without baseline coverage: ['q1_2025', 'q2_2025'].
2. Reconcile baseline splits that are not represented as trajectory splits: ['march_2026'].
3. Explain baseline trades that did not land on a full-surface trajectory decision event.
4. Use the baseline-aligned training scope for conservative neural training; keep excluded splits diagnostic until baseline coverage exists.

## Outputs

- Summary: `v4/audit/autoresearch/unified_protocol101_baseline_attachment/summary.json`
- Report: `v4/audit/autoresearch/unified_protocol101_baseline_attachment/report.md`
- Event actions: `v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions.parquet`
- Training-scope event actions: `v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet`
- Action counts: `v4/audit/autoresearch/unified_protocol101_baseline_attachment/action_counts.csv`
- Docs copy: `v4/docs/UNIFIED_PROTOCOL101_BASELINE_ATTACHMENT.md`
