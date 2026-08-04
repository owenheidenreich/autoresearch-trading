# DATASET_UNIFIED_SERIAL_DP_ORACLE_V1

What is this: dataset foundation / unified serial DP oracle training-scope manifest
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope`

## Manifest

- Training splits: `('q1_2026', 'q3_2025', 'q4_2025', 'recent_2026')`
- Included sessions: `191`
- Excluded sessions with missing holding paths: `33`
- Flat candidate rows: `1919518`
- Flat decision events: `68247`
- Flat oracle enter events: `14106`
- Holding state rows: `2856795`
- Holding oracle trades: `14106`
- Baseline event-seed rows: `341235`
- Baseline seed count: `5`
- Holding coverage of oracle entries: `1.000000`
- Baseline coverage of flat events: `1.000000`
- Disallowed baseline actions: `{}`

## Split Summary

| split | included sessions | excluded sessions | flat events | oracle enters | holding trades | holding rows |
|---|---:|---:|---:|---:|---:|---:|
| q1_2026 | 53 | 8 | 19027 | 3886 | 3886 | 785554 |
| q3_2025 | 55 | 9 | 19744 | 4114 | 4114 | 841805 |
| q4_2025 | 54 | 10 | 19066 | 3917 | 3917 | 788361 |
| recent_2026 | 29 | 6 | 10410 | 2189 | 2189 | 441075 |

## Next Required Evidence

1. This oracle scope may be used for the first preregistered conservative neural training run.

## Outputs

- summary: `v4/audit/autoresearch/unified_serial_dp_oracle/summary.json`
- report: `v4/audit/autoresearch/unified_serial_dp_oracle/report.md`
- flat_events: `v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_flat_events.parquet`
- holding_trade_manifest: `v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_holding_trade_manifest.parquet`
- session_manifest: `v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_session_manifest.csv`
- split_summary: `v4/audit/autoresearch/unified_serial_dp_oracle/split_summary.csv`
- doc: `v4/docs/UNIFIED_SERIAL_DP_ORACLE.md`
