# Round 5 — March Freeze And Reconciliation

This folder contains the corrected response packet for Fable after Fable identified the attempt131 candidate/export mismatch.

## Files

```text
00_message_to_fable.md
01_fable_round4_response.md
02_uplift_lifecycle_report_reconciled.md
03_uplift_lifecycle_summary_reconciled.json
04_strategy_uplift_rows_reconciled.csv
05_lifecycle_attribution_rows_reconciled.csv
06_candidate_null_m_rows_reconciled.csv
07_stratum_uplift_rows_reconciled.csv
08_candidate_reconciliation_rows.csv
09_uplift_lifecycle_script_reconciled.py
SHA256SUMS
```

## Key Reconciliation

Attempt131 validation now uses strict one-account replay as canonical:

```text
21 trades
$1,970 stressed PnL
```

The previous selected-candidate export had one extra row:

```text
2026-03-16T14:29:00+00:00
SPXW-20260316-06695.000-P
+$570 stressed contribution
```

Strict replay correctly skipped that row because the 2026-03-16 daily stressed PnL had already crossed the -$500 daily-loss stop.

## Verification

```text
34 targeted tests passed.
No broker, paid-data, paper-submit, training, threshold, promotion, or default-change actions occurred.
```

