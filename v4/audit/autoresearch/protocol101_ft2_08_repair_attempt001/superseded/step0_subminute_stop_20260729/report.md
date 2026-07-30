# FT2-08 Repair Attempt 1 — STEP 0 Data Feasibility

Terminal outcome: `stop_subminute_data_unavailable`

## Gate

The owner-authorized fill law requires the first executable quote at or after
`decision_time + 5 seconds`, with a report-only `+15 seconds` sensitivity. The
repair may proceed only if normalized data preserves multiple timestamped quotes
per contract per event minute for every current census session.

## Population

- Census sessions tested: **46/46**.
- Wider-corpus sessions tested: **10/10**.
- Wider sample: every normalized session outside the census, every outer-test
  slice, and the protected holdout. This yielded exactly: 2025-04-07, 2025-04-08, 2025-04-09, 2025-04-10, 2025-04-11, 2025-07-03, 2025-07-30, 2025-10-22, 2025-11-28, 2025-12-24.
- Protected holdout or outer-test session accessed: **no**.

## Result

- Sessions passing: **0/56**.
- Sessions failing: **56/56**.
- Rows inspected: **10,295,500**.
- Contract/event-minute groups: **10,295,500**.
- Groups containing at least two rows: **0**.
- Groups containing at least two distinct quote timestamps: **0**.
- Maximum rows in any contract/event-minute: **1**.
- Maximum distinct quote timestamps in any group: **1**.
- Event timestamps off the minute boundary: **0**.
- Retained snapshots whose quote timestamp is inside their event minute: **10,295,500**.
- Single retained snapshots timestamped at or after +5 seconds: **0**.
- Single retained snapshots timestamped at or after +15 seconds: **0**.
- Timestamp source: `databento_cbbo_1m_ts_recv`.

The normalized files preserve one CBBO snapshot per contract/event-minute. Many
rows retain a sub-minute `quote_time`, but no group contains the multiple quote
updates needed to identify which quote was the **first** executable state after
+5 seconds or +15 seconds. A single later snapshot is not a sub-minute quote
sequence and cannot implement A1 by construction.

## Required Stop

Per the goal, no fallback was invented. The consolidated authority, FT2-04,
FT2-05, and FT2-08 were not amended, and the census was not rerun. A separate
owner decision is required to authorize either recorder-calibrated fill
slippage or acquisition/use of sub-minute historical quote data.

Detailed per-session statistics: `step0_quote_per_minute.csv`.
Machine-readable summary: `step0_data_feasibility.json`.
