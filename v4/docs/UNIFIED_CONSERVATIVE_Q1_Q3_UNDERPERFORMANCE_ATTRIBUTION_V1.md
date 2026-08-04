# ATTRIBUTION_UNIFIED_CONSERVATIVE_Q1_Q3_UNDERPERFORMANCE_V1

What is this: attribution / flat-calibrated Q1-Q3 underperformance versus same-scope Protocol101
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost`
Replay decision: `strict_replay_complete_protocol101_challenge_still_blocked`

## Diagnosis

- q1_2026: challenger PnL $35,590 minus missed Protocol101 PnL $46,360 explains delta -$10,770.
- q3_2025: challenger PnL -$80 minus missed Protocol101 PnL $660 explains delta -$740.
- q4_2025: challenger PnL $72,280 minus missed Protocol101 PnL $32,460 explains delta $39,820.
- recent_2026: challenger PnL $5,310 minus missed Protocol101 PnL $2,130 explains delta $3,180.
- q1_2026: worst blocker net -$6,220; challenger -$640 blocked 3 Protocol101 entries worth $5,580.
- q3_2025: worst blocker net -$1,330; challenger -$480 blocked 3 Protocol101 entries worth $850.
- Zero-slippage missed Protocol101 entry reasons: {'blocked_by_challenger_open': 174, 'replaced_by_challenger_same_event': 4}.

## Component Summary

| slippage_per_side | split | challenger_pnl | missed_protocol101_pnl | component_delta_challenger_minus_missed | actual_delta | residual | challenger_entries | missed_protocol101_entries |
|---|---|---|---|---|---|---|---|---|
| 0.00 | q1_2026 | 35590.00 | 46360.00 | -10770.00 | -10770.00 | 0.00 | 45 | 96 |
| 0.00 | q3_2025 | -80.00 | 660.00 | -740.00 | -740.00 | 0.00 | 5 | 8 |
| 0.00 | q4_2025 | 72280.00 | 32460.00 | 39820.00 | 39820.00 | 0.00 | 25 | 64 |
| 0.00 | recent_2026 | 5310.00 | 2130.00 | 3180.00 | 3180.00 | -0.00 | 3 | 10 |
| 0.10 | q1_2026 | 34690.00 | 44440.00 | -9750.00 | -9750.00 | 0.00 | 45 | 96 |
| 0.10 | q3_2025 | -180.00 | 500.00 | -680.00 | -680.00 | 0.00 | 5 | 8 |
| 0.10 | q4_2025 | 71780.00 | 31180.00 | 40600.00 | 40600.00 | 0.00 | 25 | 64 |
| 0.10 | recent_2026 | 5250.00 | 1930.00 | 3320.00 | 3320.00 | -0.00 | 3 | 10 |
| 0.25 | q1_2026 | 33340.00 | 41560.00 | -8220.00 | -8220.00 | 0.00 | 45 | 96 |
| 0.25 | q3_2025 | -330.00 | 260.00 | -590.00 | -590.00 | 0.00 | 5 | 8 |
| 0.25 | q4_2025 | 71030.00 | 29260.00 | 41770.00 | 41770.00 | 0.00 | 25 | 64 |
| 0.25 | recent_2026 | 5160.00 | 1630.00 | 3530.00 | 3530.00 | -0.00 | 3 | 10 |

## Top Missed Protocol101 Entries

| slippage_per_side | split | session | decision_time | baseline_trade_pnl | stressed_baseline_pnl | miss_reason | blocking_source |
|---|---|---|---|---|---|---|---|
| 0.00 | q1_2026 | 2026-02-17 | 2026-02-17T15:47:00+00:00 | 3780.00 | 3780.00 | blocked_by_challenger_open | challenger |
| 0.10 | q1_2026 | 2026-02-17 | 2026-02-17T15:47:00+00:00 | 3780.00 | 3760.00 | blocked_by_challenger_open | challenger |
| 0.25 | q1_2026 | 2026-02-17 | 2026-02-17T15:47:00+00:00 | 3780.00 | 3730.00 | blocked_by_challenger_open | challenger |
| 0.00 | q1_2026 | 2026-01-29 | 2026-01-29T15:27:00+00:00 | 3580.00 | 3580.00 | blocked_by_challenger_open | challenger |
| 0.10 | q1_2026 | 2026-01-29 | 2026-01-29T15:27:00+00:00 | 3580.00 | 3560.00 | blocked_by_challenger_open | challenger |
| 0.25 | q1_2026 | 2026-01-29 | 2026-01-29T15:27:00+00:00 | 3580.00 | 3530.00 | blocked_by_challenger_open | challenger |
| 0.00 | q1_2026 | 2026-02-05 | 2026-02-05T16:15:00+00:00 | 3260.00 | 3260.00 | blocked_by_challenger_open | challenger |
| 0.10 | q1_2026 | 2026-02-05 | 2026-02-05T16:15:00+00:00 | 3260.00 | 3240.00 | blocked_by_challenger_open | challenger |
| 0.25 | q1_2026 | 2026-02-05 | 2026-02-05T16:15:00+00:00 | 3260.00 | 3210.00 | blocked_by_challenger_open | challenger |
| 0.00 | q4_2025 | 2025-11-20 | 2025-11-20T15:58:00+00:00 | 3120.00 | 3120.00 | blocked_by_challenger_open | challenger |
| 0.10 | q4_2025 | 2025-11-20 | 2025-11-20T15:58:00+00:00 | 3120.00 | 3100.00 | blocked_by_challenger_open | challenger |
| 0.25 | q4_2025 | 2025-11-20 | 2025-11-20T15:58:00+00:00 | 3120.00 | 3070.00 | blocked_by_challenger_open | challenger |
| 0.00 | q4_2025 | 2025-11-21 | 2025-11-21T18:48:00+00:00 | 2780.00 | 2780.00 | blocked_by_challenger_open | challenger |
| 0.10 | q4_2025 | 2025-11-21 | 2025-11-21T18:48:00+00:00 | 2780.00 | 2760.00 | blocked_by_challenger_open | challenger |
| 0.25 | q4_2025 | 2025-11-21 | 2025-11-21T18:48:00+00:00 | 2780.00 | 2730.00 | blocked_by_challenger_open | challenger |
| 0.00 | q4_2025 | 2025-11-14 | 2025-11-14T15:32:00+00:00 | 1900.00 | 1900.00 | blocked_by_challenger_open | challenger |

## Top Challenger Blockers

| slippage_per_side | split | session | challenger_pnl | blocked_protocol101_entries | blocked_protocol101_pnl | net_vs_blocked_protocol101 | challenger_duration_minutes | exit_reason |
|---|---|---|---|---|---|---|---|---|
| 0.00 | q1_2026 | 2026-02-17 | -640.00 | 3 | 5580.00 | -6220.00 | 76.00 | lifecycle_conservative_exit |
| 0.10 | q1_2026 | 2026-02-17 | -660.00 | 3 | 5520.00 | -6180.00 | 76.00 | lifecycle_conservative_exit |
| 0.25 | q1_2026 | 2026-02-17 | -690.00 | 3 | 5430.00 | -6120.00 | 76.00 | lifecycle_conservative_exit |
| 0.00 | q1_2026 | 2026-03-30 | 6390.00 | 7 | 5380.00 | 1010.00 | 381.00 | forced_flat_no_lifecycle_exit_signal |
| 0.00 | q4_2025 | 2025-11-21 | 4340.00 | 6 | 5300.00 | -960.00 | 275.00 | forced_flat_no_lifecycle_exit_signal |
| 0.10 | q1_2026 | 2026-03-30 | 6370.00 | 7 | 5240.00 | 1130.00 | 381.00 | forced_flat_no_lifecycle_exit_signal |
| 0.10 | q4_2025 | 2025-11-21 | 4320.00 | 6 | 5180.00 | -860.00 | 275.00 | forced_flat_no_lifecycle_exit_signal |
| 0.25 | q1_2026 | 2026-03-30 | 6340.00 | 7 | 5030.00 | 1310.00 | 381.00 | forced_flat_no_lifecycle_exit_signal |
| 0.25 | q4_2025 | 2025-11-21 | 4290.00 | 6 | 5000.00 | -710.00 | 275.00 | forced_flat_no_lifecycle_exit_signal |
| 0.00 | q1_2026 | 2026-01-29 | -80.00 | 6 | 4300.00 | -4380.00 | 373.00 | forced_flat_no_lifecycle_exit_signal |
| 0.10 | q1_2026 | 2026-01-29 | -100.00 | 6 | 4180.00 | -4280.00 | 373.00 | forced_flat_no_lifecycle_exit_signal |
| 0.00 | q4_2025 | 2025-11-14 | 2330.00 | 7 | 4170.00 | -1840.00 | 329.00 | forced_flat_no_lifecycle_exit_signal |
| 0.00 | q4_2025 | 2025-11-18 | -400.00 | 6 | 4070.00 | -4470.00 | 100.00 | lifecycle_conservative_exit |
| 0.10 | q4_2025 | 2025-11-14 | 2310.00 | 7 | 4030.00 | -1720.00 | 329.00 | forced_flat_no_lifecycle_exit_signal |
| 0.25 | q1_2026 | 2026-01-29 | -130.00 | 6 | 4000.00 | -4130.00 | 373.00 | forced_flat_no_lifecycle_exit_signal |
| 0.10 | q4_2025 | 2025-11-18 | -420.00 | 6 | 3950.00 | -4370.00 | 100.00 | lifecycle_conservative_exit |

## Preregistered Defer Constraints

1. `slot_opportunity_cost_margin`: Allow a challenger override only if predicted entry advantage exceeds a learned missed-Protocol101 opportunity-cost estimate plus the fixed conservative margin.
2. `split_stability_gate`: A replay candidate must have nonnegative same-scope delta on q1_2026 and q3_2025 under $0.00, $0.10, and $0.25 stress before any broader claim.
3. `blocked_baseline_budget`: Reject or penalize overrides whose predicted duration implies blocking more than one baseline entry or whose expected blocked baseline PnL exceeds challenger edge.
4. `forced_flat_dependence_audit`: Report PnL with forced-flat-no-exit-signal trades removed; a challenger cannot be promoted if gains depend mainly on forced flat.

## Next Required Evidence

1. Build the slot-opportunity-cost/defer constraint as a preregistered overlay before retraining.
2. Rerun strict replay and require nonnegative Q1/Q3 deltas under all deterministic slippage stresses.
3. Keep Protocol101 as paper default; this attribution is research-only.

## Outputs

- summary: `v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1/summary.json`
- report: `v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1/report.md`
- component_summary: `v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1/component_summary.csv`
- missed_protocol101_entries: `v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1/missed_protocol101_entries.csv`
- challenger_blocking_summary: `v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1/challenger_blocking_summary.csv`
- doc: `v4/docs/UNIFIED_CONSERVATIVE_Q1_Q3_UNDERPERFORMANCE_ATTRIBUTION_V1.md`
