# ATTRIBUTION_UNIFIED_CONSERVATIVE_NEURAL_OVERRIDES_V1

What is this: attribution / calibrated conservative neural challenger overrides
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `override_attribution_mixed_split_research_only`
Replay decision: `strict_replay_complete_protocol101_challenge_still_blocked`

## Stress Totals

| slippage | PnL | same-scope Protocol101 | delta | trades | challenger entries |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 267230.00 | 235740.00 | 31490.00 | 679 | 78 |
| 0.10 | 253650.00 | 220160.00 | 33490.00 | 679 | 78 |
| 0.25 | 233280.00 | 196790.00 | 36490.00 | 679 | 78 |

## Diagnosis

- At slippage 0.00, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].
- At slippage 0.10, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].
- At slippage 0.25, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].
- Zero-slippage challenger side PnL: {'C': 50720.0, 'P': 62380.0}.
- Zero-slippage challenger exit-reason PnL: {'lifecycle_conservative_exit': 14949.999999999998, 'forced_flat_no_lifecycle_exit_signal': 98150.0}.

## Challenger By Split

| slippage_per_side | split | rows | pnl | win_rate | median_duration_minutes |
|---|---|---|---|---|---|
| 0.00 | q1_2026 | 45 | 35590.00 | 0.33 | 132.00 |
| 0.00 | q3_2025 | 5 | -80.00 | 0.20 | 58.00 |
| 0.00 | q4_2025 | 25 | 72280.00 | 0.44 | 210.00 |
| 0.00 | recent_2026 | 3 | 5310.00 | 0.33 | 317.00 |
| 0.10 | q1_2026 | 45 | 34690.00 | 0.33 | 132.00 |
| 0.10 | q3_2025 | 5 | -180.00 | 0.20 | 58.00 |
| 0.10 | q4_2025 | 25 | 71780.00 | 0.44 | 210.00 |
| 0.10 | recent_2026 | 3 | 5250.00 | 0.33 | 317.00 |
| 0.25 | q1_2026 | 45 | 33340.00 | 0.33 | 132.00 |
| 0.25 | q3_2025 | 5 | -330.00 | 0.20 | 58.00 |
| 0.25 | q4_2025 | 25 | 71030.00 | 0.44 | 210.00 |
| 0.25 | recent_2026 | 3 | 5160.00 | 0.33 | 317.00 |

## Challenger By Exit Reason

| slippage_per_side | exit_reason | rows | pnl | win_rate | median_duration_minutes |
|---|---|---|---|---|---|
| 0.00 | forced_flat_no_lifecycle_exit_signal | 15 | 98150.00 | 0.93 | 314.00 |
| 0.00 | lifecycle_conservative_exit | 63 | 14950.00 | 0.22 | 95.00 |
| 0.10 | forced_flat_no_lifecycle_exit_signal | 15 | 97850.00 | 0.93 | 314.00 |
| 0.10 | lifecycle_conservative_exit | 63 | 13690.00 | 0.22 | 95.00 |
| 0.25 | forced_flat_no_lifecycle_exit_signal | 15 | 97400.00 | 0.93 | 314.00 |
| 0.25 | lifecycle_conservative_exit | 63 | 11800.00 | 0.22 | 95.00 |

## Challenger By Time Bucket

| slippage_per_side | time_bucket | rows | pnl | win_rate | median_duration_minutes |
|---|---|---|---|---|---|
| 0.00 | 1500_1559_utc | 16 | 34760.00 | 0.50 | 262.50 |
| 0.00 | 1600_1759_utc | 17 | 14080.00 | 0.24 | 121.00 |
| 0.00 | 1800_1959_utc | 5 | -620.00 | 0.20 | 27.00 |
| 0.00 | after_2000_utc | 2 | -60.00 | 0.50 | 29.00 |
| 0.00 | pre_1500_utc | 38 | 64940.00 | 0.37 | 181.00 |
| 0.10 | 1500_1559_utc | 16 | 34440.00 | 0.50 | 262.50 |
| 0.10 | 1600_1759_utc | 17 | 13740.00 | 0.24 | 121.00 |
| 0.10 | 1800_1959_utc | 5 | -720.00 | 0.20 | 27.00 |
| 0.10 | after_2000_utc | 2 | -100.00 | 0.50 | 29.00 |
| 0.10 | pre_1500_utc | 38 | 64180.00 | 0.37 | 181.00 |
| 0.25 | 1500_1559_utc | 16 | 33960.00 | 0.50 | 262.50 |
| 0.25 | 1600_1759_utc | 17 | 13230.00 | 0.24 | 121.00 |
| 0.25 | 1800_1959_utc | 5 | -870.00 | 0.20 | 27.00 |
| 0.25 | after_2000_utc | 2 | -160.00 | 0.50 | 29.00 |
| 0.25 | pre_1500_utc | 38 | 63040.00 | 0.37 | 181.00 |

## Top Losses

| slippage_per_side | split | session | decision_time | right | offset | entry_premium | pnl | exit_reason | duration_minutes |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | recent_2026 | 2026-05-18 | 2026-05-18T14:32:00+00:00 | P | 0.00 | 1390.00 | -860.00 | lifecycle_conservative_exit | 317.00 |
| 0.10 | recent_2026 | 2026-05-18 | 2026-05-18T14:32:00+00:00 | P | 0.00 | 1390.00 | -830.00 | lifecycle_conservative_exit | 317.00 |
| 0.00 | recent_2026 | 2026-05-18 | 2026-05-18T14:32:00+00:00 | P | 0.00 | 1390.00 | -810.00 | lifecycle_conservative_exit | 317.00 |
| 0.25 | q1_2026 | 2026-02-24 | 2026-02-24T16:20:00+00:00 | P | 20.00 | 2010.00 | -770.00 | lifecycle_conservative_exit | 50.00 |
| 0.10 | q1_2026 | 2026-02-24 | 2026-02-24T16:20:00+00:00 | P | 20.00 | 2010.00 | -740.00 | lifecycle_conservative_exit | 50.00 |
| 0.25 | q1_2026 | 2026-03-06 | 2026-03-06T16:06:00+00:00 | P | 25.00 | 3210.00 | -730.00 | lifecycle_conservative_exit | 128.00 |
| 0.00 | q1_2026 | 2026-02-24 | 2026-02-24T16:20:00+00:00 | P | 20.00 | 2010.00 | -720.00 | lifecycle_conservative_exit | 50.00 |
| 0.25 | q1_2026 | 2026-03-31 | 2026-03-31T15:26:00+00:00 | P | 35.00 | 3230.00 | -720.00 | lifecycle_conservative_exit | 59.00 |
| 0.10 | q1_2026 | 2026-03-06 | 2026-03-06T16:06:00+00:00 | P | 25.00 | 3210.00 | -700.00 | lifecycle_conservative_exit | 128.00 |
| 0.25 | q1_2026 | 2026-02-17 | 2026-02-17T14:50:00+00:00 | P | 40.00 | 3380.00 | -690.00 | lifecycle_conservative_exit | 76.00 |
| 0.10 | q1_2026 | 2026-03-31 | 2026-03-31T15:26:00+00:00 | P | 35.00 | 3230.00 | -690.00 | lifecycle_conservative_exit | 59.00 |
| 0.00 | q1_2026 | 2026-03-06 | 2026-03-06T16:06:00+00:00 | P | 25.00 | 3210.00 | -680.00 | lifecycle_conservative_exit | 128.00 |

## Next Required Evidence

1. Do not promote: explain Q1/Q3 underperformance before any additional training.
2. Add split-stability constraints or defer logic so Q4/recent gains cannot mask older-block losses.
3. Keep fill, holdout, live parity, and formal validation gates blocking Protocol101 challenge.

## Outputs

- summary: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_override_attribution_v1/summary.json`
- report: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_override_attribution_v1/report.md`
- doc: `v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_FLAT_CALIBRATED_OVERRIDE_ATTRIBUTION_V1.md`
