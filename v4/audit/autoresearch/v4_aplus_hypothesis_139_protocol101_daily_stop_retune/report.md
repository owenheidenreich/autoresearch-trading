# Protocol 139: Daily Stop Retune

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `pass_daily_stop_retune_candidate_not_live`
- Best fixed stop: `-1500.0`
- Best equity fraction: `0.005`
- Best unstressed PnL: `$379,820`
- Current unstressed PnL: `$366,960`

## Experiments

| fixed_stop | fraction | pass | pnl | inc_025 | inc_050 | max_dd | worst_day |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| -1500.0 | 0.500% | `True` | $379,820 | $45,480 | $28,730 | -$3,150 | -$1,840 |
| -1500.0 | 1.000% | `True` | $380,110 | $45,870 | $28,800 | -$3,150 | -$1,980 |
| -2000.0 | 0.500% | `True` | $378,870 | $43,810 | $28,820 | -$3,150 | -$2,150 |

## Segment Incremental

| segment | incremental | scaled | skipped |
| --- | ---: | ---: | ---: |
| q1_2026 | $9,980 | 39 | 4 |
| q3_2025 | $42,390 | 122 | 2 |
| q4_2024_external | $3,450 | 17 | 1 |
| q4_2025 | $4,950 | 15 | 1 |

## Outputs

- Experiment summary: `v4/audit/autoresearch/v4_aplus_hypothesis_139_protocol101_daily_stop_retune/experiment_summary.csv`
- Segment summary: `v4/audit/autoresearch/v4_aplus_hypothesis_139_protocol101_daily_stop_retune/segment_summary.csv`
- Attribution rows: `v4/audit/autoresearch/v4_aplus_hypothesis_139_protocol101_daily_stop_retune/attribution_rows.csv`

## Next Gate

Update account_aware_sizer_v1 to use the retuned daily stop and rerun consolidated validation.
