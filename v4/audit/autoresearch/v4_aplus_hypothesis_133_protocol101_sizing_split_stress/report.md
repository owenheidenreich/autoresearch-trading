# Protocol 133: Sizing Split And Stress Validation

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `pass_sizing_candidate_survives_split_stress_not_live`
- Best candidate: `lower_two_contract_threshold`

## Stress Summary

| stress | policy | total_pnl | max_dd | worst_day | skipped | max_qty | pass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| $0.00 | one_contract_baseline | $319,050 | -$2,790 | -$2,080 | 0 | 1 | `True` |
| $0.00 | high_conviction_profit_cushion | $332,870 | -$5,000 | -$1,840 | 64 | 3 | `True` |
| $0.00 | slow_growth_two_contract | $331,270 | -$5,000 | -$1,840 | 68 | 2 | `True` |
| $0.00 | lower_two_contract_threshold | $354,000 | -$5,000 | -$1,840 | 64 | 3 | `True` |
| $0.10 | one_contract_baseline | $298,490 | -$3,500 | -$2,160 | 0 | 1 | `True` |
| $0.10 | high_conviction_profit_cushion | $309,440 | -$5,300 | -$1,860 | 60 | 3 | `True` |
| $0.10 | slow_growth_two_contract | $307,460 | -$5,300 | -$1,860 | 68 | 2 | `True` |
| $0.10 | lower_two_contract_threshold | $326,550 | -$5,300 | -$1,860 | 60 | 3 | `True` |
| $0.25 | one_contract_baseline | $267,650 | -$4,640 | -$2,280 | 0 | 1 | `True` |
| $0.25 | high_conviction_profit_cushion | $274,790 | -$5,040 | -$1,890 | 66 | 3 | `True` |
| $0.25 | slow_growth_two_contract | $274,590 | -$5,040 | -$1,890 | 74 | 2 | `True` |
| $0.25 | lower_two_contract_threshold | $290,450 | -$5,040 | -$1,890 | 66 | 3 | `True` |
| $0.50 | one_contract_baseline | $216,250 | -$6,540 | -$2,480 | 0 | 1 | `True` |
| $0.50 | high_conviction_profit_cushion | $209,040 | -$6,400 | -$1,940 | 83 | 3 | `False` |
| $0.50 | slow_growth_two_contract | $213,160 | -$5,740 | -$1,940 | 89 | 2 | `False` |
| $0.50 | lower_two_contract_threshold | $219,530 | -$6,400 | -$1,940 | 83 | 3 | `True` |

## Outputs

- Stress summary: `v4/audit/autoresearch/v4_aplus_hypothesis_133_protocol101_sizing_split_stress/stress_summary.csv`
- Split summary: `v4/audit/autoresearch/v4_aplus_hypothesis_133_protocol101_sizing_split_stress/split_summary.csv`
- Month summary: `v4/audit/autoresearch/v4_aplus_hypothesis_133_protocol101_sizing_split_stress/month_summary.csv`
- Daily summary: `v4/audit/autoresearch/v4_aplus_hypothesis_133_protocol101_sizing_split_stress/daily_summary.csv`

## Next Gate

Run trade-set attribution for the surviving sizing policy and keep it offline until one-contract live paper parity is proven.
