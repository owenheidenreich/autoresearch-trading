# Protocol 132: Confidence-Aware Sizing Autoresearch

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `pass_confidence_sizing_research_candidate_not_live`
- Baseline total PnL: `$319,050`
- Stopped after 3 failed hypotheses: `False`

## Loop Results

| # | hypothesis | total_pnl | max_dd | worst_day | skipped | max_qty | pass |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 132a_confidence_ladder | $379,770 | -$5,000 | -$4,260 | 59 | 3 | `False` |
| 2 | 132b_high_conviction_profit_cushion | $332,870 | -$5,000 | -$1,840 | 64 | 3 | `True` |
| 3 | 132c_slow_growth_two_contract | $331,270 | -$5,000 | -$1,840 | 68 | 2 | `True` |
| 4 | 132d_lower_two_contract_threshold | $354,000 | -$5,000 | -$1,840 | 64 | 3 | `True` |
| 5 | 132e_delayed_three_contract_unlock | $372,350 | -$5,000 | -$3,220 | 57 | 3 | `False` |
| 6 | 132f_prior_131_policy_regression_check | $699,790 | -$11,540 | -$4,830 | 109 | 3 | `False` |

## Accepted

`132b_high_conviction_profit_cushion, 132c_slow_growth_two_contract, 132d_lower_two_contract_threshold`

## Outputs

- Loop results: `v4/audit/autoresearch/v4_aplus_hypothesis_132_protocol101_confidence_sizing_autoresearch/loop_results.csv`
- Trade rows: `v4/audit/autoresearch/v4_aplus_hypothesis_132_protocol101_confidence_sizing_autoresearch/sizing_trade_rows.csv`
- Daily rows: `v4/audit/autoresearch/v4_aplus_hypothesis_132_protocol101_confidence_sizing_autoresearch/sizing_daily_rows.csv`

## Next Gate

Treat the accepted policy as offline research only. It needs split-by-split attribution and live one-contract parity before any multi-contract paper test.
