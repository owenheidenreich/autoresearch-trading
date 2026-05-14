# Protocol 131: Protocol101 Multi-Contract P&L Tracking

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `reject_multi_contract_risk_not_improved_enough`
- Scope: offline research only; Tuesday paper path remains one contract.

## Policy Results

| policy | ending_cash | total_pnl | return | taken | skipped | contracts | max_qty | max_dd | worst_day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_contract_baseline | $329,050 | $319,050 | 3190.5% | 1028 | 0 | 1028 | 1 | -$2,790 | -$2,080 |
| strict_exposure_ladder | $639,210 | $629,210 | 6292.1% | 917 | 111 | 2032 | 3 | -$8,330 | -$4,050 |
| conservative_profit_ladder | $774,830 | $764,830 | 7648.3% | 913 | 115 | 2434 | 3 | -$11,540 | -$5,520 |
| recovery_lock_ladder | $709,790 | $699,790 | 6997.9% | 919 | 109 | 2222 | 3 | -$11,540 | -$4,830 |

## Best Offline Candidate

- Policy: `conservative_profit_ladder`
- Passes acceptance: `False`
- Checks: `{'total_improved': True, 'drawdown_ok': True, 'worst_day_ok': False, 'skip_ok': True, 'risk_ok': True}`

## Outputs

- Summary CSV: `v4/audit/autoresearch/v4_aplus_hypothesis_131_protocol101_multi_contract_pnl_tracking/sizing_summary.csv`
- Trade rows: `v4/audit/autoresearch/v4_aplus_hypothesis_131_protocol101_multi_contract_pnl_tracking/sizing_trade_rows.csv`
- Daily rows: `v4/audit/autoresearch/v4_aplus_hypothesis_131_protocol101_multi_contract_pnl_tracking/sizing_daily_rows.csv`

## Next Gate

Keep Tuesday and initial paper trading at one contract. Multi-contract sizing remains rejected until a safer policy improves PnL without worsening drawdown or loss clustering.
