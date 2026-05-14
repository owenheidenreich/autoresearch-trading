# Protocol 129: Protocol101 Offline Position Sizing

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 entries/exits remain frozen.

- Decision: `reject_scaling_loss_clustering_worse`
- Starting cash: `$10,000`
- Tuesday live paper remains capped at one contract regardless of this offline result.

## Results

| mode | ending_cash | total_pnl | return | taken | contracts | skipped | max_qty | max_dd | worst_day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_contract_baseline | $329,050 | $319,050 | 3190.5% | 1028 | 1028 | 0 | 1 | -$2,790 | -$2,080 |
| one_contract_20pct_cap | $302,470 | $292,470 | 2924.7% | 960 | 960 | 68 | 1 | -$5,000 | -$1,840 |
| adaptive_1_to_3_contracts | $778,050 | $768,050 | 7680.5% | 897 | 2462 | 131 | 3 | -$12,120 | -$5,520 |

## Guardrail

This protocol is intentionally isolated from Tuesday paper trading. The live-paper gate remains one contract, max one open position, and no broker endpoint without explicit approval.

## Outputs

- Trade rows: `v4/audit/autoresearch/v4_aplus_hypothesis_129_protocol101_offline_position_sizing/sizing_trade_rows.csv`
- Summary: `v4/audit/autoresearch/v4_aplus_hypothesis_129_protocol101_offline_position_sizing/summary.json`

## Next Gate

Reject scaling for initial paper trading. Keep one contract as the live-paper constraint and revisit sizing only after the one-contract path survives live shadow and paper replay.
