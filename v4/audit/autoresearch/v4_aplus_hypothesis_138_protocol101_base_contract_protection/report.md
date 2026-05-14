# Protocol 138: Base Contract Protection

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `reject_base_contract_protection_still_external_negative`
- Old total PnL: `$362,890`
- New total PnL: `$366,960`
- New skipped trades: `39`

## Segment Incremental vs One Contract

| segment | incremental | scaled | skipped |
| --- | ---: | ---: | ---: |
| q1_2026 | $9,980 | 39 | 4 |
| q3_2025 | $36,620 | 122 | 17 |
| q4_2024_external | -$3,640 | 14 | 17 |
| q4_2025 | $4,950 | 15 | 1 |

## Stress

| stress | incremental | max_dd | worst_day |
| ---: | ---: | ---: | ---: |
| $0.00 | $47,910 | -$3,060 | -$1,840 |
| $0.25 | $31,780 | -$4,420 | -$1,890 |
| $0.50 | $12,210 | -$6,520 | -$1,940 |

## Outputs

- Segment summary: `v4/audit/autoresearch/v4_aplus_hypothesis_138_protocol101_base_contract_protection/segment_summary.csv`
- Stress summary: `v4/audit/autoresearch/v4_aplus_hypothesis_138_protocol101_base_contract_protection/stress_summary.csv`
- Attribution rows: `v4/audit/autoresearch/v4_aplus_hypothesis_138_protocol101_base_contract_protection/attribution_rows.csv`

## Next Gate

Do not keep this change; inspect the segment summary for why base-contract protection failed.
