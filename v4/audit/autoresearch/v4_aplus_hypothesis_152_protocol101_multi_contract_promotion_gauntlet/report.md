# Protocol 152: Multi-Contract Promotion Gauntlet

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 entries and exits remain frozen.

- Decision: `blocked_multi_contract_promotion_incomplete_highres_coverage`
- Failed hypotheses: `0` of `5`
- Baseline PnL: `$319,050`
- Challenger PnL: `$379,820`
- Incremental PnL: `$60,770`
- Base gate: `pass`
- Timing gate: `incomplete_highres_timing_coverage`

## Timing Delay Gate

| split | delay_s | coverage | baseline_delayed | challenger_delayed | incremental_delayed | scaled | skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q1_2026 | 0 | 100.0% | $134,820 | $149,110 | $14,290 | 51 | 8 |
| q1_2026 | 1 | 74.9% | $93,980 | $100,550 | $6,570 | 37 | 8 |
| q1_2026 | 5 | 74.9% | $83,670 | $89,200 | $5,530 | 37 | 8 |
| q1_2026 | 15 | 74.9% | $63,160 | $66,070 | $2,910 | 37 | 8 |
| q1_2026 | 30 | 74.9% | $33,100 | $33,510 | $410 | 37 | 8 |
| q1_2026 | 60 | 74.9% | -$13,940 | -$17,850 | -$3,910 | 37 | 8 |
| q3_2025 | 0 | 100.0% | $68,310 | $110,320 | $42,010 | 122 | 2 |
| q3_2025 | 1 | 100.0% | $64,220 | $103,870 | $39,650 | 122 | 2 |
| q3_2025 | 5 | 100.0% | $56,500 | $91,920 | $35,420 | 122 | 2 |
| q3_2025 | 15 | 100.0% | $38,910 | $64,290 | $25,380 | 122 | 2 |
| q3_2025 | 30 | 100.0% | $21,210 | $36,980 | $15,770 | 122 | 2 |
| q3_2025 | 60 | 100.0% | -$12,670 | -$15,450 | -$2,780 | 122 | 2 |
| q4_2024_external | 0 | 100.0% | $56,650 | $60,100 | $3,450 | 17 | 1 |
| q4_2024_external | 1 | 100.0% | $55,070 | $58,270 | $3,200 | 17 | 1 |
| q4_2024_external | 5 | 100.0% | $51,360 | $54,250 | $2,890 | 17 | 1 |
| q4_2024_external | 15 | 100.0% | $39,720 | $41,360 | $1,640 | 17 | 1 |
| q4_2024_external | 30 | 100.0% | $27,950 | $28,470 | $520 | 17 | 1 |
| q4_2024_external | 60 | 100.0% | $5,485 | $4,520 | -$965 | 17 | 1 |
| q4_2025 | 0 | 100.0% | $96,800 | $101,620 | $4,820 | 15 | 1 |
| q4_2025 | 1 | 95.2% | $80,980 | $85,640 | $4,660 | 15 | 1 |
| q4_2025 | 5 | 95.2% | $67,260 | $70,150 | $2,890 | 15 | 1 |
| q4_2025 | 15 | 95.2% | $49,320 | $50,480 | $1,160 | 15 | 1 |
| q4_2025 | 30 | 95.2% | $21,640 | $22,090 | $450 | 15 | 1 |
| q4_2025 | 60 | 95.2% | -$16,040 | -$16,650 | -$610 | 15 | 1 |

## Segment Incremental

| segment | incremental | scaled | skipped |
| --- | ---: | ---: | ---: |
| q1_2026 | $9,980 | 39 | 4 |
| q3_2025 | $42,390 | 122 | 2 |
| q4_2024_external | $3,450 | 17 | 1 |
| q4_2025 | $4,950 | 15 | 1 |

## Side Incremental

| side | incremental | scaled | skipped |
| --- | ---: | ---: | ---: |
| CALL | $33,720 | 112 | 8 |
| PUT | $27,050 | 81 | 0 |

## Next Gate

Do not replace the one-contract operational baseline yet; collect live-shadow/paper evidence or more high-resolution coverage for the same frozen challenger.
