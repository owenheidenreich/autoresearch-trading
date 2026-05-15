# Protocol 151: Account-Aware Multi-Contract Validation

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 entries and exits remain frozen.

- Decision: `pass_account_aware_sizing_beats_one_contract_research_only`
- Gate: `pass`
- Baseline one-contract PnL from $10,000: `$319,050`
- Candidate account-aware PnL from $10,000: `$379,820`
- Incremental PnL: `$60,770`
- Candidate max quantity: `3`
- Candidate skipped trades: `8`
- Quantity counts: `{'0': 8, '1': 827, '2': 166, '3': 27}`
- Top day share of positive incremental PnL: `6.6%`
- Top month share of positive incremental PnL: `27.9%`

## Policy Logic

Extra contracts require cash/account growth, low drawdown, recent positive PnL, nonnegative same-day PnL, profit cushion, premium exposure limits, and score margin above the frozen Protocol101 threshold. The base one-contract trade is protected from the scaling cap when cash can afford it.

## Cash And Stress

| start | stress | baseline_pnl | candidate_pnl | incremental | baseline_dd | candidate_dd | romd_improved | worst_day_ok |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| $10,000 | $0.00 | $319,050 | $379,820 | $60,770 | -$2,790 | -$3,150 | `True` | `True` |
| $10,000 | $0.25 | $267,650 | $313,130 | $45,480 | -$4,640 | -$4,880 | `True` | `True` |
| $10,000 | $0.50 | $216,250 | $244,980 | $28,730 | -$6,540 | -$7,200 | `True` | `True` |
| $25,000 | $0.00 | $319,050 | $380,550 | $61,500 | -$2,790 | -$3,150 | `True` | `True` |
| $25,000 | $0.25 | $267,650 | $313,900 | $46,250 | -$4,640 | -$4,880 | `True` | `True` |
| $25,000 | $0.50 | $216,250 | $247,780 | $31,530 | -$6,540 | -$6,930 | `True` | `True` |
| $30,000 | $0.00 | $319,050 | $381,200 | $62,150 | -$2,790 | -$3,150 | `True` | `True` |
| $30,000 | $0.25 | $267,650 | $315,220 | $47,570 | -$4,640 | -$4,880 | `True` | `True` |
| $30,000 | $0.50 | $216,250 | $247,780 | $31,530 | -$6,540 | -$6,930 | `True` | `True` |
| $50,000 | $0.00 | $319,050 | $383,490 | $64,440 | -$2,790 | -$3,150 | `True` | `True` |
| $50,000 | $0.25 | $267,650 | $318,320 | $50,670 | -$4,640 | -$4,880 | `True` | `True` |
| $50,000 | $0.50 | $216,250 | $249,890 | $33,640 | -$6,540 | -$6,930 | `True` | `True` |
| $100,000 | $0.00 | $319,050 | $383,660 | $64,610 | -$2,790 | -$3,150 | `True` | `True` |
| $100,000 | $0.25 | $267,650 | $318,390 | $50,740 | -$4,640 | -$4,880 | `True` | `True` |
| $100,000 | $0.50 | $216,250 | $251,800 | $35,550 | -$6,540 | -$6,930 | `True` | `True` |

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

## Outputs

- Policy config: `v4/audit/autoresearch/v4_aplus_hypothesis_151_protocol101_account_aware_sizing_validation/account_aware_sizer_v1.json`
- Validation summary: `v4/audit/autoresearch/v4_aplus_hypothesis_151_protocol101_account_aware_sizing_validation/validation_summary.csv`
- Attribution rows: `v4/audit/autoresearch/v4_aplus_hypothesis_151_protocol101_account_aware_sizing_validation/attribution_rows.csv`
- Candidate trade rows: `v4/audit/autoresearch/v4_aplus_hypothesis_151_protocol101_account_aware_sizing_validation/candidate_trade_rows.csv`

## Next Gate

Keep account-aware sizing as an offline research candidate. Use one-contract paper first, then replay live paper logs through this sizer before enabling multi-contract paper order quantities.
