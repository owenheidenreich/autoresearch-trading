# Protocol 135: Starting-Cash Sensitivity

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `pass_always_on_sizer_starting_cash_sensitivity_not_live`
- Active policy: `always_on_account_aware_sizer`

## Summary

| start | stress | policy | total_pnl | inc_pnl | max_dd | worst_day | contracts | pass |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| $10,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | 1028 | `True` |
| $10,000 | $0.00 | candidate | $354,000 | $34,950 | -$5,000 | -$1,840 | 1172 | `True` |
| $10,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | 1028 | `True` |
| $10,000 | $0.25 | candidate | $290,450 | $22,800 | -$5,040 | -$1,890 | 1145 | `True` |
| $10,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | 1028 | `True` |
| $10,000 | $0.50 | candidate | $219,530 | $3,280 | -$6,400 | -$1,940 | 1091 | `True` |
| $25,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | 1028 | `True` |
| $25,000 | $0.00 | candidate | $360,780 | $41,730 | -$5,000 | -$1,840 | 1191 | `True` |
| $25,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | 1028 | `True` |
| $25,000 | $0.25 | candidate | $295,650 | $28,000 | -$5,040 | -$1,890 | 1154 | `True` |
| $25,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | 1028 | `True` |
| $25,000 | $0.50 | candidate | $225,680 | $9,430 | -$6,400 | -$1,940 | 1107 | `True` |
| $30,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | 1028 | `True` |
| $30,000 | $0.00 | candidate | $360,780 | $41,730 | -$5,000 | -$1,840 | 1191 | `True` |
| $30,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | 1028 | `True` |
| $30,000 | $0.25 | candidate | $296,040 | $28,390 | -$5,040 | -$1,890 | 1155 | `True` |
| $30,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | 1028 | `True` |
| $30,000 | $0.50 | candidate | $226,040 | $9,790 | -$6,400 | -$1,940 | 1108 | `True` |
| $50,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | 1028 | `True` |
| $50,000 | $0.00 | candidate | $363,700 | $44,650 | -$5,000 | -$1,840 | 1201 | `True` |
| $50,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | 1028 | `True` |
| $50,000 | $0.25 | candidate | $298,410 | $30,760 | -$5,040 | -$1,890 | 1167 | `True` |
| $50,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | 1028 | `True` |
| $50,000 | $0.50 | candidate | $227,350 | $11,100 | -$6,400 | -$1,940 | 1113 | `True` |
| $100,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | 1028 | `True` |
| $100,000 | $0.00 | candidate | $363,700 | $44,650 | -$5,000 | -$1,840 | 1201 | `True` |
| $100,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | 1028 | `True` |
| $100,000 | $0.25 | candidate | $300,340 | $32,690 | -$5,040 | -$1,890 | 1174 | `True` |
| $100,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | 1028 | `True` |
| $100,000 | $0.50 | candidate | $230,980 | $14,730 | -$6,340 | -$1,940 | 1132 | `True` |

## Outputs

- Starting cash summary: `v4/audit/autoresearch/v4_aplus_hypothesis_135_protocol101_starting_cash_sensitivity/starting_cash_summary.csv`
- Daily summary: `v4/audit/autoresearch/v4_aplus_hypothesis_135_protocol101_starting_cash_sensitivity/daily_summary.csv`

## Next Gate

Promote the account-aware sizer to an offline research candidate artifact, then test equity visualizations and live-shadow-compatible account-state serialization.
