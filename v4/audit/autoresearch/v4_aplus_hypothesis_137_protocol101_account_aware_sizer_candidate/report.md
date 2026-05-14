# Protocol 137: Account-Aware Sizer V1 Candidate

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `pass_account_aware_sizer_v1_research_candidate_not_live`
- Policy artifact: `v4/audit/autoresearch/v4_aplus_hypothesis_137_protocol101_account_aware_sizer_candidate/account_aware_sizer_v1.json`
- 10k candidate PnL: `$379,820`
- 10k baseline PnL: `$319,050`
- Incremental PnL: `$60,770`
- Scaled trades: `193`
- Skipped trades: `8`
- Top day share: `6.6%`
- Top month share: `25.0%`

## Validation

| start | stress | policy | total_pnl | inc_pnl | max_dd | worst_day | pass |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| $10,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | `True` |
| $10,000 | $0.00 | candidate | $379,820 | $60,770 | -$3,150 | -$1,840 | `True` |
| $10,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | `True` |
| $10,000 | $0.25 | candidate | $313,130 | $45,480 | -$4,880 | -$1,890 | `True` |
| $10,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | `True` |
| $10,000 | $0.50 | candidate | $244,980 | $28,730 | -$7,200 | -$1,940 | `True` |
| $25,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | `True` |
| $25,000 | $0.00 | candidate | $380,550 | $61,500 | -$3,150 | -$2,150 | `True` |
| $25,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | `True` |
| $25,000 | $0.25 | candidate | $313,900 | $46,250 | -$4,880 | -$1,890 | `True` |
| $25,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | `True` |
| $25,000 | $0.50 | candidate | $247,780 | $31,530 | -$6,930 | -$1,940 | `True` |
| $30,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | `True` |
| $30,000 | $0.00 | candidate | $381,200 | $62,150 | -$3,150 | -$2,150 | `True` |
| $30,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | `True` |
| $30,000 | $0.25 | candidate | $315,220 | $47,570 | -$4,880 | -$1,890 | `True` |
| $30,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | `True` |
| $30,000 | $0.50 | candidate | $247,780 | $31,530 | -$6,930 | -$1,940 | `True` |
| $50,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | `True` |
| $50,000 | $0.00 | candidate | $383,490 | $64,440 | -$3,150 | -$2,150 | `True` |
| $50,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | `True` |
| $50,000 | $0.25 | candidate | $318,320 | $50,670 | -$4,880 | -$1,890 | `True` |
| $50,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | `True` |
| $50,000 | $0.50 | candidate | $249,890 | $33,640 | -$6,930 | -$1,940 | `True` |
| $100,000 | $0.00 | baseline | $319,050 | $0 | -$2,790 | -$2,080 | `True` |
| $100,000 | $0.00 | candidate | $383,660 | $64,610 | -$3,150 | -$1,980 | `True` |
| $100,000 | $0.25 | baseline | $267,650 | $0 | -$4,640 | -$2,280 | `True` |
| $100,000 | $0.25 | candidate | $318,390 | $50,740 | -$4,880 | -$2,300 | `True` |
| $100,000 | $0.50 | baseline | $216,250 | $0 | -$6,540 | -$2,480 | `True` |
| $100,000 | $0.50 | candidate | $251,800 | $35,550 | -$6,930 | -$1,940 | `True` |

## Outputs

- Config: `v4/audit/autoresearch/v4_aplus_hypothesis_137_protocol101_account_aware_sizer_candidate/account_aware_sizer_v1.json`
- Validation summary: `v4/audit/autoresearch/v4_aplus_hypothesis_137_protocol101_account_aware_sizer_candidate/validation_summary.csv`
- Attribution rows: `v4/audit/autoresearch/v4_aplus_hypothesis_137_protocol101_account_aware_sizer_candidate/attribution_rows.csv`

## Next Gate

Use this config for offline visual/account-state reports. It is still not paper/live multi-contract approval.
