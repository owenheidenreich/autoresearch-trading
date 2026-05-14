# Protocol 126: Protocol101 Timing Fragility Hardening

No paid data was downloaded. No broker endpoint was called. No orders were placed. No model was trained.

- Decision: `blocked_incomplete_timing_coverage`
- Source replay: `v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_path_audit/report.json`
- Delay seconds: `[0, 1, 5, 15, 30, 60]`

## Split Delay Summary

| split | delay_s | coverage | original_pnl | delayed_pnl | pnl_delta | positive_trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q1_2026 | 0 | 100.0% | $624,610 | $624,610 | $0 | 68.6% |
| q1_2026 | 1 | 74.8% | $450,950 | $437,590 | -$13,360 | 68.6% |
| q1_2026 | 5 | 74.8% | $450,950 | $387,830 | -$63,120 | 65.9% |
| q1_2026 | 15 | 74.8% | $450,950 | $294,220 | -$156,730 | 59.9% |
| q1_2026 | 30 | 74.8% | $450,950 | $152,830 | -$298,120 | 54.5% |
| q1_2026 | 60 | 74.8% | $450,950 | -$63,140 | -$514,090 | 43.8% |
| q3_2025 | 0 | 100.0% | $300,750 | $300,750 | $0 | 79.5% |
| q3_2025 | 1 | 100.0% | $300,750 | $281,680 | -$19,070 | 77.3% |
| q3_2025 | 5 | 100.0% | $300,750 | $242,130 | -$58,620 | 75.2% |
| q3_2025 | 15 | 100.0% | $300,750 | $161,050 | -$139,700 | 68.7% |
| q3_2025 | 30 | 100.0% | $300,750 | $86,660 | -$214,090 | 60.2% |
| q3_2025 | 60 | 100.0% | $300,750 | -$68,720 | -$369,470 | 45.0% |
| q4_2024_external | 0 | 100.0% | $254,930 | $254,930 | $0 | 68.9% |
| q4_2024_external | 1 | 100.0% | $254,930 | $246,910 | -$8,020 | 68.0% |
| q4_2024_external | 5 | 100.0% | $254,930 | $227,060 | -$27,870 | 65.8% |
| q4_2024_external | 15 | 100.0% | $254,930 | $168,150 | -$86,780 | 60.9% |
| q4_2024_external | 30 | 100.0% | $254,930 | $106,320 | -$148,610 | 55.5% |
| q4_2024_external | 60 | 100.0% | $254,930 | -$4,420 | -$259,350 | 44.2% |
| q4_2025 | 0 | 100.0% | $455,760 | $455,760 | $0 | 75.6% |
| q4_2025 | 1 | 94.8% | $396,660 | $378,650 | -$18,010 | 73.6% |
| q4_2025 | 5 | 94.8% | $396,660 | $313,700 | -$82,960 | 69.3% |
| q4_2025 | 15 | 94.8% | $396,660 | $226,430 | -$170,230 | 61.6% |
| q4_2025 | 30 | 94.8% | $396,660 | $101,790 | -$294,870 | 56.0% |
| q4_2025 | 60 | 94.8% | $396,660 | -$71,300 | -$467,960 | 43.9% |

## Future Decision Expiry Rule

- Max option quote age: `1500ms`
- Max context age: `5000ms`
- Max entry ask move before entry: `$0.25`
- Reject missing, zero, locked, crossed, stale, non-SPXW, or non-PM-settled quotes.

## Outputs

- Delay rows: `v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/timing_delay_rows.csv`
- Split summary: `v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/split_delay_summary.csv`
- Group summary: `v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening/group_delay_summary.csv`
