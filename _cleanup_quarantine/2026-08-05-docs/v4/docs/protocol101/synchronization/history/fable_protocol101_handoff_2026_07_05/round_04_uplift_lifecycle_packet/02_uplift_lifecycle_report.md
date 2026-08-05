# Protocol101 Label Uplift And Lifecycle Attribution Packet

Generated: 2026-07-06T00:59:28.321865+00:00

Offline provisional diagnostic only. No training, threshold tuning, broker calls, paid downloads, paper-submit, default changes, promotion changes, or recorder-day selection use occurred.

## Gate / Sampler Summary

| Gate | Split | First-Eligible PnL | Slot PnL | All-Gated Winsor Mean |
|---|---:|---:|---:|---:|
| `put_near_after_0940_vwap_m2_10` | validation | $-950 | $235 | $5.63 |
| `put_near_after_0940_vwap_m2_10` | diagnostic_test | $-2,400 | $-110 | $4.92 |
| `put_near_after_0940_vwap_m2_10_near_vwap` | validation | $3,690 | $1,490 | $34.27 |
| `put_near_after_0940_vwap_m2_10_near_vwap` | diagnostic_test | $-1,950 | $-630 | $14.33 |
| `put_near_after_0940_vwap_m2_10_premium_gte_7_5` | validation | $-950 | $-310 | $-8.71 |
| `put_near_after_0940_vwap_m2_10_premium_gte_7_5` | diagnostic_test | $-2,400 | $980 | $21.16 |

## Candidate-Matched Null Rows

| Candidate | Gate | Split | Candidate PnL | Null-M p95 | Upper p | Lower p |
|---|---|---:|---:|---:|---:|---:|
| attempt131 | `put_near_after_0940_vwap_m2_10` | validation | $2,540 | $4,501 | 0.3017 | 0.7013 |
| attempt131 | `put_near_after_0940_vwap_m2_10` | diagnostic_test | $-330 | $2,295 | 0.6693 | 0.3327 |
| attempt130 | `put_near_after_0940_vwap_m2_10_near_vwap` | validation | $4,760 | $4,817 | 0.0549 | 0.9461 |
| attempt130 | `put_near_after_0940_vwap_m2_10_near_vwap` | diagnostic_test | $50 | $1,050 | 0.2627 | 0.7393 |
