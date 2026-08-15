# Q3 2025 Processed Consistency Audit

Processed dir: `data/processed/spxw_0dte_neural_q3_2025`

## Build Summary

- Sessions built: `64`
- Sessions skipped: `2025-07-04, 2025-09-01`
- Normalized rows: `10,069,430`
- Neural rows: `22,884`

## Normalized Checks

| Check | Value |
|---|---:|
| rows | 10,069,430 |
| finite_quotes | 7,204,089 |
| non_spxw | 0 |
| non_pm | 0 |
| strike_step_violations | 0 |
| bad_bid_ask | 0 |
| missing_underlying_finite_quote | 0 |
| volume_missing_finite_quote | 5,777,068 |
| oi_missing_finite_quote | 0 |

Quote gap <= 90s fraction: `n/a`
Quote gap p50/p95/p99 seconds: `n/a` / `n/a` / `n/a`

## Neural Checks

- Sessions: `64`
- Rows: `22,884`
- Candidate count p05/p50/p95: `18.0` / `28.0` / `34.0`
- Finite candidate-label fraction: `0.999945`

## Non-360 Sessions

- `2025-07-03`: `209` rows (early close; rows end at 16:59 UTC / 12:59 ET)
- `2025-07-30`: `356` rows (missing minutes: 15:20, 15:21, 15:22, 15:23)
- `2025-09-17`: `359` rows (missing minutes: 18:00)
