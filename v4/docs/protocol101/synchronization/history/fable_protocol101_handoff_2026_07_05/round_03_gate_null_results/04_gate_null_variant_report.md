# Protocol101 Gate-Only And Null Baseline Diagnostics

Generated: 2026-07-06T00:04:25.685648+00:00

This is an offline diagnostic. It did not train, tune thresholds, contact brokers/vendors, download data, change defaults, promote a model, or enable paper-submit.

## Configuration

- Design: `v4/audit/autoresearch/protocol101_fair_contract_expanded_jul_dec2025_128_q1_design/summary.json`
- Feature contract: `protocol101-live-v1`
- Policy: `ask_to_bid_stop35_target60_hold10m`
- Selection mode: `stable_abs_offset_20`
- Stress per trade: `$20.00`
- Null seeds: `1000`

## Results

### `put_near_after_0940_vwap_m2_10`

| Split | Gate Trades | Gate Stressed PnL | Gate PF | Null-U p95 | Null-M p95 | Gate p vs Null-U | Gate p vs Null-M |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 23 | $-950 | 0.838 | $3,996 | $3,670 | 0.9041 | 0.9600 |
| diagnostic_test | 25 | $-2,400 | 0.602 | $1,916 | $2,195 | 0.9411 | 0.9800 |

### `put_near_after_0940_vwap_m2_10_near_vwap`

| Split | Gate Trades | Gate Stressed PnL | Gate PF | Null-U p95 | Null-M p95 | Gate p vs Null-U | Gate p vs Null-M |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 25 | $3,690 | 1.893 | $5,101 | $5,255 | 0.2537 | 0.3177 |
| diagnostic_test | 26 | $-1,950 | 0.642 | $1,411 | $1,720 | 0.8292 | 0.9311 |

### `put_near_after_0940_vwap_m2_10_premium_gte_7_5`

| Split | Gate Trades | Gate Stressed PnL | Gate PF | Null-U p95 | Null-M p95 | Gate p vs Null-U | Gate p vs Null-M |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 23 | $-950 | 0.838 | $5,220 | $4,791 | 0.8392 | 0.8771 |
| diagnostic_test | 25 | $-2,400 | 0.602 | $3,320 | $3,301 | 0.9291 | 0.9710 |

## Interpretation

Gate-only is a possible candidate shape only if it is profitable after stress and beats the random-in-gate null distribution. A weak or null-like result means the current pocket should not be treated as proven edge, even if oracle labels look positive.

