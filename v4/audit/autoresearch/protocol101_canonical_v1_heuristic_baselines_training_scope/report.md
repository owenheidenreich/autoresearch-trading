# Protocol101 Canonical V1 Fixed-Heuristic Baselines

- Status: `complete`
- Sessions: `271`
- Candidate rows: `3412556`
- Null canary: `v4/audit/autoresearch/protocol101_canonical_v1_null_canary_training_scope/summary.json`
- Highest allowed claim: canonical v1 fixed-heuristic baseline packet complete

## Top Results

| Rank | Policy | Heuristic | PnL | Null z | Max DD % | Trades/day | Gates |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 4 | put_call_ratio_skew | $15,953 | 2.15 | 151.10% | 2.46 | fail |
| 2 | 5 | internal_delta_geometry | $5,275 | 0.70 | 113.49% | 1.00 | fail |
| 3 | 6 | internal_delta_geometry | $5,275 | 0.56 | 113.49% | 1.00 | fail |
| 4 | 1 | straddle_mid_expansion | $-975 | 1.56 | 126.42% | 3.67 | fail |
| 5 | 3 | put_call_ratio_skew | $-5,577 | -0.08 | 114.06% | 3.15 | fail |
| 6 | 5 | put_call_ratio_skew | $-6,371 | -0.20 | 158.97% | 0.99 | fail |
| 7 | 6 | near_atm_band_momentum | $-6,390 | -0.29 | 134.18% | 1.00 | fail |
| 8 | 6 | put_call_ratio_skew | $-8,896 | -0.47 | 158.97% | 0.99 | fail |
| 9 | 2 | put_call_ratio_skew | $-10,493 | -0.39 | 147.25% | 5.25 | fail |
| 10 | 3 | straddle_mid_expansion | $-11,453 | -0.71 | 171.73% | 1.72 | fail |
| 11 | 5 | near_atm_band_momentum | $-12,950 | -0.70 | 134.18% | 1.00 | fail |
| 12 | 4 | near_atm_band_momentum | $-16,052 | -1.07 | 115.37% | 2.42 | fail |

## Interpretation

This packet is a fixed-rule baseline, not a trained candidate. It is useful as a G3 reference and as a probe for whether simple canonical features carry obvious edge before model search.

The script writes no model artifacts and records all broker, paper, paid-data, launchd, runtime, promotion, and real-money side-effect flags as false.
