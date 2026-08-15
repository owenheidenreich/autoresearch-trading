# Entry Timing Pattern Audit

Non-neural audit for transferable, causal entry timing primitives.

Transfer gate pass count: `71`
Discovered pattern rules: `480`
Simulated pattern rules: `360`

## Champion

Best pattern rule: `policy1|side=C|entry_pattern=compression_breakout|time_bucket=late_afternoon|moneyness_bucket=itm_2p`
Cross-regime positive bucket fraction: `1.00`

| Split | Trades | PnL | PF | DD | Positive Days | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|
| selection | 8 | 5004 | 8.536 | -342 | 0.75 | 2674 |
| march | 21 | 3658 | 1.550 | -2908 | 0.38 | 1443 |
| q4 | 56 | 5728 | 1.668 | -1886 | 0.54 | 3763 |

## Top Pattern Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | `policy1|side=C|entry_pattern=compression_breakout|time_bucket=late_afternoon|moneyness_bucket=itm_2p` | 5004/8.536 | 3658/1.550 | 5728/1.668 | 1.00 | True |
| 2 | `policy0|side=C|entry_pattern=compression_breakout|time_bucket=late_afternoon|moneyness_bucket=itm_1` | 4084/21.218 | 4608/2.914 | 4698/2.005 | 1.00 | True |
| 3 | `policy0|side=C|entry_pattern=compression_breakout|time_bucket=late_afternoon|moneyness_bucket=itm_2p` | 4024/19.126 | 5478/2.993 | 5168/1.951 | 1.00 | True |
| 4 | `policy2|side=C|entry_pattern=pullback_resume|moneyness_bucket=itm_1|premium_bucket=very_large_20p` | 3704/1.679 | 1956/1.145 | 10946/2.011 | 1.00 | True |
| 5 | `policy0|side=C|entry_pattern=compression_breakout|time_bucket=late_afternoon|moneyness_bucket=atm` | 3124/18.954 | 4223/3.047 | 3973/2.048 | 1.00 | True |
| 6 | `policy2|side=C|entry_pattern=pullback_resume|moneyness_bucket=otm_2_3|premium_bucket=large_8_20` | 3002/1.730 | 3196/1.369 | 10534/2.087 | 1.00 | True |
| 7 | `policy0|side=P|entry_pattern=vwap_pullback_resume|premium_bucket=very_large_20p` | 2964/2.630 | 3510/1.773 | 13840/2.459 | 1.00 | True |
| 8 | `policy0|side=P|entry_pattern=pullback_resume|time_bucket=first_30` | 2892/2.871 | 1946/1.289 | 11236/2.310 | 1.00 | True |
| 9 | `policy0|side=P|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=atm` | 2892/2.871 | 1456/1.201 | 11291/2.247 | 1.00 | True |
| 10 | `policy0|side=P|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p` | 2814/2.381 | 3070/1.520 | 15120/2.654 | 1.00 | True |
| 11 | `policy0|side=P|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=otm_1` | 2642/2.827 | 1406/1.209 | 10236/2.279 | 1.00 | True |
| 12 | `policy0|side=P|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_1` | 2404/2.260 | 2800/1.502 | 11660/2.205 | 1.00 | True |
| 13 | `policy0|side=P|entry_pattern=pullback_resume|moneyness_bucket=itm_2p|spread_quality=good` | 1672/1.805 | 2046/1.299 | 10974/2.085 | 1.00 | True |
| 14 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=atm|spread_quality=good` | 1314/2.053 | 2828/1.608 | 11826/3.662 | 1.00 | True |
| 15 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=itm_1` | 1254/1.598 | 1626/1.291 | 11124/2.833 | 1.00 | True |
| 16 | `policy0|side=C|entry_pattern=vwap_pullback_resume|premium_bucket=large_8_20` | 754/1.472 | 796/1.171 | 9756/2.750 | 1.00 | True |
| 17 | `policy0|side=C|entry_pattern=pullback_resume|moneyness_bucket=itm_1|premium_bucket=very_large_20p` | 624/1.246 | 996/1.173 | 8436/3.099 | 1.00 | True |
| 18 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=atm` | 584/1.299 | 1386/1.265 | 11052/3.192 | 1.00 | True |
| 19 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=otm_1` | 514/1.280 | 1586/1.341 | 10152/3.250 | 1.00 | True |
| 20 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=otm_2_3|premium_bucket=large_8_20` | 494/1.313 | 2626/1.716 | 7948/3.569 | 1.00 | True |
| 21 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30` | 454/1.232 | 1786/1.388 | 10632/3.215 | 1.00 | True |
| 22 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=otm_2_3` | 334/1.200 | 1056/1.247 | 8192/3.068 | 1.00 | True |
| 23 | `policy0|side=C|entry_pattern=sigma_trend_continuation|moneyness_bucket=itm_2p|spread_quality=good` | 5912/4.135 | 4576/2.284 | 8164/1.931 | 0.93 | True |
| 24 | `policy0|side=C|entry_pattern=sigma_trend_continuation|moneyness_bucket=otm_1|premium_bucket=large_8_20` | 5522/5.993 | 3906/2.533 | 7838/2.633 | 0.93 | True |
| 25 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=otm_1|spread_quality=good` | 994/1.547 | 1880/1.358 | 6396/2.148 | 0.93 | True |

## Data Notes

- Pattern candidate rows: `2,918,987`.
- CBBO-1m minute-boundary audit acceptable: `True`.
- Median p95 intraminute mid range: `$5.72`.

## Interpretation

Transferable timing leads exist if transfer_gate_pass_count is positive. The next neural target should learn these pattern primitives directly, but still abstain unless the broader causal context says the pattern is worth paying the spread. These results are research leads, not live-trading permission.
