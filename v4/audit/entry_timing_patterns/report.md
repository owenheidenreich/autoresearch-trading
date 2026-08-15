# Entry Timing Pattern Audit

Non-neural audit for transferable, causal entry timing primitives.

Transfer gate pass count: `135`
Discovered pattern rules: `480`
Simulated pattern rules: `360`

## Champion

Best pattern rule: `policy0|side=C|entry_pattern=sigma_trend_continuation|moneyness_bucket=itm_1|spread_quality=good`
Cross-regime positive bucket fraction: `1.00`

| Split | Trades | PnL | PF | DD | Positive Days | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|
| selection | 18 | 7584 | 4.901 | -954 | 0.89 | 4949 |
| march | 44 | 5732 | 1.699 | -3050 | 0.68 | 2467 |
| q4 | 126 | 15948 | 2.023 | -1984 | 0.67 | 8908 |

## Top Pattern Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | `policy0|side=C|entry_pattern=sigma_trend_continuation|moneyness_bucket=itm_1|spread_quality=good` | 7584/4.901 | 5732/1.699 | 15948/2.023 | 1.00 | True |
| 2 | `policy0|side=C|entry_pattern=last10_breakout|time_bucket=first_30|moneyness_bucket=itm_2p` | 5620/3.930 | 5714/2.520 | 13486/2.493 | 1.00 | True |
| 3 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=itm_2p` | 5392/2.453 | 4598/1.591 | 22388/3.282 | 1.00 | True |
| 4 | `policy0|side=C|entry_pattern=last10_breakout|time_bucket=first_30` | 5350/4.911 | 5824/2.879 | 9555/2.099 | 1.00 | True |
| 5 | `policy0|side=C|entry_pattern=vwap_pullback_resume|time_bucket=first_30|moneyness_bucket=itm_2p` | 5310/3.084 | 1526/1.247 | 15134/3.925 | 1.00 | True |
| 6 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=itm_1` | 5072/2.528 | 3088/1.390 | 20134/2.797 | 1.00 | True |
| 7 | `policy0|side=C|entry_pattern=sigma_trend_continuation|time_bucket=post_open_morning|moneyness_bucket=itm_2p` | 4976/2.593 | 5932/1.759 | 13624/1.974 | 1.00 | True |
| 8 | `policy0|side=C|entry_pattern=vwap_pullback_resume|time_bucket=first_30` | 4850/3.345 | 1594/1.327 | 9568/2.413 | 1.00 | True |
| 9 | `policy0|side=C|entry_pattern=sigma_trend_continuation|time_bucket=post_open_morning|moneyness_bucket=itm_1` | 4806/2.672 | 5602/1.764 | 12392/1.938 | 1.00 | True |
| 10 | `policy0|side=C|entry_pattern=sigma_trend_continuation|time_bucket=post_open_morning|moneyness_bucket=atm` | 4776/3.029 | 4872/1.708 | 11190/1.945 | 1.00 | True |
| 11 | `policy0|side=C|entry_pattern=compression_breakout|time_bucket=late_afternoon|moneyness_bucket=itm_1` | 4378/13.727 | 5346/2.116 | 7776/1.849 | 1.00 | True |
| 12 | `policy0|side=C|entry_pattern=compression_breakout|time_bucket=late_afternoon|moneyness_bucket=itm_2p` | 4348/12.626 | 6676/2.333 | 8326/1.760 | 1.00 | True |
| 13 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=atm` | 4242/2.373 | 2688/1.363 | 19510/3.012 | 1.00 | True |
| 14 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=otm_1` | 4072/2.409 | 2738/1.410 | 17950/3.088 | 1.00 | True |
| 15 | `policy0|side=C|entry_pattern=last10_breakout|time_bucket=post_open_morning|moneyness_bucket=itm_2p` | 3994/2.148 | 8926/2.395 | 11686/1.945 | 1.00 | True |
| 16 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30` | 3922/2.303 | 2988/1.443 | 18308/2.938 | 1.00 | True |
| 17 | `policy0|side=C|entry_pattern=compression_breakout|time_bucket=late_afternoon|moneyness_bucket=atm` | 3718/13.150 | 4961/2.217 | 7106/1.951 | 1.00 | True |
| 18 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=otm_2_3` | 3652/2.443 | 2078/1.342 | 14728/2.985 | 1.00 | True |
| 19 | `policy0|side=C|entry_pattern=omar_retest_bounce|moneyness_bucket=itm_2p|premium_bucket=very_large_20p` | 3416/1.982 | 8156/2.010 | 16730/2.557 | 1.00 | True |
| 20 | `policy0|side=C|entry_pattern=omar_retest_bounce|moneyness_bucket=itm_2p` | 3346/1.959 | 8116/2.005 | 16330/2.591 | 1.00 | True |
| 21 | `policy1|side=P|entry_pattern=last10_breakout|time_bucket=late_afternoon|moneyness_bucket=itm_2p` | 3198/2.133 | 5044/1.481 | 16056/1.788 | 1.00 | True |
| 22 | `policy0|side=C|entry_pattern=omar_retest_bounce|spread_quality=acceptable` | 2846/1.782 | 8968/2.236 | 16639/2.625 | 1.00 | True |
| 23 | `policy0|side=C|entry_pattern=omar_retest_bounce|premium_bucket=very_large_20p` | 2556/1.923 | 7316/1.903 | 16618/2.504 | 1.00 | True |
| 24 | `policy0|side=C|entry_pattern=omar_retest_bounce|time_bucket=first_30` | 1684/1.903 | 902/1.162 | 11776/3.622 | 1.00 | True |
| 25 | `policy0|side=C|entry_pattern=omar_retest_bounce|moneyness_bucket=otm_2_3|premium_bucket=large_8_20` | 1390/1.627 | 5976/2.100 | 9562/3.048 | 1.00 | True |

## Data Notes

- Pattern candidate rows: `2,918,987`.
- CBBO-1m minute-boundary audit acceptable: `True`.
- Median p95 intraminute mid range: `$5.72`.

## Interpretation

Transferable timing leads exist if transfer_gate_pass_count is positive. The next neural target should learn these pattern primitives directly, but still abstain unless the broader causal context says the pattern is worth paying the spread. These results are research leads, not live-trading permission.
