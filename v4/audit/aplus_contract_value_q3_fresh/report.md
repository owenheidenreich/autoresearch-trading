# A+ Contract Value Audit

Greek-aware audit for whether timing patterns improve when the contract is worth paying the spread for.

Transfer gate pass count: `140`
Discovered value rules: `540`
Simulated value rules: `420`

## Champion

Best value-aware rule: `policy2|side=C|entry_pattern=last10_breakout|convexity_bucket=convexity_mid`
Cross-regime positive bucket fraction: `1.00`

| Split | Trades | PnL | PF | DD | Positive Days | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|
| selection | 9 | 6782 | 3.026 | -1452 | 0.56 | 5164 |
| march | 22 | 11146 | 2.345 | -3646 | 0.45 | 7811 |
| q4 | 64 | 9882 | 1.834 | -2136 | 0.48 | 5777 |

## Top Value-Aware Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | `policy2|side=C|entry_pattern=last10_breakout|convexity_bucket=convexity_mid` | 6782/3.026 | 11146/2.345 | 9882/1.834 | 1.00 | True |
| 2 | `policy2|side=P|entry_pattern=compression_breakout|moneyness_bucket=itm_2p|convexity_bucket=convexity_mid` | 5384/3.066 | 5010/1.720 | 8546/1.900 | 1.00 | True |
| 3 | `policy1|side=P|entry_pattern=pullback_resume|value_grade=A_value|theta_burden_bucket=theta_heavy` | 4242/1.969 | 10986/2.037 | 10592/1.786 | 1.00 | True |
| 4 | `policy0|side=C|entry_pattern=last10_breakout|value_grade=A_value|theta_burden_bucket=theta_heavy` | 3352/3.418 | 6936/2.937 | 7814/2.558 | 1.00 | True |
| 5 | `policy2|side=C|entry_pattern=pullback_resume|moneyness_bucket=atm|convexity_bucket=convexity_mid` | 3312/1.639 | 976/1.085 | 7462/1.564 | 1.00 | True |
| 6 | `policy0|side=P|entry_pattern=pullback_resume|moneyness_bucket=itm_2p|convexity_bucket=convexity_mid` | 3262/2.777 | 4986/1.696 | 11470/3.033 | 1.00 | True |
| 7 | `policy0|side=C|entry_pattern=compression_breakout|moneyness_bucket=itm_1|theta_burden_bucket=theta_extreme` | 3064/4.908 | 3508/2.663 | 2426/1.677 | 1.00 | True |
| 8 | `policy0|side=P|entry_pattern=vwap_pullback_resume|value_grade=A_value|breakeven_atr_bucket=breakeven_stretched` | 3064/2.460 | 7212/2.676 | 3850/1.479 | 1.00 | True |
| 9 | `policy0|side=P|entry_pattern=vwap_pullback_resume|value_grade=A_value` | 2904/2.538 | 5590/2.249 | 3260/1.423 | 1.00 | True |
| 10 | `policy0|side=P|entry_pattern=vwap_pullback_resume|value_grade=A_value|spread_tax_bucket=spread_tight` | 2904/2.538 | 5590/2.249 | 3260/1.423 | 1.00 | True |
| 11 | `policy0|side=P|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|convexity_bucket=convexity_low` | 2634/2.398 | 5104/2.024 | 2428/1.222 | 1.00 | True |
| 12 | `policy0|side=C|entry_pattern=last10_breakout|moneyness_bucket=itm_1|convexity_bucket=convexity_mid` | 2162/1.997 | 2946/1.584 | 4684/1.700 | 1.00 | True |
| 13 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=A_value|theta_burden_bucket=theta_heavy` | 2014/2.727 | 3702/2.019 | 6326/3.378 | 1.00 | True |
| 14 | `policy0|side=P|entry_pattern=vwap_pullback_resume|convexity_bucket=convexity_low` | 1834/1.973 | 4674/1.938 | 2698/1.263 | 1.00 | True |
| 15 | `policy0|side=C|entry_pattern=pullback_resume|value_grade=A_value|breakeven_atr_bucket=breakeven_far` | 1172/1.402 | 706/1.105 | 4972/1.710 | 1.00 | True |
| 16 | `policy0|side=C|entry_pattern=vwap_pullback_resume|premium_bucket=large_8_20|value_grade=B_value` | 694/1.419 | 626/1.142 | 5130/2.119 | 1.00 | True |
| 17 | `policy0|side=C|entry_pattern=sigma_trend_continuation|moneyness_bucket=itm_2p|theta_burden_bucket=theta_heavy` | 8002/16.629 | 2916/1.973 | 4758/1.853 | 0.93 | True |
| 18 | `policy2|side=C|entry_pattern=last10_breakout|moneyness_bucket=atm|convexity_bucket=convexity_mid` | 9512/8.573 | 10696/2.696 | 7174/1.573 | 0.93 | True |
| 19 | `policy0|side=C|entry_pattern=sigma_trend_continuation|theta_burden_bucket=theta_heavy` | 5692/5.358 | 1436/1.386 | 3292/1.558 | 0.93 | True |
| 20 | `policy0|side=C|entry_pattern=sigma_trend_continuation|value_grade=B_value|theta_burden_bucket=theta_heavy` | 5632/5.312 | 1436/1.386 | 3457/1.611 | 0.93 | True |
| 21 | `policy0|side=C|entry_pattern=sigma_trend_continuation|convexity_bucket=convexity_mid` | 5262/5.029 | 1586/1.454 | 3132/1.594 | 0.93 | True |
| 22 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=atm|value_grade=B_value` | 1974/2.047 | 4246/1.927 | 7972/3.037 | 0.93 | True |
| 23 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=atm|value_grade=B_value|spread_tax_bucket=spread_tight` | 1974/2.047 | 4246/1.927 | 7424/2.839 | 0.93 | True |
| 24 | `policy2|side=C|entry_pattern=pullback_resume|value_grade=B_value|breakeven_atr_bucket=breakeven_reachable` | 1184/1.599 | 5022/1.522 | 10094/2.381 | 0.93 | True |
| 25 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=A_value|theta_burden_bucket=theta_ok` | 1184/1.470 | 4670/1.845 | 5152/1.894 | 0.93 | True |

## Value Grades

| Grade | Rows | Median Score | Median Delta | Theta Burden | Spread Tax | Breakeven ATR |
|---|---:|---:|---:|---:|---:|---:|
| A_plus_value | 328686 | -0.194 | 0.899 | 0.029 | 0.015 | 0.66 |
| A_value | 426728 | -0.531 | 0.731 | 0.079 | 0.013 | 1.83 |
| B_value | 790645 | -1.270 | 0.489 | 0.150 | 0.014 | 4.30 |
| C_or_overpay | 1159551 | -2.084 | 0.186 | 0.573 | 0.031 | 10.78 |

## Interpretation

A positive pass count means the timing patterns are improved or confirmed by contract value conditions. These A+ filters should become explicit model targets: the network should learn pattern quality and contract value separately, then only trade when both agree. Results remain research leads, not live-trading permission.
