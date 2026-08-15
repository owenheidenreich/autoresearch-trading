# A+ Contract Value Audit

Greek-aware audit for whether timing patterns improve when the contract is worth paying the spread for.

Transfer gate pass count: `143`
Discovered value rules: `540`
Simulated value rules: `420`

## Champion

Best value-aware rule: `policy2|side=C|entry_pattern=last10_breakout|moneyness_bucket=itm_2p|convexity_bucket=convexity_mid`
Cross-regime positive bucket fraction: `1.00`

| Split | Trades | PnL | PF | DD | Positive Days | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|
| selection | 8 | 5484 | 2.760 | -1662 | 0.62 | 4676 |
| march | 22 | 5846 | 1.587 | -4898 | 0.50 | 3196 |
| q4 | 50 | 21900 | 2.800 | -2688 | 0.66 | 13040 |

## Top Value-Aware Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | `policy2|side=C|entry_pattern=last10_breakout|moneyness_bucket=itm_2p|convexity_bucket=convexity_mid` | 5484/2.760 | 5846/1.587 | 21900/2.800 | 1.00 | True |
| 2 | `policy2|side=C|entry_pattern=pullback_resume|premium_bucket=very_large_20p|value_grade=B_value` | 3824/1.786 | 916/1.074 | 15026/2.065 | 1.00 | True |
| 3 | `policy0|side=P|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|value_grade=A_value` | 3354/3.013 | 8200/3.038 | 13840/2.517 | 1.00 | True |
| 4 | `policy0|side=P|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|value_grade=A_value|spread_tax_bucket=spread_tight` | 3354/3.013 | 8200/3.038 | 13840/2.517 | 1.00 | True |
| 5 | `policy0|side=P|entry_pattern=pullback_resume|moneyness_bucket=itm_2p|convexity_bucket=convexity_mid` | 3262/2.777 | 4986/1.696 | 11336/1.984 | 1.00 | True |
| 6 | `policy0|side=C|entry_pattern=vwap_hold_continuation|moneyness_bucket=itm_1|value_grade=A_value` | 3174/5.445 | 5566/13.424 | 5424/2.347 | 1.00 | True |
| 7 | `policy0|side=C|entry_pattern=vwap_hold_continuation|moneyness_bucket=itm_1|value_grade=A_value|spread_tax_bucket=spread_tight` | 3174/5.445 | 5566/13.424 | 5256/2.306 | 1.00 | True |
| 8 | `policy0|side=P|entry_pattern=vwap_pullback_resume|value_grade=A_value|breakeven_atr_bucket=breakeven_stretched` | 3064/2.460 | 7212/2.676 | 13266/2.392 | 1.00 | True |
| 9 | `policy0|side=P|entry_pattern=vwap_pullback_resume|value_grade=A_value` | 2904/2.538 | 5590/2.249 | 12870/2.524 | 1.00 | True |
| 10 | `policy0|side=P|entry_pattern=vwap_pullback_resume|value_grade=A_value|spread_tax_bucket=spread_tight` | 2904/2.538 | 5590/2.249 | 12870/2.524 | 1.00 | True |
| 11 | `policy0|side=P|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|convexity_bucket=convexity_low` | 2634/2.398 | 5104/2.024 | 8782/1.867 | 1.00 | True |
| 12 | `policy0|side=C|entry_pattern=compression_breakout|moneyness_bucket=itm_2p|value_grade=A_plus_value` | 2362/2.927 | 3712/2.144 | 5394/1.648 | 1.00 | True |
| 13 | `policy0|side=C|entry_pattern=compression_breakout|premium_bucket=very_large_20p|value_grade=A_plus_value` | 2362/2.927 | 4072/2.255 | 4976/1.586 | 1.00 | True |
| 14 | `policy0|side=C|entry_pattern=compression_breakout|value_grade=A_plus_value` | 2362/2.927 | 3712/2.144 | 5189/1.624 | 1.00 | True |
| 15 | `policy0|side=C|entry_pattern=compression_breakout|moneyness_bucket=itm_2p|value_grade=A_plus_value|spread_tax_bucket=spread_tight` | 2362/2.927 | 3712/2.144 | 5046/1.607 | 1.00 | True |
| 16 | `policy0|side=C|entry_pattern=compression_breakout|value_grade=A_plus_value|spread_tax_bucket=spread_tight` | 2362/2.927 | 3712/2.144 | 5006/1.602 | 1.00 | True |
| 17 | `policy0|side=P|entry_pattern=vwap_pullback_resume|convexity_bucket=convexity_low` | 1834/1.973 | 4674/1.938 | 8522/1.867 | 1.00 | True |
| 18 | `policy0|side=C|entry_pattern=vwap_pullback_resume|premium_bucket=very_large_20p|value_grade=A_value` | 1274/1.543 | 4010/1.779 | 13794/3.462 | 1.00 | True |
| 19 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=A_value|breakeven_atr_bucket=breakeven_far` | 1254/1.534 | 4080/1.872 | 14414/3.763 | 1.00 | True |
| 20 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=A_value|theta_burden_bucket=theta_ok` | 1184/1.470 | 4670/1.845 | 12730/3.396 | 1.00 | True |
| 21 | `policy0|side=C|entry_pattern=pullback_resume|value_grade=A_value|breakeven_atr_bucket=breakeven_far` | 1172/1.402 | 706/1.105 | 13762/3.308 | 1.00 | True |
| 22 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|theta_burden_bucket=theta_ok` | 674/1.268 | 2808/1.388 | 11372/2.721 | 1.00 | True |
| 23 | `policy0|side=C|entry_pattern=vwap_pullback_resume|theta_burden_bucket=theta_ok` | 674/1.268 | 2808/1.388 | 11272/2.743 | 1.00 | True |
| 24 | `policy0|side=C|entry_pattern=last10_breakout|value_grade=A_value|theta_burden_bucket=theta_heavy` | 3352/3.418 | 6936/2.937 | 7836/1.965 | 0.94 | True |
| 25 | `policy0|side=C|entry_pattern=compression_breakout|moneyness_bucket=itm_1|theta_burden_bucket=theta_extreme` | 3064/4.908 | 3508/2.663 | 4248/1.800 | 0.94 | True |

## Value Grades

| Grade | Rows | Median Score | Median Delta | Theta Burden | Spread Tax | Breakeven ATR |
|---|---:|---:|---:|---:|---:|---:|
| A_plus_value | 240908 | -0.207 | 0.887 | 0.034 | 0.016 | 0.68 |
| A_value | 458156 | -0.545 | 0.717 | 0.084 | 0.013 | 1.87 |
| B_value | 947360 | -1.267 | 0.484 | 0.153 | 0.014 | 4.29 |
| C_or_overpay | 1272563 | -2.075 | 0.198 | 0.577 | 0.028 | 10.07 |

## Interpretation

A positive pass count means the timing patterns are improved or confirmed by contract value conditions. These A+ filters should become explicit model targets: the network should learn pattern quality and contract value separately, then only trade when both agree. Results remain research leads, not live-trading permission.
