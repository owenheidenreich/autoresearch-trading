# A+ Contract Value Audit

Greek-aware audit for whether timing patterns improve when the contract is worth paying the spread for.

Transfer gate pass count: `5`
Discovered value rules: `12`
Simulated value rules: `12`

## Champion

Best value-aware rule: `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|theta_burden_bucket=theta_ok`
Cross-regime positive bucket fraction: `1.00`

| Split | Trades | PnL | PF | DD | Positive Days | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|
| selection | 8 | 674 | 1.268 | -1334 | 0.50 | 874 |
| march | 21 | 2748 | 1.380 | -1990 | 0.57 | 2708 |
| q4 | 59 | 11362 | 2.717 | -1506 | 0.68 | 6502 |

## Top Value-Aware Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|theta_burden_bucket=theta_ok` | 674/1.268 | 2748/1.380 | 11362/2.717 | 1.00 | True |
| 2 | `policy0|side=C|entry_pattern=vwap_pullback_resume|theta_burden_bucket=theta_ok` | 674/1.268 | 2748/1.380 | 11212/2.731 | 1.00 | True |
| 3 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=A_plus_value|theta_burden_bucket=theta_ok` | 674/1.268 | 2748/1.380 | 11212/2.731 | 1.00 | True |
| 4 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|convexity_bucket=convexity_low` | 2754/3.287 | 5958/2.247 | 10580/2.552 | 0.79 | True |
| 5 | `policy0|side=C|entry_pattern=vwap_pullback_resume|convexity_bucket=convexity_low` | 2714/3.273 | 5926/2.286 | 10500/2.633 | 0.79 | True |
| 6 | `policy0|side=C|entry_pattern=vwap_pullback_resume|premium_bucket=very_large_20p|value_grade=A_plus_value` | 584/1.285 | 536/1.088 | 12046/2.601 | 1.00 | False |
| 7 | `policy0|side=C|entry_pattern=vwap_pullback_resume|theta_burden_bucket=theta_heavy` | 424/1.208 | 486/1.086 | 10166/2.641 | 1.00 | False |
| 8 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=A_plus_value|theta_burden_bucket=theta_heavy` | 424/1.208 | 486/1.086 | 10166/2.641 | 1.00 | False |
| 9 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|value_grade=A_plus_value` | 544/1.216 | 68/1.009 | 13388/2.989 | 0.92 | False |
| 10 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|spread_tax_bucket=spread_tight` | 544/1.216 | 68/1.009 | 13388/2.989 | 0.92 | False |
| 11 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|value_grade=A_plus_value|spread_tax_bucket=spread_tight` | 544/1.216 | 68/1.009 | 13388/2.989 | 0.92 | False |
| 12 | `policy0|side=C|entry_pattern=vwap_pullback_resume|convexity_bucket=convexity_mid` | 334/1.157 | 196/1.032 | 9966/2.521 | 0.92 | False |

## Value Grades

| Grade | Rows | Median Score | Median Delta | Theta Burden | Spread Tax | Breakeven ATR |
|---|---:|---:|---:|---:|---:|---:|
| A_plus_value | 972954 | -9.000 | 0.401 | 0.104 | 0.018 | nan |

## Interpretation

A positive pass count means the timing patterns are improved or confirmed by contract value conditions. These A+ filters should become explicit model targets: the network should learn pattern quality and contract value separately, then only trade when both agree. Results remain research leads, not live-trading permission.
