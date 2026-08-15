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
| march | 21 | 2808 | 1.388 | -1990 | 0.57 | 2708 |
| q4 | 59 | 11372 | 2.721 | -1506 | 0.68 | 6502 |

## Top Value-Aware Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|theta_burden_bucket=theta_ok` | 674/1.268 | 2808/1.388 | 11372/2.721 | 1.00 | True |
| 2 | `policy0|side=C|entry_pattern=vwap_pullback_resume|theta_burden_bucket=theta_ok` | 674/1.268 | 2808/1.388 | 11272/2.743 | 1.00 | True |
| 3 | `policy0|side=C|entry_pattern=pullback_resume|moneyness_bucket=itm_2p|convexity_bucket=convexity_mid` | 1212/1.430 | 1356/1.234 | 14286/3.368 | 0.92 | True |
| 4 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|convexity_bucket=convexity_low` | 2564/2.839 | 5958/2.247 | 10600/2.555 | 0.79 | True |
| 5 | `policy0|side=C|entry_pattern=vwap_pullback_resume|convexity_bucket=convexity_low` | 2524/2.824 | 5906/2.276 | 10470/2.594 | 0.79 | True |
| 6 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=B_value|spread_tax_bucket=spread_tight` | 654/1.341 | 286/1.053 | 10921/2.792 | 0.92 | False |
| 7 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=B_value` | 654/1.341 | 286/1.053 | 10921/2.792 | 0.92 | False |
| 8 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|spread_tax_bucket=spread_tight` | 544/1.216 | 68/1.009 | 13398/2.994 | 0.92 | False |
| 9 | `policy0|side=C|entry_pattern=vwap_pullback_resume|theta_burden_bucket=theta_heavy` | 534/1.262 | 226/1.036 | 10896/2.718 | 0.92 | False |
| 10 | `policy0|side=C|entry_pattern=vwap_pullback_resume|convexity_bucket=convexity_mid` | 444/1.209 | 536/1.087 | 10446/2.572 | 0.92 | False |
| 11 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=A_value` | -996/0.681 | 3964/1.855 | 9532/2.220 | 0.69 | False |
| 12 | `policy0|side=C|entry_pattern=vwap_pullback_resume|value_grade=A_value|spread_tax_bucket=spread_tight` | -996/0.681 | 3964/1.855 | 9532/2.220 | 0.69 | False |

## Value Grades

| Grade | Rows | Median Score | Median Delta | Theta Burden | Spread Tax | Breakeven ATR |
|---|---:|---:|---:|---:|---:|---:|
| A_plus_value | 81251 | -0.171 | 0.886 | 0.017 | 0.016 | 0.66 |
| A_value | 150103 | -0.463 | 0.720 | 0.035 | 0.013 | 1.81 |
| B_value | 309049 | -1.075 | 0.494 | 0.073 | 0.013 | 4.11 |
| C_or_overpay | 432551 | -1.673 | 0.192 | 0.206 | 0.029 | 10.45 |

## Interpretation

A positive pass count means the timing patterns are improved or confirmed by contract value conditions. These A+ filters should become explicit model targets: the network should learn pattern quality and contract value separately, then only trade when both agree. Results remain research leads, not live-trading permission.
