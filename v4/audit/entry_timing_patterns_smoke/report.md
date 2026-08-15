# Entry Timing Pattern Audit

Non-neural audit for transferable, causal entry timing primitives.

Transfer gate pass count: `8`
Discovered pattern rules: `10`
Simulated pattern rules: `10`

## Champion

Best pattern rule: `policy0|side=C|entry_pattern=vwap_pullback_resume|time_bucket=first_30`
Cross-regime positive bucket fraction: `1.00`

| Split | Trades | PnL | PF | DD | Positive Days | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|
| selection | 10 | 4850 | 3.345 | -1054 | 0.71 | 3470 |
| march | 23 | 1594 | 1.327 | -1244 | 0.53 | 879 |
| q4 | 61 | 9568 | 2.413 | -2354 | 0.70 | 7053 |

## Top Pattern Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | `policy0|side=C|entry_pattern=vwap_pullback_resume|time_bucket=first_30` | 4850/3.345 | 1594/1.327 | 9568/2.413 | 1.00 | True |
| 2 | `policy0|side=C|entry_pattern=pullback_resume|time_bucket=first_30` | 3922/2.303 | 2988/1.443 | 18308/2.938 | 1.00 | True |
| 3 | `policy0|side=P|entry_pattern=pullback_resume|time_bucket=first_30|moneyness_bucket=itm_2p` | 3606/2.542 | 4690/1.400 | 13668/1.722 | 0.85 | True |
| 4 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|spread_quality=acceptable` | 5418/2.242 | 5732/1.581 | 20964/2.763 | 0.84 | True |
| 5 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p` | 5118/2.220 | 5732/1.623 | 22634/3.039 | 0.84 | True |
| 6 | `policy0|side=C|entry_pattern=vwap_pullback_resume|premium_bucket=very_large_20p` | 4948/2.408 | 5720/1.743 | 18110/2.275 | 0.84 | True |
| 7 | `policy0|side=C|entry_pattern=vwap_pullback_resume|moneyness_bucket=itm_2p|premium_bucket=very_large_20p` | 5208/2.245 | 5732/1.623 | 22814/2.928 | 0.79 | True |
| 8 | `policy0|side=C|entry_pattern=vwap_pullback_resume|spread_quality=acceptable` | 5088/2.166 | 6580/1.776 | 16455/2.152 | 0.79 | True |
| 9 | `policy0|side=C|entry_pattern=omar_retest_bounce|time_bucket=post_open_morning` | 2226/3.262 | 5840/2.129 | 1536/1.222 | 0.92 | False |
| 10 | `policy0|side=C|entry_pattern=sigma_trend_continuation|time_bucket=first_30` | 4270/7.843 | 1196/1.367 | 6270/2.716 | 0.89 | False |

## Data Notes

- Pattern candidate rows: `972,954`.
- CBBO-1m minute-boundary audit acceptable: `True`.
- Median p95 intraminute mid range: `$5.72`.

## Interpretation

Transferable timing leads exist if transfer_gate_pass_count is positive. The next neural target should learn these pattern primitives directly, but still abstain unless the broader causal context says the pattern is worth paying the spread. These results are research leads, not live-trading permission.
