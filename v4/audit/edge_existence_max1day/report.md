# Edge Existence Audit

Non-neural audit for whether current v4 data contains transparent, picky long-call/long-put entry pockets after executable ask-entry / bid-exit labels.

Simulation gate pass count: `0`
Discovered candidate rules: `480`
Simulated rules: `360`

## Data Fidelity

- Normalized files audited: `125` with `22,735,080` rows.
- Bad root / settlement / 5-point strike fractions: `0.000000` / `0.000000` / `0.000000`.
- Bad finite quote fraction: `0.000000`.
- Quote time after event fraction: `0.000000`.
- Quote gap p95/p99 seconds on finite quotes: `nan` / `nan`.
- Candidate rows: `3,820,203` across `125` sessions.
- Median spread: `$0.20`; p95 spread: `$0.50`.
- Median spread fraction: `0.019`; p95 spread fraction: `0.074`.
- Median mid-label advantage over executable net label: `$15.00`.
- CBBO-1s audit available: `True`; minute-boundary acceptable: `True`.
- Median p95 intraminute mid range in 1s audit: `$5.72`.

## Champion

Best simulated rule: `policy0|side=C|time_bucket=post_open_morning|sigma_side=sigma_aligned|trend15_side=trend15_aligned`

| Split | Trades | PnL | PF | DD | Positive Days | Top-Day Share | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| selection | 9 | 7852 | 27.000 | -302 | 0.89 | 0.32 | 480 |
| march | 22 | 256 | 1.058 | -1536 | 0.36 | 0.22 | -829 |
| q4 | 64 | 2312 | 1.307 | -2538 | 0.55 | 0.10 | -3130 |

## Top Simulated Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Pass |
|---:|---|---:|---:|---:|---|
| 1 | `policy0|side=C|time_bucket=post_open_morning|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 7852/27.000 | 256/1.058 | 2312/1.307 | False |
| 2 | `policy2|side=C|moneyness_bucket=atm|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 7472/13.795 | 5446/1.556 | 2097/1.116 | False |
| 3 | `policy2|side=C|moneyness_bucket=otm_1|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 6982/13.164 | 4976/1.575 | 1742/1.109 | False |
| 4 | `policy2|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 6402/11.962 | 5506/1.505 | 2592/1.115 | False |
| 5 | `policy2|side=C|moneyness_bucket=itm_1|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 5812/10.623 | 4706/1.449 | 2292/1.112 | False |
| 6 | `policy1|side=C|time_bucket=post_open_morning|sigma_side=sigma_aligned|trend15_side=trend15_flat` | 4962/12.176 | 4950/2.081 | 1553/1.147 | False |
| 7 | `policy0|side=C|time_bucket=post_open_morning|moneyness_bucket=itm_2p|sigma_side=sigma_aligned` | 4712/7.790 | 566/1.095 | 5532/1.574 | False |
| 8 | `policy0|side=C|time_bucket=post_open_morning|moneyness_bucket=itm_1|sigma_side=sigma_aligned` | 4672/8.255 | 536/1.096 | 5872/1.714 | False |
| 9 | `policy1|side=C|time_bucket=post_open_morning|moneyness_bucket=itm_1|sigma_side=sigma_aligned` | 4582/3.563 | 286/1.034 | 5602/1.410 | False |
| 10 | `policy2|side=P|time_bucket=first_30|moneyness_bucket=itm_1|premium_bucket=large_8_20` | 4542/4.565 | 1076/inf | 4872/1.268 | False |
| 11 | `policy2|side=P|time_bucket=first_30|first15_side=first15_aligned|last10_break_bucket=last10_down_break` | 4532/3.978 | 964/1.175 | 4154/1.526 | False |
| 12 | `policy0|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 4452/4.312 | 3176/1.700 | 10912/2.324 | False |
| 13 | `policy0|side=C|moneyness_bucket=itm_1|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 4402/5.663 | 3026/1.719 | 10192/2.389 | False |
| 14 | `policy0|side=C|time_bucket=post_open_morning|moneyness_bucket=atm|sigma_side=sigma_aligned` | 4392/8.272 | 256/1.050 | 5222/1.697 | False |
| 15 | `policy0|side=C|time_bucket=first_30|first15_side=first15_aligned|last10_break_bucket=last10_up_break` | 4370/12.380 | 1996/2.135 | 4532/1.983 | False |
| 16 | `policy0|side=C|time_bucket=post_open_morning|moneyness_bucket=otm_1|sigma_side=sigma_aligned` | 4362/8.734 | 216/1.046 | 4172/1.638 | False |
| 17 | `policy1|side=C|time_bucket=post_open_morning|moneyness_bucket=atm|sigma_side=sigma_aligned` | 4342/3.667 | 116/1.015 | 4492/1.351 | False |
| 18 | `policy0|side=C|moneyness_bucket=atm|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 4152/5.806 | 2866/1.754 | 8342/2.129 | False |
| 19 | `policy1|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 4122/2.599 | 3746/1.473 | 4222/1.278 | False |
| 20 | `policy1|side=C|time_bucket=post_open_morning|moneyness_bucket=otm_1|sigma_side=sigma_aligned` | 4072/3.774 | 106/1.015 | 3972/1.366 | False |

## Interpretation

A pass would show at least one transparent, picky entry pocket that survives the frozen audits. A zero pass count means the current dataset may be cleaner than v3, but still does not yet show a robust exploitable long-premium edge under these trader-style cells.
