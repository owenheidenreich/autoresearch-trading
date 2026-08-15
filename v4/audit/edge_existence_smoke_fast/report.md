# Edge Existence Audit

Non-neural audit for whether current v4 data contains transparent, picky long-call/long-put entry pockets after executable ask-entry / bid-exit labels.

Simulation gate pass count: `4`
Discovered candidate rules: `10`
Simulated rules: `10`

## Data Fidelity

- Normalized files audited: `125` with `22,735,080` rows.
- Bad root / settlement / 5-point strike fractions: `0.000000` / `0.000000` / `0.000000`.
- Bad finite quote fraction: `0.000000`.
- Quote time after event fraction: `0.000000`.
- Quote gap p95/p99 seconds on finite quotes: `nan` / `nan`.
- Candidate rows: `1,273,391` across `125` sessions.
- Median spread: `$0.20`; p95 spread: `$0.50`.
- Median spread fraction: `0.019`; p95 spread fraction: `0.074`.
- Median mid-label advantage over executable net label: `$15.00`.
- CBBO-1s audit available: `True`; minute-boundary acceptable: `True`.
- Median p95 intraminute mid range in 1s audit: `$5.72`.

## Champion

Best simulated rule: `policy0|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned|trend15_side=trend15_aligned`

| Split | Trades | PnL | PF | DD | Positive Days | Top-Day Share | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| selection | 18 | 9084 | 4.478 | -1364 | 0.89 | 0.28 | 739 |
| march | 44 | 5512 | 1.722 | -3092 | 0.59 | 0.20 | -4273 |
| q4 | 128 | 11134 | 1.574 | -3532 | 0.59 | 0.08 | -4516 |

## Top Simulated Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Pass |
|---:|---|---:|---:|---:|---|
| 1 | `policy0|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 9084/4.478 | 5512/1.722 | 11134/1.574 | True |
| 2 | `policy0|side=C|moneyness_bucket=itm_1|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 8924/5.245 | 5302/1.741 | 9434/1.534 | True |
| 3 | `policy0|side=C|moneyness_bucket=atm|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 8424/5.383 | 4902/1.747 | 7774/1.476 | True |
| 4 | `policy0|side=C|time_bucket=post_open_morning|moneyness_bucket=itm_2p|sigma_side=sigma_aligned` | 5524/2.977 | 6202/1.668 | 12034/1.660 | True |
| 5 | `policy0|side=P|time_bucket=post_open_morning|first15_side=first15_aligned|last10_break_bucket=last10_down_break` | 2136/8.796 | 3596/1.849 | 10198/1.993 | False |
| 6 | `policy0|side=C|time_bucket=first_30|sigma_side=sigma_aligned` | 1254/1.342 | 2902/1.327 | 18244/2.167 | False |
| 7 | `policy0|side=C|time_bucket=post_open_morning|first15_side=first15_counter|last10_break_bucket=last10_up_break` | 218/1.194 | 5264/2.675 | 3388/1.604 | False |
| 8 | `policy0|side=P|time_bucket=first_30|vix_bucket=vix_high|range_bucket=range_mid` | 2102/14.829 | -1268/0.562 | -2824/0.272 | False |
| 9 | `policy0|side=P|time_bucket=first_30|range_bucket=range_mid` | 2102/14.829 | -1268/0.562 | -2824/0.272 | False |
| 10 | `policy0|side=P|time_bucket=first_30|vwap_side=vwap_aligned|trend15_side=trend15_aligned` | -500/0.862 | -1162/0.808 | -6836/0.557 | False |

## Interpretation

A pass would show at least one transparent, picky entry pocket that survives the frozen audits. A zero pass count means the current dataset may be cleaner than v3, but still does not yet show a robust exploitable long-premium edge under these trader-style cells.
