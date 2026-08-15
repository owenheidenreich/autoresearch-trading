# Edge Existence Audit

Non-neural audit for whether current v4 data contains transparent, picky long-call/long-put entry pockets after executable ask-entry / bid-exit labels.

Simulation gate pass count: `43`
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
- Candidate IV missing fraction: `0.000`; gamma missing fraction: `0.000`.
- Median mid-label advantage over executable net label: `$15.00`.
- CBBO-1s audit available: `True`; minute-boundary acceptable: `True`.
- Median p95 intraminute mid range in 1s audit: `$5.72`.
- Normalized contract multiplier counts: `{'2147483647': 22735080}`.
- Normalized min price increment missing fraction: `1.000`.

## Champion

Best simulated rule: `policy2|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned`

| Split | Trades | PnL | PF | DD | Positive Days | Top-Day Share | Random Same-Time PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| selection | 18 | 9694 | 2.292 | -3014 | 0.56 | 0.51 | 161 |
| march | 44 | 11632 | 1.515 | -4964 | 0.55 | 0.14 | -1500 |
| q4 | 128 | 8844 | 1.186 | -9152 | 0.56 | 0.11 | -891 |

## Top Simulated Rules

| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Pass |
|---:|---|---:|---:|---:|---|
| 1 | `policy2|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned` | 9694/2.292 | 11632/1.515 | 8844/1.186 | True |
| 2 | `policy2|side=C|moneyness_bucket=itm_1|sigma_side=sigma_aligned` | 9414/2.384 | 6922/1.309 | 7294/1.168 | True |
| 3 | `policy0|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 9084/4.478 | 5512/1.722 | 11134/1.574 | True |
| 4 | `policy0|side=C|moneyness_bucket=itm_1|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 8924/5.245 | 5302/1.741 | 9434/1.534 | True |
| 5 | `policy2|side=C|moneyness_bucket=atm|sigma_side=sigma_aligned` | 8874/2.478 | 4552/1.213 | 4484/1.117 | True |
| 6 | `policy0|side=C|moneyness_bucket=atm|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 8424/5.383 | 4902/1.747 | 7774/1.476 | True |
| 7 | `policy0|side=C|time_bucket=post_open_morning|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 7571/5.807 | 1972/1.258 | 5618/1.388 | True |
| 8 | `policy0|side=C|moneyness_bucket=otm_1|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 7514/4.810 | 4532/1.748 | 6914/1.482 | True |
| 9 | `policy2|side=C|moneyness_bucket=itm_2p|sigma_side=sigma_aligned|trend15_side=trend15_flat` | 7304/2.166 | 11682/1.575 | 7284/1.158 | True |
| 10 | `policy2|side=C|moneyness_bucket=itm_1|sigma_side=sigma_aligned|trend15_side=trend15_flat` | 7004/2.207 | 8002/1.377 | 7224/1.177 | True |
| 11 | `policy0|side=C|moneyness_bucket=otm_2_3|sigma_side=sigma_aligned|trend15_side=trend15_aligned` | 6824/4.940 | 3912/1.727 | 5374/1.448 | True |
| 12 | `policy2|side=C|moneyness_bucket=atm|sigma_side=sigma_aligned|trend15_side=trend15_flat` | 6494/2.252 | 6792/1.347 | 3919/1.104 | True |
| 13 | `policy2|side=C|time_bucket=post_open_morning|sigma_side=sigma_aligned` | 6084/2.466 | 1612/1.089 | 3656/1.129 | True |
| 14 | `policy1|side=C|time_bucket=post_open_morning|sigma_side=sigma_aligned|trend15_side=trend15_counter` | 5974/3.689 | 1824/1.187 | 9506/1.462 | True |
| 15 | `policy2|side=C|moneyness_bucket=otm_2_3|sigma_side=sigma_aligned|trend15_side=trend15_flat` | 5874/2.563 | 4352/1.263 | 3634/1.128 | True |
| 16 | `policy0|side=C|moneyness_bucket=itm_1|sigma_side=sigma_aligned|trend15_side=trend15_flat` | 5704/3.324 | 7042/1.886 | 16444/1.848 | True |
| 17 | `policy1|side=C|time_bucket=post_open_morning|moneyness_bucket=itm_1|sigma_side=sigma_aligned` | 5634/2.982 | 2842/1.206 | 8576/1.310 | True |
| 18 | `policy0|side=C|time_bucket=post_open_morning|moneyness_bucket=itm_2p|sigma_side=sigma_aligned` | 5524/2.977 | 6202/1.668 | 12034/1.660 | True |
| 19 | `policy0|side=C|time_bucket=post_open_morning|moneyness_bucket=itm_1|sigma_side=sigma_aligned` | 5404/3.091 | 5822/1.669 | 11894/1.752 | True |
| 20 | `policy1|side=C|time_bucket=post_open_morning|moneyness_bucket=atm|sigma_side=sigma_aligned` | 5164/2.985 | 2632/1.213 | 6926/1.270 | True |

## Interpretation

This audit found transparent, non-neural edge-existence leads if simulation_gate_pass_count is positive. The leads should not be promoted directly to live trading: they are templates for the next model target. The strongest theme is call-side participation when the causal SPX/VWAP sigma state is call-aligned, usually with near-ATM or ITM contracts. Remaining data caveats are the coarse 1-minute stop/target path, derived rather than official SPX/VIX context, and unusable normalized definition fields for multiplier/min tick.
