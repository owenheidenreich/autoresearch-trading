# Historical Priors From v2/v3

These are prior-discovery diagnostics, not executable edge claims.

## Read This First

- v2/v3 covers many more environments than v4: 2022-04-11 through 2026-04-24.
- v2/v3 bid/ask is proxy-derived from Polygon minute OHLC; v4 Databento CBBO remains the execution-grade validator.
- v3 action-surface coverage is morning-focused, so it cannot by itself validate late-afternoon rules.
- v2 rows are opportunity labels, not a tradable policy; the very high PF values are expected from best-contract label construction and must not be read as strategy performance.

## v2 Bar Opportunity Coverage

- Rows with valid slice labels: 234,806.
- Days: 1002 (2022-04-11 to 2026-04-24).
- Framing: v2 bar labels are broad opportunity labels from the exact-chain sidecars. They are useful for time/regime priors, not final executable PnL.

## v2 Time Buckets

| Group | N | Mean | Median | Win Rate | PF | Years PF>1 / Years |
|---|---:|---:|---:|---:|---:|---:|
| midday | 117854 | 0.293 | 0.329 | 0.920 | 58.323 | 5 / 5 |
| post_open_morning | 87736 | 0.290 | 0.315 | 0.918 | 53.857 | 5 / 5 |
| late_afternoon | 29216 | 0.292 | 0.322 | 0.912 | 44.891 | 5 / 5 |

## v2 Time x VWAP

| Group | N | Mean | Median | Win Rate | PF | Years PF>1 / Years |
|---|---:|---:|---:|---:|---:|---:|
| midday | True | 64503 | 0.291 | 0.332 | 0.918 | 67.182 | 5 / 5 |
| post_open_morning | True | 47145 | 0.294 | 0.328 | 0.919 | 57.906 | 5 / 5 |
| late_afternoon | True | 16129 | 0.288 | 0.321 | 0.914 | 51.146 | 5 / 5 |
| midday | False | 53351 | 0.295 | 0.326 | 0.923 | 50.421 | 5 / 5 |
| post_open_morning | False | 40591 | 0.285 | 0.299 | 0.917 | 49.703 | 5 / 5 |
| late_afternoon | False | 13087 | 0.297 | 0.324 | 0.909 | 39.199 | 5 / 5 |

## v2 Time x OMAR

| Group | N | Mean | Median | Win Rate | PF | Years PF>1 / Years |
|---|---:|---:|---:|---:|---:|---:|
| midday | positive | 62612 | 0.291 | 0.331 | 0.919 | 68.111 | 5 / 5 |
| post_open_morning | positive | 46513 | 0.293 | 0.325 | 0.918 | 56.372 | 5 / 5 |
| late_afternoon | positive | 15666 | 0.288 | 0.319 | 0.915 | 55.373 | 5 / 5 |
| post_open_morning | negative | 41218 | 0.287 | 0.302 | 0.918 | 51.226 | 5 / 5 |
| midday | negative | 55237 | 0.295 | 0.327 | 0.922 | 50.293 | 5 / 5 |
| late_afternoon | negative | 13550 | 0.298 | 0.325 | 0.908 | 37.108 | 5 / 5 |

## v3 Action-Surface Coverage

- Rows after side expansion: 176,975.
- Days: 986 (2022-04-11 to 2026-04-01).
- Framing: v3 action-surface priors use the best tradeable candidate per side inside the morning execution window. Labels use proxy bid/ask and should be validated against v4 CBBO before becoming policy rules.

## v3 Time x Side

| Group | N | Mean | Median | Win Rate | PF | Years PF>1 / Years |
|---|---:|---:|---:|---:|---:|---:|
| post_open_morning | C | 87683 | 260.831 | -180.224 | 0.383 | 2.532 | 5 / 5 |
| midday | C | 981 | 212.861 | -153.428 | 0.402 | 2.378 | 5 / 5 |
| post_open_morning | P | 87343 | 253.560 | -232.355 | 0.319 | 2.185 | 5 / 5 |
| midday | P | 968 | 201.481 | -200.160 | 0.326 | 2.053 | 5 / 5 |

## v3 Time x Side x VWAP

| Group | N | Mean | Median | Win Rate | PF | Years PF>1 / Years |
|---|---:|---:|---:|---:|---:|---:|
| midday | C | True | 530 | 262.637 | -124.303 | 0.440 | 2.901 | 5 / 5 |
| post_open_morning | C | True | 46707 | 278.649 | -154.661 | 0.412 | 2.789 | 5 / 5 |
| post_open_morning | C | False | 40976 | 240.520 | -203.273 | 0.351 | 2.288 | 5 / 5 |
| post_open_morning | P | False | 40599 | 279.626 | -236.717 | 0.334 | 2.273 | 5 / 5 |
| post_open_morning | P | True | 46744 | 230.921 | -229.582 | 0.306 | 2.104 | 5 / 5 |
| midday | P | True | 528 | 171.520 | -202.537 | 0.299 | 1.904 | 5 / 5 |

## v3 Time x Side x OMAR

| Group | N | Mean | Median | Win Rate | PF | Years PF>1 / Years |
|---|---:|---:|---:|---:|---:|---:|
| midday | C | positive | 525 | 272.551 | -120.744 | 0.446 | 2.999 | 5 / 5 |
| post_open_morning | C | positive | 46235 | 300.212 | -150.336 | 0.415 | 2.958 | 5 / 5 |
| post_open_morning | P | negative | 41077 | 267.981 | -234.691 | 0.337 | 2.237 | 5 / 5 |
| post_open_morning | C | negative | 41443 | 216.887 | -207.330 | 0.348 | 2.147 | 5 / 5 |
| post_open_morning | P | positive | 46261 | 240.691 | -230.818 | 0.304 | 2.137 | 5 / 5 |
| midday | P | positive | 519 | 185.735 | -201.349 | 0.310 | 1.979 | 5 / 5 |

## Translation To v4

The recurring priors to carry forward are not hard rules yet. They become candidate causal features and pre-registered evaluation slices in v4:

- time bucket, especially post-open morning; late afternoon must be validated primarily in v4 because v3 action-surface rows are morning-focused.
- above/below VWAP and OMAR sign as state variables, not as hand-forced directions.
- side-aware opportunity: calls and puts should remain separate actions rather than one generic long-option class.
- volatility/range buckets as diagnostics for whether a policy is regime brittle.

Next validation step: train a v4 action model with these priors represented explicitly, then require the March CBBO holdout to improve across seeds before buying more data.
