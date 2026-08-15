# Data Sufficiency Audit

This audit separates broad historical context from executable quote truth.

## Verdict

Enough data for a serious prototype and data-quality falsification; not enough Databento-quality months for a robust broad-market neural edge claim.

- v4 Databento CBBO is the source of truth for executable labels.
- v4 SPX/VIX context is currently derived/proxy context, not official Cboe index data.
- v3/v2 is broad and valuable, but its bid/ask is proxy-derived from Polygon minute OHLC.
- Use v3/v2 to learn durable market structure; use v4 to validate whether that structure survives real bid/ask execution.

## v4 Raw Inventory

| Dataset | Files | Sessions | Rows | First | Last |
|---|---:|---:|---:|---|---|
| definition | 61 | 61 | 1088932 | 2026-01-02 | 2026-03-31 |
| cbbo_1m | 61 | 61 | 11140845 | 2026-01-02 | 2026-03-31 |
| ohlcv_1m | 61 | 61 | 2026868 | 2026-01-02 | 2026-03-31 |
| statistics | 61 | 61 | 181526 | 2026-01-02 | 2026-03-31 |
| cbbo_1s_audit | 10 | 10 | 16635213 | 2026-01-02 | 2026-03-31 |
| spx_1m | 62 | 61 | 25094 | 2026-01-02 | 2026-03-31 |
| vix_1m | 62 | 61 | 25013 | 2026-01-02 | 2026-03-31 |

## v4 Context Source Caveat

| Context | Files | Rows | Sources | Official Index Data |
|---|---:|---:|---|---|
| spx_1m | 62 | 25094 | {'spxw_put_call_parity': 23774, 'GLBX.MDP3:ES.FUT': 1320} | False |
| vix_1m | 62 | 25013 | {'spxw_0dte_atm_iv': 23774, 'XCBF.PITCH:VX.FUT': 1239} | False |

The current neural market window is usable for prototype wiring, but it is not a substitute for official SPX and VIX bars when we make promotion-grade claims.

## v4 Normalized Quality

- Sessions: 61 (2026-01-02 to 2026-03-31)
- Rows: 11,140,845; unique contracts: 31,350
- Bad roots: 0; bad settlements: 0; bad strike alignment: 0
- Bid/ask coverage: 70.0%; invalid bid/ask rows: 0
- OHLCV volume coverage: 18.2%; OI coverage: 100.0%
- Underlying coverage: 99.7%; IV/delta/gamma coverage: 0.0% / 0.0% / 0.0%
- Quote gap populated: 0.0%; quote gap <= 90s: n/a; spread p50/p95: 0.800 / 15.800

## v4 Neural Dataset

- Decision rows: 21,959 across 61 sessions.
- Candidate slots: 922,278; valid candidate fraction: 69.2%.
- Candidate count per row p10/p50/p90: 22 / 30 / 33.
- Latest market complete: 100.0%; full 30m market window complete: 91.9%.
- Decision rows by split: {'train': 7200, 'validation': 6840, 'test': 7919}.
- Decision rows by time bucket: {'first_30': 1769, 'post_open_morning': 5490, 'midday': 7319, 'late_afternoon': 7381}.
- Neural IV/delta/gamma/theta coverage on valid candidates: 100.0% / 100.0% / 100.0% / 100.0%.

| Label Policy | Finite Net Labels | Decision Rows With Label |
|---|---:|---:|
| ask_to_bid_stop35_target60_hold10m | 638,091 | 21,959 |
| ask_to_bid_stop50_target100_hold25m | 638,087 | 21,959 |
| ask_to_bid_stop65_target150_hold45m | 638,078 | 21,959 |

## v3/v2 Historical Coverage

- v2 bars: 389,160; days: 1002 (2022-04-11 to 2026-04-24).
- v2 days by year: {'2022': 172, '2023': 250, '2024': 252, '2025': 250, '2026': 78}.
- v2 label scheme: `exact_contract_fixed_risk_dynamic_slice`; trade window: `bar 30-270`.
- Sidecars loaded: 1019; canonical dates: 1002; duplicates: 17.
- Sidecar contracts/day median: 237; mid coverage: 23.5%; bid/ask coverage: 23.5%.
- Sidecar quality valid/partial/corrupt: 23.5% / 0.0% / 76.5%.
- Bid/ask provenance: proxy: v2/pipeline/build_v2_dataset.py computes bid/ask from Polygon minute close +/- spread_fraction_proxy(close, high, low) / 2.

## v3 Action Surface

- Rows: 89,692; days: 986 (2022-04-11 to 2026-04-01).
- Days by year: {'2022': 172, '2023': 250, '2024': 252, '2025': 250, '2026': 62}.
- Execution window: {'start_bar': 15, 'end_bar': 120, 'label': '09:45-11:30 ET'}; tokens: 24.
- Tradeable actions per row mean: 13.44; rows with any tradeable action: 99.3%.
- Entry mid finite: 93.1%; entry spread finite: 93.1%.

## Practical Use

Use v3/v2 now for:
- regime priors
- time-of-day priors
- architecture pretraining or ablations
- feature importance triage

Do not use v3/v2 directly for:
- final executable PnL
- bid/ask microstructure edge claims
- broad purchase gate for Databento-quality data

Use v4 now for:
- executable ask-entry / bid-exit labels
- SPXW PM-settled 0DTE neural prototype
- small-sample holdout gates
- calibration experiments before spending more

## Next Modeling Direction

1. Mine v3/v2 for stable environment priors across 2022-2026.
2. Re-express those priors as causal v4 features, not as copied v3 labels.
3. Train/calibrate on v4 ask-entry/bid-exit labels only.
4. Require any v3-discovered prior to improve the v4 multi-seed March holdout before buying more data.
