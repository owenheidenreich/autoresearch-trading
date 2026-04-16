# v2 Feature Schema

This is the active 52-feature contract used by `v2/data.pt`.

**Authority**: `v2/core/features.py:FEATURE_NAMES` is the single source of truth.
Any discrepancy between this doc and the code means this doc is wrong.

## Overview

Feature groups:

- 29 price / market-structure features (indices 0-28)
- 3 intraday phase features (indices 29-31)
- 12 option / Greeks features (indices 32-43)
- 8 flow features (indices 44-51)

Canonical source files:

- `v2/pipeline/compute_features.py` (computation)
- `v2/core/features.py` (names, normalization, indexing)

## Canonical Feature Order

| Idx | Name | Group |
|-----|------|-------|
| 0 | ret_6 | price |
| 1 | ret_12 | price |
| 2 | volume_ratio | price |
| 3 | bar_range | price |
| 4 | realized_vol | price |
| 5 | range_ratio | price |
| 6 | vwap_dist | price |
| 7 | session_range_pct | price |
| 8 | prev_high_dist | price |
| 9 | ema_cross | price |
| 10 | consec_direction | price |
| 11 | speed_estimate | price |
| 12 | vix_roc | price |
| 13 | minutes_to_close | price |
| 14 | vix_regime | price |
| 15 | bollinger_position | price |
| 16 | rsi_7 | price |
| 17 | session_range_position | price |
| 18 | poc_dist | price |
| 19 | va_position | price |
| 20 | ib_break | price |
| 21 | atr_14 | price |
| 22 | bar_delta | price |
| 23 | session_cum_delta | price |
| 24 | macdh_slope | price |
| 25 | force_index_2 | price |
| 26 | effort_vs_result | price |
| 27 | trend_5min | price |
| 28 | vwap_slope | price |
| 29 | intraday_sin | intraday |
| 30 | intraday_cos | intraday |
| 31 | intraday_phase | intraday |
| 32 | atm_iv | option |
| 33 | vrp | option |
| 34 | iv_percentile | option |
| 35 | atm_gamma | option |
| 36 | atm_theta_per_bar | option |
| 37 | gamma_pressure | option |
| 38 | aggregate_charm | option |
| 39 | option_spread_pct | option |
| 40 | iv_skew_pct | option |
| 41 | current_moneyness_pct | option |
| 42 | near_atm_moneyness_pct | option |
| 43 | theta_acceleration | option |
| 44 | log_near_call_volume | flow |
| 45 | log_near_put_volume | flow |
| 46 | call_put_flow_ratio | flow |
| 47 | log_total_volume | flow |
| 48 | chain_call_put_ratio | flow |
| 49 | log_chain_volume | flow |
| 50 | log_near_transactions | flow |
| 51 | put_call_txn_ratio | flow |

## Current Normalization

The active normalization regime is:

- 60-day rolling z-score (expanding warm-up window at cold start)
- Clipping to `[-5, 5]`
- NaNs converted to zero after normalization

Implemented in `v2/core/features.py`.

## Features Excluded From Z-Score Normalization (19)

These are left in their bounded or categorical form:

- `minutes_to_close` — [0, 1]
- `theta_acceleration` — [0, 1]
- `iv_percentile` — [0, 1]
- `vix_regime` — {-1, -0.33, 0.33, 1}
- `ib_break` — {-1, 0, 1}
- `macdh_slope` — {-1, 0, 1}
- `bollinger_position` — ~[-2, 2]
- `rsi_7` — [0, 1]
- `session_range_position` — [0, 1]
- `va_position` — [0, 1]
- `bar_delta` — [-1, 1]
- `consec_direction` — [-1, 1]
- `effort_vs_result` — [0, 3]
- `call_put_flow_ratio` — [0, 1]
- `chain_call_put_ratio` — [0, 1]
- `put_call_txn_ratio` — [0, 1]
- `intraday_sin` — [-1, 1]
- `intraday_cos` — [-1, 1]
- `intraday_phase` — [0, 1]

## Important Semantics

- `poc_dist` and `va_position` are incremental intraday features, not full-day look-ahead values.
- `option_spread_pct`, moneyness, IV, gamma, theta, and flow features are computed from the wide-grid option data aligned to the current nearest ATM.
- The feature contract is 52 columns. Any doc that still says 47 or 39 is stale.
