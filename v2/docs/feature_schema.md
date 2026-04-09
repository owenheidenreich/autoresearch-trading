# v2 Feature Schema

This is the active 47-feature contract used by `v2/data.pt`.

## Overview

Feature groups:

- 28 price / market-structure features
- 11 option / Greeks features
- 8 flow features

Canonical source files:

- `v2/pipeline/compute_features.py`
- `v2/core/features.py`

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
| 28 | atm_iv | option |
| 29 | vrp | option |
| 30 | iv_percentile | option |
| 31 | atm_gamma | option |
| 32 | atm_theta_per_bar | option |
| 33 | gamma_pressure | option |
| 34 | option_spread_pct | option |
| 35 | iv_skew_pct | option |
| 36 | current_moneyness_pct | option |
| 37 | near_atm_moneyness_pct | option |
| 38 | theta_acceleration | option |
| 39 | log_near_call_volume | flow |
| 40 | log_near_put_volume | flow |
| 41 | call_put_flow_ratio | flow |
| 42 | log_total_volume | flow |
| 43 | chain_call_put_ratio | flow |
| 44 | log_chain_volume | flow |
| 45 | log_near_transactions | flow |
| 46 | put_call_txn_ratio | flow |

## Current Normalization

The active normalization regime is:

- 60-day rolling z-score
- expanding warm-up window at the start
- clipping to `[-5, 5]`
- NaNs converted to zero after normalization

This is implemented in `v2/core/features.py`.

## Features Excluded From Z-Score Normalization

These are left in their bounded or categorical form:

- `minutes_to_close`
- `theta_acceleration`
- `iv_percentile`
- `vix_regime`
- `ib_break`
- `macdh_slope`
- `bollinger_position`
- `rsi_7`
- `session_range_position`
- `va_position`
- `bar_delta`
- `consec_direction`
- `effort_vs_result`
- `call_put_flow_ratio`
- `chain_call_put_ratio`
- `put_call_txn_ratio`

## Important Semantics

- `poc_dist` and `va_position` are incremental intraday features, not full-day look-ahead values.
- `option_spread_pct`, moneyness, IV, gamma, theta, and flow features are computed from the wide-grid option data aligned to the current nearest ATM.
- The feature contract is 47 columns. Any doc that still says 39 is stale.
