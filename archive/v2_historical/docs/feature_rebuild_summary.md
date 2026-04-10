# Feature Pipeline Rebuild Summary

**Date**: 2026-04-07
**Dataset**: v2/data.pt (47 features, ~386K bars, 994 days)
**Previous**: 55 features with frankenstein normalization
**Trigger**: Deep feature audit found regime-destroying per-day z-score + mixed normalization

## What Changed

### Architecture
- **New file**: `v2/pipeline/compute_features.py` -- all features from raw data
- **Rewrote**: `v2/pipeline/build_v2_dataset.py` -- no dependency on v1 data.pt features
- **Updated**: `v2/core/features.py` -- rolling z-score normalization
- Features computed from raw SPX/SPY/VIX caches + spxw_wide/ option data

### Normalization Fix (the main reason for the rebuild)
- **Before**: Features 0-38 were per-day z-scored in v1's prepare.py and baked into data.pt. This destroyed regime info -- realized_vol had mean=0.0000 on every day regardless of VIX level. Features 39+ were raw with values up to 10,422. `normalize_features()` was dead code never called in training or replay.
- **After**: Rolling z-score with 60-day expanding window applied uniformly to all continuous features. High-vol days look different from low-vol days. 16 bounded/categorical features excluded from normalization.

### Features Removed (55 -> 47)
| Feature | Reason |
|---------|--------|
| iv_skew (old) | 100% zeros -- required both call+put IV simultaneously, never fired |
| charm_estimate | 99.9% zeros -- overflow guard killed 0DTE values |
| prev_close_dist | 0.9999 correlation with prev_high_dist |
| overnight_gap | 100% zeros -- cross-day dict never populated |
| intraday_drift_pct | -0.9999 correlation with current_moneyness_pct |
| near_atm_total_volume | Redundant with log-transformed components |
| near_atm_call_price_norm | Violates BS theory (options scale with sqrt(T)*vol, not SPX) |
| near_atm_put_price_norm | Same issue |

### Features Rewritten
| Feature | Before | After |
|---------|--------|-------|
| gamma_pressure | 6 fixed OTM strikes (narrow grid) | Full 82-strike chain with dynamic ATM |
| atm_iv | Fixed-at-open ATM from narrow grid | Dynamic nearest-ATM from wide grid |
| atm_gamma | Fixed ATM, stale Greeks | Dynamic ATM, fresh BS computation |
| atm_theta_per_bar | Same | Same |
| option_spread_pct | Merged from two features | Single Corwin-Schultz from wide grid |

### Features Added
| Feature | What |
|---------|------|
| iv_skew_pct | (put_iv - call_iv) / atm_iv at dynamic nearest-ATM. Properly computed. |

## 47 Feature List (Canonical Order)

### Group 1: Price/Market Structure (28)
ret_6, ret_12, volume_ratio, bar_range, realized_vol, range_ratio, vwap_dist, session_range_pct, prev_high_dist, ema_cross, consec_direction, speed_estimate, vix_roc, minutes_to_close, vix_regime, bollinger_position, rsi_7, session_range_position, poc_dist, va_position, ib_break, atr_14, bar_delta, session_cum_delta, macdh_slope, force_index_2, effort_vs_result, trend_5min

### Group 2: Option/Greeks (11)
atm_iv, vrp, iv_percentile, atm_gamma, atm_theta_per_bar, gamma_pressure, option_spread_pct, iv_skew_pct, current_moneyness_pct, near_atm_moneyness_pct, theta_acceleration

### Group 3: Volume/Flow (8)
log_near_call_volume, log_near_put_volume, call_put_flow_ratio, log_total_volume, chain_call_put_ratio, log_chain_volume, log_near_transactions, put_call_txn_ratio

## Normalization Categories

| Category | Features | Treatment |
|----------|----------|-----------|
| Continuous | 31 features | Rolling z-score, 60-day expanding window, clip [-5, 5] |
| Bounded/categorical | 16 features | No normalization (natural ranges preserved) |

Bounded features: minutes_to_close, theta_acceleration, iv_percentile, vix_regime, ib_break, macdh_slope, bollinger_position, rsi_7, session_range_position, va_position, bar_delta, consec_direction, effort_vs_result, call_put_flow_ratio, chain_call_put_ratio, put_call_txn_ratio

## Impact
- Previous model (score 5.786) is invalid -- trained on broken features
- Must train from scratch with the autoresearch loop
- First experiment should be a clean baseline to establish new score floor
