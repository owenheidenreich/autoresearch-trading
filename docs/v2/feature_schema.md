# v2 Feature Schema

## Overview

39 features computed from 1-minute OHLCV bars (SPX, SPY, VIX) plus
real-time option chain data. Preserved as-is from v1. Feature changes
are a future research question, not a Phase 0 decision.

---

## Complete Feature Table

| Idx | Name | Group | Source | Description | Normalization |
|-----|------|-------|--------|-------------|---------------|
| 0 | ret_6 | Price returns | SPX | 30-bar (30min) log return | z-score |
| 1 | ret_12 | Price returns | SPX | 60-bar (1hr) log return | z-score |
| 2 | volume_ratio | Volume | SPY | bar volume / 20-bar SMA volume | z-score |
| 3 | gamma_pressure | Gamma | Options | sum(gamma * volume * sign) across chain (GEX proxy) | z-score |
| 4 | bar_range | Volatility | SPX | (high - low) / close | z-score |
| 5 | realized_vol | Volatility | SPX | 20-bar rolling stdev of returns | z-score |
| 6 | range_ratio | Volatility | SPX | current bar range / 20-bar avg range | z-score |
| 7 | vwap_dist | VWAP | SPX | (close - session VWAP) / close | z-score |
| 8 | session_range_pct | Session | SPX | session range so far / close | z-score |
| 9 | prev_high_dist | Key levels | SPX | distance to previous day high / close | z-score |
| 10 | ema_cross | Trend | SPX | (EMA8 - EMA21) / close | z-score |
| 11 | consec_direction | Trend | SPX | consecutive same-direction bars (signed) | z-score |
| 12 | speed_estimate | Trend | SPX | |5-bar return| / realized_vol | z-score |
| 13 | vix_roc | VIX | VIX | VIX 10-bar rate of change | z-score |
| 14 | minutes_to_close | Time | Clock | log(minutes remaining + 1), normalized | None (already normalized) |
| 15 | iv_percentile | IV | Options | current IV rank vs 60-day history [0,1] | None (already 0-1) |
| 16 | atm_iv | Options | Options | ATM implied vol (avg call + put IV) | z-score |
| 17 | iv_skew | Options | Options | put IV - call IV (fear premium) | z-score |
| 18 | vix_regime | VIX/Regime | VIX | regime bucket: -1 low, -0.33 normal, 0.33 elevated, 1 crisis | None (categorical) |
| 19 | vrp | VIX/Regime | Derived | variance risk premium: atm_iv^2 - realized_vol^2 | z-score |
| 20 | atm_gamma | Greeks | Options | ATM call gamma | z-score |
| 21 | atm_theta_per_bar | Greeks | Options | ATM theta per 1-min bar | z-score |
| 22 | charm_estimate | Greeks | Options | estimated dDelta/dT | z-score |
| 23 | bollinger_position | Bollinger | SPX | 5-min Bollinger band position | None (already bounded) |
| 24 | rsi_7 | Range | SPX | 5-min RSI(7) | None (already 0-1) |
| 25 | session_range_position | Range | SPX | (close - session_low) / (session_high - session_low) | None (already 0-1) |
| 26 | poc_dist | Market structure | SPX | (close - session POC) / close | z-score |
| 27 | va_position | Market structure | SPX | position within Value Area | None (already ~0-1) |
| 28 | ib_break | Market structure | SPX | Initial Balance break state: -1/0/+1 | None (categorical) |
| 29 | atr_14 | v9 | SPX | 14-bar ATR / close | z-score |
| 30 | bar_delta | v9 | SPX | (close - open) / (high - low), intrabar pressure | None (already -1 to +1) |
| 31 | session_cum_delta | v9 | SPX | cumulative bar deltas within session | z-score |
| 32 | option_spread_width | Spread | Options | ATM option (high-low)/close (bid-ask proxy) | z-score |
| 33 | macdh_slope | v10 | SPX | 5-min MACD-H direction (Elder signal) | None (already -1/0/+1) |
| 34 | force_index_2 | v10 | SPX | 5-min Force Index (Elder) | z-score |
| 35 | prev_close_dist | v10 | SPX | (close - prev_day_close) / close | z-score |
| 36 | effort_vs_result | v10 | SPX | 5-min effort vs result (Coulling) | None (already bounded [-3,3]) |
| 37 | trend_5min | v10 | SPX | 5-min EMA(13) slope (Elder Triple Screen) | z-score |
| 38 | overnight_gap | v17 | SPX | (day open - prev close) / prev close | z-score |

---

## Feature Groups

| Group | Indices | Count |
|-------|---------|-------|
| Price returns | 0-1 | 2 |
| Volume | 2 | 1 |
| Gamma | 3 | 1 |
| Volatility | 4-6 | 3 |
| VWAP | 7 | 1 |
| Session structure | 8 | 1 |
| Key levels | 9 | 1 |
| Trend | 10-12 | 3 |
| VIX | 13 | 1 |
| Time | 14 | 1 |
| IV Percentile | 15 | 1 |
| Options | 16-17 | 2 |
| VIX/Regime | 18-19 | 2 |
| Greeks | 20-22 | 3 |
| Bollinger | 23 | 1 |
| Range | 24-25 | 2 |
| Market structure | 26-28 | 3 |
| v9 features | 29-31 | 3 |
| Spread width | 32 | 1 |
| v10 features | 33-37 | 5 |
| v17 promoted | 38 | 1 |
| **Total** | **0-38** | **39** |

---

## Normalization Contract

### Method

**Default (training):** Per-day z-score with global standard deviation.
- For each day: subtract day mean, divide by global std
- Prevents cross-day fingerprinting while preserving regime-level info

**Fallback (replay/live):** Rolling z-score.
- Window = min(len / 4, 500), clamped to min 50 bars
- Applied to non-excluded features only

### Excluded from Normalization

11 features are excluded because they are already bounded or categorical:

```
minutes_to_close     (log-normalized)
iv_percentile        (already 0-1)
vix_regime           (categorical buckets)
bollinger_position   (already bounded)
session_range_position (already 0-1)
rsi_7                (already 0-1)
va_position          (already ~0-1)
ib_break             (categorical: -1, 0, +1)
bar_delta            (already -1 to +1)
macdh_slope          (already -1/0/+1)
effort_vs_result     (already bounded [-3, 3])
```

### Post-Normalization

All features clipped to [-5.0, 5.0] after normalization.

---

## Version Tracking

Feature contract version: string identifier (v1 used "ibkr_live_v1").

Validation checks:
- Array shape: (N, 39)
- Feature names match FEATURE_NAMES list exactly
- No unexpected NaN patterns

The feature contract is enforced at:
- Dataset build time (pipeline/build_dataset.py)
- Live context bootstrap (live/market.py)
- Model load time (train.py)

Any feature schema change requires a new version string and a data rebuild.
