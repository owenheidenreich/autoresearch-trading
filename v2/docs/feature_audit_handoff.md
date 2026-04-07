# Feature Audit Handoff

**Date**: 2026-04-06
**Dataset**: v2/data.pt (55 features, 387,990 bars, 999 days)
**Fingerprint**: 52adee35373466f9
**Task**: Deep audit of every feature. Determine if each is trustworthy, useful, and correctly computed. Recommend removals/fixes before the next training run.

## How to Use This Document

Read the full feature inventory below. For each feature flagged as INVESTIGATE or SUSPECT:
1. Trace how it was computed (source file + line numbers provided)
2. Verify the computation is correct (no lookahead, no bugs)
3. Check if the feature has real signal (direction edge, correlation with P&L)
4. Recommend: KEEP, FIX, or REMOVE

## Context

The model is a P&L regression transformer that predicts call_pnl and put_pnl for SPX 0DTE options. Trading decisions derive from predictions: gate = max(pred) > threshold, direction = argmax(pred_call, pred_put). Score = min(sortino, 6.0) * positive_day_rate * dd_mult. Current best: 5.786.

Features 0-38 are the "original 39" pre-computed in v1/training/prepare.py (1,222 lines). They were z-score normalized during v1 computation with per-day walk-forward expanding std, then clipped to [-5.0, 5.0].

Features 39-54 are "enriched" features computed in v2/pipeline/build_v2_dataset.py from wide-grid option data. They are NOT normalized -- raw values passed to the model. The model's Linear(55, 64) -> LayerNorm handles scale differences.

## Key Metrics Explained

- **DirEdge**: Split feature at median. Measure % of bars where calls beat puts in HIGH vs LOW group. Higher = more directional signal.
- **Corr(39)**: Correlation with current_moneyness_pct (feature 39). High correlation = potentially redundant.
- **ZeroPct**: % of all 387,990 bars that are exactly 0. High = possibly dead or sparse.

---

## Feature Inventory

### ORIGINAL 39 FEATURES (indices 0-38)

**Source**: archive/v1/training/prepare.py (pre-computed, z-scored, clipped [-5, 5])
**Normalization**: Per-day z-score with walk-forward expanding std. 11 features excluded from normalization (already bounded).

#### Price Returns (0-1)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 0 | ret_6 | 30-bar (30min) log return of SPX | OK | std=0.09, 5.4% edge. No edge in QC signal scan (PF=0.81). |
| 1 | ret_12 | 60-bar (1hr) log return of SPX | OK | std=0.09, 7.0% edge. Slightly better than ret_6. |

#### Volume (2)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 2 | volume_ratio | SPY volume / 20-bar SMA (proxy for SPX which has no volume) | OK | std=0.96, 2.7% edge. |

#### Gamma (3)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 3 | gamma_pressure | GEX proxy: sum(gamma * volume * sign) across OTM chain | INVESTIGATE | Gamma is Black-Scholes estimated, not real. Was computed from narrow 14-strike grid in v1. v2 wide grid (82 strikes) may invalidate this feature's calibration. std=1.94, 3.6% edge. |

#### Volatility (4-6)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 4 | bar_range | (high - low) / close for current bar, z-scored | OK | std=0.11, 9.3% edge. Good volatility signal. |
| 5 | realized_vol | 20-bar rolling stdev of returns | OK | std=0.15, 9.9% edge. QC confirmed PF=1.279. |
| 6 | range_ratio | current bar range / 20-bar avg range | OK | std=0.67, 3.9% edge. |

#### Market Structure (7-9)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 7 | vwap_dist | distance from VWAP, z-scored | OK | std=0.11, **22.7% edge** -- strong signal. |
| 8 | session_range_pct | current price position in session high-low range | OK | std=0.14, 2.2% edge. QC confirmed PF=1.435. |
| 9 | prev_high_dist | distance from previous day's high | OK | std=0.11, **27.5% edge** -- strongest original feature. |

#### Trend (10-12)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 10 | ema_cross | EMA(5) - EMA(20) crossing signal | OK | std=0.12, 14.6% edge. |
| 11 | consec_direction | count of consecutive same-direction bars | OK | std=0.49, 0.1% edge -- very weak. |
| 12 | speed_estimate | 5-bar return / realized_vol (momentum relative to volatility) | OK | std=1.17, 3.6% edge. |

#### VIX (13)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 13 | vix_roc | 10-bar rate of change of VIX | OK | std=0.12, 1.1% edge -- weak but informational. |

#### Time (14)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 14 | minutes_to_close | log(minutes remaining + 1), NOT z-scored | OK | std=0.16, 6.0% edge. Excluded from normalization (already bounded). |

#### IV/Greeks (15-22)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 15 | iv_percentile | ATM IV percentile vs historical | INVESTIGATE | 51.7% zeros. Was computed from v1 option data. May be stale/incomplete. |
| 16 | atm_iv | ATM implied volatility, z-scored | OK | 23.4% zeros (bars with no IV data). std=0.31. |
| 17 | iv_skew | Call IV - Put IV at ATM | **DEAD** | **100% zeros.** Never computed or always zero. Zero direction edge. Wasting a channel. |
| 18 | vix_regime | 4-level VIX bucket (-1, -0.33, +0.33, +1) | OK | std=0.56, 6.7% edge. Not z-scored (categorical). |
| 19 | vrp | Variance risk premium: atm_iv^2 - realized_vol^2 | OK | 24.7% zeros, std=0.27. Weak edge 0.9%. |
| 20 | atm_gamma | Black-Scholes estimated gamma at ATM | SUSPECT | 28.5% zeros. **BS estimate, never validated.** std=0.10, 9.3% edge -- but is the edge real or an artifact? |
| 21 | atm_theta_per_bar | Black-Scholes estimated theta per bar | SUSPECT | 28.5% zeros. Same concerns as atm_gamma. std=0.25, 4.2% edge. |
| 22 | charm_estimate | Estimated dDelta/dTime | **DEAD** | **99.9% zeros.** Has overflow bounds check `if abs(charm_val) < 100`. Effectively never computed. Zero edge. |

#### Technical Indicators (23-28)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 23 | bollinger_position | Price position within Bollinger Bands (0-1) | OK | std=0.61, 2.6% edge. -0.44 correlated with moneyness. |
| 24 | rsi_7 | 7-bar RSI (0-1) | OK | std=0.22, 2.3% edge. Not z-scored (already bounded). |
| 25 | session_range_position | Price position in session range (0-1) | OK | std=0.32, 11.4% edge. -0.73 correlated with moneyness. |
| 26 | poc_dist | Distance from Point of Control (volume profile) | OK | std=0.10, **22.9% edge** -- strong. |
| 27 | va_position | Position within Value Area (volume profile) | OK | std=0.70, 11.0% edge. -0.65 correlated with moneyness. |
| 28 | ib_break | Initial Balance breakout indicator (-1, 0, +1) | OK | std=0.78, 14.5% edge. -0.74 correlated with moneyness. |

#### v10 Extended (29-37)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 29 | atr_14 | 14-bar Average True Range | OK | std=0.16, 10.6% edge. |
| 30 | bar_delta | Up volume - Down volume normalized (-1 to +1) | OK | std=0.64, 0.1% edge -- very weak signal. |
| 31 | session_cum_delta | Cumulative delta for the session | OK | std=2.51, **24.3% edge** -- very strong. |
| 32 | option_spread_width | ATM option (high-low)/close as bid-ask proxy | OK | std=0.36, **18.7% edge**. QC confirmed PF=1.855 (best signal). |
| 33 | macdh_slope | 5-min MACD histogram slope | OK | std=0.99, 7.5% edge. |
| 34 | force_index_2 | 2-bar Force Index | OK | std=1.26, 3.4% edge. |
| 35 | prev_close_dist | Distance from previous close | OK | std=0.11, **27.5% edge** -- identical to prev_high_dist? |
| 36 | effort_vs_result | Volume effort vs price result ratio (0-3) | OK | std=0.77, 1.4% edge -- weak. Not z-scored. |
| 37 | trend_5min | 5-min EMA(13) slope | OK | std=0.09, 3.8% edge. |

#### v17 (38)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 38 | overnight_gap | (day open - prev close) / prev close | **DEAD** | **100% zeros, std=0.** Never populated. Completely dead channel. |

### ENRICHED FEATURES (indices 39-54)

**Source**: v2/pipeline/build_v2_dataset.py:compute_bar_features() (lines 87-170)
**Normalization**: NONE -- raw values. Model's Linear projection + LayerNorm handles scaling.

#### Moneyness (39-41)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 39 | current_moneyness_pct | (ATM_strike - SPX_current) / SPX * 100 | OK | std=0.64, 15.8% edge. Relative (scale-invariant across SPX levels). |
| 40 | intraday_drift_pct | (SPX_current - ATM_strike) / ATM * 100 | **REDUNDANT** | **-0.9999 correlation with feature 39.** It's the negative of moneyness with slightly different denominator (ATM vs SPX). Same 15.8% edge. One should be removed. |
| 41 | near_atm_moneyness_pct | (nearest_strike - SPX) / SPX * 100 | OK | std=0.14, 2.1% edge. Captures how far the nearest tradeable strike is from spot. |

#### Volume (42-48)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 42 | near_atm_call_volume | Call contracts traded at nearest ATM strike | INVESTIGATE | **Raw count, std=267.** Huge scale mismatch with z-scored features. 1.3% edge -- very weak given its scale dominance. |
| 43 | near_atm_put_volume | Put contracts traded at nearest ATM strike | INVESTIGATE | **Raw count, std=242.** Same scale concern. 3.8% edge. |
| 44 | near_atm_total_volume | call + put volume at nearest ATM | INVESTIGATE | **Raw count, std=432.** Largest-scale feature by far. 2.8% edge. Redundant with sum of 42+43. |
| 45 | call_put_flow_ratio | call_vol / total_vol at nearest ATM (0 to 1) | OK | std=0.21, 2.6% edge. Properly bounded ratio. |
| 46 | log_total_volume | log(1 + total_vol) at nearest ATM | OK | std=1.08, 2.8% edge. Log-transformed, reasonable scale. |
| 47 | chain_call_put_ratio | call_vol / total_vol across entire chain | OK | std=0.12, 1.9% edge. |
| 48 | log_chain_volume | log(1 + total vol) across entire chain | OK | std=1.06, 3.9% edge. |

#### Spread/Price (49-51)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 49 | call_hl_range_pct | (high - low) / mid for nearest ATM call | OK | std=0.14, 0.5% edge -- very weak. Spread proxy. |
| 50 | near_atm_call_price_norm | call_close / SPX * 100 (normalized option price) | OK | std=0.18, 1.1% edge. Scale-invariant. |
| 51 | near_atm_put_price_norm | put_close / SPX * 100 | OK | std=0.13, 1.4% edge. Scale-invariant. |

#### Time Decay (52)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 52 | theta_acceleration | 1 / sqrt(minutes_to_close) | OK | std=0.08, 5.4% edge. 0DTE-specific: theta accelerates near close. |

#### Transaction Activity (53-54)
| Idx | Name | What It Is | Status | Notes |
|-----|------|-----------|--------|-------|
| 53 | near_atm_transactions | call_txn + put_txn at nearest ATM | INVESTIGATE | **Raw count, std=156.** Same scale concern as volume features. 3.6% edge. |
| 54 | put_call_txn_ratio | put_txn / (call_txn + put_txn) at nearest ATM | OK | std=0.20, 2.8% edge. New feature added in this rebuild. Properly bounded 0-1. |

---

## Issues to Investigate

### CRITICAL: Dead Features (wasting model capacity)
1. **[17] iv_skew**: 100% zeros, 0% edge. Never computed in v1. Remove.
2. **[22] charm_estimate**: 99.9% zeros, 0% edge. Overflow guard killed it. Remove.
3. **[38] overnight_gap**: 100% zeros, std=0. Never populated. Remove.

These 3 features are in the ORIGINAL 39 (pre-computed in v1/prepare.py). They cannot be removed by editing build_v2_dataset.py -- they're baked into the base X tensor loaded from the v1 data.pt. To remove them, either:
- Strip columns 17, 22, 38 from X after loading in build_v2_dataset.py
- Or accept 3 dead channels (model weights them at ~0, cost is small)

### HIGH: Near-Perfect Redundancy
4. **[40] intraday_drift_pct**: -0.9999 correlation with [39] current_moneyness_pct. Same information. Remove one.
5. **[9] prev_high_dist** vs **[35] prev_close_dist**: Both have 27.5% edge and -0.09 correlation with moneyness. Are these actually the same feature? (prev day high == prev day close sometimes). Check v1/prepare.py.

### HIGH: Scale Mismatch (raw counts in otherwise z-scored dataset)
6. **[42] near_atm_call_volume** (std=267), **[43] near_atm_put_volume** (std=242), **[44] near_atm_total_volume** (std=432), **[53] near_atm_transactions** (std=156): Raw counts 100-1000x larger than z-scored features. The model's input Linear(55, 64) projection weights will be dominated by these 4 features during early training.

Possible fix: log-transform volumes like [46] log_total_volume. Or normalize enriched features to similar scale as original 39.

Note: The previous model (exp_024, score 5.786) trained on 71 features WITH these same raw volumes and WITH 16 duplicate channels. So the model CAN handle it. But it's suboptimal.

### MEDIUM: Unvalidated Greek Estimates
7. **[20] atm_gamma** and **[21] atm_theta_per_bar**: Black-Scholes estimates from Polygon price data, never validated against real broker Greeks. 28.5% zeros. They have 9.3% and 4.2% direction edge respectively -- which suggests real signal, but the estimates could be systematically biased.

### LOW: Weak Features (present but minimal signal)
8. **[11] consec_direction**: 0.1% edge. Near-zero directional signal.
9. **[30] bar_delta**: 0.1% edge. Near-zero directional signal.
10. **[36] effort_vs_result**: 1.4% edge. Weak.
11. **[49] call_hl_range_pct**: 0.5% edge. Spread proxy with almost no signal.

These aren't harmful (the model learns to ignore them), but they dilute the feature space.

### INFO: High-Correlation Clusters
Several features are highly correlated with moneyness (feature 39):
- [25] session_range_position: -0.73
- [28] ib_break: -0.74
- [27] va_position: -0.65
- [24] rsi_7: -0.46
- [23] bollinger_position: -0.44

This is expected (price moving away from open correlates with moneyness drift). Not a bug, but means the model has ~5 features encoding similar "how far has price moved today" information.

---

## Verification Commands

```bash
# Load and inspect any feature
python3 -c "
import torch; d = torch.load('v2/data.pt', map_location='cpu', weights_only=False)
X = d['X'].numpy(); names = d['feature_names']
print(names[IDX], X[:, IDX].mean(), X[:, IDX].std())
"

# Check correlation between two features
python3 -c "
import torch, numpy as np
d = torch.load('v2/data.pt', map_location='cpu', weights_only=False)
X = d['X'].numpy()
print(np.corrcoef(X[:, A], X[:, B])[0,1])
"

# Check direction edge for a feature
python3 -c "
import torch, numpy as np
d = torch.load('v2/data.pt', map_location='cpu', weights_only=False)
X = d['X'].numpy(); bod = d['bar_of_day'].numpy()
call_pnl = d['label_call_pnl'].numpy(); put_pnl = d['label_put_pnl'].numpy()
t = (bod >= 30) & (bod < 270); col = X[t, IDX]
valid = (col != 0) & ~np.isnan(col); med = np.median(col[valid])
high = valid & (col > med); low = valid & (col <= med)
print('HIGH call_better:', (call_pnl[t][high] > put_pnl[t][high]).mean())
print('LOW call_better:', (call_pnl[t][low] > put_pnl[t][low]).mean())
"

# Trace feature computation in v1
grep -n 'FEATURE_NAME' archive/v1/training/prepare.py
```

## Source Files

| File | What | Lines |
|------|------|-------|
| archive/v1/training/prepare.py | Original 39 feature computation | ~1572-2200 |
| v2/pipeline/build_v2_dataset.py | Enriched feature computation | 87-170 |
| v2/core/features.py | Feature names, constants, normalization config | 1-170 |
| v2/train.py | Model architecture (how features are consumed) | 96-210 |
| v2/docs/data_audit_findings.md | Previous data audit (2026-04-03) | Full file |
