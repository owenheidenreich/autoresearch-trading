"""Compute all features from raw market data.

Ground-up rebuild: no dependency on v1 prepare.py at runtime.
All features computed from raw SPX/SPY/VIX caches + spxw_wide/ option data.

Three groups:
  1. Price/Volume/Market Structure (28 features) -- SPX/SPY/VIX only
  2. Option features (11 features) -- from wide grid with dynamic ATM + BS
  3. Volume/Flow features (8 features) -- from wide grid transaction data

Normalization is NOT applied here. The caller (build_v2_dataset.py) handles
rolling z-score normalization after all features are computed.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import norm as _norm_dist
from scipy.optimize import brentq as _brentq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BARS_PER_DAY = 390
LOOKBACK = 60            # bars for rolling windows
RISK_FREE_RATE = 0.05    # for BS pricing
IB_BARS = 30             # Initial Balance = first 30 minutes

# ---------------------------------------------------------------------------
# Black-Scholes utilities (extracted from v1 prepare.py:300-353)
# ---------------------------------------------------------------------------

def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_dist.cdf(d1) - K * math.exp(-r * T) * _norm_dist.cdf(d2)


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_dist.cdf(-d2) - S * _norm_dist.cdf(-d1)


def _bs_iv(price: float, S: float, K: float, T: float, r: float,
           is_call: bool = True) -> float:
    """Implied vol via Brent's method. Returns NaN on failure."""
    if T <= 1e-10 or price <= 0 or S <= 0 or K <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if price < intrinsic - 0.01:
        return np.nan
    fn = _bs_call if is_call else _bs_put
    try:
        return _brentq(lambda s: fn(S, K, T, r, s) - price, 0.01, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan


def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float):
    """Returns (delta, gamma, theta_per_bar, vega). NaN on bad inputs."""
    if T <= 1e-10 or sigma <= 0 or S <= 0:
        return np.nan, np.nan, np.nan, np.nan
    sqrt_T = math.sqrt(T)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt_T
    npdf_d1 = _norm_dist.pdf(d1)

    delta = _norm_dist.cdf(d1)
    gamma = npdf_d1 / (S * sigma * sqrt_T)
    theta_annual = (-(S * npdf_d1 * sigma) / (2.0 * sqrt_T)
                    - r * K * math.exp(-r * T) * _norm_dist.cdf(d2))
    theta_per_bar = theta_annual / (252.0 * BARS_PER_DAY)
    vega = S * npdf_d1 * sqrt_T / 100.0

    return delta, gamma, theta_per_bar, vega


# ---------------------------------------------------------------------------
# Helper: find nearest ATM strike in wide grid bar
# ---------------------------------------------------------------------------

def _find_nearest_atm(bar_data: dict, spx: float) -> float | None:
    """Find the strike closest to current SPX."""
    if not bar_data or np.isnan(spx):
        return None
    best_strike = None
    best_dist = float('inf')
    for strike in bar_data:
        d = abs(strike - spx)
        if d < best_dist:
            best_dist = d
            best_strike = strike
    return best_strike


# ---------------------------------------------------------------------------
# EMA helper (per-day, no cross-day bleed)
# ---------------------------------------------------------------------------

def _ema_per_day(values: np.ndarray, day_starts: list[int], span: int) -> np.ndarray:
    """Compute EMA that resets at each day boundary."""
    out = np.full_like(values, np.nan)
    alpha = 2.0 / (span + 1)
    day_ends = day_starts[1:] + [len(values)]
    for ds, de in zip(day_starts, day_ends):
        if de <= ds:
            continue
        out[ds] = values[ds]
        for i in range(ds + 1, de):
            if np.isnan(out[i - 1]):
                out[i] = values[i]
            else:
                out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _expand_5min_to_1min(vals_5m: np.ndarray, ds: int, de: int, n5: int,
                         out: np.ndarray, start_k: int = 0) -> None:
    """Expand 5-min values back to 1-min by repeating each across 5 bars."""
    for k in range(start_k, n5):
        v = vals_5m[k]
        base = ds + k * 5
        end = min(base + 5, de)
        out[base:end] = v


# ---------------------------------------------------------------------------
# Group 1: Price / Volume / Market Structure features (VECTORIZED)
# ---------------------------------------------------------------------------

def compute_price_features(
    spx_close: np.ndarray,
    spx_high: np.ndarray,
    spx_low: np.ndarray,
    spx_open: np.ndarray,
    spy_volume: np.ndarray,
    spy_close: np.ndarray,
    vix_close: np.ndarray,
    day_starts: list[int],
    bar_of_day: np.ndarray,
) -> np.ndarray:
    """Compute 28 price/volume/market structure features for all bars.

    Returns: (n_bars, 28) array of RAW features (not normalized).
    Uses vectorized numpy wherever possible for speed.
    """
    import pandas as pd

    n = len(spx_close)
    N_FEAT = 28
    feat = np.zeros((n, N_FEAT), dtype=np.float64)
    day_ends = day_starts[1:] + [n]
    day_starts_arr = np.array(day_starts)

    c = spx_close.astype(np.float64)
    h = spx_high.astype(np.float64)
    lo = spx_low.astype(np.float64)
    o = spx_open.astype(np.float64)
    vol = spy_volume.astype(np.float64)
    bod = bar_of_day

    # --- Day membership array: day_of[i] = which day index bar i belongs to ---
    day_of = np.zeros(n, dtype=np.int32)
    for di, (ds, de) in enumerate(zip(day_starts, day_ends)):
        day_of[ds:de] = di

    # --- Log returns (vectorized, zeroed at day boundaries) ---
    log_ret = np.zeros(n, dtype=np.float64)
    safe = (c[:-1] > 0) & (c[1:] > 0) & (day_of[1:] == day_of[:-1])
    log_ret[1:] = np.where(safe, np.log(c[1:] / c[:-1]), 0.0)

    # --- EMAs (per-day, inherently sequential) ---
    ema8 = _ema_per_day(c, day_starts, 8)
    ema21 = _ema_per_day(c, day_starts, 21)

    # --- Bar range and bar delta (fully vectorized) ---
    bar_range_raw = np.where(c > 0, (h - lo) / c, 0.0)
    denom = np.where((h - lo) > 0, h - lo, 1.0)
    bar_delta_raw = np.clip((c - o) / denom, -1.0, 1.0)

    # --- Session cumulative delta (per-day cumsum, vectorized per day) ---
    session_cum_delta = np.zeros(n, dtype=np.float64)
    for ds, de in zip(day_starts, day_ends):
        session_cum_delta[ds:de] = np.cumsum(bar_delta_raw[ds:de])

    # --- VWAP per session (vectorized per day) ---
    vwap_arr = np.full(n, np.nan)
    for ds, de in zip(day_starts, day_ends):
        v_day = np.maximum(vol[ds:de], 0.0)
        cum_pv = np.cumsum(c[ds:de] * v_day)
        cum_v = np.cumsum(v_day)
        valid = cum_v > 0
        vwap_arr[ds:de] = np.where(valid, cum_pv / cum_v, np.nan)

    # --- Previous day high (vectorized) ---
    prev_high_arr = np.zeros(n, dtype=np.float64)
    for di, (ds, de) in enumerate(zip(day_starts, day_ends)):
        if di > 0:
            prev_ds = day_starts[di - 1]
            prev_de = ds
            ph = np.max(h[prev_ds:prev_de])
            prev_high_arr[ds:de] = ph

    # --- Initial balance (first 30 bars per day) ---
    ib_high_arr = np.zeros(n, dtype=np.float64)
    ib_low_arr = np.zeros(n, dtype=np.float64)
    for ds, de in zip(day_starts, day_ends):
        ib_end = min(ds + IB_BARS, de)
        ib_h = np.max(h[ds:ib_end])
        ib_l = np.min(lo[ds:ib_end])
        ib_high_arr[ds:de] = ib_h
        ib_low_arr[ds:de] = ib_l

    # --- Session running high/low (for session_range_pct and session_range_position) ---
    sess_high = np.zeros(n, dtype=np.float64)
    sess_low = np.zeros(n, dtype=np.float64)
    for ds, de in zip(day_starts, day_ends):
        sess_high[ds:de] = np.maximum.accumulate(h[ds:de])
        sess_low[ds:de] = np.minimum.accumulate(lo[ds:de])

    # --- Volume profile: POC and Value Area (incremental, no look-ahead) ---
    # Each bar sees only the volume profile from bars [day_start .. current_bar].
    # The tracked range expands with the running session low/high so trend bars
    # stay represented instead of being clipped out of the profile.
    poc_arr = np.zeros(n, dtype=np.float64)
    va_lo_arr = np.zeros(n, dtype=np.float64)
    va_hi_arr = np.zeros(n, dtype=np.float64)
    N_BINS = 50
    for ds, de in zip(day_starts, day_ends):
        day_len = de - ds
        if day_len < 10:
            continue
        day_c = c[ds:de]
        day_v = vol[ds:de]
        lo_px = float(day_c[0])
        hi_px = float(day_c[0])
        profile = np.zeros(N_BINS, dtype=np.float64)
        total_v = 0.0
        poc_idx = 0
        poc_max_v = 0.0

        for i in range(ds, de):
            seen_end = i - ds + 1
            px = float(day_c[seen_end - 1])
            v = float(day_v[seen_end - 1])
            prev_lo = lo_px
            prev_hi = hi_px
            lo_px = min(lo_px, px)
            hi_px = max(hi_px, px)

            if hi_px - lo_px < 0.01:
                poc_arr[i] = px
                va_lo_arr[i] = lo_px
                va_hi_arr[i] = hi_px
                continue

            bin_idx = np.clip(
                ((day_c[:seen_end] - lo_px) / (hi_px - lo_px) * N_BINS).astype(int),
                0,
                N_BINS - 1,
            )
            if lo_px != prev_lo or hi_px != prev_hi:
                profile = np.bincount(
                    bin_idx,
                    weights=day_v[:seen_end],
                    minlength=N_BINS,
                ).astype(np.float64, copy=False)
                total_v = float(profile.sum())
                poc_idx = int(np.argmax(profile))
                poc_max_v = float(profile[poc_idx])
            elif v > 0.0:
                bidx = int(bin_idx[-1])
                profile[bidx] += v
                total_v += v
                if profile[bidx] > poc_max_v:
                    poc_max_v = float(profile[bidx])
                    poc_idx = bidx

            if total_v <= 0.0:
                poc_arr[i] = px
                va_lo_arr[i] = px
                va_hi_arr[i] = px
                continue

            bin_edges = np.linspace(lo_px, hi_px, N_BINS + 1)
            poc_arr[i] = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0

            # Value area (70% of volume) — expand from POC
            va_vol = profile[poc_idx]
            l_idx, h_idx = poc_idx, poc_idx
            target = 0.70 * total_v
            while va_vol < target:
                add_lo = profile[l_idx - 1] if l_idx > 0 else 0.0
                add_hi = profile[h_idx + 1] if h_idx < N_BINS - 1 else 0.0
                if add_lo >= add_hi and l_idx > 0:
                    l_idx -= 1
                    va_vol += add_lo
                elif h_idx < N_BINS - 1:
                    h_idx += 1
                    va_vol += add_hi
                else:
                    break
            va_lo_arr[i] = bin_edges[l_idx]
            va_hi_arr[i] = bin_edges[h_idx + 1]

    # --- 5-min technical indicators (per-day loop, ~78 bars/day, fast) ---
    boll_pos = np.zeros(n, dtype=np.float64)
    rsi_7_arr = np.full(n, 0.5, dtype=np.float64)
    macdh_slope_arr = np.zeros(n, dtype=np.float64)
    force_idx_arr = np.zeros(n, dtype=np.float64)
    effort_arr = np.ones(n, dtype=np.float64)
    trend_5m_arr = np.zeros(n, dtype=np.float64)

    for ds, de in zip(day_starts, day_ends):
        day_len = de - ds
        n5 = day_len // 5
        if n5 < 5:
            continue
        # Reshape to (n5, 5) for vectorized aggregation
        trim = n5 * 5
        c_block = c[ds:ds + trim].reshape(n5, 5)
        h_block = h[ds:ds + trim].reshape(n5, 5)
        l_block = lo[ds:ds + trim].reshape(n5, 5)
        o_block = o[ds:ds + trim].reshape(n5, 5)
        v_block = vol[ds:ds + trim].reshape(n5, 5)
        c5 = c_block[:, -1]        # close of last 1-min bar
        h5 = h_block.max(axis=1)
        l5 = l_block.min(axis=1)
        o5 = o_block[:, 0]
        v5 = v_block.sum(axis=1)

        # Bollinger (20-period SMA +/- 2*std on 5-min close)
        if n5 >= 21:
            c5_s = pd.Series(c5)
            mu20 = c5_s.rolling(20).mean().values
            std20 = c5_s.rolling(20).std().values
            boll_5m = np.where(std20 > 1e-10, (c5 - mu20) / (2 * std20), 0.0)
            boll_5m[:20] = 0.0
            _expand_5min_to_1min(boll_5m, ds, de, n5, boll_pos)

        # RSI(7) on 5-min (Wilder's smoothing, inherently sequential)
        if n5 > 7:
            diffs = np.diff(c5)
            gains = np.maximum(diffs, 0.0)
            losses_arr = np.maximum(-diffs, 0.0)
            avg_gain = gains[:7].mean()
            avg_loss = losses_arr[:7].mean()
            rsi_5m = np.full(n5, 0.5)
            for k in range(7, n5):
                avg_gain = (avg_gain * 6 + gains[k - 1]) / 7
                avg_loss = (avg_loss * 6 + losses_arr[k - 1]) / 7
                if avg_loss > 1e-10:
                    rs = avg_gain / avg_loss
                    rsi_5m[k] = rs / (1.0 + rs)
                else:
                    rsi_5m[k] = 1.0
            _expand_5min_to_1min(rsi_5m, ds, de, n5, rsi_7_arr)

        # MACD-H slope (12/26/9 EMA on 5-min, sequential)
        if n5 > 26:
            ema12 = np.copy(c5)
            ema26 = np.copy(c5)
            a12, a26 = 2.0 / 13, 2.0 / 27
            for k in range(1, n5):
                ema12[k] = a12 * c5[k] + (1 - a12) * ema12[k - 1]
                ema26[k] = a26 * c5[k] + (1 - a26) * ema26[k - 1]
            macd_line = ema12 - ema26
            signal = np.copy(macd_line)
            a9 = 2.0 / 10
            for k in range(1, n5):
                signal[k] = a9 * macd_line[k] + (1 - a9) * signal[k - 1]
            macd_h = macd_line - signal
            slope_5m = np.sign(np.diff(macd_h))
            slope_full = np.zeros(n5)
            slope_full[1:] = slope_5m
            _expand_5min_to_1min(slope_full, ds, de, n5, macdh_slope_arr)

        # Force Index (EMA(2) of volume * price change, normalized)
        if n5 > 20:
            fi_raw = np.zeros(n5)
            fi_raw[1:] = v5[1:] * np.diff(c5)
            fi_ema = np.copy(fi_raw)
            a2 = 2.0 / 3
            for k in range(1, n5):
                fi_ema[k] = a2 * fi_raw[k] + (1 - a2) * fi_ema[k - 1]
            fi_s = pd.Series(fi_ema)
            fi_std = fi_s.rolling(20).std().values
            fi_norm = np.where(fi_std > 1e-10, fi_ema / fi_std, 0.0)
            fi_norm[:20] = 0.0
            _expand_5min_to_1min(fi_norm, ds, de, n5, force_idx_arr)

        # Effort vs Result
        body5 = np.abs(c5 - o5)
        if n5 > 20:
            body_s = pd.Series(body5)
            vol_s = pd.Series(v5)
            avg_body = body_s.rolling(20).mean().values
            avg_vol = vol_s.rolling(20).mean().values
            safe_b = avg_body > 1e-10
            safe_v = avg_vol > 1e-10
            eff_5m = np.ones(n5)
            mask = safe_b & safe_v
            body_ratio = np.where(safe_b, body5 / np.maximum(avg_body, 1e-10), 1.0)
            vol_ratio = np.where(safe_v, v5 / np.maximum(avg_vol, 1e-10), 1.0)
            eff_5m = np.where(mask & (vol_ratio > 1e-10), np.minimum(body_ratio / vol_ratio, 3.0), 1.0)
            eff_5m[:20] = 1.0
            _expand_5min_to_1min(eff_5m, ds, de, n5, effort_arr)

        # Trend 5min: EMA(13) slope / close
        if n5 > 13:
            ema13 = np.copy(c5)
            a13 = 2.0 / 14
            for k in range(1, n5):
                ema13[k] = a13 * c5[k] + (1 - a13) * ema13[k - 1]
            slope_5m = np.zeros(n5)
            slope_5m[1:] = np.where(c5[1:] > 0, (ema13[1:] - ema13[:-1]) / c5[1:], 0.0)
            _expand_5min_to_1min(slope_5m, ds, de, n5, trend_5m_arr)

    # =========================================================================
    # VECTORIZED feature assignment (replaces the 394K-iteration bar loop)
    # =========================================================================
    fi = 0  # running feature column index

    # [0] ret_6: 30-bar return (zero if within first 30 bars of day)
    ret30 = np.zeros(n)
    valid30 = (bod >= 30) & (c > 0)
    idx30 = np.arange(n)
    safe30 = valid30 & (c[np.clip(idx30 - 30, 0, n - 1)] > 0)
    ret30[safe30] = c[safe30] / c[(idx30 - 30)[safe30]] - 1.0
    feat[:, fi] = ret30
    fi += 1

    # [1] ret_12: 60-bar return
    ret60 = np.zeros(n)
    valid60 = (bod >= 60) & (c > 0)
    safe60 = valid60 & (c[np.clip(idx30 - 60, 0, n - 1)] > 0)
    ret60[safe60] = c[safe60] / c[(idx30 - 60)[safe60]] - 1.0
    feat[:, fi] = ret60
    fi += 1

    # [2] volume_ratio (rolling mean, vectorized via pandas)
    vol_s = pd.Series(vol)
    # Per-day rolling: use groupby to avoid cross-day contamination
    vol_rolling_mean = np.ones(n)
    for ds, de in zip(day_starts, day_ends):
        day_vol = vol[ds:de]
        rm = pd.Series(day_vol).rolling(LOOKBACK, min_periods=5).mean().values
        rm = np.where(np.isnan(rm), 1.0, np.maximum(rm, 1.0))
        vol_rolling_mean[ds:de] = rm
    feat[:, fi] = vol / vol_rolling_mean
    feat[c <= 0, fi] = 0.0
    fi += 1

    # [3] bar_range
    feat[:, fi] = bar_range_raw
    fi += 1

    # [4] realized_vol (20-bar rolling std of log_ret, per-day)
    realized_vol_arr = np.zeros(n)
    for ds, de in zip(day_starts, day_ends):
        lr = log_ret[ds:de]
        rv = pd.Series(lr).rolling(20, min_periods=5).std().values
        rv = np.nan_to_num(rv, nan=0.0)
        realized_vol_arr[ds:de] = rv
    feat[:, fi] = realized_vol_arr
    fi += 1

    # [5] range_ratio (bar_range / 20-bar rolling mean of bar_range)
    range_rolling = np.ones(n)
    for ds, de in zip(day_starts, day_ends):
        br = bar_range_raw[ds:de]
        rm = pd.Series(br).rolling(20, min_periods=5).mean().values
        rm = np.where(np.isnan(rm), 1e-8, np.maximum(rm, 1e-8))
        range_rolling[ds:de] = rm
    feat[:, fi] = bar_range_raw / range_rolling
    feat[c <= 0, fi] = 0.0
    fi += 1

    # [6] vwap_dist
    feat[:, fi] = np.where((c > 0) & ~np.isnan(vwap_arr), (c - vwap_arr) / c, 0.0)
    fi += 1

    # [7] session_range_pct
    sess_range = sess_high - sess_low
    feat[:, fi] = np.where(c > 0, sess_range / c, 0.0)
    fi += 1

    # [8] prev_high_dist
    feat[:, fi] = np.where((c > 0) & (prev_high_arr > 0), (c - prev_high_arr) / c, 0.0)
    fi += 1

    # [9] ema_cross
    ema_valid = ~np.isnan(ema8) & ~np.isnan(ema21) & (c > 0)
    feat[:, fi] = np.where(ema_valid, (ema8 - ema21) / c, 0.0)
    fi += 1

    # [10] consec_direction (per-day loop -- inner loop is max 20 bars, fast enough)
    consec_arr = np.zeros(n, dtype=np.float64)
    for ds, de in zip(day_starts, day_ends):
        for i in range(ds + 1, de):
            direction = 1 if c[i] >= c[i - 1] else -1
            consec = 0
            for j in range(i, max(i - 20, ds) - 1, -1):
                if j <= ds:
                    break
                d = 1 if c[j] >= c[j - 1] else -1
                if d == direction:
                    consec += 1
                else:
                    break
            consec_arr[i] = direction * min(consec, 10) / 10.0
    feat[:, fi] = consec_arr
    fi += 1

    # [11] speed_estimate
    ret5 = np.zeros(n)
    valid5 = (bod >= 5) & (c > 0)
    safe5 = valid5 & (c[np.clip(idx30 - 5, 0, n - 1)] > 0)
    ret5[safe5] = np.abs(c[safe5] / c[(idx30 - 5)[safe5]] - 1.0)
    rv = realized_vol_arr
    feat[:, fi] = np.where(rv > 1e-8, ret5 / rv, 0.0)
    fi += 1

    # [12] vix_roc
    vix_roc = np.zeros(n)
    valid_vix = (bod >= 10) & ~np.isnan(vix_close)
    prev_vix = vix_close[np.clip(idx30 - 10, 0, n - 1)]
    same_day_10 = day_of == day_of[np.clip(idx30 - 10, 0, n - 1)]
    vix_ok = valid_vix & ~np.isnan(prev_vix) & (prev_vix > 0) & same_day_10
    vix_roc[vix_ok] = (vix_close[vix_ok] - prev_vix[vix_ok]) / prev_vix[vix_ok]
    feat[:, fi] = vix_roc
    fi += 1

    # [13] minutes_to_close
    mtc = np.maximum(BARS_PER_DAY - bod, 1).astype(np.float64)
    feat[:, fi] = np.log1p(mtc) / np.log1p(BARS_PER_DAY)
    fi += 1

    # [14] vix_regime
    vx = np.where(np.isnan(vix_close), 20.0, vix_close)
    regime = np.where(vx < 15, -1.0,
             np.where(vx < 20, -0.33,
             np.where(vx < 30, 0.33, 1.0)))
    feat[:, fi] = regime
    fi += 1

    # [15] bollinger_position
    feat[:, fi] = boll_pos
    fi += 1

    # [16] rsi_7
    feat[:, fi] = rsi_7_arr
    fi += 1

    # [17] session_range_position
    sr = sess_high - sess_low
    feat[:, fi] = np.where(sr > 1e-10, (c - sess_low) / sr, 0.5)
    fi += 1

    # [18] poc_dist
    feat[:, fi] = np.where((c > 0) & (poc_arr > 0), (c - poc_arr) / c, 0.0)
    fi += 1

    # [19] va_position
    va_range = va_hi_arr - va_lo_arr
    feat[:, fi] = np.where(va_range > 1e-10, (c - va_lo_arr) / va_range, 0.5)
    fi += 1

    # [20] ib_break
    ib_break = np.zeros(n)
    after_ib = bod >= IB_BARS
    ib_break[after_ib & (c > ib_high_arr)] = 1.0
    ib_break[after_ib & (c < ib_low_arr)] = -1.0
    feat[:, fi] = ib_break
    fi += 1

    # [21] atr_14 (per-day rolling, vectorized)
    atr_arr = np.zeros(n)
    for ds, de in zip(day_starts, day_ends):
        if de - ds < 15:
            continue
        day_h = h[ds:de]
        day_l = lo[ds:de]
        day_c = c[ds:de]
        tr = np.maximum(day_h - day_l,
             np.maximum(np.abs(day_h - np.roll(day_c, 1)),
                        np.abs(day_l - np.roll(day_c, 1))))
        tr[0] = day_h[0] - day_l[0]  # first bar: no prev close
        atr_14 = pd.Series(tr).rolling(14, min_periods=14).mean().values
        atr_arr[ds:de] = np.where(day_c > 0, np.nan_to_num(atr_14, nan=0.0) / day_c, 0.0)
    feat[:, fi] = atr_arr
    fi += 1

    # [22] bar_delta
    feat[:, fi] = bar_delta_raw
    fi += 1

    # [23] session_cum_delta
    feat[:, fi] = session_cum_delta
    fi += 1

    # [24] macdh_slope
    feat[:, fi] = macdh_slope_arr
    fi += 1

    # [25] force_index_2
    feat[:, fi] = force_idx_arr
    fi += 1

    # [26] effort_vs_result
    feat[:, fi] = effort_arr
    fi += 1

    # [27] trend_5min
    feat[:, fi] = trend_5m_arr
    fi += 1

    assert fi == N_FEAT, f"Expected {N_FEAT} features, assigned {fi}"
    return feat


# ---------------------------------------------------------------------------
# Group 2: Option features from wide grid (per-bar)
# ---------------------------------------------------------------------------

def compute_option_features(
    bar_data: dict,
    spx_current: float,
    atm_strike_open: float,
    minutes_to_close: float,
    iv_history: list[float] | None = None,
) -> dict:
    """Compute 11 option features for one bar from wide-grid data.

    Args:
        bar_data: {strike: {call_close, put_close, call_high, call_low, ...}}
        spx_current: current SPX price
        atm_strike_open: ATM strike set at session open (for moneyness drift)
        minutes_to_close: minutes remaining in session
        iv_history: list of historical ATM IV values for percentile calculation

    Returns: dict of feature_name -> value (NaN for missing)
    """
    features = {}
    T = minutes_to_close / (252.0 * BARS_PER_DAY)  # time to expiry in years

    if np.isnan(spx_current) or spx_current <= 0 or not bar_data:
        return _default_option_features()

    near_atm = _find_nearest_atm(bar_data, spx_current)
    if near_atm is None:
        return _default_option_features()
    near_data = bar_data.get(near_atm, {})

    # ATM IV from dynamic nearest strike
    call_close = near_data.get('call_close', np.nan) or np.nan
    put_close = near_data.get('put_close', np.nan) or np.nan
    call_iv = np.nan
    put_iv = np.nan

    if T > 1e-10 and not np.isnan(call_close) and call_close > 0:
        call_iv = _bs_iv(call_close, spx_current, near_atm, T, RISK_FREE_RATE, is_call=True)
    if T > 1e-10 and not np.isnan(put_close) and put_close > 0:
        put_iv = _bs_iv(put_close, spx_current, near_atm, T, RISK_FREE_RATE, is_call=False)

    # [0] atm_iv: average of call and put IV
    if np.isfinite(call_iv) and np.isfinite(put_iv):
        atm_iv = (call_iv + put_iv) / 2
    elif np.isfinite(call_iv):
        atm_iv = call_iv
    elif np.isfinite(put_iv):
        atm_iv = put_iv
    else:
        atm_iv = np.nan
    features['atm_iv'] = atm_iv

    # [1] vrp: variance risk premium
    # realized_vol is computed externally; caller sets this from price features
    features['vrp'] = np.nan  # caller fills in after merge with price features

    # [2] iv_percentile: rank vs 60-day history
    if np.isfinite(atm_iv) and iv_history and len(iv_history) >= 5:
        hist = np.array(iv_history)
        features['iv_percentile'] = float(np.sum(hist <= atm_iv)) / len(hist)
    else:
        features['iv_percentile'] = np.nan

    # [3] atm_gamma
    gamma_val = np.nan
    theta_val = np.nan
    if np.isfinite(atm_iv) and T > 1e-10:
        _, gamma_val, theta_val, _ = _bs_greeks(spx_current, near_atm, T, RISK_FREE_RATE, atm_iv)
    features['atm_gamma'] = gamma_val

    # [4] atm_theta_per_bar
    features['atm_theta_per_bar'] = theta_val

    # [5] gamma_pressure: sum(gamma_i * volume_i * sign_i) across chain
    gex = 0.0
    gex_valid = False
    if T > 1e-10:
        for strike, sdata in bar_data.items():
            for side, sign, is_call in [('call', 1.0, True), ('put', -1.0, False)]:
                px = sdata.get(f'{side}_close', np.nan) or np.nan
                sv = sdata.get(f'{side}_volume', 0) or 0
                if np.isnan(px) or px <= 0 or sv <= 0:
                    continue
                iv = _bs_iv(px, spx_current, strike, T, RISK_FREE_RATE, is_call=is_call)
                if not np.isfinite(iv):
                    continue
                _, g, _, _ = _bs_greeks(spx_current, strike, T, RISK_FREE_RATE, iv)
                if np.isfinite(g):
                    gex += g * sv * sign
                    gex_valid = True
    features['gamma_pressure'] = gex if gex_valid else np.nan

    # [6] option_spread_pct: (high - low) / mid for nearest ATM call
    call_high = near_data.get('call_high', np.nan) or np.nan
    call_low = near_data.get('call_low', np.nan) or np.nan
    if not np.isnan(call_high) and not np.isnan(call_low) and call_high > 0 and call_low > 0:
        mid = (call_high + call_low) / 2
        features['option_spread_pct'] = (call_high - call_low) / mid if mid > 0 else 0.0
    else:
        features['option_spread_pct'] = np.nan

    # [7] iv_skew_pct: (put_iv - call_iv) / atm_iv
    if np.isfinite(call_iv) and np.isfinite(put_iv) and np.isfinite(atm_iv) and atm_iv > 0.01:
        features['iv_skew_pct'] = (put_iv - call_iv) / atm_iv
    else:
        features['iv_skew_pct'] = np.nan

    # [8] current_moneyness_pct: drift from opening ATM
    features['current_moneyness_pct'] = (atm_strike_open - spx_current) / spx_current * 100

    # [9] near_atm_moneyness_pct
    features['near_atm_moneyness_pct'] = (near_atm - spx_current) / spx_current * 100

    # [10] theta_acceleration: 0DTE specific
    features['theta_acceleration'] = 1.0 / math.sqrt(max(minutes_to_close, 1.0))

    return features


def _default_option_features() -> dict:
    """Return NaN for all option features when data is missing."""
    return {
        'atm_iv': np.nan,
        'vrp': np.nan,
        'iv_percentile': np.nan,
        'atm_gamma': np.nan,
        'atm_theta_per_bar': np.nan,
        'gamma_pressure': np.nan,
        'option_spread_pct': np.nan,
        'iv_skew_pct': np.nan,
        'current_moneyness_pct': np.nan,
        'near_atm_moneyness_pct': np.nan,
        'theta_acceleration': np.nan,
    }


# ---------------------------------------------------------------------------
# Group 3: Volume / Flow features from wide grid (per-bar)
# ---------------------------------------------------------------------------

def compute_flow_features(bar_data: dict, spx_current: float) -> dict:
    """Compute 8 volume/flow features for one bar from wide-grid data.

    Returns: dict of feature_name -> value
    """
    features = {}

    if np.isnan(spx_current) or spx_current <= 0 or not bar_data:
        return _default_flow_features()

    near_atm = _find_nearest_atm(bar_data, spx_current)
    if near_atm is None:
        return _default_flow_features()
    near_data = bar_data.get(near_atm, {})

    # Nearest ATM volumes
    call_vol = near_data.get('call_volume', 0) or 0
    put_vol = near_data.get('put_volume', 0) or 0
    total_vol = call_vol + put_vol

    features['log_near_call_volume'] = math.log1p(call_vol)
    features['log_near_put_volume'] = math.log1p(put_vol)
    features['call_put_flow_ratio'] = call_vol / total_vol if total_vol > 0 else 0.5
    features['log_total_volume'] = math.log1p(total_vol)

    # Chain-wide volumes
    chain_call_vol = sum(bar_data[s].get('call_volume', 0) or 0 for s in bar_data)
    chain_put_vol = sum(bar_data[s].get('put_volume', 0) or 0 for s in bar_data)
    chain_total = chain_call_vol + chain_put_vol
    features['chain_call_put_ratio'] = chain_call_vol / chain_total if chain_total > 0 else 0.5
    features['log_chain_volume'] = math.log1p(chain_total)

    # Transaction counts
    call_txn = near_data.get('call_transactions', 0) or 0
    put_txn = near_data.get('put_transactions', 0) or 0
    total_txn = call_txn + put_txn
    features['log_near_transactions'] = math.log1p(total_txn)
    features['put_call_txn_ratio'] = put_txn / total_txn if total_txn > 0 else 0.5

    return features


def _default_flow_features() -> dict:
    """Return defaults for all flow features when data is missing."""
    return {
        'log_near_call_volume': 0.0,
        'log_near_put_volume': 0.0,
        'call_put_flow_ratio': 0.5,
        'log_total_volume': 0.0,
        'chain_call_put_ratio': 0.5,
        'log_chain_volume': 0.0,
        'log_near_transactions': 0.0,
        'put_call_txn_ratio': 0.5,
    }


# ---------------------------------------------------------------------------
# Feature name lists (canonical order)
# ---------------------------------------------------------------------------

PRICE_FEATURE_NAMES = [
    'ret_6', 'ret_12', 'volume_ratio', 'bar_range', 'realized_vol',
    'range_ratio', 'vwap_dist', 'session_range_pct', 'prev_high_dist',
    'ema_cross', 'consec_direction', 'speed_estimate', 'vix_roc',
    'minutes_to_close', 'vix_regime', 'bollinger_position', 'rsi_7',
    'session_range_position', 'poc_dist', 'va_position', 'ib_break',
    'atr_14', 'bar_delta', 'session_cum_delta', 'macdh_slope',
    'force_index_2', 'effort_vs_result', 'trend_5min',
]

OPTION_FEATURE_NAMES = [
    'atm_iv', 'vrp', 'iv_percentile', 'atm_gamma', 'atm_theta_per_bar',
    'gamma_pressure', 'option_spread_pct', 'iv_skew_pct',
    'current_moneyness_pct', 'near_atm_moneyness_pct', 'theta_acceleration',
]

FLOW_FEATURE_NAMES = [
    'log_near_call_volume', 'log_near_put_volume', 'call_put_flow_ratio',
    'log_total_volume', 'chain_call_put_ratio', 'log_chain_volume',
    'log_near_transactions', 'put_call_txn_ratio',
]

ALL_FEATURE_NAMES = PRICE_FEATURE_NAMES + OPTION_FEATURE_NAMES + FLOW_FEATURE_NAMES
NUM_FEATURES = len(ALL_FEATURE_NAMES)  # 47
