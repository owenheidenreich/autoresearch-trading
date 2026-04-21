"""Shared point-in-time market-structure helpers.

Single source of truth for SPY-derived VWAP σ-position, OMAR day constants,
and rolling last-10-bar state. Consumed by both v2 feature generation
(v2/pipeline/compute_features.py) and v3 research/teacher context
(v3/harness/v2_adapter.py, v3/analysis/*). Replaces the inline
_build_spy_vwap / _vwap_sigma_position / _build_omar_map copies that
previously lived in each analysis script.

The formulas here are the canonical research definitions; any divergence
between research and production σ-position must be traced to a bug in a
caller, not an alternative implementation.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np


SPY_CACHE_PATH = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"
SPX_CACHE_PATH = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"


def build_spy_vwap(spy_path: Path | str = SPY_CACHE_PATH) -> dict[str, dict[str, np.ndarray]]:
    """Per-day arrays for σ-position computation.

    Each value has keys {vwap, close, std} — std is cumulative-volume-weighted
    stddev of (close - vwap), matching research behavior.
    """
    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        vwap = g["vwap"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        volume = g["volume"].to_numpy(dtype=float)
        sq_dev = (close - vwap) ** 2
        cum_v = np.maximum(np.cumsum(volume), 1.0)
        cum_sq = np.cumsum(volume * sq_dev)
        std_run = np.sqrt(np.maximum(cum_sq / cum_v, 1e-12))
        out[day] = {"vwap": vwap, "close": close, "std": std_run}
    return out


def build_spx_bars(spx_path: Path | str = SPX_CACHE_PATH) -> dict[str, dict[str, np.ndarray]]:
    """Per-day SPX 1-minute high/low/close arrays."""
    df = pickle.load(open(spx_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue
        out[day] = {
            "high": g["spx_high"].to_numpy(dtype=float),
            "low": g["spx_low"].to_numpy(dtype=float),
            "close": g["spx_close"].to_numpy(dtype=float),
        }
    return out


def build_omar_map(spx_bars: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Per-day OMAR (first-minute 09:30-09:31) high / low / mid / range."""
    out: dict[str, dict[str, float]] = {}
    for day, arrs in spx_bars.items():
        if len(arrs["high"]) == 0:
            continue
        h = float(arrs["high"][0])
        l = float(arrs["low"][0])
        out[day] = {
            "high": h,
            "low": l,
            "mid": (h + l) / 2.0,
            "range": max(h - l, 0.01),
        }
    return out


def sigma_pos(
    spy_day: dict[str, np.ndarray],
    minute: int,
    spx_close: float,
) -> Optional[float]:
    """SPY-derived VWAP σ-position translated to SPX space.

    ratio = spx_close / spy_close; spx_vwap = spy_vwap * ratio; spx_std = spy_std * ratio;
    sigma_pos = (spx_close - spx_vwap) / spx_std.

    Returns None if the minute is out of range or any input is non-positive.
    """
    vwap = spy_day["vwap"]
    if minute < 0 or minute >= len(vwap):
        return None
    spy_vwap = float(vwap[minute])
    spy_close = float(spy_day["close"][minute])
    spy_std = float(spy_day["std"][minute])
    if not all(v > 0 for v in (spy_vwap, spy_close, spy_std, spx_close)):
        return None
    ratio = spx_close / spy_close
    spx_vwap = spy_vwap * ratio
    spx_std = spy_std * ratio
    return (spx_close - spx_vwap) / spx_std


def abs_sigma_pos(
    spy_day: dict[str, np.ndarray],
    minute: int,
    spx_close: float,
) -> Optional[float]:
    s = sigma_pos(spy_day, minute, spx_close)
    return None if s is None else abs(s)


def last10(
    spx_day: dict[str, np.ndarray],
    minute: int,
) -> Optional[dict[str, float]]:
    """Last-10-bar high/low/range as of `minute`, excluding the current bar.

    Returns None if no prior bars exist.
    """
    if minute <= 0:
        return None
    lo = max(minute - 10, 0)
    hi_slice = spx_day["high"][lo:minute]
    lw_slice = spx_day["low"][lo:minute]
    if len(hi_slice) == 0:
        return None
    last10_high = float(hi_slice.max())
    last10_low = float(lw_slice.min())
    return {
        "high": last10_high,
        "low": last10_low,
        "range": last10_high - last10_low,
    }


def last10_break_state(
    spx_day: dict[str, np.ndarray],
    minute: int,
    spx_close: float,
) -> Optional[int]:
    """Trinary break state vs last-10-bar range: +1 above, -1 below, 0 inside."""
    l10 = last10(spx_day, minute)
    if l10 is None:
        return None
    if spx_close > l10["high"]:
        return 1
    if spx_close < l10["low"]:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Vectorized (array-based) variants for the v2 feature pipeline
# ---------------------------------------------------------------------------
#
# The dict-based functions above are right for v3 research / teacher context
# where callers iterate one (day, minute) tuple at a time. The v2 feature
# pipeline operates on flat per-bar arrays and needs vectorized equivalents.
# Both paths must produce the same numbers; the vectorized variants below
# mirror the point-in-time formulas exactly so research and production
# cannot drift.


def sigma_pos_arrays(
    spx_close: np.ndarray,
    spy_close: np.ndarray,
    spy_volume: np.ndarray,
    day_starts: list[int],
) -> np.ndarray:
    """Per-bar σ-position (SPY-derived, SPX-scaled).

    Mirrors the per-minute formula in `sigma_pos()` exactly. Returns a
    float array of length n with values clipped to [-5, +5] and NaN/inf
    normalized to 0.0.
    """
    n = len(spx_close)
    out = np.zeros(n, dtype=np.float64)
    day_ends = list(day_starts[1:]) + [n]
    for ds, de in zip(day_starts, day_ends):
        spx_day = spx_close[ds:de].astype(np.float64)
        spy_cl = spy_close[ds:de].astype(np.float64)
        spy_vol = np.maximum(spy_volume[ds:de].astype(np.float64), 0.0)
        if len(spx_day) == 0:
            continue
        cum_v = np.maximum(np.cumsum(spy_vol), 1.0)
        cum_pv = np.cumsum(spy_cl * spy_vol)
        spy_vwap = cum_pv / cum_v
        sq_dev = (spy_cl - spy_vwap) ** 2
        cum_sq = np.cumsum(spy_vol * sq_dev)
        spy_std = np.sqrt(np.maximum(cum_sq / cum_v, 1e-12))
        valid = (spx_day > 0) & (spy_cl > 0) & (spy_vwap > 0) & (spy_std > 0)
        ratio = np.where(spy_cl > 0, spx_day / spy_cl, 1.0)
        spx_vwap = spy_vwap * ratio
        spx_std = np.maximum(spy_std * ratio, 1e-12)
        val = np.where(valid, (spx_day - spx_vwap) / spx_std, 0.0)
        out[ds:de] = val
    out = np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)
    return np.clip(out, -5.0, 5.0)


def omar_arrays(
    spx_high: np.ndarray,
    spx_low: np.ndarray,
    day_starts: list[int],
) -> dict[str, np.ndarray]:
    """Per-bar OMAR day constants (first-minute 09:30-09:31 high/low/mid/range),
    broadcast so every bar in a day carries that day's OMAR scalars.
    """
    n = len(spx_high)
    day_ends = list(day_starts[1:]) + [n]
    high_arr = np.zeros(n, dtype=np.float64)
    low_arr = np.zeros(n, dtype=np.float64)
    mid_arr = np.zeros(n, dtype=np.float64)
    range_arr = np.full(n, 0.01, dtype=np.float64)  # safe floor
    for ds, de in zip(day_starts, day_ends):
        if de <= ds:
            continue
        h0 = float(spx_high[ds])
        l0 = float(spx_low[ds])
        rng = max(h0 - l0, 0.01)
        high_arr[ds:de] = h0
        low_arr[ds:de] = l0
        mid_arr[ds:de] = (h0 + l0) / 2.0
        range_arr[ds:de] = rng
    return {"high": high_arr, "low": low_arr, "mid": mid_arr, "range": range_arr}


def last10_arrays(
    spx_high: np.ndarray,
    spx_low: np.ndarray,
    day_starts: list[int],
) -> dict[str, np.ndarray]:
    """Per-bar rolling last-10-bar high/low/range, computed within each day.

    At minute 0 (session open) there is no history; high/low fall back to the
    current bar values and range = 0. At minutes 1..9 the window shrinks.
    """
    n = len(spx_high)
    day_ends = list(day_starts[1:]) + [n]
    hi_arr = spx_high.astype(np.float64).copy()
    lo_arr = spx_low.astype(np.float64).copy()
    for ds, de in zip(day_starts, day_ends):
        for i in range(ds + 1, de):
            window_start = max(ds, i - 10)
            hi_arr[i] = float(np.max(spx_high[window_start:i]))
            lo_arr[i] = float(np.min(spx_low[window_start:i]))
    range_arr = np.maximum(hi_arr - lo_arr, 0.0)
    return {"high": hi_arr, "low": lo_arr, "range": range_arr}

