"""Point-in-time market-structure helpers for v4 research paths."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np


SPY_CACHE_PATH = Path.home() / ".cache/autoresearch-trading/data/spy_1min.pkl"
SPX_CACHE_PATH = Path.home() / ".cache/autoresearch-trading/data/spx_1min.pkl"


def build_spy_vwap(spy_path: Path | str = SPY_CACHE_PATH) -> dict[str, dict[str, np.ndarray]]:
    """Per-day arrays for SPY-derived VWAP sigma-position computation."""
    df = pickle.load(open(spy_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, group in df.groupby("date"):
        group = group.reset_index(drop=True)
        if len(group) == 0:
            continue
        vwap = group["vwap"].to_numpy(dtype=float)
        close = group["close"].to_numpy(dtype=float)
        volume = group["volume"].to_numpy(dtype=float)
        sq_dev = (close - vwap) ** 2
        cum_v = np.maximum(np.cumsum(volume), 1.0)
        cum_sq = np.cumsum(volume * sq_dev)
        std_run = np.sqrt(np.maximum(cum_sq / cum_v, 1e-12))
        out[day] = {"vwap": vwap, "close": close, "std": std_run}
    return out


def build_spx_bars(spx_path: Path | str = SPX_CACHE_PATH) -> dict[str, dict[str, np.ndarray]]:
    """Per-day SPX one-minute high/low/close arrays."""
    df = pickle.load(open(spx_path, "rb"))
    df["date"] = df["date"].astype(str)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day, group in df.groupby("date"):
        group = group.reset_index(drop=True)
        if len(group) == 0:
            continue
        out[day] = {
            "high": group["spx_high"].to_numpy(dtype=float),
            "low": group["spx_low"].to_numpy(dtype=float),
            "close": group["spx_close"].to_numpy(dtype=float),
        }
    return out


def build_omar_map(spx_bars: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Per-day opening-minute high, low, midpoint, and range constants."""
    out: dict[str, dict[str, float]] = {}
    for day, arrays in spx_bars.items():
        if len(arrays["high"]) == 0:
            continue
        high = float(arrays["high"][0])
        low = float(arrays["low"][0])
        out[day] = {
            "high": high,
            "low": low,
            "mid": (high + low) / 2.0,
            "range": max(high - low, 0.01),
        }
    return out


def sigma_pos(
    spy_day: dict[str, np.ndarray],
    minute: int,
    spx_close: float,
) -> Optional[float]:
    """SPY-derived VWAP sigma-position translated to SPX space."""
    vwap = spy_day["vwap"]
    if minute < 0 or minute >= len(vwap):
        return None
    spy_vwap = float(vwap[minute])
    spy_close = float(spy_day["close"][minute])
    spy_std = float(spy_day["std"][minute])
    if not all(value > 0 for value in (spy_vwap, spy_close, spy_std, spx_close)):
        return None
    ratio = spx_close / spy_close
    spx_vwap = spy_vwap * ratio
    spx_std = spy_std * ratio
    return (spx_close - spx_vwap) / spx_std


def last10(
    spx_day: dict[str, np.ndarray],
    minute: int,
) -> Optional[dict[str, float]]:
    """Last-ten-bar high, low, and range as of ``minute``, excluding the current bar."""
    if minute <= 0:
        return None
    lo = max(minute - 10, 0)
    high_slice = spx_day["high"][lo:minute]
    low_slice = spx_day["low"][lo:minute]
    if len(high_slice) == 0:
        return None
    high = float(high_slice.max())
    low = float(low_slice.min())
    return {"high": high, "low": low, "range": high - low}


def last10_break_state(
    spx_day: dict[str, np.ndarray],
    minute: int,
    spx_close: float,
) -> Optional[int]:
    """Trinary break state versus the last-ten-bar range."""
    last = last10(spx_day, minute)
    if last is None:
        return None
    if spx_close > last["high"]:
        return 1
    if spx_close < last["low"]:
        return -1
    return 0
