"""Derived market-state cache for v3 multi-scale memory."""
from __future__ import annotations

import math
import os
import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch

from shared.chain_data import to_wide_bar
from shared.market_data import (
    FULL_CHAIN_CACHE_DIR,
    SPX_PATH,
    SPY_PATH,
    VIX_PATH,
    chain_bar_for_timestamp,
    load_market_cache,
)
from shared.option_math import bs_iv, find_nearest_atm, compute_flow_features
from v3.core.schema import BARS_PER_DAY, FIVE_MINUTE_BUCKET, LOOKBACK_5M, SESSION_STATE_FEATURE_NAMES


MARKET_STATE_VERSION = "v3_market_state_v1"
DEFAULT_MARKET_STATE_PATH = os.path.join("v3", "data", "market_state_v1.pt")

FIVE_MINUTE_FEATURE_NAMES = (
    "ret_1",
    "ret_3",
    "ret_6",
    "range_frac",
    "body_frac",
    "realized_vol_6",
    "vwap_dist",
    "session_range_position",
    "dist_to_session_high_pct",
    "dist_to_session_low_pct",
    "ema_cross",
    "bollinger_position",
    "rsi_7",
    "atm_iv",
    "option_spread_pct",
    "call_put_flow_ratio",
    "chain_call_put_ratio",
    "put_call_txn_ratio",
)

FIVE_MINUTE_NO_NORMALIZE = {
    "session_range_position",
    "bollinger_position",
    "rsi_7",
    "call_put_flow_ratio",
    "chain_call_put_ratio",
    "put_call_txn_ratio",
}

FIVE_MINUTE_WINDOW = LOOKBACK_5M * 60


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    out = np.zeros_like(series, dtype=np.float64)
    if len(series) == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = float(series[0])
    for i in range(1, len(series)):
        out[i] = alpha * float(series[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_rsi(close: np.ndarray, period: int = 7) -> np.ndarray:
    if len(close) == 0:
        return np.zeros(0, dtype=np.float64)
    diff = np.diff(close, prepend=close[0])
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = np.zeros_like(close, dtype=np.float64)
    avg_loss = np.zeros_like(close, dtype=np.float64)
    for i in range(len(close)):
        start = max(0, i - period + 1)
        avg_gain[i] = gains[start : i + 1].mean()
        avg_loss[i] = losses[start : i + 1].mean()
    rs = avg_gain / np.clip(avg_loss, 1e-6, None)
    return 1.0 - (1.0 / (1.0 + rs))


def _aggregate_contract_bars(
    bars: list[dict[tuple[float, str], dict[str, float]]],
) -> dict[tuple[float, str], dict[str, float]]:
    agg: dict[tuple[float, str], dict[str, float]] = {}
    for bar in bars:
        for contract, fields in bar.items():
            close_px = float(fields.get("close", np.nan) or np.nan)
            if not np.isfinite(close_px) or close_px <= 0:
                continue
            row = agg.get(contract)
            if row is None:
                agg[contract] = {
                    "open": float(fields.get("open", close_px) or close_px),
                    "high": float(fields.get("high", close_px) or close_px),
                    "low": float(fields.get("low", close_px) or close_px),
                    "close": close_px,
                    "volume": float(fields.get("volume", 0.0) or 0.0),
                    "transactions": float(fields.get("transactions", 0.0) or 0.0),
                }
                continue
            row["high"] = max(float(row["high"]), float(fields.get("high", close_px) or close_px))
            row["low"] = min(float(row["low"]), float(fields.get("low", close_px) or close_px))
            row["close"] = close_px
            row["volume"] = float(row["volume"]) + float(fields.get("volume", 0.0) or 0.0)
            row["transactions"] = float(row["transactions"]) + float(fields.get("transactions", 0.0) or 0.0)
    return agg


def _option_volume_per_bar(
    raw_bars: list[dict[tuple[float, str], dict[str, float]]],
) -> np.ndarray:
    out = np.zeros((len(raw_bars),), dtype=np.float64)
    for i, bar in enumerate(raw_bars):
        total = 0.0
        for fields in bar.values():
            total += float(fields.get("volume", 0.0) or 0.0)
        out[i] = total
    return out


def _normalize_five_minute_features(raw: np.ndarray) -> np.ndarray:
    if len(raw) == 0:
        return raw.astype(np.float32, copy=True)
    out = raw.astype(np.float64, copy=True)
    for j, name in enumerate(FIVE_MINUTE_FEATURE_NAMES):
        if name in FIVE_MINUTE_NO_NORMALIZE:
            continue
        col = pd.Series(out[:, j], dtype=np.float64)
        mu = col.expanding(min_periods=5).mean()
        sigma = col.expanding(min_periods=5).std()
        rolling_mu = col.rolling(FIVE_MINUTE_WINDOW, min_periods=5).mean()
        rolling_sigma = col.rolling(FIVE_MINUTE_WINDOW, min_periods=5).std()
        switch_idx = min(FIVE_MINUTE_WINDOW, len(col))
        mu.iloc[switch_idx:] = rolling_mu.iloc[switch_idx:]
        sigma.iloc[switch_idx:] = rolling_sigma.iloc[switch_idx:]
        normalized = (col - mu) / sigma.clip(lower=1e-10)
        out[:, j] = normalized.fillna(0.0).values
    out = np.clip(out, -5.0, 5.0)
    out = np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)
    return out.astype(np.float32)


def _compute_five_minute_raw_features(
    *,
    day_open: np.ndarray,
    day_high: np.ndarray,
    day_low: np.ndarray,
    day_close: np.ndarray,
    day_spy_volume: np.ndarray,
    raw_bars: list[dict[tuple[float, str], dict[str, float]]],
) -> np.ndarray:
    n_bars = len(day_close)
    bucket_starts = list(range(0, n_bars, FIVE_MINUTE_BUCKET))
    n_buckets = len(bucket_starts)
    if n_buckets == 0:
        return np.zeros((0, len(FIVE_MINUTE_FEATURE_NAMES)), dtype=np.float32)

    open_5m = np.zeros((n_buckets,), dtype=np.float64)
    high_5m = np.zeros((n_buckets,), dtype=np.float64)
    low_5m = np.zeros((n_buckets,), dtype=np.float64)
    close_5m = np.zeros((n_buckets,), dtype=np.float64)
    spy_volume_5m = np.zeros((n_buckets,), dtype=np.float64)
    option_cols = {name: np.zeros((n_buckets,), dtype=np.float64) for name in ("atm_iv", "option_spread_pct")}
    flow_cols = {name: np.zeros((n_buckets,), dtype=np.float64) for name in ("call_put_flow_ratio", "chain_call_put_ratio", "put_call_txn_ratio")}
    atm_strike_open = round(float(day_open[0]) / 5.0) * 5.0

    for bucket_idx, start in enumerate(bucket_starts):
        end = min(start + FIVE_MINUTE_BUCKET, n_bars)
        open_5m[bucket_idx] = float(day_open[start])
        high_5m[bucket_idx] = float(np.max(day_high[start:end]))
        low_5m[bucket_idx] = float(np.min(day_low[start:end]))
        close_5m[bucket_idx] = float(day_close[end - 1])
        spy_volume_5m[bucket_idx] = float(np.maximum(day_spy_volume[start:end], 0.0).sum())

        agg_contracts = _aggregate_contract_bars(raw_bars[start:end])
        wide_bar = to_wide_bar(agg_contracts)
        opt = _compute_bucket_option_state(
            wide_bar=wide_bar,
            spx_current=float(close_5m[bucket_idx]),
            atm_strike_open=atm_strike_open,
            minutes_to_close=max(1.0, float(BARS_PER_DAY - end)),
        )
        flow = compute_flow_features(wide_bar, float(close_5m[bucket_idx]))
        for name in option_cols:
            option_cols[name][bucket_idx] = float(opt.get(name, 0.0) or 0.0)
        for name in flow_cols:
            flow_cols[name][bucket_idx] = float(flow.get(name, 0.0) or 0.0)

    log_ret = np.zeros((n_buckets,), dtype=np.float64)
    valid_prev = close_5m[:-1] > 0
    log_ret[1:] = np.where(valid_prev, np.log(np.clip(close_5m[1:], 1e-6, None) / np.clip(close_5m[:-1], 1e-6, None)), 0.0)
    ret_3 = np.zeros_like(log_ret)
    ret_6 = np.zeros_like(log_ret)
    for i in range(n_buckets):
        if i >= 3 and close_5m[i - 3] > 0:
            ret_3[i] = math.log(close_5m[i] / close_5m[i - 3])
        if i >= 6 and close_5m[i - 6] > 0:
            ret_6[i] = math.log(close_5m[i] / close_5m[i - 6])

    range_frac = np.where(close_5m > 0, (high_5m - low_5m) / np.clip(close_5m, 1e-6, None), 0.0)
    body_frac = np.clip(
        (close_5m - open_5m) / np.clip(high_5m - low_5m, 1e-6, None),
        -1.0,
        1.0,
    )
    realized_vol_6 = np.zeros_like(close_5m)
    for i in range(n_buckets):
        start = max(0, i - 5)
        realized_vol_6[i] = float(np.std(log_ret[start : i + 1], ddof=0))

    cum_pv = np.cumsum(close_5m * np.maximum(spy_volume_5m, 0.0))
    cum_v = np.cumsum(np.maximum(spy_volume_5m, 0.0))
    vwap = np.where(cum_v > 0, cum_pv / np.clip(cum_v, 1e-6, None), close_5m)
    vwap_dist = np.where(close_5m > 0, (close_5m - vwap) / np.clip(close_5m, 1e-6, None), 0.0)

    running_high = np.maximum.accumulate(high_5m)
    running_low = np.minimum.accumulate(low_5m)
    session_range_position = np.where(
        (running_high - running_low) > 1e-6,
        (close_5m - running_low) / np.clip(running_high - running_low, 1e-6, None),
        0.5,
    )
    dist_to_session_high = np.where(close_5m > 0, (close_5m - running_high) / np.clip(close_5m, 1e-6, None), 0.0)
    dist_to_session_low = np.where(close_5m > 0, (close_5m - running_low) / np.clip(close_5m, 1e-6, None), 0.0)

    ema_cross = np.where(
        close_5m > 0,
        (_ema(close_5m, 8) - _ema(close_5m, 21)) / np.clip(close_5m, 1e-6, None),
        0.0,
    )

    bollinger_position = np.zeros_like(close_5m)
    for i in range(n_buckets):
        start = max(0, i - 19)
        window = close_5m[start : i + 1]
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        denom = max(std * 2.0, 1e-6)
        bollinger_position[i] = float(np.clip((close_5m[i] - mean) / denom, -3.0, 3.0))

    rsi_7 = _rolling_rsi(close_5m, period=7)

    raw = np.column_stack(
        [
            log_ret,
            ret_3,
            ret_6,
            range_frac,
            body_frac,
            realized_vol_6,
            vwap_dist,
            session_range_position,
            dist_to_session_high,
            dist_to_session_low,
            ema_cross,
            bollinger_position,
            rsi_7,
            option_cols["atm_iv"],
            option_cols["option_spread_pct"],
            flow_cols["call_put_flow_ratio"],
            flow_cols["chain_call_put_ratio"],
            flow_cols["put_call_txn_ratio"],
        ]
    )
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _compute_bucket_option_state(
    *,
    wide_bar: dict[float, dict[str, float]],
    spx_current: float,
    atm_strike_open: float,
    minutes_to_close: float,
) -> dict[str, float]:
    if not wide_bar or not np.isfinite(spx_current) or spx_current <= 0:
        return {"atm_iv": 0.0, "option_spread_pct": 0.0}
    near_atm = find_nearest_atm(wide_bar, spx_current)
    if near_atm is None:
        return {"atm_iv": 0.0, "option_spread_pct": 0.0}
    near_data = wide_bar.get(near_atm, {})
    t_years = minutes_to_close / (252.0 * BARS_PER_DAY)
    call_close = float(near_data.get("call_close", np.nan) or np.nan)
    put_close = float(near_data.get("put_close", np.nan) or np.nan)
    call_iv = np.nan
    put_iv = np.nan
    if t_years > 1e-10 and np.isfinite(call_close) and call_close > 0:
        call_iv = bs_iv(call_close, spx_current, near_atm, t_years, 0.05, is_call=True)
    if t_years > 1e-10 and np.isfinite(put_close) and put_close > 0:
        put_iv = bs_iv(put_close, spx_current, near_atm, t_years, 0.05, is_call=False)

    if np.isfinite(call_iv) and np.isfinite(put_iv):
        atm_iv = float((call_iv + put_iv) / 2.0)
    elif np.isfinite(call_iv):
        atm_iv = float(call_iv)
    elif np.isfinite(put_iv):
        atm_iv = float(put_iv)
    else:
        atm_iv = 0.0

    call_high = float(near_data.get("call_high", np.nan) or np.nan)
    call_low = float(near_data.get("call_low", np.nan) or np.nan)
    if np.isfinite(call_high) and np.isfinite(call_low) and call_high > 0 and call_low > 0:
        mid = (call_high + call_low) / 2.0
        option_spread_pct = float((call_high - call_low) / max(mid, 1e-6))
    else:
        option_spread_pct = 0.0
    return {
        "atm_iv": atm_iv,
        "option_spread_pct": option_spread_pct,
        "atm_strike_open": atm_strike_open,
    }


def _bars_since_extreme(values: np.ndarray, *, find_high: bool) -> np.ndarray:
    out = np.zeros((len(values),), dtype=np.float64)
    if len(values) == 0:
        return out
    best = values[0]
    best_idx = 0
    for i, value in enumerate(values):
        improved = value >= best if find_high else value <= best
        if improved:
            best = value
            best_idx = i
        out[i] = float(i - best_idx)
    return out


def _compute_session_state(
    *,
    day_open: np.ndarray,
    day_high: np.ndarray,
    day_low: np.ndarray,
    day_close: np.ndarray,
    day_spy_volume: np.ndarray,
    option_volume: np.ndarray,
    baseline_cum_volume: np.ndarray,
) -> np.ndarray:
    n_bars = len(day_close)
    minutes_from_open = np.arange(n_bars, dtype=np.float64) / float(BARS_PER_DAY)
    minutes_to_close = np.maximum(0, n_bars - 1 - np.arange(n_bars, dtype=np.float64)) / float(BARS_PER_DAY)
    session_open = float(day_open[0]) if n_bars else 0.0
    spot_return_from_open = np.where(day_close > 0, day_close / max(session_open, 1e-6) - 1.0, 0.0)

    cum_pv = np.cumsum(day_close * np.maximum(day_spy_volume, 0.0))
    cum_v = np.cumsum(np.maximum(day_spy_volume, 0.0))
    vwap = np.where(cum_v > 0, cum_pv / np.clip(cum_v, 1e-6, None), day_close)
    spot_return_from_vwap = np.where(day_close > 0, day_close / np.clip(vwap, 1e-6, None) - 1.0, 0.0)

    session_high = np.maximum.accumulate(day_high)
    session_low = np.minimum.accumulate(day_low)
    dist_to_session_high = np.where(day_close > 0, (day_close - session_high) / np.clip(day_close, 1e-6, None), 0.0)
    dist_to_session_low = np.where(day_close > 0, (day_close - session_low) / np.clip(day_close, 1e-6, None), 0.0)

    opening_high = np.zeros((n_bars,), dtype=np.float64)
    opening_low = np.zeros((n_bars,), dtype=np.float64)
    initial_high = np.zeros((n_bars,), dtype=np.float64)
    initial_low = np.zeros((n_bars,), dtype=np.float64)
    or_high = float(day_high[0])
    or_low = float(day_low[0])
    ib_high = float(day_high[0])
    ib_low = float(day_low[0])
    for i in range(n_bars):
        if i < 30:
            or_high = max(or_high, float(day_high[i]))
            or_low = min(or_low, float(day_low[i]))
        opening_high[i] = or_high
        opening_low[i] = or_low
        if i < 60:
            ib_high = max(ib_high, float(day_high[i]))
            ib_low = min(ib_low, float(day_low[i]))
        initial_high[i] = ib_high
        initial_low[i] = ib_low

    dist_to_or_high = np.where(day_close > 0, (day_close - opening_high) / np.clip(day_close, 1e-6, None), 0.0)
    dist_to_or_low = np.where(day_close > 0, (day_close - opening_low) / np.clip(day_close, 1e-6, None), 0.0)
    dist_to_ib_high = np.where(day_close > 0, (day_close - initial_high) / np.clip(day_close, 1e-6, None), 0.0)
    dist_to_ib_low = np.where(day_close > 0, (day_close - initial_low) / np.clip(day_close, 1e-6, None), 0.0)

    session_range = session_high - session_low
    or_range = max(float(opening_high[min(29, n_bars - 1)] - opening_low[min(29, n_bars - 1)]), 1e-6)
    session_range_expansion = np.zeros((n_bars,), dtype=np.float64)
    if n_bars > 30:
        session_range_expansion[30:] = session_range[30:] / or_range - 1.0

    log_ret = np.zeros((n_bars,), dtype=np.float64)
    if n_bars > 1:
        valid_prev = day_close[:-1] > 0
        log_ret[1:] = np.where(valid_prev, np.log(np.clip(day_close[1:], 1e-6, None) / np.clip(day_close[:-1], 1e-6, None)), 0.0)
    realized_vol = np.zeros((n_bars,), dtype=np.float64)
    for i in range(n_bars):
        realized_vol[i] = float(np.std(log_ret[: i + 1], ddof=0))

    bars_since_high = _bars_since_extreme(session_high, find_high=True) / float(BARS_PER_DAY)
    bars_since_low = _bars_since_extreme(session_low, find_high=False) / float(BARS_PER_DAY)

    denom = np.clip(day_high - day_low, 1e-6, None)
    bar_delta = np.clip((day_close - day_open) / denom, -1.0, 1.0)
    cum_delta = np.tanh(np.cumsum(bar_delta) / 20.0)

    cum_option_volume = np.cumsum(option_volume)
    volume_ratio = np.zeros((n_bars,), dtype=np.float64)
    valid = baseline_cum_volume > 1e-6
    volume_ratio[valid] = (cum_option_volume[valid] - baseline_cum_volume[valid]) / baseline_cum_volume[valid]
    volume_ratio = np.clip(volume_ratio, -5.0, 5.0)

    out = np.column_stack(
        [
            minutes_from_open,
            minutes_to_close,
            spot_return_from_open,
            spot_return_from_vwap,
            dist_to_session_high,
            dist_to_session_low,
            dist_to_or_high,
            dist_to_or_low,
            dist_to_ib_high,
            dist_to_ib_low,
            session_range_expansion,
            realized_vol,
            bars_since_high,
            bars_since_low,
            cum_delta,
            volume_ratio,
        ]
    )
    return np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


def _aligned_market_arrays(data_path: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    dates_list = [str(d) for d in raw["dates"]]
    spx_df = load_market_cache(SPX_PATH, "spx")
    spy_df = load_market_cache(SPY_PATH, "spy")
    vix_df = load_market_cache(VIX_PATH, "vix")

    dataset_days = set(sorted(set(dates_list)))
    spx_mask = np.array([str(d) in dataset_days for d in spx_df["date"]], dtype=bool)
    spx_df = {k: np.asarray(v)[spx_mask] for k, v in spx_df.items()}
    if len(spx_df["date"]) != len(dates_list):
        raise ValueError("SPX cache length does not match v2 dataset after full-chain filtering")
    if [str(d) for d in spx_df["date"]] != dates_list:
        raise ValueError("SPX cache date order does not match v2 dataset")

    n = len(dates_list)
    spy_ts_to_idx = {str(ts): i for i, ts in enumerate(spy_df["timestamp"])}
    spy_volume = np.zeros(n, dtype=np.float64)
    spy_close = np.zeros(n, dtype=np.float64)
    for i, ts in enumerate(spx_df["timestamp"]):
        j = spy_ts_to_idx.get(str(ts))
        if j is not None:
            spy_volume[i] = float(spy_df["volume"][j])
            spy_close[i] = float(spy_df["close"][j])

    vix_ts_to_idx = {str(ts): i for i, ts in enumerate(vix_df["timestamp"])}
    vix_close = np.zeros(n, dtype=np.float64)
    for i, ts in enumerate(spx_df["timestamp"]):
        j = vix_ts_to_idx.get(str(ts))
        if j is not None:
            vix_close[i] = float(vix_df["vix_close"][j])

    day_to_indices: dict[str, list[int]] = {}
    for i, day in enumerate(dates_list):
        day_to_indices.setdefault(day, []).append(i)

    market = {
        "dates": dates_list,
        "timestamps": np.asarray(spx_df["timestamp"], dtype=np.int64),
        "spx_open": np.asarray(spx_df["spx_open"], dtype=np.float64),
        "spx_high": np.asarray(spx_df["spx_high"], dtype=np.float64),
        "spx_low": np.asarray(spx_df["spx_low"], dtype=np.float64),
        "spx_close": np.asarray(spx_df["spx_close"], dtype=np.float64),
        "spy_volume": spy_volume,
        "spy_close": spy_close,
        "vix_close": vix_close,
    }
    return raw, market, {k: np.asarray(v, dtype=np.int32) for k, v in day_to_indices.items()}


def build_market_state_cache(
    *,
    data_path: str = "v2/data.pt",
    output_path: str = DEFAULT_MARKET_STATE_PATH,
    force: bool = False,
) -> str:
    raw, market, day_to_indices = _aligned_market_arrays(data_path)
    dataset_fingerprint = str(raw.get("metadata", {}).get("fingerprint", "unknown"))
    if os.path.exists(output_path) and not force:
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        if (
            existing.get("version") == MARKET_STATE_VERSION
            and existing.get("dataset_fingerprint") == dataset_fingerprint
        ):
            return output_path

    train_mask = np.asarray(raw["train_mask"].numpy(), dtype=bool)
    train_days = {str(raw["dates"][i]) for i, flag in enumerate(train_mask) if flag}

    unique_days = sorted(day_to_indices)
    day_option_volume: dict[str, np.ndarray] = {}
    day_five_minute_raw: dict[str, np.ndarray] = {}
    raw_chunks: list[np.ndarray] = []

    for day in unique_days:
        if len(day_option_volume) % 50 == 0:
            print(f"[market-state] processing day {len(day_option_volume) + 1}/{len(unique_days)}: {day}", flush=True)
        idxs = day_to_indices[day]
        cache_path = os.path.join(FULL_CHAIN_CACHE_DIR, f"{day}.pkl")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"missing full-chain cache for {day}: {cache_path}; "
                "build the raw caches before creating v3 market state"
            )
        with open(cache_path, "rb") as f:
            full_day = pickle.load(f)
        day_bars = full_day.get("bars", {})
        day_timestamps = market["timestamps"][idxs]
        raw_bars = [chain_bar_for_timestamp(day_bars, int(ts)) for ts in day_timestamps]
        if len(raw_bars) != len(idxs):
            raise ValueError(f"unexpected raw bar count for {day}")

        option_volume = _option_volume_per_bar(raw_bars).astype(np.float32)
        day_option_volume[day] = option_volume
        f5_raw = _compute_five_minute_raw_features(
            day_open=market["spx_open"][idxs],
            day_high=market["spx_high"][idxs],
            day_low=market["spx_low"][idxs],
            day_close=market["spx_close"][idxs],
            day_spy_volume=market["spy_volume"][idxs],
            raw_bars=raw_bars,
        )
        day_five_minute_raw[day] = f5_raw
        raw_chunks.append(f5_raw)

    train_cums = [np.cumsum(day_option_volume[day], dtype=np.float64) for day in unique_days if day in train_days]
    if not train_cums:
        raise ValueError("no train days found while building v3 market state")
    max_bars = max(len(arr) for arr in train_cums)
    baseline_sum = np.zeros((max_bars,), dtype=np.float64)
    baseline_count = np.zeros((max_bars,), dtype=np.float64)
    for arr in train_cums:
        use = len(arr)
        baseline_sum[:use] += arr
        baseline_count[:use] += 1.0
    baseline_cum_volume = baseline_sum / np.clip(baseline_count, 1.0, None)

    normalized_f5 = _normalize_five_minute_features(np.concatenate(raw_chunks, axis=0))
    offset = 0
    days_payload: dict[str, dict[str, np.ndarray]] = {}
    for day in unique_days:
        idxs = day_to_indices[day]
        raw_f5 = day_five_minute_raw[day]
        count = raw_f5.shape[0]
        day_context_5m = normalized_f5[offset : offset + count]
        offset += count
        session_state = _compute_session_state(
            day_open=market["spx_open"][idxs],
            day_high=market["spx_high"][idxs],
            day_low=market["spx_low"][idxs],
            day_close=market["spx_close"][idxs],
            day_spy_volume=market["spy_volume"][idxs],
            option_volume=day_option_volume[day],
            baseline_cum_volume=baseline_cum_volume[: len(idxs)],
        )
        days_payload[day] = {
            "context_5m": day_context_5m.astype(np.float32),
            "session_state": session_state.astype(np.float32),
        }

    payload = {
        "version": MARKET_STATE_VERSION,
        "created_at": _utc_now_iso(),
        "dataset_fingerprint": dataset_fingerprint,
        "data_path": data_path,
        "five_minute_feature_names": list(FIVE_MINUTE_FEATURE_NAMES),
        "session_feature_names": list(SESSION_STATE_FEATURE_NAMES),
        "days": days_payload,
    }
    _ensure_parent(output_path)
    torch.save(payload, output_path)
    print(f"[market-state] wrote {output_path}", flush=True)
    return output_path


def load_market_state_cache(
    *,
    data_path: str = "v2/data.pt",
    cache_path: str = DEFAULT_MARKET_STATE_PATH,
) -> dict[str, Any]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"missing v3 market-state cache: {cache_path}. "
            f"Build it with `python3 -m v3.build_market_state --data {data_path}`"
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    dataset_fingerprint = str(raw.get("metadata", {}).get("fingerprint", "unknown"))
    if payload.get("version") != MARKET_STATE_VERSION:
        raise ValueError(f"unexpected market-state version: {payload.get('version')}")
    if payload.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError(
            f"market-state cache fingerprint {payload.get('dataset_fingerprint')} "
            f"does not match dataset fingerprint {dataset_fingerprint}"
        )
    return payload
