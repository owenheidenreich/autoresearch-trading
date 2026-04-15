"""Build the v4 exact-chain dataset manifest plus per-day chain sidecars."""
from __future__ import annotations

import argparse
import functools
import math
import multiprocessing
import os
import pickle
import subprocess
import time
import tempfile
from collections import defaultdict

import numpy as np
import torch

from v2.core.chain_data import (
    CHAIN_SCHEMA_VERSION,
    CONTRACT_FEATURE_FIELDS,
    QUALITY_VALID,
    QUALITY_PARTIAL,
    QUALITY_CORRUPT,
    build_contract_row,
    ensure_sidecar_dir,
    file_sha256,
    manifest_sidecar_digest,
    quality_from_bar,
    sidecar_path,
    spread_fraction_proxy,
    to_wide_bar,
)
from v2.core.dataset_fingerprint import compute_dataset_fingerprint
from v2.core.policy import DEFAULT_POLICY, SHORT_POLICY, EOD_POLICY
from v2.core.simulator import simulate_trade
from v2.core.schema import TradeIntent
from v2.core.features import (
    FEATURE_NAMES,
    NO_TRADE_BEFORE_BAR,
    normalize_features,
    NUM_FEATURES,
    _FEAT_IDX,
)
from v2.pipeline.compute_features import (
    ALL_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    OPTION_FEATURE_NAMES,
    PRICE_FEATURE_NAMES,
    _bs_greeks,
    _bs_iv,
    bs_greeks_vec,
    bs_iv_vec,
    compute_flow_features,
    compute_option_features,
    compute_price_features,
)


DATA_DIR = os.path.expanduser("~/.cache/autoresearch-trading/data")
FULL_CHAIN_CACHE_DIR = os.path.join(DATA_DIR, "spxw_full_chain")
SPX_PATH = os.path.join(DATA_DIR, "spx_1min.pkl")
SPY_PATH = os.path.join(DATA_DIR, "spy_1min.pkl")
VIX_PATH = os.path.join(DATA_DIR, "vix_1min.pkl")

OUTPUT_PATH = "v2/data.pt"
SIDECAR_DIR = os.path.join("v2", "data_sidecars")

VAL_DAYS = 60
PROMOTE_DAYS = 60
SHADOW_DAYS = 20
BARS_PER_DAY = 390


def _get_config_fingerprint() -> str:
    """Get RuntimeConfig fingerprint for dataset provenance."""
    try:
        from v2.core.config import RUNTIME_CONFIG
        return RUNTIME_CONFIG.fingerprint()
    except Exception:
        return "unknown"


def _get_git_sha() -> str:
    """Get current git SHA for dataset provenance."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _fallback_python() -> str | None:
    candidates = [
        os.path.join(os.getcwd(), ".venv", "bin", "python"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".venv", "bin", "python"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_market_cache(path: str, kind: str) -> dict[str, np.ndarray]:
    """Load one cached market dataframe, with a fallback through the repo venv."""

    try:
        df = pickle.load(open(path, "rb"))
        return {col: df[col].to_numpy() for col in df.columns}
    except Exception:
        py = _fallback_python()
        if py is None:
            raise
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            out_path = tmp.name
        script = r"""
import pickle, sys, numpy as np
df = pickle.load(open(sys.argv[1], 'rb'))
kind = sys.argv[2]
out = sys.argv[3]
if kind == 'spx':
    np.savez_compressed(out,
        date=np.asarray(df['date'].astype(str).to_numpy(), dtype='U16'),
        timestamp=np.asarray(df['timestamp'].astype(str).to_numpy(), dtype='U64'),
        spx_open=df['spx_open'].to_numpy(),
        spx_high=df['spx_high'].to_numpy(),
        spx_low=df['spx_low'].to_numpy(),
        spx_close=df['spx_close'].to_numpy(),
    )
elif kind == 'spy':
    np.savez_compressed(out,
        timestamp=np.asarray(df['timestamp'].astype(str).to_numpy(), dtype='U64'),
        close=df['close'].to_numpy(),
        volume=df['volume'].to_numpy(),
    )
elif kind == 'vix':
    np.savez_compressed(out,
        timestamp=np.asarray(df['timestamp'].astype(str).to_numpy(), dtype='U64'),
        vix_close=df['vix_close'].to_numpy(),
    )
else:
    raise SystemExit(f'unknown kind {kind}')
"""
        subprocess.run([py, "-c", script, path, kind, out_path], check=True)
        try:
            payload = np.load(out_path, allow_pickle=True)
            return {k: payload[k] for k in payload.files}
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


def _empty_sidecar(day: str, expiry: str, n_bars: int, bar_timestamps: np.ndarray | None = None) -> dict:
    ts = np.asarray(bar_timestamps, dtype=np.int64) if bar_timestamps is not None else np.zeros(n_bars, dtype=np.int64)
    return {
        "schema_version": CHAIN_SCHEMA_VERSION,
        "date": day,
        "expiry": expiry,
        "n_bars": n_bars,
        "bar_timestamps": ts,
        "contract_strike": np.zeros(0, dtype=np.float32),
        "contract_right": np.zeros(0, dtype=np.int8),
        "contract_mid": np.zeros((0, n_bars), dtype=np.float32),
        "contract_bid": np.zeros((0, n_bars), dtype=np.float32),
        "contract_ask": np.zeros((0, n_bars), dtype=np.float32),
        "contract_quality": np.zeros((0, n_bars), dtype=np.int8),
        "row_features": np.zeros((0, len(CONTRACT_FEATURE_FIELDS)), dtype=np.float32),
        "row_labels": np.zeros((0,), dtype=np.float32),
        "row_contract_idx": np.zeros((0,), dtype=np.int32),
        "bar_ptrs": np.zeros(n_bars + 1, dtype=np.int32),
        "bar_best_contract_idx": np.full(n_bars, -1, dtype=np.int32),
        "bar_best_pnl": np.zeros(n_bars, dtype=np.float32),
        "bar_label_trade": np.zeros(n_bars, dtype=bool),
        "bar_labelable": np.zeros(n_bars, dtype=bool),
        "bar_quality": np.full(n_bars, QUALITY_CORRUPT, dtype=np.int8),
        # Path library fields
        "row_raw_returns": np.zeros((0, N_HORIZONS), dtype=np.float32),
        "row_mfe": np.zeros((0, N_HORIZONS), dtype=np.float32),
        "row_mae": np.zeros((0, N_HORIZONS), dtype=np.float32),
        "row_bars_to_breakeven": np.zeros((0,), dtype=np.float32),
        "row_impulse_fraction": np.zeros((0,), dtype=np.float32),
        "row_labels_short": np.zeros((0,), dtype=np.float32),
        "row_labels_eod": np.zeros((0,), dtype=np.float32),
    }


def _build_masks(dates_list: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    unique_dates = sorted(set(dates_list))
    n_days = len(unique_dates)
    eval_days = VAL_DAYS + PROMOTE_DAYS + SHADOW_DAYS
    train_end_day_idx = n_days - eval_days

    train_dates = set(unique_dates[:train_end_day_idx])
    val_dates = set(unique_dates[train_end_day_idx:train_end_day_idx + VAL_DAYS])
    promote_start = train_end_day_idx + VAL_DAYS
    promote_dates = set(unique_dates[promote_start:promote_start + PROMOTE_DAYS])
    shadow_start = promote_start + PROMOTE_DAYS
    shadow_dates = set(unique_dates[shadow_start:])

    train_mask = np.array([d in train_dates for d in dates_list], dtype=bool)
    val_mask = np.array([d in val_dates for d in dates_list], dtype=bool)
    promote_mask = np.array([d in promote_dates for d in dates_list], dtype=bool)
    shadow_mask = np.array([d in shadow_dates for d in dates_list], dtype=bool)
    split_info = {
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "promote_days": len(promote_dates),
        "shadow_days": len(shadow_dates),
    }
    return train_mask, val_mask, promote_mask, shadow_mask, split_info


def _fill_chain_matrices(
    contracts: list[tuple[float, str]],
    bar_contracts_list: list[dict[tuple[float, str], dict[str, float]]],
    spot_series: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    n_contracts = len(contracts)
    n_bars = len(bar_contracts_list)
    contract_to_idx = {c: i for i, c in enumerate(contracts)}
    contract_strikes = np.asarray([c[0] for c in contracts], dtype=np.float64)
    contract_is_call = np.asarray([c[1] == "C" for c in contracts], dtype=bool)

    mats = {
        "mid": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "bid": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "ask": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "quality": np.full((n_contracts, n_bars), QUALITY_CORRUPT, dtype=np.int8),
        "iv": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "delta": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "gamma": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "theta": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "vega": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "charm": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "volume": np.zeros((n_contracts, n_bars), dtype=np.float32),
        "transactions": np.zeros((n_contracts, n_bars), dtype=np.float32),
        "spread": np.ones((n_contracts, n_bars), dtype=np.float32),
    }
    bar_quality: list[dict] = []

    for local_i, bar_contracts in enumerate(bar_contracts_list):
        spot = float(spot_series[local_i])
        mtc = max(BARS_PER_DAY - local_i, 1)
        year_frac = mtc / (252.0 * BARS_PER_DAY)

        # Phase 1: populate price/volume/quality/spread (still per-contract,
        # but these are cheap field lookups -- not the bottleneck)
        seen_idxs: list[int] = []
        for contract, fields in bar_contracts.items():
            idx = contract_to_idx.get(contract)
            if idx is None:
                continue
            close_px = float(fields.get("close", np.nan) or np.nan)
            high_px = float(fields.get("high", np.nan) or np.nan)
            low_px = float(fields.get("low", np.nan) or np.nan)
            volume = float(fields.get("volume", 0.0) or 0.0)
            transactions = float(fields.get("transactions", 0.0) or 0.0)
            quality = quality_from_bar(close_px, volume, transactions)
            spread = spread_fraction_proxy(close_px, high_px, low_px)
            bid = close_px * (1.0 - spread / 2.0) if np.isfinite(close_px) and close_px > 0 else np.nan
            ask = close_px * (1.0 + spread / 2.0) if np.isfinite(close_px) and close_px > 0 else np.nan
            mats["mid"][idx, local_i] = close_px
            mats["bid"][idx, local_i] = bid
            mats["ask"][idx, local_i] = ask
            mats["quality"][idx, local_i] = quality
            mats["volume"][idx, local_i] = volume
            mats["transactions"][idx, local_i] = transactions
            mats["spread"][idx, local_i] = spread
            seen_idxs.append(idx)

        # Phase 2: vectorized IV + Greeks for all seen contracts at this bar
        if seen_idxs:
            idxs = np.asarray(seen_idxs, dtype=np.intp)
            prices = mats["mid"][idxs, local_i].astype(np.float64)
            strikes = contract_strikes[idxs]
            is_call = contract_is_call[idxs]
            priceable = np.isfinite(prices) & (prices > 0)

            if priceable.any():
                p_idx = idxs[priceable]
                p_prices = prices[priceable]
                p_strikes = strikes[priceable]
                p_is_call = is_call[priceable]
                p_spot = np.full(len(p_prices), spot, dtype=np.float64)
                p_T = np.full(len(p_prices), year_frac, dtype=np.float64)

                iv_arr = bs_iv_vec(p_prices, p_spot, p_strikes, p_T, 0.05, p_is_call)
                mats["iv"][p_idx, local_i] = iv_arr.astype(np.float32)

                iv_ok = np.isfinite(iv_arr)
                if iv_ok.any():
                    g_idx = p_idx[iv_ok]
                    g_spot = p_spot[iv_ok]
                    g_strikes = p_strikes[iv_ok]
                    g_T = p_T[iv_ok]
                    g_iv = iv_arr[iv_ok]
                    g_is_call = p_is_call[iv_ok]
                    delta, gamma, theta, vega, charm = bs_greeks_vec(g_spot, g_strikes, g_T, 0.05, g_iv, g_is_call)
                    ok = np.isfinite(delta)
                    mats["delta"][g_idx[ok], local_i] = delta[ok].astype(np.float32)
                    mats["gamma"][g_idx[ok], local_i] = gamma[ok].astype(np.float32)
                    mats["theta"][g_idx[ok], local_i] = theta[ok].astype(np.float32)
                    mats["vega"][g_idx[ok], local_i] = vega[ok].astype(np.float32)
                    mats["charm"][g_idx[ok], local_i] = charm[ok].astype(np.float32)

        labeled = int(np.isfinite(mats["iv"][:, local_i]).sum()) if seen_idxs else 0
        bar_quality.append({"observed_contracts": len(seen_idxs), "greeked_contracts": labeled})
    return mats, bar_quality


RAW_RETURN_HORIZONS = [5, 10, 15, 30, 60]
N_HORIZONS = len(RAW_RETURN_HORIZONS)


def _compute_path_metrics(
    series: np.ndarray,
    entry_bar: int,
    entry_mid: float,
    spread_cost_est: float,
    n_bars: int,
) -> dict:
    """Compute raw forward returns, MFE, MAE at each horizon.

    Returns dict with:
      raw_returns: (N_HORIZONS,) raw price returns at [5,10,15,30,60] bars
      mfe: (N_HORIZONS,) max favorable excursion at each horizon
      mae: (N_HORIZONS,) max adverse excursion at each horizon
      bars_to_breakeven: first bar where unrealized >= spread cost (or NaN)
      impulse_fraction: fraction of 30-bar terminal PnL achieved in first 5 bars
    """
    fill_bar = entry_bar + 1
    raw_returns = np.full(N_HORIZONS, np.nan, dtype=np.float32)
    mfe = np.full(N_HORIZONS, np.nan, dtype=np.float32)
    mae = np.full(N_HORIZONS, np.nan, dtype=np.float32)
    bars_to_breakeven = np.float32(np.nan)
    impulse_fraction = np.float32(np.nan)

    if fill_bar >= n_bars or not np.isfinite(series[fill_bar]) or series[fill_bar] <= 0:
        return {
            "raw_returns": raw_returns, "mfe": mfe, "mae": mae,
            "bars_to_breakeven": bars_to_breakeven,
            "impulse_fraction": impulse_fraction,
        }

    fill_px = float(series[fill_bar])

    # Walk forward computing unrealized returns bar by bar
    max_horizon = max(RAW_RETURN_HORIZONS)
    end_bar = min(n_bars, fill_bar + max_horizon + 1)
    path_slice = series[fill_bar:end_bar]

    if len(path_slice) == 0:
        return {
            "raw_returns": raw_returns, "mfe": mfe, "mae": mae,
            "bars_to_breakeven": bars_to_breakeven,
            "impulse_fraction": impulse_fraction,
        }

    # Unrealized returns from fill price
    valid = np.isfinite(path_slice) & (path_slice > 0)
    unrealized = np.where(valid, (path_slice - fill_px) / fill_px, np.nan)

    # Raw returns at each horizon
    for hi, h in enumerate(RAW_RETURN_HORIZONS):
        if h < len(unrealized) and np.isfinite(unrealized[h]):
            raw_returns[hi] = unrealized[h]

    # MFE and MAE at each horizon (cumulative max/min up to that bar)
    running_max = np.float64(-np.inf)
    running_min = np.float64(np.inf)
    horizon_idx = 0
    found_breakeven = False
    for bar_offset in range(len(unrealized)):
        u = unrealized[bar_offset]
        if not np.isfinite(u):
            continue
        if u > running_max:
            running_max = u
        if u < running_min:
            running_min = u
        if not found_breakeven and u >= spread_cost_est:
            bars_to_breakeven = np.float32(bar_offset)
            found_breakeven = True
        # Check if we've reached the next horizon boundary
        while horizon_idx < N_HORIZONS and bar_offset >= RAW_RETURN_HORIZONS[horizon_idx]:
            mfe[horizon_idx] = np.float32(max(running_max, 0.0))
            mae[horizon_idx] = np.float32(min(running_min, 0.0))
            horizon_idx += 1
    # Fill remaining horizons that we reached
    while horizon_idx < N_HORIZONS:
        mfe[horizon_idx] = np.float32(max(running_max, 0.0))
        mae[horizon_idx] = np.float32(min(running_min, 0.0))
        horizon_idx += 1

    # Impulse fraction: how much of 30-bar terminal PnL was in first 5 bars
    ret_5_idx = RAW_RETURN_HORIZONS.index(5)
    ret_30_idx = RAW_RETURN_HORIZONS.index(30)
    if np.isfinite(raw_returns[ret_30_idx]) and abs(raw_returns[ret_30_idx]) > 1e-6:
        if np.isfinite(raw_returns[ret_5_idx]):
            impulse_fraction = np.float32(
                raw_returns[ret_5_idx] / raw_returns[ret_30_idx]
            )

    return {
        "raw_returns": raw_returns, "mfe": mfe, "mae": mae,
        "bars_to_breakeven": bars_to_breakeven,
        "impulse_fraction": impulse_fraction,
    }


def _simulate_under_policy(
    policy,
    expiry: str,
    contract_strike: float,
    right: str,
    mid_now: float,
    series: np.ndarray,
    feature_context: np.ndarray,
    n_bars: int,
    local_i: int,
    day: str,
    timestamps: np.ndarray,
) -> float:
    """Simulate a trade under a specific policy, return net_pnl_pct or NaN."""
    max_hold = min(policy.max_hold_bars, n_bars - local_i - 1)
    if max_hold < 2:
        return np.nan
    intent = TradeIntent(
        trade=True,
        expiry=expiry,
        strike=contract_strike,
        right=right,
        qty=policy.qty,
        entry_ref_price=mid_now,
        order_style=policy.order_style,
        tif=policy.tif,
        stop_price=mid_now * (1.0 - policy.stop_pct),
        take_profit_price=mid_now * (1.0 + policy.target_pct),
        max_hold_bars=max_hold,
        exit_policy=policy.exit_policy,
        confidence=0.5,
        reason_codes=("dataset_label",),
        bar_index=local_i,
        timestamp=str(int(timestamps[local_i])) if len(timestamps) else "",
        underlying_price=0.0,
    )
    trade = simulate_trade(
        intent=intent,
        option_prices=series.astype(np.float32),
        features=feature_context,
        bar_of_day=np.arange(n_bars, dtype=np.int32),
        dates=[day] * n_bars,
        global_entry_bar=local_i,
        breakeven_trigger_pct=policy.breakeven_trigger_pct if policy.breakeven_trigger_pct > 0 else None,
        extra_trailing_tiers=policy.extra_trailing_tiers,
    )
    if trade is None:
        return np.nan
    return float(trade.net_pnl_pct)


def _forward_path_complete(series: np.ndarray, start_bar: int, max_hold: int, n_bars: int) -> bool:
    fill_bar = start_bar + 1
    if fill_bar >= n_bars:
        return False
    if not np.isfinite(series[fill_bar]) or series[fill_bar] <= 0:
        return False
    end = min(n_bars, fill_bar + max_hold + 1)
    return bool(np.isfinite(series[fill_bar:end]).all())


def _chain_bar_for_timestamp(
    day_bars: dict,
    ts_ms: int,
) -> dict[tuple[float, str], dict[str, float]]:
    if ts_ms in day_bars:
        return day_bars[ts_ms]
    ts_ns = int(ts_ms) * 1_000_000
    if ts_ns in day_bars:
        return day_bars[ts_ns]
    ts_str = str(int(ts_ms))
    if ts_str in day_bars:
        return day_bars[ts_str]
    ts_ns_str = str(ts_ns)
    if ts_ns_str in day_bars:
        return day_bars[ts_ns_str]
    return {}


def _label_day_sidecar(
    *,
    day: str,
    expiry: str,
    global_indices: list[int],
    timestamps: np.ndarray,
    spot_series: np.ndarray,
    feature_context: np.ndarray,
    raw_full_bar_list: list[dict[tuple[float, str], dict[str, float]]],
    chain_mats: dict[str, np.ndarray],
) -> dict:
    n_bars = len(global_indices)
    n_contracts = chain_mats["mid"].shape[0]
    if n_contracts == 0:
        return _empty_sidecar(day, expiry, n_bars)

    row_features: list[np.ndarray] = []
    row_labels: list[float] = []
    row_labels_short: list[float] = []
    row_labels_eod: list[float] = []
    row_raw_returns: list[np.ndarray] = []
    row_mfe: list[np.ndarray] = []
    row_mae: list[np.ndarray] = []
    row_bars_to_breakeven: list[float] = []
    row_impulse_fraction: list[float] = []
    row_contract_idx: list[int] = []
    bar_ptrs = [0]
    bar_best_contract_idx = np.full(n_bars, -1, dtype=np.int32)
    bar_best_pnl = np.zeros(n_bars, dtype=np.float32)
    bar_label_trade = np.zeros(n_bars, dtype=bool)
    bar_labelable = np.zeros(n_bars, dtype=bool)
    bar_quality = np.full(n_bars, QUALITY_CORRUPT, dtype=np.int8)
    contract_strike = chain_mats["contract_strike"]
    contract_right = chain_mats["contract_right"]

    for local_i in range(n_bars):
        spot = float(spot_series[local_i])
        bod = local_i
        if bod < DEFAULT_POLICY.no_trade_before_bar or bod >= DEFAULT_POLICY.no_trade_after_bar:
            bar_ptrs.append(len(row_features))
            continue

        best_local = -1
        best_pnl = -float("inf")
        any_visible = False
        any_labelable = False

        for contract_idx in range(n_contracts):
            mid_now = float(chain_mats["mid"][contract_idx, local_i])
            if not np.isfinite(mid_now) or mid_now < DEFAULT_POLICY.min_contract_mid:
                continue
            spread_frac = float(chain_mats["spread"][contract_idx, local_i])
            if spread_frac > DEFAULT_POLICY.max_spread_fraction:
                continue
            volume = float(chain_mats["volume"][contract_idx, local_i])
            transactions = float(chain_mats["transactions"][contract_idx, local_i])
            if DEFAULT_POLICY.require_volume_or_transactions and volume <= 0 and transactions <= 0:
                continue
            any_visible = True
            quality = int(chain_mats["quality"][contract_idx, local_i])

            # Contract price momentum: % change over last 5 and 10 bars
            mid_chg_5 = 0.0
            mid_chg_10 = 0.0
            mid_series = chain_mats["mid"][contract_idx]
            if local_i >= 5:
                prev5 = float(mid_series[local_i - 5])
                if np.isfinite(prev5) and prev5 > 0:
                    mid_chg_5 = (mid_now - prev5) / prev5
            if local_i >= 10:
                prev10 = float(mid_series[local_i - 10])
                if np.isfinite(prev10) and prev10 > 0:
                    mid_chg_10 = (mid_now - prev10) / prev10

            row = build_contract_row(
                strike=float(contract_strike[contract_idx]),
                right="P" if int(contract_right[contract_idx]) == 1 else "C",
                mid=mid_now,
                spread_frac=spread_frac,
                volume=volume,
                transactions=transactions,
                iv=float(chain_mats["iv"][contract_idx, local_i]),
                delta=float(chain_mats["delta"][contract_idx, local_i]),
                gamma=float(chain_mats["gamma"][contract_idx, local_i]),
                theta=float(chain_mats["theta"][contract_idx, local_i]),
                spot=spot,
                minutes_to_close=max(BARS_PER_DAY - bod, 1),
                quality=quality,
                is_executable=True,
                vega=float(chain_mats["vega"][contract_idx, local_i]),
                charm=float(chain_mats["charm"][contract_idx, local_i]),
                mid_chg_5=mid_chg_5,
                mid_chg_10=mid_chg_10,
            )

            row_features.append(row)
            row_labels.append(np.nan)
            row_labels_short.append(np.nan)
            row_labels_eod.append(np.nan)
            row_contract_idx.append(contract_idx)

            series = chain_mats["mid"][contract_idx]

            # --- Path library: raw returns + MFE/MAE (policy-free) ---
            spread_est = float(spread_frac) * 0.5  # conservative half-spread
            path_m = _compute_path_metrics(
                series, local_i, mid_now, spread_est, n_bars,
            )
            row_raw_returns.append(path_m["raw_returns"])
            row_mfe.append(path_m["mfe"])
            row_mae.append(path_m["mae"])
            row_bars_to_breakeven.append(float(path_m["bars_to_breakeven"]))
            row_impulse_fraction.append(float(path_m["impulse_fraction"]))

            if not _forward_path_complete(series, local_i, DEFAULT_POLICY.max_hold_bars, n_bars):
                continue

            right = "P" if int(contract_right[contract_idx]) == 1 else "C"
            c_strike = float(contract_strike[contract_idx])

            # --- Long policy (current default) ---
            pnl_long = _simulate_under_policy(
                DEFAULT_POLICY, expiry, c_strike, right, mid_now,
                series, feature_context, n_bars, local_i, day, timestamps,
            )
            if not np.isfinite(pnl_long):
                continue

            any_labelable = True
            row_labels[-1] = pnl_long

            # --- Short policy overlay ---
            pnl_short = _simulate_under_policy(
                SHORT_POLICY, expiry, c_strike, right, mid_now,
                series, feature_context, n_bars, local_i, day, timestamps,
            )
            row_labels_short[-1] = pnl_short

            # --- EOD policy overlay ---
            pnl_eod = _simulate_under_policy(
                EOD_POLICY, expiry, c_strike, right, mid_now,
                series, feature_context, n_bars, local_i, day, timestamps,
            )
            row_labels_eod[-1] = pnl_eod

            if pnl_long > best_pnl:
                best_pnl = pnl_long
                best_local = len(row_labels) - bar_ptrs[-1] - 1

        if any_labelable:
            bar_quality[local_i] = QUALITY_VALID
            bar_labelable[local_i] = True
        elif any_visible:
            bar_quality[local_i] = QUALITY_PARTIAL
        bar_ptrs.append(len(row_features))
        if best_local >= 0:
            bar_best_contract_idx[local_i] = best_local
            bar_best_pnl[local_i] = float(best_pnl)
            if best_pnl > DEFAULT_POLICY.label_gate_min_pnl:
                bar_label_trade[local_i] = True

    return {
        "schema_version": CHAIN_SCHEMA_VERSION,
        "date": day,
        "expiry": expiry,
        "n_bars": n_bars,
        "bar_timestamps": timestamps.astype(np.int64),
        "contract_strike": contract_strike.astype(np.float32),
        "contract_right": contract_right.astype(np.int8),
        "contract_mid": chain_mats["mid"].astype(np.float32),
        "contract_bid": chain_mats["bid"].astype(np.float32),
        "contract_ask": chain_mats["ask"].astype(np.float32),
        "contract_quality": chain_mats["quality"].astype(np.int8),
        "row_features": np.asarray(row_features, dtype=np.float32),
        "row_labels": np.asarray(row_labels, dtype=np.float32),
        "row_contract_idx": np.asarray(row_contract_idx, dtype=np.int32),
        "bar_ptrs": np.asarray(bar_ptrs, dtype=np.int32),
        "bar_best_contract_idx": bar_best_contract_idx,
        "bar_best_pnl": bar_best_pnl.astype(np.float32),
        "bar_label_trade": bar_label_trade,
        "bar_labelable": bar_labelable,
        "bar_quality": bar_quality,
        # Path library: environment truth + policy overlays
        "row_raw_returns": np.asarray(row_raw_returns, dtype=np.float32).reshape(-1, N_HORIZONS) if row_raw_returns else np.zeros((0, N_HORIZONS), dtype=np.float32),
        "row_mfe": np.asarray(row_mfe, dtype=np.float32).reshape(-1, N_HORIZONS) if row_mfe else np.zeros((0, N_HORIZONS), dtype=np.float32),
        "row_mae": np.asarray(row_mae, dtype=np.float32).reshape(-1, N_HORIZONS) if row_mae else np.zeros((0, N_HORIZONS), dtype=np.float32),
        "row_bars_to_breakeven": np.asarray(row_bars_to_breakeven, dtype=np.float32),
        "row_impulse_fraction": np.asarray(row_impulse_fraction, dtype=np.float32),
        "row_labels_short": np.asarray(row_labels_short, dtype=np.float32),
        "row_labels_eod": np.asarray(row_labels_eod, dtype=np.float32),
    }


def _process_one_day(args: dict) -> dict:
    """Worker: process a single day's chain data into a sidecar + features.

    Runs in a child process.  All inputs are passed explicitly so that
    nothing depends on mutable parent-process state.
    """
    day = args["day"]
    global_indices = args["global_indices"]
    day_spot = args["day_spot"]
    day_X_price = args["day_X_price"]
    day_timestamps = args["day_timestamps"]
    sd = args["sidecar_dir"]

    n_bars_day = len(global_indices)
    n_opt = len(OPTION_FEATURE_NAMES)
    n_flow = len(FLOW_FEATURE_NAMES)
    X_opt_day = np.full((n_bars_day, n_opt), np.nan, dtype=np.float64)
    X_flow_day = np.zeros((n_bars_day, n_flow), dtype=np.float64)

    cache_path = os.path.join(FULL_CHAIN_CACHE_DIR, f"{day}.pkl")
    expiry = day.replace("-", "")
    day_ts_ms = np.asarray(day_timestamps, dtype=np.int64)

    if os.path.exists(cache_path):
        full_day = pickle.load(open(cache_path, "rb"))
        raw_day_bars = full_day.get("bars", {})
        raw_bars = [_chain_bar_for_timestamp(raw_day_bars, int(ts)) for ts in day_ts_ms]
        contracts = sorted(full_day.get("contracts", []))
    else:
        raw_bars = [{} for _ in range(n_bars_day)]
        contracts = []

    if contracts:
        chain_mats, _ = _fill_chain_matrices(contracts, raw_bars, day_spot)
        chain_mats["contract_strike"] = np.asarray([c[0] for c in contracts], dtype=np.float32)
        chain_mats["contract_right"] = np.asarray([1 if c[1] == "P" else 0 for c in contracts], dtype=np.int8)
    else:
        chain_mats = {k: np.zeros((0, n_bars_day), dtype=np.float32) for k in
                      ("mid", "bid", "ask", "quality", "iv", "delta", "gamma",
                       "theta", "vega", "charm", "volume", "transactions", "spread")}
        chain_mats["contract_strike"] = np.zeros(0, dtype=np.float32)
        chain_mats["contract_right"] = np.zeros(0, dtype=np.int8)

    atm_strike_open = round(float(day_spot[0]) / 5.0) * 5.0
    iv_history: list[float] = []
    for local_i in range(n_bars_day):
        mtc = max(BARS_PER_DAY - local_i, 1)
        wide_bar = to_wide_bar(raw_bars[local_i])
        opt_feats = compute_option_features(
            wide_bar, float(day_spot[local_i]), atm_strike_open, mtc,
            iv_history=iv_history[-390 * 60:] if iv_history else None,
        )
        for j, name in enumerate(OPTION_FEATURE_NAMES):
            if name in opt_feats:
                X_opt_day[local_i, j] = opt_feats[name]
        flow_feats = compute_flow_features(wide_bar, float(day_spot[local_i]))
        for j, name in enumerate(FLOW_FEATURE_NAMES):
            if name in flow_feats:
                X_flow_day[local_i, j] = flow_feats[name]
        atm_iv_val = opt_feats.get("atm_iv", np.nan)
        if np.isfinite(atm_iv_val):
            iv_history.append(atm_iv_val)

    vrp_idx = OPTION_FEATURE_NAMES.index("vrp")
    rv_idx = PRICE_FEATURE_NAMES.index("realized_vol")
    iv_idx = OPTION_FEATURE_NAMES.index("atm_iv")
    for local_i in range(n_bars_day):
        atm_iv_val = X_opt_day[local_i, iv_idx]
        rv_val = day_X_price[local_i, rv_idx]
        if np.isfinite(atm_iv_val) and rv_val > 0:
            X_opt_day[local_i, vrp_idx] = atm_iv_val ** 2 - rv_val ** 2

    # NaN audit: count per-feature NaNs before zeroing (observability)
    nan_counts_opt = np.isnan(X_opt_day).sum(axis=0).tolist()  # per option feature
    X_opt_day = np.nan_to_num(X_opt_day, nan=0.0)

    X_day_context = np.concatenate(
        [day_X_price, X_opt_day, X_flow_day], axis=1,
    ).astype(np.float32)

    sc = _label_day_sidecar(
        day=day, expiry=expiry, global_indices=global_indices,
        timestamps=day_ts_ms, spot_series=day_spot,
        feature_context=X_day_context, raw_full_bar_list=raw_bars,
        chain_mats=chain_mats,
    )

    path = sidecar_path(sd, day)
    torch.save(sc, path)

    max_c = 0
    if len(sc["bar_ptrs"]) > 1:
        day_counts = np.diff(sc["bar_ptrs"])
        if len(day_counts):
            max_c = int(day_counts.max())

    signal_bars = 0
    trade_bars = 0
    bar_info: list[dict] = []
    for local_i in range(n_bars_day):
        gi = global_indices[local_i]
        info: dict = {
            "gi": gi,
            "best_pnl": float(sc["bar_best_pnl"][local_i]),
            "label_trade": bool(sc["bar_label_trade"][local_i]),
            "labelable": bool(sc["bar_labelable"][local_i]),
            "best_strike": 0.0,
            "best_right": -1,
            "is_trade": False,
        }
        if sc["bar_best_contract_idx"][local_i] >= 0:
            start = int(sc["bar_ptrs"][local_i])
            row_local = int(sc["bar_best_contract_idx"][local_i])
            cidx = int(sc["row_contract_idx"][start + row_local])
            info["best_strike"] = float(sc["contract_strike"][cidx])
            info["best_right"] = int(sc["contract_right"][cidx])
            info["is_trade"] = True
            trade_bars += 1
        signal_bars += 1
        bar_info.append(info)

    return {
        "day": day,
        "global_indices": global_indices,
        "X_opt_day": X_opt_day,
        "X_flow_day": X_flow_day,
        "sidecar_path": path,
        "max_contracts": max_c,
        "signal_bars": signal_bars,
        "trade_bars": trade_bars,
        "bar_info": bar_info,
        "nan_counts_opt": nan_counts_opt,
    }


def build_dataset(output_path: str = OUTPUT_PATH, sidecar_dir: str = SIDECAR_DIR) -> None:
    global print
    _orig_print = print
    print = functools.partial(_orig_print, flush=True)

    t0 = time.time()
    ensure_sidecar_dir(sidecar_dir)

    print("Loading raw market data...")
    spx_df = _load_market_cache(SPX_PATH, "spx")
    spy_df = _load_market_cache(SPY_PATH, "spy")
    vix_df = _load_market_cache(VIX_PATH, "vix")

    full_chain_dates = set(f.replace(".pkl", "") for f in os.listdir(FULL_CHAIN_CACHE_DIR) if f.endswith(".pkl"))
    spx_mask = np.array([d in full_chain_dates for d in spx_df["date"]], dtype=bool)
    spx_df = {k: np.asarray(v)[spx_mask] for k, v in spx_df.items()}
    dates_list = [str(d) for d in spx_df["date"]]
    N = len(dates_list)
    print(f"  SPX bars: {N:,} across {len(set(dates_list))} days")

    spx_close = np.asarray(spx_df["spx_close"], dtype=np.float64)
    spx_high = np.asarray(spx_df["spx_high"], dtype=np.float64)
    spx_low = np.asarray(spx_df["spx_low"], dtype=np.float64)
    spx_open = np.asarray(spx_df["spx_open"], dtype=np.float64)

    spy_ts_to_idx = {str(ts): i for i, ts in enumerate(spy_df["timestamp"])}
    spy_volume = np.zeros(N, dtype=np.float64)
    spy_close = np.zeros(N, dtype=np.float64)
    for i, ts in enumerate(spx_df["timestamp"]):
        j = spy_ts_to_idx.get(str(ts))
        if j is not None:
            spy_volume[i] = float(spy_df["volume"][j])
            spy_close[i] = float(spy_df["close"][j])

    vix_ts_to_idx = {str(ts): i for i, ts in enumerate(vix_df["timestamp"])}
    vix_close = np.full(N, np.nan, dtype=np.float64)
    for i, ts in enumerate(spx_df["timestamp"]):
        j = vix_ts_to_idx.get(str(ts))
        if j is not None:
            vix_close[i] = float(vix_df["vix_close"][j])

    day_starts = [0]
    for i in range(1, N):
        if dates_list[i] != dates_list[i - 1]:
            day_starts.append(i)
    day_ends = day_starts[1:] + [N]
    bar_of_day = np.zeros(N, dtype=np.int32)
    for ds, de in zip(day_starts, day_ends):
        bar_of_day[ds:de] = np.arange(de - ds, dtype=np.int32)

    day_to_bars = defaultdict(list)
    for i, d in enumerate(dates_list):
        day_to_bars[d].append(i)
    unique_dates = sorted(day_to_bars.keys())

    print("Computing price features...")
    X_price = compute_price_features(
        spx_close, spx_high, spx_low, spx_open,
        spy_volume, spy_close, vix_close,
        day_starts, bar_of_day,
    )

    n_opt = len(OPTION_FEATURE_NAMES)
    n_flow = len(FLOW_FEATURE_NAMES)
    X_opt = np.full((N, n_opt), np.nan, dtype=np.float64)
    X_flow = np.zeros((N, n_flow), dtype=np.float64)
    spot_prices = spx_close.astype(np.float32)
    timestamps = [str(ts) for ts in spx_df["timestamp"]]

    print("Building day sidecars and option/flow features...")
    sidecar_paths: list[str] = []
    max_contracts_per_bar = 0
    total_signal_bars = 0
    total_trade_bars = 0
    # NaN audit accumulators (per option feature)
    nan_counts_opt_total = np.zeros(n_opt, dtype=np.int64)
    best_contract_pnl = np.zeros(N, dtype=np.float32)
    best_contract_strike = np.zeros(N, dtype=np.float32)
    best_contract_right = np.full(N, -1, dtype=np.int32)
    label_trade = np.zeros(N, dtype=bool)
    label_trade_valid = np.zeros(N, dtype=bool)

    # Build per-day work items
    work_items: list[dict] = []
    for day in unique_dates:
        gi = day_to_bars[day]
        work_items.append({
            "day": day,
            "global_indices": gi,
            "day_spot": spot_prices[gi],
            "day_X_price": X_price[gi],
            "day_timestamps": np.asarray(spx_df["timestamp"][gi], dtype=np.int64),
            "sidecar_dir": sidecar_dir,
        })

    n_workers = min(max(1, os.cpu_count() or 1), len(work_items))
    print(f"  Processing {len(work_items)} days with {n_workers} workers...")

    done = 0
    # Use fork context to avoid re-importing heavy modules in each worker
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        for result in pool.imap_unordered(_process_one_day, work_items, chunksize=4):
            gi = result["global_indices"]
            X_opt[gi] = result["X_opt_day"]
            X_flow[gi] = result["X_flow_day"]
            nan_counts_opt_total += np.array(result["nan_counts_opt"], dtype=np.int64)
            sidecar_paths.append(result["sidecar_path"])
            max_contracts_per_bar = max(max_contracts_per_bar, result["max_contracts"])
            total_signal_bars += result["signal_bars"]
            total_trade_bars += result["trade_bars"]
            for info in result["bar_info"]:
                g = info["gi"]
                best_contract_pnl[g] = info["best_pnl"]
                label_trade[g] = info["label_trade"]
                label_trade_valid[g] = info["labelable"]
                if info["is_trade"]:
                    best_contract_strike[g] = info["best_strike"]
                    best_contract_right[g] = info["best_right"]
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(unique_dates)} days processed")

    X_combined = np.concatenate([X_price, X_opt, X_flow], axis=1).astype(np.float32)
    assert X_combined.shape[1] == NUM_FEATURES, f"expected {NUM_FEATURES}, got {X_combined.shape[1]}"
    print("Normalizing features...")
    X_normalized = normalize_features(X_combined, np.ones(N, dtype=bool), dates=dates_list)

    train_mask, val_mask, promote_mask, shadow_mask, split_info = _build_masks(dates_list)
    sidecar_digest = manifest_sidecar_digest(sidecar_paths)
    dataset = {
        "X": torch.from_numpy(X_normalized.astype(np.float32)),
        "X_sim": torch.from_numpy(X_combined.astype(np.float32)),
        "feature_names": list(ALL_FEATURE_NAMES),
        "spot_prices": torch.from_numpy(spot_prices.astype(np.float32)),
        "label_trade": torch.from_numpy(label_trade),
        "label_trade_valid": torch.from_numpy(label_trade_valid),
        "best_contract_pnl": torch.from_numpy(best_contract_pnl.astype(np.float32)),
        "best_contract_strike": torch.from_numpy(best_contract_strike.astype(np.float32)),
        "best_contract_right": torch.from_numpy(best_contract_right.astype(np.int32)),
        "dates": dates_list,
        "bar_of_day": torch.from_numpy(bar_of_day),
        "train_mask": torch.from_numpy(train_mask),
        "val_mask": torch.from_numpy(val_mask),
        "promote_mask": torch.from_numpy(promote_mask),
        "shadow_mask": torch.from_numpy(shadow_mask),
        "metadata": {
            "version": "v4_exact_chain",
            "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fingerprint": "pending",
            "n_features": NUM_FEATURES,
            "normalization": "rolling_zscore_60day",
            "label_scheme": "exact_contract_fixed_risk",
            "label_gate_min_pnl": DEFAULT_POLICY.label_gate_min_pnl,
            "trade_window": f"bar {DEFAULT_POLICY.no_trade_before_bar}-{DEFAULT_POLICY.no_trade_after_bar}",
            "split": split_info,
            "chain_schema_version": CHAIN_SCHEMA_VERSION,
            "chain_sidecar_dir": sidecar_dir,
            "chain_sidecar_digest": sidecar_digest,
            "contract_feature_fields": CONTRACT_FEATURE_FIELDS,
            "max_contracts_per_bar": max_contracts_per_bar,
            "total_signal_bars": total_signal_bars,
            "total_trade_bars": total_trade_bars,
            "risk_policy": {
                "stop_pct": DEFAULT_POLICY.stop_pct,
                "target_pct": DEFAULT_POLICY.target_pct,
                "max_hold_bars": DEFAULT_POLICY.max_hold_bars,
                "exit_policy": DEFAULT_POLICY.exit_policy,
            },
            "execution_filters": {
                "min_contract_mid": DEFAULT_POLICY.min_contract_mid,
                "max_spread_fraction": DEFAULT_POLICY.max_spread_fraction,
                "require_volume_or_transactions": DEFAULT_POLICY.require_volume_or_transactions,
            },
            # Stage contract fields (added 2026-04-15)
            "config_fingerprint": _get_config_fingerprint(),
            "lookback": 30,  # must match LOOKBACK in train.py
            "build_git_sha": _get_git_sha(),
            "feature_names": list(ALL_FEATURE_NAMES),
        },
    }
    dataset["metadata"]["fingerprint"] = compute_dataset_fingerprint(dataset)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(dataset, output_path)
    with open(f"{output_path}.sha256", "w") as f:
        f.write(file_sha256(output_path))

    print(f"Saved {output_path}")
    print(f"  fingerprint: {dataset['metadata']['fingerprint']}")
    print(f"  sidecars: {len(sidecar_paths)} days in {sidecar_dir}")
    print(f"  max_contracts_per_bar: {max_contracts_per_bar}")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # --- Build report (observability) ---
    _save_build_report(
        dataset=dataset,
        X_combined=X_combined,
        train_mask=train_mask,
        label_trade=label_trade,
        label_trade_valid=label_trade_valid,
        nan_counts_opt_total=nan_counts_opt_total,
        feature_names=list(ALL_FEATURE_NAMES),
        option_feature_names=list(OPTION_FEATURE_NAMES),
        n_price=X_price.shape[1],
        n_opt=n_opt,
        dates_list=dates_list,
        elapsed=time.time() - t0,
    )


def _save_build_report(
    *,
    dataset: dict,
    X_combined: np.ndarray,
    train_mask: np.ndarray,
    label_trade: np.ndarray,
    label_trade_valid: np.ndarray,
    nan_counts_opt_total: np.ndarray,
    feature_names: list[str],
    option_feature_names: list[str],
    n_price: int,
    n_opt: int,
    dates_list: list[str],
    elapsed: float,
) -> None:
    """Save build_report.json: NaN audit, feature stats, label quality."""
    import json as _json
    from v2.core.observability import schema_header, identity_block

    meta = dataset["metadata"]
    dataset_fp = meta["fingerprint"]
    report_dir = os.path.join("v2", "build_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{dataset_fp}.json")

    # NaN audit: option features (price features are computed, not from raw data;
    # flow features are initialized to 0.0 so they don't have NaN)
    nan_audit = {}
    for i, name in enumerate(option_feature_names):
        nan_audit[name] = int(nan_counts_opt_total[i])
    total_opt_cells = X_combined.shape[0]  # bars * 1 (per feature)
    nan_warnings = []
    for name, count in sorted(nan_audit.items(), key=lambda x: -x[1])[:5]:
        rate = count / total_opt_cells if total_opt_cells > 0 else 0
        if rate > 0.20:
            nan_warnings.append(f"{name}: {rate:.1%} NaN ({count}/{total_opt_cells})")

    # Feature stats on train_mask (unnormalized X_combined)
    train_idx = train_mask.numpy() if hasattr(train_mask, 'numpy') else train_mask
    X_train = X_combined[train_idx.astype(bool)]
    feature_stats = {}
    for i, name in enumerate(feature_names):
        col = X_train[:, i]
        feature_stats[name] = {
            "mean": float(np.nanmean(col)),
            "std": float(np.nanstd(col)),
            "min": float(np.nanmin(col)) if len(col) > 0 else 0.0,
            "max": float(np.nanmax(col)) if len(col) > 0 else 0.0,
            "pct_zero": float((col == 0).mean()) if len(col) > 0 else 0.0,
        }

    # Label quality
    n_bars = len(label_trade)
    n_labelable = int(label_trade_valid.sum())
    n_trade = int(label_trade.sum())
    label_quality = {
        "total_bars": n_bars,
        "labelable_bars": n_labelable,
        "labelable_rate": n_labelable / n_bars if n_bars > 0 else 0,
        "trade_bars": n_trade,
        "trade_rate": n_trade / n_bars if n_bars > 0 else 0,
    }
    label_warnings = []
    if label_quality["trade_rate"] < 0.15:
        label_warnings.append(f"Low trade rate: {label_quality['trade_rate']:.1%}")
    if label_quality["trade_rate"] > 0.60:
        label_warnings.append(f"High trade rate: {label_quality['trade_rate']:.1%}")

    # Day coverage
    unique_days = sorted(set(dates_list))
    day_coverage = {
        "total_days": len(unique_days),
        "date_range": [unique_days[0], unique_days[-1]] if unique_days else [],
        "total_bars": n_bars,
    }

    report = {
        **schema_header("build_report", "1.0", "v2.pipeline.build_v2_dataset"),
        **identity_block(
            run_id=f"build_{dataset_fp[:8]}",
            stage="dataset_build",
            dataset_fingerprint=dataset_fp,
            config_fingerprint=meta.get("config_fingerprint", ""),
        ),
        "dataset_fingerprint": dataset_fp,
        "nan_audit": nan_audit,
        "nan_warnings": nan_warnings,
        "feature_stats": feature_stats,
        "label_quality": label_quality,
        "label_warnings": label_warnings,
        "day_coverage": day_coverage,
        "build_elapsed_seconds": round(elapsed, 1),
    }

    with open(report_path, "w") as f:
        f.write(_json.dumps(report, indent=2, default=str))
    print(f"  build report: {report_path}")
    if nan_warnings:
        print(f"  WARNING: NaN rate >20%: {'; '.join(nan_warnings)}")
    if label_warnings:
        print(f"  WARNING: {'; '.join(label_warnings)}")


def main():
    parser = argparse.ArgumentParser(description="Build the v4 exact-chain dataset")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--sidecar-dir", type=str, default=SIDECAR_DIR)
    args = parser.parse_args()
    build_dataset(output_path=args.output, sidecar_dir=args.sidecar_dir)


if __name__ == "__main__":
    main()
