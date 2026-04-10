"""Build the v4 exact-chain dataset manifest plus per-day chain sidecars."""
from __future__ import annotations

import argparse
import functools
import math
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
from v2.core.policy import DEFAULT_POLICY
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

    mats = {
        "mid": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "bid": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "ask": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "quality": np.full((n_contracts, n_bars), QUALITY_CORRUPT, dtype=np.int8),
        "iv": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "delta": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "gamma": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "theta": np.full((n_contracts, n_bars), np.nan, dtype=np.float32),
        "volume": np.zeros((n_contracts, n_bars), dtype=np.float32),
        "transactions": np.zeros((n_contracts, n_bars), dtype=np.float32),
        "spread": np.ones((n_contracts, n_bars), dtype=np.float32),
    }
    bar_quality: list[dict] = []

    for local_i, bar_contracts in enumerate(bar_contracts_list):
        spot = float(spot_series[local_i])
        mtc = max(BARS_PER_DAY - local_i, 1)
        year_frac = mtc / (252.0 * BARS_PER_DAY)
        seen = 0
        labeled = 0
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
            seen += 1

            strike, right = contract
            is_call = right == "C"
            iv = _bs_iv(close_px, spot, strike, year_frac, 0.05, is_call=is_call) if np.isfinite(close_px) and close_px > 0 else np.nan
            mats["iv"][idx, local_i] = iv if np.isfinite(iv) else np.nan
            if np.isfinite(iv):
                delta, gamma, theta, _ = _bs_greeks(spot, strike, year_frac, 0.05, iv)
                if np.isfinite(delta):
                    mats["delta"][idx, local_i] = delta if is_call else (delta - 1.0)
                    mats["gamma"][idx, local_i] = gamma
                    mats["theta"][idx, local_i] = theta
                    labeled += 1

        bar_quality.append({"observed_contracts": seen, "greeked_contracts": labeled})
    return mats, bar_quality


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
            )

            row_features.append(row)
            row_labels.append(np.nan)
            row_contract_idx.append(contract_idx)

            series = chain_mats["mid"][contract_idx]
            if not _forward_path_complete(series, local_i, DEFAULT_POLICY.max_hold_bars, n_bars):
                continue

            right = "P" if int(contract_right[contract_idx]) == 1 else "C"
            intent = TradeIntent(
                trade=True,
                expiry=expiry,
                strike=float(contract_strike[contract_idx]),
                right=right,
                qty=DEFAULT_POLICY.qty,
                entry_ref_price=mid_now,
                order_style=DEFAULT_POLICY.order_style,
                tif=DEFAULT_POLICY.tif,
                stop_price=mid_now * (1.0 - DEFAULT_POLICY.stop_pct),
                take_profit_price=mid_now * (1.0 + DEFAULT_POLICY.target_pct),
                max_hold_bars=DEFAULT_POLICY.max_hold_bars,
                exit_policy=DEFAULT_POLICY.exit_policy,
                confidence=0.5,
                reason_codes=("dataset_label",),
                bar_index=bod,
                timestamp=str(int(timestamps[local_i])) if len(timestamps) else "",
                underlying_price=spot,
            )
            trade = simulate_trade(
                intent=intent,
                option_prices=series.astype(np.float32),
                features=feature_context,
                bar_of_day=np.arange(n_bars, dtype=np.int32),
                dates=[day] * n_bars,
                global_entry_bar=local_i,
            )
            if trade is None:
                continue

            any_labelable = True
            pnl = float(trade.net_pnl_pct)
            row_labels[-1] = pnl
            if pnl > best_pnl:
                best_pnl = pnl
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
    best_contract_pnl = np.zeros(N, dtype=np.float32)
    best_contract_strike = np.zeros(N, dtype=np.float32)
    best_contract_right = np.full(N, -1, dtype=np.int32)
    label_trade = np.zeros(N, dtype=bool)
    label_trade_valid = np.zeros(N, dtype=bool)

    for day_idx, day in enumerate(unique_dates):
        global_indices = day_to_bars[day]
        n_bars_day = len(global_indices)
        cache_path = os.path.join(FULL_CHAIN_CACHE_DIR, f"{day}.pkl")
        expiry = day.replace("-", "")
        day_ts_ms = np.asarray(spx_df["timestamp"][global_indices], dtype=np.int64)

        if os.path.exists(cache_path):
            full_day = pickle.load(open(cache_path, "rb"))
            raw_day_bars = full_day.get("bars", {})
            raw_bars = [_chain_bar_for_timestamp(raw_day_bars, int(ts)) for ts in day_ts_ms]
            aligned_ts = day_ts_ms
            contracts = sorted(full_day.get("contracts", []))
        else:
            raw_bars = [{} for _ in range(n_bars_day)]
            aligned_ts = day_ts_ms
            contracts = []

        day_spot = spot_prices[global_indices]
        if contracts:
            chain_mats, _ = _fill_chain_matrices(contracts, raw_bars, day_spot)
            chain_mats["contract_strike"] = np.asarray([c[0] for c in contracts], dtype=np.float32)
            chain_mats["contract_right"] = np.asarray([1 if c[1] == "P" else 0 for c in contracts], dtype=np.int8)
        else:
            chain_mats = {
                "mid": np.zeros((0, n_bars_day), dtype=np.float32),
                "bid": np.zeros((0, n_bars_day), dtype=np.float32),
                "ask": np.zeros((0, n_bars_day), dtype=np.float32),
                "quality": np.zeros((0, n_bars_day), dtype=np.int8),
                "iv": np.zeros((0, n_bars_day), dtype=np.float32),
                "delta": np.zeros((0, n_bars_day), dtype=np.float32),
                "gamma": np.zeros((0, n_bars_day), dtype=np.float32),
                "theta": np.zeros((0, n_bars_day), dtype=np.float32),
                "volume": np.zeros((0, n_bars_day), dtype=np.float32),
                "transactions": np.zeros((0, n_bars_day), dtype=np.float32),
                "spread": np.zeros((0, n_bars_day), dtype=np.float32),
                "contract_strike": np.zeros(0, dtype=np.float32),
                "contract_right": np.zeros(0, dtype=np.int8),
            }

        atm_strike_open = round(float(spot_prices[global_indices[0]]) / 5.0) * 5.0 if global_indices else 0.0
        iv_history: list[float] = []
        for local_i in range(n_bars_day):
            gi = global_indices[local_i]
            mtc = max(BARS_PER_DAY - local_i, 1)
            wide_bar = to_wide_bar(raw_bars[local_i])
            opt_feats = compute_option_features(
                wide_bar,
                float(spot_prices[gi]),
                atm_strike_open,
                mtc,
                iv_history=iv_history[-390 * 60:] if iv_history else None,
            )
            for j, name in enumerate(OPTION_FEATURE_NAMES):
                if name in opt_feats:
                    X_opt[gi, j] = opt_feats[name]
            flow_feats = compute_flow_features(wide_bar, float(spot_prices[gi]))
            for j, name in enumerate(FLOW_FEATURE_NAMES):
                if name in flow_feats:
                    X_flow[gi, j] = flow_feats[name]
            atm_iv_val = opt_feats.get("atm_iv", np.nan)
            if np.isfinite(atm_iv_val):
                iv_history.append(atm_iv_val)

        vrp_idx = OPTION_FEATURE_NAMES.index("vrp")
        rv_idx = PRICE_FEATURE_NAMES.index("realized_vol")
        iv_idx = OPTION_FEATURE_NAMES.index("atm_iv")
        for gi in global_indices:
            atm_iv_val = X_opt[gi, iv_idx]
            rv_val = X_price[gi, rv_idx]
            if np.isfinite(atm_iv_val) and rv_val > 0:
                X_opt[gi, vrp_idx] = atm_iv_val ** 2 - rv_val ** 2

        X_opt[global_indices] = np.nan_to_num(X_opt[global_indices], nan=0.0)

        X_day_context = np.concatenate(
            [X_price[global_indices], X_opt[global_indices], X_flow[global_indices]],
            axis=1,
        ).astype(np.float32)
        sidecar = _label_day_sidecar(
            day=day,
            expiry=expiry,
            global_indices=global_indices,
            timestamps=aligned_ts,
            spot_series=spot_prices[global_indices],
            feature_context=X_day_context,
            raw_full_bar_list=raw_bars,
            chain_mats=chain_mats,
        )

        path = sidecar_path(sidecar_dir, day)
        torch.save(sidecar, path)
        sidecar_paths.append(path)
        if len(sidecar["bar_ptrs"]) > 1:
            day_counts = np.diff(sidecar["bar_ptrs"])
            if len(day_counts):
                max_contracts_per_bar = max(max_contracts_per_bar, int(day_counts.max()))

        for local_i, gi in enumerate(global_indices):
            if local_i >= len(sidecar["bar_best_pnl"]):
                continue
            best_contract_pnl[gi] = sidecar["bar_best_pnl"][local_i]
            label_trade[gi] = bool(sidecar["bar_label_trade"][local_i])
            label_trade_valid[gi] = bool(sidecar["bar_labelable"][local_i])
            if sidecar["bar_best_contract_idx"][local_i] >= 0:
                start = int(sidecar["bar_ptrs"][local_i])
                row_local = int(sidecar["bar_best_contract_idx"][local_i])
                contract_idx = int(sidecar["row_contract_idx"][start + row_local])
                best_contract_strike[gi] = float(sidecar["contract_strike"][contract_idx])
                best_contract_right[gi] = int(sidecar["contract_right"][contract_idx])
                total_trade_bars += 1
            total_signal_bars += 1

        if (day_idx + 1) % 100 == 0:
            print(f"  {day_idx + 1}/{len(unique_dates)} days processed")

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


def main():
    parser = argparse.ArgumentParser(description="Build the v4 exact-chain dataset")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--sidecar-dir", type=str, default=SIDECAR_DIR)
    args = parser.parse_args()
    build_dataset(output_path=args.output, sidecar_dir=args.sidecar_dir)


if __name__ == "__main__":
    main()
