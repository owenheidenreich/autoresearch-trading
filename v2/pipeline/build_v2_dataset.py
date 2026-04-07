"""Build the v2 dataset from raw market data + wide-grid option data.

Ground-up rebuild: computes ALL features from raw SPX/SPY/VIX caches
and spxw_wide/ option data. No dependency on pre-computed v1 features.

Pipeline:
  1. Load raw SPX/SPY/VIX from pickle caches
  2. Compute price features (Group 1) from raw data
  3. For each day, load spxw_wide/ and compute option + flow features (Groups 2-3)
  4. Compute triple-barrier labels (dual-direction P&L)
  5. Apply rolling z-score normalization (preserves regime info)
  6. Save to v2/data.pt with 4-way split

Usage:
    python -m v2.pipeline.build_v2_dataset [--output v2/data.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from v2.core.features import (
    compute_adaptive_spread_bps, MIN_HOLD_BARS,
    normalize_features, FEATURE_NAMES, NUM_FEATURES, _FEAT_IDX,
)
from v2.core.simulator import TRAILING_TIERS
from v2.pipeline.compute_features import (
    compute_price_features, compute_option_features, compute_flow_features,
    PRICE_FEATURE_NAMES, OPTION_FEATURE_NAMES, FLOW_FEATURE_NAMES,
    ALL_FEATURE_NAMES, _find_nearest_atm,
)

# Minimum entry price -- match simulator
MIN_ENTRY_PRICE = 0.50
# Commission per contract per side ($0.65/leg, $1.30 RT)
COMMISSION_PER_CONTRACT = 0.65

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.expanduser("~/.cache/autoresearch-trading/data")
WIDE_CACHE_DIR = os.path.join(DATA_DIR, "spxw_wide")
SPX_PATH = os.path.join(DATA_DIR, "spx_1min.pkl")
SPY_PATH = os.path.join(DATA_DIR, "spy_1min.pkl")
VIX_PATH = os.path.join(DATA_DIR, "vix_1min.pkl")
OUTPUT_PATH = "v2/data.pt"

# Split sizes (trading days from end)
VAL_DAYS = 60
PROMOTE_DAYS = 60
SHADOW_DAYS = 20

# Fixed risk parameters for triple-barrier labeling
FIXED_STOP = 0.30
FIXED_TARGET = 0.50
FIXED_HOLD = 30

# Minimum volume to consider a bar tradeable
MIN_VOLUME = 1

BARS_PER_DAY = 390


# ---------------------------------------------------------------------------
# SPX from call-put parity
# ---------------------------------------------------------------------------

def compute_spx_from_parity(atm_strike: float, call_close: float, put_close: float) -> float:
    """SPX = K + C - P (call-put parity)."""
    if any(np.isnan(x) or x <= 0 for x in [atm_strike, call_close, put_close]):
        return np.nan
    return atm_strike + call_close - put_close


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build_dataset(output_path: str = OUTPUT_PATH):
    """Build complete v2 dataset from raw market data."""
    t0 = time.time()

    # ===== STEP 1: Load raw market data =====
    print("Loading raw market data...")
    spx_df = pickle.load(open(SPX_PATH, 'rb'))
    spy_df = pickle.load(open(SPY_PATH, 'rb'))
    vix_df = pickle.load(open(VIX_PATH, 'rb'))

    # Filter SPX to only dates that have wide grid data
    wide_dates = set(f.replace('.pkl', '') for f in os.listdir(WIDE_CACHE_DIR) if f.endswith('.pkl'))
    spx_mask = spx_df['date'].isin(wide_dates)
    spx_df = spx_df[spx_mask].reset_index(drop=True)
    print(f"  Filtered to {len(wide_dates)} dates with wide grid data "
          f"(dropped {(~spx_mask).sum():,} bars)")

    # Align all three dataframes by timestamp
    spx_dates = spx_df['date'].values
    spx_times = spx_df['time'].values
    dates_list = list(spx_df['date'].values)
    N = len(spx_df)
    print(f"  SPX: {N:,} bars, {len(set(dates_list))} days")
    print(f"  SPY: {len(spy_df):,} bars")
    print(f"  VIX: {len(vix_df):,} bars")

    # Extract arrays
    spx_close = spx_df['spx_close'].values.astype(np.float64)
    spx_high = spx_df['spx_high'].values.astype(np.float64)
    spx_low = spx_df['spx_low'].values.astype(np.float64)
    spx_open = spx_df['spx_open'].values.astype(np.float64)

    # SPY: align by matching timestamps (SPX and SPY have same timestamps)
    spy_ts_to_idx = {ts: i for i, ts in enumerate(spy_df['timestamp'].values)}
    spy_volume = np.zeros(N, dtype=np.float64)
    spy_close = np.zeros(N, dtype=np.float64)
    for i, ts in enumerate(spx_df['timestamp'].values):
        j = spy_ts_to_idx.get(ts)
        if j is not None:
            spy_volume[i] = spy_df['volume'].values[j]
            spy_close[i] = spy_df['close'].values[j]

    # VIX: align by matching timestamps
    vix_ts_to_idx = {ts: i for i, ts in enumerate(vix_df['timestamp'].values)}
    vix_close = np.full(N, np.nan, dtype=np.float64)
    for i, ts in enumerate(spx_df['timestamp'].values):
        j = vix_ts_to_idx.get(ts)
        if j is not None:
            vix_close[i] = vix_df['vix_close'].values[j]

    # Compute day_starts and bar_of_day
    day_starts = [0]
    for i in range(1, N):
        if dates_list[i] != dates_list[i - 1]:
            day_starts.append(i)
    day_ends = day_starts[1:] + [N]

    bar_of_day = np.zeros(N, dtype=np.int32)
    for ds, de in zip(day_starts, day_ends):
        for i in range(ds, de):
            bar_of_day[i] = i - ds

    # Day -> bar index mapping
    day_to_bars = defaultdict(list)
    for i, d in enumerate(dates_list):
        day_to_bars[d].append(i)
    unique_dates = sorted(day_to_bars.keys())

    # ===== STEP 2: Compute Group 1 features (price/volume/market structure) =====
    print("\nComputing price features (Group 1)...")
    t1 = time.time()
    X_price = compute_price_features(
        spx_close, spx_high, spx_low, spx_open,
        spy_volume, spy_close, vix_close,
        day_starts, bar_of_day,
    )
    print(f"  {X_price.shape[1]} price features computed in {time.time()-t1:.1f}s")

    # ===== STEP 3: Compute Group 2 + 3 features (options + flow) from wide grid =====
    print("\nComputing option + flow features (Groups 2-3)...")
    t2 = time.time()
    n_opt = len(OPTION_FEATURE_NAMES)
    n_flow = len(FLOW_FEATURE_NAMES)
    X_opt = np.full((N, n_opt), np.nan, dtype=np.float64)
    X_flow = np.zeros((N, n_flow), dtype=np.float64)

    # SPX estimated from call-put parity (for labeling)
    spx_estimated = np.full(N, np.nan, dtype=np.float32)
    nearest_call_close = np.full(N, np.nan, dtype=np.float32)
    nearest_put_close = np.full(N, np.nan, dtype=np.float32)

    # Option price arrays for replay (ATM + OTM ladder)
    OTM_STEPS = [5, 10, 15, 20, 25, 30]
    atm_call_prices = np.full(N, np.nan, dtype=np.float32)
    atm_put_prices = np.full(N, np.nan, dtype=np.float32)
    otm_prices = {}
    for step in OTM_STEPS:
        otm_prices[f'otm{step}_call_prices'] = np.full(N, np.nan, dtype=np.float32)
        otm_prices[f'otm{step}_put_prices'] = np.full(N, np.nan, dtype=np.float32)

    # IV history for percentile calculation (expanding window)
    iv_history = []
    IV_HISTORY_MAX = 390 * 60  # 60 days of bars

    # Check which wide grid files exist
    wide_files = set(os.listdir(WIDE_CACHE_DIR))
    processed_days = 0

    for day_idx, day in enumerate(unique_dates):
        fname = f"{day}.pkl"
        if fname not in wide_files:
            continue

        global_indices = day_to_bars[day]
        n_bars = len(global_indices)

        wide = pickle.load(open(os.path.join(WIDE_CACHE_DIR, fname), 'rb'))
        if not wide.get('bars'):
            continue

        atm_strike_open = wide['atm_strike']
        wide_timestamps = sorted(wide['bars'].keys())
        n_wide = len(wide_timestamps)
        n_align = min(n_bars, n_wide)

        for local_i in range(n_align):
            gi = global_indices[local_i]
            bod = int(bar_of_day[gi])
            mtc = max(BARS_PER_DAY - bod, 1)

            ts = wide_timestamps[local_i]
            bar_data = wide['bars'][ts]

            # SPX from parity
            atm_data = bar_data.get(atm_strike_open, {})
            atm_call_c = atm_data.get('call_close', np.nan) or np.nan
            atm_put_c = atm_data.get('put_close', np.nan) or np.nan
            spx = compute_spx_from_parity(atm_strike_open, atm_call_c, atm_put_c)
            # Fallback to raw SPX if parity fails
            if np.isnan(spx):
                spx = spx_close[gi]
            spx_estimated[gi] = spx

            if np.isnan(spx) or spx <= 0:
                continue

            # Nearest ATM prices (for labeling)
            near_strike = _find_nearest_atm(bar_data, spx)
            if near_strike is not None:
                nd = bar_data.get(near_strike, {})
                nearest_call_close[gi] = nd.get('call_close', np.nan) or np.nan
                nearest_put_close[gi] = nd.get('put_close', np.nan) or np.nan

            # ATM + OTM price arrays (for replay baselines)
            atm_call_prices[gi] = atm_call_c
            atm_put_prices[gi] = atm_put_c
            for step in OTM_STEPS:
                call_strike = atm_strike_open + step
                put_strike = atm_strike_open - step
                cs = bar_data.get(call_strike, {})
                ps = bar_data.get(put_strike, {})
                otm_prices[f'otm{step}_call_prices'][gi] = cs.get('call_close', np.nan) or np.nan
                otm_prices[f'otm{step}_put_prices'][gi] = ps.get('put_close', np.nan) or np.nan

            # Option features
            opt_feats = compute_option_features(
                bar_data, spx, atm_strike_open, mtc,
                iv_history=iv_history[-IV_HISTORY_MAX:] if iv_history else None,
            )
            for j, fname_opt in enumerate(OPTION_FEATURE_NAMES):
                if fname_opt in opt_feats:
                    X_opt[gi, j] = opt_feats[fname_opt]

            # Track IV for percentile history
            atm_iv_val = opt_feats.get('atm_iv', np.nan)
            if np.isfinite(atm_iv_val):
                iv_history.append(atm_iv_val)

            # Flow features
            flow_feats = compute_flow_features(bar_data, spx)
            for j, fname_flow in enumerate(FLOW_FEATURE_NAMES):
                if fname_flow in flow_feats:
                    X_flow[gi, j] = flow_feats[fname_flow]

        processed_days += 1
        if (day_idx + 1) % 100 == 0:
            print(f"  {day_idx+1}/{len(unique_dates)} days processed")

    print(f"  Options/flow computed for {processed_days} days in {time.time()-t2:.1f}s")

    # Fill VRP now that we have both atm_iv and realized_vol
    vrp_idx = OPTION_FEATURE_NAMES.index('vrp')
    rv_idx = PRICE_FEATURE_NAMES.index('realized_vol')
    iv_idx = OPTION_FEATURE_NAMES.index('atm_iv')
    for i in range(N):
        atm_iv_val = X_opt[i, iv_idx]
        rv_val = X_price[i, rv_idx]
        if np.isfinite(atm_iv_val) and rv_val > 0:
            X_opt[i, vrp_idx] = atm_iv_val**2 - rv_val**2

    # Forward-fill option/flow NaN within each day
    print("\nForward-filling NaN within days...")
    for arr in [X_opt, X_flow]:
        for day in unique_dates:
            indices = day_to_bars[day]
            for j in range(arr.shape[1]):
                last_valid = np.nan
                for gi in indices:
                    if np.isnan(arr[gi, j]):
                        if not np.isnan(last_valid):
                            arr[gi, j] = last_valid
                    else:
                        last_valid = arr[gi, j]

    # Fill remaining NaN with 0
    X_opt = np.nan_to_num(X_opt, nan=0.0)

    # ===== STEP 4: Combine all features =====
    X_combined = np.concatenate([X_price, X_opt, X_flow], axis=1).astype(np.float32)
    all_feature_names = ALL_FEATURE_NAMES
    print(f"\nCombined: {X_combined.shape[1]} features ({len(PRICE_FEATURE_NAMES)} price + "
          f"{len(OPTION_FEATURE_NAMES)} option + {len(FLOW_FEATURE_NAMES)} flow)")
    assert X_combined.shape[1] == NUM_FEATURES, \
        f"Expected {NUM_FEATURES} features, got {X_combined.shape[1]}"

    # ===== STEP 5: Normalize =====
    print("\nNormalizing features (rolling z-score, 60-day window)...")
    t3 = time.time()
    valid = np.ones(N, dtype=bool)  # all bars valid for normalization
    X_normalized = normalize_features(X_combined, valid, dates=dates_list)
    print(f"  Normalized in {time.time()-t3:.1f}s")

    # Sanity check
    for j in range(X_normalized.shape[1]):
        col = X_normalized[:, j]
        vmax = np.abs(col).max()
        if vmax > 5.01:
            print(f"  WARNING: [{j}] {all_feature_names[j]} max |value| = {vmax:.2f} (expected <= 5)")

    # ===== STEP 6: Dual-direction labeling =====
    print(f"\nLabeling ({len(unique_dates)} days)...")
    t4 = time.time()

    label_trade = np.zeros(N, dtype=bool)
    label_direction = np.full(N, -1, dtype=np.int32)
    label_outcome = np.zeros(N, dtype=np.int32)
    label_pnl = np.zeros(N, dtype=np.float32)
    label_stop_pct = np.zeros(N, dtype=np.float32)
    label_target_pct = np.zeros(N, dtype=np.float32)
    label_max_hold = np.zeros(N, dtype=np.int32)
    label_confidence = np.zeros(N, dtype=np.float32)
    label_call_pnl = np.zeros(N, dtype=np.float32)
    label_put_pnl = np.zeros(N, dtype=np.float32)

    # Train split for direction medians
    n_days = len(unique_dates)
    eval_days = VAL_DAYS + PROMOTE_DAYS + SHADOW_DAYS
    train_end_day_idx = n_days - eval_days
    train_dates = set(unique_dates[:train_end_day_idx])

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_signal_bars = 0

    for day_idx, day in enumerate(unique_dates):
        global_indices = day_to_bars[day]
        n_bars_day = len(global_indices)

        wide_path = os.path.join(WIDE_CACHE_DIR, f"{day}.pkl")
        if not os.path.exists(wide_path):
            continue

        wide = pickle.load(open(wide_path, 'rb'))
        if not wide.get('bars'):
            continue

        atm_strike = wide['atm_strike']
        wide_timestamps = sorted(wide['bars'].keys())
        n_wide = len(wide_timestamps)
        n_align = min(n_bars_day, n_wide)

        # Get vix_regime for cost model
        vix_regime_idx = _FEAT_IDX.get('vix_regime')

        for local_i in range(n_align):
            gi = global_indices[local_i]
            bod = int(bar_of_day[gi])

            if bod < 30 or bod >= 270:
                continue

            # Check minimum volume
            log_vol_idx = _FEAT_IDX.get('log_total_volume')
            if log_vol_idx is not None:
                # log1p(vol) > log1p(MIN_VOLUME) means vol > MIN_VOLUME
                if X_combined[gi, log_vol_idx] < math.log1p(MIN_VOLUME):
                    continue

            spx_now = spx_estimated[gi]
            if np.isnan(spx_now):
                continue
            if local_i >= len(wide_timestamps):
                continue

            entry_ts = wide_timestamps[local_i]
            entry_bar_data = wide['bars'].get(entry_ts, {})
            entry_strike = _find_nearest_atm(entry_bar_data, spx_now)
            if entry_strike is None:
                continue

            # Get fill prices for BOTH directions
            fill_local = local_i + 1
            if fill_local >= n_align or fill_local >= len(wide_timestamps):
                continue
            fill_ts = wide_timestamps[fill_local]
            fill_bar_data = wide['bars'].get(fill_ts, {})
            fill_strike_data = fill_bar_data.get(entry_strike, {})
            call_entry_px = fill_strike_data.get('call_close', np.nan)
            put_entry_px = fill_strike_data.get('put_close', np.nan)

            if not call_entry_px or np.isnan(call_entry_px) or call_entry_px <= 0:
                call_entry_px = None
            if not put_entry_px or np.isnan(put_entry_px) or put_entry_px <= 0:
                put_entry_px = None
            if call_entry_px is None and put_entry_px is None:
                continue

            # Cost model
            is_otm = False
            vix_regime = float(X_combined[gi, vix_regime_idx]) if vix_regime_idx is not None else 0.0
            mtc = max(BARS_PER_DAY - bod, 10)

            # Pre-fetch price series for BOTH directions
            max_look = FIXED_HOLD + 2
            call_prices = []
            put_prices = []
            for k in range(max_look):
                cl = fill_local + k
                if cl >= n_align or cl >= len(wide_timestamps):
                    break
                cgi = global_indices[cl] if cl < len(global_indices) else None
                if cgi is None or dates_list[cgi] != day:
                    break
                cts = wide_timestamps[cl]
                cbd = wide['bars'].get(cts, {})
                csd = cbd.get(entry_strike, {})
                cpx = csd.get('call_close', np.nan)
                ppx = csd.get('put_close', np.nan)
                call_prices.append(cpx if cpx and not np.isnan(cpx) and cpx > 0 else np.nan)
                put_prices.append(ppx if ppx and not np.isnan(ppx) and ppx > 0 else np.nan)

            if len(call_prices) < 3:
                continue

            def _sim_one(entry_px, prices, entry_bod):
                if entry_px is None or entry_px < MIN_ENTRY_PRICE:
                    return 0.0, 0
                last_px = entry_px
                trailing_stop = -float('inf')
                exit_k = len(prices) - 1
                exit_px = entry_px
                for k in range(1, min(FIXED_HOLD + 1, len(prices))):
                    px = prices[k]
                    if np.isnan(px):
                        continue
                    last_px = px
                    unr = (px - entry_px) / entry_px
                    if k < MIN_HOLD_BARS:
                        continue
                    if unr <= -FIXED_STOP:
                        exit_k = k
                        exit_px = entry_px * (1.0 - FIXED_STOP)
                        break
                    if unr >= FIXED_TARGET:
                        exit_k = k
                        exit_px = entry_px * (1.0 + FIXED_TARGET)
                        break
                    for tier_thr, lock_pct in TRAILING_TIERS:
                        if unr >= tier_thr:
                            if lock_pct > trailing_stop:
                                trailing_stop = lock_pct
                            break
                    if trailing_stop > -float('inf') and unr <= trailing_stop:
                        exit_k = k
                        exit_px = entry_px * (1.0 + trailing_stop)
                        break
                else:
                    exit_px = last_px

                raw_pnl = (exit_px - entry_px) / entry_px
                entry_mtc = mtc
                exit_bod_val = min(entry_bod + exit_k, 389)
                exit_mtc_val = max(BARS_PER_DAY - exit_bod_val, 1)
                entry_spread = compute_adaptive_spread_bps(entry_mtc, vix_regime, is_otm) / 10000.0
                exit_spread = compute_adaptive_spread_bps(exit_mtc_val, vix_regime, is_otm) / 10000.0
                min_tick = 0.05 if entry_px < 3.00 else 0.10
                min_frac = min_tick / entry_px
                entry_spread = max(entry_spread, min_frac)
                exit_spread = max(exit_spread, min_frac)
                spread_cost = entry_spread + exit_spread
                commission_frac = (2 * COMMISSION_PER_CONTRACT) / (entry_px * 100)
                net_pnl = raw_pnl - spread_cost - commission_frac
                outcome = 1 if net_pnl > 0 else -1
                return net_pnl, outcome

            call_pnl, call_outcome = _sim_one(call_entry_px, call_prices, bod)
            put_pnl, put_outcome = _sim_one(put_entry_px, put_prices, bod)

            label_call_pnl[gi] = call_pnl
            label_put_pnl[gi] = put_pnl

            best_pnl = max(call_pnl, put_pnl)
            if call_pnl >= put_pnl:
                direction = 0
                outcome = call_outcome
            else:
                direction = 1
                outcome = put_outcome

            GATE_MIN_PNL = 0.04
            is_trade = best_pnl > GATE_MIN_PNL
            total_signal_bars += 1

            label_direction[gi] = direction
            label_pnl[gi] = best_pnl
            label_stop_pct[gi] = FIXED_STOP
            label_target_pct[gi] = FIXED_TARGET
            label_max_hold[gi] = FIXED_HOLD
            label_outcome[gi] = outcome
            label_confidence[gi] = max(0.0, min(1.0, abs(call_pnl - put_pnl) * 3.0))

            if is_trade:
                label_trade[gi] = True
                total_trades += 1
                total_wins += 1
            else:
                label_trade[gi] = False
                total_losses += 1

        if (day_idx + 1) % 100 == 0:
            print(f"  {day_idx+1}/{len(unique_dates)} days labeled, "
                  f"{total_trades:,} trades")

    print(f"  Labeling done in {time.time()-t4:.1f}s")

    # Label statistics
    gate_true_rate = label_trade.sum() / total_signal_bars if total_signal_bars > 0 else 0
    pnls_trade = label_pnl[label_trade]
    mean_pnl_trade = pnls_trade.mean() if len(pnls_trade) > 0 else 0
    call_count = int((label_direction[label_trade] == 0).sum()) if label_trade.sum() > 0 else 0
    put_count = int((label_direction[label_trade] == 1).sum()) if label_trade.sum() > 0 else 0

    print(f"\n=== LABEL STATISTICS ===")
    print(f"Signal bars: {total_signal_bars:,}")
    print(f"Gate=True: {label_trade.sum():,} ({gate_true_rate*100:.1f}%)")
    print(f"Direction: {call_count} calls, {put_count} puts")
    print(f"Mean P&L (trade=True): {mean_pnl_trade*100:.2f}%")
    if len(pnls_trade) > 0:
        gp = pnls_trade[pnls_trade > 0].sum()
        gl = abs(pnls_trade[pnls_trade < 0].sum())
        pf = gp / gl if gl > 0 else float('inf')
        wr = (pnls_trade > 0).sum() / len(pnls_trade) * 100
        print(f"PF: {pf:.3f}, WR: {wr:.1f}%")

    # ===== STEP 7: Build splits =====
    val_dates = set(unique_dates[train_end_day_idx:train_end_day_idx + VAL_DAYS])
    promote_start = train_end_day_idx + VAL_DAYS
    promote_dates = set(unique_dates[promote_start:promote_start + PROMOTE_DAYS])
    shadow_start = promote_start + PROMOTE_DAYS
    shadow_dates = set(unique_dates[shadow_start:])

    train_mask = np.array([d in train_dates for d in dates_list])
    val_mask = np.array([d in val_dates for d in dates_list])
    promote_mask = np.array([d in promote_dates for d in dates_list])
    shadow_mask = np.array([d in shadow_dates for d in dates_list])

    split_info = {
        'train_days': len(train_dates),
        'val_days': len(val_dates),
        'promote_days': len(promote_dates),
        'shadow_days': len(shadow_dates),
    }
    print(f"\nSplit: {split_info}")

    # ===== STEP 8: Save =====
    fp_str = (f"v2_rebuild:features:{X_normalized.shape}:signals:{total_signal_bars}"
              f":trades:{total_trades}:gate_rate:{gate_true_rate:.4f}"
              f":stop:{FIXED_STOP}:target:{FIXED_TARGET}:hold:{FIXED_HOLD}")
    fingerprint = hashlib.sha256(fp_str.encode()).hexdigest()[:16]

    # Build option price tensors for replay (computed from wide grid above)
    price_tensors = {
        'spot_prices': torch.from_numpy(spx_close.astype(np.float32)),
        'atm_call_prices': torch.from_numpy(atm_call_prices),
        'atm_put_prices': torch.from_numpy(atm_put_prices),
    }
    for k, v in otm_prices.items():
        price_tensors[k] = torch.from_numpy(v)
    oracle_tensors = {}  # no backward compat needed for fresh rebuild

    dataset = {
        'X': torch.from_numpy(X_normalized.astype(np.float32)),
        'feature_names': list(all_feature_names),

        'label_trade': torch.from_numpy(label_trade),
        'label_direction': torch.from_numpy(label_direction),
        'label_outcome': torch.from_numpy(label_outcome),
        'label_pnl': torch.from_numpy(label_pnl),
        'label_stop_pct': torch.from_numpy(label_stop_pct),
        'label_target_pct': torch.from_numpy(label_target_pct),
        'label_max_hold': torch.from_numpy(label_max_hold.astype(np.int32)),
        'label_confidence': torch.from_numpy(label_confidence),
        'label_call_pnl': torch.from_numpy(label_call_pnl),
        'label_put_pnl': torch.from_numpy(label_put_pnl),

        **oracle_tensors,

        'spx_estimated': torch.from_numpy(spx_estimated),
        'nearest_call_close': torch.from_numpy(nearest_call_close),
        'nearest_put_close': torch.from_numpy(nearest_put_close),

        'dates': dates_list,
        'bar_of_day': torch.from_numpy(bar_of_day),
        'train_mask': torch.from_numpy(train_mask),
        'val_mask': torch.from_numpy(val_mask),
        'promote_mask': torch.from_numpy(promote_mask),
        'shadow_mask': torch.from_numpy(shadow_mask),

        **price_tensors,

        'metadata': {
            'version': 'v2_rebuild',
            'build_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'fingerprint': fingerprint,
            'n_features': NUM_FEATURES,
            'normalization': 'rolling_zscore_60day',
            'total_signal_bars': total_signal_bars,
            'total_trades': total_trades,
            'gate_true_rate': gate_true_rate,
            'mean_pnl_trade': float(mean_pnl_trade),
            'fixed_stop': FIXED_STOP,
            'fixed_target': FIXED_TARGET,
            'fixed_hold': FIXED_HOLD,
            'split': split_info,
            'cost_model': {
                'spread_rt': 0.30,
                'commission_rt': 1.30,
            },
            'direction_signal': 'dual-direction P&L (model learns to choose)',
            'label_scheme': 'dual_direction_pnl',
        },
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"\nSaving to {output_path}...")
    torch.save(dataset, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Features: {NUM_FEATURES}")
    print(f"  Fingerprint: {fingerprint}")
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Build v2 dataset from raw data")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    build_dataset(output_path=args.output)


if __name__ == "__main__":
    main()
