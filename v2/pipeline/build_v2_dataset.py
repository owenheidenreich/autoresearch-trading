"""Build the v2 dataset from wide-grid Polygon data + existing features.

This is the COMPLETE rebuild pipeline that:
1. Loads existing 39 features from data.pt
2. Loads wide-grid OHLCV from spxw_wide/ cache (82 contracts per day)
3. Computes enriched features (moneyness, volume, spread, regime indicators)
4. Computes triple-barrier labels (with real losers)
5. Builds data.pt with 4-way split

All new features are RELATIVE (moneyness %, normalized prices) so patterns
learned at SPX 4300 transfer to SPX 6500.

Usage:
    python -m v2.pipeline.build_v2_dataset [--output v2/data.pt]
"""
from __future__ import annotations

import argparse
import hashlib  # noqa: used for fingerprint
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WIDE_CACHE_DIR = os.path.expanduser("~/.cache/autoresearch-trading/data/spxw_wide")
EXISTING_DATA = "v2/data.pt"
OUTPUT_PATH = "v2/data.pt"

# Split sizes (trading days from end)
VAL_DAYS = 60
PROMOTE_DAYS = 60
SHADOW_DAYS = 20

# Label params: fixed-hold direction labels (no stop/TP)
# Stops kill the signal (theta triggers before direction plays out).
# The model learns WHEN to enter. Exit is fixed at HOLD_BARS.
TB_STOP_PCT = 999.0     # effectively no stop (direction label only)
TB_TARGET_PCT = 999.0   # effectively no TP (direction label only)
TB_MAX_HOLD = 15        # 15-bar hold (15 minutes -- short enough to avoid theta crush)
TB_SPREAD_COST = 0.0030 # $0.30 round-trip spread as fraction ($0.30/$100 multiplier)
TB_COMMISSION = 0.0013  # $1.30 round-trip commission as fraction ($1.30/$1000 notional for $10 option)

# Minimum volume to consider a bar tradeable
MIN_VOLUME = 1


# ---------------------------------------------------------------------------
# Phase 1: Feature computation from wide grid
# ---------------------------------------------------------------------------

def compute_spx_from_parity(atm_strike: float, call_close: float, put_close: float) -> float:
    """SPX = K + C - P (call-put parity)."""
    if any(np.isnan(x) or x <= 0 for x in [atm_strike, call_close, put_close]):
        return np.nan
    return atm_strike + call_close - put_close


def find_nearest_atm_strike(bar_data: dict, spx_current: float) -> float | None:
    """Find the strike closest to current SPX from the available strikes."""
    if np.isnan(spx_current) or not bar_data:
        return None
    best_strike = None
    best_dist = float('inf')
    for strike in bar_data.keys():
        dist = abs(strike - spx_current)
        if dist < best_dist:
            best_dist = dist
            best_strike = strike
    return best_strike


def compute_bar_features(
    bar_data: dict,
    atm_strike: float,
    spx_current: float,
    minutes_to_close: float,
) -> dict:
    """Compute enriched features for one bar from wide-grid data.

    All features are RELATIVE (moneyness %, normalized) not absolute.
    """
    features = {}

    if np.isnan(spx_current) or spx_current <= 0 or not bar_data:
        return features

    # Find strike closest to current SPX
    near_atm = find_nearest_atm_strike(bar_data, spx_current)
    if near_atm is None:
        return features

    near_data = bar_data.get(near_atm, {})

    # --- Moneyness features ---
    features['current_moneyness_pct'] = (atm_strike - spx_current) / spx_current * 100
    features['intraday_drift_pct'] = (spx_current - atm_strike) / atm_strike * 100
    features['near_atm_moneyness_pct'] = (near_atm - spx_current) / spx_current * 100

    # --- Volume features (from nearest-ATM strike) ---
    call_vol = near_data.get('call_volume', 0) or 0
    put_vol = near_data.get('put_volume', 0) or 0
    total_vol = call_vol + put_vol
    features['near_atm_call_volume'] = float(call_vol)
    features['near_atm_put_volume'] = float(put_vol)
    features['near_atm_total_volume'] = float(total_vol)
    features['call_put_flow_ratio'] = call_vol / total_vol if total_vol > 0 else 0.5
    features['volume_zero_flag'] = 1.0 if total_vol == 0 else 0.0
    features['log_total_volume'] = math.log1p(total_vol)

    # --- Volume across the chain (aggregate flow) ---
    chain_call_vol = sum(bar_data[s].get('call_volume', 0) or 0 for s in bar_data)
    chain_put_vol = sum(bar_data[s].get('put_volume', 0) or 0 for s in bar_data)
    chain_total = chain_call_vol + chain_put_vol
    features['chain_call_put_ratio'] = chain_call_vol / chain_total if chain_total > 0 else 0.5
    features['log_chain_volume'] = math.log1p(chain_total)

    # --- Spread proxy (Corwin-Schultz simplified: high-low of nearest ATM) ---
    call_high = near_data.get('call_high', np.nan) or np.nan
    call_low = near_data.get('call_low', np.nan) or np.nan
    call_close = near_data.get('call_close', np.nan) or np.nan
    if not np.isnan(call_high) and not np.isnan(call_low) and call_high > 0 and call_low > 0:
        hl_range = call_high - call_low
        mid = (call_high + call_low) / 2
        features['call_hl_range_pct'] = hl_range / mid if mid > 0 else 0.0
    else:
        features['call_hl_range_pct'] = np.nan

    # --- Normalized option prices (price / SPX for scale invariance) ---
    if not np.isnan(call_close) and call_close > 0:
        features['near_atm_call_price_norm'] = call_close / spx_current * 100
    else:
        features['near_atm_call_price_norm'] = np.nan

    put_close = near_data.get('put_close', np.nan) or np.nan
    if not np.isnan(put_close) and put_close > 0:
        features['near_atm_put_price_norm'] = put_close / spx_current * 100
    else:
        features['near_atm_put_price_norm'] = np.nan

    # --- Theta acceleration (0DTE specific: theta burns faster near close) ---
    if minutes_to_close > 0:
        features['theta_acceleration'] = 1.0 / math.sqrt(max(minutes_to_close, 1.0))
    else:
        features['theta_acceleration'] = 1.0

    # --- Transaction activity ---
    call_txn = near_data.get('call_transactions', 0) or 0
    put_txn = near_data.get('put_transactions', 0) or 0
    features['near_atm_transactions'] = float(call_txn + put_txn)

    return features


# New feature names (appended to existing 39)
NEW_FEATURE_NAMES = [
    'current_moneyness_pct',
    'intraday_drift_pct',
    'near_atm_moneyness_pct',
    'near_atm_call_volume',
    'near_atm_put_volume',
    'near_atm_total_volume',
    'call_put_flow_ratio',
    'volume_zero_flag',
    'log_total_volume',
    'chain_call_put_ratio',
    'log_chain_volume',
    'call_hl_range_pct',
    'near_atm_call_price_norm',
    'near_atm_put_price_norm',
    'theta_acceleration',
    'near_atm_transactions',
]


# ---------------------------------------------------------------------------
# Phase 2: Triple-barrier labeler
# ---------------------------------------------------------------------------

def triple_barrier_label(
    entry_bar_idx: int,
    option_prices: np.ndarray,
    stop_pct: float,
    target_pct: float,
    max_hold: int,
    cost_pct: float,
    same_day_mask: np.ndarray,
) -> tuple[int, float]:
    """Simulate a trade forward with triple barrier.

    Returns (outcome, net_pnl):
        outcome: +1 (TP hit), -1 (SL hit), 0 (timeout/EOD)
        net_pnl: actual P&L as fraction of entry price, after costs
    """
    fill_bar = entry_bar_idx + 1
    if fill_bar >= len(option_prices):
        return (0, 0.0)

    entry_px = option_prices[fill_bar]
    if np.isnan(entry_px) or entry_px <= 0:
        return (0, 0.0)

    last_valid_px = entry_px
    for k in range(1, max_hold + 1):
        check = fill_bar + k
        if check >= len(option_prices):
            break
        if not same_day_mask[check]:
            break

        px = option_prices[check]
        if np.isnan(px) or px <= 0:
            continue

        last_valid_px = px
        unrealized = (px - entry_px) / entry_px

        # Stop loss (checked first)
        if unrealized <= -stop_pct:
            net = -stop_pct - cost_pct
            return (-1, net)

        # Take profit
        if unrealized >= target_pct:
            net = target_pct - cost_pct
            return (+1, net)

    # Timeout / EOD
    raw_pnl = (last_valid_px - entry_px) / entry_px
    net = raw_pnl - cost_pct
    outcome = +1 if net > 0 else -1 if net < -0.01 else 0
    return (outcome, net)


def compute_direction_signal(features_row: np.ndarray, feature_names: list[str]) -> int:
    """Determine direction from volatility-regime features.

    Returns: +1 (buy call), -1 (buy put), 0 (no trade)

    Uses session_range_pct as primary signal (strongest edge in signal scan).
    High session range = volatile = buy call (gamma regime).
    Low session range = calm = buy put (theta regime).
    """
    # Get feature indices
    idx = {name: i for i, name in enumerate(feature_names)}

    session_range = features_row[idx.get('session_range_pct', -1)] if 'session_range_pct' in idx else np.nan
    atm_iv = features_row[idx.get('atm_iv', -1)] if 'atm_iv' in idx else np.nan
    realized_vol = features_row[idx.get('realized_vol', -1)] if 'realized_vol' in idx else np.nan

    if np.isnan(session_range):
        return 0

    # Combined volatility score (simple average of available signals)
    vol_signals = []
    if not np.isnan(session_range):
        vol_signals.append(session_range)
    if not np.isnan(atm_iv):
        vol_signals.append(atm_iv)
    if not np.isnan(realized_vol):
        vol_signals.append(realized_vol)

    if not vol_signals:
        return 0

    # We'll use session_range_pct > median as the split.
    # The median is computed during label generation over the training set.
    # For now, return the raw signal; the caller applies the threshold.
    return 1  # placeholder -- actual threshold applied in label loop


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    existing_path: str = EXISTING_DATA,
    output_path: str = OUTPUT_PATH,
):
    """Build complete v2 dataset with enriched features + honest labels."""
    t0 = time.time()

    # Load existing data for the 39 original features and metadata
    print(f"Loading existing data from {existing_path}...")
    existing = torch.load(existing_path, map_location='cpu', weights_only=False)
    X_old = existing['X'].numpy()
    feature_names_old = existing.get('feature_names', [f'f{i}' for i in range(X_old.shape[1])])
    dates = existing['dates']
    bar_of_day = existing['bar_of_day'].numpy()
    N = len(dates)
    n_old_features = X_old.shape[1]

    print(f"  {N:,} bars, {len(set(dates))} days, {n_old_features} existing features")

    # Build day -> global bar indices mapping
    day_to_bars = defaultdict(list)
    for i, d in enumerate(dates):
        day_to_bars[d].append(i)
    unique_dates = sorted(day_to_bars.keys())

    # Allocate new feature arrays
    n_new = len(NEW_FEATURE_NAMES)
    X_new = np.full((N, n_new), np.nan, dtype=np.float32)

    # Allocate label arrays
    label_trade = np.zeros(N, dtype=bool)
    label_direction = np.full(N, -1, dtype=np.int32)  # 0=call, 1=put, -1=no trade
    label_outcome = np.zeros(N, dtype=np.int32)  # +1, -1, 0
    label_pnl = np.zeros(N, dtype=np.float32)
    label_exit_reason = np.full(N, '', dtype=object)

    # For nearest-ATM contract prices (used by labeler for simulation)
    nearest_call_close = np.full(N, np.nan, dtype=np.float32)
    nearest_put_close = np.full(N, np.nan, dtype=np.float32)
    spx_estimated = np.full(N, np.nan, dtype=np.float32)

    # Compute session_range_pct median from TRAINING data for direction threshold
    # Use existing feature for now; will be computed on train split only
    sr_idx = feature_names_old.index('session_range_pct') if 'session_range_pct' in feature_names_old else None

    # First pass: determine train split to compute threshold
    n_days = len(unique_dates)
    eval_days = VAL_DAYS + PROMOTE_DAYS + SHADOW_DAYS
    train_end_day_idx = n_days - eval_days
    train_dates = set(unique_dates[:train_end_day_idx])

    if sr_idx is not None:
        train_mask_arr = np.array([d in train_dates for d in dates])
        tradeable = (bar_of_day >= 30) & (bar_of_day < 300) & train_mask_arr
        sr_values = X_old[tradeable, sr_idx]
        sr_values = sr_values[~np.isnan(sr_values)]
        sr_median = float(np.median(sr_values)) if len(sr_values) > 0 else 0.0
        sr_q25 = float(np.percentile(sr_values, 25)) if len(sr_values) > 0 else 0.0
        sr_q75 = float(np.percentile(sr_values, 75)) if len(sr_values) > 0 else 0.0
        sr_iqr = sr_q75 - sr_q25
        print(f"  session_range_pct: median={sr_median:.6f} IQR={sr_iqr:.6f}")
    else:
        sr_median = 0.0
        sr_iqr = 0.01  # fallback

    # Process each day
    print(f"\nProcessing {len(unique_dates)} days...")
    processed_days = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0

    for day_idx, day in enumerate(unique_dates):
        global_indices = day_to_bars[day]
        n_bars = len(global_indices)

        # Load wide-grid data for this day
        wide_path = os.path.join(WIDE_CACHE_DIR, f"{day}.pkl")
        if not os.path.exists(wide_path):
            continue

        wide = pickle.load(open(wide_path, 'rb'))
        if not wide.get('bars'):
            continue

        atm_strike = wide['atm_strike']
        wide_timestamps = sorted(wide['bars'].keys())
        n_wide = len(wide_timestamps)
        n_align = min(n_bars, n_wide)

        # Build same-day mask for this day
        same_day = np.zeros(N, dtype=bool)
        for gi in global_indices:
            same_day[gi] = True

        # ===== PASS 1: Compute features + populate price series for ALL bars =====
        day_features_computed = {}  # gi -> feats dict

        for local_i in range(n_align):
            gi = global_indices[local_i]
            bod = int(bar_of_day[gi])

            if local_i >= len(wide_timestamps):
                continue
            ts = wide_timestamps[local_i]
            bar_data = wide['bars'][ts]

            # Compute SPX from call-put parity
            atm_data = bar_data.get(atm_strike, {})
            atm_call_c = atm_data.get('call_close', np.nan) or np.nan
            atm_put_c = atm_data.get('put_close', np.nan) or np.nan
            spx = compute_spx_from_parity(atm_strike, atm_call_c, atm_put_c)
            spx_estimated[gi] = spx

            mtc = max(390 - bod, 1)

            # Compute enriched features
            feats = compute_bar_features(bar_data, atm_strike, spx, mtc)
            day_features_computed[gi] = feats
            for j, fname in enumerate(NEW_FEATURE_NAMES):
                if fname in feats:
                    X_new[gi, j] = feats[fname]

            # Populate nearest-ATM price series (needed by labeler to look forward)
            if not np.isnan(spx):
                near_strike = find_nearest_atm_strike(bar_data, spx)
                if near_strike is not None:
                    nd = bar_data.get(near_strike, {})
                    nearest_call_close[gi] = nd.get('call_close', np.nan) or np.nan
                    nearest_put_close[gi] = nd.get('put_close', np.nan) or np.nan

        # ===== PASS 2: Triple-barrier labeling (same-contract simulation) =====
        # For each entry bar, pick the nearest-ATM strike and follow THAT
        # SPECIFIC STRIKE's price for the entire hold. No contract switching.

        for local_i in range(n_align):
            gi = global_indices[local_i]
            bod = int(bar_of_day[gi])

            if bod < 30 or bod >= 270:
                continue

            feats = day_features_computed.get(gi, {})
            total_vol = feats.get('near_atm_total_volume', 0)
            if total_vol < MIN_VOLUME:
                continue

            # Direction from volatility regime
            if sr_idx is not None:
                sr_val = X_old[gi, sr_idx]
                if np.isnan(sr_val):
                    continue
                if abs(sr_val - sr_median) < sr_iqr * 0.5:
                    continue
                direction = 0 if sr_val > sr_median else 1  # 0=call, 1=put
            else:
                continue

            # Find the entry strike (nearest ATM at this bar)
            spx_now = spx_estimated[gi]
            if np.isnan(spx_now):
                continue

            if local_i >= len(wide_timestamps):
                continue
            entry_ts = wide_timestamps[local_i]
            entry_bar_data = wide['bars'].get(entry_ts, {})
            entry_strike = find_nearest_atm_strike(entry_bar_data, spx_now)
            if entry_strike is None:
                continue

            side = 'call' if direction == 0 else 'put'

            # Get entry price (fill at next bar)
            fill_local = local_i + 1
            if fill_local >= n_align or fill_local >= len(wide_timestamps):
                continue
            fill_ts = wide_timestamps[fill_local]
            fill_bar_data = wide['bars'].get(fill_ts, {})
            fill_strike_data = fill_bar_data.get(entry_strike, {})
            entry_px = fill_strike_data.get(f'{side}_close', np.nan)
            if not entry_px or np.isnan(entry_px) or entry_px <= 0:
                continue

            # Simulate: follow the SAME STRIKE for TB_MAX_HOLD bars
            cost_frac = 1.60 / (entry_px * 100)
            last_valid_px = entry_px
            exit_outcome = 0
            exit_pnl = 0.0

            for k in range(1, TB_MAX_HOLD + 1):
                check_local = fill_local + k
                if check_local >= n_align or check_local >= len(wide_timestamps):
                    break
                check_gi = global_indices[check_local] if check_local < len(global_indices) else None
                if check_gi is None or dates[check_gi] != day:
                    break

                check_ts = wide_timestamps[check_local]
                check_bar_data = wide['bars'].get(check_ts, {})
                check_strike_data = check_bar_data.get(entry_strike, {})
                px = check_strike_data.get(f'{side}_close', np.nan)
                if not px or np.isnan(px) or px <= 0:
                    continue

                last_valid_px = px
                unrealized = (px - entry_px) / entry_px

                if unrealized <= -TB_STOP_PCT:
                    exit_pnl = -TB_STOP_PCT - cost_frac
                    exit_outcome = -1
                    break
                if unrealized >= TB_TARGET_PCT:
                    exit_pnl = TB_TARGET_PCT - cost_frac
                    exit_outcome = 1
                    break

            if exit_outcome == 0:
                raw = (last_valid_px - entry_px) / entry_px
                exit_pnl = raw - cost_frac
                exit_outcome = 1 if exit_pnl > 0 else -1 if exit_pnl < -0.01 else 0

            if exit_outcome == 0 and abs(exit_pnl) < 0.001:
                continue

            label_trade[gi] = True
            label_direction[gi] = direction
            label_outcome[gi] = exit_outcome
            label_pnl[gi] = exit_pnl
            total_trades += 1
            if exit_outcome == 1:
                total_wins += 1
            elif exit_outcome == -1:
                total_losses += 1

        processed_days += 1
        if (day_idx + 1) % 100 == 0:
            print(f"  {day_idx+1}/{len(unique_dates)} days, "
                  f"{total_trades:,} trades ({total_wins} W / {total_losses} L)")

    elapsed = time.time() - t0
    print(f"\nDone: {processed_days} days in {elapsed:.1f}s")

    # --- Label statistics ---
    trade_rate = label_trade.sum() / N
    tradeable_bars = ((bar_of_day >= 30) & (bar_of_day < 270)).sum()
    trade_rate_of_tradeable = label_trade.sum() / tradeable_bars if tradeable_bars > 0 else 0
    win_rate = total_wins / total_trades if total_trades > 0 else 0
    loss_rate = total_losses / total_trades if total_trades > 0 else 0
    pnls = label_pnl[label_trade]
    mean_pnl = pnls.mean() if len(pnls) > 0 else 0

    print(f"\n=== LABEL STATISTICS ===")
    print(f"Total trades: {total_trades:,}")
    print(f"Trade rate (all bars): {trade_rate*100:.1f}%")
    print(f"Trade rate (tradeable bars): {trade_rate_of_tradeable*100:.1f}%")
    print(f"Win rate: {win_rate*100:.1f}%")
    print(f"Loss rate: {loss_rate*100:.1f}%")
    print(f"Mean P&L: {mean_pnl*100:.2f}%")
    if total_trades > 0:
        gp = pnls[pnls > 0].sum()
        gl = abs(pnls[pnls < 0].sum())
        pf = gp / gl if gl > 0 else 0
        print(f"Profit factor: {pf:.3f}")

    # --- Validation gates ---
    errors = []
    if loss_rate < 0.20:
        errors.append(f"Loss rate {loss_rate:.1%} < 20% -- not enough losers")
    if win_rate > 0.70:
        errors.append(f"Win rate {win_rate:.1%} > 70% -- still too oracle-like")
    if trade_rate_of_tradeable > 0.30:
        errors.append(f"Trade rate {trade_rate_of_tradeable:.1%} > 30% -- too aggressive")
    if trade_rate_of_tradeable < 0.03:
        errors.append(f"Trade rate {trade_rate_of_tradeable:.1%} < 3% -- too few examples")

    if errors:
        print(f"\n*** VALIDATION WARNINGS ***")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"\nAll validation gates PASSED.")

    # --- Combine features ---
    X_combined = np.concatenate([X_old, X_new], axis=1)
    all_feature_names = list(feature_names_old) + NEW_FEATURE_NAMES
    print(f"\nCombined features: {X_combined.shape[1]} ({n_old_features} old + {n_new} new)")

    # --- Build 4-way split ---
    val_dates = set(unique_dates[train_end_day_idx:train_end_day_idx + VAL_DAYS])
    promote_start = train_end_day_idx + VAL_DAYS
    promote_dates = set(unique_dates[promote_start:promote_start + PROMOTE_DAYS])
    shadow_start = promote_start + PROMOTE_DAYS
    shadow_dates = set(unique_dates[shadow_start:])

    train_mask = np.array([d in train_dates for d in dates])
    val_mask = np.array([d in val_dates for d in dates])
    promote_mask = np.array([d in promote_dates for d in dates])
    shadow_mask = np.array([d in shadow_dates for d in dates])

    split_info = {
        'train_days': len(train_dates),
        'val_days': len(val_dates),
        'promote_days': len(promote_dates),
        'shadow_days': len(shadow_dates),
    }
    print(f"Split: {split_info}")

    # --- Fingerprint ---
    fp_str = f"wide_grid:features:{X_combined.shape}:trades:{total_trades}:wr:{win_rate:.4f}"
    fingerprint = hashlib.sha256(fp_str.encode()).hexdigest()[:16]

    # --- Build output ---
    dataset = {
        'X': torch.from_numpy(X_combined),
        'feature_names': all_feature_names,

        # Labels (triple-barrier)
        'label_trade': torch.from_numpy(label_trade),
        'label_direction': torch.from_numpy(label_direction),
        'label_outcome': torch.from_numpy(label_outcome),
        'label_pnl': torch.from_numpy(label_pnl),

        # Keep oracle labels for backward compat (marked as deprecated)
        **{k: existing[k] for k in existing if k.startswith('oracle_')},

        # Auxiliary
        'spx_estimated': torch.from_numpy(spx_estimated),
        'nearest_call_close': torch.from_numpy(nearest_call_close),
        'nearest_put_close': torch.from_numpy(nearest_put_close),

        # Splits
        'dates': dates,
        'bar_of_day': existing['bar_of_day'],
        'train_mask': torch.from_numpy(train_mask),
        'val_mask': torch.from_numpy(val_mask),
        'promote_mask': torch.from_numpy(promote_mask),
        'shadow_mask': torch.from_numpy(shadow_mask),

        # Keep option prices for replay
        'spot_prices': existing['spot_prices'],
        **{k: existing[k] for k in existing if k.endswith('_prices') and k != 'spot_prices'},

        # Metadata
        'metadata': {
            'version': 'v2_wide_grid_triple_barrier',
            'build_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'fingerprint': fingerprint,
            'n_features': X_combined.shape[1],
            'n_old_features': n_old_features,
            'n_new_features': n_new,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'trade_rate': trade_rate_of_tradeable,
            'mean_pnl': float(mean_pnl),
            'split': split_info,
            'triple_barrier': {
                'stop_pct': TB_STOP_PCT,
                'target_pct': TB_TARGET_PCT,
                'max_hold': TB_MAX_HOLD,
                'spread_cost': TB_SPREAD_COST,
                'commission': TB_COMMISSION,
            },
            'direction_signal': 'session_range_pct > train_median -> call, else put',
            'sr_median_threshold': sr_median,
        },
    }

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"\nSaving to {output_path}...")
    torch.save(dataset, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Features: {X_combined.shape[1]}")
    print(f"  Fingerprint: {fingerprint}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Build v2 dataset with wide grid + honest labels")
    parser.add_argument("--existing", type=str, default=EXISTING_DATA)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    build_dataset(existing_path=args.existing, output_path=args.output)


if __name__ == "__main__":
    main()
