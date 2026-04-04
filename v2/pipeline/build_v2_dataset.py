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

# Risk parameter grid: the labeler searches these combos per bar and picks the best.
# All values within DecisionPolicy sigmoid bounds:
#   stop: [0.10, 0.65], target: [0.15, 1.65], hold: [10, 390]
RISK_GRID_STOPS = [0.30, 0.40, 0.50, 0.60]
RISK_GRID_TARGETS = [0.20, 0.30, 0.50, 0.80]
RISK_GRID_HOLDS = [10, 15, 30, 45]
# Total combos: 4 x 4 x 4 = 64 per bar

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
    label_stop_pct = np.zeros(N, dtype=np.float32)
    label_target_pct = np.zeros(N, dtype=np.float32)
    label_max_hold = np.zeros(N, dtype=np.int32)
    label_confidence = np.zeros(N, dtype=np.float32)

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
    total_signal_bars = 0

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

        # ===== PASS 2: Risk-grid labeling (same-contract, search best combo) =====
        # For each entry bar: commit direction (causal), pick nearest-ATM strike,
        # search ALL risk combos, label with the best P&L combo.
        # Gate = True ONLY if best combo is profitable.

        for local_i in range(n_align):
            gi = global_indices[local_i]
            bod = int(bar_of_day[gi])

            if bod < 30 or bod >= 270:
                continue

            feats = day_features_computed.get(gi, {})
            total_vol = feats.get('near_atm_total_volume', 0)
            if total_vol < MIN_VOLUME:
                continue

            # Direction from volatility regime (causal, no lookahead)
            if sr_idx is not None:
                sr_val = X_old[gi, sr_idx]
                if np.isnan(sr_val):
                    continue
                if abs(sr_val - sr_median) < sr_iqr * 0.5:
                    continue
                direction = 0 if sr_val > sr_median else 1
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

            cost_frac = 1.60 / (entry_px * 100)

            # Pre-fetch the price series for this strike (avoids re-looking up per combo)
            max_look = max(RISK_GRID_HOLDS) + 2
            price_series = []
            for k in range(max_look):
                cl = fill_local + k
                if cl >= n_align or cl >= len(wide_timestamps):
                    break
                cgi = global_indices[cl] if cl < len(global_indices) else None
                if cgi is None or dates[cgi] != day:
                    break
                cts = wide_timestamps[cl]
                cbd = wide['bars'].get(cts, {})
                csd = cbd.get(entry_strike, {})
                px = csd.get(f'{side}_close', np.nan)
                price_series.append(px if px and not np.isnan(px) and px > 0 else np.nan)

            if len(price_series) < 3:
                continue

            # Search ALL risk combos
            best_pnl = float('-inf')
            best_stop = 0.0
            best_target = 0.0
            best_hold = 0
            best_outcome = 0
            n_profitable = 0

            for stop in RISK_GRID_STOPS:
                for target in RISK_GRID_TARGETS:
                    for hold in RISK_GRID_HOLDS:
                        if hold + 1 >= len(price_series):
                            continue

                        last_px = entry_px
                        outcome = 0
                        pnl = 0.0

                        for k in range(1, hold + 1):
                            if k >= len(price_series):
                                break
                            px = price_series[k]
                            if np.isnan(px):
                                continue
                            last_px = px
                            unr = (px - entry_px) / entry_px

                            if unr <= -stop:
                                pnl = -stop - cost_frac
                                outcome = -1
                                break
                            if unr >= target:
                                pnl = target - cost_frac
                                outcome = 1
                                break

                        if outcome == 0:
                            raw = (last_px - entry_px) / entry_px
                            pnl = raw - cost_frac
                            outcome = 1 if pnl > 0 else -1

                        if pnl > 0:
                            n_profitable += 1

                        if pnl > best_pnl:
                            best_pnl = pnl
                            best_stop = stop
                            best_target = target
                            best_hold = hold
                            best_outcome = outcome

            # Gate: True ONLY if best combo is profitable
            is_trade = best_pnl > 0 and best_outcome == 1
            total_signal_bars += 1

            label_direction[gi] = direction
            label_pnl[gi] = best_pnl
            label_stop_pct[gi] = best_stop
            label_target_pct[gi] = best_target
            label_max_hold[gi] = best_hold
            label_outcome[gi] = best_outcome
            label_confidence[gi] = n_profitable / 64.0  # fraction of combos profitable

            if is_trade:
                label_trade[gi] = True
                total_trades += 1
                total_wins += 1
            else:
                label_trade[gi] = False
                total_losses += 1

        processed_days += 1
        if (day_idx + 1) % 100 == 0:
            print(f"  {day_idx+1}/{len(unique_dates)} days, "
                  f"{total_trades:,} trades ({total_wins} W / {total_losses} L)")

    elapsed = time.time() - t0
    print(f"\nDone: {processed_days} days in {elapsed:.1f}s")

    # --- Label statistics ---
    tradeable_bars = ((bar_of_day >= 30) & (bar_of_day < 270)).sum()
    gate_true_rate = label_trade.sum() / total_signal_bars if total_signal_bars > 0 else 0
    pnls_trade = label_pnl[label_trade]
    pnls_all = label_pnl[label_pnl != 0]  # all bars that got a P&L (trade or not)
    mean_pnl_trade = pnls_trade.mean() if len(pnls_trade) > 0 else 0

    # Risk param diversity
    stops_unique = len(set(label_stop_pct[label_trade].tolist())) if label_trade.sum() > 0 else 0
    targets_unique = len(set(label_target_pct[label_trade].tolist())) if label_trade.sum() > 0 else 0
    holds_unique = len(set(label_max_hold[label_trade].tolist())) if label_trade.sum() > 0 else 0

    print(f"\n=== LABEL STATISTICS ===")
    print(f"Signal bars (passed filters): {total_signal_bars:,}")
    print(f"Gate=True (best combo profitable): {label_trade.sum():,} ({gate_true_rate*100:.1f}% of signal bars)")
    print(f"Gate=False (all combos lose): {total_losses:,} ({total_losses/max(total_signal_bars,1)*100:.1f}%)")
    print(f"Mean P&L (trade=True bars): {mean_pnl_trade*100:.2f}%")
    print(f"Mean P&L (all signal bars): {pnls_all.mean()*100:.2f}%" if len(pnls_all) > 0 else "")
    if len(pnls_trade) > 0:
        gp = pnls_trade[pnls_trade > 0].sum()
        gl = abs(pnls_trade[pnls_trade < 0].sum())
        pf = gp / gl if gl > 0 else float('inf')
        print(f"PF (trade=True): {pf:.3f}")
    print(f"Risk diversity: {stops_unique} stops, {targets_unique} targets, {holds_unique} holds")
    print(f"Avg confidence: {label_confidence[label_trade].mean():.3f}" if label_trade.sum() > 0 else "")

    # --- Validation gates ---
    errors = []
    if total_signal_bars > 0 and total_losses / total_signal_bars < 0.20:
        errors.append(f"Gate=False rate {total_losses/total_signal_bars:.1%} < 20% -- gate not selective enough")
    if gate_true_rate > 0.70:
        errors.append(f"Gate=True rate {gate_true_rate:.1%} > 70% -- gate not selective enough")
    if stops_unique < 3:
        errors.append(f"Only {stops_unique} unique stop values -- risk head can't learn diversity")
    if targets_unique < 3:
        errors.append(f"Only {targets_unique} unique target values -- risk head can't learn diversity")

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
    fp_str = f"wide_grid_risk:features:{X_combined.shape}:signals:{total_signal_bars}:trades:{total_trades}:gate_rate:{gate_true_rate:.4f}"
    fingerprint = hashlib.sha256(fp_str.encode()).hexdigest()[:16]

    # --- Build output ---
    dataset = {
        'X': torch.from_numpy(X_combined),
        'feature_names': all_feature_names,

        # Labels (risk-grid search)
        'label_trade': torch.from_numpy(label_trade),
        'label_direction': torch.from_numpy(label_direction),
        'label_outcome': torch.from_numpy(label_outcome),
        'label_pnl': torch.from_numpy(label_pnl),
        'label_stop_pct': torch.from_numpy(label_stop_pct),
        'label_target_pct': torch.from_numpy(label_target_pct),
        'label_max_hold': torch.from_numpy(label_max_hold.astype(np.int32)),
        'label_confidence': torch.from_numpy(label_confidence),

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
            'version': 'v2_wide_grid_risk_search',
            'build_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'fingerprint': fingerprint,
            'n_features': X_combined.shape[1],
            'n_old_features': n_old_features,
            'n_new_features': n_new,
            'total_signal_bars': total_signal_bars,
            'total_trades': total_trades,
            'gate_true_rate': gate_true_rate,
            'gate_false_count': total_losses,
            'mean_pnl_trade': float(mean_pnl_trade),
            'risk_stops': RISK_GRID_STOPS,
            'risk_targets': RISK_GRID_TARGETS,
            'risk_holds': RISK_GRID_HOLDS,
            'split': split_info,
            'cost_model': {
                'spread_rt': 0.30,
                'commission_rt': 1.30,
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
