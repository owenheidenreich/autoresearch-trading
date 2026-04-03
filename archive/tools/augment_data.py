#!/usr/bin/env python3
"""Augment existing data.pt with tournament features WITHOUT full rebuild.

Loads 38-feature data.pt, computes ONLY the new features, appends columns.
Much faster than full prepare.py rebuild (~1-2 min vs ~40 min).

Usage:
  python3 tools/augment_data.py --features vix_roc
  python3 tools/augment_data.py --features vix_roc,overnight_gap,event_day
  python3 tools/augment_data.py --output /tmp/data_39.pt --features vix_roc
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from prepare import (
    BARS_PER_DAY, _ECON_EVENT_DATES, _bs_iv, _bs_greeks, _safe_float,
    OTM_STRIKE_STEPS, _TOURNAMENT_FEATURES,
)

CACHE_DIR = os.path.expanduser("~/.cache/autoresearch-trading")
DATA_DIR = os.path.join(CACHE_DIR, "data")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")


def load_source_data():
    """Load cached source data needed for feature computation."""
    # SPY/SPX bars
    spx_path = os.path.join(DATA_DIR, "spx_1min.pkl")
    spy_path = os.path.join(DATA_DIR, "spy_1min.pkl")
    if os.path.exists(spx_path):
        with open(spx_path, "rb") as f:
            df = pickle.load(f)
        print(f"  SPX bars: {len(df)}")
    else:
        with open(spy_path, "rb") as f:
            df = pickle.load(f)
        print(f"  SPY bars: {len(df)}")

    # VIX
    vix_path = os.path.join(DATA_DIR, "vix_1min.pkl")
    vix_data = {}
    if os.path.exists(vix_path):
        with open(vix_path, "rb") as f:
            vix_df = pickle.load(f)
        for _, row in vix_df.iterrows():
            vix_data[row['timestamp']] = {
                'vix_close': row.get('vix_close', row.get('close', np.nan)),
            }
        print(f"  VIX bars: {len(vix_data)}")

    # Options
    opts_path = os.path.join(DATA_DIR, "spxw_full.pkl")
    options_data = {}
    if os.path.exists(opts_path):
        with open(opts_path, "rb") as f:
            options_data = pickle.load(f)
        print(f"  Options: {len(options_data)} entries")

    # Chain
    chain_path = os.path.join(DATA_DIR, "spxw_chain_full.pkl")
    chain_data = {}
    if os.path.exists(chain_path):
        with open(chain_path, "rb") as f:
            chain_data = pickle.load(f)
        print(f"  Chain: {len(chain_data)} entries")

    return df, vix_data, options_data, chain_data


def compute_feature_column(feature_name: str, data_pt: dict, df, vix_data, options_data, chain_data) -> np.ndarray:
    """Compute a single feature column for all bars."""
    N = data_pt['features'].shape[0]
    col = np.full(N, np.nan, dtype=np.float32)

    # We need dates and timestamps aligned with data.pt
    dates = data_pt['dates']
    if isinstance(dates, torch.Tensor):
        dates = dates.tolist()
    ts_strings = data_pt.get('timestamps', dates)
    if isinstance(ts_strings, torch.Tensor):
        ts_strings = ts_strings.tolist()

    # Convert human-readable timestamps ('2022-03-16 09:30') to millisecond timestamps
    # for option/chain data lookup
    from datetime import datetime as _dt
    import pytz
    _et = pytz.timezone('US/Eastern')
    ms_timestamps = np.zeros(N, dtype=np.int64)
    for i in range(N):
        try:
            if len(ts_strings[i]) > 10:  # has time component
                dt = _et.localize(_dt.strptime(ts_strings[i], '%Y-%m-%d %H:%M'))
            else:
                dt = _et.localize(_dt.strptime(ts_strings[i] + ' 09:30', '%Y-%m-%d %H:%M'))
            ms_timestamps[i] = int(dt.timestamp() * 1000)
        except (ValueError, IndexError):
            ms_timestamps[i] = 0

    # Build day indices
    from collections import defaultdict
    day_indices = defaultdict(list)
    for i, d in enumerate(dates):
        day_indices[d].append(i)

    # Build aligned OHLC arrays from SPX data indexed by (date, ms_timestamp)
    # Create a lookup from the raw SPX dataframe
    if 'spx_close' in df.columns:
        _close_col, _open_col, _high_col, _low_col = 'spx_close', 'spx_open', 'spx_high', 'spx_low'
    else:
        _close_col, _open_col, _high_col, _low_col = 'close', 'open', 'high', 'low'

    _spx_lookup = {}
    for _, row in df.iterrows():
        _spx_lookup[int(row['timestamp'])] = {
            'close': float(row[_close_col]),
            'open': float(row[_open_col]),
            'high': float(row[_high_col]),
            'low': float(row[_low_col]),
        }

    close = np.zeros(N, dtype=np.float64)
    opn = np.zeros(N, dtype=np.float64)
    high = np.zeros(N, dtype=np.float64)
    low = np.zeros(N, dtype=np.float64)
    for i in range(N):
        bar = _spx_lookup.get(int(ms_timestamps[i]))
        if bar is not None:
            close[i] = bar['close']
            opn[i] = bar['open']
            high[i] = bar['high']
            low[i] = bar['low']

    timestamps = ms_timestamps

    print(f"  Computing {feature_name} for {N} bars...")
    t0 = time.time()

    if feature_name == 'vix_roc':
        for i in range(N):
            ts = int(timestamps[i])
            vix_bar = vix_data.get(ts)
            if vix_bar is None:
                continue
            vix_now = vix_bar['vix_close']
            if np.isnan(vix_now):
                continue
            # 10-bar lookback
            if i >= 10 and dates[i - 10] == dates[i]:
                ts_prev = int(timestamps[i - 10])
                vix_prev = vix_data.get(ts_prev)
                if vix_prev is not None:
                    vp = vix_prev['vix_close']
                    if not np.isnan(vp) and vp > 0:
                        col[i] = (vix_now - vp) / vp

    elif feature_name == 'overnight_gap':
        # Build prev_day_close
        unique_dates = list(dict.fromkeys(dates))  # ordered unique
        prev_close = {}
        for di, day in enumerate(unique_dates):
            if di > 0:
                prev_day = unique_dates[di - 1]
                pidx = day_indices[prev_day]
                prev_close[day] = close[pidx[-1]]

        for i in range(N):
            day = dates[i]
            if day not in prev_close:
                continue
            pc = prev_close[day]
            if pc <= 0:
                continue
            day_open = opn[day_indices[day][0]]
            col[i] = (day_open - pc) / pc

    elif feature_name == 'event_day':
        for i in range(N):
            col[i] = 1.0 if dates[i] in _ECON_EVENT_DATES else 0.0

    elif feature_name == 'option_spread_width':
        for i in range(N):
            opt_key = (dates[i], int(timestamps[i]))
            opt = options_data.get(opt_key)
            if opt is None:
                continue
            ch = _safe_float(opt.get('call_high', np.nan))
            cl = _safe_float(opt.get('call_low', np.nan))
            cc = _safe_float(opt.get('call_close', np.nan))
            if np.isfinite(ch) and np.isfinite(cl) and np.isfinite(cc) and cc > 0:
                col[i] = (ch - cl) / cc

    elif feature_name == 'pc_volume_ratio':
        for i in range(N):
            opt_key = (dates[i], int(timestamps[i]))
            chain_bar = chain_data.get(opt_key)
            opt = options_data.get(opt_key)
            if chain_bar is None:
                continue
            total_put = 0.0
            total_call = 0.0
            for step in OTM_STRIKE_STEPS:
                cv = _safe_float(chain_bar.get(f'otm{step}_call_volume', 0.0), default=0.0)
                pv = _safe_float(chain_bar.get(f'otm{step}_put_volume', 0.0), default=0.0)
                total_call += cv
                total_put += pv
            if opt is not None:
                total_call += _safe_float(opt.get('call_volume', 0.0), default=0.0)
                total_put += _safe_float(opt.get('put_volume', 0.0), default=0.0)
            if total_call > 0:
                col[i] = total_put / total_call

    elif feature_name == 'gamma_pressure':
        for i in range(N):
            opt_key = (dates[i], int(timestamps[i]))
            chain_bar = chain_data.get(opt_key)
            opt = options_data.get(opt_key)
            if chain_bar is None or opt is None:
                continue
            # Compute T
            didx = day_indices[dates[i]]
            day_pos = np.searchsorted(didx, i)
            minutes_remaining = max(BARS_PER_DAY - day_pos, 1)
            T = minutes_remaining / (252.0 * 390.0)
            if T <= 1e-10:
                continue
            gp = 0.0
            gp_valid = False
            spx = close[i]
            for step in OTM_STRIKE_STEPS:
                for side, sign in [('call', 1.0), ('put', -1.0)]:
                    k = f'otm{step}_{side}'
                    px = _safe_float(chain_bar.get(f'{k}_close', np.nan))
                    vol = _safe_float(chain_bar.get(f'{k}_volume', 0.0), default=0.0)
                    strike = _safe_float(chain_bar.get(f'{k}_strike', np.nan))
                    if np.isfinite(px) and np.isfinite(strike) and px > 0 and vol > 0:
                        iv = _bs_iv(px, spx, strike, T, 0.05, is_call=(side == 'call'))
                        if np.isfinite(iv) and iv > 0:
                            _, gamma, _, _ = _bs_greeks(spx, strike, T, 0.05, iv)
                            if np.isfinite(gamma):
                                gp += gamma * vol * sign
                                gp_valid = True
            if gp_valid:
                col[i] = gp

    elif feature_name == 'iv_percentile':
        # Build rolling 60-day IV history
        unique_dates = list(dict.fromkeys(dates))
        daily_close_iv = {}
        for day in unique_dates:
            didx = day_indices[day]
            for bi in reversed(didx):
                opt_key = (day, int(timestamps[bi]))
                opt = options_data.get(opt_key)
                if opt is not None and not np.isnan(opt.get('call_close', np.nan)):
                    spx = close[bi]
                    K = opt['strike']
                    day_pos = np.searchsorted(didx, bi)
                    T = max(BARS_PER_DAY - day_pos, 1) / (252.0 * 390.0)
                    iv = _bs_iv(opt['call_close'], spx, K, T, 0.05, is_call=True)
                    if np.isfinite(iv):
                        daily_close_iv[day] = iv
                    break

        iv_history_by_day = {}
        day_list = unique_dates
        for di, day in enumerate(day_list):
            start = max(0, di - 60)
            hist = [daily_close_iv[day_list[j]] for j in range(start, di) if day_list[j] in daily_close_iv]
            if hist:
                iv_history_by_day[day] = np.array(hist)

        # Get atm_iv from existing features (index 16)
        atm_iv = data_pt['features'][:, 16].numpy()
        for i in range(N):
            if np.isnan(atm_iv[i]):
                continue
            day = dates[i]
            if day not in iv_history_by_day:
                continue
            hist = iv_history_by_day[day]
            if len(hist) >= 5:
                col[i] = float(np.sum(hist <= atm_iv[i])) / len(hist)

    elif feature_name == 'rsi_15min':
        unique_dates = list(dict.fromkeys(dates))
        for day in unique_dates:
            didx = day_indices[day]
            if len(didx) < 15:
                continue
            closes_15 = []
            bar_ranges = []
            for k in range(0, len(didx), 15):
                end = min(k + 15, len(didx))
                closes_15.append(close[didx[end - 1]])
                bar_ranges.append((didx[k], didx[end - 1]))
            if len(closes_15) < 3:
                continue
            c15 = np.array(closes_15)
            deltas = np.diff(c15)
            period = min(7, len(deltas))
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            for j in range(len(deltas)):
                if j >= period:
                    avg_gain = (avg_gain * (period - 1) + gains[j]) / period
                    avg_loss = (avg_loss * (period - 1) + losses[j]) / period
                if j >= period - 1:
                    if avg_loss == 0:
                        rsi = 1.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi = rs / (1.0 + rs)
                    bi_start, bi_end = bar_ranges[j + 1]
                    for bi in range(bi_start, bi_end + 1):
                        if bi < N:
                            col[bi] = rsi

    else:
        raise ValueError(f"Unknown feature: {feature_name}")

    valid = np.isfinite(col)
    print(f"    Done in {time.time()-t0:.1f}s. Coverage: {valid.sum()}/{N} ({valid.mean()*100:.1f}%)")
    return col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Comma-separated feature names")
    parser.add_argument("--input", default=os.path.join(FEATURES_DIR, "data.pt"))
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    args = parser.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    for f in features:
        if f not in _TOURNAMENT_FEATURES:
            print(f"Unknown feature: {f}. Available: {list(_TOURNAMENT_FEATURES.keys())}")
            sys.exit(1)

    output = args.output or args.input

    print(f"Loading data.pt from {args.input}...")
    data_pt = torch.load(args.input, weights_only=False, map_location="cpu")
    orig_shape = data_pt['features'].shape
    print(f"  Original: {orig_shape[0]} bars x {orig_shape[1]} features")

    print("Loading source data...")
    df, vix_data, options_data, chain_data = load_source_data()

    # Ensure df is aligned with data.pt
    N = orig_shape[0]
    if len(df) < N:
        print(f"WARNING: df has {len(df)} rows but data.pt has {N} bars")
        N = min(N, len(df))

    # Compute new columns
    new_cols = []
    for feat_name in features:
        col = compute_feature_column(feat_name, data_pt, df, vix_data, options_data, chain_data)
        new_cols.append(torch.tensor(col, dtype=torch.float32).unsqueeze(1))

    # Append columns
    new_features = torch.cat([data_pt['features']] + new_cols, dim=1)
    data_pt['features'] = new_features

    # Update provenance
    prov = data_pt.get('_provenance', {})
    old_names = prov.get('feature_names', [])
    prov['feature_names'] = old_names + features
    prov['num_features'] = new_features.shape[1]
    prov['tournament_features'] = features
    data_pt['_provenance'] = prov

    # Update norm buffers if present
    if 'norm_raw_buffer' in data_pt:
        raw = data_pt['norm_raw_buffer']
        # Extend with NaN columns for new features (they'll be normalized at inference)
        n_new = len(features)
        pad = torch.full((raw.shape[0], n_new), float('nan'))
        # Recompute from source for the buffer window
        for fi, feat_name in enumerate(features):
            buf_start = N - raw.shape[0]
            col = torch.tensor(
                compute_feature_column(feat_name, data_pt, df, vix_data, options_data, chain_data)[buf_start:N],
                dtype=torch.float32
            )
            pad[:len(col), fi] = col
        data_pt['norm_raw_buffer'] = torch.cat([raw, pad], dim=1)

    print(f"\nSaving to {output}...")
    print(f"  New shape: {new_features.shape[0]} bars x {new_features.shape[1]} features")
    torch.save(data_pt, output)
    print("Done!")


if __name__ == "__main__":
    main()
