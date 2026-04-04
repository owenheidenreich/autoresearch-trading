"""Extract full OHLCV per strike from raw Polygon cache.

The existing pipeline (v1 prepare.py) threw away high, low, volume per strike
and only kept close prices. This extractor recovers ALL fields from the 999-day
raw cache, adding:
- Per-strike high/low (bid-ask proxy: the traded range within 1-min bars)
- Per-strike volume (real per-contract activity, not aggregate SPY)
- Absolute strike prices (not just relative offsets)
- Real-time SPX estimated via call-put parity (fixes moneyness drift)

Usage:
    python -m v2.pipeline.extract_raw [--cache-dir ~/.cache/autoresearch-trading/data]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch


# Strike offsets tracked in the chain cache
CHAIN_PREFIXES = ['otm5', 'otm10', 'otm15', 'otm20', 'otm25', 'otm30']
SIDES = ['call', 'put']
OHLCV_FIELDS = ['open', 'high', 'low', 'close', 'volume']


def _ts_to_bar_of_day(ts_ms: int) -> int:
    """Convert Polygon millisecond timestamp to bar-of-day index (0-389)."""
    # Market open = 9:30 ET = 13:30 UTC = 48600 seconds from midnight UTC
    # But timestamps may be in ET already depending on Polygon config
    # The safest approach: compute relative to the first bar of the day
    # This is handled by the caller grouping bars per day
    return -1  # placeholder, computed by caller


def load_day_atm(path: str) -> dict:
    """Load ATM cache file, return {bar_index: {field: value}}."""
    data = pickle.load(open(path, 'rb'))
    # Keys are (date_str, timestamp_ms), sorted by timestamp
    bars = {}
    sorted_keys = sorted(data.keys(), key=lambda k: k[1])
    for i, key in enumerate(sorted_keys):
        bar = data[key]
        bars[i] = {
            'atm_strike': bar.get('strike', np.nan),
            'atm_call_open': bar.get('call_open', np.nan),
            'atm_call_high': bar.get('call_high', np.nan),
            'atm_call_low': bar.get('call_low', np.nan),
            'atm_call_close': bar.get('call_close', np.nan),
            'atm_call_volume': bar.get('call_volume', 0),
            'atm_put_open': bar.get('put_open', np.nan),
            'atm_put_high': bar.get('put_high', np.nan),
            'atm_put_low': bar.get('put_low', np.nan),
            'atm_put_close': bar.get('put_close', np.nan),
            'atm_put_volume': bar.get('put_volume', 0),
        }
    return bars


def load_day_chain(path: str) -> dict:
    """Load chain cache file, return {bar_index: {field: value}}."""
    data = pickle.load(open(path, 'rb'))
    bars = {}
    sorted_keys = sorted(data.keys(), key=lambda k: k[1])
    for i, key in enumerate(sorted_keys):
        bar = data[key]
        row = {'atm_strike_chain': bar.get('atm_strike', np.nan)}
        for prefix in CHAIN_PREFIXES:
            for side in SIDES:
                tag = f'{prefix}_{side}'
                for field in OHLCV_FIELDS:
                    row[f'{tag}_{field}'] = bar.get(f'{tag}_{field}', np.nan)
                row[f'{tag}_strike'] = bar.get(f'{tag}_strike', np.nan)
        bars[i] = row
    return bars


def compute_spx_from_parity(
    atm_strike: float,
    call_close: float,
    put_close: float,
) -> float:
    """Estimate current SPX from call-put parity: S = K + C - P."""
    if np.isnan(atm_strike) or np.isnan(call_close) or np.isnan(put_close):
        return np.nan
    if call_close <= 0 or put_close <= 0:
        return np.nan
    return atm_strike + call_close - put_close


def compute_moneyness(strike: float, spx_est: float) -> float:
    """Moneyness = (strike - SPX) / SPX. Positive = OTM call, negative = ITM call."""
    if np.isnan(strike) or np.isnan(spx_est) or spx_est <= 0:
        return np.nan
    return (strike - spx_est) / spx_est


def extract_all_days(
    cache_dir: str,
    existing_data_path: str = 'v2/data.pt',
) -> dict:
    """Extract full OHLCV from raw cache and merge with existing features.

    Returns a dict ready to be saved as the new data.pt.
    """
    atm_dir = os.path.join(cache_dir, 'spxw')
    chain_dir = os.path.join(cache_dir, 'spxw_chain')

    # Load existing data.pt for features and metadata
    print(f"Loading existing data from {existing_data_path}...")
    existing = torch.load(existing_data_path, map_location='cpu', weights_only=False)
    features = existing['X'].numpy()
    dates = existing['dates']
    bar_of_day = existing['bar_of_day'].numpy()
    N = len(dates)

    print(f"  {N:,} bars, {len(set(dates))} days")

    # Build day -> global bar index mapping
    day_to_global = defaultdict(list)
    for i, d in enumerate(dates):
        day_to_global[d].append(i)

    # Output arrays for new fields
    # ATM OHLCV
    atm_fields = {}
    for side in SIDES:
        for field in OHLCV_FIELDS:
            atm_fields[f'atm_{side}_{field}'] = np.full(N, np.nan, dtype=np.float32)
    atm_fields['atm_strike'] = np.full(N, np.nan, dtype=np.float32)

    # Chain OHLCV + strikes
    chain_fields = {}
    for prefix in CHAIN_PREFIXES:
        for side in SIDES:
            tag = f'{prefix}_{side}'
            for field in OHLCV_FIELDS:
                chain_fields[f'{tag}_{field}'] = np.full(N, np.nan, dtype=np.float32)
            chain_fields[f'{tag}_strike'] = np.full(N, np.nan, dtype=np.float32)

    # Derived fields
    spx_estimated = np.full(N, np.nan, dtype=np.float32)
    atm_moneyness = np.full(N, np.nan, dtype=np.float32)

    # Per-strike moneyness (how far each tracked strike is from current SPX)
    for prefix in CHAIN_PREFIXES:
        for side in SIDES:
            tag = f'{prefix}_{side}'
            chain_fields[f'{tag}_moneyness'] = np.full(N, np.nan, dtype=np.float32)

    # Process each day
    unique_dates = sorted(set(dates))
    atm_files = set(os.listdir(atm_dir)) if os.path.exists(atm_dir) else set()
    chain_files = set(os.listdir(chain_dir)) if os.path.exists(chain_dir) else set()

    t0 = time.time()
    processed = 0
    missing_atm = 0
    missing_chain = 0

    for day_idx, day in enumerate(unique_dates):
        global_indices = day_to_global[day]
        n_bars_day = len(global_indices)

        # Load ATM data
        atm_fname = f'{day}.pkl'
        if atm_fname in atm_files:
            atm_bars = load_day_atm(os.path.join(atm_dir, atm_fname))
            for local_i, global_i in enumerate(global_indices):
                if local_i not in atm_bars:
                    continue
                bar = atm_bars[local_i]
                atm_fields['atm_strike'][global_i] = bar['atm_strike']
                for side in SIDES:
                    for field in OHLCV_FIELDS:
                        key = f'atm_{side}_{field}'
                        val = bar.get(key, np.nan)
                        if val is not None:
                            atm_fields[key][global_i] = float(val) if val == val else np.nan

                # Compute SPX from call-put parity
                spx = compute_spx_from_parity(
                    bar['atm_strike'],
                    bar.get('atm_call_close', np.nan),
                    bar.get('atm_put_close', np.nan),
                )
                spx_estimated[global_i] = spx

                # ATM moneyness (how far the fixed opening strike is from current SPX)
                atm_moneyness[global_i] = compute_moneyness(bar['atm_strike'], spx)
        else:
            missing_atm += 1

        # Load chain data
        chain_fname = f'{day}.pkl'
        if chain_fname in chain_files:
            chain_bars = load_day_chain(os.path.join(chain_dir, chain_fname))
            for local_i, global_i in enumerate(global_indices):
                if local_i not in chain_bars:
                    continue
                bar = chain_bars[local_i]
                for prefix in CHAIN_PREFIXES:
                    for side in SIDES:
                        tag = f'{prefix}_{side}'
                        for field in OHLCV_FIELDS:
                            key = f'{tag}_{field}'
                            val = bar.get(key, np.nan)
                            if val is not None:
                                chain_fields[key][global_i] = float(val) if val == val else np.nan
                        strike_val = bar.get(f'{tag}_strike', np.nan)
                        if strike_val is not None:
                            chain_fields[f'{tag}_strike'][global_i] = float(strike_val) if strike_val == strike_val else np.nan

                        # Per-strike moneyness
                        spx = spx_estimated[global_i]
                        chain_fields[f'{tag}_moneyness'][global_i] = compute_moneyness(
                            chain_fields[f'{tag}_strike'][global_i], spx
                        )
        else:
            missing_chain += 1

        processed += 1
        if processed % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {processed}/{len(unique_dates)} days ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {processed} days in {elapsed:.1f}s")
    print(f"  Missing ATM cache: {missing_atm} days")
    print(f"  Missing chain cache: {missing_chain} days")

    # Coverage report
    spx_valid = (~np.isnan(spx_estimated)).sum()
    print(f"\n=== COVERAGE ===")
    print(f"  SPX estimated (call-put parity): {spx_valid:,}/{N:,} ({spx_valid/N*100:.1f}%)")
    for key in ['atm_call_volume', 'atm_put_volume']:
        arr = atm_fields[key]
        valid = (~np.isnan(arr)).sum()
        print(f"  {key}: {valid:,}/{N:,} ({valid/N*100:.1f}%)")
    for key in ['atm_call_high', 'atm_call_low']:
        arr = atm_fields[key]
        valid = (~np.isnan(arr)) & (arr > 0)
        print(f"  {key}: {valid.sum():,}/{N:,} ({valid.sum()/N*100:.1f}%)")

    # Build output
    result = {
        # Carry forward existing features and metadata
        'X': existing['X'],
        'feature_names': existing.get('feature_names', []),
        'dates': dates,
        'bar_of_day': existing['bar_of_day'],
        'train_mask': existing['train_mask'],
        'val_mask': existing['val_mask'],
        'promote_mask': existing.get('promote_mask', existing['val_mask']),
        'shadow_mask': existing.get('shadow_mask', torch.zeros(N, dtype=torch.bool)),

        # NEW: Real-time SPX estimate and moneyness
        'spx_estimated': torch.from_numpy(spx_estimated),
        'atm_moneyness': torch.from_numpy(atm_moneyness),

        # NEW: Full ATM OHLCV
        **{k: torch.from_numpy(v) for k, v in atm_fields.items()},

        # NEW: Full chain OHLCV + strikes + moneyness
        **{k: torch.from_numpy(v) for k, v in chain_fields.items()},

        # Carry forward auxiliary labels (pred_return_*)
        **{k: existing[k] for k in existing if k.startswith('pred_')},

        # Carry forward old oracle labels (will be replaced by triple-barrier later)
        **{k: existing[k] for k in existing if k.startswith('oracle_')},

        # Old spot_prices (kept for backward compat, but spx_estimated is better)
        'spot_prices': existing['spot_prices'],

        # Metadata
        'metadata': {
            **existing.get('metadata', {}),
            'enrichment': 'raw_ohlcv_extracted',
            'enrichment_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'spx_estimated_coverage': f'{spx_valid/N*100:.1f}%',
            'new_fields': [
                'spx_estimated', 'atm_moneyness',
                'atm_*_open', 'atm_*_high', 'atm_*_low', 'atm_*_volume',
                'otm*_*_open', 'otm*_*_high', 'otm*_*_low', 'otm*_*_volume',
                'otm*_*_strike', 'otm*_*_moneyness',
            ],
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract full OHLCV from raw Polygon cache")
    parser.add_argument("--cache-dir", type=str,
                        default=os.path.expanduser("~/.cache/autoresearch-trading/data"),
                        help="Path to raw cache directory")
    parser.add_argument("--existing", type=str, default="v2/data.pt",
                        help="Existing data.pt to enrich")
    parser.add_argument("--output", type=str, default="v2/data_enriched.pt",
                        help="Output path")
    args = parser.parse_args()

    result = extract_all_days(args.cache_dir, args.existing)

    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(result, args.output)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Keys: {len(result)}")
    print("Done.")


if __name__ == "__main__":
    main()
