"""Download wide-grid SPXW 0DTE option data from Polygon S3 flat files.

Extracts ATM +/- 100pt (41 strikes per side, 82 contracts) instead of
the narrow 14-contract grid in v1. Stores by ABSOLUTE strike with full
OHLCV per bar, so moneyness can be computed relative to current SPX.

The Polygon flat files contain ALL SPXW strikes traded each day (209+
unique strikes). We filter to the 82 nearest ATM to keep storage manageable
while providing coverage after 50pt intraday moves.

Usage:
    python -m v2.pipeline.download_wide_grid [--days 999] [--radius 100]
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np


CACHE_DIR = os.path.expanduser("~/.cache/autoresearch-trading/data")
WIDE_CACHE_DIR = os.path.join(CACHE_DIR, "spxw_wide")
SPX_CACHE = os.path.join(CACHE_DIR, "spx_1min.pkl")
SPY_CACHE = os.path.join(CACHE_DIR, "spy_1min.pkl")

# Strike grid: ATM +/- RADIUS in 5pt steps
DEFAULT_RADIUS = 100  # points from ATM
STRIKE_STEP = 5


def _load_env() -> dict:
    """Load .env file for Polygon credentials."""
    env = {}
    for path in ['.env', os.path.join(os.path.dirname(__file__), '../../.env')]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        k, v = line.split('=', 1)
                        env[k.strip()] = v.strip()
    return env


def _s3_client():
    """Create S3 client for Polygon flat files."""
    import boto3
    env = _load_env()
    key_id = env.get('POLYGON_S3_KEY_ID')
    secret = env.get('POLYGON_S3_SECRET')
    endpoint = env.get('POLYGON_S3_ENDPOINT', 'https://files.massive.com')
    if not key_id or not secret:
        return None
    return boto3.client('s3', endpoint_url=endpoint,
                        aws_access_key_id=key_id, aws_secret_access_key=secret)


def _get_trading_days() -> list[str]:
    """Get list of trading days from existing SPX/SPY cache."""
    for cache in [SPX_CACHE, SPY_CACHE]:
        if os.path.exists(cache):
            import pandas as pd
            df = pickle.load(open(cache, 'rb'))
            if isinstance(df, pd.DataFrame) and 'date' in df.columns:
                return sorted(df['date'].unique().tolist())
    # Fallback: use existing narrow cache dates
    narrow_dir = os.path.join(CACHE_DIR, 'spxw')
    if os.path.exists(narrow_dir):
        return sorted(f.replace('.pkl', '') for f in os.listdir(narrow_dir) if f.endswith('.pkl'))
    return []


def _get_opening_atm(day_str: str) -> float | None:
    """Get opening ATM strike for a day from existing cache."""
    narrow_path = os.path.join(CACHE_DIR, 'spxw', f'{day_str}.pkl')
    if os.path.exists(narrow_path):
        data = pickle.load(open(narrow_path, 'rb'))
        if data:
            first_bar = data[sorted(data.keys())[0]]
            strike = first_bar.get('strike')
            if strike and not np.isnan(strike):
                return float(strike)
    return None


def _is_0dte_day(day_str: str) -> bool:
    """Check if this date had 0DTE SPXW options."""
    d = datetime.strptime(day_str, '%Y-%m-%d')
    # Before May 11, 2022: MWF only. After: daily.
    cutoff = datetime(2022, 5, 11)
    if d >= cutoff:
        return d.weekday() < 5  # Mon-Fri
    return d.weekday() in (0, 2, 4)  # Mon, Wed, Fri


def _parse_strike_from_ticker(ticker: str) -> tuple[str, float] | None:
    """Extract (call/put, strike) from OPRA ticker.

    Example: O:SPXW260326C06555000 -> ('C', 6555.0)
    """
    # Find C or P after the date portion
    # Format: O:SPXW{YYMMDD}{C|P}{strike*1000 zero-padded}
    # or O:SPX{YYMMDD}{C|P}{strike*1000}
    try:
        # Strip O: prefix
        tk = ticker[2:] if ticker.startswith('O:') else ticker
        # Find the C or P
        for i in range(len(tk) - 1, 5, -1):
            if tk[i - 1] in 'CP' and tk[i:].isdigit():
                cp = tk[i - 1]
                strike = int(tk[i:]) / 1000
                return (cp, strike)
    except (ValueError, IndexError):
        pass
    return None


def download_day(
    s3,
    day_str: str,
    atm_strike: float,
    radius: int = DEFAULT_RADIUS,
) -> dict | None:
    """Download and extract wide-grid data for one day.

    Returns dict: {bar_timestamp_ms: {strike: {call_open, call_high, ...}}}
    """
    day_dt = datetime.strptime(day_str, '%Y-%m-%d')
    yymmdd = day_dt.strftime('%y%m%d')
    s3_key = f"us_options_opra/minute_aggs_v1/{day_dt.year}/{day_dt.month:02d}/{day_str}.csv.gz"

    # Target strikes: ATM +/- radius in STRIKE_STEP increments
    min_strike = atm_strike - radius
    max_strike = atm_strike + radius
    target_strikes = set(
        atm_strike + i * STRIKE_STEP
        for i in range(-radius // STRIKE_STEP, radius // STRIKE_STEP + 1)
    )

    # Target ticker prefixes
    target_prefix_w = f"O:SPXW{yymmdd}"
    target_prefix_spx = f"O:SPX{yymmdd}"

    try:
        obj = s3.get_object(Bucket='flatfiles', Key=s3_key)
        raw = gzip.decompress(obj['Body'].read())
    except Exception:
        return None

    # Parse: ticker,volume,open,close,high,low,window_start,transactions
    # Group by timestamp -> strike -> OHLCV
    bars = defaultdict(lambda: defaultdict(dict))

    for line in raw.decode('utf-8', errors='replace').split('\n'):
        if not line:
            continue
        if not (line.startswith(target_prefix_w) or line.startswith(target_prefix_spx)):
            continue

        parts = line.split(',')
        if len(parts) < 8:
            continue

        ticker = parts[0]
        parsed = _parse_strike_from_ticker(ticker)
        if not parsed:
            continue
        cp, strike = parsed

        # Filter to target strike range
        if strike < min_strike or strike > max_strike:
            continue
        # Snap to 5pt grid
        if strike % STRIKE_STEP != 0:
            continue

        try:
            volume = int(parts[1]) if parts[1] else 0
            o = float(parts[2]) if parts[2] else np.nan
            c = float(parts[3]) if parts[3] else np.nan
            h = float(parts[4]) if parts[4] else np.nan
            l = float(parts[5]) if parts[5] else np.nan
            ts = int(parts[6]) if parts[6] else 0
            transactions = int(parts[7]) if parts[7] else 0
        except (ValueError, IndexError):
            continue

        if ts == 0:
            continue

        side = 'call' if cp == 'C' else 'put'
        bar = bars[ts][strike]
        bar[f'{side}_open'] = o
        bar[f'{side}_close'] = c
        bar[f'{side}_high'] = h
        bar[f'{side}_low'] = l
        bar[f'{side}_volume'] = volume
        bar[f'{side}_transactions'] = transactions

    if not bars:
        return None

    # Convert to final format with metadata
    result = {
        'atm_strike': atm_strike,
        'radius': radius,
        'day': day_str,
        'bars': dict(bars),  # {ts: {strike: {field: value}}}
        'strikes': sorted(set(
            strike for ts_data in bars.values() for strike in ts_data.keys()
        )),
    }
    return result


def run_download(
    max_days: int | None = None,
    radius: int = DEFAULT_RADIUS,
    force: bool = False,
):
    """Download wide-grid data for all trading days."""
    s3 = _s3_client()
    if s3 is None:
        print("ERROR: No Polygon S3 credentials. Set POLYGON_S3_KEY_ID + POLYGON_S3_SECRET in .env")
        return

    os.makedirs(WIDE_CACHE_DIR, exist_ok=True)

    trading_days = _get_trading_days()
    if not trading_days:
        print("ERROR: No trading days found. Run v1 prepare.py first or check cache.")
        return

    # Filter to 0DTE days
    trading_days = [d for d in trading_days if _is_0dte_day(d)]

    if max_days:
        trading_days = trading_days[-max_days:]

    # Skip already cached
    if not force:
        needed = []
        for day in trading_days:
            cache_path = os.path.join(WIDE_CACHE_DIR, f"{day}.pkl")
            if not os.path.exists(cache_path):
                needed.append(day)
        print(f"Total 0DTE days: {len(trading_days)}")
        print(f"Already cached: {len(trading_days) - len(needed)}")
        print(f"Need to download: {len(needed)}")
        trading_days = needed

    if not trading_days:
        print("All days already cached.")
        return

    print(f"Downloading {len(trading_days)} days with ATM +/- {radius}pt grid...")
    t0 = time.time()
    downloaded = 0
    failed = 0
    total_strikes = 0

    for i, day in enumerate(trading_days):
        atm = _get_opening_atm(day)
        if atm is None:
            # Try to estimate from SPX cache
            atm = 5500.0  # fallback, will be overridden by actual data
            failed += 1
            continue

        result = download_day(s3, day, atm, radius)

        if result is None:
            # Cache empty result so we don't retry
            cache_path = os.path.join(WIDE_CACHE_DIR, f"{day}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump({'atm_strike': atm, 'bars': {}, 'strikes': [], 'day': day, 'radius': radius}, f)
            failed += 1
        else:
            cache_path = os.path.join(WIDE_CACHE_DIR, f"{day}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            downloaded += 1
            total_strikes += len(result['strikes'])

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(trading_days) - i - 1) / rate
            print(f"  {i+1}/{len(trading_days)} ({downloaded} OK, {failed} failed) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t0
    avg_strikes = total_strikes / max(downloaded, 1)
    print(f"\nDone: {downloaded} downloaded, {failed} failed in {elapsed:.0f}s")
    print(f"Average strikes per day: {avg_strikes:.0f}")
    print(f"Cache: {WIDE_CACHE_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Download wide-grid SPXW data from Polygon")
    parser.add_argument("--days", type=int, default=None,
                        help="Limit to last N trading days")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                        help=f"Points from ATM (default {DEFAULT_RADIUS})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached")
    args = parser.parse_args()

    run_download(max_days=args.days, radius=args.radius, force=args.force)


if __name__ == "__main__":
    main()
