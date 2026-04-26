"""One-off: extend cached pkl data through 2026-04-24 using Polygon flat files.

Pulls 17 missing trading days (2026-04-02 → 2026-04-24, skipping
weekends + 2026-04-03 Good Friday holiday) for:
  - SPXW/SPX option chains  → ~/.cache/.../spxw_full_chain/{day}.pkl
                              + spxw/{day}.pkl + spxw_wide/{day}.pkl
                              (the build pipeline reads from spxw_wide
                              for day discovery)
  - SPX, SPY, VIX minute bars → append to {symbol}_1min.pkl

Writes only to the cache; does not modify v2/data.pt or sidecars
(those are rebuilt by build_v2_dataset.py downstream).

Usage:
    .venv/bin/python -m scripts.extend_caches_late_april_2026
"""
from __future__ import annotations

import gzip
import os
import pickle
from collections import defaultdict
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd

from v2.pipeline.download_full_chain import (
    _parse_option_ticker,
    download_day,
    CACHE_DIR,
    FULL_CHAIN_DIR,
)


SPX_PATH = os.path.join(CACHE_DIR, "spx_1min.pkl")
SPY_PATH = os.path.join(CACHE_DIR, "spy_1min.pkl")
VIX_PATH = os.path.join(CACHE_DIR, "vix_1min.pkl")

SPXW_DIR = os.path.join(CACHE_DIR, "spxw")
SPXW_WIDE_DIR = os.path.join(CACHE_DIR, "spxw_wide")


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in open(os.path.join(os.path.dirname(__file__), "..", ".env")):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _s3_client():
    env = _load_env()
    return boto3.client(
        "s3",
        endpoint_url=env["POLYGON_S3_ENDPOINT"],
        aws_access_key_id=env["POLYGON_S3_KEY_ID"],
        aws_secret_access_key=env["POLYGON_S3_SECRET"],
    )


def _missing_chain_days() -> list[str]:
    """Trading days we need: every weekday 2026-04-02 → 2026-04-24
    that doesn't already have a chain pkl."""
    out = []
    start = datetime(2026, 4, 2)
    end = datetime(2026, 4, 24)
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon-Fri
            d = cur.strftime("%Y-%m-%d")
            chain_path = os.path.join(FULL_CHAIN_DIR, f"{d}.pkl")
            if not os.path.exists(chain_path):
                out.append(d)
        cur = cur.replace(day=cur.day + 1) if cur.day < 30 else datetime(cur.year, cur.month + 1, 1)
    return out


def _list_missing_chain_days() -> list[str]:
    """Find all weekdays 2026-04-02 → 2026-04-24 that aren't already cached."""
    import datetime as _dt
    out = []
    d = _dt.date(2026, 4, 2)
    end = _dt.date(2026, 4, 24)
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            ds = d.isoformat()
            chain_path = os.path.join(FULL_CHAIN_DIR, f"{ds}.pkl")
            if not os.path.exists(chain_path):
                out.append(ds)
        d += _dt.timedelta(days=1)
    return out


def _download_minute_aggs_for_symbol(s3, day_str: str, ticker: str, prefix_root: str) -> pd.DataFrame:
    """Pull a single symbol's 1-min bars from a daily flat file.

    prefix_root: 'us_stocks_sip' for stocks (SPY) or 'us_indices' for indices (SPX, VIX, NDX, etc.)
    ticker: full Polygon ticker, e.g. 'SPY' for SPY, 'I:SPX' for SPX, 'I:VIX' for VIX.
    """
    day_dt = datetime.strptime(day_str, "%Y-%m-%d")
    s3_key = f"{prefix_root}/minute_aggs_v1/{day_dt.year}/{day_dt.month:02d}/{day_str}.csv.gz"
    obj = s3.get_object(Bucket="flatfiles", Key=s3_key)
    raw = gzip.decompress(obj["Body"].read())

    matches = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line:
            continue
        parts = line.split(",")
        if not parts or parts[0] != ticker:
            continue
        try:
            volume = int(parts[1]) if parts[1] else 0
            open_px = float(parts[2]) if parts[2] else np.nan
            close_px = float(parts[3]) if parts[3] else np.nan
            high_px = float(parts[4]) if parts[4] else np.nan
            low_px = float(parts[5]) if parts[5] else np.nan
            ts = int(parts[6]) if parts[6] else 0
        except (ValueError, IndexError):
            continue
        matches.append({
            "timestamp": ts,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "volume": volume,
        })
    if not matches:
        return pd.DataFrame()
    df = pd.DataFrame(matches).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    return df


def _extend_symbol_cache(s3, days: list[str], cache_path: str, ticker: str, prefix_root: str, symbol_short: str) -> None:
    """Load existing cache, append new days, save back. Renames columns to symbol-prefixed (spx_open etc.)."""
    if os.path.exists(cache_path):
        existing = pickle.load(open(cache_path, "rb"))
    else:
        existing = pd.DataFrame()
    existing_dates = set(existing["date"].unique()) if not existing.empty else set()
    new_frames = []
    for day in days:
        if day in existing_dates:
            continue
        try:
            df = _download_minute_aggs_for_symbol(s3, day, ticker, prefix_root)
        except Exception as e:
            print(f"  {symbol_short} {day}: ERROR {e}")
            continue
        if df.empty:
            print(f"  {symbol_short} {day}: NO DATA")
            continue
        # Rename to symbol-prefixed
        rename = {
            "open": f"{symbol_short}_open",
            "high": f"{symbol_short}_high",
            "low": f"{symbol_short}_low",
            "close": f"{symbol_short}_close",
            "volume": f"{symbol_short}_volume",
        }
        df = df.rename(columns=rename)
        new_frames.append(df)
        print(f"  {symbol_short} {day}: {len(df)} bars")
    if not new_frames:
        print(f"  {symbol_short}: nothing to add")
        return
    combined = pd.concat([existing] + new_frames, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    with open(cache_path, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {symbol_short}: cache extended to {combined['date'].max()}, {len(combined)} bars total")


def _save_legacy_chain_caches(day_str: str, result: dict) -> None:
    """The build pipeline also reads from spxw/ and spxw_wide/.
    Save the same payload there to keep day-discovery working."""
    for d in (SPXW_DIR, SPXW_WIDE_DIR):
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{day_str}.pkl")
        with open(path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    s3 = _s3_client()

    missing = _list_missing_chain_days()
    print(f"Missing chain days: {len(missing)} ({missing[0] if missing else 'none'} -> "
          f"{missing[-1] if missing else 'none'})")

    # 1. SPXW/SPX option chain per missing day
    print("\n=== Downloading SPXW option chains ===")
    for i, day in enumerate(missing, 1):
        chain_path = os.path.join(FULL_CHAIN_DIR, f"{day}.pkl")
        if os.path.exists(chain_path):
            print(f"  {i}/{len(missing)} {day}: already cached")
            continue
        result = download_day(s3, day)
        if isinstance(result, dict) and "error" in result:
            print(f"  {i}/{len(missing)} {day}: ERROR {result['error']}")
            continue
        if result is None or not result.get("bars"):
            print(f"  {i}/{len(missing)} {day}: no bars")
            continue
        with open(chain_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        _save_legacy_chain_caches(day, result)
        print(f"  {i}/{len(missing)} {day}: {len(result['contracts'])} contracts, "
              f"{len(result['timestamps'])} bars")

    # 2. Underlying SPX, SPY, VIX
    print("\n=== Extending underlying SPX 1-min cache ===")
    _extend_symbol_cache(s3, missing, SPX_PATH, "I:SPX", "us_indices", "spx")

    print("\n=== Extending underlying SPY 1-min cache ===")
    _extend_symbol_cache(s3, missing, SPY_PATH, "SPY", "us_stocks_sip", "spy")

    print("\n=== Extending underlying VIX 1-min cache ===")
    _extend_symbol_cache(s3, missing, VIX_PATH, "I:VIX", "us_indices", "vix")

    print("\nDone. Next: run v2.pipeline.build_v2_dataset and "
          "v3.layer2.export_action_surface_dataset to extend the dataset.")


if __name__ == "__main__":
    main()
