"""Download full same-day SPXW 0DTE option chain data from Polygon flat files.

The resulting cache stores every observed SPX/SPXW contract on the session that
expires the same day. This replaces the opening-ATM filtered wide-grid cache as
the canonical raw option source for the v4 exact-chain harness.
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np


CACHE_DIR = os.path.expanduser("~/.cache/autoresearch-trading/data")
FULL_CHAIN_DIR = os.path.join(CACHE_DIR, "spxw_full_chain")
SPX_CACHE = os.path.join(CACHE_DIR, "spx_1min.pkl")
SPY_CACHE = os.path.join(CACHE_DIR, "spy_1min.pkl")


def _load_env() -> dict[str, str]:
    env = {}
    for path in [".env", os.path.join(os.path.dirname(__file__), "../../.env")]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip()
    return env


def _s3_client():
    import boto3

    env = _load_env()
    key_id = env.get("POLYGON_S3_KEY_ID")
    secret = env.get("POLYGON_S3_SECRET")
    endpoint = env.get("POLYGON_S3_ENDPOINT", "https://files.massive.com")
    if not key_id or not secret:
        return None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )


def _get_trading_days() -> list[str]:
    for dirname in ("spxw_full_chain", "spxw_wide", "spxw"):
        path = os.path.join(CACHE_DIR, dirname)
        if os.path.isdir(path):
            days = sorted(f.replace(".pkl", "") for f in os.listdir(path) if f.endswith(".pkl"))
            if days:
                return days
    for cache in [SPX_CACHE, SPY_CACHE]:
        if os.path.exists(cache):
            try:
                df = pickle.load(open(cache, "rb"))
            except Exception:
                continue
            if hasattr(df, "columns") and "date" in df.columns:
                return sorted(df["date"].unique().tolist())
    return []


def _is_0dte_day(day_str: str) -> bool:
    d = datetime.strptime(day_str, "%Y-%m-%d")
    cutoff = datetime(2022, 5, 11)
    if d >= cutoff:
        return d.weekday() < 5
    return d.weekday() in (0, 2, 4)


def _parse_option_ticker(ticker: str) -> tuple[str, float] | None:
    """Return (right, strike) from a Polygon OPRA ticker."""

    tk = ticker[2:] if ticker.startswith("O:") else ticker
    try:
        for i in range(len(tk) - 1, 5, -1):
            if tk[i - 1] in "CP" and tk[i:].isdigit():
                return tk[i - 1], int(tk[i:]) / 1000.0
    except (IndexError, ValueError):
        return None
    return None


def download_day(s3, day_str: str) -> dict | None:
    day_dt = datetime.strptime(day_str, "%Y-%m-%d")
    yymmdd = day_dt.strftime("%y%m%d")
    s3_key = f"us_options_opra/minute_aggs_v1/{day_dt.year}/{day_dt.month:02d}/{day_str}.csv.gz"

    try:
        obj = s3.get_object(Bucket="flatfiles", Key=s3_key)
        raw = gzip.decompress(obj["Body"].read())
    except Exception:
        return None

    prefix_spxw = f"O:SPXW{yymmdd}"
    prefix_spx = f"O:SPX{yymmdd}"

    bars: dict[int, dict[tuple[float, str], dict[str, float]]] = defaultdict(dict)
    contracts: set[tuple[float, str]] = set()

    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line:
            continue
        if not (line.startswith(prefix_spxw) or line.startswith(prefix_spx)):
            continue

        parts = line.split(",")
        if len(parts) < 8:
            continue

        parsed = _parse_option_ticker(parts[0])
        if parsed is None:
            continue
        right, strike = parsed

        try:
            volume = int(parts[1]) if parts[1] else 0
            open_px = float(parts[2]) if parts[2] else np.nan
            close_px = float(parts[3]) if parts[3] else np.nan
            high_px = float(parts[4]) if parts[4] else np.nan
            low_px = float(parts[5]) if parts[5] else np.nan
            ts = int(parts[6]) if parts[6] else 0
            transactions = int(parts[7]) if parts[7] else 0
        except (ValueError, IndexError):
            continue

        if ts <= 0 or strike <= 0 or strike % 5.0 != 0:
            continue

        bars[ts][(float(strike), right)] = {
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "volume": volume,
            "transactions": transactions,
        }
        contracts.add((float(strike), right))

    if not bars:
        return None

    return {
        "day": day_str,
        "bars": dict(bars),
        "contracts": sorted(contracts),
        "timestamps": sorted(bars.keys()),
        "source": "polygon_flatfiles_full_0dte",
    }


def run_download(max_days: int | None = None, force: bool = False) -> None:
    s3 = _s3_client()
    if s3 is None:
        raise SystemExit("Missing Polygon S3 credentials in .env")

    os.makedirs(FULL_CHAIN_DIR, exist_ok=True)

    trading_days = [d for d in _get_trading_days() if _is_0dte_day(d)]
    if max_days:
        trading_days = trading_days[-max_days:]

    needed = []
    for day in trading_days:
        cache_path = os.path.join(FULL_CHAIN_DIR, f"{day}.pkl")
        if force or not os.path.exists(cache_path):
            needed.append(day)

    print(f"Total 0DTE days: {len(trading_days)}")
    print(f"Need to download: {len(needed)}")
    if not needed:
        print("All days already cached.")
        return

    ok = 0
    failed = 0
    for i, day in enumerate(needed, start=1):
        result = download_day(s3, day)
        cache_path = os.path.join(FULL_CHAIN_DIR, f"{day}.pkl")
        if result is None:
            failed += 1
            print(f"  {i}/{len(needed)} {day}: FAILED")
            continue
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        ok += 1
        print(
            f"  {i}/{len(needed)} {day}: {len(result['contracts'])} contracts, "
            f"{len(result['timestamps'])} bars"
        )

    print(f"Done: {ok} downloaded, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Download full same-day SPXW 0DTE chain cache")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_download(max_days=args.days, force=args.force)


if __name__ == "__main__":
    main()
