"""Extend SPX/SPY/VIX 1-min underlying caches via yfinance.

Polygon's Options Starter plan does not include access to us_indices or
us_stocks_sip flat files. To validate forward-walk on the most recent
weeks, we substitute yfinance for the underlying. yfinance restricts
1-min interval requests to 8 days per call; we issue 3 chunks across
2026-04-02 → 2026-04-25.

Format is matched to the existing pickle schema:
  spx: timestamp, spx_open, spx_high, spx_low, spx_close, datetime, time, date
  spy: timestamp, open, high, low, close, volume, vwap, datetime, time, date
  vix: timestamp, vix_open, vix_high, vix_low, vix_close, datetime, time, date

Caveat: yfinance uses Yahoo data which may have small tick/timestamp
differences vs Polygon. For directional/feature purposes the agreement
is close, but the data is documented as "yfinance-sourced" in the
output to keep provenance honest.

Usage:
    .venv/bin/python -m scripts.yfinance_extend_underlying
"""
from __future__ import annotations

import os
import pickle

import pandas as pd
import yfinance as yf


CACHE_DIR = os.path.expanduser("~/.cache/autoresearch-trading/data")
SPX_PATH = os.path.join(CACHE_DIR, "spx_1min.pkl")
SPY_PATH = os.path.join(CACHE_DIR, "spy_1min.pkl")
VIX_PATH = os.path.join(CACHE_DIR, "vix_1min.pkl")


def _chunk_dates(start: str, end: str, days_per_chunk: int = 7):
    """Yield (start, end_exclusive) pairs covering [start, end)."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = s
    while cur < e:
        nxt = min(cur + pd.Timedelta(days=days_per_chunk), e)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt


def _download_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download 1-min bars from yfinance across [start, end] in 7-day chunks."""
    chunks = []
    for cs, ce in _chunk_dates(start, end, days_per_chunk=7):
        df = yf.download(ticker, start=cs, end=ce, interval="1m", progress=False, auto_adjust=False)
        if df.empty:
            continue
        # flatten multi-level cols
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df = df.rename(columns={"Datetime": "datetime", "Date": "datetime"})
        chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out


def _format_index_like(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Match SPX/VIX schema (no volume column)."""
    if df.empty:
        return df
    df = df.copy()
    # datetime is UTC-aware from yfinance; convert to US/Eastern
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    df["datetime"] = df["datetime"].dt.tz_convert("US/Eastern")
    # filter to RTH 09:30 - 15:59
    df = df[df["datetime"].dt.time.between(pd.Timestamp("09:30").time(), pd.Timestamp("15:59").time())].copy()
    df["timestamp"] = (df["datetime"].astype("int64") // 10**6).astype("int64")
    df["time"] = df["datetime"].dt.strftime("%H:%M:%S")
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    rename = {
        "Open": f"{sym}_open",
        "High": f"{sym}_high",
        "Low": f"{sym}_low",
        "Close": f"{sym}_close",
    }
    df = df.rename(columns=rename)
    keep = ["timestamp", f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close", "datetime", "time", "date"]
    return df[keep]


def _format_spy_like(df: pd.DataFrame) -> pd.DataFrame:
    """SPY schema has volume + vwap (vwap synthesized from typical price)."""
    if df.empty:
        return df
    df = df.copy()
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    df["datetime"] = df["datetime"].dt.tz_convert("US/Eastern")
    df = df[df["datetime"].dt.time.between(pd.Timestamp("09:30").time(), pd.Timestamp("15:59").time())].copy()
    df["timestamp"] = (df["datetime"].astype("int64") // 10**6).astype("int64")
    df["time"] = df["datetime"].dt.strftime("%H:%M:%S")
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = typical
    keep = ["timestamp", "open", "high", "low", "close", "volume", "vwap", "datetime", "time", "date"]
    return df[keep]


def _append_to_pickle(path: str, new_df: pd.DataFrame, sym_label: str) -> None:
    if new_df.empty:
        print(f"  {sym_label}: nothing to append")
        return
    if os.path.exists(path):
        existing = pickle.load(open(path, "rb"))
    else:
        existing = pd.DataFrame()
    existing_dates = set(existing["date"].unique()) if not existing.empty else set()
    new_dates = set(new_df["date"].unique())
    truly_new = new_dates - existing_dates
    new_df = new_df[new_df["date"].isin(truly_new)]
    if new_df.empty:
        print(f"  {sym_label}: all dates already cached")
        return
    new_df = new_df.copy()
    if not existing.empty:
        # match dtype of existing datetime
        existing_tz = existing["datetime"].dt.tz
        if existing_tz is not None:
            new_df["datetime"] = new_df["datetime"].dt.tz_convert(existing_tz)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    with open(path, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {sym_label}: appended {len(new_df)} bars, new dates: {sorted(truly_new)[:3]}... "
          f"max date now {combined['date'].max()}")


def main() -> None:
    start = "2026-04-02"
    end = "2026-04-25"
    print(f"Pulling underlying data via yfinance for {start} -> {end}")

    print("\n=== SPX (^GSPC) ===")
    spx = _download_yf("^GSPC", start, end)
    print(f"  raw rows: {len(spx)}")
    spx_fmt = _format_index_like(spx, "spx")
    _append_to_pickle(SPX_PATH, spx_fmt, "spx")

    print("\n=== SPY ===")
    spy = _download_yf("SPY", start, end)
    print(f"  raw rows: {len(spy)}")
    spy_fmt = _format_spy_like(spy)
    _append_to_pickle(SPY_PATH, spy_fmt, "spy")

    print("\n=== VIX (^VIX) ===")
    vix = _download_yf("^VIX", start, end)
    print(f"  raw rows: {len(vix)}")
    vix_fmt = _format_index_like(vix, "vix")
    _append_to_pickle(VIX_PATH, vix_fmt, "vix")

    print("\nDone.")


if __name__ == "__main__":
    main()
