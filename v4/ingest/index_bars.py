"""Index-bar loaders for SPX/VIX side data.

The option chain comes from OPRA, but the prototype should keep index bars as
their own source family. That makes it explicit when SPX/VIX came from Cboe,
ThetaData, or another licensed feed instead of silently blending it into OPRA.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


_TIME_COLUMNS = ("event_time", "timestamp", "datetime", "date_time", "time", "ts")
_CLOSE_COLUMNS = ("close", "price", "value", "last", "index_value")


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported index-bar file extension: {path.suffix}")


def normalize_index_bars(
    frame: pd.DataFrame,
    *,
    symbol: str,
    time_col: str | None = None,
    close_col: str | None = None,
) -> pd.DataFrame:
    """Normalize SPX/VIX bars to the columns the neural dataset expects.

    Accepted input can be vendor CSV/parquet with either full OHLCV bars or a
    timestamp + index value series. Missing OHLC columns are filled from close.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=["event_time", "symbol", "open", "high", "low", "close", "volume"]
        )

    time_col = time_col or _first_existing(frame.columns, _TIME_COLUMNS)
    close_col = close_col or _first_existing(frame.columns, _CLOSE_COLUMNS)
    if time_col is None:
        raise ValueError(f"could not identify time column in {list(frame.columns)}")
    if close_col is None:
        raise ValueError(f"could not identify close/value column in {list(frame.columns)}")

    out = pd.DataFrame()
    out["event_time"] = pd.to_datetime(frame[time_col], utc=True)
    out["symbol"] = symbol.upper()
    out["close"] = pd.to_numeric(frame[close_col], errors="coerce")

    for name in ("open", "high", "low"):
        source = _first_existing(frame.columns, (name,))
        out[name] = (
            pd.to_numeric(frame[source], errors="coerce")
            if source is not None
            else out["close"]
        )

    volume_col = _first_existing(frame.columns, ("volume", "vol"))
    out["volume"] = (
        pd.to_numeric(frame[volume_col], errors="coerce").fillna(0).astype("int64")
        if volume_col is not None
        else 0
    )

    out = out[["event_time", "symbol", "open", "high", "low", "close", "volume"]]
    out = out.dropna(subset=["event_time", "close"]).sort_values("event_time")
    return out.reset_index(drop=True)


def load_index_bars(
    path: str | Path,
    *,
    symbol: str,
    time_col: str | None = None,
    close_col: str | None = None,
) -> pd.DataFrame:
    """Load a vendor file and normalize it into causal one-minute index bars."""
    return normalize_index_bars(
        _read_frame(path), symbol=symbol, time_col=time_col, close_col=close_col
    )


def load_spx_1m(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load SPX one-minute index bars."""
    return load_index_bars(path, symbol="SPX", **kwargs)


def load_vix_1m(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load VIX one-minute index bars."""
    return load_index_bars(path, symbol="VIX", **kwargs)
