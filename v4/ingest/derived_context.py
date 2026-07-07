"""Derived SPX/volatility context from SPXW option quotes.

This is an economic bridge for the pilot. True SPX/VIX index bars are still the
promotion-grade target, but put-call parity lets us build a causal SPX context
directly from the same SPXW quotes we already purchased.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v4.greeks.black_scholes import implied_vol


SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class DerivedContextResult:
    spx_bars: pd.DataFrame
    vol_bars: pd.DataFrame
    normalized: pa.Table


def _frame(table_or_frame: pa.Table | pd.DataFrame) -> pd.DataFrame:
    if isinstance(table_or_frame, pa.Table):
        return table_or_frame.to_pandas()
    return table_or_frame.copy()


def _years_to_settlement(row: pd.Series, quote_time: pd.Timestamp) -> float:
    expiry = pd.Timestamp(row["settlement_time_utc"])
    if expiry.tzinfo is None:
        expiry = expiry.tz_localize("UTC")
    else:
        expiry = expiry.tz_convert("UTC")
    seconds = (expiry - quote_time).total_seconds()
    return max(seconds / SECONDS_PER_YEAR, 1.0 / SECONDS_PER_YEAR / 24.0 / 60.0)


def _atm_iv_from_straddle(
    *,
    spot: float,
    strike: float,
    t_years: float,
    call_mid: float,
    put_mid: float,
    risk_free_rate: float,
) -> float:
    vols = []
    for price, is_call in ((call_mid, True), (put_mid, False)):
        try:
            vols.append(
                implied_vol(
                    market_price=price,
                    S=spot,
                    K=strike,
                    T=t_years,
                    r=risk_free_rate,
                    q=0.0,
                    is_call=is_call,
                )
            )
        except ValueError:
            continue
    if vols:
        return float(np.mean(vols))

    # Brenner-Subrahmanyam ATM approximation as a fallback.
    straddle = call_mid + put_mid
    return float(straddle / max(spot, 1.0) * math.sqrt(math.pi / (2.0 * t_years)))


def derive_context_bars_from_spxw_quotes(
    normalized: pa.Table | pd.DataFrame,
    *,
    risk_free_rate: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive SPX and volatility-context bars from paired call/put quotes.

    For each minute, the strike with the lowest call+put mid is treated as the
    near-ATM pair. Put-call parity estimates spot:

        S ~= C - P + K * exp(-rT)

    The volatility context is the near-ATM IV in VIX-like points. It is not VIX;
    it is a causal 0DTE chain-implied volatility proxy.
    """
    df = _frame(normalized)
    if df.empty:
        empty = pd.DataFrame(columns=["event_time", "symbol", "open", "high", "low", "close", "volume"])
        return empty.copy(), empty.copy()

    df = df[(df["root"] == "SPXW") & (df["settlement_style"] == "PM")].copy()
    df["quote_time"] = pd.to_datetime(
        df["quote_time"].where(df["quote_time"].notna(), df["event_time"]), utc=True
    )
    df["_minute"] = df["quote_time"].dt.floor("min")
    df["strike_float"] = df["strike"].astype(float)
    df["mid"] = df["mid"].where(df["mid"].notna(), (df["bid"] + df["ask"]) / 2.0)
    df = df.dropna(subset=["bid", "ask", "mid", "settlement_time_utc"])
    df = df[(df["ask"] >= df["bid"]) & (df["mid"] > 0)]

    spx_rows = []
    vol_rows = []
    for minute, group in df.groupby("_minute", sort=True):
        latest = group.sort_values("quote_time").groupby(["strike_float", "right"]).tail(1)
        calls = latest[latest["right"] == "C"].set_index("strike_float")
        puts = latest[latest["right"] == "P"].set_index("strike_float")
        common = calls.index.intersection(puts.index)
        if len(common) == 0:
            continue

        pairs = pd.DataFrame(
            {
                "call_mid": calls.loc[common, "mid"].astype(float),
                "put_mid": puts.loc[common, "mid"].astype(float),
                "call_spread": (calls.loc[common, "ask"] - calls.loc[common, "bid"]).astype(float),
                "put_spread": (puts.loc[common, "ask"] - puts.loc[common, "bid"]).astype(float),
            },
            index=common,
        )
        pairs["straddle"] = pairs["call_mid"] + pairs["put_mid"]
        pairs["total_spread"] = pairs["call_spread"] + pairs["put_spread"]
        pairs = pairs[(pairs["straddle"] > 0) & (pairs["total_spread"] / pairs["straddle"] < 0.60)]
        if pairs.empty:
            continue

        atm_strike = float(pairs["straddle"].idxmin())
        pair = pairs.loc[atm_strike]
        sample_row = latest[latest["strike_float"] == atm_strike].iloc[0]
        t_years = _years_to_settlement(sample_row, minute)
        discount_k = atm_strike * math.exp(-risk_free_rate * t_years)
        spot = float(pair["call_mid"] - pair["put_mid"] + discount_k)
        if not np.isfinite(spot) or spot <= 0:
            continue
        atm_iv = _atm_iv_from_straddle(
            spot=spot,
            strike=atm_strike,
            t_years=t_years,
            call_mid=float(pair["call_mid"]),
            put_mid=float(pair["put_mid"]),
            risk_free_rate=risk_free_rate,
        )

        spx_rows.append(
            {
                "event_time": minute,
                "symbol": "SPX",
                "open": spot,
                "high": spot,
                "low": spot,
                "close": spot,
                "volume": 0,
                "context_source": "spxw_put_call_parity",
                "is_derived": True,
            }
        )
        vol_rows.append(
            {
                "event_time": minute,
                "symbol": "VIX",
                "open": atm_iv * 100.0,
                "high": atm_iv * 100.0,
                "low": atm_iv * 100.0,
                "close": atm_iv * 100.0,
                "volume": 0,
                "context_source": "spxw_0dte_atm_iv",
                "is_derived": True,
            }
        )

    spx = pd.DataFrame(spx_rows).sort_values("event_time").reset_index(drop=True)
    vol = pd.DataFrame(vol_rows).sort_values("event_time").reset_index(drop=True)
    return spx, vol


def attach_underlying_from_context(
    normalized: pa.Table,
    spx_bars: pd.DataFrame,
) -> pa.Table:
    """Fill ``underlying_price`` from derived/context SPX bars at-or-before row time."""
    df = normalized.to_pandas()
    if df.empty or spx_bars.empty:
        return normalized

    df["quote_time"] = pd.to_datetime(
        df["quote_time"].where(df["quote_time"].notna(), df["event_time"]), utc=True
    )
    context = spx_bars[["event_time", "close"]].copy()
    context["event_time"] = pd.to_datetime(context["event_time"], utc=True)
    context = context.sort_values("event_time")
    merged = pd.merge_asof(
        df.sort_values("quote_time"),
        context.rename(columns={"event_time": "_context_time", "close": "_context_close"}),
        left_on="quote_time",
        right_on="_context_time",
        direction="backward",
    )
    merged["underlying_price"] = merged["_context_close"].where(
        merged["_context_close"].notna(), merged["underlying_price"]
    )
    merged = merged.drop(columns=["_context_time", "_context_close"])
    merged = merged[df.columns]
    return pa.Table.from_pandas(merged, schema=normalized.schema, preserve_index=False)


def write_derived_context(
    normalized: pa.Table,
    *,
    session: str,
    spx_path: str | Path,
    vol_path: str | Path,
    normalized_out: str | Path,
) -> DerivedContextResult:
    """Derive, write, and return context bars plus normalized table."""
    spx, vol = derive_context_bars_from_spxw_quotes(normalized)
    spx_path = Path(spx_path)
    vol_path = Path(vol_path)
    normalized_out = Path(normalized_out)
    spx_path.parent.mkdir(parents=True, exist_ok=True)
    vol_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_out.parent.mkdir(parents=True, exist_ok=True)
    spx.to_parquet(spx_path, index=False)
    vol.to_parquet(vol_path, index=False)
    enriched = attach_underlying_from_context(normalized, spx)
    pq.write_table(enriched, normalized_out)
    return DerivedContextResult(spx_bars=spx, vol_bars=vol, normalized=enriched)
