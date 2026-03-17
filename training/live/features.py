from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from training.prepare import compute_features, normalize_features_with_context
from training.live.contracts import LiveContextBundle


@dataclass
class LiveFeatureSnapshot:
    timestamp_ms: int
    normalized_window: np.ndarray
    latest_normalized_row: np.ndarray
    latest_raw_row: np.ndarray
    completeness: float
    staleness_seconds: dict[str, float]


class FiveSecondMinuteAggregator:
    """Aggregates 5-second bars into completed 1-minute bars."""

    def __init__(self) -> None:
        self.current_minute: dt.datetime | None = None
        self.current: dict[str, dict[str, float]] = {}
        self.completed: list[tuple[dt.datetime, dict[str, dict[str, float]]]] = []

    @staticmethod
    def _minute(ts: dt.datetime) -> dt.datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return ts.replace(second=0, microsecond=0)

    def update(self, symbol: str, bar_time: dt.datetime, o: float, h: float, l: float, c: float, v: float) -> None:
        minute = self._minute(bar_time)
        if self.current_minute is None:
            self.current_minute = minute
        if minute > self.current_minute:
            self.completed.append((self.current_minute, self.current))
            self.current = {}
            self.current_minute = minute
        rec = self.current.get(symbol)
        if rec is None:
            self.current[symbol] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        else:
            rec["high"] = max(rec["high"], h)
            rec["low"] = min(rec["low"], l)
            rec["close"] = c
            rec["volume"] += v

    def pop_completed(self) -> tuple[dt.datetime, dict[str, dict[str, float]]] | None:
        if not self.completed:
            return None
        return self.completed.pop(0)


class LiveFeatureEngine:
    """Computes model-ready live features by appending only live minute bars."""

    def __init__(self, context_bundle: LiveContextBundle) -> None:
        self.bundle = context_bundle
        self.df = pd.DataFrame(context_bundle.market_rows).copy()
        if "datetime" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], utc=True).dt.tz_convert("US/Eastern")
        else:
            self.df["datetime"] = pd.to_datetime(self.df["timestamp"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
        self.df["date"] = self.df["datetime"].dt.date.astype(str)
        self.options_data = dict(context_bundle.options_data or {})
        self.chain_data = dict(context_bundle.chain_data or {})
        self.vix_data = dict(context_bundle.vix_rows or {})
        self.feature_history: np.ndarray | None = None
        self.norm_history: np.ndarray | None = None
        self.valid_history: np.ndarray | None = None
        self.timestamps: list[str] = list(context_bundle.timestamps)
        self.staleness: dict[str, float] = {}

    def append_live_minute(
        self,
        minute_ts: dt.datetime,
        spx_bar: dict[str, float],
        spy_bar: dict[str, float],
        vix_bar: dict[str, float] | None,
        option_snapshot: dict[str, dict[str, Any]],
    ) -> int:
        if minute_ts.tzinfo is None:
            minute_ts = minute_ts.replace(tzinfo=dt.timezone.utc)
        ts_ms = int(minute_ts.timestamp() * 1000)
        et = minute_ts.astimezone(dt.timezone(dt.timedelta(hours=-5)))
        date_str = et.date().isoformat()
        row = {
            "timestamp": ts_ms,
            "datetime": et,
            "date": date_str,
            "open": float(spx_bar["open"]),
            "high": float(spx_bar["high"]),
            "low": float(spx_bar["low"]),
            "close": float(spx_bar["close"]),
            "volume": float(spy_bar.get("volume", 0.0)),
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df = self.df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if vix_bar is not None:
            self.vix_data[ts_ms] = {
                "vix_open": float(vix_bar["open"]),
                "vix_high": float(vix_bar["high"]),
                "vix_low": float(vix_bar["low"]),
                "vix_close": float(vix_bar["close"]),
            }

        # Build ATM and OTM cache entries for feature computation.
        atm = option_snapshot.get("CALL_ATM"), option_snapshot.get("PUT_ATM")
        if atm[0] is not None or atm[1] is not None:
            call = atm[0] or {}
            put = atm[1] or {}
            strike = float(call.get("strike") or put.get("strike") or round(spx_bar["close"] / 5.0) * 5.0)
            self.options_data[(date_str, ts_ms)] = {
                "strike": strike,
                "call_close": _mid_or_nan(call),
                "call_open": _mid_or_nan(call),
                "call_high": _mid_or_nan(call),
                "call_low": _mid_or_nan(call),
                "call_volume": int(call.get("size", 0) or 0),
                "put_close": _mid_or_nan(put),
                "put_open": _mid_or_nan(put),
                "put_high": _mid_or_nan(put),
                "put_low": _mid_or_nan(put),
                "put_volume": int(put.get("size", 0) or 0),
            }
        chain_rec: dict[str, float] = {}
        for label, key in [
            ("CALL_OTM5", "otm5_call"),
            ("CALL_OTM10", "otm10_call"),
            ("PUT_OTM5", "otm5_put"),
            ("PUT_OTM10", "otm10_put"),
        ]:
            quote = option_snapshot.get(label)
            if quote is None:
                continue
            chain_rec[f"{key}_close"] = _mid_or_nan(quote)
            chain_rec[f"{key}_volume"] = int(quote.get("size", 0) or 0)
            chain_rec[f"{key}_strike"] = float(quote.get("strike", np.nan))
        if chain_rec:
            chain_rec["atm_strike"] = float(
                option_snapshot.get("CALL_ATM", {}).get("strike", round(spx_bar["close"] / 5.0) * 5.0)
            )
            self.chain_data[(date_str, ts_ms)] = chain_rec

        return ts_ms

    def compute_snapshot(self, lookback: int) -> LiveFeatureSnapshot | None:
        if len(self.df) < lookback + 5:
            return None
        features, _, _, valid, _, timestamps = compute_features(
            self.df,
            options_data=self.options_data if self.options_data else None,
            vix_data=self.vix_data if self.vix_data else None,
            chain_data=self.chain_data if self.chain_data else None,
        )
        norm = normalize_features_with_context(
            features.copy(),
            valid,
            self.bundle.norm_raw_buffer,
            self.bundle.norm_valid_buffer,
        )
        self.feature_history = features
        self.norm_history = norm
        self.valid_history = valid
        self.timestamps = [str(x) for x in timestamps]

        last_idx = len(norm) - 1
        if last_idx < lookback:
            return None
        window = norm[last_idx - lookback:last_idx]
        raw_last = features[last_idx]
        norm_last = norm[last_idx]
        completeness = float(np.mean(~np.isnan(raw_last)))
        ts = int(self.df.iloc[last_idx]["timestamp"])
        return LiveFeatureSnapshot(
            timestamp_ms=ts,
            normalized_window=window,
            latest_normalized_row=norm_last,
            latest_raw_row=raw_last,
            completeness=completeness,
            staleness_seconds=dict(self.staleness),
        )


def _mid_or_nan(quote: dict[str, Any]) -> float:
    bid = quote.get("bid")
    ask = quote.get("ask")
    last = quote.get("last")
    if _finite(bid) and _finite(ask) and float(ask) >= float(bid):
        return (float(bid) + float(ask)) / 2.0
    if _finite(last):
        return float(last)
    return float("nan")


def _finite(v: Any) -> bool:
    try:
        return v is not None and not math.isnan(float(v))
    except Exception:
        return False

