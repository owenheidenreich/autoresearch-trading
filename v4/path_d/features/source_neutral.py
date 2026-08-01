"""Shared feature builder operating on canonical values, never vendor clients."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np


FEATURE_CONTRACT_VERSION = "pathd.source-neutral-feature-builder.v1"
MARKET_FEATURE_NAMES = (
    "spx_close", "vix_close", "spx_vwap", "omar", "session_range", "momentum_5m", "momentum_15m",
)
OPTION_FEATURE_NAMES = (
    "option_bid", "option_ask", "option_mid", "option_spread", "option_spread_frac",
    "option_bid_size", "option_ask_size", "moneyness_points",
)
LEGACY_OPTION_FEATURE_NAMES = (
    "bid", "ask", "mid", "spread", "spread_frac", "bid_size", "ask_size",
    "option_ohlcv_volume", "stat_open_interest", "iv", "delta", "gamma", "theta",
    "distance_points", "breakeven_distance",
)


@dataclass(frozen=True)
class IndexObservation:
    received_timestamp_utc: str
    close: float
    volume: float = 0.0


class SourceNeutralFeatureBuilder:
    """Pure calculations shared by historical replay and future feed adapters."""

    feature_contract_version = FEATURE_CONTRACT_VERSION

    @staticmethod
    def market_features(
        spx: Sequence[IndexObservation],
        vix: Sequence[IndexObservation],
        decision_available_at_utc: str,
    ) -> np.ndarray:
        cutoff = _timestamp(decision_available_at_utc)
        spx_hist = [item for item in spx if _timestamp(item.received_timestamp_utc) <= cutoff]
        if not spx_hist:
            return np.full(len(MARKET_FEATURE_NAMES), np.nan, dtype=np.float64)
        vix_hist = [item for item in vix if _timestamp(item.received_timestamp_utc) <= cutoff]
        close = np.asarray([float(item.close) for item in spx_hist], dtype=np.float64)
        volume = np.asarray([float(item.volume) for item in spx_hist], dtype=np.float64)
        spx_close = float(close[-1])
        vix_close = float(vix_hist[-1].close) if vix_hist else np.nan
        volume_sum = float(volume.sum())
        vwap = float((close * volume).sum() / volume_sum) if volume_sum > 0 else float(close.mean())
        session_open = float(close[0])
        session_high = float(close.max())
        session_low = float(close.min())
        session_range = session_high - session_low
        omar = (spx_close - session_open) / session_range if session_range > 0 else 0.0
        momentum_5 = spx_close - float(close[-6]) if len(close) >= 6 else 0.0
        momentum_15 = spx_close - float(close[-16]) if len(close) >= 16 else 0.0
        return np.asarray(
            [spx_close, vix_close, vwap, omar, session_range, momentum_5, momentum_15],
            dtype=np.float64,
        )

    @staticmethod
    def option_features(*, quote: Mapping[str, int | float], spx_price: float, strike: float) -> dict[str, float]:
        bid = float(quote["bid_price_micros"]) / 1_000_000.0
        ask = float(quote["ask_price_micros"]) / 1_000_000.0
        mid = (bid + ask) / 2.0
        spread = ask - bid
        return {
            "option_bid": bid,
            "option_ask": ask,
            "option_mid": mid,
            "option_spread": spread,
            "option_spread_frac": spread / mid if mid > 0 else 0.0,
            "option_bid_size": float(quote.get("bid_size") or 0),
            "option_ask_size": float(quote.get("ask_size") or 0),
            "moneyness_points": float(spx_price) - float(strike),
        }

    @staticmethod
    def legacy_option_features(
        quote: Mapping[str, int | float],
        *,
        underlying_price: float,
        strike: float,
        right: str,
        atm_strike: int,
    ) -> np.ndarray:
        """Canonical form of the legacy historical option vector for shim parity."""
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        mid = float(quote["mid"])
        spread = ask - bid
        spread_frac = spread / mid if mid > 0 else np.nan
        is_call = str(right).upper() == "C"
        breakeven = float(strike) + ask if is_call else float(strike) - ask
        breakeven_distance = breakeven - float(underlying_price) if is_call else float(underlying_price) - breakeven
        return np.asarray(
            [
                bid, ask, mid, spread, spread_frac,
                float(quote["bid_size"]), float(quote["ask_size"]),
                float(quote["option_ohlcv_volume"]), float(quote["stat_open_interest"]),
                float(quote["iv"]), float(quote["delta"]), float(quote["gamma"]), float(quote["theta"]),
                float(strike) - int(atm_strike), breakeven_distance,
            ],
            dtype=np.float64,
        )

    def feature_map(
        self,
        *,
        spx: Sequence[IndexObservation],
        vix: Sequence[IndexObservation],
        decision_available_at_utc: str,
        option_quote: Mapping[str, int | float],
        strike: float,
    ) -> dict[str, float]:
        market = self.market_features(spx, vix, decision_available_at_utc)
        result = {name: float(market[index]) for index, name in enumerate(MARKET_FEATURE_NAMES)}
        result.update(self.option_features(quote=option_quote, spx_price=result["spx_close"], strike=strike))
        return result


def feature_bytes(values: Iterable[float]) -> bytes:
    return np.ascontiguousarray(tuple(values), dtype=np.float64).tobytes(order="C")


def feature_hash(values: Iterable[float]) -> str:
    return "sha256:" + hashlib.sha256(feature_bytes(values)).hexdigest()


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
