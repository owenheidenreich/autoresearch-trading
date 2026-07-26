"""Protocol101 live-reproducible feature contract.

This module is intentionally small and dependency-light so both historical
dataset builders and live IBKR row builders can share the same runtime feature
rules without importing each other.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
import math
from typing import Any, Mapping

import pandas as pd

from v4.greeks.repair import compute_repaired_greeks
from v4.schema.types import OptionRight


FEATURE_CONTRACT_VERSION = "protocol101-live-v1"
HISTORICAL_FEATURE_CONTRACT_VERSION = "historical-default"
RUNTIME_ZERO_FEATURES = frozenset({"option_ohlcv_volume", "stat_open_interest"})
REQUIRED_GREEK_FIELDS = ("iv", "delta", "gamma", "theta")
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class Protocol101LiveFeatureContractV1:
    """Causal feature rules shared by historical replay and live IBKR runtime."""

    version: str = FEATURE_CONTRACT_VERSION
    ladder_dollars: int = 50
    strike_step: int = 5
    market_window_minutes: int = 30
    index_context_lag_minutes: int = 1
    max_quote_age_seconds: float = 90.0
    min_mid: float = 0.50
    max_mid: float = 35.0
    max_spread_abs: float = 0.50
    max_spread_frac: float = 0.25
    min_bid_size: int = 1
    min_ask_size: int = 1
    no_new_entries_after: time = time(15, 30)
    forced_flat_before: time = time(15, 55)
    contract_multiplier: int = 100
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    greek_policy: str = "shared_repaired_greeks_vendor_greeks_audit_only"
    unavailable_runtime_fields_zeroed: tuple[str, ...] = tuple(sorted(RUNTIME_ZERO_FEATURES))
    timestamp_policy: str = "interval_end_option_quote_at_t_completed_index_context_through_t_minus_1m"
    opening_context_policy: str = "no_leading_backfill_prior_close_only_when_available"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT = Protocol101LiveFeatureContractV1()


def is_live_feature_contract(name: str | None) -> bool:
    return str(name or "").strip().lower() == FEATURE_CONTRACT_VERSION


def feature_contract_version(name: str | None) -> str:
    return FEATURE_CONTRACT_VERSION if is_live_feature_contract(name) else HISTORICAL_FEATURE_CONTRACT_VERSION


def feature_contract_metadata(name: str | None) -> dict[str, Any]:
    if is_live_feature_contract(name):
        return DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT.to_dict()
    return {
        "version": HISTORICAL_FEATURE_CONTRACT_VERSION,
        "timestamp_policy": "legacy_historical_builder",
    }


def round_to_strike_step(value: float, step: int = 5) -> int:
    return int(round(float(value) / float(step)) * int(step))


def runtime_feature_value(name: str, value: Any, *, live_contract: bool) -> float:
    if live_contract and name in RUNTIME_ZERO_FEATURES:
        return 0.0
    return finite_or_nan(value)


def option_feature_values(
    quote: Mapping[str, Any],
    *,
    decision_time: pd.Timestamp,
    spx: float,
    strike: float,
    right: str,
    atm_strike: int,
    feature_names: list[str] | tuple[str, ...],
    live_contract: bool,
    risk_free_rate: float,
    dividend_yield: float,
) -> dict[str, float]:
    bid = finite_or_nan(quote.get("bid"))
    ask = finite_or_nan(quote.get("ask"))
    mid = finite_or_nan(quote.get("mid"))
    if not math.isfinite(mid) and math.isfinite(bid) and math.isfinite(ask):
        mid = (bid + ask) / 2.0
    spread = ask - bid if math.isfinite(ask) and math.isfinite(bid) else math.nan
    spread_frac = spread / mid if math.isfinite(spread) and math.isfinite(mid) and mid > 0 else math.nan
    is_call = _is_call(right)
    breakeven = strike + ask if is_call else strike - ask
    breakeven_distance = (
        breakeven - spx if is_call and math.isfinite(breakeven) else spx - breakeven if math.isfinite(breakeven) else math.nan
    )
    greek_quote = dict(quote)
    if math.isfinite(float(spx)):
        greek_quote["underlying_price"] = float(spx)
    if live_contract:
        greek_quote["prefer_repaired_greeks"] = True
    iv, delta, gamma, theta = greeks_for_quote(
        greek_quote,
        decision_time=decision_time,
        strike=strike,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    values = {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "spread_frac": spread_frac,
        "bid_size": runtime_feature_value("bid_size", quote.get("bid_size"), live_contract=live_contract),
        "ask_size": runtime_feature_value("ask_size", quote.get("ask_size"), live_contract=live_contract),
        "option_ohlcv_volume": runtime_feature_value("option_ohlcv_volume", quote.get("option_ohlcv_volume"), live_contract=live_contract),
        "stat_open_interest": runtime_feature_value("stat_open_interest", quote.get("stat_open_interest"), live_contract=live_contract),
        "iv": iv,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "distance_points": float(strike) - float(atm_strike),
        "breakeven_distance": breakeven_distance,
    }
    return {name: values.get(name, math.nan) for name in feature_names}


def greeks_for_quote(
    quote: Mapping[str, Any],
    *,
    decision_time: pd.Timestamp,
    strike: float,
    right: str,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[float, float, float, float]:
    direct = tuple(finite_or_nan(quote.get(name)) for name in REQUIRED_GREEK_FIELDS)
    prefer_repaired = bool(quote.get("prefer_repaired_greeks"))
    if not prefer_repaired and all(math.isfinite(value) for value in direct):
        return direct  # type: ignore[return-value]
    underlying = finite_or_nan(quote.get("underlying_price"))
    t_years = time_to_expiry_years(quote, decision_time)
    if not math.isfinite(underlying) or t_years is None:
        return direct if all(math.isfinite(value) for value in direct) else (math.nan, math.nan, math.nan, math.nan)  # type: ignore[return-value]
    estimate = compute_repaired_greeks(
        S=underlying,
        K=float(strike),
        T=t_years,
        is_call=_is_call(right),
        mid=quote.get("mid"),
        ask=quote.get("ask"),
        bid=quote.get("bid"),
        r=float(risk_free_rate),
        q=float(dividend_yield),
    )
    if estimate is None:
        if prefer_repaired:
            # Vendor Greeks are retained in raw audit fields, but they cannot
            # rescue a contract that fails the shared causal repair.  Falling
            # back here made the IBKR universe less strict than historical.
            return (math.nan, math.nan, math.nan, math.nan)
        return direct if all(math.isfinite(value) for value in direct) else (math.nan, math.nan, math.nan, math.nan)  # type: ignore[return-value]
    return estimate.iv, estimate.delta, estimate.gamma, estimate.theta_per_day


def time_to_expiry_years(quote: Mapping[str, Any], decision_time: pd.Timestamp) -> float | None:
    expiry_ts = quote.get("settlement_time_utc")
    if expiry_ts is None or pd.isna(expiry_ts):
        return None
    expiry = pd.Timestamp(expiry_ts)
    if expiry.tzinfo is None:
        expiry = expiry.tz_localize("UTC")
    else:
        expiry = expiry.tz_convert("UTC")
    ts = pd.Timestamp(decision_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    seconds = (expiry - ts).total_seconds()
    if seconds <= 0:
        return None
    return seconds / SECONDS_PER_YEAR


def candidate_is_tradable_values(
    values: Mapping[str, Any],
    contract: Protocol101LiveFeatureContractV1 | None = None,
    *,
    require_greeks: bool = True,
) -> bool:
    contract = contract or DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT
    bid = finite_or_nan(values.get("bid"))
    ask = finite_or_nan(values.get("ask"))
    mid = finite_or_nan(values.get("mid"))
    spread = ask - bid if math.isfinite(ask) and math.isfinite(bid) else math.nan
    if not all(math.isfinite(value) for value in (bid, ask, mid)):
        return False
    if bid < 0 or ask <= 0 or ask < bid:
        return False
    if mid < contract.min_mid or mid > contract.max_mid:
        return False
    if spread > contract.max_spread_abs:
        return False
    if mid > 0 and spread / mid > contract.max_spread_frac:
        return False
    bid_size = finite_or_nan(values.get("bid_size"))
    ask_size = finite_or_nan(values.get("ask_size"))
    if math.isfinite(bid_size) and int(bid_size) < contract.min_bid_size:
        return False
    if math.isfinite(ask_size) and int(ask_size) < contract.min_ask_size:
        return False
    if not require_greeks:
        return True
    return all(math.isfinite(finite_or_nan(values.get(name))) for name in REQUIRED_GREEK_FIELDS)


def quote_source_metadata(
    quote: Mapping[str, Any],
    *,
    decision_time: pd.Timestamp,
    feature_contract_name: str | None,
) -> dict[str, Any]:
    ts = pd.Timestamp(decision_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    quote_ts = first_timestamp(
        quote,
        ("raw_quote_timestamp_utc", "quote_timestamp", "quote_time", "event_time", "receive_time"),
    )
    received_ts = first_timestamp(quote, ("received_timestamp_utc", "received_timestamp", "receive_time"))
    age_ms = quote.get("quote_age_ms")
    if age_ms is None and quote_ts is not None:
        age_ms = max((ts - quote_ts).total_seconds() * 1000.0, 0.0)
    return {
        "feature_contract_version": feature_contract_version(feature_contract_name),
        "source_quote_time": quote_ts.isoformat() if quote_ts is not None else None,
        "source_quote_ts": quote_ts.isoformat() if quote_ts is not None else None,
        "source_context_time": None,
        "source_context_ts": None,
        "quote_age_ms": finite_or_none(age_ms),
        "quote_age_source": quote.get("quote_age_source") or ("computed_from_source_quote_time" if quote_ts is not None else None),
        "raw_quote_timestamp_utc": quote_ts.isoformat() if quote_ts is not None else quote.get("raw_quote_timestamp_utc"),
        "received_timestamp_utc": received_ts.isoformat() if received_ts is not None else quote.get("received_timestamp_utc"),
        "decision_timestamp_utc": ts.isoformat(),
    }


def first_timestamp(quote: Mapping[str, Any], names: tuple[str, ...]) -> pd.Timestamp | None:
    for name in names:
        value = quote.get(name)
        if value is None or value == "":
            continue
        try:
            ts = pd.Timestamp(value)
        except (TypeError, ValueError):
            continue
        if pd.isna(ts):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    return None


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def finite_or_none(value: Any) -> float | None:
    number = finite_or_nan(value)
    return number if math.isfinite(number) else None


def _is_call(right: str) -> bool:
    return str(right).upper() in {"C", OptionRight.CALL.value}
