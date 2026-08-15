"""Live-style full-action candidate features for research challengers.

This module is intentionally model-agnostic. It builds the flat-state candidate
rows expected by the full-action challengers from point-in-time quote/context
inputs, without labels, exits, or future path fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.scripts.run_protocol101_event_history_policy import HISTORY_FEATURE_COLUMNS
from v4.scripts.run_protocol164_full_action_space_dataset import (
    CONTRACT_MULTIPLIER,
    FULL_ACTION_FEATURE_COLUMNS,
    NO_NEW_ENTRIES_AFTER_MINUTE,
    SESSION_MINUTES,
    SESSION_OPEN_MINUTE,
)


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_FEATURE_COLUMNS = [
    "market_spx_close",
    "market_vix_close",
    "market_spx_vwap",
    "market_omar",
    "market_session_range",
    "market_momentum_5m",
    "market_momentum_15m",
]
BASE_FEATURE_COLUMNS = list(FULL_ACTION_FEATURE_COLUMNS)
ALL_FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *HISTORY_FEATURE_COLUMNS]


@dataclass
class FullActionHistoryState:
    """Causal per-session summaries used by the full-action challenger."""

    summaries: list[dict[str, float | pd.Timestamp]] = field(default_factory=list)
    session: str | None = None

    def reset_if_new_session(self, session: str) -> None:
        if self.session != session:
            self.session = session
            self.summaries.clear()

    def features_for(self, decision_time: datetime | pd.Timestamp) -> dict[str, float]:
        ts = _utc_timestamp(decision_time)
        previous = self.summaries[-1] if self.summaries else _empty_summary()
        roll3 = self.summaries[-3:]
        if roll3:
            rolling_mean = {key: float(np.mean([float(item[key]) for item in roll3])) for key in _MEAN_KEYS}
            rolling_max = {key: float(np.max([float(item[key]) for item in roll3])) for key in _MAX_KEYS}
            rolling_min = {key: float(np.min([float(item[key]) for item in roll3])) for key in _MIN_KEYS}
        else:
            rolling_mean = _empty_summary()
            rolling_max = _empty_summary()
            rolling_min = _empty_summary()
        prev_time = previous.get("decision_dt")
        minutes_since_prev = 999.0
        if isinstance(prev_time, pd.Timestamp):
            minutes_since_prev = max(0.0, (ts - prev_time).total_seconds() / 60.0)
        return {
            "hist_events_seen": float(min(len(self.summaries), 50)),
            "hist_minutes_since_prev_event": minutes_since_prev,
            "hist_prev_candidate_count": float(previous["candidate_count"]),
            "hist_prev_max_edge": float(previous["max_edge"]),
            "hist_prev_mean_edge": float(previous["mean_edge"]),
            "hist_prev_max_gamma": float(previous["max_gamma"]),
            "hist_prev_mean_theta_burden": float(previous["mean_theta_burden"]),
            "hist_prev_min_spread_over_mid": float(previous["min_spread_over_mid"]),
            "hist_prev_call_count": float(previous["call_count"]),
            "hist_prev_put_count": float(previous["put_count"]),
            "hist_prev_call_minus_put_edge": float(previous["call_minus_put_edge"]),
            "hist_roll3_candidate_count_mean": float(rolling_mean["candidate_count"]),
            "hist_roll3_max_edge": float(rolling_max["max_edge"]),
            "hist_roll3_mean_edge": float(rolling_mean["mean_edge"]),
            "hist_roll3_max_gamma": float(rolling_max["max_gamma"]),
            "hist_roll3_mean_theta_burden": float(rolling_mean["mean_theta_burden"]),
            "hist_roll3_min_spread_over_mid": float(rolling_min["min_spread_over_mid"]),
            "hist_roll3_call_minus_put_edge": float(rolling_mean["call_minus_put_edge"]),
        }

    def update(self, candidates: pd.DataFrame, decision_time: datetime | pd.Timestamp | None = None) -> None:
        if candidates.empty:
            return
        frame = candidates.copy()
        frame["right"] = frame["right"].astype(str)
        edge = pd.to_numeric(frame.get("surface_edge", frame.get("edge", 0.0)), errors="coerce").fillna(0.0)
        gamma = pd.to_numeric(frame.get("entry_gamma", 0.0), errors="coerce").fillna(0.0)
        theta_burden = pd.to_numeric(frame.get("entry_theta_burden", 0.0), errors="coerce").fillna(0.0)
        spread = pd.to_numeric(frame.get("entry_spread_over_mid", 0.0), errors="coerce").fillna(0.0)
        call_edge = edge.where(frame["right"].eq("C"))
        put_edge = edge.where(frame["right"].eq("P"))
        max_call = _finite_float(call_edge.max(), 0.0)
        max_put = _finite_float(put_edge.max(), 0.0)
        ts = _utc_timestamp(decision_time if decision_time is not None else frame["decision_dt"].iloc[0])
        self.summaries.append(
            {
                "decision_dt": ts,
                "candidate_count": float(len(frame)),
                "max_edge": _finite_float(edge.max(), 0.0),
                "mean_edge": _finite_float(edge.mean(), 0.0),
                "max_gamma": _finite_float(gamma.max(), 0.0),
                "mean_theta_burden": _finite_float(theta_burden.mean(), 0.0),
                "min_spread_over_mid": _finite_float(spread.min(), 0.0),
                "call_count": float(frame["right"].eq("C").sum()),
                "put_count": float(frame["right"].eq("P").sum()),
                "call_minus_put_edge": max_call - max_put,
            }
        )


def build_full_action_candidates_from_quotes(
    *,
    decision_time: datetime | pd.Timestamp,
    spx: float,
    vix: float,
    option_quotes: Iterable[dict[str, Any]],
    history: FullActionHistoryState | None = None,
    account_cash: float = 10_000.0,
    starting_cash: float = 10_000.0,
    market_features: dict[str, float] | None = None,
    session: str | None = None,
) -> pd.DataFrame:
    """Build one live-style full-action candidate row per valid SPXW quote."""

    ts = _utc_timestamp(decision_time)
    session_id = session or ts.tz_convert(NY).date().isoformat()
    if history is not None:
        history.reset_if_new_session(session_id)
    hist = history.features_for(ts) if history is not None else _empty_history_features()
    market = _market_features(spx=float(spx), vix=float(vix), override=market_features)
    atm = _round_to_5(float(spx))
    minutes = _minute_of_day(ts)
    elapsed = _elapsed(minutes)
    rows: list[dict[str, Any]] = []
    for raw in option_quotes:
        quote = dict(raw)
        if not _valid_live_quote(quote):
            continue
        right = str(quote.get("right")).upper()
        strike = _finite_float(quote.get("strike"), math.nan)
        bid = _finite_float(quote.get("bid"), math.nan)
        ask = _finite_float(quote.get("ask"), math.nan)
        mid = _finite_float(quote.get("mid"), (bid + ask) / 2.0)
        spread = _finite_float(quote.get("spread"), ask - bid)
        bid_size = _finite_float(quote.get("bid_size"), 0.0)
        ask_size = _finite_float(quote.get("ask_size"), 0.0)
        delta = _finite_float(quote.get("delta"), 0.0)
        gamma = _finite_float(quote.get("gamma"), 0.0)
        theta = _finite_float(quote.get("theta"), 0.0)
        iv = _finite_float(quote.get("iv"), 0.0)
        underlying = _finite_float(quote.get("underlying_price"), float(spx))
        entry_premium = ask * CONTRACT_MULTIPLIER
        offset = strike - atm
        contract_id = str(quote.get("contract_id") or _contract_id(expiry=str(quote.get("expiry") or ts.strftime("%Y%m%d")), strike=strike, right=right))
        row = {
            "session": session_id,
            "decision_time": ts.isoformat(),
            "decision_dt": ts,
            "candidate_uid": str(quote.get("candidate_uid") or f"live_full_action|{session_id}|{ts.isoformat()}|{contract_id}"),
            "contract_id": contract_id,
            "root": "SPXW",
            "settlement_style": "PM",
            "right": right,
            "offset": float(offset),
            "entry_quote_time": str(quote.get("quote_time") or ts.isoformat()),
            "entry_bid": bid,
            "entry_ask": ask,
            "entry_mid": mid,
            "entry_spread": spread,
            "entry_spread_frac": _safe_ratio(spread, mid, 0.0),
            "entry_bid_size": bid_size,
            "entry_ask_size": ask_size,
            "entry_underlying_price": underlying,
            "entry_iv": iv,
            "entry_delta": delta,
            "entry_gamma": gamma,
            "entry_theta": theta,
            "entry_minutes_since_open": elapsed,
            "entry_minutes_to_forced_flat": max(NO_NEW_ENTRIES_AFTER_MINUTE - minutes, 0.0),
            "entry_progress": _safe_ratio(elapsed, SESSION_MINUTES, 0.0),
            "entry_progress_sin": math.sin(2.0 * math.pi * _safe_ratio(elapsed, SESSION_MINUTES, 0.0)),
            "entry_progress_cos": math.cos(2.0 * math.pi * _safe_ratio(elapsed, SESSION_MINUTES, 0.0)),
            "entry_is_first_30m": float(minutes < 10 * 60),
            "entry_is_post_open_morning": float(10 * 60 <= minutes < 11 * 60 + 30),
            "entry_is_midday": float(11 * 60 + 30 <= minutes < 13 * 60 + 30),
            "entry_is_late_afternoon": float(minutes >= 13 * 60 + 30),
            "right_is_call": float(right == "C"),
            "right_is_put": float(right == "P"),
            "abs_offset": abs(float(offset)),
            "entry_abs_delta": abs(delta),
            "entry_abs_theta": abs(theta),
            "entry_gamma_theta_ratio": _safe_ratio(gamma, abs(theta), 0.0),
            "entry_theta_over_mid": _safe_ratio(abs(theta), abs(mid), 0.0),
            "entry_theta_burden": _safe_ratio(abs(theta) * max(NO_NEW_ENTRIES_AFTER_MINUTE - minutes, 0.0), abs(mid), 0.0),
            "entry_gamma_per_premium": _safe_ratio(gamma, ask, 0.0),
            "entry_premium_over_underlying": _safe_ratio(ask, abs(underlying), 0.0),
            "entry_spread_over_mid": _safe_ratio(spread, abs(mid), 0.0),
            "entry_size_imbalance": _safe_ratio(bid_size - ask_size, bid_size + ask_size, 0.0),
            "entry_call_delta_signed": delta if right == "C" else 0.0,
            "entry_put_delta_signed": delta if right == "P" else 0.0,
            "entry_premium": entry_premium,
            "entry_premium_frac_10k": _safe_ratio(entry_premium, starting_cash, 0.0),
            "entry_affordable_10k": float(entry_premium > 0.0 and entry_premium <= account_cash),
            "surface_edge": _finite_float(quote.get("surface_edge", quote.get("edge", 0.0)), 0.0),
            "edge": _finite_float(quote.get("edge", quote.get("surface_edge", 0.0)), 0.0),
            **market,
            **hist,
        }
        for column in ALL_FEATURE_COLUMNS:
            row[column] = _finite_float(row.get(column), 0.0)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["session", "decision_dt", "contract_id", *ALL_FEATURE_COLUMNS])
    return pd.DataFrame(rows).sort_values(["decision_dt", "contract_id"]).reset_index(drop=True)


def feature_source_coverage(feature_columns: Iterable[str]) -> dict[str, Any]:
    """Explain whether feature columns have a live-style source."""

    sources = {}
    missing = []
    for column in feature_columns:
        source = _FEATURE_SOURCES.get(column)
        if source is None:
            missing.append(str(column))
        else:
            sources[str(column)] = source
    return {
        "feature_count": len(list(feature_columns)),
        "mapped_feature_count": len(sources),
        "missing_feature_sources": missing,
        "status": "pass" if not missing else "fail",
        "sources": sources,
    }


def _market_features(*, spx: float, vix: float, override: dict[str, float] | None) -> dict[str, float]:
    base = {
        "market_spx_close": float(spx),
        "market_vix_close": float(vix),
        "market_spx_vwap": float(spx),
        "market_omar": 0.0,
        "market_session_range": 0.0,
        "market_momentum_5m": 0.0,
        "market_momentum_15m": 0.0,
    }
    if override:
        for key in MARKET_FEATURE_COLUMNS:
            if key in override:
                base[key] = _finite_float(override[key], base[key])
    return base


def _valid_live_quote(quote: dict[str, Any]) -> bool:
    return (
        str(quote.get("root") or quote.get("trading_class") or "SPXW") == "SPXW"
        and str(quote.get("settlement_style") or "PM") == "PM"
        and str(quote.get("right") or "").upper() in {"C", "P"}
        and math.isfinite(_finite_float(quote.get("strike"), math.nan))
        and _finite_float(quote.get("bid"), 0.0) > 0.0
        and _finite_float(quote.get("ask"), 0.0) > 0.0
        and _finite_float(quote.get("ask"), 0.0) >= _finite_float(quote.get("bid"), 0.0)
    )


def _empty_history_features() -> dict[str, float]:
    return {column: 999.0 if column == "hist_minutes_since_prev_event" else 0.0 for column in HISTORY_FEATURE_COLUMNS}


def _empty_summary() -> dict[str, float | pd.Timestamp]:
    return {
        "candidate_count": 0.0,
        "max_edge": 0.0,
        "mean_edge": 0.0,
        "max_gamma": 0.0,
        "mean_theta_burden": 0.0,
        "min_spread_over_mid": 0.0,
        "call_count": 0.0,
        "put_count": 0.0,
        "call_minus_put_edge": 0.0,
    }


_MEAN_KEYS = [
    "candidate_count",
    "mean_edge",
    "mean_theta_burden",
    "call_minus_put_edge",
]
_MAX_KEYS = ["max_edge", "max_gamma"]
_MIN_KEYS = ["min_spread_over_mid"]


_FEATURE_SOURCES = {
    **{column: "clock" for column in FULL_ACTION_FEATURE_COLUMNS if column.startswith("entry_minutes") or column.startswith("entry_progress") or column.startswith("entry_is_")},
    **{column: "option_quote" for column in FULL_ACTION_FEATURE_COLUMNS if column.startswith("entry_") and column not in {"entry_minutes_since_open", "entry_minutes_to_forced_flat", "entry_progress", "entry_progress_sin", "entry_progress_cos", "entry_is_first_30m", "entry_is_post_open_morning", "entry_is_midday", "entry_is_late_afternoon"}},
    "right_is_call": "contract_metadata",
    "right_is_put": "contract_metadata",
    "offset": "contract_metadata_plus_underlying",
    "abs_offset": "contract_metadata_plus_underlying",
    **{column: "live_index_context" for column in MARKET_FEATURE_COLUMNS},
    **{column: "causal_candidate_history" for column in HISTORY_FEATURE_COLUMNS},
}


def _elapsed(minutes: float) -> float:
    return float(min(max(minutes - SESSION_OPEN_MINUTE, 0.0), SESSION_MINUTES))


def _minute_of_day(value: pd.Timestamp) -> int:
    local = value.tz_convert(NY)
    return int(local.hour * 60 + local.minute)


def _round_to_5(value: float) -> int:
    return int(round(value / 5.0) * 5)


def _contract_id(*, expiry: str, strike: float, right: str) -> str:
    return f"SPXW-{expiry}-{float(strike):09.3f}-{right}"


def _utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)) or abs(float(denominator)) < 1e-9:
        return float(default)
    return float(numerator) / float(denominator)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)
