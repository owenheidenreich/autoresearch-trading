"""Deterministic bridge from causal episode tables to architecture tensors.

The bridge selects only completed candles whose ``knowable_at`` boundary is at
or before the decision and only the contemporaneous whole live ladder snapshot.
Scaling/imputation is deliberately not learned here; a future walk-forward fit
must derive those transforms from its earlier training prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
import pandas as pd
import torch

from v5.ops.build_causal_day_dataset import minute_number
from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_policy_gate import ROLES


CANDLE_FEATURES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "body_points",
    "upper_wick_points",
    "lower_wick_points",
    "close_from_session_open_points",
    "close_from_running_high_points",
    "close_from_running_low_points",
    "range_position",
    "return_1m",
    "realised_vol_5m",
    "realised_vol_15m",
    "realised_vol_30m",
    "realised_vol_60m",
    "volume_vs_expanding_median",
)
LADDER_FEATURES = (
    "bid",
    "ask",
    "mid",
    "spread",
    "bid_size",
    "ask_size",
    "volume",
    "volume_observed",
    "open_interest",
    "open_interest_observed",
    "self_iv",
    "self_iv_observed",
    "self_delta",
    "self_delta_observed",
    "self_gamma",
    "self_gamma_observed",
    "self_theta_per_minute",
    "self_theta_observed",
    "self_vega",
    "self_vega_observed",
    "moneyness_itm_points",
    "is_call",
    "minutes_to_expiry_fraction",
)
MAX_CANDLE_PREFIX = 390

# The three state-vector widths, named rather than left as literals inside the
# builders below. Architecture parameter counts scale with these, and on
# 2026-08-14 a governance ruling was signed against transcribed counts that were
# wrong by roughly 2.4x. Anything that needs these must derive them from here.
ACCOUNT_FEATURES = 5
POSITION_FEATURES = 10
CLOCK_FEATURES = 5
STARTING_EQUITY_USD = 10_000.0


class TensorizationError(RuntimeError):
    """The causal tables cannot produce the declared observation."""


@dataclass(frozen=True)
class AccountObservation:
    cash_usd: float
    realised_pnl_usd: float
    trades_opened: int
    trade_cap: int
    breaker_triggered: bool


@dataclass(frozen=True)
class PositionObservation:
    right: str
    strike: float
    origin_regime: str
    entry_ask_usd: float
    entry_moneyness_itm_points: float
    unrealised_pnl_usd: float
    maximum_favourable_usd: float
    maximum_adverse_usd: float
    minutes_held: int


@dataclass(frozen=True)
class TensorizedObservation:
    batch: CausalPolicyBatch
    contract_ids: tuple[str, ...]
    candle_minutes: tuple[str, ...]


def _finite_matrix(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> np.ndarray:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise TensorizationError(f"{name} columns missing: {missing}")
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise TensorizationError(
            f"{name} contains missing/non-finite values; fold-fitted imputation is required"
        )
    return values


def _candle_feature_frame(prefix: pd.DataFrame) -> pd.DataFrame:
    """Derive chart-reader channels from the visible prefix only."""

    raw_columns = ("open", "high", "low", "close", "volume")
    raw = prefix.loc[:, raw_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(raw.to_numpy(float)).all():
        raise TensorizationError("completed candle prefix contains missing/non-finite OHLCV")
    out = raw.copy()
    open_ = raw["open"]
    high = raw["high"]
    low = raw["low"]
    close = raw["close"]
    volume = raw["volume"]
    body_top = pd.concat((open_, close), axis=1).max(axis=1)
    body_bottom = pd.concat((open_, close), axis=1).min(axis=1)
    running_high = high.cummax()
    running_low = low.cummin()
    running_range = running_high - running_low
    returns = close.pct_change(fill_method=None).fillna(0.0)
    out["body_points"] = close - open_
    out["upper_wick_points"] = high - body_top
    out["lower_wick_points"] = body_bottom - low
    out["close_from_session_open_points"] = close - float(open_.iloc[0])
    out["close_from_running_high_points"] = close - running_high
    out["close_from_running_low_points"] = close - running_low
    out["range_position"] = ((close - running_low) / running_range).where(
        running_range.gt(0.0), 0.5
    )
    out["return_1m"] = returns
    for window in (5, 15, 30, 60):
        out[f"realised_vol_{window}m"] = (
            returns.rolling(window, min_periods=1).std(ddof=0).fillna(0.0)
        )
    expanding_median = volume.expanding(min_periods=1).median()
    out["volume_vs_expanding_median"] = (volume / expanding_median).where(
        expanding_median.gt(0.0), 0.0
    )
    return out


def _ladder_feature_frame(current: pd.DataFrame) -> pd.DataFrame:
    """Encode the whole visible chain and preserve observed-missing information."""

    required = {
        "bid",
        "ask",
        "mid",
        "bid_size",
        "ask_size",
        "volume",
        "open_interest",
        "self_iv",
        "self_delta",
        "self_gamma",
        "self_theta_per_minute",
        "self_vega",
        "moneyness_itm_points",
        "right",
        "minute",
        "entry_eligible",
    }
    missing = sorted(required - set(current.columns))
    if missing:
        raise TensorizationError(f"whole ladder columns missing: {missing}")
    out = pd.DataFrame(index=current.index)
    for column in ("bid", "ask", "mid", "bid_size", "ask_size", "moneyness_itm_points"):
        out[column] = pd.to_numeric(current[column], errors="coerce")
    out["spread"] = out["ask"] - out["bid"]
    for column in (
        "volume",
        "open_interest",
        "self_iv",
        "self_delta",
        "self_gamma",
        "self_theta_per_minute",
        "self_vega",
    ):
        values = pd.to_numeric(current[column], errors="coerce")
        observed_name = (
            "self_theta_observed"
            if column == "self_theta_per_minute"
            else f"{column}_observed"
        )
        out[column] = values.fillna(0.0)
        out[observed_name] = values.notna().astype(float)
    out["is_call"] = current["right"].astype(str).eq("C").astype(float)
    minutes = current["minute"].astype(str)
    out["minutes_to_expiry_fraction"] = np.asarray(
        [(960 - minute_number(value)) / 390.0 for value in minutes], dtype=float
    )
    return out


def _account_vector(value: AccountObservation) -> np.ndarray:
    if value.trade_cap not in (1, 2, 3) or not 0 <= value.trades_opened <= value.trade_cap:
        raise TensorizationError("account trade count/cap violates the declared family")
    return np.asarray(
        [
            value.cash_usd / STARTING_EQUITY_USD,
            value.realised_pnl_usd / STARTING_EQUITY_USD,
            value.trades_opened / 3.0,
            (value.trade_cap - value.trades_opened) / value.trade_cap,
            float(value.breaker_triggered),
        ],
        dtype=np.float32,
    )


def _position_vector(
    value: PositionObservation | None, spot: float
) -> np.ndarray:
    if value is None:
        return np.zeros(POSITION_FEATURES, dtype=np.float32)
    if value.right not in ("C", "P") or value.origin_regime not in ("morning", "afternoon"):
        raise TensorizationError("position side/origin is outside the declared router")
    side = 1.0 if value.right == "C" else -1.0
    origin = 1.0 if value.origin_regime == "morning" else -1.0
    return np.asarray(
        [
            1.0,
            side,
            (value.strike - spot) / 25.0,
            value.entry_ask_usd / STARTING_EQUITY_USD,
            value.unrealised_pnl_usd / STARTING_EQUITY_USD,
            value.maximum_favourable_usd / STARTING_EQUITY_USD,
            value.maximum_adverse_usd / STARTING_EQUITY_USD,
            value.minutes_held / 120.0,
            value.entry_moneyness_itm_points / 25.0,
            origin,
        ],
        dtype=np.float32,
    )


def _clock_vector(minute: str) -> np.ndarray:
    value = minute_number(minute)
    if not 570 <= value <= 960:
        raise TensorizationError("decision minute is outside the regular session")
    elapsed = (value - 570) / 390.0
    angle = 2.0 * pi * elapsed
    return np.asarray(
        [elapsed, (960 - value) / 390.0, float(value < 766), sin(angle), cos(angle)],
        dtype=np.float32,
    )


def tensorize_observation(
    candles: pd.DataFrame,
    ladder: pd.DataFrame,
    *,
    session: str,
    minute: str,
    role: str,
    account: AccountObservation,
    position: PositionObservation | None = None,
) -> TensorizedObservation:
    if role not in ROLES:
        raise TensorizationError(f"unknown routed role: {role}")
    candle_session = candles[candles["session"].astype(str).eq(session)].copy()
    if "knowable_at" not in candle_session or "bar_minute" not in candle_session:
        raise TensorizationError("candles require knowable_at and bar_minute clocks")
    prefix = candle_session[candle_session["knowable_at"].astype(str).le(minute)].sort_values(
        "knowable_at"
    ).tail(MAX_CANDLE_PREFIX)
    if prefix.empty:
        raise TensorizationError("decision has no completed candle prefix")
    if str(prefix.iloc[-1]["knowable_at"]) != minute:
        raise TensorizationError("latest completed candle is missing at the decision boundary")
    candle_features = _candle_feature_frame(prefix)
    candle_values = _finite_matrix(candle_features, CANDLE_FEATURES, "candle prefix")

    current = ladder[
        ladder["session"].astype(str).eq(session)
        & ladder["minute"].astype(str).eq(minute)
    ].copy()
    current = current.sort_values(["right", "strike", "contract_id"])
    ladder_features = _ladder_feature_frame(current)
    ladder_values = _finite_matrix(ladder_features, LADDER_FEATURES, "ladder snapshot")
    contract_ids = tuple(current["contract_id"].astype(str))
    spot_values = pd.to_numeric(current["underlying_price"], errors="coerce").dropna().unique()
    if len(spot_values) != 1:
        raise TensorizationError("ladder snapshot requires one unique finite SPX value")
    spot = float(spot_values[0])

    candle_tensor = torch.tensor(candle_values, dtype=torch.float32).unsqueeze(0)
    candle_mask = torch.ones((1, len(prefix)), dtype=torch.bool)
    ladder_tensor = torch.tensor(ladder_values, dtype=torch.float32).unsqueeze(0)
    ladder_mask = torch.ones((1, len(current)), dtype=torch.bool)
    entry_actions = current["entry_eligible"].fillna(False).astype(bool).to_numpy(copy=True)
    entry_actions &= (
        pd.to_numeric(current["ask"], errors="coerce").to_numpy(float) * 100.0
        <= account.cash_usd
    )
    if (
        role not in ("morning_entry", "afternoon_entry")
        or account.trades_opened >= account.trade_cap
        or account.breaker_triggered
    ):
        entry_actions[:] = False
    batch = CausalPolicyBatch(
        candles=candle_tensor,
        candle_mask=candle_mask,
        ladder=ladder_tensor,
        ladder_mask=ladder_mask,
        entry_action_mask=torch.tensor(entry_actions, dtype=torch.bool).unsqueeze(0),
        account=torch.tensor(_account_vector(account)).unsqueeze(0),
        position=torch.tensor(_position_vector(position, spot)).unsqueeze(0),
        clock=torch.tensor(_clock_vector(minute)).unsqueeze(0),
        roles=(role,),
    )
    return TensorizedObservation(
        batch=batch,
        contract_ids=contract_ids,
        candle_minutes=tuple(prefix["bar_minute"].astype(str)),
    )
