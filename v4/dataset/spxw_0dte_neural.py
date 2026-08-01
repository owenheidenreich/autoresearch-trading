"""SPXW 0DTE neural dataset builder.

The first prototype is an entry-quality dataset, not a live trading engine. It
creates one decision row per minute, centered on the SPX $5 strike ladder, and
labels each tradable one-contract long call/put candidate using executable
ask-entry / bid-exit prices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from bisect import bisect_right
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from v4.greeks.repair import compute_repaired_greeks
from v4.live.protocol101_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT,
    candidate_filter_diagnostics,
    candidate_is_tradable_values,
    candidate_ladder_slots,
    feature_contract_metadata,
    feature_contract_requires_model_scoring_greeks,
    feature_contract_version,
    is_live_feature_contract,
    missing_candidate_slot_diagnostics,
    option_feature_values,
    quote_source_metadata,
    round_to_strike_step,
    strike_ladder_context,
)
from v4.model.protocol101_regimen_repair import (
    INT64_MISSING,
    LEGACY_PROCESSED_ROW_SCHEMA,
    TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    ExitReason,
    InvalidReason,
    Protocol101DuplicatePathQuoteIdentityError,
    TwoClockLabel,
)
from v4.schema.types import OptionRight


_NY_TZ = "America/New_York"
_SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
DECISION_GRID_VERSION = "calendar_v2"
MARKET_OPEN_ET = time(9, 30)
EARLY_CLOSE_TIMES_ET = {
    # Cboe U.S. Options RTH early closes only. Keep in sync with the
    # acceptance verifier; do not import bond-market early closes here.
    "2024-11-29": time(13, 0),
    "2024-12-24": time(13, 15),
    "2025-07-03": time(13, 0),
    "2025-11-28": time(13, 0),
    "2025-12-24": time(13, 15),
}


@dataclass(frozen=True)
class LabelPolicy:
    """One executable outcome definition for a long option entry."""

    stop_loss_pct: float = 0.50
    take_profit_pct: float = 1.00
    max_hold_minutes: int = 25

    @property
    def name(self) -> str:
        stop = int(round(self.stop_loss_pct * 100))
        target = int(round(self.take_profit_pct * 100))
        return f"ask_to_bid_stop{stop}_target{target}_hold{self.max_hold_minutes}m"


@dataclass(frozen=True)
class NeuralDatasetConfig:
    """Prototype filters and market-shape parameters."""

    ladder_dollars: int = 50
    strike_step: int = 5
    market_window_minutes: int = 30
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
    fee_per_contract: float = 0.00
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    feature_contract: str | None = None
    diagnostic_index_context_lag_minutes: int | None = None
    diagnostic_source_policy: str | None = None
    compute_policy_labels: bool = True
    processed_row_schema_version: str = LEGACY_PROCESSED_ROW_SCHEMA
    # Trade-shape menu v2, owner-approved 2026-07-07 (see
    # v4/docs/PROTOCOL101_TRADE_SHAPE_MENU_V2_PROPOSAL.md). Shapes 3-6 add
    # patient/asymmetric profiles: stop=1.00 means the premium is the stop
    # (exit only at zero bid); target=99.0 is the no-practical-target
    # sentinel; hold=384 always reaches the 15:55 ET forced-flat cap, i.e.
    # "hold to forced flat" for any entry time.
    label_policies: tuple[LabelPolicy, ...] = field(
        default_factory=lambda: (
            LabelPolicy(0.35, 0.60, 10),
            LabelPolicy(0.50, 1.00, 25),
            LabelPolicy(0.65, 1.50, 45),
            LabelPolicy(0.50, 2.00, 90),
            LabelPolicy(1.00, 3.00, 120),
            LabelPolicy(1.00, 9.99, 384),
            LabelPolicy(1.00, 99.0, 384),
        )
    )


@dataclass(frozen=True)
class ContractQuotePath:
    """Sorted quote arrays for fast causal policy-label evaluation."""

    quote_ns: np.ndarray
    bid: np.ndarray
    mid: np.ndarray


OPTION_FEATURE_NAMES = [
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_frac",
    "bid_size",
    "ask_size",
    "option_ohlcv_volume",
    "stat_open_interest",
    "iv",
    "delta",
    "gamma",
    "theta",
    "distance_points",
    "breakeven_distance",
]

MARKET_FEATURE_NAMES = [
    "spx_close",
    "vix_close",
    "spx_vwap",
    "omar",
    "session_range",
    "momentum_5m",
    "momentum_15m",
]


def _to_frame(normalized: pa.Table | pd.DataFrame) -> pd.DataFrame:
    if isinstance(normalized, pa.Table):
        return normalized.to_pandas()
    return normalized.copy()


def _utc_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _strike_float(value) -> float:
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _round_to_step(value: float, step: int) -> int:
    return round_to_strike_step(value, step)


def _time_to_expiry_years(row: pd.Series, decision_time: pd.Timestamp) -> float | None:
    expiry_ts = row.get("settlement_time_utc")
    if pd.isna(expiry_ts):
        return None
    expiry = pd.Timestamp(expiry_ts)
    if expiry.tzinfo is None:
        expiry = expiry.tz_localize("UTC")
    else:
        expiry = expiry.tz_convert("UTC")
    seconds = (expiry - decision_time).total_seconds()
    if seconds <= 0:
        return None
    return seconds / _SECONDS_PER_YEAR


def _computed_greeks(row: pd.Series, decision_time: pd.Timestamp, config: NeuralDatasetConfig):
    delta = row.get("delta")
    gamma = row.get("gamma")
    theta = row.get("theta")
    iv = row.get("iv")
    if all(pd.notna(v) for v in (delta, gamma, theta, iv)):
        return float(iv), float(delta), float(gamma), float(theta)

    underlying = row.get("underlying_price")
    strike = _strike_float(row.get("strike"))
    if pd.isna(underlying):
        return np.nan, np.nan, np.nan, np.nan
    t_years = _time_to_expiry_years(row, decision_time)
    if t_years is None:
        return np.nan, np.nan, np.nan, np.nan

    estimate = compute_repaired_greeks(
        S=underlying,
        K=strike,
        T=t_years,
        is_call=row.get("right") == OptionRight.CALL.value,
        mid=row.get("mid"),
        ask=row.get("ask"),
        bid=row.get("bid"),
        r=config.risk_free_rate,
        q=config.dividend_yield,
    )
    if estimate is None:
        return np.nan, np.nan, np.nan, np.nan

    return estimate.iv, estimate.delta, estimate.gamma, estimate.theta_per_day


def _normalize_index_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["event_time", "symbol", "close", "volume"])
    out = frame.copy()
    if "symbol" not in out.columns:
        out["symbol"] = symbol
    out = out[out["symbol"].astype(str).str.upper() == symbol.upper()]
    out["event_time"] = _utc_series(out["event_time"])
    if "close" not in out.columns:
        raise ValueError(f"{symbol} bars must include close")
    if "volume" not in out.columns:
        out["volume"] = 0
    return out.sort_values("event_time").reset_index(drop=True)


def _index_close_at(frame: pd.DataFrame, decision_time: pd.Timestamp) -> float:
    if frame.empty:
        return np.nan
    eligible = frame[frame["event_time"] <= decision_time]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[-1]["close"])


def _market_features(
    spx_bars: pd.DataFrame,
    vix_bars: pd.DataFrame,
    decision_time: pd.Timestamp,
) -> np.ndarray:
    spx_hist = spx_bars[spx_bars["event_time"] <= decision_time]
    if spx_hist.empty:
        return np.full(len(MARKET_FEATURE_NAMES), np.nan, dtype=float)

    spx_close = float(spx_hist.iloc[-1]["close"])
    vix_close = _index_close_at(vix_bars, decision_time)
    volume = spx_hist["volume"].astype(float)
    if volume.sum() > 0:
        vwap = float((spx_hist["close"].astype(float) * volume).sum() / volume.sum())
    else:
        vwap = float(spx_hist["close"].astype(float).mean())

    session_open = float(spx_hist.iloc[0]["close"])
    session_high = float(spx_hist["close"].astype(float).max())
    session_low = float(spx_hist["close"].astype(float).min())
    session_range = session_high - session_low
    omar = (spx_close - session_open) / session_range if session_range > 0 else 0.0
    momentum_5 = spx_close - float(spx_hist.iloc[-6]["close"]) if len(spx_hist) >= 6 else 0.0
    momentum_15 = spx_close - float(spx_hist.iloc[-16]["close"]) if len(spx_hist) >= 16 else 0.0
    return np.array(
        [spx_close, vix_close, vwap, omar, session_range, momentum_5, momentum_15],
        dtype=float,
    )


def _market_window(
    spx_bars: pd.DataFrame,
    vix_bars: pd.DataFrame,
    decision_time: pd.Timestamp,
    config: NeuralDatasetConfig,
    *,
    session_only: bool = False,
) -> np.ndarray:
    if session_only:
        local_day = decision_time.tz_convert(_NY_TZ).date()
        if not spx_bars.empty:
            spx_local = pd.to_datetime(spx_bars["event_time"], utc=True).dt.tz_convert(_NY_TZ)
            spx_bars = spx_bars[spx_local.dt.date == local_day]
        if not vix_bars.empty:
            vix_local = pd.to_datetime(vix_bars["event_time"], utc=True).dt.tz_convert(_NY_TZ)
            vix_bars = vix_bars[vix_local.dt.date == local_day]
    start = decision_time - pd.Timedelta(minutes=config.market_window_minutes - 1)
    minutes = pd.date_range(start=start, end=decision_time, freq="min", tz="UTC")
    rows = []
    for ts in minutes:
        rows.append(_market_features(spx_bars, vix_bars, ts))
    return np.vstack(rows)


def _market_context_summary(
    spx_bars: pd.DataFrame,
    decision_time: pd.Timestamp,
    config: NeuralDatasetConfig,
) -> dict:
    local_day = decision_time.tz_convert(_NY_TZ).date()
    eligible = spx_bars[spx_bars["event_time"] <= decision_time].copy()
    if not eligible.empty:
        local = eligible["event_time"].dt.tz_convert(_NY_TZ)
        eligible = eligible[local.dt.date == local_day].copy()
    if eligible.empty:
        return {
            "context_ready": False,
            "context_required_minutes": float(config.market_window_minutes),
            "context_minute_rows": 0,
            "context_span_minutes": 0.0,
            "context_start_timestamp": None,
            "context_last_timestamp": None,
        }
    minutes = eligible["event_time"].dt.floor("min").drop_duplicates().sort_values()
    first = pd.Timestamp(minutes.iloc[0])
    last = pd.Timestamp(minutes.iloc[-1])
    span = float((last - first).total_seconds() / 60.0)
    required = max(float(config.market_window_minutes), 0.0)
    required_span = max(required - 1.0, 0.0)
    minute_rows = int(len(minutes))
    return {
        "context_ready": bool(span >= required_span and minute_rows >= int(np.ceil(required))),
        "context_required_minutes": required,
        "context_minute_rows": minute_rows,
        "context_span_minutes": span,
        "context_start_timestamp": first.to_pydatetime(),
        "context_last_timestamp": last.to_pydatetime(),
    }


def _latest_quotes_at(
    options: pd.DataFrame,
    decision_time: pd.Timestamp,
    config: NeuralDatasetConfig,
) -> pd.DataFrame:
    eligible = options[options["quote_time"] <= decision_time].copy()
    if eligible.empty:
        return eligible
    eligible["_age"] = (decision_time - eligible["quote_time"]).dt.total_seconds()
    eligible = eligible[eligible["_age"] <= config.max_quote_age_seconds]
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(["contract_id", "quote_time"])
    return eligible.groupby("contract_id", as_index=False).tail(1)


def _session_dates_from_options(options: pd.DataFrame) -> list[date]:
    if options.empty:
        return []
    local_dates = options["quote_time"].dt.tz_convert(_NY_TZ).dt.date
    return sorted(set(local_dates))


def _session_decision_grid(
    session_date: date,
    config: NeuralDatasetConfig,
) -> pd.DatetimeIndex:
    open_plus_one = pd.Timestamp.combine(session_date, MARKET_OPEN_ET).tz_localize(
        _NY_TZ
    ) + pd.Timedelta(minutes=1)
    early_close = EARLY_CLOSE_TIMES_ET.get(session_date.isoformat())
    if early_close is not None:
        last_local = pd.Timestamp.combine(session_date, early_close).tz_localize(
            _NY_TZ
        ) - pd.Timedelta(minutes=1)
    else:
        last_local = pd.Timestamp.combine(
            session_date,
            config.no_new_entries_after,
        ).tz_localize(_NY_TZ)
    if last_local < open_plus_one:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(
        start=open_plus_one.tz_convert("UTC"),
        end=last_local.tz_convert("UTC"),
        freq="min",
        tz="UTC",
    )


def _decision_grid(options: pd.DataFrame, config: NeuralDatasetConfig) -> pd.DatetimeIndex:
    grids = [
        _session_decision_grid(session_date, config)
        for session_date in _session_dates_from_options(options)
    ]
    if not grids:
        return pd.DatetimeIndex([], tz="UTC")
    return grids[0].append(grids[1:]) if len(grids) > 1 else grids[0]


def _candidate_is_tradable(row: pd.Series, config: NeuralDatasetConfig) -> bool:
    if is_live_feature_contract(config.feature_contract):
        return candidate_is_tradable_values(
            {
                "bid": row.get("bid"),
                "ask": row.get("ask"),
                "mid": row.get("mid"),
                "bid_size": row.get("bid_size"),
                "ask_size": row.get("ask_size"),
                "iv": row.get("iv"),
                "delta": row.get("delta"),
                "gamma": row.get("gamma"),
                "theta": row.get("theta"),
            },
            DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT,
            require_greeks=False,
        )
    if any(pd.isna(row.get(c)) for c in ("bid", "ask", "mid")):
        return False
    bid = float(row["bid"])
    ask = float(row["ask"])
    mid = float(row["mid"])
    spread = ask - bid
    if bid < 0 or ask <= 0 or ask < bid:
        return False
    if mid < config.min_mid or mid > config.max_mid:
        return False
    if spread > config.max_spread_abs:
        return False
    if mid > 0 and spread / mid > config.max_spread_frac:
        return False
    bid_size = row.get("bid_size")
    ask_size = row.get("ask_size")
    if pd.notna(bid_size) and int(bid_size) < config.min_bid_size:
        return False
    if pd.notna(ask_size) and int(ask_size) < config.min_ask_size:
        return False
    return True


def _candidate_has_required_greeks(features: np.ndarray) -> bool:
    by_name = {name: idx for idx, name in enumerate(OPTION_FEATURE_NAMES)}
    required = ("iv", "delta", "gamma", "theta")
    return bool(np.all(np.isfinite([features[by_name[name]] for name in required])))


def _option_features(
    row: pd.Series,
    *,
    decision_time: pd.Timestamp,
    atm_strike: int,
    config: NeuralDatasetConfig,
    underlying_price: float | None = None,
) -> np.ndarray:
    live_contract = is_live_feature_contract(config.feature_contract)
    if live_contract:
        values = option_feature_values(
            row,
            decision_time=decision_time,
            spx=float(underlying_price)
            if underlying_price is not None and np.isfinite(underlying_price)
            else float(row.get("underlying_price"))
            if pd.notna(row.get("underlying_price"))
            else np.nan,
            strike=_strike_float(row.get("strike")),
            right=str(row.get("right")),
            atm_strike=atm_strike,
            feature_names=OPTION_FEATURE_NAMES,
            live_contract=True,
            risk_free_rate=config.risk_free_rate,
            dividend_yield=config.dividend_yield,
        )
        return np.asarray([values[name] for name in OPTION_FEATURE_NAMES], dtype=float)

    bid = float(row["bid"])
    ask = float(row["ask"])
    mid = float(row["mid"])
    spread = ask - bid
    spread_frac = spread / mid if mid > 0 else np.nan
    strike = _strike_float(row["strike"])
    is_call = row["right"] == OptionRight.CALL.value
    breakeven = (strike + ask) if is_call else (strike - ask)
    underlying = row.get("underlying_price")
    breakeven_distance = (
        breakeven - float(underlying)
        if pd.notna(underlying) and is_call
        else float(underlying) - breakeven
        if pd.notna(underlying)
        else np.nan
    )
    iv, delta, gamma, theta = _computed_greeks(row, decision_time, config)
    return np.array(
        [
            bid,
            ask,
            mid,
            spread,
            spread_frac,
            float(row.get("bid_size")) if pd.notna(row.get("bid_size")) else np.nan,
            float(row.get("ask_size")) if pd.notna(row.get("ask_size")) else np.nan,
            float(row.get("option_ohlcv_volume"))
            if pd.notna(row.get("option_ohlcv_volume"))
            else np.nan,
            float(row.get("stat_open_interest"))
            if pd.notna(row.get("stat_open_interest"))
            else np.nan,
            iv,
            delta,
            gamma,
            theta,
            strike - atm_strike,
            breakeven_distance,
        ],
        dtype=float,
    )


def _policy_exit_deadline(
    decision_time: pd.Timestamp,
    policy: LabelPolicy,
    config: NeuralDatasetConfig,
) -> pd.Timestamp:
    max_hold = decision_time + pd.Timedelta(minutes=policy.max_hold_minutes)
    local_day = decision_time.tz_convert(_NY_TZ).date()
    forced_local = pd.Timestamp.combine(local_day, config.forced_flat_before).tz_localize(
        _NY_TZ
    )
    forced = forced_local.tz_convert("UTC")
    return min(max_hold, forced)


def _label_for_policy(
    contract_quotes: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    entry_row: pd.Series,
    policy: LabelPolicy,
    config: NeuralDatasetConfig,
) -> tuple[float, float]:
    label = label_for_policy_two_clock_reference(
        contract_quotes,
        decision_time=decision_time,
        entry_row=entry_row,
        policy=policy,
        config=config,
        enforce_unique_path=False,
    )
    return label.net_pnl, label.mid_pnl


def _deadline_exit_reason(
    decision_time: pd.Timestamp,
    policy: LabelPolicy,
    config: NeuralDatasetConfig,
) -> ExitReason:
    max_hold = decision_time + pd.Timedelta(minutes=policy.max_hold_minutes)
    local_day = decision_time.tz_convert(_NY_TZ).date()
    forced_flat = pd.Timestamp.combine(
        local_day,
        config.forced_flat_before,
    ).tz_localize(_NY_TZ).tz_convert("UTC")
    return (
        ExitReason.FORCED_FLAT
        if forced_flat <= max_hold
        else ExitReason.MAX_HOLD
    )


def _label_pnl(
    *,
    entry_ask: float,
    entry_mid: float,
    exit_bid: float,
    exit_mid: float,
    config: NeuralDatasetConfig,
) -> tuple[float, float]:
    net = (exit_bid - entry_ask) * config.contract_multiplier
    if config.fee_per_contract:
        net -= 2.0 * config.fee_per_contract
    mid_net = (
        (exit_mid - entry_mid) * config.contract_multiplier
        if np.isfinite(exit_mid)
        else np.nan
    )
    if np.isfinite(mid_net) and config.fee_per_contract:
        mid_net -= 2.0 * config.fee_per_contract
    return float(net), float(mid_net)


def _valid_two_clock_label(
    *,
    decision_time: pd.Timestamp,
    deadline: pd.Timestamp,
    source_time_ns: int,
    reason: ExitReason,
    entry_ask: float,
    entry_mid: float,
    exit_bid: float,
    exit_mid: float,
    config: NeuralDatasetConfig,
) -> TwoClockLabel:
    source_time = pd.Timestamp(source_time_ns, unit="ns", tz="UTC")
    realized_time = (
        source_time
        if reason
        in {
            ExitReason.STOP_LOSS,
            ExitReason.TAKE_PROFIT,
            ExitReason.NO_BID_STOP,
        }
        else deadline
    )
    if source_time <= decision_time:
        return TwoClockLabel.invalid(
            InvalidReason.SOURCE_QUOTE_NOT_STRICTLY_AFTER_ENTRY,
            policy_deadline_ns=int(deadline.value),
        )
    if (
        source_time.tz_convert(_NY_TZ).date()
        != decision_time.tz_convert(_NY_TZ).date()
        or realized_time.tz_convert(_NY_TZ).date()
        != decision_time.tz_convert(_NY_TZ).date()
    ):
        return TwoClockLabel.invalid(
            InvalidReason.EXIT_CLOCK_CROSSES_SESSION,
            policy_deadline_ns=int(deadline.value),
        )
    if not np.isfinite(exit_bid) or exit_bid < 0.0:
        return TwoClockLabel.invalid(
            InvalidReason.NONFINITE_EXECUTABLE_EXIT_BID,
            policy_deadline_ns=int(deadline.value),
        )
    net, mid_net = _label_pnl(
        entry_ask=entry_ask,
        entry_mid=entry_mid,
        exit_bid=exit_bid,
        exit_mid=exit_mid,
        config=config,
    )
    age_ms = (int(realized_time.value) - int(source_time.value)) / 1_000_000.0
    return TwoClockLabel(
        net_pnl=net,
        mid_pnl=mid_net,
        realized_exit_time_ns=int(realized_time.value),
        source_exit_quote_time_ns=int(source_time.value),
        exit_quote_age_ms=float(age_ms),
        exit_reason_code=int(reason),
        executable_exit_bid=float(exit_bid),
        policy_deadline_ns=int(deadline.value),
        invalid_reason_code=int(InvalidReason.NONE),
    )


def label_for_policy_two_clock_reference(
    contract_quotes: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    entry_row: pd.Series,
    policy: LabelPolicy,
    config: NeuralDatasetConfig,
    enforce_unique_path: bool = True,
) -> TwoClockLabel:
    """Reference two-clock implementation over a quote DataFrame."""

    entry_ask = float(entry_row["ask"])
    entry_mid = float(entry_row["mid"])
    deadline = _policy_exit_deadline(decision_time, policy, config)
    if not np.isfinite(entry_ask) or entry_ask <= 0.0:
        return TwoClockLabel.invalid(
            InvalidReason.NONFINITE_OR_NONPOSITIVE_ENTRY_ASK,
            policy_deadline_ns=int(deadline.value),
        )
    if deadline <= decision_time:
        return TwoClockLabel.invalid(
            InvalidReason.DEADLINE_BEFORE_OR_AT_ENTRY,
            policy_deadline_ns=int(deadline.value),
        )
    future = contract_quotes[
        (contract_quotes["quote_time"] > decision_time)
        & (contract_quotes["quote_time"] <= deadline)
    ].sort_values("quote_time", kind="mergesort")
    if future.empty:
        return TwoClockLabel.invalid(
            InvalidReason.NO_CAUSAL_FUTURE_QUOTE_AT_OR_BEFORE_DEADLINE,
            policy_deadline_ns=int(deadline.value),
        )
    if enforce_unique_path and future["quote_time"].duplicated(keep=False).any():
        duplicate_time = future.loc[
            future["quote_time"].duplicated(keep=False), "quote_time"
        ].iloc[0]
        raise Protocol101DuplicatePathQuoteIdentityError(
            "duplicate contract/source-time path quote identity",
            canonical_key=(
                str(entry_row.get("contract_id") or ""),
                pd.Timestamp(duplicate_time).isoformat(),
            ),
            observed_count=int((future["quote_time"] == duplicate_time).sum()),
            boundary="two-clock reference label path",
        )

    stop_bid = entry_ask * (1.0 - policy.stop_loss_pct)
    target_bid = entry_ask * (1.0 + policy.take_profit_pct)
    exit_row = future.iloc[-1]
    exit_reason = _deadline_exit_reason(decision_time, policy, config)
    for _, row in future.iterrows():
        # No-bid convention: absent bid is an executable 0.00 on the exit path.
        bid = 0.0 if pd.isna(row.get("bid")) else float(row.get("bid"))
        if bid <= 0.0:
            exit_row = row
            exit_reason = ExitReason.NO_BID_STOP
            break
        if bid <= stop_bid:
            exit_row = row
            exit_reason = ExitReason.STOP_LOSS
            break
        if bid >= target_bid:
            exit_row = row
            exit_reason = ExitReason.TAKE_PROFIT
            break

    exit_bid = 0.0 if pd.isna(exit_row["bid"]) else float(exit_row["bid"])
    exit_mid = float(exit_row["mid"]) if pd.notna(exit_row["mid"]) else np.nan
    return _valid_two_clock_label(
        decision_time=decision_time,
        deadline=deadline,
        source_time_ns=int(pd.Timestamp(exit_row["quote_time"]).value),
        reason=exit_reason,
        entry_ask=entry_ask,
        entry_mid=entry_mid,
        exit_bid=exit_bid,
        exit_mid=exit_mid,
        config=config,
    )


def _contract_quote_path(
    contract_quotes: pd.DataFrame,
    *,
    enforce_unique_path: bool = False,
) -> ContractQuotePath:
    """Convert one contract quote path into arrays for label computation.

    No-bid convention (pinned 2026-07-07): an absent bid on the exit path is
    an executable value of 0.00. Databento CBBO encoded absent bids as 0.00
    through 2025-02-19 and as null from 2025-02-20; both mean "nobody will
    pay anything for this contract right now", and 0.00 is the worst-case
    honest exit for a long option. Applies to the LABEL path only — features
    and entry tradability keep NaN semantics.
    """
    quotes = contract_quotes.sort_values("quote_time", kind="mergesort")
    quote_ns = (
        pd.to_datetime(quotes["quote_time"], utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy()
    )
    if enforce_unique_path and len(quote_ns) != len(np.unique(quote_ns)):
        duplicate_ns, counts = np.unique(quote_ns, return_counts=True)
        selected = int(duplicate_ns[np.flatnonzero(counts > 1)[0]])
        raise Protocol101DuplicatePathQuoteIdentityError(
            "duplicate contract/source-time path quote identity",
            canonical_key=(
                str(quotes.iloc[0].get("contract_id") or ""),
                selected,
            ),
            observed_count=int(counts[np.flatnonzero(counts > 1)[0]]),
            boundary="two-clock vectorized label path",
        )
    return ContractQuotePath(
        quote_ns=quote_ns,
        bid=pd.to_numeric(quotes["bid"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        mid=pd.to_numeric(quotes["mid"], errors="coerce").to_numpy(dtype=float),
    )


def _label_for_policy_from_path(
    quote_path: ContractQuotePath,
    *,
    decision_time: pd.Timestamp,
    entry_ask: float,
    entry_mid: float,
    policy: LabelPolicy,
    config: NeuralDatasetConfig,
) -> tuple[float, float]:
    """Legacy return shape backed by the two-clock vectorized kernel."""
    label = label_for_policy_two_clock_from_path(
        quote_path,
        decision_time=decision_time,
        entry_ask=entry_ask,
        entry_mid=entry_mid,
        policy=policy,
        config=config,
    )
    return label.net_pnl, label.mid_pnl


def label_for_policy_two_clock_from_path(
    quote_path: ContractQuotePath,
    *,
    decision_time: pd.Timestamp,
    entry_ask: float,
    entry_mid: float,
    policy: LabelPolicy,
    config: NeuralDatasetConfig,
) -> TwoClockLabel:
    """Vectorized two-clock label kernel for bulk historical builds."""

    deadline = _policy_exit_deadline(decision_time, policy, config)
    if not np.isfinite(entry_ask) or entry_ask <= 0.0:
        return TwoClockLabel.invalid(
            InvalidReason.NONFINITE_OR_NONPOSITIVE_ENTRY_ASK,
            policy_deadline_ns=int(deadline.value),
        )
    if deadline <= decision_time:
        return TwoClockLabel.invalid(
            InvalidReason.DEADLINE_BEFORE_OR_AT_ENTRY,
            policy_deadline_ns=int(deadline.value),
        )
    start = int(np.searchsorted(quote_path.quote_ns, decision_time.value, side="right"))
    end = int(np.searchsorted(quote_path.quote_ns, deadline.value, side="right"))
    if start >= end:
        return TwoClockLabel.invalid(
            InvalidReason.NO_CAUSAL_FUTURE_QUOTE_AT_OR_BEFORE_DEADLINE,
            policy_deadline_ns=int(deadline.value),
        )

    future_bid = quote_path.bid[start:end]
    finite_bid = np.isfinite(future_bid)
    stop_bid = entry_ask * (1.0 - policy.stop_loss_pct)
    target_bid = entry_ask * (1.0 + policy.take_profit_pct)
    hit_mask = finite_bid & (
        (future_bid <= 0.0)
        | (future_bid <= stop_bid)
        | (future_bid >= target_bid)
    )
    if hit_mask.any():
        exit_idx = start + int(np.flatnonzero(hit_mask)[0])
    else:
        exit_idx = end - 1

    exit_bid = float(quote_path.bid[exit_idx])
    exit_mid = float(quote_path.mid[exit_idx])
    if hit_mask.any():
        reason = (
            ExitReason.NO_BID_STOP
            if exit_bid <= 0.0
            else ExitReason.STOP_LOSS
            if exit_bid <= stop_bid
            else ExitReason.TAKE_PROFIT
        )
    else:
        reason = _deadline_exit_reason(decision_time, policy, config)
    return _valid_two_clock_label(
        decision_time=decision_time,
        deadline=deadline,
        source_time_ns=int(quote_path.quote_ns[exit_idx]),
        reason=reason,
        entry_ask=float(entry_ask),
        entry_mid=float(entry_mid),
        exit_bid=exit_bid,
        exit_mid=exit_mid,
        config=config,
    )


def label_for_policy_two_clock_scalar_reference_from_path(
    quote_path: ContractQuotePath,
    *,
    decision_time: pd.Timestamp,
    entry_ask: float,
    entry_mid: float,
    policy: LabelPolicy,
    config: NeuralDatasetConfig,
) -> TwoClockLabel:
    """Independent scalar array oracle used by non-economic corpus checks."""

    deadline = _policy_exit_deadline(decision_time, policy, config)
    if not np.isfinite(entry_ask) or entry_ask <= 0.0:
        return TwoClockLabel.invalid(
            InvalidReason.NONFINITE_OR_NONPOSITIVE_ENTRY_ASK,
            policy_deadline_ns=int(deadline.value),
        )
    if deadline <= decision_time:
        return TwoClockLabel.invalid(
            InvalidReason.DEADLINE_BEFORE_OR_AT_ENTRY,
            policy_deadline_ns=int(deadline.value),
        )
    stop_bid = entry_ask * (1.0 - policy.stop_loss_pct)
    target_bid = entry_ask * (1.0 + policy.take_profit_pct)
    selected_index: int | None = None
    selected_reason: ExitReason | None = None
    start = bisect_right(quote_path.quote_ns, int(decision_time.value))
    for index in range(start, len(quote_path.quote_ns)):
        quote_time_ns = int(quote_path.quote_ns[index])
        if quote_time_ns > int(deadline.value):
            break
        selected_index = index
        bid = float(quote_path.bid[index])
        if np.isfinite(bid) and bid <= 0.0:
            selected_reason = ExitReason.NO_BID_STOP
            break
        if np.isfinite(bid) and bid <= stop_bid:
            selected_reason = ExitReason.STOP_LOSS
            break
        if np.isfinite(bid) and bid >= target_bid:
            selected_reason = ExitReason.TAKE_PROFIT
            break
    if selected_index is None:
        return TwoClockLabel.invalid(
            InvalidReason.NO_CAUSAL_FUTURE_QUOTE_AT_OR_BEFORE_DEADLINE,
            policy_deadline_ns=int(deadline.value),
        )
    reason = selected_reason or _deadline_exit_reason(
        decision_time, policy, config
    )
    return _valid_two_clock_label(
        decision_time=decision_time,
        deadline=deadline,
        source_time_ns=int(quote_path.quote_ns[selected_index]),
        reason=reason,
        entry_ask=entry_ask,
        entry_mid=entry_mid,
        exit_bid=float(quote_path.bid[selected_index]),
        exit_mid=float(quote_path.mid[selected_index]),
        config=config,
    )


def _prepare_options(normalized: pa.Table | pd.DataFrame, config: NeuralDatasetConfig) -> pd.DataFrame:
    options = _to_frame(normalized)
    if options.empty:
        return options

    options = options.copy()
    options["quote_time"] = _utc_series(
        options["quote_time"].where(options["quote_time"].notna(), options["event_time"])
    )
    options["event_time"] = _utc_series(options["event_time"])
    options["strike_float"] = options["strike"].map(_strike_float)
    options = options[
        (options["root"] == "SPXW")
        & (options["settlement_style"] == "PM")
        & (np.isclose(options["strike_float"] % config.strike_step, 0.0))
    ].copy()
    if options.empty:
        return options

    options["mid"] = options["mid"].where(
        options["mid"].notna(), (options["bid"] + options["ask"]) / 2.0
    )
    local_times = options["quote_time"].dt.tz_convert(_NY_TZ).dt.time
    options = options[local_times <= config.forced_flat_before].copy()
    return options.sort_values(["quote_time", "contract_id"]).reset_index(drop=True)


def build_neural_dataset(
    normalized: pa.Table | pd.DataFrame,
    spx_bars: pd.DataFrame,
    vix_bars: pd.DataFrame | None = None,
    *,
    config: NeuralDatasetConfig | None = None,
    label_policies: Sequence[LabelPolicy] | None = None,
) -> list[dict]:
    """Create SPXW 0DTE decision rows with executable long-option labels.

    Each output row contains the market window, option ladder tensor, candidate
    mask, contract IDs, and ask-entry/bid-exit labels for each candidate.
    """
    config = config or NeuralDatasetConfig()
    if config.processed_row_schema_version not in {
        LEGACY_PROCESSED_ROW_SCHEMA,
        TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    }:
        raise ValueError(
            "unsupported processed_row_schema_version: "
            f"{config.processed_row_schema_version}"
        )
    two_clock_rows = (
        config.processed_row_schema_version == TWO_CLOCK_PROCESSED_ROW_SCHEMA
    )
    if two_clock_rows and not config.compute_policy_labels:
        raise ValueError(
            "two-clock processed rows require computed offline policy labels"
        )
    policies = tuple(label_policies) if label_policies is not None else config.label_policies
    if two_clock_rows and policies != NeuralDatasetConfig().label_policies:
        raise ValueError(
            "two-clock processed rows require the exact signed seven-policy axis"
        )
    options = _prepare_options(normalized, config)
    if options.empty:
        return []

    spx = _normalize_index_frame(spx_bars, "SPX")
    vix = _normalize_index_frame(
        vix_bars if vix_bars is not None else pd.DataFrame(), "VIX"
    )
    if spx.empty:
        raise ValueError("SPX one-minute bars are required")

    by_contract = {
        cid: group.sort_values("quote_time").reset_index(drop=True)
        for cid, group in options.groupby("contract_id")
    }
    by_contract_path = {
        cid: _contract_quote_path(
            group,
            enforce_unique_path=two_clock_rows,
        )
        for cid, group in by_contract.items()
    }
    strike_offsets = np.arange(
        -config.ladder_dollars,
        config.ladder_dollars + config.strike_step,
        config.strike_step,
    )
    rights = (OptionRight.CALL.value, OptionRight.PUT.value)
    decision_times = list(_decision_grid(options, config))
    rows: list[dict] = []
    live_contract = is_live_feature_contract(config.feature_contract)
    contract_version = feature_contract_version(config.feature_contract)
    require_model_scoring_greeks = feature_contract_requires_model_scoring_greeks(contract_version)
    if config.diagnostic_index_context_lag_minutes is not None and not live_contract:
        raise ValueError("diagnostic_index_context_lag_minutes is only supported for live feature-contract rows")
    context_lag_minutes = (
        DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT.index_context_lag_minutes
        if live_contract
        else 0
    )
    if config.diagnostic_index_context_lag_minutes is not None:
        context_lag_minutes = int(config.diagnostic_index_context_lag_minutes)
        if context_lag_minutes < 0:
            raise ValueError("diagnostic_index_context_lag_minutes must be >= 0")
    contract_payload = feature_contract_metadata(config.feature_contract)
    if config.diagnostic_index_context_lag_minutes is not None or config.diagnostic_source_policy:
        contract_payload = {
            **contract_payload,
            "diagnostic_only": True,
            "diagnostic_index_context_lag_minutes": context_lag_minutes,
            "diagnostic_source_policy": config.diagnostic_source_policy
            or f"diagnostic_index_context_lag_{context_lag_minutes}m",
            "production_contract_mutation": False,
        }

    for decision_time in decision_times:
        context_time = (
            decision_time - pd.Timedelta(minutes=context_lag_minutes)
            if live_contract
            else decision_time
        )
        spx_close = _index_close_at(spx, context_time)
        if not np.isfinite(spx_close):
            continue
        atm_strike = _round_to_step(spx_close, config.strike_step)
        ladder_context = strike_ladder_context(
            spx_for_ladder=spx_close,
            strike_step=config.strike_step,
        )
        context_summary = _market_context_summary(spx, context_time, config)
        ladder_quotes = _latest_quotes_at(options, decision_time, config)
        by_key = (
            {
                (int(round(row["strike_float"])), row["right"]): row
                for _, row in ladder_quotes.iterrows()
            }
            if not ladder_quotes.empty
            else {}
        )

        option_ladder = np.full(
            (len(strike_offsets), len(rights), len(OPTION_FEATURE_NAMES)),
            np.nan,
            dtype=float,
        )
        candidate_mask = np.zeros((len(strike_offsets), len(rights)), dtype=bool)
        contract_ids = np.full((len(strike_offsets), len(rights)), None, dtype=object)
        label_shape = (len(strike_offsets), len(rights), len(policies))
        labels_net = (
            np.full(label_shape, np.nan)
            if config.compute_policy_labels
            else np.zeros(label_shape, dtype=float)
        )
        labels_mid = labels_net.copy()
        label_policy_index = np.arange(len(policies), dtype=np.uint8)
        label_realized_exit_time_ns = np.full(
            label_shape, INT64_MISSING, dtype=np.int64
        )
        label_source_exit_quote_time_ns = np.full(
            label_shape, INT64_MISSING, dtype=np.int64
        )
        label_exit_quote_age_ms = np.full(label_shape, np.nan, dtype=np.float64)
        label_exit_reason_code = np.full(
            label_shape, int(ExitReason.INVALID), dtype=np.uint8
        )
        label_executable_exit_bid = np.full(
            label_shape, np.nan, dtype=np.float64
        )
        label_policy_deadline_ns = np.full(
            label_shape, INT64_MISSING, dtype=np.int64
        )
        label_invalid_reason_code = np.full(
            label_shape,
            int(InvalidReason.AXIS_OR_POLICY_ALIGNMENT_FAILURE),
            dtype=np.uint8,
        )
        candidate_quote_metadata: dict[str, dict] = {}

        for slot in candidate_ladder_slots(
            spx_for_ladder=spx_close,
            ladder_dollars=config.ladder_dollars,
            strike_step=config.strike_step,
            rights=rights,
        ):
            strike_idx = int(slot["strike_idx"])
            right_idx = int(slot["right_idx"])
            offset = float(slot["offset"])
            strike = int(slot["strike"])
            right = str(slot["right"])
            row = by_key.get((strike, right))
            if row is None:
                contract_id = f"SPXW-{decision_time.tz_convert(_NY_TZ).strftime('%Y%m%d')}-{float(strike):09.3f}-{right}"
                contract_ids[strike_idx, right_idx] = contract_id
                no_quote = missing_candidate_slot_diagnostics("no_historical_quote_at_decision")
                candidate_quote_metadata[contract_id] = {
                    "feature_contract_version": contract_version,
                    "contract_id": contract_id,
                    **slot,
                    "source_quote_time": None,
                    "source_quote_ts": None,
                    "source_context_time": None,
                    "source_context_ts": None,
                    "quote_age_ms": None,
                    "quote_age_source": "no_historical_quote_at_decision",
                    "raw_quote_timestamp_utc": None,
                    "received_timestamp_utc": None,
                    "decision_timestamp_utc": decision_time.isoformat(),
                    "bid": None,
                    "ask": None,
                    "mid": None,
                    "bid_size": None,
                    "ask_size": None,
                    "option_ohlcv_volume": None,
                    "stat_open_interest": None,
                    "pre_filter_candidate": False,
                    "post_filter_candidate": False,
                    "candidate_filter": no_quote,
                    "filter_reasons": list(no_quote["reasons"]),
                    "tradability_pass": False,
                    "freshness_pass": None,
                }
                continue
            contract_ids[strike_idx, right_idx] = row["contract_id"]
            metadata = quote_source_metadata(
                row,
                decision_time=decision_time,
                feature_contract_name=config.feature_contract,
            )
            metadata.update(
                {
                    "contract_id": row["contract_id"],
                    "strike": float(strike),
                    "right": str(right),
                    "offset": float(offset),
                    "strike_idx": int(strike_idx),
                    "right_idx": int(right_idx),
                    **ladder_context,
                    "bid": float(row["bid"]) if pd.notna(row.get("bid")) else None,
                    "ask": float(row["ask"]) if pd.notna(row.get("ask")) else None,
                    "mid": float(row["mid"]) if pd.notna(row.get("mid")) else None,
                    "bid_size": float(row["bid_size"]) if pd.notna(row.get("bid_size")) else None,
                    "ask_size": float(row["ask_size"]) if pd.notna(row.get("ask_size")) else None,
                    "option_ohlcv_volume": 0.0
                    if live_contract
                    else (float(row["option_ohlcv_volume"]) if pd.notna(row.get("option_ohlcv_volume")) else None),
                    "stat_open_interest": 0.0
                    if live_contract
                    else (float(row["stat_open_interest"]) if pd.notna(row.get("stat_open_interest")) else None),
                }
            )
            filter_values = {
                "bid": metadata.get("bid"),
                "ask": metadata.get("ask"),
                "mid": metadata.get("mid"),
                "bid_size": metadata.get("bid_size"),
                "ask_size": metadata.get("ask_size"),
                "quote_age_ms": metadata.get("quote_age_ms"),
                "iv": row.get("iv"),
                "delta": row.get("delta"),
                "gamma": row.get("gamma"),
                "theta": row.get("theta"),
            }
            filter_diagnostics = candidate_filter_diagnostics(
                filter_values,
                DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT,
                require_greeks=False,
                enforce_freshness=True,
            )
            metadata.update(
                {
                    "pre_filter_candidate": True,
                    "post_filter_candidate": False,
                    "candidate_filter": filter_diagnostics,
                    "filter_reasons": list(filter_diagnostics.get("reasons") or []),
                    "tradability_pass": filter_diagnostics.get("tradability_pass"),
                    "freshness_pass": filter_diagnostics.get("freshness_pass"),
                }
            )
            if not _candidate_is_tradable(row, config):
                candidate_quote_metadata[str(row["contract_id"])] = metadata
                continue
            features = _option_features(
                row,
                decision_time=decision_time,
                atm_strike=atm_strike,
                config=config,
                underlying_price=spx_close if live_contract else None,
            )
            if require_model_scoring_greeks and not _candidate_has_required_greeks(features):
                reasons = sorted(set((metadata.get("filter_reasons") or []) + ["missing_model_scoring_greeks"]))
                metadata["filter_reasons"] = reasons
                metadata["model_scoring_greek_pass"] = False
                if isinstance(metadata.get("candidate_filter"), dict):
                    metadata["candidate_filter"]["reasons"] = reasons
                    metadata["candidate_filter"]["passed"] = False
                candidate_quote_metadata[str(row["contract_id"])] = metadata
                continue
            for idx, name in enumerate(OPTION_FEATURE_NAMES):
                metadata[name] = float(features[idx]) if np.isfinite(features[idx]) else None
            metadata["post_filter_candidate"] = True
            metadata["model_scoring_greek_pass"] = True
            candidate_mask[strike_idx, right_idx] = True
            option_ladder[strike_idx, right_idx, :] = features
            candidate_quote_metadata[str(row["contract_id"])] = metadata
            if config.compute_policy_labels:
                contract_path = by_contract_path[row["contract_id"]]
                for policy_idx, policy in enumerate(policies):
                    two_clock_label = label_for_policy_two_clock_from_path(
                        contract_path,
                        decision_time=decision_time,
                        entry_ask=float(row["ask"]),
                        entry_mid=float(row["mid"]),
                        policy=policy,
                        config=config,
                    )
                    labels_net[strike_idx, right_idx, policy_idx] = (
                        two_clock_label.net_pnl
                    )
                    labels_mid[strike_idx, right_idx, policy_idx] = (
                        two_clock_label.mid_pnl
                    )
                    if two_clock_rows:
                        index = (strike_idx, right_idx, policy_idx)
                        label_realized_exit_time_ns[index] = (
                            two_clock_label.realized_exit_time_ns
                        )
                        label_source_exit_quote_time_ns[index] = (
                            two_clock_label.source_exit_quote_time_ns
                        )
                        label_exit_quote_age_ms[index] = (
                            two_clock_label.exit_quote_age_ms
                        )
                        label_exit_reason_code[index] = (
                            two_clock_label.exit_reason_code
                        )
                        label_executable_exit_bid[index] = (
                            two_clock_label.executable_exit_bid
                        )
                        label_policy_deadline_ns[index] = (
                            two_clock_label.policy_deadline_ns
                        )
                        label_invalid_reason_code[index] = (
                            two_clock_label.invalid_reason_code
                        )

        if not candidate_mask.any():
            labels_net = np.full(label_shape, np.nan)
            labels_mid = np.full(label_shape, np.nan)
        source_quote_times = [
            pd.Timestamp(item["source_quote_time"])
            for item in candidate_quote_metadata.values()
            if item.get("source_quote_time")
        ]
        source_quote_time = max(source_quote_times).to_pydatetime() if source_quote_times else decision_time.to_pydatetime()
        source_context_time = context_summary.get("context_last_timestamp")
        for metadata in candidate_quote_metadata.values():
            metadata["source_context_time"] = (
                source_context_time.isoformat() if hasattr(source_context_time, "isoformat") else source_context_time
            )
            metadata["source_context_ts"] = metadata["source_context_time"]
        ladder_context["source_context_time"] = (
            source_context_time.isoformat() if hasattr(source_context_time, "isoformat") else source_context_time
        )
        ladder_context["source_context_ts"] = ladder_context["source_context_time"]
        candidate_filter_trace = sorted(
            [dict(item) for item in candidate_quote_metadata.values()],
            key=lambda item: (
                int(item.get("strike_idx") or 0),
                int(item.get("right_idx") or 0),
                str(item.get("contract_id") or ""),
            ),
        )
        max_quote_age_ms = max(
            [
                float(item["quote_age_ms"])
                for item in candidate_quote_metadata.values()
                if item.get("quote_age_ms") is not None
            ],
            default=np.nan,
        )
        two_clock_payload = (
            {
                "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
                "label_realized_exit_time_ns": label_realized_exit_time_ns,
                "label_source_exit_quote_time_ns": (
                    label_source_exit_quote_time_ns
                ),
                "label_exit_quote_age_ms": label_exit_quote_age_ms,
                "label_exit_reason_code": label_exit_reason_code,
                "label_executable_exit_bid": label_executable_exit_bid,
                "label_policy_deadline_ns": label_policy_deadline_ns,
                "label_policy_index": label_policy_index,
                "label_invalid_reason_code": label_invalid_reason_code,
            }
            if two_clock_rows
            else {}
        )
        rows.append(
            {
                "decision_time": decision_time.to_pydatetime(),
                "source_quote_time": source_quote_time,
                "source_context_time": source_context_time,
                "max_quote_age_ms": max_quote_age_ms,
                "feature_contract_version": contract_version,
                "feature_contract": contract_payload,
                "decision_grid": DECISION_GRID_VERSION,
                "position_state": "flat",
                "atm_strike": atm_strike,
                "strike_offsets": strike_offsets.copy(),
                "rights": rights,
                "feature_names": tuple(OPTION_FEATURE_NAMES),
                "option_ladder": option_ladder,
                "candidate_mask": candidate_mask,
                "contract_ids": contract_ids,
                "label_names": tuple(policy.name for policy in policies),
                "labels_net_pnl": labels_net,
                "labels_mid_pnl": labels_mid,
                **two_clock_payload,
                "market_feature_names": tuple(MARKET_FEATURE_NAMES),
                "market_window": _market_window(
                    spx,
                    vix,
                    context_time,
                    config,
                    session_only=live_contract,
                ),
                "contract_quote_metadata": candidate_quote_metadata,
                "candidate_filter_trace": candidate_filter_trace,
                "ladder_context": dict(ladder_context),
                **context_summary,
            }
        )

    return rows
