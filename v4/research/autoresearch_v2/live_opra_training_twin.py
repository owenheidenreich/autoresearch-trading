"""Shared clock law for a historical/live-identical OPRA entry regimen.

The module is intentionally model-free.  It defines the market-time cutoffs
that a future distinct model generation must use on both recorded history and
live shadow data.  A minute boundary ``t`` represents the closed interval
``[t-60s, t)``.  The official SPX bar for that interval is stamped ``t-60s``;
the OPRA CBBO-1m snapshot is stamped ``t``.  Both must be received before the
fixed emission clock.  Entry execution and labels begin later, from a fresh
CBBO-1s quote at the frozen arrival clock.

This is not the rejected crossed-time workaround for the immutable model: no
feature from ``[t, t+60s)`` is paired with the option state at ``t``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd


SECOND_NS = 1_000_000_000
MINUTE_NS = 60 * SECOND_NS
DEFAULT_MAX_ENTRY_QUOTE_AGE_NS = 2 * SECOND_NS
SCHEMA_VERSION = "autoresearch.live-opra-training-twin.v1"


class LiveTwinClockError(RuntimeError):
    """Raised when a source row cannot lawfully enter the shared regimen."""


@dataclass(frozen=True)
class TrainingTwinClockV1:
    """The four ordered clocks for one compiled entry example."""

    interval_start_ns: int
    feature_boundary_ns: int
    decision_emission_ns: int
    entry_arrival_ns: int
    label_deadline_ns: int
    hold_minutes: int

    def __post_init__(self) -> None:
        if self.feature_boundary_ns % MINUTE_NS:
            raise LiveTwinClockError("feature boundary must be an exact UTC minute")
        if self.interval_start_ns != self.feature_boundary_ns - MINUTE_NS:
            raise LiveTwinClockError("feature interval must be exactly one closed minute")
        if not (
            self.feature_boundary_ns
            <= self.decision_emission_ns
            <= self.entry_arrival_ns
            < self.label_deadline_ns
        ):
            raise LiveTwinClockError("training-twin clocks are not monotone")
        if self.label_deadline_ns != self.entry_arrival_ns + self.hold_minutes * MINUTE_NS:
            raise LiveTwinClockError("label deadline must start from executable arrival")


@dataclass(frozen=True)
class SelectedSourceRowV1:
    source: str
    stable_contract_id: str | None
    represented_interval_start_ns: int
    represented_interval_end_ns: int
    source_timestamp_ns: int
    received_at_ns: int
    age_at_consumption_ns: int
    row_index: int


@dataclass(frozen=True)
class CompiledEntrySourceReceiptV1:
    schema_version: str
    clock: TrainingTwinClockV1
    official_spx: SelectedSourceRowV1
    option_feature_quote: SelectedSourceRowV1
    executable_entry_quote: SelectedSourceRowV1
    stable_contract_id: str
    receipt_sha256: str


@dataclass(frozen=True)
class CurrentSessionContractV1:
    stable_contract_id: str
    current_instrument_id: int
    expiration_date: str
    right: str
    strike: float
    definition_received_at_ns: int


_OSI_RE = re.compile(r"^SPXW  (?P<expiry>[0-9]{6})(?P<right>[CP])(?P<strike>[0-9]{8})$")


def current_session_definition_universe(
    frame: pd.DataFrame,
    *,
    session_date: str,
    as_of_ns: int,
) -> dict[str, CurrentSessionContractV1]:
    """Apply add/modify/delete definition updates through one causal cutoff."""

    required = {
        "ts_recv",
        "raw_symbol",
        "expiration",
        "asset",
        "instrument_id",
        "instrument_class",
        "strike_price",
        "security_update_action",
    }
    missing = required - set(frame.columns)
    if missing:
        raise LiveTwinClockError(f"definition frame missing columns: {sorted(missing)}")
    working = frame.copy()
    working["_ts_recv_ns"] = _timestamp_ns(working, "ts_recv")
    working["_expiration"] = pd.to_datetime(
        working["expiration"], utc=True, errors="coerce"
    ).dt.date.astype(str)
    working = working[
        (working["_ts_recv_ns"] <= int(as_of_ns))
        & working["_expiration"].eq(session_date)
        & working["asset"].astype(str).eq("SPXW")
        & working["raw_symbol"].astype(str).str.startswith("SPXW  ")
    ].sort_values(["_ts_recv_ns", "raw_symbol"])
    latest = working.drop_duplicates("raw_symbol", keep="last")
    latest = latest[~latest["security_update_action"].astype(str).isin({"D", "DELETE"})]
    universe: dict[str, CurrentSessionContractV1] = {}
    for _, row in latest.iterrows():
        symbol = str(row["raw_symbol"])
        match = _OSI_RE.fullmatch(symbol)
        if match is None:
            raise LiveTwinClockError(f"invalid SPXW OSI raw symbol: {symbol}")
        expiry = pd.Timestamp(session_date).strftime("%y%m%d")
        right = match.group("right")
        strike = int(match.group("strike")) / 1000.0
        if (
            match.group("expiry") != expiry
            or str(row["instrument_class"]) != right
            or not np.isclose(float(row["strike_price"]), strike, atol=0.0, rtol=0.0)
        ):
            raise LiveTwinClockError(f"definition/OSI geometry mismatch: {symbol}")
        instrument_id = int(row["instrument_id"])
        if instrument_id <= 0 or symbol in universe:
            raise LiveTwinClockError(f"invalid or duplicate live definition: {symbol}")
        universe[symbol] = CurrentSessionContractV1(
            stable_contract_id=symbol,
            current_instrument_id=instrument_id,
            expiration_date=session_date,
            right=right,
            strike=strike,
            definition_received_at_ns=int(row["_ts_recv_ns"]),
        )
    if not universe:
        raise LiveTwinClockError("current-session definition universe is empty")
    return dict(sorted(universe.items()))


def make_clock(
    *,
    feature_boundary_ns: int,
    emission_lag_ms: int,
    order_latency_ms: int,
    hold_minutes: int = 25,
) -> TrainingTwinClockV1:
    """Compile the fixed clocks; parameters must be frozen before any fit."""

    if not 0 <= emission_lag_ms <= 10_000:
        raise LiveTwinClockError("emission lag must be between 0 and 10 seconds")
    if not 0 <= order_latency_ms <= 10_000:
        raise LiveTwinClockError("order latency must be between 0 and 10 seconds")
    if not 1 <= hold_minutes <= 390:
        raise LiveTwinClockError("hold minutes out of bounds")
    emission_ns = int(feature_boundary_ns) + int(emission_lag_ms) * 1_000_000
    arrival_ns = emission_ns + int(order_latency_ms) * 1_000_000
    return TrainingTwinClockV1(
        interval_start_ns=int(feature_boundary_ns) - MINUTE_NS,
        feature_boundary_ns=int(feature_boundary_ns),
        decision_emission_ns=emission_ns,
        entry_arrival_ns=arrival_ns,
        label_deadline_ns=arrival_ns + int(hold_minutes) * MINUTE_NS,
        hold_minutes=int(hold_minutes),
    )


def _timestamp_ns(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise LiveTwinClockError(f"missing timestamp column: {name}")
    parsed = pd.to_datetime(frame[name], utc=True, errors="coerce")
    if parsed.isna().any():
        raise LiveTwinClockError(f"invalid timestamp in {name}")
    return (
        parsed.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )


def _received_ns(
    frame: pd.DataFrame, *, fallback: np.ndarray, received_column: str
) -> np.ndarray:
    if received_column not in frame.columns:
        return fallback.copy()
    values = pd.to_numeric(frame[received_column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise LiveTwinClockError(f"invalid receipt timestamp in {received_column}")
    return values.astype(np.int64)


def _valid_bbo(row: pd.Series) -> None:
    bid = float(row["bid_px_00"])
    ask = float(row["ask_px_00"])
    if not np.isfinite([bid, ask]).all() or bid < 0.0 or ask <= bid:
        raise LiveTwinClockError("selected BBO is missing, locked, or crossed")


def select_official_spx(
    frame: pd.DataFrame,
    *,
    clock: TrainingTwinClockV1,
    received_column: str = "received_at_ns",
) -> tuple[pd.Series, SelectedSourceRowV1]:
    """Select the SPX bar for exactly ``[t-60s, t)`` and never the next bar."""

    event_ns = _timestamp_ns(frame, "event_time")
    represented_end = event_ns + MINUTE_NS
    received = _received_ns(
        frame, fallback=represented_end, received_column=received_column
    )
    eligible = np.flatnonzero(
        (represented_end == clock.feature_boundary_ns)
        & (received <= clock.decision_emission_ns)
    )
    if len(eligible) != 1:
        raise LiveTwinClockError(
            f"expected one exact completed SPX bar, found {len(eligible)}"
        )
    index = int(eligible[0])
    row = frame.iloc[index]
    close = float(row["close"])
    if not np.isfinite(close) or close <= 0.0:
        raise LiveTwinClockError("selected SPX close is invalid")
    receipt = SelectedSourceRowV1(
        source="THETADATA_OFFICIAL_SPX_1M",
        stable_contract_id=None,
        represented_interval_start_ns=int(event_ns[index]),
        represented_interval_end_ns=int(represented_end[index]),
        source_timestamp_ns=int(event_ns[index]),
        received_at_ns=int(received[index]),
        age_at_consumption_ns=clock.decision_emission_ns - int(received[index]),
        row_index=index,
    )
    return row, receipt


def _select_option_quote(
    frame: pd.DataFrame,
    *,
    stable_contract_id: str,
    represented_end_ns: int,
    consumed_at_ns: int,
    exact_boundary: bool,
    max_age_ns: int,
    source: str,
    received_column: str,
) -> tuple[pd.Series, SelectedSourceRowV1]:
    required = {"symbol", "ts_recv", "bid_px_00", "ask_px_00"}
    missing = required - set(frame.columns)
    if missing:
        raise LiveTwinClockError(f"option frame missing columns: {sorted(missing)}")
    ts_recv = _timestamp_ns(frame, "ts_recv")
    received = _received_ns(frame, fallback=ts_recv, received_column=received_column)
    same = frame["symbol"].astype(str).to_numpy() == stable_contract_id
    if exact_boundary:
        time_ok = ts_recv == represented_end_ns
    else:
        time_ok = (ts_recv >= represented_end_ns) & (ts_recv <= consumed_at_ns)
    eligible = np.flatnonzero(same & time_ok & (received <= consumed_at_ns))
    if exact_boundary and len(eligible) != 1:
        raise LiveTwinClockError(
            f"expected one exact CBBO-1m row for {stable_contract_id}, found {len(eligible)}"
        )
    if not len(eligible):
        raise LiveTwinClockError(f"no causal option quote for {stable_contract_id}")
    index = int(eligible[np.argmax(ts_recv[eligible])])
    age = consumed_at_ns - int(ts_recv[index])
    if age < 0 or age > max_age_ns:
        raise LiveTwinClockError(f"option quote age out of bounds: {age}")
    row = frame.iloc[index]
    _valid_bbo(row)
    receipt = SelectedSourceRowV1(
        source=source,
        stable_contract_id=stable_contract_id,
        represented_interval_start_ns=(
            int(ts_recv[index]) - (MINUTE_NS if exact_boundary else SECOND_NS)
        ),
        represented_interval_end_ns=int(ts_recv[index]),
        source_timestamp_ns=int(ts_recv[index]),
        received_at_ns=int(received[index]),
        age_at_consumption_ns=age,
        row_index=index,
    )
    return row, receipt


def select_option_feature_quote(
    frame: pd.DataFrame,
    *,
    stable_contract_id: str,
    clock: TrainingTwinClockV1,
    received_column: str = "received_at_ns",
) -> tuple[pd.Series, SelectedSourceRowV1]:
    """Select native CBBO-1m at exactly ``t`` for model features."""

    return _select_option_quote(
        frame,
        stable_contract_id=stable_contract_id,
        represented_end_ns=clock.feature_boundary_ns,
        consumed_at_ns=clock.decision_emission_ns,
        exact_boundary=True,
        max_age_ns=clock.decision_emission_ns - clock.feature_boundary_ns,
        source="DATABENTO_OPRA_CBBO_1M",
        received_column=received_column,
    )


def select_executable_entry_quote(
    frame: pd.DataFrame,
    *,
    stable_contract_id: str,
    clock: TrainingTwinClockV1,
    max_age_ns: int = DEFAULT_MAX_ENTRY_QUOTE_AGE_NS,
    received_column: str = "received_at_ns",
) -> tuple[pd.Series, SelectedSourceRowV1]:
    """Select a fresh CBBO-1s quote at the frozen order-arrival cutoff."""

    return _select_option_quote(
        frame,
        stable_contract_id=stable_contract_id,
        represented_end_ns=clock.feature_boundary_ns,
        consumed_at_ns=clock.entry_arrival_ns,
        exact_boundary=False,
        max_age_ns=max_age_ns,
        source="DATABENTO_OPRA_CBBO_1S",
        received_column=received_column,
    )


def compile_source_receipt(
    *,
    stable_contract_id: str,
    clock: TrainingTwinClockV1,
    spx_frame: pd.DataFrame,
    cbbo_1m_frame: pd.DataFrame,
    cbbo_1s_frame: pd.DataFrame,
    received_column: str = "received_at_ns",
) -> CompiledEntrySourceReceiptV1:
    """Run all three selectors and bind their exact source/clock receipts."""

    _, spx = select_official_spx(
        spx_frame, clock=clock, received_column=received_column
    )
    _, feature = select_option_feature_quote(
        cbbo_1m_frame,
        stable_contract_id=stable_contract_id,
        clock=clock,
        received_column=received_column,
    )
    _, entry = select_executable_entry_quote(
        cbbo_1s_frame,
        stable_contract_id=stable_contract_id,
        clock=clock,
        received_column=received_column,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "clock": asdict(clock),
        "official_spx": asdict(spx),
        "option_feature_quote": asdict(feature),
        "executable_entry_quote": asdict(entry),
        "stable_contract_id": stable_contract_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CompiledEntrySourceReceiptV1(
        schema_version=SCHEMA_VERSION,
        clock=clock,
        official_spx=spx,
        option_feature_quote=feature,
        executable_entry_quote=entry,
        stable_contract_id=stable_contract_id,
        receipt_sha256=digest,
    )
