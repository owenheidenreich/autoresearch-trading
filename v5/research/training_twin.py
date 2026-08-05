"""Source-neutral clock and parity contracts for future v5 model inputs.

This module is deliberately model-free and network-free.  Historical replay
and live shadow must call the same selectors.  A minute boundary ``t`` means
the closed interval ``[t-60s, t)``; a row is usable only when it was received
by the frozen decision-emission clock.

The implementation was promoted from the reviewed Path-D training-twin law.
It is native v5 code and does not import the frozen v4 tree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd


SECOND_NS = 1_000_000_000
MINUTE_NS = 60 * SECOND_NS
DEFAULT_MAX_ENTRY_QUOTE_AGE_NS = 2 * SECOND_NS
CLOCK_SCHEMA_VERSION = "v5.training-twin-clock.v1"
PARITY_SCHEMA_VERSION = "v5.training-live-parity-receipt.v1"
LATENCY_SCHEMA_VERSION = "v5.source-latency-receipt.v1"
REVIEWED_ANTECEDENT = {
    "path": "v4/research/autoresearch_v2/live_opra_training_twin.py",
    "sha256": "30322c68019b29a90b3e056da86fa7db2b42f496e1f30b5751351f5321e78755",
}


class TrainingTwinError(RuntimeError):
    """A source row or paired replay failed the frozen causal contract."""


@dataclass(frozen=True)
class TrainingTwinClock:
    interval_start_ns: int
    feature_boundary_ns: int
    decision_emission_ns: int
    entry_arrival_ns: int
    label_deadline_ns: int
    hold_minutes: int

    def __post_init__(self) -> None:
        if self.feature_boundary_ns % MINUTE_NS:
            raise TrainingTwinError("feature boundary must be an exact UTC minute")
        if self.interval_start_ns != self.feature_boundary_ns - MINUTE_NS:
            raise TrainingTwinError("feature interval must be exactly one closed minute")
        if not (
            self.feature_boundary_ns
            <= self.decision_emission_ns
            <= self.entry_arrival_ns
            < self.label_deadline_ns
        ):
            raise TrainingTwinError("training-twin clocks are not monotone")
        expected_deadline = self.entry_arrival_ns + self.hold_minutes * MINUTE_NS
        if self.label_deadline_ns != expected_deadline:
            raise TrainingTwinError("label deadline must start from executable arrival")


@dataclass(frozen=True)
class CurrentSessionContract:
    stable_contract_id: str
    current_instrument_id: int
    expiration_date: str
    right: str
    strike: float
    definition_received_at_ns: int


@dataclass(frozen=True)
class SelectedSourceRow:
    source: str
    stable_contract_id: str | None
    represented_interval_start_ns: int
    represented_interval_end_ns: int
    source_timestamp_ns: int
    received_at_ns: int
    age_at_consumption_ns: int
    row_index: int


@dataclass(frozen=True)
class EntrySourceReceipt:
    schema_version: str
    clock: TrainingTwinClock
    official_spx: SelectedSourceRow
    option_feature_quote: SelectedSourceRow
    executable_entry_quote: SelectedSourceRow
    stable_contract_id: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedParityReceipt:
    schema_version: str
    status: str
    key_columns: tuple[str, ...]
    value_columns: tuple[str, ...]
    live_rows: int
    historical_rows: int
    matched_rows: int
    missing_live_keys: int
    missing_historical_keys: int
    dtype_mismatches: int
    mismatched_cells: int
    tolerances: dict[str, float]
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_passed(self) -> "PairedParityReceipt":
        if self.status != "PASS":
            raise TrainingTwinError(
                "paired replay failed: "
                f"missing_live={self.missing_live_keys}, "
                f"missing_historical={self.missing_historical_keys}, "
                f"dtype_mismatches={self.dtype_mismatches}, "
                f"mismatched_cells={self.mismatched_cells}"
            )
        return self


@dataclass
class StreamReadiness:
    """Fail-closed warm-up state shared by recording and live shadow."""

    definitions_loaded: bool = False
    completed_intervals: int = 0
    reconnect_count: int = 0

    def load_current_session_definitions(self) -> None:
        self.definitions_loaded = True

    def reconnect(self) -> None:
        self.reconnect_count += 1
        self.completed_intervals = 0

    def observe_completed_interval(self) -> bool:
        """Return whether the interval may emit; the first is always warm-up."""

        if not self.definitions_loaded:
            raise TrainingTwinError("current-session definitions are not loaded")
        self.completed_intervals += 1
        return self.completed_intervals > 1


_OSI_RE = re.compile(r"^SPXW  (?P<expiry>[0-9]{6})(?P<right>[CP])(?P<strike>[0-9]{8})$")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def make_clock(
    *,
    feature_boundary_ns: int,
    emission_lag_ms: int,
    order_latency_ms: int,
    hold_minutes: int = 25,
) -> TrainingTwinClock:
    """Compile the clock law; all parameters must be frozen before fitting."""

    if not 0 <= emission_lag_ms <= 10_000:
        raise TrainingTwinError("emission lag must be between 0 and 10 seconds")
    if not 0 <= order_latency_ms <= 10_000:
        raise TrainingTwinError("order latency must be between 0 and 10 seconds")
    if not 1 <= hold_minutes <= 390:
        raise TrainingTwinError("hold minutes out of bounds")
    emission_ns = int(feature_boundary_ns) + int(emission_lag_ms) * 1_000_000
    arrival_ns = emission_ns + int(order_latency_ms) * 1_000_000
    return TrainingTwinClock(
        interval_start_ns=int(feature_boundary_ns) - MINUTE_NS,
        feature_boundary_ns=int(feature_boundary_ns),
        decision_emission_ns=emission_ns,
        entry_arrival_ns=arrival_ns,
        label_deadline_ns=arrival_ns + int(hold_minutes) * MINUTE_NS,
        hold_minutes=int(hold_minutes),
    )


def _timestamp_ns(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise TrainingTwinError(f"missing timestamp column: {column}")
    parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if parsed.isna().any():
        raise TrainingTwinError(f"invalid timestamp in {column}")
    # Pandas may retain microsecond resolution for parsed ISO strings.  The
    # contract is explicitly nanoseconds, so use Timestamp.value row by row.
    return parsed.map(lambda value: value.value).astype("int64")


def _received_ns(
    frame: pd.DataFrame, *, fallback: pd.Series, received_column: str
) -> pd.Series:
    if received_column not in frame.columns:
        return fallback.copy()
    values = pd.to_numeric(frame[received_column], errors="coerce")
    if values.isna().any():
        raise TrainingTwinError(f"invalid receipt timestamp in {received_column}")
    return values.astype("int64")


def current_session_definition_universe(
    frame: pd.DataFrame, *, session_date: str, as_of_ns: int
) -> dict[str, CurrentSessionContract]:
    """Apply current-session add/modify/delete updates through one cutoff."""

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
        raise TrainingTwinError(f"definition frame missing columns: {sorted(missing)}")
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
    latest = latest[
        ~latest["security_update_action"].astype(str).str.upper().isin({"D", "DELETE"})
    ]
    universe: dict[str, CurrentSessionContract] = {}
    for _, row in latest.iterrows():
        symbol = str(row["raw_symbol"])
        match = _OSI_RE.fullmatch(symbol)
        if match is None:
            raise TrainingTwinError(f"invalid SPXW OSI raw symbol: {symbol}")
        expected_expiry = pd.Timestamp(session_date).strftime("%y%m%d")
        right = match.group("right")
        strike = int(match.group("strike")) / 1000.0
        if (
            match.group("expiry") != expected_expiry
            or str(row["instrument_class"]) != right
            or not math.isclose(
                float(row["strike_price"]), strike, abs_tol=0.0, rel_tol=0.0
            )
        ):
            raise TrainingTwinError(f"definition/OSI geometry mismatch: {symbol}")
        instrument_id = int(row["instrument_id"])
        if instrument_id <= 0:
            raise TrainingTwinError(f"invalid instrument ID: {symbol}")
        universe[symbol] = CurrentSessionContract(
            stable_contract_id=symbol,
            current_instrument_id=instrument_id,
            expiration_date=session_date,
            right=right,
            strike=strike,
            definition_received_at_ns=int(row["_ts_recv_ns"]),
        )
    if not universe:
        raise TrainingTwinError("current-session definition universe is empty")
    return dict(sorted(universe.items()))


def _valid_bbo(row: pd.Series) -> None:
    bid = float(row["bid_px_00"])
    ask = float(row["ask_px_00"])
    if not math.isfinite(bid) or not math.isfinite(ask) or bid < 0.0 or ask <= bid:
        raise TrainingTwinError("selected BBO is missing, locked, or crossed")


def select_official_spx(
    frame: pd.DataFrame,
    *,
    clock: TrainingTwinClock,
    received_column: str = "received_at_ns",
) -> tuple[pd.Series, SelectedSourceRow]:
    """Select the official bar for ``[t-60s, t)`` and never the next bar."""

    frame = frame.reset_index(drop=True)
    event_ns = _timestamp_ns(frame, "event_time")
    represented_end = event_ns + MINUTE_NS
    received = _received_ns(
        frame, fallback=represented_end, received_column=received_column
    )
    eligible = frame.index[
        (represented_end == clock.feature_boundary_ns)
        & (received <= clock.decision_emission_ns)
    ].tolist()
    if len(eligible) != 1:
        raise TrainingTwinError(
            f"expected one exact completed SPX bar, found {len(eligible)}"
        )
    label = eligible[0]
    position = int(label)
    row = frame.loc[label]
    close = float(row["close"])
    if not math.isfinite(close) or close <= 0.0:
        raise TrainingTwinError("selected SPX close is invalid")
    receipt = SelectedSourceRow(
        source="THETADATA_OFFICIAL_SPX_1M",
        stable_contract_id=None,
        represented_interval_start_ns=int(event_ns.loc[label]),
        represented_interval_end_ns=int(represented_end.loc[label]),
        source_timestamp_ns=int(event_ns.loc[label]),
        received_at_ns=int(received.loc[label]),
        age_at_consumption_ns=clock.decision_emission_ns - int(received.loc[label]),
        row_index=position,
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
) -> tuple[pd.Series, SelectedSourceRow]:
    frame = frame.reset_index(drop=True)
    required = {"symbol", "ts_recv", "bid_px_00", "ask_px_00"}
    missing = required - set(frame.columns)
    if missing:
        raise TrainingTwinError(f"option frame missing columns: {sorted(missing)}")
    ts_recv = _timestamp_ns(frame, "ts_recv")
    received = _received_ns(frame, fallback=ts_recv, received_column=received_column)
    same = frame["symbol"].astype(str).eq(stable_contract_id)
    if exact_boundary:
        time_ok = ts_recv.eq(represented_end_ns)
    else:
        time_ok = ts_recv.ge(represented_end_ns) & ts_recv.le(consumed_at_ns)
    eligible = frame.index[same & time_ok & received.le(consumed_at_ns)].tolist()
    if exact_boundary and len(eligible) != 1:
        raise TrainingTwinError(
            f"expected one exact CBBO-1m row for {stable_contract_id}, found {len(eligible)}"
        )
    if not eligible:
        raise TrainingTwinError(f"no causal option quote for {stable_contract_id}")
    label = max(eligible, key=lambda item: int(ts_recv.loc[item]))
    position = int(label)
    age = consumed_at_ns - int(ts_recv.loc[label])
    if age < 0 or age > max_age_ns:
        raise TrainingTwinError(f"option quote age out of bounds: {age}")
    row = frame.loc[label]
    _valid_bbo(row)
    receipt = SelectedSourceRow(
        source=source,
        stable_contract_id=stable_contract_id,
        represented_interval_start_ns=int(ts_recv.loc[label])
        - (MINUTE_NS if exact_boundary else SECOND_NS),
        represented_interval_end_ns=int(ts_recv.loc[label]),
        source_timestamp_ns=int(ts_recv.loc[label]),
        received_at_ns=int(received.loc[label]),
        age_at_consumption_ns=age,
        row_index=position,
    )
    return row, receipt


def select_option_feature_quote(
    frame: pd.DataFrame,
    *,
    stable_contract_id: str,
    clock: TrainingTwinClock,
    received_column: str = "received_at_ns",
) -> tuple[pd.Series, SelectedSourceRow]:
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
    clock: TrainingTwinClock,
    max_age_ns: int = DEFAULT_MAX_ENTRY_QUOTE_AGE_NS,
    received_column: str = "received_at_ns",
) -> tuple[pd.Series, SelectedSourceRow]:
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


def compile_entry_source_receipt(
    *,
    stable_contract_id: str,
    clock: TrainingTwinClock,
    spx_frame: pd.DataFrame,
    cbbo_1m_frame: pd.DataFrame,
    cbbo_1s_frame: pd.DataFrame,
    received_column: str = "received_at_ns",
) -> EntrySourceReceipt:
    """Bind exact source rows and clocks without loading or fitting a model."""

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
    unsigned = {
        "schema_version": CLOCK_SCHEMA_VERSION,
        "clock": asdict(clock),
        "official_spx": asdict(spx),
        "option_feature_quote": asdict(feature),
        "executable_entry_quote": asdict(entry),
        "stable_contract_id": stable_contract_id,
    }
    return EntrySourceReceipt(
        schema_version=CLOCK_SCHEMA_VERSION,
        clock=clock,
        official_spx=spx,
        option_feature_quote=feature,
        executable_entry_quote=entry,
        stable_contract_id=stable_contract_id,
        receipt_sha256=hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    )


def compare_paired_frames(
    historical: pd.DataFrame,
    live: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    tolerances: Mapping[str, float] | None = None,
    allow_historical_superset: bool = True,
) -> PairedParityReceipt:
    """Compare the same captured events after historical and live decoding."""

    keys = tuple(map(str, key_columns))
    values = tuple(map(str, value_columns))
    if not keys or not values or set(keys) & set(values):
        raise TrainingTwinError("parity keys and values must be non-empty and disjoint")
    required = set(keys) | set(values)
    for label, frame in (("historical", historical), ("live", live)):
        missing = required - set(frame.columns)
        if missing:
            raise TrainingTwinError(f"{label} frame missing columns: {sorted(missing)}")
        if frame.duplicated(list(keys)).any():
            raise TrainingTwinError(f"{label} frame has duplicate parity keys")
    tolerance_map = {name: float((tolerances or {}).get(name, 0.0)) for name in values}
    if any(value < 0.0 or not math.isfinite(value) for value in tolerance_map.values()):
        raise TrainingTwinError("parity tolerances must be finite and non-negative")

    historical_indexed = historical.set_index(list(keys)).sort_index()
    live_indexed = live.set_index(list(keys)).sort_index()
    historical_keys = set(historical_indexed.index.tolist())
    live_keys = set(live_indexed.index.tolist())
    missing_historical = live_keys - historical_keys
    missing_live = historical_keys - live_keys
    shared = historical_indexed.index.intersection(live_indexed.index)
    dtype_mismatches = sum(
        str(historical[column].dtype) != str(live[column].dtype)
        for column in values
    )
    mismatches = 0
    for column in values:
        left = historical_indexed.loc[shared, column]
        right = live_indexed.loc[shared, column]
        tolerance = tolerance_map[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_numeric = pd.to_numeric(left, errors="coerce")
            right_numeric = pd.to_numeric(right, errors="coerce")
            equal = (left_numeric - right_numeric).abs().le(tolerance) | (
                left_numeric.isna() & right_numeric.isna()
            )
        else:
            equal = left.eq(right) | (left.isna() & right.isna())
        mismatches += int((~equal).sum())
    passed = (
        len(live) > 0
        and len(shared) > 0
        and not missing_historical
        and not dtype_mismatches
        and not mismatches
    )
    if not allow_historical_superset:
        passed = passed and not missing_live
    unsigned = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "key_columns": keys,
        "value_columns": values,
        "live_rows": len(live),
        "historical_rows": len(historical),
        "matched_rows": len(shared),
        "missing_live_keys": len(missing_live),
        "missing_historical_keys": len(missing_historical),
        "dtype_mismatches": dtype_mismatches,
        "mismatched_cells": mismatches,
        "tolerances": tolerance_map,
    }
    return PairedParityReceipt(
        **unsigned,
        receipt_sha256=hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    )


@dataclass(frozen=True)
class LatencyReceipt:
    """A signed, dated measurement of how late one source stream actually is.

    Historical rows carry no arrival time.  Injecting one is only honest if the
    injected number was measured on the same stream the features come from, is
    recent enough to still describe that stream, and is recorded with the
    sessions behind it so a thin sample cannot masquerade as a bound.
    """

    schema_version: str
    source_family: str
    p50_ms: float
    p99_ms: float
    max_ms: float
    session_count: int
    measured_on: str
    valid_until: str
    evidence_path: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def unsigned(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("receipt_sha256")
        return payload

    def assert_usable(self, *, source_family: str, as_of: str) -> "LatencyReceipt":
        """Refuse a receipt that does not license this stream on this date."""

        if self.schema_version != LATENCY_SCHEMA_VERSION:
            raise TrainingTwinError(f"unknown latency receipt schema: {self.schema_version}")
        if _hash_payload(self.to_dict()) != self.receipt_sha256:
            raise TrainingTwinError("latency receipt self-hash mismatch; the bytes were edited")
        if self.source_family != source_family:
            raise TrainingTwinError(
                f"latency receipt covers {self.source_family}, not {source_family}; "
                "a measurement on one feed does not license another"
            )
        if self.session_count < 1:
            raise TrainingTwinError("latency receipt records no sessions")
        if not (0.0 <= self.p50_ms <= self.p99_ms <= self.max_ms):
            raise TrainingTwinError("latency percentiles are not ordered")
        if as_of > self.valid_until:
            raise TrainingTwinError(
                f"latency receipt expired on {self.valid_until}; re-measure before fitting"
            )
        return self


def make_latency_receipt(
    *,
    source_family: str,
    p50_ms: float,
    p99_ms: float,
    max_ms: float,
    session_count: int,
    measured_on: str,
    valid_until: str,
    evidence_path: str,
) -> LatencyReceipt:
    """Build a content-addressed latency receipt from a completed measurement."""

    unsigned = {
        "schema_version": LATENCY_SCHEMA_VERSION,
        "source_family": source_family,
        "p50_ms": float(p50_ms),
        "p99_ms": float(p99_ms),
        "max_ms": float(max_ms),
        "session_count": int(session_count),
        "measured_on": measured_on,
        "valid_until": valid_until,
        "evidence_path": evidence_path,
    }
    return LatencyReceipt(
        **unsigned, receipt_sha256=hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    )


def assert_no_zero_lag(
    frame: pd.DataFrame,
    *,
    interval_end_column: str = "event_time",
    received_column: str = "received_at_ns",
) -> None:
    """Reject a frame that still claims data arrived the instant it existed.

    The owned corpus stores ``receive_time == event_time`` on every one of its
    47,707,186 rows, which is the interval close rather than an arrival.  Fitting
    on that assumes timing the live system cannot deliver, so it is refused
    rather than silently accepted.
    """

    if received_column not in frame.columns:
        raise TrainingTwinError(f"frame has no arrival column: {received_column}")
    interval_end = _timestamp_ns(frame, interval_end_column) + MINUTE_NS
    received = pd.to_numeric(frame[received_column], errors="coerce")
    if received.isna().any():
        raise TrainingTwinError(f"invalid arrival timestamp in {received_column}")
    lag = received.astype("int64") - interval_end
    if (lag < 0).any():
        raise TrainingTwinError("arrival precedes the interval it represents")
    if bool((lag == 0).all()):
        raise TrainingTwinError(
            "every row claims zero arrival lag; call simulate_historical_arrival "
            "with a latency receipt before using historical rows for a fit"
        )


def simulate_historical_arrival(
    frame: pd.DataFrame,
    *,
    receipt: LatencyReceipt,
    source_family: str,
    as_of: str,
    interval_end_column: str = "event_time",
    received_column: str = "received_at_ns",
    percentile: str = "p99_ms",
) -> pd.DataFrame:
    """Stamp a causal arrival time onto historical rows and return a new frame.

    Historical replay and live shadow already share the same selectors.  Once
    arrival is injected here, they also share the same timing distribution, so a
    model cannot distinguish the two by how fresh its inputs are.  The injected
    lag defaults to the measured p99 because a selector must survive the slow
    case, not the typical one.
    """

    receipt.assert_usable(source_family=source_family, as_of=as_of)
    if percentile not in {"p50_ms", "p99_ms", "max_ms"}:
        raise TrainingTwinError(f"unsupported latency percentile: {percentile}")
    lag_ms = float(getattr(receipt, percentile))
    if lag_ms <= 0.0:
        raise TrainingTwinError("a simulated arrival lag must be positive")
    result = frame.reset_index(drop=True).copy()
    interval_end = _timestamp_ns(result, interval_end_column) + MINUTE_NS
    result[received_column] = interval_end + int(round(lag_ms * 1_000_000))
    result["arrival_source"] = "SIMULATED_FROM_LATENCY_RECEIPT"
    result["arrival_receipt_sha256"] = receipt.receipt_sha256
    return result


def write_parity_receipt(
    receipt: PairedParityReceipt, path: Path, *, allow_overwrite: bool = False
) -> None:
    """Write a content-addressed local receipt; a failed comparison is refused."""

    receipt.assert_passed()
    payload = receipt.to_dict()
    if _hash_payload(payload) != receipt.receipt_sha256:
        raise TrainingTwinError("parity receipt self-hash mismatch")
    if path.exists() and not allow_overwrite:
        raise TrainingTwinError(f"refusing to overwrite parity receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
