"""Causal signed-17 entry feature machinery for the quarantined Path-D study.

This module deliberately contains no file access, fitting, broker integration, or
result-dependent behavior.  Both historical rows and future source-neutral inputs
are reduced to the same immutable snapshot before the single signed-17 kernel runs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES, feature_matrix
from v4.research import pathd_entry_exit as prereg


_OFFICIAL_SPX_MARKET_FEATURE_NAMES = (
    "spx_close",
    "vix_close",
    "spx_vwap",
    "omar",
    "session_range",
    "momentum_5m",
    "momentum_15m",
)
_MINUTE_NS = 60_000_000_000
_OFFICIAL_SPX_MAX_AGE_NS = 90_000_000_000


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-safe representation, including NaN masks."""

    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": _jsonable(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "+inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _readonly_array(value: Any, *, dtype: Any, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    expected = np.dtype(dtype)
    if array.dtype == expected and array.flags.c_contiguous and not array.flags.writeable:
        return array
    result = np.array(array, dtype=expected, copy=True, order="C")
    result.setflags(write=False)
    return result


def _object_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    array = np.array(array, dtype=object, copy=True, order="C")
    if any(not isinstance(item, str) or not item for item in array.reshape(-1)):
        raise ValueError(f"{name} must contain nonempty strings")
    array.setflags(write=False)
    return array


def _utc_ns(value: Any, *, name: str) -> int:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return int(timestamp.tz_convert("UTC").value)


def official_spx_market_window_from_rows(
    rows: Any,
    /,
    *,
    session: str,
    decision_time_ns: int,
    history_minutes: int = 30,
) -> tuple[np.ndarray, tuple[str, ...], int]:
    """Build causal signed-17 market context from official ThetaData bars only.

    ThetaData ``event_time`` is the bar-open timestamp.  A row therefore becomes
    available exactly one minute later; neither a same-timestamp close nor any
    legacy processed VIX value can enter this adapter.
    """

    if type(session) is not str or pd.Timestamp(session).strftime("%Y-%m-%d") != session:
        raise ValueError("official SPX session must be YYYY-MM-DD")
    if type(decision_time_ns) is not int:
        raise TypeError("official SPX decision clock must be an exact integer")
    if type(history_minutes) is not int or history_minutes <= 0:
        raise ValueError("official SPX history_minutes must be a positive integer")
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict("records")
    elif type(rows) in (tuple, list):
        records = list(rows)
    else:
        raise TypeError("official SPX rows must be a DataFrame or exact sequence")
    required = {
        "event_time",
        "symbol",
        "close",
        "volume",
        "context_source",
        "is_derived",
        "is_proxy",
        "is_official_index_data",
    }
    normalized: list[tuple[int, int, float, int]] = []
    for record in records:
        if type(record) is not dict or not required.issubset(record):
            raise ValueError("official SPX row schema drift")
        if (
            record["symbol"] != "SPX"
            or record["context_source"] != "thetadata_index_history_ohlc"
            or record["is_derived"] is not False
            or record["is_proxy"] is not False
            or record["is_official_index_data"] is not True
        ):
            raise ValueError("official SPX row invariants drift")
        event_time_ns = _utc_ns(record["event_time"], name="official SPX event_time")
        available_at_ns = event_time_ns + _MINUTE_NS
        local_session = (
            pd.Timestamp(event_time_ns, unit="ns", tz="UTC")
            .tz_convert("America/New_York")
            .strftime("%Y-%m-%d")
        )
        if local_session != session:
            raise ValueError("official SPX row crossed session")
        try:
            close = float(record["close"])
        except (TypeError, ValueError) as exc:
            raise ValueError("official SPX close is malformed") from exc
        volume = record["volume"]
        if (
            not math.isfinite(close)
            or close <= 0.0
            or isinstance(volume, (bool, np.bool_))
            or not isinstance(volume, (int, np.integer))
            or int(volume) < 0
        ):
            raise ValueError("official SPX close/volume is invalid")
        normalized.append((available_at_ns, event_time_ns, close, int(volume)))
    normalized.sort(key=lambda item: (item[0], item[1]))
    if not normalized or len({item[0] for item in normalized}) != len(normalized):
        raise ValueError("official SPX availability clocks are empty or duplicated")

    available = np.asarray([item[0] for item in normalized], dtype=np.int64)
    closes = np.asarray([item[2] for item in normalized], dtype=np.float64)
    volumes = np.asarray([item[3] for item in normalized], dtype=np.int64)
    output = np.full(
        (history_minutes, len(_OFFICIAL_SPX_MARKET_FEATURE_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    current_available_at_ns: int | None = None
    for output_index in range(history_minutes):
        boundary = decision_time_ns - (history_minutes - 1 - output_index) * _MINUTE_NS
        selected_index = int(np.searchsorted(available, boundary, side="right") - 1)
        if selected_index < 0:
            continue
        selected_available = int(available[selected_index])
        age = boundary - selected_available
        if age < 0 or age > _OFFICIAL_SPX_MAX_AGE_NS:
            continue
        prefix_close = closes[: selected_index + 1]
        prefix_volume = volumes[: selected_index + 1]
        total_volume = int(prefix_volume.sum(dtype=np.int64))
        if total_volume > 0:
            vwap = float(
                np.dot(prefix_close, prefix_volume.astype(np.float64))
                / float(total_volume)
            )
        else:
            vwap = float(np.mean(prefix_close))
        session_open = float(prefix_close[0])
        session_range = float(np.max(prefix_close) - np.min(prefix_close))
        close = float(prefix_close[-1])
        omar = 0.0 if session_range == 0.0 else (close - session_open) / session_range
        momentum_5m = (
            close - float(closes[selected_index - 5])
            if selected_index >= 5
            else float("nan")
        )
        momentum_15m = (
            close - float(closes[selected_index - 15])
            if selected_index >= 15
            else float("nan")
        )
        output[output_index] = (
            close,
            float("nan"),
            vwap,
            omar,
            session_range,
            momentum_5m,
            momentum_15m,
        )
        if output_index == history_minutes - 1:
            current_available_at_ns = selected_available
    if current_available_at_ns is None:
        raise ValueError("official SPX has no fresh row at the decision boundary")
    output.setflags(write=False)
    return output, _OFFICIAL_SPX_MARKET_FEATURE_NAMES, current_available_at_ns


@dataclass(frozen=True)
class EntrySnapshotV1:
    SCHEMA_VERSION = "pathd.entry_snapshot.v1"

    schema_version: str
    session: str
    decision_time_ns: int
    decision_watermark_ns: int
    atm_strike: float
    strike_offsets: np.ndarray
    rights: tuple[str, ...]
    option_ladder: np.ndarray
    option_feature_names: tuple[str, ...]
    market_window: np.ndarray
    market_feature_names: tuple[str, ...]
    contract_ids: np.ndarray
    contract_quote_watermark_ns: np.ndarray
    declared_alpha_sources: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("entry snapshot schema drift")
        if not isinstance(self.session, str) or not self.session:
            raise TypeError("session must be a nonempty string")
        if type(self.decision_time_ns) is not int or type(self.decision_watermark_ns) is not int:
            raise TypeError("decision clocks must be exact integers")
        if self.decision_watermark_ns > self.decision_time_ns:
            raise ValueError("decision watermark is post-decision")
        offsets = _readonly_array(
            self.strike_offsets, dtype=np.float64, ndim=1, name="strike_offsets"
        )
        rights = tuple(self.rights)
        if rights != ("C", "P"):
            raise ValueError("Path-D entry ladder freezes rights=(C,P)")
        ladder = _readonly_array(
            self.option_ladder, dtype=np.float64, ndim=3, name="option_ladder"
        )
        market = _readonly_array(
            self.market_window, dtype=np.float64, ndim=2, name="market_window"
        )
        identities = _object_array(self.contract_ids, ndim=2, name="contract_ids")
        quote_clocks = _readonly_array(
            self.contract_quote_watermark_ns,
            dtype=np.int64,
            ndim=2,
            name="contract_quote_watermark_ns",
        )
        option_names = tuple(self.option_feature_names)
        market_names = tuple(self.market_feature_names)
        if len(set(option_names)) != len(option_names) or not option_names:
            raise ValueError("option feature names must be unique and nonempty")
        if len(set(market_names)) != len(market_names) or not market_names:
            raise ValueError("market feature names must be unique and nonempty")
        expected_prefix = (len(offsets), len(rights))
        if ladder.shape != (*expected_prefix, len(option_names)):
            raise ValueError("option ladder shape does not match names/geometry")
        if identities.shape != expected_prefix or quote_clocks.shape != expected_prefix:
            raise ValueError("contract identities or quote clocks do not match ladder")
        if market.shape[1] != len(market_names) or market.shape[0] == 0:
            raise ValueError("market window shape does not match names")
        if np.any(quote_clocks > self.decision_time_ns):
            raise ValueError("post-decision option receipt in entry snapshot")
        sources = tuple(self.declared_alpha_sources)
        for record in sources:
            validate_alpha_source_record(record)
        expected_sources = tuple(prereg.feature_lineage()["features"])
        if _canonical_sha256(sources) != _canonical_sha256(expected_sources):
            raise ValueError("declared alpha source registry is not exact")
        object.__setattr__(self, "strike_offsets", offsets)
        object.__setattr__(self, "rights", rights)
        object.__setattr__(self, "option_ladder", ladder)
        object.__setattr__(self, "option_feature_names", option_names)
        object.__setattr__(self, "market_window", market)
        object.__setattr__(self, "market_feature_names", market_names)
        object.__setattr__(self, "contract_ids", identities)
        object.__setattr__(self, "contract_quote_watermark_ns", quote_clocks)
        object.__setattr__(self, "declared_alpha_sources", sources)

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self)


@dataclass(frozen=True)
class Signed17FrameV1:
    SCHEMA_VERSION = "pathd.signed17_frame.v1"

    schema_version: str
    feature_names: tuple[str, ...]
    values: np.ndarray
    finite: np.ndarray
    contract_ids: np.ndarray
    decision_time_ns: int
    decision_watermark_ns: int

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("signed-17 frame schema drift")
        names = tuple(self.feature_names)
        if names != tuple(FEATURE_NAMES):
            raise ValueError("signed-17 feature names/order drift")
        values = _readonly_array(self.values, dtype=np.float64, ndim=3, name="values")
        finite = _readonly_array(self.finite, dtype=np.bool_, ndim=3, name="finite")
        identities = _object_array(self.contract_ids, ndim=2, name="contract_ids")
        if values.shape != (*identities.shape, len(names)) or finite.shape != values.shape:
            raise ValueError("signed-17 frame shape mismatch")
        if not np.array_equal(finite, np.isfinite(values)):
            raise ValueError("signed-17 finite mask mismatch")
        if type(self.decision_time_ns) is not int or type(self.decision_watermark_ns) is not int:
            raise TypeError("signed-17 clocks must be exact integers")
        if self.decision_watermark_ns > self.decision_time_ns:
            raise ValueError("signed-17 frame is post-decision")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "finite", finite)
        object.__setattr__(self, "contract_ids", identities)

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self)


@dataclass(frozen=True)
class Signed17HistoryV1:
    SCHEMA_VERSION = "pathd.signed17_history.v1"

    schema_version: str
    values: np.ndarray
    finite: np.ndarray
    contract_present: np.ndarray
    minute_available: np.ndarray
    current_contract_ids: np.ndarray
    decision_time_ns: int

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("signed-17 history schema drift")
        values = _readonly_array(self.values, dtype=np.float64, ndim=3, name="values")
        finite = _readonly_array(self.finite, dtype=np.bool_, ndim=3, name="finite")
        present = _readonly_array(
            self.contract_present, dtype=np.bool_, ndim=2, name="contract_present"
        )
        available = _readonly_array(
            self.minute_available, dtype=np.bool_, ndim=1, name="minute_available"
        )
        identities = _object_array(
            self.current_contract_ids, ndim=1, name="current_contract_ids"
        )
        if values.shape[1:] != (len(identities), len(FEATURE_NAMES)):
            raise ValueError("signed-17 history geometry mismatch")
        if finite.shape != values.shape or present.shape != values.shape[:2]:
            raise ValueError("signed-17 history mask mismatch")
        if available.shape != (values.shape[0],):
            raise ValueError("signed-17 minute mask mismatch")
        expected_finite = np.isfinite(values) & present[..., None] & available[:, None, None]
        if not np.array_equal(finite, expected_finite):
            raise ValueError("signed-17 history finite mask is not reconstructible")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "finite", finite)
        object.__setattr__(self, "contract_present", present)
        object.__setattr__(self, "minute_available", available)
        object.__setattr__(self, "current_contract_ids", identities)

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self)


def validate_alpha_source_record(record: Any, /) -> None:
    if type(record) is not dict:
        raise TypeError("alpha lineage record must be an exact mapping")
    rows = prereg.feature_lineage()["features"]
    name = record.get("name")
    matches = [row for row in rows if row["name"] == name]
    if len(matches) != 1 or _canonical_sha256(record) != _canonical_sha256(matches[0]):
        raise ValueError("unregistered or altered alpha lineage record")
    if record.get("raw_vendor_greek") is not False:
        raise ValueError("raw vendor Greeks are forbidden entry alpha")
    if record.get("ibkr_market_alpha") is not False:
        raise ValueError("IBKR market values are forbidden entry alpha")
    if not record.get("future_live_twin_adapter"):
        raise ValueError("source-neutral future twin is mandatory")
    if "available_after_decision" in record:
        raise ValueError("post-decision alpha metadata is forbidden")


def source_neutral_snapshot_from_causal_inputs(
    *,
    session: str,
    decision_time_ns: int,
    decision_watermark_ns: int,
    atm_strike: float,
    strike_offsets: Any,
    rights: Any,
    option_ladder: Any,
    option_feature_names: Any,
    market_window: Any,
    market_feature_names: Any,
    contract_ids: Any,
    contract_quote_watermark_ns: Any,
    declared_alpha_sources: Any,
) -> EntrySnapshotV1:
    return EntrySnapshotV1(
        schema_version=EntrySnapshotV1.SCHEMA_VERSION,
        session=session,
        decision_time_ns=decision_time_ns,
        decision_watermark_ns=decision_watermark_ns,
        atm_strike=float(atm_strike),
        strike_offsets=strike_offsets,
        rights=tuple(rights),
        option_ladder=option_ladder,
        option_feature_names=tuple(option_feature_names),
        market_window=market_window,
        market_feature_names=tuple(market_feature_names),
        contract_ids=contract_ids,
        contract_quote_watermark_ns=contract_quote_watermark_ns,
        declared_alpha_sources=tuple(declared_alpha_sources),
    )


def historical_snapshot_from_processed_row(row: Any, /) -> EntrySnapshotV1:
    if type(row) is not dict:
        raise TypeError("historical processed row must be an exact mapping")
    required = (
        "session",
        "decision_time_ns",
        "decision_watermark_ns",
        "atm_strike",
        "strike_offsets",
        "rights",
        "option_ladder",
        "market_window",
        "market_feature_names",
        "contract_ids",
        "contract_quote_watermark_ns",
        "declared_alpha_sources",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"historical processed row is missing {missing}")
    option_names = row.get("option_feature_names", row.get("feature_names"))
    if option_names is None:
        raise ValueError("historical processed row has no option feature names")
    return source_neutral_snapshot_from_causal_inputs(
        session=row["session"],
        decision_time_ns=int(row["decision_time_ns"]),
        decision_watermark_ns=int(row["decision_watermark_ns"]),
        atm_strike=float(row["atm_strike"]),
        strike_offsets=row["strike_offsets"],
        rights=row["rights"],
        option_ladder=row["option_ladder"],
        option_feature_names=option_names,
        market_window=row["market_window"],
        market_feature_names=row["market_feature_names"],
        contract_ids=row["contract_ids"],
        contract_quote_watermark_ns=row["contract_quote_watermark_ns"],
        declared_alpha_sources=row["declared_alpha_sources"],
    )


def signed17_from_snapshot(snapshot: Any, /) -> Signed17FrameV1:
    if type(snapshot) is not EntrySnapshotV1:
        raise TypeError("signed-17 kernel requires EntrySnapshotV1")
    row = {
        "decision_time": pd.Timestamp(snapshot.decision_time_ns, unit="ns", tz="UTC"),
        "atm_strike": snapshot.atm_strike,
        "strike_offsets": snapshot.strike_offsets,
        "rights": snapshot.rights,
        "option_ladder": snapshot.option_ladder,
        "feature_names": snapshot.option_feature_names,
        "market_window": snapshot.market_window,
        "market_feature_names": snapshot.market_feature_names,
    }
    values = np.asarray(feature_matrix(row), dtype=np.float64)
    return Signed17FrameV1(
        schema_version=Signed17FrameV1.SCHEMA_VERSION,
        feature_names=tuple(FEATURE_NAMES),
        values=values,
        finite=np.isfinite(values),
        contract_ids=snapshot.contract_ids,
        decision_time_ns=snapshot.decision_time_ns,
        decision_watermark_ns=snapshot.decision_watermark_ns,
    )


def build_identity_joined_history(
    frames: Any,
    /,
    *,
    current_session: str,
    current_decision_time_ns: int,
    current_contract_ids: Any,
    history_minutes: int = 90,
) -> Signed17HistoryV1:
    if type(history_minutes) is not int or history_minutes <= 0:
        raise ValueError("history_minutes must be a positive exact integer")
    identities_2d = _object_array(
        current_contract_ids, ndim=2, name="current_contract_ids"
    )
    identities = np.array(identities_2d.reshape(-1), dtype=object, copy=True)
    if len(set(identities.tolist())) != len(identities):
        raise ValueError("current action identities must be unique")
    by_time: dict[int, Signed17FrameV1] = {}
    for row in frames:
        if type(row) not in (tuple, list) or len(row) != 2:
            raise TypeError("history rows must be (session, Signed17FrameV1)")
        session, frame = row
        if type(frame) is not Signed17FrameV1:
            raise TypeError("history frame has wrong type")
        if session != current_session:
            continue
        if frame.decision_time_ns > current_decision_time_ns:
            raise ValueError("post-decision history frame")
        if frame.decision_time_ns in by_time:
            raise ValueError("duplicate history minute")
        by_time[frame.decision_time_ns] = frame
    minute_ns = 60_000_000_000
    times = [
        current_decision_time_ns - (history_minutes - 1 - index) * minute_ns
        for index in range(history_minutes)
    ]
    values = np.full(
        (history_minutes, len(identities), len(FEATURE_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    present = np.zeros((history_minutes, len(identities)), dtype=np.bool_)
    available = np.zeros(history_minutes, dtype=np.bool_)
    wanted = {identity: index for index, identity in enumerate(identities.tolist())}
    for minute_index, timestamp in enumerate(times):
        frame = by_time.get(timestamp)
        if frame is None:
            continue
        available[minute_index] = True
        frame_ids = frame.contract_ids.reshape(-1).tolist()
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("historical action identities must be unique")
        frame_values = frame.values.reshape(-1, len(FEATURE_NAMES))
        for source_index, identity in enumerate(frame_ids):
            target_index = wanted.get(identity)
            if target_index is None:
                continue
            present[minute_index, target_index] = True
            values[minute_index, target_index] = frame_values[source_index]
    finite = np.isfinite(values) & present[..., None] & available[:, None, None]
    return Signed17HistoryV1(
        schema_version=Signed17HistoryV1.SCHEMA_VERSION,
        values=values,
        finite=finite,
        contract_present=present,
        minute_available=available,
        current_contract_ids=identities,
        decision_time_ns=current_decision_time_ns,
    )


def hgb_signed17_summaries(
    history: Any,
    /,
    *,
    current_offsets: Any,
    current_rights: Any,
) -> np.ndarray:
    if type(history) is not Signed17HistoryV1:
        raise TypeError("HGB summaries require Signed17HistoryV1")
    offsets = np.asarray(current_offsets, dtype=np.float64)
    rights = np.asarray(current_rights, dtype=object)
    action_count = history.values.shape[1]
    if offsets.shape != (action_count,) or rights.shape != (action_count,):
        raise ValueError("current geometry does not match history actions")
    if any(value not in ("C", "P") for value in rights.tolist()):
        raise ValueError("unrecognized option right")
    windows = (5, 15, 30, 60, 90)
    output = np.full((action_count, 444), np.nan, dtype=np.float64)
    for action_index in range(action_count):
        cursor = 0
        for feature_index in range(len(FEATURE_NAMES)):
            current_value = history.values[-1, action_index, feature_index]
            if history.finite[-1, action_index, feature_index]:
                output[action_index, cursor] = current_value
            cursor += 1
            for window in windows:
                take = min(window, history.values.shape[0])
                values = history.values[-take:, action_index, feature_index]
                valid = history.finite[-take:, action_index, feature_index]
                finite_values = values[valid]
                valid_count = int(valid.sum())
                if valid_count:
                    output[action_index, cursor] = float(np.mean(finite_values))
                    output[action_index, cursor + 4] = valid_count / float(window)
                if valid_count >= 2:
                    positions = np.flatnonzero(valid).astype(np.float64)
                    output[action_index, cursor + 1] = float(np.std(finite_values, ddof=0))
                    centered = positions - float(np.mean(positions))
                    denominator = float(np.dot(centered, centered))
                    if denominator > 0.0:
                        output[action_index, cursor + 2] = float(
                            np.dot(centered, finite_values - float(np.mean(finite_values)))
                            / denominator
                        )
                    output[action_index, cursor + 3] = float(
                        finite_values[-1] - finite_values[0]
                    )
                cursor += 5
        if cursor != 442:
            raise AssertionError("signed-17 HGB summary column drift")
        output[action_index, 442] = offsets[action_index] / 50.0
        output[action_index, 443] = float(rights[action_index] == "C")
    return output


__all__ = [
    "EntrySnapshotV1",
    "Signed17FrameV1",
    "Signed17HistoryV1",
    "historical_snapshot_from_processed_row",
    "source_neutral_snapshot_from_causal_inputs",
    "validate_alpha_source_record",
    "signed17_from_snapshot",
    "build_identity_joined_history",
    "hgb_signed17_summaries",
]
