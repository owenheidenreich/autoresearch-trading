"""Bounded, fail-closed CMBP-1 streaming semantics for Job 51.

Historical ``DBNStore`` iterators and live callbacks both feed this module one
record at a time through :meth:`CmbpStreamDecoder.accept`.  The implementation
does not import a Databento client (or pandas), and deliberately keeps no
session-sized collection: only frozen contract mappings, one prior record per
instrument, clock diagnostics, and counters are retained.

Prices remain in Databento's fixed-point integer representation.  In
particular, comparisons never round-trip through float.
"""
from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal


CMBP1_RTYPE = 177
UNDEF_PRICE = (1 << 63) - 1

F_LAST = 1 << 7
F_TOB = 1 << 6
F_SNAPSHOT = 1 << 5
F_MBP = 1 << 4
F_BAD_TS_RECV = 1 << 3
F_MAYBE_BAD_BOOK = 1 << 2
F_PUBLISHER_SPECIFIC = 1 << 1

_KNOWN_FLAGS = (
    ("LAST", F_LAST),
    ("TOB", F_TOB),
    ("SNAPSHOT", F_SNAPSHOT),
    ("MBP", F_MBP),
    ("BAD_TS_RECV", F_BAD_TS_RECV),
    ("MAYBE_BAD_BOOK", F_MAYBE_BAD_BOOK),
    ("PUBLISHER_SPECIFIC", F_PUBLISHER_SPECIFIC),
)
_KNOWN_FLAG_MASK = sum(value for _, value in _KNOWN_FLAGS)


class CmbpStreamError(RuntimeError):
    """Named, sticky decoder failure with no vendor payload in its message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CmbpRecord:
    """SDK-independent CMBP-1 input used by synthetic and replay producers."""

    instrument_id: int
    ts_event: int
    ts_recv: int
    action: str
    side: str
    price: int
    size: int
    publisher_id: int = 0
    flags: int = 0
    bid_px: int | None = None
    ask_px: int | None = None
    bid_sz: int | None = None
    ask_sz: int | None = None
    bid_pb: int | None = None
    ask_pb: int | None = None
    rtype: int = CMBP1_RTYPE


@dataclass(frozen=True, slots=True)
class SymbolMappingRecord:
    """SDK-independent live raw-symbol to instrument mapping event."""

    instrument_id: int
    raw_symbol: str
    start_ts: int | None = None
    end_ts: int | None = None


@dataclass(frozen=True, slots=True)
class SystemRecord:
    """SDK-independent non-error control event."""

    code: str = "NONE"
    heartbeat: bool = False


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """SDK-independent fatal stream error; the server message is not retained."""

    code: str = "NONE"


@dataclass(frozen=True, slots=True)
class ContractMapping:
    instrument_id: int
    raw_symbol: str


@dataclass(frozen=True, slots=True)
class NamedCount:
    name: str
    count: int


@dataclass(frozen=True, slots=True)
class InstrumentSilence:
    instrument_id: int
    max_silence_ns: int


@dataclass(frozen=True, slots=True)
class SignedTrade:
    """A trade whose immediate prior touch passed the strict causal law."""

    session: str
    instrument_id: int
    raw_symbol: str
    direction: Literal["BUY", "SELL"]
    classification: Literal["AT_ASK", "AT_BID"]
    trade_ts_event: int
    trade_ts_recv: int
    trade_price: int
    trade_size: int
    trade_flags: int
    prior_ts_event: int
    prior_ts_recv: int
    prior_bid_px: int
    prior_ask_px: int
    prior_flags: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CmbpStreamSummary:
    """Immutable, canonical-JSON-safe terminal QC summary."""

    artifact_type: str
    session: str
    source_kind: Literal["historical", "live"]
    window_start_ns: int | None
    window_end_ns: int | None
    expected_mappings: tuple[ContractMapping, ...]
    active_mappings: tuple[ContractMapping, ...]
    seen_instrument_ids: tuple[int, ...]
    mapping_records: int
    mapping_reconciled: bool
    all_expected_instruments_seen: bool
    total_records_accepted: int
    cmbp1_records: int
    expected_cmbp1_records: int | None
    record_count_reconciled: bool | Literal["UNKNOWN"]
    trade_records: int
    strict_prior_trades: int
    tied_prior_trades_excluded: int
    no_prior_trades: int
    signed_trades: int
    at_bid_trades: int
    at_ask_trades: int
    inside_trades: int
    outside_trades: int
    ambiguous_trades: int
    undefined_trade_price_excluded: int
    missing_prior_book_excluded: int
    undefined_prior_book_excluded: int
    locked_prior_book_excluded: int
    crossed_prior_book_excluded: int
    trade_bad_ts_recv_excluded: int
    prior_bad_ts_recv_excluded: int
    prior_maybe_bad_book_excluded: int
    all_causal_priors_strict: bool
    tied_receive_priors_excluded: bool
    global_receive_ties: int
    global_receive_regressions: int
    global_event_regressions: int
    instrument_event_regressions: int
    first_ts_recv: int | None
    last_ts_recv: int | None
    max_stream_silence_ns: int | None
    max_instrument_silence_ns: int | None
    instrument_silences: tuple[InstrumentSilence, ...]
    disconnect_count: int
    reconnect_count: int
    gap_count: int
    book_state_clear_count: int
    gaps_with_known_bounds: int
    total_known_gap_ns: int
    max_known_gap_ns: int | None
    explicit_connection_telemetry: Literal["UNKNOWN", "NONE_OBSERVED", "OBSERVED"]
    system_records: int
    heartbeat_records: int
    system_code_counts: tuple[NamedCount, ...]
    rows_with_flags: int
    unknown_flag_rows: int
    flag_counts: tuple[NamedCount, ...]
    flag_value_counts: tuple[NamedCount, ...]
    action_counts: tuple[NamedCount, ...]
    side_counts: tuple[NamedCount, ...]
    classification_reconciled: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain value accepted by ``json.dumps(..., allow_nan=False)``."""

        return asdict(self)

    def market_semantics(self) -> dict[str, Any]:
        """Projection used to prove historical/live record-path parity."""

        value = self.to_dict()
        for key in (
            "source_kind",
            "mapping_records",
            "disconnect_count",
            "reconnect_count",
            "gap_count",
            "book_state_clear_count",
            "gaps_with_known_bounds",
            "total_known_gap_ns",
            "max_known_gap_ns",
            "explicit_connection_telemetry",
            "system_records",
            "heartbeat_records",
            "system_code_counts",
            "total_records_accepted",
        ):
            value.pop(key)
        return value


@dataclass(frozen=True, slots=True)
class _Prior:
    ts_event: int
    ts_recv: int
    bid_px: int | None
    ask_px: int | None
    flags: int


def _index(value: Any, field: str) -> int:
    try:
        result = operator.index(value)
    except (TypeError, ValueError) as exc:
        raise CmbpStreamError("STOP_MALFORMED_RECORD", f"{field} is not an integer") from exc
    return int(result)


def _optional_index(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _index(value, field)


def _text(value: Any, field: str) -> str:
    enum_value = getattr(value, "value", value)
    result = str(enum_value)
    if not result:
        raise CmbpStreamError("STOP_MALFORMED_RECORD", f"{field} is empty")
    return result


def _normalized_mapping_pairs(
    mappings: Mapping[int, str] | Iterable[tuple[int, str]],
) -> tuple[ContractMapping, ...]:
    items = mappings.items() if isinstance(mappings, Mapping) else mappings
    by_id: dict[int, str] = {}
    by_symbol: dict[str, int] = {}
    for raw_id, raw_symbol in items:
        instrument_id = _index(raw_id, "instrument_id")
        symbol = str(raw_symbol)
        if instrument_id <= 0 or not symbol:
            raise CmbpStreamError("STOP_CONTRACT_MAPPING", "mapping contains an invalid ID or symbol")
        if instrument_id in by_id and by_id[instrument_id] != symbol:
            raise CmbpStreamError("STOP_CONTRACT_MAPPING", "instrument ID maps to multiple raw symbols")
        if symbol in by_symbol and by_symbol[symbol] != instrument_id:
            raise CmbpStreamError("STOP_CONTRACT_MAPPING", "raw symbol maps to multiple instrument IDs")
        by_id[instrument_id] = symbol
        by_symbol[symbol] = instrument_id
    if not by_id:
        raise CmbpStreamError("STOP_CONTRACT_MAPPING", "expected mapping set is empty")
    return tuple(ContractMapping(instrument_id, by_id[instrument_id]) for instrument_id in sorted(by_id))


def expected_mappings_from_resolution(
    resolved_symbols: Mapping[str, Sequence[str | int]],
) -> tuple[ContractMapping, ...]:
    """Normalize Job-50 ``raw_symbol -> [instrument_id]`` resolution evidence."""

    pairs: list[tuple[int, str]] = []
    for raw_symbol, values in resolved_symbols.items():
        if isinstance(values, (str, bytes)) or len(values) != 1:
            raise CmbpStreamError("STOP_CONTRACT_MAPPING", "resolution is not exactly one-to-one")
        value = values[0]
        if isinstance(value, str):
            if not value.isdecimal() or str(int(value)) != value:
                raise CmbpStreamError("STOP_CONTRACT_MAPPING", "resolution instrument ID is not canonical")
            instrument_id = int(value)
        else:
            instrument_id = _index(value, "instrument_id")
        pairs.append((instrument_id, str(raw_symbol)))
    return _normalized_mapping_pairs(pairs)


def _iso_date(value: Any, field: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value)
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise CmbpStreamError("STOP_CONTRACT_MAPPING", f"historical mapping {field} is invalid") from exc


def historical_mappings_for_session(
    mappings: Mapping[str, Sequence[Mapping[str, Any] | Any]],
    session: str | date,
) -> tuple[ContractMapping, ...]:
    """Select exact DBN metadata mappings whose half-open date range covers a session."""

    session_date = _iso_date(session, "session")
    pairs: list[tuple[int, str]] = []
    for raw_symbol, raw_intervals in mappings.items():
        if isinstance(raw_intervals, Mapping):
            intervals: Sequence[Mapping[str, Any] | Any] = (raw_intervals,)
        else:
            intervals = raw_intervals
        selected: set[int] = set()
        for interval in intervals:
            if isinstance(interval, Mapping):
                start_value = interval.get("start_date")
                end_value = interval.get("end_date")
                symbol_value = interval.get("symbol")
            else:
                start_value = getattr(interval, "start_date", None)
                end_value = getattr(interval, "end_date", None)
                symbol_value = getattr(interval, "symbol", None)
            start_date = _iso_date(start_value, "start_date")
            end_date = _iso_date(end_value, "end_date")
            if end_date <= start_date:
                raise CmbpStreamError("STOP_CONTRACT_MAPPING", "historical mapping range is empty or reversed")
            if start_date <= session_date < end_date:
                text = str(symbol_value)
                if not text.isdecimal() or str(int(text)) != text:
                    raise CmbpStreamError("STOP_CONTRACT_MAPPING", "historical instrument ID is not canonical")
                selected.add(int(text))
        if len(selected) != 1:
            raise CmbpStreamError(
                "STOP_CONTRACT_MAPPING",
                "historical raw symbol has missing or ambiguous session mapping",
            )
        pairs.append((next(iter(selected)), str(raw_symbol)))
    return _normalized_mapping_pairs(pairs)


class CmbpStreamDecoder:
    """One-record shared state machine for historical and live CMBP-1."""

    def __init__(
        self,
        *,
        session: str,
        expected_mappings: Mapping[int, str] | Iterable[tuple[int, str]],
        source_kind: Literal["historical", "live"],
        window_start_ns: int | None = None,
        window_end_ns: int | None = None,
    ) -> None:
        if source_kind not in {"historical", "live"}:
            raise CmbpStreamError("STOP_DECODER_CONFIGURATION", "source_kind must be historical or live")
        try:
            session_date = date.fromisoformat(str(session))
        except ValueError as exc:
            raise CmbpStreamError("STOP_DECODER_CONFIGURATION", "session must be an ISO date") from exc
        self.session = session_date.isoformat()
        self.source_kind = source_kind
        self.window_start_ns = _optional_index(window_start_ns, "window_start_ns")
        self.window_end_ns = _optional_index(window_end_ns, "window_end_ns")
        if (self.window_start_ns is None) != (self.window_end_ns is None):
            raise CmbpStreamError("STOP_DECODER_CONFIGURATION", "both window bounds must be supplied")
        if self.window_start_ns is not None and self.window_start_ns >= self.window_end_ns:
            raise CmbpStreamError("STOP_DECODER_CONFIGURATION", "window bounds are empty or reversed")

        self._expected = _normalized_mapping_pairs(expected_mappings)
        self._expected_by_id = {item.instrument_id: item.raw_symbol for item in self._expected}
        self._expected_by_symbol = {item.raw_symbol: item.instrument_id for item in self._expected}
        self._active_by_id: dict[int, str] = {}
        self._active_by_symbol: dict[str, int] = {}
        self._seen_instruments: set[int] = set()
        self._prior: dict[int, _Prior] = {}

        self._failure: CmbpStreamError | None = None
        self._summary: CmbpStreamSummary | None = None

        self._total_records = 0
        self._cmbp_records = 0
        self._mapping_records = 0
        self._system_records = 0
        self._heartbeat_records = 0
        self._system_codes: Counter[str] = Counter()
        self._action_counts: Counter[str] = Counter()
        self._side_counts: Counter[str] = Counter()
        self._flag_value_counts: Counter[str] = Counter()
        self._flag_counts: Counter[str] = Counter()
        self._rows_with_flags = 0
        self._unknown_flag_rows = 0

        self._trades = 0
        self._strict_prior = 0
        self._tied_prior = 0
        self._no_prior = 0
        self._at_bid = 0
        self._at_ask = 0
        self._inside = 0
        self._outside = 0
        self._ambiguous = 0
        self._undefined_trade_price = 0
        self._missing_prior_book = 0
        self._undefined_prior_book = 0
        self._locked_prior_book = 0
        self._crossed_prior_book = 0
        self._trade_bad_ts_recv = 0
        self._prior_bad_ts_recv = 0
        self._prior_maybe_bad_book = 0

        self._first_ts_recv: int | None = None
        self._last_ts_recv: int | None = None
        self._last_ts_event: int | None = None
        self._last_instrument_recv: dict[int, int] = {}
        self._last_instrument_event: dict[int, int] = {}
        self._max_stream_silence: int | None = None
        self._max_instrument_silence: dict[int, int] = {}
        self._global_receive_ties = 0
        self._global_receive_regressions = 0
        self._global_event_regressions = 0
        self._instrument_event_regressions = 0

        self._disconnects = 0
        self._reconnects = 0
        self._gaps = 0
        self._book_state_clears = 0
        self._known_gaps = 0
        self._total_gap_ns = 0
        self._max_gap_ns: int | None = None

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def _fail(self, code: str, message: str) -> None:
        error = CmbpStreamError(code, message)
        self._failure = error
        raise error

    def _require_open(self) -> None:
        if self._failure is not None:
            raise CmbpStreamError(self._failure.code, str(self._failure))
        if self._summary is not None:
            raise CmbpStreamError("STOP_DECODER_FINALIZED", "decoder was already finalized")

    def preload_mappings(
        self,
        mappings: Mapping[int, str] | Iterable[tuple[int, str] | ContractMapping],
    ) -> None:
        """Load a complete historical mapping set before the first market row."""

        self._require_open()
        if self._cmbp_records or self._mapping_records:
            self._fail("STOP_CONTRACT_MAPPING", "mapping preload occurred after stream records")
        pairs = (
            (item.instrument_id, item.raw_symbol) if isinstance(item, ContractMapping) else item
            for item in (mappings.items() if isinstance(mappings, Mapping) else mappings)
        )
        try:
            normalized = _normalized_mapping_pairs(pairs)
        except CmbpStreamError as exc:
            self._failure = exc
            raise
        if normalized != self._expected:
            self._fail("STOP_CONTRACT_MAPPING", "preloaded mappings differ from frozen mappings")
        self._active_by_id = {item.instrument_id: item.raw_symbol for item in normalized}
        self._active_by_symbol = {item.raw_symbol: item.instrument_id for item in normalized}

    def preload_historical_mappings(
        self,
        mappings: Mapping[str, Sequence[Mapping[str, Any] | Any]],
    ) -> None:
        try:
            selected = historical_mappings_for_session(mappings, self.session)
        except CmbpStreamError as exc:
            self._failure = exc
            raise
        self.preload_mappings(selected)

    def _register_mapping(self, record: SymbolMappingRecord) -> None:
        instrument_id = _index(record.instrument_id, "instrument_id")
        raw_symbol = str(record.raw_symbol)
        if record.start_ts is not None or record.end_ts is not None:
            if record.start_ts is None or record.end_ts is None:
                self._fail("STOP_CONTRACT_MAPPING", "mapping validity bounds are incomplete")
            start_ts = _index(record.start_ts, "mapping start_ts")
            end_ts = _index(record.end_ts, "mapping end_ts")
            if end_ts < start_ts:
                self._fail("STOP_CONTRACT_MAPPING", "mapping validity range is reversed")
        if self._expected_by_id.get(instrument_id) != raw_symbol:
            self._fail("STOP_CONTRACT_MAPPING", "live mapping is outside the frozen ID-symbol pairs")
        if self._expected_by_symbol.get(raw_symbol) != instrument_id:
            self._fail("STOP_CONTRACT_MAPPING", "live raw symbol mapping is not one-to-one")
        active_symbol = self._active_by_id.get(instrument_id)
        active_id = self._active_by_symbol.get(raw_symbol)
        if (active_symbol is not None and active_symbol != raw_symbol) or (
            active_id is not None and active_id != instrument_id
        ):
            self._fail("STOP_CONTRACT_MAPPING", "live mapping conflicts with active mapping")
        self._active_by_id[instrument_id] = raw_symbol
        self._active_by_symbol[raw_symbol] = instrument_id
        self._mapping_records += 1
        self._total_records += 1

    def _parse_sdk_mapping(self, record: Any) -> SymbolMappingRecord:
        try:
            stype_in = _text(record.stype_in, "stype_in")
            stype_out = _text(record.stype_out, "stype_out")
            raw_symbol = str(record.stype_in_symbol)
            out_symbol = str(record.stype_out_symbol)
            instrument_id = _index(record.instrument_id, "instrument_id")
            start_ts = _index(record.start_ts, "mapping start_ts")
            end_ts = _index(record.end_ts, "mapping end_ts")
        except (AttributeError, CmbpStreamError) as exc:
            if isinstance(exc, CmbpStreamError):
                raise
            self._fail("STOP_MALFORMED_RECORD", "SymbolMappingMsg is missing required fields")
        if stype_in != "raw_symbol" or stype_out != "instrument_id":
            self._fail("STOP_CONTRACT_MAPPING", "SymbolMappingMsg symbology types differ from contract")
        if not out_symbol.isdecimal() or str(int(out_symbol)) != out_symbol or int(out_symbol) != instrument_id:
            self._fail("STOP_CONTRACT_MAPPING", "SymbolMappingMsg instrument fields disagree")
        return SymbolMappingRecord(instrument_id, raw_symbol, start_ts, end_ts)

    def _parse_sdk_cmbp(self, record: Any) -> CmbpRecord:
        try:
            levels = record.levels
            level = levels[0] if len(levels) else None
            return CmbpRecord(
                rtype=_index(record.rtype, "rtype"),
                publisher_id=_index(record.publisher_id, "publisher_id"),
                instrument_id=_index(record.instrument_id, "instrument_id"),
                ts_event=_index(record.ts_event, "ts_event"),
                ts_recv=_index(record.ts_recv, "ts_recv"),
                action=_text(record.action, "action"),
                side=_text(record.side, "side"),
                price=_index(record.price, "price"),
                size=_index(record.size, "size"),
                flags=_index(record.flags, "flags"),
                bid_px=None if level is None else _index(level.bid_px, "bid_px"),
                ask_px=None if level is None else _index(level.ask_px, "ask_px"),
                bid_sz=None if level is None else _index(level.bid_sz, "bid_sz"),
                ask_sz=None if level is None else _index(level.ask_sz, "ask_sz"),
                bid_pb=None if level is None else _optional_index(level.bid_pb, "bid_pb"),
                ask_pb=None if level is None else _optional_index(level.ask_pb, "ask_pb"),
            )
        except (AttributeError, IndexError, TypeError) as exc:
            raise CmbpStreamError("STOP_MALFORMED_RECORD", "CMBP1Msg is missing required fields") from exc

    def accept(self, record: Any) -> SignedTrade | None:
        """Consume exactly one normalized or pinned-SDK DBN record."""

        self._require_open()
        try:
            return self._accept_one(record)
        except CmbpStreamError as exc:
            self._failure = exc
            raise

    def _accept_one(self, record: Any) -> SignedTrade | None:
        if isinstance(record, CmbpRecord):
            return self._accept_cmbp(record)
        if isinstance(record, SymbolMappingRecord):
            self._register_mapping(record)
            return None
        if isinstance(record, SystemRecord):
            self._accept_system(record.code, record.heartbeat)
            return None
        if isinstance(record, ErrorRecord):
            self._fail("STOP_STREAM_ERROR", f"stream emitted ErrorRecord code={record.code}")

        record_name = type(record).__name__
        if record_name == "CMBP1Msg":
            return self._accept_cmbp(self._parse_sdk_cmbp(record))
        if record_name == "SymbolMappingMsg":
            self._register_mapping(self._parse_sdk_mapping(record))
            return None
        if record_name == "SystemMsg":
            try:
                code = _text(getattr(record, "code", None), "system code")
                message = str(getattr(record, "msg", ""))
            except CmbpStreamError:
                code = "NONE"
                message = ""
            self._accept_system(code, code.upper() == "HEARTBEAT" or message.lower() == "heartbeat")
            return None
        if record_name == "ErrorMsg":
            code = str(getattr(record, "code", "NONE"))
            self._fail("STOP_STREAM_ERROR", f"stream emitted ErrorMsg code={code}")
        self._fail("STOP_WRONG_RTYPE", f"unsupported DBN record class {record_name}")

    def _accept_system(self, code: str, heartbeat: bool) -> None:
        normalized_code = str(code) or "NONE"
        if len(normalized_code) > 128:
            self._fail("STOP_MALFORMED_RECORD", "system code exceeds bounded length")
        self._system_records += 1
        self._total_records += 1
        self._system_codes[normalized_code] += 1
        if heartbeat:
            self._heartbeat_records += 1

    def _validate_market_record(self, record: CmbpRecord) -> None:
        if _index(record.rtype, "rtype") != CMBP1_RTYPE:
            self._fail("STOP_WRONG_RTYPE", "data record is not CMBP-1 rtype 177")
        if record.instrument_id <= 0 or record.ts_event < 0 or record.ts_recv < 0:
            self._fail("STOP_MALFORMED_RECORD", "CMBP-1 identity or clock is invalid")
        if record.size < 0 or record.publisher_id < 0 or not 0 <= record.flags <= 0xFF:
            self._fail("STOP_MALFORMED_RECORD", "CMBP-1 size, publisher, or flags are invalid")
        # Historical get_range filters CMBP on ts_recv. Exchange event time may
        # legitimately precede the receive boundary by transport latency.
        if self.window_start_ns is not None and not (
            self.window_start_ns <= record.ts_recv < self.window_end_ns
        ):
            self._fail("STOP_OUT_OF_WINDOW", "CMBP-1 receive timestamp is outside frozen request bounds")
        if self._active_by_id.get(record.instrument_id) != self._expected_by_id.get(record.instrument_id):
            self._fail("STOP_CONTRACT_MAPPING", "CMBP-1 instrument is unmapped or outside frozen scope")

    def _observe_clocks(self, record: CmbpRecord) -> None:
        if self._last_ts_recv is None:
            self._first_ts_recv = record.ts_recv
        else:
            delta = record.ts_recv - self._last_ts_recv
            if delta < 0:
                self._global_receive_regressions += 1
                self._fail("STOP_TS_RECV_REGRESSION", "global CMBP-1 receive clock regressed")
            if delta == 0:
                self._global_receive_ties += 1
            self._max_stream_silence = delta if self._max_stream_silence is None else max(
                self._max_stream_silence, delta
            )
        if self._last_ts_event is not None and record.ts_event < self._last_ts_event:
            self._global_event_regressions += 1

        instrument_recv = self._last_instrument_recv.get(record.instrument_id)
        if instrument_recv is not None:
            instrument_delta = record.ts_recv - instrument_recv
            if instrument_delta < 0:
                self._global_receive_regressions += 1
                self._fail("STOP_TS_RECV_REGRESSION", "per-instrument CMBP-1 receive clock regressed")
            self._max_instrument_silence[record.instrument_id] = max(
                self._max_instrument_silence.get(record.instrument_id, 0), instrument_delta
            )
        instrument_event = self._last_instrument_event.get(record.instrument_id)
        if instrument_event is not None and record.ts_event < instrument_event:
            self._instrument_event_regressions += 1

        self._last_ts_recv = record.ts_recv
        self._last_ts_event = record.ts_event
        self._last_instrument_recv[record.instrument_id] = record.ts_recv
        self._last_instrument_event[record.instrument_id] = record.ts_event

    def _count_record_shape(self, record: CmbpRecord) -> None:
        self._action_counts[record.action] += 1
        self._side_counts[record.side] += 1
        self._flag_value_counts[str(record.flags)] += 1
        if record.flags:
            self._rows_with_flags += 1
        if record.flags & ~_KNOWN_FLAG_MASK:
            self._unknown_flag_rows += 1
        for name, value in _KNOWN_FLAGS:
            if record.flags & value:
                self._flag_counts[name] += 1

    def _accept_cmbp(self, raw_record: CmbpRecord) -> SignedTrade | None:
        try:
            record = CmbpRecord(
                instrument_id=_index(raw_record.instrument_id, "instrument_id"),
                ts_event=_index(raw_record.ts_event, "ts_event"),
                ts_recv=_index(raw_record.ts_recv, "ts_recv"),
                action=_text(raw_record.action, "action"),
                side=_text(raw_record.side, "side"),
                price=_index(raw_record.price, "price"),
                size=_index(raw_record.size, "size"),
                publisher_id=_index(raw_record.publisher_id, "publisher_id"),
                flags=_index(raw_record.flags, "flags"),
                bid_px=_optional_index(raw_record.bid_px, "bid_px"),
                ask_px=_optional_index(raw_record.ask_px, "ask_px"),
                bid_sz=_optional_index(raw_record.bid_sz, "bid_sz"),
                ask_sz=_optional_index(raw_record.ask_sz, "ask_sz"),
                bid_pb=_optional_index(raw_record.bid_pb, "bid_pb"),
                ask_pb=_optional_index(raw_record.ask_pb, "ask_pb"),
                rtype=_index(raw_record.rtype, "rtype"),
            )
        except CmbpStreamError as exc:
            self._failure = exc
            raise
        self._validate_market_record(record)
        self._observe_clocks(record)
        self._cmbp_records += 1
        self._total_records += 1
        self._seen_instruments.add(record.instrument_id)
        self._count_record_shape(record)

        signed: SignedTrade | None = None
        if record.action == "T":
            signed = self._classify_trade(record, self._prior.get(record.instrument_id))
        self._prior[record.instrument_id] = _Prior(
            ts_event=record.ts_event,
            ts_recv=record.ts_recv,
            bid_px=record.bid_px,
            ask_px=record.ask_px,
            flags=record.flags,
        )
        return signed

    def _exclude_ambiguous(self) -> None:
        self._ambiguous += 1

    def _classify_trade(self, trade: CmbpRecord, prior: _Prior | None) -> SignedTrade | None:
        self._trades += 1
        if prior is None:
            self._no_prior += 1
            self._exclude_ambiguous()
            return None
        if prior.ts_recv == trade.ts_recv:
            self._tied_prior += 1
            self._exclude_ambiguous()
            return None
        if prior.ts_recv > trade.ts_recv:
            self._fail("STOP_TS_RECV_REGRESSION", "same-instrument causal prior receive clock regressed")
        self._strict_prior += 1

        if trade.flags & F_BAD_TS_RECV:
            self._trade_bad_ts_recv += 1
            self._exclude_ambiguous()
            return None
        if prior.flags & F_BAD_TS_RECV:
            self._prior_bad_ts_recv += 1
            self._exclude_ambiguous()
            return None
        if prior.flags & F_MAYBE_BAD_BOOK:
            self._prior_maybe_bad_book += 1
            self._exclude_ambiguous()
            return None
        if trade.price == UNDEF_PRICE:
            self._undefined_trade_price += 1
            self._exclude_ambiguous()
            return None
        if prior.bid_px is None or prior.ask_px is None:
            self._missing_prior_book += 1
            self._exclude_ambiguous()
            return None
        if prior.bid_px == UNDEF_PRICE or prior.ask_px == UNDEF_PRICE:
            self._undefined_prior_book += 1
            self._exclude_ambiguous()
            return None
        if prior.bid_px == prior.ask_px:
            self._locked_prior_book += 1
            self._exclude_ambiguous()
            return None
        if prior.bid_px > prior.ask_px:
            self._crossed_prior_book += 1
            self._exclude_ambiguous()
            return None

        if trade.price == prior.ask_px:
            self._at_ask += 1
            return SignedTrade(
                session=self.session,
                instrument_id=trade.instrument_id,
                raw_symbol=self._active_by_id[trade.instrument_id],
                direction="BUY",
                classification="AT_ASK",
                trade_ts_event=trade.ts_event,
                trade_ts_recv=trade.ts_recv,
                trade_price=trade.price,
                trade_size=trade.size,
                trade_flags=trade.flags,
                prior_ts_event=prior.ts_event,
                prior_ts_recv=prior.ts_recv,
                prior_bid_px=prior.bid_px,
                prior_ask_px=prior.ask_px,
                prior_flags=prior.flags,
            )
        if trade.price == prior.bid_px:
            self._at_bid += 1
            return SignedTrade(
                session=self.session,
                instrument_id=trade.instrument_id,
                raw_symbol=self._active_by_id[trade.instrument_id],
                direction="SELL",
                classification="AT_BID",
                trade_ts_event=trade.ts_event,
                trade_ts_recv=trade.ts_recv,
                trade_price=trade.price,
                trade_size=trade.size,
                trade_flags=trade.flags,
                prior_ts_event=prior.ts_event,
                prior_ts_recv=prior.ts_recv,
                prior_bid_px=prior.bid_px,
                prior_ask_px=prior.ask_px,
                prior_flags=prior.flags,
            )
        if prior.bid_px < trade.price < prior.ask_px:
            self._inside += 1
        else:
            self._outside += 1
        return None

    def consume(
        self,
        records: Iterable[Any],
        *,
        on_signed_trade: Callable[[SignedTrade], None] | None = None,
    ) -> None:
        """Stream an iterable without materializing it or retaining signed events."""

        for record in records:
            signed = self.accept(record)
            if signed is not None and on_signed_trade is not None:
                on_signed_trade(signed)

    def _clear_book_state(self) -> None:
        self._prior.clear()
        self._book_state_clears += 1

    def note_disconnect(self) -> None:
        self._require_open()
        self._disconnects += 1
        self._clear_book_state()

    def note_reconnect(self) -> None:
        self._require_open()
        self._reconnects += 1
        self._clear_book_state()

    def note_gap(self, start_ts: int | None = None, end_ts: int | None = None) -> None:
        self._require_open()
        self._gaps += 1
        if start_ts is not None or end_ts is not None:
            if start_ts is None or end_ts is None:
                self._fail("STOP_MALFORMED_CONTROL", "gap bounds are incomplete")
            try:
                start = _index(start_ts, "gap start_ts")
                end = _index(end_ts, "gap end_ts")
            except CmbpStreamError as exc:
                self._failure = exc
                raise
            if end < start:
                self._fail("STOP_MALFORMED_CONTROL", "gap bounds are reversed")
            duration = end - start
            self._known_gaps += 1
            self._total_gap_ns += duration
            self._max_gap_ns = duration if self._max_gap_ns is None else max(self._max_gap_ns, duration)
        self._clear_book_state()

    @staticmethod
    def _named_counts(counter: Mapping[str, int], *, include_zero: Iterable[str] = ()) -> tuple[NamedCount, ...]:
        values = dict(counter)
        for name in include_zero:
            values.setdefault(name, 0)
        return tuple(NamedCount(name, int(values[name])) for name in sorted(values))

    def finalize(
        self,
        *,
        expected_record_count: int | None = None,
        require_all_expected_mappings: bool = True,
        require_all_expected_instruments: bool = False,
    ) -> CmbpStreamSummary:
        """Reconcile terminal counts and freeze a canonical-JSON-safe summary."""

        self._require_open()
        try:
            expected_count = _optional_index(expected_record_count, "expected_record_count")
        except CmbpStreamError as exc:
            self._failure = exc
            raise
        if expected_count is not None and expected_count < 0:
            self._fail("STOP_RECORD_COUNT_RECONCILIATION", "expected record count is negative")
        active = tuple(
            ContractMapping(instrument_id, self._active_by_id[instrument_id])
            for instrument_id in sorted(self._active_by_id)
        )
        mapping_reconciled = active == self._expected
        if require_all_expected_mappings and not mapping_reconciled:
            self._fail("STOP_CONTRACT_MAPPING", "terminal active mappings do not match frozen mappings")
        all_instruments_seen = self._seen_instruments == set(self._expected_by_id)
        if require_all_expected_instruments and not all_instruments_seen:
            self._fail("STOP_CONTRACT_MAPPING", "not every frozen instrument appeared in CMBP-1 records")
        reconciled: bool | Literal["UNKNOWN"] = "UNKNOWN"
        if expected_count is not None:
            reconciled = self._cmbp_records == expected_count
            if not reconciled:
                self._fail(
                    "STOP_RECORD_COUNT_RECONCILIATION",
                    "decoded CMBP-1 count differs from frozen census count",
                )

        terminal_classifications = self._at_bid + self._at_ask + self._inside + self._outside + self._ambiguous
        classification_reconciled = self._trades == terminal_classifications
        strict_reconciled = self._strict_prior == (
            self._at_bid
            + self._at_ask
            + self._inside
            + self._outside
            + self._undefined_trade_price
            + self._missing_prior_book
            + self._undefined_prior_book
            + self._locked_prior_book
            + self._crossed_prior_book
            + self._trade_bad_ts_recv
            + self._prior_bad_ts_recv
            + self._prior_maybe_bad_book
        )
        if not classification_reconciled or not strict_reconciled:
            self._fail("STOP_CLASSIFICATION_RECONCILIATION", "trade classification counters do not reconcile")

        instrument_silences = tuple(
            InstrumentSilence(instrument_id, self._max_instrument_silence[instrument_id])
            for instrument_id in sorted(self._max_instrument_silence)
        )
        max_instrument_silence = max(self._max_instrument_silence.values(), default=None)
        explicit_telemetry: Literal["UNKNOWN", "NONE_OBSERVED", "OBSERVED"]
        if self._disconnects or self._reconnects or self._gaps:
            explicit_telemetry = "OBSERVED"
        elif self.source_kind == "historical":
            explicit_telemetry = "UNKNOWN"
        else:
            explicit_telemetry = "NONE_OBSERVED"

        self._summary = CmbpStreamSummary(
            artifact_type="JOB51_CMBP_STREAM_SUMMARY_V1",
            session=self.session,
            source_kind=self.source_kind,
            window_start_ns=self.window_start_ns,
            window_end_ns=self.window_end_ns,
            expected_mappings=self._expected,
            active_mappings=active,
            seen_instrument_ids=tuple(sorted(self._seen_instruments)),
            mapping_records=self._mapping_records,
            mapping_reconciled=mapping_reconciled,
            all_expected_instruments_seen=all_instruments_seen,
            total_records_accepted=self._total_records,
            cmbp1_records=self._cmbp_records,
            expected_cmbp1_records=expected_count,
            record_count_reconciled=reconciled,
            trade_records=self._trades,
            strict_prior_trades=self._strict_prior,
            tied_prior_trades_excluded=self._tied_prior,
            no_prior_trades=self._no_prior,
            signed_trades=self._at_bid + self._at_ask,
            at_bid_trades=self._at_bid,
            at_ask_trades=self._at_ask,
            inside_trades=self._inside,
            outside_trades=self._outside,
            ambiguous_trades=self._ambiguous,
            undefined_trade_price_excluded=self._undefined_trade_price,
            missing_prior_book_excluded=self._missing_prior_book,
            undefined_prior_book_excluded=self._undefined_prior_book,
            locked_prior_book_excluded=self._locked_prior_book,
            crossed_prior_book_excluded=self._crossed_prior_book,
            trade_bad_ts_recv_excluded=self._trade_bad_ts_recv,
            prior_bad_ts_recv_excluded=self._prior_bad_ts_recv,
            prior_maybe_bad_book_excluded=self._prior_maybe_bad_book,
            all_causal_priors_strict=True,
            tied_receive_priors_excluded=True,
            global_receive_ties=self._global_receive_ties,
            global_receive_regressions=self._global_receive_regressions,
            global_event_regressions=self._global_event_regressions,
            instrument_event_regressions=self._instrument_event_regressions,
            first_ts_recv=self._first_ts_recv,
            last_ts_recv=self._last_ts_recv,
            max_stream_silence_ns=self._max_stream_silence,
            max_instrument_silence_ns=max_instrument_silence,
            instrument_silences=instrument_silences,
            disconnect_count=self._disconnects,
            reconnect_count=self._reconnects,
            gap_count=self._gaps,
            book_state_clear_count=self._book_state_clears,
            gaps_with_known_bounds=self._known_gaps,
            total_known_gap_ns=self._total_gap_ns,
            max_known_gap_ns=self._max_gap_ns,
            explicit_connection_telemetry=explicit_telemetry,
            system_records=self._system_records,
            heartbeat_records=self._heartbeat_records,
            system_code_counts=self._named_counts(self._system_codes),
            rows_with_flags=self._rows_with_flags,
            unknown_flag_rows=self._unknown_flag_rows,
            flag_counts=self._named_counts(self._flag_counts, include_zero=(name for name, _ in _KNOWN_FLAGS)),
            flag_value_counts=self._named_counts(self._flag_value_counts),
            action_counts=self._named_counts(self._action_counts),
            side_counts=self._named_counts(self._side_counts),
            classification_reconciled=classification_reconciled and strict_reconciled,
        )
        return self._summary
