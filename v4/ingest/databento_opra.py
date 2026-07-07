"""Databento OPRA ingest for the SPXW 0DTE pilot.

This module intentionally accepts a Databento client object instead of importing
the paid SDK. Tests can pass a stub client, and production code can pass
``databento.Historical(API_KEY)``.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa

from v4.ingest.fingerprint import new_run_id
from v4.parser import ContractId, parse
from v4.schema.normalized import NORMALIZED_SCHEMA, validate_normalized_table
from v4.schema.types import SCHEMA_VERSION, ContractRoot, OptionRight, VendorSource


_NY = ZoneInfo("America/New_York")
_DATASET = "OPRA.PILLAR"
_PARENT_SYMBOL = "SPXW.OPT"


def _as_session_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _day_bounds(session_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(session_date, time(0, 0), tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _to_frame(result: Any) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if hasattr(result, "to_df"):
        return result.to_df()
    if isinstance(result, pa.Table):
        return result.to_pandas()
    return pd.DataFrame(result)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _first_present(row: Mapping[str, Any], candidates: Iterable[str]) -> Any:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            return row[candidate]
    return None


def _normal_price(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, Decimal):
        return float(value)
    value_f = float(value)
    if abs(value_f) > 9_000_000_000_000_000:
        return None
    if abs(value_f) >= 1_000_000:
        return value_f / 1_000_000_000.0
    return value_f


def _normal_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _normal_timestamp(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _expiration_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def _decimal_strike(value: Any) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    try:
        price = _normal_price(value)
        if price is None:
            return None
        return Decimal(str(price)).quantize(Decimal("0.000001"))
    except (InvalidOperation, ValueError):
        return None


def _is_five_point_strike(strike: Decimal | float | int | None) -> bool:
    if strike is None:
        return False
    strike_d = strike if isinstance(strike, Decimal) else Decimal(str(strike))
    return strike_d % Decimal("5") == 0


def _pm_settlement_time_utc(expiry: date) -> datetime:
    local_close = datetime.combine(expiry, time(16, 0), tzinfo=_NY)
    return local_close.astimezone(timezone.utc)


def _right_from_value(value: Any) -> OptionRight | None:
    if value is None or pd.isna(value):
        return None
    value_s = str(value).strip().upper()
    if value_s in {"C", "CALL"}:
        return OptionRight.CALL
    if value_s in {"P", "PUT"}:
        return OptionRight.PUT
    return None


def _raw_symbol_from_row(row: Mapping[str, Any]) -> str | None:
    raw = _first_present(row, ("raw_symbol", "symbol", "ts_symbol"))
    if raw is None:
        return None
    return str(raw).strip()


def _contract_from_definition_row(row: Mapping[str, Any]) -> ContractId | None:
    raw_symbol = _raw_symbol_from_row(row)
    if raw_symbol:
        try:
            return parse(raw_symbol)
        except ValueError:
            pass

    root = _first_present(row, ("root_symbol", "root", "underlying_symbol"))
    expiry = _expiration_date(_first_present(row, ("expiration", "expiration_date", "expiry")))
    strike = _decimal_strike(_first_present(row, ("strike_price", "strike")))
    right = _right_from_value(_first_present(row, ("instrument_class", "right", "put_call")))
    if root is None or expiry is None or strike is None or right is None:
        return None
    try:
        return ContractId(
            root=ContractRoot(str(root).strip().upper()),
            expiry=expiry,
            strike=strike,
            right=right,
        )
    except ValueError:
        return None


def _instrument_id(row: Mapping[str, Any]) -> int | None:
    return _normal_int(_first_present(row, ("instrument_id", "instrument_id_00")))


def filter_0dte_definitions(
    definitions: pd.DataFrame,
    session_date: str | date | datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Keep only PM-settled SPXW contracts expiring on the session date."""
    session = _as_session_date(session_date)
    if definitions.empty:
        return definitions.copy()

    records: list[dict[str, Any]] = []
    for row in definitions.to_dict("records"):
        cid = _contract_from_definition_row(row)
        if cid is None:
            continue
        if cid.root.value != "SPXW":
            continue
        if cid.expiry != session:
            continue
        if not _is_five_point_strike(cid.strike):
            continue

        raw_symbol = _raw_symbol_from_row(row) or cid.to_occ21()
        record = dict(row)
        record["raw_symbol"] = raw_symbol
        record["contract_id"] = cid.to_canonical()
        record["root"] = cid.root.value
        record["expiry"] = cid.expiry
        record["strike"] = cid.strike
        record["right"] = cid.right.value
        record["settlement_style"] = "PM"
        record["settlement_time_utc"] = _pm_settlement_time_utc(cid.expiry)
        records.append(record)

    return pd.DataFrame.from_records(records)


def _range_request(
    client: Any,
    *,
    schema: str,
    symbols: str | Sequence[str],
    start: datetime,
    end: datetime | None = None,
    stype_in: str = "raw_symbol",
    stype_out: str | None = None,
) -> pd.DataFrame:
    kwargs: dict[str, Any] = {
        "dataset": _DATASET,
        "schema": schema,
        "symbols": symbols,
        "stype_in": stype_in,
        "start": start,
    }
    if stype_out is not None:
        kwargs["stype_out"] = stype_out
    if end is not None:
        kwargs["end"] = end
    return _to_frame(client.timeseries.get_range(**kwargs))


def fetch_0dte_definitions(
    client: Any,
    session_date: str | date | datetime | pd.Timestamp,
    *,
    parent_symbol: str = _PARENT_SYMBOL,
) -> pd.DataFrame:
    """Download and filter Databento OPRA definitions for one SPXW 0DTE day."""
    session = _as_session_date(session_date)
    raw = _range_request(
        client,
        schema="definition",
        symbols=parent_symbol,
        stype_in="parent",
        start=datetime.combine(session, time(0, 0), tzinfo=timezone.utc),
        end=None,
    )
    return filter_0dte_definitions(raw, session)


def fetch_cbbo_1m(
    client: Any,
    session_date: str | date | datetime | pd.Timestamp,
    raw_symbols: Sequence[str],
) -> pd.DataFrame:
    """Download Databento consolidated best bid/offer one-minute records."""
    session = _as_session_date(session_date)
    start, end = _day_bounds(session)
    return _range_request(
        client,
        schema="cbbo-1m",
        symbols=list(raw_symbols),
        start=start,
        end=end,
    )


def fetch_option_ohlcv_1m(
    client: Any,
    session_date: str | date | datetime | pd.Timestamp,
    raw_symbols: Sequence[str],
) -> pd.DataFrame:
    """Download Databento option trade bars for actual minute volume."""
    session = _as_session_date(session_date)
    start, end = _day_bounds(session)
    return _range_request(
        client,
        schema="ohlcv-1m",
        symbols=list(raw_symbols),
        start=start,
        end=end,
    )


def fetch_open_interest(
    client: Any,
    session_date: str | date | datetime | pd.Timestamp,
    raw_symbols: Sequence[str],
) -> pd.DataFrame:
    """Download Databento statistics rows and keep start-of-day OI only."""
    session = _as_session_date(session_date)
    start, end = _day_bounds(session)
    stats = _range_request(
        client,
        schema="statistics",
        symbols=list(raw_symbols),
        start=start,
        end=end,
    )
    if stats.empty:
        return stats
    stat_col = _first_existing(stats.columns, ("stat_type",))
    if stat_col is None:
        raise ValueError("statistics frame must include stat_type")
    return stats[pd.to_numeric(stats[stat_col], errors="coerce") == 9].copy()


def _minute_key(frame: pd.DataFrame, time_candidates: Sequence[str]) -> pd.Series:
    time_col = _first_existing(frame.columns, time_candidates)
    if time_col is None:
        raise ValueError(f"missing timestamp column; have {list(frame.columns)}")
    return pd.to_datetime(frame[time_col], utc=True).dt.floor("min")


def _raw_symbol_series(frame: pd.DataFrame) -> pd.Series:
    raw_col = _first_existing(frame.columns, ("raw_symbol", "symbol", "ts_symbol"))
    if raw_col is not None:
        return frame[raw_col].astype(str).str.strip()
    raise ValueError(f"missing raw symbol column; have {list(frame.columns)}")


def _symbol_or_instrument_key(frame: pd.DataFrame) -> pd.Series:
    raw_col = _first_existing(frame.columns, ("raw_symbol", "symbol", "ts_symbol"))
    if raw_col is not None:
        return frame[raw_col].astype(str).str.strip()
    inst_col = _first_existing(frame.columns, ("instrument_id",))
    if inst_col is not None:
        return pd.to_numeric(frame[inst_col], errors="coerce").astype("Int64").astype(str)
    raise ValueError(f"missing raw symbol/instrument column; have {list(frame.columns)}")


def _build_ohlcv_lookup(ohlcv: pd.DataFrame | None) -> dict[tuple[str, pd.Timestamp], int]:
    if ohlcv is None or ohlcv.empty:
        return {}
    frame = ohlcv.copy()
    frame["_raw_symbol"] = _symbol_or_instrument_key(frame)
    # Databento CBBO timestamps mark the end of the interval, while OHLCV
    # timestamps mark the start of the trade aggregation interval. Align OHLCV
    # to the interval end so a 14:30 OHLCV bar is available to a 14:31 decision.
    frame["_minute"] = _minute_key(
        frame, ("event_time", "ts_event", "ts_recv", "timestamp")
    ) + pd.Timedelta(minutes=1)
    volume_col = _first_existing(frame.columns, ("volume",))
    if volume_col is None:
        return {}
    lookup: dict[tuple[str, pd.Timestamp], int] = {}
    for row in frame.to_dict("records"):
        volume = _normal_int(row.get(volume_col))
        if volume is not None:
            lookup[(row["_raw_symbol"], row["_minute"])] = volume
    return lookup


def _build_oi_lookup(stats: pd.DataFrame | None) -> dict[str, int]:
    if stats is None or stats.empty:
        return {}
    stat_col = _first_existing(stats.columns, ("stat_type",))
    if stat_col is not None:
        stats = stats[pd.to_numeric(stats[stat_col], errors="coerce") == 9].copy()
    if stats.empty:
        return {}

    value_cols = ("open_interest", "quantity", "value", "volume", "count")
    lookup: dict[str, int] = {}
    for row in stats.assign(_raw_symbol=_symbol_or_instrument_key(stats)).to_dict("records"):
        oi = _normal_int(_first_present(row, value_cols))
        if oi is not None:
            lookup[row["_raw_symbol"]] = oi
    return lookup


def _build_underlying_lookup(index_bars: pd.DataFrame | None):
    if index_bars is None or index_bars.empty:
        return lambda _: None
    frame = index_bars.copy()
    symbol_col = _first_existing(frame.columns, ("symbol",))
    if symbol_col is not None:
        frame = frame[frame[symbol_col].astype(str).str.upper() == "SPX"]
    time_col = _first_existing(frame.columns, ("event_time", "timestamp", "datetime", "time"))
    close_col = _first_existing(frame.columns, ("close", "price", "value", "last"))
    if frame.empty or time_col is None or close_col is None:
        return lambda _: None

    frame = frame[[time_col, close_col]].dropna().copy()
    frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
    frame = frame.sort_values(time_col)
    # Parquet vendors may round-trip timestamps as us/ms/ns resolution. Pandas
    # astype("int64") preserves the dtype unit, but Timestamp.value is always
    # nanoseconds, matching the lookup key below.
    times = frame[time_col].map(lambda ts: pd.Timestamp(ts).value).to_numpy(dtype=np.int64)
    closes = pd.to_numeric(frame[close_col], errors="coerce").to_numpy(dtype=float)

    def lookup(ts: pd.Timestamp | datetime | None) -> float | None:
        if ts is None or pd.isna(ts):
            return None
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")
        else:
            ts_pd = ts_pd.tz_convert("UTC")
        idx = np.searchsorted(times, ts_pd.value, side="right") - 1
        if idx < 0:
            return None
        return float(closes[idx])

    return lookup


def _definition_lookup(definitions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in definitions.to_dict("records"):
        cid = _contract_from_definition_row(row)
        if cid is None:
            continue
        raw_symbol = _raw_symbol_from_row(row) or cid.to_occ21()
        definition = {
            "cid": cid,
            "instrument_id": _instrument_id(row),
            "settlement_time_utc": row.get("settlement_time_utc")
            or _pm_settlement_time_utc(cid.expiry),
            "min_price_increment": _normal_price(
                _first_present(row, ("min_price_increment", "price_increment", "min_tick"))
            ),
            "contract_multiplier": _normal_int(
                _first_present(row, ("contract_multiplier", "multiplier", "unit_of_trade"))
            )
            or 100,
        }
        out[raw_symbol] = definition
        if definition["instrument_id"] is not None:
            out[str(definition["instrument_id"])] = definition
    return out


def _last_trade_time(row: Mapping[str, Any]) -> datetime | None:
    value = _first_present(
        row,
        (
            "last_trade_time",
            "last_trade_ts",
            "last_ts_event",
            "last_sale_time",
            "trade_time",
        ),
    )
    return _normal_timestamp(value)


def normalize_spxw_0dte_day(
    definitions: pd.DataFrame,
    cbbo_1m: pd.DataFrame,
    *,
    ohlcv_1m: pd.DataFrame | None = None,
    statistics: pd.DataFrame | None = None,
    index_bars: pd.DataFrame | None = None,
    ingest_run_id: str | None = None,
) -> pa.Table:
    """Normalize one day of SPXW 0DTE OPRA rows into v4's Normalized schema."""
    run_id = ingest_run_id or new_run_id()
    defs_by_raw = _definition_lookup(definitions)
    ohlcv_volume = _build_ohlcv_lookup(ohlcv_1m)
    oi_lookup = _build_oi_lookup(statistics)
    spx_at_or_before = _build_underlying_lookup(index_bars)

    if cbbo_1m.empty:
        table = pa.Table.from_pylist([], schema=NORMALIZED_SCHEMA)
        validate_normalized_table(table)
        return table

    frame = cbbo_1m.copy()
    frame["_raw_symbol"] = _symbol_or_instrument_key(frame)
    frame["_minute"] = _minute_key(frame, ("event_time", "ts_recv", "ts_event", "timestamp"))

    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        source_key = row["_raw_symbol"]
        definition = defs_by_raw.get(source_key)
        if definition is None:
            continue
        cid: ContractId = definition["cid"]
        raw_symbol = cid.to_occ21()
        if cid.root.value != "SPXW" or not _is_five_point_strike(cid.strike):
            continue

        quote_time = _normal_timestamp(
            _first_present(row, ("ts_recv", "event_time", "timestamp", "ts_event"))
        )
        event_time = quote_time or _normal_timestamp(_first_present(row, ("ts_event",)))
        last_time = _last_trade_time(row)
        last_trade = _normal_price(_first_present(row, ("price", "last", "last_trade")))
        last_size = _normal_int(_first_present(row, ("size", "last_size", "last_trade_size")))
        bid = _normal_price(_first_present(row, ("bid_px_00", "bid_price", "bid")))
        ask = _normal_price(_first_present(row, ("ask_px_00", "ask_price", "ask")))
        bid_size = _normal_int(_first_present(row, ("bid_sz_00", "bid_size", "bid_sz")))
        ask_size = _normal_int(_first_present(row, ("ask_sz_00", "ask_size", "ask_sz")))
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
        option_volume = ohlcv_volume.get((source_key, row["_minute"])) or ohlcv_volume.get(
            (raw_symbol, row["_minute"])
        )
        open_interest = oi_lookup.get(source_key) or oi_lookup.get(raw_symbol)

        quote_gap_seconds = None
        if quote_time is not None and last_time is not None:
            quote_gap_seconds = (quote_time - last_time).total_seconds()

        records.append(
            {
                "event_time": event_time,
                "receive_time": quote_time,
                "timestamp_source": (
                    "databento_cbbo_1m_ts_recv"
                    if _first_present(row, ("ts_recv",)) is not None
                    else "databento_cbbo_1m_ts_event"
                ),
                "contract_id": cid.to_canonical(),
                "raw_symbol": raw_symbol,
                "instrument_id": definition["instrument_id"],
                "root": cid.root.value,
                "expiry": cid.expiry,
                "strike": cid.strike,
                "right": cid.right.value,
                "settlement_style": "PM",
                "settlement_time_utc": definition["settlement_time_utc"],
                "min_price_increment": definition["min_price_increment"],
                "contract_multiplier": definition["contract_multiplier"],
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "mid": mid,
                "last_trade": last_trade,
                "last_trade_size": last_size,
                "quote_time": quote_time,
                "last_trade_time": last_time,
                "quote_age_ms": 0 if quote_time is not None else None,
                "quote_gap_seconds": quote_gap_seconds,
                "open_interest": open_interest,
                "open_interest_asof_date": cid.expiry,
                "stat_open_interest": open_interest,
                "volume": option_volume,
                "volume_asof_time": event_time if option_volume is not None else None,
                "option_ohlcv_volume": option_volume,
                "underlying_price": spx_at_or_before(event_time),
                "iv": None,
                "iv_source": None,
                "delta": None,
                "gamma": None,
                "theta": None,
                "vega": None,
                "rho": None,
                "charm": None,
                "vanna": None,
                "vomma": None,
                "greek_source": None,
                "greek_computation_ts": None,
                "risk_free_rate_used": None,
                "dividend_yield_used": None,
                "vendor_source": VendorSource.DATABENTO_OPRA.value,
                "ingest_run_id": run_id,
                "schema_version": SCHEMA_VERSION,
            }
        )

    table = pa.Table.from_pylist(records, schema=NORMALIZED_SCHEMA)
    validate_normalized_table(table)
    return table
