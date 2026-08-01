"""OptionsDX ingest.

OptionsDX provides free historical SPX/SPY option chains 2010–2023 with
bid/ask/last/IV/Greeks at minute resolution. Used in v4 Phase 0 for:
- pipeline scaffolding
- simulator skeleton
- Greeks reconciliation against our Black-Scholes computation

Used NOT for: final edge proof, recent-regime validation (their coverage
ends in 2023, before the modern 2024+ 0DTE regime).

CSV format (publicly documented; see tests/fixtures/optionsdx_spx_sample.csv):
Single CSV row contains BOTH the call and put quotes for one (timestamp,
expiry, strike) — wide format. We expand to one row per (timestamp, contract)
during ingest, producing the long format the Normalized layer expects.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pyarrow as pa

from v4.ingest.fingerprint import file_sha256, new_run_id
from v4.parser import ContractId, from_columns
from v4.schema.normalized import NORMALIZED_SCHEMA, validate_normalized_table
from v4.schema.raw import RAW_PROVENANCE_SCHEMA, RawProvenance
from v4.schema.types import (
    SCHEMA_VERSION,
    ContractRoot,
    GreekSource,
    OptionRight,
    VendorSource,
)


# Column-name strip rules: OptionsDX ships columns wrapped like "[QUOTE_DATE]"
# and prefixes call/put metrics with C_ / P_.
_BRACKET_RE = ("[", "]")
_NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class OptionsDXIngestResult:
    raw_provenance: pa.Table
    normalized: pa.Table
    ingest_run_id: str
    input_sha256: str
    input_path: str
    rows_emitted: int


def _strip_brackets(s: str) -> str:
    s = s.strip()
    if s.startswith(_BRACKET_RE[0]) and s.endswith(_BRACKET_RE[1]):
        return s[1:-1]
    return s


def _parse_quote_unixtime(seconds: str) -> datetime:
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


def _parse_expiry(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _parse_decimal(s: str) -> Decimal:
    s = s.strip()
    if not s:
        raise ValueError("empty decimal field")
    return Decimal(s)


def _parse_float_or_none(s: str) -> float | None:
    s = s.strip()
    if s == "" or s.lower() in ("nan", "null"):
        return None
    return float(s)


def _parse_int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _settlement_style(root: ContractRoot) -> str | None:
    if root == ContractRoot.SPXW:
        return "PM"
    if root == ContractRoot.SPX:
        return "AM"
    return None


def _pm_settlement_time_utc(expiry: date) -> datetime | None:
    local_close = datetime.combine(expiry, time(16, 0), tzinfo=_NY)
    return local_close.astimezone(timezone.utc)


def _row_to_records(
    row: dict[str, str],
    *,
    root: ContractRoot,
    ingest_run_id: str,
) -> Iterable[dict]:
    """Yield two records per OptionsDX row: one call, one put.

    Skips a side if both bid and ask are missing (vendor convention for
    "no quote" on an OTM strike).
    """
    event_dt = _parse_quote_unixtime(row["QUOTE_UNIXTIME"])
    expiry = _parse_expiry(row["EXPIRE_DATE"])
    strike_d = _parse_decimal(row["STRIKE"])
    underlying = _parse_float_or_none(row["UNDERLYING_LAST"])

    for side, right in (("C", OptionRight.CALL), ("P", OptionRight.PUT)):
        bid = _parse_float_or_none(row[f"{side}_BID"])
        ask = _parse_float_or_none(row[f"{side}_ASK"])
        if bid is None and ask is None:
            continue
        last = _parse_float_or_none(row[f"{side}_LAST"])
        volume = _parse_int_or_none(row[f"{side}_VOLUME"])
        iv = _parse_float_or_none(row[f"{side}_IV"])
        delta = _parse_float_or_none(row[f"{side}_DELTA"])
        gamma = _parse_float_or_none(row[f"{side}_GAMMA"])
        theta = _parse_float_or_none(row[f"{side}_THETA"])
        vega = _parse_float_or_none(row[f"{side}_VEGA"])
        rho = _parse_float_or_none(row[f"{side}_RHO"])

        cid: ContractId = from_columns(
            root=root, expiry=expiry, strike=strike_d, right=right
        )
        mid = None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0

        yield {
            "event_time": event_dt,
            "receive_time": event_dt,  # OptionsDX is EOD-published; no separate receive
            "timestamp_source": "optionsdx_quote_unixtime",
            "contract_id": cid.to_canonical(),
            "raw_symbol": cid.to_occ21(),
            "instrument_id": None,
            "root": root.value,
            "expiry": expiry,
            "strike": strike_d,
            "right": right.value,
            "settlement_style": _settlement_style(root),
            "settlement_time_utc": (
                _pm_settlement_time_utc(expiry)
                if root == ContractRoot.SPXW
                else None
            ),
            "min_price_increment": None,
            "contract_multiplier": 100,
            "bid": bid,
            "ask": ask,
            "bid_size": None,
            "ask_size": None,
            "mid": mid,
            "last_trade": last,
            "last_trade_size": None,
            "quote_time": event_dt,
            "last_trade_time": event_dt if last is not None else None,
            "quote_age_ms": None,
            "quote_gap_seconds": 0.0 if last is not None else None,
            "open_interest": None,  # OptionsDX intraday does not ship OI per row
            "open_interest_asof_date": None,
            "stat_open_interest": None,
            "volume": volume,
            "volume_asof_time": event_dt,
            "option_ohlcv_volume": volume,
            "underlying_price": underlying,
            "iv": iv,
            "iv_source": GreekSource.OPTIONSDX.value if iv is not None else None,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "charm": None,  # not provided by OptionsDX; computed by greeks/ later
            "vanna": None,
            "vomma": None,
            "greek_source": GreekSource.OPTIONSDX.value if delta is not None else None,
            "greek_computation_ts": None,
            "risk_free_rate_used": None,
            "dividend_yield_used": None,
            "vendor_source": VendorSource.OPTIONSDX.value,
            "ingest_run_id": ingest_run_id,
            "schema_version": SCHEMA_VERSION,
        }


def ingest_optionsdx_file(
    path: str | Path,
    *,
    root: ContractRoot = ContractRoot.SPXW,
    ingest_run_id: str | None = None,
) -> OptionsDXIngestResult:
    """Read an OptionsDX CSV; return raw provenance + normalized pyarrow Tables.

    Note: this is the ingest-as-data-pipeline. Persistence (writing to
    raw/, normalized/) is the caller's responsibility — keeps ingest pure
    and testable.
    """
    path = Path(path)
    run_id = ingest_run_id or new_run_id()
    file_hash = file_sha256(path)
    file_size = path.stat().st_size

    records: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [_strip_brackets(c) for c in next(reader)]
        for raw_row in reader:
            if not raw_row or all(not c.strip() for c in raw_row):
                continue
            row = dict(zip(header, raw_row, strict=True))
            records.extend(_row_to_records(row, root=root, ingest_run_id=run_id))

    normalized = pa.Table.from_pylist(records, schema=NORMALIZED_SCHEMA)
    validate_normalized_table(normalized)

    raw_prov = RawProvenance(
        ingest_run_id=run_id,
        vendor_source=VendorSource.OPTIONSDX,
        vendor_file_path=str(path),
        vendor_file_sha256=file_hash,
        vendor_file_size_bytes=file_size,
        vendor_received_at=datetime.now(timezone.utc).isoformat(),
    )
    raw_table = pa.table(
        {
            "ingest_run_id": [raw_prov.ingest_run_id],
            "vendor_source": [raw_prov.vendor_source.value],
            "vendor_file_path": [raw_prov.vendor_file_path],
            "vendor_file_sha256": [raw_prov.vendor_file_sha256],
            "vendor_file_size_bytes": [raw_prov.vendor_file_size_bytes],
            "vendor_received_at": pa.array(
                [datetime.fromisoformat(raw_prov.vendor_received_at)],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "schema_version": [raw_prov.schema_version],
        },
        schema=RAW_PROVENANCE_SCHEMA,
    )

    return OptionsDXIngestResult(
        raw_provenance=raw_table,
        normalized=normalized,
        ingest_run_id=run_id,
        input_sha256=file_hash,
        input_path=str(path),
        rows_emitted=len(records),
    )
