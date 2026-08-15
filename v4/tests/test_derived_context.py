"""Tests for option-derived SPX/volatility context bars."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pyarrow as pa

from v4.ingest.derived_context import (
    attach_underlying_from_context,
    derive_context_bars_from_spxw_quotes,
)
from v4.schema.normalized import NORMALIZED_SCHEMA
from v4.schema.types import SCHEMA_VERSION, VendorSource


def _row(ts: datetime, strike: Decimal, right: str, bid: float, ask: float) -> dict:
    mid = (bid + ask) / 2.0
    return {
        "event_time": ts,
        "receive_time": ts,
        "timestamp_source": "test",
        "contract_id": f"SPXW-20260102-{strike:09.3f}-{right}",
        "raw_symbol": f"SPXW  260102{right}{int(strike * 1000):08d}",
        "instrument_id": 1,
        "root": "SPXW",
        "expiry": date(2026, 1, 2),
        "strike": strike,
        "right": right,
        "settlement_style": "PM",
        "settlement_time_utc": datetime(2026, 1, 2, 21, 0, tzinfo=timezone.utc),
        "min_price_increment": 0.05,
        "contract_multiplier": 100,
        "bid": bid,
        "ask": ask,
        "bid_size": 10,
        "ask_size": 10,
        "mid": mid,
        "last_trade": mid,
        "last_trade_size": 1,
        "quote_time": ts,
        "last_trade_time": None,
        "quote_age_ms": 0,
        "quote_gap_seconds": None,
        "open_interest": None,
        "open_interest_asof_date": None,
        "stat_open_interest": None,
        "volume": None,
        "volume_asof_time": None,
        "option_ohlcv_volume": None,
        "underlying_price": None,
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
        "ingest_run_id": "test",
        "schema_version": SCHEMA_VERSION,
    }


def test_derive_context_bars_from_put_call_parity() -> None:
    ts = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    table = pa.Table.from_pylist(
        [
            _row(ts, Decimal("6500.000000"), "C", 9.90, 10.10),
            _row(ts, Decimal("6500.000000"), "P", 9.80, 10.00),
            _row(ts, Decimal("6510.000000"), "C", 5.10, 5.30),
            _row(ts, Decimal("6510.000000"), "P", 14.80, 15.00),
        ],
        schema=NORMALIZED_SCHEMA,
    )

    spx, vol = derive_context_bars_from_spxw_quotes(table)
    enriched = attach_underlying_from_context(table, spx)

    assert len(spx) == 1
    assert len(vol) == 1
    assert spx.iloc[0]["symbol"] == "SPX"
    assert bool(spx.iloc[0]["is_derived"]) is True
    assert 6490 < spx.iloc[0]["close"] < 6510
    assert vol.iloc[0]["close"] > 0
    assert all(x is not None for x in enriched["underlying_price"].to_pylist())
