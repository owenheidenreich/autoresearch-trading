"""Tests for Databento OPRA SPXW 0DTE ingest."""
from __future__ import annotations

from datetime import date

import pandas as pd

from v4.ingest.databento_opra import filter_0dte_definitions, normalize_spxw_0dte_day


SESSION = date(2026, 1, 2)
RAW_C = "SPXW  260102C06550000"


def test_filter_0dte_definitions_keeps_only_spxw_pm_five_point_strikes() -> None:
    defs = pd.DataFrame(
        {
            "raw_symbol": [
                RAW_C,
                "SPX   260102C06550000",
                "SPXW  260105C06550000",
                "SPXW  260102C06552500",
            ],
            "instrument_id": [1, 2, 3, 4],
        }
    )

    filtered = filter_0dte_definitions(defs, SESSION)

    assert filtered["raw_symbol"].tolist() == [RAW_C]
    assert filtered.iloc[0]["root"] == "SPXW"
    assert filtered.iloc[0]["settlement_style"] == "PM"
    assert filtered.iloc[0]["strike"] % 5 == 0


def test_normalize_spxw_0dte_day_uses_ohlcv_volume_not_cbbo_last_size() -> None:
    defs = filter_0dte_definitions(
        pd.DataFrame({"raw_symbol": [RAW_C], "instrument_id": [1001]}), SESSION
    )
    cbbo = pd.DataFrame(
        {
            "symbol": [RAW_C],
            "ts_recv": [pd.Timestamp("2026-01-02T14:31:00Z")],
            "ts_event": [pd.Timestamp("2026-01-02T14:30:59Z")],
            "bid_px_00": [2.95],
            "ask_px_00": [3.10],
            "bid_sz_00": [12],
            "ask_sz_00": [8],
            "price": [3.00],
            "size": [7],
        }
    )
    ohlcv = pd.DataFrame(
        {
            "symbol": [RAW_C],
            "event_time": [pd.Timestamp("2026-01-02T14:30:00Z")],
            "volume": [123],
        }
    )
    stats = pd.DataFrame(
        {
            "symbol": [RAW_C, RAW_C],
            "stat_type": [9, 1],
            "quantity": [44, 999],
        }
    )
    index = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2026-01-02T14:31:00Z")],
            "symbol": ["SPX"],
            "close": [6550.0],
        }
    )

    table = normalize_spxw_0dte_day(
        defs, cbbo, ohlcv_1m=ohlcv, statistics=stats, index_bars=index
    )
    row = table.to_pydict()

    assert row["root"] == ["SPXW"]
    assert row["settlement_style"] == ["PM"]
    assert row["bid_size"] == [12]
    assert row["ask_size"] == [8]
    assert row["last_trade_size"] == [7]
    assert row["option_ohlcv_volume"] == [123]
    assert row["volume"] == [123]
    assert row["stat_open_interest"] == [44]
    assert row["open_interest"] == [44]
    assert row["underlying_price"] == [6550.0]


def test_normalize_spxw_0dte_day_handles_microsecond_index_timestamps() -> None:
    defs = filter_0dte_definitions(
        pd.DataFrame({"raw_symbol": [RAW_C], "instrument_id": [1001]}), SESSION
    )
    cbbo = pd.DataFrame(
        {
            "symbol": [RAW_C, RAW_C],
            "ts_recv": [
                pd.Timestamp("2026-01-02T14:31:00Z"),
                pd.Timestamp("2026-01-02T14:32:00Z"),
            ],
            "bid_px_00": [2.95, 3.10],
            "ask_px_00": [3.10, 3.25],
        }
    )
    index = pd.DataFrame(
        {
            "event_time": pd.Series(
                [
                    pd.Timestamp("2026-01-02T14:31:00Z"),
                    pd.Timestamp("2026-01-02T14:32:00Z"),
                ],
                dtype="datetime64[us, UTC]",
            ),
            "symbol": ["SPX", "SPX"],
            "close": [6550.0, 6560.0],
        }
    )

    table = normalize_spxw_0dte_day(defs, cbbo, index_bars=index)
    row = table.to_pydict()

    assert row["underlying_price"] == [6550.0, 6560.0]


def test_normalize_spxw_0dte_day_does_not_promote_cbbo_size_to_volume() -> None:
    defs = filter_0dte_definitions(
        pd.DataFrame({"raw_symbol": [RAW_C], "instrument_id": [1001]}), SESSION
    )
    cbbo = pd.DataFrame(
        {
            "symbol": [RAW_C],
            "ts_recv": [pd.Timestamp("2026-01-02T14:31:00Z")],
            "bid_px_00": [2.95],
            "ask_px_00": [3.10],
            "price": [3.00],
            "size": [7],
        }
    )

    table = normalize_spxw_0dte_day(defs, cbbo)
    row = table.to_pydict()

    assert row["last_trade_size"] == [7]
    assert row["option_ohlcv_volume"] == [None]
    assert row["volume"] == [None]
