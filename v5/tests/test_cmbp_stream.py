"""Synthetic no-network tests for the Job-51 shared CMBP stream decoder."""
from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError, replace
from datetime import date
from types import SimpleNamespace

import databento_dbn as dbn
import pytest

from v5.research import cmbp_stream as stream


EXPECTED = {101: "SPXW  260824C06000000", 202: "SPXW  260824P06000000"}
SESSION = "2026-08-24"


def row(
    instrument_id: int,
    ts_recv: int,
    *,
    ts_event: int | None = None,
    action: str = "A",
    price: int = 105,
    size: int = 1,
    bid: int | None = 100,
    ask: int | None = 110,
    flags: int = 0,
    rtype: int = stream.CMBP1_RTYPE,
) -> stream.CmbpRecord:
    return stream.CmbpRecord(
        instrument_id=instrument_id,
        ts_event=ts_recv if ts_event is None else ts_event,
        ts_recv=ts_recv,
        action=action,
        side="N",
        price=price,
        size=size,
        flags=flags,
        bid_px=bid,
        ask_px=ask,
        rtype=rtype,
    )


def decoder(
    *,
    source_kind: str = "historical",
    expected: dict[int, str] = EXPECTED,
    preload: bool = True,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
) -> stream.CmbpStreamDecoder:
    result = stream.CmbpStreamDecoder(
        session=SESSION,
        expected_mappings=expected,
        source_kind=source_kind,  # type: ignore[arg-type]
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
    )
    if preload:
        result.preload_mappings(expected)
    return result


def test_resolution_and_historical_metadata_mapping_are_exact_one_to_one() -> None:
    resolved = {
        EXPECTED[101]: ["101"],
        EXPECTED[202]: ["202"],
    }
    assert stream.expected_mappings_from_resolution(resolved) == (
        stream.ContractMapping(101, EXPECTED[101]),
        stream.ContractMapping(202, EXPECTED[202]),
    )

    metadata = {
        EXPECTED[101]: [
            {"start_date": date(2026, 8, 23), "end_date": date(2026, 8, 24), "symbol": "999"},
            {"start_date": date(2026, 8, 24), "end_date": date(2026, 8, 25), "symbol": "101"},
        ],
        EXPECTED[202]: [
            {"start_date": "2026-08-24", "end_date": "2026-08-25", "symbol": "202"}
        ],
    }
    assert stream.historical_mappings_for_session(metadata, SESSION) == (
        stream.ContractMapping(101, EXPECTED[101]),
        stream.ContractMapping(202, EXPECTED[202]),
    )

    with pytest.raises(stream.CmbpStreamError, match="one-to-one"):
        stream.expected_mappings_from_resolution({EXPECTED[101]: ["101", "102"]})
    ambiguous = {EXPECTED[101]: [
        {"start_date": SESSION, "end_date": "2026-08-25", "symbol": "101"},
        {"start_date": SESSION, "end_date": "2026-08-25", "symbol": "102"},
    ]}
    with pytest.raises(stream.CmbpStreamError, match="ambiguous"):
        stream.historical_mappings_for_session(ambiguous, SESSION)


def test_historical_and_live_records_share_identical_market_semantics() -> None:
    records = [
        row(101, 10),
        row(101, 11, action="T", price=110),
        row(202, 12),
        row(202, 13, action="T", price=100),
        row(101, 14),
        row(101, 14, ts_event=15, action="T", price=110),
    ]
    historical = decoder()
    historical_signed: list[stream.SignedTrade] = []
    historical.consume(records, on_signed_trade=historical_signed.append)

    live = decoder(source_kind="live", preload=False)
    live.accept(stream.SymbolMappingRecord(101, EXPECTED[101], 0, 100))
    live.accept(stream.SymbolMappingRecord(202, EXPECTED[202], 0, 100))
    live_signed: list[stream.SignedTrade] = []
    live.consume(iter(records), on_signed_trade=live_signed.append)

    historical_summary = historical.finalize(
        expected_record_count=len(records),
        require_all_expected_instruments=True,
    )
    live_summary = live.finalize(
        expected_record_count=len(records),
        require_all_expected_instruments=True,
    )
    assert historical_signed == live_signed
    assert [item.direction for item in historical_signed] == ["BUY", "SELL"]
    assert historical_summary.market_semantics() == live_summary.market_semantics()
    assert historical_summary.tied_prior_trades_excluded == 1
    assert historical_summary.signed_trades == 2
    assert historical_summary.explicit_connection_telemetry == "UNKNOWN"
    assert live_summary.explicit_connection_telemetry == "NONE_OBSERVED"


def test_immediate_prior_is_per_instrument_and_classifies_all_price_regions() -> None:
    subject = decoder()
    events: list[stream.SignedTrade] = []
    records = [
        row(101, 1),
        row(202, 2, bid=200, ask=220),
        row(101, 3, action="T", price=110),
        row(101, 4),
        row(101, 5, action="T", price=100),
        row(101, 6),
        row(101, 7, action="T", price=105),
        row(101, 8),
        row(101, 9, action="T", price=120),
    ]
    subject.consume(records, on_signed_trade=events.append)
    summary = subject.finalize(expected_record_count=len(records), require_all_expected_instruments=True)
    assert [(event.direction, event.classification) for event in events] == [
        ("BUY", "AT_ASK"),
        ("SELL", "AT_BID"),
    ]
    assert summary.at_ask_trades == 1
    assert summary.at_bid_trades == 1
    assert summary.inside_trades == 1
    assert summary.outside_trades == 1
    assert summary.ambiguous_trades == 0
    assert summary.classification_reconciled is True


def test_receive_tie_is_excluded_even_when_event_time_suggests_an_order() -> None:
    subject = decoder(expected={101: EXPECTED[101]})
    subject.accept(row(101, 50, ts_event=100))
    assert subject.accept(row(101, 50, ts_event=99, action="T", price=110)) is None
    summary = subject.finalize(expected_record_count=2, require_all_expected_instruments=True)
    assert summary.tied_prior_trades_excluded == 1
    assert summary.strict_prior_trades == 0
    assert summary.signed_trades == 0
    assert summary.ambiguous_trades == 1
    assert summary.global_receive_ties == 1
    assert summary.global_event_regressions == 1
    assert summary.instrument_event_regressions == 1
    assert summary.tied_receive_priors_excluded is True


def test_receive_regression_is_sticky_fatal_but_event_regression_is_diagnostic() -> None:
    fatal = decoder(expected={101: EXPECTED[101]})
    fatal.accept(row(101, 20))
    with pytest.raises(stream.CmbpStreamError) as caught:
        fatal.accept(row(101, 19))
    assert caught.value.code == "STOP_TS_RECV_REGRESSION"
    assert fatal.failed is True
    with pytest.raises(stream.CmbpStreamError) as repeated:
        fatal.accept(row(101, 21))
    assert repeated.value.code == "STOP_TS_RECV_REGRESSION"

    diagnostic = decoder(expected={101: EXPECTED[101]})
    diagnostic.accept(row(101, 30, ts_event=40))
    diagnostic.accept(row(101, 31, ts_event=39))
    summary = diagnostic.finalize(expected_record_count=2)
    assert summary.global_receive_regressions == 0
    assert summary.global_event_regressions == 1
    assert summary.instrument_event_regressions == 1


def test_disconnect_reconnect_and_gap_each_clear_prior_book_state() -> None:
    subject = decoder(expected={101: EXPECTED[101]}, source_kind="live")
    subject.accept(row(101, 1))
    subject.note_disconnect()
    subject.accept(row(101, 2, action="T", price=110))
    subject.accept(row(101, 3))
    subject.note_reconnect()
    subject.accept(row(101, 4, action="T", price=110))
    subject.accept(row(101, 5))
    subject.note_gap(5, 9)
    subject.accept(row(101, 10, action="T", price=110))
    summary = subject.finalize(expected_record_count=6, require_all_expected_instruments=True)
    assert summary.no_prior_trades == 3
    assert summary.signed_trades == 0
    assert summary.disconnect_count == 1
    assert summary.reconnect_count == 1
    assert summary.gap_count == 1
    assert summary.book_state_clear_count == 3
    assert summary.gaps_with_known_bounds == 1
    assert summary.total_known_gap_ns == 4
    assert summary.max_known_gap_ns == 4
    assert summary.explicit_connection_telemetry == "OBSERVED"


def test_bad_and_unusable_prior_books_are_counted_and_excluded_not_fatal() -> None:
    subject = decoder(expected={101: EXPECTED[101]})
    records = [
        row(101, 1, bid=None, ask=None),
        row(101, 2, action="T", price=110),
        row(101, 3, bid=stream.UNDEF_PRICE, ask=110),
        row(101, 4, action="T", price=110),
        row(101, 5, bid=100, ask=100),
        row(101, 6, action="T", price=100),
        row(101, 7, bid=111, ask=110),
        row(101, 8, action="T", price=110),
        row(101, 9, flags=stream.F_BAD_TS_RECV),
        row(101, 10, action="T", price=110),
        row(101, 11, flags=stream.F_MAYBE_BAD_BOOK),
        row(101, 12, action="T", price=110),
        row(101, 13),
        row(101, 14, action="T", price=110, flags=stream.F_BAD_TS_RECV),
        row(101, 15),
        row(101, 16, action="T", price=stream.UNDEF_PRICE),
    ]
    subject.consume(records)
    summary = subject.finalize(expected_record_count=len(records), require_all_expected_instruments=True)
    assert summary.trade_records == 8
    assert summary.strict_prior_trades == 8
    assert summary.ambiguous_trades == 8
    assert summary.missing_prior_book_excluded == 1
    assert summary.undefined_prior_book_excluded == 1
    assert summary.locked_prior_book_excluded == 1
    assert summary.crossed_prior_book_excluded == 1
    assert summary.prior_bad_ts_recv_excluded == 1
    assert summary.prior_maybe_bad_book_excluded == 1
    assert summary.trade_bad_ts_recv_excluded == 1
    assert summary.undefined_trade_price_excluded == 1
    assert summary.signed_trades == 0
    flag_counts = {item.name: item.count for item in summary.flag_counts}
    assert flag_counts["MAYBE_BAD_BOOK"] == 1
    assert flag_counts["BAD_TS_RECV"] == 2


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        (lambda item: item.accept(row(101, 1, rtype=1)), "STOP_WRONG_RTYPE"),
        (lambda item: item.accept(row(101, 9)), "STOP_OUT_OF_WINDOW"),
    ],
)
def test_wrong_rtype_and_out_of_window_records_fail_closed(operation: object, code: str) -> None:
    subject = decoder(
        expected={101: EXPECTED[101]},
        window_start_ns=10,
        window_end_ns=20,
    )
    with pytest.raises(stream.CmbpStreamError) as caught:
        operation(subject)  # type: ignore[operator]
    assert caught.value.code == code

    receive_clock_subject = decoder(
        expected={101: EXPECTED[101]},
        window_start_ns=10,
        window_end_ns=20,
    )
    receive_clock_subject.accept(row(101, 10, ts_event=1))
    assert receive_clock_subject.finalize(expected_record_count=1).record_count_reconciled is True


def test_mapping_mismatch_unmapped_data_and_incomplete_terminal_set_fail_closed() -> None:
    mismatch = decoder(source_kind="live", preload=False)
    with pytest.raises(stream.CmbpStreamError) as caught:
        mismatch.accept(stream.SymbolMappingRecord(101, EXPECTED[202]))
    assert caught.value.code == "STOP_CONTRACT_MAPPING"
    assert mismatch.failed is True
    with pytest.raises(stream.CmbpStreamError) as repeated:
        mismatch.accept(stream.SymbolMappingRecord(101, EXPECTED[101]))
    assert repeated.value.code == "STOP_CONTRACT_MAPPING"

    unmapped = decoder(source_kind="live", preload=False)
    with pytest.raises(stream.CmbpStreamError) as caught:
        unmapped.accept(row(101, 1))
    assert caught.value.code == "STOP_CONTRACT_MAPPING"

    incomplete = decoder(source_kind="live", preload=False)
    incomplete.accept(stream.SymbolMappingRecord(101, EXPECTED[101]))
    incomplete.accept(row(101, 1))
    with pytest.raises(stream.CmbpStreamError) as caught:
        incomplete.finalize(expected_record_count=1)
    assert caught.value.code == "STOP_CONTRACT_MAPPING"


def test_sdk_shaped_mapping_and_cmbp_records_use_levels_zero_without_pandas() -> None:
    subject = decoder(source_kind="live", expected={101: EXPECTED[101]}, preload=False)
    mapping = dbn.SymbolMappingMsg(
        publisher_id=1,
        instrument_id=101,
        ts_event=1,
        stype_in=dbn.SType.RAW_SYMBOL,
        stype_in_symbol=EXPECTED[101],
        stype_out=dbn.SType.INSTRUMENT_ID,
        stype_out_symbol="101",
        start_ts=0,
        end_ts=100,
    )
    subject.accept(mapping)
    quote = dbn.CMBP1Msg(
        rtype=dbn.RType.CMBP_1,
        publisher_id=1,
        instrument_id=101,
        ts_event=10,
        price=105,
        size=1,
        action=dbn.Action.ADD,
        side=dbn.Side.NONE,
        ts_recv=10,
        flags=0,
        levels=dbn.ConsolidatedBidAskPair(bid_px=100, ask_px=110, bid_sz=1, ask_sz=1),
    )
    trade = dbn.CMBP1Msg(
        rtype=dbn.RType.CMBP_1,
        publisher_id=1,
        instrument_id=101,
        ts_event=11,
        price=110,
        size=2,
        action=dbn.Action.TRADE,
        side=dbn.Side.NONE,
        ts_recv=11,
        flags=0,
        levels=dbn.ConsolidatedBidAskPair(bid_px=100, ask_px=110, bid_sz=1, ask_sz=1),
    )
    subject.accept(quote)
    signed = subject.accept(trade)
    assert signed is not None
    assert signed.direction == "BUY"
    assert signed.prior_bid_px == 100
    assert signed.prior_ask_px == 110
    summary = subject.finalize(expected_record_count=2, require_all_expected_instruments=True)
    assert summary.mapping_reconciled is True
    assert summary.at_ask_trades == 1

    source = inspect.getsource(stream)
    for forbidden in ("import pandas", ".to_df(", "read_parquet("):
        assert forbidden not in source


def test_sdk_shaped_undefined_participant_counts_do_not_break_book_decode() -> None:
    sdk_shaped_type = type("CMBP1Msg", (), {})
    sdk_shaped = sdk_shaped_type()
    sdk_shaped.rtype = stream.CMBP1_RTYPE
    sdk_shaped.publisher_id = 1
    sdk_shaped.instrument_id = 101
    sdk_shaped.ts_event = 10
    sdk_shaped.ts_recv = 10
    sdk_shaped.action = "A"
    sdk_shaped.side = "N"
    sdk_shaped.price = 105
    sdk_shaped.size = 1
    sdk_shaped.flags = 0
    sdk_shaped.levels = [
        SimpleNamespace(bid_px=100, ask_px=110, bid_sz=1, ask_sz=1, bid_pb=None, ask_pb=None)
    ]
    subject = decoder(expected={101: EXPECTED[101]})
    subject.accept(sdk_shaped)
    assert subject.finalize(expected_record_count=1).cmbp1_records == 1


def test_system_controls_are_counted_and_error_records_are_fatal() -> None:
    subject = decoder(expected={101: EXPECTED[101]}, source_kind="live")
    subject.accept(stream.SystemRecord("HEARTBEAT", heartbeat=True))
    subject.accept(stream.SystemRecord("REPLAY"))
    subject.accept(row(101, 1))
    summary = subject.finalize(expected_record_count=1)
    assert summary.system_records == 2
    assert summary.heartbeat_records == 1
    assert {item.name: item.count for item in summary.system_code_counts} == {
        "HEARTBEAT": 1,
        "REPLAY": 1,
    }

    fatal = decoder(expected={101: EXPECTED[101]}, source_kind="live")
    with pytest.raises(stream.CmbpStreamError) as caught:
        fatal.accept(stream.ErrorRecord("AUTH"))
    assert caught.value.code == "STOP_STREAM_ERROR"


def test_record_count_reconciliation_summary_is_frozen_and_canonical_json_safe() -> None:
    subject = decoder(expected={101: EXPECTED[101]})
    subject.accept(row(101, 1))
    with pytest.raises(stream.CmbpStreamError) as caught:
        subject.finalize(expected_record_count=2)
    assert caught.value.code == "STOP_RECORD_COUNT_RECONCILIATION"

    complete = decoder(expected={101: EXPECTED[101]})
    complete.accept(row(101, 1))
    summary = complete.finalize(expected_record_count=1, require_all_expected_instruments=True)
    assert summary.record_count_reconciled is True
    assert summary.all_expected_instruments_seen is True
    assert json.loads(json.dumps(summary.to_dict(), sort_keys=True, allow_nan=False))["cmbp1_records"] == 1
    with pytest.raises(FrozenInstanceError):
        summary.cmbp1_records = 2  # type: ignore[misc]
    with pytest.raises(stream.CmbpStreamError) as caught:
        complete.accept(row(101, 2))
    assert caught.value.code == "STOP_DECODER_FINALIZED"


def test_missing_expected_instrument_and_preload_drift_are_terminal_failures() -> None:
    missing = decoder()
    missing.accept(row(101, 1))
    with pytest.raises(stream.CmbpStreamError) as caught:
        missing.finalize(expected_record_count=1, require_all_expected_instruments=True)
    assert caught.value.code == "STOP_CONTRACT_MAPPING"

    subject = decoder(preload=False)
    with pytest.raises(stream.CmbpStreamError) as caught:
        subject.preload_mappings({101: EXPECTED[101]})
    assert caught.value.code == "STOP_CONTRACT_MAPPING"


def test_inputs_and_emitted_events_are_frozen_values() -> None:
    record = row(101, 1)
    with pytest.raises(FrozenInstanceError):
        record.ts_recv = 2  # type: ignore[misc]
    assert replace(record, ts_recv=2).ts_recv == 2

    subject = decoder(expected={101: EXPECTED[101]})
    subject.accept(record)
    event = subject.accept(row(101, 2, action="T", price=110))
    assert event is not None
    with pytest.raises(FrozenInstanceError):
        event.direction = "SELL"  # type: ignore[misc]
    assert json.dumps(event.to_dict(), allow_nan=False)


def test_malformed_public_ingress_failures_are_sticky() -> None:
    malformed_type = type("CMBP1Msg", (), {})
    malformed = malformed_type()
    malformed.rtype = stream.CMBP1_RTYPE
    malformed.publisher_id = 1
    malformed.instrument_id = 101
    malformed.ts_event = 1
    malformed.ts_recv = 1.5
    malformed.action = "A"
    malformed.side = "N"
    malformed.price = 105
    malformed.size = 1
    malformed.flags = 0
    malformed.levels = [
        SimpleNamespace(bid_px=100, ask_px=110, bid_sz=1, ask_sz=1, bid_pb=1, ask_pb=1)
    ]
    parsed = decoder(expected={101: EXPECTED[101]})
    with pytest.raises(stream.CmbpStreamError) as caught:
        parsed.accept(malformed)
    assert caught.value.code == "STOP_MALFORMED_RECORD"
    assert parsed.failed is True

    preloaded = decoder(expected={101: EXPECTED[101]}, preload=False)
    with pytest.raises(stream.CmbpStreamError):
        preloaded.preload_mappings([(1.5, EXPECTED[101])])  # type: ignore[list-item]
    assert preloaded.failed is True

    gapped = decoder(expected={101: EXPECTED[101]}, source_kind="live")
    with pytest.raises(stream.CmbpStreamError):
        gapped.note_gap(1.5, 2)  # type: ignore[arg-type]
    assert gapped.failed is True

    finalized = decoder(expected={101: EXPECTED[101]})
    with pytest.raises(stream.CmbpStreamError):
        finalized.finalize(expected_record_count=1.5)  # type: ignore[arg-type]
    assert finalized.failed is True
