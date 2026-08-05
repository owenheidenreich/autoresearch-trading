from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from v5.research.training_twin import (
    MINUTE_NS,
    StreamReadiness,
    TrainingTwinError,
    compare_paired_frames,
    compile_entry_source_receipt,
    current_session_definition_universe,
    make_clock,
    select_official_spx,
    write_parity_receipt,
)


def _ns(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").value)


def _source_frames(receipt_delay_ms: int = 100):
    boundary = _ns("2026-08-03 16:08:00")
    symbol = "SPXW  260803C07585000"
    spx = pd.DataFrame(
        {
            "event_time": [
                "2026-08-03 16:07:00+00:00",
                "2026-08-03 16:08:00+00:00",
            ],
            "close": [7582.8, 7583.2],
            "received_at_ns": [
                boundary + receipt_delay_ms * 1_000_000,
                boundary + MINUTE_NS + receipt_delay_ms * 1_000_000,
            ],
        }
    )
    cbbo_1m = pd.DataFrame(
        {
            "ts_recv": ["2026-08-03 16:08:00+00:00"],
            "symbol": [symbol],
            "bid_px_00": [5.4],
            "ask_px_00": [5.6],
            "received_at_ns": [boundary + receipt_delay_ms * 1_000_000],
        }
    )
    cbbo_1s = pd.DataFrame(
        {
            "ts_recv": [
                "2026-08-03 16:08:01+00:00",
                "2026-08-03 16:08:02+00:00",
            ],
            "symbol": [symbol, symbol],
            "bid_px_00": [5.41, 5.42],
            "ask_px_00": [5.61, 5.62],
            "received_at_ns": [
                boundary + 1_100_000_000,
                boundary + 2_100_000_000,
            ],
        }
    )
    return boundary, symbol, spx, cbbo_1m, cbbo_1s


def test_compiled_receipt_uses_previous_completed_spx_minute() -> None:
    boundary, symbol, spx, cbbo_1m, cbbo_1s = _source_frames()
    clock = make_clock(
        feature_boundary_ns=boundary,
        emission_lag_ms=2336,
        order_latency_ms=100,
    )

    receipt = compile_entry_source_receipt(
        stable_contract_id=symbol,
        clock=clock,
        spx_frame=spx,
        cbbo_1m_frame=cbbo_1m,
        cbbo_1s_frame=cbbo_1s,
    )

    assert receipt.official_spx.source_timestamp_ns == boundary - MINUTE_NS
    assert receipt.option_feature_quote.source_timestamp_ns == boundary
    assert receipt.executable_entry_quote.source_timestamp_ns == boundary + 2_000_000_000
    unsigned = receipt.to_dict()
    digest = unsigned.pop("receipt_sha256")
    expected = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == expected


def test_next_spx_bar_is_never_used_at_current_boundary() -> None:
    boundary, _, spx, _, _ = _source_frames()
    clock = make_clock(
        feature_boundary_ns=boundary,
        emission_lag_ms=2336,
        order_latency_ms=0,
    )

    row, _ = select_official_spx(spx, clock=clock)

    assert row["close"] == 7582.8


def test_late_or_crossed_quotes_fail_closed() -> None:
    boundary, symbol, spx, cbbo_1m, cbbo_1s = _source_frames(receipt_delay_ms=2500)
    clock = make_clock(
        feature_boundary_ns=boundary,
        emission_lag_ms=2336,
        order_latency_ms=100,
    )
    with pytest.raises(TrainingTwinError, match="completed SPX bar"):
        compile_entry_source_receipt(
            stable_contract_id=symbol,
            clock=clock,
            spx_frame=spx,
            cbbo_1m_frame=cbbo_1m,
            cbbo_1s_frame=cbbo_1s,
        )

    _, symbol, spx, cbbo_1m, cbbo_1s = _source_frames()
    cbbo_1m.loc[0, "bid_px_00"] = 5.7
    with pytest.raises(TrainingTwinError, match="locked, or crossed"):
        compile_entry_source_receipt(
            stable_contract_id=symbol,
            clock=clock,
            spx_frame=spx,
            cbbo_1m_frame=cbbo_1m,
            cbbo_1s_frame=cbbo_1s,
        )


def test_current_session_definitions_apply_modify_and_delete() -> None:
    as_of = _ns("2026-08-03 13:31:00")
    call = "SPXW  260803C07585000"
    put = "SPXW  260803P07585000"
    frame = pd.DataFrame(
        {
            "ts_recv": [
                "2026-08-03 13:29:00+00:00",
                "2026-08-03 13:30:00+00:00",
                "2026-08-03 13:29:00+00:00",
                "2026-08-03 13:30:30+00:00",
            ],
            "raw_symbol": [call, call, put, put],
            "expiration": ["2026-08-03"] * 4,
            "asset": ["SPXW"] * 4,
            "instrument_id": [1, 9, 2, 2],
            "instrument_class": ["C", "C", "P", "P"],
            "strike_price": [7585.0] * 4,
            "security_update_action": ["A", "M", "A", "D"],
        }
    )

    universe = current_session_definition_universe(
        frame, session_date="2026-08-03", as_of_ns=as_of
    )

    assert list(universe) == [call]
    assert universe[call].current_instrument_id == 9


def test_first_interval_and_first_after_reconnect_are_warmup_only() -> None:
    state = StreamReadiness()
    with pytest.raises(TrainingTwinError, match="definitions"):
        state.observe_completed_interval()
    state.load_current_session_definitions()
    assert state.observe_completed_interval() is False
    assert state.observe_completed_interval() is True
    state.reconnect()
    assert state.observe_completed_interval() is False
    assert state.observe_completed_interval() is True


def test_paired_same_session_replay_checks_values_scores_and_decisions(tmp_path) -> None:
    historical = pd.DataFrame(
        {
            "minute": [1, 2, 3],
            "symbol": ["a", "a", "a"],
            "bid": [1.0, 1.1, 1.2],
            "score": [0.2, 0.7, 0.8],
            "decision": ["ABSTAIN", "CALL", "CALL"],
        }
    )
    live = historical.iloc[1:].copy()
    receipt = compare_paired_frames(
        historical,
        live,
        key_columns=("minute", "symbol"),
        value_columns=("bid", "score", "decision"),
    ).assert_passed()
    assert receipt.matched_rows == 2
    assert receipt.missing_live_keys == 1

    path = tmp_path / "parity.json"
    write_parity_receipt(receipt, path)
    assert json.loads(path.read_text())["status"] == "PASS"
    with pytest.raises(TrainingTwinError, match="refusing to overwrite"):
        write_parity_receipt(receipt, path)

    live.loc[live.index[-1], "decision"] = "PUT"
    failed = compare_paired_frames(
        historical,
        live,
        key_columns=("minute", "symbol"),
        value_columns=("bid", "score", "decision"),
    )
    with pytest.raises(TrainingTwinError, match="mismatched_cells=1"):
        failed.assert_passed()


def test_paired_replay_requires_every_live_key_and_honors_float_tolerance() -> None:
    historical = pd.DataFrame({"key": [1], "value": [1.0]})
    live = pd.DataFrame({"key": [1, 2], "value": [1.00001, 2.0]})
    failed = compare_paired_frames(
        historical,
        live,
        key_columns=("key",),
        value_columns=("value",),
        tolerances={"value": 0.001},
    )
    assert failed.status == "FAIL"
    assert failed.missing_historical_keys == 1


def test_paired_replay_rejects_empty_and_dtype_drift() -> None:
    empty = pd.DataFrame({"key": pd.Series(dtype="int64"), "value": pd.Series(dtype="float64")})
    assert compare_paired_frames(
        empty,
        empty,
        key_columns=("key",),
        value_columns=("value",),
    ).status == "FAIL"

    historical = pd.DataFrame({"key": [1], "value": pd.Series([1], dtype="int64")})
    live = pd.DataFrame({"key": [1], "value": pd.Series([1.0], dtype="float64")})
    receipt = compare_paired_frames(
        historical,
        live,
        key_columns=("key",),
        value_columns=("value",),
    )
    assert receipt.status == "FAIL"
    assert receipt.dtype_mismatches == 1
