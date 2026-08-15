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
    select_executable_entry_quote,
    select_official_spx,
    select_option_feature_quote,
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


# --- reconnects_and_gaps -----------------------------------------------------
# Four boundary conditions that legitimately change what input has arrived. None
# waives parity: each must fail closed rather than emit a decision on ambiguous
# or absent data. Settled as the reconnects_and_gaps axis 2026-08-12.


def _entry_clock(boundary: int):
    return make_clock(
        feature_boundary_ns=boundary, emission_lag_ms=2336, order_latency_ms=250
    )


def test_a_reconnect_forces_a_fresh_warm_up_before_anything_may_emit() -> None:
    readiness = StreamReadiness()
    readiness.load_current_session_definitions()

    assert readiness.observe_completed_interval() is False  # first is warm-up
    assert readiness.observe_completed_interval() is True

    readiness.reconnect()
    assert readiness.reconnect_count == 1
    assert readiness.observe_completed_interval() is False, (
        "the first interval after a reconnect must be warm-up: the stream may "
        "have missed updates while disconnected"
    )
    assert readiness.observe_completed_interval() is True


def test_an_interval_cannot_be_observed_before_definitions_load() -> None:
    with pytest.raises(TrainingTwinError, match="definitions are not loaded"):
        StreamReadiness().observe_completed_interval()


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("duplicate", "expected one exact CBBO-1m row"),
        ("gap", "expected one exact CBBO-1m row"),
        ("correction", "expected one exact CBBO-1m row"),
    ],
)
def test_the_feature_path_refuses_a_duplicate_gap_or_correction(
    mutation, expected
) -> None:
    """The exact-boundary rule makes all three the same refusal, by design.

    A completed minute has exactly one CBBO-1m row. Zero means the interval did
    not arrive; two means a duplicate or a correction, and the selector cannot
    know which of them the live system would have acted on.
    """

    boundary, symbol, _, cbbo_1m, _ = _source_frames()
    if mutation == "duplicate":
        frame = pd.concat([cbbo_1m, cbbo_1m], ignore_index=True)
    elif mutation == "gap":
        frame = cbbo_1m.iloc[0:0]
    else:
        corrected = cbbo_1m.copy()
        corrected.loc[0, "bid_px_00"] = 5.45
        frame = pd.concat([cbbo_1m, corrected], ignore_index=True)

    with pytest.raises(TrainingTwinError, match=expected):
        select_option_feature_quote(
            frame, stable_contract_id=symbol, clock=_entry_clock(boundary)
        )


def test_the_entry_path_refuses_a_correction_it_cannot_disambiguate() -> None:
    """Repaired 2026-08-12; this previously emitted the value being corrected.

    The windowed entry selector takes the newest quote via ``max``, which returns
    the *first* maximal element. Two rows sharing the newest timestamp therefore
    resolved by delivery order, so a correction was discarded in favour of the
    stale value it corrected. This row sets the fill price, so the failure was a
    wrong fill rather than a wrong feature.
    """

    boundary, symbol, _, _, cbbo_1s = _source_frames()
    clock = _entry_clock(boundary)

    corrected = cbbo_1s.copy()
    corrected.loc[1, "bid_px_00"] = 9.99
    with pytest.raises(TrainingTwinError, match="ambiguous entry quote"):
        select_executable_entry_quote(
            pd.concat([cbbo_1s, corrected], ignore_index=True),
            stable_contract_id=symbol,
            clock=clock,
        )


def test_the_entry_path_still_accepts_a_byte_identical_duplicate() -> None:
    """A true duplicate is not ambiguous, so refusing it would fail closed on
    ordinary vendor behaviour rather than on a real divergence."""

    boundary, symbol, _, _, cbbo_1s = _source_frames()
    row, _receipt = select_executable_entry_quote(
        pd.concat([cbbo_1s, cbbo_1s], ignore_index=True),
        stable_contract_id=symbol,
        clock=_entry_clock(boundary),
    )
    assert float(row["bid_px_00"]) == pytest.approx(5.42)


def test_the_entry_path_refuses_a_sequence_gap() -> None:
    boundary, symbol, _, _, cbbo_1s = _source_frames()
    with pytest.raises(TrainingTwinError, match="no causal option quote"):
        select_executable_entry_quote(
            cbbo_1s.iloc[0:0],
            stable_contract_id=symbol,
            clock=_entry_clock(boundary),
        )


# --- reconnects_and_gaps: the four boundaries, each failing closed ------------
# Settles the divergence axis of that name. Reconnects, duplicates, corrections
# and sequence gaps legitimately change what input has arrived. None of them
# waives parity: the live path must refuse to emit rather than guess.


def _quote_frame(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def _boundary_clock():
    from v5.research import training_twin as tt

    return tt.make_clock(
        feature_boundary_ns=1_800_000_000_000_000_000,
        emission_lag_ms=2_336,
        order_latency_ms=100,
    )


def _row(ts_ns, *, symbol="SPXW  260803C05000000", bid=1.0, ask=1.2, received=None):
    return {
        "symbol": symbol,
        "ts_recv": pd.Timestamp(ts_ns, unit="ns", tz="UTC").isoformat(),
        "bid_px_00": bid,
        "ask_px_00": ask,
        "received_at_ns": received if received is not None else ts_ns,
    }


def test_a_reconnect_discards_the_first_interval_rather_than_emitting() -> None:
    from v5.research import training_twin as tt

    readiness = tt.StreamReadiness()
    readiness.load_current_session_definitions()
    assert readiness.observe_completed_interval() is False  # warm-up
    assert readiness.observe_completed_interval() is True

    readiness.reconnect()
    assert readiness.reconnect_count == 1
    assert readiness.observe_completed_interval() is False, (
        "the first interval after a reconnect is warm-up; emitting on it would "
        "use a stream whose completeness has not been re-established"
    )
    assert readiness.observe_completed_interval() is True


def test_emitting_before_definitions_are_loaded_is_refused() -> None:
    from v5.research import training_twin as tt

    readiness = tt.StreamReadiness()
    with pytest.raises(tt.TrainingTwinError, match="definitions are not loaded"):
        readiness.observe_completed_interval()


def test_a_duplicate_boundary_row_is_refused_rather_than_deduplicated() -> None:
    """Two rows claiming the same completed interval is ambiguity, not a choice."""

    from v5.research import training_twin as tt

    clock = _boundary_clock()
    frame = _quote_frame([_row(clock.feature_boundary_ns), _row(clock.feature_boundary_ns)])
    with pytest.raises(tt.TrainingTwinError, match="expected one exact"):
        tt.select_option_feature_quote(
            frame, stable_contract_id="SPXW  260803C05000000", clock=clock
        )


def test_a_sequence_gap_at_the_boundary_is_refused_rather_than_interpolated() -> None:
    """A missing completed interval must not be filled from a neighbouring one."""

    from v5.research import training_twin as tt

    clock = _boundary_clock()
    frame = _quote_frame(
        [
            _row(clock.feature_boundary_ns - tt.MINUTE_NS),
            _row(clock.feature_boundary_ns + tt.MINUTE_NS),
        ]
    )
    with pytest.raises(tt.TrainingTwinError, match="expected one exact"):
        tt.select_option_feature_quote(
            frame, stable_contract_id="SPXW  260803C05000000", clock=clock
        )


def test_a_correction_arriving_after_the_decision_cannot_be_consumed() -> None:
    """A row received after the emission instant is not evidence the bot had."""

    from v5.research import training_twin as tt

    clock = _boundary_clock()
    late = _row(
        clock.feature_boundary_ns,
        received=clock.decision_emission_ns + 1_000_000_000,
    )
    with pytest.raises(tt.TrainingTwinError, match="expected one exact"):
        tt.select_option_feature_quote(
            frame := _quote_frame([late]),
            stable_contract_id="SPXW  260803C05000000",
            clock=clock,
        )
    # ...and the same row received in time is accepted, so the refusal above is
    # about arrival, not about the row being malformed.
    ok = _row(clock.feature_boundary_ns, received=clock.feature_boundary_ns)
    row, receipt = tt.select_option_feature_quote(
        _quote_frame([ok]), stable_contract_id="SPXW  260803C05000000", clock=clock
    )
    assert receipt.age_at_consumption_ns >= 0
