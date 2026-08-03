from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from v4.research.autoresearch_v2.live_opra_training_twin import (
    LiveTwinClockError,
    MINUTE_NS,
    compile_source_receipt,
    current_session_definition_universe,
    make_clock,
    select_official_spx,
)


def _ns(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").value)


def _frames(*, receipt_delay_ms: int = 100):
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
            "instrument_id": [111],
        }
    )
    cbbo_1s = pd.DataFrame(
        {
            "ts_recv": [
                "2026-08-03 16:08:00+00:00",
                "2026-08-03 16:08:01+00:00",
            ],
            "symbol": [symbol, symbol],
            "bid_px_00": [5.4, 5.3],
            "ask_px_00": [5.6, 5.5],
            "received_at_ns": [
                boundary + receipt_delay_ms * 1_000_000,
                boundary + 1_000_000_000 + receipt_delay_ms * 1_000_000,
            ],
            "instrument_id": [222, 222],
        }
    )
    return boundary, symbol, spx, cbbo_1m, cbbo_1s


def test_compiled_regimen_uses_aligned_closed_minute_and_fresh_entry_quote() -> None:
    boundary, symbol, spx, cbbo_1m, cbbo_1s = _frames()
    clock = make_clock(
        feature_boundary_ns=boundary,
        emission_lag_ms=500,
        order_latency_ms=600,
    )
    receipt = compile_source_receipt(
        stable_contract_id=symbol,
        clock=clock,
        spx_frame=spx,
        cbbo_1m_frame=cbbo_1m,
        cbbo_1s_frame=cbbo_1s,
    )
    assert receipt.official_spx.represented_interval_start_ns == boundary - MINUTE_NS
    assert receipt.official_spx.represented_interval_end_ns == boundary
    assert receipt.option_feature_quote.source_timestamp_ns == boundary
    assert receipt.executable_entry_quote.source_timestamp_ns == boundary + 1_000_000_000
    assert receipt.clock.label_deadline_ns == receipt.clock.entry_arrival_ns + 25 * MINUTE_NS
    assert len(receipt.receipt_sha256) == 64


def test_next_theta_bar_is_never_used_at_current_boundary() -> None:
    boundary, _, spx, _, _ = _frames()
    clock = make_clock(
        feature_boundary_ns=boundary,
        emission_lag_ms=500,
        order_latency_ms=0,
    )
    row, receipt = select_official_spx(spx, clock=clock)
    assert float(row["close"]) == 7582.8
    assert receipt.source_timestamp_ns == boundary - MINUTE_NS


def test_source_arriving_after_frozen_emission_fails_closed() -> None:
    boundary, symbol, spx, cbbo_1m, cbbo_1s = _frames(receipt_delay_ms=700)
    clock = make_clock(
        feature_boundary_ns=boundary,
        emission_lag_ms=500,
        order_latency_ms=600,
    )
    with pytest.raises(LiveTwinClockError, match="completed SPX bar"):
        compile_source_receipt(
            stable_contract_id=symbol,
            clock=clock,
            spx_frame=spx,
            cbbo_1m_frame=cbbo_1m,
            cbbo_1s_frame=cbbo_1s,
        )


def test_stable_raw_symbol_not_daily_instrument_id_binds_contract() -> None:
    boundary, symbol, spx, cbbo_1m, cbbo_1s = _frames()
    cbbo_1m["instrument_id"] = 333333
    cbbo_1s["instrument_id"] = 444444
    clock = make_clock(
        feature_boundary_ns=boundary,
        emission_lag_ms=500,
        order_latency_ms=600,
    )
    receipt = compile_source_receipt(
        stable_contract_id=symbol,
        clock=clock,
        spx_frame=spx,
        cbbo_1m_frame=cbbo_1m,
        cbbo_1s_frame=cbbo_1s,
    )
    assert receipt.stable_contract_id == symbol


def test_crossed_time_theta_row_is_rejected_not_delayed_into_parity() -> None:
    boundary, _, spx, _, _ = _frames()
    future_only = spx.iloc[[1]].copy()
    future_only["received_at_ns"] = boundary + MINUTE_NS + 100_000_000
    clock = make_clock(
        feature_boundary_ns=boundary,
        emission_lag_ms=10_000,
        order_latency_ms=0,
    )
    with pytest.raises(LiveTwinClockError, match="completed SPX bar"):
        select_official_spx(future_only, clock=clock)


def test_current_definition_universe_applies_add_modify_delete_causally() -> None:
    frame = pd.DataFrame(
        {
            "ts_recv": [
                "2026-08-03 12:00:00+00:00",
                "2026-08-03 12:01:00+00:00",
                "2026-08-03 12:02:00+00:00",
                "2026-08-03 12:03:00+00:00",
            ],
            "raw_symbol": [
                "SPXW  260803C07585000",
                "SPXW  260803C07585000",
                "SPXW  260803P07585000",
                "SPXW  260803P07585000",
            ],
            "expiration": ["2026-08-03"] * 4,
            "asset": ["SPXW"] * 4,
            "instrument_id": [1, 2, 3, 3],
            "instrument_class": ["C", "C", "P", "P"],
            "strike_price": [7585.0] * 4,
            "security_update_action": ["A", "M", "A", "D"],
        }
    )
    before_delete = current_session_definition_universe(
        frame,
        session_date="2026-08-03",
        as_of_ns=_ns("2026-08-03 12:02:30"),
    )
    assert before_delete["SPXW  260803C07585000"].current_instrument_id == 2
    assert "SPXW  260803P07585000" in before_delete
    after_delete = current_session_definition_universe(
        frame,
        session_date="2026-08-03",
        as_of_ns=_ns("2026-08-03 12:03:30"),
    )
    assert set(after_delete) == {"SPXW  260803C07585000"}


def test_unfit_foundation_binds_implementation_and_cannot_authorize_fit() -> None:
    root = Path(__file__).resolve().parents[2]
    foundation_path = (
        root
        / "v4/research/autoresearch_v2/foundations/live_opra_training_twin_v1_2026_08_03.json"
    )
    foundation = json.loads(foundation_path.read_text())
    implementation = root / foundation["implementation"]["path"]
    assert hashlib.sha256(implementation.read_bytes()).hexdigest() == foundation[
        "implementation"
    ]["sha256"]
    catalog = root / foundation["entry_feature_catalog"]["path"]
    assert hashlib.sha256(catalog.read_bytes()).hexdigest() == foundation[
        "entry_feature_catalog"
    ]["sha256"]
    assert foundation["fit_authorized"] is False
    assert foundation["holdout_open_authorized"] is False
    assert foundation["unfrozen_fit_blocking_parameters"]["decision_emission_lag_ms"] is None
