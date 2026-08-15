from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from v4.live.protocol101_live_entry import (
    LiveIndexState,
    build_live_surface_row,
    live_surface_decision,
    order_intent_from_prediction,
    selected_contract_payload,
)
from v4.live.protocol101_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
)
from v4.model.hypothesis_protocol import SurfaceVariant


def _quotes() -> list[dict]:
    rows = []
    for strike in (5995.0, 6000.0, 6005.0):
        for right in ("C", "P"):
            rows.append(
                {
                    "contract_id": f"SPXW-20260102-{strike:09.3f}-{right}",
                    "symbol": "SPX",
                    "trading_class": "SPXW",
                    "expiry": "20260102",
                    "strike": strike,
                    "right": right,
                    "bid": 9.8,
                    "ask": 10.0,
                    "bid_size": 20,
                    "ask_size": 25,
                    "iv": 0.20,
                    "delta": 0.50 if right == "C" else -0.50,
                    "gamma": 0.01,
                    "theta": -0.30,
                    "quote_age_ms": 100,
                }
            )
    return rows


def test_build_live_surface_row_creates_ladder_and_lookup() -> None:
    state = LiveIndexState()
    ts = datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc)

    row, lookup = build_live_surface_row(
        decision_time=ts,
        spx=6000.0,
        vix=18.0,
        option_quotes=_quotes(),
        index_state=state,
    )

    assert row["option_ladder"].shape == (21, 2, 15)
    assert row["candidate_mask"].sum() == 6
    assert row["strike_offsets"][0] == -50
    assert row["strike_offsets"][-1] == 50
    assert row["market_window"].shape == (30, 7)
    assert np.isfinite(row["labels_net_pnl"]).all()
    assert lookup["SPXW-20260102-06000.000-C"]["ask"] == 10.0
    assert np.isclose(lookup["SPXW-20260102-06000.000-C"]["mid"], 9.9)
    assert row["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert lookup["SPXW-20260102-06000.000-C"]["feature_contract_version"] == FEATURE_CONTRACT_VERSION


def test_build_live_surface_row_filters_stale_quotes_like_historical_dataset() -> None:
    state = LiveIndexState()
    ts = datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc)
    quotes = _quotes()
    for quote in quotes:
        quote["quote_age_ms"] = 91_000

    row, lookup = build_live_surface_row(
        decision_time=ts,
        spx=6000.0,
        vix=18.0,
        option_quotes=quotes,
        index_state=state,
    )

    assert row["candidate_mask"].sum() == 0
    assert lookup == {}
    assert len(row["candidate_filter_trace"]) == 42
    reason_counts = {
        reason: sum(reason in item["filter_reasons"] for item in row["candidate_filter_trace"])
        for reason in {"stale_quote", "no_ibkr_quote_at_decision"}
    }
    assert reason_counts == {"stale_quote": 6, "no_ibkr_quote_at_decision": 36}
    assert row["ladder_context"]["atm_strike"] == 6000


def test_microstructure_masked_live_row_keeps_raw_tradable_quote_when_greeks_missing() -> None:
    ts = datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc)
    quotes = [
        {
            "contract_id": "SPXW-20260102-06450.000-C",
            "symbol": "SPX",
            "trading_class": "SPXW",
            "expiry": "20260102",
            "strike": 6450.0,
            "right": "C",
            "bid": 0.9,
            "ask": 1.1,
            "bid_size": 10,
            "ask_size": 10,
            "settlement_time_utc": "2026-01-02T21:00:00Z",
            "quote_age_ms": 100,
        }
    ]
    v1_row, _ = build_live_surface_row(
        decision_time=ts,
        spx=6500.0,
        vix=18.0,
        option_quotes=quotes,
        index_state=LiveIndexState(),
    )
    v2_row, lookup = build_live_surface_row(
        decision_time=ts,
        spx=6500.0,
        vix=18.0,
        option_quotes=quotes,
        index_state=LiveIndexState(),
        feature_contract_name=FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
    )

    assert v1_row["candidate_mask"].sum() == 0
    assert v2_row["feature_contract_version"] == FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
    assert lookup["SPXW-20260102-06450.000-C"]["feature_contract_version"] == FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
    assert v2_row["candidate_mask"].sum() == 1
    assert lookup["SPXW-20260102-06450.000-C"]["post_filter_candidate"] is True
    assert v2_row["candidate_filter_trace"][0]["candidate_filter"]["thresholds"]["require_greeks"] is False


def test_live_index_state_keeps_enough_intraday_context() -> None:
    state = LiveIndexState()
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    for idx in range(900):
        state.add(timestamp=start + timedelta(seconds=idx * 5), spx=6000.0 + idx * 0.01, vix=18.0)

    summary = state.session_context_summary(datetime(2026, 1, 2, 15, 45, tzinfo=timezone.utc))

    assert len(state.rows) == 900
    assert summary["row_count"] == 900
    assert summary["minute_row_count"] == 75
    assert summary["span_minutes"] > 70.0


def test_live_context_summary_counts_opening_minute_with_first_tick_delay() -> None:
    state = LiveIndexState()
    start = datetime(2026, 1, 2, 14, 30, 11, tzinfo=timezone.utc)
    for idx in range(30):
        state.add(timestamp=start + timedelta(minutes=idx), spx=6000.0 + idx, vix=18.0)
    state.add(timestamp=datetime(2026, 1, 2, 15, 0, 0, tzinfo=timezone.utc), spx=6030.0, vix=18.0)

    summary = state.session_context_summary(datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc))

    assert summary["first_timestamp"] == "2026-01-02T14:30:00+00:00"
    assert summary["minute_row_count"] == 30
    assert summary["span_minutes"] == 29.0
    assert summary["opening_context_ready"] is True
    assert summary["missing_opening_minutes"] == 0


def test_live_structure_uses_available_completed_bars_for_atr_and_first15() -> None:
    state = LiveIndexState()
    start = pd.Timestamp("2026-01-02 09:30", tz="America/New_York").tz_convert("UTC")
    for minute in range(15):
        state.add(timestamp=start + pd.Timedelta(minutes=minute), spx=6000.0 + minute, vix=18.0)
        state.add(timestamp=start + pd.Timedelta(minutes=minute, seconds=30), spx=6001.0 + minute, vix=18.0)

    early = state.structure_features(start + pd.Timedelta(minutes=1))
    first15 = state.structure_features(start + pd.Timedelta(minutes=15))

    assert early[22] > 0.0
    assert first15[14] == 1.0


def test_live_context_summary_marks_late_start_missing_opening_context() -> None:
    state = LiveIndexState()
    start = datetime(2026, 1, 2, 14, 39, tzinfo=timezone.utc)
    for idx in range(31):
        state.add(timestamp=start + timedelta(minutes=idx), spx=6000.0 + idx, vix=18.0)

    summary = state.session_context_summary(datetime(2026, 1, 2, 15, 9, tzinfo=timezone.utc))

    assert summary["first_timestamp"] == "2026-01-02T14:39:00+00:00"
    assert summary["expected_first_timestamp"] == "2026-01-02T14:30:00+00:00"
    assert summary["opening_context_ready"] is False
    assert summary["missing_opening_minutes"] == 9


def test_live_structure_features_fail_closed_when_opening_context_is_missing() -> None:
    state = LiveIndexState()
    start = datetime(2026, 1, 2, 14, 39, tzinfo=timezone.utc)
    for idx in range(31):
        state.add(timestamp=start + timedelta(minutes=idx), spx=6000.0 + idx, vix=18.0)

    structure = state.structure_features(datetime(2026, 1, 2, 15, 9, tzinfo=timezone.utc))

    assert np.allclose(structure, 0.0)


def test_live_structure_features_use_opening_minute_omar_not_session_range() -> None:
    state = LiveIndexState()
    opening = datetime(2026, 1, 2, 14, 30, 5, tzinfo=timezone.utc)
    state.add(timestamp=opening, spx=6000.0, vix=18.0)
    state.add(timestamp=opening + timedelta(seconds=15), spx=6004.0, vix=18.0)
    state.add(timestamp=opening + timedelta(seconds=30), spx=5998.0, vix=18.0)
    for minute in range(1, 32):
        state.add(timestamp=opening + timedelta(minutes=minute), spx=6025.0 + minute, vix=18.0)

    structure = state.structure_features(datetime(2026, 1, 2, 15, 1, tzinfo=timezone.utc))

    assert structure[7] == 6004.0
    assert structure[8] == 5998.0
    assert structure[10] == 6.0
    assert structure[10] != 57.0


def test_live_structure_last10_excludes_current_minute_like_historical_cache() -> None:
    state = LiveIndexState()
    opening = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    state.add(timestamp=opening, spx=6000.0, vix=18.0)
    state.add(timestamp=opening + timedelta(seconds=20), spx=6004.0, vix=18.0)
    state.add(timestamp=opening + timedelta(seconds=40), spx=5998.0, vix=18.0)
    for minute in range(1, 12):
        state.add(timestamp=opening + timedelta(minutes=minute), spx=6000.0 + minute, vix=18.0)
    state.add(timestamp=opening + timedelta(minutes=12), spx=6050.0, vix=18.0)

    structure = state.structure_features(opening + timedelta(minutes=12))

    assert structure[20] < 2.0
    assert structure[21] == 1.0


def test_live_index_state_resamples_intraday_context_to_one_minute_features() -> None:
    state = LiveIndexState()
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    for idx in range(360):
        state.add(timestamp=start + timedelta(seconds=idx * 5), spx=6000.0 + idx * 0.01, vix=18.0)

    window = state.market_window(datetime(2026, 1, 2, 14, 59, 55, tzinfo=timezone.utc))

    assert window.shape == (30, 7)
    assert window[-1, 6] > 1.0  # 15-minute momentum, not 15 raw five-second ticks.


def test_live_market_window_keeps_missing_preopen_minutes_fully_missing() -> None:
    state = LiveIndexState()
    opening = datetime(2026, 1, 2, 14, 30, 10, tzinfo=timezone.utc)
    state.add(timestamp=opening, spx=6000.0, vix=18.0)

    window = state.market_window(datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc))

    assert np.isnan(window[:-1]).all()
    assert np.isfinite(window[-1]).all()


def test_live_surface_decision_uses_spxw_quotes_without_future_labels() -> None:
    state = LiveIndexState()
    ts = datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc)
    row, _ = build_live_surface_row(
        decision_time=ts,
        spx=6000.0,
        vix=18.0,
        option_quotes=_quotes(),
        index_state=state,
    )
    variant = SurfaceVariant(
        name="surface_structure_aplus_side_value_rank",
        action_space="surface",
        market_mode="structure",
        token_mode="aplus",
        loss_mode="aplus_side_value_rank",
    )

    decision = live_surface_decision(
        session="2026-01-02",
        row=row,
        variant=variant,
        policy_index=1,
        index_state=state,
    )

    assert decision.session == "2026-01-02"
    assert decision.token_mask.sum() == 6
    assert decision.labels[decision.token_mask].shape == (6,)
    assert np.isfinite(decision.labels).all()


def test_order_intent_from_prediction_uses_selected_ask() -> None:
    state = LiveIndexState()
    ts = datetime(2026, 1, 2, 15, 5, tzinfo=timezone.utc)
    _, lookup = build_live_surface_row(
        decision_time=ts,
        spx=6000.0,
        vix=18.0,
        option_quotes=_quotes(),
        index_state=state,
    )
    prediction = {
        "action": "enter",
        "selected": {"contract_id": "SPXW-20260102-06000.000-C"},
    }

    intent = order_intent_from_prediction(prediction, lookup, quantity=1)
    payload = selected_contract_payload(intent, lookup)

    assert intent is not None
    assert intent.symbol == "SPX"
    assert intent.trading_class == "SPXW"
    assert intent.expiry == "20260102"
    assert intent.strike == 6000.0
    assert intent.right == "C"
    assert intent.limit_price == 10.0
    assert payload["contract_id"] == "SPXW-20260102-06000.000-C"
