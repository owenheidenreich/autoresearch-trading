from __future__ import annotations

import pandas as pd

from v4.live.full_action_challenger_adapter import (
    ALL_FEATURE_COLUMNS,
    FullActionHistoryState,
    build_full_action_candidates_from_quotes,
    feature_source_coverage,
)


def test_feature_source_coverage_maps_all_model_features() -> None:
    coverage = feature_source_coverage(ALL_FEATURE_COLUMNS)
    assert coverage["status"] == "pass"
    assert coverage["missing_feature_sources"] == []
    assert coverage["mapped_feature_count"] == coverage["feature_count"]


def test_live_style_candidate_builder_tracks_causal_history() -> None:
    history = FullActionHistoryState()
    first_time = pd.Timestamp("2026-05-20T14:00:00Z")
    quotes = [
        {
            "contract_id": "SPXW-20260520-07400.000-C",
            "strike": 7400.0,
            "right": "C",
            "bid": 10.0,
            "ask": 10.2,
            "bid_size": 5.0,
            "ask_size": 4.0,
            "underlying_price": 7401.0,
            "iv": 0.2,
            "delta": 0.52,
            "gamma": 0.01,
            "theta": -30.0,
            "surface_edge": 12.0,
        },
        {
            "contract_id": "SPXW-20260520-07400.000-P",
            "strike": 7400.0,
            "right": "P",
            "bid": 8.0,
            "ask": 8.2,
            "bid_size": 8.0,
            "ask_size": 6.0,
            "underlying_price": 7401.0,
            "iv": 0.21,
            "delta": -0.48,
            "gamma": 0.012,
            "theta": -31.0,
            "surface_edge": 4.0,
        },
    ]
    first = build_full_action_candidates_from_quotes(
        decision_time=first_time,
        spx=7401.0,
        vix=18.0,
        option_quotes=quotes,
        history=history,
        market_features={
            "market_spx_close": 7401.0,
            "market_vix_close": 18.0,
            "market_spx_vwap": 7400.0,
            "market_omar": 0.25,
            "market_session_range": 18.0,
            "market_momentum_5m": 3.0,
            "market_momentum_15m": 7.0,
        },
    )
    assert set(ALL_FEATURE_COLUMNS).issubset(first.columns)
    assert first["hist_events_seen"].eq(0.0).all()
    assert first["entry_affordable_10k"].eq(1.0).all()
    history.update(first, decision_time=first_time)

    second = build_full_action_candidates_from_quotes(
        decision_time=first_time + pd.Timedelta(minutes=1),
        spx=7406.0,
        vix=17.9,
        option_quotes=quotes,
        history=history,
    )
    assert second["hist_events_seen"].eq(1.0).all()
    assert second["hist_prev_candidate_count"].eq(2.0).all()
    assert second["hist_prev_max_edge"].eq(12.0).all()
    assert second["hist_prev_call_minus_put_edge"].eq(8.0).all()
