from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from v4.research.autoresearch_v2 import runtime_decision_parity as parity


def _official_spx() -> pd.DataFrame:
    times = pd.date_range("2026-06-09 13:30:00+00:00", periods=31, freq="min")
    return pd.DataFrame(
        {
            "event_time": times,
            "symbol": "SPX",
            "close": np.arange(31, dtype=float) + 7400.0,
            "volume": np.zeros(31, dtype=np.int64),
            "context_source": "thetadata_index_history_ohlc",
            "is_derived": False,
            "is_proxy": False,
            "is_official_index_data": True,
        }
    )


def test_frozen_binding_fails_loud_on_model_hash_drift(monkeypatch) -> None:
    monkeypatch.setattr(parity, "EXPECTED_MODEL_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="FROZEN MODEL DRIFT"):
        parity.assert_frozen_binding()


def test_live_contract_is_exact_signed18_and_has_no_ibkr_input() -> None:
    contract = parity.live_decision_feature_contract()
    assert [row["feature"] for row in contract["features"]] == list(
        parity.EXPECTED_FEATURES
    )
    assert contract["ordered_feature_count"] == 18
    assert contract["ibkr_decision_features"] is False
    assert contract["live_network_required"] is False
    assert "not consumed" in contract["vix_context_disposition"]


def test_thetadata_bar_open_clock_waits_for_completed_minute() -> None:
    frame = parity._validate_index(_official_spx(), symbol="SPX")
    decision = int(pd.Timestamp("2026-06-09 14:00:00+00:00").value)
    causal = parity._context_at(frame, decision, completion_lag_seconds=60)
    noncausal_diagnostic = parity._context_at(
        frame, decision, completion_lag_seconds=0
    )
    assert causal is not None and noncausal_diagnostic is not None
    assert causal["event_time_ns"] == int(
        pd.Timestamp("2026-06-09 13:59:00+00:00").value
    )
    assert causal["available_at_ns"] == decision
    assert causal["spx"] == 7429.0
    assert noncausal_diagnostic["event_time_ns"] == decision
    assert noncausal_diagnostic["spx"] == 7430.0


def test_feature_comparison_treats_missing_candidates_as_failures() -> None:
    common = {
        "candidate_uid": ["s|1|a"],
        **{name: [float(index)] for index, name in enumerate(parity.EXPECTED_FEATURES)},
    }
    training = pd.DataFrame(common)
    live = pd.DataFrame(
        {
            "candidate_uid": ["s|1|a", "s|1|b"],
            **{
                name: [float(index), float(index)]
                for index, name in enumerate(parity.EXPECTED_FEATURES)
            },
        }
    )
    rows, summary = parity.compare_features(training, live)
    assert summary["candidate_set_identical"] is False
    assert summary["feature_within_tolerance_rate"] == 0.5
    assert all(row["within_tolerance_rate_on_matched"] == 1.0 for row in rows)
    assert all(row["missing_candidate_count"] == 1 for row in rows)


def test_block_policy_comparison_does_not_hide_entry_drift_behind_waits() -> None:
    training = pd.DataFrame(
        [
            {
                "session": "2026-06-09",
                "block": 0,
                "action": "ENTER",
                "signal_time_ns": 1,
                "score_bits": parity._score_bits(1.0),
                "score": 1.0,
                "side": "C",
                "selected_contract": "A",
            }
        ]
    )
    live = training.copy()
    live.loc[0, "signal_time_ns"] = 2
    live.loc[0, "side"] = "P"
    live.loc[0, "selected_contract"] = "B"
    _, summary = parity.compare_block_policies(training, live)
    assert summary["action_match_rate"] == 1.0
    assert summary["signal_time_match_rate"] == 0.0
    assert summary["side_match_rate"] == 0.0
    assert summary["selected_contract_match_rate"] == 0.0
    assert summary["complete_block_policy_match_rate"] == 0.0
