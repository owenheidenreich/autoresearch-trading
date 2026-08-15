from __future__ import annotations

import pandas as pd
import pytest

from v4.scripts import run_protocol101_hold_exit_action_advantage_foundation as foundation


def _trade() -> pd.Series:
    return pd.Series(
        {
            "reported_split": "q1_2026",
            "fold": "test",
            "seed": 1,
            "session": "2026-01-02",
            "candidate_uid": "candidate-1",
            "contract_id": "SPXW-test-C",
            "right": "C",
            "offset": -20.0,
            "decision_ts": pd.Timestamp("2026-01-02T15:00:00+00:00"),
            "exit_ts": pd.Timestamp("2026-01-02T15:02:00+00:00"),
            "entry_ask": 10.0,
            "entry_bid": 9.8,
            "score": 1.0,
            "threshold": 0.5,
            "exit_reason": "sequence_residual_override",
            "time_bucket": "post_open_morning",
        }
    )


def test_labels_for_trade_builds_hold_exit_advantage_rows() -> None:
    quotes = pd.DataFrame(
        {
            "quote_time": pd.to_datetime(
                [
                    "2026-01-02T15:00:00+00:00",
                    "2026-01-02T15:01:00+00:00",
                    "2026-01-02T15:02:00+00:00",
                ],
                utc=True,
            ),
            "contract_id": ["SPXW-test-C"] * 3,
            "bid": [9.8, 12.0, 11.0],
            "ask": [10.0, 12.3, 11.4],
            "underlying_price": [5000.0, 5005.0, 5004.0],
        }
    )

    rows, skip = foundation.labels_for_trade(
        _trade(),
        quotes,
        pd.Timestamp("2026-01-02T20:55:00+00:00"),
    )

    assert skip is None
    assert len(rows) == 3
    assert rows[0]["current_pnl"] == pytest.approx(-20.0)
    assert rows[0]["a_hold"] == pytest.approx(220.0)
    assert rows[0]["oracle_holding_action"] == "hold"
    assert rows[2]["is_protocol101_exit_state"] is True


def test_feature_contract_excludes_label_columns() -> None:
    contract = foundation.build_feature_contract()

    assert contract["status"] == "pass"
    assert contract["forbidden_overlap"] == []
    assert "a_hold" in contract["label_only_columns"]
    assert "current_pnl" in contract["causal_feature_columns"]


def test_exit_decision_summary_groups_protocol101_exit_state() -> None:
    frame = pd.DataFrame(
        {
            "reported_split": ["q1_2026", "q1_2026"],
            "candidate_uid": ["a", "b"],
            "is_protocol101_exit_state": [True, False],
            "exit_reason": ["target", "target"],
            "oracle_holding_action": ["exit", "hold"],
            "oracle_one_step_action": ["exit", "hold"],
            "a_hold": [0.0, 100.0],
            "current_pnl": [200.0, 100.0],
            "mfe_to_now": [200.0, 120.0],
            "giveback_from_mfe": [0.0, 20.0],
            "future_worst_before_best_pnl": [200.0, 100.0],
        }
    )

    audit = foundation.build_exit_state_audit(frame)
    summary = foundation.build_protocol101_exit_decision_summary(audit)

    assert len(audit) == 1
    assert len(summary) == 1
    assert summary.iloc[0]["oracle_holding_action"] == "exit"


def test_summary_blocks_training_until_slot_and_fill_are_resolved() -> None:
    frame = pd.DataFrame(
        {
            "candidate_uid": ["a"],
            "oracle_holding_action": ["hold"],
            "a_hold": [100.0],
            "is_protocol101_exit_state": [True],
        }
    )
    skips = pd.DataFrame()

    summary = foundation.build_summary(
        frame,
        skips,
        source_trades=1,
        output_dataset=pd.Timestamp("2026-01-02"),
        feature_contract=foundation.build_feature_contract(),
    )

    assert summary["training_allowed"] is False
    assert summary["challenge_allowed"] is False
    assert "no_counterfactual_flat_slot_opportunity_cost" in summary["limitations"]
