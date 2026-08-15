from __future__ import annotations

import pandas as pd

from v4.model.unified_policy_trajectory import (
    HOLDING_TRAJECTORY_FEATURE_COLUMNS,
    flat_decision_states_from_action_rows,
    holding_decision_states_from_rows,
    summarize_flat_coverage,
    summarize_holding_coverage,
    trajectory_contract_payload,
    trajectory_foundation_decision,
    validate_trajectory_feature_contract,
)


def test_flat_rows_extract_unified_decision_states_with_candidate_set() -> None:
    states = flat_decision_states_from_action_rows(pd.DataFrame([_flat_row("a"), _flat_row("b")]))

    assert len(states) == 1
    assert states[0].position_state == "flat"
    assert len(states[0].candidates) == 2
    assert states[0].candidates[0].contract_id.startswith("SPXW-")


def test_holding_rows_extract_position_state_without_candidates() -> None:
    states = holding_decision_states_from_rows(pd.DataFrame([_holding_row()]))

    assert len(states) == 1
    assert states[0].position_state == "holding"
    assert states[0].position is not None
    assert states[0].position.current_pnl == 150.0
    assert states[0].candidates == ()


def test_trajectory_feature_contract_excludes_label_derived_holding_inputs() -> None:
    result = validate_trajectory_feature_contract()
    payload = trajectory_contract_payload()

    assert result["status"] == "pass"
    assert "entry_a_enter" not in HOLDING_TRAJECTORY_FEATURE_COLUMNS
    assert "entry_q_wait" not in HOLDING_TRAJECTORY_FEATURE_COLUMNS
    assert "entry_q_enter" not in HOLDING_TRAJECTORY_FEATURE_COLUMNS
    assert payload["excluded_holding_label_derived_columns"] == ["entry_a_enter", "entry_q_enter", "entry_q_wait"]


def test_coverage_summaries_report_flat_and_holding_shape() -> None:
    flat = summarize_flat_coverage(pd.DataFrame([_flat_row("a"), _flat_row("b", decision="2026-01-02T15:05:00+00:00")]))
    holding = summarize_holding_coverage(pd.DataFrame([_holding_row(), _holding_row(state_index=1)]))

    assert flat.loc[0, "candidate_rows"] == 2
    assert flat.loc[0, "decision_events"] == 2
    assert holding.loc[0, "state_rows"] == 2
    assert holding.loc[0, "oracle_hold_fraction"] == 1.0


def test_trajectory_foundation_decision_blocks_training_until_gates_are_ready() -> None:
    decision = trajectory_foundation_decision(
        flat_rows=10,
        holding_rows=10,
        feature_contract_status="pass",
        fill_model_ready=False,
        untouched_holdout_ready=False,
        live_parity_ready=False,
    )

    assert decision == "unified_trajectory_foundation_ready_training_blocked_by_foundation_gates"


def _flat_row(uid: str, *, decision: str = "2026-01-02T15:00:00+00:00") -> dict:
    return {
        "split": "unit",
        "session": "2026-01-02",
        "decision_time": decision,
        "decision_dt": decision,
        "candidate_uid": uid,
        "trade_uid": f"trade-{uid}",
        "contract_id": "SPXW-20260102-04000.000-C",
        "root": "SPXW",
        "settlement_style": "PM",
        "right": "C",
        "offset": 0.0,
        "entry_quote_time": decision,
        "entry_bid": 10.0,
        "entry_ask": 10.2,
        "entry_mid": 10.1,
        "entry_spread": 0.2,
        "entry_bid_size": 10.0,
        "entry_ask_size": 12.0,
        "entry_premium": 1020.0,
        "entry_delta": 0.45,
        "entry_gamma": 0.01,
        "entry_theta": -0.2,
        "entry_iv": 0.2,
        "oracle_action": "wait",
        "a_enter": -10.0,
    }


def _holding_row(*, state_index: int = 0) -> dict:
    return {
        "split": "unit",
        "session": "2026-01-02",
        "candidate_uid": "candidate-a",
        "trade_uid": "trade-a",
        "contract_id": "SPXW-20260102-04000.000-C",
        "right": "C",
        "entry_time": "2026-01-02T15:00:00+00:00",
        "state_time": f"2026-01-02T15:0{state_index}:00+00:00",
        "state_index": state_index,
        "entry_ask": 10.0,
        "bid": 11.5,
        "ask": 11.8,
        "current_pnl": 150.0,
        "mfe_to_now": 200.0,
        "mae_to_now": -50.0,
        "giveback_from_mfe": 50.0,
        "minutes_since_entry": float(state_index),
        "minutes_to_forced_flat": 300.0,
        "oracle_holding_action": "hold",
    }
