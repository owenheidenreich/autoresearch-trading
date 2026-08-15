from __future__ import annotations

import pandas as pd

from v4.model.unified_serial_dp_oracle import (
    BLOCKED_DECISION,
    READY_DECISION,
    build_serial_dp_oracle_scope,
)


def test_serial_dp_oracle_excludes_sessions_with_missing_holding_paths() -> None:
    scope = build_serial_dp_oracle_scope(
        pd.DataFrame([_flat("keep", "a"), _flat("skip", "b")]),
        pd.DataFrame([_holding("keep", "a")]),
        pd.DataFrame([_baseline("keep"), _baseline("skip")]),
        pd.DataFrame([{"split": "unit", "session": "skip", "skip_reason": "missing_contract_quotes"}]),
        training_splits=("unit",),
    )

    manifest = scope["manifest"]

    assert manifest.decision == READY_DECISION
    assert manifest.included_sessions == 1
    assert manifest.excluded_sessions_with_missing_holding_paths == 1
    assert manifest.holding_coverage_of_oracle_entries == 1.0


def test_serial_dp_oracle_blocks_when_holding_coverage_is_missing() -> None:
    scope = build_serial_dp_oracle_scope(
        pd.DataFrame([_flat("keep", "a")]),
        pd.DataFrame(columns=["split", "session", "candidate_uid", "entry_time", "state_time", "oracle_holding_action", "q_exit", "q_hold", "a_hold"]),
        pd.DataFrame([_baseline("keep")]),
        pd.DataFrame(columns=["split", "session", "skip_reason"]),
        training_splits=("unit",),
    )

    assert scope["manifest"].decision == BLOCKED_DECISION
    assert scope["manifest"].holding_coverage_of_oracle_entries == 0.0


def test_serial_dp_oracle_blocks_baseline_not_available_rows() -> None:
    baseline = _baseline("keep")
    baseline["protocol101_action"] = "baseline_not_available_for_split"
    scope = build_serial_dp_oracle_scope(
        pd.DataFrame([_flat("keep", "a")]),
        pd.DataFrame([_holding("keep", "a")]),
        pd.DataFrame([baseline]),
        pd.DataFrame(columns=["split", "session", "skip_reason"]),
        training_splits=("unit",),
    )

    assert scope["manifest"].decision == BLOCKED_DECISION
    assert scope["manifest"].disallowed_baseline_actions == {"baseline_not_available_for_split": 1}


def _flat(session: str, uid: str) -> dict:
    return {
        "split": "unit",
        "session": session,
        "decision_time": "2026-01-02T15:00:00+00:00",
        "decision_dt": "2026-01-02T15:00:00+00:00",
        "candidate_uid": uid,
        "contract_id": f"SPXW-{uid}",
        "entry_premium": 1000.0,
        "q_wait": 0.0,
        "q_enter": 100.0,
        "a_enter": 100.0,
        "session_oracle_value": 100.0,
        "best_enter_value_at_decision": 100.0,
        "oracle_action": "enter",
        "oracle_action_uid": uid,
    }


def _holding(session: str, uid: str) -> dict:
    return {
        "split": "unit",
        "session": session,
        "candidate_uid": uid,
        "trade_uid": f"trade-{uid}",
        "entry_time": "2026-01-02T15:00:00+00:00",
        "state_time": "2026-01-02T15:01:00+00:00",
        "oracle_holding_action": "exit",
        "q_exit": 100.0,
        "q_hold": 80.0,
        "a_hold": -20.0,
        "a_switch": 20.0,
        "current_pnl": 100.0,
    }


def _baseline(session: str) -> dict:
    return {
        "split": "unit",
        "session": session,
        "decision_time": "2026-01-02T15:00:00+00:00",
        "decision_dt": "2026-01-02T15:00:00+00:00",
        "seed": 1,
        "protocol101_action": "wait",
        "contract_id": "",
    }
