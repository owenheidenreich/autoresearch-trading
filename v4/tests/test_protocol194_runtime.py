from __future__ import annotations

import pandas as pd

from v4.live.protocol194_runtime import (
    build_runtime_event,
    candidate_from_row,
    live_candidate_mask,
    validate_runtime_event,
    validate_runtime_stream,
)


def test_protocol194_runtime_event_is_no_order_and_valid() -> None:
    row = build_runtime_event(
        session="2026-05-22",
        timestamp="2026-05-22T14:31:00+00:00",
        selected_action="wait",
        candidate_set=_candidate_set(),
        model_decision={"score": -1.0, "threshold": 0.5, "future_exit_fields_used": False},
        latency={
            "candidate_validation_ms": 1.0,
            "model_inference_ms": 2.0,
            "total_decision_ms": 4.0,
            "budget_passed": True,
        },
        risk_gate={"passed": True, "reason": "pass"},
        paper_account_state=_account(),
    )

    result = validate_runtime_event(row)

    assert result.status == "pass"
    assert result.errors == []


def test_protocol194_runtime_schema_rejects_broker_order_fields() -> None:
    row = build_runtime_event(
        session="2026-05-22",
        timestamp="2026-05-22T14:31:00+00:00",
        selected_action="wait",
        candidate_set=_candidate_set(),
        model_decision={"score": -1.0, "threshold": 0.5},
        latency={
            "candidate_validation_ms": 1.0,
            "model_inference_ms": 2.0,
            "total_decision_ms": 4.0,
            "budget_passed": True,
        },
        risk_gate={"passed": True, "reason": "pass"},
        paper_account_state=_account(),
    )
    row["order_id"] = "should-not-exist"

    result = validate_runtime_event(row)

    assert result.status == "fail"
    assert any("broker/order field" in error for error in result.errors)


def test_protocol194_enter_requires_protocol166_valid_contract() -> None:
    row = build_runtime_event(
        session="2026-05-22",
        timestamp="2026-05-22T14:31:00+00:00",
        selected_action="enter",
        candidate_set=_candidate_set(),
        model_decision={"score": 1.0, "threshold": 0.5},
        latency={
            "candidate_validation_ms": 1.0,
            "model_inference_ms": 2.0,
            "total_decision_ms": 4.0,
            "budget_passed": True,
        },
        risk_gate={"passed": True, "reason": "pass"},
        paper_account_state=_account(),
        selected_contract=_candidate(),
    )

    assert validate_runtime_event(row).status == "pass"

    row["selected_contract"]["root"] = "SPX"
    assert validate_runtime_event(row).status == "fail"


def test_live_candidate_mask_does_not_require_future_exit_fields() -> None:
    frame = pd.DataFrame([_candidate()])
    frame["candidate_exit_dt"] = pd.NaT
    mask = live_candidate_mask(frame, account_state=_account())

    assert mask.tolist() == [True]


def test_live_candidate_mask_blocks_unaffordable_candidates() -> None:
    candidate = _candidate()
    candidate["entry_ask"] = 200.0
    candidate["entry_premium"] = 20_000.0
    frame = pd.DataFrame([candidate])

    mask = live_candidate_mask(frame, account_state=_account())

    assert mask.tolist() == [False]


def test_validate_runtime_stream_summarizes_counts() -> None:
    row = build_runtime_event(
        session="2026-05-22",
        timestamp="2026-05-22T14:31:00+00:00",
        selected_action="wait",
        candidate_set=_candidate_set(),
        model_decision={"score": -1.0, "threshold": 0.5},
        latency={
            "candidate_validation_ms": 1.0,
            "model_inference_ms": 2.0,
            "total_decision_ms": 4.0,
            "budget_passed": True,
        },
        risk_gate={"passed": True, "reason": "pass"},
        paper_account_state=_account(),
    )

    summary = validate_runtime_stream([row])

    assert summary["status"] == "pass"
    assert summary["event_counts"] == {"model_decision": 1}
    assert summary["action_counts"] == {"wait": 1}


def _candidate() -> dict:
    return {
        "decision_time": "2026-05-22T14:31:00+00:00",
        "contract_id": "SPXW-20260522-07385.000-C",
        "root": "SPXW",
        "settlement_style": "PM",
        "right": "C",
        "offset": 5.0,
        "entry_bid": 10.0,
        "entry_ask": 10.1,
        "entry_mid": 10.05,
        "entry_spread": 0.1,
        "entry_bid_size": 10.0,
        "entry_ask_size": 11.0,
        "entry_premium": 1010.0,
        "entry_delta": 0.45,
        "entry_gamma": 0.02,
        "entry_theta": -0.3,
        "entry_iv": 0.20,
    }


def _candidate_set() -> dict:
    return {
        "candidate_count": 42,
        "valid_candidate_count": 42,
        "root": "SPXW",
        "settlement_style": "PM",
        "max_abs_offset": 50.0,
        "call_count": 21,
        "put_count": 21,
    }


def _account() -> dict:
    return {
        "starting_cash": 10_000.0,
        "account_equity": 10_000.0,
        "cash_available": 10_000.0,
        "open_positions": 0,
        "open_position_count": 0,
        "max_concurrent_positions": 1,
        "max_contracts": 1,
    }
