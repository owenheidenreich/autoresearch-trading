from __future__ import annotations

from v4.live.protocol166_parity_contract import default_protocol166_contract, validate_candidate, validate_position_action
from v4.model.unified_serial_game import (
    action_mask_for_candidates,
    build_flat_action_advantage_labels,
    hold_exit_advantage_path,
    hold_exit_opportunity_advantage_path,
    validate_unified_candidate,
)

import pandas as pd


def test_protocol166_contract_disallows_protocol101_entry_gates() -> None:
    contract = default_protocol166_contract()

    assert "protocol101_min_edge_gate" in contract.disallowed_pre_entry_gates
    assert "protocol101_allowed_time_bucket_gate" in contract.disallowed_pre_entry_gates
    assert "protocol101_selected_candidate_dependency" in contract.disallowed_pre_entry_gates


def test_protocol166_candidate_validation_accepts_live_training_parity_row() -> None:
    result = validate_candidate(_candidate(), _account())

    assert result == {"status": "pass", "errors": []}


def test_protocol166_candidate_validation_rejects_wrong_root_and_unaffordable_trade() -> None:
    candidate = _candidate()
    candidate["root"] = "SPX"
    candidate["contract_id"] = "SPX-20260102-04000.000-C"
    account = _account()
    account["cash_available"] = 500.0

    result = validate_candidate(candidate, account)

    assert result["status"] == "fail"
    assert "wrong_root" in result["errors"]
    assert "unaffordable" in result["errors"]


def test_unified_contract_rejects_stale_quote_and_late_entry() -> None:
    candidate = _candidate()
    candidate["quote_age_ms"] = 2_000
    candidate["context_age_ms"] = 100
    candidate["decision_time"] = "2026-01-02T20:35:00+00:00"

    result = validate_unified_candidate(candidate, _account())

    assert result["status"] == "fail"
    assert "stale_option_quote" in result["errors"]
    assert "after_no_new_entries_cutoff" in result["errors"]


def test_position_action_rejects_enter_put_with_call_candidate() -> None:
    result = validate_position_action(
        position_state="flat",
        action="enter_put",
        selected_candidate=_candidate(),
        account_state=_account(),
    )

    assert result["status"] == "fail"
    assert "action_candidate_side_mismatch" in result["errors"]


def test_action_mask_rejects_unaffordable_candidate() -> None:
    candidate = _candidate()
    candidate["quote_age_ms"] = 0
    candidate["context_age_ms"] = 0
    frame = pd.DataFrame([candidate])
    account = _account()
    account["cash_available"] = 500.0

    mask = action_mask_for_candidates(frame, account)

    assert mask.tolist() == [False]


def test_flat_action_advantage_penalizes_early_trade_that_blocks_later_winner() -> None:
    rows = pd.DataFrame(
        [
            {
                **_candidate(),
                "split": "unit",
                "session": "2026-01-02",
                "candidate_uid": "early_weak",
                "decision_time": "2026-01-02T15:00:00+00:00",
                "decision_dt": "2026-01-02T15:00:00+00:00",
                "candidate_exit_time": "2026-01-02T15:10:00+00:00",
                "candidate_pnl": 10.0,
            },
            {
                **_candidate(),
                "split": "unit",
                "session": "2026-01-02",
                "candidate_uid": "later_winner",
                "decision_time": "2026-01-02T15:05:00+00:00",
                "decision_dt": "2026-01-02T15:05:00+00:00",
                "candidate_exit_time": "2026-01-02T15:06:00+00:00",
                "candidate_pnl": 500.0,
            },
        ]
    )

    labels = build_flat_action_advantage_labels(rows)
    early = labels[labels["candidate_uid"].eq("early_weak")].iloc[0]
    later = labels[labels["candidate_uid"].eq("later_winner")].iloc[0]

    assert early["a_enter"] < 0.0
    assert later["oracle_action"] == "enter"


def test_hold_exit_advantage_allows_winner_to_run_and_exits_when_no_upside() -> None:
    labels = hold_exit_advantage_path([10.0, 12.0, 15.0, 13.0], entry_ask=10.0)

    assert labels.iloc[0]["oracle_holding_action"] == "hold"
    assert labels.iloc[2]["oracle_holding_action"] == "exit"


def test_hold_exit_opportunity_advantage_prices_next_slot_value() -> None:
    labels = hold_exit_opportunity_advantage_path(
        [10.0, 10.5, 10.2],
        entry_ask=10.0,
        future_flat_values=[500.0, 0.0, 0.0],
    )

    assert labels.iloc[0]["oracle_holding_action"] == "exit"
    assert labels.iloc[0]["a_switch"] > 0.0


def _candidate() -> dict:
    return {
        "decision_time": "2026-01-02T15:00:00+00:00",
        "contract_id": "SPXW-20260102-04000.000-C",
        "root": "SPXW",
        "settlement_style": "PM",
        "right": "C",
        "offset": 0.0,
        "entry_bid": 10.0,
        "entry_ask": 10.2,
        "entry_mid": 10.1,
        "entry_spread": 0.2,
        "entry_bid_size": 10.0,
        "entry_ask_size": 12.0,
        "entry_premium": 1020.0,
        "entry_delta": 0.45,
        "entry_gamma": 0.01,
        "entry_theta": -0.20,
        "entry_iv": 0.20,
    }


def _account() -> dict:
    return {
        "account_equity": 10_000.0,
        "cash_available": 10_000.0,
        "open_position_count": 0,
        "max_concurrent_positions": 1,
        "max_contracts": 1,
    }
