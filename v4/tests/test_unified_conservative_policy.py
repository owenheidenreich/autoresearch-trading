from __future__ import annotations

import pytest

from v4.model.unified_conservative_policy import (
    CandidateSnapshotV1,
    ConservativePolicyGateConfig,
    ExecutionModelV1,
    ExecutionModelV1Config,
    PositionSnapshotV1,
    UnifiedDecisionStateV1,
    action_mask_for_decision_state,
    build_flat_action_advantage_label_v1,
    build_holding_action_advantage_label_v1,
    conservative_policy_improvement_decision,
    validate_no_future_feature_columns,
)

import pandas as pd


def test_flat_state_masks_unaffordable_and_stale_candidates() -> None:
    state = UnifiedDecisionStateV1(
        split="unit",
        session="2026-01-02",
        decision_time="2026-01-02T15:00:00+00:00",
        position_state="flat",
        account_equity=10_000.0,
        cash_available=1_000.0,
        candidates=(
            _candidate("ok", ask=8.0, quote_age_ms=100.0),
            _candidate("too_expensive", ask=20.0, quote_age_ms=100.0),
            _candidate("stale", ask=8.0, quote_age_ms=2_000.0),
        ),
    )

    mask = action_mask_for_decision_state(state)

    assert mask.wait is True
    assert mask.enter_candidates == (True, False, False)
    assert "unaffordable" in mask.reasons_by_candidate[1]
    assert "stale_option_quote" in mask.reasons_by_candidate[2]


def test_holding_state_forces_exit_after_forced_flat_cutoff() -> None:
    state = UnifiedDecisionStateV1(
        split="unit",
        session="2026-01-02",
        decision_time="2026-01-02T20:56:00+00:00",
        position_state="holding",
        account_equity=10_000.0,
        cash_available=9_000.0,
        position=PositionSnapshotV1(
            contract_id="SPXW-20260102-04000.000-C",
            right="C",
            entry_time="2026-01-02T15:00:00+00:00",
            entry_ask=10.0,
            current_bid=12.0,
            current_ask=12.4,
            current_pnl=200.0,
            mfe_to_now=300.0,
            mae_to_now=-50.0,
            giveback_from_mfe=100.0,
            minutes_since_entry=356.0,
            minutes_to_forced_flat=0.0,
        ),
    )

    mask = action_mask_for_decision_state(state)

    assert mask.hold is False
    assert mask.exit is True
    assert mask.forced_exit_required is True


def test_execution_model_blocks_stochastic_fill_without_observations() -> None:
    model = ExecutionModelV1(
        ExecutionModelV1Config(fill_model="stochastic_calibrated", fill_observations=0, required_fill_observations=30)
    )

    result = model.validate()

    assert result["status"] == "fail"
    assert "stochastic_fill_model_insufficient_observations" in result["errors"]


def test_no_future_feature_guard_rejects_label_columns() -> None:
    with pytest.raises(ValueError, match="future/label columns"):
        validate_no_future_feature_columns(["entry_delta", "q_enter", "candidate_exit_time"])


def test_action_advantage_label_v1_prices_wait_and_hold_correctly() -> None:
    flat = build_flat_action_advantage_label_v1(
        pd.DataFrame(
            [
                _candidate_row("early", "2026-01-02T15:00:00+00:00", "2026-01-02T15:20:00+00:00", pnl=50.0),
                _candidate_row("later", "2026-01-02T15:05:00+00:00", "2026-01-02T15:10:00+00:00", pnl=600.0),
            ]
        )
    )
    early = flat[flat["candidate_uid"].eq("early")].iloc[0]
    later = flat[flat["candidate_uid"].eq("later")].iloc[0]

    assert early["a_enter"] < 0.0
    assert later["oracle_action"] == "enter"
    assert flat["label_contract"].eq("ActionAdvantageLabelV1").all()

    holding = build_holding_action_advantage_label_v1(
        [10.0, 12.0, 11.0],
        entry_ask=10.0,
        future_flat_values=[500.0, 0.0, 0.0],
    )

    assert holding.iloc[0]["oracle_holding_action"] == "exit"
    assert holding.iloc[0]["a_exit"] > 0.0


def test_conservative_gate_defers_when_uncertainty_consumes_edge() -> None:
    decision = conservative_policy_improvement_decision(
        challenger_advantage=400.0,
        uncertainty=200.0,
        ood_penalty=100.0,
        config=ConservativePolicyGateConfig(min_advantage_margin=250.0),
    )

    assert decision["decision"] == "defer_to_protocol101"
    assert decision["allowed"] is False


def _candidate(uid: str, *, ask: float, quote_age_ms: float) -> CandidateSnapshotV1:
    return CandidateSnapshotV1(
        candidate_uid=uid,
        contract_id="SPXW-20260102-04000.000-C",
        right="C",
        entry_bid=max(ask - 0.2, 0.01),
        entry_ask=ask,
        entry_premium=ask * 100.0,
        quote_age_ms=quote_age_ms,
        entry_delta=0.45,
        entry_gamma=0.01,
        entry_theta=-0.20,
        entry_iv=0.20,
    )


def _candidate_row(uid: str, decision: str, exit_time: str, *, pnl: float) -> dict:
    candidate = _candidate(uid, ask=10.0, quote_age_ms=0.0).to_candidate_dict(decision)
    return {
        **candidate,
        "split": "unit",
        "session": "2026-01-02",
        "candidate_uid": uid,
        "decision_time": decision,
        "decision_dt": decision,
        "candidate_exit_time": exit_time,
        "candidate_pnl": pnl,
    }
