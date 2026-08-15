"""Tests for fair-contract candidate validation before shadow/paper gates."""
from __future__ import annotations

from v4.live.protocol101_synchronization import (
    Protocol101FairContractCandidateValidationGateV1,
)
from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED


def _runner_plan(**overrides) -> dict:
    plan = {
        "mode": "train",
        "selected_feature_contract": FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
        "model_training_executed": True,
        "threshold_selection_executed": True,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
    }
    plan.update(overrides)
    return plan


def _training_result(*, validation_pnl: float = 500.0, diagnostic_pnl: float = 600.0) -> dict:
    return {
        "chosen_threshold": 42.0,
        "model_out": "/tmp/protocol101/model.pt",
        "neural": {
            "validation": {
                "metrics": {
                    "trades": 25,
                    "total_pnl": validation_pnl,
                    "profit_factor": 1.5,
                    "max_drawdown": -100.0,
                }
            },
            "diagnostic_test": {
                "metrics": {
                    "trades": 30,
                    "total_pnl": diagnostic_pnl,
                    "profit_factor": 1.6,
                    "max_drawdown": -120.0,
                }
            },
        },
    }


def test_candidate_gate_waits_when_training_result_is_absent() -> None:
    packet = Protocol101FairContractCandidateValidationGateV1.evaluate(
        runner_plan=_runner_plan(mode="dry-run", model_training_executed=False),
        training_result=None,
    )

    assert packet.status == "waiting_for_owner_approved_training_result"
    assert packet.model_training_executed is False
    assert packet.paper_submit_allowed is False
    assert packet.checks["training_result_present"]["pass"] is False


def test_candidate_gate_passes_safe_positive_candidate() -> None:
    packet = Protocol101FairContractCandidateValidationGateV1.evaluate(
        runner_plan=_runner_plan(),
        training_result=_training_result(),
    )

    assert packet.status == "pass"
    assert packet.decision == "candidate_can_proceed_to_strict_serial_lifecycle_replay"
    assert packet.checks["broker_endpoint_called"]["pass"] is True
    assert packet.checks["validation_positive_pnl"]["pass"] is True
    assert packet.checks["diagnostic_positive_pnl"]["pass"] is True


def test_candidate_gate_fails_if_broker_or_metrics_fail() -> None:
    packet = Protocol101FairContractCandidateValidationGateV1.evaluate(
        runner_plan=_runner_plan(broker_endpoint_called=True),
        training_result=_training_result(validation_pnl=-10.0),
    )

    assert packet.status == "fail"
    assert packet.checks["broker_endpoint_called"]["pass"] is False
    assert packet.checks["validation_positive_pnl"]["pass"] is False
