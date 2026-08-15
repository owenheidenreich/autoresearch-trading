from __future__ import annotations

from v4.model.neural_training_readiness import (
    BLOCKED,
    PASS,
    CHALLENGE_BLOCKED_DECISION,
    CHALLENGE_READY_DECISION,
    TRAINING_BLOCKED_DECISION,
    TRAINING_READY_DECISION,
    evaluate_neural_training_readiness,
    gate,
    make_default_holdout_reservation,
    validate_untouched_holdout_reservation,
)


def test_readiness_blocks_training_when_required_gate_is_blocked() -> None:
    evaluation = evaluate_neural_training_readiness(
        [
            gate("contract", PASS, "ok", "none"),
            gate("full serial dp oracle", BLOCKED, "missing", "build oracle"),
            gate(
                "live parity",
                BLOCKED,
                "missing",
                "collect parity",
                blocks_training=False,
                blocks_protocol101_challenge=True,
            ),
        ]
    )

    assert evaluation.training_decision == TRAINING_BLOCKED_DECISION
    assert evaluation.protocol101_challenge_decision == CHALLENGE_BLOCKED_DECISION
    assert evaluation.training_blockers == ("full serial dp oracle",)
    assert evaluation.challenge_blockers == ("full serial dp oracle", "live parity")


def test_readiness_can_allow_training_while_later_challenge_remains_blocked() -> None:
    evaluation = evaluate_neural_training_readiness(
        [
            gate("contract", PASS, "ok", "none"),
            gate(
                "live parity",
                BLOCKED,
                "historical proxy only",
                "collect live parity",
                blocks_training=False,
                blocks_protocol101_challenge=True,
            ),
        ]
    )

    assert evaluation.training_decision == TRAINING_READY_DECISION
    assert evaluation.protocol101_challenge_decision == CHALLENGE_BLOCKED_DECISION


def test_readiness_allows_protocol101_challenge_only_when_all_challenge_gates_pass() -> None:
    evaluation = evaluate_neural_training_readiness(
        [
            gate("contract", PASS, "ok", "none"),
            gate(
                "live parity",
                PASS,
                "ok",
                "none",
                blocks_training=False,
                blocks_protocol101_challenge=True,
            ),
        ]
    )

    assert evaluation.training_decision == TRAINING_READY_DECISION
    assert evaluation.protocol101_challenge_decision == CHALLENGE_READY_DECISION


def test_default_holdout_reservation_is_valid_but_data_is_pending() -> None:
    reservation = make_default_holdout_reservation(reserved_on="2026-05-24")
    validation = validate_untouched_holdout_reservation(reservation)

    assert validation["status"] == PASS
    assert validation["data_available"] is False
    assert validation["data_status"] == "pending_new_data_collection"


def test_holdout_validation_rejects_reused_diagnostic_split() -> None:
    payload = make_default_holdout_reservation(reserved_on="2026-05-24").to_dict()
    payload["split_label"] = "q3_2025"

    validation = validate_untouched_holdout_reservation(payload)

    assert validation["status"] == BLOCKED
    assert "holdout_uses_repeated_diagnostic_split" in validation["errors"]
