"""Readiness gates for the unified conservative neural policy.

This module keeps the next model run behind explicit foundation gates. It
separates the narrower question "may we start a preregistered neural training
run?" from the stronger question "may this challenge Protocol101?".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


ROLE_LABEL = "FOUNDATION_UNIFIED_NEURAL_TRAINING_READINESS_V1"
HOLDOUT_ROLE_LABEL = "FOUNDATION_UNTOUCHED_EVAL_BLOCK_RESERVATION_V1"
UNTOUCHED_EVAL_BLOCK_NAME = "UNTOUCHED_EVAL_BLOCK_V1"
PAPER_DEFAULT_BASELINE = "PAPER_DEFAULT_PROTOCOL101"

PASS = "pass"
PARTIAL = "partial"
BLOCKED = "blocked"

TRAINING_BLOCKED_DECISION = "neural_training_not_ready_foundation_gates_blocked"
TRAINING_READY_DECISION = "neural_training_ready_for_preregistered_conservative_policy_run"
CHALLENGE_BLOCKED_DECISION = "protocol101_challenge_not_ready_foundation_gates_blocked"
CHALLENGE_READY_DECISION = "protocol101_challenge_ready_for_untouched_strict_serial_replay"

EXPOSED_DIAGNOSTIC_SPLITS = (
    "q3_2025",
    "q4_2025",
    "q1_2026",
    "march_2026",
    "recent_2026",
    "q1_2025",
    "q1_2025_partial",
    "q2_2025",
    "q4_2024",
)

REQUIRED_FORBIDDEN_HOLDOUT_USES = (
    "feature_selection",
    "architecture_selection",
    "threshold_selection",
    "objective_selection",
    "sizing_selection",
    "exit_rule_selection",
    "model_selection",
    "training",
)


@dataclass(frozen=True)
class FoundationGateV1:
    name: str
    status: str
    evidence: str
    required_action: str
    blocks_training: bool = True
    blocks_protocol101_challenge: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UntouchedHoldoutReservationV1:
    name: str = UNTOUCHED_EVAL_BLOCK_NAME
    status: str = "reserved_pending_collection"
    reserved_on: str = "2026-05-24"
    start_after: str = "2026-05-24"
    data_status: str = "pending_new_data_collection"
    intended_use: str = "final_evaluation_only"
    split_label: str = "future_unseen_block_after_2026_05_24"
    paper_default_baseline: str = PAPER_DEFAULT_BASELINE
    exposed_diagnostic_splits: tuple[str, ...] = EXPOSED_DIAGNOSTIC_SPLITS
    forbidden_uses: tuple[str, ...] = REQUIRED_FORBIDDEN_HOLDOUT_USES
    allowed_uses: tuple[str, ...] = ("single_final_protocol101_challenge", "post_freeze_audit_report")
    notes: tuple[str, ...] = (
        "Existing Q3/Q4/Q1/March/recent splits remain diagnostics only.",
        "The reserved block must not influence model, label, threshold, or sizing choices.",
        "Final claims require the block to be collected/frozen before scoring.",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessEvaluationV1:
    role_label: str
    training_decision: str
    protocol101_challenge_decision: str
    training_blockers: tuple[str, ...]
    challenge_blockers: tuple[str, ...]
    gates: tuple[FoundationGateV1, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gates"] = [gate.to_dict() for gate in self.gates]
        return payload


def make_default_holdout_reservation(*, reserved_on: str = "2026-05-24") -> UntouchedHoldoutReservationV1:
    return UntouchedHoldoutReservationV1(reserved_on=reserved_on, start_after=reserved_on)


def validate_untouched_holdout_reservation(reservation: dict[str, Any] | UntouchedHoldoutReservationV1) -> dict[str, Any]:
    payload = reservation.to_dict() if isinstance(reservation, UntouchedHoldoutReservationV1) else dict(reservation)
    errors: list[str] = []
    warnings: list[str] = []

    name = str(payload.get("name", "")).strip()
    split_label = str(payload.get("split_label", "")).strip().lower()
    intended_use = str(payload.get("intended_use", "")).strip()
    data_status = str(payload.get("data_status", "")).strip()
    forbidden = {str(item) for item in payload.get("forbidden_uses", [])}

    if not name:
        errors.append("missing_holdout_name")
    if intended_use != "final_evaluation_only":
        errors.append("holdout_must_be_final_evaluation_only")
    if split_label in set(EXPOSED_DIAGNOSTIC_SPLITS):
        errors.append("holdout_uses_repeated_diagnostic_split")
    if "future" not in split_label and "unseen" not in split_label and data_status == "pending_new_data_collection":
        warnings.append("pending_holdout_split_label_should_identify_future_or_unseen_data")

    missing_forbidden = sorted(set(REQUIRED_FORBIDDEN_HOLDOUT_USES) - forbidden)
    if missing_forbidden:
        errors.append(f"missing_forbidden_uses:{','.join(missing_forbidden)}")

    data_available = data_status in {"available_frozen", "collected_frozen", "scored_once_frozen"}
    reservation_status = PASS if not errors else BLOCKED
    return {
        "status": reservation_status,
        "errors": errors,
        "warnings": warnings,
        "data_available": bool(data_available),
        "data_status": data_status,
        "split_label": split_label,
        "is_exposed_diagnostic_split": split_label in set(EXPOSED_DIAGNOSTIC_SPLITS),
    }


def evaluate_neural_training_readiness(gates: Iterable[FoundationGateV1 | dict[str, Any]]) -> ReadinessEvaluationV1:
    normalized = tuple(_coerce_gate(gate) for gate in gates)
    training_blockers = tuple(gate.name for gate in normalized if gate.blocks_training and gate.status != PASS)
    challenge_blockers = tuple(gate.name for gate in normalized if gate.blocks_protocol101_challenge and gate.status != PASS)
    return ReadinessEvaluationV1(
        role_label=ROLE_LABEL,
        training_decision=TRAINING_BLOCKED_DECISION if training_blockers else TRAINING_READY_DECISION,
        protocol101_challenge_decision=CHALLENGE_BLOCKED_DECISION if challenge_blockers else CHALLENGE_READY_DECISION,
        training_blockers=training_blockers,
        challenge_blockers=challenge_blockers,
        gates=normalized,
    )


def gate(
    name: str,
    status: str,
    evidence: str,
    required_action: str,
    *,
    blocks_training: bool = True,
    blocks_protocol101_challenge: bool = True,
) -> FoundationGateV1:
    clean_status = str(status)
    if clean_status not in {PASS, PARTIAL, BLOCKED}:
        raise ValueError(f"unknown gate status: {status}")
    return FoundationGateV1(
        name=name,
        status=clean_status,
        evidence=evidence,
        required_action=required_action,
        blocks_training=bool(blocks_training),
        blocks_protocol101_challenge=bool(blocks_protocol101_challenge),
    )


def gate_summary(gates: Iterable[FoundationGateV1 | dict[str, Any]]) -> dict[str, Any]:
    normalized = tuple(_coerce_gate(gate) for gate in gates)
    counts = {PASS: 0, PARTIAL: 0, BLOCKED: 0}
    for item in normalized:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {
        "counts": counts,
        "training_blockers": [item.name for item in normalized if item.blocks_training and item.status != PASS],
        "protocol101_challenge_blockers": [
            item.name for item in normalized if item.blocks_protocol101_challenge and item.status != PASS
        ],
    }


def _coerce_gate(item: FoundationGateV1 | dict[str, Any]) -> FoundationGateV1:
    if isinstance(item, FoundationGateV1):
        return item
    return FoundationGateV1(
        name=str(item["name"]),
        status=str(item["status"]),
        evidence=str(item.get("evidence", "")),
        required_action=str(item.get("required_action", "")),
        blocks_training=bool(item.get("blocks_training", True)),
        blocks_protocol101_challenge=bool(item.get("blocks_protocol101_challenge", True)),
    )
