from __future__ import annotations

import json
from pathlib import Path

from v4.foundation.model_experiment_preflight import (
    BLOCKED,
    BLOCKED_DECISION,
    PASS,
    READY_DECISION,
    REQUIRED_HYPOTHESIS_TOKENS,
    WARN,
    build_model_experiment_preflight,
)
from v4.model.neural_training_readiness import CHALLENGE_READY_DECISION, TRAINING_READY_DECISION


def _write_section_1_2_files(root: Path) -> None:
    for relative in (
        "v5/STATUS.md",
        "v5/AGENTS.md",
        "v4/docs/PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md",
        "v4/docs/DATA_CONTRACT.md",
        "v4/checks/paid_data_guard.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n")


def _write_neural_summary(root: Path, *, training_ready: bool = False, challenge_ready: bool = False) -> None:
    path = root / "v4/audit/autoresearch/unified_neural_training_readiness/summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "training_decision": TRAINING_READY_DECISION if training_ready else "neural_training_not_ready_foundation_gates_blocked",
                "protocol101_challenge_decision": CHALLENGE_READY_DECISION
                if challenge_ready
                else "protocol101_challenge_not_ready_foundation_gates_blocked",
                "training_blockers": [] if training_ready else ["Additional neural training pause"],
                "challenge_blockers": [] if challenge_ready else ["Formal validation controls"],
            }
        )
        + "\n"
    )


def test_section3_preflight_uses_neural_readiness_training_blockers(tmp_path: Path) -> None:
    _write_section_1_2_files(tmp_path)
    _write_neural_summary(tmp_path, training_ready=False)

    payload = build_model_experiment_preflight(tmp_path)
    checks = {check["name"]: check for check in payload["checks"]}

    assert payload["section3_model_experiment_decision"] == BLOCKED_DECISION
    assert payload["training_blockers"] == ["neural_training_readiness"]
    assert checks["neural_training_readiness"]["status"] == BLOCKED
    assert checks["hypothesis_packet_preregistered"]["status"] == WARN


def test_section3_preflight_requires_hypothesis_packet_for_actual_model_run(tmp_path: Path) -> None:
    _write_section_1_2_files(tmp_path)
    _write_neural_summary(tmp_path, training_ready=True, challenge_ready=True)

    payload = build_model_experiment_preflight(tmp_path, require_hypothesis=True)

    assert payload["section3_model_experiment_decision"] == BLOCKED_DECISION
    assert payload["training_blockers"] == ["hypothesis_packet_preregistered"]


def test_section3_preflight_accepts_complete_preregistration_packet(tmp_path: Path) -> None:
    _write_section_1_2_files(tmp_path)
    _write_neural_summary(tmp_path, training_ready=True, challenge_ready=True)
    packet = tmp_path / "hypothesis.md"
    packet.write_text("\n".join(REQUIRED_HYPOTHESIS_TOKENS) + "\n")

    payload = build_model_experiment_preflight(tmp_path, hypothesis_packet=packet, require_hypothesis=True)
    checks = {check["name"]: check for check in payload["checks"]}

    assert payload["section3_model_experiment_decision"] == READY_DECISION
    assert payload["training_blockers"] == []
    assert checks["hypothesis_packet_preregistered"]["status"] == PASS
