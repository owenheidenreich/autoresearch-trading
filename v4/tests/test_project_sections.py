from __future__ import annotations

from pathlib import Path

from v4.foundation.project_sections import (
    BLOCKED,
    PASS,
    build_readiness_payload,
    project_sections,
    unguarded_paid_download_scripts,
)


def test_project_sections_are_explicit_and_ordered() -> None:
    sections = project_sections()

    assert [section.id for section in sections] == [1, 2, 3, 4, 5]
    assert sections[0].name == "architecture_foundation"
    assert sections[1].name == "data_acquisition_preparation"
    assert sections[2].name == "model_experiments_training_testing_validating"
    assert "paid market-data endpoint calls" in sections[1].forbidden_without_approval


def test_paid_download_scan_flags_unguarded_paid_endpoint(tmp_path: Path) -> None:
    scripts = tmp_path / "v4" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "download_bad.py").write_text("client.timeseries.get_range()\n")
    (scripts / "download_good.py").write_text(
        "from v4.checks.paid_data_guard import require_paid_data_approval\n"
        "require_paid_data_approval\n"
        "client.timeseries.get_range()\n"
    )

    assert unguarded_paid_download_scripts(tmp_path) == ["v4/scripts/download_bad.py"]


def test_current_paid_download_scripts_are_guarded() -> None:
    assert unguarded_paid_download_scripts(Path(".")) == []


def test_section_readiness_marks_sections_one_two_and_four_ready_but_blocks_model_hill_climb() -> None:
    payload = build_readiness_payload(Path("."))

    assert payload["sections"][1]["status"] == PASS
    assert payload["sections"][2]["status"] == PASS
    assert payload["sections"][4]["status"] == PASS
    assert payload["sections"][3]["status"] == BLOCKED
    assert payload["section_1_2_decision"] == "section_1_2_ready"
    assert payload["model_hill_climb_decision"] == "model_hill_climb_blocked_until_truth_gates_pass"
