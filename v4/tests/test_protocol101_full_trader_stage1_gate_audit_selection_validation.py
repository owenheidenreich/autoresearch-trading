from __future__ import annotations

import json

from v4.scripts.run_protocol101_full_trader_stage1_gate_audit_selection_validation import (
    build_ft1c3_terminal_packet,
    build_terminal_packet,
    evidence_schema,
    run_authority_root_chain_matrix,
    run_synthetic_matrix,
)


def test_complete_synthetic_matrix_has_exact_required_cases() -> None:
    result = run_synthetic_matrix()
    assert result["case_count"] == 32
    assert result["scenario_count"] == 52
    assert result["passed_case_count"] == 32
    assert result["failed_case_count"] == 0
    assert result["all_pass"] is True
    assert result["baseline"]["unit_count"] == 420
    assert result["baseline"]["row_count"] == 28


def test_evidence_schema_requires_primary_rungs_and_G8_report_only() -> None:
    schema = evidence_schema()
    assert schema["primary_evidence"] == {
        "fee": "3.00",
        "fill": "pessimistic_executable",
        "noise": "1.0x",
    }
    assert schema["G8_role"] == "report_only"
    assert schema["G9"] is False


def test_terminal_packet_contains_all_required_outputs_without_running_tests(
    tmp_path,
) -> None:
    source = (
        "v4/audit/autoresearch/"
        "protocol101_full_trader_stage1_gate_audit_selection_machinery_attempt001"
    )
    from pathlib import Path
    from shutil import copy2

    root = Path(__file__).resolve().parents[2]
    for name in (
        "preregistration.json",
        "source_inventory.json",
        "progress.json",
        "progress_initial.json",
        "preregistration_freeze.sha256",
    ):
        copy2(root / source / name, tmp_path / name)
    summary = build_terminal_packet(
        out_dir=tmp_path,
        execute_tests=False,
    )
    assert summary["routing_decision"].endswith(
        "complete_pending_independent_acceptance"
    )
    required = {
        "preregistration.json",
        "source_inventory.json",
        "implementation_manifest.json",
        "changed_files.json",
        "gate_contract.json",
        "evidence_schema.json",
        "gate_truth_table.json",
        "audit_independence_report.json",
        "selection_contract.json",
        "graph_contract.json",
        "synthetic_420_unit_manifest.json",
        "synthetic_case_results.json",
        "readiness_matrix.csv",
        "test_matrix.csv",
        "test_results.json",
        "progress.json",
        "summary.json",
        "routing_decision.json",
        "report.md",
        "hashes.sha256",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    decision = json.loads((tmp_path / "routing_decision.json").read_text())
    assert decision["campaign_execution_authorized"] is False
    assert decision["G9_authorized"] is False


def test_ft1c3_root_matrix_closes_exact_and_additional_attacks() -> None:
    result = run_authority_root_chain_matrix()
    assert result["honest_chain_passed"] is True
    assert result["exact_reproducer_count"] == 13
    assert result["exact_reproducer_passed_count"] == 13
    assert result["additional_case_count"] >= 80
    assert result["additional_passed_count"] == result["additional_case_count"]
    assert result["all_pass"] is True


def test_ft1c3_terminal_packet_remains_checksum_valid() -> None:
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "v4/audit/autoresearch/"
        "protocol101_full_trader_stage1_authority_root_chain_repair_attempt003"
    )
    completed = subprocess.run(
        ["shasum", "-a", "256", "-c", "hashes.sha256"],
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    summary = json.loads((source / "summary.json").read_text())
    assert summary["routing_decision"] == (
        "authority_root_chain_repair_complete_pending_reacceptance"
    )
