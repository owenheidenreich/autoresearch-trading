from __future__ import annotations

import json
from pathlib import Path

from v4.model.protocol101_repair_artifacts import sha256_file
from v4.scripts import (
    run_protocol101_full_trader_stage1_reference_multiplicity_validation
    as validation,
)


REQUIRED_OUTPUTS = {
    "preregistration.json",
    "source_inventory.json",
    "implementation_manifest.json",
    "changed_files.json",
    "reference_v5_call_graph.json",
    "synthetic_reference_validation.json",
    "d1_contract_validation.json",
    "d5_contract_validation.json",
    "d6_authority_receipt.json",
    "maxT_contract.json",
    "maxT_schedule_manifest.json",
    "maxT_synthetic_validation.json",
    "readiness_matrix.csv",
    "test_matrix.csv",
    "test_results.json",
    "progress.json",
    "summary.json",
    "routing_decision.json",
    "report.md",
    "hashes.sha256",
}


def _copy_preregistration(target: Path) -> None:
    validation.seed_preregistration_namespace(
        validation.DEFAULT_OUT,
        target,
    )


def test_reference_call_graph_is_v5_only_and_legacy_is_labeled() -> None:
    result = validation.validate_reference_call_graph()
    assert result["status"] == "PASS"
    assert result["predicates"]["reference_replay_calls_v5_simulator"]
    assert result["predicates"]["reference_replay_does_not_call_v4"]
    assert result["predicates"]["legacy_cli_explicitly_labeled"]


def test_reference_call_graph_detects_injected_v4_edge(monkeypatch) -> None:
    actual = validation._called_names

    def injected(function):
        names = actual(function)
        if function.__name__ == "replay_reference_v5":
            names.discard("simulate_serial_candidates_v5")
            names.add("simulate_serial_candidates")
        return names

    monkeypatch.setattr(validation, "_called_names", injected)
    result = validation.validate_reference_call_graph()
    assert result["status"] == "FAIL"


def test_no_fit_readiness_packet_is_truthful_complete_and_hashed(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "attempt001"
    _copy_preregistration(out_dir)
    result = validation.build_terminal_packet(
        out_dir,
        replicates=32,
        execute_tests=False,
    )
    assert result["status"] == "PASS"
    assert result["route"] == (
        "reference_multiplicity_machinery_repair_complete_"
        "pending_independent_acceptance"
    )
    assert REQUIRED_OUTPUTS.issubset(
        {path.name for path in out_dir.iterdir() if path.is_file()}
    )
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["status"] == (
        "reference_multiplicity_machinery_ready_"
        "pending_independent_acceptance"
    )
    assert summary["side_effects"] == {
        "campaign_model_fit": False,
        "campaign_model_score": False,
        "campaign_economic_replay": False,
        "real_reference_execution": False,
        "G1_G8_aggregation": False,
        "ranking_or_selection": False,
        "seed_45_or_G9": False,
        "protected_holdout_access": False,
        "sealed_recorder_evidence_access": False,
        "learned_exits": False,
        "broker_or_API_call": False,
        "paper_submit": False,
        "paid_download": False,
        "promotion_or_default_change": False,
        "runtime_or_launchd_change": False,
        "real_money_path": False,
    }
    routing = json.loads((out_dir / "routing_decision.json").read_text())
    assert routing["independent_acceptance_started"] is False
    assert routing["campaign_economics_started"] is False
    hashes = {}
    for line in (out_dir / "hashes.sha256").read_text().splitlines():
        digest, separator, name = line.partition("  ")
        assert separator
        hashes[name] = digest
    expected = {
        path.name
        for path in out_dir.iterdir()
        if path.is_file() and path.name != "hashes.sha256"
    }
    assert set(hashes) == expected
    assert all(
        sha256_file(out_dir / name) == digest
        for name, digest in hashes.items()
    )
    resumed = validation.build_terminal_packet(
        out_dir,
        replicates=32,
        execute_tests=False,
    )
    assert resumed["status"] == "PASS"
    assert json.loads(
        (out_dir / "synthetic_reference_validation.json").read_text()
    )["immutable_random_packet"]["status"] == (
        "verified_existing_complete_packet_skipped"
    )
    assert json.loads(
        (out_dir / "d5_contract_validation.json").read_text()
    )["immutable_packet"]["status"] == (
        "verified_existing_complete_packet_skipped"
    )
