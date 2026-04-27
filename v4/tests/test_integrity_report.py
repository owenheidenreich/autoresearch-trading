"""End-to-end test of the pipeline-integrity report generator."""
from __future__ import annotations

from pathlib import Path

import pytest

from v4.runner import run_pipeline_integrity_report

FIXTURE = Path(__file__).parent / "fixtures" / "optionsdx_spx_sample.csv"


def test_pipeline_integrity_report_runs_end_to_end(tmp_path: Path) -> None:
    report = run_pipeline_integrity_report(
        optionsdx_input=FIXTURE,
        output_dir=tmp_path,
    )

    # Fundamental Phase-0 milestone: the substrate runs end-to-end without
    # exceptions and produces an audit-ready report.
    md = tmp_path / "PIPELINE_INTEGRITY_REPORT.md"
    assert md.exists()
    assert md.read_text().startswith("# Pipeline Integrity Report")

    jsonl = tmp_path / "integrity_runs.jsonl"
    assert jsonl.exists()

    # Report fields populated
    assert report.normalized_rows == 10  # 5 fixture rows × 2 sides
    assert len(report.normalized_fingerprint) == 64
    assert report.deterministic_rebuild_match is True


def test_pipeline_integrity_phase0_substrate_is_green(tmp_path: Path) -> None:
    """Phase-0 exit criterion: the substrate is GREEN on the OptionsDX
    fixture. This is THE Phase-0 milestone test."""
    report = run_pipeline_integrity_report(
        optionsdx_input=FIXTURE,
        output_dir=tmp_path,
    )
    assert report.all_passed, (
        f"Phase-0 substrate is RED. Sanity: "
        f"{[(c.name, c.passed) for c in report.sanity_checks]}; "
        f"Integrity: {[(c.name, c.passed) for c in report.integrity_checks]}; "
        f"Greeks: {report.greeks_reconciliation}; "
        f"deterministic: {report.deterministic_rebuild_match}; "
        f"leak detector: {report.leak_detector_works}"
    )


def test_jsonl_audit_record_appended(tmp_path: Path) -> None:
    """Multiple runs append to the JSONL — an audit log is preserved."""
    run_pipeline_integrity_report(optionsdx_input=FIXTURE, output_dir=tmp_path)
    run_pipeline_integrity_report(optionsdx_input=FIXTURE, output_dir=tmp_path)
    jsonl = tmp_path / "integrity_runs.jsonl"
    lines = jsonl.read_text().strip().split("\n")
    assert len(lines) == 2
