"""Tests for the Protocol101 owned raw acceptance batch runner."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from v4.scripts.run_protocol101_owned_raw_acceptance_batch import (
    SessionBatchRecord,
    build_payload,
    builder_command,
    run_command,
    sessions_between,
    status_from_build_summary,
    terminal_build_failures,
)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        start_date="2024-10-01",
        end_date="2024-10-02",
        out_dir=tmp_path / "out",
        raw_root=tmp_path / "raw",
        normalized_dir=tmp_path / "normalized",
        processed_dir=tmp_path / "processed",
        official_spx_dir=tmp_path / "spx",
        official_vix_dir=tmp_path / "vix",
        context_mode="official",
        feature_contract="protocol101-live-v1",
        timeout_seconds=10,
        acceptance_timeout_seconds=10,
        role="diagnostics_only",
        skip_existing=True,
        compute_live_policy_labels=True,
        run_acceptance=False,
    )


def test_sessions_between_excludes_weekends_and_market_holidays() -> None:
    assert sessions_between("2024-10-04", "2024-10-08") == [
        "2024-10-04",
        "2024-10-07",
        "2024-10-08",
    ]
    assert sessions_between("2024-11-27", "2024-11-29") == [
        "2024-11-27",
        "2024-11-29",
    ]


def test_status_from_build_summary_classifies_builder_outcomes() -> None:
    assert status_from_build_summary({"sessions_built": 1}, "2024-10-01") == "built"
    assert (
        status_from_build_summary({"sessions_existing_skipped": ["2024-10-01"]}, "2024-10-01")
        == "skipped_existing"
    )
    assert (
        status_from_build_summary({"sessions_skipped": ["2024-10-01"]}, "2024-10-01")
        == "skipped_missing_raw"
    )
    assert status_from_build_summary({}, "2024-10-01") == "no_rows_built"


def test_builder_command_preserves_live_contract_and_label_flags(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cmd = builder_command(args, "2024-10-01", tmp_path / "summary.json")

    assert cmd[:3] == [sys.executable, "-m", "v4.scripts.build_databento_neural_dataset"]
    assert "--feature-contract" in cmd
    assert "protocol101-live-v1" in cmd
    assert "--compute-live-policy-labels" in cmd
    assert "--skip-existing" in cmd


def test_build_payload_fails_when_acceptance_fails(tmp_path: Path) -> None:
    args = _args(tmp_path)
    record = SessionBatchRecord(
        session="2024-10-01",
        status="built",
        command=["python"],
        timeout_seconds=10,
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        returncode=0,
        stdout_log="stdout",
        stderr_log="stderr",
        build_summary_path="summary",
        sessions_built=1,
        sessions_skipped=[],
        sessions_existing_skipped=[],
    )

    payload = build_payload(args, [record], acceptance={"status": "fail"})

    assert payload["status"] == "fail"
    assert payload["counts_by_status"] == {"built": 1}


def test_build_payload_skips_acceptance_status_without_double_counting_build_failure(tmp_path: Path) -> None:
    args = _args(tmp_path)
    record = SessionBatchRecord(
        session="2024-10-01",
        status="timeout",
        command=["python"],
        timeout_seconds=10,
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:10+00:00",
        duration_seconds=10.0,
        returncode=None,
        stdout_log="stdout",
        stderr_log="stderr",
        build_summary_path="summary",
        sessions_built=0,
        sessions_skipped=[],
        sessions_existing_skipped=[],
        error="timeout_after_10s",
    )

    payload = build_payload(
        args,
        [record],
        acceptance={
            "status": "not_run_due_to_build_failures",
            "terminal_failures": ["2024-10-01"],
        },
    )

    assert payload["status"] == "fail"
    assert terminal_build_failures([record]) == ["2024-10-01"]
    assert payload["acceptance"]["status"] == "not_run_due_to_build_failures"


def test_run_command_records_timeout(tmp_path: Path) -> None:
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"

    returncode, error = run_command(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_seconds=1,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )

    assert returncode is None
    assert error == "timeout_after_1s"
    assert "TIMEOUT after 1s" in stderr_log.read_text()
