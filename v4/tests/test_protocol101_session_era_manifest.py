"""Tests for Protocol101 session era manifest generation."""
from __future__ import annotations

import json
from pathlib import Path

from v4.scripts.build_protocol101_session_era_manifest import (
    UNASSIGNED_NO_MATCHING_RULE,
    UNASSIGNED_OVERLAPPING_RULES,
    UNASSIGNED_ERA,
    EraRule,
    build_manifest,
    collect_raw_vendor_sessions,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_build_manifest_assigns_known_eras_and_recorder_days(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    capture_root = tmp_path / "captures"
    _write_json(
        audit_root / "design" / "canonical_processed_session_manifest.json",
        {
            "status": "pass",
            "included_sessions": [
                {"session": "2025-07-01", "processed_file": "p1.pkl"},
                {"session": "2026-01-02", "processed_file": "p2.pkl"},
            ],
            "excluded_recorder_or_parity_sessions": ["2026-07-01"],
        },
    )
    _write_json(
        capture_root
        / "2026-07-01"
        / "protocol101-recorder-2026-07-01"
        / "ibkr_capture_quality.json",
        {
            "capture_id": "protocol101-recorder-2026-07-01",
            "checks": {"complete_regular_session": True},
            "evidence": {
                "checkpoint_count": 390,
                "expected_checkpoint_count": 390,
                "broker_order_endpoint_called": False,
            },
        },
    )

    payload = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[
            EraRule.parse("pre_program_oct2024_jun2025:2024-10-01:2025-06-30"),
            EraRule.parse("owned_jul_dec2025:2025-07-01:2025-12-31"),
            EraRule.parse("q1_2026_development:2026-01-01:2026-03-31"),
            EraRule.parse("post_q1_gap_apr_may2026:2026-04-01:2026-05-31"),
            EraRule.parse("confirmation_jun_jul2026:2026-06-01:2026-07-31"),
        ],
        allow_unassigned=False,
        raw_vendor_roots={},
    )

    eras = {record["session"]: record["era"] for record in payload["sessions"]}
    assert payload["status"] == "pass"
    assert eras == {
        "2025-07-01": "owned_jul_dec2025",
        "2026-01-02": "q1_2026_development",
        "2026-07-01": "confirmation_jun_jul2026",
    }
    july = next(record for record in payload["sessions"] if record["session"] == "2026-07-01")
    assert "ibkr_recorder_capture" in july["source_types"]
    assert "complete" in july["source_statuses"]
    assert payload["manifest_hash"]


def test_build_manifest_fails_closed_for_unassigned_sessions(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    capture_root = tmp_path / "captures"
    _write_json(
        audit_root / "design" / "canonical_processed_session_manifest.json",
        {"included_sessions": [{"session": "2024-01-02", "processed_file": "p.pkl"}]},
    )

    payload = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[EraRule.parse("owned_jul_dec2025:2025-07-01:2025-12-31")],
        allow_unassigned=False,
        raw_vendor_roots={},
    )

    assert payload["status"] == "fail"
    assert payload["unassigned_sessions"] == ["2024-01-02"]
    assert payload["unassigned_sessions_by_reason"] == {UNASSIGNED_NO_MATCHING_RULE: ["2024-01-02"]}
    assert payload["sessions"][0]["era"] == UNASSIGNED_ERA
    assert payload["sessions"][0]["era_assignment_reason"] == UNASSIGNED_NO_MATCHING_RULE


def test_build_manifest_reports_overlapping_rules_separately(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    capture_root = tmp_path / "captures"
    _write_json(
        audit_root / "design" / "canonical_processed_session_manifest.json",
        {"included_sessions": [{"session": "2025-07-02", "processed_file": "p.pkl"}]},
    )

    payload = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[
            EraRule.parse("first:2025-07-01:2025-07-31"),
            EraRule.parse("second:2025-07-02:2025-08-01"),
        ],
        allow_unassigned=False,
        raw_vendor_roots={},
    )

    assert payload["status"] == "fail"
    assert payload["unassigned_sessions_by_reason"] == {UNASSIGNED_OVERLAPPING_RULES: ["2025-07-02"]}
    assert payload["sessions"][0]["era_assignment_reason"] == UNASSIGNED_OVERLAPPING_RULES


def test_manifest_hash_changes_when_era_rule_changes(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    capture_root = tmp_path / "captures"
    _write_json(
        audit_root / "design" / "canonical_processed_session_manifest.json",
        {"included_sessions": [{"session": "2025-07-01", "processed_file": "p.pkl"}]},
    )

    first = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[EraRule.parse("owned_jul_dec2025:2025-07-01:2025-12-31")],
        allow_unassigned=False,
        raw_vendor_roots={},
    )
    second = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[EraRule.parse("alternate_era:2025-07-01:2025-12-31")],
        allow_unassigned=False,
        raw_vendor_roots={},
    )

    assert first["manifest_hash"] != second["manifest_hash"]


def test_collect_raw_vendor_sessions_adds_raw_products(tmp_path: Path) -> None:
    cbbo = tmp_path / "raw" / "cbbo"
    ohlcv = tmp_path / "raw" / "ohlcv"
    cbbo.mkdir(parents=True)
    ohlcv.mkdir(parents=True)
    (cbbo / "2024-10-01.cbbo-1m.parquet").write_text("placeholder")
    (cbbo / "2024-10-01.cbbo-1m.dbn.zst").write_text("placeholder")
    (ohlcv / "2024-10-01.ohlcv-1m.parquet").write_text("placeholder")

    sessions = collect_raw_vendor_sessions(
        {
            "databento_opra_cbbo_1m": cbbo,
            "databento_opra_ohlcv_1m": ohlcv,
        }
    )

    assert sorted(sessions) == ["2024-10-01"]
    item = sessions["2024-10-01"]
    assert item.source_types == {"raw_vendor_session"}
    assert item.source_statuses == {"raw_file_present"}
    assert item.evidence["databento_opra_cbbo_1m"]["file_count"] == 2
    assert item.evidence["databento_opra_ohlcv_1m"]["file_count"] == 1
