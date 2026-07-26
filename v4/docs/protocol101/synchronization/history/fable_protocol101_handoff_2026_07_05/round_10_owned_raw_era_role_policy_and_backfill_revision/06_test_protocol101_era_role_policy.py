"""Tests for Protocol101 era role policy artifact."""
from __future__ import annotations

import json
from pathlib import Path

from v4.scripts.build_protocol101_era_role_policy import (
    DEFAULT_POLICY,
    build_policy_artifact,
    validate_policy,
)


def test_validate_policy_requires_every_manifest_era() -> None:
    manifest = {"counts_by_era": {"pre_program_oct2024_jun2025": 1, "new_era": 1}}

    result = validate_policy(DEFAULT_POLICY, manifest)

    assert result["pass"] is False
    assert result["missing_policy_eras"] == ["new_era"]


def test_build_policy_artifact_passes_for_default_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "summary.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_hash": "abc123",
                "counts_by_era": {
                    "pre_program_oct2024_jun2025": 10,
                    "owned_jul_dec2025": 20,
                    "q1_2026_development": 30,
                    "post_q1_gap_apr_may2026": 5,
                    "confirmation_jun_jul2026": 3,
                },
            }
        )
    )

    payload = build_policy_artifact(manifest_path)

    assert payload["status"] == "pass"
    assert payload["session_manifest_hash"] == "abc123"
    assert payload["policy"]["confirmation_jun_jul2026"]["permitted_roles"] == [
        "confirmation_one_shot",
        "report_only",
    ]
    assert "train" not in payload["policy"]["q1_2026_development"]["permitted_roles"]


def test_default_policy_never_allows_unassigned_sessions() -> None:
    assert DEFAULT_POLICY["unassigned_requires_decision"]["permitted_roles"] == []
