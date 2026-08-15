"""Tests for Protocol 101 targeted high-resolution download plumbing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


APPROVAL = "I approve this exact Protocol 101 high-resolution batch."


def _manifest(tmp_path: Path) -> tuple[Path, Path]:
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "session": "2025-07-01",
                        "raw_symbol": "SPXW  250701C06165000",
                        "audit_status": "missing_1s_session",
                    }
                ]
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "116_protocol101_targeted_1s_request",
                "source_replay_json": str(replay),
                "requested_data": {
                    "sessions": ["2025-07-01"],
                    "dataset": "OPRA.PILLAR",
                    "schemas": ["cbbo-1s"],
                    "session_count": 1,
                    "session_count_by_schema": {"cbbo-1s": 1},
                },
                "requests": [
                    {
                        "session": "2025-07-01",
                        "schema": "cbbo-1s",
                        "estimated_cost_usd": 0.01,
                    }
                ],
                "cost_estimate": {"hard_cap_usd": 1.0, "estimated_total_usd": 0.01},
                "approval_required": {"exact_approval_text": APPROVAL},
            }
        )
    )
    return manifest, replay


def test_targeted_highres_download_requires_approval_before_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import v4.scripts.download_protocol101_targeted_highres as script

    manifest, _ = _manifest(tmp_path)

    def fail_download(*_: object, **__: object) -> pd.DataFrame:
        raise AssertionError("download endpoint should not be reached before approval")

    monkeypatch.setattr(script, "_client", lambda: object())
    monkeypatch.setattr(script, "_download", fail_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_protocol101_targeted_highres",
            "--manifest",
            str(manifest),
            "--approval-manifest",
            str(manifest),
            "--out-root",
            str(tmp_path / "raw"),
            "--audit-out",
            str(tmp_path / "audit.jsonl"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--use-manifest-estimates",
        ],
    )

    with pytest.raises(SystemExit, match="paid-data approval required"):
        script.main()


def test_targeted_highres_download_accepts_exact_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import v4.scripts.download_protocol101_targeted_highres as script

    manifest, _ = _manifest(tmp_path)
    audit_out = tmp_path / "audit.jsonl"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(script, "_client", lambda: object())
    monkeypatch.setattr(script, "_download", lambda *_, **__: pd.DataFrame({"row": [1, 2]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_protocol101_targeted_highres",
            "--manifest",
            str(manifest),
            "--approval-manifest",
            str(manifest),
            "--approval-text",
            APPROVAL,
            "--out-root",
            str(tmp_path / "raw"),
            "--audit-out",
            str(audit_out),
            "--summary-out",
            str(summary_out),
            "--use-manifest-estimates",
        ],
    )

    assert script.main() == 0
    assert audit_out.exists()
    assert summary_out.exists()
    record = json.loads(audit_out.read_text().splitlines()[0])
    assert record["schema"] == "cbbo-1s"
    assert record["rows"] == 2
