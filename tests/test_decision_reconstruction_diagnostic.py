import csv
import json
import subprocess
import sys

from research_ops.diagnostics.decision_reconstruction import matrix_row


def test_decision_row_with_missing_fields_is_insufficient():
    row = {
        "timestamp": "2026-05-21T14:30:00+00:00",
        "session": "2026-05-21",
        "run_id": "unit",
        "mode": "paper-submit",
        "event_type": "model_decision",
        "model_decision": {"action": "wait", "reason": "below_threshold"},
        "risk_gate": {"passed": True, "reason": "pass"},
    }

    result = matrix_row("sample.jsonl", 1, row)

    assert result["decision_event"] == "true"
    assert result["reconstruction_status"] == "insufficient"
    assert "candidate_count" in result["missing_fields"]
    assert "quote_timestamp" in result["missing_fields"]


def test_cli_writes_decision_reconstruction_artifacts(tmp_path):
    log_root = tmp_path / "logs"
    out_dir = tmp_path / "artifacts"
    log_root.mkdir()
    (log_root / "sample.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-21T14:30:00+00:00",
                "session": "2026-05-21",
                "run_id": "unit",
                "mode": "paper-submit",
                "event_type": "candidate_set",
                "candidate_count": 2,
                "candidate_sample": [{"contract_id": "A"}, {"contract_id": "B"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "research_ops.diagnostics.decision_reconstruction",
            "--log-root",
            str(log_root),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    matrix_path = out_dir / "decision_reconstruction_matrix.csv"
    assert matrix_path.exists()
    assert (out_dir / "missing_fields_report.md").exists()
    assert (out_dir / "log_schema_gap_report.md").exists()
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["reconstruction_status"] == "insufficient"
    report = (out_dir / "missing_fields_report.md").read_text(encoding="utf-8")
    assert "schema patch required" in report
