import csv
import json
import subprocess
import sys

from research_ops.diagnostics.feature_parity import candidate_row, logit_row


def test_candidate_row_without_replay_reference_is_not_comparable():
    row = {
        "timestamp": "2026-05-21T14:30:00+00:00",
        "run_id": "unit",
        "mode": "paper-submit",
        "event_type": "candidate_set",
        "candidate_count": 1,
        "candidate_sample": [{"contract_id": "A", "edge": 42.0, "features": {"entry_ask": 1.2}}],
        "candidate_gate_diagnostics": {"flat_score": 0.0},
    }

    result = candidate_row("sample.jsonl", 1, row)

    assert result is not None
    assert result["has_full_candidate_features"] == "true"
    assert result["has_surface_scores"] == "true"
    assert result["has_replay_candidate_set"] == "false"
    assert result["status"] == "not_comparable"


def test_logit_row_without_replay_logits_is_not_comparable():
    row = {
        "timestamp": "2026-05-21T14:30:00+00:00",
        "run_id": "unit",
        "mode": "paper-submit",
        "event_type": "model_decision",
        "model_decision": {"action": "wait", "score": 1.0, "threshold": 2.0, "wait_logit": 0.5},
    }

    result = logit_row("sample.jsonl", 1, row)

    assert result is not None
    assert result["has_replay_logits"] == "false"
    assert result["status"] == "not_comparable"


def test_cli_writes_feature_parity_artifacts(tmp_path):
    log_root = tmp_path / "logs"
    out_dir = tmp_path / "artifacts"
    log_root.mkdir()
    (log_root / "sample.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-21T14:30:00+00:00",
                        "run_id": "unit",
                        "mode": "paper-submit",
                        "event_type": "candidate_set",
                        "candidate_count": 1,
                        "candidate_sample": [{"contract_id": "A", "edge": 42.0}],
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-21T14:31:00+00:00",
                        "run_id": "unit",
                        "mode": "paper-submit",
                        "event_type": "model_decision",
                        "model_decision": {"action": "wait", "score": 1.0, "threshold": 2.0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "research_ops.diagnostics.feature_parity",
            "--log-root",
            str(log_root),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (out_dir / "feature_parity_report.md").exists()
    assert (out_dir / "feature_diff.csv").exists()
    assert (out_dir / "logit_diff.csv").exists()
    assert (out_dir / "candidate_set_diff.csv").exists()
    with (out_dir / "candidate_set_diff.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "not_comparable"
    report = (out_dir / "feature_parity_report.md").read_text(encoding="utf-8")
    assert "replay metrics not yet usable" in report
