import csv
import json
import subprocess
import sys
from pathlib import Path

from research_ops.diagnostics.quote_age_truth import classify_event


def classify(row):
    return classify_event(
        row,
        source_file="test.jsonl",
        line_number=1,
        max_age_ms=1500.0,
        tolerance_ms=250.0,
    )


def test_matching_raw_timestamps_are_trustworthy():
    row = {
        "event_type": "risk_gate",
        "timestamp": "2026-05-21T14:30:01.000000+00:00",
        "market_snapshot": {
            "option_nbbo": {
                "bid": 1.0,
                "ask": 1.1,
                "quote_age_ms": 1000,
                "timestamp": "2026-05-21T14:30:00.000000+00:00",
            }
        },
    }

    result = classify(row)

    assert result["classification"] == "trustworthy"
    assert result["trust_status"] == "pass"
    assert result["recomputed_quote_age_ms"] == "1000.000"


def test_zero_age_without_raw_timestamp_is_placeholder_unknown():
    row = {
        "event_type": "risk_gate",
        "timestamp": "2026-05-21T14:30:01.000000+00:00",
        "market_snapshot": {"option_nbbo": {"bid": 1.0, "ask": 1.1, "quote_age_ms": 0}},
    }

    result = classify(row)

    assert result["classification"] == "placeholder"
    assert result["trust_status"] == "unknown"
    assert result["reason"] == "missing_raw_quote_timestamp_persisted_zero"


def test_missing_persisted_quote_age_is_missing():
    row = {
        "event_type": "risk_gate",
        "timestamp": "2026-05-21T14:30:01.000000+00:00",
        "market_snapshot": {
            "option_nbbo": {
                "bid": 1.0,
                "ask": 1.1,
                "timestamp": "2026-05-21T14:30:00.000000+00:00",
            }
        },
    }

    result = classify(row)

    assert result["classification"] == "missing"
    assert result["trust_status"] == "unknown"


def test_stale_age_is_classified_as_stale():
    row = {
        "event_type": "risk_gate",
        "timestamp": "2026-05-21T14:30:03.000000+00:00",
        "market_snapshot": {
            "option_nbbo": {
                "quote_age_ms": 3000,
                "timestamp": "2026-05-21T14:30:00.000000+00:00",
            }
        },
    }

    result = classify(row)

    assert result["classification"] == "stale"
    assert result["trust_status"] == "fail"


def test_missing_raw_timestamp_with_nonzero_age_is_unreconstructable():
    row = {
        "event_type": "risk_gate",
        "timestamp": "2026-05-21T14:30:01.000000+00:00",
        "market_snapshot": {"option_nbbo": {"quote_age_ms": 500}},
    }

    result = classify(row)

    assert result["classification"] == "unreconstructable"
    assert result["trust_status"] == "unknown"
    assert result["reason"] == "missing_raw_quote_timestamp"


def test_cli_writes_csv_markdown_and_json_artifacts(tmp_path):
    log_root = tmp_path / "logs"
    out_dir = tmp_path / "artifacts"
    log_root.mkdir()
    log_path = log_root / "sample.jsonl"
    rows = [
        {
            "event_type": "risk_gate",
            "timestamp": "2026-05-21T14:30:01.000000+00:00",
            "run_id": "unit",
            "mode": "paper",
            "broker_order_endpoint_called": False,
            "market_snapshot": {
                "option_nbbo": {
                    "quote_age_ms": 1000,
                    "timestamp": "2026-05-21T14:30:00.000000+00:00",
                }
            },
        },
        {
            "event_type": "risk_gate",
            "timestamp": "2026-05-21T14:30:01.000000+00:00",
            "run_id": "unit",
            "mode": "paper",
            "broker_order_endpoint_called": False,
            "market_snapshot": {"option_nbbo": {"quote_age_ms": 0}},
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "research_ops.diagnostics.quote_age_truth",
            "--log-root",
            str(log_root),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    csv_path = out_dir / "quote_age_rows.csv"
    markdown_path = out_dir / "quote_age_summary.md"
    json_path = out_dir / "quote_age_summary.json"
    assert csv_path.exists()
    assert markdown_path.exists()
    assert json_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["classification"] for row in csv_rows] == ["trustworthy", "placeholder"]

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["counts"]["classification_counts"]["trustworthy"] == 1
    assert summary["counts"]["classification_counts"]["placeholder"] == 1
    assert "Missing raw quote timestamp is never treated as pass." in markdown_path.read_text(encoding="utf-8")
