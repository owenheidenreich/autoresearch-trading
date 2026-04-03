from __future__ import annotations

import json

from tools.live_order_parity_report import build_report


def _write_rows(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_live_order_parity_report_passes_with_linked_events(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    rows = [
        {
            "event": "entry_intent",
            "payload": {"intent_id": "i-1", "decision_id": "d-1", "action": 1, "qty": 1},
        },
        {
            "event": "entry_intent_applied",
            "payload": {"intent_id": "i-1", "decision_id": "d-1", "position_id": "pos-1"},
        },
        {
            "event": "entry_live",
            "payload": {"intent_id": "i-1", "decision_id": "d-1", "position_id": "pos-1"},
        },
        {
            "event": "ib_order_status",
            "payload": {"intent_id": "i-1", "decision_id": "d-1", "status": "Submitted"},
        },
        {
            "event": "ib_exec_details",
            "payload": {"intent_id": "i-1", "decision_id": "d-1", "exec_id": "abc"},
        },
    ]
    _write_rows(audit_path, rows)

    report = build_report(str(audit_path))
    assert report["pass"] is True
    assert report["counts"]["entry_intents"] == 1
    assert report["counts"]["ib_error_events"] == 0
    assert report["mismatches"] == []


def test_live_order_parity_report_flags_missing_execution_and_errors(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    rows = [
        {
            "event": "entry_intent",
            "payload": {"intent_id": "i-2", "decision_id": "d-2", "action": 2, "qty": 1},
        },
        {
            "event": "ib_error_event",
            "payload": {"error_code": 201, "error_string": "Order rejected"},
        },
    ]
    _write_rows(audit_path, rows)

    report = build_report(str(audit_path))
    assert report["pass"] is False
    assert report["counts"]["entry_intents"] == 1
    assert report["counts"]["ib_error_events"] == 1
    assert any(m.startswith("missing_entry_execution:") for m in report["mismatches"])
