from __future__ import annotations

import json
from pathlib import Path

from v4.scripts.run_protocol147_protocol101_morning_session import main as morning_main


def test_protocol147_dry_run_writes_analyzable_trade_log(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    args = [
        "prog",
        "--dry-run",
        "--session-date",
        "2026-05-15",
        "--run-id",
        "test_morning_session",
        "--out-root",
        str(tmp_path / "audit"),
        "--trade-log-root",
        str(tmp_path / "logs"),
        "--max-cycles",
        "0",
    ]
    monkeypatch.setattr("sys.argv", args)

    assert morning_main() == 0

    summary = json.loads(
        (tmp_path / "audit" / "2026-05-15" / "test_morning_session" / "summary.json").read_text()
    )
    log_path = Path(summary["trade_log"]["jsonl"])

    assert summary["decision"] == "dry_run_logged_timing_evidence"
    assert summary["broker_order_endpoint_called"] is False
    assert summary["paper_orders_submitted"] is False
    assert summary["trade_log"]["validation"]["status"] == "pass"
    assert summary["timing_evidence"]["summary"]["decision"] == "blocked_protocol155_no_closed_one_contract_paper_trades_yet"
    assert log_path.exists()
    assert log_path.with_suffix(".csv").exists()
