from __future__ import annotations

from pathlib import Path

from v4.scripts.run_protocol272_fill_model_readiness import should_scan_jsonl


def test_fill_readiness_scans_only_paper_live_runtime_logs() -> None:
    assert should_scan_jsonl(Path("v4/logs/paper_trading/session.jsonl")) is True
    assert should_scan_jsonl(Path("v4/audit/autoresearch/runtime_shadow/session.jsonl")) is True
    assert should_scan_jsonl(Path("v4/audit/autoresearch/unified_replay/decisions_slippage_0_00.jsonl")) is False
    assert should_scan_jsonl(Path("v4/audit/databento/raw_download.jsonl")) is False
