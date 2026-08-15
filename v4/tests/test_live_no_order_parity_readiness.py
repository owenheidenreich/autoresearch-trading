from __future__ import annotations

from datetime import datetime
from pathlib import Path

from v4.foundation.live_no_order_parity_readiness import build_live_no_order_parity_readiness


def _write_scaffolds(root: Path) -> None:
    for relative in (
        "v4/live/protocol166_parity_contract.py",
        "v4/scripts/run_protocol166_live_training_parity_contract.py",
        "v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py",
        "v4/audit/autoresearch/formal_validation_governance/summary.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "quote_freshness_from_ticker\n" if "protocol158" in relative else "{}\n"
        path.write_text(text)


def test_live_no_order_parity_waits_when_market_is_closed(tmp_path: Path) -> None:
    _write_scaffolds(tmp_path)
    now = datetime.fromisoformat("2026-05-24T12:00:00-04:00")

    payload = build_live_no_order_parity_readiness(tmp_path, now=now)

    assert payload["decision"] == "live_no_order_full_action_parity_waiting_for_open_market_session"
    assert payload["market_status"]["weekday"] == "Sunday"
    assert payload["broker_endpoint_called"] is False
    assert payload["live_orders"] is False


def test_live_no_order_parity_ready_to_run_during_regular_session(tmp_path: Path) -> None:
    _write_scaffolds(tmp_path)
    now = datetime.fromisoformat("2026-05-26T10:00:00-04:00")

    payload = build_live_no_order_parity_readiness(tmp_path, now=now)

    assert payload["decision"] == "live_no_order_full_action_parity_ready_to_run_no_order_session"
    assert payload["market_status"]["is_regular_market_session"] is True
