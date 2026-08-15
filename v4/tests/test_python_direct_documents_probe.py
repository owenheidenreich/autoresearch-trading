from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import socket

import pytest

from v4.ops.tracka.probe_python_documents_access import (
    _network_blocked,
    _stable_hash,
    run_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFINITION_PATH = (
    REPO_ROOT
    / "v4/audit/autoresearch/pathd_phase0b_tracka_live_capture_2026_08_04"
    / "2026-08-05/midday/definitions/current_session_definitions.parquet"
)


def test_network_guard_refuses_socket_connections() -> None:
    with _network_blocked(), pytest.raises(RuntimeError, match="network disabled"):
        socket.create_connection(("127.0.0.1", 9))


def test_probe_runs_both_capture_entrypoints_without_network(tmp_path: Path) -> None:
    if not DEFINITION_PATH.is_file():
        pytest.skip("owned 2026-08-05 definition capture is unavailable")
    receipt = tmp_path / "probe.json"
    payload = run_probe(
        repo_root=REPO_ROOT,
        receipt_path=receipt,
        session=date(2026, 8, 5),
        definition_path=DEFINITION_PATH,
        require_receipt_in_repo=False,
    )
    stored = json.loads(receipt.read_text())
    assert payload["status"] == "PASS"
    assert payload["checks"]["market_dry_run"]["symbol_count"] == 510
    assert payload["hard_stops"]["network_allowed"] is False
    assert stored["receipt_sha256"] == _stable_hash(stored)


def test_probe_rejects_receipt_outside_repo(tmp_path: Path) -> None:
    if not DEFINITION_PATH.is_file():
        pytest.skip("owned 2026-08-05 definition capture is unavailable")
    with pytest.raises(RuntimeError, match="inside the repository"):
        run_probe(
            repo_root=REPO_ROOT,
            receipt_path=tmp_path / "probe.json",
            session=date(2026, 8, 5),
            definition_path=DEFINITION_PATH,
        )
