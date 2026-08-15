from __future__ import annotations

from pathlib import Path

import pandas as pd

from v4.scripts.run_protocol116_protocol101_targeted_1s_request import (
    build_manifest,
    build_requests,
    missing_rows,
)


def test_missing_rows_excludes_audited_and_keeps_critical_scope() -> None:
    rows = pd.DataFrame(
        [
            {"split": "q3_2025", "session": "2025-07-01", "audit_status": "missing_1s_session", "raw_symbol": "A"},
            {"split": "q1_2026", "session": "2026-01-02", "audit_status": "missing_1s_session", "raw_symbol": "B"},
            {"split": "q4_2025", "session": "2025-10-17", "audit_status": "audited", "raw_symbol": "C"},
        ]
    )

    critical = missing_rows(rows, scope="critical_missing")
    all_missing = missing_rows(rows, scope="all_missing")

    assert critical["raw_symbol"].tolist() == ["A"]
    assert set(all_missing["raw_symbol"]) == {"A", "B"}


def test_build_requests_groups_symbols_and_marks_existing_file(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {"split": "q3_2025", "session": "2025-07-01", "audit_status": "missing_1s_session", "raw_symbol": "A"},
            {"split": "q3_2025", "session": "2025-07-01", "audit_status": "missing_1s_session", "raw_symbol": "A"},
            {"split": "q3_2025", "session": "2025-07-01", "audit_status": "missing_symbol", "raw_symbol": "B"},
        ]
    )
    (tmp_path / "2025-07-01.cbbo-1s.parquet").write_text("placeholder")

    [request] = build_requests(
        rows,
        cbbo_1s_dir=tmp_path,
        estimates={"2025-07-01": 0.12},
        fallback_rates={"cbbo-1s": 0.01, "cmbp-1": 0.02},
    )

    assert request.schema == "cbbo-1s"
    assert request.symbols == 2
    assert request.selected_trades == 3
    assert request.missing_1s_session_rows == 2
    assert request.missing_symbol_rows == 1
    assert request.existing_1s_file is True
    assert request.estimated_cost_usd == 0.12


def test_manifest_requires_exact_approval_text() -> None:
    request = []
    manifest = build_manifest(
        decision="ready_for_explicit_user_approval",
        approval_text="exact approval",
        requests=request,
        estimated_total=1.25,
        hard_cap=3.0,
        metadata_status="ok",
        fallback_rates={"cbbo-1s": 0.01, "cmbp-1": 0.02},
        replay_json=Path("report.json"),
        scope="all_missing",
    )

    assert manifest["paid_data_downloaded"] is False
    assert manifest["market_data_download_endpoint_called"] is False
    assert manifest["approval_required"]["exact_approval_text"] == "exact approval"
    assert manifest["cost_estimate"]["hard_cap_usd"] == 3.0
