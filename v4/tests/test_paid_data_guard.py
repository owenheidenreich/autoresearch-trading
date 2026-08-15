"""Tests for paid market-data download guardrails."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from v4.checks.paid_data_guard import exact_approval_text, require_paid_data_approval


APPROVAL = "I approve this exact paid data batch."


def _manifest(path: Path, approval: str = APPROVAL) -> Path:
    path.write_text(
        (
            "{\n"
            '  "approval_required": {\n'
            f'    "exact_approval_text": "{approval}"\n'
            "  }\n"
            "}\n"
        )
    )
    return path


def test_paid_data_guard_accepts_only_exact_manifest_text(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")

    assert exact_approval_text(manifest) == APPROVAL
    require_paid_data_approval(
        manifest_path=manifest,
        approval_text=APPROVAL,
        operation="unit-test download",
    )

    with pytest.raises(SystemExit, match="paid-data approval required"):
        require_paid_data_approval(
            manifest_path=manifest,
            approval_text="approve",
            operation="unit-test download",
        )


def test_thetadata_download_mode_requires_approval_before_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import v4.scripts.download_thetadata_index_bars as script

    manifest = _manifest(tmp_path / "manifest.json")

    def fail_client() -> object:
        raise AssertionError("ThetaData client should not initialize before approval")

    monkeypatch.setattr(script, "_client", fail_client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_thetadata_index_bars",
            "--start-date",
            "2024-07-01",
            "--end-date",
            "2024-07-02",
            "--approval-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(SystemExit, match="paid-data approval required"):
        script.main()


def test_databento_selected_1s_download_requires_approval_before_get_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import v4.scripts.download_databento_cbbo_1s_selected as script

    manifest = _manifest(tmp_path / "manifest.json")
    selected_dir = tmp_path / "selected"
    selected_dir.mkdir()
    (selected_dir / "selected_seed1.json").write_text(
        '[{"session":"2025-07-01","contract_id":"SPXW-20250701-06165.000-C"}]'
    )

    class _Metadata:
        def get_cost(self, **_: object) -> float:
            return 0.01

    class _Client:
        metadata = _Metadata()

    def fail_download(*_: object, **__: object) -> object:
        raise AssertionError("Databento get_range should not be reached before approval")

    monkeypatch.setattr(script, "_client", lambda: _Client())
    monkeypatch.setattr(
        script,
        "_raw_symbol_map",
        lambda normalized_dir, session: {"SPXW-20250701-06165.000-C": "SPXW  250701C06165000"},
    )
    monkeypatch.setattr(script, "_download", fail_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_databento_cbbo_1s_selected",
            "--selected-trades-dir",
            str(selected_dir),
            "--sessions",
            "2025-07-01",
            "--max-cost",
            "1",
            "--approval-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(SystemExit, match="paid-data approval required"):
        script.main()


def test_databento_cbbo_1s_audit_download_requires_approval_before_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import v4.scripts.download_databento_cbbo_1s_audit as script

    manifest = _manifest(tmp_path / "manifest.json")
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    frame = __import__("pandas").DataFrame(
        {
            "raw_symbol": ["SPXW  260102C06700000"],
            "strike": [6700.0],
            "underlying_price": [6701.0],
            "root": ["SPXW"],
            "settlement_style": ["PM"],
        }
    )
    frame.to_parquet(normalized / "databento_spxw_0dte_2026-01-02_derived_context.parquet")

    class _Metadata:
        def get_cost(self, **_: object) -> float:
            return 0.01

    class _Client:
        metadata = _Metadata()

    def fail_download(*_: object, **__: object) -> object:
        raise AssertionError("Databento download should not be reached before approval")

    monkeypatch.setattr(script, "_client", lambda: _Client())
    monkeypatch.setattr(script, "_download", fail_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_databento_cbbo_1s_audit",
            "--sessions",
            "2026-01-02",
            "--normalized-dir",
            str(normalized),
            "--raw-root",
            str(tmp_path / "raw"),
            "--approval-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(SystemExit, match="paid-data approval required"):
        script.main()


def test_databento_context_proxy_download_requires_approval_before_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import v4.scripts.download_databento_context_proxies as script

    manifest = _manifest(tmp_path / "manifest.json")

    class _Metadata:
        def get_cost(self, **_: object) -> float:
            return 0.01

    class _Client:
        metadata = _Metadata()

    def fail_download(*_: object, **__: object) -> object:
        raise AssertionError("Databento context proxy download should not be reached before approval")

    monkeypatch.setattr(script, "_client", lambda: _Client())
    monkeypatch.setattr(script, "_download_parent_ohlcv", fail_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_databento_context_proxies",
            "--start-date",
            "2026-01-02",
            "--days",
            "1",
            "--raw-root",
            str(tmp_path / "raw"),
            "--approval-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(SystemExit, match="paid-data approval required"):
        script.main()


def test_databento_es_vwap_download_requires_approval_before_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import v4.scripts.download_databento_es_vwap as script

    manifest = _manifest(tmp_path / "manifest.json")
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "2025-01-02.pkl").write_bytes(b"placeholder")

    class _Metadata:
        def get_cost(self, **_: object) -> float:
            return 0.01

    class _Client:
        metadata = _Metadata()

    def fail_download(*_: object, **__: object) -> object:
        raise AssertionError("Databento ES download should not be reached before approval")

    monkeypatch.setattr(script, "_client", lambda: _Client())
    monkeypatch.setattr(script, "_download_range", fail_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_databento_es_vwap",
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-02",
            "--processed-dirs",
            str(processed),
            "--raw-root",
            str(tmp_path / "raw"),
            "--approval-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(SystemExit, match="paid-data approval required"):
        script.main()
