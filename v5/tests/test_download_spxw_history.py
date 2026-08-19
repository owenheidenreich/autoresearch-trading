from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from v5.ops import download_spxw_history as subject


def _write_ohlcv(path: Path, rows: int) -> None:
    pd.DataFrame({"x": list(range(rows))}).to_parquet(path)


def test_source_sessions_uses_only_nonempty_files_inside_declared_span(tmp_path: Path) -> None:
    _write_ohlcv(tmp_path / "2022-06-01.spxw_0dte.ohlcv-1m.parquet", 1)
    _write_ohlcv(tmp_path / "2022-06-02.spxw_0dte.ohlcv-1m.parquet", 0)
    _write_ohlcv(tmp_path / "2022-06-03.spxw_0dte.ohlcv-1m.parquet", 2)
    _write_ohlcv(tmp_path / "2022-05-31.spxw_0dte.ohlcv-1m.parquet", 9)

    sessions, digest = subject.source_sessions(
        tmp_path, start="2022-06-01", end="2022-06-03"
    )

    assert sessions == ["2022-06-01", "2022-06-03"]
    assert digest == hashlib.sha256(subject.canonical_json(sessions)).hexdigest()


def test_definition_uses_full_utc_day_and_cbbo_uses_regular_hours() -> None:
    definition_start, definition_end = subject._bounds("2024-01-02", "definition")
    cbbo_start, cbbo_end = subject._bounds("2024-01-02", "cbbo-1m")

    assert (definition_start, definition_end) == (
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
    )
    assert (cbbo_start, cbbo_end) == (
        "2024-01-02T14:30:00Z",
        "2024-01-02T21:00:00Z",
    )


def test_zero_dte_filters_by_osi_expiry_not_calendar_assumption() -> None:
    frame = pd.DataFrame(
        {
            "symbol": [
                "SPXW  240102C04700000",
                "SPXW  240103P04700000",
                "SPXW  240102P04695000",
            ],
            "bid_px_00": [1.0, 2.0, 3.0],
        }
    )

    got = subject._zero_dte(frame, "2024-01-02")

    assert got["symbol"].tolist() == ["SPXW  240102C04700000", "SPXW  240102P04695000"]
    assert got["right"].tolist() == ["C", "P"]
    assert got["strike"].tolist() == [4700.0, 4695.0]


class _Metadata:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_cost(self, **kwargs: object) -> float:
        self.calls.append(dict(kwargs))
        return 0.25


class _Client:
    def __init__(self) -> None:
        self.metadata = _Metadata()


def _declaration(path: Path, source: Path) -> dict:
    sessions, digest = subject.source_sessions(
        source, start="2022-06-01", end="2022-06-02"
    )
    payload = {
        "schema_version": "v5.lifecycle-quote-backfill-declaration.v1",
        "implementation_sha256": subject.file_sha256(Path(subject.__file__)),
        "semantic_freeze": {
            "path": "v5/work/lifecycle-training/PREACQUISITION_SEMANTIC_FREEZE_V1.json",
            "freeze_sha256": "71463cc0eeb4e242e43307e19a926ea4e258fd45a254c389e5a27110923893c4",
        },
        "hard_cap_usd": 75.0,
        "request": {"dataset": subject.DATASET, "parent": subject.PARENT, "schemas": list(subject.SCHEMAS)},
        "source_ohlcv_inventory": {
            "root": str(source),
            "start": "2022-06-01",
            "end": "2022-06-02",
            "nonempty_sessions": len(sessions),
            "sessions_sha256": digest,
        },
        "destination": {"root": str(path.parent / "destination")},
    }
    payload["declaration_sha256"] = subject._payload_sha256(payload, "declaration_sha256")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def test_preflight_prices_every_session_schema_before_writing_passing_receipt(tmp_path: Path) -> None:
    source = tmp_path / "ohlcv"
    source.mkdir()
    _write_ohlcv(source / "2022-06-01.spxw_0dte.ohlcv-1m.parquet", 1)
    _write_ohlcv(source / "2022-06-02.spxw_0dte.ohlcv-1m.parquet", 1)
    declaration = tmp_path / "declaration.json"
    declared = _declaration(declaration, source)
    output = tmp_path / "cost.json"
    client = _Client()

    receipt = subject.write_preflight(
        declaration_path=declaration, output_path=output, client=client
    )

    assert receipt["gate"] == "PASS"
    assert receipt["estimated_total_usd"] == 1.0
    assert len(client.metadata.calls) == 4
    assert {call["schema"] for call in client.metadata.calls} == set(subject.SCHEMAS)
    assert output.is_file()
    assert receipt["declaration_sha256"] == declared["declaration_sha256"]
    assert receipt["receipt_sha256"] == subject._payload_sha256(receipt, "receipt_sha256")
