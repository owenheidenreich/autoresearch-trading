"""Tests for Protocol101 data materialization helper."""
from __future__ import annotations

from pathlib import Path

from v4.scripts.materialize_protocol101_data_tree import build_payload


def test_materialization_inventory_does_not_stream_files(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "sample.parquet").write_bytes(b"abc")

    payload = build_payload(roots=[root], execute=False, max_files=0)

    assert payload["status"] == "pass"
    assert payload["execute"] is False
    assert payload["file_count"] == 1
    assert payload["bytes_streamed"] == 0


def test_materialization_execute_streams_files(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "sample.parquet").write_bytes(b"abc")

    payload = build_payload(roots=[root], execute=True, max_files=0)

    assert payload["status"] == "pass"
    assert payload["execute"] is True
    assert payload["before_dataless_count"] == 0
    assert payload["bytes_streamed"] == 0
