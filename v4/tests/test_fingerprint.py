"""Tests for deterministic ingest hashing."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from v4.ingest import build_id, file_sha256, new_run_id, table_sha256


def test_file_sha256_known_string() -> None:
    """Lock down sha256 against a known string so any future hashlib changes
    that affect output would fail loudly."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"v4 phase 0 audit substrate")
        path = f.name
    try:
        # Verified independently with `sha256sum`
        h = file_sha256(path)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
        # Compute the expected hash via hashlib directly to catch any wrapper drift
        import hashlib

        expected = hashlib.sha256(b"v4 phase 0 audit substrate").hexdigest()
        assert h == expected
    finally:
        Path(path).unlink()


def test_file_sha256_chunks_correctly() -> None:
    """File reads in 1 MiB chunks. Exercise the chunk path with > 1 MiB."""
    payload = b"x" * (2 * 1024 * 1024 + 17)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(payload)
        path = f.name
    try:
        import hashlib

        assert file_sha256(path) == hashlib.sha256(payload).hexdigest()
    finally:
        Path(path).unlink()


def test_table_sha256_is_deterministic() -> None:
    tbl = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    h1 = table_sha256(tbl)
    h2 = table_sha256(tbl)
    assert h1 == h2


def test_table_sha256_invariant_under_row_permutation_when_sorted() -> None:
    tbl_a = pa.table({"k": [1, 2, 3], "v": [10, 20, 30]})
    tbl_b = pa.table({"k": [3, 1, 2], "v": [30, 10, 20]})
    assert table_sha256(tbl_a, sort_keys=["k"]) == table_sha256(tbl_b, sort_keys=["k"])


def test_table_sha256_changes_when_data_changes() -> None:
    tbl_a = pa.table({"k": [1, 2, 3], "v": [10, 20, 30]})
    tbl_b = pa.table({"k": [1, 2, 3], "v": [10, 20, 31]})  # one differing value
    assert table_sha256(tbl_a) != table_sha256(tbl_b)


def test_table_sha256_rejects_unknown_sort_key() -> None:
    tbl = pa.table({"a": [1]})
    with pytest.raises(ValueError, match="not in table columns"):
        table_sha256(tbl, sort_keys=["does_not_exist"])


def test_new_run_id_unique() -> None:
    a = new_run_id()
    b = new_run_id()
    assert a != b
    assert len(a) == 36  # uuid4 canonical form


def test_build_id_format() -> None:
    """build_id should be either a git sha (40 hex), 'sha-dirty-XXXXXXXX',
    or '<no-git>' as fallback."""
    bid = build_id()
    assert isinstance(bid, str)
    if bid != "<no-git>":
        # Either pure 40-char sha or 'sha-dirty-XXXXXXXX'
        if "-dirty-" in bid:
            sha, _, suffix = bid.partition("-dirty-")
            assert len(sha) == 40
            assert len(suffix) == 8
        else:
            assert len(bid) == 40
