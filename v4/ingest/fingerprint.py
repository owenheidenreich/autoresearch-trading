"""Deterministic hashing primitives.

Per the data contract, the Normalized layer must be a deterministic function
of Raw: re-running ingest produces byte-identical output. CI compares the
normalized-table fingerprint across runs; mismatch without a documented
schema bump is a critical error.

This module provides:
- file_sha256: hash an input file (for raw provenance)
- table_sha256: deterministic hash of a pyarrow Table after canonical sorting
- build_id: git sha + uncommitted-diff hash (for audit records)
- new_run_id: stable UUID per ingest run
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


_HASH_CHUNK = 1 << 20  # 1 MiB


def file_sha256(path: str | Path) -> str:
    """Compute sha256 of a file. Returns 64-char lowercase hex string."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def table_sha256(table: pa.Table, sort_keys: list[str] | None = None) -> str:
    """Deterministic sha256 of a pyarrow Table.

    Process:
    1. Optionally sort rows by `sort_keys` (recommended: a unique row key
       like ['decision_time', 'contract_id']). If None, the table is hashed
       in its current row order — caller is responsible for determinism.
    2. Serialize to Arrow IPC stream with no compression. IPC is byte-stable
       given a fixed schema and row order.
    3. sha256 the resulting bytes.

    Returns 64-char lowercase hex string.

    Note: table.cast() / set_column ops can produce non-canonical metadata.
    If two semantically-equal tables produce different hashes, normalize
    via `table.combine_chunks()` and ensure the schema's metadata is empty
    (`pa.schema(fields, metadata=None)`).
    """
    if sort_keys:
        for k in sort_keys:
            if k not in table.column_names:
                raise ValueError(f"sort key {k!r} not in table columns")
        table = table.sort_by([(k, "ascending") for k in sort_keys])

    table = table.combine_chunks()
    sink = io.BytesIO()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def new_run_id() -> str:
    """UUID4 for a fresh ingest run. Used as `ingest_run_id` everywhere."""
    return str(uuid.uuid4())


def build_id() -> str:
    """Identifier for the running build: <git_sha>[-dirty-<diff_sha8>].

    If the working tree has uncommitted changes, those changes are hashed
    and appended so the build_id captures exactly what was running. This
    matters for audit records: a green pipeline-integrity-report from a
    "dirty" working tree is identifiable.

    Falls back to '<no-git>' if not in a git repo.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout

        if diff.strip():
            diff_sha = hashlib.sha256(diff.encode()).hexdigest()[:8]
            return f"{sha}-dirty-{diff_sha}"
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "<no-git>"
