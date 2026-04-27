"""Audit layer schema.

Append-only. Hashes, source versions, vendor licenses, build IDs, schema
versions, leakage tests. Failed runs stay in the audit log forever — they
are evidence the discipline was followed.

Audit records are stored as JSONL (one JSON object per line) for human
auditability, with a parallel Parquet index for fast querying.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pyarrow as pa

from .types import SCHEMA_VERSION


AUDIT_INDEX_SCHEMA = pa.schema(
    [
        pa.field("ingest_run_id", pa.string(), nullable=False),
        pa.field("started_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("finished_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("build_id", pa.string(), nullable=False),
        pa.field("vendor_source", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("input_files_count", pa.int32(), nullable=False),
        pa.field("output_files_count", pa.int32(), nullable=False),
        pa.field("output_rows_total", pa.int64(), nullable=False),
        pa.field("deterministic_rebuild_hash", pa.string(), nullable=False),
        pa.field("leakage_tests_passed", pa.bool_(), nullable=False),
        pa.field("status", pa.string(), nullable=False),  # 'green' | 'red' | 'partial'
    ]
)


@dataclass
class FileRef:
    path: str
    sha256: str
    size_bytes: int
    row_count: int | None = None


@dataclass
class LeakageTestResult:
    name: str
    passed: bool
    metric_name: str
    metric_value: float
    notes: str = ""


@dataclass
class AuditRecord:
    """One entry in the audit log. Append-only. Immutable after write."""

    ingest_run_id: str
    started_at: str  # ISO8601 UTC
    build_id: str    # git sha + uncommitted-diff hash
    vendor_source: str
    schema_version: str = SCHEMA_VERSION
    finished_at: str | None = None
    input_files: list[FileRef] = field(default_factory=list)
    output_files: list[FileRef] = field(default_factory=list)
    leakage_test_results: list[LeakageTestResult] = field(default_factory=list)
    deterministic_rebuild_hash: str = ""
    status: str = "running"  # 'running' | 'green' | 'red' | 'partial'
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def output_rows_total(self) -> int:
        return sum(f.row_count or 0 for f in self.output_files)

    @property
    def leakage_tests_passed(self) -> bool:
        if not self.leakage_test_results:
            return False
        return all(r.passed for r in self.leakage_test_results)
