"""Raw layer schema.

Raw rows are vendor-original messages, stored unchanged. The Raw layer is
immutable, hash-stamped on ingest, and never modified after first write.
A hash mismatch on re-read is a critical error.

Schema is intentionally minimal: only the provenance fields v4 needs to
locate, audit, and version vendor-original payloads. The vendor's own column
set is stored as an opaque payload (typically a Parquet file written from
pandas/pyarrow with vendor-native columns preserved).
"""
from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from .types import SCHEMA_VERSION, VendorSource


RAW_PROVENANCE_SCHEMA = pa.schema(
    [
        pa.field("ingest_run_id", pa.string(), nullable=False),
        pa.field("vendor_source", pa.string(), nullable=False),
        pa.field("vendor_file_path", pa.string(), nullable=False),
        pa.field("vendor_file_sha256", pa.string(), nullable=False),
        pa.field("vendor_file_size_bytes", pa.int64(), nullable=False),
        pa.field("vendor_received_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)
"""Provenance index written alongside every raw payload.

The vendor's own data is stored as a separate file (CSV / Parquet / etc.)
hashed by sha256 and referenced by `vendor_file_sha256`. This index lets
audit consumers locate any raw payload from a run id without opening the
payload itself.
"""


@dataclass(frozen=True)
class RawProvenance:
    """One row in the raw provenance index. Frozen so it cannot be mutated
    after construction — the Raw layer is immutable by contract."""

    ingest_run_id: str
    vendor_source: VendorSource
    vendor_file_path: str
    vendor_file_sha256: str
    vendor_file_size_bytes: int
    vendor_received_at: str  # ISO8601 with tz; converted to pa.timestamp on write
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.vendor_file_size_bytes < 0:
            raise ValueError("vendor_file_size_bytes must be non-negative")
        if len(self.vendor_file_sha256) != 64:
            raise ValueError(
                f"vendor_file_sha256 must be 64 hex chars; got {len(self.vendor_file_sha256)}"
            )


def validate_raw_provenance_table(table: pa.Table) -> None:
    """Verify a pyarrow Table conforms to RAW_PROVENANCE_SCHEMA."""
    if not table.schema.equals(RAW_PROVENANCE_SCHEMA, check_metadata=False):
        raise ValueError(
            f"Raw provenance schema mismatch.\nExpected:\n{RAW_PROVENANCE_SCHEMA}\n"
            f"Got:\n{table.schema}"
        )
    valid_vendors = set(VendorSource.values())
    for vendor in table["vendor_source"].to_pylist():
        if vendor not in valid_vendors:
            raise ValueError(f"Unknown vendor_source: {vendor!r}")
