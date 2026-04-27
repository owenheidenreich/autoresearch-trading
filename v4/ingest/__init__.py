"""v4.ingest — deterministic vendor ingest + fingerprinting.

See fingerprint.py for hashing primitives. Vendor-specific ingestors live
in this package: optionsdx.py (Phase 0), databento.py (Phase 1+).
"""
from .fingerprint import build_id, file_sha256, new_run_id, table_sha256

__all__ = ["build_id", "file_sha256", "new_run_id", "table_sha256"]
