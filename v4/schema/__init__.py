"""v4.schema — schema descriptors for the five data layers.

See ../docs/DATA_CONTRACT.md for the operational contract.
"""
from .audit import (
    AUDIT_INDEX_SCHEMA,
    AuditRecord,
    FileRef,
    LeakageTestResult,
)
from .feature import FEATURE_SCHEMA, REQUIRED_METADATA_FIELDS, validate_feature_table
from .label import KNOWN_LABEL_NAMES, LABEL_SCHEMA, validate_label_table
from .normalized import (
    NORMALIZED_SCHEMA,
    REQUIRED_FIELDS_NORMALIZED,
    validate_normalized_table,
)
from .raw import RAW_PROVENANCE_SCHEMA, RawProvenance, validate_raw_provenance_table
from .types import (
    SCHEMA_VERSION,
    ContractRoot,
    GreekSource,
    OptionRight,
    TimestampPrecision,
    VendorSource,
)

__all__ = [
    "AUDIT_INDEX_SCHEMA",
    "AuditRecord",
    "ContractRoot",
    "FEATURE_SCHEMA",
    "FileRef",
    "GreekSource",
    "KNOWN_LABEL_NAMES",
    "LABEL_SCHEMA",
    "LeakageTestResult",
    "NORMALIZED_SCHEMA",
    "OptionRight",
    "REQUIRED_FIELDS_NORMALIZED",
    "REQUIRED_METADATA_FIELDS",
    "RAW_PROVENANCE_SCHEMA",
    "RawProvenance",
    "SCHEMA_VERSION",
    "TimestampPrecision",
    "VendorSource",
    "validate_feature_table",
    "validate_label_table",
    "validate_normalized_table",
    "validate_raw_provenance_table",
]
