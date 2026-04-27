"""Feature layer schema.

Causal features available as of `decision_time` only. Every Feature row
must have `is_live_reproducible=True`; if a feature could not exist live
at decision_time, it does not belong in this layer.

Per the Data Contract: features are recomputable from Normalized; CI fails
on any commit that introduces a leak (a feature that depends on data with
event_time > decision_time).

This schema is intentionally extensible: feature columns are stored as a
sparse map (`feature_name`, `feature_value`) so adding a feature does not
require a schema-version bump. The required metadata fields are fixed.
"""
from __future__ import annotations

import pyarrow as pa

from .types import SCHEMA_VERSION


FEATURE_SCHEMA = pa.schema(
    [
        # --- identity ---
        pa.field("decision_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("contract_id", pa.string(), nullable=False),
        # --- feature payload ---
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("feature_value", pa.float64(), nullable=True),
        # --- causality / freshness (Section 4.0 of protocol; Section 2.6 of contract) ---
        pa.field("is_live_reproducible", pa.bool_(), nullable=False),
        pa.field("is_label", pa.bool_(), nullable=False),
        pa.field("feature_age_seconds", pa.float64(), nullable=False),
        pa.field("feature_source", pa.string(), nullable=False),
        pa.field("feature_refresh_interval", pa.float64(), nullable=True),
        pa.field("feature_is_estimated", pa.bool_(), nullable=False),
        pa.field("feature_is_revised", pa.bool_(), nullable=False),
        # --- provenance ---
        pa.field("ingest_run_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)


REQUIRED_METADATA_FIELDS = [
    "is_live_reproducible",
    "is_label",
    "feature_age_seconds",
    "feature_source",
    "feature_is_estimated",
    "feature_is_revised",
]


def validate_feature_table(table: pa.Table) -> None:
    """Verify a pyarrow Table conforms to FEATURE_SCHEMA and the leak contract.

    Hard rules:
    - Every row's `is_live_reproducible` must be True. False rows must not
      enter the feature layer; they belong in the label or audit layer.
    - Every row's `is_label` must be False. Label rows belong in the label layer.
    - `feature_age_seconds` must be non-negative.
    """
    if not table.schema.equals(FEATURE_SCHEMA, check_metadata=False):
        raise ValueError(
            "Feature schema mismatch. See docs/DATA_CONTRACT.md.\n"
            f"Expected:\n{FEATURE_SCHEMA}\nGot:\n{table.schema}"
        )

    is_live = table["is_live_reproducible"].to_pylist()
    if not all(is_live):
        bad_rows = [i for i, v in enumerate(is_live) if not v]
        raise ValueError(
            f"Feature layer requires is_live_reproducible=True on every row. "
            f"Violators at rows: {bad_rows[:5]}{'...' if len(bad_rows) > 5 else ''}"
        )

    is_label = table["is_label"].to_pylist()
    if any(is_label):
        bad_rows = [i for i, v in enumerate(is_label) if v]
        raise ValueError(
            f"Feature layer requires is_label=False on every row. "
            f"Label rows belong in the label layer. Violators: {bad_rows[:5]}"
        )

    ages = table["feature_age_seconds"].to_pylist()
    if any(a is not None and a < 0 for a in ages):
        raise ValueError("feature_age_seconds must be non-negative")
