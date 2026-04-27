"""Smoke tests for schema descriptors and per-layer validators."""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pytest

from v4.schema import (
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    NORMALIZED_SCHEMA,
    RAW_PROVENANCE_SCHEMA,
    RawProvenance,
    SCHEMA_VERSION,
    VendorSource,
    validate_feature_table,
    validate_label_table,
    validate_normalized_table,
    validate_raw_provenance_table,
)


# ---------- raw ----------

def _good_raw_provenance() -> pa.Table:
    received = datetime(2023, 1, 2, 15, 30, tzinfo=timezone.utc)
    return pa.table(
        {
            "ingest_run_id": ["run-1"],
            "vendor_source": [VendorSource.OPTIONSDX.value],
            "vendor_file_path": ["/tmp/spx_20230102.csv"],
            "vendor_file_sha256": ["a" * 64],
            "vendor_file_size_bytes": [12345],
            "vendor_received_at": pa.array(
                [received], type=pa.timestamp("us", tz="UTC")
            ),
            "schema_version": [SCHEMA_VERSION],
        },
        schema=RAW_PROVENANCE_SCHEMA,
    )


def test_raw_provenance_dataclass_validates_hash_length() -> None:
    with pytest.raises(ValueError, match="64 hex chars"):
        RawProvenance(
            ingest_run_id="run-1",
            vendor_source=VendorSource.OPTIONSDX,
            vendor_file_path="/tmp/x.csv",
            vendor_file_sha256="abc",
            vendor_file_size_bytes=10,
            vendor_received_at="2023-01-02T15:30:00Z",
        )


def test_raw_provenance_dataclass_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RawProvenance(
            ingest_run_id="run-1",
            vendor_source=VendorSource.OPTIONSDX,
            vendor_file_path="/tmp/x.csv",
            vendor_file_sha256="a" * 64,
            vendor_file_size_bytes=-1,
            vendor_received_at="2023-01-02T15:30:00Z",
        )


def test_validate_raw_provenance_accepts_good_table() -> None:
    validate_raw_provenance_table(_good_raw_provenance())


def test_validate_raw_provenance_rejects_unknown_vendor() -> None:
    good = _good_raw_provenance()
    bad = good.set_column(
        good.column_names.index("vendor_source"),
        pa.field("vendor_source", pa.string(), nullable=False),
        pa.array(["MADE_UP_VENDOR"], type=pa.string()),
    )
    with pytest.raises(ValueError, match="Unknown vendor_source"):
        validate_raw_provenance_table(bad)


# ---------- normalized ----------

def _good_normalized() -> pa.Table:
    return pa.table(
        {f.name: pa.array([_default_for(f)], type=f.type) for f in NORMALIZED_SCHEMA},
        schema=NORMALIZED_SCHEMA,
    )


def _default_for(field: pa.Field):
    name = field.name
    if name == "right":
        return "C"
    if name == "vendor_source":
        return VendorSource.OPTIONSDX.value
    if name == "iv_source":
        return None
    if name == "greek_source":
        return None
    if name == "schema_version":
        return SCHEMA_VERSION
    if name == "ingest_run_id":
        return "run-1"
    if name == "contract_id":
        return "SPXW-20230102-04000.000-C"
    if name == "root":
        return "SPXW"
    if pa.types.is_timestamp(field.type):
        return None if field.nullable else 1672671600000000  # 2023-01-02T15:00 UTC
    if pa.types.is_date32(field.type):
        return 19359 if not field.nullable else None  # 2023-01-02
    if pa.types.is_decimal(field.type):
        from decimal import Decimal
        return Decimal("4000.000000")
    if pa.types.is_string(field.type):
        return name if not field.nullable else None
    return None


def test_validate_normalized_accepts_minimal_good_table() -> None:
    validate_normalized_table(_good_normalized())


def test_validate_normalized_rejects_bad_right() -> None:
    tbl = _good_normalized()
    idx = tbl.column_names.index("right")
    bad = tbl.set_column(
        idx,
        pa.field("right", pa.string(), nullable=False),
        pa.array(["X"], type=pa.string()),
    )
    with pytest.raises(ValueError, match="Invalid right"):
        validate_normalized_table(bad)


# ---------- feature ----------

def _good_feature(
    is_live: bool = True, is_label: bool = False, age: float = 0.0
) -> pa.Table:
    return pa.table(
        {
            "decision_time": pa.array(
                [1672671600000000], type=pa.timestamp("us", tz="UTC")
            ),
            "contract_id": ["SPXW-20230102-04000.000-C"],
            "feature_name": ["spread_bps"],
            "feature_value": [12.5],
            "is_live_reproducible": [is_live],
            "is_label": [is_label],
            "feature_age_seconds": [age],
            "feature_source": ["computed_from_normalized_v4"],
            "feature_refresh_interval": [60.0],
            "feature_is_estimated": [False],
            "feature_is_revised": [False],
            "ingest_run_id": ["run-1"],
            "schema_version": [SCHEMA_VERSION],
        },
        schema=FEATURE_SCHEMA,
    )


def test_validate_feature_accepts_good() -> None:
    validate_feature_table(_good_feature())


def test_validate_feature_rejects_non_live_reproducible() -> None:
    with pytest.raises(ValueError, match="is_live_reproducible=True"):
        validate_feature_table(_good_feature(is_live=False))


def test_validate_feature_rejects_label_in_feature_layer() -> None:
    with pytest.raises(ValueError, match="is_label=False"):
        validate_feature_table(_good_feature(is_label=True))


def test_validate_feature_rejects_negative_age() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_feature_table(_good_feature(age=-1.0))


# ---------- label ----------

def test_validate_label_rejects_unknown_name() -> None:
    tbl = pa.table(
        {
            "decision_time": pa.array(
                [1672671600000000], type=pa.timestamp("us", tz="UTC")
            ),
            "contract_id": ["SPXW-20230102-04000.000-C"],
            "label_name": ["nonsense_label"],
            "label_value": [1.0],
            "label_definition_version": ["v1"],
            "reference_exit_policy": ["time_stop_25min_OR_minus50pct_OR_session_close"],
            "simulator_version": ["sim-v1"],
            "ingest_run_id": ["run-1"],
            "schema_version": [SCHEMA_VERSION],
        },
        schema=LABEL_SCHEMA,
    )
    with pytest.raises(ValueError, match="Unknown label_name"):
        validate_label_table(tbl)


def test_validate_label_accepts_known_name() -> None:
    tbl = pa.table(
        {
            "decision_time": pa.array(
                [1672671600000000], type=pa.timestamp("us", tz="UTC")
            ),
            "contract_id": ["SPXW-20230102-04000.000-C"],
            "label_name": ["utility_lambda_0.50"],
            "label_value": [12.34],
            "label_definition_version": ["v1"],
            "reference_exit_policy": ["time_stop_25min_OR_minus50pct_OR_session_close"],
            "simulator_version": ["sim-v1"],
            "ingest_run_id": ["run-1"],
            "schema_version": [SCHEMA_VERSION],
        },
        schema=LABEL_SCHEMA,
    )
    validate_label_table(tbl)
