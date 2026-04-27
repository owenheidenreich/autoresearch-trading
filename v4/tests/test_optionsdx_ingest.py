"""Tests for OptionsDX ingest."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from v4.ingest.fingerprint import table_sha256
from v4.ingest.optionsdx import ingest_optionsdx_file
from v4.schema.types import ContractRoot, GreekSource, OptionRight, VendorSource

FIXTURE = Path(__file__).parent / "fixtures" / "optionsdx_spx_sample.csv"


def test_fixture_exists() -> None:
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"


def test_ingest_returns_expected_shape() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    # 5 input rows × 2 sides per row = 10 normalized records
    assert result.rows_emitted == 10
    assert result.normalized.num_rows == 10
    assert result.raw_provenance.num_rows == 1


def test_normalized_columns_populated() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    tbl = result.normalized
    # All vendor_source rows are OPTIONSDX
    assert all(v == VendorSource.OPTIONSDX.value for v in tbl["vendor_source"].to_pylist())
    # Every row has bid AND ask
    bids = tbl["bid"].to_pylist()
    asks = tbl["ask"].to_pylist()
    assert all(b is not None for b in bids)
    assert all(a is not None for a in asks)
    # Mid is computed correctly
    mids = tbl["mid"].to_pylist()
    for b, a, m in zip(bids, asks, mids, strict=True):
        assert m == pytest.approx((b + a) / 2.0)


def test_normalized_has_both_call_and_put_rows() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    rights = result.normalized["right"].to_pylist()
    assert OptionRight.CALL.value in rights
    assert OptionRight.PUT.value in rights


def test_iv_and_greek_source_populated() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    iv_sources = {s for s in result.normalized["iv_source"].to_pylist() if s}
    greek_sources = {s for s in result.normalized["greek_source"].to_pylist() if s}
    assert iv_sources == {GreekSource.OPTIONSDX.value}
    assert greek_sources == {GreekSource.OPTIONSDX.value}


def test_contract_id_is_canonical_form() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    cids = result.normalized["contract_id"].to_pylist()
    # Canonical form: ROOT-YYYYMMDD-STRIKE.padded-RIGHT
    assert "SPXW-20230102-03840.000-C" in cids
    assert "SPXW-20230102-03840.000-P" in cids
    assert "SPXW-20230102-03850.000-C" in cids


def test_ingest_is_deterministic() -> None:
    """Re-running ingest with the same run_id produces the same fingerprint."""
    result_a = ingest_optionsdx_file(FIXTURE, ingest_run_id="fixed-id")
    result_b = ingest_optionsdx_file(FIXTURE, ingest_run_id="fixed-id")
    sort_keys = ["event_time", "contract_id"]
    assert table_sha256(result_a.normalized, sort_keys=sort_keys) == table_sha256(
        result_b.normalized, sort_keys=sort_keys
    )


def test_input_file_hash_recorded() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    assert len(result.input_sha256) == 64
    assert result.raw_provenance["vendor_file_sha256"][0].as_py() == result.input_sha256


def test_expiry_is_parsed_as_date() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    expiries = result.normalized["expiry"].to_pylist()
    assert all(e == date(2023, 1, 2) for e in expiries)
