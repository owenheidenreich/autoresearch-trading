from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from v5.research.feature_admission import (
    FeatureAdmissionError,
    LEGACY_LEDGER_PATH,
    admitted_feature_matrix,
    admission_summary,
    assert_features_admitted,
    ledger_sha256,
    verify_ledger,
)


def _write_receipt(path: Path, value: str = "receipt") -> str:
    path.write_text(value)
    return hashlib.sha256(value.encode()).hexdigest()


def _write_ledger(
    tmp_path: Path,
    *,
    status: str = "ADMITTED",
    valid_until: str = "2099-01-01T00:00:00+00:00",
) -> Path:
    receipt = tmp_path / "receipt.json"
    receipt_hash = _write_receipt(receipt)
    row = {
        "name": "feature_a",
        "family": "entry.test.v1",
        "contract_id": "entry.test.v1",
        "status": status,
        "receipts": [{"path": str(receipt), "sha256": receipt_hash}],
        "availability_clock_ms": 10.0 if status == "ADMITTED" else None,
        "valid_until": valid_until,
        "barred_reason": None if status == "ADMITTED" else "missing_parity",
    }
    payload = {
        "schema_version": "v5.feature-admission-ledger.v1",
        "features": [row],
    }
    payload["ledger_sha256"] = ledger_sha256(payload)
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_legacy_ledger_verifies_as_evidence_but_cannot_authorize_new_fit() -> None:
    summary = admission_summary(LEGACY_LEDGER_PATH)
    assert summary == {
        "ledger_sha256": "04b5e9584b295c970847870d32111efce1268f0b7302f3ce830970f2411923e3",
        "features": 83,
        "admitted": 8,
        "barred": 75,
        "admitted_families": ["entry.contract_clock.v1"],
    }
    with pytest.raises(FeatureAdmissionError, match="missing_validity_window"):
        assert_features_admitted(("is_call",), ledger_path=LEGACY_LEDGER_PATH)


def test_current_signed_ledger_returns_only_requested_admitted_columns(tmp_path) -> None:
    ledger = _write_ledger(tmp_path)
    source = pd.DataFrame({"feature_a": [1.0, 2.0], "unapproved": [8.0, 9.0]})

    matrix = admitted_feature_matrix(
        source,
        ("feature_a",),
        ledger_path=ledger,
        as_of=datetime(2026, 8, 5, tzinfo=timezone.utc),
    )

    assert list(matrix.columns) == ["feature_a"]
    assert matrix.to_dict("list") == {"feature_a": [1.0, 2.0]}


def test_barred_unknown_stale_and_missing_values_fail_before_use(tmp_path) -> None:
    barred = _write_ledger(tmp_path, status="BARRED")
    with pytest.raises(FeatureAdmissionError, match="feature_not_admitted"):
        assert_features_admitted(("feature_a",), ledger_path=barred)
    with pytest.raises(FeatureAdmissionError, match="MISSING"):
        assert_features_admitted(("unknown",), ledger_path=barred)

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    stale = _write_ledger(stale_dir, valid_until="2020-01-01T00:00:00+00:00")
    with pytest.raises(FeatureAdmissionError, match="receipt_stale"):
        assert_features_admitted(("feature_a",), ledger_path=stale)

    current_dir = tmp_path / "current"
    current_dir.mkdir()
    current = _write_ledger(current_dir)
    with pytest.raises(FeatureAdmissionError, match="missing_values"):
        admitted_feature_matrix(
            pd.DataFrame({"feature_a": [1.0, None]}),
            ("feature_a",),
            ledger_path=current,
        )


def test_tampered_ledger_and_receipt_fail_closed(tmp_path) -> None:
    ledger = _write_ledger(tmp_path)
    payload = json.loads(ledger.read_text())
    payload["features"][0]["availability_clock_ms"] = 11.0
    ledger.write_text(json.dumps(payload))
    with pytest.raises(FeatureAdmissionError, match="sha256_mismatch"):
        verify_ledger(ledger)

    receipt_dir = tmp_path / "receipt-tamper"
    receipt_dir.mkdir()
    ledger = _write_ledger(receipt_dir)
    (receipt_dir / "receipt.json").write_text("changed")
    with pytest.raises(FeatureAdmissionError, match="receipt_hash_mismatch"):
        verify_ledger(ledger)

