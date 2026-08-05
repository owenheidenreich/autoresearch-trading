"""Fail-closed feature admission for future v5 estimator inputs.

Verification is read-only.  The historical Path-D ledger can be inspected as
evidence, but its admitted rows have no current validity window and therefore
cannot authorize a new v5 fit.  A future v5 certification must issue a signed
ledger with unexpired receipts before ``admitted_feature_matrix`` succeeds.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_LEDGER_PATH = (
    REPO_ROOT
    / "v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04"
    / "feature_admission_ledger.json"
)
SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {"pathd.feature-admission-ledger.v1", "v5.feature-admission-ledger.v1"}
)
ADMITTED = "ADMITTED"
BARRED = "BARRED"
REVIEWED_ANTECEDENT = {
    "path": "v4/research/pathd_feature_admission_ledger.py",
    "sha256": "57fde34a488e15b7876d9efe84225e266fb3aa9645d0e58722d1ed21628663d4",
}

# Dependency edges are part of the safety law, not research conclusions.
DEFAULT_PARENT_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "entry.opra_cbbo1m_cross_section.v1": ("entry.opra_cbbo1m_native.v1",),
    "entry.opra_implied_spot.v1": ("entry.opra_cbbo1m_native.v1",),
    "entry.opra_implied_volatility.v1": ("entry.opra_implied_spot.v1",),
    "entry.self_computed_greeks.v1": ("entry.opra_implied_spot.v1",),
}


class FeatureAdmissionError(RuntimeError):
    """A ledger, receipt, or requested estimator input failed closed."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def ledger_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("ledger_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_receipt_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise FeatureAdmissionError(f"missing_{label}")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise FeatureAdmissionError(f"invalid_{label}:{value}") from exc
    if parsed.tzinfo is None:
        raise FeatureAdmissionError(f"timezone_required_{label}:{value}")
    return parsed.astimezone(timezone.utc)


def verify_ledger(
    path: Path = LEGACY_LEDGER_PATH,
    *,
    parent_families: Mapping[str, Sequence[str]] = DEFAULT_PARENT_FAMILIES,
) -> dict[str, Any]:
    """Verify ledger self-hash, receipt bytes, rows, and family dependencies."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FeatureAdmissionError(f"feature_admission_ledger_missing_or_empty:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureAdmissionError(f"feature_admission_ledger_unreadable:{path}") from exc
    if payload.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise FeatureAdmissionError("feature_admission_ledger_schema_mismatch")
    if payload.get("ledger_sha256") != ledger_sha256(payload):
        raise FeatureAdmissionError("feature_admission_ledger_sha256_mismatch")
    rows = payload.get("features")
    if not isinstance(rows, list) or not rows:
        raise FeatureAdmissionError("feature_admission_ledger_has_no_features")

    seen: set[str] = set()
    statuses_by_family: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise FeatureAdmissionError("feature_admission_row_is_not_an_object")
        name = str(row.get("name", ""))
        family = str(row.get("contract_id", row.get("family", "")))
        status = str(row.get("status", ""))
        if not name or name in seen:
            raise FeatureAdmissionError(f"duplicate_or_empty_ledger_feature:{name}")
        seen.add(name)
        if not family:
            raise FeatureAdmissionError(f"missing_feature_family:{name}")
        if status not in {ADMITTED, BARRED}:
            raise FeatureAdmissionError(f"invalid_ledger_status:{name}:{status}")
        statuses_by_family.setdefault(family, set()).add(status)
        receipts = row.get("receipts")
        if not isinstance(receipts, list):
            raise FeatureAdmissionError(f"invalid_receipt_list:{name}")
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise FeatureAdmissionError(f"invalid_receipt:{name}")
            receipt_path = _resolve_receipt_path(str(receipt.get("path", "")))
            expected_hash = str(receipt.get("sha256", ""))
            if (
                not receipt_path.is_file()
                or not expected_hash
                or sha256_file(receipt_path) != expected_hash
            ):
                raise FeatureAdmissionError(
                    f"receipt_hash_mismatch:{name}:{receipt_path}"
                )
        if status == ADMITTED:
            if not receipts or row.get("availability_clock_ms") is None:
                raise FeatureAdmissionError(
                    f"admitted_feature_missing_receipt_or_clock:{name}"
                )
        elif not row.get("barred_reason"):
            raise FeatureAdmissionError(f"barred_feature_missing_reason:{name}")

    for child, parents in parent_families.items():
        if statuses_by_family.get(child) == {ADMITTED}:
            for parent in parents:
                if statuses_by_family.get(str(parent)) != {ADMITTED}:
                    raise FeatureAdmissionError(
                        f"parent_family_not_admitted:{child}:{parent}"
                    )
    return payload


def admission_summary(path: Path = LEGACY_LEDGER_PATH) -> dict[str, Any]:
    payload = verify_ledger(path)
    rows = payload["features"]
    admitted = [row for row in rows if row["status"] == ADMITTED]
    return {
        "ledger_sha256": payload["ledger_sha256"],
        "features": len(rows),
        "admitted": len(admitted),
        "barred": len(rows) - len(admitted),
        "admitted_families": sorted(
            {str(row.get("contract_id", row.get("family"))) for row in admitted}
        ),
    }


def _assert_row_current(row: Mapping[str, Any], *, as_of: datetime) -> None:
    """Require a predeclared validity window for every future fit input."""

    valid_until = row.get("valid_until")
    if valid_until is None:
        raise FeatureAdmissionError(
            f"admitted_feature_missing_validity_window:{row.get('name')}"
        )
    if _parse_utc(valid_until, label="valid_until") < as_of:
        raise FeatureAdmissionError(f"admitted_feature_receipt_stale:{row.get('name')}")


def assert_features_admitted(
    feature_names: Iterable[str],
    *,
    ledger_path: Path = LEGACY_LEDGER_PATH,
    as_of: datetime | None = None,
    require_current_receipts: bool = True,
) -> tuple[str, ...]:
    """Fail before estimator construction for unknown, barred, or stale inputs."""

    payload = verify_ledger(ledger_path)
    rows = {str(row["name"]): row for row in payload["features"]}
    names = tuple(map(str, feature_names))
    if not names:
        raise FeatureAdmissionError("feature_matrix_cannot_be_empty")
    if len(set(names)) != len(names):
        raise FeatureAdmissionError("feature_matrix_contains_duplicate_names")
    check_time = as_of or datetime.now(timezone.utc)
    if check_time.tzinfo is None:
        raise FeatureAdmissionError("as_of_timezone_required")
    check_time = check_time.astimezone(timezone.utc)
    for name in names:
        row = rows.get(name)
        if row is None:
            raise FeatureAdmissionError(f"feature_not_in_admission_ledger:{name}:MISSING")
        if row["status"] != ADMITTED:
            raise FeatureAdmissionError(
                f"feature_not_admitted:{name}:{row['status']}:{row.get('barred_reason')}"
            )
        if require_current_receipts:
            _assert_row_current(row, as_of=check_time)
    return names


def admitted_feature_matrix(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    ledger_path: Path = LEGACY_LEDGER_PATH,
    as_of: datetime | None = None,
    require_current_receipts: bool = True,
) -> pd.DataFrame:
    """Return only admitted columns after every fail-closed check succeeds."""

    names = assert_features_admitted(
        feature_names,
        ledger_path=ledger_path,
        as_of=as_of,
        require_current_receipts=require_current_receipts,
    )
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise FeatureAdmissionError(
            "feature_matrix_source_columns_missing:" + ",".join(missing)
        )
    matrix = frame.loc[:, list(names)].copy()
    if matrix.empty:
        raise FeatureAdmissionError("feature_matrix_has_no_rows")
    if matrix.isna().any().any():
        missing_columns = matrix.columns[matrix.isna().any()].tolist()
        raise FeatureAdmissionError(
            "feature_matrix_contains_missing_values:" + ",".join(missing_columns)
        )
    return matrix
