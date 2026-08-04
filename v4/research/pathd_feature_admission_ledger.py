"""Signed, fail-closed Path-D feature-admission law.

This module does not fit a model.  It verifies the immutable Phase-0 ledger,
checks receipt bytes, and refuses construction of a feature matrix containing
anything other than an ``ADMITTED`` feature.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from v4.research.autoresearch_v2.entry_live_feature_catalog import (
    BARRED as CATALOG_BARRED,
    CONTRACTS_BY_ID,
)
from v4.research.pathd_contract_clock import (
    EARLY_CLOSE_TIMES_ET,
    contract_clock_features,
    historical_contract_clock_features,
    live_contract_clock_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "pathd.feature-admission-ledger.v1"
ADMITTED = "ADMITTED"
BARRED = "BARRED"
DEFAULT_LEDGER_PATH = REPO_ROOT / "v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/feature_admission_ledger.json"
DEFAULT_RECEIPT_DIR = REPO_ROOT / "v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/receipts"
DEFAULT_REPORT_PATH = REPO_ROOT / "v4/docs/protocol101/training/research/PATHD_PHASE0_FEATURE_CERTIFICATION_REPORT_2026_08_04.md"

PHASE0_CONTRACT_IDS = (
    "entry.contract_clock.v1",
    "entry.opra_cbbo1m_native.v1",
    "entry.opra_cbbo1m_cross_section.v1",
    "entry.opra_cbbo1s_rolling.v1",
    "entry.opra_ohlcv1m_sparse.v1",
    "entry.opra_implied_spot.v1",
    "entry.opra_implied_volatility.v1",
    "entry.self_computed_greeks.v1",
    "entry.causal_account_state.v1",
)
PERMANENT_BARRED_CONTRACT_ID = "entry.barred_no_live_twin.v1"
PARENTS = {
    "entry.opra_cbbo1m_cross_section.v1": ("entry.opra_cbbo1m_native.v1",),
    "entry.opra_implied_volatility.v1": ("entry.opra_implied_spot.v1",),
    "entry.self_computed_greeks.v1": ("entry.opra_implied_spot.v1",),
    # Corrected 2026-08-04. The Phase-0 spec listed implied_spot as having no
    # dependency, but its estimator consumes ['ask','bid','raw_symbol','right',
    # 'strike'] -- the OPRA CBBO-1m quote substrate that cbbo1m_native describes
    # and whose multi-session arrival latency is uncertified. Without this edge,
    # implied_spot (and through it IV and the greeks) was admitted carrying a
    # 4.9 ms LOCAL COMPUTE p99 as if it were an availability clock, while the
    # quotes it is derived from arrive at ~319.5 ms p99. Training on that would
    # decide earlier than is live-possible: the signed18 look-ahead class.
    # See test_parent_graph_covers_declared_substrate_dependencies.
    "entry.opra_implied_spot.v1": ("entry.opra_cbbo1m_native.v1",),
}

# Clock kinds that may back an ADMITTED row. A local compute p99 measures how
# long an adapter takes to RUN; it is not when the feature becomes available at
# the decision boundary, and may only be an additive component of one.
ARRIVAL_CLOCK_KINDS = frozenset(
    {
        "measured_definition_replay_warmup",
        "measured_upstream_arrival_p99",
        "composed_parent_arrival_plus_local_compute",
    }
)

EVIDENCE_ROOT = REPO_ROOT / "v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03"
DEFINITION_SUMMARY = EVIDENCE_ROOT / "definition_attempt001/definition_capture_summary.json"
SAME_SESSION_COMPARISON = EVIDENCE_ROOT / "comparison_same_session_attempt002/comparison_result.json"
HISTORICAL_CBBO = EVIDENCE_ROOT / "historical_attempt001/2026-08-03.cbbo-1m.parquet"


class AdmissionLedgerError(RuntimeError):
    """The signed ledger or a requested feature failed closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ledger_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("ledger_sha256", None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # Test and independent-verifier ledgers may live outside the checkout.
        # Absolute receipt paths remain unambiguous and are still byte-hashed.
        return resolved.as_posix()


def _write_signed_json(path: Path, payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[hash_field] = hashlib.sha256(_canonical(payload)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _contract_clock_receipts(receipt_dir: Path) -> tuple[list[dict[str, str]], float]:
    summary = json.loads(DEFINITION_SUMMARY.read_text(encoding="utf-8"))
    comparison = json.loads(SAME_SESSION_COMPARISON.read_text(encoding="utf-8"))
    if summary.get("status") != "CAPTURED_CURRENT_SESSION_DEFINITIONS":
        raise AdmissionLedgerError("current-session definition capture is not valid")
    schema = comparison.get("opra_schema_comparison", {})
    if schema.get("same_session_value_identity_verdict") != "PASS_BYTE_VALUE_IDENTICAL_AFTER_DECODE":
        raise AdmissionLedgerError("same-session OPRA value identity did not pass")

    source_path = Path(inspect.getsourcefile(contract_clock_features) or "")
    source_hash = sha256_file(source_path)
    captured_ms = (
        int(summary["capture_finished_unix_ns"]) - int(summary["capture_started_unix_ns"])
    ) / 1_000_000.0
    implementation = _write_signed_json(
        receipt_dir / "contract_clock_definition_implementation_hash.json",
        {
            "schema_version": "pathd.contract-clock-implementation-receipt.v1",
            "status": "PASS",
            "shared_source_path": _repo_relative(source_path),
            "shared_source_sha256": source_hash,
            "historical_callable": "historical_contract_clock_features",
            "live_callable": "live_contract_clock_features",
            "shared_callable": "contract_clock_features",
            "definition_replay_warmup_ms": captured_ms,
            "availability_clock_measurement": "capture_finished_unix_ns-capture_started_unix_ns",
            "source_definition_receipt": {
                "path": _repo_relative(DEFINITION_SUMMARY),
                "sha256": sha256_file(DEFINITION_SUMMARY),
            },
        },
        "receipt_sha256",
    )

    historical_symbols = sorted(
        pd.read_parquet(HISTORICAL_CBBO, columns=["symbol"])["symbol"].astype(str).unique()
    )
    additions = comparison["contract_identity"]["live_definition_evidence"]["same_day_added_symbols"]
    live_symbols = sorted(set(historical_symbols) | set(map(str, additions)))
    decision_time = datetime(2026, 8, 3, 16, 13, tzinfo=timezone.utc)
    mismatches = []
    for symbol in historical_symbols:
        historical = historical_contract_clock_features(symbol, decision_time)
        live = live_contract_clock_features(symbol, decision_time)
        if historical != live:
            mismatches.append(symbol)
    for symbol in live_symbols:
        live_contract_clock_features(symbol, decision_time)
    if mismatches or len(historical_symbols) != 492 or len(live_symbols) != 510:
        raise AdmissionLedgerError("historical/live OSI geometry receipt failed")
    identity = _write_signed_json(
        receipt_dir / "contract_clock_historical_live_osi_geometry_invariance.json",
        {
            "schema_version": "pathd.contract-clock-osi-identity-receipt.v1",
            "status": "PASS",
            "session": "2026-08-03",
            "decision_time": decision_time.isoformat(),
            "historical_overlap_contracts": len(historical_symbols),
            "live_current_session_contracts": len(live_symbols),
            "same_day_additions": len(additions),
            "exact_feature_vector_mismatches": len(mismatches),
            "tolerance": 0.0,
            "historical_cbbo_source": {
                "path": _repo_relative(HISTORICAL_CBBO),
                "sha256": sha256_file(HISTORICAL_CBBO),
            },
            "same_session_comparison_source": {
                "path": _repo_relative(SAME_SESSION_COMPARISON),
                "sha256": sha256_file(SAME_SESSION_COMPARISON),
            },
        },
        "receipt_sha256",
    )

    calendar_payload = {
        key: value.isoformat() for key, value in sorted(EARLY_CLOSE_TIMES_ET.items())
    }
    regular = contract_clock_features("SPXW  260803C07665000", decision_time)
    short = contract_clock_features(
        "SPXW  250703P06000000", datetime(2025, 7, 3, 16, 0, tzinfo=timezone.utc)
    )
    if regular["is_early_close"] != 0.0 or short["is_early_close"] != 1.0:
        raise AdmissionLedgerError("exchange-calendar vector check failed")
    calendar = _write_signed_json(
        receipt_dir / "contract_clock_exchange_calendar_hash.json",
        {
            "schema_version": "pathd.contract-clock-calendar-receipt.v1",
            "status": "PASS",
            "calendar_payload": calendar_payload,
            "calendar_sha256": hashlib.sha256(_canonical(calendar_payload)).hexdigest(),
            "shared_source_sha256": source_hash,
            "regular_session_vector": regular,
            "early_close_session_vector": short,
            "tolerance": 0.0,
        },
        "receipt_sha256",
    )
    rows = []
    for name, path, payload in (
        ("definition implementation hash", receipt_dir / "contract_clock_definition_implementation_hash.json", implementation),
        ("historical/live OSI geometry invariance", receipt_dir / "contract_clock_historical_live_osi_geometry_invariance.json", identity),
        ("exchange-calendar hash", receipt_dir / "contract_clock_exchange_calendar_hash.json", calendar),
    ):
        assert payload["status"] == "PASS"
        rows.append({"name": name, "path": _repo_relative(path), "sha256": sha256_file(path)})
    return rows, captured_ms


def generate_ledger(
    *, ledger_path: Path = DEFAULT_LEDGER_PATH, receipt_dir: Path = DEFAULT_RECEIPT_DIR
) -> dict[str, Any]:
    contract_receipts, availability_ms = _contract_clock_receipts(receipt_dir)
    family_status: dict[str, str] = {contract_id: BARRED for contract_id in PHASE0_CONTRACT_IDS}
    family_status["entry.contract_clock.v1"] = ADMITTED
    partial_receipts = {
        "entry.opra_cbbo1m_native.v1": [
            {
                "name": "same-session live-versus-Historical-API value identity",
                "path": _repo_relative(SAME_SESSION_COMPARISON),
                "sha256": sha256_file(SAME_SESSION_COMPARISON),
            }
        ]
    }
    features: list[dict[str, Any]] = []
    for contract_id in PHASE0_CONTRACT_IDS:
        contract = CONTRACTS_BY_ID[contract_id]
        status = family_status[contract_id]
        parents = PARENTS.get(contract_id, ())
        missing_receipts = list(contract.required_parity_receipts)
        barred_reason = None
        receipts: list[dict[str, str]] = list(partial_receipts.get(contract_id, ()))
        clock_ms = None
        tolerance = 0.0
        if status == ADMITTED:
            receipts = contract_receipts
            missing_receipts = []
            clock_ms = availability_ms
        else:
            produced_names = {receipt["name"] for receipt in receipts}
            missing_receipts = [
                name for name in missing_receipts if name not in produced_names
            ]
            parent_blockers = [parent for parent in parents if family_status[parent] != ADMITTED]
            if parent_blockers:
                barred_reason = "parent_family_not_admitted:" + ",".join(parent_blockers)
            else:
                barred_reason = "missing_required_receipts:" + "|".join(missing_receipts)
        for name in contract.feature_names:
            features.append(
                {
                    "name": name,
                    "family": contract_id,
                    "contract_id": contract_id,
                    "status": status,
                    "receipts": receipts,
                    "availability_clock_ms": clock_ms,
                    "tolerance": tolerance,
                    "barred_reason": barred_reason,
                }
            )
    permanent = CONTRACTS_BY_ID[PERMANENT_BARRED_CONTRACT_ID]
    for name in permanent.feature_names:
        features.append(
            {
                "name": name,
                "family": PERMANENT_BARRED_CONTRACT_ID,
                "contract_id": PERMANENT_BARRED_CONTRACT_ID,
                "status": BARRED,
                "receipts": [],
                "availability_clock_ms": None,
                "tolerance": 0.0,
                "barred_reason": "permanently_barred_no_matching_live_twin",
            }
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": sorted(features, key=lambda row: row["name"]),
    }
    payload = dict(unsigned)
    payload["ledger_sha256"] = ledger_sha256(unsigned)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_ledger(ledger_path)
    return payload


def verify_ledger(path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise AdmissionLedgerError(f"feature admission ledger missing_or_empty:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AdmissionLedgerError(f"feature admission ledger unreadable:{path}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AdmissionLedgerError("feature admission ledger schema mismatch")
    expected = ledger_sha256(payload)
    if payload.get("ledger_sha256") != expected:
        raise AdmissionLedgerError("feature admission ledger_sha256 mismatch")
    rows = payload.get("features")
    if not isinstance(rows, list) or not rows:
        raise AdmissionLedgerError("feature admission ledger has no features")
    seen: set[str] = set()
    statuses_by_family: dict[str, set[str]] = {}
    for row in rows:
        name = str(row.get("name", ""))
        if not name or name in seen:
            raise AdmissionLedgerError(f"duplicate_or_empty_ledger_feature:{name}")
        seen.add(name)
        status = row.get("status")
        if status not in {ADMITTED, BARRED}:
            raise AdmissionLedgerError(f"invalid_ledger_status:{name}:{status}")
        statuses_by_family.setdefault(str(row.get("contract_id")), set()).add(str(status))
        receipts = row.get("receipts")
        if not isinstance(receipts, list):
            raise AdmissionLedgerError(f"invalid_receipt_list:{name}")
        for receipt in receipts:
            receipt_path = Path(str(receipt.get("path", "")))
            if not receipt_path.is_absolute():
                receipt_path = REPO_ROOT / receipt_path
            if not receipt_path.is_file() or sha256_file(receipt_path) != receipt.get("sha256"):
                raise AdmissionLedgerError(f"receipt_hash_mismatch:{name}:{receipt_path}")
        if status == ADMITTED:
            if not receipts or row.get("availability_clock_ms") is None:
                raise AdmissionLedgerError(f"admitted_feature_missing_receipt_or_clock:{name}")
        elif not row.get("barred_reason"):
            raise AdmissionLedgerError(f"barred_feature_missing_reason:{name}")
    for child, parents in PARENTS.items():
        if statuses_by_family.get(child) == {ADMITTED}:
            for parent in parents:
                if statuses_by_family.get(parent) != {ADMITTED}:
                    raise AdmissionLedgerError(f"tier2_parent_not_admitted:{child}:{parent}")
    return payload


def assert_feature_matrix_admitted(
    feature_names: Iterable[str], *, ledger_path: Path = DEFAULT_LEDGER_PATH
) -> tuple[str, ...]:
    payload = verify_ledger(ledger_path)
    index = {str(row["name"]): row for row in payload["features"]}
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise AdmissionLedgerError("feature matrix cannot be empty")
    for name in names:
        row = index.get(name)
        if row is None:
            raise AdmissionLedgerError(f"feature_not_in_admission_ledger:{name}:MISSING")
        if row["status"] != ADMITTED:
            raise AdmissionLedgerError(
                f"feature_not_admitted:{name}:{row['status']}:{row.get('barred_reason')}"
            )
    return names


def admitted_feature_matrix(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> pd.DataFrame:
    names = assert_feature_matrix_admitted(feature_names, ledger_path=ledger_path)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise AdmissionLedgerError("feature matrix source columns missing:" + ",".join(missing))
    return frame.loc[:, list(names)]


def write_report(payload: Mapping[str, Any], path: Path = DEFAULT_REPORT_PATH) -> None:
    families: dict[str, list[Mapping[str, Any]]] = {}
    for row in payload["features"]:
        if row["contract_id"] == PERMANENT_BARRED_CONTRACT_ID:
            continue
        families.setdefault(str(row["contract_id"]), []).append(row)
    lines = [
        "# Path-D Phase 0 feature certification report — 2026-08-04",
        "",
        "Status: **STOP_FOR_CLAUDE_VERIFICATION**",
        "",
        f"Ledger SHA-256: `{payload['ledger_sha256']}`",
        "",
        "No model was loaded, trained, or fitted. The protected holdout open count remained `0`. No paid data, live capture, broker, order, paper runtime, promotion, default, runtime flag, or launchd path was accessed.",
        "",
        "| Family | Features | Certification | Availability clock | Tolerance | Evidence / blocker |",
        "|---|---:|---|---:|---:|---|",
    ]
    for family, rows in families.items():
        first = rows[0]
        status = "FIT_READY / ADMITTED" if first["status"] == ADMITTED else "BARRED"
        clock = f"{first['availability_clock_ms']:.3f} ms measured definition replay warm-up" if first["availability_clock_ms"] is not None else "N/A"
        produced = ", ".join(receipt["name"] for receipt in first["receipts"])
        evidence = "; ".join(value for value in (produced, str(first["barred_reason"] or "")) if value)
        lines.append(f"| `{family}` | {len(rows)} | **{status}** | {clock} | {first['tolerance']} | {evidence} |")
    admitted = sum(row["status"] == ADMITTED for row in payload["features"])
    scoped = sum(row["contract_id"] in PHASE0_CONTRACT_IDS for row in payload["features"])
    lines.extend(
        [
            "",
            "## Exit-gate result",
            "",
            f"- OPRA-only scoped features: `{scoped}`; admitted: `{admitted}`; barred: `{scoped-admitted}`.",
            "- `entry.contract_clock.v1` is the sole certified family. Its availability clock is measured from the owned definition capture (`capture_finished_unix_ns - capture_started_unix_ns`), not inherited from ThetaData.",
            "- `entry.opra_cbbo1m_native.v1` remains barred because the owned evidence is one session and the contract requires a multi-session local receipt-latency distribution. The observed OPRA same-session p99 of about 319.5 ms is preserved as evidence but is not promoted into a multi-session receipt.",
            "- Every Tier-2 family remains barred until both its own receipts and its parent admission exist.",
            "- The 10 permanently barred controls, including `intraday_open_interest`, are included in the executable ledger so they fail with an explicit `BARRED` status.",
            "",
            "## Enforcement",
            "",
            "The admission check executes immediately before feature-matrix construction in the autoresearch v2 OOF fit path. Missing, empty, tampered, unknown, or barred ledger state raises before any estimator is created or fitted.",
            "",
            "`STOP_FOR_CLAUDE_VERIFICATION`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    payload = generate_ledger(ledger_path=args.ledger, receipt_dir=args.receipt_dir)
    write_report(payload, args.report)
    print(json.dumps({"ledger": str(args.ledger), "ledger_sha256": payload["ledger_sha256"], "status": "STOP_FOR_CLAUDE_VERIFICATION"}, sort_keys=True))


if __name__ == "__main__":
    main()
