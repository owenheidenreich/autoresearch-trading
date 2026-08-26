"""Deterministic verifier for Job 49's local, outcome-blind foundation.

This module binds code, contracts, the exact local OSI catalogue scope, one
synthetic human journal, the already-completed parser-only fixture identity,
and a zero-failure test report.  It has no network, vendor, broker, market-data,
model-fitting, or order path.  A pass is local infrastructure evidence only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

from v5.research import cmbp_catalogue_preflight as catalogue
from v5.research import human_decision_log as human


ARTIFACT_TYPE = "JOB49_LOCAL_FOUNDATION_RECEIPT_V2"
STATUS = "JOB49_LOCAL_FOUNDATION_PASS_ONLY_V2"
NEXT_STATE = "BLOCKED_AWAITING_JOB50_OWNER_GATE"
HUMAN_STATUS = "HUMAN_INSTRUMENTATION_READY_LOCAL_V2"
CATALOGUE_STATUS = "CMBP_CATALOGUE_READY_LOCAL"
STOP_STATUS = "STOP_LOCAL_RECEIPT_INVALID"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

INTEGRITY_ZERO = {
    "network_calls": 0,
    "vendor_calls": 0,
    "broker_calls": 0,
    "downloads": 0,
    "spend_usd": 0,
    "strategy_outcomes_read": False,
    "reserved_economics_read": False,
    "real_human_decisions_recorded": False,
    "models_fit": 0,
    "orders_submitted": 0,
}

BOUND_FILES = (
    "v5/research/human_decision_log.py",
    "v5/ops/record_human_decision.py",
    "v5/tests/test_human_decision_log.py",
    "v5/research/cmbp_catalogue_preflight.py",
    "v5/ops/prepare_cmbp_catalogue_preflight.py",
    "v5/tests/test_cmbp_catalogue_preflight.py",
    "v5/research/human_policy_foundation_receipt.py",
    "v5/ops/record_human_policy_foundation.py",
    "v5/tests/test_human_policy_foundation_receipt.py",
)


class FoundationReceiptError(RuntimeError):
    """The local packet cannot support its claimed pass."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FoundationReceiptError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise FoundationReceiptError(f"nonfinite JSON constant: {value}")


def strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoundationReceiptError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FoundationReceiptError(f"JSON root is not an object: {path}")
    return value


def contract_semantic_sha256(contract: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(contract))
    self_hash = normalized.get("self_hash")
    if not isinstance(self_hash, dict):
        raise FoundationReceiptError("program contract self_hash is missing")
    normalized_status = self_hash.get("normalized_status_for_hash")
    if normalized_status != "NORMALIZED_FOR_HASH":
        raise FoundationReceiptError("program contract hash normalization drifted")
    self_hash["value"] = None
    self_hash["status"] = normalized_status
    if contract.get("schema_version") == (
        "v5.human-policy-foundation-program-contract.v2"
    ):
        try:
            normalized["human_policy_record_contract_v2_amendment"][
                "program_contract_identity"
            ]["decision_program_contract_sha256"] = "THIS_V2_SELF_HASH"
        except (KeyError, TypeError) as exc:
            raise FoundationReceiptError(
                "V2 program-contract identity normalization is missing"
            ) from exc
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def verify_program_contract(path: Path) -> dict[str, Any]:
    contract = strict_json(path)
    if contract.get("job_id") != 49:
        raise FoundationReceiptError("program contract is not Job 49")
    self_hash = contract.get("self_hash")
    if not isinstance(self_hash, Mapping) or self_hash.get("status") != "SEALED":
        raise FoundationReceiptError("program contract is not sealed")
    claimed = self_hash.get("value")
    if not isinstance(claimed, str) or SHA256_RE.fullmatch(claimed) is None:
        raise FoundationReceiptError("program contract self-hash is invalid")
    if claimed != contract_semantic_sha256(contract):
        raise FoundationReceiptError("program contract self-hash mismatch")
    if contract.get("state") != "LOCAL_OUTCOME_BLIND_FOUNDATION_ONLY":
        raise FoundationReceiptError("program contract authority state drifted")
    if contract.get("schema_version") == (
        "v5.human-policy-foundation-program-contract.v2"
    ):
        identity = contract.get("human_policy_record_contract_v2_amendment", {}).get(
            "program_contract_identity", {}
        )
        if identity.get("decision_program_contract_sha256") != claimed:
            raise FoundationReceiptError(
                "V2 decision program-contract identity differs from its self-hash"
            )
        if identity.get("risk_contract_sha256_unchanged") != human.RISK_CONTRACT_SHA256:
            raise FoundationReceiptError("V2 risk-contract identity drifted")
    return contract


def _verify_v1_receipt_as_sealed_bytes(
    path: Path,
    *,
    expected_semantic_sha256: str,
) -> dict[str, Any]:
    receipt = strict_json(path)
    if receipt.get("artifact_type") != "JOB49_LOCAL_FOUNDATION_RECEIPT_V1":
        raise FoundationReceiptError("base receipt is not Job 49 V1")
    if receipt.get("receipt_sha256") != expected_semantic_sha256:
        raise FoundationReceiptError("base receipt semantic identity drifted")
    if payload_sha256(receipt, "receipt_sha256") != expected_semantic_sha256:
        raise FoundationReceiptError("base receipt self-hash mismatch")
    return receipt


def _verify_scope_manifest(path: Path, expected_sessions: int) -> dict[str, Any]:
    manifest = strict_json(path)
    claimed = manifest.get("manifest_sha256")
    if not isinstance(claimed, str) or claimed != catalogue.self_hash(
        manifest, "manifest_sha256"
    ):
        raise FoundationReceiptError("CMBP scope-manifest self-hash mismatch")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != expected_sessions:
        raise FoundationReceiptError("CMBP scope session count drifted")
    source_policy = manifest.get("source_policy")
    if not isinstance(source_policy, Mapping) or source_policy.get(
        "parquet_columns_read"
    ) != ["raw_symbol"]:
        raise FoundationReceiptError("CMBP scope projection widened")
    if source_policy.get("outcome_columns_read") != []:
        raise FoundationReceiptError("CMBP scope opened an outcome column")
    return manifest


def _verify_declaration(
    path: Path,
    *,
    manifest_path: Path,
    expected_sessions: int,
) -> dict[str, Any]:
    declaration = strict_json(path)
    expected_code_hashes = {
        name: file_sha256(path.parents[3] / name)
        for name in declaration.get("code_hashes", {})
    }
    catalogue.validate_catalogue_declaration(
        declaration, expected_code_hashes=expected_code_hashes
    )
    if declaration.get("local_gate_status") != CATALOGUE_STATUS:
        raise FoundationReceiptError("catalogue local-gate status drifted")
    if declaration.get("source_session_count") != expected_sessions:
        raise FoundationReceiptError("catalogue declaration source count drifted")
    if declaration.get("source_manifest_file_sha256") != file_sha256(manifest_path):
        raise FoundationReceiptError("catalogue declaration does not bind scope bytes")
    if declaration.get("claim_policy") != {
        "actual_vendor_availability": "UNKNOWN",
        "exact_available_session_count": "UNKNOWN",
        "exact_vendor_cost": "UNKNOWN",
        "zero_price_boundary": "UNKNOWN",
        "population_event_prevalence": "UNKNOWN",
        "strategy_economics": "NOT_READ",
    }:
        raise FoundationReceiptError("catalogue declaration made an external claim")
    return declaration


def _verify_parser_fixture(path: Path) -> dict[str, Any]:
    fixture = strict_json(path)
    claimed = fixture.get("manifest_sha256")
    if not isinstance(claimed, str) or claimed != payload_sha256(
        fixture, "manifest_sha256"
    ):
        raise FoundationReceiptError("parser fixture manifest self-hash mismatch")
    sessions = fixture.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 64:
        raise FoundationReceiptError("parser fixture session identity drifted")
    if fixture.get("session_symbol_pairs") != 248:
        raise FoundationReceiptError("parser fixture pair identity drifted")
    if fixture.get("total_rows") != 173_470_783:
        raise FoundationReceiptError("parser fixture row identity drifted")
    return fixture


def verify_junit_report(
    path: Path, *, required_test_names: Sequence[str]
) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise FoundationReceiptError(f"invalid JUnit report: {exc}") from exc
    cases = list(root.iter("testcase"))
    names = {case.get("name") for case in cases}
    missing = sorted(
        required
        for required in set(required_test_names)
        if required not in names
        and not any(
            isinstance(observed, str) and observed.startswith(required + "[")
            for observed in names
        )
    )
    failures = sum(bool(list(case.iter("failure"))) for case in cases)
    errors = sum(bool(list(case.iter("error"))) for case in cases)
    if not cases or failures or errors or missing:
        raise FoundationReceiptError(
            f"test report failed: tests={len(cases)} failures={failures} "
            f"errors={errors} missing={missing}"
        )
    return {
        "tests": len(cases),
        "failures": failures,
        "errors": errors,
        "required_tests_present": len(required_test_names),
        "report_file_sha256": file_sha256(path),
    }


def _relative_hashes(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in BOUND_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FoundationReceiptError(f"required implementation file is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def build_local_foundation_receipt(
    *,
    repo_root: Path,
    test_report_path: Path,
    scope_manifest_path: Path,
    catalogue_declaration_path: Path,
    synthetic_journal_path: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    work = repo_root / "v5/work/human-policy-foundation"
    base_contract_path = work / "PROGRAM_CONTRACT_V1.json"
    contract_path = work / "PROGRAM_CONTRACT_V2.json"
    base_receipt_path = work / "LOCAL_FOUNDATION_RECEIPT_V1.json"
    plan_path = work / "PLAN.md"
    fixture_path = (
        repo_root
        / "v5/work/lifecycle-training/CMBP_SEMANTIC_GATE_MANIFEST_2026_08_23.json"
    )
    base_contract = verify_program_contract(base_contract_path)
    contract = verify_program_contract(contract_path)
    supersession = contract.get("supersession")
    if not isinstance(supersession, Mapping):
        raise FoundationReceiptError("V2 supersession binding is missing")
    if supersession.get("base_contract_semantic_sha256") != base_contract[
        "self_hash"
    ]["value"]:
        raise FoundationReceiptError("V2 base-contract semantic binding drifted")
    if supersession.get("base_contract_file_sha256") != file_sha256(
        base_contract_path
    ):
        raise FoundationReceiptError("V2 base-contract byte binding drifted")
    if supersession.get("base_receipt_file_sha256") != file_sha256(
        base_receipt_path
    ):
        raise FoundationReceiptError("V2 base-receipt byte binding drifted")
    base_receipt = _verify_v1_receipt_as_sealed_bytes(
        base_receipt_path,
        expected_semantic_sha256=supersession[
            "base_receipt_semantic_sha256"
        ],
    )
    risk_bytes = canonical_json_bytes(base_contract["risk_contract_v1"])
    risk_sha256 = hashlib.sha256(risk_bytes).hexdigest()
    if risk_sha256 != human.RISK_CONTRACT_SHA256:
        raise FoundationReceiptError("human logger risk-contract identity drifted")
    if contract["self_hash"]["value"] != human.PROGRAM_CONTRACT_SHA256:
        raise FoundationReceiptError("human logger program-contract identity drifted")

    expected_sessions = base_contract["evidence_gates"]["C_offline_cmbp_catalogue"][
        "frozen_request_semantics"
    ]["source_session_count"]
    scope = _verify_scope_manifest(scope_manifest_path, expected_sessions)
    declaration = _verify_declaration(
        catalogue_declaration_path,
        manifest_path=scope_manifest_path,
        expected_sessions=expected_sessions,
    )
    fixture = _verify_parser_fixture(fixture_path)
    journal = human.verify_log(synthetic_journal_path)
    watermark = human.verify_watermark(synthetic_journal_path)
    watermark_file = human.watermark_path(synthetic_journal_path)
    if (
        journal.log_id != watermark.log_id
        or journal.terminal_sequence != watermark.terminal_sequence
        or journal.head != watermark.terminal_head
        or watermark.journal_schema_version != human.SCHEMA_VERSION
        or watermark.program_contract_sha256 != contract["self_hash"]["value"]
    ):
        raise FoundationReceiptError("journal/watermark/contract identity diverged")
    if (watermark_file.stat().st_mode & 0o777) != 0o600:
        raise FoundationReceiptError("watermark mode is not 0600")
    if journal.monitoring_state != "OFF" or journal.position_state != "FLAT":
        raise FoundationReceiptError("synthetic journal did not finish safely closed")
    if journal.explicit_waits < 1 or journal.decisions < 4:
        raise FoundationReceiptError("synthetic journal does not exercise the policy vocabulary")

    gates = base_contract["evidence_gates"]
    required_tests = (
        gates["H_human_instrumentation"]["required_tests"]
        + gates["C_offline_cmbp_catalogue"]["required_tests"]
        + contract["evidence_gate_H_v2_delta"][
            "required_tests_added_to_all_v1_gate_H_tests"
        ]
    )
    tests = verify_junit_report(
        test_report_path, required_test_names=required_tests
    )
    receipt: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": "v5.human-policy-foundation-receipt.v2",
        "job_id": 49,
        "status": STATUS,
        "status_meaning": (
            "Both Job 49 interfaces passed local outcome-blind validation only. "
            "No external, research-outcome, capture, model, broker, or order authority follows."
        ),
        "next_state": NEXT_STATE,
        "human_gate": {
            "status": HUMAN_STATUS,
            "schema_version": human.SCHEMA_VERSION,
            "synthetic_only": True,
            "journal_file_sha256": file_sha256(synthetic_journal_path),
            "watermark_file_sha256": file_sha256(watermark_file),
            "watermark_sha256": watermark.watermark_sha256,
            "watermark_file_mode": "0600",
            "log_id": journal.log_id,
            "terminal_sequence": journal.terminal_sequence,
            "terminal_head": journal.head,
            "decisions": journal.decisions,
            "explicit_waits": journal.explicit_waits,
            "final_monitoring_state": journal.monitoring_state,
            "final_position_state": journal.position_state,
            "program_contract_sha256": contract["self_hash"]["value"],
            "risk_contract_sha256": risk_sha256,
        },
        "catalogue_gate": {
            "status": CATALOGUE_STATUS,
            "primary_artifact_status": declaration["status"],
            "external_preflight_achieved": False,
            "source_session_count": declaration["source_session_count"],
            "request_session_count": declaration["request_session_count"],
            "excluded_precoverage_session_count": declaration[
                "excluded_pre_event_era_session_count"
            ],
            "session_symbol_membership_count": declaration[
                "session_symbol_membership_count"
            ],
            "scope_manifest_sha256": scope["manifest_sha256"],
            "scope_file_sha256": file_sha256(scope_manifest_path),
            "declaration_sha256": declaration["declaration_sha256"],
            "declaration_file_sha256": file_sha256(catalogue_declaration_path),
        },
        "bindings": {
            "plan_file_sha256": file_sha256(plan_path),
            "base_program_contract_semantic_sha256": base_contract["self_hash"][
                "value"
            ],
            "base_program_contract_file_sha256": file_sha256(base_contract_path),
            "program_contract_semantic_sha256": contract["self_hash"]["value"],
            "program_contract_file_sha256": file_sha256(contract_path),
            "prior_receipt": {
                "semantic_sha256": base_receipt["receipt_sha256"],
                "file_sha256": file_sha256(base_receipt_path),
                "use": "SEALED_HISTORICAL_EVIDENCE_ONLY",
            },
            "implementation_file_sha256": _relative_hashes(repo_root),
            "parser_fixture": {
                "manifest_sha256": fixture["manifest_sha256"],
                "file_sha256": file_sha256(fixture_path),
                "sessions": 64,
                "session_symbol_pairs": 248,
                "rows": 173_470_783,
                "use": "PARSER_ONLY",
            },
            "test_report": tests,
        },
        "integrity": dict(INTEGRITY_ZERO),
        "authority_effect": "NONE",
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = payload_sha256(receipt, "receipt_sha256")
    return receipt


def validate_local_foundation_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    test_report_path: Path,
    scope_manifest_path: Path,
    catalogue_declaration_path: Path,
    synthetic_journal_path: Path,
) -> None:
    if receipt.get("artifact_type") != ARTIFACT_TYPE or receipt.get("status") != STATUS:
        raise FoundationReceiptError("wrong Job 49 receipt type or status")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or claimed != payload_sha256(
        receipt, "receipt_sha256"
    ):
        raise FoundationReceiptError("Job 49 receipt self-hash mismatch")
    if receipt.get("integrity") != INTEGRITY_ZERO:
        raise FoundationReceiptError("Job 49 zero-external-action attestation drifted")
    if receipt.get("authority_effect") != "NONE" or receipt.get("next_state") != NEXT_STATE:
        raise FoundationReceiptError("Job 49 receipt claims unauthorized effect")
    rebuilt = build_local_foundation_receipt(
        repo_root=repo_root,
        test_report_path=test_report_path,
        scope_manifest_path=scope_manifest_path,
        catalogue_declaration_path=catalogue_declaration_path,
        synthetic_journal_path=synthetic_journal_path,
    )
    if dict(receipt) != rebuilt:
        raise FoundationReceiptError("Job 49 receipt no longer matches bound local artifacts")
