"""Audit-superseding local-readiness receipt for Job 50.

Generation V2 preserves the complete first seal in place, requires the sealed
eleven-gap independent audit record, and reconstructs only side-by-side V2
evidence under the effective V4 contract.  This module has no credential,
vendor-client, network, market-data, broker, fitting, or order path.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from v5.research import cmbp_metadata_census as census
from v5.research import cmbp_metadata_census_receipt as legacy


ARTIFACT_TYPE = "JOB50_LOCAL_READINESS_RECEIPT_V2"
SCHEMA_VERSION = "v5.cmbp-metadata-census-local-readiness-receipt.v2"
STATUS = "JOB50_LOCAL_RUNNER_READY_ONLY"
NEXT_STATE = "BLOCKED_AWAITING_JOB50_METADATA_VENDOR_CALL_AUTHORIZATION"

WORK_RELATIVE = Path("v5/work/cmbp-metadata-census")
PLAN_RELATIVE = WORK_RELATIVE / "PLAN.md"
V1_CONTRACT_RELATIVE = WORK_RELATIVE / "PROGRAM_CONTRACT_V1.json"
V2_CONTRACT_RELATIVE = WORK_RELATIVE / "PROGRAM_CONTRACT_V2.json"
V3_CONTRACT_RELATIVE = WORK_RELATIVE / "PROGRAM_CONTRACT_V3.json"
V4_CONTRACT_RELATIVE = WORK_RELATIVE / "PROGRAM_CONTRACT_V4.json"
AUDIT_RELATIVE = WORK_RELATIVE / "INDEPENDENT_TEST_AUDIT_V1.json"
TEST_REPORT_RELATIVE = WORK_RELATIVE / "TEST_RESULTS_V2.xml"
SYNTHETIC_JOURNAL_RELATIVE = WORK_RELATIVE / "SYNTHETIC_CALL_JOURNAL_V2.jsonl"
SYNTHETIC_RESPONSE_RELATIVE = WORK_RELATIVE / "SYNTHETIC_METADATA_RESPONSES_V2.json"
READINESS_RECEIPT_RELATIVE = WORK_RELATIVE / "LOCAL_READINESS_RECEIPT_V2.json"

CURRENT_IMPLEMENTATION_FILES = (
    "v5/research/cmbp_catalogue_preflight.py",
    "v5/research/cmbp_metadata_census.py",
    "v5/ops/run_cmbp_metadata_census.py",
    "v5/tests/test_cmbp_metadata_census.py",
    "v5/research/cmbp_metadata_census_receipt.py",
    "v5/ops/record_cmbp_metadata_census_readiness.py",
    "v5/research/cmbp_metadata_census_receipt_v2.py",
    "v5/ops/record_cmbp_metadata_census_readiness_v2.py",
)

V2_RUNTIME_FILES = (
    "v5/research/cmbp_metadata_census_receipt_v2.py",
    "v5/ops/record_cmbp_metadata_census_readiness_v2.py",
)

FORBIDDEN_RUNTIME_IMPORTS = legacy.FORBIDDEN_RUNTIME_IMPORTS


class MetadataCensusReceiptV2Error(legacy.MetadataCensusReceiptError):
    """Current V2 evidence cannot support the local-readiness claim."""


def _null_field_sha256(value: Mapping[str, Any], field: str) -> str:
    """Hash the complete canonical object with one retained field set to null."""

    unsigned = copy.deepcopy(dict(value))
    _require(field in unsigned, f"self-hash field is missing: {field}")
    unsigned[field] = None
    return hashlib.sha256(legacy.canonical_json_bytes(unsigned)).hexdigest()


def _require(
    condition: bool,
    message: str,
    *,
    status: str = legacy.STOP_STATUS,
) -> None:
    if not condition:
        raise MetadataCensusReceiptV2Error(message, status=status)


def _canonical_path(
    repo_root: Path,
    relative: Path,
    name: str,
    *,
    supplied: Path | None = None,
    must_exist: bool = True,
) -> Path:
    try:
        return legacy.canonical_repository_artifact_path(
            repo_root,
            supplied if supplied is not None else Path(repo_root) / relative,
            relative,
            name,
            must_exist=must_exist,
        )
    except legacy.MetadataCensusReceiptError as exc:
        raise MetadataCensusReceiptV2Error(str(exc), status=exc.status) from exc


def _verify_contract(path: Path, *, version: int) -> dict[str, Any]:
    contract = legacy.strict_json(path)
    _require(
        contract.get("artifact_type") == f"JOB50_PROGRAM_CONTRACT_V{version}",
        f"wrong Job-50 V{version} contract artifact",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    _require(
        contract.get("schema_version")
        == f"v5.cmbp-metadata-census-program-contract.v{version}",
        f"wrong Job-50 V{version} contract schema",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    _require(
        contract.get("job_id") == 50 and contract.get("state") == "LOCAL_BUILD_ONLY",
        f"Job-50 V{version} authority state drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    block = contract.get("self_hash")
    _require(
        isinstance(block, Mapping) and block.get("status") == "SEALED",
        f"Job-50 V{version} contract is not sealed",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    claimed = legacy._require_sha(block.get("value"), f"Job-50 V{version} contract self-hash")
    _require(
        claimed == legacy.contract_semantic_sha256(contract),
        f"Job-50 V{version} contract self-hash mismatch",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    if version == 4:
        _require(
            contract.get("seal_state") == "SEALED_AFTER_INDEPENDENT_AUDIT_PASS",
            "V4 audit seal state is not final",
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        )
        _require(
            claimed == census.PROGRAM_CONTRACT_SHA256,
            "runner and V4 semantic identities differ",
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        )
        _require(
            legacy.file_sha256(path) == census.PROGRAM_CONTRACT_FILE_SHA256,
            "runner and V4 raw identities differ",
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        )
    return contract


def _contract_chain(repo_root: Path) -> tuple[dict[str, Any], ...]:
    paths = tuple(
        _canonical_path(repo_root, relative, f"V{version} contract")
        for version, relative in (
            (1, V1_CONTRACT_RELATIVE),
            (2, V2_CONTRACT_RELATIVE),
            (3, V3_CONTRACT_RELATIVE),
            (4, V4_CONTRACT_RELATIVE),
        )
    )
    contracts = tuple(
        _verify_contract(path, version=version)
        for version, path in enumerate(paths, start=1)
    )
    v1, v2, v3, v4 = contracts
    identities = tuple(str(item["self_hash"]["value"]) for item in contracts)
    raw = tuple(legacy.file_sha256(path) for path in paths)
    _require(
        v2["supersession"]["base_contract_semantic_sha256"] == identities[0]
        and v2["supersession"]["base_contract_file_sha256"] == raw[0],
        "V2/V1 contract chain drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    _require(
        v3["supersession"]["base_contract_semantic_sha256"] == identities[1]
        and v3["supersession"]["base_contract_file_sha256"] == raw[1]
        and v3["supersession"]["v1_contract_semantic_sha256"] == identities[0]
        and v3["supersession"]["v1_contract_file_sha256"] == raw[0],
        "V3 preserved contract chain drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    _require(
        v4["supersession"]["base_contract_semantic_sha256"] == identities[2]
        and v4["supersession"]["base_contract_file_sha256"] == raw[2]
        and v4["supersession"]["v2_contract_semantic_sha256"] == identities[1]
        and v4["supersession"]["v2_contract_file_sha256"] == raw[1]
        and v4["supersession"]["v1_contract_semantic_sha256"] == identities[0]
        and v4["supersession"]["v1_contract_file_sha256"] == raw[0],
        "V4 preserved contract chain drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    runner_projection = {
        "PRIOR_PROGRAM_CONTRACT_SHA256": identities[2],
        "PRIOR_PROGRAM_CONTRACT_FILE_SHA256": raw[2],
        "INTERMEDIATE_PROGRAM_CONTRACT_SHA256": identities[1],
        "INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256": raw[1],
        "BASE_PROGRAM_CONTRACT_SHA256": identities[0],
        "BASE_PROGRAM_CONTRACT_FILE_SHA256": raw[0],
    }
    for attribute, expected in runner_projection.items():
        _require(
            getattr(census, attribute, None) == expected,
            f"runner preserved contract projection drifted: {attribute}",
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        )
    return (*contracts, paths, identities, raw)


def _required_tests(
    v1: Mapping[str, Any],
    v2: Mapping[str, Any],
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> list[str]:
    groups = (
        v1.get("local_evidence_gate", {}).get("required_tests"),
        v2.get("local_readiness_receipt_v2_delta", {}).get("required_tests_added_to_v1"),
        v3.get("local_readiness_v3_delta", {}).get("required_tests_added_to_v2"),
    )
    _require(
        all(isinstance(group, list) and all(isinstance(item, str) for item in group) for group in groups),
        "effective required-test lists are invalid",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    required = [item for group in groups for item in group]
    _require(
        len(required) == len(set(required)) == v4["effective_test_gate"]["required_test_total"] == 26,
        "effective V4 required-test total drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    return required


def _verify_interim_v1(repo_root: Path, v4: Mapping[str, Any]) -> dict[str, Any]:
    disposition = v4.get("interim_v1_evidence_disposition")
    _require(isinstance(disposition, Mapping), "V4 interim V1 disposition is missing")
    _require(
        disposition.get("classification") == "INTERIM_SUPERSEDED_AFTER_INDEPENDENT_AUDIT"
        and disposition.get("terminal_or_current_readiness") is False
        and disposition.get("external_precredential_acceptable") is False,
        "interim V1 disposition drifted",
        status="STOP_LOCAL_READINESS_RECEIPT_INVALID",
    )
    artifacts = disposition.get("artifacts")
    _require(isinstance(artifacts, Mapping), "interim V1 artifact bindings are missing")
    observed: dict[str, Any] = {}
    for name, spec in artifacts.items():
        _require(isinstance(spec, Mapping), f"interim V1 {name} binding is invalid")
        relative = Path(str(spec.get("path")))
        path = _canonical_path(repo_root, relative, f"interim V1 {name}")
        raw = legacy.file_sha256(path)
        _require(raw == spec.get("file_sha256"), f"interim V1 {name} bytes drifted")
        observed[name] = {"path": relative.as_posix(), "file_sha256": raw}
    receipt_spec = artifacts["local_readiness_receipt"]
    receipt_path = Path(repo_root) / str(receipt_spec["path"])
    receipt = legacy.strict_json(receipt_path)
    _require(receipt.get("artifact_type") == legacy.ARTIFACT_TYPE, "wrong interim V1 receipt artifact")
    _require(receipt.get("schema_version") == legacy.SCHEMA_VERSION, "wrong interim V1 receipt schema")
    receipt_sha = legacy._require_sha(receipt.get("receipt_sha256"), "interim V1 receipt self-hash")
    _require(receipt_sha == legacy.payload_sha256(receipt, "receipt_sha256"), "interim V1 receipt self-hash mismatch")
    _require(receipt_sha == receipt_spec.get("receipt_sha256"), "interim V1 receipt semantic binding drifted")
    local_gate = receipt.get("local_gate")
    bindings = receipt.get("bindings")
    _require(isinstance(local_gate, Mapping) and isinstance(bindings, Mapping), "interim V1 receipt bindings are missing")
    _require(
        bindings.get("test_report", {}).get("report_file_sha256")
        == artifacts["test_report"]["file_sha256"],
        "interim V1 receipt/JUnit binding drifted",
    )
    _require(
        local_gate.get("journal", {}).get("file_sha256")
        == artifacts["synthetic_call_journal"]["file_sha256"],
        "interim V1 receipt/journal binding drifted",
    )
    _require(
        local_gate.get("response", {}).get("file_sha256")
        == artifacts["synthetic_response"]["file_sha256"],
        "interim V1 receipt/response binding drifted",
    )
    observed["classification"] = disposition["classification"]
    observed["terminal_or_current_readiness"] = False
    observed["receipt_sha256"] = receipt_sha
    return observed


def _verify_audit_record(
    repo_root: Path,
    v4: Mapping[str, Any],
    required_tests: Sequence[str],
) -> dict[str, Any]:
    contract = v4.get("independent_test_audit_contract")
    _require(isinstance(contract, Mapping), "independent test-audit contract is missing")
    path = _canonical_path(repo_root, AUDIT_RELATIVE, "independent test-audit record")
    record = legacy.strict_json(path)
    required_fields = contract.get("required_exact_fields")
    _require(isinstance(required_fields, list) and set(record) == set(required_fields), "independent audit exact-field set drifted")
    _require(record.get("artifact_type") == contract.get("artifact_type"), "wrong independent audit artifact")
    _require(record.get("schema_version") == contract.get("schema_version"), "wrong independent audit schema")
    _require(record.get("job_id") == 50, "independent audit is not Job 50")
    claimed = legacy._require_sha(record.get("audit_sha256"), "independent audit self-hash")
    _require(claimed == _null_field_sha256(record, "audit_sha256"), "independent audit self-hash mismatch")
    raw = legacy.file_sha256(path)
    _require(
        claimed == contract.get("semantic_sha256") and raw == contract.get("file_sha256"),
        "V4 independent-audit binding drifted",
    )
    identifiers = contract.get("finding_identifiers_in_frozen_order")
    _require(record.get("finding_count") == contract.get("finding_count") == 11, "independent audit finding count drifted")
    _require(record.get("finding_identifiers") == identifiers, "independent audit identifiers drifted")
    _require(record.get("required_test_names") == list(required_tests), "independent audit required-test names drifted")
    fixed_values = contract.get("fixed_record_values")
    _require(isinstance(fixed_values, Mapping), "independent audit fixed values are missing")
    for field, expected in fixed_values.items():
        _require(record.get(field) == expected, f"independent audit fixed field drifted: {field}")
    audited_at = record.get("audited_at_utc")
    _require(
        isinstance(audited_at, str) and bool(audited_at) and audited_at.endswith("Z"),
        "independent audit completion timestamp is not RFC3339 UTC",
    )
    try:
        parsed_audited_at = datetime.fromisoformat(audited_at[:-1] + "+00:00")
    except ValueError as exc:
        raise MetadataCensusReceiptV2Error(
            "independent audit completion timestamp is invalid"
        ) from exc
    _require(
        parsed_audited_at.utcoffset() is not None
        and parsed_audited_at.utcoffset().total_seconds() == 0,
        "independent audit completion timestamp is not UTC",
    )
    test_path = _canonical_path(repo_root, Path("v5/tests/test_cmbp_metadata_census.py"), "strengthened test source")
    _require(record.get("test_source_file_sha256") == legacy.file_sha256(test_path), "independent audit test-source bytes drifted")
    rows = record.get("closure_rows")
    _require(isinstance(rows, list) and len(rows) == 11, "independent audit closure rows drifted")
    row_fields = set(contract.get("closure_row_exact_fields", ()))
    for identifier, row in zip(identifiers, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == row_fields, f"audit closure row schema drifted: {identifier}")
        _require(row.get("identifier") == identifier, f"audit closure-row order drifted: {identifier}")
        _require(row.get("initial_disposition") == "MATERIAL_COVERAGE_GAP", f"audit initial disposition drifted: {identifier}")
        names = row.get("closure_test_names")
        expected_names = v4["gap_to_strengthened_test_coverage"][identifier]
        _require(names == expected_names, f"audit closure tests drifted: {identifier}")
        _require(
            isinstance(row.get("closure_evidence"), str)
            and bool(row["closure_evidence"].strip()),
            f"audit closure evidence is empty: {identifier}",
        )
        _require(row.get("closure_status") == "CLOSED_ON_AUDITED_BYTES", f"audit gap remains open: {identifier}")
    _require(
        record.get("focused_test_count") == len(required_tests)
        and record.get("focused_failures") == 0
        and record.get("focused_errors") == 0
        and record.get("focused_skips") == 0
        and record.get("positive_external_pass_executed_locally") is False
        and record.get("audit_verdict") == "PASS_CURRENT_BYTES",
        "independent audit did not pass current test bytes",
    )
    provenance_contract = contract.get("external_provenance_closure_contract")
    _require(
        isinstance(provenance_contract, Mapping),
        "independent audit provenance-closure contract is missing",
    )
    provenance = record.get(provenance_contract.get("record_field"))
    _require(
        isinstance(provenance, Mapping)
        and set(provenance) == set(provenance_contract.get("exact_fields", ()))
        and dict(provenance) == provenance_contract.get("required_value"),
        "independent audit external-provenance closure drifted",
    )
    return {
        "path": AUDIT_RELATIVE.as_posix(),
        "audit_sha256": claimed,
        "file_sha256": raw,
        "finding_count": 11,
        "finding_identifiers": list(identifiers),
        "audit_verdict": record["audit_verdict"],
        "test_source_file_sha256": record["test_source_file_sha256"],
        "positive_external_pass_executed_locally": False,
        "external_provenance_closure": dict(provenance),
        "manual_auditor_attestation_boundary": record["manual_auditor_attestation_boundary"],
    }


def _implementation_hashes(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for value in CURRENT_IMPLEMENTATION_FILES:
        relative = Path(value)
        path = _canonical_path(repo_root, relative, "current Job-50 implementation")
        values[value] = legacy.file_sha256(path)
    return values


def _runtime_import_audit(repo_root: Path) -> dict[str, Any]:
    roots: dict[str, list[str]] = {}
    for value in V2_RUNTIME_FILES:
        relative = Path(value)
        path = _canonical_path(repo_root, relative, "V2 receipt runtime")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = sorted(imported & FORBIDDEN_RUNTIME_IMPORTS)
        _require(not forbidden, f"V2 receipt runtime imports forbidden dependencies: {forbidden}")
        roots[value] = sorted(imported)
    return {
        "audited_files": list(V2_RUNTIME_FILES),
        "direct_import_roots": roots,
        "forbidden_import_roots": sorted(FORBIDDEN_RUNTIME_IMPORTS),
        "forbidden_imports_observed": [],
    }


def build_local_readiness_receipt_v2(
    *,
    repo_root: Path,
    test_report_path: Path,
    synthetic_journal_path: Path,
    synthetic_response_path: Path,
    require_vendor_authorization_absent: bool = True,
) -> dict[str, Any]:
    """Reconstruct the current, audit-cleared V2 readiness receipt."""

    repo_root = Path(repo_root).resolve(strict=True)
    test_report_path = _canonical_path(repo_root, TEST_REPORT_RELATIVE, "V2 JUnit report", supplied=test_report_path)
    synthetic_journal_path = _canonical_path(repo_root, SYNTHETIC_JOURNAL_RELATIVE, "V2 synthetic journal", supplied=synthetic_journal_path)
    synthetic_response_path = _canonical_path(repo_root, SYNTHETIC_RESPONSE_RELATIVE, "V2 synthetic response", supplied=synthetic_response_path)
    plan_path = _canonical_path(repo_root, PLAN_RELATIVE, "Job-50 plan")

    v1, v2, v3, v4, paths, identities, raw = _contract_chain(repo_root)
    required_tests = _required_tests(v1, v2, v3, v4)
    audit = _verify_audit_record(repo_root, v4, required_tests)
    interim_v1 = _verify_interim_v1(repo_root, v4)
    declaration, job49_bindings = legacy._verify_job49_bindings(repo_root, v1)
    tests = legacy.verify_junit_report(test_report_path, required_test_names=required_tests)
    source_audit = legacy._required_test_source_audit(repo_root, required_tests)
    _require(
        source_audit["file_sha256"] == audit["test_source_file_sha256"],
        "JUnit/test source differs from the independently audited bytes",
    )
    _, execution = legacy._verify_synthetic_execution(
        declaration=declaration,
        response_path=synthetic_response_path,
        journal_path=synthetic_journal_path,
        contract_sha256=identities[3],
    )
    execution["journal"]["path"] = SYNTHETIC_JOURNAL_RELATIVE.as_posix()
    execution["response"]["path"] = SYNTHETIC_RESPONSE_RELATIVE.as_posix()

    work = repo_root / WORK_RELATIVE
    if require_vendor_authorization_absent:
        _require(not list(work.rglob("VENDOR_RUN_AUTHORIZATION*.json")), "V2 build found a vendor authorization", status="JOB50_AUTHORITY_VIOLATION")
        _require(not list((work / "authorization-consumptions").glob("*.json")) if (work / "authorization-consumptions").exists() else True, "V2 build found authorization consumption", status="JOB50_AUTHORITY_VIOLATION")
        _require(not list((work / "external-attempts").iterdir()) if (work / "external-attempts").exists() else True, "V2 build found external attempt evidence", status="JOB50_AUTHORITY_VIOLATION")

    try:
        sdk_identity = legacy._sdk_source_identity(v1, v2, v3)
    except legacy.MetadataCensusReceiptError as exc:
        raise MetadataCensusReceiptV2Error(str(exc), status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT") from exc

    receipt: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "job_id": 50,
        "status": STATUS,
        "status_meaning": "Eleven independently audited coverage gaps are closed and the exact V4-bound request population passed a fresh side-by-side V2 synthetic rehearsal. No external preflight or downstream authority follows.",
        "next_state": NEXT_STATE,
        "local_gate": {
            "generation": 2,
            "synthetic_only": True,
            "external_preflight_achieved": False,
            "positive_external_pass_executed_locally": False,
            "external_positive_path_status": v4["external_provenance_v4_law"]["positive_path_status"],
            "vendor_authorization_artifact_present_at_seal": False,
            "interim_v1_is_current": False,
            "dataset": declaration["dataset"],
            "schema": declaration["schema"],
            "stype_in": declaration["stype_in"],
            "source_session_count": declaration["source_session_count"],
            "request_session_count": declaration["request_session_count"],
            "excluded_known_precoverage_session_count": declaration["excluded_pre_event_era_session_count"],
            "session_symbol_membership_count": declaration["session_symbol_membership_count"],
            "sdk_package": census.EXPECTED_SDK_PACKAGE,
            "sdk_version": census.EXPECTED_SDK_VERSION,
            **execution,
        },
        "bindings": {
            "plan_path": PLAN_RELATIVE.as_posix(),
            "plan_file_sha256": legacy.file_sha256(plan_path),
            "program_contract": {"path": V4_CONTRACT_RELATIVE.as_posix(), "semantic_sha256": identities[3], "file_sha256": raw[3]},
            "prior_program_contract": {"path": V3_CONTRACT_RELATIVE.as_posix(), "semantic_sha256": identities[2], "file_sha256": raw[2]},
            "intermediate_program_contract": {"path": V2_CONTRACT_RELATIVE.as_posix(), "semantic_sha256": identities[1], "file_sha256": raw[1]},
            "base_program_contract": {"path": V1_CONTRACT_RELATIVE.as_posix(), "semantic_sha256": identities[0], "file_sha256": raw[0]},
            "interim_v1_evidence": interim_v1,
            "independent_test_audit": audit,
            "sealed_job49": job49_bindings,
            "implementation_file_sha256": _implementation_hashes(repo_root),
            "sdk_source_identity": sdk_identity,
            "test_report": {"path": TEST_REPORT_RELATIVE.as_posix(), **tests},
            "test_source_audit": source_audit,
            "synthetic_evidence_paths": {"journal": SYNTHETIC_JOURNAL_RELATIVE.as_posix(), "response": SYNTHETIC_RESPONSE_RELATIVE.as_posix()},
            "receipt_runtime_import_audit": _runtime_import_audit(repo_root),
        },
        "claim_boundary": {
            "interim_v1_disposition": "INTERIM_SUPERSEDED_AFTER_INDEPENDENT_AUDIT",
            "vendor_availability": "UNKNOWN",
            "exact_external_session_count": "UNKNOWN",
            "exact_external_cost": "UNKNOWN",
            "zero_price_boundary": "UNKNOWN",
            "population_event_prevalence": "UNKNOWN",
            "strategy_economics": "NOT_READ",
            "external_provenance": {
                "caller_selected_source_allowed": False,
                "private_execution_default": v4["external_provenance_v4_law"]["private_execution_default"],
                "external_context_mechanism": v4["external_provenance_v4_law"]["external_context_mechanism"],
                "receipt_context_mechanism": v4["external_provenance_v4_law"]["receipt_context_mechanism"],
                "capability_test_exception_to_v3": v4["external_provenance_v4_law"]["capability_test_exception_to_v3"],
                "production_capability_provenance_law": v4["external_provenance_v4_law"]["production_capability_provenance_law"],
                "negative_fixture_structural_law": v4["external_provenance_v4_law"]["negative_fixture_structural_law"],
                "local_positive_external_pass_allowed": False,
                "positive_external_pass_executed_locally": False,
                "positive_path_status": v4["external_provenance_v4_law"]["positive_path_status"],
                "local_test_boundary": v4["external_provenance_v4_law"]["local_test_boundary"],
                "bounded_tamper_boundary": v4["external_provenance_v4_law"]["bounded_tamper_boundary"],
            },
            "data_acquisition_authorized": False,
            "local_credential_zero_definition": v3["local_readiness_v3_delta"]["local_credential_zero_definition"],
            "tamper_evidence": v2["tamper_evidence_boundary"]["required_plain_language_claim"],
            "audit_bootstrap_boundary": v4["independent_test_audit_contract"]["post_seal_bootstrap_law"],
        },
        "integrity": dict(legacy.INTEGRITY_ZERO),
        "authority_effect": "NONE",
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = legacy.payload_sha256(receipt, "receipt_sha256")
    return receipt


def validate_local_readiness_receipt_v2(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    test_report_path: Path,
    synthetic_journal_path: Path,
    synthetic_response_path: Path,
    require_vendor_authorization_absent: bool = True,
) -> None:
    _require(receipt.get("artifact_type") == ARTIFACT_TYPE, "wrong V2 readiness receipt artifact")
    _require(receipt.get("schema_version") == SCHEMA_VERSION, "wrong V2 readiness receipt schema")
    _require(receipt.get("status") == STATUS and receipt.get("next_state") == NEXT_STATE, "V2 readiness state drifted")
    claimed = legacy._require_sha(receipt.get("receipt_sha256"), "V2 readiness receipt self-hash")
    _require(claimed == legacy.payload_sha256(receipt, "receipt_sha256"), "V2 readiness receipt self-hash mismatch")
    _require(receipt.get("integrity") == legacy.INTEGRITY_ZERO, "V2 readiness zero-action integrity drifted")
    _require(receipt.get("authority_effect") == "NONE", "V2 readiness receipt claims authority")
    rebuilt = build_local_readiness_receipt_v2(
        repo_root=repo_root,
        test_report_path=test_report_path,
        synthetic_journal_path=synthetic_journal_path,
        synthetic_response_path=synthetic_response_path,
        require_vendor_authorization_absent=require_vendor_authorization_absent,
    )
    _require(dict(receipt) == rebuilt, "V2 readiness receipt no longer matches current bound evidence")


# Stable post-V4 integration surface.  The explicit V2 names remain available
# for artifact-generation code; the compatibility names let the production CLI
# change only its imported module at the mechanical post-seal projection.
def _sdk_source_identity(
    base_contract: Mapping[str, Any],
    intermediate_contract: Mapping[str, Any],
    prior_contract: Mapping[str, Any],
    effective_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the four-contract V4 projection, then rebuild SDK identity."""

    contracts = (base_contract, intermediate_contract, prior_contract, effective_contract)
    expected_artifacts = tuple(f"JOB50_PROGRAM_CONTRACT_V{version}" for version in range(1, 5))
    expected_schemas = tuple(
        f"v5.cmbp-metadata-census-program-contract.v{version}"
        for version in range(1, 5)
    )
    identities: list[str] = []
    for version, (contract, artifact, schema) in enumerate(
        zip(contracts, expected_artifacts, expected_schemas, strict=True),
        start=1,
    ):
        _require(
            contract.get("artifact_type") == artifact
            and contract.get("schema_version") == schema
            and contract.get("job_id") == 50
            and contract.get("state") == "LOCAL_BUILD_ONLY",
            f"post-key V{version} contract identity drifted",
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        )
        block = contract.get("self_hash")
        _require(
            isinstance(block, Mapping) and block.get("status") == "SEALED",
            f"post-key V{version} contract is not sealed",
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        )
        claimed = legacy._require_sha(
            block.get("value"), f"post-key V{version} contract self-hash"
        )
        _require(
            claimed == legacy.contract_semantic_sha256(contract),
            f"post-key V{version} contract semantic hash drifted",
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        )
        identities.append(claimed)
    _require(
        identities
        == [
            census.BASE_PROGRAM_CONTRACT_SHA256,
            census.INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
            census.PRIOR_PROGRAM_CONTRACT_SHA256,
            census.PROGRAM_CONTRACT_SHA256,
        ],
        "post-key runner contract projection drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    supersession = effective_contract.get("supersession")
    _require(
        isinstance(supersession, Mapping)
        and supersession.get("base_contract_semantic_sha256") == identities[2]
        and supersession.get("v2_contract_semantic_sha256") == identities[1]
        and supersession.get("v1_contract_semantic_sha256") == identities[0],
        "post-key V4 preserved semantic chain drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    try:
        return legacy._sdk_source_identity(
            base_contract,
            intermediate_contract,
            prior_contract,
        )
    except legacy.MetadataCensusReceiptError as exc:
        raise MetadataCensusReceiptV2Error(
            str(exc), status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT"
        ) from exc


build_local_readiness_receipt = build_local_readiness_receipt_v2
validate_local_readiness_receipt = validate_local_readiness_receipt_v2
strict_json = legacy.strict_json
sdk_source_identity = _sdk_source_identity
