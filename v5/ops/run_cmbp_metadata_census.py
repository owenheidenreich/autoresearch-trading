#!/usr/bin/env python3
"""Job-50 metadata runner CLI.

``inspect``, ``validate``, and ``synthetic`` are local-only and never import
Databento or read a credential.  ``external-run`` is the sole production
client-construction route and remains fail-closed until a separately sealed
authorization artifact exists.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from v5.research import cmbp_metadata_census as census
from v5.research import cmbp_metadata_census_receipt_v2 as readiness


REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "v5/work/cmbp-metadata-census"
DEFAULT_DECLARATION = REPO / "v5/work/human-policy-foundation/CMBP_CATALOGUE_DECLARATION_V1.json"
DEFAULT_CONTRACT = WORK / "PROGRAM_CONTRACT_V4.json"
DEFAULT_PRIOR_CONTRACT = WORK / "PROGRAM_CONTRACT_V3.json"
DEFAULT_INTERMEDIATE_CONTRACT = WORK / "PROGRAM_CONTRACT_V2.json"
DEFAULT_BASE_CONTRACT = WORK / "PROGRAM_CONTRACT_V1.json"
DEFAULT_SYNTHETIC_JOURNAL = WORK / "SYNTHETIC_CALL_JOURNAL_V2.jsonl"
DEFAULT_SYNTHETIC_RESPONSE = WORK / "SYNTHETIC_METADATA_RESPONSES_V2.json"
DEFAULT_EXTERNAL_ATTEMPTS = WORK / "external-attempts"
DEFAULT_READINESS_RECEIPT = WORK / "LOCAL_READINESS_RECEIPT_V2.json"
DEFAULT_TEST_REPORT = WORK / "TEST_RESULTS_V2.xml"

AuthorizedExternalClient = census.AuthorizedExternalClient


def _default_client_factory(api_key: str) -> Any:
    # This is intentionally the only SDK import in the Job-50 runner.  Every
    # caller reaches it only after the sealed authorization and credential
    # gates below pass.
    client_module = importlib.import_module("databento.historical.client")
    metadata_api = importlib.import_module("databento.historical.api.metadata")
    symbology_api = importlib.import_module("databento.historical.api.symbology")
    historical = getattr(client_module, "Historical", None)
    if (
        not inspect.isclass(historical)
        or historical.__module__ != "databento.historical.client"
        or historical.__qualname__ != "Historical"
    ):
        raise census.MetadataCensusError(
            "pinned Databento Historical class identity drifted",
            status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
        )
    expected_parameters = {
        metadata_api.MetadataHttpAPI.get_dataset_range: ("self", "dataset"),
        symbology_api.SymbologyHttpAPI.resolve: (
            "self", "dataset", "symbols", "stype_in", "stype_out", "start_date", "end_date"
        ),
        metadata_api.MetadataHttpAPI.get_cost: (
            "self", "dataset", "start", "end", "mode", "symbols", "schema", "stype_in", "limit"
        ),
        metadata_api.MetadataHttpAPI.get_record_count: (
            "self", "dataset", "start", "end", "symbols", "schema", "stype_in", "limit"
        ),
    }
    for method, expected in expected_parameters.items():
        if tuple(inspect.signature(method).parameters) != expected:
            raise census.MetadataCensusError(
                "pinned Databento SDK method signature drifted",
                status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
            )
    client = historical(key=api_key)
    if type(client) is not historical:
        raise census.MetadataCensusError(
            "constructed Databento client exact type drifted",
            status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
        )
    return client


_SEALED_PRODUCTION_CLIENT_FACTORY = _default_client_factory


def _read_databento_api_key() -> str | None:
    """Private key-read seam; production calls it only after durable gates."""

    return os.environ.get("DATABENTO_API_KEY")


def _verify_post_key_sdk_identity(
    *,
    base_contract: Mapping[str, Any],
    intermediate_contract: Mapping[str, Any],
    prior_contract: Mapping[str, Any],
    contract: Mapping[str, Any],
    local_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the entire pinned source/AST identity immediately pre-client."""

    observed = readiness._sdk_source_identity(
        base_contract,
        intermediate_contract,
        prior_contract,
        contract,
    )
    expected = local_receipt.get("bindings", {}).get("sdk_source_identity")
    if observed != expected:
        raise census.MetadataCensusError(
            "post-key SDK/source identity no longer matches readiness",
            status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
        )
    return observed


def _canonical_attempt_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise census.MetadataCensusError(
            "external attempt id is not canonical UUIDv4",
            status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
        ) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise census.MetadataCensusError(
            "external attempt id is not lowercase canonical UUIDv4",
            status="STOP_AUTHORIZATION_MISSING_OR_INVALID",
        )
    return value


def construct_authorized_external_client(
    *,
    attempt_id: str,
) -> AuthorizedExternalClient:
    """Mint the sole production capability after every precredential gate."""

    attempt_id = _canonical_attempt_id(attempt_id)
    declaration = census.load_json(DEFAULT_DECLARATION)
    contract = census.load_json(DEFAULT_CONTRACT)
    prior_contract = census.load_json(DEFAULT_PRIOR_CONTRACT)
    intermediate_contract = census.load_json(DEFAULT_INTERMEDIATE_CONTRACT)
    base_contract = census.load_json(DEFAULT_BASE_CONTRACT)
    census._contract_semantic_sha256(contract)
    census._prior_contract_semantic_sha256(prior_contract)
    census._intermediate_contract_semantic_sha256(intermediate_contract)
    census._base_contract_semantic_sha256(base_contract)
    if census.file_sha256(DEFAULT_CONTRACT) != census.PROGRAM_CONTRACT_FILE_SHA256:
        raise census.MetadataCensusError("effective V4 contract raw-file identity drifted")
    if census.file_sha256(DEFAULT_PRIOR_CONTRACT) != census.PRIOR_PROGRAM_CONTRACT_FILE_SHA256:
        raise census.MetadataCensusError("preserved V3 contract raw-file identity drifted")
    if census.file_sha256(DEFAULT_INTERMEDIATE_CONTRACT) != census.INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256:
        raise census.MetadataCensusError("preserved V2 contract raw-file identity drifted")
    if census.file_sha256(DEFAULT_BASE_CONTRACT) != census.BASE_PROGRAM_CONTRACT_FILE_SHA256:
        raise census.MetadataCensusError("preserved V1 contract raw-file identity drifted")

    try:
        local_receipt = readiness.strict_json(DEFAULT_READINESS_RECEIPT)
        readiness.validate_local_readiness_receipt(
            local_receipt,
            repo_root=REPO,
            test_report_path=DEFAULT_TEST_REPORT,
            synthetic_journal_path=DEFAULT_SYNTHETIC_JOURNAL,
            synthetic_response_path=DEFAULT_SYNTHETIC_RESPONSE,
            require_vendor_authorization_absent=False,
        )
    except Exception as exc:
        raise census.MetadataCensusError(
            "sealed local-readiness receipt does not reconstruct",
            status="STOP_LOCAL_READINESS_RECEIPT_INVALID",
        ) from exc

    authorization_path = (
        WORK
        / "authorizations"
        / attempt_id
        / "VENDOR_RUN_AUTHORIZATION_V2.json"
    )
    authorization = census.load_json(authorization_path)
    census.verify_vendor_run_authorization(
        authorization,
        declaration,
        contract,
        prior_contract=prior_contract,
        intermediate_contract=intermediate_contract,
        base_contract=base_contract,
        local_readiness_receipt=local_receipt,
        local_readiness_receipt_file_sha256=census.file_sha256(DEFAULT_READINESS_RECEIPT),
        authorization_path=authorization_path,
        repo_root=REPO,
        attempt_id=attempt_id,
        declaration_file_sha256=census.file_sha256(DEFAULT_DECLARATION),
    )
    consumption, consumption_path, attempt_directory = (
        census.consume_vendor_run_authorization(
            authorization,
            authorization_path=authorization_path,
            repo_root=REPO,
            local_readiness_receipt=local_receipt,
        )
    )

    # These accesses are deliberately after local-readiness verification,
    # strict authorization verification, and durable one-use consumption.
    try:
        if _default_client_factory is not _SEALED_PRODUCTION_CLIENT_FACTORY:
            raise census.MetadataCensusError(
                "production client factory was replaced in memory",
                status="JOB50_AUTHORITY_VIOLATION",
            )
        api_key = _read_databento_api_key()
        if not isinstance(api_key, str) or not api_key.strip():
            raise census.MetadataCensusError(
                "authorized attempt lacks a usable Databento credential",
                status="STOP_VENDOR_AUTH_OR_ENTITLEMENT",
            )
        try:
            sdk_version = importlib.metadata.version(census.EXPECTED_SDK_PACKAGE)
        except importlib.metadata.PackageNotFoundError as exc:
            raise census.MetadataCensusError(
                "pinned Databento SDK is not installed",
                status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
            ) from exc
        if sdk_version != census.EXPECTED_SDK_VERSION:
            raise census.MetadataCensusError(
                "pinned Databento SDK version drifted",
                status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
            )
        # Repeat the complete source/AST projection after credential access and
        # immediately before importing/constructing Historical.  The object is
        # compared wholesale, not by a caller-selected subset.
        _verify_post_key_sdk_identity(
            base_contract=base_contract,
            intermediate_contract=intermediate_contract,
            prior_contract=prior_contract,
            contract=contract,
            local_receipt=local_receipt,
        )
        try:
            client = _SEALED_PRODUCTION_CLIENT_FACTORY(api_key)
        except census.MetadataCensusError:
            raise
        except Exception as exc:
            raise census.MetadataCensusError(
                "authorized external client construction failed closed",
                status="STOP_VENDOR_AUTH_OR_ENTITLEMENT",
            ) from exc
    except census.MetadataCensusError as exc:
        try:
            _write_stop_diagnostic(
                attempt_directory / "ATTEMPT_STOP_V1.json",
                attempt_id=attempt_id,
                status=exc.status,
            )
        except Exception:
            pass
        raise

    bindings = {
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": census.file_sha256(authorization_path),
        "attempt_id": attempt_id,
        "attempt_directory": attempt_directory,
        "consumption_record_path": consumption_path,
        "consumption_marker_sha256": census.file_sha256(consumption_path),
        "local_readiness_receipt_sha256": local_receipt["receipt_sha256"],
        "local_readiness_receipt_file_sha256": census.file_sha256(DEFAULT_READINESS_RECEIPT),
        "base_program_contract_sha256": census.BASE_PROGRAM_CONTRACT_SHA256,
        "intermediate_program_contract_sha256": census.INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
        "prior_program_contract_sha256": census.PRIOR_PROGRAM_CONTRACT_SHA256,
        "program_contract_sha256": census.PROGRAM_CONTRACT_SHA256,
        "declaration_sha256": declaration["declaration_sha256"],
        "declaration_file_sha256": census.file_sha256(DEFAULT_DECLARATION),
        "sdk_identity_sha256": census._readiness_sdk_identity_sha256(local_receipt),
        "quoted_acquisition_ceiling_usd": authorization["quoted_acquisition_ceiling_usd"],
        "repo_root": REPO,
        "declaration_path": DEFAULT_DECLARATION,
        "program_contract_path": DEFAULT_CONTRACT,
        "prior_program_contract_path": DEFAULT_PRIOR_CONTRACT,
        "intermediate_program_contract_path": DEFAULT_INTERMEDIATE_CONTRACT,
        "base_program_contract_path": DEFAULT_BASE_CONTRACT,
        "local_readiness_receipt_path": DEFAULT_READINESS_RECEIPT,
        "readiness_test_report_path": DEFAULT_TEST_REPORT,
        "synthetic_journal_path": DEFAULT_SYNTHETIC_JOURNAL,
        "synthetic_response_path": DEFAULT_SYNTHETIC_RESPONSE,
        "authorization_path": authorization_path,
    }
    # Recheck the retained marker before sealing the client handle into the
    # capability; its payload is not trusted merely because this process wrote it.
    census.verify_authorization_consumption(
        census.load_json(consumption_path),
        record_path=consumption_path,
        authorization=authorization,
        authorization_path=authorization_path,
        repo_root=REPO,
        local_readiness_receipt=local_receipt,
    )
    return census._mint_authorized_external_client(client, bindings=bindings)


def _json_out(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration_path = Path(args.declaration)
    declaration = census.load_json(declaration_path)
    contract = census.load_json(Path(args.contract))
    contract_sha = census._contract_semantic_sha256(contract)
    if contract_sha != census.PROGRAM_CONTRACT_SHA256:
        raise census.MetadataCensusError("program contract identity drifted")
    base_contract = census.load_json(DEFAULT_BASE_CONTRACT)
    census._base_contract_semantic_sha256(base_contract)
    sealed = base_contract.get("sealed_job49_input")
    if not isinstance(sealed, Mapping) or census.file_sha256(declaration_path) != sealed.get("declaration_file_sha256"):
        raise census.MetadataCensusError("sealed Job-49 declaration raw-file identity drifted")
    return declaration, contract


def _inspect(args: argparse.Namespace) -> int:
    declaration, contract = _load_inputs(args)
    _json_out(
        census.inspect_declaration(
            declaration,
            expected_contract_sha256=str(contract["self_hash"]["value"]),
        )
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    declaration, _ = _load_inputs(args)
    response = census.load_json(Path(args.response))
    census.validate_execution_response(
        response,
        declaration,
        call_journal_path=Path(args.call_journal),
    )
    _json_out(
        {
            "status": "VALID",
            "response_file_sha256": census.file_sha256(Path(args.response)),
            "call_journal_file_sha256": census.file_sha256(Path(args.call_journal)),
            "network_calls_performed_by_validator": 0,
            "authorization_effect": "NONE",
        }
    )
    return 0


def _synthetic(args: argparse.Namespace) -> int:
    declaration, _ = _load_inputs(args)
    response = census.run_synthetic_metadata_census(
        declaration,
        call_journal_path=Path(args.call_journal),
        response_path=Path(args.response),
        attempt_id=args.attempt_id,
    )
    _json_out(
        {
            "status": "SYNTHETIC_COMPLETE",
            "attempt_id": response["execution_audit"]["attempt_id"],
            "completed_call_count": response["execution_audit"]["completed_call_count"],
            "call_journal": str(Path(args.call_journal)),
            "response": str(Path(args.response)),
            "network_calls_performed": 0,
            "credential_read": False,
            "authorization_effect": "NONE",
        }
    )
    return 0


def _write_stop_diagnostic(path: Path, *, attempt_id: str, status: str) -> None:
    census._write_json_exclusive(
        path,
        {
            "artifact_type": "JOB50_METADATA_ATTEMPT_STOP_V1",
            "schema_version": "v5.cmbp-metadata-attempt-stop.v1",
            "attempt_id": attempt_id,
            "status": census.normalize_job50_status(status),
            "success_response_emitted": False,
            "resume_allowed": False,
            "authorization_effect": "NONE",
        },
        mode=0o600,
    )


def _external_run(args: argparse.Namespace) -> int:
    if (
        Path(args.declaration).resolve() != DEFAULT_DECLARATION
        or Path(args.contract).resolve() != DEFAULT_CONTRACT
    ):
        raise census.MetadataCensusError(
            "external-run does not permit declaration or contract path overrides",
            status="STOP_FORBIDDEN_METHOD_OR_SCOPE",
        )
    attempt_id = _canonical_attempt_id(args.attempt_id)
    attempt_dir = WORK / "external-attempts" / attempt_id
    receipt_path = attempt_dir / "EXTERNAL_METADATA_RECEIPT_V1.json"
    try:
        capability = construct_authorized_external_client(attempt_id=attempt_id)
        response = census.run_authorized_external_metadata_census(capability)
        receipt = census.finalize_external_metadata_receipt(capability)
    except census.MetadataCensusError as exc:
        stop_path = attempt_dir / "ATTEMPT_STOP_V1.json"
        if attempt_dir.is_dir() and not stop_path.exists():
            try:
                _write_stop_diagnostic(
                    stop_path,
                    attempt_id=attempt_id,
                    status=exc.status,
                )
            except Exception:
                pass
        raise
    _json_out(
        {
            "status": receipt["status"],
            "attempt_id": attempt_id,
            "attempt_directory": str(attempt_dir),
            "response_file_sha256": census.file_sha256(
                attempt_dir / "EXTERNAL_METADATA_RESPONSES_V1.json"
            ),
            "receipt_file_sha256": census.file_sha256(receipt_path),
            "call_journal_file_sha256": response["execution_audit"]["call_journal"]["file_sha256"],
            "authorization_effect": census.EXTERNAL_RECEIPT_AUTHORIZATION_EFFECT,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_command = commands.add_parser("inspect", help="inspect the sealed call population locally")
    inspect_command.set_defaults(handler=_inspect)

    validate_command = commands.add_parser("validate", help="validate an existing response and call journal locally")
    validate_command.add_argument("--response", type=Path, required=True)
    validate_command.add_argument("--call-journal", type=Path, required=True)
    validate_command.set_defaults(handler=_validate)

    synthetic_command = commands.add_parser("synthetic", help="run the deterministic no-network fake client")
    synthetic_command.add_argument("--call-journal", type=Path, default=DEFAULT_SYNTHETIC_JOURNAL)
    synthetic_command.add_argument("--response", type=Path, default=DEFAULT_SYNTHETIC_RESPONSE)
    synthetic_command.add_argument("--attempt-id", default=None)
    synthetic_command.set_defaults(handler=_synthetic)

    external_command = commands.add_parser(
        "external-run",
        help="future gated four-method vendor run; blocked without a sealed fresh authorization",
    )
    external_command.add_argument(
        "--attempt-id",
        required=True,
        help="lowercase UUIDv4 already bound by the canonical owner authorization",
    )
    external_command.set_defaults(handler=_external_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except census.MetadataCensusError as exc:
        sys.stderr.write(json.dumps({"status": exc.status, "error": str(exc)}, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
