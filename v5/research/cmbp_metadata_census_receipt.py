"""Deterministic local-readiness receipt for the Job-50 metadata runner.

This module reads only versioned local artifacts.  It imports no Databento
client and has no credential, network, vendor, market-data, broker, fitting,
or order path.  A pass proves that the sealed Job-49 request population was
exercised through the Job-50 synthetic client with complete runner-owned call
accounting; it does not prove an external metadata preflight occurred.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

from v5.research import cmbp_catalogue_preflight as catalogue
from v5.research import cmbp_metadata_census as census


ARTIFACT_TYPE = "JOB50_LOCAL_READINESS_RECEIPT_V1"
SCHEMA_VERSION = "v5.cmbp-metadata-census-local-readiness-receipt.v1"
STATUS = "JOB50_LOCAL_RUNNER_READY_ONLY"
NEXT_STATE = "BLOCKED_AWAITING_JOB50_METADATA_VENDOR_CALL_AUTHORIZATION"
STOP_STATUS = "STOP_RESPONSE_OR_RECEIPT_INVALID"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

WORK_RELATIVE = Path("v5/work/cmbp-metadata-census")
BASE_CONTRACT_RELATIVE = WORK_RELATIVE / "PROGRAM_CONTRACT_V1.json"
INTERMEDIATE_CONTRACT_RELATIVE = WORK_RELATIVE / "PROGRAM_CONTRACT_V2.json"
EFFECTIVE_CONTRACT_RELATIVE = WORK_RELATIVE / "PROGRAM_CONTRACT_V3.json"
PLAN_RELATIVE = WORK_RELATIVE / "PLAN.md"
TEST_REPORT_RELATIVE = WORK_RELATIVE / "TEST_RESULTS_V1.xml"
SYNTHETIC_JOURNAL_RELATIVE = WORK_RELATIVE / "SYNTHETIC_CALL_JOURNAL_V1.jsonl"
SYNTHETIC_RESPONSE_RELATIVE = WORK_RELATIVE / "SYNTHETIC_METADATA_RESPONSES_V1.json"
READINESS_RECEIPT_RELATIVE = WORK_RELATIVE / "LOCAL_READINESS_RECEIPT_V1.json"

BOUND_IMPLEMENTATION_FILES = (
    "v5/research/cmbp_catalogue_preflight.py",
    "v5/research/cmbp_metadata_census.py",
    "v5/ops/run_cmbp_metadata_census.py",
    "v5/tests/test_cmbp_metadata_census.py",
    "v5/research/cmbp_metadata_census_receipt.py",
    "v5/ops/record_cmbp_metadata_census_readiness.py",
)

RECEIPT_RUNTIME_FILES = (
    "v5/research/cmbp_metadata_census_receipt.py",
    "v5/ops/record_cmbp_metadata_census_readiness.py",
)

FORBIDDEN_RUNTIME_IMPORTS = frozenset(
    {
        "databento",
        "requests",
        "aiohttp",
        "httpx",
        "socket",
        "urllib",
        "urllib3",
        "ib_insync",
        "ibapi",
    }
)

INTEGRITY_ZERO = {
    "credentials_read": 0,
    "authenticated_clients_constructed": 0,
    "network_calls": 0,
    "vendor_calls": 0,
    "timeseries_calls": 0,
    "download_calls": 0,
    "downloads": 0,
    "spend_usd": 0,
    "strategy_outcomes_read": False,
    "reserved_economics_read": False,
    "fills_read": False,
    "returns_read": False,
    "pnl_read": False,
    "models_fit": 0,
    "orders_submitted": 0,
}


class MetadataCensusReceiptError(RuntimeError):
    """The local artifacts cannot support the claimed readiness pass."""

    def __init__(self, message: str, *, status: str = STOP_STATUS) -> None:
        super().__init__(message)
        self.status = census.normalize_job50_status(status)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's compact, deterministic JSON encoding."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    unsigned = copy.deepcopy(dict(value))
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
            raise MetadataCensusReceiptError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise MetadataCensusReceiptError(f"nonfinite JSON constant: {value}")


def strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataCensusReceiptError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MetadataCensusReceiptError(f"JSON root is not an object: {path}")
    return value


def _require(
    condition: bool,
    message: str,
    *,
    status: str = STOP_STATUS,
) -> None:
    if not condition:
        raise MetadataCensusReceiptError(message, status=status)


def _require_sha(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{name} is not a lowercase SHA-256",
    )
    return value


def _repo_path(repo_root: Path, relative: Any, name: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"{name} path is invalid")
    candidate = Path(relative)
    _require(not candidate.is_absolute(), f"{name} path must be repository-relative")
    return canonical_repository_artifact_path(
        repo_root,
        Path(repo_root) / candidate,
        candidate,
        name,
    )


def canonical_repository_artifact_path(
    repo_root: Path,
    supplied: Path,
    relative: Path,
    name: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve one frozen repository path and reject every symlink component."""

    try:
        root = Path(repo_root).resolve(strict=True)
    except OSError as exc:
        raise MetadataCensusReceiptError(f"repository root is unavailable: {exc}") from exc
    _require(root.is_dir(), "repository root is not a directory")
    _require(not relative.is_absolute() and ".." not in relative.parts, f"{name} relative path is invalid")
    expected = root / relative
    supplied_path = Path(supplied)
    if not supplied_path.is_absolute():
        supplied_path = root / supplied_path
    _require(
        supplied_path.absolute() == expected.absolute(),
        f"{name} must use the frozen path {relative.as_posix()}",
    )
    current = root
    components = relative.parts if must_exist else relative.parts[:-1]
    for component in components:
        current = current / component
        _require(not current.is_symlink(), f"{name} path contains a symlink: {current}")
        _require(current.exists(), f"{name} path component is missing: {current}")
    if must_exist:
        _require(expected.is_file(), f"{name} file is missing: {relative.as_posix()}")
        try:
            resolved = expected.resolve(strict=True)
        except OSError as exc:
            raise MetadataCensusReceiptError(f"cannot resolve {name}: {exc}") from exc
    else:
        _require(not expected.is_symlink(), f"{name} target may not be a symlink")
        try:
            resolved = expected.parent.resolve(strict=True) / expected.name
        except OSError as exc:
            raise MetadataCensusReceiptError(f"cannot resolve {name} parent: {exc}") from exc
    _require(resolved.is_relative_to(root), f"{name} resolves outside the repository")
    _require(
        resolved == (root / relative).resolve(strict=must_exist),
        f"{name} resolved path differs from its frozen path",
    )
    return resolved


def contract_semantic_sha256(contract: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(contract))
    self_hash = normalized.get("self_hash")
    _require(isinstance(self_hash, dict), "program contract self_hash is missing")
    _require(
        self_hash.get("normalized_status_for_hash") == "NORMALIZED_FOR_HASH",
        "program contract normalization law drifted",
    )
    self_hash["value"] = None
    self_hash["status"] = "NORMALIZED_FOR_HASH"
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def verify_program_contract(path: Path) -> dict[str, Any]:
    contract = strict_json(path)
    version = contract.get("schema_version")
    expected_artifact = {
        "v5.cmbp-metadata-census-program-contract.v1": "JOB50_PROGRAM_CONTRACT_V1",
        "v5.cmbp-metadata-census-program-contract.v2": "JOB50_PROGRAM_CONTRACT_V2",
        "v5.cmbp-metadata-census-program-contract.v3": "JOB50_PROGRAM_CONTRACT_V3",
    }.get(version)
    _require(expected_artifact is not None, "wrong Job-50 contract schema")
    _require(contract.get("artifact_type") == expected_artifact, "wrong Job-50 contract artifact")
    _require(contract.get("job_id") == 50, "program contract is not Job 50")
    _require(contract.get("state") == "LOCAL_BUILD_ONLY", "program contract authority state drifted")
    block = contract.get("self_hash")
    _require(isinstance(block, Mapping) and block.get("status") == "SEALED", "program contract is not sealed")
    claimed = _require_sha(block.get("value"), "program contract self-hash")
    _require(claimed == contract_semantic_sha256(contract), "program contract semantic self-hash mismatch")
    if version == "v5.cmbp-metadata-census-program-contract.v3":
        _require(claimed == census.PROGRAM_CONTRACT_SHA256, "runner and effective program-contract identities differ")
        _require(
            file_sha256(path) == census.PROGRAM_CONTRACT_FILE_SHA256,
            "runner and effective program-contract raw identities differ",
        )
    return contract


def _job49_contract_semantic_sha256(contract: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(contract))
    block = normalized.get("self_hash")
    _require(isinstance(block, dict), "Job-49 contract self_hash is missing")
    normalized_status = block.get("normalized_status_for_hash")
    _require(normalized_status == "NORMALIZED_FOR_HASH", "Job-49 hash normalization drifted")
    block["value"] = None
    block["status"] = normalized_status
    if contract.get("schema_version") == "v5.human-policy-foundation-program-contract.v2":
        try:
            normalized["human_policy_record_contract_v2_amendment"][
                "program_contract_identity"
            ]["decision_program_contract_sha256"] = "THIS_V2_SELF_HASH"
        except (KeyError, TypeError) as exc:
            raise MetadataCensusReceiptError("Job-49 V2 identity normalization is missing") from exc
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _verify_job49_bindings(
    repo_root: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed = contract.get("sealed_job49_input")
    _require(isinstance(sealed, Mapping), "sealed Job-49 input block is missing")

    job49_contract_path = _repo_path(
        repo_root, sealed.get("program_contract_path"), "Job-49 program contract"
    )
    job49_contract = strict_json(job49_contract_path)
    _require(job49_contract.get("job_id") == 49, "bound foundation contract is not Job 49")
    job49_claimed = _require_sha(
        job49_contract.get("self_hash", {}).get("value"), "Job-49 contract self-hash"
    )
    _require(job49_contract.get("self_hash", {}).get("status") == "SEALED", "Job-49 contract is not sealed")
    _require(job49_claimed == _job49_contract_semantic_sha256(job49_contract), "Job-49 contract self-hash mismatch")
    _require(job49_claimed == sealed.get("program_contract_semantic_sha256"), "Job-49 semantic binding drifted")
    _require(file_sha256(job49_contract_path) == sealed.get("program_contract_file_sha256"), "Job-49 contract byte binding drifted")

    foundation_receipt_path = _repo_path(
        repo_root, sealed.get("local_foundation_receipt_path"), "Job-49 foundation receipt"
    )
    foundation_receipt = strict_json(foundation_receipt_path)
    _require(foundation_receipt.get("artifact_type") == "JOB49_LOCAL_FOUNDATION_RECEIPT_V2", "wrong Job-49 receipt artifact")
    _require(foundation_receipt.get("status") == "JOB49_LOCAL_FOUNDATION_PASS_ONLY_V2", "Job-49 foundation did not pass")
    foundation_claimed = _require_sha(foundation_receipt.get("receipt_sha256"), "Job-49 receipt self-hash")
    _require(foundation_claimed == payload_sha256(foundation_receipt, "receipt_sha256"), "Job-49 receipt self-hash mismatch")
    _require(foundation_claimed == sealed.get("local_foundation_receipt_semantic_sha256"), "Job-49 receipt semantic binding drifted")
    _require(file_sha256(foundation_receipt_path) == sealed.get("local_foundation_receipt_file_sha256"), "Job-49 receipt byte binding drifted")
    foundation_bindings = foundation_receipt.get("bindings")
    _require(isinstance(foundation_bindings, Mapping), "Job-49 receipt bindings are missing")
    foundation_implementation = foundation_bindings.get("implementation_file_sha256")
    _require(
        isinstance(foundation_implementation, Mapping),
        "Job-49 implementation bindings are missing",
    )
    parser_relative = "v5/research/cmbp_catalogue_preflight.py"
    parser_expected = _require_sha(
        foundation_implementation.get(parser_relative),
        "Job-49 catalogue-parser frozen hash",
    )
    parser_path = canonical_repository_artifact_path(
        repo_root,
        Path(repo_root) / parser_relative,
        Path(parser_relative),
        "Job-49 catalogue parser",
    )
    parser_observed = file_sha256(parser_path)
    _require(
        parser_observed == parser_expected,
        "live Job-49 catalogue parser differs from the sealed foundation receipt",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )

    declaration_path = _repo_path(
        repo_root, sealed.get("declaration_path"), "Job-49 catalogue declaration"
    )
    declaration = strict_json(declaration_path)
    try:
        catalogue.validate_catalogue_declaration(declaration)
    except catalogue.CataloguePreflightError as exc:
        raise MetadataCensusReceiptError(f"Job-49 declaration is invalid: {exc}") from exc
    declaration_claimed = _require_sha(declaration.get("declaration_sha256"), "declaration self-hash")
    _require(declaration_claimed == catalogue.self_hash(declaration, "declaration_sha256"), "declaration self-hash mismatch")
    _require(declaration_claimed == sealed.get("declaration_semantic_sha256"), "declaration semantic binding drifted")
    _require(file_sha256(declaration_path) == sealed.get("declaration_file_sha256"), "declaration byte binding drifted")

    scope_path = _repo_path(repo_root, sealed.get("scope_manifest_path"), "Job-49 scope manifest")
    scope = strict_json(scope_path)
    scope_claimed = _require_sha(scope.get("manifest_sha256"), "scope-manifest self-hash")
    _require(scope_claimed == catalogue.self_hash(scope, "manifest_sha256"), "scope-manifest self-hash mismatch")
    _require(scope_claimed == sealed.get("scope_manifest_semantic_sha256"), "scope semantic binding drifted")
    scope_file_hash = file_sha256(scope_path)
    _require(scope_file_hash == sealed.get("scope_manifest_file_sha256"), "scope byte binding drifted")
    _require(declaration.get("source_manifest_file_sha256") == scope_file_hash, "declaration/scope byte link drifted")

    exact_fields = {
        "request_manifest_sha256": declaration.get("request_manifest_sha256"),
        "session_manifest_sha256": declaration.get("session_manifest_sha256"),
        "symbol_manifest_sha256": declaration.get("symbol_manifest_sha256"),
        "dataset": declaration.get("dataset"),
        "schema": declaration.get("schema"),
        "stype_in": declaration.get("stype_in"),
        "coverage_start": declaration.get("coverage_start"),
        "source_session_count": declaration.get("source_session_count"),
        "request_session_count": declaration.get("request_session_count"),
        "excluded_known_precoverage_session_count": declaration.get(
            "excluded_pre_event_era_session_count"
        ),
        "session_symbol_membership_count": declaration.get(
            "session_symbol_membership_count"
        ),
    }
    for name, observed in exact_fields.items():
        _require(observed == sealed.get(name), f"sealed Job-49 field drifted: {name}")

    return declaration, {
        "program_contract": {
            "semantic_sha256": job49_claimed,
            "file_sha256": file_sha256(job49_contract_path),
        },
        "local_foundation_receipt": {
            "semantic_sha256": foundation_claimed,
            "file_sha256": file_sha256(foundation_receipt_path),
        },
        "catalogue_parser": {
            "path": parser_relative,
            "foundation_receipt_file_sha256": parser_expected,
            "live_file_sha256": parser_observed,
            "exact_equality_verified": True,
        },
        "catalogue_declaration": {
            "semantic_sha256": declaration_claimed,
            "file_sha256": file_sha256(declaration_path),
        },
        "scope_manifest": {
            "semantic_sha256": scope_claimed,
            "file_sha256": scope_file_hash,
        },
    }


def _find_method(
    source_path: Path,
    *,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise MetadataCensusReceiptError(f"cannot inspect SDK source {source_path}: {exc}") from exc
    matches: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            matches.extend(
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == method_name
            )
    _require(len(matches) == 1, f"SDK source does not contain exactly one {class_name}.{method_name}")
    return matches[0]


def _distribution_metadata(
    distribution: importlib.metadata.Distribution,
) -> tuple[str, Path]:
    for item in distribution.files or ():
        if str(item).endswith(".dist-info/METADATA"):
            candidate = Path(distribution.locate_file(item))
            if candidate.is_file():
                return str(item), candidate
    raise MetadataCensusReceiptError(
        f"distribution METADATA identity is missing: {distribution.metadata['Name']}"
    )


def _sdk_source_identity(
    base_contract: Mapping[str, Any],
    intermediate_contract: Mapping[str, Any],
    effective_contract: Mapping[str, Any],
) -> dict[str, Any]:
    sdk_contract = base_contract.get("sdk_adapter_contract")
    _require(isinstance(sdk_contract, Mapping), "SDK adapter contract is missing")
    package = sdk_contract.get("package")
    expected_version = sdk_contract.get("exact_version")
    _require(package == census.EXPECTED_SDK_PACKAGE, "runner/contract SDK package drifted")
    _require(expected_version == census.EXPECTED_SDK_VERSION, "runner/contract SDK version drifted")
    try:
        distribution = importlib.metadata.distribution(str(package))
    except importlib.metadata.PackageNotFoundError as exc:
        raise MetadataCensusReceiptError(f"required local SDK distribution is missing: {package}") from exc
    _require(distribution.version == expected_version, "installed SDK distribution version drifted")

    source_specs: dict[str, dict[str, Any]] = {
        "databento/historical/api/metadata.py": {
            "class": "MetadataHttpAPI",
            "methods": {
                "get_dataset_range": (["self", "dataset"], 0, "_get"),
                "get_cost": (
                    ["self", "dataset", "start", "end", "mode", "symbols", "schema", "stype_in", "limit"],
                    6,
                    "_post",
                ),
                "get_record_count": (
                    ["self", "dataset", "start", "end", "symbols", "schema", "stype_in", "limit"],
                    5,
                    "_post",
                ),
            },
        },
        "databento/historical/api/symbology.py": {
            "class": "SymbologyHttpAPI",
            "methods": {
                "resolve": (
                    ["self", "dataset", "symbols", "stype_in", "stype_out", "start_date", "end_date"],
                    1,
                    "_post",
                )
            },
        },
    }
    files: dict[str, Any] = {}
    observed_methods: dict[str, Any] = {}
    for relative, spec in source_specs.items():
        source_path = Path(distribution.locate_file(relative))
        _require(source_path.is_file(), f"SDK source file is missing: {relative}")
        files[relative] = {"file_sha256": file_sha256(source_path)}
        class_name = str(spec["class"])
        for method_name, (
            expected_args,
            expected_defaults,
            expected_transport_helper,
        ) in spec["methods"].items():
            node = _find_method(
                source_path,
                class_name=class_name,
                method_name=method_name,
            )
            observed_args = [item.arg for item in node.args.args]
            _require(observed_args == expected_args, f"SDK arguments drifted for {class_name}.{method_name}")
            _require(len(node.args.defaults) == expected_defaults, f"SDK defaults drifted for {class_name}.{method_name}")
            transport_helpers = [
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "self"
                and child.func.attr in {"_get", "_post"}
            ]
            _require(
                transport_helpers == [expected_transport_helper],
                f"SDK transport helper drifted for {class_name}.{method_name}",
            )
            observed_methods[
                f"{'symbology' if method_name == 'resolve' else 'metadata'}.{method_name}"
            ] = {
                "source_file": relative,
                "class": class_name,
                "arguments": observed_args[1:],
                "defaulted_argument_count": expected_defaults,
                "transport_helper": expected_transport_helper,
            }

    metadata_relative, metadata_path = _distribution_metadata(distribution)

    expected_signatures = sdk_contract.get("inspected_method_signatures")
    _require(isinstance(expected_signatures, Mapping), "contract SDK signatures are missing")
    _require(set(expected_signatures) == set(census.METHOD_ORDER), "contract SDK method set drifted")
    transport_contract = intermediate_contract.get("sdk_transport_identity_v2")
    _require(isinstance(transport_contract, Mapping), "V2 transport identity contract is missing")
    databento_transport = transport_contract.get("databento_distribution")
    requests_contract = transport_contract.get("requests_distribution")
    _require(isinstance(databento_transport, Mapping), "Databento transport contract is missing")
    _require(isinstance(requests_contract, Mapping), "requests transport contract is missing")
    _require(databento_transport.get("name") == package, "Databento transport package drifted")
    _require(databento_transport.get("version") == distribution.version, "Databento transport version drifted")
    _require(
        databento_transport.get("distribution_metadata_file_sha256")
        == file_sha256(metadata_path),
        "Databento distribution METADATA bytes drifted",
    )
    http_relative = str(databento_transport.get("transport_source_path"))
    http_path = Path(distribution.locate_file(http_relative))
    _require(http_path.is_file(), "Databento transport source is missing")
    http_hash = file_sha256(http_path)
    _require(http_hash == databento_transport.get("transport_source_file_sha256"), "Databento transport source bytes drifted")
    transport_methods: dict[str, Any] = {}
    for method_name, request_name in (("_get", "get"), ("_post", "post")):
        node = _find_method(http_path, class_name="BentoHttpAPI", method_name=method_name)
        calls = [
            (child.func.value.id, child.func.attr)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "requests"
            and child.func.attr in {"get", "post"}
        ]
        _require(calls == [("requests", request_name)], f"BentoHttpAPI.{method_name} transport-call count drifted")
        _require(
            not any(isinstance(child, (ast.For, ast.AsyncFor, ast.While)) for child in ast.walk(node)),
            f"BentoHttpAPI.{method_name} contains a retry-capable loop",
        )
        suspicious = [
            child.func.attr.lower()
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and any(token in child.func.attr.lower() for token in ("retry", "mount"))
        ]
        _require(not suspicious and not node.decorator_list, f"BentoHttpAPI.{method_name} retry wrapper drifted")
        transport_methods[method_name] = {
            "requests_call": f"requests.{request_name}",
            "requests_call_count": 1,
            "loop_count": 0,
            "retry_or_mount_call_count": 0,
            "decorator_count": 0,
        }

    requests_name = requests_contract.get("name")
    try:
        requests_distribution = importlib.metadata.distribution(str(requests_name))
    except importlib.metadata.PackageNotFoundError as exc:
        raise MetadataCensusReceiptError("required local requests distribution is missing") from exc
    _require(requests_distribution.version == requests_contract.get("version"), "requests distribution version drifted")
    requests_metadata_relative, requests_metadata_path = _distribution_metadata(requests_distribution)
    _require(
        file_sha256(requests_metadata_path)
        == requests_contract.get("distribution_metadata_file_sha256"),
        "requests distribution METADATA bytes drifted",
    )
    requests_sources = requests_contract.get("source_file_sha256")
    _require(isinstance(requests_sources, Mapping), "requests source identity is missing")
    observed_requests_sources: dict[str, str] = {}
    for relative, expected_hash in requests_sources.items():
        _require(isinstance(relative, str), "requests source path is invalid")
        source_path = Path(requests_distribution.locate_file(relative))
        _require(source_path.is_file(), f"requests source file is missing: {relative}")
        observed = file_sha256(source_path)
        _require(observed == expected_hash, f"requests source bytes drifted: {relative}")
        observed_requests_sources[relative] = observed

    adapters_relative = "requests/adapters.py"
    adapters_path = Path(requests_distribution.locate_file(adapters_relative))
    try:
        adapters_tree = ast.parse(adapters_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise MetadataCensusReceiptError(f"cannot inspect requests adapter source: {exc}") from exc
    default_retry_assignments = [
        node
        for node in adapters_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_RETRIES"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value == 0
    ]
    _require(len(default_retry_assignments) == 1, "requests DEFAULT_RETRIES=0 source law drifted")
    adapter_init = _find_method(
        adapters_path,
        class_name="HTTPAdapter",
        method_name="__init__",
    )
    retry_zero_calls = [
        call
        for call in ast.walk(adapter_init)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "Retry"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 0
        and any(
            keyword.arg == "read"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
    ]
    _require(len(retry_zero_calls) == 1, "requests Retry(0, read=False) source law drifted")

    urllib_contract = effective_contract.get("urllib3_transport_identity")
    _require(isinstance(urllib_contract, Mapping), "urllib3 identity contract is missing")
    urllib_name = urllib_contract.get("distribution")
    try:
        urllib_distribution = importlib.metadata.distribution(str(urllib_name))
    except importlib.metadata.PackageNotFoundError as exc:
        raise MetadataCensusReceiptError("required local urllib3 distribution is missing") from exc
    _require(
        urllib_distribution.version == urllib_contract.get("version"),
        "urllib3 distribution version drifted",
    )
    urllib_metadata_relative, urllib_metadata_path = _distribution_metadata(
        urllib_distribution
    )
    _require(
        file_sha256(urllib_metadata_path)
        == urllib_contract.get("distribution_metadata_file_sha256"),
        "urllib3 distribution METADATA bytes drifted",
    )
    urllib_sources = urllib_contract.get("source_file_sha256")
    _require(isinstance(urllib_sources, Mapping), "urllib3 source identity is missing")
    observed_urllib_sources: dict[str, str] = {}
    for relative, expected_hash in urllib_sources.items():
        _require(isinstance(relative, str), "urllib3 source path is invalid")
        source_path = Path(urllib_distribution.locate_file(relative))
        _require(source_path.is_file(), f"urllib3 source file is missing: {relative}")
        observed = file_sha256(source_path)
        _require(observed == expected_hash, f"urllib3 source bytes drifted: {relative}")
        observed_urllib_sources[relative] = observed

    client_contract = effective_contract.get("databento_client_construction_identity")
    _require(isinstance(client_contract, Mapping), "Databento client identity contract is missing")
    _require(client_contract.get("distribution") == package, "client identity package drifted")
    _require(client_contract.get("version") == distribution.version, "client identity version drifted")
    _require(
        client_contract.get("exact_class") == "databento.historical.client.Historical",
        "Databento client exact-class law drifted",
    )
    _require(
        client_contract.get("already_bound_constructor_dependencies")
        == [
            "databento/common/http.py",
            "databento/historical/api/metadata.py",
            "databento/historical/api/symbology.py",
        ],
        "Databento constructor-dependency law drifted",
    )
    client_sources = client_contract.get("source_file_sha256")
    _require(isinstance(client_sources, Mapping), "Databento client source identity is missing")
    observed_client_sources: dict[str, str] = {}
    for relative, expected_hash in client_sources.items():
        _require(isinstance(relative, str), "Databento client source path is invalid")
        source_path = Path(distribution.locate_file(relative))
        _require(source_path.is_file(), f"Databento client source file is missing: {relative}")
        observed = file_sha256(source_path)
        _require(observed == expected_hash, f"Databento client source bytes drifted: {relative}")
        observed_client_sources[relative] = observed

    root_path = Path(distribution.locate_file("databento/__init__.py"))
    try:
        root_tree = ast.parse(root_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise MetadataCensusReceiptError(f"cannot inspect Databento export source: {exc}") from exc
    historical_exports = [
        alias
        for node in root_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "databento.historical.client"
        for alias in node.names
        if alias.name == "Historical" and alias.asname is None
    ]
    _require(len(historical_exports) == 1, "Databento Historical export path drifted")

    client_path = Path(distribution.locate_file("databento/historical/client.py"))
    historical_init = _find_method(
        client_path,
        class_name="Historical",
        method_name="__init__",
    )
    _require(
        [argument.arg for argument in historical_init.args.args]
        == ["self", "key", "gateway"],
        "Historical constructor arguments drifted",
    )
    _require(len(historical_init.args.defaults) == 2, "Historical constructor defaults drifted")

    def call_target(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{call_target(node.value)}.{node.attr}"
        if isinstance(node, ast.Call):
            return f"{call_target(node.func)}()"
        return ast.dump(node, include_attributes=False)

    historical_calls = [
        call_target(node.func)
        for node in ast.walk(historical_init)
        if isinstance(node, ast.Call)
    ]
    expected_historical_calls = [
        "BatchHttpAPI",
        "MetadataHttpAPI",
        "SymbologyHttpAPI",
        "TimeseriesHttpAPI",
        "logger.info",
        "os.environ.get",
        "key.isspace",
        "ValueError",
        "HistoricalGateway",
        "isinstance",
        "validate_gateway",
        "type",
        "str",
    ]
    _require(
        historical_calls == expected_historical_calls,
        "Historical constructor call-target allowlist drifted",
    )

    constructor_specs = {
        "databento/historical/api/metadata.py": "MetadataHttpAPI",
        "databento/historical/api/symbology.py": "SymbologyHttpAPI",
        "databento/historical/api/batch.py": "BatchHttpAPI",
        "databento/historical/api/timeseries.py": "TimeseriesHttpAPI",
    }
    endpoint_constructors: dict[str, Any] = {}
    for relative, class_name in constructor_specs.items():
        source_path = Path(distribution.locate_file(relative))
        constructor = _find_method(
            source_path,
            class_name=class_name,
            method_name="__init__",
        )
        _require(
            [argument.arg for argument in constructor.args.args]
            == ["self", "key", "gateway"],
            f"{class_name} constructor arguments drifted",
        )
        calls = [
            call_target(node.func)
            for node in ast.walk(constructor)
            if isinstance(node, ast.Call)
        ]
        _require(calls == ["super().__init__", "super"], f"{class_name} constructor call targets drifted")
        endpoint_constructors[class_name] = {
            "source_file": relative,
            "arguments": ["key", "gateway"],
            "call_targets": calls,
        }

    identity = {
        "databento": {
            "distribution": str(package),
            "version": distribution.version,
            "distribution_metadata": {
                "path": metadata_relative,
                "file_sha256": file_sha256(metadata_path),
            },
            "endpoint_source_files": files,
            "transport_source": {
                "path": http_relative,
                "file_sha256": http_hash,
                "observed_methods": transport_methods,
            },
            "client_construction": {
                "exact_class": "databento.historical.client.Historical",
                "source_files": observed_client_sources,
                "root_export_count": len(historical_exports),
                "constructor_call_targets": historical_calls,
                "endpoint_constructors": endpoint_constructors,
            },
        },
        "requests": {
            "distribution": str(requests_name),
            "version": requests_distribution.version,
            "distribution_metadata": {
                "path": requests_metadata_relative,
                "file_sha256": file_sha256(requests_metadata_path),
            },
            "source_files": observed_requests_sources,
            "default_retries": 0,
            "retry_zero_read_false_call_count": len(retry_zero_calls),
        },
        "urllib3": {
            "distribution": str(urllib_name),
            "version": urllib_distribution.version,
            "distribution_metadata": {
                "path": urllib_metadata_relative,
                "file_sha256": file_sha256(urllib_metadata_path),
            },
            "source_files": observed_urllib_sources,
        },
        "endpoint_contract_signatures": dict(expected_signatures),
        "observed_endpoint_source_arguments": observed_methods,
        "bounded_claims": {
            "transport": urllib_contract["bounded_claim"],
            "client_construction": client_contract["bounded_claim"],
        },
        "client_imported": False,
        "static_verification_only": True,
    }
    identity["identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    return identity


def verify_junit_report(
    path: Path,
    *,
    required_test_names: Sequence[str],
) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise MetadataCensusReceiptError(f"invalid JUnit report: {exc}") from exc
    cases = list(root.iter("testcase"))
    observed_names = [case.get("name") for case in cases]
    missing = sorted(
        required
        for required in required_test_names
        if required not in observed_names
        and not any(
            isinstance(observed, str) and observed.startswith(required + "[")
            for observed in observed_names
        )
    )
    failures = sum(bool(list(case.iter("failure"))) for case in cases)
    errors = sum(bool(list(case.iter("error"))) for case in cases)
    skipped = sum(bool(list(case.iter("skipped"))) for case in cases)
    _require(bool(cases), "JUnit report contains no tests")
    _require(not failures and not errors and not skipped, f"JUnit report is not all-pass: failures={failures} errors={errors} skipped={skipped}")
    _require(not missing, f"JUnit report lacks required tests: {missing}")
    _require(len(set(required_test_names)) == len(required_test_names), "program contract duplicates a required test")
    return {
        "tests": len(cases),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "required_tests_present": len(required_test_names),
        "required_test_names": list(required_test_names),
        "report_file_sha256": file_sha256(path),
    }


def _required_test_source_audit(
    repo_root: Path,
    required_test_names: Sequence[str],
) -> dict[str, Any]:
    """Refuse an all-pass JUnit receipt backed by empty named scaffolds."""

    relative = Path("v5/tests/test_cmbp_metadata_census.py")
    path = canonical_repository_artifact_path(
        repo_root,
        Path(repo_root) / relative,
        relative,
        "Job-50 test source",
    )
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise MetadataCensusReceiptError(f"cannot inspect Job-50 test source: {exc}") from exc
    tests = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }
    _require(
        set(tests) == set(required_test_names),
        "Job-50 test source does not contain exactly the sealed test names",
    )
    node_counts: dict[str, int] = {}
    for name in required_test_names:
        node = tests[name]
        substantive = [
            statement
            for statement in node.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        _require(bool(substantive), f"required test is an empty scaffold: {name}")
        _require(
            not all(
                isinstance(statement, ast.Pass)
                or (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and statement.value.value is Ellipsis
                )
                for statement in substantive
            ),
            f"required test is a pass/ellipsis scaffold: {name}",
        )
        node_counts[name] = sum(1 for _ in ast.walk(node))
    return {
        "path": relative.as_posix(),
        "file_sha256": file_sha256(path),
        "exact_required_test_names": list(required_test_names),
        "non_scaffold_ast_node_counts": node_counts,
    }


def _implementation_hashes(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for relative in BOUND_IMPLEMENTATION_FILES:
        path = canonical_repository_artifact_path(
            repo_root,
            Path(repo_root) / relative,
            Path(relative),
            "bound implementation",
        )
        values[relative] = file_sha256(path)
    return values


def _receipt_runtime_import_audit(repo_root: Path) -> dict[str, Any]:
    roots: dict[str, list[str]] = {}
    for relative in RECEIPT_RUNTIME_FILES:
        path = canonical_repository_artifact_path(
            repo_root,
            Path(repo_root) / relative,
            Path(relative),
            "receipt runtime",
        )
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetadataCensusReceiptError(f"cannot inspect receipt runtime {relative}: {exc}") from exc
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = sorted(imported & FORBIDDEN_RUNTIME_IMPORTS)
        _require(not forbidden, f"receipt runtime imports forbidden dependency {relative}: {forbidden}")
        roots[relative] = sorted(imported)
    return {
        "audited_files": list(RECEIPT_RUNTIME_FILES),
        "direct_import_roots": roots,
        "forbidden_import_roots": sorted(FORBIDDEN_RUNTIME_IMPORTS),
        "forbidden_imports_observed": [],
    }


def _verify_synthetic_execution(
    *,
    declaration: Mapping[str, Any],
    response_path: Path,
    journal_path: Path,
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = strict_json(response_path)
    try:
        census.validate_execution_response(
            response,
            declaration,
            call_journal_path=journal_path,
        )
    except (census.MetadataCensusError, catalogue.CataloguePreflightError) as exc:
        raise MetadataCensusReceiptError(
            f"synthetic execution does not verify: {exc}",
            status=getattr(exc, "status", STOP_STATUS),
        ) from exc
    audit = response.get("execution_audit")
    _require(isinstance(audit, Mapping), "synthetic execution audit is missing")
    _require(response.get("source") == "synthetic", "readiness response is not synthetic")
    _require(audit.get("response_source") == "synthetic", "execution audit falsely claims external source")
    _require(audit.get("contract_sha256") == contract_sha256, "synthetic execution contract link drifted")
    _require(audit.get("authorization_sha256") is None, "synthetic execution claims vendor authorization")
    _require(audit.get("authorization_effect") == "NONE", "synthetic execution claims authority")
    _require(audit.get("expected_call_count") == census.EXPECTED_CALL_COUNT, "synthetic expected-call count drifted")
    _require(audit.get("completed_call_count") == census.EXPECTED_CALL_COUNT, "synthetic execution is incomplete")
    _require(audit.get("method_counts") == census.EXPECTED_METHOD_COUNTS, "synthetic method counts drifted")
    _require_sha(audit.get("raw_results_sha256"), "synthetic raw-results aggregate hash")
    _require_sha(audit.get("normalized_results_sha256"), "synthetic normalized-results aggregate hash")
    _require(audit.get("sdk_package") == census.EXPECTED_SDK_PACKAGE, "synthetic SDK package drifted")
    _require(audit.get("sdk_version") == census.EXPECTED_SDK_VERSION, "synthetic SDK version drifted")
    _require(audit.get("automatic_retries") == 0, "synthetic execution used retries")
    _require(audit.get("timeseries_calls") == 0, "synthetic execution recorded time-series calls")
    _require(audit.get("download_calls") == 0, "synthetic execution recorded download calls")
    _require(audit.get("data_downloaded") is False, "synthetic execution downloaded data")
    _require(audit.get("outcomes_read") is False, "synthetic execution read outcomes")
    attestation = response.get("method_attestation")
    _require(isinstance(attestation, Mapping), "synthetic method-shape attestation is missing")
    _require(attestation.get("methods_used") == [], "synthetic response claims external method use")
    _require(attestation.get("request_shapes_exercised") == list(census.METHOD_ORDER), "synthetic request shapes drifted")
    _require(attestation.get("timeseries_calls") == 0 and attestation.get("download_calls") == 0, "synthetic response widens method scope")
    binding = audit.get("call_journal")
    _require(isinstance(binding, Mapping), "synthetic journal binding is missing")
    _require(binding.get("file_sha256") == file_sha256(journal_path), "synthetic journal raw hash drifted")
    _require((Path(journal_path).stat().st_mode & 0o777) == 0o600, "synthetic journal mode is not 0600")
    return response, {
        "attempt_id": audit.get("attempt_id"),
        "expected_call_count": audit.get("expected_call_count"),
        "completed_call_count": audit.get("completed_call_count"),
        "method_counts": dict(audit.get("method_counts", {})),
        "call_order_sha256": audit.get("call_order_sha256"),
        "raw_results_sha256": audit.get("raw_results_sha256"),
        "normalized_results_sha256": audit.get("normalized_results_sha256"),
        "journal": {
            "file_sha256": binding.get("file_sha256"),
            "terminal_sequence": binding.get("terminal_sequence"),
            "terminal_head": binding.get("terminal_head"),
            "response_core_sha256": binding.get("response_core_sha256"),
            "file_mode": "0600",
        },
        "response": {
            "semantic_sha256": census.json_sha256(response),
            "file_sha256": file_sha256(response_path),
        },
    }


def build_local_readiness_receipt(
    *,
    repo_root: Path,
    test_report_path: Path,
    synthetic_journal_path: Path,
    synthetic_response_path: Path,
    require_vendor_authorization_absent: bool = True,
) -> dict[str, Any]:
    """Reconstruct one Job-50 local-only readiness receipt from local evidence."""

    repo_root = Path(repo_root).resolve()
    work = repo_root / WORK_RELATIVE
    base_contract_path = canonical_repository_artifact_path(
        repo_root,
        repo_root / BASE_CONTRACT_RELATIVE,
        BASE_CONTRACT_RELATIVE,
        "V1 contract",
    )
    intermediate_contract_path = canonical_repository_artifact_path(
        repo_root,
        repo_root / INTERMEDIATE_CONTRACT_RELATIVE,
        INTERMEDIATE_CONTRACT_RELATIVE,
        "V2 contract",
    )
    effective_contract_path = canonical_repository_artifact_path(
        repo_root,
        repo_root / EFFECTIVE_CONTRACT_RELATIVE,
        EFFECTIVE_CONTRACT_RELATIVE,
        "V3 contract",
    )
    plan_path = canonical_repository_artifact_path(
        repo_root,
        repo_root / PLAN_RELATIVE,
        PLAN_RELATIVE,
        "Job-50 plan",
    )
    test_report_path = canonical_repository_artifact_path(
        repo_root,
        test_report_path,
        TEST_REPORT_RELATIVE,
        "JUnit report",
    )
    synthetic_journal_path = canonical_repository_artifact_path(
        repo_root,
        synthetic_journal_path,
        SYNTHETIC_JOURNAL_RELATIVE,
        "synthetic call journal",
    )
    synthetic_response_path = canonical_repository_artifact_path(
        repo_root,
        synthetic_response_path,
        SYNTHETIC_RESPONSE_RELATIVE,
        "synthetic metadata response",
    )

    try:
        base_contract = verify_program_contract(base_contract_path)
        intermediate_contract = verify_program_contract(intermediate_contract_path)
        effective_contract = verify_program_contract(effective_contract_path)
    except MetadataCensusReceiptError as exc:
        raise MetadataCensusReceiptError(
            str(exc),
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        ) from exc
    base_semantic = str(base_contract["self_hash"]["value"])
    intermediate_semantic = str(intermediate_contract["self_hash"]["value"])
    effective_semantic = str(effective_contract["self_hash"]["value"])
    base_raw = file_sha256(base_contract_path)
    intermediate_raw = file_sha256(intermediate_contract_path)
    effective_raw = file_sha256(effective_contract_path)

    v2_supersession = intermediate_contract.get("supersession")
    _require(isinstance(v2_supersession, Mapping), "V2 supersession block is missing", status="STOP_CONTRACT_OR_DECLARATION_DRIFT")
    _require(
        v2_supersession.get("base_contract_semantic_sha256") == base_semantic
        and v2_supersession.get("base_contract_file_sha256") == base_raw,
        "V2/V1 contract chain drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    v3_supersession = effective_contract.get("supersession")
    _require(isinstance(v3_supersession, Mapping), "V3 supersession block is missing", status="STOP_CONTRACT_OR_DECLARATION_DRIFT")
    _require(
        v3_supersession.get("base_contract_semantic_sha256") == intermediate_semantic
        and v3_supersession.get("base_contract_file_sha256") == intermediate_raw,
        "V3/V2 contract chain drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    _require(
        v3_supersession.get("v1_contract_semantic_sha256") == base_semantic
        and v3_supersession.get("v1_contract_file_sha256") == base_raw,
        "V3/V1 preserved contract chain drifted",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )

    try:
        declaration, job49_bindings = _verify_job49_bindings(repo_root, base_contract)
    except MetadataCensusReceiptError as exc:
        raise MetadataCensusReceiptError(
            str(exc),
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        ) from exc
    evidence_gate = base_contract.get("local_evidence_gate")
    _require(
        isinstance(evidence_gate, Mapping),
        "local evidence-gate contract is missing",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    v1_tests = evidence_gate.get("required_tests")
    v2_delta = intermediate_contract.get("local_readiness_receipt_v2_delta")
    v3_delta = effective_contract.get("local_readiness_v3_delta")
    _require(
        isinstance(v2_delta, Mapping),
        "V2 local-readiness delta is missing",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    _require(
        isinstance(v3_delta, Mapping),
        "V3 local-readiness delta is missing",
        status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
    )
    v2_tests = v2_delta.get("required_tests_added_to_v1")
    v3_tests = v3_delta.get("required_tests_added_to_v2")
    for label, value in (
        ("V1", v1_tests),
        ("V2", v2_tests),
        ("V3", v3_tests),
    ):
        _require(
            isinstance(value, list) and all(isinstance(item, str) for item in value),
            f"{label} required-test contract is invalid",
            status="STOP_CONTRACT_OR_DECLARATION_DRIFT",
        )
    required_tests = [*v1_tests, *v2_tests, *v3_tests]
    _require(v2_delta.get("required_test_total") == 22, "V2 required-test total drifted", status="STOP_CONTRACT_OR_DECLARATION_DRIFT")
    _require(v3_delta.get("required_test_total") == 26, "V3 required-test total drifted", status="STOP_CONTRACT_OR_DECLARATION_DRIFT")
    _require(len(required_tests) == 26, "effective Job-50 contract must freeze exactly 26 tests", status="STOP_CONTRACT_OR_DECLARATION_DRIFT")
    _require(len(set(required_tests)) == 26, "effective Job-50 contract duplicates required tests", status="STOP_CONTRACT_OR_DECLARATION_DRIFT")
    tests = verify_junit_report(test_report_path, required_test_names=required_tests)
    _, execution = _verify_synthetic_execution(
        declaration=declaration,
        response_path=synthetic_response_path,
        journal_path=synthetic_journal_path,
        contract_sha256=effective_semantic,
    )
    execution["journal"]["path"] = SYNTHETIC_JOURNAL_RELATIVE.as_posix()
    execution["response"]["path"] = SYNTHETIC_RESPONSE_RELATIVE.as_posix()
    _require(execution["expected_call_count"] == 2440, "receipt does not bind 2,440 planned calls")
    _require(
        execution["method_counts"]
        == {
            "metadata.get_dataset_range": 1,
            "symbology.resolve": 813,
            "metadata.get_cost": 813,
            "metadata.get_record_count": 813,
        },
        "receipt method-count population drifted",
    )

    if require_vendor_authorization_absent:
        authorization_paths = list(work.rglob("VENDOR_RUN_AUTHORIZATION*.json"))
        consumption_paths = list(
            (work / "authorization-consumptions").glob("*.json")
        ) if (work / "authorization-consumptions").exists() else []
        attempt_paths = list((work / "external-attempts").iterdir()) if (
            work / "external-attempts"
        ).exists() else []
        _require(not authorization_paths, "local readiness build found a vendor-run authorization artifact", status="JOB50_AUTHORITY_VIOLATION")
        _require(not consumption_paths, "local readiness build found an authorization-consumption artifact", status="JOB50_AUTHORITY_VIOLATION")
        _require(not attempt_paths, "local readiness build found an external-attempt artifact", status="JOB50_AUTHORITY_VIOLATION")

    try:
        sdk_source_identity = _sdk_source_identity(
            base_contract,
            intermediate_contract,
            effective_contract,
        )
    except MetadataCensusReceiptError as exc:
        raise MetadataCensusReceiptError(
            str(exc),
            status="STOP_SDK_VERSION_OR_SIGNATURE_DRIFT",
        ) from exc

    receipt: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "job_id": 50,
        "status": STATUS,
        "status_meaning": (
            "The exact sealed Job-49 metadata request population passed a complete local synthetic "
            "runner rehearsal with durable call accounting. No Databento authentication, vendor call, "
            "external preflight, acquisition, outcome access, or trading authority follows."
        ),
        "next_state": NEXT_STATE,
        "local_gate": {
            "synthetic_only": True,
            "external_preflight_achieved": False,
            "vendor_authorization_artifact_present": False,
            "authorization_absence_law": (
                "Verified at the original local seal. Historical native verification may set "
                "require_vendor_authorization_absent=false after a separately authorized later attempt; "
                "that mode does not rewrite this frozen fact."
            ),
            "dataset": declaration["dataset"],
            "schema": declaration["schema"],
            "stype_in": declaration["stype_in"],
            "source_session_count": declaration["source_session_count"],
            "request_session_count": declaration["request_session_count"],
            "excluded_known_precoverage_session_count": declaration[
                "excluded_pre_event_era_session_count"
            ],
            "session_symbol_membership_count": declaration[
                "session_symbol_membership_count"
            ],
            "sdk_package": census.EXPECTED_SDK_PACKAGE,
            "sdk_version": census.EXPECTED_SDK_VERSION,
            **execution,
        },
        "bindings": {
            "plan_file_sha256": file_sha256(plan_path),
            "plan_path": PLAN_RELATIVE.as_posix(),
            "program_contract_path": EFFECTIVE_CONTRACT_RELATIVE.as_posix(),
            "program_contract_semantic_sha256": effective_semantic,
            "program_contract_file_sha256": effective_raw,
            "intermediate_program_contract_path": INTERMEDIATE_CONTRACT_RELATIVE.as_posix(),
            "intermediate_program_contract_semantic_sha256": intermediate_semantic,
            "intermediate_program_contract_file_sha256": intermediate_raw,
            "base_program_contract_path": BASE_CONTRACT_RELATIVE.as_posix(),
            "base_program_contract_semantic_sha256": base_semantic,
            "base_program_contract_file_sha256": base_raw,
            "sealed_job49": job49_bindings,
            "implementation_file_sha256": _implementation_hashes(repo_root),
            "sdk_source_identity": sdk_source_identity,
            "test_report": {
                "path": TEST_REPORT_RELATIVE.as_posix(),
                **tests,
            },
            "test_source_audit": _required_test_source_audit(
                repo_root,
                required_tests,
            ),
            "synthetic_evidence_paths": {
                "journal": SYNTHETIC_JOURNAL_RELATIVE.as_posix(),
                "response": SYNTHETIC_RESPONSE_RELATIVE.as_posix(),
            },
            "receipt_runtime_import_audit": _receipt_runtime_import_audit(repo_root),
        },
        "claim_boundary": {
            "vendor_availability": "UNKNOWN",
            "exact_external_session_count": "UNKNOWN",
            "exact_external_cost": "UNKNOWN",
            "zero_price_boundary": "UNKNOWN",
            "population_event_prevalence": "UNKNOWN",
            "strategy_economics": "NOT_READ",
            "data_acquisition_authorized": False,
            "local_credential_zero_definition": v3_delta[
                "local_credential_zero_definition"
            ],
            "tamper_evidence": intermediate_contract["tamper_evidence_boundary"][
                "required_plain_language_claim"
            ],
        },
        "integrity": dict(INTEGRITY_ZERO),
        "authority_effect": "NONE",
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = payload_sha256(receipt, "receipt_sha256")
    return receipt


def validate_local_readiness_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    test_report_path: Path,
    synthetic_journal_path: Path,
    synthetic_response_path: Path,
    require_vendor_authorization_absent: bool = True,
) -> None:
    """Verify the self-hash and reconstruct every current local binding."""

    _require(receipt.get("artifact_type") == ARTIFACT_TYPE, "wrong Job-50 receipt artifact")
    _require(receipt.get("schema_version") == SCHEMA_VERSION, "wrong Job-50 receipt schema")
    _require(receipt.get("status") == STATUS, "wrong Job-50 receipt status")
    _require(receipt.get("next_state") == NEXT_STATE, "Job-50 next state drifted")
    claimed = _require_sha(receipt.get("receipt_sha256"), "Job-50 receipt self-hash")
    _require(claimed == payload_sha256(receipt, "receipt_sha256"), "Job-50 receipt self-hash mismatch")
    _require(receipt.get("integrity") == INTEGRITY_ZERO, "Job-50 zero-action evidence drifted")
    _require(receipt.get("authority_effect") == "NONE", "Job-50 receipt claims authority")
    rebuilt = build_local_readiness_receipt(
        repo_root=repo_root,
        test_report_path=test_report_path,
        synthetic_journal_path=synthetic_journal_path,
        synthetic_response_path=synthetic_response_path,
        require_vendor_authorization_absent=require_vendor_authorization_absent,
    )
    _require(dict(receipt) == rebuilt, "Job-50 receipt no longer matches bound local artifacts")
