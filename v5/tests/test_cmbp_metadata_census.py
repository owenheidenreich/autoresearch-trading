"""No-network refusal, provenance, and accounting tests for Job 50."""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import inspect
import json
import shutil
import socket
import stat
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import pytest

from v5.research import cmbp_catalogue_preflight as catalogue
from v5.research import cmbp_metadata_census as census
from v5.research import cmbp_metadata_census_receipt as readiness


REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "v5/work/cmbp-metadata-census"
DECLARATION_PATH = (
    REPO / "v5/work/human-policy-foundation/CMBP_CATALOGUE_DECLARATION_V1.json"
)
V1_PATH = WORK / "PROGRAM_CONTRACT_V1.json"
V2_PATH = WORK / "PROGRAM_CONTRACT_V2.json"
V3_PATH = WORK / "PROGRAM_CONTRACT_V3.json"
FIXED_NOW = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")
    path.chmod(mode)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _declaration() -> dict[str, Any]:
    return _load(DECLARATION_PATH)


def _first_request_session(declaration: Mapping[str, Any]) -> str:
    return next(
        str(row["session"])
        for row in declaration["sessions"]
        if row["requests"]
    )


def _journal_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    records = [json.loads(line) for line in raw.splitlines()]
    previous = census.GENESIS_HASH
    for sequence, (line, record) in enumerate(zip(raw.splitlines(), records, strict=True)):
        assert line == _canonical_bytes(record)
        assert record["sequence"] == sequence
        assert record["previous_hash"] == previous
        unsigned = dict(record)
        claimed = unsigned.pop("record_hash")
        assert claimed == _sha256(unsigned)
        previous = claimed
    return records


def _run_fake(
    root: Path,
    declaration: dict[str, Any],
    *,
    name: str,
    overrides: Mapping[tuple[str, str | None], Any] | None = None,
    contract_sha256: str = census.PROGRAM_CONTRACT_SHA256,
) -> tuple[dict[str, Any], census.SyntheticMetadataClient, Path, Path]:
    client = census.SyntheticMetadataClient(declaration, overrides=overrides)
    journal = root / f"{name}.jsonl"
    response_path = root / f"{name}.json"
    response = census._run_metadata_census_with_injected_synthetic_client(
        client,
        declaration,
        call_journal_path=journal,
        response_path=response_path,
        contract_sha256=contract_sha256,
        attempt_id=f"synthetic-{name}",
    )
    return response, client, journal, response_path


@pytest.fixture(scope="module")
def complete_fake_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path]:
    declaration = _declaration()
    response, client, journal, response_path = _run_fake(
        tmp_path_factory.mktemp("job50-complete"),
        declaration,
        name="complete",
    )
    return declaration, response, client, journal, response_path


class _EnvironmentProbe:
    def __init__(self, value: str | None) -> None:
        self.value = value
        self.reads = 0

    def read(self) -> str | None:
        self.reads += 1
        return self.value


def _stage_authorization_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cap: str = "1",
) -> dict[str, Any]:
    from v5.ops import run_cmbp_metadata_census as cli

    root = tmp_path / "repo"
    work = root / "v5/work/cmbp-metadata-census"
    foundation = root / "v5/work/human-policy-foundation"
    work.mkdir(parents=True)
    foundation.mkdir(parents=True)
    active_contract_sources = {
        "program": Path(cli.DEFAULT_CONTRACT),
        "intermediate": Path(cli.DEFAULT_INTERMEDIATE_CONTRACT),
        "base": Path(cli.DEFAULT_BASE_CONTRACT),
    }
    if hasattr(cli, "DEFAULT_PRIOR_CONTRACT"):
        active_contract_sources["prior"] = Path(cli.DEFAULT_PRIOR_CONTRACT)
    active_contract_paths = {
        role: work / source.name
        for role, source in active_contract_sources.items()
    }
    copied: set[Path] = set()
    for source, target in (
        *(
            (source, active_contract_paths[role])
            for role, source in active_contract_sources.items()
        ),
        (DECLARATION_PATH, foundation / DECLARATION_PATH.name),
    ):
        if target in copied:
            continue
        shutil.copy2(source, target)
        copied.add(target)

    identity: dict[str, Any] = {"synthetic_test_identity": True}
    identity["identity_sha256"] = _sha256(identity)
    local_receipt = {
        "receipt_sha256": "b" * 64,
        "bindings": {"sdk_source_identity": identity},
    }
    receipt_module = cli.readiness
    readiness_path = work / Path(cli.DEFAULT_READINESS_RECEIPT).name
    _write_json(readiness_path, local_receipt)
    declaration_path = foundation / DECLARATION_PATH.name
    declaration = _load(declaration_path)
    attempt_id = "11111111-1111-4111-8111-111111111111"
    authorization_id = "22222222-2222-4222-8222-222222222222"
    authorization: dict[str, Any] = {
        "artifact_type": census.VENDOR_AUTH_ARTIFACT,
        "schema_version": census.VENDOR_AUTH_SCHEMA_VERSION,
        "authorization_id": authorization_id,
        "attempt_id": attempt_id,
        "issued_at_utc": "2026-08-24T18:00:00.000000Z",
        "expires_at_utc": "2026-08-24T23:00:00.000000Z",
        "program_contract_sha256": census.PROGRAM_CONTRACT_SHA256,
        "intermediate_program_contract_sha256": census.INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
        "intermediate_program_contract_file_sha256": census.INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256,
        "base_program_contract_sha256": census.BASE_PROGRAM_CONTRACT_SHA256,
        "base_program_contract_file_sha256": census.BASE_PROGRAM_CONTRACT_FILE_SHA256,
        "local_readiness_receipt_sha256": local_receipt["receipt_sha256"],
        "local_readiness_receipt_file_sha256": census.file_sha256(readiness_path),
        "declaration_sha256": declaration["declaration_sha256"],
        "declaration_file_sha256": census.file_sha256(declaration_path),
        "credential_rotation_attested": True,
        "credential_attestation_recorded_at_utc": "2026-08-24T17:59:00.000000Z",
        "quoted_acquisition_ceiling_usd": cap,
        "metadata_only": True,
        "authorized_methods": list(census.METHOD_ORDER),
        "current_conversation_authorization_sha256": "c" * 64,
        "authorization_effect": census.AUTHORIZATION_EFFECT,
        "one_attempt_only": True,
        "authorization_sha256": None,
    }
    if hasattr(census, "PRIOR_PROGRAM_CONTRACT_SHA256"):
        authorization.update(
            {
                "prior_program_contract_sha256": census.PRIOR_PROGRAM_CONTRACT_SHA256,
                "prior_program_contract_file_sha256": census.PRIOR_PROGRAM_CONTRACT_FILE_SHA256,
            }
        )
    authorization["authorization_sha256"] = census.self_hash(
        authorization, "authorization_sha256"
    )
    authorization_path = (
        work
        / "authorizations"
        / attempt_id
        / "VENDOR_RUN_AUTHORIZATION_V2.json"
    )
    _write_json(authorization_path, authorization, mode=0o600)

    monkeypatch.setattr(cli, "REPO", root)
    monkeypatch.setattr(cli, "WORK", work)
    monkeypatch.setattr(cli, "DEFAULT_DECLARATION", declaration_path)
    monkeypatch.setattr(cli, "DEFAULT_CONTRACT", active_contract_paths["program"])
    monkeypatch.setattr(
        cli,
        "DEFAULT_INTERMEDIATE_CONTRACT",
        active_contract_paths["intermediate"],
    )
    monkeypatch.setattr(cli, "DEFAULT_BASE_CONTRACT", active_contract_paths["base"])
    if "prior" in active_contract_paths:
        monkeypatch.setattr(
            cli,
            "DEFAULT_PRIOR_CONTRACT",
            active_contract_paths["prior"],
        )
    monkeypatch.setattr(cli, "DEFAULT_READINESS_RECEIPT", readiness_path)
    monkeypatch.setattr(
        cli,
        "DEFAULT_TEST_REPORT",
        work / Path(cli.DEFAULT_TEST_REPORT).name,
    )
    monkeypatch.setattr(
        cli,
        "DEFAULT_SYNTHETIC_JOURNAL",
        work / Path(cli.DEFAULT_SYNTHETIC_JOURNAL).name,
    )
    monkeypatch.setattr(
        cli,
        "DEFAULT_SYNTHETIC_RESPONSE",
        work / Path(cli.DEFAULT_SYNTHETIC_RESPONSE).name,
    )
    monkeypatch.setattr(
        receipt_module,
        "validate_local_readiness_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(census, "_authorization_now_utc", lambda: FIXED_NOW)
    return {
        "cli": cli,
        "root": root,
        "work": work,
        "declaration": declaration,
        "authorization": authorization,
        "authorization_path": authorization_path,
        "attempt_id": attempt_id,
        "local_receipt": local_receipt,
        "readiness_path": readiness_path,
        "receipt_module": receipt_module,
        "contract_paths": active_contract_paths,
        "test_report_path": Path(cli.DEFAULT_TEST_REPORT),
        "synthetic_journal_path": Path(cli.DEFAULT_SYNTHETIC_JOURNAL),
        "synthetic_response_path": Path(cli.DEFAULT_SYNTHETIC_RESPONSE),
    }


def _verify_staged_authorization(
    staged: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    attempt_id: str | None = None,
) -> Decimal:
    paths = staged["contract_paths"]
    kwargs: dict[str, Any] = {
        "intermediate_contract": _load(paths["intermediate"]),
        "base_contract": _load(paths["base"]),
        "local_readiness_receipt": staged["local_receipt"],
        "local_readiness_receipt_file_sha256": census.file_sha256(
            staged["readiness_path"]
        ),
        "authorization_path": staged["authorization_path"],
        "repo_root": staged["root"],
        "attempt_id": attempt_id or staged["attempt_id"],
        "declaration_file_sha256": staged["authorization"][
            "declaration_file_sha256"
        ],
    }
    if "prior" in paths:
        assert "prior_contract" in inspect.signature(
            census.verify_vendor_run_authorization
        ).parameters
        kwargs["prior_contract"] = _load(paths["prior"])
    return census.verify_vendor_run_authorization(
        candidate,
        staged["declaration"],
        _load(paths["program"]),
        **kwargs,
    )


def _mint_fake_capability_for_refusal(
    staged: Mapping[str, Any],
) -> tuple[
    census.AuthorizedExternalClient,
    census.SyntheticMetadataClient,
    Path,
    Path,
]:
    """Mint an exact-class capability that tests refusal before call one.

    The caller must invalidate a retained binding before passing this object to
    the public external core.  This helper never executes or finalizes an
    external attempt and therefore cannot create counterfeit vendor evidence.
    """

    authorization = staged["authorization"]
    consumption, consumption_path, attempt_directory = census.consume_vendor_run_authorization(
        authorization,
        authorization_path=staged["authorization_path"],
        repo_root=staged["root"],
        local_readiness_receipt=staged["local_receipt"],
        consumed_at_utc="2026-08-24T20:00:00.000000Z",
    )
    client = census.SyntheticMetadataClient(staged["declaration"])
    bindings = {
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": census.file_sha256(staged["authorization_path"]),
        "attempt_id": staged["attempt_id"],
        "attempt_directory": attempt_directory,
        "consumption_record_path": consumption_path,
        "consumption_marker_sha256": census.file_sha256(consumption_path),
        "local_readiness_receipt_sha256": staged["local_receipt"]["receipt_sha256"],
        "local_readiness_receipt_file_sha256": census.file_sha256(
            staged["readiness_path"]
        ),
        "base_program_contract_sha256": census.BASE_PROGRAM_CONTRACT_SHA256,
        "intermediate_program_contract_sha256": census.INTERMEDIATE_PROGRAM_CONTRACT_SHA256,
        "program_contract_sha256": census.PROGRAM_CONTRACT_SHA256,
        "declaration_sha256": staged["declaration"]["declaration_sha256"],
        "declaration_file_sha256": authorization["declaration_file_sha256"],
        "sdk_identity_sha256": census._readiness_sdk_identity_sha256(staged["local_receipt"]),
        "quoted_acquisition_ceiling_usd": authorization["quoted_acquisition_ceiling_usd"],
        "repo_root": staged["root"],
        "declaration_path": staged["root"] / "v5/work/human-policy-foundation" / DECLARATION_PATH.name,
        "program_contract_path": staged["contract_paths"]["program"],
        "intermediate_program_contract_path": staged["contract_paths"]["intermediate"],
        "base_program_contract_path": staged["contract_paths"]["base"],
        "local_readiness_receipt_path": staged["readiness_path"],
        "readiness_test_report_path": staged["test_report_path"],
        "synthetic_journal_path": staged["synthetic_journal_path"],
        "synthetic_response_path": staged["synthetic_response_path"],
        "authorization_path": staged["authorization_path"],
    }
    optional_bindings = {
        "program_contract_file_sha256": getattr(
            census, "PROGRAM_CONTRACT_FILE_SHA256", None
        ),
        "prior_program_contract_sha256": getattr(
            census, "PRIOR_PROGRAM_CONTRACT_SHA256", None
        ),
        "prior_program_contract_file_sha256": getattr(
            census, "PRIOR_PROGRAM_CONTRACT_FILE_SHA256", None
        ),
        "prior_program_contract_path": staged["contract_paths"].get("prior"),
        "intermediate_program_contract_file_sha256": getattr(
            census, "INTERMEDIATE_PROGRAM_CONTRACT_FILE_SHA256", None
        ),
        "base_program_contract_file_sha256": getattr(
            census, "BASE_PROGRAM_CONTRACT_FILE_SHA256", None
        ),
    }
    for field, value in optional_bindings.items():
        if field in census._AUTHORIZED_CAPABILITY_FIELDS:
            assert value is not None
            bindings[field] = value
    assert set(bindings) == census._AUTHORIZED_CAPABILITY_FIELDS
    capability = census._mint_nonproduction_authorized_client_for_refusal_test(
        client,
        bindings=bindings,
    )
    assert consumption["attempt_id"] == staged["attempt_id"]
    return capability, client, consumption_path, attempt_directory


def test_fake_census_calls_exact_four_methods_in_frozen_order_and_arguments(
    complete_fake_run: tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path],
) -> None:
    declaration, response, client, journal, response_path = complete_fake_run
    expected: list[tuple[str, str | None, dict[str, Any]]] = []
    expected.append(
        (
            "metadata.get_dataset_range",
            None,
            {"dataset": "OPRA.PILLAR"},
        )
    )
    for row in declaration["sessions"]:
        for request in row["requests"]:
            parameters = request["parameters"]
            if request["method"] == "symbology.resolve":
                exact = {
                    "dataset": parameters["dataset"],
                    "symbols": parameters["symbols"],
                    "stype_in": parameters["stype_in"],
                    "stype_out": parameters["stype_out"],
                    "start_date": parameters["start"],
                    "end_date": parameters["end"],
                }
            else:
                exact = {
                    "dataset": parameters["dataset"],
                    "schema": parameters["schema"],
                    "symbols": parameters["symbols"],
                    "stype_in": parameters["stype_in"],
                    "start": parameters["start"],
                    "end": parameters["end"],
                }
            expected.append(
                (request["method"], row["session"], exact)
            )
    observed = [
        (call["method"], call["session"], call["parameters"])
        for call in client.calls
    ]
    assert observed == expected
    assert len(observed) == census.EXPECTED_CALL_COUNT == 2_440
    assert Counter(call[0] for call in observed) == census.EXPECTED_METHOD_COUNTS
    assert stat.S_IMODE(journal.stat().st_mode) == 0o600
    assert stat.S_IMODE(response_path.stat().st_mode) == 0o644
    assert response["execution_audit"]["contract_sha256"] == census.PROGRAM_CONTRACT_SHA256
    census.validate_execution_response(response, declaration, call_journal_path=journal)


def test_symbology_adapter_renames_start_end_only_after_descriptor_hash_validation() -> None:
    declaration = _declaration()
    request = next(
        request
        for row in declaration["sessions"]
        for request in row["requests"]
        if request["method"] == "symbology.resolve"
    )
    frozen = copy.deepcopy(request)
    adapted = census.sdk_parameters(request)
    assert request == frozen
    assert set(adapted) == {
        "dataset", "symbols", "stype_in", "stype_out", "start_date", "end_date"
    }
    assert adapted["start_date"] == request["parameters"]["start"]
    assert adapted["end_date"] == request["parameters"]["end"]
    tampered = copy.deepcopy(request)
    tampered["parameters"]["start"] = "2023-03-27"
    with pytest.raises(census.MetadataCensusError, match="hash"):
        census.sdk_parameters(tampered)


def test_descriptor_hash_tamper_forbidden_method_and_scope_widening_fail_before_calls(
    tmp_path: Path,
) -> None:
    declaration = _declaration()
    request = next(
        request
        for row in declaration["sessions"]
        for request in row["requests"]
        if request["method"] == "metadata.get_cost"
    )
    widened = copy.deepcopy(request)
    widened["parameters"]["limit"] = 1
    widened["request_sha256"] = _sha256(
        {"method": widened["method"], "parameters": widened["parameters"]}
    )
    with pytest.raises(census.MetadataCensusError, match="argument|shape"):
        census.sdk_parameters(widened)
    parent = copy.deepcopy(request)
    parent["parameters"]["symbols"] = ["SPXW.OPT"]
    parent["request_sha256"] = _sha256(
        {"method": parent["method"], "parameters": parent["parameters"]}
    )
    with pytest.raises(census.MetadataCensusError, match="symbol|SPXW"):
        census.sdk_parameters(parent)
    forbidden = {
        "method": "timeseries.get_range",
        "parameters": {"dataset": "OPRA.PILLAR"},
    }
    forbidden["request_sha256"] = _sha256(forbidden)
    with pytest.raises(census.MetadataCensusError, match="method"):
        census.sdk_parameters(forbidden)

    drifted_declaration = copy.deepcopy(declaration)
    drifted_request = next(
        request
        for row in drifted_declaration["sessions"]
        for request in row["requests"]
        if request["method"] == "metadata.get_cost"
    )
    drifted_request["parameters"]["symbols"] = ["SPXW.OPT"]
    drifted_request["request_sha256"] = _sha256(
        {
            "method": drifted_request["method"],
            "parameters": drifted_request["parameters"],
        }
    )
    drifted_declaration["declaration_sha256"] = catalogue.self_hash(
        drifted_declaration, "declaration_sha256"
    )
    scope_client = census.SyntheticMetadataClient(declaration)
    with pytest.raises(census.MetadataCensusError, match="request|symbol|scope|drift"):
        census._run_metadata_census_with_injected_synthetic_client(
            scope_client,
            drifted_declaration,
            call_journal_path=tmp_path / "scope.jsonl",
            response_path=tmp_path / "scope.json",
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-rehashed-scope",
        )
    assert scope_client.calls == []

    client = census.SyntheticMetadataClient(declaration)
    with pytest.raises(census.MetadataCensusError, match="contract|identity"):
        census._run_metadata_census_with_injected_synthetic_client(
            client,
            declaration,
            call_journal_path=tmp_path / "wrong.jsonl",
            response_path=tmp_path / "wrong.json",
            contract_sha256="0" * 64,
            attempt_id="synthetic-wrong-contract",
        )
    assert client.calls == []


def test_sdk_version_mismatch_fails_before_calls(tmp_path: Path) -> None:
    declaration = _declaration()
    client = census.SyntheticMetadataClient(declaration)
    with pytest.raises(census.MetadataCensusError, match="SDK version"):
        census._execute_metadata_census(
            client,
            declaration,
            call_journal_path=tmp_path / "version.jsonl",
            response_path=tmp_path / "version.json",
            sdk_version="0.76.0",
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
        )
    assert client.calls == []
    assert not (tmp_path / "version.jsonl").exists()


def test_vendor_exception_leaves_durable_error_journal_and_no_final_artifact(
    tmp_path: Path,
) -> None:
    declaration = _declaration()
    session = _first_request_session(declaration)
    secret = RuntimeError("synthetic-secret-must-not-enter-journal")
    client = census.SyntheticMetadataClient(
        declaration,
        overrides={("metadata.get_record_count", session): secret},
    )
    journal = tmp_path / "error.jsonl"
    response = tmp_path / "error.json"
    with pytest.raises(census.MetadataCensusError, match="failed at call"):
        census._run_metadata_census_with_injected_synthetic_client(
            client,
            declaration,
            call_journal_path=journal,
            response_path=response,
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-error",
        )
    records = _journal_records(journal)
    assert [row["record_type"] for row in records[-3:]] == [
        "CALL_START", "CALL_ERROR", "ATTEMPT_STOP"
    ]
    assert records[-1]["closed_error_class"] == "STOP_VENDOR_AUTH_OR_ENTITLEMENT"
    serialized = journal.read_text(encoding="utf-8")
    assert "synthetic-secret" not in serialized and "RuntimeError" not in serialized
    assert not response.exists()


def test_partial_or_malformed_vendor_response_fails_without_silent_drop(
    tmp_path: Path,
    complete_fake_run: tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path],
) -> None:
    declaration, valid, _, journal, _ = complete_fake_run
    dropped = copy.deepcopy(valid)
    dropped["sessions"].pop()
    with pytest.raises(census.MetadataCensusError, match="incomplete|missing"):
        census.validate_execution_response(dropped, declaration, call_journal_path=journal)
    session = _first_request_session(declaration)
    client = census.SyntheticMetadataClient(
        declaration,
        overrides={("symbology.resolve", session): {}},
    )
    with pytest.raises(census.MetadataCensusError, match="mapping|missing|symbology|incomplete"):
        census._run_metadata_census_with_injected_synthetic_client(
            client,
            declaration,
            call_journal_path=tmp_path / "malformed.jsonl",
            response_path=tmp_path / "malformed.json",
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-malformed",
        )
    assert not (tmp_path / "malformed.json").exists()


def test_zero_cost_with_positive_count_is_retained_and_zero_count_stops(
    tmp_path: Path,
    complete_fake_run: tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path],
) -> None:
    declaration, response, _, journal, _ = complete_fake_run
    assert all(Decimal(row["cost_usd"]) == 0 for row in response["sessions"])
    assert all(row["record_count"] > 0 for row in response["sessions"])
    census.validate_execution_response(response, declaration, call_journal_path=journal)
    session = _first_request_session(declaration)
    client = census.SyntheticMetadataClient(
        declaration,
        overrides={("metadata.get_record_count", session): 0},
    )
    with pytest.raises(census.MetadataCensusError, match="zero") as stopped:
        census._run_metadata_census_with_injected_synthetic_client(
            client,
            declaration,
            call_journal_path=tmp_path / "zero.jsonl",
            response_path=tmp_path / "zero.json",
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-zero",
        )
    assert stopped.value.status == "STOP_ZERO_RECORDS"
    assert not (tmp_path / "zero.json").exists()


def test_attempt_paths_are_exclusive_and_partial_attempt_cannot_resume_or_overwrite(
    tmp_path: Path,
) -> None:
    declaration = _declaration()
    occupied = tmp_path / "occupied.jsonl"
    occupied.write_text("owner-evidence\n", encoding="utf-8")
    client = census.SyntheticMetadataClient(declaration)
    with pytest.raises(census.MetadataCensusError, match="exclusive|new"):
        census._run_metadata_census_with_injected_synthetic_client(
            client,
            declaration,
            call_journal_path=occupied,
            response_path=tmp_path / "occupied.json",
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-occupied",
        )
    assert occupied.read_text(encoding="utf-8") == "owner-evidence\n"
    assert client.calls == []

    session = _first_request_session(declaration)
    partial = census.SyntheticMetadataClient(
        declaration,
        overrides={("symbology.resolve", session): RuntimeError("stop")},
    )
    journal = tmp_path / "partial.jsonl"
    response = tmp_path / "partial.json"
    with pytest.raises(census.MetadataCensusError):
        census._run_metadata_census_with_injected_synthetic_client(
            partial,
            declaration,
            call_journal_path=journal,
            response_path=response,
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-partial",
        )
    retained = journal.read_bytes()
    with pytest.raises(census.MetadataCensusError, match="exclusive|new"):
        census._run_metadata_census_with_injected_synthetic_client(
            census.SyntheticMetadataClient(declaration),
            declaration,
            call_journal_path=journal,
            response_path=response,
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-partial",
        )
    assert journal.read_bytes() == retained and not response.exists()


def test_external_client_construction_requires_fresh_authorization_credential_and_numeric_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_authorization_repo(tmp_path, monkeypatch)
    authorization = staged["authorization"]
    cap = _verify_staged_authorization(staged, authorization)
    assert cap == Decimal("1")
    invalid = dict(authorization)
    invalid["quoted_acquisition_ceiling_usd"] = "NaN"
    invalid["authorization_sha256"] = census.self_hash(invalid, "authorization_sha256")
    with pytest.raises(census.MetadataCensusError) as stopped:
        _verify_staged_authorization(staged, invalid)
    assert stopped.value.status == "STOP_QUOTED_CEILING_MISSING_OR_INVALID"

    probe = _EnvironmentProbe(None)
    monkeypatch.setattr(staged["cli"], "_read_databento_api_key", probe.read)
    with pytest.raises(census.MetadataCensusError) as missing:
        staged["cli"].construct_authorized_external_client(
            attempt_id=staged["attempt_id"]
        )
    assert missing.value.status == "STOP_VENDOR_AUTH_OR_ENTITLEMENT"
    assert probe.reads == 1


def test_fake_suite_never_imports_sdk_authenticates_or_opens_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v5.ops import run_cmbp_metadata_census as cli

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "databento" not in imported

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("synthetic path attempted network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(cli, "_read_databento_api_key", forbidden)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == "databento":
            raise AssertionError("synthetic path imported Databento")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    declaration = _declaration()
    response = census.run_synthetic_metadata_census(
        declaration,
        call_journal_path=tmp_path / "firewall.jsonl",
        response_path=tmp_path / "firewall.json",
        attempt_id="synthetic-firewall",
    )
    assert response["source"] == "synthetic"
    assert response["execution_audit"]["completed_call_count"] == 2_440


def test_every_invocation_has_durable_start_and_result_pair_bound_to_response(
    complete_fake_run: tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path],
) -> None:
    declaration, response, client, journal, _ = complete_fake_run
    records = _journal_records(journal)
    starts = [row for row in records if row["record_type"] == "CALL_START"]
    results = [row for row in records if row["record_type"] == "CALL_RESULT"]
    assert len(starts) == len(results) == len(client.calls) == 2_440
    assert records[0]["record_type"] == "ATTEMPT_HEADER"
    assert records[-1]["record_type"] == "ATTEMPT_COMPLETE"
    for ordinal, (start, result) in enumerate(zip(starts, results, strict=True), 1):
        assert start["call_ordinal"] == result["call_ordinal"] == ordinal
        assert start["request_sha256"] == result["request_sha256"]
        assert result["result_sha256"] == result["raw_result_sha256"]
        assert len(result["normalized_result_sha256"]) == 64
    audit = response["execution_audit"]
    assert audit["call_journal"]["terminal_head"] == records[-1]["record_hash"]
    assert audit["raw_results_sha256"] == records[-1]["raw_results_sha256"]
    assert audit["normalized_results_sha256"] == records[-1]["normalized_results_sha256"]
    census.validate_execution_response(response, declaration, call_journal_path=journal)


def test_external_receipt_wraps_structural_parser_receipt_and_scopes_claims(
    tmp_path: Path,
    complete_fake_run: tuple[
        dict[str, Any],
        dict[str, Any],
        census.SyntheticMetadataClient,
        Path,
        Path,
    ],
) -> None:
    declaration, response, _, _, _ = complete_fake_run
    nested = catalogue.build_preflight_receipt(
        declaration,
        response,
        hard_cap_usd=Decimal("0"),
    )
    assert nested["job49_integration_disposition"] == "VALIDATOR_REHEARSAL_ONLY"
    assert nested["external_preflight_achieved_by_job49"] is False
    assert nested["claims"]["actual_vendor_availability"] is False
    catalogue.validate_preflight_receipt(nested, declaration=declaration)

    # External provenance is closure-captured by the fully revalidating public
    # core.  The synthetic executor exposes no source, authority, cap, hash,
    # token, seal, or separately callable external engine.
    synthetic_parameters = inspect.signature(
        census._execute_metadata_census
    ).parameters
    assert not {
        "source",
        "authorization_sha256",
        "quoted_acquisition_ceiling_usd",
        "provenance",
        "seal",
        "_external_execution_seal",
    } & set(synthetic_parameters)
    authorized_source = inspect.getsource(
        census.run_authorized_external_metadata_census
    )
    assert authorized_source.index(
        "prepared = prepare_external(capability)"
    ) < authorized_source.index(
        'source="externally_supplied"'
    )
    for fake_entrypoint in (
        census._run_metadata_census_with_injected_synthetic_client,
        census._run_test_only_authorized_external_client,
        census.run_synthetic_metadata_census,
    ):
        source = inspect.getsource(fake_entrypoint)
        assert "externally_supplied" not in source
        assert "external_execution_seal" not in source

    for hidden_name in (
        "_execute_metadata_census_engine",
        "_prepare_authorized_external_metadata_census",
        "_bind_metadata_execution_entrypoints",
        "_AUTHORIZED_CORE_ACCESS_SEAL",
        "_EXTERNAL_RECEIPT_CONTEXT_SEAL",
        "_external_receipt_context_engine",
        "_build_nested_structural_receipt_engine",
        "_external_wrapper_from_context_engine",
        "_bind_external_receipt_entrypoints",
    ):
        assert not hasattr(census, hidden_name)

    # A caller-assembled context cannot create even an in-memory external
    # wrapper: no callable context/wrapper factory remains in module globals.
    assert not hasattr(census, "_external_receipt_context")
    assert not hasattr(census, "_build_nested_structural_receipt")
    assert not hasattr(census, "_external_wrapper_from_context")
    assert not list(tmp_path.rglob("NESTED_STRUCTURAL_PARSER_RECEIPT_V1.json"))
    assert not list(tmp_path.rglob("EXTERNAL_METADATA_RECEIPT_V1.json"))


def test_external_run_verifies_local_readiness_and_sdk_before_credential_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_authorization_repo(tmp_path, monkeypatch)
    events: list[str] = []

    def readiness_gate(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        events.append("readiness")

    class OrderedEnvironment(_EnvironmentProbe):
        def read(self) -> str | None:
            events.append("credential")
            return super().read()

    probe = OrderedEnvironment(None)
    monkeypatch.setattr(
        staged["receipt_module"],
        "validate_local_readiness_receipt",
        readiness_gate,
    )
    monkeypatch.setattr(staged["cli"], "_read_databento_api_key", probe.read)
    with pytest.raises(census.MetadataCensusError) as stopped:
        staged["cli"].construct_authorized_external_client(
            attempt_id=staged["attempt_id"]
        )
    assert stopped.value.status == "STOP_VENDOR_AUTH_OR_ENTITLEMENT"
    assert events == ["readiness", "credential"]
    marker_dir = staged["work"] / "authorization-consumptions"
    assert len(list(marker_dir.glob("*.json"))) == 1
    assert (staged["work"] / "external-attempts" / staged["attempt_id"]).is_dir()

    # A second fresh authorization traverses the public production constructor
    # through credential read and the post-key *full* SDK/source rehash.  The
    # private test seam stops there, before any Databento import/construction.
    traversed = _stage_authorization_repo(tmp_path / "post-key", monkeypatch)
    production_events: list[str] = []

    def fake_key() -> str:
        production_events.append("credential")
        return "synthetic-test-key"

    expected_identity = copy.deepcopy(
        traversed["local_receipt"]["bindings"]["sdk_source_identity"]
    )
    drifted_identity = copy.deepcopy(expected_identity)
    drifted_identity["synthetic_test_identity"] = False
    drifted_identity["identity_sha256"] = _sha256(
        {
            key: value
            for key, value in drifted_identity.items()
            if key != "identity_sha256"
        }
    )

    def mismatched_identity_rebuild(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert len(args) == len(traversed["contract_paths"])
        assert kwargs == {}
        production_events.append("post_key_full_identity")
        return drifted_identity

    def forbidden_import(name: str, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError(f"Historical import occurred before test-safe stop: {name}")

    monkeypatch.setattr(traversed["cli"], "_read_databento_api_key", fake_key)
    monkeypatch.setattr(
        traversed["receipt_module"],
        "_sdk_source_identity",
        mismatched_identity_rebuild,
    )
    monkeypatch.setattr(traversed["cli"].importlib, "import_module", forbidden_import)
    with pytest.raises(census.MetadataCensusError) as post_key_stop:
        traversed["cli"].construct_authorized_external_client(
            attempt_id=traversed["attempt_id"]
        )
    assert post_key_stop.value.status == "STOP_SDK_VERSION_OR_SIGNATURE_DRIFT"
    assert production_events == ["credential", "post_key_full_identity"]


def test_external_receipt_binds_auth_raw_normalized_cost_count_and_ceiling(
    tmp_path: Path,
    complete_fake_run: tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path],
) -> None:
    declaration, response, _, journal, _ = complete_fake_run
    census.validate_execution_response(
        response,
        declaration,
        call_journal_path=journal,
    )
    audit = response["execution_audit"]
    terminal = _journal_records(journal)[-1]
    assert audit["raw_results_sha256"] == terminal["raw_results_sha256"]
    assert audit["normalized_results_sha256"] == terminal["normalized_results_sha256"]

    wrapper_fields = census.EXTERNAL_RECEIPT_FIELDS
    assert {
        "authorization_sha256",
        "authorization_file_sha256",
        "authorization_consumption",
        "response",
        "call_journal",
        "exact_total_record_count",
        "exact_total_cost_usd",
        "quoted_acquisition_ceiling_usd",
        "within_quoted_acquisition_ceiling",
        "receipt_sha256",
    } <= wrapper_fields
    module_source = Path(census.__file__).read_text(encoding="utf-8")
    wrapper_start = module_source.index(
        "def _external_wrapper_from_context_engine("
    )
    wrapper_end = module_source.index(
        "def _bind_external_receipt_entrypoints("
    )
    wrapper_source = module_source[wrapper_start:wrapper_end]
    for binding in (
        '"raw_results_sha256": audit["raw_results_sha256"]',
        '"normalized_results_sha256": audit["normalized_results_sha256"]',
        '"quoted_acquisition_ceiling_usd": authorization[',
        '"exact_total_record_count": nested_receipt[',
        '"exact_total_cost_usd": nested_receipt[',
    ):
        assert binding in wrapper_source

    # Re-labeling a synthetic response cannot become external evidence: the
    # retained journal header and audit remain synthetic and validation fails.
    counterfeit = copy.deepcopy(response)
    counterfeit["source"] = "externally_supplied"
    counterfeit["execution_audit"]["response_source"] = "externally_supplied"
    counterfeit["execution_audit"]["authorization_sha256"] = "a" * 64
    with pytest.raises(census.MetadataCensusError):
        census.validate_execution_response(
            counterfeit,
            declaration,
            call_journal_path=journal,
        )
    assert not list(tmp_path.rglob("EXTERNAL_METADATA_RECEIPT_V1.json"))


def test_vendor_authorization_is_fresh_attempt_bound_and_consumed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_authorization_repo(tmp_path, monkeypatch)

    def verify(candidate: Mapping[str, Any], *, attempt_id: str | None = None) -> Decimal:
        return _verify_staged_authorization(
            staged,
            candidate,
            attempt_id=attempt_id or staged["attempt_id"],
        )

    assert "now_utc" not in inspect.signature(
        staged["cli"].construct_authorized_external_client
    ).parameters
    assert "now_utc" not in inspect.signature(
        census.run_authorized_external_metadata_census
    ).parameters
    monkeypatch.setattr(
        census,
        "_authorization_now_utc",
        lambda: datetime(2026, 8, 25, tzinfo=timezone.utc),
    )
    with pytest.raises(census.MetadataCensusError, match="fresh"):
        verify(staged["authorization"])
    monkeypatch.setattr(
        census,
        "_authorization_now_utc",
        lambda: datetime(2026, 8, 24, 17, tzinfo=timezone.utc),
    )
    with pytest.raises(census.MetadataCensusError, match="fresh"):
        verify(staged["authorization"])
    monkeypatch.setattr(census, "_authorization_now_utc", lambda: FIXED_NOW)
    false_rotation = dict(staged["authorization"])
    false_rotation["credential_rotation_attested"] = False
    false_rotation["authorization_sha256"] = census.self_hash(
        false_rotation, "authorization_sha256"
    )
    with pytest.raises(census.MetadataCensusError) as rotation:
        verify(false_rotation)
    assert rotation.value.status == "STOP_CREDENTIAL_ROTATION_UNRESOLVED"
    wrong_conversation = dict(staged["authorization"])
    wrong_conversation["current_conversation_authorization_sha256"] = "not-a-digest"
    wrong_conversation["authorization_sha256"] = census.self_hash(
        wrong_conversation, "authorization_sha256"
    )
    with pytest.raises(census.MetadataCensusError, match="conversation"):
        verify(wrong_conversation)
    with pytest.raises(census.MetadataCensusError, match="attempt"):
        verify(
            staged["authorization"],
            attempt_id="55555555-5555-4555-8555-555555555555",
        )

    probe = _EnvironmentProbe("synthetic-test-key")
    monkeypatch.setattr(staged["cli"], "_read_databento_api_key", probe.read)
    monkeypatch.setattr(
        staged["cli"],
        "_default_client_factory",
        lambda key: pytest.fail(f"replacement factory called with {key}"),
    )
    with pytest.raises(census.MetadataCensusError) as first:
        staged["cli"].construct_authorized_external_client(
            attempt_id=staged["attempt_id"]
        )
    assert first.value.status == "JOB50_AUTHORITY_VIOLATION"
    assert probe.reads == 0
    marker = (
        staged["work"]
        / "authorization-consumptions"
        / f"{staged['authorization']['authorization_sha256']}.json"
    )
    assert marker.is_file() and stat.S_IMODE(marker.stat().st_mode) == 0o600
    retained = marker.read_bytes()
    with pytest.raises(census.MetadataCensusError) as replay:
        staged["cli"].construct_authorized_external_client(
            attempt_id=staged["attempt_id"]
        )
    assert replay.value.status == "STOP_AUTHORIZATION_MISSING_OR_INVALID"
    assert marker.read_bytes() == retained

    symlinked = _stage_authorization_repo(tmp_path / "symlink-case", monkeypatch)
    outside = tmp_path / "outside-consumptions"
    outside.mkdir()
    (symlinked["work"] / "authorization-consumptions").symlink_to(
        outside, target_is_directory=True
    )
    symlink_probe = _EnvironmentProbe("synthetic-test-key")
    monkeypatch.setattr(
        symlinked["cli"], "_read_databento_api_key", symlink_probe.read
    )
    monkeypatch.setattr(
        symlinked["cli"],
        "_default_client_factory",
        symlinked["cli"]._SEALED_PRODUCTION_CLIENT_FACTORY,
    )
    with pytest.raises(census.MetadataCensusError) as symlink_stop:
        symlinked["cli"].construct_authorized_external_client(
            attempt_id=symlinked["attempt_id"]
        )
    assert symlink_stop.value.status == "STOP_ATTEMPT_PATH_EXISTS"
    assert symlink_probe.reads == 0

    attempt_symlinked = _stage_authorization_repo(
        tmp_path / "attempt-symlink-case",
        monkeypatch,
    )
    outside_attempts = tmp_path / "outside-attempts"
    outside_attempts.mkdir()
    (attempt_symlinked["work"] / "external-attempts").symlink_to(
        outside_attempts,
        target_is_directory=True,
    )
    attempt_probe = _EnvironmentProbe("synthetic-test-key")
    monkeypatch.setattr(
        attempt_symlinked["cli"],
        "_read_databento_api_key",
        attempt_probe.read,
    )
    with pytest.raises(census.MetadataCensusError) as attempt_symlink_stop:
        attempt_symlinked["cli"].construct_authorized_external_client(
            attempt_id=attempt_symlinked["attempt_id"]
        )
    assert attempt_symlink_stop.value.status == "STOP_ATTEMPT_PATH_EXISTS"
    assert attempt_probe.reads == 0


def test_core_external_execution_reverifies_full_authorization_not_bare_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _declaration()
    raw_client = census.SyntheticMetadataClient(declaration)
    with pytest.raises(census.MetadataCensusError) as raw:
        census.run_authorized_external_metadata_census(raw_client)  # type: ignore[arg-type]
    assert raw.value.status == "STOP_AUTHORIZATION_MISSING_OR_INVALID"
    with pytest.raises(census.MetadataCensusError):
        census.run_authorized_external_metadata_census("a" * 64)  # type: ignore[arg-type]
    assert raw_client.calls == []

    valid_disk_case = _stage_authorization_repo(
        tmp_path / "valid-disk-nonproduction",
        monkeypatch,
    )
    nonproduction, valid_fake, _, valid_attempt = (
        _mint_fake_capability_for_refusal(valid_disk_case)
    )
    with pytest.raises(census.MetadataCensusError) as nonproduction_stop:
        census.run_authorized_external_metadata_census(nonproduction)
    assert nonproduction_stop.value.status == "STOP_AUTHORIZATION_MISSING_OR_INVALID"
    assert valid_fake.calls == []
    assert not (valid_attempt / "CALL_JOURNAL_V1.jsonl").exists()
    assert not (valid_attempt / "EXTERNAL_METADATA_RESPONSES_V1.json").exists()
    with pytest.raises(census.MetadataCensusError, match="non-production") as build_stop:
        census.build_external_metadata_receipt(nonproduction)
    assert build_stop.value.status == "STOP_AUTHORIZATION_MISSING_OR_INVALID"
    with pytest.raises(census.MetadataCensusError, match="non-production") as finalize_stop:
        census.finalize_external_metadata_receipt(nonproduction)
    assert finalize_stop.value.status == "STOP_AUTHORIZATION_MISSING_OR_INVALID"
    assert not (valid_attempt / "NESTED_STRUCTURAL_PARSER_RECEIPT_V1.json").exists()
    assert not (valid_attempt / "EXTERNAL_METADATA_RECEIPT_V1.json").exists()

    # Enter the public external core with the exact capability class, but
    # mutate each retained authority layer after minting.  Full disk
    # reconstruction must refuse before the fake adapter receives call one.
    auth_case = _stage_authorization_repo(tmp_path / "auth-drift", monkeypatch)
    auth_capability, auth_fake, _, auth_attempt = _mint_fake_capability_for_refusal(
        auth_case
    )
    drifted_auth = _load(auth_case["authorization_path"])
    drifted_auth["credential_rotation_attested"] = False
    drifted_auth["authorization_sha256"] = census.self_hash(
        drifted_auth,
        "authorization_sha256",
    )
    _write_json(auth_case["authorization_path"], drifted_auth, mode=0o600)
    with pytest.raises(census.MetadataCensusError):
        census.run_authorized_external_metadata_census(auth_capability)
    assert auth_fake.calls == []
    assert not (auth_attempt / "CALL_JOURNAL_V1.jsonl").exists()
    assert not (auth_attempt / "EXTERNAL_METADATA_RESPONSES_V1.json").exists()

    marker_case = _stage_authorization_repo(tmp_path / "marker-drift", monkeypatch)
    marker_capability, marker_fake, marker_path, marker_attempt = (
        _mint_fake_capability_for_refusal(marker_case)
    )
    drifted_marker = _load(marker_path)
    drifted_marker["attempt_id"] = "99999999-9999-4999-8999-999999999999"
    _write_json(marker_path, drifted_marker, mode=0o600)
    with pytest.raises(census.MetadataCensusError):
        census.run_authorized_external_metadata_census(marker_capability)
    assert marker_fake.calls == []
    assert not (marker_attempt / "CALL_JOURNAL_V1.jsonl").exists()
    assert not (marker_attempt / "EXTERNAL_METADATA_RESPONSES_V1.json").exists()


def test_real_client_cannot_use_synthetic_public_entrypoint(tmp_path: Path) -> None:
    signature = inspect.signature(census.run_synthetic_metadata_census)
    assert "client" not in signature.parameters
    with pytest.raises(TypeError):
        census.run_synthetic_metadata_census(  # type: ignore[call-arg]
            _declaration(),
            client=object(),
            call_journal_path=Path("unused.jsonl"),
            response_path=Path("unused.json"),
        )
    fake = census.SyntheticMetadataClient(_declaration())
    test_capability = census._construct_test_only_authorized_external_client(fake)
    synthetic = census._run_test_only_authorized_external_client(
        test_capability,
        _declaration(),
        call_journal_path=tmp_path / "test-capability.jsonl",
        response_path=tmp_path / "test-capability.json",
        attempt_id="synthetic-test-capability",
    )
    assert synthetic["source"] == "synthetic"
    assert synthetic["execution_audit"]["authorization_sha256"] is None
    assert synthetic["method_attestation"]["methods_used"] == []
    with pytest.raises(census.MetadataCensusError):
        census.run_authorized_external_metadata_census(test_capability)  # type: ignore[arg-type]
    with pytest.raises(census.MetadataCensusError):
        census.finalize_external_metadata_receipt(test_capability)  # type: ignore[arg-type]
    assert len(fake.calls) == 2_440


def test_sdk_transport_source_and_requests_zero_retry_identity_are_bound() -> None:
    identity = readiness._sdk_source_identity(_load(V1_PATH), _load(V2_PATH), _load(V3_PATH))
    assert identity["identity_sha256"] == census._readiness_sdk_identity_sha256(
        {"bindings": {"sdk_source_identity": identity}}
    )
    assert identity["databento"]["transport_source"]["observed_methods"]["_get"]["requests_call_count"] == 1
    assert identity["databento"]["transport_source"]["observed_methods"]["_post"]["requests_call_count"] == 1
    assert identity["requests"]["default_retries"] == 0
    assert identity["requests"]["retry_zero_read_false_call_count"] == 1
    assert identity["client_imported"] is False


def test_job49_stop_aliases_are_normalized_at_every_job50_boundary(
    tmp_path: Path,
) -> None:
    from v5.ops import run_cmbp_metadata_census as cli

    contract_mapping = _load(V3_PATH)["total_job50_stop_normalization"][
        "mapping"
    ]
    inherited_names = (
        "STOP_SCOPE_OR_CODE_DRIFT",
        "STOP_MISSING_KEY_OR_ENTITLEMENT",
        "STOP_OVER_HARD_CAP",
        "STOP_SYMBOL_OR_EXPIRY_MISMATCH",
        "STOP_SCHEMA_UNAVAILABLE",
        "STOP_ZERO_RECORDS",
        "STOP_NONFINITE_COST",
        "STOP_MISSING_SESSION_RESPONSE",
    )
    expected_inherited = {
        source: contract_mapping[source]
        for source in inherited_names
    }
    assert census.JOB49_STOP_NORMALIZATION == expected_inherited
    assert census.JOB50_STOP_STATUSES == frozenset(contract_mapping.values())
    for source, target in contract_mapping.items():
        assert census.normalize_job50_status(source) == target
        assert census.normalize_job50_status(target) == target
    for status in sorted(set(contract_mapping.values())):
        assert census.normalize_job50_status(status) == status
    assert census.normalize_job50_status("UNKNOWN_STOP") == "STOP_CONTRACT_OR_DECLARATION_DRIFT"
    assert census.normalize_job50_status([]) == "STOP_CONTRACT_OR_DECLARATION_DRIFT"
    assert census.MetadataCensusError("x", status=catalogue.STOP_OVER_HARD_CAP).status == "STOP_OVER_QUOTED_CEILING"

    declaration = _declaration()
    session = _first_request_session(declaration)
    client = census.SyntheticMetadataClient(
        declaration,
        overrides={
            ("symbology.resolve", session): census.MetadataCensusError(
                "closed alias",
                status=catalogue.STOP_OVER_HARD_CAP,
            )
        },
    )
    journal = tmp_path / "alias.jsonl"
    response = tmp_path / "alias.json"
    with pytest.raises(census.MetadataCensusError) as stopped:
        census._run_metadata_census_with_injected_synthetic_client(
            client,
            declaration,
            call_journal_path=journal,
            response_path=response,
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-alias",
        )
    assert stopped.value.status == "STOP_OVER_QUOTED_CEILING"
    assert _journal_records(journal)[-1]["closed_error_class"] == "STOP_OVER_QUOTED_CEILING"
    classified = census.classify_metadata_attempt(
        declaration, call_journal_path=journal, response_path=response
    )
    assert classified["status"] == "STOP_OVER_QUOTED_CEILING"
    diagnostic = tmp_path / "diagnostic.json"
    cli._write_stop_diagnostic(
        diagnostic,
        attempt_id="synthetic-alias",
        status=catalogue.STOP_OVER_HARD_CAP,
    )
    assert census._load_canonical_json_output(diagnostic)["status"] == "STOP_OVER_QUOTED_CEILING"


def test_external_receipt_rejects_counterfeit_external_pass(
    tmp_path: Path,
) -> None:
    declaration = _declaration()
    fake = census.SyntheticMetadataClient(declaration)
    journal = tmp_path / "counterfeit.jsonl"
    response_path = tmp_path / "counterfeit.json"

    # The old bypass was a caller-selected source string on the private
    # executor.  That keyword no longer exists, so the attempt stops before a
    # journal, response, nested receipt, or top-level receipt can be emitted.
    for forbidden_selector in (
        {"source": "externally_supplied"},
        {"_external_execution_seal": object()},
        {"provenance": "externally_supplied"},
        {"authorization_sha256": "a" * 64},
        {"quoted_acquisition_ceiling_usd": "0"},
    ):
        with pytest.raises(TypeError):
            census._execute_metadata_census(  # type: ignore[arg-type]
                fake,
                declaration,
                call_journal_path=journal,
                response_path=response_path,
                contract_sha256=census.PROGRAM_CONTRACT_SHA256,
                **forbidden_selector,
            )
    assert fake.calls == []
    assert not journal.exists()
    assert not response_path.exists()

    test_capability = census._construct_test_only_authorized_external_client(fake)
    with pytest.raises(census.MetadataCensusError):
        census.build_external_metadata_receipt(test_capability)  # type: ignore[arg-type]
    with pytest.raises(census.MetadataCensusError):
        census.finalize_external_metadata_receipt(test_capability)  # type: ignore[arg-type]

    counterfeit = {field: None for field in census.EXTERNAL_RECEIPT_FIELDS}
    counterfeit.update(
        {
            "artifact_type": "JOB50_EXTERNAL_METADATA_RECEIPT_V1",
            "schema_version": "v5.job50-external-metadata-receipt.v1",
            "attempt_id": "33333333-3333-4333-8333-333333333333",
            "claims": {
                "owner_attestation_verification": "CRYPTOGRAPHICALLY_VERIFIED"
            },
        }
    )
    counterfeit["receipt_sha256"] = census.self_hash(
        counterfeit,
        "receipt_sha256",
    )
    with pytest.raises(census.MetadataCensusError):
        census.validate_external_metadata_receipt(
            counterfeit,
            repo_root=tmp_path,
        )
    assert not list(tmp_path.rglob("NESTED_STRUCTURAL_PARSER_RECEIPT_V1.json"))
    assert not list(tmp_path.rglob("EXTERNAL_METADATA_RECEIPT_V1.json"))


def test_external_receipt_rejects_raw_or_normalized_digest_tamper(
    complete_fake_run: tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path],
) -> None:
    declaration, response, _, journal, _ = complete_fake_run
    raw_tamper = copy.deepcopy(response)
    raw_tamper["dataset_range"]["raw_value"]["end"] = "2099-01-01"
    with pytest.raises(census.MetadataCensusError, match="raw"):
        census.validate_execution_response(raw_tamper, declaration, call_journal_path=journal)
    normalized_tamper = copy.deepcopy(response)
    normalized_tamper["sessions"][0]["record_count"] += 1
    with pytest.raises(census.MetadataCensusError, match="normalized|hash"):
        census.validate_execution_response(normalized_tamper, declaration, call_journal_path=journal)


def test_partial_attempt_classifier_rereads_on_disk_journal_and_never_passes(
    tmp_path: Path,
    complete_fake_run: tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path],
) -> None:
    declaration = _declaration()
    session = _first_request_session(declaration)
    client = census.SyntheticMetadataClient(
        declaration,
        overrides={("symbology.resolve", session): RuntimeError("crash")},
    )
    journal = tmp_path / "partial.jsonl"
    response = tmp_path / "partial.json"
    with pytest.raises(census.MetadataCensusError):
        census._run_metadata_census_with_injected_synthetic_client(
            client,
            declaration,
            call_journal_path=journal,
            response_path=response,
            contract_sha256=census.PROGRAM_CONTRACT_SHA256,
            attempt_id="synthetic-classifier",
        )
    classification = census.classify_metadata_attempt(
        declaration, call_journal_path=journal, response_path=response
    )
    assert classification["complete"] is False
    assert classification["status"] == "STOP_VENDOR_AUTH_OR_ENTITLEMENT"

    _, _, _, complete_journal, _ = complete_fake_run
    valid_prefix_lines = complete_journal.read_bytes().splitlines()[:2]
    call_start_only = tmp_path / "call-start-only.jsonl"
    call_start_only.write_bytes(b"\n".join(valid_prefix_lines) + b"\n")
    prefix_classification = census.classify_metadata_attempt(
        declaration,
        call_journal_path=call_start_only,
        response_path=tmp_path / "never-emitted.json",
    )
    assert prefix_classification["status"] == "STOP_PARTIAL_OR_CRASHED_ATTEMPT"
    assert prefix_classification["complete"] is False
    assert prefix_classification["success_response_valid"] is False
    lines = journal.read_bytes().splitlines()
    tampered = json.loads(lines[1])
    tampered["method"] = "timeseries.get_range"
    lines[1] = _canonical_bytes(tampered)
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_bytes(b"\n".join(lines) + b"\n")
    refused = census.classify_metadata_attempt(
        declaration, call_journal_path=corrupt, response_path=response
    )
    assert refused["status"] == "STOP_CALL_JOURNAL_INVALID"
    assert refused["success_response_valid"] is False


def test_nested_parser_receipt_is_attempt_local_durable_and_bound(
    tmp_path: Path,
    complete_fake_run: tuple[dict[str, Any], dict[str, Any], census.SyntheticMetadataClient, Path, Path],
) -> None:
    declaration, response, _, _, _ = complete_fake_run
    rehearsal = catalogue.build_preflight_receipt(
        declaration,
        response,
        hard_cap_usd=Decimal("0"),
    )
    assert rehearsal["claims"]["actual_vendor_availability"] is False
    assert rehearsal["job49_integration_disposition"] == "VALIDATOR_REHEARSAL_ONLY"

    module_source = Path(census.__file__).read_text(encoding="utf-8")
    context_source = module_source[
        module_source.index("def _external_receipt_context_engine("):
        module_source.index("def _build_nested_structural_receipt_engine(")
    ]
    nested_source = module_source[
        module_source.index("def _build_nested_structural_receipt_engine("):
        module_source.index("def _external_wrapper_from_context_engine(")
    ]
    wrapper_source = module_source[
        module_source.index("def _external_wrapper_from_context_engine("):
        module_source.index("def _bind_external_receipt_entrypoints(")
    ]
    assert (
        'paths["nested_receipt"] = verified_attempt / '
        '"NESTED_STRUCTURAL_PARSER_RECEIPT_V1.json"'
    ) in context_source
    assert "_write_json_exclusive(path, expected)" in nested_source
    assert '"path": _repository_relative(paths["nested_receipt"], root)' in wrapper_source
    assert '"receipt_sha256": nested_receipt["receipt_sha256"]' in wrapper_source
    assert '"file_sha256": nested_file_sha256' in wrapper_source

    assert not hasattr(census, "_external_receipt_context_engine")
    assert not hasattr(census, "_build_nested_structural_receipt_engine")
    assert not hasattr(census, "_external_wrapper_from_context_engine")

    finalizer_source = inspect.getsource(census.finalize_external_metadata_receipt)
    build_entrypoint_source = inspect.getsource(
        census.build_external_metadata_receipt
    )
    assert build_entrypoint_source.index(
        "capability._require_production_usable()"
    ) < build_entrypoint_source.index(
        "nested, nested_raw_sha = nested_builder(context, write_new=True)"
    )
    assert finalizer_source.index("wrapper = build(capability)") < finalizer_source.index(
        "reread = _write_json_exclusive(path, wrapper)"
    )
    assert finalizer_source.index(
        "reread = _write_json_exclusive(path, wrapper)"
    ) < finalizer_source.index("validate(")
    assert "receipt_path=path" in finalizer_source
    assert not list(tmp_path.rglob("NESTED_STRUCTURAL_PARSER_RECEIPT_V1.json"))
    assert not list(tmp_path.rglob("EXTERNAL_METADATA_RECEIPT_V1.json"))


def test_external_integrity_and_authorized_client_capability_are_exact(
    tmp_path: Path,
) -> None:
    assert census.EXTERNAL_PASS_INTEGRITY == _load(V3_PATH)[
        "external_pass_integrity_law"
    ]["exact_integrity_object"]
    assert census.EXTERNAL_PASS_INTEGRITY["sdk_metadata_method_invocations"] == 2_440
    assert census.EXTERNAL_PASS_INTEGRITY["low_level_network_request_count"] == "UNOBSERVED"
    assert (
        census.EXTERNAL_RECEIPT_AUTHORIZATION_EFFECT
        == _load(V3_PATH)["external_pass_integrity_law"]["exact_authorization_effect"]
    )
    assert not hasattr(census.AuthorizedExternalClient, "_sealed_contents")
    assert not hasattr(census, "_AUTHORIZED_CORE_ACCESS_SEAL")
    assert not hasattr(census, "_AUTHORIZED_CAPABILITY_SEAL")
    assert not hasattr(census, "_bind_authorized_capability_type")
    assert tuple(census.AuthorizedExternalClient.__slots__) == (
        "__adapter",
        "__bindings",
        "__construction_seal",
        "__production_state",
    )
    assert tuple(
        inspect.signature(
            census.AuthorizedExternalClient._dispatch_surface_for_core
        ).parameters
    ) == ("self",)
    with pytest.raises(census.MetadataCensusError):
        census.AuthorizedExternalClient(
            object(),
            {},
            _construction_seal=object(),
            _production_state=object(),
        )
    fake = census.SyntheticMetadataClient(_declaration())
    bindings = {
        field: "test-only-binding"
        for field in census._AUTHORIZED_CAPABILITY_FIELDS
    }
    databento_modules_before = {
        name for name in sys.modules if name == "databento" or name.startswith("databento.")
    }
    with pytest.raises(census.MetadataCensusError) as fake_production:
        census._mint_authorized_external_client(fake, bindings=bindings)
    assert fake_production.value.status == "STOP_SDK_VERSION_OR_SIGNATURE_DRIFT"
    assert {
        name for name in sys.modules if name == "databento" or name.startswith("databento.")
    } == databento_modules_before
    assert fake.calls == []
    adapter = census._MetadataOnlyClientAdapter(fake)
    assert set(adapter.metadata.__slots__) == {
        "get_dataset_range", "get_cost", "get_record_count"
    }
    assert set(adapter.symbology.__slots__) == {"resolve"}
    assert not hasattr(adapter, "timeseries")
    assert not hasattr(adapter, "batch")
    test_capability = census._construct_test_only_authorized_external_client(fake)
    assert type(test_capability) is not census.AuthorizedExternalClient
    with pytest.raises(census.MetadataCensusError):
        census.run_authorized_external_metadata_census(test_capability)  # type: ignore[arg-type]


def test_owner_attestations_are_explicit_manual_governance_inputs(
    tmp_path: Path,
) -> None:
    v3 = _load(V3_PATH)
    boundary = v3["owner_attestation_epistemic_boundary"]
    assert boundary["manual_governance_inputs"] == [
        "credential_rotation_attested and credential_attestation_recorded_at_utc",
        "current_conversation_authorization_sha256",
        "the claim that the authorization was authored/approved by the owner",
    ]
    assert "cannot cryptographically prove" in boundary["not_machine_verifiable"]
    disclosure = "OWNER_ATTESTED_NOT_CRYPTOGRAPHICALLY_VERIFIED"
    assert boundary["required_receipt_disclosure"] == disclosure
    module_source = Path(census.__file__).read_text(encoding="utf-8")
    build_source = module_source[
        module_source.index("def _external_wrapper_from_context_engine("):
        module_source.index("def _bind_external_receipt_entrypoints(")
    ]
    validate_source = module_source[
        module_source.index("def _bind_external_receipt_entrypoints("):
        module_source.index("class SyntheticMetadataClient:")
    ]
    assert disclosure in build_source
    assert disclosure in validate_source
    validate_tree = ast.parse(validate_source)
    validate_object = next(
        node
        for node in ast.walk(validate_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "validate_object"
    )
    literals = {
        node.value
        for node in ast.walk(validate_object)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "owner_attestation_verification" in literals
    assert disclosure in literals
    assert "owner-attestation epistemic disclosure drifted" in literals

    # Local code can verify the exact disclosure law, but cannot create an
    # external receipt without an authorized external response on disk.
    fake = census.SyntheticMetadataClient(_declaration())
    test_capability = census._construct_test_only_authorized_external_client(fake)
    with pytest.raises(census.MetadataCensusError):
        census.finalize_external_metadata_receipt(test_capability)  # type: ignore[arg-type]
    assert fake.calls == []
    assert not list(tmp_path.rglob("EXTERNAL_METADATA_RECEIPT_V1.json"))


def test_urllib3_default_retry_identity_is_bound() -> None:
    identity = readiness._sdk_source_identity(_load(V1_PATH), _load(V2_PATH), _load(V3_PATH))
    urllib = identity["urllib3"]
    assert urllib["version"] == "2.6.3"
    assert set(urllib["source_files"]) == {
        "urllib3/util/retry.py",
        "urllib3/connectionpool.py",
        "urllib3/poolmanager.py",
    }
    assert all(len(value) == 64 for value in urllib["source_files"].values())
    v3 = _load(V3_PATH)
    assert urllib["distribution_metadata"]["file_sha256"] == (
        v3["urllib3_transport_identity"]["distribution_metadata_file_sha256"]
    )
