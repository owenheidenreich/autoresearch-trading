from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
from typing import Any

import pytest

from v4.research import pathd_entry_exit as foundation
from v4.research import pathd_evidence_gate as gate


SESSION = "2026-01-02"
HEX = "a" * 64


@dataclass(frozen=True)
class _Input:
    session: str


@dataclass(frozen=True)
class _Example:
    model_input: _Input
    canonical_sha256: str


@dataclass(frozen=True)
class _Dataset:
    source_receipts: tuple[dict[str, Any], ...]
    examples: tuple[_Example, ...]
    dataset_sha256: str


@pytest.fixture(autouse=True)
def _clear_process_capabilities() -> Any:
    for key in list(gate._ACTIVE_CAPABILITIES):
        gate._release_capability(key)
    yield
    for key in list(gate._ACTIVE_CAPABILITIES):
        gate._release_capability(key)


@pytest.fixture
def isolated_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "repo"
    root.mkdir()
    folds = root / "audit" / "entry_outer_folds"
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(gate, "ENTRY_FOLD_ARTIFACT_ROOT", folds)
    monkeypatch.setattr(gate, "PREREG_PATH", root / "preregistration.json")
    monkeypatch.setattr(gate, "SESSION_PATH", root / "session_assignments.json")
    monkeypatch.setattr(gate, "PLAN_PATH", root / "plan.md")
    monkeypatch.setattr(
        gate,
        "CORPUS_INTEGRITY_RECEIPT_PATH",
        root / "corpus_integrity_receipt.json",
    )
    monkeypatch.setattr(
        gate,
        "LINEAGE_IMPLEMENTATION_RECEIPT_PATH",
        root / "lineage_receipt.json",
    )
    monkeypatch.setattr(
        gate,
        "ENTRY_MACHINERY_RECEIPT_PATH",
        root / "machinery_receipt.json",
    )
    monkeypatch.setattr(gate, "_validate_claim_foundation", lambda *_a, **_k: None)
    monkeypatch.setattr(
        gate,
        "_result_context",
        lambda: {
            "plan_sha256": "b" * 64,
            "preregistration_sha256": "c" * 64,
            "fill_law_hash": "d" * 64,
        },
    )

    def create_preopen(
        role: str = "nested_validation",
        outer_fold: int = 1,
        inner_fold: int | None = 1,
    ) -> gate._ScopePaths:
        paths = gate._scope_paths(
            role=role, outer_fold=outer_fold, inner_fold=inner_fold
        )
        paths.preopen.parent.mkdir(parents=True, exist_ok=True)
        paths.preopen.write_bytes(b"preopen\n")
        scope = (
            f"NESTED_OUTER_{outer_fold}_INNER_{inner_fold}"
            if role == "nested_validation"
            else f"OUTER_{outer_fold}"
        )
        node_ids = foundation.entry_required_calibration_node_ids(scope)
        gate_semantic = {
            "schema_version": "pathd.calibration_scope_gate_receipt.v1",
            "scope": scope,
            "status": "VALID",
            "required_node_count": len(node_ids),
            "required_node_ids_sha256": gate._stable_hash(list(node_ids)),
            "ordered_node_sha256s": [
                f"{index + 1:064x}" for index in range(len(node_ids))
            ],
            "node_vector_sha256": "a" * 64,
            "failure_node_ids": [],
            "evidence_access_count": 0,
            "holdout_open_count": 0,
            "forbidden_rescue_applied": False,
        }
        gate._write_exclusive_json(
            gate._calibration_scope_gate_path(paths),
            {**gate_semantic, "receipt_sha256": gate._stable_hash(gate_semantic)},
        )
        return paths

    def claim(
        role: str = "nested_validation",
        outer_fold: int = 1,
        inner_fold: int | None = 1,
        sessions: tuple[str, ...] = (SESSION,),
    ) -> dict[str, Any]:
        paths = gate._scope_paths(
            role=role, outer_fold=outer_fold, inner_fold=inner_fold
        )
        gates = gate._gate_paths(paths)
        gate_hashes = tuple(
            hashlib.sha256(path.read_bytes()).hexdigest() for path in gates
        )
        return {
            "role": role,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "sessions": sessions,
            "sessions_sha256_newline": foundation.canonical_session_hash(sessions),
            "preregistration_sha256": "c" * 64,
            "session_assignments_sha256": "e" * 64,
            "source_hash_policy_sha256": "f" * 64,
            "corpus_integrity_receipt_sha256": "1" * 64,
            "lineage_receipt_sha256": "2" * 64,
            "machinery_receipt_sha256": "3" * 64,
            "fit_environment_sha256": "4" * 64,
            "open_gate_receipts_sha256": gate_hashes,
        }

    return {
        "root": root,
        "folds": folds,
        "create_preopen": create_preopen,
        "claim": claim,
        "monkeypatch": monkeypatch,
    }


def _install_claim(
    fixture: dict[str, Any], *, role: str = "nested_validation",
    outer_fold: int = 1, inner_fold: int | None = 1,
) -> gate._ScopePaths:
    paths = fixture["create_preopen"](role, outer_fold, inner_fold)
    frozen_claim = fixture["claim"](role, outer_fold, inner_fold)
    fixture["monkeypatch"].setattr(
        gate,
        "_prepare_entry_evidence_authorization_claim",
        lambda **_kwargs: dict(frozen_claim),
    )
    return paths


def _dataset() -> _Dataset:
    source = {
        "session": SESSION,
        "source_files": [],
        "source_files_sha256": "5" * 64,
        "example_count": 1,
        "ordered_example_list_sha256": "6" * 64,
        "session_content_sha256": "7" * 64,
        "receipt_sha256": "8" * 64,
    }
    return _Dataset(
        source_receipts=(source,),
        examples=(_Example(_Input(SESSION), "9" * 64),),
        dataset_sha256="0" * 64,
    )


def _nested_result(
    authorization: foundation.FrozenEvidenceAuthorization,
    dataset: _Dataset,
) -> dict[str, Any]:
    semantic = {
        "schema_version": "pathd.entry_nested_family_evaluation.v1",
        "holdout_caveat": foundation.HOLDOUT_CAVEAT,
        "outer_fold": authorization.outer_fold,
        "inner_fold": authorization.inner_fold,
        "authorization_sha256": gate._stable_hash(authorization.to_dict()),
        "dataset_sha256": dataset.dataset_sha256,
        "source_receipts_root_sha256": gate._stable_hash(
            list(dataset.source_receipts)
        ),
        "access_receipt_sha256": authorization.access_receipt_sha256,
        "hgb": {
            "policy_id": "HGB",
            "holdout_caveat": foundation.HOLDOUT_CAVEAT,
            "result_sha256": "a" * 64,
        },
        "neural": {
            "policy_id": "NEURAL",
            "holdout_caveat": foundation.HOLDOUT_CAVEAT,
            "result_sha256": "b" * 64,
        },
    }
    return {**semantic, "result_sha256": gate._stable_hash(semantic)}


def _shortened_result(
    authorization: foundation.FrozenEvidenceAuthorization,
    dataset: _Dataset,
) -> dict[str, Any]:
    fields = foundation.entry_future_api_contract()["dataclass_fields"][
        "EntryPolicyEvaluationV1"
    ]
    semantic = {field: None for field in fields if field != "result_sha256"}
    semantic.update(
        {
            "schema_version": "pathd.entry_policy_evaluation.v1",
            "holdout_caveat": foundation.HOLDOUT_CAVEAT,
            "evidence_role": "outer_test_shortened_diagnostic",
            "authorization_sha256": gate._stable_hash(authorization.to_dict()),
            "dataset_sha256": dataset.dataset_sha256,
        }
    )
    return {**semantic, "result_sha256": gate._stable_hash(semantic)}


def _abandon_live_capability(
    authorization: foundation.FrozenEvidenceAuthorization,
) -> None:
    key = id(authorization)
    capability = gate._ACTIVE_CAPABILITIES.pop(key)
    gate._release_lock(capability.lock_fd)


def test_access_count_zero_to_one_and_decode_precedence(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _install_claim(isolated_gate)
    assert not paths.access.exists()

    authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=1, inner_fold=1
    )

    assert paths.access.exists()
    receipt = gate._read_access_receipt(paths)
    assert receipt["validation_access_count"] == 1
    assert receipt["source_decode_started"] is False
    assert authorization.access_receipt_sha256 == gate._sha256_nofollow(paths.access)
    assert gate.inspect_entry_evidence_state(
        role="nested_validation", outer_fold=1, inner_fold=1
    )["state"] == gate.OPEN_ACTIVE

    observed: list[bool] = []

    def decode_probe() -> None:
        observed.append(paths.access.exists())

    artifact_only = gate.read_frozen_entry_evidence_authorization(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    assert artifact_only == authorization and artifact_only is not authorization
    with pytest.raises(gate.EntryEvidenceGateError, match="process-local"):
        gate.validate_entry_evidence_access(artifact_only)
    gate.validate_entry_evidence_access(authorization)
    gate.validate_entry_evidence_access(authorization)
    gate.claim_entry_evidence_decode_once(authorization)
    gate.validate_entry_evidence_access(authorization)
    gate.validate_entry_evidence_access(authorization)
    decode_probe()
    assert observed == [True]
    with pytest.raises(gate.EntryEvidenceGateError, match="already consumed"):
        gate.claim_entry_evidence_decode_once(authorization)


@pytest.mark.parametrize("failure", ("ABSENT", "WRONG_SCOPE", "FAILED"))
def test_direct_gate_call_requires_exact_valid_br_before_any_open(
    isolated_gate: dict[str, Any], failure: str
) -> None:
    paths = _install_claim(isolated_gate)
    scope_path = gate._calibration_scope_gate_path(paths)
    receipt = gate._read_canonical_json(scope_path)
    scope_path.unlink()
    if failure != "ABSENT":
        semantic = dict(receipt)
        semantic.pop("receipt_sha256")
        if failure == "WRONG_SCOPE":
            semantic["scope"] = "NESTED_OUTER_1_INNER_2"
        else:
            semantic["status"] = "FAILED_CLOSED"
            semantic["failure_node_ids"] = [
                "ENTRY::NESTED_OUTER_1_INNER_1::HGB::ACTION_COMPOSITE::WAIT"
            ]
        gate._write_exclusive_json(
            scope_path,
            {**semantic, "receipt_sha256": gate._stable_hash(semantic)},
        )

    with pytest.raises(gate.EntryEvidenceGateError):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=1, inner_fold=1
        )

    assert not paths.access.exists()
    assert not paths.lock.exists()
    assert gate._ACTIVE_CAPABILITIES == {}


def test_concurrent_and_repeated_open_are_impossible(
    isolated_gate: dict[str, Any]
) -> None:
    _install_claim(isolated_gate)
    authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    errors: list[BaseException] = []

    def contender() -> None:
        try:
            gate.begin_entry_evidence_once(
                role="nested_validation", outer_fold=1, inner_fold=1
            )
        except BaseException as exc:  # capture from the contender thread
            errors.append(exc)

    thread = threading.Thread(target=contender)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], gate.EntryEvidenceGateError)

    _abandon_live_capability(authorization)
    with pytest.raises(gate.EntryEvidenceGateError, match="not pristine"):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=1, inner_fold=1
        )


def test_invalid_nested_block_seals_skip_without_decode(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = gate._scope_paths(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    payload = {
        "binding_plan": {"sha256": "b" * 64},
        "fill_law": {"fill_law_hash": "d" * 64},
    }
    role = {
        "calibration_valid": False,
        "model_fit_sha256_newline": "1" * 64,
        "calibration_sha256_newline": "2" * 64,
        "validation_sha256_newline": "3" * 64,
        "calibration": ["2025-01-02"],
        "calibration_minimum_sessions": 10,
    }
    monkeypatch.setattr(
        gate,
        "_prepare_invalid_nested_skip_claim",
        lambda **_kwargs: {
            "payload": payload,
            "role": role,
            "preregistration_sha256": "c" * 64,
            "session_assignments_sha256": "e" * 64,
            "source_hash_policy_sha256": "f" * 64,
        },
    )
    decode_calls: list[bool] = []
    monkeypatch.setattr(
        gate,
        "_validate_exact_dataset",
        lambda *_a, **_k: decode_calls.append(True),
    )

    receipt = gate.seal_invalid_nested_block_skip(
        outer_fold=1, inner_fold=1
    )

    assert receipt["validation_access_count"] == 0
    assert receipt["dataset_opened"] is False
    assert receipt["result_generated"] is False
    assert paths.skip_receipt is not None and paths.skip_receipt.exists()
    assert not paths.access.exists()
    assert decode_calls == []
    assert gate.inspect_entry_evidence_state(
        role="nested_validation", outer_fold=1, inner_fold=1
    )["state"] == gate.INVALID_SKIPPED
    with pytest.raises(gate.EntryEvidenceGateError, match="not pristine"):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=1, inner_fold=1
        )


def test_result_is_bound_to_exact_dataset_and_revokes_capability(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _install_claim(isolated_gate)
    authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    gate.claim_entry_evidence_decode_once(authorization)
    dataset = _dataset()
    result = _nested_result(authorization, dataset)
    gate._validate_result_document_binding(
        result,
        authorization=authorization,
        dataset_sha256=dataset.dataset_sha256,
    )
    for caveat_value in (None, "FORGED_HOLDOUT_CAVEAT"):
        forged = dict(result)
        if caveat_value is None:
            forged.pop("holdout_caveat")
        else:
            forged["holdout_caveat"] = caveat_value
        semantic = dict(forged)
        semantic.pop("result_sha256")
        forged["result_sha256"] = gate._stable_hash(semantic)
        with pytest.raises(
            gate.EntryEvidenceGateError, match="nested result binding drift"
        ):
            gate._validate_result_document_binding(
                forged,
                authorization=authorization,
                dataset_sha256=dataset.dataset_sha256,
            )
    caveat_free = dict(result)
    caveat_free.pop("holdout_caveat")
    with pytest.raises(
        gate.EntryEvidenceGateError, match="result payload holdout caveat drift"
    ):
        gate._build_result_document(
            caveat_free, authorization=authorization, dataset=dataset
        )
    monkeypatch.setattr(gate, "_validate_exact_dataset", lambda *_a, **_k: None)
    monkeypatch.setattr(gate, "_validate_exact_evaluation", lambda *_a, **_k: None)
    monkeypatch.setattr(
        gate, "_build_result_document", lambda *_a, **_k: dict(result)
    )
    monkeypatch.setattr(
        gate, "_validate_terminal_result", lambda *_a, **_k: {"status": "ok"}
    )

    receipt = gate.seal_entry_evidence_result(
        authorization, dataset=dataset, evaluation=object()
    )

    assert receipt["validation_access_count"] == 1
    assert receipt["dataset_sha256"] == dataset.dataset_sha256
    assert receipt["authorization_sha256"] == gate._stable_hash(
        authorization.to_dict()
    )
    assert paths.dataset_receipt.exists()
    assert paths.result.exists()
    assert paths.result_receipt.exists()
    assert id(authorization) not in gate._ACTIVE_CAPABILITIES
    assert gate.inspect_entry_evidence_state(
        role="nested_validation", outer_fold=1, inner_fold=1
    )["state"] == gate.SEALED
    with pytest.raises(gate.EntryEvidenceGateError):
        gate.seal_entry_evidence_result(
            authorization, dataset=dataset, evaluation=object()
        )
    with pytest.raises(gate.EntryEvidenceGateError):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=1, inner_fold=1
        )


def test_invalid_typed_result_burns_scope(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _install_claim(isolated_gate)
    authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    gate.claim_entry_evidence_decode_once(authorization)
    monkeypatch.setattr(gate, "_validate_exact_dataset", lambda *_a, **_k: None)

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("forged evaluation")

    monkeypatch.setattr(gate, "_validate_exact_evaluation", reject)
    with pytest.raises(ValueError, match="forged"):
        gate.seal_entry_evidence_result(
            authorization, dataset=_dataset(), evaluation={"PASS": True}
        )
    assert paths.burned_receipt.exists()
    assert not paths.dataset_receipt.exists()
    assert gate.inspect_entry_evidence_state(
        role="nested_validation", outer_fold=1, inner_fold=1
    )["state"] == gate.BURNED


def test_later_block_outer_and_diagnostic_order_are_fail_closed(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = isolated_gate["create_preopen"]("nested_validation", 1, 1)
    (paths.preopen.parent / "nested_inner_2_unexpected.json").write_text("x")
    monkeypatch.setattr(
        gate,
        "_prepare_entry_evidence_authorization_claim",
        lambda **_kwargs: pytest.fail("prerequisites must fail before authorization"),
    )
    with pytest.raises(gate.EntryEvidenceGateError, match="later nested"):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=1, inner_fold=1
        )

    isolated_gate["create_preopen"]("nested_validation", 2, 2)
    with pytest.raises(gate.EntryEvidenceGateError, match="terminate in order"):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=2, inner_fold=2
        )

    isolated_gate["create_preopen"](
        "outer_test_shortened_diagnostic", 3, None
    )
    with pytest.raises(gate.EntryEvidenceGateError, match="precedes primary"):
        gate.begin_entry_evidence_once(
            role="outer_test_shortened_diagnostic", outer_fold=3
        )

    isolated_gate["create_preopen"]("outer_test_primary", 4, None)
    with pytest.raises(gate.EntryEvidenceGateError, match="four nested terminals"):
        gate.begin_entry_evidence_once(
            role="outer_test_primary", outer_fold=4
        )

    with pytest.raises(gate.EntryEvidenceGateError, match="unsupported"):
        gate.begin_entry_evidence_once(role="pooled_outer_primary", outer_fold=1)

    live_paths = _install_claim(
        isolated_gate, outer_fold=5, inner_fold=1
    )
    live_authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=5, inner_fold=1
    )
    (live_paths.preopen.parent / "nested_inner_2_leak.json").write_text("leak")
    with pytest.raises(gate.EntryEvidenceGateError, match="appeared after open"):
        gate.validate_entry_evidence_access(live_authorization)


def test_future_outer_and_premature_shortened_artifacts_are_rejected(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_gate["create_preopen"]("nested_validation", 1, 1)
    later = isolated_gate["folds"] / "fold_2"
    later.mkdir(parents=True)
    (later / "future_outcome.json").write_text("future")
    monkeypatch.setattr(
        gate,
        "_prepare_entry_evidence_authorization_claim",
        lambda **_kwargs: pytest.fail("future artifacts must block before claim"),
    )
    with pytest.raises(gate.EntryEvidenceGateError, match="later outer-fold"):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=1, inner_fold=1
        )

    for child in later.iterdir():
        child.unlink()
    later.rmdir()
    primary = isolated_gate["create_preopen"]("outer_test_primary", 3, None)
    for inner in range(1, 5):
        (primary.preopen.parent / f"nested_inner_{inner}_skip_receipt.json").write_text(
            "terminal"
        )
    (primary.preopen.parent / "shortened_preopen_receipt.json").write_text(
        "premature"
    )
    with pytest.raises(gate.EntryEvidenceGateError, match="before primary seal"):
        gate.begin_entry_evidence_once(role="outer_test_primary", outer_fold=3)


def test_shortened_diagnostic_opens_only_after_primary(
    isolated_gate: dict[str, Any]
) -> None:
    paths = isolated_gate["create_preopen"](
        "outer_test_shortened_diagnostic", 1, None
    )
    primary = paths.preopen.parent / "outer_result_receipt.json"
    primary.write_bytes(b"primary\n")
    frozen_claim = isolated_gate["claim"](
        "outer_test_shortened_diagnostic", 1, None
    )
    isolated_gate["monkeypatch"].setattr(
        gate,
        "_prepare_entry_evidence_authorization_claim",
        lambda **_kwargs: dict(frozen_claim),
    )
    authorization = gate.begin_entry_evidence_once(
        role="outer_test_shortened_diagnostic", outer_fold=1
    )
    access = gate._read_access_receipt(paths)
    assert access["diagnostic_access_count"] == 1
    assert authorization.role == "outer_test_shortened_diagnostic"
    assert authorization.open_gate_receipts_sha256 == tuple(
        frozen_claim["open_gate_receipts_sha256"]
    )
    gate.claim_entry_evidence_decode_once(authorization)
    dataset = _dataset()
    shortened_result = _shortened_result(authorization, dataset)
    gate._validate_result_document_binding(
        shortened_result,
        authorization=authorization,
        dataset_sha256=dataset.dataset_sha256,
    )
    for caveat_value in (None, "FORGED_HOLDOUT_CAVEAT"):
        forged = dict(shortened_result)
        if caveat_value is None:
            forged.pop("holdout_caveat")
        else:
            forged["holdout_caveat"] = caveat_value
        semantic = dict(forged)
        semantic.pop("result_sha256")
        forged["result_sha256"] = gate._stable_hash(semantic)
        with pytest.raises(
            gate.EntryEvidenceGateError, match="shortened result binding drift"
        ):
            gate._validate_result_document_binding(
                forged,
                authorization=authorization,
                dataset_sha256=dataset.dataset_sha256,
            )
    caveat_free = dict(shortened_result)
    caveat_free.pop("holdout_caveat")
    with pytest.raises(
        gate.EntryEvidenceGateError, match="result payload holdout caveat drift"
    ):
        gate._build_result_document(
            caveat_free, authorization=authorization, dataset=dataset
        )
    membership_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    monkeypatch = isolated_gate["monkeypatch"]
    monkeypatch.setattr(gate, "_validate_exact_dataset", lambda *_a, **_k: None)
    monkeypatch.setattr(gate, "_validate_exact_evaluation", lambda *_a, **_k: None)
    monkeypatch.setattr(
        gate,
        "_build_result_document",
        lambda *_a, **_k: dict(shortened_result),
    )
    monkeypatch.setattr(
        foundation,
        "_validate_policy_evaluation_dataset_membership",
        lambda evaluation, dataset_receipt: membership_calls.append(
            (evaluation, dataset_receipt)
        ),
        raising=False,
    )
    receipt = gate.seal_entry_evidence_result(
        authorization, dataset=dataset, evaluation=object()
    )
    assert receipt["status"] == "FROZEN_SHORTENED_DIAGNOSTIC_RESULT"
    assert len(membership_calls) == 1
    assert membership_calls[0][0] == shortened_result
    assert membership_calls[0][1]["dataset_sha256"] == dataset.dataset_sha256


def test_crash_without_complete_result_is_permanently_burned(
    isolated_gate: dict[str, Any]
) -> None:
    paths = _install_claim(isolated_gate)
    authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    gate.claim_entry_evidence_decode_once(authorization)
    _abandon_live_capability(authorization)

    state = gate.recover_entry_evidence_after_crash(
        role="nested_validation", outer_fold=1, inner_fold=1
    )

    assert state["state"] == gate.BURNED
    assert state["access_count"] == 1
    assert paths.burned_receipt.exists()
    with pytest.raises(gate.EntryEvidenceGateError):
        gate.begin_entry_evidence_once(
            role="nested_validation", outer_fold=1, inner_fold=1
        )


def test_crash_with_durable_unsealed_result_is_permanently_burned(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Preserve the registered JUnit identity: the hardened rule now proves that
    # even otherwise-valid durable bytes cannot be sealed after lease loss.
    paths = _install_claim(isolated_gate)
    authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    gate.claim_entry_evidence_decode_once(authorization)
    dataset = _dataset()
    dataset_receipt = gate._dataset_receipt(
        dataset, authorization=authorization
    )
    result = _nested_result(authorization, dataset)
    gate._write_exclusive_json(paths.dataset_receipt, dataset_receipt)
    gate._write_exclusive_json(paths.result, result)
    _abandon_live_capability(authorization)
    decode_calls: list[bool] = []
    monkeypatch.setattr(
        gate,
        "_validate_exact_dataset",
        lambda *_a, **_k: decode_calls.append(True),
    )
    monkeypatch.setattr(
        gate,
        "_rehash_dataset_source_files",
        lambda *_a, **_k: decode_calls.append(True),
    )
    monkeypatch.setattr(
        gate,
        "_result_receipt",
        lambda *_a, **_k: pytest.fail("crash recovery must never seal"),
    )

    state = gate.recover_entry_evidence_after_crash(
        role="nested_validation", outer_fold=1, inner_fold=1
    )

    assert state["state"] == gate.BURNED
    assert not paths.result_receipt.exists()
    assert paths.burned_receipt.exists()
    burned = gate._read_canonical_json(paths.burned_receipt)
    assert burned["reason"] == "LEASE_LOST_WITH_DURABLE_UNSEALED_RESULT"
    assert paths.dataset_receipt.exists() and paths.result.exists()
    assert decode_calls == []
    with pytest.raises(gate.EntryEvidenceGateError):
        gate.validate_entry_evidence_access(authorization)


def test_access_receipt_is_canonical_nofollow_and_tamper_evident(
    isolated_gate: dict[str, Any]
) -> None:
    paths = _install_claim(isolated_gate)
    authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=1, inner_fold=1
    )
    raw = paths.access.read_bytes()
    assert raw.endswith(b"\n")
    assert b" " not in raw
    _abandon_live_capability(authorization)

    paths.access.write_bytes(raw.replace(b'"holdout_open_count":0', b'"holdout_open_count":1'))
    with pytest.raises(gate.EntryEvidenceGateError):
        gate.read_frozen_entry_evidence_authorization(
            role="nested_validation", outer_fold=1, inner_fold=1
        )

    other_paths = isolated_gate["create_preopen"]("nested_validation", 2, 1)
    target = isolated_gate["root"] / "outside.json"
    target.write_text("{}\n")
    other_paths.access.symlink_to(target)
    with pytest.raises(gate.EntryEvidenceGateError):
        gate.read_frozen_entry_evidence_authorization(
            role="nested_validation", outer_fold=2, inner_fold=1
        )


def test_foundation_drift_after_open_burns_before_decode(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _install_claim(isolated_gate)
    authorization = gate.begin_entry_evidence_once(
        role="nested_validation", outer_fold=1, inner_fold=1
    )

    def drift(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("machinery source drift")

    monkeypatch.setattr(gate, "_validate_claim_foundation", drift)
    with pytest.raises(RuntimeError, match="machinery source drift"):
        gate.validate_entry_evidence_access(authorization)
    assert paths.burned_receipt.exists()
    assert id(authorization) not in gate._ACTIVE_CAPABILITIES
    assert not paths.dataset_receipt.exists()


def test_sealed_source_bytes_are_rehashed_without_decode(
    isolated_gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = isolated_gate["root"] / "corpus"
    corpus.mkdir()
    source = corpus / "session" / "quotes.dbn"
    source.parent.mkdir()
    source.write_bytes(b"frozen bytes")
    monkeypatch.setattr(gate, "CORPUS_ROOT", corpus)
    receipt = {
        "source_receipts": [
            {
                "session": SESSION,
                "source_files": [
                    {
                        "relative_path": "session/quotes.dbn",
                        "bytes": len(b"frozen bytes"),
                        "sha256": hashlib.sha256(b"frozen bytes").hexdigest(),
                    }
                ],
            }
        ]
    }
    gate._rehash_dataset_source_files(receipt)
    source.write_bytes(b"changed bytes")
    with pytest.raises(gate.EntryEvidenceGateError, match="source bytes drifted"):
        gate._rehash_dataset_source_files(receipt)
