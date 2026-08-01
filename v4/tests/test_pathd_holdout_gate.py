from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from v4.research import pathd_holdout_gate as gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _artifact_row(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _drop_originating_process_lease(
    authorization: gate.ActiveProtectedHoldoutAuthorizationV1,
) -> None:
    gate._deactivate_authorization(authorization)


@pytest.fixture
def frozen_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    for authorization in list(gate._ACTIVE_AUTH_OBJECTS.values()):
        gate._deactivate_authorization(authorization)
    gate._ACTIVE_AUTH_OBJECTS.clear()

    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    protected = audit / "protected_holdout"
    monkeypatch.setattr(gate, "AUDIT_ROOT", audit)
    monkeypatch.setattr(
        gate, "PRE_HOLDOUT_PACKET_PATH", audit / "complete_pre_holdout_packet.json"
    )
    monkeypatch.setattr(gate, "PROTECTED_HOLDOUT_ROOT", protected)
    monkeypatch.setattr(gate, "HOLDOUT_LOCK_PATH", protected / ".transaction.lock")
    monkeypatch.setattr(
        gate,
        "HOLDOUT_ACCESS_RECEIPT_PATH",
        protected / "holdout_access_receipt.json",
    )
    monkeypatch.setattr(gate, "HOLDOUT_RESULT_PATH", protected / "holdout_result.json")
    monkeypatch.setattr(
        gate,
        "HOLDOUT_SEAL_RECEIPT_PATH",
        protected / "holdout_seal_receipt.json",
    )
    monkeypatch.setattr(
        gate,
        "HOLDOUT_ABORT_RECEIPT_PATH",
        protected / "holdout_abort_receipt.json",
    )
    monkeypatch.setattr(gate, "HOLDOUT_TRACE_PATH", protected / "holdout_trace.jsonl")
    monkeypatch.setattr(
        gate,
        "HOLDOUT_EVALUATOR_RECEIPT_PATH",
        protected / "holdout_evaluator_receipt.json",
    )

    foundation = audit / "foundation"
    preregistration_path = foundation / "preregistration.json"
    session_path = foundation / "session_assignments.json"
    corpus_receipt_path = foundation / "corpus_receipt.json"
    lineage_receipt_path = foundation / "lineage_receipt.json"
    machinery_receipt_path = foundation / "machinery_receipt.json"
    monkeypatch.setattr(gate, "PREREGISTRATION_PATH", preregistration_path)
    monkeypatch.setattr(gate, "SESSION_ASSIGNMENTS_PATH", session_path)
    monkeypatch.setattr(
        gate, "CORPUS_INTEGRITY_RECEIPT_PATH", corpus_receipt_path
    )
    monkeypatch.setattr(gate, "LINEAGE_RECEIPT_PATH", lineage_receipt_path)
    monkeypatch.setattr(gate, "MACHINERY_RECEIPT_PATH", machinery_receipt_path)
    monkeypatch.setattr(gate, "_assert_preregistration_current", lambda _digest: None)
    fixture_environment = {"schema_version": "pathd.test_fit_environment.v1"}
    monkeypatch.setattr(
        gate._foundation,
        "assert_entry_fit_environment_current",
        lambda: fixture_environment,
    )

    preregistration = {
        "binding_plan": {"sha256": "1" * 64},
        "fill_law": {"fill_law_hash": "2" * 64},
        "source_hash_policy": {"closed": True, "version": 1},
    }
    _write_fixture_json(preregistration_path, preregistration)
    protected_sessions = [f"2026-07-{day:02d}" for day in range(1, 30)] + [
        "2026-07-31"
    ]
    _write_fixture_json(
        session_path, {"protected_holdout_30": protected_sessions}
    )
    _write_fixture_json(corpus_receipt_path, {"status": "PASS", "kind": "corpus"})
    _write_fixture_json(lineage_receipt_path, {"status": "PASS", "kind": "lineage"})
    _write_fixture_json(
        machinery_receipt_path, {"status": "PASS", "kind": "machinery"}
    )

    artifact_dir = audit / "preopen_artifacts"
    artifact_dir.mkdir()
    outer_paths = tuple(
        artifact_dir / f"outer_fold_{index}_result_receipt.json"
        for index in range(1, 6)
    )
    monkeypatch.setattr(gate, "OUTER_RESULT_RECEIPT_PATHS", outer_paths)
    for index, path in enumerate(outer_paths, start=1):
        _write_fixture_json(
            path,
            {
                "schema_version": "pathd.entry_outer_result_receipt.v1",
                "outer_fold": index,
                "status": "FROZEN_PRIMARY_OUTER_RESULT",
            },
        )
    monkeypatch.setattr(
        gate,
        "_validate_outer_result_semantics",
        lambda path, fold: {
            **json.loads(path.read_text(encoding="utf-8")),
            "validated_outer_fold": fold,
        },
    )

    named_paths = {
        "outer_entry_exit": artifact_dir / "outer_entry_exit.json",
        "four_box": artifact_dir / "four_box.json",
        "guard": artifact_dir / "guard.json",
        "acceptance": artifact_dir / "acceptance.json",
        "full_entry": artifact_dir / "full_entry.json",
        "full_exit": artifact_dir / "full_exit.json",
    }
    fixed_named_constants = {
        "outer_entry_exit": "OUTER_ENTRY_EXIT_ARTIFACTS_PATH",
        "four_box": "FOUR_BOX_PACKET_PATH",
        "guard": "GUARD_PANEL_PATH",
        "acceptance": "OUTER_ACCEPTANCE_PACKET_PATH",
        "full_entry": "FULL_FIT_ENTRY_ARTIFACTS_PATH",
        "full_exit": "FULL_FIT_EXIT_ARTIFACTS_PATH",
    }
    for name, constant in fixed_named_constants.items():
        monkeypatch.setattr(gate, constant, named_paths[name])
    semantic_names = {
        "outer_entry_exit": "outer_entry_exit_artifacts",
        "four_box": "four_box_packet",
        "guard": "guard_panel",
        "acceptance": "outer_acceptance_packet",
        "full_entry": "full_fit_entry_artifacts",
        "full_exit": "full_fit_exit_artifacts",
    }
    statuses = {
        "outer_entry_exit": "FROZEN_COMPLETE",
        "four_box": "FROZEN_COMPLETE",
        "guard": "PASS",
        "acceptance": "PASS",
        "full_entry": "FROZEN_COMPLETE",
        "full_exit": "FROZEN_COMPLETE",
    }
    for name, path in named_paths.items():
        value = {
            "schema_version": f"pathd.{semantic_names[name]}.v1",
            "artifact_kind": semantic_names[name],
            "status": statuses[name],
        }
        if name in {"four_box", "acceptance"}:
            value["selected_box_d_policy_id"] = "BOX_D::FROZEN"
        if name == "acceptance":
            value["selected_comparator_policy_id"] = "COMPARATOR::FROZEN"
        _write_fixture_json(path, value)

    def validator_for(name: str):
        def validate(path: Path) -> dict[str, object]:
            value = json.loads(path.read_text(encoding="utf-8"))
            assert value["artifact_kind"] == name
            return {**value, "artifact_sha256": _sha(path)}

        return validate

    monkeypatch.setattr(
        gate,
        "_registered_prepacket_artifact_validators",
        lambda: {
            semantic_name: validator_for(semantic_name)
            for semantic_name in semantic_names.values()
        },
    )

    outer_rows = [_artifact_row(path) for path in outer_paths]
    outer_entry_row = _artifact_row(named_paths["outer_entry_exit"])
    four_box = _artifact_row(named_paths["four_box"])
    guard = _artifact_row(named_paths["guard"])
    acceptance = _artifact_row(named_paths["acceptance"])
    full_entry = _artifact_row(named_paths["full_entry"])
    full_exit = _artifact_row(named_paths["full_exit"])
    closure = {
        row["path"]: row
        for row in (
            outer_rows
            + [outer_entry_row]
            + [four_box, guard, acceptance]
            + [full_entry]
            + [full_exit]
        )
    }
    artifact_rows = [closure[path] for path in sorted(closure)]
    primary_sessions = protected_sessions[:-1]
    semantic = {
        "schema_version": gate.PACKET_SCHEMA,
        "status": "FROZEN_PRE_HOLDOUT",
        "frozen_at_utc": "2026-08-01T16:00:00.000000Z",
        "preregistration_sha256": _sha(preregistration_path),
        "session_assignments_sha256": _sha(session_path),
        "source_hash_policy_sha256": gate._stable_hash(
            preregistration["source_hash_policy"]
        ),
        "corpus_integrity_receipt_sha256": _sha(corpus_receipt_path),
        "lineage_receipt_sha256": _sha(lineage_receipt_path),
        "machinery_receipt_sha256": _sha(machinery_receipt_path),
        "fit_environment_sha256": gate._foundation.stable_hash(
            fixture_environment
        ),
        "outer_result_receipts": outer_rows,
        "outer_entry_exit_artifacts": outer_entry_row,
        "four_box_packet": four_box,
        "guard_panel": guard,
        "outer_acceptance_packet": acceptance,
        "full_fit_entry_artifacts": full_entry,
        "full_fit_exit_artifacts": full_exit,
        "selected_box_d_policy_id": "BOX_D::FROZEN",
        "selected_comparator_policy_id": "COMPARATOR::FROZEN",
        "artifact_rows": artifact_rows,
        "artifact_root_sha256": gate._stable_hash(artifact_rows),
        "prerequisite_checks": {
            name: "PASS" for name in gate.PRE_HOLDOUT_PREREQUISITE_CHECK_NAMES
        },
        "protected_sessions_sha256_newline": gate._foundation.canonical_session_hash(
            protected_sessions
        ),
        "primary_sessions_sha256_newline": gate._foundation.canonical_session_hash(
            primary_sessions
        ),
        "holdout_open_count": 0,
        "quarantine_labels": list(gate._foundation.QUARANTINE_LABELS),
        "claim_boundary": gate._foundation.CLAIM_BOUNDARY,
        "holdout_caveat": gate._foundation.HOLDOUT_CAVEAT,
    }
    packet = {**semantic, "packet_sha256": gate._stable_hash(semantic)}
    gate._freeze_complete_pre_holdout_packet(packet)
    yield {
        "packet": packet,
        "protected_sessions": protected_sessions,
        "artifact_paths": named_paths,
    }
    for authorization in list(gate._ACTIVE_AUTH_OBJECTS.values()):
        gate._deactivate_authorization(authorization)
    gate._ACTIVE_AUTH_OBJECTS.clear()


@dataclass(frozen=True)
class _FakeDataset:
    schema_version: str
    authorization_sha256: str
    role: str
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    source_receipts: tuple[dict[str, object], ...]
    dataset_sha256: str


def _install_dataset_api(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    fail: bool = False,
) -> None:
    def loader(
        authorization: gate.ActiveProtectedHoldoutAuthorizationV1,
    ) -> _FakeDataset:
        assert gate.HOLDOUT_ACCESS_RECEIPT_PATH.exists()
        access_bytes = gate.HOLDOUT_ACCESS_RECEIPT_PATH.read_bytes()
        assert access_bytes == gate._canonical_json_bytes(json.loads(access_bytes))
        events.append("decode")
        if fail:
            raise RuntimeError("simulated protected decoder crash")
        receipts = tuple(
            {"session": session, "receipt_sha256": hashlib.sha256(session.encode()).hexdigest()}
            for session in authorization.sessions
        )
        return _FakeDataset(
            schema_version="pathd.entry_evidence_dataset.v1",
            authorization_sha256=authorization.authorization_sha256,
            role="protected_holdout_once",
            sessions=authorization.sessions,
            sessions_sha256_newline=authorization.sessions_sha256_newline,
            source_receipts=receipts,
            dataset_sha256="a" * 64,
        )

    def validator(
        dataset: _FakeDataset,
        *,
        authorization: gate.ActiveProtectedHoldoutAuthorizationV1,
    ) -> _FakeDataset:
        assert dataset.authorization_sha256 == authorization.authorization_sha256
        events.append("validate")
        return dataset

    monkeypatch.setattr(
        gate,
        "_registered_dataset_api",
        lambda: (_FakeDataset, loader, validator),
    )


@dataclass(frozen=True)
class _FakeEvaluation:
    semantic: dict[str, object]


def _valid_payload() -> dict[str, object]:
    guards = {name: True for name in gate.HOLDOUT_GUARD_KEYS}
    survival = {name: 0 for name in gate.HOLDOUT_SURVIVAL_VIOLATION_KEYS}
    criteria = {
        "primary_non_degraded_sessions_at_least_25": True,
        "completed_box_d_trades_on_primary_29_at_least_50": True,
        "box_d_net_pnl_on_primary_29_positive": True,
        "paired_comparator_delta_on_primary_29_positive": True,
        "all_guards_pass": True,
        "all_survival_violations_zero": True,
    }
    return {
        "schema_version": "pathd.protected_holdout_evaluation_payload.v1",
        "verdict": "PASS",
        "protected_session_count": 30,
        "primary_non_degraded_session_count": 29,
        "completed_box_d_trades_on_primary_29": 50,
        "box_d_net_pnl_micros_on_primary_29": 1,
        "box_d_minus_comparator_paired_net_pnl_micros_on_primary_29": 1,
        "guards": guards,
        "survival_violation_counts": survival,
        "owner_facing_metrics": {
            name: {"status": "REPORTED"} for name in gate.OWNER_FACING_METRIC_NAMES
        },
        "degraded_sensitivity": {
            "session": "2026-07-31",
            "excluded_from_primary": True,
            "metrics": {"status": "DIAGNOSTIC_ONLY"},
        },
        "pass_criteria_recomputed": criteria,
    }


def _install_evaluator_api(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    attempt_post_decode_abort: bool = False,
    attempt_forged_seal: bool = False,
    payload_mutator=None,
) -> None:
    last_semantic: dict[str, object] = {}

    def evaluator(
        authorization: gate.ActiveProtectedHoldoutAuthorizationV1,
        dataset: _FakeDataset,
    ) -> _FakeEvaluation:
        events.append("evaluate")
        if attempt_post_decode_abort:
            with pytest.raises(
                gate.ProtectedHoldoutError,
                match="authorization is inactive",
            ):
                gate._abort_protected_holdout(
                    authorization,
                    "0" * 64,
                    reason_code="OPERATOR_ABORT",
                )
            events.append("post_decode_abort_rejected")
        if attempt_forged_seal:
            with pytest.raises(
                gate.ProtectedHoldoutError,
                match="authorization is inactive",
            ):
                gate._seal_protected_holdout_result(
                    authorization,
                    {"payload": _valid_payload()},
                    "0" * 64,
                )
            events.append("forged_seal_rejected")
        payload = _valid_payload()
        if payload_mutator is not None:
            payload_mutator(payload)
        payload_sha256 = gate._stable_hash(payload)
        gate.write_json_exclusive_durable(
            gate.HOLDOUT_TRACE_PATH,
            {
                "schema_version": "pathd.protected_holdout_trace.v1",
                "authorization_sha256": authorization.authorization_sha256,
                "dataset_sha256": dataset.dataset_sha256,
                "session_count": 30,
            },
        )
        trace_sha256 = gate._sha256_regular(
            gate.HOLDOUT_TRACE_PATH, require_mode_600=True
        )
        receipt_semantic = {
            "schema_version": gate.EVALUATOR_RECEIPT_SCHEMA,
            "status": "EVALUATION_COMPLETE",
            "transaction_id": authorization.transaction_id,
            "authorization_sha256": authorization.authorization_sha256,
            "dataset_sha256": dataset.dataset_sha256,
            "source_receipts_root_sha256": authorization._source_receipts_root_sha256,
            "machinery_receipt_sha256": authorization.machinery_receipt_sha256,
            "evaluator_source_path": gate._path_label(gate.EVALUATOR_SOURCE_PATH),
            "evaluator_source_sha256": gate._sha256_regular(
                gate.EVALUATOR_SOURCE_PATH
            ),
            "trace_path": gate._path_label(gate.HOLDOUT_TRACE_PATH),
            "trace_artifact_sha256": trace_sha256,
            "trace_root_sha256": trace_sha256,
            "payload_sha256": payload_sha256,
        }
        evaluator_receipt = {
            **receipt_semantic,
            "receipt_sha256": gate._stable_hash(receipt_semantic),
        }
        gate.write_json_exclusive_durable(
            gate.HOLDOUT_EVALUATOR_RECEIPT_PATH, evaluator_receipt
        )
        semantic = {
            "schema_version": gate.EVALUATION_SCHEMA,
            "authorization_sha256": authorization.authorization_sha256,
            "access_receipt_sha256": authorization.access_receipt_sha256,
            "preopen_packet_sha256": authorization.preopen_packet_sha256,
            "artifact_root_sha256": authorization.artifact_root_sha256,
            "fit_environment_sha256": authorization.fit_environment_sha256,
            "sessions_sha256_newline": authorization.sessions_sha256_newline,
            "primary_sessions_sha256_newline": authorization.primary_sessions_sha256_newline,
            "dataset_sha256": dataset.dataset_sha256,
            "source_receipts_root_sha256": authorization._source_receipts_root_sha256,
            "box_d_policy_id": authorization.box_d_policy_id,
            "comparator_policy_id": authorization.comparator_policy_id,
            "trace_root_sha256": trace_sha256,
            "trace_artifact_sha256": trace_sha256,
            "evaluator_receipt_sha256": gate._sha256_regular(
                gate.HOLDOUT_EVALUATOR_RECEIPT_PATH, require_mode_600=True
            ),
            "payload_sha256": payload_sha256,
            "payload": payload,
        }
        last_semantic.clear()
        last_semantic.update(semantic)
        return _FakeEvaluation(semantic=semantic)

    def validator(
        evaluation: _FakeEvaluation,
        *,
        authorization: gate.ActiveProtectedHoldoutAuthorizationV1,
        dataset: _FakeDataset,
    ) -> dict[str, object]:
        assert evaluation.semantic["authorization_sha256"] == (
            authorization.authorization_sha256
        )
        assert evaluation.semantic["dataset_sha256"] == dataset.dataset_sha256
        events.append("validate_evaluation")
        return dict(evaluation.semantic)

    monkeypatch.setattr(
        gate,
        "_registered_evaluator_api",
        lambda: (
            _FakeEvaluation,
            evaluator,
            validator,
            lambda authorization, dataset: dict(last_semantic),
        ),
    )


def test_access_is_exclusive_durable_and_precedes_decode(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert inspect.signature(gate.execute_protected_holdout_once).parameters == {}
    assert gate.execute_protected_holdout_once.__closure__ is None
    assert not hasattr(gate, "begin_protected_holdout_once")
    assert not hasattr(gate, "abort_protected_holdout")
    assert not hasattr(gate, "_begin_protected_holdout_once")
    assert gate.inspect_protected_holdout_state().state == gate.UNOPENED
    events: list[str] = []
    _install_dataset_api(monkeypatch, events)
    _install_evaluator_api(monkeypatch, events)

    seal = gate.execute_protected_holdout_once()
    assert seal["status"] == "SEALED"
    assert events[0] == "decode"
    assert "evaluate" in events
    access = json.loads(gate.HOLDOUT_ACCESS_RECEIPT_PATH.read_bytes())
    assert access["protected_sessions"] == frozen_packet["protected_sessions"]
    assert stat_mode(gate.HOLDOUT_ACCESS_RECEIPT_PATH) == 0o600
    assert gate.inspect_protected_holdout_state().state == gate.SEALED
    with pytest.raises(gate.ProtectedHoldoutError):
        gate.execute_protected_holdout_once()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_result_is_bound_and_durably_sealed(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _install_dataset_api(monkeypatch, events)
    _install_evaluator_api(
        monkeypatch, events, attempt_post_decode_abort=True
    )
    seal = gate.execute_protected_holdout_once()
    assert seal["status"] == "SEALED"
    assert "post_decode_abort_rejected" in events
    state = gate.inspect_protected_holdout_state()
    assert state.state == gate.SEALED
    assert state.holdout_open_count == 1
    result = json.loads(gate.HOLDOUT_RESULT_PATH.read_bytes())
    assert result["holdout_open_count"] == 1
    assert result["dataset_sha256"] == "a" * 64
    assert result["authorization_sha256"] == seal["authorization_sha256"]
    assert result["payload_sha256"] == gate._stable_hash(result["payload"])
    assert result["box_d_policy_id"] == "BOX_D::FROZEN"
    assert result["comparator_policy_id"] == "COMPARATOR::FROZEN"
    with pytest.raises(gate.ProtectedHoldoutError):
        gate.execute_protected_holdout_once()


def test_crash_after_access_burns_and_recovery_never_decodes(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    decode_calls: list[str] = []
    _install_dataset_api(monkeypatch, decode_calls)

    def crash_after_access(
        authorization: gate.ActiveProtectedHoldoutAuthorizationV1,
        _transaction_token: str,
        /,
    ) -> dict[str, object]:
        _drop_originating_process_lease(authorization)
        raise RuntimeError("simulated crash after durable access receipt")

    monkeypatch.setattr(
        gate, "_execute_protected_holdout_transaction", crash_after_access
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        gate.execute_protected_holdout_once()
    assert gate.inspect_protected_holdout_state().state == gate.BURNED_INCOMPLETE

    state = gate.recover_protected_holdout_after_crash()
    assert state.state == gate.ABORTED
    assert decode_calls == []
    abort = json.loads(gate.HOLDOUT_ABORT_RECEIPT_PATH.read_bytes())
    assert abort["reason_code"] == "CRASH_RECOVERY_INCOMPLETE"
    assert abort["corpus_reopened"] is False
    assert abort["artifacts_rewritten"] is False
    assert gate.recover_protected_holdout_after_crash().state == gate.ABORTED


def test_post_access_validation_failure_releases_live_lease(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_after_access(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected post-access validation failure")

    monkeypatch.setattr(gate, "_validate_live_authorization", reject_after_access)
    with pytest.raises(RuntimeError, match="post-access validation failure"):
        gate.execute_protected_holdout_once()

    assert gate._ACTIVE_AUTH_OBJECTS == {}
    assert gate.inspect_protected_holdout_state().state == gate.BURNED_INCOMPLETE
    assert gate.recover_protected_holdout_after_crash().state == gate.ABORTED
    assert gate.HOLDOUT_ABORT_RECEIPT_PATH.exists()
    assert not gate.HOLDOUT_SEAL_RECEIPT_PATH.exists()


def test_crash_after_durable_result_burns_without_seal_or_reopen(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _install_dataset_api(monkeypatch, events)
    _install_evaluator_api(monkeypatch, events)
    durable_writer = gate.write_json_exclusive_durable

    def crash_before_seal(path: Path, payload: object) -> None:
        if path == gate.HOLDOUT_SEAL_RECEIPT_PATH:
            raise RuntimeError("simulated process crash after durable result")
        durable_writer(path, payload)

    monkeypatch.setattr(gate, "write_json_exclusive_durable", crash_before_seal)
    with pytest.raises(RuntimeError, match="simulated process crash"):
        gate.execute_protected_holdout_once()
    events_before_recovery = list(events)
    assert (
        gate.inspect_protected_holdout_state().state
        == gate.RESULT_DURABLE_PENDING_ABORT
    )

    monkeypatch.setattr(gate, "write_json_exclusive_durable", durable_writer)
    state = gate.recover_protected_holdout_after_crash()
    assert state.state == gate.ABORTED
    assert events == events_before_recovery
    assert not gate.HOLDOUT_SEAL_RECEIPT_PATH.exists()
    assert gate.HOLDOUT_ABORT_RECEIPT_PATH.exists()
    abort = json.loads(gate.HOLDOUT_ABORT_RECEIPT_PATH.read_bytes())
    assert abort["reason_code"] == "CORRUPT_BURNED"


def test_decode_failure_aborts_once_and_never_reopens(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _install_dataset_api(monkeypatch, events, fail=True)
    _install_evaluator_api(monkeypatch, events)
    with pytest.raises(RuntimeError, match="decoder crash"):
        gate.execute_protected_holdout_once()
    assert events == ["decode"]
    state = gate.inspect_protected_holdout_state()
    assert state.state == gate.ABORTED
    abort = json.loads(gate.HOLDOUT_ABORT_RECEIPT_PATH.read_bytes())
    assert abort["reason_code"] == "DATASET_DECODE_FAILED"
    with pytest.raises(gate.ProtectedHoldoutError):
        gate.execute_protected_holdout_once()
    assert gate.recover_protected_holdout_after_crash().state == gate.ABORTED
    assert events == ["decode"]


def test_forged_verdict_is_recomputed_and_burned(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _install_dataset_api(monkeypatch, events)

    def forge_verdict(payload: dict[str, object]) -> None:
        payload["verdict"] = "no_genuine_signal"

    _install_evaluator_api(monkeypatch, events, payload_mutator=forge_verdict)
    with pytest.raises(gate.ProtectedHoldoutError, match="verdict does not recompute"):
        gate.execute_protected_holdout_once()
    assert gate.inspect_protected_holdout_state().state == gate.ABORTED
    assert not gate.HOLDOUT_RESULT_PATH.exists()
    abort = json.loads(gate.HOLDOUT_ABORT_RECEIPT_PATH.read_bytes())
    assert abort["reason_code"] == "RESULT_COMMIT_FAILED"


def test_arbitrary_evaluation_object_cannot_enter_seal(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _install_dataset_api(monkeypatch, events)
    _install_evaluator_api(monkeypatch, events, attempt_forged_seal=True)
    seal = gate.execute_protected_holdout_once()
    assert seal["status"] == "SEALED"
    assert "forged_seal_rejected" in events
    assert gate.inspect_protected_holdout_state().state == gate.SEALED


def test_crash_with_partial_result_stays_corrupt_and_never_decodes(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _install_dataset_api(monkeypatch, events)

    def crash_with_partial_result(
        authorization: gate.ActiveProtectedHoldoutAuthorizationV1,
        _transaction_token: str,
        /,
    ) -> dict[str, object]:
        fd = os.open(
            gate.HOLDOUT_RESULT_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.write(fd, b"{")
        os.fsync(fd)
        os.close(fd)
        _drop_originating_process_lease(authorization)
        raise RuntimeError("simulated crash with partial result")

    monkeypatch.setattr(
        gate,
        "_execute_protected_holdout_transaction",
        crash_with_partial_result,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        gate.execute_protected_holdout_once()

    assert gate.inspect_protected_holdout_state().state == gate.CORRUPT_BURNED
    assert gate.recover_protected_holdout_after_crash().state == gate.CORRUPT_BURNED
    assert events == []
    assert gate.HOLDOUT_ABORT_RECEIPT_PATH.exists()


@pytest.mark.parametrize(
    "corruption",
    ["broken_access_symlink", "partial_access", "premature_result", "unexpected_file"],
)
def test_symlink_partial_reserved_and_unexpected_paths_burn_fail_closed(
    frozen_packet: dict[str, object], corruption: str
) -> None:
    gate.PROTECTED_HOLDOUT_ROOT.mkdir(mode=0o700)
    if corruption == "broken_access_symlink":
        gate.HOLDOUT_ACCESS_RECEIPT_PATH.symlink_to(
            gate.PROTECTED_HOLDOUT_ROOT / "missing-target"
        )
        expected_count = 1
    elif corruption == "partial_access":
        fd = os.open(
            gate.HOLDOUT_ACCESS_RECEIPT_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.write(fd, b"{")
        os.close(fd)
        expected_count = 1
    elif corruption == "premature_result":
        gate.write_json_exclusive_durable(gate.HOLDOUT_RESULT_PATH, {"premature": True})
        expected_count = 0
    else:
        (gate.PROTECTED_HOLDOUT_ROOT / "surprise.bin").write_bytes(b"surprise")
        expected_count = 0

    state = gate.inspect_protected_holdout_state()
    assert state.state == gate.CORRUPT_BURNED
    assert state.holdout_open_count == expected_count
    with pytest.raises(gate.ProtectedHoldoutError):
        gate.execute_protected_holdout_once()
    assert gate.recover_protected_holdout_after_crash().state == gate.CORRUPT_BURNED


def test_packet_and_artifact_tamper_block_before_access(
    frozen_packet: dict[str, object]
) -> None:
    artifact = frozen_packet["artifact_paths"]["guard"]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    state = gate.inspect_protected_holdout_state()
    assert state.state == gate.CORRUPT_BURNED
    assert state.holdout_open_count == 0
    with pytest.raises(gate.ProtectedHoldoutError):
        gate.execute_protected_holdout_once()
    assert not os.path.lexists(gate.HOLDOUT_ACCESS_RECEIPT_PATH)


def test_packet_rejects_declared_pass_or_policy_without_semantic_proof(
    frozen_packet: dict[str, object]
) -> None:
    packet = dict(frozen_packet["packet"])
    packet["prerequisite_checks"] = {"caller_says_pass": "PASS"}
    semantic = dict(packet)
    semantic.pop("packet_sha256")
    packet["packet_sha256"] = gate._stable_hash(semantic)
    with pytest.raises(gate.ProtectedHoldoutError, match="prerequisite"):
        gate._validate_complete_pre_holdout_packet(packet)

    packet = dict(frozen_packet["packet"])
    packet["selected_box_d_policy_id"] = "CALLER_SELECTED_AFTER_RESULTS"
    semantic = dict(packet)
    semantic.pop("packet_sha256")
    packet["packet_sha256"] = gate._stable_hash(semantic)
    with pytest.raises(gate.ProtectedHoldoutError, match="policy identity"):
        gate._validate_complete_pre_holdout_packet(packet)


def test_preregistration_revalidation_semantically_validates_corpus_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    preregistration_receipt = {"preregistration_sha256": digest}
    payload = {"frozen": True}
    lineage_receipt = {"lineage": True}
    machinery_receipt = {"machinery": True}
    events: list[str] = []

    monkeypatch.setattr(
        gate._foundation,
        "assert_preregistration_frozen",
        lambda: preregistration_receipt,
    )
    monkeypatch.setattr(gate._foundation, "read_json", lambda _path: payload)
    monkeypatch.setattr(
        gate._foundation,
        "_verify_immutable_sources",
        lambda current: events.append("immutable")
        if current is payload
        else pytest.fail("immutable validation input drifted"),
    )
    monkeypatch.setattr(
        gate._foundation,
        "assert_lineage_implementation_frozen",
        lambda: lineage_receipt,
    )

    def validate_machinery(
        prereg: object, current: object, lineage: object
    ) -> dict[str, bool]:
        assert (prereg, current, lineage) == (
            preregistration_receipt,
            payload,
            lineage_receipt,
        )
        events.append("machinery")
        return machinery_receipt

    monkeypatch.setattr(
        gate._foundation, "_assert_entry_machinery_frozen", validate_machinery
    )

    def validate_corpus(
        current: object, prereg: object, machinery: object
    ) -> dict[str, bool]:
        assert (current, prereg, machinery) == (
            payload,
            preregistration_receipt,
            machinery_receipt,
        )
        events.append("corpus")
        return {"validated": True}

    monkeypatch.setattr(
        gate._foundation, "_validated_corpus_integrity_receipt", validate_corpus
    )
    monkeypatch.setattr(
        gate._foundation,
        "_verify_authorized_dependency_closure",
        lambda current, *, allow_missing_future: events.append("dependencies")
        if current is payload and allow_missing_future is False
        else pytest.fail("dependency validation inputs drifted"),
    )
    monkeypatch.setattr(
        gate._foundation,
        "assert_entry_fit_environment_current",
        lambda: events.append("environment"),
    )
    monkeypatch.setattr(
        gate._foundation,
        "assert_research_foundation_stable",
        lambda: events.append("stability"),
    )

    gate._assert_preregistration_current(digest)
    assert events == [
        "immutable",
        "machinery",
        "corpus",
        "dependencies",
        "environment",
        "stability",
    ]


def test_parent_directory_fsync_failure_prevents_access_receipt(
    frozen_packet: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = gate.PROTECTED_HOLDOUT_ROOT.parent.stat()
    real_fsync = gate.os.fsync

    def fail_parent_fsync(fd: int) -> None:
        opened = os.fstat(fd)
        if opened.st_dev == parent.st_dev and opened.st_ino == parent.st_ino:
            raise OSError("injected audit-parent fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(gate.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="injected audit-parent"):
        gate.execute_protected_holdout_once()
    assert not os.path.lexists(gate.HOLDOUT_ACCESS_RECEIPT_PATH)


def test_nonblocking_flock_excludes_a_second_process(
    frozen_packet: dict[str, object]
) -> None:
    lock_fd = gate._open_lock(create=True)
    assert lock_fd is not None
    try:
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import fcntl, os, sys; "
                    "fd=os.open(sys.argv[1], os.O_RDWR|os.O_NOFOLLOW); "
                    "\ntry:\n fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)"
                    "\nexcept BlockingIOError:\n print('BUSY')"
                    "\nelse:\n print('ESCAPED')"
                    "\nos.close(fd)"
                ),
                str(gate.HOLDOUT_LOCK_PATH),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        gate._close_lock(lock_fd)
    assert child.stdout.strip() == "BUSY"


def test_exclusive_writer_rejects_nonfinite_and_every_occupied_inode(
    frozen_packet: dict[str, object]
) -> None:
    root = gate.PROTECTED_HOLDOUT_ROOT
    root.mkdir(mode=0o700)
    target = root / "standalone.json"
    with pytest.raises(gate.ProtectedHoldoutError):
        gate.write_json_exclusive_durable(target, {"bad": float("nan")})
    gate.write_json_exclusive_durable(target, {"b": 2, "a": 1})
    assert target.read_bytes() == b'{"a":1,"b":2}'
    assert stat_mode(target) == 0o600
    with pytest.raises(gate.ProtectedHoldoutError):
        gate.write_json_exclusive_durable(target, {"a": 1})

    broken = root / "broken.json"
    broken.symlink_to(root / "does-not-exist")
    with pytest.raises(gate.ProtectedHoldoutError):
        gate.write_json_exclusive_durable(broken, {"a": 1})


def test_opaque_types_reject_reconstruction_and_bad_abort_reason(
    frozen_packet: dict[str, object]
) -> None:
    with pytest.raises(TypeError):
        gate.ActiveProtectedHoldoutAuthorizationV1(
            object(),
            transaction_token_sha256="0" * 64,
            lock_fd=-1,
            semantic={},
        )
    assert "ActiveProtectedHoldoutAuthorizationV1" not in gate.__all__
    assert "begin_protected_holdout_once" not in gate.__all__
    assert "abort_protected_holdout" not in gate.__all__
    assert inspect.signature(gate.execute_protected_holdout_once).parameters == {}
