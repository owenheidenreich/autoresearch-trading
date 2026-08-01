"""Focused regressions for the process-local Path-D evidence capability gate."""
from __future__ import annotations

from dataclasses import fields, replace
import json
import os
from pathlib import Path

import pytest

from v4.research import pathd_evidence_gate as gate


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


@pytest.fixture(autouse=True)
def _isolated_capability_registry() -> None:
    gate._ACTIVE_CAPABILITIES.clear()
    yield
    gate._ACTIVE_CAPABILITIES.clear()


def _configure_tmp_campaign(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        gate,
        "ENTRY_FOLD_ARTIFACT_ROOT",
        tmp_path / "campaign" / "entry_outer_folds",
    )


def _claim(*, generation_sha256: str = _B) -> dict[str, object]:
    value: dict[str, object] = {
        "role": "outer_test_primary",
        "outer_fold": 1,
        "inner_fold": None,
        "sessions": ("2026-01-02",),
        "sessions_sha256_newline": _A,
        "preregistration_sha256": _A,
        "session_assignments_sha256": _A,
        "source_hash_policy_sha256": _A,
        "corpus_integrity_receipt_sha256": _A,
        "lineage_receipt_sha256": _A,
        "machinery_receipt_sha256": _A,
        "fit_environment_sha256": _A,
        "open_gate_receipts_sha256": (_A,),
    }
    authorization_fields = {
        item.name for item in fields(gate._foundation.FrozenEvidenceAuthorization)
    }
    if "foundation_generation_sha256" in authorization_fields:
        value["foundation_generation_sha256"] = generation_sha256
    if "foundation_stability_receipt_sha256" in authorization_fields:
        value["foundation_stability_receipt_sha256"] = _C
    return value


def _issue_durable_access() -> tuple[gate._ScopePaths, object]:
    paths = gate._scope_paths(
        role="outer_test_primary", outer_fold=1, inner_fold=None
    )
    transaction_id = "00000000-0000-4000-8000-000000000001"
    access = gate._access_receipt(
        paths, claim=_claim(), transaction_id=transaction_id
    )
    gate._write_exclusive_json(paths.access, access)
    authorization = gate._authorization_from_access(paths, access)
    return paths, authorization


@pytest.mark.parametrize(
    "capability_fault",
    ["detached", "wrong_identity", "wrong_pid", "wrong_hash", "inactive"],
)
def test_untrusted_exact_authorization_cannot_read_or_burn_fixed_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capability_fault: str,
) -> None:
    _configure_tmp_campaign(monkeypatch, tmp_path)
    paths, issued = _issue_durable_access()
    authorization = replace(issued)
    assert authorization == issued and authorization is not issued

    if capability_fault != "detached":
        active_authorization = (
            issued if capability_fault == "wrong_identity" else authorization
        )
        gate._ACTIVE_CAPABILITIES[id(authorization)] = gate._ActiveCapability(
            pid=(os.getpid() + 1 if capability_fault == "wrong_pid" else os.getpid()),
            lock_fd=-1,
            authorization=active_authorization,
            authorization_sha256=(
                _C
                if capability_fault == "wrong_hash"
                else gate._stable_hash(authorization.to_dict())
            ),
            state=("REVOKED" if capability_fault == "inactive" else "ISSUED_BEFORE_DECODE"),
        )

    access_before = paths.access.read_bytes()

    def _unexpected_scope_io(*args: object, **kwargs: object) -> object:
        raise AssertionError("fixed-scope read or burn occurred before capability authentication")

    monkeypatch.setattr(
        gate, "read_frozen_entry_evidence_authorization", _unexpected_scope_io
    )
    monkeypatch.setattr(gate, "_write_burned", _unexpected_scope_io)
    with pytest.raises(
        gate.EntryEvidenceGateError,
        match=(
            "no active process-local decode capability"
            if capability_fault != "inactive"
            else "evidence capability is not active"
        ),
    ):
        gate.validate_entry_evidence_access(authorization)

    assert paths.access.read_bytes() == access_before
    assert not paths.burned_receipt.exists()
    assert sorted(path.name for path in paths.access.parent.iterdir()) == [
        paths.access.name
    ]


def test_active_capability_foundation_drift_burns_and_releases_transaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from v4.research import pathd_holdout_gate

    _configure_tmp_campaign(monkeypatch, tmp_path)
    paths, authorization = _issue_durable_access()
    capability_key = id(authorization)
    gate._ACTIVE_CAPABILITIES[capability_key] = gate._ActiveCapability(
        pid=os.getpid(),
        lock_fd=731,
        authorization=authorization,
        authorization_sha256=gate._stable_hash(authorization.to_dict()),
        state="ISSUED_BEFORE_DECODE",
    )
    released: list[int] = []
    monkeypatch.setattr(gate, "_release_lock", released.append)

    def _foundation_drift(**kwargs: object) -> object:
        raise gate.EntryEvidenceGateError("synthetic frozen foundation drift")

    monkeypatch.setattr(
        pathd_holdout_gate, "_assert_protected_holdout_unopened", lambda: None
    )
    monkeypatch.setattr(
        gate._foundation,
        "assert_research_foundation_stable",
        _foundation_drift,
        raising=False,
    )
    with pytest.raises(
        gate.EntryEvidenceGateError, match="synthetic frozen foundation drift"
    ):
        gate.validate_entry_evidence_access(authorization)

    burned = json.loads(paths.burned_receipt.read_text(encoding="utf-8"))
    assert burned["status"] == "BURNED_NO_REOPEN"
    assert burned["reason"] == "FROZEN_FOUNDATION_DRIFT_BEFORE_DECODE"
    assert burned["transaction_id"] == authorization.transaction_id
    assert burned["access_scope_id"] == authorization.access_scope_id
    assert burned["access_count"] == 1
    assert capability_key not in gate._ACTIVE_CAPABILITIES
    assert released == [731]


def test_scope_id_is_bound_to_frozen_foundation_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tmp_campaign(monkeypatch, tmp_path)
    paths = gate._scope_paths(
        role="outer_test_primary", outer_fold=1, inner_fold=None
    )
    first = _claim(generation_sha256=_A)
    second = _claim(generation_sha256=_B)
    if "foundation_generation_sha256" not in first:
        second["preregistration_sha256"] = _B

    assert gate._scope_id(paths, claim=first) != gate._scope_id(
        paths, claim=second
    )
