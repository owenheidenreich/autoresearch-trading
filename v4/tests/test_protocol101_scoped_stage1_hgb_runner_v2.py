from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as base
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner_v2 as runner


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        mode="train-hypothesis",
        out_dir=tmp_path,
        readiness=base.DEFAULT_READINESS,
        hypothesis="H1",
        owner_approved_plumbing_smoke=False,
        owner_approved_offline_training=True,
        force=False,
        smoke_rows_per_session=20,
    )


def _plan() -> dict:
    return {
        "fold_governance_hash": "fold",
        "acceptance_registry_hash": "registry",
        "code_hashes": {"runner": "hash"},
        "identity_receipt": {"status": "PASS"},
        "provenance": {
            "campaign_contract_sha256": "a" * 64,
            "campaign_preregistration_sha256": "b" * 64,
            "fold_governance_sha256": "c" * 64,
            "acceptance_registry_sha256": "d" * 64,
            "simulator_source_sha256": "e" * 64,
        },
    }


def test_resume_preserves_existing_preregistration(tmp_path: Path) -> None:
    args = _args(tmp_path)
    first = runner.load_or_create_preregistration(args, plan=_plan())
    before = (tmp_path / "preregistration.json").read_bytes()
    second = runner.load_or_create_preregistration(args, plan=_plan())
    assert first == second
    assert (tmp_path / "preregistration.json").read_bytes() == before


def test_resume_rejects_changed_contract(tmp_path: Path) -> None:
    args = _args(tmp_path)
    runner.load_or_create_preregistration(args, plan=_plan())
    changed = dict(_plan())
    changed["provenance"] = {
        **changed["provenance"],
        "fold_governance_sha256": "f" * 64,
    }
    with pytest.raises(RuntimeError, match="differs"):
        runner.load_or_create_preregistration(args, plan=changed)


def test_resume_rejects_model_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / base.FRESH_CAMPAIGN_NAMESPACE
    root.mkdir()
    model = root / "model.pkl"
    model.write_bytes(b"model")
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "campaign_namespace": base.FRESH_CAMPAIGN_NAMESPACE,
                "campaign_contract_sha256": "a" * 64,
                "campaign_preregistration_sha256": "b" * 64,
                "runner_preregistration_sha256": "p" * 64,
                "fold": 0,
                "fold_id": "expanding_fold_01",
                "fit_sessions": ["a"],
                "calibration_sessions": ["b"],
                "validation_sessions": ["c"],
                "unit": {
                    "hypothesis": "H1",
                    "policy_index": 0,
                    "seed": 42,
                    "feature_names": list(base.HYPOTHESES["H1"]),
                    "config": {
                        "hypothesis": "H1",
                        "policy_index": 0,
                        "seed": 42,
                    },
                    "simulator_version": (
                        base.PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
                    ),
                },
                "model_artifact": {
                    "path": str(model),
                    "sha256": "wrong",
                },
            }
        )
    )
    with pytest.raises(RuntimeError, match="model hash mismatch"):
        runner.verify_resumable_unit(
            summary_path,
            hypothesis="H1",
            policy=0,
            seed=42,
            fold={"fold": 0, "fold_id": "expanding_fold_01"},
            fit_sessions=["a"],
            calibration_sessions=["b"],
            validation_sessions=["c"],
            preregistration_payload={"preregistration_hash": "p" * 64},
            provenance={
                "campaign_contract_sha256": "a" * 64,
                "campaign_preregistration_sha256": "b" * 64,
            },
        )


def test_code_hashes_freeze_wrapper_and_delegated_runner() -> None:
    hashes = runner.code_hashes()
    assert str(runner.RUNNER_PATH) in hashes
    assert str(runner.ORIGINAL_RUNNER_PATH) in hashes
    assert hashes[str(runner.RUNNER_PATH)] != hashes[str(runner.ORIGINAL_RUNNER_PATH)]
