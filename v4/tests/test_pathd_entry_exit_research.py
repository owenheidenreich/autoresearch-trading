from __future__ import annotations

import ast
import copy
from dataclasses import replace
import pytest
from pathlib import Path

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
import v4.research.pathd_entry_exit as research
from v4.research.pathd_entry_exit import (
    BURNED_SMOKE_DATES,
    OPRA_DEGRADED_DATES,
    fill_law,
    feature_lineage,
    forward_inner_folds,
    floor_law,
    require_all_negative_fixtures_rejected,
    preregistration_payload,
    assert_no_evidence_firewall_sessions,
    assert_no_holdout_sessions,
    session_assignments,
    validate_preregistration_payload,
)


def test_pathd_session_assignment_is_exact_and_firewalled() -> None:
    assignments = session_assignments()
    assert len(assignments["all_251_sessions"]) == 251
    assert len(assignments["folds"]) == 5
    assert [len(row["outer_test"]) for row in assignments["folds"]] == [28] * 5
    assert len(assignments["firewall_raw_indices_216_251"]) == 36
    assert len(assignments["protected_holdout_30"]) == 30
    assert not set(BURNED_SMOKE_DATES) & set(assignments["protected_holdout_30"])
    assert set(OPRA_DEGRADED_DATES).isdisjoint(
        session
        for fold in assignments["folds"]
        for session in fold["outer_test"]
    )
    hashes = assignments["ordered_list_hashes_sha256_newline"]
    assert hashes["all_251_sessions"] == "c9f61585bd4a566c4facba4816165eed56c60d4dbe7fdeb2d6af0f6c104d7c64"
    assert hashes["protected_holdout_30"] == "3f88231a0c32ed47821a91f3fc8da8f80caf9702b8c29fec7fb4aa16e7506e67"


def test_pathd_signed_entry_feature_contract_remains_exactly_17() -> None:
    assert len(FEATURE_NAMES) == 17
    assert len(set(FEATURE_NAMES)) == 17
    assert "bid" not in FEATURE_NAMES
    assert "ask" not in FEATURE_NAMES
    assert "iv" not in FEATURE_NAMES


def test_pathd_locked_fill_and_floor_laws() -> None:
    fill = fill_law()
    assert fill["headline_delay_ms"] == 1_000
    assert fill["completed_round_trip_fee_cash_micros"] == 3_000_000
    assert fill["delay_sensitivity_ms"] == [0, 1_000, 2_000, 5_000]
    assert fill["fill_price_mode"] == "SUBMITTED_HARD_LIMIT"
    assert fill["forced_flat"]["delay_ms"] == 0
    floor = floor_law()
    assert floor["formula"] == "floor_bid=max(0,0.50*P_entry+3/100)"
    assert floor["trailing"] is False
    assert floor["learned"] is False
    assert floor["intent_origin"] == "DETERMINISTIC_EXIT"


def test_pathd_all_mandatory_lineage_negative_fixtures_fail_closed() -> None:
    result = require_all_negative_fixtures_rejected()
    assert len(result) == 5
    assert all(result.values())


@pytest.mark.parametrize("fold", range(5))
def test_pathd_fold_calibration_is_strictly_before_outer_test(fold: int) -> None:
    row = session_assignments()["folds"][fold]
    assert max(row["model_fit"]) < row["calibration_embargo"][0]
    assert row["calibration_embargo"][0] < min(row["calibration_last_20_percent"])
    assert max(row["calibration_last_20_percent"]) < row["embargo"][0]
    assert row["embargo"][0] < min(row["outer_test"])


def test_pathd_inner_blocks_are_forward_chaining_and_embargoed() -> None:
    assignments = session_assignments()
    assert [
        fold["exit_training_oof_mapping"]["weight_session_count"]
        for fold in assignments["folds"]
    ] == [0, 0, 19, 48, 56]
    assert [
        fold["exit_training_oof_mapping"]["calibration_session_count"]
        for fold in assignments["folds"]
    ] == [0, 15, 19, 23, 28]
    for outer in assignments["folds"]:
        inner = outer["inner_forward_folds"]
        assert len(inner["blocks"]) == 5
        assert max(inner["block_sizes"]) - min(inner["block_sizes"]) <= 1
        assert len(inner["scored_forward_folds"]) == 4
        for fold in inner["scored_forward_folds"]:
            assert max(fold["model_fit"]) < fold["calibration_embargo"][0]
            assert fold["calibration_embargo"][0] < min(fold["calibration"])
            assert max(fold["calibration"]) < fold["outer_validation_embargo"][0]
            assert fold["outer_validation_embargo"][0] < min(fold["validation"])
            assert fold["calibration_valid"] is (len(fold["calibration"]) >= 10)
        mapping = outer["exit_training_oof_mapping"]
        invalid_validations = {
            session
            for row in inner["scored_forward_folds"]
            if not row["calibration_valid"]
            for session in row["validation"]
        }
        assert invalid_validations.isdisjoint(mapping["exit_model_weight_sessions"])
        assert invalid_validations.isdisjoint(mapping["exit_calibration_sessions"])


def test_pathd_lineage_requires_implementation_receipt_before_fit() -> None:
    lineage = feature_lineage()
    assert lineage["implementation_receipt_required_before_fit"] is True
    assert lineage["training_allowed_at_preregistration"] is False
    assert len(lineage["features"]) == 17
    assert all(row["future_live_twin_adapter"] for row in lineage["features"])


def test_pathd_semantic_preregistration_validates_and_has_no_wall_clock() -> None:
    payload, assignments, lineage = preregistration_payload()
    validate_preregistration_payload(payload, assignments, lineage)
    assert "frozen_at_utc" not in payload
    integrity = payload["corpus"]["integrity_contract"]
    assert integrity["partitions_in_order"] == [
        research.CORE_INTEGRITY_PARTITION,
        research.CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION,
        research.UNCONSUMED_VIX_CACHE_PARTITION,
    ]
    exact = {
        research.CORE_INTEGRITY_PARTITION: (
            2_766, 21_471_285_394,
            "31e1abb88b9a5539f9445a63c9af650385f8ded207a94021422a53a364a29497",
        ),
        research.CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION: (
            604, 8_843_589,
            "11605b5f8cd88957b5b0227ef45ec8380d80cfbfae16ff0f0111c55857d8ae1c",
        ),
        research.UNCONSUMED_VIX_CACHE_PARTITION: (
            338, 4_663_695,
            "61bfb0daf39de84879ee31df9d9023d0563b04c41e36196c91580c35155df56f",
        ),
    }
    assert sum(value[0] for value in exact.values()) == 3_708
    assert sum(value[1] for value in exact.values()) == 21_484_792_678
    for partition_id, expected in exact.items():
        partition = integrity["partitions"][partition_id]
        assert (
            partition["file_count"], partition["total_bytes"],
            partition["ordered_entries_semantic_sha256"],
        ) == expected
    assert len(
        research.integrity_manifest_entries_for_partition(
            research.CORE_INTEGRITY_PARTITION
        )
    ) == 2_766
    assert len(
        research.integrity_manifest_entries_for_partition(
            research.CONTEXT_DIAGNOSTIC_INTEGRITY_PARTITION
        )
    ) == 604
    with pytest.raises(RuntimeError, match="not loader-authorized"):
        research.integrity_manifest_entries_for_partition(
            research.UNCONSUMED_VIX_CACHE_PARTITION
        )
    core_receipt = research.core_corpus_integrity_receipt_spec()
    assert core_receipt["status"] == "PASS_CORE_AUTHORITATIVE"
    assert core_receipt["partition_id"] == research.CORE_INTEGRITY_PARTITION
    assert core_receipt["verification"] == {
        "file_count": 2_766,
        "total_bytes": 21_471_285_394,
        "ordered_entries_semantic_sha256": (
            "31e1abb88b9a5539f9445a63c9af650385f8ded207a94021422a53a364a29497"
        ),
        "diagnostic_files_verified": 0,
        "cache_files_verified": 0,
        "mismatches": [],
        "unreceipted_paths": [],
        "loader_rehash_required": True,
    }
    assert "no diagnostic" in core_receipt["authority"]
    assert payload["source_hash_policy"]["receipt_contracts"][
        "core_corpus_integrity_receipt"
    ] == core_receipt
    official_spx = lineage["raw_leaf_registry"]["official_spx"]
    assert official_spx["path"] == (
        "vendor/thetadata/index/spx_1m/{session}.parquet"
    )
    assert official_spx["timestamp_semantics"] == "BAR_OPEN_TIMESTAMP"
    assert official_spx["available_at"] == "event_time+60 seconds"
    assert official_spx["row_invariants"] == {
        "symbol": "SPX",
        "context_source": "thetadata_index_history_ohlc",
        "is_derived": False,
        "is_proxy": False,
        "is_official_index_data": True,
    }
    assert lineage["source_mutation_canaries"]["raw_index_vix_cache_mutation"].endswith(
        "cache_path_never_opened"
    )


def test_pathd_evidence_guard_blocks_all_36_firewall_sessions() -> None:
    assignments = session_assignments()
    with pytest.raises(RuntimeError, match="firewall session"):
        assert_no_evidence_firewall_sessions(assignments["burned_smoke_dates"][:1])
    with pytest.raises(RuntimeError, match="firewall session"):
        assert_no_evidence_firewall_sessions(assignments["protected_holdout_30"][:1])
    assert_no_evidence_firewall_sessions(assignments["folds"][0]["outer_test"])


def test_pathd_holdout_guard_is_narrower_for_permitted_schema_audits() -> None:
    assignments = session_assignments()
    assert_no_holdout_sessions(assignments["burned_smoke_dates"])
    with pytest.raises(RuntimeError, match="protected holdout"):
        assert_no_holdout_sessions(assignments["protected_holdout_30"][:1])


def _redirect_freeze(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    strict_repo_file = research._canonical_repo_regular_file

    def allow_test_audit_file(label: object) -> Path:
        candidate = Path(str(label))
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError as exc:
                raise RuntimeError("test fixed artifact is absent") from exc
            if not resolved.is_relative_to(root.resolve()):
                raise RuntimeError("test fixed artifact escaped temporary audit root")
            if candidate.is_symlink() or not resolved.is_file():
                raise RuntimeError("test fixed artifact is not a regular file")
            return resolved
        return strict_repo_file(label)

    monkeypatch.setattr(
        research, "_canonical_repo_regular_file", allow_test_audit_file
    )
    monkeypatch.setattr(research, "AUDIT_ROOT", root)
    monkeypatch.setattr(research, "PREREG_PATH", root / "preregistration.json")
    monkeypatch.setattr(research, "PREREG_HASH_PATH", root / "preregistration.sha256")
    monkeypatch.setattr(research, "SESSION_PATH", root / "session_assignments.json")
    monkeypatch.setattr(research, "LINEAGE_PATH", root / "feature_lineage.json")
    monkeypatch.setattr(research, "FREEZE_RECEIPT_PATH", root / "preregistration_freeze_receipt.json")
    monkeypatch.setattr(
        research,
        "LINEAGE_IMPLEMENTATION_RECEIPT_PATH",
        root / "feature_lineage_implementation_receipt.json",
    )
    monkeypatch.setattr(
        research,
        "ENTRY_MACHINERY_RECEIPT_PATH",
        root / "entry_machinery_receipt.json",
    )
    monkeypatch.setattr(
        research,
        "ENTRY_MACHINERY_TEST_EVIDENCE_PATH",
        root / "entry_machinery_test_evidence.json",
    )


def test_pathd_freeze_is_idempotent_and_prereg_alone_cannot_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _redirect_freeze(monkeypatch, root)
    first = research.freeze_preregistration()
    second = research.freeze_preregistration()
    assert first["preregistration_sha256"] == second["preregistration_sha256"]
    with pytest.raises(RuntimeError, match="lineage implementation receipt is absent"):
        research.assert_lineage_implementation_frozen()
    with pytest.raises(RuntimeError, match="lineage implementation receipt is absent"):
        research.assert_entry_fit_ready(role="outer_weights", outer_fold=1)


def test_pathd_existing_freeze_detects_tampered_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audit"
    _redirect_freeze(monkeypatch, root)
    research.freeze_preregistration()
    research.SESSION_PATH.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="session-assignment freeze integrity"):
        research.freeze_preregistration()


@pytest.mark.parametrize(
    "label",
    [
        "../outside.py",
        "/tmp/outside.py",
        "./v4/research/pathd_entry_exit.py",
        "v4/research/../research/pathd_entry_exit.py",
        "v4\\research\\pathd_entry_exit.py",
    ],
)
def test_pathd_fit_receipt_paths_reject_noncanonical_labels(label: str) -> None:
    with pytest.raises(RuntimeError, match="implementation path"):
        research._canonical_repo_regular_file(label)


def test_pathd_fit_receipt_path_accepts_one_exact_repo_regular_file() -> None:
    path = research._canonical_repo_regular_file(
        "v4/research/pathd_entry_exit.py"
    )
    assert path == research.REPO_ROOT / "v4/research/pathd_entry_exit.py"


@pytest.mark.parametrize(
    ("role", "outer_fold", "inner_fold"),
    [
        ("outer_weights", 1, None),
        ("outer_calibration", 5, None),
        ("nested_weights", 3, 4),
        ("nested_calibration", 5, 4),
        ("full_weights", None, None),
        ("full_calibration", None, None),
    ],
)
def test_pathd_closed_fit_roles_resolve_only_frozen_sessions(
    role: str, outer_fold: int | None, inner_fold: int | None
) -> None:
    assignments = session_assignments()
    sessions = research._resolve_frozen_fit_sessions(
        assignments,
        role=role,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    assert sessions
    assert list(sessions) == sorted(sessions)
    assert set(sessions).issubset(assignments["pre_holdout_session_indices_1_215"])
    assert set(sessions).isdisjoint(assignments["firewall_raw_indices_216_251"])


def test_pathd_invalid_or_under_calibrated_fit_roles_fail_closed() -> None:
    assignments = session_assignments()
    with pytest.raises(RuntimeError, match="unsupported frozen fit role"):
        research._resolve_frozen_fit_sessions(
            assignments, role="arbitrary", outer_fold=None, inner_fold=None
        )
    with pytest.raises(RuntimeError, match="fewer than 10 calibration sessions"):
        research._resolve_frozen_fit_sessions(
            assignments, role="nested_weights", outer_fold=1, inner_fold=1
        )
    tampered = research.json.loads(research.json.dumps(assignments))
    tampered["folds"][0]["model_fit"][0] = tampered["protected_holdout_30"][0]
    with pytest.raises(RuntimeError, match="pre-holdout era|protected session|firewall session"):
        research._resolve_frozen_fit_sessions(
            tampered, role="outer_weights", outer_fold=1, inner_fold=None
        )


def test_pathd_fit_receipt_rejects_symlinked_source() -> None:
    with pytest.raises(RuntimeError, match="symlinked implementation path"):
        research._canonical_repo_regular_file(
            "v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md"
        )


def test_pathd_implementation_receipt_requires_exact_unique_coverage() -> None:
    label = "v4/research/pathd_entry_exit.py"
    digest = research.sha256_path(research.REPO_ROOT / label)
    valid = {"implementations": [{"path": label, "sha256": digest}]}
    assert research._implementation_receipt_map(
        valid, expected_paths=[label], owner="test"
    ) == {label: digest}
    duplicate = {
        "implementations": [
            {"path": label, "sha256": digest},
            {"path": label, "sha256": digest},
        ]
    }
    with pytest.raises(RuntimeError, match="duplicate"):
        research._implementation_receipt_map(
            duplicate, expected_paths=[label], owner="test"
        )
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        research._implementation_receipt_map(
            {"implementations": []}, expected_paths=[label], owner="test"
        )
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        research._implementation_receipt_map(
            valid,
            expected_paths=[label, "v4/model/protocol101_walking_skeleton.py"],
            owner="test",
        )
    with pytest.raises(RuntimeError, match="hash drift"):
        research._implementation_receipt_map(
            {"implementations": [{"path": label, "sha256": "0" * 64}]},
            expected_paths=[label],
            owner="test",
        )


def test_pathd_immutable_source_drift_cannot_be_receipted() -> None:
    payload, _, _ = preregistration_payload()
    tampered = research.json.loads(research.json.dumps(payload))
    immutable = tampered["source_hash_policy"][
        "immutable_and_rehashed_at_every_fit"
    ][0]
    tampered["source_hashes_at_freeze"][immutable] = "0" * 64
    with pytest.raises(RuntimeError, match="immutable source drift"):
        research._verify_immutable_sources(tampered)


def test_pathd_production_dependency_closure_rejects_external_escape_imports() -> None:
    assert research.FROZEN_OFFLINE_IMPORT_ROOTS == (
        "__future__",
        "argparse",
        "ast",
        "contextlib",
        "dataclasses",
        "datetime",
        "enum",
        "errno",
        "fcntl",
        "hashlib",
        "joblib",
        "json",
        "math",
        "numpy",
        "os",
        "pandas",
        "pathlib",
        "platform",
        "pyarrow",
        "random",
        "re",
        "scipy",
        "sklearn",
        "stat",
        "sys",
        "threadpoolctl",
        "torch",
        "types",
        "typing",
        "uuid",
        "v4",
        "xml",
        "zoneinfo",
    )
    forbidden_modules = (
        "aiohttp",
        "boto3",
        "builtins",
        "ctypes",
        "ftplib",
        "grpc",
        "http.client",
        "httpx",
        "ibapi",
        "ib_insync",
        "importlib",
        "inspect",
        "multiprocessing",
        "webbrowser",
        "paramiko",
        "pkgutil",
        "requests",
        "runpy",
        "smtplib",
        "socket",
        "subprocess",
        "urllib",
        "urllib3",
        "websockets",
    )
    for module in forbidden_modules:
        for source in (f"import {module}", f"from {module} import example"):
            with pytest.raises(
                RuntimeError, match="prohibited external dependency import"
            ):
                research._reject_dynamic_dependency_escape(
                    "v4/research/synthetic_pathd_module.py", ast.parse(source)
                )


def test_pathd_production_dependency_closure_rejects_os_process_launches() -> None:
    forbidden_sources = (
        "import os\nos.system('command')",
        "import os\nos.popen('command')",
        "import os\nos.spawnlp(0, 'command')",
        "import os\nos.execv('command', ['command'])",
        "import os as operating_system\noperating_system.execve('command', [], {})",
        "from os import system as launch\nlaunch('command')",
        "import os\nlaunch=os.system\npropagated=launch\npropagated('command')",
        "import os as operating_system\ngetter=getattr\nalias=getter\nalias(operating_system, 'system')('command')",
        "import os\nname='system'\ngetattr(os, name)('command')",
        "import os\nos.__dict__['system']('command')",
        "from os import __dict__ as namespace\nnamespace['system']('command')",
        "import os\nos.fork()",
        "from os import fork as launch\nlaunch()",
        "import os\nos.posix_spawn('/bin/true', ['/bin/true'], {})",
        "import sys as system_state\nalias=system_state\nalias.path.append('/tmp')",
        "import sys as system_state\nsystem_state.modules['socket']",
        "from sys import path as import_path\nimport_path.append('/tmp')",
        "loader=__import__\nalias=loader\nalias('socket')",
        "runner=eval\nalias=runner\nalias('1+1')",
        "namespace=globals\nalias=namespace\nalias()",
        "import os\nmutator=setattr\nalias=mutator\nalias(os, 'system', lambda: None)",
        "import pandas as pd\npd.read_csv('https://example.invalid/data.csv')",
        "import pandas as pd\nurl='s3://bucket/data.parquet'\npd.read_parquet(url)",
        "import pyarrow as pa\npa.fs.S3FileSystem()",
        "import joblib\njoblib.Parallel(n_jobs=2)([])",
        "import joblib\njoblib.delayed(lambda: 1)()",
        "from joblib import Parallel\nParallel(n_jobs=1)([])",
        "from joblib import delayed\ndelayed(lambda: 1)()",
        "from sklearn.ensemble import RandomForestRegressor\nworkers=1\nRandomForestRegressor(n_jobs=workers)",
        "from sklearn.ensemble import RandomForestRegressor\nRandomForestRegressor(n_jobs=-1)",
    )
    for source in forbidden_sources:
        with pytest.raises(RuntimeError, match="dependency escape"):
            research._reject_dynamic_dependency_escape(
                "v4/research/synthetic_pathd_module.py", ast.parse(source)
            )


def test_pathd_dependency_closure_allows_offline_imports_and_scans_extensible_tests() -> None:
    safe_source = """
import json
import os as operating_system
import sys as system_state
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import sklearn
import threadpoolctl
import torch
from xml.etree import ElementTree

path_module = operating_system.path
path_module.join('a', 'b')
getter = getattr
getter_alias = getter
getter_alias(operating_system, 'O_NOFOLLOW', 0)
getter_alias(operating_system, 'O_DIRECTORY', 0)
getter_alias(model, 'predict_proba', None)
getter_alias(dataset, name, None)
system_state.version_info
Path('artifact.json')
joblib.dump({'x': 1}, Path('artifact.joblib'))
joblib.load(Path('artifact.joblib'))
sklearn.ensemble.RandomForestRegressor(n_jobs=1)
sklearn.ensemble.RandomForestRegressor(n_jobs=None)
pd.read_parquet(Path('local-artifact.parquet'))
np.asarray([1.0])
"""
    research._reject_dynamic_dependency_escape(
        "v4/research/synthetic_pathd_module.py", ast.parse(safe_source)
    )
    safe_test_source = "from pathlib import Path\nimport pytest\nimport v4"
    research._reject_dynamic_dependency_escape(
        "v4/tests/test_pathd_fixed_science_contract.py", ast.parse(safe_test_source)
    )
    with pytest.raises(RuntimeError, match="prohibited external dependency import"):
        research._reject_dynamic_dependency_escape(
            "v4/tests/test_pathd_fixed_science_contract.py",
            ast.parse("import subprocess"),
        )


def test_pathd_test_evidence_rejects_arbitrary_success_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = research.receipt_contract_spec()["required_test_commands"]
    junit = tmp_path / "gate.junit.xml"
    junit.write_text(
        "<testsuites><testsuite>"
        + "".join(
            f'<testcase name="{name}" />'
            for name in research.FROZEN_GATE_TEST_NAMES
        )
        + "</testsuite></testsuites>",
        encoding="utf-8",
    )
    monkeypatch.setattr(research, "_canonical_repo_regular_file", lambda label: junit)
    valid = [
        {
            "argv": expected[0]["argv"],
            "cwd": expected[0]["cwd"],
            "env": expected[0]["env"],
            "exit_code": 0,
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "0" * 64,
            "junit_path": expected[0]["junit_path"],
            "junit_sha256": research.sha256_path(junit),
            "junit_summary": expected[0]["expected_junit_summary"],
        }
    ]
    research._validate_registered_test_commands(valid, expected[:1])
    forged = [
        {
            **valid[0],
            "argv": ["true"],
        }
    ]
    with pytest.raises(RuntimeError, match="unregistered machinery test command"):
        research._validate_registered_test_commands(forged, expected[:1])


def test_pathd_semantic_validator_rejects_role_or_receipt_contract_drift() -> None:
    payload, assignments, lineage = preregistration_payload()
    role_drift = research.json.loads(research.json.dumps(payload))
    role_drift["source_hash_policy"]["fit_session_roles"][
        "outer_weights"
    ] = "caller.sessions"
    with pytest.raises(ValueError, match="fit-session role"):
        validate_preregistration_payload(role_drift, assignments, lineage)
    command_drift = research.json.loads(research.json.dumps(payload))
    command_drift["source_hash_policy"]["receipt_contracts"][
        "required_test_commands"
    ][0]["argv"] = ["true"]
    with pytest.raises(ValueError, match="receipt/evidence contract"):
        validate_preregistration_payload(command_drift, assignments, lineage)


def test_pathd_result_envelope_fails_closed() -> None:
    payload, _, _ = preregistration_payload()
    prereg_sha = "1" * 64
    valid = {
        "quarantine_labels": list(research.QUARANTINE_LABELS),
        "claim_boundary": research.CLAIM_BOUNDARY,
        "holdout_caveat": research.HOLDOUT_CAVEAT,
        "plan_sha256": payload["binding_plan"]["sha256"],
        "preregistration_sha256": prereg_sha,
        "fill_law_hash": payload["fill_law"]["fill_law_hash"],
        "holdout_open_count": 0,
    }
    research._validate_result_envelope_against(
        valid,
        prereg_payload=payload,
        preregistration_sha256=prereg_sha,
        expected_holdout_open_count=0,
    )
    for name in payload["result_envelope"]["required_fields_every_result"]:
        broken = dict(valid)
        broken.pop(name)
        with pytest.raises(ValueError, match="invalid_result"):
            research._validate_result_envelope_against(
                broken,
                prereg_payload=payload,
                preregistration_sha256=prereg_sha,
                expected_holdout_open_count=0,
            )


def test_pathd_stale_or_altered_fit_authorization_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = research.FrozenFitAuthorization(
        role="outer_weights",
        outer_fold=1,
        inner_fold=None,
        sessions=("2025-08-01",),
        sessions_sha256_newline=research.canonical_session_hash(["2025-08-01"]),
        preregistration_sha256="1" * 64,
        session_assignments_sha256="2" * 64,
        source_hash_policy_sha256="3" * 64,
        corpus_integrity_receipt_sha256="4" * 64,
        lineage_receipt_sha256="5" * 64,
        machinery_receipt_sha256="6" * 64,
        fit_environment_sha256="7" * 64,
        entry_pooled_acceptance_receipt_sha256=None,
    )
    monkeypatch.setattr(research, "assert_entry_fit_ready", lambda **_: base)
    assert research.assert_fit_authorization_current(base) == base
    altered = replace(base, machinery_receipt_sha256="7" * 64)
    with pytest.raises(RuntimeError, match="stale or altered"):
        research.assert_fit_authorization_current(altered)


def test_pathd_owner_locked_operational_contract_tamper_matrix() -> None:
    payload, assignments, lineage = preregistration_payload()
    assert payload["owner_authorization"] == research.validated_owner_authorization()
    assert payload["owner_authorization"]["execution_authorization"]["authorized"] is True

    def owner_receipt(row: dict[str, object]) -> None:
        row["owner_authorization"]["execution_authorization"]["authorized"] = False

    def feature_count(row: dict[str, object]) -> None:
        row["entry"]["feature_count"] = 16

    def representation(row: dict[str, object]) -> None:
        row["entry"]["representation"]["history_minutes"] = 89

    def entry_family(row: dict[str, object]) -> None:
        row["entry"]["global_family_freeze"]["selectable_candidate"] = "NEURAL"

    def exit_horizon(row: dict[str, object]) -> None:
        row["exit"]["local_horizon_seconds"] = 299

    def exit_penalty(row: dict[str, object]) -> None:
        row["exit"]["composer"]["downside_term"]["penalty"] = 0.5

    def exit_isolation(row: dict[str, object]) -> None:
        row["exit"]["diagnostic_target_isolation"]["diagnostic_family_count"] = 5

    def exit_family(row: dict[str, object]) -> None:
        row["exit"]["family_evaluation"]["selectable_candidates"] = ["HGB", "NEURAL"]

    def replay_fee(row: dict[str, object]) -> None:
        row["economics"]["serial_account_contract"]["floor_fee_constant"] = "$4"

    def floor_self_rehashed(row: dict[str, object]) -> None:
        floor = row["floor_law"]
        floor["formula"] = "floor_bid=max(0,0.40*P_entry+3/100)"
        semantic = dict(floor)
        semantic.pop("floor_law_hash")
        floor["floor_law_hash"] = research.stable_hash(semantic)

    def diagnostics(row: dict[str, object]) -> None:
        row["context_diagnostics"]["alpha_or_gate"] = True

    for mutate in (
        owner_receipt,
        feature_count,
        representation,
        entry_family,
        exit_horizon,
        exit_penalty,
        exit_isolation,
        exit_family,
        replay_fee,
        floor_self_rehashed,
        diagnostics,
    ):
        tampered = copy.deepcopy(payload)
        mutate(tampered)
        with pytest.raises(ValueError):
            validate_preregistration_payload(tampered, assignments, lineage)

    tampered_assignments = copy.deepcopy(assignments)
    tampered_assignments["protected_holdout_30"][0] = assignments[
        "pre_holdout_session_indices_1_215"
    ][0]
    tampered_payload = copy.deepcopy(payload)
    tampered_payload["sessions_hash"] = research.stable_hash(tampered_assignments)
    with pytest.raises(ValueError, match="session assignment reconstruction"):
        validate_preregistration_payload(tampered_payload, tampered_assignments, lineage)

    tampered_lineage = copy.deepcopy(lineage)
    tampered_lineage["features"][0]["feature"] = "forged"
    tampered_payload = copy.deepcopy(payload)
    tampered_payload["feature_lineage_hash"] = research.stable_hash(tampered_lineage)
    with pytest.raises(ValueError, match="feature lineage reconstruction"):
        validate_preregistration_payload(tampered_payload, assignments, tampered_lineage)


def test_pathd_vix_es_vx_diagnostic_contract_is_closed_and_nonalpha() -> None:
    spec = research.context_diagnostics_spec()
    assert spec["role"] == "diagnostics_and_strata_only"
    assert spec["alpha_or_gate"] is False
    authority = spec["execution_authority"]
    assert authority["current_preregistration"] == (
        "AUTHORIZED_POST_ENTRY_PRIMARY_READ_ONLY"
    )
    assert "604-file diagnostic inventory receipt" in authority["blocking_precondition"]
    assert authority["authorization_type"] == "FrozenContextDiagnosticsAuthorizationV1"
    assert authority["four_box_and_protected_holdout"].startswith(
        "UNAUTHORIZED_AND_UNREACHABLE"
    )
    artifact_contract = spec["artifact_contract"]
    assert len(artifact_contract["outer_artifact_paths"]) == 5
    assert len(artifact_contract["outer_receipt_paths"]) == 5
    assert artifact_contract["payload_spec"] == research.context_diagnostic_payload_spec()
    assert artifact_contract["forbidden_caller_inputs"]
    assert set(spec["sources"]) == {"VIX", "ES", "VX", "SPX_REFERENCE"}
    assert artifact_contract["payload_spec"]["source_order"] == [
        "VIX", "ES", "VX", "SPX_REFERENCE"
    ]
    assert artifact_contract["access_receipt_path"].endswith(
        "entry_context_diagnostics_access_receipt.json"
    )
    assert artifact_contract["transaction"]["crash_rule"].startswith(
        "an access receipt without the pooled terminal receipt"
    )
    assert spec["sources"]["VIX"]["clock"] == {
        "kind": "HISTORICAL_BAR_END_CLOCK_ONLY",
        "timestamp_semantics": "BAR_OPEN_TIMESTAMP",
        "available_at": "event_time+60 seconds",
        "eligible": "latest available_at<=decision_time and age_seconds from available_at in [0,90]",
        "live_parity_claim": False,
    }
    assert spec["sources"]["VIX"]["row_invariants"]["symbol"] == "VIX"
    assert spec["sources"]["VIX"]["row_invariants"]["is_proxy"] is False
    assert spec["sources"]["SPX_REFERENCE"]["integrity_partition"] == (
        research.CORE_INTEGRITY_PARTITION
    )
    assert spec["sources"]["SPX_REFERENCE"]["clock"]["available_at"] == (
        "event_time+60 seconds"
    )
    assert spec["sources"]["SPX_REFERENCE"]["row_invariants"]["symbol"] == "SPX"
    assert spec["sources"]["VX"]["availability_boundary_session"] == "2026-04-01"
    assert spec["sources"]["VX"]["pre_boundary_status"] == (
        "NOT_REQUESTED_PRE_VX_BOUNDARY"
    )
    assert all(
        source["clock"]["live_parity_claim"] is False
        for source in spec["sources"].values()
    )
    assert spec["publisher_consolidation"]["within_publisher_duplicate"] == (
        "invalid:DUPLICATE_WITHIN_PUBLISHER"
    )
    assert spec["anchors"]["joint_or_multidimensional_strata"] is False
    assert spec["anchors"]["anchor_kinds_in_order"] == [
        "SESSION_1000", "ENTRY_DECISION"
    ]
    assert len(spec["anchors"]["policy_contract"]["policy_ids_in_order"]) == 11
    assert spec["source_status_vocabulary"] == [
        "FRESH", *spec["missing_status_priority"]
    ]
    bins = spec["bins"]
    assert bins["status"] == spec["source_status_vocabulary"]
    assert len(bins["vix_15m_point_change"]) == len(set(bins["vix_15m_point_change"])) == 5
    assert len(bins["edges_float64_hex"]["vix_15m_point_change"]) == 6
    assert research.context_age_quantile_type7([0, 1_000_000_000], 0.5) == 0.5
    assert research.context_age_quantile_type7([0, 1_000_000_000], 0.95) == 0.95
    assert "model feature" in spec["forbidden_consumers"]
    assert "gate" in spec["forbidden_consumers"]

    payload, assignments, lineage = preregistration_payload()
    tampered = copy.deepcopy(payload)
    tampered["context_diagnostics"]["sources"]["ES"]["clock"][
        "live_parity_claim"
    ] = True
    with pytest.raises(ValueError, match="diagnostics operational binding"):
        validate_preregistration_payload(tampered, assignments, lineage)


def test_pathd_every_postfreeze_result_carries_exact_holdout_caveat() -> None:
    contract = research.entry_future_api_contract()
    classification = research.result_artifact_classification()
    research.validate_result_artifact_classification(
        {"result_envelope": {"artifact_classification": classification}}
    )
    classes = classification["classes"]
    classified = [
        name
        for class_names in classes.values()
        for name in class_names
    ]
    assert len(classified) == len(set(classified))
    assert set(classified) == set(contract["schema_versions"])
    for schema_name in classes["standalone_scientific_result"]:
        assert "holdout_caveat" in contract["dataclass_fields"][schema_name]
    assert set(classes["embedded_scientific_result"]) == {
        "EntryOuterPrimaryResultV1",
        "ProtectedHoldoutEvaluationV1",
    }
    assert classification["fixed_named_result_paths"]


def test_pathd_immutable_pooled_gate_rejects_runner_approved_forged_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v4.scripts import run_pathd_entry_exit_research as runner

    spec = research.entry_pooled_gate_input_spec()
    fold_keys = spec["closed_shapes"]["per_fold"]["row_keys_in_order"]
    per_fold = []
    for fold in range(1, 6):
        row = {key: 10 for key in fold_keys}
        row.update(
            outer_fold=fold,
            primary_session_count=25,
            shared_exit_candidate_completed_trades=40,
            time300_oof_trajectory_count=60,
        )
        per_fold.append(row)
    pooled = {
        key: sum(row[key] for row in per_fold)
        for key in spec["closed_shapes"]["pooled"]["keys_in_order"]
    }
    action_row = {
        "distinct_trajectory_count": 100,
        "coverage_numerator": 90,
        "coverage_denominator": 100,
        "decile_trajectory_counts": [10] * 10,
        "decile_distinct_session_counts": [3] * 10,
        "adjacent_valid_bootstrap_replicates": [5_000] * 9,
        "adjacent_total_draws": [5_000] * 9,
        "adjacent_upper_bounds_micros": [-1.0] * 9,
    }
    random_spec = spec["closed_shapes"]["matched_random"]
    random_shape = (
        len(random_spec["policy_ids"]), 5, 8, 4
    )
    random_complete = [
        [[[True for _ in range(random_shape[3])] for _ in range(random_shape[2])] for _ in range(random_shape[1])]
        for _ in range(random_shape[0])
    ]
    random_hashes = [
        [
            [
                [
                    research.stable_hash([policy, fold, seed, channel])
                    for channel in range(4)
                ]
                for seed in range(8)
            ]
            for fold in range(5)
        ]
        for policy in range(random_shape[0])
    ]
    negative_complete = [
        [[True for _ in range(4)] for _ in range(5)] for _ in range(22)
    ]
    negative_hashes = [
        [
            [research.stable_hash(["negative", control, fold, channel]) for channel in range(4)]
            for fold in range(5)
        ]
        for control in range(22)
    ]
    payload, assignments, _lineage = research.preregistration_payload()
    required_checks = payload["metrics_and_gates"]["large_improvement_audit"][
        "required_checks"
    ]
    forged_inputs = {
        "schema_version": spec["schema_version"],
        "fold_order": [1, 2, 3, 4, 5],
        "per_fold": per_fold,
        "pooled": pooled,
        "action_calibration": {"ENTER": dict(action_row), "WAIT": dict(action_row)},
        "matched_random": {
            "policy_ids": random_spec["policy_ids"],
            "seeds": random_spec["seeds"],
            "channels": random_spec["channels"],
            "complete_by_policy_fold_seed_channel": random_complete,
            "evaluation_receipt_sha256s_by_policy_fold_seed_channel": random_hashes,
        },
        "negative_controls": {
            "control_ids": spec["closed_shapes"]["negative_controls"]["control_ids"],
            "channels": spec["closed_shapes"]["negative_controls"]["channels"],
            "complete_by_control_fold_channel": negative_complete,
            "full_gate_passed_by_control": [False] * 22,
            "evaluation_sha256s_by_control_fold_channel": negative_hashes,
        },
        "large_improvement_audit": {
            "ratio_trigger": False,
            "bootstrap_trigger": False,
            "triggered": False,
            "required_check_names": required_checks,
            "required_check_statuses": ["NOT_TRIGGERED"] * len(required_checks),
            "required_check_receipt_sha256s": [None] * len(required_checks),
            "complete": True,
        },
        "survival_violation_counts": {
            key: 0
            for key in spec["closed_shapes"]["survival_violation_counts"][
                "keys_in_order"
            ]
        },
    }
    _inputs, criteria, verdict, stop, full = research._validate_entry_pooled_gate_inputs(
        forged_inputs
    )
    assert verdict == "PASS" and stop is None and full is True
    assert all(criteria.values())

    genuine_inputs = copy.deepcopy(forged_inputs)
    genuine_inputs["per_fold"][0]["primary_session_count"] = 0
    genuine_inputs["pooled"]["primary_session_count"] = sum(
        row["primary_session_count"] for row in genuine_inputs["per_fold"]
    )
    authorization = research.FrozenResultAggregationAuthorization(
        role="pooled_outer_primary",
        sessions=("2026-01-02",),
        sessions_sha256_newline=research.canonical_session_hash(["2026-01-02"]),
        preregistration_sha256="1" * 64,
        session_assignments_sha256="2" * 64,
        source_hash_policy_sha256="3" * 64,
        machinery_receipt_sha256="4" * 64,
        outer_result_receipts_sha256=tuple(str(index) * 64 for index in range(1, 6)),
    )
    monkeypatch.setattr(
        runner,
        "validate_entry_pooled_acceptance_result",
        lambda *_args, **_kwargs: {"verdict": "PASS", "full_fit_authorized": True},
        raising=False,
    )
    monkeypatch.setattr(
        research, "assert_entry_result_aggregation_ready", lambda **_kwargs: authorization
    )
    monkeypatch.setattr(
        research, "_validate_outer_result_receipt_one", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        research,
        "_reconstruct_entry_pooled_gate_inputs_from_outer_artifacts",
        lambda **_kwargs: genuine_inputs,
    )

    def fake_read_json(path: Path) -> dict[str, object]:
        path = Path(path)
        if path == research.PREREG_PATH:
            return payload
        if path == research.SESSION_PATH:
            return assignments
        if path.name == "outer_primary_result.json":
            fold = int(path.parent.name.removeprefix("fold_"))
            return {"payload": {"hgb": {"result_sha256": str(fold) * 64}}}
        raise AssertionError(f"unexpected read: {path}")

    monkeypatch.setattr(research, "read_json", fake_read_json)
    monkeypatch.setattr(
        research,
        "sha256_path",
        lambda path: research.stable_hash(str(Path(path))),
    )
    forged_result = {
        "schema_version": "pathd.entry_pooled_acceptance_result.v1",
        "holdout_caveat": research.HOLDOUT_CAVEAT,
        "aggregation_authorization_sha256": research.stable_hash(authorization.to_dict()),
        "outer_result_receipts_sha256": research.stable_hash(
            list(authorization.outer_result_receipts_sha256)
        ),
        "sessions": list(authorization.sessions),
        "sessions_sha256_newline": authorization.sessions_sha256_newline,
        "candidate_family": "HGB",
        "candidate_outer_evaluation_sha256s": [str(fold) * 64 for fold in range(1, 6)],
        "control_exit_sha256s": [
            research.sha256_path(research._outer_fold_artifact_path(fold, "control_exit.json"))
            for fold in range(1, 6)
        ],
        "control_replay_result_sha256s": [
            research.sha256_path(research._outer_fold_artifact_path(fold, "control_replay_result.json"))
            for fold in range(1, 6)
        ],
        "negative_control_panel_sha256s": [
            research.sha256_path(research._outer_fold_artifact_path(fold, "negative_control_panel.json"))
            for fold in range(1, 6)
        ],
        "gate_spec_sha256": research._entry_pooled_gate_spec_sha256(payload),
        "reconstructed_gate_inputs": forged_inputs,
        "gate_inputs_sha256": research.stable_hash(forged_inputs),
        "pass_criteria_recomputed": criteria,
        "verdict": "PASS",
        "stop_reason": None,
        "full_fit_authorized": True,
        "result_sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="do not reconstruct"):
        research.validate_entry_pooled_acceptance_result(
            forged_result, authorization=authorization
        )
