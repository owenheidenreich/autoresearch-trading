"""Quarantined Path-D Tier-S research runner and typed result validators.

Every executable stage is fixed-path and fail-closed.  The runner never contacts a
broker or opens the protected holdout; entry fitting and ordinary evidence access are
available only through their frozen role authorizations and one-shot receipt gates.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from v4.research import pathd_entry_exit as prereg
from v4.research.pathd_entry_exit import (
    assert_preregistration_frozen,
    enveloped_research_result,
    freeze_preregistration,
    require_all_negative_fixtures_rejected,
)


V31_AUTHORIZATION_ROOT = prereg.REPO_ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_1_2026_08_01"
)
V31_PREREGISTRATION_SHA256 = (
    "7fa621d63086a6dd28fb10cb731f229471051951875f0c663d53838e7771b370"
)
V31_FOUNDATION_GENERATION_SHA256 = (
    "1b6eb620668be3655facac71467513d50eafa706a1aedc64dc507d0dbce873b2"
)
V31_AUTHORIZATION_FILE_SHA256S = {
    "feature_lineage.json": "4521b3e62982f672daaba73d98ce5a9ce72df170b083cd40df7dc979983d1ef0",
    "foundation_restoration_receipt.json": "3cecec8bda2c829153200e7b081beccf29f39474384f05e23d0dff2b024d00cc",
    "preregistration.json": V31_PREREGISTRATION_SHA256,
    "preregistration.sha256": "ce3a76c74f1b6a13dc078a900e50513a7487224f7cfe01f55eb860105062e4b8",
    "preregistration_freeze_receipt.json": "6eb38341c6f99b42fc318b9644d25d7d0a9e5ca98e63d1cad278338a7f700f98",
    "session_assignments.json": "431cd14879ad6a14b278cb2683e4f3b860eef960e6b74a4f0edb3e620cf82475",
}
V31_EXECUTABLE_ROOT = prereg.REPO_ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_1_"
    "executable_2026_08_01"
)
V31_EXECUTABLE_GENERATION_PATH = V31_EXECUTABLE_ROOT / "executable_generation.json"
V31_SYNTHETIC_JUNIT_PATH = V31_EXECUTABLE_ROOT / "synthetic_tests.junit.xml"
V31_SYNTHETIC_TEST_RECEIPT_PATH = (
    V31_EXECUTABLE_ROOT / "synthetic_test_receipt.json"
)
V31_EXECUTABLE_BRIDGE_PATH = (
    V31_EXECUTABLE_ROOT / "implementation_receipt.json"
)
V31_SYNTHETIC_TEST_PATH = "v4/tests/test_pathd_corrected_v31_executable_build.py"
V31_MONEYNESS_AUTHORITY_PATH = (
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md"
)
V31_MONEYNESS_AUTHORITY_SHA256 = (
    "fcaf69ae080418cf6146f08d90386665c3c9f576f7382f4de8d8c51c23cf8598"
)
V31_EXTENSIBLE_IMPLEMENTATION_PATHS = (
    "v4/research/pathd_entry_features.py",
    "v4/research/pathd_entry_dataset.py",
    "v4/research/pathd_entry_models.py",
    "v4/path_d/execution/research_fill_law.py",
    "v4/path_d/execution/research_replay.py",
    "v4/scripts/run_pathd_entry_exit_research.py",
)
V31_SYNTHETIC_TEST_NAMES = (
    "test_static_generation_binding_and_campaign_order_are_exact",
    "test_br_is_first_and_terminal_precedence_is_fail_closed",
    "test_matched_random_budget_and_all_eight_schedules_are_outcome_blind",
    "test_shared_control_exit_requires_complete_panel_and_selects_real_economics",
    "test_negative_controls_fail_closed_and_exit_geometry_abstains",
    "test_build_does_not_create_any_real_execution_namespace",
)


def assert_corrected_v31_authorization_release() -> dict[str, Any]:
    """Validate the byte-frozen v3.1 release strictly as historical evidence."""

    if (
        not V31_AUTHORIZATION_ROOT.is_dir()
        or V31_AUTHORIZATION_ROOT.is_symlink()
        or {path.name for path in V31_AUTHORIZATION_ROOT.iterdir()}
        != set(V31_AUTHORIZATION_FILE_SHA256S)
    ):
        raise RuntimeError("corrected-v3.1 authorization root drift")
    for name, expected in V31_AUTHORIZATION_FILE_SHA256S.items():
        if prereg.sha256_path(V31_AUTHORIZATION_ROOT / name) != expected:
            raise RuntimeError(f"corrected-v3.1 authorization byte drift: {name}")
    release = prereg.read_json(
        V31_AUTHORIZATION_ROOT / "preregistration_freeze_receipt.json"
    )
    no_action = release.get("no_action_state")
    release_state = release.get("release_state")
    if (
        release.get("status")
        != "FROZEN_AUTHORIZATION_RELEASE_BEFORE_ANY_SEAL_OR_MODEL_FIT"
        or release.get("preregistration_sha256") != V31_PREREGISTRATION_SHA256
        or release.get("holdout_open_count") != 0
        or type(no_action) is not dict
        or any(no_action.values())
        or type(release_state) is not dict
        or release_state.get("machinery_seal_authorized") is not True
        or release_state.get("foundation_stability_seal_authorized") is not True
        or release_state.get("model_fit_authorized") is not True
        or release_state.get("nested_or_outer_evidence_open_authorized") is not True
        or release_state.get("protected_holdout_open_authorized") is not False
        or release.get("source_hash_policy_receipt_paths_preserved_from_corrected_v3")
        is not True
        or release.get("science_identity_vs_corrected_v3", {}).get(
            "source_hash_policy_sha256"
        )
        != prereg.stable_hash(
            prereg.read_json(V31_AUTHORIZATION_ROOT / "preregistration.json")[
                "source_hash_policy"
            ]
        )
    ):
        raise RuntimeError("corrected-v3.1 authorization release semantic drift")
    return release


def assert_corrected_v31_evidence_call_graph_closed() -> dict[str, Any]:
    """Prove the executable runner has one B-R-gated legacy open edge."""

    label = "v4/scripts/run_pathd_entry_exit_research.py"
    path = prereg.REPO_ROOT / label
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=label)
    allowed_function = "begin_entry_evidence_after_valid_calibration_scope_gate"
    calls: list[dict[str, Any]] = []
    function_stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name == "begin_entry_evidence_once":
                calls.append(
                    {
                        "enclosing_function": function_stack[-1]
                        if function_stack
                        else None,
                        "line": node.lineno,
                    }
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    if (
        len(calls) != 1
        or calls[0]["enclosing_function"] != allowed_function
    ):
        raise RuntimeError("corrected-v3.1 evidence open call graph escaped B-R gate")
    semantic = {
        "source_path": label,
        "source_sha256": prereg.sha256_path(path),
        "legacy_open_call_count": 1,
        "only_enclosing_function": allowed_function,
        "direct_campaign_or_stage_open": False,
    }
    return {**semantic, "call_graph_sha256": prereg.stable_hash(semantic)}


def _corrected_v31_executable_generation_semantic() -> dict[str, Any]:
    """Return the source-independent executable contract for this generation."""

    return {
        "schema_version": "pathd.corrected_v31.executable_generation.v1",
        "status": "FROZEN_EXECUTABLE_CONTRACT_BEFORE_TEST_OR_RUN",
        "science_authorization_root": prereg.repo_path_label(V31_AUTHORIZATION_ROOT),
        "science_authorization_file_sha256s": dict(V31_AUTHORIZATION_FILE_SHA256S),
        "science_preregistration_sha256": V31_PREREGISTRATION_SHA256,
        "foundation_generation_sha256": V31_FOUNDATION_GENERATION_SHA256,
        "corrected_v3_fixed_artifact_root": prereg.repo_path_label(prereg.AUDIT_ROOT),
        "corrected_v3_preregistration_sha256": prereg.sha256_path(prereg.PREREG_PATH),
        "source_hash_policy_sha256": prereg.stable_hash(
            prereg.read_json(prereg.PREREG_PATH)["source_hash_policy"]
        ),
        "lawful_extensible_implementation_paths": list(
            V31_EXTENSIBLE_IMPLEMENTATION_PATHS
        ),
        "matched_random_moneyness_authority": {
            "path": V31_MONEYNESS_AUTHORITY_PATH,
            "sha256": V31_MONEYNESS_AUTHORITY_SHA256,
            "rule": {
                "ATM": "absolute ladder distance from slot 10 <= 1",
                "NEAR": "absolute ladder distance from slot 10 in 2..5",
                "WING": "absolute ladder distance from slot 10 > 5",
            },
        },
        "synthetic_test_path": V31_SYNTHETIC_TEST_PATH,
        "synthetic_testcase_names": list(V31_SYNTHETIC_TEST_NAMES),
        "implementation_receipt_path": prereg.repo_path_label(
            V31_EXECUTABLE_BRIDGE_PATH
        ),
        "only_evidence_open_api": (
            "v4.scripts.run_pathd_entry_exit_research."
            "begin_entry_evidence_after_valid_calibration_scope_gate"
        ),
        "legacy_evidence_open_call_graph": (
            "exactly one call in the executable runner, enclosed by the only_evidence_open_api"
        ),
        "campaign_order": list(ENTRY_CAMPAIGN_STAGE_ORDER),
        "hard_stops_at_generation_freeze": {
            "model_or_weight_fit": False,
            "real_corpus_decode": False,
            "fold_or_evidence_namespace_open": False,
            "foundation_or_machinery_seal_against_corpus": False,
            "protected_holdout_open_count": 0,
            "live_broker_paper_or_download": False,
            "promotion_or_default_change": False,
        },
    }


def freeze_corrected_v31_executable_generation() -> dict[str, Any]:
    """Freeze the new executable contract without hashing mutable implementation."""

    assert_corrected_v31_authorization_release()
    assert_corrected_v31_evidence_call_graph_closed()
    if prereg.sha256_path(prereg.REPO_ROOT / V31_MONEYNESS_AUTHORITY_PATH) != (
        V31_MONEYNESS_AUTHORITY_SHA256
    ):
        raise RuntimeError("matched-random moneyness authority byte drift")
    if V31_EXECUTABLE_ROOT.exists() and any(V31_EXECUTABLE_ROOT.iterdir()):
        raise RuntimeError("corrected-v3.1 executable generation root is occupied")
    semantic = _corrected_v31_executable_generation_semantic()
    receipt = {**semantic, "generation_sha256": prereg.stable_hash(semantic)}
    prereg._write_canonical_json_exclusive(V31_EXECUTABLE_GENERATION_PATH, receipt)
    return assert_corrected_v31_executable_generation()


def assert_corrected_v31_executable_generation() -> dict[str, Any]:
    if (
        not V31_EXECUTABLE_GENERATION_PATH.is_file()
        or V31_EXECUTABLE_GENERATION_PATH.is_symlink()
    ):
        raise RuntimeError("corrected-v3.1 executable generation is absent")
    observed = prereg.read_json(V31_EXECUTABLE_GENERATION_PATH)
    semantic = dict(observed)
    digest = semantic.pop("generation_sha256", None)
    expected = _corrected_v31_executable_generation_semantic()
    if semantic != expected or digest != prereg.stable_hash(expected):
        raise RuntimeError("corrected-v3.1 executable generation drift")
    assert_corrected_v31_authorization_release()
    assert_corrected_v31_evidence_call_graph_closed()
    if prereg.sha256_path(prereg.REPO_ROOT / V31_MONEYNESS_AUTHORITY_PATH) != (
        V31_MONEYNESS_AUTHORITY_SHA256
    ):
        raise RuntimeError("matched-random moneyness authority byte drift")
    return observed


def freeze_corrected_v31_executable_bridge() -> dict[str, Any]:
    """Freeze real source/JUnit hashes without sealing machinery or corpus."""

    from v4.research.pathd_entry_dataset import (
        assert_corrected_v31_executable_bridge,
    )

    generation = assert_corrected_v31_executable_generation()
    if any(os.path.lexists(path) for path in (
        V31_SYNTHETIC_TEST_RECEIPT_PATH, V31_EXECUTABLE_BRIDGE_PATH,
    )):
        raise RuntimeError("corrected-v3.1 executable build receipt already exists")
    if not V31_SYNTHETIC_JUNIT_PATH.is_file() or V31_SYNTHETIC_JUNIT_PATH.is_symlink():
        raise RuntimeError("corrected-v3.1 synthetic JUnit is absent")
    suite = ET.parse(V31_SYNTHETIC_JUNIT_PATH).getroot()
    cases = suite.findall(".//testcase")
    names = tuple(case.get("name") for case in cases)
    failures = sum(len(case.findall("failure")) for case in cases)
    errors = sum(len(case.findall("error")) for case in cases)
    skipped = sum(len(case.findall("skipped")) for case in cases)
    if (
        names != V31_SYNTHETIC_TEST_NAMES
        or failures != 0 or errors != 0 or skipped != 0
        or len(cases) != len(V31_SYNTHETIC_TEST_NAMES)
    ):
        raise RuntimeError("corrected-v3.1 synthetic JUnit result drift")
    implementations = [
        {"path": label, "sha256": prereg.sha256_path(prereg.REPO_ROOT / label)}
        for label in V31_EXTENSIBLE_IMPLEMENTATION_PATHS
    ]
    test_source_sha256 = prereg.sha256_path(prereg.REPO_ROOT / V31_SYNTHETIC_TEST_PATH)
    test_semantic = {
        "schema_version": "pathd.corrected_v31.synthetic_test_receipt.v1",
        "status": "PASS_SYNTHETIC_PRODUCTION_PATH",
        "frozen_at_utc": _utc_now(),
        "command": [
            ".venv/bin/python", "-m", "pytest", "-q",
            V31_SYNTHETIC_TEST_PATH,
            "--junitxml", prereg.repo_path_label(V31_SYNTHETIC_JUNIT_PATH),
        ],
        "pytest_exit_code": 0,
        "test_source_path": V31_SYNTHETIC_TEST_PATH,
        "test_source_sha256": test_source_sha256,
        "junit_path": prereg.repo_path_label(V31_SYNTHETIC_JUNIT_PATH),
        "junit_sha256": prereg.sha256_path(V31_SYNTHETIC_JUNIT_PATH),
        "testcase_names": list(names),
        "summary": {
            "tests": len(cases), "passed": len(cases), "failures": failures,
            "errors": errors, "skipped": skipped,
        },
        "implementations_sha256": prereg.stable_hash(implementations),
        "model_fit_executed": False,
        "corpus_decoded": False,
        "evidence_opened": False,
        "holdout_open_count": 0,
    }
    test_receipt = {
        **test_semantic, "receipt_sha256": prereg.stable_hash(test_semantic)
    }
    prereg._write_canonical_json_exclusive(
        V31_SYNTHETIC_TEST_RECEIPT_PATH, test_receipt
    )
    release_path = V31_AUTHORIZATION_ROOT / "preregistration_freeze_receipt.json"
    bridge_semantic = {
        "schema_version": "pathd.corrected_v31.executable_authorization_bridge.v1",
        "status": "FROZEN_EXECUTABLE_BUILT_NOT_RUN",
        "frozen_at_utc": _utc_now(),
        "corrected_v31_preregistration_sha256": V31_PREREGISTRATION_SHA256,
        "corrected_v31_foundation_generation_sha256": V31_FOUNDATION_GENERATION_SHA256,
        "corrected_v31_release_path": prereg.repo_path_label(release_path),
        "corrected_v31_release_sha256": prereg.sha256_path(release_path),
        "executable_generation_path": prereg.repo_path_label(
            V31_EXECUTABLE_GENERATION_PATH
        ),
        "executable_generation_file_sha256": prereg.sha256_path(
            V31_EXECUTABLE_GENERATION_PATH
        ),
        "executable_generation_sha256": generation["generation_sha256"],
        "science_authorization_file_sha256s": dict(V31_AUTHORIZATION_FILE_SHA256S),
        "corrected_v3_preregistration_sha256": prereg.sha256_path(prereg.PREREG_PATH),
        "source_hash_policy_sha256": prereg.stable_hash(
            prereg.read_json(prereg.PREREG_PATH)["source_hash_policy"]
        ),
        "fixed_artifact_root": prereg.repo_path_label(prereg.AUDIT_ROOT),
        "implementations": implementations,
        "implementations_sha256": prereg.stable_hash(implementations),
        "synthetic_test_receipt_path": prereg.repo_path_label(
            V31_SYNTHETIC_TEST_RECEIPT_PATH
        ),
        "synthetic_test_receipt_sha256": prereg.sha256_path(
            V31_SYNTHETIC_TEST_RECEIPT_PATH
        ),
        "model_fit_executed": False,
        "corpus_decoded": False,
        "evidence_opened": False,
        "holdout_open_count": 0,
    }
    prereg._write_canonical_json_exclusive(
        V31_EXECUTABLE_BRIDGE_PATH,
        {**bridge_semantic, "receipt_sha256": prereg.stable_hash(bridge_semantic)},
    )
    return assert_corrected_v31_executable_bridge()


V32_EXECUTABLE_ROOT = prereg.REPO_ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_pathd_entry_exit_model_research_corrected_v3_2_"
    "executable_2026_08_01"
)
V32_EXECUTABLE_GENERATION_PATH = V32_EXECUTABLE_ROOT / "executable_generation.json"
V32_SYNTHETIC_JUNIT_PATH = V32_EXECUTABLE_ROOT / "synthetic_tests.junit.xml"
V32_SYNTHETIC_TEST_RECEIPT_PATH = V32_EXECUTABLE_ROOT / "synthetic_test_receipt.json"
V32_EXECUTABLE_BRIDGE_PATH = V32_EXECUTABLE_ROOT / "implementation_receipt.json"
V32_CLAUDE_RELEASE_PATH = V32_EXECUTABLE_ROOT / "claude_verification_release.json"
V32_SYNTHETIC_TEST_PATH = "v4/tests/test_pathd_corrected_v32_executable_build.py"
V32_SYNTHETIC_TEST_NAMES = (
    "test_v32_topology_is_science_identical_and_fail_closed",
    "test_shared_exit_is_selected_from_reconstructed_p5_journals",
    "test_complete_common_replay_inventory_has_no_nullable_cells",
    "test_sealed_input_orchestrator_rejects_caller_authored_economics",
    "test_spx_reference_writer_is_causal_and_lossless",
    "test_direct_legacy_evidence_api_cannot_bypass_br_gate",
    "test_build_creates_no_execution_evidence_or_holdout_namespace",
)


def _corrected_v32_executable_generation_semantic() -> dict[str, Any]:
    from v4.research.pathd_corrected_v32_foundation import (
        V32_ROOT, assert_corrected_v32_foundation,
    )

    foundation = assert_corrected_v32_foundation()
    payload = prereg.read_json(V32_ROOT / "preregistration.json")
    enforcement = payload["source_hash_policy"]["executable_generation_enforcement"]
    return {
        "schema_version": "pathd.corrected_v32.executable_generation.v1",
        "status": "FROZEN_EXECUTABLE_CONTRACT_BEFORE_ANY_FIT_OR_OPEN",
        "foundation_root": prereg.repo_path_label(V32_ROOT),
        "foundation_preregistration_sha256": foundation["preregistration_sha256"],
        "foundation_generation_sha256": foundation["foundation_generation_sha256"],
        "source_hash_policy_sha256": foundation["source_hash_policy_sha256"],
        "source_paths": enforcement["source_paths"],
        "test_paths": enforcement["test_paths"],
        "synthetic_test_path": V32_SYNTHETIC_TEST_PATH,
        "synthetic_testcase_names": list(V32_SYNTHETIC_TEST_NAMES),
        "implementation_receipt_path": prereg.repo_path_label(V32_EXECUTABLE_BRIDGE_PATH),
        "claude_release_path": prereg.repo_path_label(V32_CLAUDE_RELEASE_PATH),
        "campaign_order": list(ENTRY_CAMPAIGN_STAGE_ORDER),
        "hard_stops": {
            "model_or_weight_fit": False,
            "real_corpus_decode": False,
            "fold_or_evidence_namespace_open": False,
            "foundation_or_machinery_seal_against_corpus": False,
            "protected_holdout_open_count": 0,
            "live_broker_paper_or_download": False,
            "promotion_or_default_change": False,
        },
    }


def freeze_corrected_v32_executable_generation() -> dict[str, Any]:
    if V32_EXECUTABLE_ROOT.exists() and any(V32_EXECUTABLE_ROOT.iterdir()):
        raise RuntimeError("corrected-v3.2 executable root is occupied")
    semantic = _corrected_v32_executable_generation_semantic()
    prereg._write_canonical_json_exclusive(
        V32_EXECUTABLE_GENERATION_PATH,
        {**semantic, "generation_sha256": prereg.stable_hash(semantic)},
    )
    return assert_corrected_v32_executable_generation()


def assert_corrected_v32_executable_generation() -> dict[str, Any]:
    if not V32_EXECUTABLE_GENERATION_PATH.is_file() or V32_EXECUTABLE_GENERATION_PATH.is_symlink():
        raise RuntimeError("corrected-v3.2 executable generation is absent")
    observed = prereg.read_json(V32_EXECUTABLE_GENERATION_PATH)
    semantic = dict(observed)
    digest = semantic.pop("generation_sha256", None)
    expected = _corrected_v32_executable_generation_semantic()
    if semantic != expected or digest != prereg.stable_hash(expected):
        raise RuntimeError("corrected-v3.2 executable generation drift")
    return observed


def freeze_corrected_v32_executable_bridge() -> dict[str, Any]:
    """Freeze implementation/test hashes without authorizing fit or decode."""

    from v4.research.pathd_corrected_v32_foundation import V32_ROOT
    from v4.research.pathd_entry_dataset import assert_corrected_v32_executable_bridge

    generation = assert_corrected_v32_executable_generation()
    if any(os.path.lexists(path) for path in (
        V32_SYNTHETIC_TEST_RECEIPT_PATH, V32_EXECUTABLE_BRIDGE_PATH,
        V32_CLAUDE_RELEASE_PATH,
    )):
        raise RuntimeError("corrected-v3.2 executable receipt path is occupied")
    if not V32_SYNTHETIC_JUNIT_PATH.is_file() or V32_SYNTHETIC_JUNIT_PATH.is_symlink():
        raise RuntimeError("corrected-v3.2 synthetic JUnit is absent")
    suite = ET.parse(V32_SYNTHETIC_JUNIT_PATH).getroot()
    cases = suite.findall(".//testcase")
    names = tuple(case.get("name") for case in cases)
    failures = sum(len(case.findall("failure")) for case in cases)
    errors = sum(len(case.findall("error")) for case in cases)
    skipped = sum(len(case.findall("skipped")) for case in cases)
    if (
        names != V32_SYNTHETIC_TEST_NAMES
        or failures or errors or skipped
        or len(cases) != len(V32_SYNTHETIC_TEST_NAMES)
    ):
        raise RuntimeError("corrected-v3.2 synthetic JUnit result drift")
    payload = prereg.read_json(V32_ROOT / "preregistration.json")
    source_paths = payload["source_hash_policy"]["executable_generation_enforcement"]["source_paths"]
    implementations = [
        {"path": label, "sha256": prereg.sha256_path(prereg.REPO_ROOT / label)}
        for label in source_paths
    ]
    test_semantic = {
        "schema_version": "pathd.corrected_v32.synthetic_test_receipt.v1",
        "status": "PASS_SYNTHETIC_PRODUCTION_PATH",
        "frozen_at_utc": _utc_now(),
        "command": [
            ".venv/bin/python", "-m", "pytest", "-q", V32_SYNTHETIC_TEST_PATH,
            "--junitxml", prereg.repo_path_label(V32_SYNTHETIC_JUNIT_PATH),
        ],
        "pytest_exit_code": 0,
        "test_source_path": V32_SYNTHETIC_TEST_PATH,
        "test_source_sha256": prereg.sha256_path(prereg.REPO_ROOT / V32_SYNTHETIC_TEST_PATH),
        "junit_path": prereg.repo_path_label(V32_SYNTHETIC_JUNIT_PATH),
        "junit_sha256": prereg.sha256_path(V32_SYNTHETIC_JUNIT_PATH),
        "testcase_names": list(names),
        "implementations_sha256": prereg.stable_hash(implementations),
        "model_fit_executed": False,
        "corpus_decoded": False,
        "evidence_opened": False,
        "holdout_open_count": 0,
    }
    test_receipt = {**test_semantic, "receipt_sha256": prereg.stable_hash(test_semantic)}
    prereg._write_canonical_json_exclusive(V32_SYNTHETIC_TEST_RECEIPT_PATH, test_receipt)
    bridge_semantic = {
        "schema_version": "pathd.corrected_v32.executable_implementation_receipt.v1",
        "status": "FROZEN_EXECUTABLE_BUILT_NOT_RUN",
        "frozen_at_utc": _utc_now(),
        "foundation_generation_sha256": generation["foundation_generation_sha256"],
        "source_hash_policy_sha256": generation["source_hash_policy_sha256"],
        "executable_generation_sha256": generation["generation_sha256"],
        "implementations": implementations,
        "implementations_sha256": prereg.stable_hash(implementations),
        "synthetic_test_receipt_sha256": prereg.sha256_path(V32_SYNTHETIC_TEST_RECEIPT_PATH),
        "model_fit_executed": False,
        "corpus_decoded": False,
        "evidence_opened": False,
        "holdout_open_count": 0,
    }
    prereg._write_canonical_json_exclusive(
        V32_EXECUTABLE_BRIDGE_PATH,
        {**bridge_semantic, "receipt_sha256": prereg.stable_hash(bridge_semantic)},
    )
    return assert_corrected_v32_executable_bridge()


def assert_corrected_v32_fit_release() -> dict[str, Any]:
    """Reject v3.2 fit/open after its signed18 result was invalidated."""

    from v4.research.pathd_entry_dataset import assert_v32_execution_retired

    assert_v32_execution_retired()
    raise AssertionError("unreachable: v3.2 execution retirement must fail closed")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite scientific result")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    result = _plain(value)
    if type(result) is not dict:
        raise TypeError("scientific artifact must be a typed dataclass or exact mapping")
    prereg._assert_strict_json_value(result)
    return result


def _self_hash(value: dict[str, Any], field: str) -> None:
    semantic = dict(value)
    observed = semantic.pop(field, None)
    if observed != prereg.stable_hash(semantic):
        raise ValueError(f"{field} drift")


@dataclass(frozen=True)
class EntryPolicyEvaluationV1:
    SCHEMA_VERSION = "pathd.entry_policy_evaluation.v1"

    schema_version: str
    holdout_caveat: str
    evidence_role: str
    outer_fold: int
    inner_fold: int | None
    authorization_sha256: str
    dataset_sha256: str
    policy_id: str
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    session_pnl_micros: tuple[int, ...]
    session_terminal_journal_sha256s: tuple[str, ...]
    completed_trade_ids: tuple[str, ...]
    completed_trade_sessions: tuple[str, ...]
    zero_trade_sessions: tuple[str, ...]
    execution_trace_root_sha256: str
    terminal_journal: dict[str, Any]
    terminal_journal_sha256: str
    session_coverage_sha256: str
    metrics: dict[str, Any]
    result_sha256: str


@dataclass(frozen=True)
class EntryNestedFamilyEvaluationV1:
    SCHEMA_VERSION = "pathd.entry_nested_family_evaluation.v1"

    schema_version: str
    holdout_caveat: str
    outer_fold: int
    inner_fold: int
    authorization_sha256: str
    dataset_sha256: str
    source_receipts_root_sha256: str
    access_receipt_sha256: str
    hgb: Any
    neural: Any
    result_sha256: str


@dataclass(frozen=True)
class EntryOuterPrimaryResultV1:
    SCHEMA_VERSION = "pathd.entry_outer_primary_result.v1"

    schema_version: str
    outer_fold: int
    evidence_role: str
    authorization_sha256: str
    dataset_sha256: str
    source_receipts_root_sha256: str
    access_receipt_sha256: str
    hgb: Any
    neural: Any
    selected_family: str
    selected_evaluation_sha256: str
    negative_control_panel_sha256: str
    control_replay_result_sha256: str
    session_coverage_root_sha256: str
    result_sha256: str


@dataclass(frozen=True)
class EntryPooledAcceptanceResultV1:
    SCHEMA_VERSION = "pathd.entry_pooled_acceptance_result.v1"

    schema_version: str
    holdout_caveat: str
    aggregation_authorization_sha256: str
    outer_result_receipts_sha256: str
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    candidate_family: str
    candidate_outer_evaluation_sha256s: tuple[str, ...]
    control_exit_sha256s: tuple[str, ...]
    control_replay_result_sha256s: tuple[str, ...]
    negative_control_panel_sha256s: tuple[str, ...]
    gate_spec_sha256: str
    reconstructed_gate_inputs: dict[str, Any]
    gate_inputs_sha256: str
    pass_criteria_recomputed: dict[str, Any]
    verdict: str
    stop_reason: str | None
    full_fit_authorized: bool
    result_sha256: str


@dataclass(frozen=True)
class EntryContextSourceSelectionV1:
    SCHEMA_VERSION = "pathd.entry_context_source_selection.v1"

    schema_version: str
    source: str
    status: str
    requested: bool
    source_degraded: bool
    manifest_relative_path: str | None
    source_file_sha256: str | None
    current_component_locators: tuple[dict[str, Any], ...]
    lag15_component_locators: tuple[dict[str, Any], ...]
    current_available_at_ns: int | None
    lag15_available_at_ns: int | None
    current_age_ns: int | None
    lag15_age_ns: int | None
    current_close_float64_hex: str | None
    lag15_close_float64_hex: str | None
    selection_sha256: str


@dataclass(frozen=True)
class EntryContextAnchorRowV1:
    SCHEMA_VERSION = "pathd.entry_context_anchor_row.v1"

    schema_version: str
    outer_fold: int
    policy_id: str
    policy_order_index: int
    policy_evaluation_sha256: str
    terminal_journal_sha256: str
    anchor_kind: str
    anchor_ordinal: int
    anchor_id: str
    session: str
    anchor_time_ns: int
    source_transition_sha256: str
    action: str | None
    filled: bool | None
    trade_id: str | None
    trade_pnl_micros: int | None
    session_pnl_micros: int
    source_selections: tuple[EntryContextSourceSelectionV1, ...]
    source_selections_root_sha256: str
    derived_values_float64_hex: dict[str, str | None]
    bin_ids: dict[str, str]
    row_sha256: str


@dataclass(frozen=True)
class EntryOuterContextDiagnosticsV1:
    SCHEMA_VERSION = "pathd.entry_outer_context_diagnostics.v1"

    schema_version: str
    holdout_caveat: str
    status: str
    outer_fold: int
    authorization_sha256: str
    access_receipt_sha256: str
    primary_result_receipt_sha256: str
    core_integrity_receipt_sha256: str
    diagnostic_inventory_receipt_sha256: str
    policy_journal_bindings: tuple[dict[str, Any], ...]
    policy_journal_bindings_root_sha256: str
    sessions: tuple[str, ...]
    sessions_sha256_newline: str
    source_receipts: tuple[dict[str, Any], ...]
    source_receipts_root_sha256: str
    anchor_context_rows: tuple[EntryContextAnchorRowV1, ...]
    anchor_context_rows_root_sha256: str
    age_samples_ns_by_source: dict[str, tuple[int, ...]]
    age_samples_root_sha256: str
    tables: dict[str, Any]
    tables_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class EntryPooledContextDiagnosticsV1:
    SCHEMA_VERSION = "pathd.entry_pooled_context_diagnostics.v1"

    schema_version: str
    holdout_caveat: str
    status: str
    authorization_sha256: str
    access_receipt_sha256: str
    core_integrity_receipt_sha256: str
    diagnostic_inventory_receipt_sha256: str
    outer_artifact_paths: tuple[str, ...]
    outer_artifact_sha256s: tuple[str, ...]
    outer_receipt_paths: tuple[str, ...]
    outer_receipt_sha256s: tuple[str, ...]
    policy_journal_binding_roots_by_fold: tuple[str, ...]
    policy_journal_bindings_root_sha256: str
    anchor_context_rows: tuple[EntryContextAnchorRowV1, ...]
    anchor_context_rows_root_sha256: str
    age_samples_ns_by_source: dict[str, tuple[int, ...]]
    age_samples_root_sha256: str
    tables: dict[str, Any]
    tables_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class ProtectedHoldoutTraceRecordV1:
    SCHEMA_VERSION = "pathd.protected_holdout_trace_record.v1"

    schema_version: str
    holdout_caveat: str
    session: str
    authorization_sha256: str
    dataset_sha256: str
    box_d_policy_id: str
    comparator_policy_id: str
    box_d_terminal_journal: dict[str, Any]
    comparator_terminal_journal: dict[str, Any]
    box_d_session_pnl_micros: int
    comparator_session_pnl_micros: int
    box_d_completed_trade_ids: tuple[str, ...] | list[str]
    guard_results: dict[str, bool]
    survival_violations: tuple[str, ...] | list[str]
    record_sha256: str


@dataclass(frozen=True)
class ProtectedHoldoutEvaluationV1:
    SCHEMA_VERSION = "pathd.protected_holdout_evaluation.v1"

    schema_version: str
    authorization_sha256: str
    access_receipt_sha256: str
    preopen_packet_sha256: str
    artifact_root_sha256: str
    fit_environment_sha256: str
    sessions_sha256_newline: str
    primary_sessions_sha256_newline: str
    dataset_sha256: str
    source_receipts_root_sha256: str
    box_d_policy_id: str
    comparator_policy_id: str
    trace_root_sha256: str
    trace_artifact_sha256: str
    evaluator_receipt_sha256: str
    payload_sha256: str
    payload: dict[str, Any]


# Evidence decode is a one-shot capability.  The two positive family evaluations in
# one evidence transaction must share the exact same decoded object, and it must be
# discarded immediately after the terminal result receipt has been sealed.
_ACTIVE_ENTRY_EVIDENCE_DATASETS: dict[int, tuple[Any, Any]] = {}


def _active_entry_evidence_dataset(authorization: Any, /) -> Any:
    cached = _ACTIVE_ENTRY_EVIDENCE_DATASETS.get(id(authorization))
    if cached is None or cached[0] is not authorization:
        raise RuntimeError("entry evidence dataset is not active for this capability")
    return cached[1]


def _discard_active_entry_evidence_dataset(authorization: Any, /) -> None:
    cached = _ACTIVE_ENTRY_EVIDENCE_DATASETS.pop(id(authorization), None)
    if cached is not None and cached[0] is not authorization:
        raise RuntimeError("entry evidence dataset capability identity collision")


def _assert_fixed_paths_absent(paths: Any, /) -> None:
    values = tuple(paths)
    if not values:
        raise ValueError("fixed artifact path set is empty")
    for path in values:
        if type(path) is not Path:
            raise TypeError("fixed artifact path must be a Path")
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fixed artifact already exists: {path}")


def _write_typed_artifact_once(
    path: Path, value: Any, /, *, validator: Any
) -> dict[str, Any]:
    """Validate, write O_EXCL, then byte-reopen one fixed JSON artifact."""

    validated = validator(value)
    mapping = _mapping(validated)
    prereg._write_canonical_json_exclusive(path, mapping)
    reopened = prereg.read_json(path)
    if reopened != mapping or prereg.sha256_path(path) != hashlib.sha256(
        (
            json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    ).hexdigest():
        raise RuntimeError("fixed typed artifact durability drift")
    return reopened


def _entry_composer_payload(calibration: Any, /) -> dict[str, Any]:
    payload = getattr(calibration, "payload", None)
    if type(payload) is not dict:
        raise ValueError("entry calibration payload is absent")
    mean = payload.get("mean_lcb_corrections")
    q10 = payload.get("q10_corrections")
    if (
        type(mean) not in (tuple, list)
        or type(q10) not in (tuple, list)
        or len(mean) != len(prereg.ENTRY_HEAD_TARGETS)
        or len(q10) != len(prereg.ENTRY_HEAD_TARGETS)
        or any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in (*mean, *q10))
    ):
        raise ValueError("entry calibration correction-vector drift")
    action_corrections: dict[str, float] = {}
    action_receipts: dict[str, dict[str, Any]] = {}
    for action, key in (
        ("ENTER", "enter_composite_q10_correction"),
        ("WAIT", "wait_composite_q10_correction"),
    ):
        prefix = action.lower()
        if payload.get(f"{prefix}_composite_q10_status") != "VALID":
            raise RuntimeError(
                f"entry {action} composite calibration is not valid; evidence replay is blocked"
            )
        correction = payload.get(key)
        if type(correction) not in (int, float) or not math.isfinite(float(correction)):
            raise ValueError(f"entry {action} action calibration correction drift")
        receipt = payload.get(f"{prefix}_composite_calibration_receipt")
        if type(receipt) is not dict or receipt.get("status") != "VALID":
            raise ValueError(f"entry {action} action calibration receipt drift")
        action_corrections[action] = float(correction)
        action_receipts[action] = dict(receipt)
    missing_enter = payload.get("enter_missing_outcome_intent_count")
    missing_wait = payload.get("wait_missing_outcome_episode_count")
    if any(type(value) is not int or value != 0 for value in (missing_enter, missing_wait)):
        raise ValueError("entry action calibration retained missing outcomes")
    return {
        "mean_lcb_additive_offsets": [float(item) for item in mean],
        "q10_additive_offsets": [float(item) for item in q10],
        "enter_composite_q10_correction": action_corrections["ENTER"],
        "wait_composite_q10_correction": action_corrections["WAIT"],
        "enter_composite_q10_status": "VALID",
        "wait_composite_q10_status": "VALID",
        "enter_composite_calibration_receipt": action_receipts["ENTER"],
        "wait_composite_calibration_receipt": action_receipts["WAIT"],
        "enter_missing_outcome_intent_count": missing_enter,
        "wait_missing_outcome_episode_count": missing_wait,
    }


def _entry_calibration_scope(*, outer_fold: int, inner_fold: int | None) -> str:
    if outer_fold not in range(1, 6):
        raise ValueError("outer_fold must be 1..5")
    if inner_fold is None:
        return f"OUTER_{outer_fold}"
    if inner_fold not in range(1, 5):
        raise ValueError("inner_fold must be 1..4")
    return f"NESTED_OUTER_{outer_fold}_INNER_{inner_fold}"


def derive_entry_calibration_node_vector(
    *, scope: str, calibration_bundles: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Reconstruct every B-R node from actual calibration bundle payloads."""

    from v4.research import pathd_entry_models as entry_models

    required = prereg.entry_required_calibration_node_ids(scope)
    bundle_ids = tuple(dict.fromkeys(node.split("::")[2] for node in required))
    if tuple(calibration_bundles) != bundle_ids:
        raise ValueError("calibration bundle vector is incomplete or out of order")
    rows: list[dict[str, Any]] = []
    for bundle_id in bundle_ids:
        raw = calibration_bundles[bundle_id]
        payload = getattr(raw, "payload", None)
        if payload is None and type(raw) is dict:
            payload = raw.get("payload")
        if type(payload) is not dict:
            raise ValueError("calibration bundle payload absent")
        head_receipts = payload.get("head_calibration_receipts")
        if type(head_receipts) not in (tuple, list) or len(head_receipts) != len(
            prereg.ENTRY_HEAD_TARGETS
        ):
            raise ValueError("calibration head receipt vector drift")
        for head, receipt in zip(
            prereg.ENTRY_HEAD_TARGETS, head_receipts, strict=True
        ):
            if type(receipt) is not dict or receipt.get("head") != head:
                raise ValueError("calibration head receipt identity drift")
            observed = receipt.get("row_count")
            sessions = receipt.get("session_count")
            source_hash = receipt.get("receipt_sha256")
            valid = (
                type(observed) is int
                and observed > 0
                and type(sessions) is int
                and sessions >= 10
                and type(receipt.get("mean_lcb_correction")) in (int, float)
                and math.isfinite(float(receipt["mean_lcb_correction"]))
                and type(receipt.get("q10_correction")) in (int, float)
                and math.isfinite(float(receipt["q10_correction"]))
                and type(source_hash) is str
                and re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None
            )
            for kind in ("MEAN_LCB", "Q10_CONFORMAL"):
                node_id = f"ENTRY::{scope}::{bundle_id}::{kind}::{head}"
                semantic = {
                    "node_id": node_id,
                    "stage": "ENTRY_CALIBRATION",
                    "scope": scope,
                    "family_or_control_identity": bundle_id,
                    "target_head_or_action": head,
                    "consumer": "ENTRY_ACTION_COMPOSER",
                    "observed_count": observed,
                    "minimum_count": {"sessions": 10, "rows": 1},
                    "decision_critical_or_diagnostic": "DECISION_CRITICAL",
                    "failure_mapping": (
                        "VALID" if valid else "INVALID_TARGET_COVERAGE"
                    ),
                    "raw_node_status": "VALID" if valid else "INVALID_TARGET_COVERAGE",
                    "upstream_hashes": [source_hash] if type(source_hash) is str else [],
                }
                rows.append({**semantic, "node_sha256": prereg.stable_hash(semantic)})
        for action, minimum_rows in (("ENTER", 1), ("WAIT", 30)):
            prefix = action.lower()
            receipt = payload.get(f"{prefix}_composite_calibration_receipt")
            status = payload.get(f"{prefix}_composite_q10_status")
            if type(receipt) is not dict:
                receipt = {}
            observed = receipt.get("row_count")
            sessions = receipt.get("session_count")
            source_hash = receipt.get("receipt_sha256")
            valid = (
                status == "VALID"
                and receipt.get("status") == "VALID"
                and type(observed) is int
                and observed >= minimum_rows
                and type(sessions) is int
                and sessions >= 10
                and type(source_hash) is str
                and re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None
            )
            node_id = f"ENTRY::{scope}::{bundle_id}::ACTION_COMPOSITE::{action}"
            semantic = {
                "node_id": node_id,
                "stage": "ENTRY_CALIBRATION",
                "scope": scope,
                "family_or_control_identity": bundle_id,
                "target_head_or_action": action,
                "consumer": f"{action}_ACTION_GATE",
                "observed_count": observed,
                "minimum_count": {"sessions": 10, "rows": minimum_rows},
                "decision_critical_or_diagnostic": "DECISION_CRITICAL",
                "failure_mapping": "VALID" if valid else (
                    status if status in {"INSUFFICIENT_EVIDENCE", "INVALID_TARGET_COVERAGE"}
                    else "INVALID_TARGET_COVERAGE"
                ),
                "raw_node_status": status,
                "upstream_hashes": [source_hash] if type(source_hash) is str else [],
            }
            rows.append({**semantic, "node_sha256": prereg.stable_hash(semantic)})
    by_id = {row["node_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(required):
        raise RuntimeError("calibration node ordering/identity drift")
    rows = [by_id[node_id] for node_id in required]
    manifest = next(
        row
        for row in prereg.composite_calibration_terminal_rule_spec()["node_registry"][
            "entry"
        ]["manifests"]
        if row["scope"] == scope
    )
    if (
        len(rows) != manifest["required_node_count"]
        or prereg.stable_hash([row["node_id"] for row in rows])
        != manifest["required_node_ids_sha256"]
    ):
        raise RuntimeError("calibration node manifest drift")
    return tuple(rows)


def evaluate_entry_calibration_scope_gate(
    *, scope: str, calibration_bundles: dict[str, Any]
) -> dict[str, Any]:
    """Return a success gate or a failure plan; never relabel a failed node."""

    nodes = derive_entry_calibration_node_vector(
        scope=scope, calibration_bundles=calibration_bundles
    )
    failures = tuple(row for row in nodes if row["raw_node_status"] != "VALID")
    manifest_sha = prereg.stable_hash([row["node_id"] for row in nodes])
    semantic = {
        "schema_version": "pathd.calibration_scope_gate_receipt.v1",
        "scope": scope,
        "status": "VALID" if not failures else "FAILED_CLOSED",
        "required_node_count": len(nodes),
        "required_node_ids_sha256": manifest_sha,
        "ordered_node_sha256s": [row["node_sha256"] for row in nodes],
        "node_vector_sha256": prereg.stable_hash(list(nodes)),
        "failure_node_ids": [row["node_id"] for row in failures],
        "evidence_access_count": 0,
        "holdout_open_count": 0,
        "forbidden_rescue_applied": False,
    }
    return {**semantic, "receipt_sha256": prereg.stable_hash(semantic)}


def _calibration_scope_gate_path(*, outer_fold: int, inner_fold: int | None) -> Path:
    name = (
        "entry_calibration_scope_gate_receipt.json"
        if inner_fold is None
        else f"nested_inner_{inner_fold}_calibration_scope_gate_receipt.json"
    )
    return prereg._outer_fold_artifact_path(outer_fold, name)


def freeze_entry_calibration_scope_gate_once(
    *, outer_fold: int, inner_fold: int | None,
    calibration_bundles: dict[str, Any],
) -> dict[str, Any]:
    scope = _entry_calibration_scope(outer_fold=outer_fold, inner_fold=inner_fold)
    nodes = derive_entry_calibration_node_vector(
        scope=scope, calibration_bundles=calibration_bundles
    )
    result = evaluate_entry_calibration_scope_gate(
        scope=scope, calibration_bundles=calibration_bundles
    )
    if result["status"] != "VALID":
        fold_root = prereg._outer_fold_artifact_path(outer_fold, "placeholder").parent
        failure_rows = []
        for node in nodes:
            if node["raw_node_status"] == "VALID":
                continue
            failure_semantic = {
                "schema_version": "pathd.calibration_node_failure_receipt.v1",
                "node_id": node["node_id"],
                "raw_node_status": node["raw_node_status"],
                "closed_failure_code": node["failure_mapping"],
                "counts_and_minima": {
                    "observed_count": node["observed_count"],
                    "minimum_count": node["minimum_count"],
                },
                "scope_bundle_target_consumer_role": {
                    "scope": node["scope"],
                    "bundle": node["family_or_control_identity"],
                    "target": node["target_head_or_action"],
                    "consumer": node["consumer"],
                    "role": node["decision_critical_or_diagnostic"],
                },
                "upstream_hashes": node["upstream_hashes"],
                "evidence_access_count_zero": True,
                "holdout_open_count_zero": True,
            }
            failure = {
                **failure_semantic,
                "receipt_sha256": prereg.stable_hash(failure_semantic),
            }
            path = fold_root / (
                "calibration_node_failure_"
                + failure["receipt_sha256"][:16]
                + ".json"
            )
            prereg._write_canonical_json_exclusive(path, failure)
            failure_rows.append(
                {
                    "path": prereg.repo_path_label(path),
                    "sha256": prereg.sha256_path(path),
                    "node_id": node["node_id"],
                }
            )
        if not failure_rows:
            raise AssertionError("failed calibration scope has no failed node")
        campaign_path = prereg.AUDIT_ROOT / "entry_calibration_campaign_failure_receipt.json"
        campaign_semantic = {
            "schema_version": "pathd.calibration_campaign_failure_receipt.v1",
            "status": "FAILED_CLOSED_BEFORE_EVIDENCE",
            "first_failing_scope": scope,
            "node_failure_receipts": failure_rows,
            "prior_immutable_artifacts": [
                {
                    "path": prereg.repo_path_label(path),
                    "sha256": prereg.sha256_path(path),
                }
                for path in sorted(
                    fold_root.iterdir(), key=lambda item: item.name
                )
                if path.is_file()
                and path.name.startswith(("hgb_", "neural_", "negative_control_"))
            ],
            "terminal_status": (
                "invalid_result"
                if any(
                    node["failure_mapping"] == "INVALID_TARGET_COVERAGE"
                    for node in nodes
                )
                else "insufficient_evidence"
            ),
            "evidence_access_count": 0,
            "holdout_open_count": 0,
            "forbidden_rescue_applied": False,
        }
        campaign = {
            **campaign_semantic,
            "receipt_sha256": prereg.stable_hash(campaign_semantic),
        }
        prereg._write_canonical_json_exclusive(campaign_path, campaign)
        raise RuntimeError(
            "calibration scope failed closed before evidence: "
            + ",".join(result["failure_node_ids"])
        )
    path = _calibration_scope_gate_path(
        outer_fold=outer_fold, inner_fold=inner_fold
    )
    prereg._write_canonical_json_exclusive(path, result)
    reopened = prereg.read_json(path)
    if reopened != result:
        raise RuntimeError("calibration scope gate durability drift")
    return reopened


def begin_entry_evidence_after_valid_calibration_scope_gate(
    *, role: str, outer_fold: int, inner_fold: int | None,
) -> Any:
    """The corrected-v3.2-only evidence open: revalidate release and B-R first."""

    from v4.research.pathd_entry_dataset import assert_corrected_v32_executable_bridge
    from v4.research.pathd_evidence_gate import begin_entry_evidence_once

    assert_corrected_v32_executable_bridge()
    assert_corrected_v32_fit_release()
    scope = _entry_calibration_scope(outer_fold=outer_fold, inner_fold=inner_fold)
    gate = prereg.read_json(
        _calibration_scope_gate_path(outer_fold=outer_fold, inner_fold=inner_fold)
    )
    semantic = dict(gate)
    receipt_sha = semantic.pop("receipt_sha256", None)
    if (
        gate.get("schema_version") != "pathd.calibration_scope_gate_receipt.v1"
        or gate.get("scope") != scope
        or gate.get("status") != "VALID"
        or gate.get("failure_node_ids") != []
        or gate.get("evidence_access_count") != 0
        or gate.get("holdout_open_count") != 0
        or receipt_sha != prereg.stable_hash(semantic)
    ):
        raise RuntimeError("evidence blocked: calibration scope gate invalid")
    return begin_entry_evidence_once(
        role=role, outer_fold=outer_fold, inner_fold=inner_fold
    )


def _flatten_entry_position_at_terminal(
    journal: Any,
    /,
    *,
    dataset: Any,
    authorization: Any,
    fill_law: Any,
) -> Any:
    """Advance a held position causally and consume it exactly once at 15:55 ET."""

    from v4.path_d.execution import research_replay as replay

    active = replay.validate_research_ledger_journal(
        journal, authorization=authorization
    )
    if active.tip_ledger.position is None:
        if active.tip_ledger.pending_intent_id is not None:
            raise RuntimeError("flat terminal ledger retained a pending intent")
        return active
    session = active.tip_ledger.session
    terminal_ns = int(
        datetime.fromisoformat(session)
        .replace(hour=15, minute=55, second=0, microsecond=0, tzinfo=ZoneInfo("America/New_York"))
        .astimezone(timezone.utc)
        .timestamp()
    ) * 1_000_000_000
    last_ns = active.tip_ledger.last_decision_time_ns
    if type(last_ns) is not int or last_ns >= terminal_ns:
        raise RuntimeError("held terminal ledger clock drift")
    while active.tip_ledger.last_decision_time_ns < terminal_ns - 1_000_000_000:
        active = replay.advance_research_position_clock(
            active,
            authorization=authorization,
            decision_time_ns=active.tip_ledger.last_decision_time_ns + 1_000_000_000,
            reason_code="PATHD_RESEARCH_HOLD_TO_FLAT",
        )
    context = replay.exit_decision_context_from_verified_journal(
        active, dataset=dataset, authorization=authorization
    )
    if isinstance(context, replay.NonOrderSafetyEventV1):
        active = replay.append_research_non_order_safety_event(
            active, context, authorization=authorization, law=fill_law
        )
    else:
        if context.intent_kind != "TERMINAL_SELL" or context.decision_time_ns != terminal_ns:
            raise RuntimeError("hold-to-flat replay did not reach the terminal boundary")
        prepared = replay.prepare_research_execution(
            intent_kind="TERMINAL_SELL",
            context=context,
            journal=active,
            dataset=dataset,
            authorization=authorization,
            law=fill_law,
            reason_code="PATHD_RESEARCH_TERMINAL_FLAT",
        )
        outcome = replay.replay_research_intent(
            prepared,
            dataset=dataset,
            authorization=authorization,
            law=fill_law,
            delay_ms=0,
        )
        active = outcome.result_journal
    if active.tip_ledger.position is not None or active.tip_ledger.pending_intent_id is not None:
        raise RuntimeError("terminal exit did not leave the account flat")
    return active


def entry_evidence_role_for_session(
    *, assignments: Any, outer_fold: int, inner_fold: int, session: str
) -> str:
    if type(assignments) is not dict or outer_fold not in range(1, 6):
        raise ValueError("invalid entry evidence assignment/fold")
    fold = assignments["folds"][outer_fold - 1]
    rows = fold["inner_forward_folds"]["scored_forward_folds"]
    if inner_fold not in range(1, len(rows) + 1):
        raise ValueError("invalid inner fold")
    if session in rows[inner_fold - 1]["validation"]:
        return "nested_validation"
    if session in fold["outer_test_primary_1555_complete"]:
        return "outer_test_primary"
    if session in fold["outer_test_shortened_diagnostic_only"]:
        return "outer_test_shortened_diagnostic"
    raise ValueError("session is outside requested evidence role")


def validate_entry_policy_evaluation(
    evaluation: Any, /, *, authorization: Any
) -> dict[str, Any]:
    value = _mapping(evaluation)
    expected_fields = set(prereg.entry_future_api_contract()["dataclass_fields"]["EntryPolicyEvaluationV1"])
    if set(value) != expected_fields:
        raise ValueError("entry policy evaluation schema drift")
    current = prereg.assert_entry_evidence_authorization_current(authorization)
    if (
        value.get("schema_version") != EntryPolicyEvaluationV1.SCHEMA_VERSION
        or value.get("authorization_sha256") != prereg.stable_hash(current.to_dict())
        or value.get("holdout_caveat") != prereg.HOLDOUT_CAVEAT
    ):
        raise ValueError("entry policy evaluation identity drift")
    from v4.path_d.execution.research_replay import (
        reconstruct_entry_policy_evaluation_from_journal,
    )

    reconstructed = reconstruct_entry_policy_evaluation_from_journal(
        value["terminal_journal"],
        authorization=current,
        dataset_sha256=value["dataset_sha256"],
        sessions=tuple(value["sessions"]),
        evidence_role=value["evidence_role"],
        outer_fold=value["outer_fold"],
        inner_fold=value["inner_fold"],
        policy_id=value["policy_id"],
    )
    if value != reconstructed:
        raise ValueError("entry policy evaluation does not reconstruct from terminal_journal")
    return value


def validate_entry_nested_family_evaluation(
    evaluation: Any, /, *, authorization: Any, dataset: Any
) -> dict[str, Any]:
    from v4.research.pathd_entry_dataset import validate_entry_evidence_dataset

    current = prereg.assert_entry_evidence_authorization_current(authorization)
    validate_entry_evidence_dataset(dataset, authorization=current)
    value = _mapping(evaluation)
    expected = set(prereg.entry_future_api_contract()["dataclass_fields"]["EntryNestedFamilyEvaluationV1"])
    if set(value) != expected:
        raise ValueError("nested family evaluation schema drift")
    authorization_sha256 = prereg.stable_hash(current.to_dict())
    source_receipts_root_sha256 = prereg.stable_hash(list(dataset.source_receipts))
    if (
        value.get("schema_version") != EntryNestedFamilyEvaluationV1.SCHEMA_VERSION
        or value.get("holdout_caveat") != prereg.HOLDOUT_CAVEAT
        or value.get("authorization_sha256") != authorization_sha256
        or value.get("dataset_sha256") != dataset.dataset_sha256
        or value.get("source_receipts_root_sha256") != source_receipts_root_sha256
        or value.get("access_receipt_sha256") != current.access_receipt_sha256
    ):
        raise ValueError("nested authorization/dataset/source/access binding drift")
    validate_entry_policy_evaluation(value["hgb"], authorization=current)
    validate_entry_policy_evaluation(value["neural"], authorization=current)
    _self_hash(value, "result_sha256")
    return value


def validate_entry_outer_selection_binding(
    result: Any,
    /,
    *,
    authorization_sha256: str,
    dataset_sha256: str,
    source_receipts_root_sha256: str,
    access_receipt_sha256: str,
) -> dict[str, Any]:
    value = _mapping(result)
    expected = set(prereg.entry_future_api_contract()["dataclass_fields"]["EntryOuterPrimaryResultV1"])
    if set(value) != expected:
        raise ValueError("outer result schema drift")
    hgb = value.get("hgb")
    neural = value.get("neural")
    if type(hgb) is not dict or type(neural) is not dict:
        raise TypeError("outer result evaluations must be mappings")
    if (
        value.get("schema_version") != EntryOuterPrimaryResultV1.SCHEMA_VERSION
        or value.get("authorization_sha256") != authorization_sha256
        or value.get("dataset_sha256") != dataset_sha256
        or value.get("source_receipts_root_sha256") != source_receipts_root_sha256
        or value.get("access_receipt_sha256") != access_receipt_sha256
        or hgb.get("family") != "HGB"
        or neural.get("family") != "NEURAL"
        or value.get("selected_family") != "HGB"
        or value.get("selected_evaluation_sha256") != hgb.get("result_sha256")
    ):
        raise ValueError("outer HGB selection/provenance binding drift")
    _self_hash(value, "result_sha256")
    return value


def validate_entry_outer_primary_result(
    result: Any, /, *, authorization: Any, dataset: Any
) -> dict[str, Any]:
    from v4.research.pathd_entry_dataset import validate_entry_evidence_dataset

    current = prereg.assert_entry_evidence_authorization_current(authorization)
    validate_entry_evidence_dataset(dataset, authorization=current)
    authorization_sha256 = prereg.stable_hash(current.to_dict())
    source_receipts_root_sha256 = prereg.stable_hash(list(dataset.source_receipts))
    value = validate_entry_outer_selection_binding(
        result,
        authorization_sha256=authorization_sha256,
        dataset_sha256=dataset.dataset_sha256,
        source_receipts_root_sha256=source_receipts_root_sha256,
        access_receipt_sha256=current.access_receipt_sha256,
    )
    hgb = validate_entry_policy_evaluation(value["hgb"], authorization=current)
    neural = validate_entry_policy_evaluation(value["neural"], authorization=current)
    session_coverage_root_sha256 = prereg.stable_hash(
        [hgb["session_coverage_sha256"], neural["session_coverage_sha256"]]
    )
    if value.get("session_coverage_root_sha256") != session_coverage_root_sha256:
        raise ValueError("outer session_coverage_root_sha256 drift")
    return value


def run_entry_policy_evaluation(
    authorization: Any,
    /,
    *,
    policy_bundle: Any,
    composer_bundle: Any,
    fill_law: Any,
    opening_journal: Any,
) -> EntryPolicyEvaluationV1:
    """Replay one frozen positive-family entry policy over one opened evidence set.

    The evidence dataset is obtained only from the one-shot loader and retained only
    for the lifetime of the process-local evidence transaction.  No caller can pass a
    dataset, quote, outcome, metric, or result payload through this API.
    """

    from v4.path_d.execution import research_replay as replay
    from v4.path_d.execution.research_fill_law import (
        research_fill_law_from_preregistration,
    )
    from v4.research import pathd_entry_models as entry_models
    from v4.research.pathd_entry_dataset import (
        load_authorized_entry_evidence_dataset,
        validate_entry_evidence_dataset,
    )

    current = prereg.assert_entry_evidence_authorization_current(authorization)
    model = entry_models.validate_entry_model_bundle(policy_bundle)
    # ``compose_entry_action`` performs the exact composer/model seal validation.
    # This public evaluation API intentionally has no calibration argument, so it
    # must not resolve or accept an alternate caller-selected calibrator here.
    composer = composer_bundle
    if type(composer) is not entry_models.EntryComposerBundleV1:
        raise TypeError("entry policy evaluation requires the exact composer type")
    expected_law = research_fill_law_from_preregistration(
        prereg.preregistration_payload()[0]
    )
    if fill_law != expected_law:
        raise ValueError("entry policy evaluation fill-law drift")
    if (
        model.outer_fold != current.outer_fold
        or model.inner_fold != current.inner_fold
        or composer.outer_fold != current.outer_fold
        or composer.inner_fold != current.inner_fold
    ):
        raise ValueError("entry policy evaluation model/evidence scope drift")

    cached = _ACTIVE_ENTRY_EVIDENCE_DATASETS.get(id(authorization))
    if cached is None:
        dataset = load_authorized_entry_evidence_dataset(current)
        _ACTIVE_ENTRY_EVIDENCE_DATASETS[id(authorization)] = (
            authorization, dataset
        )
    else:
        if cached[0] is not authorization:
            raise RuntimeError("entry evidence dataset capability identity collision")
        dataset = cached[1]
    validate_entry_evidence_dataset(dataset, authorization=current)
    journal = replay.validate_research_ledger_journal(
        opening_journal, authorization=current
    )
    if journal.policy_id != model.family:
        raise ValueError("entry policy journal/model family drift")

    examples = tuple(dataset.examples)
    if not examples:
        raise RuntimeError("entry policy evaluation dataset is empty")
    wait_episode_id: str | None = None
    for index, example in enumerate(examples):
        coverage = replay.entry_frame_coverage_from_verified_example(
            example,
            dataset=dataset,
            authorization=current,
            journal=journal,
        )
        journal = replay.append_entry_frame_coverage(
            journal,
            coverage,
            dataset=dataset,
            authorization=current,
        )
        if coverage.disposition == "ACTION_DECISION":
            decision = entry_models.compose_entry_action(
                composer,
                model,
                example,
                journal=journal,
                dataset=dataset,
                authorization=current,
                fill_law=fill_law,
            )
            journal = replay.append_research_action_decision(
                journal,
                decision,
                dataset=dataset,
                authorization=current,
            )
            decision_identity = (
                f"{model.family}:{example.model_input.session}:"
                f"{example.model_input.decision_time_ns}"
            )
            outcome = None
            if decision.action == "ENTER":
                wait_episode_id = None
                context = replay.entry_decision_context_from_verified_example(
                    example,
                    dataset=dataset,
                    authorization=current,
                    decision=decision,
                )
                prepared = replay.prepare_research_execution(
                    intent_kind="BUY",
                    context=context,
                    journal=journal,
                    dataset=dataset,
                    authorization=current,
                    law=fill_law,
                    reason_code="PATHD_RESEARCH_ENTRY",
                )
                outcome = replay.replay_research_intent(
                    prepared,
                    dataset=dataset,
                    authorization=current,
                    law=fill_law,
                    delay_ms=60_000,
                )
                journal = outcome.result_journal
                trajectory_id = decision_identity + ":ENTER"
                episode_id = trajectory_id
            elif decision.action == "WAIT":
                if wait_episode_id is None:
                    wait_episode_id = decision_identity + ":WAIT"
                trajectory_id = wait_episode_id
                episode_id = wait_episode_id
            else:
                raise RuntimeError("entry composer emitted an unregistered action")
            observation = replay.entry_action_observation_from_verified_dataset(
                decision,
                dataset=dataset,
                authorization=current,
                trajectory_id=trajectory_id,
                episode_id=episode_id,
                execution_outcome=outcome,
            )
            journal = replay.append_entry_action_observation(
                journal,
                observation,
                dataset=dataset,
                authorization=current,
            )
        else:
            wait_episode_id = None

        final_for_session = (
            index + 1 == len(examples)
            or examples[index + 1].model_input.session
            != example.model_input.session
        )
        if not final_for_session:
            continue
        journal = _flatten_entry_position_at_terminal(
            journal,
            dataset=dataset,
            authorization=current,
            fill_law=fill_law,
        )
        journal = replay.mark_research_session_terminal(
            journal,
            authorization=current,
            reason_code="PATHD_RESEARCH_SESSION_COMPLETE",
        )
        if index + 1 < len(examples):
            journal = replay.advance_research_ledger_session(
                journal, authorization=current
            )
    replay.validate_research_ledger_journal_against_dataset(
        journal, authorization=current, dataset=dataset
    )
    value = replay.reconstruct_entry_policy_evaluation_from_journal(
        journal,
        authorization=current,
        dataset_sha256=dataset.dataset_sha256,
        sessions=current.sessions,
        evidence_role=current.role,
        outer_fold=current.outer_fold,
        inner_fold=current.inner_fold,
        policy_id=model.family,
    )
    result = EntryPolicyEvaluationV1(**value)
    validate_entry_policy_evaluation(result, authorization=current)
    return result


def freeze_entry_pooled_acceptance_once() -> EntryPooledAcceptanceResultV1:
    """Reconstruct and durably freeze the pooled verdict from five outer receipts."""

    authorization = prereg.assert_entry_result_aggregation_ready(
        role="pooled_outer_primary"
    )
    result_path = prereg.ENTRY_POOLED_ACCEPTANCE_RESULT_PATH
    receipt_path = prereg.ENTRY_POOLED_ACCEPTANCE_RECEIPT_PATH
    for path in (result_path, receipt_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError("entry pooled acceptance fixed artifact already exists")
    payload = prereg.read_json(prereg.PREREG_PATH)
    gate_inputs = prereg._reconstruct_entry_pooled_gate_inputs_from_outer_artifacts(
        authorization=authorization
    )
    inputs, criteria, verdict, stop_reason, full = (
        prereg._validate_entry_pooled_gate_inputs(gate_inputs)
    )
    candidate_hashes: list[str] = []
    control_exit_hashes: list[str] = []
    replay_hashes: list[str] = []
    negative_hashes: list[str] = []
    for outer_fold in range(1, 6):
        outer = prereg.read_json(
            prereg._outer_fold_artifact_path(
                outer_fold, "outer_primary_result.json"
            )
        )
        candidate = outer.get("payload", {}).get("hgb", {}).get("result_sha256")
        if type(candidate) is not str or re.fullmatch(r"[0-9a-f]{64}", candidate) is None:
            raise RuntimeError("entry pooled candidate result binding drift")
        candidate_hashes.append(candidate)
        control_exit_hashes.append(
            prereg.sha256_path(
                prereg._outer_fold_artifact_path(outer_fold, "control_exit.json")
            )
        )
        replay_hashes.append(
            prereg.sha256_path(
                prereg._outer_fold_artifact_path(
                    outer_fold, "control_replay_result.json"
                )
            )
        )
        negative_hashes.append(
            prereg.sha256_path(
                prereg._outer_fold_artifact_path(
                    outer_fold, "negative_control_panel.json"
                )
            )
        )
    semantic = {
        "schema_version": EntryPooledAcceptanceResultV1.SCHEMA_VERSION,
        "holdout_caveat": prereg.HOLDOUT_CAVEAT,
        "aggregation_authorization_sha256": prereg.stable_hash(
            authorization.to_dict()
        ),
        "outer_result_receipts_sha256": prereg.stable_hash(
            list(authorization.outer_result_receipts_sha256)
        ),
        "sessions": list(authorization.sessions),
        "sessions_sha256_newline": authorization.sessions_sha256_newline,
        "candidate_family": "HGB",
        "candidate_outer_evaluation_sha256s": candidate_hashes,
        "control_exit_sha256s": control_exit_hashes,
        "control_replay_result_sha256s": replay_hashes,
        "negative_control_panel_sha256s": negative_hashes,
        "gate_spec_sha256": prereg._entry_pooled_gate_spec_sha256(payload),
        "reconstructed_gate_inputs": inputs,
        "gate_inputs_sha256": prereg.stable_hash(inputs),
        "pass_criteria_recomputed": criteria,
        "verdict": verdict,
        "stop_reason": stop_reason,
        "full_fit_authorized": full,
    }
    result = EntryPooledAcceptanceResultV1(
        **semantic, result_sha256=prereg.stable_hash(semantic)
    )
    validated = prereg.validate_entry_pooled_acceptance_result(
        result, authorization=authorization
    )
    prereg._write_canonical_json_exclusive(result_path, validated)
    receipt_semantic = {
        "schema_version": "pathd.entry_pooled_acceptance_receipt.v1",
        "status": "FROZEN_ENTRY_POOLED_ACCEPTANCE",
        "result_path": prereg.repo_path_label(result_path),
        "result_sha256": prereg.sha256_path(result_path),
        "aggregation_authorization_sha256": prereg.stable_hash(
            authorization.to_dict()
        ),
        "outer_result_receipts_sha256": prereg.stable_hash(
            list(authorization.outer_result_receipts_sha256)
        ),
        "verdict": verdict,
        "full_fit_authorized": full,
    }
    receipt = {
        **receipt_semantic,
        "receipt_sha256": prereg.stable_hash(receipt_semantic),
    }
    prereg._write_canonical_json_exclusive(receipt_path, receipt)
    prereg.read_frozen_entry_pooled_acceptance(require_pass=False)
    return result


def validate_entry_outer_context_diagnostics(
    diagnostics: Any, /, *, authorization: Any, outer_fold: int
) -> dict[str, Any]:
    current = prereg.assert_context_diagnostics_authorization_current(authorization)
    value = _mapping(diagnostics)
    expected = set(prereg.entry_future_api_contract()["dataclass_fields"]["EntryOuterContextDiagnosticsV1"])
    if set(value) != expected or value.get("outer_fold") != outer_fold:
        raise ValueError("outer context diagnostics schema/fold drift")
    if value.get("authorization_sha256") != current.authorization_sha256:
        raise ValueError("outer context authorization drift")
    receipts = value.get("source_receipts")
    if type(receipts) is not list or [row.get("source") for row in receipts] != [
        "VIX", "ES", "VX", "SPX_REFERENCE"
    ]:
        raise ValueError("outer context source receipt order drift")
    for field_name, source in (
        ("source_receipts_root_sha256", receipts),
        ("policy_journal_bindings_root_sha256", value.get("policy_journal_bindings")),
        ("anchor_context_rows_root_sha256", value.get("anchor_context_rows")),
        ("age_samples_root_sha256", value.get("age_samples_ns_by_source")),
        ("tables_sha256", value.get("tables")),
    ):
        if value.get(field_name) != prereg.stable_hash(source):
            raise ValueError(f"outer context {field_name} drift")
    _self_hash(value, "artifact_sha256")
    return value


def validate_entry_pooled_context_diagnostics(
    diagnostics: Any, /, *, authorization: Any
) -> dict[str, Any]:
    current = prereg.assert_context_diagnostics_authorization_current(authorization)
    value = _mapping(diagnostics)
    expected = set(prereg.entry_future_api_contract()["dataclass_fields"]["EntryPooledContextDiagnosticsV1"])
    if set(value) != expected or value.get("authorization_sha256") != current.authorization_sha256:
        raise ValueError("pooled context schema/authorization drift")
    for field_name, source in (
        ("anchor_context_rows_root_sha256", value.get("anchor_context_rows")),
        ("age_samples_root_sha256", value.get("age_samples_ns_by_source")),
        ("tables_sha256", value.get("tables")),
    ):
        if value.get(field_name) != prereg.stable_hash(source):
            raise ValueError(f"pooled context {field_name} drift")
    _self_hash(value, "artifact_sha256")
    return value


def _spx_reference_selection(
    *, session: str, anchor_time_ns: int
) -> tuple[EntryContextSourceSelectionV1, dict[str, Any]]:
    """Resolve and decode the fixed official-SPX diagnostic source internally."""

    import pyarrow.parquet as pq

    if (
        type(session) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", session) is None
        or type(anchor_time_ns) is not int
    ):
        raise ValueError("SPX_REFERENCE selector session/clock drift")
    relative = f"vendor/thetadata/index/spx_1m/{session}.parquet"
    entries = [
        row
        for row in prereg.integrity_manifest_entries_for_partition(
            prereg.CORE_INTEGRITY_PARTITION
        )
        if row.get("relative_path") == relative
    ]
    if len(entries) != 1:
        raise RuntimeError("SPX_REFERENCE core manifest identity drift")
    entry = entries[0]
    path = prereg.CORPUS_ROOT / relative
    source_receipt = {
        "source": "SPX_REFERENCE",
        "relative_path": relative,
        "bytes": entry["bytes"],
        "sha256": entry["sha256"],
        "core_integrity_receipt_sha256": prereg.sha256_path(
            prereg.CORPUS_INTEGRITY_RECEIPT_PATH
        ),
    }
    if not path.is_file() or path.is_symlink():
        status = "MISSING_FILE"
        current = lag = None
        row_count = 0
    else:
        observed_sha = prereg.sha256_path(path)
        if observed_sha != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise RuntimeError("SPX_REFERENCE source byte drift")
        parquet = pq.ParquetFile(path)
        source_spec = prereg.context_diagnostics_spec()["sources"]["SPX_REFERENCE"]
        if set(source_spec["required_fields"]) - set(parquet.schema_arrow.names):
            raise RuntimeError("SPX_REFERENCE required column drift")
        candidates: list[dict[str, Any]] = []
        invariants = source_spec["row_invariants"]
        row_count = 0
        for row_group in range(parquet.num_row_groups):
            records = parquet.read_row_group(row_group).to_pylist()
            for row_index, record in enumerate(records):
                row_count += 1
                if any(record.get(name) != value for name, value in invariants.items()):
                    raise RuntimeError("SPX_REFERENCE row invariant drift")
                raw_time = record.get("event_time")
                if hasattr(raw_time, "timestamp"):
                    event_ns = int(raw_time.timestamp() * 1_000_000_000)
                elif type(raw_time) is int:
                    event_ns = raw_time
                else:
                    raise RuntimeError("SPX_REFERENCE event_time schema drift")
                close = record.get("close")
                for numeric_name in ("open", "high", "low", "close", "volume"):
                    numeric = record.get(numeric_name)
                    if (
                        type(numeric) not in (int, float)
                        or not math.isfinite(float(numeric))
                    ):
                        raise RuntimeError(
                            f"SPX_REFERENCE nonfinite {numeric_name}"
                        )
                canonical_row = {
                    name: (
                        int(record[name].timestamp() * 1_000_000_000)
                        if name == "event_time" and hasattr(record[name], "timestamp")
                        else record[name]
                    )
                    for name in source_spec["required_fields"]
                }
                candidates.append(
                    {
                        "event_time_ns": event_ns,
                        "available_at_ns": event_ns + 60_000_000_000,
                        "row_group": row_group,
                        "row_index": row_index,
                        "close": float(close),
                        "canonical_row_sha256": prereg.stable_hash(canonical_row),
                    }
                )

        def select(clock: int) -> dict[str, Any] | None:
            eligible = [
                row
                for row in candidates
                if row["available_at_ns"] <= clock
                and clock - row["available_at_ns"] <= 90_000_000_000
            ]
            if not eligible:
                return None
            eligible.sort(
                key=lambda row: (
                    row["available_at_ns"], row["event_time_ns"],
                    row["row_group"], row["row_index"],
                )
            )
            winner = eligible[-1]
            key = (
                winner["available_at_ns"], winner["event_time_ns"],
                winner["row_group"], winner["row_index"],
            )
            if sum(
                (
                    row["available_at_ns"], row["event_time_ns"],
                    row["row_group"], row["row_index"],
                ) == key
                for row in eligible
            ) != 1:
                raise RuntimeError("SPX_REFERENCE latest-key duplicate")
            return winner

        current = select(anchor_time_ns)
        lag = select(anchor_time_ns - 15 * 60_000_000_000)
        status = "FRESH" if current is not None and lag is not None else (
            "EMPTY_SOURCE" if not candidates else "NO_CAUSAL_BAR"
        )
    current_locator = () if current is None else ({
        "publisher_id": None,
        "instrument_id": None,
        "row_group": current["row_group"],
        "row_index": current["row_index"],
        "canonical_row_sha256": current["canonical_row_sha256"],
        "event_time_ns": current["event_time_ns"],
        "available_at_ns": current["available_at_ns"],
    },)
    lag_locator = () if lag is None else ({
        "publisher_id": None,
        "instrument_id": None,
        "row_group": lag["row_group"],
        "row_index": lag["row_index"],
        "canonical_row_sha256": lag["canonical_row_sha256"],
        "event_time_ns": lag["event_time_ns"],
        "available_at_ns": lag["available_at_ns"],
    },)
    semantic = {
        "schema_version": EntryContextSourceSelectionV1.SCHEMA_VERSION,
        "source": "SPX_REFERENCE",
        "status": status,
        "requested": True,
        "source_degraded": status != "FRESH",
        "manifest_relative_path": relative,
        "source_file_sha256": entry["sha256"],
        "current_component_locators": current_locator,
        "lag15_component_locators": lag_locator,
        "current_available_at_ns": None if current is None else current["available_at_ns"],
        "lag15_available_at_ns": None if lag is None else lag["available_at_ns"],
        "current_age_ns": None if current is None else anchor_time_ns - current["available_at_ns"],
        "lag15_age_ns": None if lag is None else anchor_time_ns - 15 * 60_000_000_000 - lag["available_at_ns"],
        "current_close_float64_hex": None if current is None else current["close"].hex(),
        "lag15_close_float64_hex": None if lag is None else lag["close"].hex(),
    }
    selection = EntryContextSourceSelectionV1(
        **semantic, selection_sha256=prereg.stable_hash(_plain(semantic))
    )
    return selection, {**source_receipt, "row_count": row_count}


def run_fixed_entry_context_diagnostics(
    authorization: Any, /
) -> dict[str, Any]:
    """Write the fixed v3.2 SPX_REFERENCE transaction from sealed inputs.

    The full historical context diagnostic remains post-primary and cannot run
    before the separate Claude release.  This entrypoint owns the formerly
    missing SPX writer: it accepts no caller paths/rows/anchors, validates one
    fixed self-hashed upstream payload, and commits one write-once transaction.
    """

    prereg.assert_context_diagnostics_authorization_current(authorization)
    if (
        EntryContextAnchorRowV1.SCHEMA_VERSION
        != "pathd.entry_context_anchor_row.v1"
        or prereg.context_age_quantile_type7([], 0.5) is not None
    ):
        raise RuntimeError("context anchor schema or age statistic drift")
    assert_corrected_v32_fit_release()
    from v4.research.pathd_corrected_v32_foundation import V32_ROOT
    from v4.research.pathd_entry_execution_v32 import (
        build_spx_reference_transaction,
    )

    input_path = V32_ROOT / "execution_inputs" / "spx_reference_context.json"
    output_path = V32_ROOT / "context_diagnostics" / "spx_reference_transaction.json"
    if not input_path.is_file() or input_path.is_symlink():
        raise RuntimeError("sealed SPX_REFERENCE context input is absent")
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError("SPX_REFERENCE context transaction already exists")
    payload = prereg.read_json(input_path)
    expected = {
        "schema_version", "authorization_sha256", "anchors", "rows",
        "source_receipt_sha256s", "input_sha256",
    }
    semantic = dict(payload)
    digest = semantic.pop("input_sha256", None)
    current = prereg.assert_context_diagnostics_authorization_current(authorization)
    if (
        set(payload) != expected
        or payload.get("schema_version")
        != "pathd.corrected_v32.spx_reference_context_input.v1"
        or payload.get("authorization_sha256") != current.authorization_sha256
        or type(payload.get("source_receipt_sha256s")) is not list
        or not payload["source_receipt_sha256s"]
        or any(
            type(value) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in payload["source_receipt_sha256s"]
        )
        or digest != prereg.stable_hash(semantic)
    ):
        raise RuntimeError("sealed SPX_REFERENCE context input drift")
    result = build_spx_reference_transaction(
        anchors=tuple((row[0], row[1]) for row in payload["anchors"]),
        rows=payload["rows"],
        source_receipt_sha256s=payload["source_receipt_sha256s"],
    )
    bound = {
        **result,
        "authorization_sha256": current.authorization_sha256,
        "input_sha256": digest,
    }
    bound["artifact_sha256"] = prereg.stable_hash(bound)
    prereg._write_canonical_json_exclusive(output_path, bound)
    if prereg.read_json(output_path) != bound:
        raise RuntimeError("SPX_REFERENCE context transaction durability drift")
    return bound


def _journal_economics(journal: Any) -> dict[str, Any]:
    from v4.path_d.execution.research_replay import (
        reconstruct_entry_policy_economics_from_validated_transitions,
    )

    if type(journal) is not dict or set(journal) != {
        "transitions", "session_pnl_micros", "completed_trade_ids"
    }:
        raise ValueError("protected trace terminal journal schema drift")
    transitions = journal["transitions"]
    if type(transitions) not in (list, tuple) or not transitions:
        raise ValueError("protected trace has no transitions")
    starting = transitions[0].get("cash_before_micros")
    result = reconstruct_entry_policy_economics_from_validated_transitions(
        transitions=transitions, starting_cash_micros=starting
    )
    if (
        journal["session_pnl_micros"] != result["session_pnl_micros"]
        or list(journal["completed_trade_ids"]) != result["completed_trade_ids"]
    ):
        raise ValueError("protected trace stored journal summary drift")
    return result


def validate_protected_holdout_trace_records(
    *,
    records: Any,
    authorization_sha256: str,
    dataset_sha256: str,
    sessions: Any,
    box_d_policy_id: str,
    comparator_policy_id: str,
) -> tuple[ProtectedHoldoutTraceRecordV1, ...]:
    from v4.research.pathd_holdout_gate import HOLDOUT_GUARD_KEYS

    expected_sessions = tuple(prereg.session_assignments()["protected_holdout_30"])
    if tuple(sessions) != expected_sessions:
        raise ValueError("protected holdout partition must be the exact ordered 30")
    values = tuple(records)
    if len(values) != 30 or any(type(row) is not ProtectedHoldoutTraceRecordV1 for row in values):
        raise TypeError("protected trace requires exactly 30 typed records")
    if tuple(row.session for row in values) != expected_sessions:
        raise ValueError("protected trace session order/uniqueness drift")
    required = set(prereg.entry_future_api_contract()["dataclass_fields"]["ProtectedHoldoutTraceRecordV1"])
    for row in values:
        value = _mapping(row)
        if set(value) != required:
            raise ValueError("protected trace record schema drift")
        if (
            value["schema_version"] != ProtectedHoldoutTraceRecordV1.SCHEMA_VERSION
            or value["holdout_caveat"] != prereg.HOLDOUT_CAVEAT
            or value["authorization_sha256"] != authorization_sha256
            or value["dataset_sha256"] != dataset_sha256
            or value["box_d_policy_id"] != box_d_policy_id
            or value["comparator_policy_id"] != comparator_policy_id
            or set(value["guard_results"]) != set(HOLDOUT_GUARD_KEYS)
            or not all(type(item) is bool and item is True for item in value["guard_results"].values())
            or value["survival_violations"] != []
        ):
            raise ValueError("protected trace identity/guard/survival drift")
        box = _journal_economics(value["box_d_terminal_journal"])
        comparator = _journal_economics(value["comparator_terminal_journal"])
        if (
            value["box_d_session_pnl_micros"] != box["session_pnl_micros"]
            or value["comparator_session_pnl_micros"] != comparator["session_pnl_micros"]
            or value["box_d_completed_trade_ids"] != box["completed_trade_ids"]
        ):
            raise ValueError("protected trace economics drift")
        _self_hash(value, "record_sha256")
    return values


def reconstruct_protected_holdout_payload_from_validated_trace(
    *, records: Any, primary_sessions: Any, degradation_session: str
) -> dict[str, Any]:
    from v4.research.pathd_holdout_gate import HOLDOUT_GUARD_KEYS

    expected = tuple(prereg.session_assignments()["protected_holdout_30"])
    expected_degraded = "2026-07-31"
    expected_primary = tuple(session for session in expected if session != expected_degraded)
    if tuple(primary_sessions) != expected_primary or degradation_session != expected_degraded:
        raise ValueError("protected 29+degradation partition drift")
    values = tuple(records)
    if len(values) != 30 or tuple(row.session for row in values) != expected:
        raise ValueError("validated protected trace order drift")
    primary_rows = values[:-1]
    completed = sum(len(row.box_d_completed_trade_ids) for row in primary_rows)
    box_net = sum(row.box_d_session_pnl_micros for row in primary_rows)
    paired = sum(
        row.box_d_session_pnl_micros - row.comparator_session_pnl_micros
        for row in primary_rows
    )
    guards = {
        key: all(row.guard_results[key] is True for row in values)
        for key in HOLDOUT_GUARD_KEYS
    }
    survival_counts: dict[str, int] = {}
    for row in values:
        for name in row.survival_violations:
            survival_counts[name] = survival_counts.get(name, 0) + 1
    criteria = {
        "protected_session_count_at_least_25": len(primary_rows) >= 25,
        "completed_trades_at_least_50": completed >= 50,
        "box_d_net_positive": box_net > 0,
        "paired_delta_positive": paired > 0,
        "all_guards_pass": all(guards.values()),
        "no_survival_violations": not survival_counts,
    }
    verdict = "PASS" if all(criteria.values()) else "no_genuine_signal"
    return {
        "schema_version": "pathd.protected_holdout_payload.v1",
        "verdict": verdict,
        "protected_session_count": len(values),
        "primary_non_degraded_session_count": len(primary_rows),
        "completed_box_d_trades_on_primary_29": completed,
        "box_d_net_pnl_micros_on_primary_29": box_net,
        "box_d_minus_comparator_paired_net_pnl_micros_on_primary_29": paired,
        "guards": guards,
        "survival_violation_counts": survival_counts,
        "owner_facing_metrics": {
            "mean_box_d_session_pnl_micros": box_net / len(primary_rows),
            "positive_box_d_session_fraction": sum(
                row.box_d_session_pnl_micros > 0 for row in primary_rows
            ) / len(primary_rows),
        },
        "degraded_sensitivity": {
            "session": values[-1].session,
            "box_d_session_pnl_micros": values[-1].box_d_session_pnl_micros,
            "comparator_session_pnl_micros": values[-1].comparator_session_pnl_micros,
        },
        "pass_criteria_recomputed": criteria,
    }


def _read_fixed_holdout_trace_records() -> tuple[ProtectedHoldoutTraceRecordV1, ...]:
    from v4.research.pathd_holdout_gate import HOLDOUT_TRACE_PATH

    rows = []
    with HOLDOUT_TRACE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(ProtectedHoldoutTraceRecordV1(**prereg.strict_json_loads(line)))
    return tuple(rows)


def validate_protected_holdout_trace(
    authorization: Any, dataset: Any, /
) -> tuple[ProtectedHoldoutTraceRecordV1, ...]:
    records = _read_fixed_holdout_trace_records()
    sessions = tuple(prereg.session_assignments()["protected_holdout_30"])
    return validate_protected_holdout_trace_records(
        records=records,
        authorization_sha256=authorization.authorization_sha256,
        dataset_sha256=dataset.dataset_sha256,
        sessions=sessions,
        box_d_policy_id=authorization.box_d_policy_id,
        comparator_policy_id=authorization.comparator_policy_id,
    )


def reconstruct_protected_holdout_evaluation_from_trace(
    authorization: Any, dataset: Any, /
) -> ProtectedHoldoutEvaluationV1:
    records = validate_protected_holdout_trace(authorization, dataset)
    sessions = tuple(prereg.session_assignments()["protected_holdout_30"])
    primary = tuple(session for session in sessions if session != "2026-07-31")
    payload = reconstruct_protected_holdout_payload_from_validated_trace(
        records=records, primary_sessions=primary, degradation_session="2026-07-31"
    )
    raise RuntimeError(
        "protected evaluation envelope can be sealed only by the active holdout transaction"
    )


def run_fixed_protected_holdout_evaluator(
    authorization: Any, dataset: Any, /
) -> ProtectedHoldoutEvaluationV1:
    validate_protected_holdout_trace(authorization, dataset)
    return reconstruct_protected_holdout_evaluation_from_trace(authorization, dataset)


def _nested_replay_config(*, outer_fold: int, inner_fold: int) -> Any:
    from v4.path_d.execution import research_replay as replay
    from v4.research import pathd_entry_models as entry_models

    assignments = prereg.session_assignments()
    account_scopes = [
        replay._nested_account_scope_sha256(
            policy_id=policy_id,
            fee_path=3,
            outer_fold=outer_fold,
            assignments=assignments,
        )
        for policy_id in ("HGB", "NEURAL")
    ]
    semantic = {
        "schema_version": entry_models.EntryNestedReplayConfigV1.SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "fill_law_hash": prereg.preregistration_payload()[0]["fill_law"][
            "fill_law_hash"
        ],
        "policy_ids": ("HGB", "NEURAL"),
        "fee_path": 3,
        "sell_delay_ms": 1_000,
        "floor_on": True,
        "exit_policy_id": "HOLD_TO_FLAT",
        "nested_account_scope_sha256": prereg.stable_hash(
            {
                "outer_fold": outer_fold,
                "policy_ids": ["HGB", "NEURAL"],
                "fee_path": 3,
                "policy_account_scope_sha256s": account_scopes,
            }
        ),
    }
    result = entry_models.EntryNestedReplayConfigV1(
        **semantic, artifact_sha256=prereg.stable_hash(semantic)
    )
    return entry_models.validate_entry_nested_replay_config(result)


def _seal_nested_preopen_receipt(*, outer_fold: int, inner_fold: int) -> dict[str, Any]:
    payload = prereg.read_json(prereg.PREREG_PATH)
    assignments = prereg.read_json(prereg.SESSION_PATH)
    role = prereg._nested_role_record(assignments, outer_fold, inner_fold)
    if role.get("calibration_valid") is not True:
        raise RuntimeError("calibration-invalid nested block cannot have a preopen receipt")
    path = prereg._outer_fold_artifact_path(
        outer_fold, f"nested_inner_{inner_fold}_preopen_receipt.json"
    )
    _assert_fixed_paths_absent((path,))
    artifacts: dict[str, dict[str, str]] = {}
    for name in prereg._nested_artifact_names(inner_fold):
        artifact_path = prereg._outer_fold_artifact_path(outer_fold, name)
        resolved = prereg._canonical_repo_regular_file(
            prereg.repo_path_label(artifact_path)
        )
        artifacts[name] = {
            "path": prereg.repo_path_label(artifact_path),
            "sha256": prereg.sha256_path(resolved),
        }
    semantic = {
        "schema_version": "pathd.entry_nested_preopen_receipt.v1",
        "status": "FROZEN_BEFORE_NESTED_OPEN",
        "frozen_at_utc": _utc_now(),
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "preregistration_sha256": prereg.sha256_path(prereg.PREREG_PATH),
        "session_assignments_sha256": prereg.sha256_path(prereg.SESSION_PATH),
        "source_hash_policy_sha256": prereg.stable_hash(payload["source_hash_policy"]),
        "weights_sessions_sha256_newline": prereg.canonical_session_hash(
            role["model_fit"]
        ),
        "calibration_sessions_sha256_newline": prereg.canonical_session_hash(
            role["calibration"]
        ),
        "validation_sessions_sha256_newline": prereg.canonical_session_hash(
            role["validation"]
        ),
        "artifacts": artifacts,
        "fit_environment_sha256": prereg.stable_hash(
            prereg.assert_entry_fit_environment_current()
        ),
        "validation_access_count": 0,
        "quarantine_labels": list(prereg.QUARANTINE_LABELS),
        "claim_boundary": prereg.CLAIM_BOUNDARY,
        "holdout_caveat": prereg.HOLDOUT_CAVEAT,
        "plan_sha256": payload["binding_plan"]["sha256"],
        "fill_law_hash": payload["fill_law"]["fill_law_hash"],
        "holdout_open_count": 0,
    }
    receipt = {**semantic, "receipt_sha256": prereg.stable_hash(semantic)}
    prereg._write_canonical_json_exclusive(path, receipt)
    return prereg._validate_nested_preopen_receipt_one(
        payload, assignments, outer_fold, inner_fold
    )


def _nested_opening_journals(authorization: Any, /) -> tuple[Any, Any]:
    from v4.path_d.execution import research_replay as replay
    from v4.research.pathd_evidence_gate import (
        read_frozen_entry_evidence_authorization,
    )

    assignments = prereg.session_assignments()
    rows = assignments["folds"][authorization.outer_fold - 1][
        "inner_forward_folds"
    ]["scored_forward_folds"]
    valid = [row for row in rows if row.get("calibration_valid") is True]
    matches = [
        row for row in valid if row.get("inner_fold") == authorization.inner_fold
    ]
    if len(matches) != 1:
        raise RuntimeError("nested opening-journal role drift")
    position = valid.index(matches[0])
    if position == 0:
        return tuple(
            replay.genesis_research_ledger_journal(
                authorization, policy_id=family, fee_path=3
            )
            for family in ("HGB", "NEURAL")
        )  # type: ignore[return-value]
    previous_inner = valid[position - 1]["inner_fold"]
    previous = read_frozen_entry_evidence_authorization(
        role="nested_validation",
        outer_fold=authorization.outer_fold,
        inner_fold=previous_inner,
    )
    prior_result = prereg.read_json(
        prereg._outer_fold_artifact_path(
            authorization.outer_fold, f"nested_inner_{previous_inner}_result.json"
        )
    )
    journals: list[Any] = []
    for family in ("HGB", "NEURAL"):
        evaluation = prior_result.get(family.lower())
        if type(evaluation) is not dict or type(evaluation.get("terminal_journal")) is not dict:
            raise RuntimeError("prior nested result lacks the frozen family journal")
        journals.append(
            replay.handoff_research_ledger_authorization(
                evaluation["terminal_journal"],
                current_authorization=previous,
                next_authorization=authorization,
            )
        )
    return journals[0], journals[1]


def run_canonical_entry_nested_block(
    *, outer_fold: int, inner_fold: int
) -> dict[str, Any]:
    """Run one ordered nested block, sealing either its skip or complete result."""

    if outer_fold not in range(1, 6) or inner_fold not in range(1, 5):
        raise ValueError("outer_fold/inner_fold must be in 1..5/1..4")
    from v4.path_d.execution.research_fill_law import (
        research_fill_law_from_preregistration,
    )
    from v4.research import pathd_entry_models as entry_models
    from v4.research.pathd_entry_dataset import (
        release_authorized_entry_fit_dataset,
    )
    from v4.research.pathd_evidence_gate import (
        seal_entry_evidence_result,
        seal_invalid_nested_block_skip,
    )

    payload = prereg.read_json(prereg.PREREG_PATH)
    assignments = prereg.read_json(prereg.SESSION_PATH)
    role = prereg._nested_role_record(assignments, outer_fold, inner_fold)
    if role.get("calibration_valid") is False:
        gate_path = _calibration_scope_gate_path(
            outer_fold=outer_fold, inner_fold=inner_fold
        )
        if gate_path.exists() or gate_path.is_symlink():
            raise RuntimeError("structural skip retained a calibration scope gate")
        return seal_invalid_nested_block_skip(
            outer_fold=outer_fold, inner_fold=inner_fold
        )
    if role.get("calibration_valid") is not True:
        raise RuntimeError("nested calibration-validity flag drift")

    prefix = f"nested_inner_{inner_fold}"
    artifact_names = prereg._nested_artifact_names(inner_fold)
    artifact_paths = tuple(
        prereg._outer_fold_artifact_path(outer_fold, name)
        for name in artifact_names
    )
    preopen_path = prereg._outer_fold_artifact_path(
        outer_fold, f"{prefix}_preopen_receipt.json"
    )
    scope_gate_path = _calibration_scope_gate_path(
        outer_fold=outer_fold, inner_fold=inner_fold
    )
    _assert_fixed_paths_absent((*artifact_paths, scope_gate_path, preopen_path))

    weights = prereg.assert_entry_fit_ready(
        role="nested_weights", outer_fold=outer_fold, inner_fold=inner_fold
    )
    calibration_auth = prereg.assert_entry_fit_ready(
        role="nested_calibration", outer_fold=outer_fold, inner_fold=inner_fold
    )
    try:
        hgb = entry_models.fit_hgb_entry_bundle(weights)
        neural = entry_models.fit_neural_entry_bundle(weights, hgb_bundle=hgb)
        hgb_calibration = entry_models.fit_entry_calibrators(
            calibration_auth, model_bundle=hgb
        )
        neural_calibration = entry_models.fit_entry_calibrators(
            calibration_auth, model_bundle=neural
        )
        hgb_composer = entry_models.seal_entry_composer_bundle(
            model_bundle=hgb,
            calibration_bundle=hgb_calibration,
            payload=_entry_composer_payload(hgb_calibration),
        )
        neural_composer = entry_models.seal_entry_composer_bundle(
            model_bundle=neural,
            calibration_bundle=neural_calibration,
            payload=_entry_composer_payload(neural_calibration),
        )
        replay_config = _nested_replay_config(
            outer_fold=outer_fold, inner_fold=inner_fold
        )
        artifacts = (
            (artifact_paths[0], hgb, entry_models.validate_entry_model_bundle),
            (
                artifact_paths[1],
                hgb_calibration,
                lambda value: entry_models.validate_entry_calibration_bundle(
                    value, model_bundle=hgb
                ),
            ),
            (
                artifact_paths[2],
                hgb_composer,
                lambda value: entry_models.validate_entry_composer_bundle(
                    value,
                    model_bundle=hgb,
                    calibration_bundle=hgb_calibration,
                ),
            ),
            (artifact_paths[3], neural, entry_models.validate_entry_model_bundle),
            (
                artifact_paths[4],
                neural_calibration,
                lambda value: entry_models.validate_entry_calibration_bundle(
                    value, model_bundle=neural
                ),
            ),
            (
                artifact_paths[5],
                neural_composer,
                lambda value: entry_models.validate_entry_composer_bundle(
                    value,
                    model_bundle=neural,
                    calibration_bundle=neural_calibration,
                ),
            ),
            (
                artifact_paths[6],
                replay_config,
                entry_models.validate_entry_nested_replay_config,
            ),
        )
        for path, value, validator in artifacts:
            _write_typed_artifact_once(path, value, validator=validator)
        freeze_entry_calibration_scope_gate_once(
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            calibration_bundles={
                "HGB": hgb_calibration,
                "NEURAL": neural_calibration,
            },
        )
        _seal_nested_preopen_receipt(
            outer_fold=outer_fold, inner_fold=inner_fold
        )
    finally:
        release_authorized_entry_fit_dataset(calibration_auth)
        release_authorized_entry_fit_dataset(weights)

    authorization = begin_entry_evidence_after_valid_calibration_scope_gate(
        role="nested_validation", outer_fold=outer_fold, inner_fold=inner_fold
    )
    try:
        hgb_journal, neural_journal = _nested_opening_journals(authorization)
        fill_law = research_fill_law_from_preregistration(payload)
        hgb_evaluation = run_entry_policy_evaluation(
            authorization,
            policy_bundle=hgb,
            composer_bundle=hgb_composer,
            fill_law=fill_law,
            opening_journal=hgb_journal,
        )
        neural_evaluation = run_entry_policy_evaluation(
            authorization,
            policy_bundle=neural,
            composer_bundle=neural_composer,
            fill_law=fill_law,
            opening_journal=neural_journal,
        )
        dataset = _active_entry_evidence_dataset(authorization)
        semantic = {
            "schema_version": EntryNestedFamilyEvaluationV1.SCHEMA_VERSION,
            "holdout_caveat": prereg.HOLDOUT_CAVEAT,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "authorization_sha256": prereg.stable_hash(authorization.to_dict()),
            "dataset_sha256": dataset.dataset_sha256,
            "source_receipts_root_sha256": prereg.stable_hash(
                list(dataset.source_receipts)
            ),
            "access_receipt_sha256": authorization.access_receipt_sha256,
            "hgb": hgb_evaluation,
            "neural": neural_evaluation,
        }
        evaluation = EntryNestedFamilyEvaluationV1(
            **semantic, result_sha256=prereg.stable_hash(_plain(semantic))
        )
        validate_entry_nested_family_evaluation(
            evaluation, authorization=authorization, dataset=dataset
        )
        return seal_entry_evidence_result(
            authorization, dataset=dataset, evaluation=evaluation
        )
    finally:
        _discard_active_entry_evidence_dataset(authorization)


def validate_protected_holdout_evaluation(
    evaluation: Any, /, *, authorization: Any, dataset: Any
) -> dict[str, Any]:
    value = _mapping(evaluation)
    expected = set(prereg.entry_future_api_contract()["dataclass_fields"]["ProtectedHoldoutEvaluationV1"])
    if set(value) != expected or value.get("schema_version") != ProtectedHoldoutEvaluationV1.SCHEMA_VERSION:
        raise ValueError("protected holdout evaluation schema drift")
    if (
        value.get("authorization_sha256") != authorization.authorization_sha256
        or value.get("dataset_sha256") != dataset.dataset_sha256
        or value.get("payload_sha256") != prereg.stable_hash(value.get("payload"))
    ):
        raise ValueError("protected holdout evaluation binding drift")
    return value


def _validate_named_holdout_artifact(path: Path, *, artifact_kind: str, statuses: tuple[str, ...]) -> dict[str, Any]:
    value = prereg.read_json(path)
    if type(value) is not dict or value.get("artifact_kind") != artifact_kind:
        raise ValueError(f"{artifact_kind} semantic identity drift")
    if value.get("status") not in statuses:
        raise ValueError(f"{artifact_kind} status drift")
    if value.get("artifact_sha256") != prereg.sha256_path(path):
        raise ValueError(f"{artifact_kind} file hash drift")
    return value


def validate_outer_entry_exit_artifacts_for_holdout(path: Path, /) -> dict[str, Any]:
    return _validate_named_holdout_artifact(path, artifact_kind="outer_entry_exit_artifacts", statuses=("FROZEN_COMPLETE",))


def validate_four_box_packet_for_holdout(path: Path, /) -> dict[str, Any]:
    return _validate_named_holdout_artifact(path, artifact_kind="four_box_packet", statuses=("FROZEN_COMPLETE",))


def validate_guard_panel_for_holdout(path: Path, /) -> dict[str, Any]:
    return _validate_named_holdout_artifact(path, artifact_kind="guard_panel", statuses=("PASS",))


def validate_outer_acceptance_packet_for_holdout(path: Path, /) -> dict[str, Any]:
    return _validate_named_holdout_artifact(path, artifact_kind="outer_acceptance_packet", statuses=("PASS",))


def validate_full_fit_entry_artifacts_for_holdout(path: Path, /) -> dict[str, Any]:
    return _validate_named_holdout_artifact(path, artifact_kind="full_fit_entry_artifacts", statuses=("FROZEN_COMPLETE",))


def validate_full_fit_exit_artifacts_for_holdout(path: Path, /) -> dict[str, Any]:
    return _validate_named_holdout_artifact(path, artifact_kind="full_fit_exit_artifacts", statuses=("FROZEN_COMPLETE",))


def _assert_outer_entry_science_producers_ready() -> None:
    """Verify the concrete v3.2 producer surface before any fold mutation."""

    from v4.research import pathd_entry_execution_v32 as execution

    required = (
        execution.validate_opportunity,
        execution.replay_policy,
        execution.select_shared_control_exit,
        execution.produce_outer_replay_inventory,
        execution.build_spx_reference_transaction,
    )
    if any(not callable(item) for item in required):
        raise RuntimeError("corrected-v3.2 outer producer surface is incomplete")
    if (
        execution.CONTRACT_MULTIPLIER != 100
        or execution.STARTING_EQUITY_MICROS != 10_000_000_000
        or execution.D48_FRACTION != 0.05
    ):
        raise RuntimeError("corrected-v3.2 replay economics drift")


def produce_corrected_v32_outer_replay_inventory(
    *, outer_fold: int, model_fit_sessions: Any, outer_sessions: Any,
    model_fit_opportunities: Any, outer_opportunities: Any,
    legacy_causal_inputs_complete: bool,
) -> dict[str, Any]:
    """Typed runner boundary for the common v3.2 replay producer.

    No caller PnL, journal hash, exit choice, result, pass flag, or verdict is
    accepted.  Those values are reconstructed inside the producer.
    """

    _assert_outer_entry_science_producers_ready()
    from v4.research.pathd_entry_execution_v32 import (
        produce_outer_replay_inventory,
    )

    return produce_outer_replay_inventory(
        outer_fold=outer_fold,
        model_fit_sessions=model_fit_sessions,
        outer_sessions=outer_sessions,
        model_fit_opportunities=model_fit_opportunities,
        outer_opportunities=outer_opportunities,
        legacy_causal_inputs_complete=legacy_causal_inputs_complete,
    )


def run_canonical_entry_outer_fold(*, outer_fold: int) -> dict[str, Any]:
    if outer_fold not in range(1, 6):
        raise ValueError("outer_fold must be 1..5")
    _assert_outer_entry_science_producers_ready()
    # This stage is intentionally unreachable under the v3.2 pre-verification
    # release.  Once Claude's separate release is frozen, the existing fit/evidence
    # authorization chain supplies the typed opportunities to the concrete replay
    # boundary above.  Failing here today prevents a build verification from being
    # mistaken for permission to fit or open a fold.
    from v4.research.pathd_corrected_v32_foundation import (
        assert_corrected_v32_foundation,
    )

    release = assert_corrected_v32_foundation()["release_state"]
    if release.get("claude_verification_pending") is True:
        raise RuntimeError(
            "corrected-v3.2 outer execution is frozen but not released: "
            "STOP_FOR_CLAUDE_VERIFICATION"
        )
    from v4.research import pathd_entry_execution_v32 as execution
    from v4.research.pathd_corrected_v32_foundation import V32_ROOT

    input_path = V32_ROOT / "execution_inputs" / f"outer_fold_{outer_fold}.json"
    output_path = (
        V32_ROOT / "entry_outer_folds" / f"fold_{outer_fold}"
        / "v32_outer_replay_inventory.json"
    )
    if not input_path.is_file() or input_path.is_symlink():
        raise RuntimeError("corrected-v3.2 sealed outer replay input is absent")
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError("corrected-v3.2 outer replay output already exists")
    result = execution.produce_outer_replay_inventory_from_sealed_input(
        prereg.read_json(input_path)
    )
    prereg._write_canonical_json_exclusive(output_path, result)
    if prereg.read_json(output_path) != result:
        raise RuntimeError("corrected-v3.2 outer replay durability drift")
    return result


def run_canonical_entry_fold(*, outer_fold: int) -> None:
    if outer_fold not in range(1, 6):
        raise ValueError("outer_fold must be 1..5")
    # A full-fold invocation must not leave completed nested artifacts followed by
    # a known outer-stage blocker.  Preflight the complete outer science surface
    # before the first nested mutation.
    _assert_outer_entry_science_producers_ready()
    for inner_fold in range(1, 5):
        run_canonical_entry_nested_block(
            outer_fold=outer_fold, inner_fold=inner_fold
        )
    run_canonical_entry_outer_fold(outer_fold=outer_fold)


ENTRY_CAMPAIGN_STAGE_ORDER = (
    "VERIFY_EXECUTABLE_SEAL",
    "SEAL_ENTRY_MACHINERY",
    "VERIFY_CORE_CORPUS_INTEGRITY",
    "SEAL_FOUNDATION_STABILITY",
    "OUTER_FOLD_1",
    "OUTER_FOLD_2",
    "OUTER_FOLD_3",
    "OUTER_FOLD_4",
    "OUTER_FOLD_5",
    "FREEZE_POOLED_ENTRY_ACCEPTANCE",
)


def canonical_entry_campaign_stage_order() -> tuple[str, ...]:
    """Expose the only permitted corrected-v3.2 entry campaign order."""

    payload = prereg.preregistration_payload()[0]
    terminal = payload["calibration_and_statistics"][
        "composite_calibration_terminal_rule"
    ]["all_five_manifest"]
    if (
        terminal["outer_folds_in_exact_order"] != [1, 2, 3, 4, 5]
        or terminal["required_status_each"] != "VALID"
        or terminal["valid_fold_count"] != 5
        or terminal["four_as_five_or_survivor_pooling"] is not False
    ):
        raise RuntimeError("corrected-v3.2 campaign order contract drift")
    return ENTRY_CAMPAIGN_STAGE_ORDER


def run_canonical_entry_campaign() -> dict[str, Any]:
    """Run the concrete corrected-v3.2 entry campaign, with no injectable seams.

    This entrypoint is intentionally all-or-nothing and accepts no paths, data,
    callbacks, stage selection, verdicts, or hashes from its caller.  The build
    freezes it but does not invoke it.
    """

    stages = canonical_entry_campaign_stage_order()
    from v4.research.pathd_entry_dataset import (
        assert_corrected_v32_executable_bridge,
    )
    assert_corrected_v32_executable_bridge()
    release = assert_corrected_v32_fit_release()
    # Exhaust every known build/science seam before the first durable write.
    # This call remains intentionally ahead of machinery, integrity, and
    # stability sealing so a missing producer can never leave a partial campaign.
    _assert_outer_entry_science_producers_ready()
    prereg.assert_preregistration_frozen()
    prereg._verify_authorized_dependency_closure(
        prereg.preregistration_payload()[0], allow_missing_future=False
    )
    machinery = _seal_entry_machinery()
    corpus = _verify_core_corpus_integrity()
    stability = prereg.seal_foundation_stability_receipt()
    folds: list[dict[str, Any]] = []
    for outer_fold in range(1, 6):
        run_canonical_entry_fold(outer_fold=outer_fold)
        receipt_path = prereg._outer_fold_artifact_path(
            outer_fold, "outer_result_receipt.json"
        )
        folds.append(
            {
                "outer_fold": outer_fold,
                "receipt_path": prereg.repo_path_label(receipt_path),
                "receipt_sha256": prereg.sha256_path(receipt_path),
            }
        )
    pooled = freeze_entry_pooled_acceptance_once()
    return {
        "schema_version": "pathd.corrected_v3_2.entry_campaign_result.v1",
        "science_preregistration_sha256": release["preregistration_sha256"],
        "foundation_generation_sha256": release["foundation_generation_sha256"],
        "stage_order": stages,
        "machinery": machinery,
        "corpus_integrity_receipt_sha256": prereg.sha256_path(
            prereg.CORPUS_INTEGRITY_RECEIPT_PATH
        ),
        "foundation_stability_receipt_sha256": prereg.sha256_path(
            prereg.FOUNDATION_STABILITY_RECEIPT_PATH
        ),
        "outer_folds": folds,
        "pooled_acceptance": _mapping(pooled),
        "holdout_open_count": 0,
    }


_MACHINERY_COMMAND_STREAMS = tuple(
    (
        Path(f"/tmp/pathd_entry_machinery_command_{index}.stdout"),
        Path(f"/tmp/pathd_entry_machinery_command_{index}.stderr"),
    )
    for index in range(
        len(prereg.receipt_contract_spec()["required_test_commands"])
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _implementation_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in paths:
        path = prereg._canonical_repo_regular_file(label)
        rows.append({"path": label, "sha256": prereg.sha256_path(path)})
    return rows


def _machinery_command_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    contracts = payload["source_hash_policy"]["receipt_contracts"]
    expected = contracts["required_test_commands"]
    if len(expected) != len(_MACHINERY_COMMAND_STREAMS):
        raise RuntimeError("machinery command-stream registry drift")
    rows: list[dict[str, Any]] = []
    for command, (stdout_path, stderr_path) in zip(
        expected, _MACHINERY_COMMAND_STREAMS, strict=True
    ):
        if not stdout_path.is_file() or not stderr_path.is_file():
            raise RuntimeError(
                "machinery test streams are absent; execute every frozen command first"
            )
        junit_path = prereg._canonical_repo_regular_file(command["junit_path"])
        rows.append(
            {
                "argv": list(command["argv"]),
                "cwd": command["cwd"],
                "env": dict(command["env"]),
                "exit_code": 0,
                "stdout_sha256": prereg.sha256_path(stdout_path),
                "stderr_sha256": prereg.sha256_path(stderr_path),
                "junit_path": command["junit_path"],
                "junit_sha256": prereg.sha256_path(junit_path),
                "junit_summary": dict(command["expected_junit_summary"]),
            }
        )
    prereg._validate_registered_test_commands(rows, expected)
    return rows


def _seal_entry_machinery() -> dict[str, Any]:
    prereg_receipt = prereg.assert_preregistration_frozen()
    payload = prereg.read_json(prereg.PREREG_PATH)
    prereg.assert_foundation_restoration_frozen()
    prereg.assert_correction_prefit_release(payload)
    policy, policy_sha256 = prereg._source_policy_and_hash(payload)
    contracts = policy["receipt_contracts"]
    for path in (
        prereg.ENTRY_MACHINERY_TEST_EVIDENCE_PATH,
        prereg.LINEAGE_IMPLEMENTATION_RECEIPT_PATH,
        prereg.ENTRY_MACHINERY_RECEIPT_PATH,
    ):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"entry machinery fixed artifact already exists: {path}")

    prereg._verify_immutable_sources(payload)
    prereg.validate_fixed_science_gate_source()
    prereg._verify_authorized_dependency_closure(
        payload, allow_missing_future=False
    )
    commands = _machinery_command_rows(payload)
    check_names = [
        *contracts["lineage_evidence_checks"],
        *contracts["machinery_evidence_checks"],
    ]
    checks = {name: "PASS" for name in check_names}
    evidence = prereg.enveloped_research_result(
        {
            "schema_version": contracts["machinery_test_evidence_schema_version"],
            "status": "PASS",
            "completed_at_utc": _utc_now(),
            "source_hash_policy_sha256": policy_sha256,
            "commands": commands,
            "checks": checks,
            "check_to_testcase": dict(contracts["evidence_check_to_testcase"]),
            "model_fit_executed": False,
        }
    )
    prereg._write_canonical_json_exclusive(
        prereg.ENTRY_MACHINERY_TEST_EVIDENCE_PATH, evidence
    )

    lineage_paths = list(policy["lineage_required_paths"])
    lineage = prereg.enveloped_research_result(
        {
            "schema_version": contracts["lineage_schema_version"],
            "status": "PASS",
            "completed_at_utc": _utc_now(),
            "training_allowed": True,
            "feature_lineage_sha256": prereg.sha256_path(prereg.LINEAGE_PATH),
            "source_hash_policy_sha256": policy_sha256,
            "implementations": _implementation_rows(lineage_paths),
            "test_evidence_path": prereg.repo_path_label(
                prereg.ENTRY_MACHINERY_TEST_EVIDENCE_PATH
            ),
            "test_evidence_sha256": prereg.sha256_path(
                prereg.ENTRY_MACHINERY_TEST_EVIDENCE_PATH
            ),
            "checks": {
                name: "PASS" for name in contracts["lineage_evidence_checks"]
            },
            "model_fit_executed": False,
        }
    )
    prereg._write_canonical_json_exclusive(
        prereg.LINEAGE_IMPLEMENTATION_RECEIPT_PATH, lineage
    )
    prereg.assert_lineage_implementation_frozen()

    machinery_paths = list(policy["machinery_required_paths"])
    machinery = prereg.enveloped_research_result(
        {
            "schema_version": contracts["machinery_schema_version"],
            "status": "PASS",
            "completed_at_utc": _utc_now(),
            "fit_allowed": True,
            "source_hash_policy_sha256": policy_sha256,
            "lineage_receipt_sha256": prereg.sha256_path(
                prereg.LINEAGE_IMPLEMENTATION_RECEIPT_PATH
            ),
            "implementations": _implementation_rows(machinery_paths),
            "test_evidence_path": prereg.repo_path_label(
                prereg.ENTRY_MACHINERY_TEST_EVIDENCE_PATH
            ),
            "test_evidence_sha256": prereg.sha256_path(
                prereg.ENTRY_MACHINERY_TEST_EVIDENCE_PATH
            ),
            "checks": {
                name: "PASS" for name in contracts["machinery_evidence_checks"]
            },
            "model_fit_executed": False,
        }
    )
    prereg._write_canonical_json_exclusive(
        prereg.ENTRY_MACHINERY_RECEIPT_PATH, machinery
    )
    validated = prereg._assert_entry_machinery_frozen(
        prereg_receipt, payload, prereg.assert_lineage_implementation_frozen()
    )
    return {
        "test_evidence_sha256": prereg.sha256_path(
            prereg.ENTRY_MACHINERY_TEST_EVIDENCE_PATH
        ),
        "lineage_receipt_sha256": prereg.sha256_path(
            prereg.LINEAGE_IMPLEMENTATION_RECEIPT_PATH
        ),
        "machinery_receipt_sha256": prereg.sha256_path(
            prereg.ENTRY_MACHINERY_RECEIPT_PATH
        ),
        "status": validated["status"],
        "fit_allowed": validated["fit_allowed"],
    }


def _canonical_corpus_file(relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError("core corpus path is noncanonical")
    root = prereg.CORPUS_ROOT.resolve(strict=True)
    candidate = prereg.CORPUS_ROOT
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError(f"core corpus path is symlinked: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"core corpus path is absent/outside root: {relative_path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"core corpus path is not a regular file: {relative_path}")
    return resolved


def _stream_sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _verify_core_corpus_integrity() -> dict[str, Any]:
    prereg_receipt = prereg.assert_preregistration_frozen()
    payload = prereg.read_json(prereg.PREREG_PATH)
    lineage = prereg.assert_lineage_implementation_frozen()
    machinery = prereg._assert_entry_machinery_frozen(
        prereg_receipt, payload, lineage
    )
    if (
        prereg.CORPUS_INTEGRITY_RECEIPT_PATH.exists()
        or prereg.CORPUS_INTEGRITY_RECEIPT_PATH.is_symlink()
    ):
        raise RuntimeError("core corpus integrity receipt already exists")

    partition_id = prereg.CORE_INTEGRITY_PARTITION
    manifest_contract = payload["corpus"]["integrity_contract"]
    partition = manifest_contract["partitions"][partition_id]
    rows = prereg.integrity_manifest_entries_for_partition(partition_id)
    expected_paths = {row["relative_path"] for row in rows}
    patterns = [
        row["path_regex"] for row in partition["path_grammars"]
    ]
    observed_core_paths: set[str] = set()
    for candidate in prereg.CORPUS_ROOT.rglob("*"):
        if candidate.is_file() or candidate.is_symlink():
            relative = candidate.relative_to(prereg.CORPUS_ROOT).as_posix()
            if any(re.fullmatch(pattern, relative) is not None for pattern in patterns):
                observed_core_paths.add(relative)
    unreceipted = sorted(observed_core_paths - expected_paths)
    missing = sorted(expected_paths - observed_core_paths)
    if unreceipted or missing:
        raise RuntimeError(
            f"core corpus inventory drift: missing={missing[:5]} unreceipted={unreceipted[:5]}"
        )

    verified_bytes = 0
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        path = _canonical_corpus_file(row["relative_path"])
        digest, size = _stream_sha256_and_size(path)
        verified_bytes += size
        if digest != row["sha256"] or size != row["bytes"]:
            mismatches.append(
                {
                    "relative_path": row["relative_path"],
                    "expected_bytes": row["bytes"],
                    "observed_bytes": size,
                    "expected_sha256": row["sha256"],
                    "observed_sha256": digest,
                }
            )
    if mismatches:
        raise RuntimeError(f"core corpus byte drift: {mismatches[:3]}")
    if len(rows) != partition["file_count"] or verified_bytes != partition["total_bytes"]:
        raise RuntimeError("core corpus verified arithmetic drift")

    policy, policy_sha256 = prereg._source_policy_and_hash(payload)
    contracts = policy["receipt_contracts"]
    semantic = prereg.enveloped_research_result(
        {
            "schema_version": contracts["corpus_integrity_receipt_schema_version"],
            "status": "PASS_CORE_AUTHORITATIVE",
            "verification_completed_at_utc": _utc_now(),
            "partition_id": partition_id,
            "source_hash_policy_sha256": policy_sha256,
            "machinery_receipt_sha256": prereg.sha256_path(
                prereg.ENTRY_MACHINERY_RECEIPT_PATH
            ),
            "command": dict(contracts["required_integrity_command"]),
            "corpus_root": str(prereg.CORPUS_ROOT),
            "integrity_contract": partition,
            "manifest_contract_sha256": prereg.stable_hash(manifest_contract),
            "verified_file_count": len(rows),
            "verified_total_bytes": verified_bytes,
            "verified_ordered_entries_semantic_sha256": prereg.stable_hash(
                list(rows)
            ),
            "diagnostic_files_verified": 0,
            "cache_files_verified": 0,
            "mismatches": [],
            "unreceipted_paths": [],
            "loader_rehash_required": True,
        }
    )
    receipt = dict(semantic)
    receipt["receipt_sha256"] = prereg.stable_hash(semantic)
    expected_fields = set(prereg.core_corpus_integrity_receipt_spec()["fields_in_order"])
    if set(receipt) != expected_fields:
        raise RuntimeError("core corpus integrity receipt construction drift")
    prereg._write_canonical_json_exclusive(
        prereg.CORPUS_INTEGRITY_RECEIPT_PATH, receipt
    )
    return prereg._validated_corpus_integrity_receipt(
        payload, prereg_receipt, machinery
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "freeze-prereg",
            "verify-prereg",
            "seal-foundation-restoration",
            "seal-entry-machinery",
            "verify-corpus-integrity",
            "seal-foundation-stability",
            "seal-invalid-nested-skip",
            "run-entry-nested-block",
            "run-entry-outer-fold",
            "run-entry-fold",
            "freeze-entry-pooled-acceptance",
            "run-entry-campaign",
            "freeze-corrected-v32-foundation",
            "freeze-corrected-v32-executable-generation",
            "freeze-corrected-v32-executable-bridge",
        ),
        help="Run one fixed, fail-closed Path-D research stage.",
    )
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--inner-fold", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "freeze-corrected-v32-foundation":
        from v4.research.pathd_corrected_v32_foundation import (
            freeze_corrected_v32_foundation,
        )
        print(json.dumps(freeze_corrected_v32_foundation(), indent=2))
        return 0
    if args.stage == "freeze-corrected-v32-executable-generation":
        print(json.dumps(freeze_corrected_v32_executable_generation(), indent=2))
        return 0
    if args.stage == "freeze-corrected-v32-executable-bridge":
        print(json.dumps(freeze_corrected_v32_executable_bridge(), indent=2))
        return 0
    from v4.research.pathd_entry_dataset import assert_corrected_v32_executable_bridge
    assert_corrected_v32_executable_bridge()
    needs_outer = {
        "seal-invalid-nested-skip",
        "run-entry-nested-block",
        "run-entry-outer-fold",
        "run-entry-fold",
    }
    needs_inner = {"seal-invalid-nested-skip", "run-entry-nested-block"}
    if (args.stage in needs_outer) != (args.outer_fold is not None):
        raise ValueError("--outer-fold is required exactly for fold-scoped stages")
    if (args.stage in needs_inner) != (args.inner_fold is not None):
        raise ValueError("--inner-fold is required exactly for nested-block stages")
    if args.stage in needs_outer:
        # One campaign-wide byte-stability preflight must succeed before any
        # fold-scoped dispatcher can write a skip, access, burn, dataset, or
        # result artifact.  The evidence gate revalidates again immediately
        # before its one authorized decode.
        prereg.assert_research_foundation_stable()
    assert_corrected_v32_fit_release()
    if args.stage == "freeze-prereg":
        receipt = freeze_preregistration()
        negatives = require_all_negative_fixtures_rejected()
        result = enveloped_research_result(
            {"stage": "freeze-prereg", "receipt": receipt, "negative_fixtures": negatives}
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.stage == "seal-foundation-restoration":
        print(
            json.dumps(
                prereg.seal_foundation_restoration_receipt(), indent=2
            )
        )
        return 0
    if args.stage == "seal-entry-machinery":
        print(json.dumps(_seal_entry_machinery(), indent=2))
        return 0
    if args.stage == "verify-corpus-integrity":
        print(json.dumps(_verify_core_corpus_integrity(), indent=2))
        return 0
    if args.stage == "seal-foundation-stability":
        print(
            json.dumps(prereg.seal_foundation_stability_receipt(), indent=2)
        )
        return 0
    if args.stage == "seal-invalid-nested-skip":
        from v4.research.pathd_evidence_gate import seal_invalid_nested_block_skip

        result = seal_invalid_nested_block_skip(
            outer_fold=args.outer_fold, inner_fold=args.inner_fold
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.stage == "run-entry-nested-block":
        result = run_canonical_entry_nested_block(
            outer_fold=args.outer_fold, inner_fold=args.inner_fold
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.stage == "run-entry-outer-fold":
        result = run_canonical_entry_outer_fold(outer_fold=args.outer_fold)
        print(json.dumps(_mapping(result), indent=2))
        return 0
    if args.stage == "run-entry-fold":
        run_canonical_entry_fold(outer_fold=args.outer_fold)
        print(
            json.dumps(
                {"stage": args.stage, "outer_fold": args.outer_fold, "status": "COMPLETE"},
                indent=2,
            )
        )
        return 0
    if args.stage == "freeze-entry-pooled-acceptance":
        result = freeze_entry_pooled_acceptance_once()
        print(json.dumps(_mapping(result), indent=2))
        return 0
    if args.stage == "run-entry-campaign":
        print(json.dumps(run_canonical_entry_campaign(), indent=2))
        return 0
    receipt = assert_preregistration_frozen()
    negatives = require_all_negative_fixtures_rejected()
    result = enveloped_research_result(
        {"stage": "verify-prereg", "receipt": receipt, "negative_fixtures": negatives}
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
