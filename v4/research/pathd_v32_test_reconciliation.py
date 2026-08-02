"""Write-once, test-only reconciliation for the unreleased corrected-v3.2 build.

This module cannot release, fit, decode, open evidence, or access the holdout.  It
only binds the post-freeze governance/test corrections and their registered JUnit
result to the immutable corrected-v3.2 foundation and executable receipts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from v4.research import pathd_entry_exit as prereg
from v4.research.pathd_entry_dataset import (
    CORRECTED_V32_EXECUTABLE_BRIDGE_PATH,
    CORRECTED_V32_EXECUTABLE_GENERATION_PATH,
    CORRECTED_V32_RECONCILER_SOURCE_PATH,
    CORRECTED_V32_REGISTERED_TEST_PATHS,
    CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH,
    CORRECTED_V32_TEST_RECONCILIATION_RECEIPT_PATH,
)


JUNIT_PATH = (
    CORRECTED_V32_TEST_RECONCILIATION_RECEIPT_PATH.parent
    / "registered_test_reconciliation.junit.xml"
)
EXPECTED_TEST_COUNT = 151
NO_ACTION_STATE = {
    "model_fit_executed": False,
    "corpus_decoded": False,
    "evidence_opened": False,
    "foundation_or_machinery_sealed_against_corpus": False,
    "holdout_opened": False,
    "live_or_broker_action_executed": False,
    "holdout_open_count": 0,
}


def _self_hashed(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "receipt_sha256": prereg.stable_hash(payload)}


def _foundation_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    generation = prereg.read_json(CORRECTED_V32_EXECUTABLE_GENERATION_PATH)
    semantic = dict(generation)
    digest = semantic.pop("generation_sha256", None)
    if digest != prereg.stable_hash(semantic):
        raise RuntimeError("corrected-v3.2 executable generation drift")
    root = prereg.REPO_ROOT / generation["foundation_root"]
    preregistration = prereg.read_json(root / "preregistration.json")
    if prereg.sha256_path(root / "preregistration.json") != generation[
        "foundation_preregistration_sha256"
    ]:
        raise RuntimeError("corrected-v3.2 foundation preregistration drift")
    return generation, preregistration


def _changes(
    generation: dict[str, Any], preregistration: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str | None]]]:
    source_changes = []
    for label in generation["source_paths"]:
        frozen = preregistration["source_hashes_at_freeze"][label]
        current = prereg.sha256_path(prereg.REPO_ROOT / label)
        if frozen != current:
            source_changes.append(
                {"path": label, "frozen_sha256": frozen, "current_sha256": current}
            )
    test_changes = []
    for label in CORRECTED_V32_REGISTERED_TEST_PATHS:
        frozen = preregistration["source_hashes_at_freeze"].get(label)
        current = prereg.sha256_path(prereg.REPO_ROOT / label)
        if frozen != current:
            test_changes.append(
                {"path": label, "frozen_sha256": frozen, "current_sha256": current}
            )
    if not source_changes or not test_changes:
        raise RuntimeError("corrected-v3.2 reconciliation has no source/test delta")
    return source_changes, test_changes


def authorize() -> dict[str, Any]:
    """Freeze a no-execution authorization for the exact current corrections."""

    generation, preregistration = _foundation_inputs()
    source_changes, test_changes = _changes(generation, preregistration)
    release_path = prereg.REPO_ROOT / generation["claude_release_path"]
    if release_path.exists():
        raise RuntimeError("test-only reconciliation forbidden after Claude release")
    payload = _self_hashed(
        {
            "schema_version": (
                "pathd.corrected_v32.registered_test_reconciliation_authorization.v1"
            ),
            "status": "AUTHORIZED_TEST_ONLY_UNRELEASED_NO_EXECUTION",
            "foundation_generation_sha256": generation[
                "foundation_generation_sha256"
            ],
            "foundation_preregistration_sha256": generation[
                "foundation_preregistration_sha256"
            ],
            "source_hash_policy_sha256": generation["source_hash_policy_sha256"],
            "base_implementation_receipt_path": str(
                CORRECTED_V32_EXECUTABLE_BRIDGE_PATH.relative_to(prereg.REPO_ROOT)
            ),
            "base_implementation_receipt_sha256": prereg.sha256_path(
                CORRECTED_V32_EXECUTABLE_BRIDGE_PATH
            ),
            "reconciler_source_path": CORRECTED_V32_RECONCILER_SOURCE_PATH,
            "reconciler_source_sha256": prereg.sha256_path(
                prereg.REPO_ROOT / CORRECTED_V32_RECONCILER_SOURCE_PATH
            ),
            "registered_test_paths": list(CORRECTED_V32_REGISTERED_TEST_PATHS),
            "source_changes": source_changes,
            "test_changes": test_changes,
            "claude_release_absent": True,
            **NO_ACTION_STATE,
        }
    )
    prereg._write_canonical_json_exclusive(
        CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH, payload
    )
    return payload


def _junit_summary(path: Path) -> dict[str, int]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("registered-suite JUnit is absent or unsafe")
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    failures = sum(len(case.findall("failure")) for case in cases)
    errors = sum(len(case.findall("error")) for case in cases)
    skipped = sum(len(case.findall("skipped")) for case in cases)
    summary = {
        "tests": len(cases),
        "passed": len(cases) - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }
    if summary != {
        "tests": EXPECTED_TEST_COUNT,
        "passed": EXPECTED_TEST_COUNT,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }:
        raise RuntimeError(f"registered Path-D suite did not pass exactly: {summary}")
    return summary


def finalize() -> dict[str, Any]:
    """Bind the passing exact registered suite without changing release state."""

    authorization = prereg.read_json(
        CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH
    )
    semantic = dict(authorization)
    digest = semantic.pop("receipt_sha256", None)
    if digest != prereg.stable_hash(semantic):
        raise RuntimeError("reconciliation authorization self-hash drift")
    generation, preregistration = _foundation_inputs()
    source_changes, test_changes = _changes(generation, preregistration)
    if (
        authorization.get("source_changes") != source_changes
        or authorization.get("test_changes") != test_changes
        or authorization.get("reconciler_source_sha256")
        != prereg.sha256_path(prereg.REPO_ROOT / CORRECTED_V32_RECONCILER_SOURCE_PATH)
    ):
        raise RuntimeError("sources changed after reconciliation authorization")
    if (prereg.REPO_ROOT / generation["claude_release_path"]).exists():
        raise RuntimeError("test-only reconciliation forbidden after Claude release")
    summary = _junit_summary(JUNIT_PATH)
    payload = _self_hashed(
        {
            "schema_version": (
                "pathd.corrected_v32.registered_test_reconciliation_receipt.v1"
            ),
            "status": "PASS_FULL_REGISTERED_PATHD_SUITE_UNRELEASED",
            "authorization_sha256": prereg.sha256_path(
                CORRECTED_V32_TEST_RECONCILIATION_AUTHORIZATION_PATH
            ),
            "registered_test_paths": list(CORRECTED_V32_REGISTERED_TEST_PATHS),
            "source_changes": source_changes,
            "test_changes": test_changes,
            "junit_path": str(JUNIT_PATH.relative_to(prereg.REPO_ROOT)),
            "junit_sha256": prereg.sha256_path(JUNIT_PATH),
            "summary": summary,
            "claude_release_absent": True,
            **NO_ACTION_STATE,
        }
    )
    prereg._write_canonical_json_exclusive(
        CORRECTED_V32_TEST_RECONCILIATION_RECEIPT_PATH, payload
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("authorize", "finalize"))
    args = parser.parse_args()
    result = authorize() if args.action == "authorize" else finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
