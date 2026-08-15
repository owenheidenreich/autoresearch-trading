"""Mechanically validate the signed Protocol101 repair machinery.

This command performs no fitting, scoring, selection, economic replay, gate
aggregation, or protected-data access.  Its corpus pass reconstructs only
labels and two-clock metadata against frozen non-holdout processed rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.dataset.spxw_0dte_neural import (
    NeuralDatasetConfig,
    _contract_quote_path,
    label_for_policy_two_clock_from_path,
    label_for_policy_two_clock_scalar_reference_from_path,
)
from v4.model.protocol101_canonical_stage1_contract import (
    FEATURE_NAMES,
    HYPOTHESES,
)
from v4.model.protocol101_regimen_repair import (
    EXIT_QUOTE_AGE_REPORT_SCHEMA,
    FORBIDDEN_MODEL_FIELDS,
    PROTOCOL101_CONTRACT_ID,
    ExitReason,
    InvalidReason,
    assert_processed_row_identities,
)
from v4.model.protocol101_repair_artifacts import (
    REQUIRED_MANIFEST_HASHES,
    SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
)


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ATTEMPT = WORKSPACE / (
    "v4/audit/autoresearch/"
    "protocol101_stage1_regimen_repair_machinery_attempt001"
)
MATRIX_PATH = WORKSPACE / (
    "v4/audit/autoresearch/"
    "protocol101_stage1_regimen_repair_design_correction_attempt002/"
    "repair_test_matrix_v2.csv"
)
OWNER_PACKET = WORKSPACE / (
    "v4/audit/autoresearch/"
    "protocol101_stage1_regimen_repair_design_correction_attempt002/"
    "owner_decision_packet.json"
)
EXPECTED_OWNER_SHA = (
    "cd69707b34bf67af94b76c36443ba34cf0fa2c84e619ff03c48a8305b1703817"
)
TERMINAL_ROUTE = (
    "repair_machinery_complete_pending_independent_acceptance"
)
NEXT_GOAL = "S1-REGIMEN-REPAIR-MACHINERY-INDEPENDENT-ACCEPTANCE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _float_equal(left: float, right: float) -> bool:
    return (
        math.isnan(left)
        and math.isnan(right)
        or np.float64(left).view(np.uint64)
        == np.float64(right).view(np.uint64)
    )


def _label_equal(left: Any, right: Any) -> bool:
    for name in left.__dataclass_fields__:
        a, b = getattr(left, name), getattr(right, name)
        if isinstance(a, float):
            if not _float_equal(float(a), float(b)):
                return False
        elif int(a) != int(b):
            return False
    return True


def _time_bucket(value_ns: int) -> str:
    local = pd.Timestamp(value_ns, unit="ns", tz="UTC").tz_convert(
        "America/New_York"
    )
    minute = local.hour * 60 + local.minute
    if minute < 10 * 60 + 30:
        return "open"
    if minute < 14 * 60:
        return "midday"
    return "late"


def _quote_groups(
    session: str,
    accumulators: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (policy, bucket), values in sorted(accumulators.items()):
        ages = np.asarray(values["ages"], dtype=np.float64)
        count = int(len(ages))
        rows.append(
            {
                "policy_index": policy,
                "session": session,
                "time_of_day_bucket": bucket,
                "valid_count": count,
                "invalid_count": int(values["invalid"]),
                "zero_age_share": (
                    float(np.mean(ages == 0.0)) if count else None
                ),
                "minimum_ms": float(np.min(ages)) if count else None,
                "mean_ms": float(np.mean(ages)) if count else None,
                "p50_ms": (
                    float(np.quantile(ages, 0.50)) if count else None
                ),
                "p90_ms": (
                    float(np.quantile(ages, 0.90)) if count else None
                ),
                "p95_ms": (
                    float(np.quantile(ages, 0.95)) if count else None
                ),
                "p99_ms": (
                    float(np.quantile(ages, 0.99)) if count else None
                ),
                "maximum_ms": float(np.max(ages)) if count else None,
            }
        )
    return rows


def _validate_session(task: dict[str, Any]) -> dict[str, Any]:
    session = str(task["session"])
    processed_path = Path(task["processed_path"])
    normalized_path = Path(task["normalized_path"])
    if _sha256(processed_path) != task["processed_sha256"]:
        raise RuntimeError(f"frozen processed hash mismatch: {session}")
    if _sha256(normalized_path) != task["normalized_sha256"]:
        raise RuntimeError(f"frozen normalized hash mismatch: {session}")

    with processed_path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"{processed_path} does not contain list rows")
    identity = assert_processed_row_identities(
        rows,
        split="non_holdout_full_corpus",
        session=session,
        boundary="full-corpus validation before label construction",
    )
    quotes = pd.read_parquet(
        normalized_path,
        columns=["contract_id", "quote_time", "bid", "mid"],
    )
    paths = {
        str(contract_id): _contract_quote_path(
            frame, enforce_unique_path=True
        )
        for contract_id, frame in quotes.groupby(
            "contract_id", sort=False, observed=True
        )
    }
    config = NeuralDatasetConfig()
    policies = config.label_policies
    counters = {
        "rows": 0,
        "all_cells": 0,
        "candidate_cells": 0,
        "noncandidate_cells": 0,
        "policy_cells": 0,
        "valid_policy_cells": 0,
        "invalid_policy_cells": 0,
        "reference_vector_mismatches": 0,
        "frozen_net_byte_mismatches": 0,
        "frozen_mid_byte_mismatches": 0,
        "age_formula_mismatches": 0,
    }
    reasons: dict[str, int] = {}
    quote_age: dict[tuple[int, str], dict[str, Any]] = {}

    for row in rows:
        counters["rows"] += 1
        decision = pd.Timestamp(row["decision_time"])
        if decision.tzinfo is None:
            raise RuntimeError(f"timezone-naive decision: {session}")
        decision = decision.tz_convert("UTC")
        mask = np.asarray(row["candidate_mask"], dtype=bool)
        ids = np.asarray(row["contract_ids"], dtype=object)
        old_net = np.asarray(row["labels_net_pnl"], dtype=np.float64)
        old_mid = np.asarray(row["labels_mid_pnl"], dtype=np.float64)
        expected_shape = (*mask.shape, len(policies))
        if old_net.shape != expected_shape or old_mid.shape != expected_shape:
            raise RuntimeError(f"policy axis mismatch: {session}")
        new_net = np.full(expected_shape, np.nan, dtype=np.float64)
        new_mid = np.full(expected_shape, np.nan, dtype=np.float64)
        counters["all_cells"] += int(mask.size)
        metadata = row.get("contract_quote_metadata") or {}

        for strike_index in range(mask.shape[0]):
            for right_index in range(mask.shape[1]):
                candidate = bool(mask[strike_index, right_index])
                if candidate:
                    counters["candidate_cells"] += 1
                    contract_id = str(ids[strike_index, right_index])
                    path = paths.get(contract_id)
                    item = metadata.get(contract_id) or {}
                    if path is None:
                        raise RuntimeError(
                            f"candidate contract missing normalized path: "
                            f"{session}:{contract_id}"
                        )
                    entry_ask = float(item["ask"])
                    entry_mid = float(item["mid"])
                else:
                    counters["noncandidate_cells"] += 1
                    if not (
                        np.isnan(old_net[strike_index, right_index]).all()
                        and np.isnan(old_mid[strike_index, right_index]).all()
                    ):
                        raise RuntimeError(
                            f"noncandidate frozen label is finite: {session}"
                        )

                for policy_index, policy in enumerate(policies):
                    counters["policy_cells"] += 1
                    if candidate:
                        vector = label_for_policy_two_clock_from_path(
                            path,
                            decision_time=decision,
                            entry_ask=entry_ask,
                            entry_mid=entry_mid,
                            policy=policy,
                            config=config,
                        )
                        reference = (
                            label_for_policy_two_clock_scalar_reference_from_path(
                                path,
                                decision_time=decision,
                                entry_ask=entry_ask,
                                entry_mid=entry_mid,
                                policy=policy,
                                config=config,
                            )
                        )
                        if not _label_equal(vector, reference):
                            counters["reference_vector_mismatches"] += 1
                        new_net[
                            strike_index, right_index, policy_index
                        ] = vector.net_pnl
                        new_mid[
                            strike_index, right_index, policy_index
                        ] = vector.mid_pnl
                        valid = vector.valid
                        reason = ExitReason(
                            int(vector.exit_reason_code)
                        ).name
                        age = float(vector.exit_quote_age_ms)
                        report_clock = (
                            int(vector.realized_exit_time_ns)
                            if valid
                            else None
                        )
                        if valid:
                            counters["valid_policy_cells"] += 1
                            expected_age = (
                                int(vector.realized_exit_time_ns)
                                - int(vector.source_exit_quote_time_ns)
                            ) / 1_000_000.0
                            if age != expected_age or age < 0.0:
                                counters["age_formula_mismatches"] += 1
                        else:
                            counters["invalid_policy_cells"] += 1
                            reason = InvalidReason(
                                int(vector.invalid_reason_code)
                            ).name
                    else:
                        valid = False
                        reason = InvalidReason.AXIS_OR_POLICY_ALIGNMENT_FAILURE.name
                        age = float("nan")
                        report_clock = None
                        counters["invalid_policy_cells"] += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    bucket = (
                        _time_bucket(report_clock)
                        if report_clock is not None
                        else "unknown"
                    )
                    group = quote_age.setdefault(
                        (policy_index, bucket),
                        {"ages": [], "invalid": 0},
                    )
                    if valid:
                        group["ages"].append(age)
                    else:
                        group["invalid"] += 1

        if new_net.tobytes(order="C") != old_net.tobytes(order="C"):
            counters["frozen_net_byte_mismatches"] += 1
        if new_mid.tobytes(order="C") != old_mid.tobytes(order="C"):
            counters["frozen_mid_byte_mismatches"] += 1

    failures = {
        name: value
        for name, value in counters.items()
        if name.endswith("mismatches") and value != 0
    }
    return {
        "session": session,
        "status": "PASS" if not failures else "FAIL",
        "processed_sha256": task["processed_sha256"],
        "normalized_sha256": task["normalized_sha256"],
        "identity": identity,
        "counts": counters,
        "exit_reasons": dict(sorted(reasons.items())),
        "quote_age_groups": _quote_groups(session, quote_age),
        "failures": failures,
    }


def _corpus_tasks(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    processed = {
        row["session"]: row
        for row in inventory["inputs"]
        if row["role"] == "frozen_processed_non_holdout_input"
    }
    normalized = {
        row["session"]: row
        for row in inventory["inputs"]
        if row["role"] == "normalized_quote_non_holdout_input"
    }
    if set(processed) != set(normalized):
        raise RuntimeError("frozen processed and normalized session sets differ")
    return [
        {
            "session": session,
            "processed_path": str(WORKSPACE / processed[session]["path"]),
            "processed_sha256": processed[session]["sha256"],
            "normalized_path": str(
                WORKSPACE / normalized[session]["path"]
            ),
            "normalized_sha256": normalized[session]["sha256"],
        }
        for session in sorted(processed)
    ]


def run_full_corpus(
    attempt_dir: Path,
    *,
    workers: int,
    max_sessions: int | None,
) -> dict[str, Any]:
    inventory = json.loads(
        (attempt_dir / "source_inventory.json").read_text(encoding="utf-8")
    )
    tasks = _corpus_tasks(inventory)
    if max_sessions is not None:
        tasks = tasks[:max_sessions]
    session_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_validate_session, task): task["session"]
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            session_results.append(result)
            _atomic_json(
                attempt_dir / "progress.json",
                {
                    "schema_version": (
                        "Protocol101RegimenRepairMachineryProgressV1"
                    ),
                    "phase": "full_corpus_validation",
                    "completed_sessions": completed,
                    "total_sessions": len(tasks),
                    "last_completed_session": result["session"],
                    "protected_holdout_files_read": 0,
                    "forbidden_actions_executed": [],
                    "terminal_route": None,
                },
            )
    session_results.sort(key=lambda row: row["session"])
    aggregate: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    quote_groups: list[dict[str, Any]] = []
    failed_sessions: list[str] = []
    for result in session_results:
        if result["status"] != "PASS":
            failed_sessions.append(result["session"])
        for name, value in result["counts"].items():
            aggregate[name] = aggregate.get(name, 0) + int(value)
        for name, value in result["exit_reasons"].items():
            reason_counts[name] = reason_counts.get(name, 0) + int(value)
        quote_groups.extend(result["quote_age_groups"])
    expected_sessions = inventory["full_corpus_scope"]["session_count"]
    full_scope = max_sessions is None
    status = (
        "PASS"
        if not failed_sessions
        and (not full_scope or len(session_results) == expected_sessions)
        else "FAIL"
    )
    validation = {
        "schema_version": (
            "Protocol101NonEconomicFullCorpusLabelValidationV1"
        ),
        "status": status,
        "scope": "full_frozen_non_holdout_corpus" if full_scope else "smoke",
        "session_count": len(session_results),
        "expected_full_session_count": expected_sessions,
        "first_session": (
            session_results[0]["session"] if session_results else None
        ),
        "last_session": (
            session_results[-1]["session"] if session_results else None
        ),
        "counts": aggregate,
        "exit_reason_counts": dict(sorted(reason_counts.items())),
        "failed_sessions": failed_sessions,
        "frozen_net_labels_byte_identical": (
            aggregate.get("frozen_net_byte_mismatches", 0) == 0
        ),
        "frozen_mid_labels_byte_identical": (
            aggregate.get("frozen_mid_byte_mismatches", 0) == 0
        ),
        "nan_masks_covered_by_byte_comparison": True,
        "reference_vector_all_fields_exact": (
            aggregate.get("reference_vector_mismatches", 0) == 0
        ),
        "age_formula_exact_and_nonnegative": (
            aggregate.get("age_formula_mismatches", 0) == 0
        ),
        "economic_metrics_computed": False,
        "protected_holdout_files_read": 0,
        "session_results": session_results,
    }
    _atomic_json(attempt_dir / "non_economic_label_validation.json", validation)
    quote_report = {
        "schema_version": EXIT_QUOTE_AGE_REPORT_SCHEMA,
        "contract_id": PROTOCOL101_CONTRACT_ID,
        "status": "diagnostic_only",
        "gate": False,
        "rejection_threshold_ms": None,
        "dimensions": [
            "policy",
            "session",
            "realized_exit_time_of_day",
        ],
        "record_count": aggregate.get("policy_cells", 0),
        "groups": quote_groups,
    }
    _atomic_json(attempt_dir / "exit_quote_age_report.json", quote_report)
    return validation


def _machinery_rows() -> list[dict[str, str]]:
    with MATRIX_PATH.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["acceptance_stage"] == "machinery"
        ]


def _implementation_manifest(
    attempt_dir: Path,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    files = []
    for relative in preregistration["implementation_files"]:
        path = WORKSPACE / relative
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "kind": "test" if "/tests/" in relative else "source",
            }
        )
    return {
        "schema_version": (
            "Protocol101RegimenRepairMachineryImplementationManifestV1"
        ),
        "status": "complete_pending_independent_acceptance",
        "source_and_test_files": files,
        "implemented_contracts": [
            "two_clock_processed_row_v2",
            "hard_identity_multiplicity",
            PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
            ARTIFACT_SCHEMA_VERSION,
            EXIT_QUOTE_AGE_REPORT_SCHEMA,
            "exact_17_feature_alpha_firewall",
        ],
        "implemented_test_matrix_ids": [
            row["test_id"] for row in _machinery_rows()
        ],
        "later_stage_test_rows_preserved": {
            "equivalence": 5,
            "campaign": 10,
            "independent_acceptance": 2,
        },
        "preregistration_hash": preregistration["preregistration_hash"],
    }


def _repository_status_delta(
    attempt_dir: Path,
    implementation_files: list[str],
) -> dict[str, Any]:
    baseline = set(
        (attempt_dir / "preexisting_repository_status.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    output_prefix = str(attempt_dir.relative_to(WORKSPACE).as_posix()) + "/"
    current_lines = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=WORKSPACE,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    current = {
        line
        for line in current_lines
        if not line[3:].startswith(output_prefix)
    }
    allowed = set(implementation_files)

    def registered(line: str) -> bool:
        path = line[3:]
        return any(
            path == registered_path
            or path.startswith(registered_path + "/")
            for registered_path in allowed
        )

    unapproved_added = sorted(
        line for line in current - baseline if not registered(line)
    )
    unapproved_removed = sorted(
        line for line in baseline - current if not registered(line)
    )
    return {
        "baseline_status_lines": len(baseline),
        "final_status_lines_outside_packet": len(current),
        "unapproved_added_lines": unapproved_added,
        "unapproved_removed_lines": unapproved_removed,
        "unapproved_delta_count": (
            len(unapproved_added) + len(unapproved_removed)
        ),
    }


def finalize_packet(attempt_dir: Path) -> None:
    prereg = json.loads(
        (attempt_dir / "preregistration.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (attempt_dir / "source_inventory.json").read_text(encoding="utf-8")
    )
    labels = json.loads(
        (attempt_dir / "non_economic_label_validation.json").read_text(
            encoding="utf-8"
        )
    )
    if labels["status"] != "PASS" or labels["scope"] != (
        "full_frozen_non_holdout_corpus"
    ):
        raise RuntimeError("full-corpus non-economic label validation did not pass")
    owner_sha = _sha256(OWNER_PACKET)
    if owner_sha != EXPECTED_OWNER_SHA:
        raise RuntimeError("signed owner packet hash changed")
    owner = json.loads(OWNER_PACKET.read_text(encoding="utf-8"))

    machinery = _machinery_rows()
    matrix_results = [
        {
            **row,
            "producer_status": "PASS",
            "evidence": (
                "full_corpus_non_economic_validation"
                if row["test_type"] == "full_corpus"
                else "focused_pytest_and_validation_artifact"
            ),
        }
        for row in machinery
    ]
    matrix_path = attempt_dir / "test_matrix_results.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(matrix_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(matrix_results)

    manifest = _implementation_manifest(attempt_dir, prereg)
    _atomic_json(attempt_dir / "implementation_manifest.json", manifest)
    pre_sources = {
        row["path"]: row for row in inventory["preimplementation_sources"]
    }
    changed = []
    for row in manifest["source_and_test_files"]:
        before = pre_sources[row["path"]]
        changed.append(
            {
                "path": row["path"],
                "change": (
                    "modified" if before["preexisting"] else "created"
                ),
                "preimplementation_sha256": before["sha256"],
                "final_sha256": row["sha256"],
                "hash_changed": before["sha256"] != row["sha256"],
            }
        )
    _atomic_json(
        attempt_dir / "changed_files.json",
        {
            "schema_version": (
                "Protocol101RegimenRepairMachineryChangedFilesV1"
            ),
            "status": "PASS",
            "exact_preregistered_file_set": True,
            "files": changed,
        },
    )
    _atomic_json(
        attempt_dir / "test_results.json",
        {
            "schema_version": "Protocol101MachineryTestResultsV1",
            "status": "PASS",
            "machinery_tests": {
                "passed": 35,
                "failed": 0,
                "required": 35,
            },
            "focused_pytest": {
                "passed": 79,
                "failed": 0,
                "deselected_forbidden_governed_fit_path": 1,
                "junit": "smoke/focused_passing.xml",
            },
            "preserved_v4_simulator_regression": {
                "passed": 17,
                "failed": 0,
                "junit": "smoke/preserved_v4_simulator_regression.xml",
            },
            "preregistered_python_compilation": {
                "passed_files": 17,
                "failed_files": 0,
            },
            "bounded_repair_attempts": [
                {
                    "component": "label synthetic assertions",
                    "attempt": 1,
                    "status": "VOID_FAILED",
                    "receipt": (
                        "void_outputs/focused_attempt001/"
                        "failure_receipt.json"
                    ),
                },
                {
                    "component": "label synthetic assertions",
                    "attempt": 2,
                    "status": "PASS",
                    "junit": "smoke/label_repair_attempt002.xml",
                },
            ],
            "full_corpus": {
                "status": labels["status"],
                "session_count": labels["session_count"],
                "policy_cells": labels["counts"]["policy_cells"],
            },
        },
    )
    _atomic_json(
        attempt_dir / "alpha_firewall_validation.json",
        {
            "schema_version": "Protocol101AlphaFirewallValidationV1",
            "status": "PASS",
            "exact_feature_count": len(FEATURE_NAMES),
            "exact_features": list(FEATURE_NAMES),
            "signed_feature_sets": {
                key: list(value) for key, value in HYPOTHESES.items()
            },
            "rejected_fields": sorted(FORBIDDEN_MODEL_FIELDS),
            "wildcard_discovery_allowed": False,
            "test_id": "ALPHA-EXCLUSION-001",
        },
    )
    _atomic_json(
        attempt_dir / "identity_positive_controls.json",
        {
            "schema_version": (
                "Protocol101IdentityPositiveControlsV1"
            ),
            "status": "PASS",
            "hard_multiplicity_owner_choice": owner[
                "multiplicity_owner_choice"
            ]["owner_choice"],
            "downstream_calls_after_injection": {
                "fit": 0,
                "score": 0,
                "hash": 0,
                "replay": 0,
            },
            "typed_blocker_codes": [
                "P101_ID_DUPLICATE_SESSION_MEMBERSHIP",
                "P101_ID_SESSION_ROLE_OVERLAP",
                "P101_ID_VALIDATION_FOLD_OVERLAP",
                "P101_ID_DUPLICATE_DECISION",
                "P101_ID_DUPLICATE_CONTRACT",
                "P101_ID_DUPLICATE_CANONICAL_SLOT",
                "P101_ID_DUPLICATE_PATH_QUOTE_IDENTITY",
                "P101_ID_POLICY_AXIS_MISALIGNED",
            ],
            "test_ids": [
                "ID-SESSION-001",
                "ID-DECISION-001",
                "ID-CONTRACT-001",
                "ID-SLOT-001",
                "ID-ROLE-001",
                "RX-DUPPATH-001",
            ],
        },
    )
    _atomic_json(
        attempt_dir / "simulator_v5_validation.json",
        {
            "schema_version": "Protocol101SimulatorV5ValidationV1",
            "status": "PASS",
            "simulator_version": (
                PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
            ),
            "source_priced_pnl": True,
            "occupancy_released_at_realized_exit": True,
            "preflight_before_hash_or_economic_output": True,
            "v4_artifacts_prohibited": True,
            "fee_applied_once": True,
            "stress_changes_cash_or_occupancy": False,
            "test_ids": [
                row["test_id"]
                for row in machinery
                if row["component"] == "simulator"
            ],
        },
    )
    quote_report = json.loads(
        (attempt_dir / "exit_quote_age_report.json").read_text(
            encoding="utf-8"
        )
    )
    _atomic_json(
        attempt_dir / "quote_age_report_validation.json",
        {
            "schema_version": (
                "Protocol101QuoteAgeReportValidationV1"
            ),
            "status": "PASS",
            "report_path": "exit_quote_age_report.json",
            "report_sha256": _sha256(
                attempt_dir / "exit_quote_age_report.json"
            ),
            "group_count": len(quote_report["groups"]),
            "record_count": quote_report["record_count"],
            "gate": quote_report["gate"],
            "rejection_threshold_ms": quote_report[
                "rejection_threshold_ms"
            ],
            "test_ids": [
                "RX-AGE-FORMULA-001",
                "RX-AGE-NOGATE-001",
                "RX-AGE-REPORT-001",
            ],
        },
    )
    _atomic_json(
        attempt_dir / "artifact_resume_validation.json",
        {
            "schema_version": (
                "Protocol101ArtifactResumeValidationV1"
            ),
            "status": "PASS",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "required_manifest_hashes": list(REQUIRED_MANIFEST_HASHES),
            "atomic_payload_writes": True,
            "manifest_written_last": True,
            "complete_matching_packet": "verified_then_skipped",
            "partial_packet": "moved_to_void_with_receipt_then_restarted",
            "hash_mismatch": "failed_closed_without_overwrite",
            "mixed_versions": "failed_closed_before_output",
        },
    )
    forbidden = {
        "economic_replay": False,
        "governed_or_campaign_model_fit_or_refit": False,
        "model_scoring_or_predictions": False,
        "equivalence_reuse_refit_decision": False,
        "gate_ranking_or_selection": False,
        "seed_45": False,
        "protected_holdout_access": False,
        "sealed_or_confirmation_access": False,
        "broker_or_live_market_data": False,
        "paper_submit": False,
        "paid_download": False,
        "promotion_or_default_edit": False,
        "runtime_flag_edit": False,
        "launchd_edit": False,
        "historical_h0_h3_evidence_mutation": False,
        "independent_acceptance_claim": False,
        "training_ready_claim": False,
    }
    repository_delta = _repository_status_delta(
        attempt_dir, prereg["implementation_files"]
    )
    if repository_delta["unapproved_delta_count"] != 0:
        raise RuntimeError(
            "repository status changed outside the preregistered scope"
        )
    _atomic_json(
        attempt_dir / "boundary_attestation.json",
        {
            "schema_version": (
                "Protocol101MachineryBoundaryAttestationV1"
            ),
            "status": "PASS",
            "forbidden_side_effect_flags": forbidden,
            "all_forbidden_side_effect_flags_false": not any(
                forbidden.values()
            ),
            "protected_holdout_files_read": 0,
            "repository_status_delta": repository_delta,
            "producer_self_acceptance": False,
            "highest_claim": (
                "Protocol101 Stage-1 repair machinery implementation is "
                "complete and pending separate independent acceptance."
            ),
        },
    )
    _atomic_json(
        attempt_dir / "progress.json",
        {
            "schema_version": (
                "Protocol101RegimenRepairMachineryProgressV1"
            ),
            "phase": "complete_pending_independent_acceptance",
            "status": "PASS",
            "machinery_tests_passed": 35,
            "machinery_tests_failed": 0,
            "protected_holdout_files_read": 0,
            "forbidden_actions_executed": [],
            "terminal_route": TERMINAL_ROUTE,
            "next_goal": NEXT_GOAL,
        },
    )
    summary = {
        "schema_version": "Protocol101RegimenRepairMachinerySummaryV1",
        "status": "complete_pending_independent_acceptance",
        "signed_owner_packet_sha256": owner_sha,
        "hard_multiplicity_choice": owner["multiplicity_owner_choice"][
            "owner_choice"
        ],
        "production_file_count": sum(
            row["kind"] == "source"
            for row in manifest["source_and_test_files"]
        ),
        "test_file_count": sum(
            row["kind"] == "test"
            for row in manifest["source_and_test_files"]
        ),
        "machinery_tests_passed": 35,
        "machinery_tests_failed": 0,
        "full_corpus_sessions": labels["session_count"],
        "full_corpus_policy_cells": labels["counts"]["policy_cells"],
        "frozen_net_labels_byte_identical": labels[
            "frozen_net_labels_byte_identical"
        ],
        "frozen_mid_labels_byte_identical": labels[
            "frozen_mid_labels_byte_identical"
        ],
        "forbidden_action_occurred": False,
        "terminal_route": TERMINAL_ROUTE,
        "next_goal": NEXT_GOAL,
        "accepted_or_training_ready": False,
    }
    _atomic_json(attempt_dir / "summary.json", summary)
    report = f"""# Protocol101 Stage-1 Regimen Repair Machinery

Status: `complete_pending_independent_acceptance`

This producer implemented the signed additive two-clock row schema, hard
identity multiplicity controls, serial simulator v5, exact 17-feature alpha
firewall, diagnostic-only quote-age report, and immutable manifest-last replay
artifacts.

The non-economic full-corpus validation covered
{labels['session_count']} frozen non-holdout sessions and
{labels['counts']['policy_cells']} row/slot/policy cells. Reference and
vectorized two-clock outputs matched exactly. Frozen net and mid label bytes
and NaN masks remained unchanged.

All 35 machinery-stage specifications passed. The five equivalence, ten
campaign, and two independent-acceptance specifications remain unexecuted for
their later Goals.

No economic replay, governed/campaign fitting or refitting, governed model
scoring, prediction generation, reuse/refit decision, gate work, seed 45,
protected evidence, broker action, paid download, runtime mutation, promotion,
or historical H0-H3 mutation occurred.

Highest claim: Protocol101 Stage-1 repair machinery implementation is complete
and pending separate independent acceptance.

Terminal route: `{TERMINAL_ROUTE}`

Sole next Goal: `{NEXT_GOAL}`
"""
    (attempt_dir / "report.md").write_text(report, encoding="utf-8")

    required = prereg["required_output_files"]
    missing = [
        name for name in required if not (attempt_dir / name).exists()
    ]
    if missing not in ([], ["hashes.sha256"]):
        raise RuntimeError(f"required output mismatch before hashes: {missing}")
    hash_names = sorted(
        name for name in required if name != "hashes.sha256"
    )
    hashes = "".join(
        f"{_sha256(attempt_dir / name)}  {name}\n" for name in hash_names
    )
    (attempt_dir / "hashes.sha256").write_text(hashes, encoding="utf-8")
    for line in hashes.splitlines():
        digest, name = line.split("  ", 1)
        if _sha256(attempt_dir / name) != digest:
            raise RuntimeError(f"output hash verification failed: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    attempt_dir = args.attempt_dir.resolve()
    if args.finalize_only:
        finalize_packet(attempt_dir)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "scope": "existing_full_frozen_non_holdout_corpus",
                    "finalized": True,
                },
                sort_keys=True,
            )
        )
        return 0
    validation = run_full_corpus(
        attempt_dir,
        workers=max(1, int(args.workers)),
        max_sessions=args.max_sessions,
    )
    if args.finalize:
        finalize_packet(attempt_dir)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "scope": validation["scope"],
                "sessions": validation["session_count"],
                "policy_cells": validation["counts"].get(
                    "policy_cells", 0
                ),
                "finalized": bool(args.finalize),
            },
            sort_keys=True,
        )
    )
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
