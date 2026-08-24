"""Create immutable evidence for the selector-quality requirement curve.

This wrapper consumes the already-computed lifecycle P&L CSV.  It never
rebuilds P&L, fits a model, searches a market feature, contacts a vendor or
broker, reads a reserved confirmation session, or spends alpha.  The research
module owns every score law, selection rate, split, bootstrap, and inverse
requirement rule.

An attempt directory is immutable.  Once it is created, source archival and
all later work occur inside the caught failure region so even an archival
failure leaves a traceback log and a self-hashed failure receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import traceback
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Direct ``python v5/ops/...py`` execution starts with v5/ops on sys.path.
# Add the repository root before dynamically importing the analysis module.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "v5.selector-quality-requirement-receipt.v1"
CREATED_ON = "2026-08-23"
DEFAULT_PNL_CSV = Path(
    "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
    "pnl_sweep_2026_08_23.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "v4/audit/autoresearch/"
    "selector_quality_requirement_curve_2026_08_23_attempt001"
)

PINNED_PNL_SHA256 = (
    "30d628892be455e4bd31e4932a35244daf2bdee194851b2737eba42fbfd1d2d2"
)
PINNED_REPO_FILES = {
    "pnl_producer": (
        Path(
            "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
            "pnl_sweep_and_jackknife_2026_08_23.py"
        ),
        "ca109c6fbcacb9abb8e1bd2f19c0e1061c25bab303ea8f48632c06a5498f9792",
    ),
    "economics_module": (
        Path("v5/research/economics_race_join.py"),
        "0b223855dc6d4ae8d4a0d146b2e8a1ee8838e1d9f7597cf47d1716d1cb6a36b9",
    ),
    "alpha_ledger": (
        Path("v5/work/lifecycle-training/ALPHA_LEDGER.json"),
        "40517c1e19d2442e6969286e98e9e16c24a24c788b2f966eb2418771105e840c",
    ),
    "semantic_freeze_file": (
        Path(
            "v5/work/lifecycle-training/"
            "PREACQUISITION_SEMANTIC_FREEZE_V1.json"
        ),
        "d9beb72ffe7e5cbaf3003b269acd1faec8ddcc36f3a41ac481c0b3a3975c6ad0",
    ),
}

SEMANTIC_FREEZE = PINNED_REPO_FILES["semantic_freeze_file"][0]
SEMANTIC_FREEZE_SHA256 = (
    "71463cc0eeb4e242e43307e19a926ea4e258fd45a254c389e5a27110923893c4"
)
SEMANTIC_FREEZE_SOURCE_COUNT = 12
ALPHA_LEDGER = PINNED_REPO_FILES["alpha_ledger"][0]
ALPHA_EXPECTED = {
    "schema_version": "v5.autoresearch-alpha-ledger.v1",
    "experiments_run": 6,
    "head": "4551e28252af0838678424b63ab7deedcfc430f84825a92399d1ed76ef203b57",
    "next_bar": 0.66059463,
}

EXPECTED_PNL_COLUMNS = ("session", "start", "offset", "pnl", "why")
EXPECTED_PNL_ROWS = 10_078
EXPECTED_PNL_SESSIONS = 1_011
EXPECTED_PNL_SESSION_MIN = "2022-06-01"
EXPECTED_PNL_SESSION_MAX = "2026-07-30"
EXPECTED_REASON_COUNTS = {"horizon": 5_524, "stop": 4_554}
EXPECTED_CELL_COUNTS = {
    ("09:35", 0.0): 1_011,
    ("09:35", 10.0): 1_011,
    ("09:35", 25.0): 502,
    ("11:30", 0.0): 1_011,
    ("11:30", 10.0): 1_011,
    ("11:30", 25.0): 495,
    ("13:30", 0.0): 1_011,
    ("13:30", 10.0): 1_010,
    ("13:30", 25.0): 499,
    ("15:00", 0.0): 1_010,
    ("15:00", 10.0): 1_010,
    ("15:00", 25.0): 497,
}

TABLE_FILES = {
    "input_manifest": "input_manifest.csv",
    "requirement_curve": "requirement_curve.csv",
    "correlation_diagnostics": "correlation_diagnostics.csv",
    "inverse_requirement": "inverse_requirement.csv",
    "inverse_trace": "inverse_trace.csv",
}
EXPECTED_RESULT_ROWS = {
    "input_manifest": 12,
    "requirement_curve": 60,
    "correlation_diagnostics": 12,
    "inverse_requirement": 5,
    "inverse_trace": 505,
}
RESULT_KEYS = {
    "input_manifest": ("start", "offset"),
    "requirement_curve": (
        "partition",
        "nominal_gaussian_rho",
        "target_selection_rate",
    ),
    "correlation_diagnostics": ("partition", "nominal_gaussian_rho"),
    "inverse_requirement": ("target_selection_rate",),
    "inverse_trace": ("nominal_gaussian_rho", "target_selection_rate"),
}
REQUIRED_RESULT_COLUMNS = {
    "input_manifest": {
        "start",
        "offset",
        "sessions",
        "calibration_sessions",
        "later_sessions",
        "unselected_mean_pnl",
        "unselected_drop_best_mean_pnl",
        "analyzed_for_selector_curve",
    },
    "requirement_curve": {
        "partition",
        "nominal_gaussian_rho",
        "target_selection_rate",
        "mean_pnl_per_selected_ticket",
        "mean_pnl_ci_95_low",
        "mean_pnl_ci_95_high",
        "drop_best_mean_pnl_per_selected_ticket",
        "drop_best_mean_pnl_ci_95_low",
        "drop_best_mean_pnl_ci_95_high",
        "constructed_score_uses_same_partition_outcome",
        "predictive_signal_evidence",
        "genuine_oof_generalization_gap",
    },
    "correlation_diagnostics": {
        "partition",
        "nominal_gaussian_rho",
        "achieved_pearson_median",
        "achieved_spearman_median",
        "interval_type",
    },
    "inverse_requirement": {
        "target_selection_rate",
        "minimum_nominal_rho_grid",
        "minimum_world_robust_nominal_rho_grid",
        "genuine_outcome_blind_requirement",
        "predictive_signal_evidence",
        "world_robust_requirement_status",
    },
    "inverse_trace": {
        "nominal_gaussian_rho",
        "target_selection_rate",
        "drop_best_simultaneous_one_sided_95_lcb",
        "drop_best_selector_world_5th_percentile",
        "clears_session_and_selector_world_robustness",
        "predictive_signal_evidence",
    },
}
JSON_FILES = {
    "sanity_floor": "sanity_floor.json",
    "claim_audit": "claim_audit.json",
    "analysis_metadata": "analysis_metadata.json",
}


class SelectorRequirementRunError(RuntimeError):
    """The evidence wrapper cannot preserve its declared audit contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert scalar containers without silently accepting non-finite JSON."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path, date, datetime)):
        return value.isoformat() if isinstance(value, (date, datetime)) else str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _signed_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise SelectorRequirementRunError("receipt payload is already signed")
    normalized = _json_safe(payload)
    signed = dict(normalized)
    signed["receipt_sha256"] = hashlib.sha256(canonical_json(normalized)).hexdigest()
    return signed


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _archive_sources(
    output_dir: Path,
    repo_root: Path,
    archived: dict[str, tuple[Path, Path]],
) -> None:
    """Archive wrapper then analysis, retaining either copy on later failure."""

    sources = [
        ("archived_wrapper", Path(__file__).resolve()),
        (
            "archived_analysis_module",
            (repo_root / "v5/research/selector_quality_requirement.py").resolve(),
        ),
    ]
    for name, source in sources:
        if not source.is_file():
            raise SelectorRequirementRunError(f"analysis source is missing: {source}")
        destination = output_dir / source.name
        archived[name] = (source, destination)
        shutil.copyfile(source, destination)
        if file_sha256(destination) != file_sha256(source):
            raise SelectorRequirementRunError(f"source archive hash mismatch: {source}")


def _verify_archived_sources(
    archived: Mapping[str, tuple[Path, Path]],
) -> dict[str, dict[str, Any]]:
    expected = {"archived_wrapper", "archived_analysis_module"}
    if set(archived) != expected:
        raise SelectorRequirementRunError("required source archive is incomplete")
    verified: dict[str, dict[str, Any]] = {}
    for name, (source, destination) in archived.items():
        if not source.is_file() or not destination.is_file():
            raise SelectorRequirementRunError(f"source archive disappeared: {source}")
        source_hash = file_sha256(source)
        archived_hash = file_sha256(destination)
        if source_hash != archived_hash:
            raise SelectorRequirementRunError(f"source changed after archival: {source}")
        verified[name] = {
            "live_path": str(source),
            "archived_path": str(destination),
            "bytes": source.stat().st_size,
            "sha256": source_hash,
        }
    return verified


def _verify_pinned_repo_files(repo_root: Path) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, (relative, expected_hash) in PINNED_REPO_FILES.items():
        path = repo_root / relative
        if not path.is_file():
            raise SelectorRequirementRunError(f"pinned file is missing: {relative}")
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise SelectorRequirementRunError(f"pinned file drift: {relative}")
        verified[name] = {
            "path": str(relative),
            "bytes": path.stat().st_size,
            "sha256": observed_hash,
        }
    return verified


def _verify_semantic_freeze(repo_root: Path) -> dict[str, Any]:
    """Reproduce the canonical pre-acquisition self/source hash checks."""

    path = repo_root / SEMANTIC_FREEZE
    if not path.is_file():
        raise SelectorRequirementRunError(
            f"pre-acquisition semantic freeze is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorRequirementRunError(
            f"pre-acquisition semantic freeze is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SelectorRequirementRunError(
            "pre-acquisition semantic freeze is not a JSON object"
        )
    semantic = dict(payload)
    semantic.pop("freeze_sha256", None)
    actual = hashlib.sha256(canonical_json(semantic)).hexdigest()
    if payload.get("freeze_sha256") != actual or actual != SEMANTIC_FREEZE_SHA256:
        raise SelectorRequirementRunError(
            "pre-acquisition semantic freeze self-hash mismatch"
        )

    declared_sources = payload.get("source_files")
    if not isinstance(declared_sources, dict):
        raise SelectorRequirementRunError("semantic freeze source_files is not an object")
    if len(declared_sources) != SEMANTIC_FREEZE_SOURCE_COUNT:
        raise SelectorRequirementRunError(
            "semantic freeze source count drift: "
            f"expected {SEMANTIC_FREEZE_SOURCE_COUNT}, found {len(declared_sources)}"
        )
    verified: dict[str, str] = {}
    for relative, declared_hash in sorted(declared_sources.items()):
        source = repo_root / relative
        if not source.is_file():
            raise SelectorRequirementRunError(
                f"semantic freeze source is missing: {relative}"
            )
        observed_hash = file_sha256(source)
        if observed_hash != declared_hash:
            raise SelectorRequirementRunError(f"semantic freeze source drift: {relative}")
        verified[str(relative)] = observed_hash
    return {
        "path": str(SEMANTIC_FREEZE),
        "bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "freeze_sha256": actual,
        "verified_source_count": len(verified),
        "verified_source_sha256": verified,
    }


def _verify_alpha_ledger(repo_root: Path) -> dict[str, Any]:
    path = repo_root / ALPHA_LEDGER
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorRequirementRunError(f"alpha ledger is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise SelectorRequirementRunError("alpha ledger is not a JSON object")
    for field, expected in ALPHA_EXPECTED.items():
        observed = payload.get(field)
        if isinstance(expected, float):
            matches = isinstance(observed, (int, float)) and math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = observed == expected
        if not matches:
            raise SelectorRequirementRunError(
                f"alpha ledger {field} drift: expected {expected!r}, found {observed!r}"
            )
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != ALPHA_EXPECTED["experiments_run"]:
        raise SelectorRequirementRunError("alpha ledger entry count is inconsistent")
    return {
        "path": str(ALPHA_LEDGER),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        **ALPHA_EXPECTED,
        "entry_count": len(entries),
        "unchanged_by_study": True,
    }


def _verify_pnl_input(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectorRequirementRunError(f"P&L CSV is missing: {path}")
    observed_hash = file_sha256(path)
    if observed_hash != PINNED_PNL_SHA256:
        raise SelectorRequirementRunError(
            f"P&L CSV hash drift: expected {PINNED_PNL_SHA256}, found {observed_hash}"
        )
    frame = pd.read_csv(path)
    if tuple(frame.columns) != EXPECTED_PNL_COLUMNS:
        raise SelectorRequirementRunError(
            f"P&L schema drift: expected {EXPECTED_PNL_COLUMNS}, found {tuple(frame.columns)}"
        )
    if len(frame) != EXPECTED_PNL_ROWS:
        raise SelectorRequirementRunError(
            f"P&L row-count drift: expected {EXPECTED_PNL_ROWS}, found {len(frame)}"
        )
    if frame[list(EXPECTED_PNL_COLUMNS)].isna().any().any():
        raise SelectorRequirementRunError("P&L CSV contains missing values")
    sessions = sorted(frame["session"].astype(str).unique())
    if (
        len(sessions) != EXPECTED_PNL_SESSIONS
        or sessions[0] != EXPECTED_PNL_SESSION_MIN
        or sessions[-1] != EXPECTED_PNL_SESSION_MAX
    ):
        raise SelectorRequirementRunError("P&L session population drift")
    if any(session >= "2026-08-06" for session in sessions):
        raise SelectorRequirementRunError("P&L CSV opens a confirmation-reserved session")
    if frame.duplicated(["session", "start", "offset"]).any():
        raise SelectorRequirementRunError("P&L CSV contains duplicate session/cell keys")
    reasons = {str(key): int(value) for key, value in frame["why"].value_counts().items()}
    if reasons != EXPECTED_REASON_COUNTS:
        raise SelectorRequirementRunError("P&L exit-reason population drift")
    cell_counts = {
        (str(start), float(offset)): int(len(block))
        for (start, offset), block in frame.groupby(["start", "offset"], sort=True)
    }
    if cell_counts != EXPECTED_CELL_COUNTS:
        raise SelectorRequirementRunError("P&L cell population drift")
    numeric = frame[["offset", "pnl"]].to_numpy(float)
    if not math.isfinite(float(numeric.min())) or not math.isfinite(float(numeric.max())):
        raise SelectorRequirementRunError("P&L CSV contains non-finite numeric values")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": observed_hash,
        "schema": list(EXPECTED_PNL_COLUMNS),
        "rows": len(frame),
        "sessions": len(sessions),
        "first_session": sessions[0],
        "last_session": sessions[-1],
        "reserved_sessions_used": 0,
        "exit_reason_counts": reasons,
        "cell_counts": {
            f"{start}|{offset:g}": count
            for (start, offset), count in sorted(cell_counts.items())
        },
    }


def _integrity_snapshot(pnl_csv: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "pnl_input": _verify_pnl_input(pnl_csv),
        "pinned_repo_files": _verify_pinned_repo_files(repo_root),
        "alpha_ledger": _verify_alpha_ledger(repo_root),
        "semantic_freeze": _verify_semantic_freeze(repo_root),
    }


def _require_analysis_result(analysis: Any, result: Any) -> None:
    result_type = getattr(analysis, "StudyResult", None)
    if result_type is None or not isinstance(result, result_type):
        raise SelectorRequirementRunError("run_analysis did not return StudyResult")
    declared_schemas = getattr(analysis, "RESULT_TABLE_COLUMNS", None)
    if declared_schemas is not None and not isinstance(declared_schemas, Mapping):
        raise SelectorRequirementRunError("RESULT_TABLE_COLUMNS is not a mapping")
    for name in TABLE_FILES:
        frame = getattr(result, name, None)
        if not isinstance(frame, pd.DataFrame):
            raise SelectorRequirementRunError(f"StudyResult.{name} is not a DataFrame")
        if frame.empty:
            raise SelectorRequirementRunError(f"StudyResult.{name} is empty")
        if frame.columns.empty or frame.columns.has_duplicates:
            raise SelectorRequirementRunError(f"StudyResult.{name} schema is invalid")
        if any(not isinstance(column, str) or not column for column in frame.columns):
            raise SelectorRequirementRunError(
                f"StudyResult.{name} has a non-string or empty column name"
            )
        if len(frame) != EXPECTED_RESULT_ROWS[name]:
            raise SelectorRequirementRunError(
                f"StudyResult.{name} row drift: expected {EXPECTED_RESULT_ROWS[name]}, "
                f"found {len(frame)}"
            )
        missing_columns = REQUIRED_RESULT_COLUMNS[name] - set(frame.columns)
        if missing_columns:
            raise SelectorRequirementRunError(
                f"StudyResult.{name} is missing required columns: {sorted(missing_columns)}"
            )
        keys = list(RESULT_KEYS[name])
        if frame.duplicated(keys).any():
            raise SelectorRequirementRunError(
                f"StudyResult.{name} has duplicate keys: {keys}"
            )
        if name != "inverse_requirement":
            numeric = frame.select_dtypes(include="number").to_numpy(float)
            if numeric.size and not np.isfinite(numeric).all():
                raise SelectorRequirementRunError(
                    f"StudyResult.{name} contains non-finite numeric evidence"
                )
        if declared_schemas is not None:
            expected_columns = declared_schemas.get(name)
            if not isinstance(expected_columns, (list, tuple)) or not expected_columns:
                raise SelectorRequirementRunError(
                    f"RESULT_TABLE_COLUMNS has no schema for {name}"
                )
            if tuple(frame.columns) != tuple(expected_columns):
                raise SelectorRequirementRunError(
                    f"StudyResult.{name} schema drift: expected "
                    f"{tuple(expected_columns)}, found {tuple(frame.columns)}"
                )
    for name in JSON_FILES:
        value = getattr(result, name.removeprefix("analysis_"), None)
        if not isinstance(value, Mapping):
            raise SelectorRequirementRunError(f"StudyResult.{name} is not a mapping")
    if not isinstance(result.readable_tables, str) or not result.readable_tables.strip():
        raise SelectorRequirementRunError("StudyResult.readable_tables is empty")

    sanity = result.sanity_floor
    if (
        sanity.get("all_pass") is not True
        or sanity.get("passed_rows") != 10
        or sanity.get("total_rows") != 10
    ):
        raise SelectorRequirementRunError("rho=0 sanity floor did not PASS")
    metadata = result.metadata
    required_metadata = {
        "model_fit": False,
        "alpha_spent": False,
        "constructed_score_uses_same_partition_outcome": True,
        "predictive_signal_evidence": False,
        "genuine_outcome_blind_oof": False,
        "adoption": "ADOPT NOTHING",
    }
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            raise SelectorRequirementRunError(
                f"analysis metadata {field} must equal {expected!r}"
            )
    if result.claim_audit.get("predictive_signal_evidence") is not False:
        raise SelectorRequirementRunError(
            "claim audit must not classify the construction as predictive evidence"
        )
    reconstruction = result.claim_audit.get("exact_nonunique_reconstruction", {})
    if not math.isclose(
        float(reconstruction.get("mean_pnl", math.nan)),
        43.46890568042436,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise SelectorRequirementRunError("+$43.47 known-answer reconstruction drift")
    if not math.isclose(
        float(reconstruction.get("drop_best_mean_pnl", math.nan)),
        29.032078662938,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise SelectorRequirementRunError("claim drop-best known answer drift")

    manifest = result.input_manifest
    primary = manifest[(manifest["start"] == "09:35") & (manifest["offset"] == 0.0)]
    if len(primary) != 1:
        raise SelectorRequirementRunError("input manifest has no unique primary cell")
    primary_row = primary.iloc[0]
    if (
        int(primary_row["sessions"]) != 1_011
        or int(primary_row["calibration_sessions"]) != 768
        or int(primary_row["later_sessions"]) != 243
        or bool(primary_row["analyzed_for_selector_curve"]) is not True
    ):
        raise SelectorRequirementRunError("primary source-seam population drift")

    curve = result.requirement_curve
    if set(curve["partition"]) != {
        "calibration_in_sample_oracle",
        "later_outcome_conditioned_oracle",
    }:
        raise SelectorRequirementRunError("requirement-curve partition law drift")
    if set(curve["nominal_gaussian_rho"]) != {0.0, 0.02, 0.05, 0.10, 0.20, 0.40}:
        raise SelectorRequirementRunError("requirement-curve rho law drift")
    if set(curve["target_selection_rate"]) != {0.05, 0.10, 0.20, 0.50, 1.00}:
        raise SelectorRequirementRunError("requirement-curve rate law drift")
    if not curve["constructed_score_uses_same_partition_outcome"].all():
        raise SelectorRequirementRunError("curve lost its outcome-conditioned label")
    if curve["predictive_signal_evidence"].any():
        raise SelectorRequirementRunError("curve improperly claims predictive evidence")

    inverse = result.inverse_requirement
    if set(inverse["genuine_outcome_blind_requirement"]) != {"UNKNOWN"}:
        raise SelectorRequirementRunError("inverse improperly claims a causal requirement")
    if inverse["predictive_signal_evidence"].any():
        raise SelectorRequirementRunError("inverse improperly claims predictive evidence")
    unknown = inverse["world_robust_requirement_status"] == "UNKNOWN_ABOVE_GRID_MAXIMUM"
    known = inverse["world_robust_requirement_status"] == "FINITE_WITHIN_GRID"
    if not (unknown | known).all():
        raise SelectorRequirementRunError("inverse requirement status is undeclared")
    if inverse.loc[known, "minimum_world_robust_nominal_rho_grid"].isna().any():
        raise SelectorRequirementRunError("finite inverse requirement is missing its rho")
    if inverse.loc[unknown, "minimum_world_robust_nominal_rho_grid"].notna().any():
        raise SelectorRequirementRunError("unknown inverse requirement contains false precision")


def _table_artifacts(output_dir: Path, result: Any) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, filename in TABLE_FILES.items():
        path = output_dir / filename
        frame = getattr(result, name)
        frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")
        artifacts[name] = _artifact(path, rows=len(frame))
    for name, filename in JSON_FILES.items():
        path = output_dir / filename
        attribute = name.removeprefix("analysis_")
        _write_json(path, getattr(result, attribute))
        artifacts[name] = _artifact(path)
    readable_path = output_dir / "readable_tables.md"
    readable = result.readable_tables
    if not readable.endswith("\n"):
        readable += "\n"
    readable_path.write_text(readable, encoding="utf-8")
    artifacts["readable_tables"] = _artifact(readable_path)
    return artifacts


def _success_payload(
    *,
    pnl_csv: Path,
    result: Any,
    artifacts: dict[str, dict[str, Any]],
    integrity_before: dict[str, Any],
    integrity_after: dict[str, Any],
    archived_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_on": CREATED_ON,
        "status": "PASS",
        "purpose": (
            "outcome-conditioned selector-quality requirement curve over an existing "
            "P&L table; a requirements study, not a signal test"
        ),
        "pnl_csv": str(pnl_csv.resolve()),
        "population": {
            "rows": EXPECTED_PNL_ROWS,
            "sessions": EXPECTED_PNL_SESSIONS,
            "cells": len(EXPECTED_CELL_COUNTS),
            "reserved_sessions_used": 0,
        },
        "scope_law": {
            "pnl_rebuilt": False,
            "model_fit": False,
            "calibration_cutoff_frozen_before_later_partition": True,
            "market_feature_tested": False,
            "predictive_signal_evidence": False,
            "alpha_spent": False,
            "vendor_or_broker_contact": False,
            "purchase_or_order": False,
        },
        "interpretation": {
            "real_signal_supplied_or_tested": False,
            "real_signal_existence": "UNKNOWN_NOT_TESTED",
            "constructed_score_uses_same_partition_outcome": True,
            "genuine_outcome_blind_oof": False,
            "later_partition_role": (
                "outcome-conditioned counterfactual stability check, not predictive OOF"
            ),
            "correlation_is_predictability": False,
            "adoption": "ADOPT NOTHING",
        },
        "quality_control": {
            "status": "PASS",
            "strict_pnl_population": True,
            "analysis_result_contract": "PASS",
            "rho_zero_sanity_floor": "PASS",
            "input_and_governance_reverified_after_analysis": True,
            "live_sources_match_archives_after_analysis": True,
        },
        "failure_path": (
            "after attempt creation, preserve archived sources, all partial artifacts, a "
            "traceback run.log, and a canonical self-hashed failure_receipt.json; before "
            "attempt creation, refuse collisions without mutation and exit nonzero"
        ),
        "table_rows": {
            name: len(getattr(result, name)) for name in TABLE_FILES
        },
        "analysis_metadata": dict(result.metadata),
        "sanity_floor_summary": {
            "all_pass": result.sanity_floor["all_pass"],
            "passed_rows": result.sanity_floor["passed_rows"],
            "total_rows": result.sanity_floor["total_rows"],
        },
        "claim_audit": dict(result.claim_audit),
        "integrity_before": integrity_before,
        "integrity_after": integrity_after,
        "integrity_before_equals_after": integrity_before == integrity_after,
        "archived_sources": archived_sources,
        "artifacts": artifacts,
    }


def _failure_artifacts(
    output_dir: Path,
    archived: Mapping[str, tuple[Path, Path]],
    log_path: Path,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, Path] = {
        name: destination for name, (_, destination) in archived.items()
    }
    candidates["run_log"] = log_path
    candidates.update(
        {
            f"partial_{name}": output_dir / filename
            for name, filename in {**TABLE_FILES, **JSON_FILES}.items()
        }
    )
    candidates["partial_readable_tables"] = output_dir / "readable_tables.md"
    candidates["partial_pass_receipt"] = output_dir / "receipt.json"
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in candidates.items():
        if not path.is_file():
            continue
        rows: int | None = None
        if path.suffix == ".csv":
            try:
                rows = len(pd.read_csv(path))
            except Exception:  # noqa: BLE001 - malformed partials remain evidence
                rows = None
        artifacts[name] = _artifact(path, rows=rows)
    return artifacts


def run(
    pnl_csv: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run one immutable attempt and return its PASS/FAIL receipt."""

    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise SelectorRequirementRunError(
            f"refusing to overwrite existing attempt: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    pnl_path = Path(pnl_csv)
    if not pnl_path.is_absolute():
        pnl_path = root / pnl_path
    pnl_path = pnl_path.resolve()
    log_path = output_dir / "run.log"
    archived: dict[str, tuple[Path, Path]] = {}
    integrity_before: dict[str, Any] = {}
    integrity_after: dict[str, Any] = {}
    try:
        # This must remain the first in-attempt action.
        _archive_sources(output_dir, root, archived)
        archived_before = _verify_archived_sources(archived)
        integrity_before = _integrity_snapshot(pnl_path, root)

        analysis = importlib.import_module("v5.research.selector_quality_requirement")
        analysis_path = Path(analysis.__file__).resolve()
        expected_path = (root / "v5/research/selector_quality_requirement.py").resolve()
        if analysis_path != expected_path:
            raise SelectorRequirementRunError(
                f"imported analysis module from {analysis_path}, expected {expected_path}"
            )
        entrypoint = getattr(analysis, "run_analysis", None)
        if not callable(entrypoint):
            raise SelectorRequirementRunError("analysis module has no callable run_analysis")
        result = entrypoint(pnl_path, strict_population=True)
        _require_analysis_result(analysis, result)

        artifacts = _table_artifacts(output_dir, result)
        integrity_after = _integrity_snapshot(pnl_path, root)
        if integrity_before != integrity_after:
            raise SelectorRequirementRunError(
                "pinned input or governance state changed during analysis"
            )
        archived_after = _verify_archived_sources(archived)
        if archived_before != archived_after:
            raise SelectorRequirementRunError(
                "live or archived analysis source changed during analysis"
            )

        log_lines = [
            "PASS",
            f"pnl_rows={EXPECTED_PNL_ROWS}",
            f"pnl_sessions={EXPECTED_PNL_SESSIONS}",
            *[
                f"{name}_rows={len(getattr(result, name))}"
                for name in TABLE_FILES
            ],
            "rho_zero_sanity_floor=PASS",
            "predictive_signal_evidence=false",
            "alpha_spent=false",
            "adoption=ADOPT NOTHING",
        ]
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        artifacts.update(
            {
                name: _artifact(destination)
                for name, (_, destination) in archived.items()
            }
        )
        artifacts["run_log"] = _artifact(log_path)
        receipt = _signed_receipt(
            _success_payload(
                pnl_csv=pnl_path,
                result=result,
                artifacts=artifacts,
                integrity_before=integrity_before,
                integrity_after=integrity_after,
                archived_sources=archived_after,
            )
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    except Exception as exc:  # noqa: BLE001 - preserve every failed attempt
        failure_traceback = traceback.format_exc()
        log_path.write_text(f"FAIL\n{failure_traceback}", encoding="utf-8")
        failure_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_on": CREATED_ON,
            "status": "FAIL",
            "purpose": "preserved failed selector-quality requirement attempt",
            "pnl_csv": str(pnl_path),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": failure_traceback,
            "failure_behavior": (
                "attempt directory, all available source archives, traceback log, "
                "partial artifacts, and self-hashed failure receipt preserved"
            ),
            "artifacts": _failure_artifacts(output_dir, archived, log_path),
        }
        if integrity_before:
            failure_payload["integrity_before"] = integrity_before
        if integrity_after:
            failure_payload["integrity_after"] = integrity_after
        failure = _signed_receipt(failure_payload)
        _write_json(output_dir / "failure_receipt.json", failure)
        return failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnl-csv", type=Path, default=DEFAULT_PNL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        receipt = run(args.pnl_csv, args.output_dir)
    except Exception as exc:  # no directory exists in the pre-attempt failure path
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    receipt_name = "receipt.json" if receipt["status"] == "PASS" else "failure_receipt.json"
    print(args.output_dir / receipt_name)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
