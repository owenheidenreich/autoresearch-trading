"""Create immutable evidence for the unconditional SPX barrier-race census.

This wrapper exposes only input and attempt paths.  The reviewed research
module owns every clock, horizon, threshold, population, and estimand.  It
never contacts a vendor or broker, fits a model, reads reserved confirmation
sessions, or reconstructs historical option P&L.

An attempt directory is immutable.  Once it is created, source archival and
all later work occur inside the caught failure region so even an archival
failure leaves a traceback log and a self-hashed failure receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import traceback
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Direct ``python v5/ops/...py`` execution starts with v5/ops on sys.path.
# Add the repository root before dynamically importing the analysis module.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "v5.unconditional-spx-race-receipt.v1"
CREATED_ON = "2026-08-23"
DEFAULT_TAPE_ROOT = Path(
    "/Volumes/AR_TRADING_DATA/spx_parity_tape_2022-06-01_2026-07-31"
)
DEFAULT_LADDER_ROOT = Path(
    "/Volumes/AR_TRADING_DATA/"
    "lifecycle_corpus_spx_tape_2022-06-01_2026-07-31/ladder"
)
DEFAULT_OUTPUT_DIR = Path(
    "v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001"
)

SEMANTIC_FREEZE = Path(
    "v5/work/lifecycle-training/PREACQUISITION_SEMANTIC_FREEZE_V1.json"
)
SEMANTIC_FREEZE_SHA256 = (
    "71463cc0eeb4e242e43307e19a926ea4e258fd45a254c389e5a27110923893c4"
)
SEMANTIC_FREEZE_SOURCE_COUNT = 12

UPSTREAM_EVIDENCE = {
    "parity_tape_build_receipt": Path(
        "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
        "parity_spot_tape_receipt.json"
    ),
    "corpus_build_receipt": Path(
        "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
        "corpus_build_spx_tape_receipt.json"
    ),
    "interior_freeze_audit_receipt": Path(
        "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
        "interior_full_book_freeze_audit_2026_08_22_attempt002.json"
    ),
    "two_era_structural_audit_receipt": Path(
        "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/"
        "two_era_structural_audit_2026_08_22_attempt001.json"
    ),
}

TABLE_FILES = {
    "race_surface": "race_surface.csv",
    "overshoot_distribution": "overshoot_distribution.csv",
    "time_to_event_distribution": "time_to_event_distribution.csv",
    "stop_compatibility": "stop_compatibility.csv",
    "conditional_quantiles": "conditional_quantiles.csv",
    "iv_by_time": "iv_by_time.csv",
    "iv_clock_change": "iv_clock_change.csv",
    "iv_session_values": "iv_session_values.csv",
}


class RaceRunError(RuntimeError):
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _signed_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise RaceRunError("receipt payload is already signed")
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
    """Copy sources one at a time while retaining partial failure evidence."""

    sources = [("archived_wrapper", Path(__file__).resolve())]
    required = repo_root / "v5/research/unconditional_spx_race.py"
    optional_iv = repo_root / "v5/research/unconditional_spx_iv.py"
    for name, source in sources:
        destination = output_dir / source.name
        archived[name] = (source, destination)
        shutil.copyfile(source, destination)
        if file_sha256(destination) != file_sha256(source):
            raise RaceRunError(f"source archive hash mismatch: {source}")
    if not required.is_file():
        raise RaceRunError(f"analysis source is missing: {required}")
    remaining = [("archived_analysis_module", required.resolve())]
    if optional_iv.is_file():
        remaining.append(("archived_iv_module", optional_iv.resolve()))
    for name, source in remaining:
        destination = output_dir / source.name
        archived[name] = (source, destination)
        shutil.copyfile(source, destination)
        if file_sha256(destination) != file_sha256(source):
            raise RaceRunError(f"source archive hash mismatch: {source}")


def _verify_archived_sources(
    archived: Mapping[str, tuple[Path, Path]],
    repo_root: Path,
) -> None:
    expected_names = {"archived_wrapper", "archived_analysis_module"}
    if not expected_names.issubset(archived):
        raise RaceRunError("required source archive is incomplete")
    optional_iv = repo_root / "v5/research/unconditional_spx_iv.py"
    if optional_iv.is_file() and "archived_iv_module" not in archived:
        raise RaceRunError("present IV companion source was not archived")
    for source, destination in archived.values():
        if not source.is_file() or not destination.is_file():
            raise RaceRunError(f"source archive disappeared: {source}")
        if file_sha256(source) != file_sha256(destination):
            raise RaceRunError(f"source changed after archival: {source}")


def _upstream_hashes(repo_root: Path) -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, relative in UPSTREAM_EVIDENCE.items():
        path = repo_root / relative
        if not path.is_file():
            missing.append(str(relative))
            continue
        hashes[name] = {
            "path": str(relative),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    if missing:
        raise RaceRunError(f"upstream evidence is missing: {missing}")
    return hashes


def _verify_semantic_freeze(repo_root: Path) -> dict[str, Any]:
    """Reproduce the canonical pre-acquisition self/source hash checks."""

    path = repo_root / SEMANTIC_FREEZE
    if not path.is_file():
        raise RaceRunError(f"pre-acquisition semantic freeze is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RaceRunError(f"pre-acquisition semantic freeze is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RaceRunError("pre-acquisition semantic freeze is not a JSON object")
    semantic = dict(payload)
    semantic.pop("freeze_sha256", None)
    actual = hashlib.sha256(canonical_json(semantic)).hexdigest()
    if payload.get("freeze_sha256") != actual or actual != SEMANTIC_FREEZE_SHA256:
        raise RaceRunError("pre-acquisition semantic freeze self-hash mismatch")

    declared_sources = payload.get("source_files")
    if not isinstance(declared_sources, dict):
        raise RaceRunError("semantic freeze source_files is not an object")
    if len(declared_sources) != SEMANTIC_FREEZE_SOURCE_COUNT:
        raise RaceRunError(
            "semantic freeze source count drift: "
            f"expected {SEMANTIC_FREEZE_SOURCE_COUNT}, found {len(declared_sources)}"
        )
    verified: dict[str, str] = {}
    for relative, declared_hash in sorted(declared_sources.items()):
        source = repo_root / relative
        if not source.is_file():
            raise RaceRunError(f"semantic freeze source is missing: {relative}")
        observed_hash = file_sha256(source)
        if observed_hash != declared_hash:
            raise RaceRunError(f"semantic freeze source drift: {relative}")
        verified[str(relative)] = observed_hash
    return {
        "path": str(SEMANTIC_FREEZE),
        "bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "freeze_sha256": actual,
        "verified_source_count": len(verified),
        "verified_source_sha256": verified,
    }


def _require_analysis_result(analysis: Any, result: Any) -> None:
    result_type = getattr(analysis, "RaceAnalysisRun", None)
    if result_type is None or not isinstance(result, result_type):
        raise RaceRunError("run_analysis did not return RaceAnalysisRun")
    if not isinstance(result.input_manifest, pd.DataFrame):
        raise RaceRunError("RaceAnalysisRun.input_manifest is not a DataFrame")
    if not isinstance(result.defect_disposition, pd.DataFrame):
        raise RaceRunError("RaceAnalysisRun.defect_disposition is not a DataFrame")
    if not isinstance(result.readable_tables, str) or not result.readable_tables.strip():
        raise RaceRunError("RaceAnalysisRun.readable_tables is empty")
    if not isinstance(result.quality_control, Mapping):
        raise RaceRunError("RaceAnalysisRun.quality_control is not a mapping")
    if result.quality_control.get("status") != "PASS":
        raise RaceRunError("RaceAnalysisRun quality control did not PASS")
    for name in TABLE_FILES:
        frame = getattr(result.tables, name, None)
        if not isinstance(frame, pd.DataFrame):
            raise RaceRunError(f"AnalysisTables.{name} is not a DataFrame")


def _table_artifacts(output_dir: Path, result: Any) -> dict[str, dict[str, Any]]:
    paths = {
        "input_manifest": output_dir / "input_manifest.csv",
        "defect_disposition": output_dir / "defect_disposition.csv",
        "readable_tables": output_dir / "readable_tables.md",
    }
    result.input_manifest.to_csv(paths["input_manifest"], index=False)
    result.defect_disposition.to_csv(paths["defect_disposition"], index=False)
    readable = result.readable_tables
    if not readable.endswith("\n"):
        readable += "\n"
    paths["readable_tables"].write_text(readable, encoding="utf-8")

    rows = {
        "input_manifest": len(result.input_manifest),
        "defect_disposition": len(result.defect_disposition),
    }
    for name, filename in TABLE_FILES.items():
        path = output_dir / filename
        frame = getattr(result.tables, name)
        frame.to_csv(path, index=False)
        paths[name] = path
        rows[name] = len(frame)
    return {
        name: _artifact(path, rows=rows.get(name))
        for name, path in paths.items()
    }


def _success_payload(
    *,
    tape_root: Path,
    ladder_root: Path,
    result: Any,
    artifacts: dict[str, dict[str, Any]],
    upstream: dict[str, dict[str, Any]],
    semantic_freeze: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_on": CREATED_ON,
        "status": "PASS",
        "purpose": (
            "unconditional SPX snapshot barrier races, path distributions, and "
            "descriptive causal-ladder ATM IV clock census; not a strategy test"
        ),
        "tape_root": str(tape_root.resolve()),
        "ladder_root": str(ladder_root.resolve()),
        "population": {
            "input_manifest_rows": len(result.input_manifest),
            "defect_disposition_rows": len(result.defect_disposition),
            "reserved_sessions_used": 0,
        },
        "scope_law": {
            "path_selection": "exact clock and horizon only",
            "feature_conditioning": False,
            "signal_conditioning": False,
            "outcome_conditioning": False,
            "model_fit": False,
            "threshold_tuning": False,
            "historical_option_pnl_join": False,
            "vendor_or_broker_contact": False,
        },
        "measurement_law": {
            "race_states": ["favourable_first", "adverse_first", "neither"],
            "touch_definition": "first observed one-minute parity snapshot crossing",
            "unit_of_inference": "session",
            "iv_primary_estimand": (
                "equal-weighted call/put median self-IV in the pre-existing "
                "+/-10 SPX-point ATM band"
            ),
            "iv_interpretation": "descriptive ATM clock curve only; no OTM-wing extrapolation",
        },
        "claim_classification": {
            "unconditional_market_structure": True,
            "strategy_proposed": False,
            "adoption": "ADOPT NOTHING",
        },
        "limitations": [
            "Snapshot races can miss intraminute touches and cannot recover intraminute order.",
            "Unconditional path frequencies do not establish option profitability or signal value.",
            "The IV companion describes the observed ATM band and cannot identify "
            "absent 40- or 60-point OTM wing IV.",
            "Source era and calendar time remain confounded in cross-era descriptions.",
        ],
        "table_rows": {
            name: len(getattr(result.tables, name)) for name in TABLE_FILES
        },
        "quality_control": dict(result.quality_control),
        "semantic_freeze": semantic_freeze,
        "upstream_evidence": upstream,
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
            "partial_input_manifest": output_dir / "input_manifest.csv",
            "partial_defect_disposition": output_dir / "defect_disposition.csv",
            "partial_readable_tables": output_dir / "readable_tables.md",
            **{
                f"partial_{name}": output_dir / filename
                for name, filename in TABLE_FILES.items()
            },
        }
    )
    return {
        name: _artifact(path)
        for name, path in candidates.items()
        if path.is_file()
    }


def run(
    tape_root: Path,
    ladder_root: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run one immutable attempt and return its PASS/FAIL receipt."""

    tape_root = Path(tape_root)
    ladder_root = Path(ladder_root)
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise RaceRunError(f"refusing to overwrite existing attempt: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    log_path = output_dir / "run.log"
    archived: dict[str, tuple[Path, Path]] = {}
    upstream: dict[str, dict[str, Any]] = {}
    semantic_freeze: dict[str, Any] = {}
    try:
        _archive_sources(output_dir, root, archived)
        semantic_freeze = _verify_semantic_freeze(root)
        upstream = _upstream_hashes(root)
        if not tape_root.is_dir():
            raise RaceRunError(f"tape root is missing: {tape_root}")
        if not ladder_root.is_dir():
            raise RaceRunError(f"ladder root is missing: {ladder_root}")

        analysis = importlib.import_module("v5.research.unconditional_spx_race")
        analysis_path = Path(analysis.__file__).resolve()
        expected_path = (root / "v5/research/unconditional_spx_race.py").resolve()
        if analysis_path != expected_path:
            raise RaceRunError(
                f"imported analysis module from {analysis_path}, expected {expected_path}"
            )
        _verify_archived_sources(archived, root)
        entrypoint = getattr(analysis, "run_analysis", None)
        if not callable(entrypoint):
            raise RaceRunError("analysis module has no callable run_analysis")
        result = entrypoint(tape_root, ladder_root, strict_population=True)
        _require_analysis_result(analysis, result)
        _verify_archived_sources(archived, root)

        artifacts = _table_artifacts(output_dir, result)
        log_lines = [
            "PASS",
            f"input_manifest_rows={len(result.input_manifest)}",
            f"defect_disposition_rows={len(result.defect_disposition)}",
            *[
                f"{name}_rows={len(getattr(result.tables, name))}"
                for name in TABLE_FILES
            ],
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
                tape_root=tape_root,
                ladder_root=ladder_root,
                result=result,
                artifacts=artifacts,
                upstream=upstream,
                semantic_freeze=semantic_freeze,
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
            "purpose": "preserved failed unconditional SPX barrier-race attempt",
            "tape_root": str(tape_root.resolve()),
            "ladder_root": str(ladder_root.resolve()),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": failure_traceback,
            "failure_behavior": (
                "attempt directory, all available source archives, traceback log, "
                "partial artifacts, and self-hashed failure receipt preserved"
            ),
            "artifacts": _failure_artifacts(output_dir, archived, log_path),
        }
        if semantic_freeze:
            failure_payload["semantic_freeze"] = semantic_freeze
        if upstream:
            failure_payload["upstream_evidence"] = upstream
        failure = _signed_receipt(failure_payload)
        _write_json(output_dir / "failure_receipt.json", failure)
        return failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape-root", type=Path, default=DEFAULT_TAPE_ROOT)
    parser.add_argument("--ladder-root", type=Path, default=DEFAULT_LADDER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        receipt = run(args.tape_root, args.ladder_root, args.output_dir)
    except Exception as exc:  # no directory exists in the pre-attempt failure path
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    receipt_name = "receipt.json" if receipt["status"] == "PASS" else "failure_receipt.json"
    print(args.output_dir / receipt_name)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
