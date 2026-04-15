"""Observability helpers: identity blocks, schema headers, run scoping.

Every observability artifact (training logs, build reports, diagnostics)
uses these helpers so that:
- Artifacts are run-scoped (no accidental overwrites)
- Every file carries a schema version (no silent format drift)
- Every file carries an identity block (cross-stage joining)
- Stage lifecycle is tracked (started/completed/failed)

Created 2026-04-15 as part of the pipeline failure observability plan.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from v2.core.config import RUNTIME_CONFIG


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def generate_run_id(experiment_id: str | None = None, stage: str = "unknown") -> str:
    """Generate a run ID.

    If experiment_id is provided (e.g. 'exp_160'), use it directly.
    Otherwise, generate '{stage}_{timestamp}'.
    """
    if experiment_id:
        return experiment_id
    return f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}"


def ensure_run_dir(run_id: str, base: str = "v2/runs") -> Path:
    """Create and return the run directory for a given run_id."""
    d = Path(base) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Git SHA
# ---------------------------------------------------------------------------

def get_git_sha() -> str:
    """Return short git SHA, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Identity block
# ---------------------------------------------------------------------------

def identity_block(
    run_id: str,
    stage: str,
    *,
    dataset_fingerprint: str = "",
    config_fingerprint: str | None = None,
    evaluator_fingerprint: str = "",
    policy_fingerprint: str = "",
    source_checkpoint: str = "",
    **extra: str,
) -> dict:
    """Standard identity block included in every observability artifact."""
    block = {
        "run_id": run_id,
        "git_sha": get_git_sha(),
        "dataset_fingerprint": dataset_fingerprint,
        "config_fingerprint": config_fingerprint or RUNTIME_CONFIG.fingerprint(),
        "stage": stage,
        "source_checkpoint": source_checkpoint,
    }
    if evaluator_fingerprint:
        block["evaluator_fingerprint"] = evaluator_fingerprint
    if policy_fingerprint:
        block["policy_fingerprint"] = policy_fingerprint
    block.update(extra)
    return block


# ---------------------------------------------------------------------------
# Schema header
# ---------------------------------------------------------------------------

def schema_header(
    schema_name: str,
    schema_version: str,
    producer: str,
) -> dict:
    """Schema + provenance header for any JSON/JSONL artifact."""
    return {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "producer": producer,
    }


# ---------------------------------------------------------------------------
# JSONL log helpers
# ---------------------------------------------------------------------------

def write_jsonl_header(
    path: Path | str,
    schema_name: str,
    schema_version: str,
    producer: str,
    identity: dict,
    **extra,
) -> None:
    """Write the header line of a JSONL log file.

    The header has record_type='header', status='started', plus schema
    and identity information.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "record_type": "header",
        "status": "started",
        **schema_header(schema_name, schema_version, producer),
        **identity,
        **extra,
    }
    with open(path, "w") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def append_jsonl(path: Path | str, record: dict) -> None:
    """Append a single JSON line and flush immediately (survives lease death)."""
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def write_jsonl_summary(
    path: Path | str,
    status: str,
    *,
    error_message: str | None = None,
    **extra,
) -> None:
    """Write the final summary line of a JSONL log file."""
    record = {
        "record_type": "summary",
        "status": status,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if error_message:
        record["error_message"] = error_message
    record.update(extra)
    append_jsonl(path, record)


# ---------------------------------------------------------------------------
# Triage index
# ---------------------------------------------------------------------------

def write_triage_index(
    run_dir: Path | str,
    run_id: str,
    identity: dict,
    *,
    verdict: str = "",
    score: float | None = None,
    gate_failure: str | None = None,
    artifacts: dict[str, str] | None = None,
) -> Path:
    """Write triage_index.json -- the single entry point for a run's evidence."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    index = {
        **schema_header("triage_index", "1.0", "v2.core.observability"),
        "run_id": run_id,
        "verdict": verdict,
        "score": score,
        "gate_failure": gate_failure,
        "artifacts": artifacts or {},
        "fingerprints": identity,
    }
    path = run_dir / "triage_index.json"
    path.write_text(json.dumps(index, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# File checksums
# ---------------------------------------------------------------------------

def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Artifact presence validation
# ---------------------------------------------------------------------------

def validate_artifact_presence(
    run_dir: Path | str,
    required: list[str] | None = None,
) -> list[str]:
    """Check that required observability artifacts exist and are valid JSON.

    Returns list of error strings (empty = all present and valid).
    """
    run_dir = Path(run_dir)
    if required is None:
        required = ["training_log.jsonl", "eval_report.json", "replay_diagnostics.json"]

    errors = []
    for name in required:
        p = run_dir / name
        if not p.exists():
            errors.append(f"missing: {name}")
            continue
        # Validate JSON/JSONL is parseable
        try:
            text = p.read_text().strip()
            if name.endswith(".jsonl"):
                for i, line in enumerate(text.split("\n")):
                    if line.strip():
                        json.loads(line)
            else:
                json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            errors.append(f"malformed: {name} ({e})")
    return errors
