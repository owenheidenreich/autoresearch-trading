"""Acquire the owner-authorized clean Walking-Skeleton Stage-0 CBBO-1s slice.

This runner is intentionally fail-closed and narrow.  It can only:

1. prove the fixed 13-session slice is in the governed census and outside
   every enumerated protected role;
2. extract exact SPXW PM-settled 0DTE raw symbols for the 11 missing sessions;
3. estimate the complete request before any paid download; and
4. download those 11 requests as OPRA.PILLAR ``cbbo-1s``.

It cannot select another date, schema, dataset, or symbol source.  It does not
train a model, build tensors, contact a broker, or mutate runtime/promotion
state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


GOAL = "WALKING-SKELETON-STAGE-0-CORRECTION-CLEAN-1S-ACQUISITION"
DATASET = "OPRA.PILLAR"
SCHEMA = "cbbo-1s"
STYPE_IN = "raw_symbol"
HARD_CAP_USD = 15.0

AUTHORITY_PATH = Path(
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
AUTHORITY_SHA256 = (
    "edcbee06ebfc5ac3a26fa13da043754589ba55fbbd11b207906e19459d4103f3"
)
GRAPH_PATH = Path(
    "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2.json"
)
GRAPH_SHA256 = (
    "9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08"
)
CENSUS_PATH = Path(
    "v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/"
    "census_sessions.json"
)
CENSUS_SHA256 = (
    "76cc65e87d26dd2c4d40fd10df3abc9d6565d36bf0f20f876816489a67c9f6e0"
)
SEALED_ASSIGNMENT_PATH = Path(
    "v4/audit/autoresearch/protocol101_sealed_day_assignment/"
    "last_assignment_run.json"
)
SEALED_ASSIGNMENT_SHA256 = (
    "d6d622c5d6d06149d25c77676ce7f313f2bfead22867c09cf3292e655fff1369"
)
WALKING_PLAN_PATH = Path(
    "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_WALKING_SKELETON_DRYRUN_PLAN_2026_07_30.md"
)
WALKING_PLAN_SHA256 = (
    "5826baa7d369f9af6e18a96cf3536ed95711d705d7775cfa48943c3d2f40ca61"
)

OWNED_D55_DATES = ("2025-02-20", "2025-03-03")
DOWNLOAD_DATES = (
    "2025-02-21",
    "2025-02-24",
    "2025-02-25",
    "2025-02-26",
    "2025-02-27",
    "2025-02-28",
    "2025-03-04",
    "2025-03-05",
    "2025-03-06",
    "2025-03-07",
    "2025-03-10",
)
SLICE_DATES = (
    "2025-02-20",
    "2025-02-21",
    "2025-02-24",
    "2025-02-25",
    "2025-02-26",
    "2025-02-27",
    "2025-02-28",
    "2025-03-03",
    "2025-03-04",
    "2025-03-05",
    "2025-03-06",
    "2025-03-07",
    "2025-03-10",
)

DEFAULT_NORMALIZED_DIR = Path("v4/normalized")
DEFAULT_D55_ROOT = Path("v4/raw/opra_1s_pilot")
DEFAULT_RAW_ROOT = Path("v4/raw/opra_1s_walking_skeleton")
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_walking_skeleton_stage0"
)
DEFAULT_ENV_FILE = Path("v4/.env")

APPROVAL_TEXT = (
    "I authorize the Walking-Skeleton Stage-0 correction download for exactly "
    "11 firewall-safe SPXW 0DTE sessions using Databento OPRA.PILLAR cbbo-1s "
    "with a hard cap of $15; no other paid pull is permitted."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prepare", "cost-gate", "download", "verify", "review"),
        required=True,
    )
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--d55-root", type=Path, default=DEFAULT_D55_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    add_paid_data_approval_args(
        parser,
        default_manifest=DEFAULT_OUT_DIR / "acquisition_plan.json",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual} != {expected}")


def verify_pinned_authority() -> None:
    _assert_hash(AUTHORITY_PATH, AUTHORITY_SHA256, "consolidated authority")
    _assert_hash(GRAPH_PATH, GRAPH_SHA256, "Graph V2")
    _assert_hash(CENSUS_PATH, CENSUS_SHA256, "FT2-04 census")
    _assert_hash(
        SEALED_ASSIGNMENT_PATH,
        SEALED_ASSIGNMENT_SHA256,
        "sealed-day assignment",
    )
    _assert_hash(WALKING_PLAN_PATH, WALKING_PLAN_SHA256, "walking-skeleton plan")


def _fold_intersections(
    session: str,
    fold_sessions: dict[str, list[str]],
) -> list[str]:
    return sorted(
        fold for fold, sessions in fold_sessions.items() if session in sessions
    )


def build_firewall_proof() -> dict[str, Any]:
    verify_pinned_authority()
    census = json.loads(CENSUS_PATH.read_text())
    assignment = json.loads(SEALED_ASSIGNMENT_PATH.read_text())
    census_sessions = set(map(str, census["census_sessions"]))
    embargo_sessions = set(map(str, census["embargo_sessions"]))
    protected_sessions = set(map(str, census["protected_holdout_sessions"]))
    outer_by_fold = {
        str(fold): list(map(str, sessions))
        for fold, sessions in census["outer_test_sessions_by_fold"].items()
    }
    recorder_protected = {
        str(row["session"])
        for row in assignment["actions"]
        if str(row.get("class")) in {"sealed", "burned"}
    }
    rows: list[dict[str, Any]] = []
    for session in SLICE_DATES:
        outer_folds = _fold_intersections(session, outer_by_fold)
        row = {
            "date": session,
            "in_governed_census": session in census_sessions,
            "outer_test_folds": outer_folds,
            "outer_test_selected": bool(outer_folds),
            "embargo_selected": session in embargo_sessions,
            "protected_holdout_selected": session in protected_sessions,
            "fresh_confirmation_selected": False,
            "fresh_confirmation_evidence": (
                "The pinned D42 authority defines the governed census after "
                "subtracting any owner-reserved confirmation sessions once "
                "frozen. This date is present in the pinned census manifest."
            ),
            "sealed_or_burned_recorder_selected": session in recorder_protected,
        }
        row["firewall_safe"] = bool(
            row["in_governed_census"]
            and not row["outer_test_selected"]
            and not row["embargo_selected"]
            and not row["protected_holdout_selected"]
            and not row["fresh_confirmation_selected"]
            and not row["sealed_or_burned_recorder_selected"]
        )
        rows.append(row)
    failures = [row["date"] for row in rows if not row["firewall_safe"]]
    proof = {
        "schema_version": "Protocol101WalkingSkeletonFirewallSafetyProofV1",
        "goal": GOAL,
        "created_at_utc": utc_now(),
        "authority": {
            "path": str(AUTHORITY_PATH),
            "sha256": AUTHORITY_SHA256,
            "d42_lines": "593-599",
        },
        "governed_sources": {
            "graph_v2": {
                "path": str(GRAPH_PATH),
                "sha256": GRAPH_SHA256,
            },
            "ft2_04_census": {
                "path": str(CENSUS_PATH),
                "sha256": CENSUS_SHA256,
                "derivation": str(census["derivation"]),
            },
            "sealed_day_assignment": {
                "path": str(SEALED_ASSIGNMENT_PATH),
                "sha256": SEALED_ASSIGNMENT_SHA256,
            },
        },
        "slice": {
            "session_count": len(SLICE_DATES),
            "sessions": list(SLICE_DATES),
            "already_owned_d55": list(OWNED_D55_DATES),
            "authorized_download": list(DOWNLOAD_DATES),
        },
        "confirmation_status": {
            "concrete_date_manifest_found": False,
            "disposition": (
                "No separate frozen confirmation-date manifest exists. "
                "Membership in the pinned governed census is dispositive under "
                "D42 because any confirmation dates are subtracted once frozen."
            ),
        },
        "sessions": rows,
        "summary": {
            "all_13_in_census": all(row["in_governed_census"] for row in rows),
            "outer_test_intersection_count": sum(
                bool(row["outer_test_selected"]) for row in rows
            ),
            "embargo_intersection_count": sum(
                bool(row["embargo_selected"]) for row in rows
            ),
            "protected_holdout_intersection_count": sum(
                bool(row["protected_holdout_selected"]) for row in rows
            ),
            "fresh_confirmation_intersection_count": sum(
                bool(row["fresh_confirmation_selected"]) for row in rows
            ),
            "sealed_or_burned_intersection_count": sum(
                bool(row["sealed_or_burned_recorder_selected"]) for row in rows
            ),
            "failure_dates": failures,
            "gate": "PASS" if not failures else "STOP_FIREWALL_FAILURE",
        },
        "side_effects": {
            "market_data_rows_read": False,
            "paid_download_performed": False,
            "broker_contacted": False,
            "model_training": False,
            "tensor_build": False,
            "stage_1_started": False,
        },
    }
    if failures:
        raise RuntimeError(f"firewall proof failed for: {failures}")
    return proof


def _session_symbols(path: Path, session: str) -> tuple[list[str], int]:
    frame = pd.read_parquet(
        path,
        columns=["raw_symbol", "root", "expiry", "settlement_style"],
    )
    valid = (
        frame["root"].astype(str).eq("SPXW")
        & frame["expiry"].astype(str).eq(session)
        & frame["settlement_style"].astype(str).eq("PM")
    )
    if not bool(valid.all()):
        raise RuntimeError(
            f"{path} has {int((~valid).sum())} rows outside exact SPXW PM 0DTE"
        )
    symbols = sorted(
        {
            value
            for value in frame["raw_symbol"].dropna().astype(str)
            if value and value != "nan"
        }
    )
    if not symbols or any(not symbol.startswith("SPXW") for symbol in symbols):
        raise RuntimeError(f"{session}: invalid or empty exact raw-symbol set")
    return symbols, int(len(frame))


def _owned_d55_manifest(d55_root: Path, session: str) -> dict[str, Any]:
    path = d55_root / session / "manifest.json"
    if not path.exists():
        raise RuntimeError(f"owned D55 manifest missing: {path}")
    manifest = json.loads(path.read_text())
    if manifest["request"]["dataset"] != DATASET:
        raise RuntimeError(f"{session}: owned D55 dataset mismatch")
    if manifest["request"]["schema"] != SCHEMA:
        raise RuntimeError(f"{session}: owned D55 schema mismatch")
    if manifest["request"]["stype_in"] != STYPE_IN:
        raise RuntimeError(f"{session}: owned D55 stype mismatch")
    for key in ("dbn", "parquet"):
        file_row = manifest["files"][key]
        artifact = Path(str(file_row["path"]))
        if not artifact.exists():
            raise RuntimeError(f"{session}: owned D55 artifact missing: {artifact}")
        if sha256_file(artifact) != file_row["sha256"]:
            raise RuntimeError(f"{session}: owned D55 {key} hash mismatch")
    return {
        "date": session,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "request": manifest["request"],
        "files": manifest["files"],
    }


def build_plan(
    normalized_dir: Path,
    d55_root: Path,
    proof_path: Path,
) -> dict[str, Any]:
    proof = json.loads(proof_path.read_text())
    if proof["summary"]["gate"] != "PASS":
        raise RuntimeError("firewall proof did not pass")
    if proof["slice"]["sessions"] != list(SLICE_DATES):
        raise RuntimeError("firewall proof slice drift")
    sessions: list[dict[str, Any]] = []
    all_symbols: list[str] = []
    for sequence, session in enumerate(DOWNLOAD_DATES, 1):
        path = normalized_dir / f"databento_spxw_0dte_{session}.parquet"
        if not path.exists():
            raise RuntimeError(f"normalized symbol source missing: {path}")
        symbols, row_count = _session_symbols(path, session)
        row = {
            "sequence": sequence,
            "date": session,
            "normalized_path": str(path),
            "normalized_sha256": sha256_file(path),
            "normalized_rows": row_count,
            "symbol_count": len(symbols),
            "symbols_sha256": canonical_sha256(symbols),
            "symbols": symbols,
        }
        sessions.append(row)
        all_symbols.extend(symbols)
    if len(all_symbols) != len(set(all_symbols)):
        raise RuntimeError("expiry-specific raw symbols overlap across sessions")
    owned = [_owned_d55_manifest(d55_root, date) for date in OWNED_D55_DATES]
    return {
        "schema_version": "Protocol101WalkingSkeletonClean1sAcquisitionPlanV1",
        "goal": GOAL,
        "created_at_utc": utc_now(),
        "authority": {
            "path": str(AUTHORITY_PATH),
            "sha256": AUTHORITY_SHA256,
        },
        "firewall_safety_proof": {
            "path": str(proof_path),
            "sha256": sha256_file(proof_path),
        },
        "approval_required": {
            "exact_approval_text": APPROVAL_TEXT,
            "owner_authorization_date": "2026-07-30",
            "scope": (
                "exactly 11 listed census sessions; OPRA.PILLAR cbbo-1s; "
                "exact owned-minute-parquet SPXW PM-settled 0DTE raw symbols; "
                "$15 hard cap"
            ),
        },
        "request": {
            "dataset": DATASET,
            "schema": SCHEMA,
            "stype_in": STYPE_IN,
            "hard_cap_usd": HARD_CAP_USD,
            "session_count": len(sessions),
            "sessions": list(DOWNLOAD_DATES),
            "symbol_session_count": len(all_symbols),
            "unique_raw_symbol_count": len(set(all_symbols)),
        },
        "already_owned_clean_sessions": owned,
        "sessions": sessions,
        "forbidden_scope": {
            "other_datasets": True,
            "other_schemas": True,
            "other_dates": True,
            "non_exact_symbols": True,
            "training": True,
            "tensor_build": True,
            "broker": True,
            "runtime_or_promotion_mutation": True,
            "stage_1": True,
        },
        "side_effects": {
            "paid_download_performed": False,
            "broker_contacted": False,
            "model_training": False,
            "tensor_build": False,
            "stage_1_started": False,
        },
    }


def _proof_file(out_dir: Path) -> Path:
    return out_dir / "firewall_safety_proof.json"


def _plan_file(out_dir: Path) -> Path:
    return out_dir / "acquisition_plan.json"


def _cost_file(out_dir: Path) -> Path:
    return out_dir / "cost_receipt.json"


def prepare(normalized_dir: Path, d55_root: Path, out_dir: Path) -> dict[str, Any]:
    proof = build_firewall_proof()
    write_json(_proof_file(out_dir), proof)
    plan = build_plan(normalized_dir, d55_root, _proof_file(out_dir))
    write_json(_plan_file(out_dir), plan)
    return {
        "firewall_proof": {
            "path": str(_proof_file(out_dir)),
            "sha256": sha256_file(_proof_file(out_dir)),
            "gate": proof["summary"]["gate"],
        },
        "acquisition_plan": {
            "path": str(_plan_file(out_dir)),
            "sha256": sha256_file(_plan_file(out_dir)),
            "download_sessions": len(plan["sessions"]),
            "symbol_session_count": plan["request"]["symbol_session_count"],
        },
    }


def _normalized_without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(value))
    out.pop("created_at_utc", None)
    return out


def load_and_verify_plan(
    normalized_dir: Path,
    d55_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    verify_pinned_authority()
    proof_path = _proof_file(out_dir)
    plan_path = _plan_file(out_dir)
    if not proof_path.exists() or not plan_path.exists():
        raise RuntimeError("run --mode prepare first")
    recorded_proof = json.loads(proof_path.read_text())
    rebuilt_proof = build_firewall_proof()
    if canonical_sha256(_normalized_without_timestamp(recorded_proof)) != (
        canonical_sha256(_normalized_without_timestamp(rebuilt_proof))
    ):
        raise RuntimeError("firewall proof no longer matches governed manifests")
    recorded = json.loads(plan_path.read_text())
    rebuilt = build_plan(normalized_dir, d55_root, proof_path)
    if canonical_sha256(_normalized_without_timestamp(recorded)) != (
        canonical_sha256(_normalized_without_timestamp(rebuilt))
    ):
        raise RuntimeError("acquisition plan no longer matches exact inputs")
    return recorded


def _load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


def _client(env_file: Path) -> Any:
    _load_env_file(env_file)
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError("databento is not installed in this runtime") from exc
    return db.Historical()


def _bounds(session: str) -> tuple[str, str]:
    start = pd.Timestamp(session, tz="UTC")
    end = start + pd.Timedelta(days=1)
    return start.isoformat(), end.isoformat()


def cost_gate(
    client: Any,
    normalized_dir: Path,
    d55_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    plan = load_and_verify_plan(normalized_dir, d55_root, out_dir)
    all_symbols = [
        symbol for session in plan["sessions"] for symbol in session["symbols"]
    ]
    full_start, _ = _bounds(DOWNLOAD_DATES[0])
    _, full_end = _bounds(DOWNLOAD_DATES[-1])
    try:
        one_shot_cost = float(
            client.metadata.get_cost(
                dataset=DATASET,
                schema=SCHEMA,
                symbols=all_symbols,
                stype_in=STYPE_IN,
                start=full_start,
                end=full_end,
            )
        )
    except Exception as exc:
        if "2,000 symbols" not in str(exc):
            raise
        one_shot = {
            "attempted_first": True,
            "outcome": "rejected_vendor_2000_symbol_limit",
            "requested_symbol_sessions": len(all_symbols),
            "error_type": type(exc).__name__,
            "adaptation": (
                "sum metadata.get_cost over all 11 exact session-date requests "
                "before any timeseries.get_range call"
            ),
        }
        one_shot_cost = None
    else:
        one_shot = {
            "attempted_first": True,
            "outcome": "accepted",
            "requested_symbol_sessions": len(all_symbols),
            "cost_estimate_usd": one_shot_cost,
        }
        if one_shot_cost > HARD_CAP_USD:
            receipt = {
                "schema_version": (
                    "Protocol101WalkingSkeletonClean1sCostReceiptV1"
                ),
                "goal": GOAL,
                "checked_at_utc": utc_now(),
                "authority_sha256": AUTHORITY_SHA256,
                "acquisition_plan": {
                    "path": str(_plan_file(out_dir)),
                    "sha256": sha256_file(_plan_file(out_dir)),
                },
                "one_shot_full_set_attempt": one_shot,
                "estimated_total_usd": one_shot_cost,
                "hard_cap_usd": HARD_CAP_USD,
                "gate": "STOP_OVER_CAP",
                "download_performed_before_gate": False,
            }
            write_json(_cost_file(out_dir), receipt)
            raise SystemExit(
                f"STOP: full-set estimate ${one_shot_cost:.6f} exceeds "
                f"${HARD_CAP_USD:.2f}"
            )

    estimates: list[dict[str, Any]] = []
    running = 0.0
    for row in plan["sessions"]:
        start, end = _bounds(str(row["date"]))
        value = float(
            client.metadata.get_cost(
                dataset=DATASET,
                schema=SCHEMA,
                symbols=list(row["symbols"]),
                stype_in=STYPE_IN,
                start=start,
                end=end,
            )
        )
        running += value
        estimate = {
            "sequence": int(row["sequence"]),
            "date": str(row["date"]),
            "symbol_count": int(row["symbol_count"]),
            "symbols_sha256": str(row["symbols_sha256"]),
            "start": start,
            "end": end,
            "cost_estimate_usd": value,
            "running_total_usd": running,
        }
        estimates.append(estimate)
        print(json.dumps(estimate, sort_keys=True), flush=True)
    if one_shot_cost is not None and not math.isclose(
        one_shot_cost,
        running,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "accepted one-shot full-set cost does not reconcile to exact "
            f"session-scoped costs: {one_shot_cost} != {running}"
        )
    gate = "PASS" if running <= HARD_CAP_USD else "STOP_OVER_CAP"
    receipt = {
        "schema_version": "Protocol101WalkingSkeletonClean1sCostReceiptV1",
        "goal": GOAL,
        "checked_at_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "acquisition_plan": {
            "path": str(_plan_file(out_dir)),
            "sha256": sha256_file(_plan_file(out_dir)),
        },
        "request": {
            "dataset": DATASET,
            "schema": SCHEMA,
            "stype_in": STYPE_IN,
            "sessions": list(DOWNLOAD_DATES),
            "selected_sessions": len(estimates),
            "symbol_session_count": sum(x["symbol_count"] for x in estimates),
        },
        "one_shot_full_set_attempt": one_shot,
        "full_set_estimation": (
            "metadata.get_cost attempted for the full union first; after the "
            "vendor's 2,000-symbol limit, the complete cost is the sum of all "
            "11 exact session-date requests, completed before any download"
        ),
        "session_estimates": estimates,
        "estimated_total_usd": running,
        "hard_cap_usd": HARD_CAP_USD,
        "headroom_usd": HARD_CAP_USD - running,
        "gate": gate,
        "download_performed_before_gate": False,
    }
    write_json(_cost_file(out_dir), receipt)
    if gate != "PASS":
        raise SystemExit(
            f"STOP: estimated cost ${running:.6f} exceeds ${HARD_CAP_USD:.2f}"
        )
    return receipt


def load_and_verify_cost(out_dir: Path) -> dict[str, Any]:
    path = _cost_file(out_dir)
    if not path.exists():
        raise RuntimeError("run --mode cost-gate first")
    receipt = json.loads(path.read_text())
    if receipt.get("gate") != "PASS":
        raise RuntimeError("cost gate did not pass")
    request = receipt.get("request", {})
    if (
        request.get("dataset") != DATASET
        or request.get("schema") != SCHEMA
        or request.get("stype_in") != STYPE_IN
        or request.get("sessions") != list(DOWNLOAD_DATES)
    ):
        raise RuntimeError("cost receipt request scope drift")
    if float(receipt.get("estimated_total_usd", math.inf)) > HARD_CAP_USD:
        raise RuntimeError("cost receipt exceeds hard cap")
    if receipt["acquisition_plan"]["sha256"] != sha256_file(
        _plan_file(out_dir)
    ):
        raise RuntimeError("cost receipt acquisition-plan hash drift")
    return receipt


def _raw_paths(raw_root: Path, session: str) -> tuple[Path, Path, Path]:
    directory = raw_root / session
    return (
        directory / f"{session}.{SCHEMA}.dbn.zst",
        directory / f"{session}.{SCHEMA}.parquet",
        directory / "manifest.json",
    )


def _partial_paths(raw_root: Path, session: str) -> tuple[Path, Path]:
    directory = raw_root / session
    return (
        directory / f"{session}.partial.{SCHEMA}.dbn.zst",
        directory / f"{session}.partial.{SCHEMA}.parquet",
    )


def _failure_file(out_dir: Path) -> Path:
    return out_dir / "failed_download_attempts.json"


def _load_failures(out_dir: Path) -> list[dict[str, Any]]:
    path = _failure_file(out_dir)
    if not path.exists():
        return []
    value = json.loads(path.read_text())
    rows = value.get("attempts", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid failed-attempt ledger: {path}")
    return rows


def _archive_partial_attempt(
    raw_root: Path,
    out_dir: Path,
    session: str,
    cost_row: dict[str, Any],
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    failures = _load_failures(out_dir)
    attempt_number = (
        sum(str(row.get("date")) == session for row in failures) + 1
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_dir = raw_root / session / "failed_attempts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for partial in _partial_paths(raw_root, session):
        if not partial.exists():
            continue
        target = archive_dir / (
            f"attempt{attempt_number:02d}_{stamp}_{partial.name}"
        )
        partial.replace(target)
        artifacts.append(
            {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    row = {
        "attempt_number": attempt_number,
        "date": session,
        "recorded_at_utc": utc_now(),
        "dataset": DATASET,
        "schema": SCHEMA,
        "symbol_count": int(cost_row["symbol_count"]),
        "symbols_sha256": str(cost_row["symbols_sha256"]),
        "preflight_session_cost_usd": float(cost_row["cost_estimate_usd"]),
        "conservative_billable_cost_upper_bound_usd": float(
            cost_row["cost_estimate_usd"]
        ),
        "cost_bound_method": "full_preflight_session_cost",
        "error_type": error_type,
        "error_message": re.sub(
            r"(?i)(api[_ -]?key|authorization)[^,;\\s]*",
            "[redacted-credential-field]",
            error_message,
        ),
        "archived_partial_files": artifacts,
    }
    failures.append(row)
    payload = {
        "schema_version": (
            "Protocol101WalkingSkeletonClean1sFailedDownloadAttemptsV1"
        ),
        "goal": GOAL,
        "updated_at_utc": utc_now(),
        "cost_bound_method": (
            "Every failed request is conservatively charged its full "
            "preflight session estimate, regardless of partial byte count."
        ),
        "attempts": failures,
        "conservative_billable_cost_upper_bound_usd": sum(
            float(item["conservative_billable_cost_upper_bound_usd"])
            for item in failures
        ),
    }
    write_json(_failure_file(out_dir), payload)
    return row


def _failure_cost_upper_bound(out_dir: Path) -> float:
    return sum(
        float(row["conservative_billable_cost_upper_bound_usd"])
        for row in _load_failures(out_dir)
    )


def _batch_jobs_file(out_dir: Path) -> Path:
    return out_dir / "batch_fallback_jobs.json"


def _load_batch_jobs(out_dir: Path) -> list[dict[str, Any]]:
    path = _batch_jobs_file(out_dir)
    if not path.exists():
        return []
    rows = json.loads(path.read_text()).get("jobs", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid batch-job ledger: {path}")
    return rows


def _write_batch_jobs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(
        _batch_jobs_file(out_dir),
        {
            "schema_version": (
                "Protocol101WalkingSkeletonClean1sBatchFallbackJobsV1"
            ),
            "goal": GOAL,
            "updated_at_utc": utc_now(),
            "jobs": rows,
        },
    )


def _job_id(job: dict[str, Any]) -> str:
    value = job.get("id", job.get("job_id"))
    if not value:
        raise RuntimeError(f"Databento batch job has no id: {sorted(job)}")
    return str(value)


def _job_state(job: dict[str, Any]) -> str:
    return str(job.get("state", job.get("status", "UNKNOWN"))).lower()


def _frame_with_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index()
    if "ts_recv" not in out.columns and len(out.columns):
        out = out.rename(columns={out.columns[0]: "ts_recv"})
    return out


def _valid_quote_mask(frame: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(frame.get("bid_px_00"), errors="coerce")
    ask = pd.to_numeric(frame.get("ask_px_00"), errors="coerce")
    return bid.gt(0) & ask.gt(0) & ask.ge(bid)


def session_manifest(
    plan_row: dict[str, Any],
    cost_row: dict[str, Any],
    dbn_path: Path,
    parquet_path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    inspect = _frame_with_timestamp(frame)
    required = {"ts_recv", "symbol", "bid_px_00", "ask_px_00"}
    missing_columns = required - set(inspect.columns)
    if missing_columns:
        raise RuntimeError(
            f"{plan_row['date']}: CBBO data lacks {sorted(missing_columns)}"
        )
    requested = set(map(str, plan_row["symbols"]))
    returned = set(inspect["symbol"].dropna().astype(str))
    unexpected = sorted(returned - requested)
    if unexpected:
        raise RuntimeError(
            f"{plan_row['date']}: vendor returned unexpected symbols"
        )
    ts = pd.to_datetime(inspect["ts_recv"], utc=True, errors="coerce")
    return {
        "schema_version": (
            "Protocol101WalkingSkeletonClean1sSessionManifestV1"
        ),
        "goal": GOAL,
        "created_at_utc": utc_now(),
        "date": str(plan_row["date"]),
        "authority_sha256": AUTHORITY_SHA256,
        "request": {
            "dataset": DATASET,
            "schema": SCHEMA,
            "stype_in": STYPE_IN,
            "start": str(cost_row["start"]),
            "end": str(cost_row["end"]),
            "symbol_count": int(plan_row["symbol_count"]),
            "symbols_sha256": str(plan_row["symbols_sha256"]),
            "symbols": list(plan_row["symbols"]),
            "estimated_cost_usd": float(cost_row["cost_estimate_usd"]),
        },
        "source_normalized": {
            "path": str(plan_row["normalized_path"]),
            "sha256": str(plan_row["normalized_sha256"]),
            "rows": int(plan_row["normalized_rows"]),
        },
        "download": {
            "rows": int(len(inspect)),
            "returned_symbol_count": len(returned),
            "missing_requested_symbol_count": len(requested - returned),
            "missing_requested_symbols": sorted(requested - returned),
            "valid_executable_bbo_rows": int(_valid_quote_mask(inspect).sum()),
            "first_ts_recv": (
                ts.min().isoformat() if bool(ts.notna().any()) else None
            ),
            "last_ts_recv": (
                ts.max().isoformat() if bool(ts.notna().any()) else None
            ),
        },
        "files": {
            "dbn": {
                "path": str(dbn_path),
                "bytes": dbn_path.stat().st_size,
                "sha256": sha256_file(dbn_path),
            },
            "parquet": {
                "path": str(parquet_path),
                "bytes": parquet_path.stat().st_size,
                "sha256": sha256_file(parquet_path),
            },
        },
        "cost_reconciliation": {
            "preflight_estimate_usd": float(cost_row["cost_estimate_usd"]),
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "reason": (
                "Databento Historical metadata exposes request cost estimates "
                "but no account invoice/usage endpoint. Exact executed request "
                "scope and downloaded artifact bytes/hashes are recorded."
            ),
        },
        "side_effects": {
            "broker_contacted": False,
            "model_training": False,
            "tensor_build": False,
            "stage_1_started": False,
        },
    }


def verify_existing_session(
    raw_root: Path,
    plan_row: dict[str, Any],
    cost_row: dict[str, Any],
) -> dict[str, Any] | None:
    session = str(plan_row["date"])
    dbn_path, parquet_path, manifest_path = _raw_paths(raw_root, session)
    existing = [path.exists() for path in (dbn_path, parquet_path, manifest_path)]
    if not any(existing):
        return None
    if not all(existing):
        raise RuntimeError(f"{session}: partial final files; refusing overwrite")
    manifest = json.loads(manifest_path.read_text())
    request = manifest["request"]
    if (
        request["dataset"] != DATASET
        or request["schema"] != SCHEMA
        or request["stype_in"] != STYPE_IN
        or request["symbols_sha256"] != plan_row["symbols_sha256"]
        or request["start"] != cost_row["start"]
        or request["end"] != cost_row["end"]
    ):
        raise RuntimeError(f"{session}: existing request scope mismatch")
    if manifest["files"]["dbn"]["sha256"] != sha256_file(dbn_path):
        raise RuntimeError(f"{session}: DBN hash mismatch")
    if manifest["files"]["parquet"]["sha256"] != sha256_file(parquet_path):
        raise RuntimeError(f"{session}: Parquet hash mismatch")
    if not math.isclose(
        float(request["estimated_cost_usd"]),
        float(cost_row["cost_estimate_usd"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{session}: estimated-cost allocation mismatch")
    return manifest


def _download_one_batch(
    client: Any,
    raw_root: Path,
    out_dir: Path,
    plan_row: dict[str, Any],
    cost_row: dict[str, Any],
) -> dict[str, Any]:
    """Submit or resume the exact request through the project batch fallback."""
    session = str(plan_row["date"])
    jobs = _load_batch_jobs(out_dir)
    matches = [row for row in jobs if str(row.get("date")) == session]
    if len(matches) > 1:
        raise RuntimeError(f"{session}: multiple batch fallback jobs exist")
    if matches:
        ledger_row = matches[0]
        expected = {
            "dataset": DATASET,
            "schema": SCHEMA,
            "stype_in": STYPE_IN,
            "start": str(cost_row["start"]),
            "end": str(cost_row["end"]),
            "symbols_sha256": str(plan_row["symbols_sha256"]),
        }
        for key, value in expected.items():
            if ledger_row.get(key) != value:
                raise RuntimeError(f"{session}: batch job scope drift at {key}")
        job_id = str(ledger_row["job_id"])
    else:
        submitted = client.batch.submit_job(
            dataset=DATASET,
            schema=SCHEMA,
            symbols=list(plan_row["symbols"]),
            stype_in=STYPE_IN,
            stype_out="instrument_id",
            start=str(cost_row["start"]),
            end=str(cost_row["end"]),
            encoding="dbn",
            compression="zstd",
            map_symbols=False,
            split_symbols=False,
            split_duration="day",
            delivery="download",
        )
        job_id = _job_id(submitted)
        ledger_row = {
            "date": session,
            "created_at_utc": utc_now(),
            "job_id": job_id,
            "dataset": DATASET,
            "schema": SCHEMA,
            "stype_in": STYPE_IN,
            "start": str(cost_row["start"]),
            "end": str(cost_row["end"]),
            "symbol_count": int(plan_row["symbol_count"]),
            "symbols_sha256": str(plan_row["symbols_sha256"]),
            "preflight_session_cost_usd": float(
                cost_row["cost_estimate_usd"]
            ),
            "delivery": "download",
            "submission": submitted,
            "status_history": [
                {
                    "checked_at_utc": utc_now(),
                    "state": _job_state(submitted),
                }
            ],
        }
        jobs.append(ledger_row)
        _write_batch_jobs(out_dir, jobs)
        print(
            json.dumps(
                {
                    "date": session,
                    "status": "batch_fallback_submitted",
                    "job_id": job_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    consecutive_poll_failures = 0
    while True:
        try:
            listed = client.batch.list_jobs(states="queued,processing,done")
        except Exception as exc:
            consecutive_poll_failures += 1
            ledger_row.setdefault("poll_errors", []).append(
                {
                    "checked_at_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "consecutive": consecutive_poll_failures,
                }
            )
            _write_batch_jobs(out_dir, jobs)
            if consecutive_poll_failures >= 10:
                raise RuntimeError(
                    f"{session}: ten consecutive batch polls failed"
                ) from exc
            time.sleep(15)
            continue
        consecutive_poll_failures = 0
        current = [job for job in listed if _job_id(job) == job_id]
        if len(current) != 1:
            raise RuntimeError(f"{session}: batch job is not uniquely visible")
        state = _job_state(current[0])
        ledger_row["latest_job"] = current[0]
        ledger_row.setdefault("status_history", []).append(
            {"checked_at_utc": utc_now(), "state": state}
        )
        _write_batch_jobs(out_dir, jobs)
        print(
            json.dumps(
                {
                    "date": session,
                    "status": "batch_fallback_poll",
                    "job_id": job_id,
                    "state": state,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if state == "done":
            break
        if state not in {"queued", "processing"}:
            raise RuntimeError(
                f"{session}: batch job entered terminal state {state}"
            )
        time.sleep(15)

    files = client.batch.list_files(job_id)
    dbn_files = [
        row
        for row in files
        if str(row.get("filename", row.get("name", ""))).endswith(".dbn.zst")
    ]
    if len(dbn_files) != 1:
        raise RuntimeError(
            f"{session}: expected one batch DBN file, found {len(dbn_files)}"
        )
    filename = str(dbn_files[0].get("filename", dbn_files[0].get("name")))
    staging = raw_root / session / "batch_fallback_download"
    paths = client.batch.download(
        job_id,
        output_dir=staging,
        filename_to_download=filename,
    )
    downloaded = [Path(path) for path in paths if Path(path).name == filename]
    if len(downloaded) != 1:
        raise RuntimeError(f"{session}: expected batch DBN was not downloaded")
    dbn_path, parquet_path, manifest_path = _raw_paths(raw_root, session)
    temp_dbn, temp_parquet = _partial_paths(raw_root, session)
    if any(
        path.exists()
        for path in (
            dbn_path,
            parquet_path,
            manifest_path,
            temp_dbn,
            temp_parquet,
        )
    ):
        raise RuntimeError(f"{session}: final or partial path appeared")
    downloaded[0].replace(temp_dbn)
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError("databento is required to read batch DBN") from exc
    frame = db.DBNStore.from_file(temp_dbn).to_df()
    inspect = _frame_with_timestamp(frame)
    returned = set(inspect["symbol"].dropna().astype(str))
    requested = set(map(str, plan_row["symbols"]))
    if returned - requested:
        raise RuntimeError(f"{session}: batch returned unexpected symbols")
    frame.to_parquet(temp_parquet, index=True)
    temp_dbn.replace(dbn_path)
    temp_parquet.replace(parquet_path)
    manifest = session_manifest(
        plan_row,
        cost_row,
        dbn_path,
        parquet_path,
        frame,
    )
    manifest["request"]["delivery"] = (
        "batch_fallback_after_repeated_stream_failures"
    )
    manifest["request"]["batch_job_id"] = job_id
    manifest["request"]["batch_file_manifest"] = dbn_files[0]
    write_json(manifest_path, manifest)
    ledger_row["completed_at_utc"] = utc_now()
    ledger_row["downloaded_file"] = manifest["files"]["dbn"]
    ledger_row["session_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    _write_batch_jobs(out_dir, jobs)
    return manifest


def download(
    client: Any,
    normalized_dir: Path,
    d55_root: Path,
    raw_root: Path,
    out_dir: Path,
    approval_manifest: Path,
    approval_text: str | None,
    approval_env_var: str,
) -> dict[str, Any]:
    plan = load_and_verify_plan(normalized_dir, d55_root, out_dir)
    cost = load_and_verify_cost(out_dir)
    cost_by_date = {
        str(row["date"]): row for row in cost["session_estimates"]
    }
    manifests: list[dict[str, Any]] = []
    for plan_row in plan["sessions"]:
        session = str(plan_row["date"])
        cost_row = cost_by_date[session]
        existing = verify_existing_session(raw_root, plan_row, cost_row)
        if existing is not None:
            print(
                json.dumps({"date": session, "status": "verified_existing"}),
                flush=True,
            )
            manifests.append(existing)
            continue
        dbn_path, parquet_path, manifest_path = _raw_paths(raw_root, session)
        temp_dbn, temp_parquet = _partial_paths(raw_root, session)
        dbn_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_dbn.exists() or temp_parquet.exists():
            _archive_partial_attempt(
                raw_root,
                out_dir,
                session,
                cost_row,
                "RecoveredUnledgeredPartial",
                (
                    "A previous process ended with a partial artifact before "
                    "the failure ledger was written."
                ),
            )
        failure_bound = _failure_cost_upper_bound(out_dir)
        worst_case_completed_total = (
            float(cost["estimated_total_usd"]) + failure_bound
        )
        if worst_case_completed_total > HARD_CAP_USD:
            raise RuntimeError(
                "STOP: conservative failed-attempt bound plus the complete "
                f"authorized request is ${worst_case_completed_total:.6f}, "
                f"above the ${HARD_CAP_USD:.2f} cap"
            )
        session_failures = [
            row
            for row in _load_failures(out_dir)
            if str(row.get("date")) == session
        ]
        batch_required_by_cap = (
            HARD_CAP_USD - worst_case_completed_total
            < float(cost_row["cost_estimate_usd"])
        )
        use_batch = len(session_failures) >= 2 or batch_required_by_cap
        require_paid_data_approval(
            manifest_path=approval_manifest,
            approval_text=approval_text,
            approval_env_var=approval_env_var,
            operation=(
                f"{DATASET} {SCHEMA} exact-symbol "
                f"{'batch fallback' if use_batch else 'download'} "
                f"for {session}"
            ),
        )
        if use_batch:
            manifest = _download_one_batch(
                client,
                raw_root,
                out_dir,
                plan_row,
                cost_row,
            )
            manifests.append(manifest)
            print(
                json.dumps(
                    {
                        "date": session,
                        "status": "batch_download_complete",
                        "batch_required_by_cap": batch_required_by_cap,
                        "rows": manifest["download"]["rows"],
                        "dbn_bytes": manifest["files"]["dbn"]["bytes"],
                        "parquet_bytes": manifest["files"]["parquet"]["bytes"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "date": session,
                    "status": "download_start",
                    "estimated_cost_usd": cost_row["cost_estimate_usd"],
                    "symbol_count": plan_row["symbol_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            store = client.timeseries.get_range(
                dataset=DATASET,
                schema=SCHEMA,
                symbols=list(plan_row["symbols"]),
                stype_in=STYPE_IN,
                start=str(cost_row["start"]),
                end=str(cost_row["end"]),
                path=temp_dbn,
            )
        except Exception as exc:
            failed = _archive_partial_attempt(
                raw_root,
                out_dir,
                session,
                cost_row,
                type(exc).__name__,
                str(exc),
            )
            projected = (
                float(cost["estimated_total_usd"])
                + _failure_cost_upper_bound(out_dir)
            )
            raise RuntimeError(
                f"{session}: vendor download failed and was archived as "
                f"attempt {failed['attempt_number']}; conservative completed-"
                f"scope cost would now be ${projected:.6f}"
            ) from exc
        frame = store.to_df()
        inspect = _frame_with_timestamp(frame)
        returned = set(inspect["symbol"].dropna().astype(str))
        requested = set(map(str, plan_row["symbols"]))
        if returned - requested:
            raise RuntimeError(f"{session}: unexpected returned symbols")
        frame.to_parquet(temp_parquet, index=True)
        temp_dbn.replace(dbn_path)
        temp_parquet.replace(parquet_path)
        manifest = session_manifest(
            plan_row,
            cost_row,
            dbn_path,
            parquet_path,
            frame,
        )
        write_json(manifest_path, manifest)
        manifests.append(manifest)
        print(
            json.dumps(
                {
                    "date": session,
                    "status": "download_complete",
                    "rows": manifest["download"]["rows"],
                    "dbn_bytes": manifest["files"]["dbn"]["bytes"],
                    "parquet_bytes": manifest["files"]["parquet"]["bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "downloaded_or_verified_sessions": len(manifests),
        "estimated_total_usd": sum(
            float(row["request"]["estimated_cost_usd"]) for row in manifests
        ),
    }


def verify(
    normalized_dir: Path,
    d55_root: Path,
    raw_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    plan = load_and_verify_plan(normalized_dir, d55_root, out_dir)
    cost = load_and_verify_cost(out_dir)
    cost_by_date = {
        str(row["date"]): row for row in cost["session_estimates"]
    }
    manifests: list[dict[str, Any]] = []
    for row in plan["sessions"]:
        manifest = verify_existing_session(
            raw_root,
            row,
            cost_by_date[str(row["date"])],
        )
        if manifest is None:
            raise RuntimeError(f"{row['date']}: download is missing")
        manifests.append(manifest)
    request_cost = sum(
        float(manifest["request"]["estimated_cost_usd"])
        for manifest in manifests
    )
    if not math.isclose(
        request_cost,
        float(cost["estimated_total_usd"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("manifest costs do not reconcile to cost receipt")
    owned = [_owned_d55_manifest(d55_root, date) for date in OWNED_D55_DATES]
    failure_cost = _failure_cost_upper_bound(out_dir)
    conservative_total = request_cost + failure_cost
    if conservative_total > HARD_CAP_USD:
        raise RuntimeError(
            "conservative completed-scope cost exceeds hard cap: "
            f"${conservative_total:.6f} > ${HARD_CAP_USD:.2f}"
        )
    failures = _load_failures(out_dir)
    batch_jobs = _load_batch_jobs(out_dir)
    summary_path = out_dir / "acquisition_summary.json"
    prior_verified_at = None
    if summary_path.exists():
        prior_verified_at = json.loads(summary_path.read_text()).get(
            "verified_at_utc"
        )
    summary = {
        "schema_version": (
            "Protocol101WalkingSkeletonClean1sAcquisitionSummaryV1"
        ),
        "goal": GOAL,
        "verified_at_utc": prior_verified_at or utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "firewall_safety_proof": {
            "path": str(_proof_file(out_dir)),
            "sha256": sha256_file(_proof_file(out_dir)),
            "gate": "PASS",
        },
        "acquisition_plan": {
            "path": str(_plan_file(out_dir)),
            "sha256": sha256_file(_plan_file(out_dir)),
        },
        "cost_receipt": {
            "path": str(_cost_file(out_dir)),
            "sha256": sha256_file(_cost_file(out_dir)),
        },
        "already_owned_clean_sessions": owned,
        "downloaded_sessions": [
            {
                "date": manifest["date"],
                "manifest_path": str(
                    raw_root / manifest["date"] / "manifest.json"
                ),
                "manifest_sha256": sha256_file(
                    raw_root / manifest["date"] / "manifest.json"
                ),
                "symbol_count": manifest["request"]["symbol_count"],
                "estimated_cost_usd": manifest["request"][
                    "estimated_cost_usd"
                ],
                "rows": manifest["download"]["rows"],
                "dbn_bytes": manifest["files"]["dbn"]["bytes"],
                "dbn_sha256": manifest["files"]["dbn"]["sha256"],
                "parquet_bytes": manifest["files"]["parquet"]["bytes"],
                "parquet_sha256": manifest["files"]["parquet"]["sha256"],
            }
            for manifest in manifests
        ],
        "reconciliation": {
            "preflight_estimated_total_usd": cost["estimated_total_usd"],
            "executed_request_manifest_estimated_total_usd": request_cost,
            "delta_usd": request_cost - float(cost["estimated_total_usd"]),
            "failed_stream_attempt_count": len(failures),
            "failed_stream_billable_cost_upper_bound_usd": failure_cost,
            "conservative_all_in_cost_upper_bound_usd": conservative_total,
            "hard_cap_usd": HARD_CAP_USD,
            "headroom_under_conservative_bound_usd": (
                HARD_CAP_USD - conservative_total
            ),
            "under_cap": conservative_total <= HARD_CAP_USD,
            "actual_vendor_invoice_cost_usd": "UNKNOWN",
            "actual_invoice_disposition": (
                "UNKNOWN per the project evidence standard: Databento "
                "Historical exposes no account invoice/usage endpoint. The "
                "preflight estimate reconciles exactly to all 11 executed "
                "request manifests, whose artifact bytes and hashes are sealed. "
                "Every interrupted stream is conservatively charged its full "
                "session estimate in the all-in upper bound."
            ),
        },
        "failed_download_attempts": {
            "path": (
                str(_failure_file(out_dir))
                if _failure_file(out_dir).exists()
                else None
            ),
            "sha256": (
                sha256_file(_failure_file(out_dir))
                if _failure_file(out_dir).exists()
                else None
            ),
            "count": len(failures),
        },
        "batch_fallback_jobs": {
            "path": (
                str(_batch_jobs_file(out_dir))
                if _batch_jobs_file(out_dir).exists()
                else None
            ),
            "sha256": (
                sha256_file(_batch_jobs_file(out_dir))
                if _batch_jobs_file(out_dir).exists()
                else None
            ),
            "count": len(batch_jobs),
            "completed_count": sum(
                bool(row.get("completed_at_utc")) for row in batch_jobs
            ),
        },
        "completeness": {
            "corrected_slice_sessions": list(SLICE_DATES),
            "corrected_slice_session_count": len(SLICE_DATES),
            "already_owned_count": len(owned),
            "downloaded_count": len(manifests),
            "all_13_firewall_safe_and_present": (
                len(owned) == len(OWNED_D55_DATES)
                and len(manifests) == len(DOWNLOAD_DATES)
            ),
            "gate": "PASS",
        },
        "side_effects": {
            "broker_contacted": False,
            "model_training": False,
            "tensor_build": False,
            "runtime_or_promotion_changed": False,
            "stage_1_started": False,
        },
    }
    write_json(summary_path, summary)
    return summary


def review(
    normalized_dir: Path,
    d55_root: Path,
    raw_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run the local-only, delta-scoped correction/acquisition review."""
    acquisition = verify(normalized_dir, d55_root, raw_root, out_dir)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "evidence": evidence,
            }
        )
        if not passed:
            raise RuntimeError(f"delta-scoped review failed: {name}: {evidence}")

    data_slice_path = out_dir / "data_slice.json"
    receipt_path = out_dir / "receipt.json"
    superseded_path = (
        out_dir
        / "superseded"
        / "data_slice_pre_firewall_correction_2026_07_30.json"
    )
    proof_path = _proof_file(out_dir)
    for path in (
        data_slice_path,
        receipt_path,
        superseded_path,
        proof_path,
        out_dir / "STAGE0_SPEC.md",
    ):
        check(f"required_file:{path.name}", path.exists(), str(path))

    data_slice = json.loads(data_slice_path.read_text())
    superseded = json.loads(superseded_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    proof = json.loads(proof_path.read_text())

    check(
        "authority_hash",
        sha256_file(AUTHORITY_PATH) == AUTHORITY_SHA256,
        AUTHORITY_SHA256,
    )
    check(
        "superseded_slice_preserved",
        sha256_file(superseded_path)
        == "4e3da0bc4d1dbcb904e9465bbec716e1bbe4e7e25612153634ffed0c46e06b2e",
        sha256_file(superseded_path),
    )
    check(
        "entry_slice_unchanged",
        data_slice["entry_minute_slice"]["sessions"]
        == superseded["entry_minute_slice"]["sessions"]
        and data_slice["entry_minute_slice"]["frozen_partition"]
        == superseded["entry_minute_slice"]["frozen_partition"],
        data_slice["entry_minute_slice"]["session_count"],
    )
    exit_rows = data_slice["exit_cbbo_1s_slice"]["sessions"]
    exit_dates = [str(row["date"]) for row in exit_rows]
    check("exact_13_exit_dates", exit_dates == list(SLICE_DATES), exit_dates)
    partition = data_slice["exit_cbbo_1s_slice"]["frozen_partition"]
    partition_dates = (
        list(partition["fit_first_8_sessions"])
        + list(partition["calibration_next_2_sessions"])
        + list(partition["plumbing_replay_last_3_sessions"])
    )
    check(
        "partition_exact_disjoint_cover",
        partition_dates == list(SLICE_DATES)
        and len(partition_dates) == len(set(partition_dates)),
        partition_dates,
    )
    check(
        "firewall_proof_pass",
        proof["summary"]["gate"] == "PASS"
        and not proof["summary"]["failure_dates"]
        and all(row["firewall_safe"] for row in proof["sessions"]),
        proof["summary"],
    )
    firewall = data_slice["exit_cbbo_1s_slice"]["firewall_assertions"]
    check(
        "zero_protected_role_intersections",
        firewall["all_sessions_in_governed_census"]
        and all(
            int(firewall[key]) == 0
            for key in (
                "outer_test_intersection_count",
                "protected_holdout_intersection_count",
                "embargo_intersection_count",
                "fresh_confirmation_intersection_count",
                "sealed_or_burned_recorder_intersection_count",
            )
        ),
        firewall,
    )
    for row in exit_rows:
        session = str(row["date"])
        manifest_path = Path(str(row["manifest_path"]))
        dbn_path = Path(str(row["dbn_path"]))
        parquet_path = Path(str(row["parquet_path"]))
        check(
            f"{session}:manifest_hash",
            manifest_path.exists()
            and sha256_file(manifest_path) == row["manifest_sha256"],
            str(manifest_path),
        )
        manifest = json.loads(manifest_path.read_text())
        request = manifest["request"]
        check(
            f"{session}:exact_request",
            request["dataset"] == DATASET
            and request["schema"] == SCHEMA
            and request["stype_in"] == STYPE_IN,
            {
                "dataset": request["dataset"],
                "schema": request["schema"],
                "stype_in": request["stype_in"],
            },
        )
        check(
            f"{session}:dbn_hash",
            dbn_path.exists() and sha256_file(dbn_path) == row["dbn_sha256"],
            str(dbn_path),
        )
        check(
            f"{session}:parquet_hash",
            parquet_path.exists()
            and sha256_file(parquet_path) == row["parquet_sha256"],
            str(parquet_path),
        )

    check(
        "raw_root_exact_date_directories",
        sorted(
            path.name for path in raw_root.iterdir() if path.is_dir()
        )
        == sorted(DOWNLOAD_DATES),
        sorted(path.name for path in raw_root.iterdir() if path.is_dir()),
    )
    final_partials = [
        str(path)
        for session in DOWNLOAD_DATES
        for path in _partial_paths(raw_root, session)
        if path.exists()
    ]
    check("no_unsealed_final_partials", not final_partials, final_partials)
    reconciliation = acquisition["reconciliation"]
    check(
        "cost_reconciles_under_cap",
        abs(float(reconciliation["delta_usd"])) <= 1e-12
        and reconciliation["under_cap"]
        and float(reconciliation["conservative_all_in_cost_upper_bound_usd"])
        <= HARD_CAP_USD,
        reconciliation,
    )
    check(
        "all_13_present",
        acquisition["completeness"]["all_13_firewall_safe_and_present"],
        acquisition["completeness"],
    )
    for relative, expected in receipt["deliverables"].items():
        path = out_dir / relative
        check(
            f"receipt_deliverable:{relative}",
            path.exists() and sha256_file(path) == expected,
            {"path": str(path), "expected_sha256": expected},
        )
    check(
        "receipt_claim_and_stop",
        receipt["outcome"] == "STOP_FOR_OWNER_AND_FABLE_REVIEW"
        and receipt["next"] == "STOP_FOR_OWNER_AND_FABLE_REVIEW"
        and receipt["highest_allowed_claim"]
        == (
            "the walking-skeleton exit slice is firewall-safe and complete; "
            "Stage 1 (entry model build) is unblocked."
        ),
        {
            "outcome": receipt["outcome"],
            "next": receipt["next"],
            "highest_allowed_claim": receipt["highest_allowed_claim"],
        },
    )
    side_effects = receipt["side_effects"]
    check(
        "forbidden_side_effects_absent",
        all(
            side_effects[key] is False
            for key in (
                "model_training_or_fitting",
                "tensor_build",
                "broker_contact",
                "protected_confirmation_or_sealed_evidence_access",
                "paper_order_submission",
                "runtime_or_promotion_change",
                "stage_1_started",
            )
        ),
        side_effects,
    )
    report = {
        "schema_version": (
            "Protocol101WalkingSkeletonStage0DeltaScopedReviewV1"
        ),
        "reviewed_at_utc": utc_now(),
        "scope": (
            "Stage-0 exit-slice firewall correction, exact 11-session "
            "OPRA.PILLAR cbbo-1s acquisition, cost reconciliation, and packet "
            "reissue only"
        ),
        "authority_sha256": AUTHORITY_SHA256,
        "receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        },
        "acquisition_summary": {
            "path": str(out_dir / "acquisition_summary.json"),
            "sha256": sha256_file(out_dir / "acquisition_summary.json"),
        },
        "runner": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "checks": checks,
        "check_count": len(checks),
        "failed_check_count": sum(not row["passed"] for row in checks),
        "outcome": "PASS",
        "next": "STOP_FOR_OWNER_AND_FABLE_REVIEW",
    }
    write_json(out_dir / "delta_scoped_review.json", report)
    return {
        "outcome": report["outcome"],
        "check_count": report["check_count"],
        "failed_check_count": report["failed_check_count"],
        "report": {
            "path": str(out_dir / "delta_scoped_review.json"),
            "sha256": sha256_file(out_dir / "delta_scoped_review.json"),
        },
        "next": report["next"],
    }


def main() -> int:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare(args.normalized_dir, args.d55_root, args.out_dir)
    elif args.mode == "cost-gate":
        result = cost_gate(
            _client(args.env_file),
            args.normalized_dir,
            args.d55_root,
            args.out_dir,
        )
    elif args.mode == "download":
        result = download(
            _client(args.env_file),
            args.normalized_dir,
            args.d55_root,
            args.raw_root,
            args.out_dir,
            args.approval_manifest,
            args.approval_text,
            args.approval_env_var,
        )
    elif args.mode == "verify":
        result = verify(
            args.normalized_dir,
            args.d55_root,
            args.raw_root,
            args.out_dir,
        )
    else:
        result = review(
            args.normalized_dir,
            args.d55_root,
            args.raw_root,
            args.out_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
