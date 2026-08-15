"""Acquire the owner-authorized FT2-D55 30-session OPRA CBBO-1s pilot.

This runner is deliberately narrow.  It can only:

1. deterministically select the 30 owner-authorized sessions,
2. estimate the complete session-scoped request before any paid download, and
3. download OPRA.PILLAR ``cbbo-1s`` for the exact selected 0DTE raw symbols.

It does not train, score a holdout, contact a broker, or mutate any signed
contract, graph, census, runtime, promotion, or paper-trading state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


GOAL = "FT2-D55-EXIT-REALISM-PILOT"
DATASET = "OPRA.PILLAR"
SCHEMA = "cbbo-1s"
STYPE_IN = "raw_symbol"
UNIVERSE_START = "2025-02-20"
UNIVERSE_END = "2025-12-31"
EXPECTED_UNIVERSE_SESSIONS = 218
SELECTED_SESSION_COUNT = 30
HARD_CAP_USD = 30.0
OWNER_REFERENCE_COST_USD = 24.12
DECIMAL_BYTES_PER_GB = 1_000_000_000
MAX_HEADER_FRAME_BYTES = 10_000_000
DATABENTO_STREAMING_BILLING_URL = (
    "https://databento.com/docs/faqs/usage-pricing-and-data-credits"
)
AUTHORITY_PATH = Path(
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
AUTHORITY_SHA256 = "d115b953d8959fe777923ca5c1e375246754a181847ae77b57d37d24f0a279ca"
DEFAULT_NORMALIZED_DIR = Path("v4/normalized")
DEFAULT_RAW_ROOT = Path("v4/raw/opra_1s_pilot")
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_ft2_d55_exit_realism_pilot"
)
DEFAULT_ENV_FILE = Path("v4/.env")
APPROVAL_TEXT = (
    "I authorize the FT2-D55 30-session Databento OPRA.PILLAR cbbo-1s "
    "exit-realism pilot with a hard cap of $30; no other paid pull is permitted."
)
NORMALIZED_RE = re.compile(
    r"^databento_spxw_0dte_(?P<date>\d{4}-\d{2}-\d{2})\.parquet$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prepare", "cost-gate", "download", "verify"),
        required=True,
    )
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    add_paid_data_approval_args(
        parser,
        default_manifest=DEFAULT_OUT_DIR / "acquisition_plan.json",
    )
    return parser.parse_args()


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise SystemExit(
            "databento is not installed in the selected runtime environment"
        ) from exc
    return db.Historical()


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    """Return rounded linspace indices with deterministic nearest-unused repair."""
    if n < k or k <= 0:
        raise ValueError(f"cannot select {k} unique indices from {n}")
    targets = np.linspace(0, n - 1, k)
    chosen: list[int] = []
    used: set[int] = set()
    for target in targets:
        rounded = int(np.rint(target))
        if rounded not in used:
            pick = rounded
        else:
            pick = min(
                (index for index in range(n) if index not in used),
                key=lambda index: (abs(float(index) - float(target)), index),
            )
        chosen.append(pick)
        used.add(pick)
    if len(chosen) != k or len(set(chosen)) != k:
        raise AssertionError("deterministic session selection did not remain unique")
    return chosen


def discover_universe(normalized_dir: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for path in sorted(normalized_dir.glob("databento_spxw_0dte_*.parquet")):
        match = NORMALIZED_RE.match(path.name)
        if match is None:
            continue
        session = match.group("date")
        if UNIVERSE_START <= session <= UNIVERSE_END:
            rows.append((session, path))
    if len(rows) != EXPECTED_UNIVERSE_SESSIONS:
        raise RuntimeError(
            "D55 selection universe drift: "
            f"expected {EXPECTED_UNIVERSE_SESSIONS}, found {len(rows)}"
        )
    if rows[0][0] != UNIVERSE_START or rows[-1][0] != UNIVERSE_END:
        raise RuntimeError(
            f"D55 universe endpoints drifted: {rows[0][0]} -> {rows[-1][0]}"
        )
    return rows


def session_symbols(path: Path, session: str) -> tuple[list[str], int]:
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
            f"{path} contains {int((~valid).sum())} rows outside the exact "
            "SPXW PM-settled 0DTE universe"
        )
    symbols = sorted(
        {
            value
            for value in frame["raw_symbol"].dropna().astype(str)
            if value and value != "nan"
        }
    )
    if not symbols:
        raise RuntimeError(f"{session}: no raw symbols in {path}")
    if any(not symbol.startswith("SPXW") for symbol in symbols):
        raise RuntimeError(f"{session}: a selected raw symbol is not SPXW")
    return symbols, int(len(frame))


def build_plan(normalized_dir: Path) -> dict[str, Any]:
    authority_actual = sha256_file(AUTHORITY_PATH)
    if authority_actual != AUTHORITY_SHA256:
        raise RuntimeError(
            f"authority mismatch: {authority_actual} != {AUTHORITY_SHA256}"
        )
    universe = discover_universe(normalized_dir)
    indices = evenly_spaced_indices(len(universe), SELECTED_SESSION_COUNT)
    sessions: list[dict[str, Any]] = []
    all_symbols: list[str] = []
    for sequence, index in enumerate(indices, 1):
        session, path = universe[index]
        symbols, row_count = session_symbols(path, session)
        symbol_hash = canonical_sha256(symbols)
        sessions.append(
            {
                "sequence": sequence,
                "selection_index": index,
                "date": session,
                "normalized_path": str(path),
                "normalized_sha256": sha256_file(path),
                "normalized_rows": row_count,
                "symbol_count": len(symbols),
                "symbols_sha256": symbol_hash,
                "symbols": symbols,
            }
        )
        all_symbols.extend(symbols)
    if len(all_symbols) != len(set(all_symbols)):
        raise RuntimeError(
            "D55 expiry-specific raw symbols unexpectedly overlap across sessions"
        )
    return {
        "schema_version": "Protocol101FT2D55AcquisitionPlanV1",
        "goal": GOAL,
        "created_at_utc": utc_now(),
        "authority": {
            "path": str(AUTHORITY_PATH),
            "sha256": AUTHORITY_SHA256,
        },
        "approval_required": {
            "exact_approval_text": APPROVAL_TEXT,
            "owner_authorization_date": "2026-07-30",
            "scope": (
                "30 selected owned SPXW 0DTE sessions; Databento OPRA.PILLAR "
                "cbbo-1s; exact normalized raw symbols; hard cap $30"
            ),
        },
        "selection": {
            "source": str(normalized_dir),
            "universe_start": UNIVERSE_START,
            "universe_end": UNIVERSE_END,
            "universe_session_count": len(universe),
            "rule": (
                "np.linspace(0,N-1,30), np.rint (half-to-even), deterministic "
                "nearest-unused repair if rounding duplicates"
            ),
            "indices": indices,
            "selected_session_count": len(sessions),
        },
        "request": {
            "dataset": DATASET,
            "schema": SCHEMA,
            "stype_in": STYPE_IN,
            "hard_cap_usd": HARD_CAP_USD,
            "symbol_session_count": len(all_symbols),
            "unique_raw_symbol_count": len(set(all_symbols)),
            "forbidden_schemas": [
                "definition",
                "ohlcv",
                "trades",
                "mbp",
                "cmbp",
                "tcbbo",
                "statistics",
            ],
        },
        "sessions": sessions,
        "side_effects": {
            "paid_download_performed": False,
            "broker_contacted": False,
            "model_training": False,
            "protected_data_accessed": False,
            "signed_contract_modified": False,
            "graph_modified": False,
            "census_modified": False,
        },
    }


def _bounds(session: str) -> tuple[str, str]:
    start = pd.Timestamp(session, tz="UTC")
    end = start + pd.Timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _plan_file(out_dir: Path) -> Path:
    return out_dir / "acquisition_plan.json"


def _cost_file(out_dir: Path) -> Path:
    return out_dir / "cost_receipt.json"


def prepare(normalized_dir: Path, out_dir: Path) -> dict[str, Any]:
    plan = build_plan(normalized_dir)
    write_json(_plan_file(out_dir), plan)
    return plan


def load_and_verify_plan(normalized_dir: Path, out_dir: Path) -> dict[str, Any]:
    path = _plan_file(out_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    recorded = json.loads(path.read_text())
    rebuilt = build_plan(normalized_dir)
    for value in (recorded, rebuilt):
        value.pop("created_at_utc", None)
    if canonical_sha256(recorded) != canonical_sha256(rebuilt):
        raise RuntimeError("acquisition plan no longer matches normalized inputs")
    return json.loads(path.read_text())


def cost_gate(
    *,
    client: Any,
    normalized_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    plan = load_and_verify_plan(normalized_dir, out_dir)
    all_symbols = [
        symbol for session in plan["sessions"] for symbol in session["symbols"]
    ]
    one_shot: dict[str, Any]
    try:
        client.metadata.get_cost(
            dataset=DATASET,
            schema=SCHEMA,
            symbols=all_symbols,
            stype_in=STYPE_IN,
            start=f"{UNIVERSE_START}T00:00:00Z",
            end=(
                pd.Timestamp(UNIVERSE_END, tz="UTC") + pd.Timedelta(days=1)
            ).isoformat(),
        )
    except Exception as exc:
        text = str(exc)
        if "maximum limit of 2,000 symbols" not in text:
            raise
        one_shot = {
            "attempted_first": True,
            "outcome": "rejected_vendor_2000_symbol_limit",
            "requested_symbols": len(all_symbols),
            "error_type": type(exc).__name__,
            "adaptation": (
                "sum metadata.get_cost over the exact 30 session-date requests; "
                "this is the request topology used by the downloader"
            ),
        }
    else:
        raise RuntimeError(
            "unexpectedly accepted cross-session one-shot estimate; D55 requires "
            "session-scoped reconciliation before download"
        )

    estimates: list[dict[str, Any]] = []
    running = 0.0
    for row in plan["sessions"]:
        start, end = _bounds(str(row["date"]))
        cost = float(
            client.metadata.get_cost(
                dataset=DATASET,
                schema=SCHEMA,
                symbols=list(row["symbols"]),
                stype_in=STYPE_IN,
                start=start,
                end=end,
            )
        )
        running += cost
        estimate = {
            "sequence": int(row["sequence"]),
            "selection_index": int(row["selection_index"]),
            "date": str(row["date"]),
            "symbol_count": int(row["symbol_count"]),
            "symbols_sha256": str(row["symbols_sha256"]),
            "start": start,
            "end": end,
            "cost_estimate_usd": cost,
            "running_total_usd": running,
        }
        estimates.append(estimate)
        print(json.dumps(estimate, sort_keys=True), flush=True)
    gate = "PASS" if running <= HARD_CAP_USD else "STOP_OVER_CAP"
    receipt = {
        "schema_version": "Protocol101FT2D55CostReceiptV1",
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
            "selected_sessions": len(estimates),
            "symbol_session_count": sum(x["symbol_count"] for x in estimates),
        },
        "one_shot_full_set_attempt": one_shot,
        "full_set_estimation": (
            "sum of metadata.get_cost for all 30 exact session-date requests "
            "before any timeseries.get_range call"
        ),
        "session_estimates": estimates,
        "estimated_total_usd": running,
        "owner_reference_cost_usd": OWNER_REFERENCE_COST_USD,
        "reference_delta_usd": running - OWNER_REFERENCE_COST_USD,
        "hard_cap_usd": HARD_CAP_USD,
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
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text())
    if receipt.get("gate") != "PASS":
        raise RuntimeError("cost receipt did not pass")
    if receipt.get("request", {}).get("dataset") != DATASET:
        raise RuntimeError("cost receipt dataset drift")
    if receipt.get("request", {}).get("schema") != SCHEMA:
        raise RuntimeError("cost receipt schema drift")
    if float(receipt.get("estimated_total_usd", math.inf)) > HARD_CAP_USD:
        raise RuntimeError("cost receipt exceeds D55 hard cap")
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


def _batch_jobs_file(out_dir: Path) -> Path:
    return out_dir / "batch_fallback_jobs.json"


def load_failures(out_dir: Path) -> list[dict[str, Any]]:
    path = _failure_file(out_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows = payload.get("attempts", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid failed-attempt ledger: {path}")
    return rows


def historical_streaming_unit_price(client: Any) -> float:
    """Read the current vendor-published OPRA CBBO-1s streaming $/GB."""
    rows = client.metadata.list_unit_prices(dataset=DATASET)
    matches = [
        row
        for row in rows
        if str(row.get("mode")) == "historical-streaming"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one OPRA.PILLAR historical-streaming unit-price row"
        )
    value = float(matches[0].get("unit_prices", {}).get(SCHEMA, math.nan))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid {DATASET} {SCHEMA} unit price: {value}")
    return value


def pinned_or_current_streaming_unit_price(
    *,
    client: Any,
    out_dir: Path,
) -> float:
    """Reuse a validated ledger price so network resumes are deterministic."""
    path = _failure_file(out_dir)
    if path.exists():
        payload = json.loads(path.read_text())
        evidence = payload.get("vendor_billing_evidence", {})
        if evidence:
            expected = {
                "dataset": DATASET,
                "schema": SCHEMA,
                "mode": "historical-streaming",
                "decimal_bytes_per_gb": DECIMAL_BYTES_PER_GB,
                "source": "Databento metadata.list_unit_prices",
            }
            for key, value in expected.items():
                if evidence.get(key) != value:
                    raise RuntimeError(
                        f"failed-attempt unit-price evidence drift: {key}"
                    )
            unit_price = float(evidence.get("unit_price_usd_per_gb", math.nan))
            if not math.isfinite(unit_price) or unit_price <= 0:
                raise RuntimeError("invalid pinned streaming unit price")
            return unit_price
    return historical_streaming_unit_price(client)


def partial_stream_billable_bound(path: Path) -> dict[str, Any]:
    """Bound a header-only interrupted Zstandard stream in uncompressed bytes.

    Databento streams DBN as concatenated Zstandard frames.  A transport that
    stops inside the first frame cannot have sent more uncompressed DBN bytes
    than that frame's declared content size.  Anything other than that narrow,
    inspectable shape falls back to the full preflight session estimate.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    compressed_bytes = path.stat().st_size
    if compressed_bytes == 0:
        return {
            "method": "empty_file",
            "eligible_for_byte_bound": True,
            "compressed_bytes": 0,
            "uncompressed_billable_bytes_upper_bound": 0,
        }
    if not path.name.endswith(".dbn.zst"):
        return {
            "method": "not_vendor_stream",
            "eligible_for_byte_bound": False,
            "compressed_bytes": compressed_bytes,
            "uncompressed_billable_bytes_upper_bound": None,
        }
    try:
        import zstandard as zstd

        data = path.read_bytes()
        magic = b"\x28\xb5\x2f\xfd"
        params = zstd.get_frame_parameters(data)
        content_size = int(params.content_size)
        decompressor = zstd.ZstdDecompressor().decompressobj()
        decompressor.decompress(data)
        first_frame_complete = bool(decompressor.eof)
    except Exception as exc:
        return {
            "method": "zstd_inspection_failed",
            "eligible_for_byte_bound": False,
            "compressed_bytes": compressed_bytes,
            "uncompressed_billable_bytes_upper_bound": None,
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }
    header_only = (
        data.startswith(magic)
        and data.find(magic, len(magic)) == -1
        and not first_frame_complete
        and 0 < content_size <= MAX_HEADER_FRAME_BYTES
    )
    return {
        "method": (
            "interrupted_first_zstd_frame_declared_content_size"
            if header_only
            else "not_proven_header_only"
        ),
        "eligible_for_byte_bound": header_only,
        "compressed_bytes": compressed_bytes,
        "uncompressed_billable_bytes_upper_bound": (
            content_size if header_only else None
        ),
        "declared_first_frame_content_size": content_size,
        "first_frame_complete": first_frame_complete,
        "zstd_magic_count": data.count(magic),
    }


def reconcile_failures(
    *,
    raw_root: Path,
    out_dir: Path,
    unit_price_usd_per_gb: float,
) -> list[dict[str, Any]]:
    """Refresh post-close file evidence and conservative billable-cost bounds."""
    failures = load_failures(out_dir)
    if not failures:
        return []
    reconciled: list[dict[str, Any]] = []
    for original in failures:
        row = dict(original)
        archived: list[dict[str, Any]] = []
        dbn_bounds: list[dict[str, Any]] = []
        for original_file in row.get("archived_partial_files", []):
            path = Path(str(original_file["path"]))
            if not path.exists():
                raise RuntimeError(f"failed-attempt evidence disappeared: {path}")
            refreshed = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.name.endswith(".dbn.zst"):
                refreshed["stream_billable_bound"] = partial_stream_billable_bound(
                    path
                )
                dbn_bounds.append(refreshed["stream_billable_bound"])
            archived.append(refreshed)
        eligible = bool(dbn_bounds) and all(
            bound["eligible_for_byte_bound"] for bound in dbn_bounds
        )
        if eligible:
            billable_bytes = int(
                sum(
                    int(bound["uncompressed_billable_bytes_upper_bound"])
                    for bound in dbn_bounds
                )
            )
            billable_cost = (
                billable_bytes
                * float(unit_price_usd_per_gb)
                / DECIMAL_BYTES_PER_GB
            )
            cost_method = "vendor_streamed_bytes_upper_bound"
        else:
            billable_bytes = None
            legacy_cost = row.get("preflight_session_cost_usd")
            if legacy_cost is None:
                legacy_cost = row["reserved_cost_usd"]
            billable_cost = float(legacy_cost)
            cost_method = "full_preflight_session_cost_fallback"
        row["archived_partial_files"] = archived
        preflight_cost = row.get("preflight_session_cost_usd")
        if preflight_cost is None:
            preflight_cost = row["reserved_cost_usd"]
        row["preflight_session_cost_usd"] = float(preflight_cost)
        row.pop("reserved_cost_usd", None)
        row["billable_uncompressed_bytes_upper_bound"] = billable_bytes
        row["billable_cost_upper_bound_usd"] = billable_cost
        row["billable_cost_method"] = cost_method
        reconciled.append(row)
    payload = {
        "schema_version": "Protocol101FT2D55FailedDownloadAttemptsV2",
        "goal": GOAL,
        "reconciled_at_utc": utc_now(),
        "vendor_billing_evidence": {
            "dataset": DATASET,
            "schema": SCHEMA,
            "mode": "historical-streaming",
            "unit_price_usd_per_gb": unit_price_usd_per_gb,
            "decimal_bytes_per_gb": DECIMAL_BYTES_PER_GB,
            "source": "Databento metadata.list_unit_prices",
            "billing_rule": (
                "Streaming is billed for outbound data bytes actually sent; "
                "an interrupted request is not billed for unsent remainder."
            ),
            "billing_rule_url": DATABENTO_STREAMING_BILLING_URL,
        },
        "attempts": reconciled,
        "billable_uncompressed_bytes_upper_bound": sum(
            int(row["billable_uncompressed_bytes_upper_bound"] or 0)
            for row in reconciled
            if row["billable_cost_method"]
            == "vendor_streamed_bytes_upper_bound"
        ),
        "billable_cost_upper_bound_usd": float(
            sum(row["billable_cost_upper_bound_usd"] for row in reconciled)
        ),
    }
    write_json(_failure_file(out_dir), payload)
    return reconciled


def _archive_failed_attempt(
    *,
    raw_root: Path,
    out_dir: Path,
    session: str,
    cost_row: dict[str, Any],
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    failures = load_failures(out_dir)
    attempt_number = (
        sum(str(row.get("date")) == session for row in failures) + 1
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_dir = raw_root / session / "failed_attempts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_files: list[dict[str, Any]] = []
    for partial in _partial_paths(raw_root, session):
        if not partial.exists():
            continue
        target = archive_dir / f"attempt{attempt_number:02d}_{stamp}_{partial.name}"
        partial.replace(target)
        archived_files.append(
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
        "billable_cost_upper_bound_usd": float(cost_row["cost_estimate_usd"]),
        "billable_cost_method": "pending_post_close_reconciliation",
        "error_type": error_type,
        "error_message": error_message,
        "archived_partial_files": archived_files,
    }
    failures.append(row)
    write_json(
        _failure_file(out_dir),
        {
            "schema_version": "Protocol101FT2D55FailedDownloadAttemptsV2",
            "goal": GOAL,
            "attempts": failures,
            "billable_cost_upper_bound_usd": float(
                sum(
                    item["billable_cost_upper_bound_usd"]
                    for item in failures
                )
            ),
        },
    )
    return row


def _worst_case_spend(cost: dict[str, Any], failures: Sequence[dict[str, Any]]) -> float:
    return float(cost["estimated_total_usd"]) + float(
        sum(row["billable_cost_upper_bound_usd"] for row in failures)
    )


def _valid_quote_mask(frame: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(frame.get("bid_px_00"), errors="coerce")
    ask = pd.to_numeric(frame.get("ask_px_00"), errors="coerce")
    return bid.gt(0) & ask.gt(0) & ask.ge(bid)


def _frame_with_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index()
    if "ts_recv" not in out.columns and len(out.columns):
        out = out.rename(columns={out.columns[0]: "ts_recv"})
    return out


def session_manifest(
    *,
    plan_row: dict[str, Any],
    cost_row: dict[str, Any],
    dbn_path: Path,
    parquet_path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    inspect = _frame_with_timestamp(frame)
    required = {"ts_recv", "symbol", "bid_px_00", "ask_px_00"}
    missing = required - set(inspect.columns)
    if missing:
        raise RuntimeError(f"downloaded CBBO-1s missing columns: {sorted(missing)}")
    requested = set(map(str, plan_row["symbols"]))
    returned = set(inspect["symbol"].dropna().astype(str))
    unexpected = sorted(returned - requested)
    if unexpected:
        raise RuntimeError(
            f"{plan_row['date']}: vendor returned {len(unexpected)} unexpected symbols"
        )
    ts = pd.to_datetime(inspect["ts_recv"], utc=True, errors="coerce")
    event_fraction: float | None = None
    if "ts_event" in inspect.columns and len(inspect):
        event_fraction = float(
            pd.to_datetime(
                inspect["ts_event"], utc=True, errors="coerce"
            ).notna().mean()
        )
    return {
        "schema_version": "Protocol101FT2D55SessionManifestV1",
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
            "ts_event_nonnull_fraction": event_fraction,
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
        "side_effects": {
            "broker_contacted": False,
            "model_training": False,
            "protected_data_accessed": False,
            "signed_contract_modified": False,
            "graph_modified": False,
            "census_modified": False,
        },
    }


def verify_existing_session(
    *,
    raw_root: Path,
    plan_row: dict[str, Any],
    cost_row: dict[str, Any],
) -> dict[str, Any] | None:
    dbn_path, parquet_path, manifest_path = _raw_paths(
        raw_root, str(plan_row["date"])
    )
    existing = [path.exists() for path in (dbn_path, parquet_path, manifest_path)]
    if not any(existing):
        return None
    if not all(existing):
        raise RuntimeError(
            f"{plan_row['date']}: partial final D55 files exist; refusing overwrite"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest["request"]["dataset"] != DATASET:
        raise RuntimeError(f"{plan_row['date']}: existing dataset mismatch")
    if manifest["request"]["schema"] != SCHEMA:
        raise RuntimeError(f"{plan_row['date']}: existing schema mismatch")
    if manifest["request"]["symbols_sha256"] != plan_row["symbols_sha256"]:
        raise RuntimeError(f"{plan_row['date']}: existing symbol-set mismatch")
    if manifest["files"]["dbn"]["sha256"] != sha256_file(dbn_path):
        raise RuntimeError(f"{plan_row['date']}: existing DBN hash mismatch")
    if manifest["files"]["parquet"]["sha256"] != sha256_file(parquet_path):
        raise RuntimeError(f"{plan_row['date']}: existing Parquet hash mismatch")
    if not math.isclose(
        float(manifest["request"]["estimated_cost_usd"]),
        float(cost_row["cost_estimate_usd"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{plan_row['date']}: existing cost allocation mismatch")
    return manifest


def _download_one(
    *,
    client: Any,
    raw_root: Path,
    plan_row: dict[str, Any],
    cost_row: dict[str, Any],
) -> dict[str, Any]:
    session = str(plan_row["date"])
    dbn_path, parquet_path, manifest_path = _raw_paths(raw_root, session)
    dbn_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dbn, temp_parquet = _partial_paths(raw_root, session)
    if dbn_path.exists() or parquet_path.exists() or manifest_path.exists():
        raise RuntimeError(f"{session}: refusing to overwrite final D55 raw files")
    if temp_dbn.exists() or temp_parquet.exists():
        raise RuntimeError(
            f"{session}: partial file exists from an earlier failed attempt; "
            "inspect it before retrying"
        )
    store = client.timeseries.get_range(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=list(plan_row["symbols"]),
        stype_in=STYPE_IN,
        start=str(cost_row["start"]),
        end=str(cost_row["end"]),
        path=temp_dbn,
    )
    frame = store.to_df()
    inspect = _frame_with_timestamp(frame)
    returned = set(inspect["symbol"].dropna().astype(str))
    requested = set(map(str, plan_row["symbols"]))
    unexpected = returned - requested
    if unexpected:
        raise RuntimeError(
            f"{session}: downloaded data contains unexpected symbols"
        )
    frame.to_parquet(temp_parquet, index=True)
    temp_dbn.replace(dbn_path)
    temp_parquet.replace(parquet_path)
    manifest = session_manifest(
        plan_row=plan_row,
        cost_row=cost_row,
        dbn_path=dbn_path,
        parquet_path=parquet_path,
        frame=frame,
    )
    write_json(manifest_path, manifest)
    return manifest


def _load_batch_jobs(out_dir: Path) -> list[dict[str, Any]]:
    path = _batch_jobs_file(out_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows = payload.get("jobs", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid batch fallback ledger: {path}")
    return rows


def _write_batch_jobs(out_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    write_json(
        _batch_jobs_file(out_dir),
        {
            "schema_version": "Protocol101FT2D55BatchFallbackJobsV1",
            "goal": GOAL,
            "updated_at_utc": utc_now(),
            "jobs": list(rows),
        },
    )


def _job_id(job: dict[str, Any]) -> str:
    value = job.get("id", job.get("job_id"))
    if not value:
        raise RuntimeError(f"Databento batch job has no id: {sorted(job)}")
    return str(value)


def _job_state(job: dict[str, Any]) -> str:
    return str(job.get("state", job.get("status", "UNKNOWN"))).lower()


def _download_one_batch(
    *,
    client: Any,
    raw_root: Path,
    out_dir: Path,
    plan_row: dict[str, Any],
    cost_row: dict[str, Any],
) -> dict[str, Any]:
    """Submit or resume one exact-symbol batch fallback after stream exhaustion."""
    session = str(plan_row["date"])
    jobs = _load_batch_jobs(out_dir)
    matches = [row for row in jobs if str(row.get("date")) == session]
    if len(matches) > 1:
        raise RuntimeError(f"{session}: multiple D55 batch fallback jobs exist")
    if matches:
        ledger_row = matches[0]
        if ledger_row["symbols_sha256"] != plan_row["symbols_sha256"]:
            raise RuntimeError(f"{session}: batch fallback symbol hash drift")
        if ledger_row["dataset"] != DATASET or ledger_row["schema"] != SCHEMA:
            raise RuntimeError(f"{session}: batch fallback request drift")
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
            print(
                json.dumps(
                    {
                        "date": session,
                        "status": "batch_fallback_poll_retry",
                        "job_id": job_id,
                        "error_type": type(exc).__name__,
                        "consecutive": consecutive_poll_failures,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if consecutive_poll_failures >= 10:
                raise RuntimeError(
                    f"{session}: ten consecutive batch status polls failed"
                ) from exc
            time.sleep(15)
            continue
        consecutive_poll_failures = 0
        current = [job for job in listed if _job_id(job) == job_id]
        if len(current) != 1:
            raise RuntimeError(
                f"{session}: batch job {job_id} not uniquely visible"
            )
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
                f"{session}: batch fallback entered terminal state {state}"
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
    filename = str(
        dbn_files[0].get("filename", dbn_files[0].get("name"))
    )
    staging = raw_root / session / "batch_fallback_download"
    paths = client.batch.download(
        job_id,
        output_dir=staging,
        filename_to_download=filename,
    )
    matching_paths = [Path(path) for path in paths if Path(path).name == filename]
    if len(matching_paths) != 1:
        raise RuntimeError(
            f"{session}: batch download did not return the expected DBN file"
        )
    downloaded = matching_paths[0]
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
        raise RuntimeError(
            f"{session}: final or partial path appeared during batch fallback"
        )
    downloaded.replace(temp_dbn)
    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError("databento is required to read batch DBN") from exc
    store = db.DBNStore.from_file(temp_dbn)
    frame = store.to_df()
    inspect = _frame_with_timestamp(frame)
    returned = set(inspect["symbol"].dropna().astype(str))
    requested = set(map(str, plan_row["symbols"]))
    if returned - requested:
        raise RuntimeError(
            f"{session}: batch data contains unexpected symbols"
        )
    frame.to_parquet(temp_parquet, index=True)
    temp_dbn.replace(dbn_path)
    temp_parquet.replace(parquet_path)
    manifest = session_manifest(
        plan_row=plan_row,
        cost_row=cost_row,
        dbn_path=dbn_path,
        parquet_path=parquet_path,
        frame=frame,
    )
    manifest["request"]["delivery"] = "batch_fallback_after_three_stream_failures"
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


def acquisition_summary(
    *,
    out_dir: Path,
    raw_root: Path,
    plan: dict[str, Any],
    cost: dict[str, Any],
    manifests: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    estimated = float(sum(x["request"]["estimated_cost_usd"] for x in manifests))
    expected = float(cost["estimated_total_usd"])
    if not math.isclose(estimated, expected, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"manifest cost total {estimated} does not reconcile with {expected}"
        )
    if len(manifests) != SELECTED_SESSION_COUNT:
        raise RuntimeError(
            f"expected {SELECTED_SESSION_COUNT} session manifests, got {len(manifests)}"
        )
    summary = {
        "schema_version": "Protocol101FT2D55AcquisitionSummaryV1",
        "goal": GOAL,
        "completed_at_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "dataset": DATASET,
        "schema": SCHEMA,
        "raw_root": str(raw_root),
        "selected_dates": [x["date"] for x in manifests],
        "selected_sessions": len(manifests),
        "requested_symbol_sessions": sum(
            int(x["request"]["symbol_count"]) for x in manifests
        ),
        "returned_symbols": sum(
            int(x["download"]["returned_symbol_count"]) for x in manifests
        ),
        "rows": sum(int(x["download"]["rows"]) for x in manifests),
        "bytes": {
            "dbn": sum(int(x["files"]["dbn"]["bytes"]) for x in manifests),
            "parquet": sum(
                int(x["files"]["parquet"]["bytes"]) for x in manifests
            ),
        },
        "estimated_cost_usd": estimated,
        "failed_attempt_billable_cost_upper_bound_usd": float(
            sum(
                row["billable_cost_upper_bound_usd"]
                for row in load_failures(out_dir)
            )
        ),
        "worst_case_estimated_spend_usd": _worst_case_spend(
            cost, load_failures(out_dir)
        ),
        "cost_receipt_total_usd": expected,
        "cost_reconciliation_delta_usd": estimated - expected,
        "hard_cap_usd": HARD_CAP_USD,
        "actual_vendor_invoice_cost_usd": "UNKNOWN",
        "plan": {
            "path": str(_plan_file(out_dir)),
            "sha256": sha256_file(_plan_file(out_dir)),
        },
        "cost_receipt": {
            "path": str(_cost_file(out_dir)),
            "sha256": sha256_file(_cost_file(out_dir)),
        },
        "session_manifests": [
            {
                "date": x["date"],
                "path": str(_raw_paths(raw_root, x["date"])[2]),
                "sha256": sha256_file(_raw_paths(raw_root, x["date"])[2]),
            }
            for x in manifests
        ],
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
            "count": len(load_failures(out_dir)),
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
            "count": len(_load_batch_jobs(out_dir)),
        },
        "side_effects": {
            "paid_download_performed": True,
            "broker_contacted": False,
            "model_training": False,
            "protected_data_accessed": False,
            "signed_contract_modified": False,
            "graph_modified": False,
            "census_modified": False,
            "runtime_or_promotion_modified": False,
        },
    }
    write_json(out_dir / "acquisition_summary.json", summary)
    return summary


def download(
    *,
    client: Any,
    normalized_dir: Path,
    raw_root: Path,
    out_dir: Path,
    approval_manifest: Path,
    approval_text: str | None,
    approval_env_var: str,
) -> dict[str, Any]:
    plan = load_and_verify_plan(normalized_dir, out_dir)
    cost = load_and_verify_cost(out_dir)
    if sha256_file(_plan_file(out_dir)) != cost["acquisition_plan"]["sha256"]:
        raise RuntimeError("cost receipt does not pin the active acquisition plan")
    cost_by_date = {
        str(row["date"]): row for row in cost["session_estimates"]
    }
    unit_price = pinned_or_current_streaming_unit_price(
        client=client,
        out_dir=out_dir,
    )
    reconcile_failures(
        raw_root=raw_root,
        out_dir=out_dir,
        unit_price_usd_per_gb=unit_price,
    )
    manifests: list[dict[str, Any]] = []
    for row in plan["sessions"]:
        session = str(row["date"])
        cost_row = cost_by_date.get(session)
        if cost_row is None:
            raise RuntimeError(f"{session}: missing cost allocation")
        existing = verify_existing_session(
            raw_root=raw_root,
            plan_row=row,
            cost_row=cost_row,
        )
        if existing is not None:
            manifests.append(existing)
            print(
                json.dumps(
                    {
                        "date": session,
                        "status": "reused_exact_existing",
                        "rows": existing["download"]["rows"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        partials = [path for path in _partial_paths(raw_root, session) if path.exists()]
        if partials:
            _archive_failed_attempt(
                raw_root=raw_root,
                out_dir=out_dir,
                session=session,
                cost_row=cost_row,
                error_type="OrphanedPartialFromPriorInvocation",
                error_message=(
                    "A prior identical get_range invocation ended before a "
                    "session manifest could be sealed."
                ),
            )
            reconcile_failures(
                raw_root=raw_root,
                out_dir=out_dir,
                unit_price_usd_per_gb=unit_price,
            )
        while True:
            failures = reconcile_failures(
                raw_root=raw_root,
                out_dir=out_dir,
                unit_price_usd_per_gb=unit_price,
            )
            if _worst_case_spend(cost, failures) > HARD_CAP_USD:
                raise SystemExit(
                    "STOP: failed-attempt billable upper bound raises worst-case "
                    f"spend to ${_worst_case_spend(cost, failures):.6f}, above "
                    f"the ${HARD_CAP_USD:.2f} cap"
                )
            session_failures = [
                item for item in failures if str(item.get("date")) == session
            ]
            if len(session_failures) >= 3:
                require_paid_data_approval(
                    manifest_path=approval_manifest,
                    approval_text=approval_text,
                    approval_env_var=approval_env_var,
                    operation=(
                        f"Databento {DATASET} {SCHEMA} exact D55 batch "
                        f"fallback for {session}"
                    ),
                )
                manifest = _download_one_batch(
                    client=client,
                    raw_root=raw_root,
                    out_dir=out_dir,
                    plan_row=row,
                    cost_row=cost_row,
                )
                break
            require_paid_data_approval(
                manifest_path=approval_manifest,
                approval_text=approval_text,
                approval_env_var=approval_env_var,
                operation=(
                    f"Databento {DATASET} {SCHEMA} exact D55 download for {session}"
                ),
            )
            try:
                manifest = _download_one(
                    client=client,
                    raw_root=raw_root,
                    plan_row=row,
                    cost_row=cost_row,
                )
            except Exception as exc:
                _archive_failed_attempt(
                    raw_root=raw_root,
                    out_dir=out_dir,
                    session=session,
                    cost_row=cost_row,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                reconcile_failures(
                    raw_root=raw_root,
                    out_dir=out_dir,
                    unit_price_usd_per_gb=unit_price,
                )
                continue
            break
        manifests.append(manifest)
        print(
            json.dumps(
                {
                    "date": session,
                    "status": "downloaded",
                    "symbols": manifest["request"]["symbol_count"],
                    "rows": manifest["download"]["rows"],
                    "dbn_bytes": manifest["files"]["dbn"]["bytes"],
                    "parquet_bytes": manifest["files"]["parquet"]["bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return acquisition_summary(
        out_dir=out_dir,
        raw_root=raw_root,
        plan=plan,
        cost=cost,
        manifests=manifests,
    )


def verify(
    *,
    normalized_dir: Path,
    raw_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    plan = load_and_verify_plan(normalized_dir, out_dir)
    cost = load_and_verify_cost(out_dir)
    cost_by_date = {
        str(row["date"]): row for row in cost["session_estimates"]
    }
    manifests = []
    for row in plan["sessions"]:
        existing = verify_existing_session(
            raw_root=raw_root,
            plan_row=row,
            cost_row=cost_by_date[str(row["date"])],
        )
        if existing is None:
            raise RuntimeError(f"{row['date']}: missing D55 session files")
        manifests.append(existing)
    summary = acquisition_summary(
        out_dir=out_dir,
        raw_root=raw_root,
        plan=plan,
        cost=cost,
        manifests=manifests,
    )
    return {
        "outcome": "pass",
        "sessions": len(manifests),
        "estimated_cost_usd": summary["estimated_cost_usd"],
        "dbn_bytes": summary["bytes"]["dbn"],
        "parquet_bytes": summary["bytes"]["parquet"],
    }


def main() -> int:
    args = parse_args()
    if args.mode == "prepare":
        plan = prepare(args.normalized_dir, args.out_dir)
        print(
            json.dumps(
                {
                    "outcome": "prepared",
                    "sessions": len(plan["sessions"]),
                    "symbol_sessions": plan["request"]["symbol_session_count"],
                    "plan": str(_plan_file(args.out_dir)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "cost-gate":
        receipt = cost_gate(
            client=_client(args.env_file),
            normalized_dir=args.normalized_dir,
            out_dir=args.out_dir,
        )
        print(
            json.dumps(
                {
                    "outcome": "cost_gate_passed",
                    "estimated_total_usd": receipt["estimated_total_usd"],
                    "hard_cap_usd": receipt["hard_cap_usd"],
                    "cost_receipt": str(_cost_file(args.out_dir)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "download":
        summary = download(
            client=_client(args.env_file),
            normalized_dir=args.normalized_dir,
            raw_root=args.raw_root,
            out_dir=args.out_dir,
            approval_manifest=args.approval_manifest,
            approval_text=args.approval_text,
            approval_env_var=args.approval_env_var,
        )
        print(
            json.dumps(
                {
                    "outcome": "download_complete",
                    "sessions": summary["selected_sessions"],
                    "estimated_cost_usd": summary["estimated_cost_usd"],
                    "rows": summary["rows"],
                    "raw_root": summary["raw_root"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = verify(
        normalized_dir=args.normalized_dir,
        raw_root=args.raw_root,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
