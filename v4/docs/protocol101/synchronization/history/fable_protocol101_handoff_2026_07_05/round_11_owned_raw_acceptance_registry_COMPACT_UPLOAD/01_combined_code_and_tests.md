# Combined Code And Tests For Round 11

## code/02_run_protocol101_owned_raw_acceptance_verifier.py

```python
"""Verify owned raw data before Protocol101 fair-contract fold placement.

This data-plane-only verifier turns "raw files exist" into an acceptance
registry. A future fold scaffold must join all three predicates before placing a
session: era role permits the requested role, canonical processed rows exist,
and this acceptance registry marks the session pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCHEMA_VERSION = "Protocol101OwnedRawAcceptanceRegistryV1"
PREDICATE_SCHEMA_VERSION = "Protocol101FoldPlacementPredicateV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_owned_raw_acceptance")
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_SPX_DIR = Path("data/vendor/thetadata/index/spx_1m")
DEFAULT_VIX_DIR = Path("data/vendor/thetadata/index/vix_1m")
DEFAULT_PROCESSED_DIR = Path("data/processed/spxw_0dte_neural_protocol101_owned_raw_acceptance_2024_10_live_v1")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_protocol101_owned_raw_acceptance_2024_10_live_v1")
DEFAULT_ERA_MANIFEST = Path("v4/audit/autoresearch/protocol101_session_era_manifest/summary.json")
DEFAULT_ROLE_POLICY = Path("v4/audit/autoresearch/protocol101_era_role_policy/summary.json")
NY = ZoneInfo("America/New_York")

EARLY_CLOSE_TIMES_ET = {
    # Observed exchange-calendar behavior in local processed artifacts.
    "2024-11-29": "13:00",
    "2024-12-24": "13:15",
    "2025-07-03": "13:00",
    "2025-11-28": "13:00",
    "2025-12-24": "13:15",
    "2026-07-02": "13:00",
}

PRODUCTS = {
    "definition": (
        ("databento", "opra_spxw_definition", "definition"),
        "definition",
    ),
    "cbbo_1m": (
        ("databento", "opra_spxw_cbbo_1m", "cbbo-1m"),
        "cbbo-1m",
    ),
    "ohlcv_1m": (
        ("databento", "opra_spxw_ohlcv_1m", "ohlcv-1m"),
        "ohlcv-1m",
    ),
    "statistics": (
        ("databento", "opra_spxw_statistics", "statistics"),
        "statistics",
    ),
}


@dataclass(frozen=True)
class AcceptanceThresholds:
    min_full_ladder_share: float = 0.70
    min_tradable_minute_share: float = 0.50
    max_missing_processed_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--official-spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--official-vix-dir", type=Path, default=DEFAULT_VIX_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--era-manifest", type=Path, default=DEFAULT_ERA_MANIFEST)
    parser.add_argument("--role-policy", type=Path, default=DEFAULT_ROLE_POLICY)
    parser.add_argument("--role", default="diagnostics_only")
    parser.add_argument("--min-full-ladder-share", type=float, default=0.70)
    parser.add_argument("--min-tradable-minute-share", type=float, default=0.50)
    return parser.parse_args()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def sessions_between(start: str, end: str) -> list[str]:
    cursor = pd.Timestamp(start).date()
    final = pd.Timestamp(end).date()
    sessions: list[str] = []
    while cursor <= final:
        if cursor.weekday() < 5:
            sessions.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return sessions


def product_paths(raw_root: Path, session: str) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for product, ((_, directory, _), suffix) in PRODUCTS.items():
        root = raw_root / "databento" / directory
        out[product] = {
            "parquet": root / f"{session}.{suffix}.parquet",
            "dbn": root / f"{session}.{suffix}.dbn.zst",
        }
    return out


def parquet_row_count(path: Path) -> int | None:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def dbn_row_count(path: Path) -> int | None:
    try:
        import databento as db

        return int(len(db.DBNStore.from_file(str(path)).to_df()))
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_decision_minutes(session: str) -> int:
    close_text = EARLY_CLOSE_TIMES_ET.get(session, "16:00")
    hour, minute = [int(part) for part in close_text.split(":", 1)]
    session_date = date.fromisoformat(session)
    first = datetime.combine(session_date, time(9, 31), tzinfo=NY)
    if close_text == "16:00":
        last = datetime.combine(session_date, time(15, 30), tzinfo=NY)
    else:
        last = datetime.combine(session_date, time(hour, minute), tzinfo=NY) - timedelta(minutes=1)
    return max(int((last - first).total_seconds() // 60) + 1, 0)


def expected_decision_bounds(session: str) -> tuple[datetime, datetime]:
    close_text = EARLY_CLOSE_TIMES_ET.get(session, "16:00")
    hour, minute = [int(part) for part in close_text.split(":", 1)]
    session_date = date.fromisoformat(session)
    first = datetime.combine(session_date, time(9, 31), tzinfo=NY)
    if close_text == "16:00":
        last = datetime.combine(session_date, time(15, 30), tzinfo=NY)
    else:
        last = datetime.combine(session_date, time(hour, minute), tzinfo=NY) - timedelta(minutes=1)
    return first.astimezone(ZoneInfo("UTC")), last.astimezone(ZoneInfo("UTC"))


def to_utc_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def load_pickle_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            rows = pickle.load(handle)
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def finite_share(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).mean())


def processed_quality(session: str, processed_dir: Path) -> dict[str, Any]:
    path = processed_dir / f"{session}.pkl"
    rows = load_pickle_rows(path)
    expected = expected_decision_minutes(session)
    if not rows:
        return {
            "processed_path": str(path),
            "processed_exists": path.exists(),
            "neural_rows": 0,
            "expected_decision_minutes": expected,
            "full_ladder_share": 0.0,
            "tradable_minute_share": 0.0,
            "label_finite_share": 0.0,
            "first_decision_time": None,
            "last_decision_time": None,
            "feature_contract_version": "",
        }
    full_ladder = 0
    tradable_minutes = 0
    label_values: list[np.ndarray] = []
    feature_contract_versions: set[str] = set()
    for row in rows:
        ladder = np.asarray(row.get("option_ladder"))
        if ladder.shape[:2] == (21, 2):
            full_ladder += 1
        mask = np.asarray(row.get("candidate_mask"))
        if mask.size and bool(mask.any()):
            tradable_minutes += 1
        labels = np.asarray(row.get("labels_net_pnl"))
        if labels.size:
            label_values.append(labels.reshape(-1))
        version = row.get("feature_contract_version") or row.get("feature_contract")
        if version:
            feature_contract_versions.add(str(version))
    labels_flat = np.concatenate(label_values) if label_values else np.asarray([], dtype=float)
    return {
        "processed_path": str(path),
        "processed_exists": path.exists(),
        "neural_rows": int(len(rows)),
        "expected_decision_minutes": expected,
        "full_ladder_share": float(full_ladder / len(rows)),
        "tradable_minute_share": float(tradable_minutes / len(rows)),
        "label_finite_share": finite_share(labels_flat),
        "first_decision_time": rows[0].get("decision_time").isoformat() if hasattr(rows[0].get("decision_time"), "isoformat") else str(rows[0].get("decision_time")),
        "last_decision_time": rows[-1].get("decision_time").isoformat() if hasattr(rows[-1].get("decision_time"), "isoformat") else str(rows[-1].get("decision_time")),
        "feature_contract_version": ",".join(sorted(feature_contract_versions)),
    }


def index_quality(session: str, spx_dir: Path, vix_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for symbol, directory in (("spx", spx_dir), ("vix", vix_dir)):
        path = directory / f"{session}.parquet"
        count = parquet_row_count(path) if path.exists() else None
        out[f"{symbol}_path"] = str(path)
        out[f"{symbol}_exists"] = path.exists()
        out[f"{symbol}_rows"] = count or 0
        try:
            frame = pd.read_parquet(path, columns=["event_time"])
            times = pd.to_datetime(frame["event_time"], utc=True)
            out[f"{symbol}_first_event_time"] = times.min().isoformat()
            out[f"{symbol}_last_event_time"] = times.max().isoformat()
        except Exception:
            out[f"{symbol}_first_event_time"] = None
            out[f"{symbol}_last_event_time"] = None
    return out


def raw_quality(session: str, raw_root: Path) -> dict[str, Any]:
    paths = product_paths(raw_root, session)
    products: dict[str, Any] = {}
    for product, files in paths.items():
        parquet_path = files["parquet"]
        dbn_path = files["dbn"]
        parquet_rows = parquet_row_count(parquet_path) if parquet_path.exists() else None
        dbn_rows = dbn_row_count(dbn_path) if dbn_path.exists() else None
        products[product] = {
            "parquet_path": str(parquet_path),
            "dbn_path": str(dbn_path),
            "parquet_exists": parquet_path.exists(),
            "dbn_exists": dbn_path.exists(),
            "parquet_bytes": parquet_path.stat().st_size if parquet_path.exists() else 0,
            "dbn_bytes": dbn_path.stat().st_size if dbn_path.exists() else 0,
            "parquet_rows": parquet_rows,
            "dbn_rows": dbn_rows,
            "row_count_match": parquet_rows is not None and dbn_rows is not None and parquet_rows == dbn_rows,
            "parquet_sha256": file_sha256(parquet_path) if parquet_path.exists() else "",
            "dbn_sha256": file_sha256(dbn_path) if dbn_path.exists() else "",
        }
    return products


def context_causality_quality(session: str, processed_dir: Path) -> dict[str, Any]:
    rows = load_pickle_rows(processed_dir / f"{session}.pkl")
    expected_first, expected_last = expected_decision_bounds(session)
    if not rows:
        return {
            "first_decision_time": None,
            "last_decision_time": None,
            "expected_first_decision_time": expected_first.isoformat(),
            "expected_last_decision_time": expected_last.isoformat(),
            "first_decision_matches_calendar": False,
            "last_decision_matches_calendar": False,
            "one_minute_decision_steps": False,
            "context_lag_exact_one_minute_share": 0.0,
            "future_context_row_count": 0,
            "opening_no_leading_backfill": False,
        }
    decision_times = [to_utc_datetime(row.get("decision_time")) for row in rows]
    source_context_times = [to_utc_datetime(row.get("source_context_time")) for row in rows]
    context_last_times = [to_utc_datetime(row.get("context_last_timestamp")) for row in rows]
    valid_decision_times = [item for item in decision_times if item is not None]
    first_decision = valid_decision_times[0] if valid_decision_times else None
    last_decision = valid_decision_times[-1] if valid_decision_times else None
    one_minute_steps = all(
        (right - left) == timedelta(minutes=1)
        for left, right in zip(valid_decision_times, valid_decision_times[1:])
    )
    exact_lag_count = 0
    comparable_lag_count = 0
    future_context_row_count = 0
    for decision_time, source_context_time, context_last_time in zip(
        decision_times, source_context_times, context_last_times
    ):
        if decision_time is None:
            continue
        expected_source_time = decision_time - timedelta(minutes=1)
        if source_context_time is not None:
            comparable_lag_count += 1
            if source_context_time == expected_source_time:
                exact_lag_count += 1
            if source_context_time > expected_source_time:
                future_context_row_count += 1
        if context_last_time is not None and context_last_time > expected_source_time:
            future_context_row_count += 1
    first_row = rows[0]
    opening_context_start = to_utc_datetime(first_row.get("context_start_timestamp"))
    opening_source_context = to_utc_datetime(first_row.get("source_context_time"))
    opening_no_leading_backfill = bool(
        first_decision == expected_first
        and opening_source_context == expected_first - timedelta(minutes=1)
        and opening_context_start == opening_source_context
        and int(first_row.get("context_minute_rows") or 0) <= 1
        and not bool(first_row.get("context_ready"))
    )
    return {
        "first_decision_time": first_decision.isoformat() if first_decision else None,
        "last_decision_time": last_decision.isoformat() if last_decision else None,
        "expected_first_decision_time": expected_first.isoformat(),
        "expected_last_decision_time": expected_last.isoformat(),
        "first_decision_matches_calendar": first_decision == expected_first,
        "last_decision_matches_calendar": last_decision == expected_last,
        "one_minute_decision_steps": one_minute_steps,
        "context_lag_exact_one_minute_share": float(
            exact_lag_count / comparable_lag_count if comparable_lag_count else 0.0
        ),
        "future_context_row_count": int(future_context_row_count),
        "opening_no_leading_backfill": opening_no_leading_backfill,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def role_policy_allows(role_policy: dict[str, Any], era: str, role: str) -> bool:
    policy = role_policy.get("policy") or {}
    item = policy.get(era) or {}
    return str(role) in set(item.get("permitted_roles") or [])


def acceptance_passes(record: dict[str, Any]) -> bool:
    return record.get("status") == "pass"


def canonical_processed_rows_exist(record: dict[str, Any]) -> bool:
    return bool(record.get("processed", {}).get("processed_exists")) and int(record.get("processed", {}).get("neural_rows") or 0) > 0


def fold_placement_predicate(
    *,
    session: str,
    role: str,
    era_manifest: dict[str, Any],
    role_policy: dict[str, Any],
    acceptance_registry: dict[str, Any],
) -> dict[str, Any]:
    session_records = {
        str(item.get("session")): item for item in era_manifest.get("sessions", []) if item.get("session")
    }
    acceptance_records = {
        str(item.get("session")): item for item in acceptance_registry.get("sessions", []) if item.get("session")
    }
    session_record = session_records.get(session, {})
    acceptance_record = acceptance_records.get(session, {})
    era = str(session_record.get("era") or "")
    checks = {
        "era_permits_role": role_policy_allows(role_policy, era, role),
        "canonical_processed_rows_exist": canonical_processed_rows_exist(acceptance_record),
        "acceptance_status_pass": acceptance_passes(acceptance_record),
    }
    return {
        "schema_version": PREDICATE_SCHEMA_VERSION,
        "session": session,
        "role": role,
        "era": era,
        "placeable": all(checks.values()),
        "checks": checks,
    }


def verify_session(
    *,
    session: str,
    raw_root: Path,
    spx_dir: Path,
    vix_dir: Path,
    processed_dir: Path,
    thresholds: AcceptanceThresholds,
) -> dict[str, Any]:
    raw = raw_quality(session, raw_root)
    index = index_quality(session, spx_dir, vix_dir)
    processed = processed_quality(session, processed_dir)
    context = context_causality_quality(session, processed_dir)
    checks = {
        "raw_product_files_present": all(
            item["parquet_exists"] and item["dbn_exists"] and item["parquet_bytes"] > 0 and item["dbn_bytes"] > 0
            for item in raw.values()
        ),
        "raw_dbn_parquet_row_counts_match": all(item["row_count_match"] for item in raw.values()),
        "index_products_present": bool(index.get("spx_exists")) and bool(index.get("vix_exists")),
        "processed_rows_exist": bool(processed.get("processed_exists")) and int(processed.get("neural_rows") or 0) > 0,
        "expected_decision_row_count": abs(
            int(processed.get("neural_rows") or 0) - int(processed.get("expected_decision_minutes") or 0)
        )
        <= thresholds.max_missing_processed_rows,
        "full_ladder_share": float(processed.get("full_ladder_share") or 0.0) >= thresholds.min_full_ladder_share,
        "tradable_minute_share": float(processed.get("tradable_minute_share") or 0.0) >= thresholds.min_tradable_minute_share,
        "labels_present": float(processed.get("label_finite_share") or 0.0) > 0.0,
        "feature_contract_version_present": processed.get("feature_contract_version") == "protocol101-live-v1",
        "decision_timestamps_match_calendar": bool(context.get("first_decision_matches_calendar"))
        and bool(context.get("last_decision_matches_calendar"))
        and bool(context.get("one_minute_decision_steps")),
        "context_lag_exact_one_minute": float(context.get("context_lag_exact_one_minute_share") or 0.0) == 1.0,
        "no_future_context": int(context.get("future_context_row_count") or 0) == 0,
        "no_leading_backfill_at_open": bool(context.get("opening_no_leading_backfill")),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": SCHEMA_VERSION,
        "session": session,
        "status": status,
        "evidence_grade": "data_plane_only",
        "labels_used_for_strategy_selection": False,
        "pnl_used_for_strategy_selection": False,
        "strategy_metrics_used": False,
        "checks": checks,
        "raw": raw,
        "index": index,
        "processed": processed,
        "context_causality": context,
        "registry_record_hash": stable_hash(
            {
                "session": session,
                "checks": checks,
                "raw": raw,
                "index": index,
                "processed": processed,
                "context_causality": context,
            }
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Owned Raw Acceptance Registry",
        "",
        f"- Status: `{payload['status']}`",
        f"- Batch id: `{payload['batch_id']}`",
        f"- Session count: `{payload['session_count']}`",
        f"- Pass count: `{payload['pass_count']}`",
        f"- Fail count: `{payload['fail_count']}`",
        f"- Evidence grade: `{payload['evidence_grade']}`",
        "",
        "## Sessions",
        "",
    ]
    for record in payload["sessions"]:
        failed = [name for name, passed in record["checks"].items() if not passed]
        lines.append(
            f"- `{record['session']}`: status=`{record['status']}`, rows=`{record['processed']['neural_rows']}`, "
            f"expected=`{record['processed']['expected_decision_minutes']}`, full_ladder=`{record['processed']['full_ladder_share']:.3f}`, "
            f"tradable_minutes=`{record['processed']['tradable_minute_share']:.3f}`, "
            f"context_lag=`{record['context_causality']['context_lag_exact_one_minute_share']:.3f}`, failed=`{failed}`"
        )
    lines.extend(["", "## Guardrails", ""])
    lines.append("- No broker endpoints, paid downloads, model training, threshold tuning, promotion, or real-money paths are used.")
    lines.append("- Registry records are data-plane-only and are not strategy-performance claims.")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "session",
        "status",
        "neural_rows",
        "expected_decision_minutes",
        "full_ladder_share",
        "tradable_minute_share",
        "label_finite_share",
        "feature_contract_version",
        "context_lag_exact_one_minute_share",
        "future_context_row_count",
        "opening_no_leading_backfill",
        "failed_checks",
        "record_hash",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "session": record["session"],
                    "status": record["status"],
                    "neural_rows": record["processed"]["neural_rows"],
                    "expected_decision_minutes": record["processed"]["expected_decision_minutes"],
                    "full_ladder_share": record["processed"]["full_ladder_share"],
                    "tradable_minute_share": record["processed"]["tradable_minute_share"],
                    "label_finite_share": record["processed"]["label_finite_share"],
                    "feature_contract_version": record["processed"]["feature_contract_version"],
                    "context_lag_exact_one_minute_share": record["context_causality"]["context_lag_exact_one_minute_share"],
                    "future_context_row_count": record["context_causality"]["future_context_row_count"],
                    "opening_no_leading_backfill": record["context_causality"]["opening_no_leading_backfill"],
                    "failed_checks": ";".join(name for name, passed in record["checks"].items() if not passed),
                    "record_hash": record["registry_record_hash"],
                }
            )


def main() -> int:
    args = parse_args()
    thresholds = AcceptanceThresholds(
        min_full_ladder_share=float(args.min_full_ladder_share),
        min_tradable_minute_share=float(args.min_tradable_minute_share),
    )
    sessions = sessions_between(args.start_date, args.end_date)
    records = [
        verify_session(
            session=session,
            raw_root=args.raw_root,
            spx_dir=args.official_spx_dir,
            vix_dir=args.official_vix_dir,
            processed_dir=args.processed_dir,
            thresholds=thresholds,
        )
        for session in sessions
    ]
    era_manifest = load_json(args.era_manifest)
    role_policy = load_json(args.role_policy)
    registry_for_predicate = {"sessions": records}
    placement = [
        fold_placement_predicate(
            session=record["session"],
            role=str(args.role),
            era_manifest=era_manifest,
            role_policy=role_policy,
            acceptance_registry=registry_for_predicate,
        )
        for record in records
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if all(record["status"] == "pass" for record in records) else "fail",
        "batch_id": f"owned_raw_acceptance_{args.start_date}_to_{args.end_date}",
        "evidence_grade": "data_plane_only",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "session_count": len(records),
        "pass_count": sum(1 for record in records if record["status"] == "pass"),
        "fail_count": sum(1 for record in records if record["status"] != "pass"),
        "thresholds": {
            "min_full_ladder_share": thresholds.min_full_ladder_share,
            "min_tradable_minute_share": thresholds.min_tradable_minute_share,
            "max_missing_processed_rows": thresholds.max_missing_processed_rows,
        },
        "labels_used_for_strategy_selection": False,
        "pnl_used_for_strategy_selection": False,
        "strategy_metrics_used": False,
        "sessions": records,
        "placement_predicates": placement,
    }
    payload["registry_hash"] = stable_hash(
        {
            "schema_version": payload["schema_version"],
            "batch_id": payload["batch_id"],
            "sessions": records,
            "placement_predicates": placement,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    registry_path = args.out_dir / "acceptance_registry.json"
    csv_path = args.out_dir / "acceptance_registry.csv"
    placement_path = args.out_dir / "fold_placement_predicates.json"
    report_path = args.out_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    placement_path.write_text(json.dumps(placement, indent=2, sort_keys=True) + "\n")
    write_csv(csv_path, records)
    report_path.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "session_count": payload["session_count"],
                "pass_count": payload["pass_count"],
                "fail_count": payload["fail_count"],
                "registry_hash": payload["registry_hash"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

```

## code/03_build_protocol101_era_role_policy.py

```python
"""Build Protocol101 era role policy artifact.

The session era manifest is a facts layer. This artifact is the policy layer:
it maps era names to permitted fold/evidence roles without changing the facts
manifest hash whenever governance rules are revised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "Protocol101EraRolePolicyV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_era_role_policy")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/protocol101_session_era_manifest/summary.json")
KNOWN_ROLES = {
    "train",
    "test",
    "diagnostics_only",
    "report_only",
    "confirmation_one_shot",
}
ROLE_TAXONOMY = {
    "train": "May appear in model-tier training windows when all acceptance and fold predicates pass.",
    "test": "May appear in model-tier test windows when all acceptance and fold predicates pass.",
    "diagnostics_only": "May appear in diagnostics-tier fold test windows for gates, samplers, nulls, and uplift; never model-tier train/test.",
    "report_only": "May be summarized for context but not used for model selection, diagnostics gates, or promotion claims.",
    "confirmation_one_shot": "May be used only for one-shot live/parity confirmation, not training or tuning.",
}
PROMOTION_GUARDS = {
    "pre_program_systematic_negative_guard": {
        "if": "pooled criteria pass but pre_program_oct2024_jun2025 test folds are systematically negative",
        "status": "regime_bound_requires_owner_review",
        "reason": "Pooled fold success must not hide failure in the cleanest owned pre-program era.",
    }
}


DEFAULT_POLICY: dict[str, dict[str, Any]] = {
    "pre_program_oct2024_jun2025": {
        "permitted_roles": ["train", "test", "diagnostics_only"],
        "evidence_tier": "owned_pre_program_raw_pending_acceptance",
        "notes": "Cleanest owned pre-program era after raw acceptance gates pass.",
    },
    "owned_jul_dec2025": {
        "permitted_roles": ["train", "test", "diagnostics_only"],
        "evidence_tier": "owned_mechanically_clean_for_new_fold_models_with_family_selection_caveat",
        "notes": "Mechanically valid for new fold-trained candidates; interpret family-level claims with historical program-ancestry caveat.",
    },
    "q1_2026_development": {
        "permitted_roles": ["diagnostics_only", "report_only"],
        "evidence_tier": "development_report_only_until_split_ancestry_audit",
        "notes": "Default report-only. Jan-Feb may be upgraded only after explicit design/threshold ancestry audit.",
    },
    "post_q1_gap_apr_may2026": {
        "permitted_roles": ["report_only"],
        "evidence_tier": "post_q1_nonfold_report_only",
        "notes": "Post-Q1 gap stays out of fold selection unless separately governed.",
    },
    "confirmation_jun_jul2026": {
        "permitted_roles": ["confirmation_one_shot", "report_only"],
        "evidence_tier": "recorder_confirmation_only",
        "notes": "Recorder/live-parity era. Never train/tune on these sessions.",
    },
    "unassigned_requires_decision": {
        "permitted_roles": [],
        "evidence_tier": "blocked",
        "notes": "Fail-closed. No fold scaffold may place this era.",
    },
    "extension_2024h1": {
        "permitted_roles": ["train", "test", "diagnostics_only"],
        "evidence_tier": "placeholder_pending_owner_approval_and_acceptance",
        "notes": "Placeholder for a possible 2024-01 through 2024-09 purchase tier. No sessions should use this until approved and accepted.",
    },
    "extension_2023": {
        "permitted_roles": ["train", "test", "diagnostics_only"],
        "evidence_tier": "placeholder_pending_owner_approval_and_acceptance",
        "notes": "Placeholder for a possible 2023 extension tier. No sessions should use this until approved and accepted.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_session_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def validate_policy(policy: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    role_errors: list[str] = []
    for era, item in sorted(policy.items()):
        roles = item.get("permitted_roles") or []
        unknown = sorted(set(roles) - KNOWN_ROLES)
        if unknown:
            role_errors.append(f"{era}:unknown_roles:{','.join(unknown)}")
    manifest_eras = set((manifest.get("counts_by_era") or {}).keys())
    missing_policy_eras = sorted(era for era in manifest_eras if era not in policy)
    return {
        "known_roles": sorted(KNOWN_ROLES),
        "role_errors": role_errors,
        "manifest_eras": sorted(manifest_eras),
        "missing_policy_eras": missing_policy_eras,
        "pass": not role_errors and not missing_policy_eras,
    }


def build_policy_artifact(session_manifest: Path) -> dict[str, Any]:
    manifest = load_session_manifest(session_manifest)
    validation = validate_policy(DEFAULT_POLICY, manifest)
    policy_hash = stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "policy": DEFAULT_POLICY,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if validation["pass"] else "fail",
        "policy_hash": policy_hash,
        "session_manifest": str(session_manifest),
        "session_manifest_hash": manifest.get("manifest_hash", ""),
        "known_roles": validation["known_roles"],
        "role_taxonomy": ROLE_TAXONOMY,
        "promotion_guards": PROMOTION_GUARDS,
        "policy": DEFAULT_POLICY,
        "validation": validation,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Era Role Policy",
        "",
        f"- Status: `{payload['status']}`",
        f"- Policy hash: `{payload['policy_hash']}`",
        f"- Session manifest hash: `{payload.get('session_manifest_hash', '')}`",
        "",
        "## Policy",
        "",
    ]
    for era, item in payload["policy"].items():
        lines.append(
            f"- `{era}`: roles=`{item['permitted_roles']}`, tier=`{item['evidence_tier']}`"
        )
    lines.extend(["", "## Role Taxonomy", ""])
    for role, meaning in payload["role_taxonomy"].items():
        lines.append(f"- `{role}`: {meaning}")
    lines.extend(["", "## Promotion Guards", ""])
    for name, guard in payload["promotion_guards"].items():
        lines.append(f"- `{name}`: if `{guard['if']}` then `{guard['status']}`.")
    if payload["validation"]["missing_policy_eras"]:
        lines.extend(["", "## Missing Policy Eras", ""])
        lines.extend(f"- `{era}`" for era in payload["validation"]["missing_policy_eras"])
    if payload["validation"]["role_errors"]:
        lines.extend(["", "## Role Errors", ""])
        lines.extend(f"- `{item}`" for item in payload["validation"]["role_errors"])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The fold scaffold should consume this policy alongside the session era manifest.",
            "- Policy changes should update this artifact without mutating the session facts manifest.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = build_policy_artifact(args.session_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.out_dir / "era_role_policy.json"
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    policy_path.write_text(json.dumps(payload["policy"], indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "policy_hash": payload["policy_hash"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

```

## code/04_build_protocol101_session_era_manifest.py

```python
"""Build a fail-closed Protocol101 session era manifest.

The fold scaffold should consume an explicit session-era manifest instead of
recomputing split/era logic internally. This script scans existing canonical
processed-session manifests and recorder capture quality files, assigns each
session through explicit date-range rules, and marks anything unmatched as
``unassigned_requires_decision``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "Protocol101SessionEraManifestV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_session_era_manifest")
DEFAULT_AUDIT_ROOT = Path("v4/audit/autoresearch")
DEFAULT_CAPTURE_ROOT = Path("~/.autoresearch-trading/live_runtime/ibkr_capture").expanduser()
UNASSIGNED_ERA = "unassigned_requires_decision"
UNASSIGNED_NO_MATCHING_RULE = "no_matching_rule"
UNASSIGNED_OVERLAPPING_RULES = "overlapping_rules"
SESSION_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class EraRule:
    era: str
    start: date
    end: date

    @classmethod
    def parse(cls, value: str) -> "EraRule":
        parts = value.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "era rules must have form era_name:YYYY-MM-DD:YYYY-MM-DD"
            )
        era, start_text, end_text = parts
        return cls(era=era, start=date.fromisoformat(start_text), end=date.fromisoformat(end_text))

    def contains(self, session: str) -> bool:
        session_date = date.fromisoformat(session)
        return self.start <= session_date <= self.end

    def as_dict(self) -> dict[str, str]:
        return {"era": self.era, "start": self.start.isoformat(), "end": self.end.isoformat()}


DEFAULT_ERA_RULES = (
    EraRule.parse("pre_program_oct2024_jun2025:2024-10-01:2025-06-30"),
    EraRule.parse("owned_jul_dec2025:2025-07-01:2025-12-31"),
    EraRule.parse("q1_2026_development:2026-01-01:2026-03-31"),
    EraRule.parse("post_q1_gap_apr_may2026:2026-04-01:2026-05-31"),
    EraRule.parse("confirmation_jun_jul2026:2026-06-01:2026-07-31"),
)

RAW_VENDOR_ROOTS = {
    "databento_opra_definition": Path("data/raw/databento/opra_spxw_definition"),
    "databento_opra_cbbo_1m": Path("data/raw/databento/opra_spxw_cbbo_1m"),
    "databento_opra_ohlcv_1m": Path("data/raw/databento/opra_spxw_ohlcv_1m"),
    "databento_opra_statistics": Path("data/raw/databento/opra_spxw_statistics"),
    "thetadata_spx_1m": Path("data/vendor/thetadata/index/spx_1m"),
    "thetadata_vix_1m": Path("data/vendor/thetadata/index/vix_1m"),
}


@dataclass
class SessionEvidence:
    session: str
    source_types: set[str] = field(default_factory=set)
    source_paths: set[str] = field(default_factory=set)
    source_statuses: set[str] = field(default_factory=set)
    evidence: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        *,
        source_type: str,
        source_path: Path,
        source_status: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.source_types.add(source_type)
        self.source_paths.add(str(source_path))
        if source_status:
            self.source_statuses.add(source_status)
        if evidence:
            for key, value in evidence.items():
                self.evidence.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--era-rule",
        type=EraRule.parse,
        action="append",
        default=None,
        help="Explicit era rule as era_name:YYYY-MM-DD:YYYY-MM-DD. May repeat.",
    )
    parser.add_argument(
        "--allow-unassigned",
        action="store_true",
        help="Write manifest with unassigned sessions and status warn instead of fail.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def session_from_text(value: Any) -> str | None:
    match = SESSION_RE.search(str(value or ""))
    if not match:
        return None
    session = match.group(0)
    try:
        date.fromisoformat(session)
    except ValueError:
        return None
    return session


def collect_processed_manifests(audit_root: Path) -> dict[str, SessionEvidence]:
    sessions: dict[str, SessionEvidence] = {}
    for manifest_path in sorted(audit_root.glob("**/canonical_processed_session_manifest.json")):
        payload = read_json(manifest_path)
        if not payload:
            continue
        for item in payload.get("included_sessions") or []:
            session = session_from_text(item.get("session") if isinstance(item, dict) else item)
            if not session:
                continue
            sessions.setdefault(session, SessionEvidence(session=session)).add(
                source_type="canonical_processed_session",
                source_path=manifest_path,
                source_status=str(payload.get("status") or ""),
                evidence={
                    "processed_file": item.get("processed_file") if isinstance(item, dict) else "",
                    "normalized_base_file": item.get("normalized_base_file") if isinstance(item, dict) else "",
                },
            )
        for item in payload.get("excluded_recorder_or_parity_sessions") or []:
            session = session_from_text(item)
            if not session:
                continue
            sessions.setdefault(session, SessionEvidence(session=session)).add(
                source_type="excluded_recorder_or_parity_session",
                source_path=manifest_path,
                source_status=str(payload.get("status") or ""),
            )
    return sessions


def collect_recorder_manifests(capture_root: Path) -> dict[str, SessionEvidence]:
    sessions: dict[str, SessionEvidence] = {}
    if not capture_root.exists():
        return sessions
    for quality_path in sorted(capture_root.glob("*/protocol101-recorder-*/ibkr_capture_quality.json")):
        session = session_from_text(quality_path)
        payload = read_json(quality_path) or {}
        if not session:
            session = session_from_text(payload.get("capture_id"))
        if not session:
            continue
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        complete = checks.get("complete_regular_session")
        status = "complete" if complete is True else "partial_or_failed"
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
        sessions.setdefault(session, SessionEvidence(session=session)).add(
            source_type="ibkr_recorder_capture",
            source_path=quality_path,
            source_status=status,
            evidence={
                "capture_id": payload.get("capture_id", ""),
                "checkpoint_count": evidence.get("checkpoint_count"),
                "expected_checkpoint_count": evidence.get("expected_checkpoint_count"),
                "broker_order_endpoint_called": evidence.get("broker_order_endpoint_called"),
            },
        )
    return sessions


def collect_raw_vendor_sessions(raw_vendor_roots: dict[str, Path]) -> dict[str, SessionEvidence]:
    sessions: dict[str, SessionEvidence] = {}
    for product, root in sorted(raw_vendor_roots.items()):
        if not root.exists():
            continue
        by_session: dict[str, list[str]] = {}
        for path in sorted(root.glob("*")):
            if not path.is_file():
                continue
            session = session_from_text(path)
            if not session:
                continue
            by_session.setdefault(session, []).append(str(path))
        for session, paths in sorted(by_session.items()):
            sessions.setdefault(session, SessionEvidence(session=session)).add(
                source_type="raw_vendor_session",
                source_path=root,
                source_status="raw_file_present",
                evidence={product: {"file_count": len(paths), "sample_path": paths[0]}},
            )
    return sessions


def merge_session_maps(*maps: dict[str, SessionEvidence]) -> dict[str, SessionEvidence]:
    merged: dict[str, SessionEvidence] = {}
    for session_map in maps:
        for session, item in session_map.items():
            target = merged.setdefault(session, SessionEvidence(session=session))
            target.source_types.update(item.source_types)
            target.source_paths.update(item.source_paths)
            target.source_statuses.update(item.source_statuses)
            target.evidence.update(item.evidence)
    return merged


def assign_era(session: str, rules: Iterable[EraRule]) -> tuple[str, str]:
    matches = [rule.era for rule in rules if rule.contains(session)]
    if len(matches) == 1:
        return matches[0], "matched_single_rule"
    if not matches:
        return UNASSIGNED_ERA, UNASSIGNED_NO_MATCHING_RULE
    return UNASSIGNED_ERA, UNASSIGNED_OVERLAPPING_RULES


def manifest_hash(records: list[dict[str, Any]], rules: list[EraRule]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "era_rules": [rule.as_dict() for rule in rules],
        "sessions": records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    *,
    audit_root: Path,
    capture_root: Path,
    era_rules: list[EraRule],
    allow_unassigned: bool,
    raw_vendor_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    processed = collect_processed_manifests(audit_root)
    recorder = collect_recorder_manifests(capture_root)
    raw_vendor = collect_raw_vendor_sessions(RAW_VENDOR_ROOTS if raw_vendor_roots is None else raw_vendor_roots)
    merged = merge_session_maps(processed, recorder, raw_vendor)
    records: list[dict[str, Any]] = []
    for session, item in sorted(merged.items()):
        era, era_assignment_reason = assign_era(session, era_rules)
        records.append(
            {
                "session": session,
                "era": era,
                "era_assignment_reason": era_assignment_reason,
                "source_types": sorted(item.source_types),
                "source_statuses": sorted(status for status in item.source_statuses if status),
                "source_paths": sorted(item.source_paths),
                "evidence": item.evidence,
            }
        )
    unassigned = [record["session"] for record in records if record["era"] == UNASSIGNED_ERA]
    unassigned_by_reason: dict[str, list[str]] = {}
    for record in records:
        if record["era"] != UNASSIGNED_ERA:
            continue
        reason = str(record.get("era_assignment_reason") or UNASSIGNED_NO_MATCHING_RULE)
        unassigned_by_reason.setdefault(reason, []).append(str(record["session"]))
    digest = manifest_hash(records, era_rules)
    status = "pass"
    if unassigned:
        status = "warn" if allow_unassigned else "fail"
    counts_by_era: dict[str, int] = {}
    for record in records:
        counts_by_era[record["era"]] = counts_by_era.get(record["era"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "manifest_hash": digest,
        "fail_closed_default_era": UNASSIGNED_ERA,
        "allow_unassigned": bool(allow_unassigned),
        "audit_root": str(audit_root),
        "capture_root": str(capture_root),
        "era_rules": [rule.as_dict() for rule in era_rules],
        "session_count": len(records),
        "counts_by_era": dict(sorted(counts_by_era.items())),
        "unassigned_sessions": unassigned,
        "unassigned_sessions_by_reason": {
            key: sorted(value) for key, value in sorted(unassigned_by_reason.items())
        },
        "sessions": records,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ("session", "era", "era_assignment_reason", "source_types", "source_statuses", "source_paths")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "session": record["session"],
                    "era": record["era"],
                    "era_assignment_reason": record["era_assignment_reason"],
                    "source_types": ";".join(record["source_types"]),
                    "source_statuses": ";".join(record["source_statuses"]),
                    "source_paths": ";".join(record["source_paths"]),
                }
            )


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Session Era Manifest",
        "",
        f"- Status: `{payload['status']}`",
        f"- Manifest hash: `{payload['manifest_hash']}`",
        f"- Session count: `{payload['session_count']}`",
        f"- Fail-closed default: `{payload['fail_closed_default_era']}`",
        f"- Allow unassigned: `{str(payload['allow_unassigned']).lower()}`",
        "",
        "## Counts By Era",
        "",
    ]
    for era, count in payload["counts_by_era"].items():
        lines.append(f"- `{era}`: `{count}`")
    lines.extend(["", "## Era Rules", ""])
    for rule in payload["era_rules"]:
        lines.append(f"- `{rule['era']}`: `{rule['start']}` to `{rule['end']}`")
    if payload["unassigned_sessions"]:
        lines.extend(["", "## Unassigned Sessions", ""])
        for reason, sessions in payload["unassigned_sessions_by_reason"].items():
            lines.append(f"- `{reason}`: `{len(sessions)}`")
            lines.extend(f"  - `{session}`" for session in sessions[:20])
            if len(sessions) > 20:
                lines.append(f"  - ... `{len(sessions) - 20}` more")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The fold scaffold should consume this manifest and refuse to place `unassigned_requires_decision` sessions.",
            "- Recorder sessions are included as confirmation-era evidence and retain their complete/partial source status.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    era_rules = list(args.era_rule or DEFAULT_ERA_RULES)
    payload = build_manifest(
        audit_root=args.audit_root,
        capture_root=args.capture_root,
        era_rules=era_rules,
        allow_unassigned=bool(args.allow_unassigned),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = args.out_dir / "sessions_manifest.json"
    summary_path = args.out_dir / "summary.json"
    csv_path = args.out_dir / "sessions_manifest.csv"
    report_path = args.out_dir / "report.md"
    sessions_path.write_text(json.dumps(payload["sessions"], indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(csv_path, payload["sessions"])
    report_path.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "session_count": payload["session_count"],
                "manifest_hash": payload["manifest_hash"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] in {"pass", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

```

## tests/05_test_protocol101_owned_raw_acceptance_verifier.py

```python
"""Tests for Protocol101 owned raw acceptance verifier."""
from __future__ import annotations

from pathlib import Path
import pickle

from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    AcceptanceThresholds,
    context_causality_quality,
    expected_decision_bounds,
    expected_decision_minutes,
    fold_placement_predicate,
    processed_quality,
    role_policy_allows,
)


def test_expected_decision_minutes_handles_full_and_early_close_days() -> None:
    assert expected_decision_minutes("2024-10-01") == 360
    assert expected_decision_minutes("2024-11-29") == 209
    assert expected_decision_minutes("2024-12-24") == 224


def test_expected_decision_bounds_match_calendar() -> None:
    first, last = expected_decision_bounds("2024-10-01")

    assert first.isoformat() == "2024-10-01T13:31:00+00:00"
    assert last.isoformat() == "2024-10-01T19:30:00+00:00"


def test_context_causality_quality_requires_one_minute_lag_and_no_open_backfill(tmp_path: Path) -> None:
    rows = []
    for minute in range(3):
        decision = f"2024-10-01T13:{31 + minute:02d}:00+00:00"
        source = f"2024-10-01T13:{30 + minute:02d}:00+00:00"
        rows.append(
            {
                "decision_time": decision,
                "source_context_time": source,
                "context_start_timestamp": "2024-10-01T13:30:00+00:00",
                "context_last_timestamp": source,
                "context_minute_rows": minute + 1,
                "context_ready": False,
            }
        )
    path = tmp_path / "2024-10-01.pkl"
    with path.open("wb") as handle:
        pickle.dump(rows, handle)

    quality = context_causality_quality("2024-10-01", tmp_path)

    assert quality["first_decision_matches_calendar"] is True
    assert quality["context_lag_exact_one_minute_share"] == 1.0
    assert quality["future_context_row_count"] == 0
    assert quality["opening_no_leading_backfill"] is True


def test_fold_placement_requires_role_processed_rows_and_acceptance() -> None:
    era_manifest = {
        "sessions": [
            {"session": "2024-10-01", "era": "pre_program_oct2024_jun2025"},
            {"session": "2026-07-01", "era": "confirmation_jun_jul2026"},
        ]
    }
    role_policy = {
        "policy": {
            "pre_program_oct2024_jun2025": {"permitted_roles": ["diagnostics_only", "train", "test"]},
            "confirmation_jun_jul2026": {"permitted_roles": ["confirmation_one_shot", "report_only"]},
        }
    }
    registry = {
        "sessions": [
            {
                "session": "2024-10-01",
                "status": "pass",
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
            {
                "session": "2024-10-02",
                "status": "pass",
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
            {
                "session": "2026-07-01",
                "status": "pass",
                "processed": {"processed_exists": True, "neural_rows": 360},
            },
        ]
    }

    assert role_policy_allows(role_policy, "pre_program_oct2024_jun2025", "diagnostics_only")
    assert fold_placement_predicate(
        session="2024-10-01",
        role="diagnostics_only",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )["placeable"] is True
    missing_era = fold_placement_predicate(
        session="2024-10-02",
        role="diagnostics_only",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )
    assert missing_era["placeable"] is False
    assert missing_era["checks"]["era_permits_role"] is False
    confirmation_as_train = fold_placement_predicate(
        session="2026-07-01",
        role="train",
        era_manifest=era_manifest,
        role_policy=role_policy,
        acceptance_registry=registry,
    )
    assert confirmation_as_train["placeable"] is False
    assert confirmation_as_train["checks"]["era_permits_role"] is False


def test_processed_quality_handles_missing_file(tmp_path: Path) -> None:
    quality = processed_quality("2024-10-01", tmp_path)

    assert quality["processed_exists"] is False
    assert quality["neural_rows"] == 0
    assert quality["expected_decision_minutes"] == 360
    assert quality["full_ladder_share"] == 0.0
    assert quality["tradable_minute_share"] == 0.0


def test_threshold_defaults_are_data_plane_only() -> None:
    thresholds = AcceptanceThresholds()

    assert thresholds.min_full_ladder_share == 0.70
    assert thresholds.min_tradable_minute_share == 0.50
    assert thresholds.max_missing_processed_rows == 0

```

## tests/06_test_protocol101_era_role_policy.py

```python
"""Tests for Protocol101 era role policy artifact."""
from __future__ import annotations

import json
from pathlib import Path

from v4.scripts.build_protocol101_era_role_policy import (
    DEFAULT_POLICY,
    build_policy_artifact,
    validate_policy,
)


def test_validate_policy_requires_every_manifest_era() -> None:
    manifest = {"counts_by_era": {"pre_program_oct2024_jun2025": 1, "new_era": 1}}

    result = validate_policy(DEFAULT_POLICY, manifest)

    assert result["pass"] is False
    assert result["missing_policy_eras"] == ["new_era"]


def test_build_policy_artifact_passes_for_default_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "summary.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_hash": "abc123",
                "counts_by_era": {
                    "pre_program_oct2024_jun2025": 10,
                    "owned_jul_dec2025": 20,
                    "q1_2026_development": 30,
                    "post_q1_gap_apr_may2026": 5,
                    "confirmation_jun_jul2026": 3,
                },
            }
        )
    )

    payload = build_policy_artifact(manifest_path)

    assert payload["status"] == "pass"
    assert payload["session_manifest_hash"] == "abc123"
    assert payload["role_taxonomy"]["diagnostics_only"].startswith("May appear in diagnostics-tier")
    assert payload["promotion_guards"]["pre_program_systematic_negative_guard"]["status"] == "regime_bound_requires_owner_review"
    assert payload["policy"]["confirmation_jun_jul2026"]["permitted_roles"] == [
        "confirmation_one_shot",
        "report_only",
    ]
    assert payload["policy"]["extension_2023"]["evidence_tier"] == "placeholder_pending_owner_approval_and_acceptance"
    assert "train" not in payload["policy"]["q1_2026_development"]["permitted_roles"]


def test_default_policy_never_allows_unassigned_sessions() -> None:
    assert DEFAULT_POLICY["unassigned_requires_decision"]["permitted_roles"] == []

```

## tests/07_test_protocol101_session_era_manifest.py

```python
"""Tests for Protocol101 session era manifest generation."""
from __future__ import annotations

import json
from pathlib import Path

from v4.scripts.build_protocol101_session_era_manifest import (
    UNASSIGNED_NO_MATCHING_RULE,
    UNASSIGNED_OVERLAPPING_RULES,
    UNASSIGNED_ERA,
    EraRule,
    build_manifest,
    collect_raw_vendor_sessions,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_build_manifest_assigns_known_eras_and_recorder_days(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    capture_root = tmp_path / "captures"
    _write_json(
        audit_root / "design" / "canonical_processed_session_manifest.json",
        {
            "status": "pass",
            "included_sessions": [
                {"session": "2025-07-01", "processed_file": "p1.pkl"},
                {"session": "2026-01-02", "processed_file": "p2.pkl"},
            ],
            "excluded_recorder_or_parity_sessions": ["2026-07-01"],
        },
    )
    _write_json(
        capture_root
        / "2026-07-01"
        / "protocol101-recorder-2026-07-01"
        / "ibkr_capture_quality.json",
        {
            "capture_id": "protocol101-recorder-2026-07-01",
            "checks": {"complete_regular_session": True},
            "evidence": {
                "checkpoint_count": 390,
                "expected_checkpoint_count": 390,
                "broker_order_endpoint_called": False,
            },
        },
    )

    payload = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[
            EraRule.parse("pre_program_oct2024_jun2025:2024-10-01:2025-06-30"),
            EraRule.parse("owned_jul_dec2025:2025-07-01:2025-12-31"),
            EraRule.parse("q1_2026_development:2026-01-01:2026-03-31"),
            EraRule.parse("post_q1_gap_apr_may2026:2026-04-01:2026-05-31"),
            EraRule.parse("confirmation_jun_jul2026:2026-06-01:2026-07-31"),
        ],
        allow_unassigned=False,
        raw_vendor_roots={},
    )

    eras = {record["session"]: record["era"] for record in payload["sessions"]}
    assert payload["status"] == "pass"
    assert eras == {
        "2025-07-01": "owned_jul_dec2025",
        "2026-01-02": "q1_2026_development",
        "2026-07-01": "confirmation_jun_jul2026",
    }
    july = next(record for record in payload["sessions"] if record["session"] == "2026-07-01")
    assert "ibkr_recorder_capture" in july["source_types"]
    assert "complete" in july["source_statuses"]
    assert payload["manifest_hash"]


def test_build_manifest_fails_closed_for_unassigned_sessions(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    capture_root = tmp_path / "captures"
    _write_json(
        audit_root / "design" / "canonical_processed_session_manifest.json",
        {"included_sessions": [{"session": "2024-01-02", "processed_file": "p.pkl"}]},
    )

    payload = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[EraRule.parse("owned_jul_dec2025:2025-07-01:2025-12-31")],
        allow_unassigned=False,
        raw_vendor_roots={},
    )

    assert payload["status"] == "fail"
    assert payload["unassigned_sessions"] == ["2024-01-02"]
    assert payload["unassigned_sessions_by_reason"] == {UNASSIGNED_NO_MATCHING_RULE: ["2024-01-02"]}
    assert payload["sessions"][0]["era"] == UNASSIGNED_ERA
    assert payload["sessions"][0]["era_assignment_reason"] == UNASSIGNED_NO_MATCHING_RULE


def test_build_manifest_reports_overlapping_rules_separately(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    capture_root = tmp_path / "captures"
    _write_json(
        audit_root / "design" / "canonical_processed_session_manifest.json",
        {"included_sessions": [{"session": "2025-07-02", "processed_file": "p.pkl"}]},
    )

    payload = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[
            EraRule.parse("first:2025-07-01:2025-07-31"),
            EraRule.parse("second:2025-07-02:2025-08-01"),
        ],
        allow_unassigned=False,
        raw_vendor_roots={},
    )

    assert payload["status"] == "fail"
    assert payload["unassigned_sessions_by_reason"] == {UNASSIGNED_OVERLAPPING_RULES: ["2025-07-02"]}
    assert payload["sessions"][0]["era_assignment_reason"] == UNASSIGNED_OVERLAPPING_RULES


def test_manifest_hash_changes_when_era_rule_changes(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    capture_root = tmp_path / "captures"
    _write_json(
        audit_root / "design" / "canonical_processed_session_manifest.json",
        {"included_sessions": [{"session": "2025-07-01", "processed_file": "p.pkl"}]},
    )

    first = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[EraRule.parse("owned_jul_dec2025:2025-07-01:2025-12-31")],
        allow_unassigned=False,
        raw_vendor_roots={},
    )
    second = build_manifest(
        audit_root=audit_root,
        capture_root=capture_root,
        era_rules=[EraRule.parse("alternate_era:2025-07-01:2025-12-31")],
        allow_unassigned=False,
        raw_vendor_roots={},
    )

    assert first["manifest_hash"] != second["manifest_hash"]


def test_collect_raw_vendor_sessions_adds_raw_products(tmp_path: Path) -> None:
    cbbo = tmp_path / "raw" / "cbbo"
    ohlcv = tmp_path / "raw" / "ohlcv"
    cbbo.mkdir(parents=True)
    ohlcv.mkdir(parents=True)
    (cbbo / "2024-10-01.cbbo-1m.parquet").write_text("placeholder")
    (cbbo / "2024-10-01.cbbo-1m.dbn.zst").write_text("placeholder")
    (ohlcv / "2024-10-01.ohlcv-1m.parquet").write_text("placeholder")

    sessions = collect_raw_vendor_sessions(
        {
            "databento_opra_cbbo_1m": cbbo,
            "databento_opra_ohlcv_1m": ohlcv,
        }
    )

    assert sorted(sessions) == ["2024-10-01"]
    item = sessions["2024-10-01"]
    assert item.source_types == {"raw_vendor_session"}
    assert item.source_statuses == {"raw_file_present"}
    assert item.evidence["databento_opra_cbbo_1m"]["file_count"] == 2
    assert item.evidence["databento_opra_ohlcv_1m"]["file_count"] == 1

```

