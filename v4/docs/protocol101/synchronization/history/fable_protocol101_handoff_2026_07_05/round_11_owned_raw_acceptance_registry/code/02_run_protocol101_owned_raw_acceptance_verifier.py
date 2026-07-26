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
