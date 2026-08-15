"""Protocol 163 recent data catch-up request and cost estimate.

This script is deliberately non-billable. It may call Databento metadata cost
endpoints for definition estimates, inspect local audit logs, and write request
artifacts. It must not call Databento ``timeseries.get_range`` or ThetaData
historical download endpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOOP_ID = "v4_aplus_hypothesis_163_recent_data_catchup_request"
DATASET = "OPRA.PILLAR"
PARENT_SYMBOL = "SPXW.OPT"
SCHEMAS = ("definition", "cbbo-1m", "ohlcv-1m", "statistics")
RAW_ROOT = Path("data/raw")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_MANIFEST = Path("v4/promotion/PROTOCOL_163_RECENT_DATA_CATCHUP_REQUEST.json")
DEFAULT_START = "2026-04-01"
DEFAULT_END = "2026-05-20"
DEFAULT_PAID_END = "2026-05-18"
KNOWN_MARKET_HOLIDAYS = {
    "2026-04-03",  # Good Friday
}


@dataclass(frozen=True)
class LocalSchemaCostStats:
    schema: str
    observations: int
    mean: float
    median: float
    p75: float
    p90: float
    max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--paid-end-date", default=DEFAULT_PAID_END)
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration-audit-dir", type=Path, default=Path("v4/audit"))
    parser.add_argument(
        "--skip-databento-metadata",
        dest="skip_databento_metadata",
        action="store_true",
        default=True,
        help="Use local calibrated estimates only; still writes a request manifest. This is the default.",
    )
    parser.add_argument(
        "--use-databento-metadata",
        dest="skip_databento_metadata",
        action="store_false",
        help="Optionally call Databento metadata.get_cost for definition estimates. Never downloads market data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp(args.start_date).date()
    end_date = pd.Timestamp(args.end_date).date()
    paid_end_date = pd.Timestamp(args.paid_end_date).date()
    expected_sessions = expected_trading_sessions(start_date, end_date)
    paid_candidate_sessions = [session for session in expected_sessions if session <= paid_end_date]
    existing_reuse_sessions = [
        session
        for session in expected_sessions
        if session > paid_end_date and has_complete_raw_session(args.raw_root, session)
    ]
    missing_paid_sessions = [
        session
        for session in paid_candidate_sessions
        if not has_complete_raw_session(args.raw_root, session)
    ]

    local_records = read_local_cost_records(args.calibration_audit_dir)
    schema_stats = schema_cost_stats(local_records)
    total_stats = per_session_total_stats(local_records)
    metadata_status, definition_estimates = estimate_definition_costs(
        missing_paid_sessions,
        skip_metadata=bool(args.skip_databento_metadata),
    )
    schema_estimates = estimate_schema_costs(
        missing_paid_sessions,
        schema_stats=schema_stats,
        definition_estimates=definition_estimates,
    )
    estimated_total = float(sum(row["estimated_total_usd"] for row in schema_estimates))
    conservative_total = float(sum(row["conservative_total_usd"] for row in schema_estimates))
    proposed_cap = proposed_hard_cap(conservative_total)
    exact_approval = exact_approval_text(
        start_date=start_date,
        paid_end_date=paid_end_date,
        combined_end_date=end_date,
        cap=proposed_cap,
    )
    payload = {
        "protocol": "163_recent_data_catchup_request",
        "paid_data_downloaded": False,
        "download_endpoint_called": False,
        "theta_historical_download_called": False,
        "source": {
            "databento": {
                "dataset": DATASET,
                "schemas": list(SCHEMAS),
                "symbols": (
                    "SPXW.OPT parent definitions, then filtered SPXW 0DTE raw symbols. "
                    "The current known downloader uses all same-day SPXW 0DTE raw symbols; "
                    "a stricter ATM +/- $50 downloader should be implemented before download if cost minimization is preferred."
                ),
            },
            "thetadata": {
                "products": ["SPX 1-minute index bars", "VIX 1-minute index bars"],
                "incremental_subscription_cost_usd": 0.0,
                "approval_still_required_before_download": True,
            },
        },
        "date_plan": {
            "combined_processed_start": start_date.isoformat(),
            "combined_processed_end": end_date.isoformat(),
            "paid_databento_start": start_date.isoformat(),
            "paid_databento_end": paid_end_date.isoformat(),
            "excluded_current_day": "2026-05-21",
            "known_market_holidays_excluded": sorted(KNOWN_MARKET_HOLIDAYS),
            "expected_sessions": [session.isoformat() for session in expected_sessions],
            "missing_paid_sessions": [session.isoformat() for session in missing_paid_sessions],
            "reuse_existing_sessions": [session.isoformat() for session in existing_reuse_sessions],
        },
        "cost_estimate": {
            "method": (
                "Databento metadata.get_cost for parent definition where available, plus local calibrated "
                "schema estimates from prior filtered SPXW 0DTE downloads for cbbo-1m, ohlcv-1m, and statistics. "
                "Exact raw-symbol costs require paid definition downloads and are therefore deferred until approval."
            ),
            "metadata_status": metadata_status,
            "estimated_total_usd": round(estimated_total, 4),
            "conservative_total_usd": round(conservative_total, 4),
            "proposed_hard_cap_usd": proposed_cap,
            "schema_estimates": schema_estimates,
            "local_schema_cost_stats": [asdict(stat) for stat in schema_stats],
            "local_total_cost_stats": total_stats,
        },
        "why_needed": (
            "Protocol163 changes training to account-level serial opportunity cost. The official-context training "
            "stream currently ends at 2026-03-31, and recent April-May 2026 market action should be available "
            "before retraining so the new target is not fitted only to stale regimes."
        ),
        "post_approval_build_steps": [
            "Download only missing Databento OPRA.PILLAR files for missing_paid_sessions.",
            "Download ThetaData SPX/VIX 1-minute bars for the same missing sessions.",
            "Reuse existing local 2026-05-19 and 2026-05-20 raw/index files.",
            "Build normalized official-context rows for 2026-04-01 through 2026-05-20.",
            "Build processed neural rows for the same combined block.",
            "Run continuity, contract, quote, index-context, and OI-source checks before retraining.",
        ],
        "approval_required": {
            "required_before": [
                "Any Databento timeseries.get_range call",
                "Any ThetaData historical index_history_ohlc call",
                "Any paid retry of failed market-data downloads",
            ],
            "exact_approval_text": exact_approval,
        },
        "decision": "approval_required_before_paid_download",
    }
    args.manifest_out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_report(args.out_dir / "request.md", payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "estimated_total_usd": payload["cost_estimate"]["estimated_total_usd"],
                "conservative_total_usd": payload["cost_estimate"]["conservative_total_usd"],
                "proposed_hard_cap_usd": payload["cost_estimate"]["proposed_hard_cap_usd"],
                "missing_paid_sessions": len(missing_paid_sessions),
                "manifest": str(args.manifest_out),
                "request": str(args.out_dir / "request.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def expected_trading_sessions(start: date, end: date) -> list[date]:
    sessions = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5 and cursor.isoformat() not in KNOWN_MARKET_HOLIDAYS:
            sessions.append(cursor)
        cursor = (pd.Timestamp(cursor) + pd.Timedelta(days=1)).date()
    return sessions


def has_complete_raw_session(raw_root: Path, session: date) -> bool:
    stem = session.isoformat()
    paths = {
        "definition": raw_root / "databento" / "opra_spxw_definition" / f"{stem}.definition.parquet",
        "cbbo-1m": raw_root / "databento" / "opra_spxw_cbbo_1m" / f"{stem}.cbbo-1m.parquet",
        "ohlcv-1m": raw_root / "databento" / "opra_spxw_ohlcv_1m" / f"{stem}.ohlcv-1m.parquet",
        "statistics": raw_root / "databento" / "opra_spxw_statistics" / f"{stem}.statistics.parquet",
    }
    return all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def read_local_cost_records(audit_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(audit_dir.glob("databento_*downloads.jsonl")):
        if "1s" in path.name or "highres" in path.name or "es_vwap" in path.name or "context_proxy" in path.name:
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            schema = str(row.get("schema") or "")
            if schema not in SCHEMAS:
                continue
            cost = _finite(row.get("cost_estimate_usd"))
            session = str(row.get("date") or row.get("session") or "")
            if cost is None or not session:
                continue
            records.append({"session": session, "schema": schema, "cost_estimate_usd": cost})
    return records


def schema_cost_stats(records: list[dict[str, Any]]) -> list[LocalSchemaCostStats]:
    stats = []
    for schema in SCHEMAS:
        values = np.array(
            [float(row["cost_estimate_usd"]) for row in records if row["schema"] == schema],
            dtype=float,
        )
        if values.size == 0:
            stats.append(LocalSchemaCostStats(schema, 0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        stats.append(
            LocalSchemaCostStats(
                schema=schema,
                observations=int(values.size),
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                p75=float(np.quantile(values, 0.75)),
                p90=float(np.quantile(values, 0.90)),
                max=float(np.max(values)),
            )
        )
    return stats


def per_session_total_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_session: dict[str, float] = {}
    for row in records:
        by_session[str(row["session"])] = by_session.get(str(row["session"]), 0.0) + float(row["cost_estimate_usd"])
    values = np.array(list(by_session.values()), dtype=float)
    if values.size == 0:
        return {"observations": 0}
    return {
        "observations": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def estimate_definition_costs(
    sessions: list[date],
    *,
    skip_metadata: bool,
) -> tuple[str, dict[str, float]]:
    if skip_metadata or not sessions:
        return ("skipped" if skip_metadata else "not_needed", {})
    try:
        import databento as db
    except ImportError:
        return "failed_import_databento", {}
    client = db.Historical()
    estimates: dict[str, float] = {}
    failures: list[str] = []
    for session in sessions:
        start, end = day_bounds(session)
        try:
            estimates[session.isoformat()] = float(
                client.metadata.get_cost(
                    dataset=DATASET,
                    schema="definition",
                    symbols=PARENT_SYMBOL,
                    stype_in="parent",
                    start=start,
                    end=end,
                )
            )
        except Exception as exc:  # metadata failures should not block the manifest
            failures.append(f"{session.isoformat()}:{type(exc).__name__}:{exc}")
    if failures:
        return f"partial_failure:{'; '.join(failures[:3])}", estimates
    return "ok", estimates


def day_bounds(session: date) -> tuple[str, str]:
    start = datetime.combine(session, time(0, 0), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def estimate_schema_costs(
    sessions: list[date],
    *,
    schema_stats: list[LocalSchemaCostStats],
    definition_estimates: dict[str, float],
) -> list[dict[str, Any]]:
    stats_by_schema = {stat.schema: stat for stat in schema_stats}
    rows = []
    for session in sessions:
        session_key = session.isoformat()
        row: dict[str, Any] = {
            "session": session_key,
            "schema_estimates": {},
            "estimated_total_usd": 0.0,
            "conservative_total_usd": 0.0,
        }
        for schema in SCHEMAS:
            stat = stats_by_schema[schema]
            if schema == "definition" and session_key in definition_estimates:
                estimate = float(definition_estimates[session_key])
                conservative = max(estimate, float(stat.p90 or estimate))
                method = "databento_metadata_get_cost"
            else:
                estimate = float(stat.p75)
                conservative = float(stat.p90 or stat.p75)
                method = "local_calibrated_prior"
            row["schema_estimates"][schema] = {
                "estimated_usd": round(estimate, 6),
                "conservative_usd": round(conservative, 6),
                "method": method,
            }
            row["estimated_total_usd"] += estimate
            row["conservative_total_usd"] += conservative
        row["estimated_total_usd"] = round(float(row["estimated_total_usd"]), 6)
        row["conservative_total_usd"] = round(float(row["conservative_total_usd"]), 6)
        rows.append(row)
    return rows


def proposed_hard_cap(conservative_total: float) -> float:
    if conservative_total <= 0:
        return 10.0
    return float(max(10.0, math.ceil((conservative_total * 1.25) / 5.0) * 5.0))


def exact_approval_text(*, start_date: date, paid_end_date: date, combined_end_date: date, cap: float) -> str:
    return (
        "I approve Protocol 163 recent data catch-up: Databento OPRA.PILLAR "
        f"definition/cbbo-1m/ohlcv-1m/statistics for missing SPXW 0DTE sessions "
        f"{start_date.isoformat()} through {paid_end_date.isoformat()}, plus ThetaData SPX/VIX "
        f"1-minute bars for the same missing sessions, reusing existing local May 19-20 files "
        f"for the combined block through {combined_end_date.isoformat()}, with a hard Databento "
        f"spend cap of ${cap:.2f}."
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    cost = payload["cost_estimate"]
    dates = payload["date_plan"]
    lines = [
        "# Protocol 163 Recent Data Catch-Up Request",
        "",
        "No paid market data was downloaded. No Databento `timeseries.get_range` or ThetaData historical download endpoint was called.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Missing paid sessions: `{len(dates['missing_paid_sessions'])}`",
        f"- Reuse existing sessions: `{dates['reuse_existing_sessions']}`",
        f"- Estimated Databento cost: `${cost['estimated_total_usd']:.4f}`",
        f"- Conservative Databento estimate: `${cost['conservative_total_usd']:.4f}`",
        f"- Proposed hard Databento cap: `${cost['proposed_hard_cap_usd']:.2f}`",
        f"- Metadata status: `{cost['metadata_status']}`",
        "",
        "## Request",
        "",
        f"- Databento dataset: `{payload['source']['databento']['dataset']}`",
        f"- Databento schemas: `{', '.join(payload['source']['databento']['schemas'])}`",
        f"- Databento symbols: `{payload['source']['databento']['symbols']}`",
        f"- ThetaData products: `{', '.join(payload['source']['thetadata']['products'])}`",
        f"- Paid date range: `{dates['paid_databento_start']}` through `{dates['paid_databento_end']}`",
        f"- Combined processed range after reuse: `{dates['combined_processed_start']}` through `{dates['combined_processed_end']}`",
        "",
        "## Why",
        "",
        payload["why_needed"],
        "",
        "## Approval Text",
        "",
        "Use this exact text to approve the paid download batch:",
        "",
        "```text",
        payload["approval_required"]["exact_approval_text"],
        "```",
        "",
        "## Notes",
        "",
        "- Exact filtered raw-symbol costs require paid definitions, so non-definition schema costs are calibrated from prior local SPXW 0DTE downloads.",
        "- May 19-20 are already present locally and should be reused rather than redownloaded.",
        "- 2026-05-21 is excluded until it is a complete session.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


if __name__ == "__main__":
    raise SystemExit(main())
