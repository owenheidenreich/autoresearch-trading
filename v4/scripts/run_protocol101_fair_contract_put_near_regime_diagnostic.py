"""Attribute put-near fair-contract failures to day and context regimes.

This offline packet joins strict replay trades back to the approved
protocol101-live-v1 processed rows so selected trades can be inspected against
causal context available at the decision time. It is diagnostic only: it does
not train, tune thresholds, contact brokers/vendors, download data, change
defaults, or promote a candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from v4.scripts.run_protocol101_fair_contract_failure_diagnostic import (
    context_regime,
    offset_bucket,
    premium_bucket,
    safe_float,
    time_bucket,
)
from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    load_json_optional,
    load_rows,
)


DEFAULT_SEARCH_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_sep2025_64_q1_put_near_gate"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_put_near_regime_diagnostic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument(
        "--attempt-ids",
        default=(
            "attempt_092_policy0_hgb_teacher_put_near_cap2_s42,"
            "attempt_093_policy0_hgb_profit_classifier_put_near_cap2_s42"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def trade_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("split") or ""),
        str(row.get("session") or ""),
        str(row.get("decision_time") or ""),
    )


def local_time(value: Any) -> str:
    dt = parse_time(value)
    if dt is None:
        return ""
    return dt.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")


def market_context_fields(row: dict[str, Any]) -> dict[str, Any]:
    window = np.asarray(row.get("market_window"), dtype=np.float32)
    if window.ndim != 2 or window.shape[0] == 0 or window.shape[1] < 7:
        return {
            "spx_close": None,
            "spx_vwap": None,
            "vwap_gap": None,
            "omar": None,
            "session_range": None,
            "momentum5": None,
            "momentum15": None,
            "put_momentum_alignment": "unknown",
            "vwap_gap_bucket": "unknown",
        }
    last = window[-1]
    spx_close = safe_float(last[0])
    spx_vwap = safe_float(last[2])
    omar = safe_float(last[3])
    session_range = safe_float(last[4])
    momentum5 = safe_float(last[5])
    momentum15 = safe_float(last[6])
    vwap_gap = None if spx_close is None or spx_vwap is None else float(spx_close - spx_vwap)
    if vwap_gap is None:
        gap_bucket = "unknown"
    elif vwap_gap <= -10:
        gap_bucket = "below_vwap_gte_10"
    elif vwap_gap < -2:
        gap_bucket = "below_vwap_2_10"
    elif vwap_gap <= 2:
        gap_bucket = "near_vwap"
    elif vwap_gap < 10:
        gap_bucket = "above_vwap_2_10"
    else:
        gap_bucket = "above_vwap_gte_10"
    if momentum15 is None:
        put_alignment = "unknown"
    elif momentum15 < 0:
        put_alignment = "put_aligned_mom15_down"
    elif momentum15 > 0:
        put_alignment = "put_counter_mom15_up"
    else:
        put_alignment = "put_neutral_mom15_flat"
    return {
        "spx_close": spx_close,
        "spx_vwap": spx_vwap,
        "vwap_gap": vwap_gap,
        "omar": omar,
        "session_range": session_range,
        "momentum5": momentum5,
        "momentum15": momentum15,
        "put_momentum_alignment": put_alignment,
        "vwap_gap_bucket": gap_bucket,
    }


def context_by_selected_key(split_files: dict[str, list[str]], selected_keys: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for split, paths in split_files.items():
        for path_str in paths:
            path = Path(path_str)
            session = path.name.removesuffix(".pkl")
            needed_for_session = {
                key for key in selected_keys if key[0] == split and key[1] == session
            }
            if not needed_for_session:
                continue
            for row in load_rows(path):
                decision_time = row.get("decision_time")
                decision_time_s = decision_time.isoformat() if hasattr(decision_time, "isoformat") else str(decision_time)
                key = (split, session, decision_time_s)
                if key not in needed_for_session:
                    continue
                out[key] = {
                    **context_regime(row),
                    **market_context_fields(row),
                }
    return out


def pnl_metrics(rows: list[dict[str, Any]], pnl_key: str = "stressed_pnl") -> dict[str, Any]:
    values = [safe_float(row.get(pnl_key)) for row in rows]
    pnl = np.asarray([value for value in values if value is not None], dtype=float)
    if len(pnl) == 0:
        return {"trades": 0, "total_pnl": 0.0, "profit_factor": 0.0, "win_rate": 0.0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": int(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else (float("inf") if wins.sum() > 0 else 0.0),
    }


def add_trade_sequence(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("attempt_id")), str(row.get("split")), str(row.get("session"))),
            [],
        ).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: str(row.get("decision_time") or ""))
        session_pnl = sum(float(safe_float(row.get("stressed_pnl")) or 0.0) for row in group_rows)
        for idx, row in enumerate(group_rows, start=1):
            row["trade_index_in_session"] = idx
            row["session_trade_count"] = len(group_rows)
            row["session_stressed_pnl"] = session_pnl
            row["session_outcome"] = "positive_day" if session_pnl > 0 else "negative_day" if session_pnl < 0 else "flat_day"
            row["first_trade_stressed_pnl"] = float(safe_float(group_rows[0].get("stressed_pnl")) or 0.0)
            row["first_trade_outcome"] = (
                "first_trade_win"
                if row["first_trade_stressed_pnl"] > 0
                else "first_trade_loss"
                if row["first_trade_stressed_pnl"] < 0
                else "first_trade_flat"
            )


def enrich_attempt_trades(search_dir: Path, attempt_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    attempt_dir = search_dir / "attempts" / attempt_id
    runner_plan = load_json_optional(attempt_dir / "training_runner" / "runner_plan.json")
    training_result = load_json_optional(attempt_dir / "training_runner" / "training_result.json")
    trades = read_csv_rows(attempt_dir / "selected_candidate_replay_gate" / "strict_replay_trades.csv")
    blockers: list[str] = []
    if not runner_plan:
        blockers.append(f"missing_runner_plan:{attempt_id}")
        return [], blockers
    if not training_result:
        blockers.append(f"missing_training_result:{attempt_id}")
    selected_keys = {trade_key(row) for row in trades}
    context_map = context_by_selected_key(
        runner_plan.get("split_files") or {},
        selected_keys,
    )
    config = (training_result or {}).get("config") or {}
    enriched: list[dict[str, Any]] = []
    for row in trades:
        key = trade_key(row)
        context = context_map.get(key)
        if context is None:
            blockers.append(f"missing_context:{attempt_id}:{key}")
            context = {}
        item = dict(row)
        item["attempt_id"] = attempt_id
        item["target_mode"] = str(config.get("target_mode") or "")
        item["entry_filter"] = str(config.get("entry_filter") or "")
        item["local_time"] = local_time(row.get("decision_time"))
        item["time_bucket"] = time_bucket(row.get("decision_time"))
        item["offset_bucket"] = offset_bucket(row.get("offset"))
        item["premium_bucket"] = premium_bucket(row.get("entry_ask"))
        item.update(context)
        enriched.append(item)
    add_trade_sequence(enriched)
    return enriched, blockers


def session_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("attempt_id")), str(row.get("split")), str(row.get("session"))),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for (attempt_id, split, session), group_rows in sorted(grouped.items()):
        group_rows.sort(key=lambda row: str(row.get("decision_time") or ""))
        metrics = pnl_metrics(group_rows)
        first = group_rows[0] if group_rows else {}
        out.append(
            {
                "attempt_id": attempt_id,
                "split": split,
                "session": session,
                **metrics,
                "first_trade_time": first.get("local_time"),
                "first_trade_stressed_pnl": safe_float(first.get("stressed_pnl")),
                "first_trade_vwap_side": first.get("vwap_side"),
                "first_trade_omar_side": first.get("omar_side"),
                "first_trade_momentum15_side": first.get("momentum15_side"),
                "first_trade_put_momentum_alignment": first.get("put_momentum_alignment"),
                "first_trade_vwap_gap_bucket": first.get("vwap_gap_bucket"),
                "outcome": "positive_day" if metrics["total_pnl"] > 0 else "negative_day" if metrics["total_pnl"] < 0 else "flat_day",
            }
        )
    return out


def bucket_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = [
        "time_bucket",
        "trade_index_in_session",
        "first_trade_outcome",
        "vwap_side",
        "omar_side",
        "momentum15_side",
        "put_momentum_alignment",
        "vwap_gap_bucket",
        "range_bucket",
        "premium_bucket",
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for dimension in dimensions:
            key = (
                str(row.get("attempt_id")),
                str(row.get("split")),
                dimension,
                str(row.get(dimension) or "unknown"),
            )
            grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (attempt_id, split, dimension, bucket), group_rows in sorted(grouped.items()):
        out.append(
            {
                "attempt_id": attempt_id,
                "split": split,
                "dimension": dimension,
                "bucket": bucket,
                **pnl_metrics(group_rows),
            }
        )
    return out


def top_negative_diagnostic_buckets(
    buckets: list[dict[str, Any]],
    *,
    min_trades: int = 3,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Return largest diagnostic losses with validation counterparts.

    This helps distinguish causal guard candidates from split-unstable
    hindsight patterns. Buckets that are diagnostic-negative but
    validation-positive are evidence of instability, not a safe repair.
    """
    by_key = {
        (
            str(row.get("attempt_id")),
            str(row.get("split")),
            str(row.get("dimension")),
            str(row.get("bucket")),
        ): row
        for row in buckets
    }
    rows: list[dict[str, Any]] = []
    for row in buckets:
        if row.get("split") != "diagnostic_test":
            continue
        trades = int(row.get("trades") or 0)
        total_pnl = float(row.get("total_pnl") or 0.0)
        if trades < min_trades or total_pnl >= 0.0:
            continue
        validation = by_key.get(
            (
                str(row.get("attempt_id")),
                "validation",
                str(row.get("dimension")),
                str(row.get("bucket")),
            )
        )
        validation_pnl = float(validation.get("total_pnl") or 0.0) if validation else None
        if validation is None:
            relation = "missing_validation_bucket"
        elif validation_pnl is not None and validation_pnl < 0.0:
            relation = "negative_in_both_splits"
        elif validation_pnl is not None and validation_pnl > 0.0:
            relation = "diagnostic_negative_validation_positive"
        else:
            relation = "diagnostic_negative_validation_flat"
        rows.append(
            {
                "attempt_id": row.get("attempt_id"),
                "dimension": row.get("dimension"),
                "bucket": row.get("bucket"),
                "diagnostic_trades": trades,
                "diagnostic_total_pnl": total_pnl,
                "diagnostic_profit_factor": row.get("profit_factor"),
                "validation_trades": validation.get("trades") if validation else None,
                "validation_total_pnl": validation_pnl,
                "validation_profit_factor": validation.get("profit_factor") if validation else None,
                "relation": relation,
            }
        )
    rows.sort(key=lambda item: float(item.get("diagnostic_total_pnl") or 0.0))
    return rows[:limit]


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Put-Near Regime Diagnostic",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Attempts inspected: `{payload['attempts_inspected']}`",
        f"- Trade rows: `{payload['trade_rows']}`",
        f"- Broker endpoint called: `false`",
        f"- Paper-submit allowed: `false`",
        "",
        "## Session Failures",
        "",
        "| Attempt | Split | Session | Trades | PnL | First Trade | First Context | Outcome |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in payload.get("session_summary", []):
        if row.get("split") != "diagnostic_test" or float(row.get("total_pnl") or 0.0) >= 0.0:
            continue
        context = "/".join(
            str(row.get(key) or "")
            for key in (
                "first_trade_vwap_side",
                "first_trade_omar_side",
                "first_trade_momentum15_side",
                "first_trade_put_momentum_alignment",
            )
        )
        lines.append(
            f"| `{row['attempt_id']}` | `{row['split']}` | `{row['session']}` | "
            f"{row['trades']} | {float(row['total_pnl']):.2f} | "
            f"{float(row.get('first_trade_stressed_pnl') or 0.0):.2f} | `{context}` | `{row['outcome']}` |"
        )
    lines.extend(
        [
            "",
            "## Largest Negative Diagnostic Buckets",
            "",
            "| Attempt | Dimension | Bucket | Diagnostic Trades | Diagnostic PnL | Diagnostic PF | Validation Trades | Validation PnL | Validation PF | Relation |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload.get("top_negative_diagnostic_buckets", []):
        lines.append(
            f"| `{row['attempt_id']}` | `{row['dimension']}` | `{row['bucket']}` | "
            f"{row['diagnostic_trades']} | {float(row['diagnostic_total_pnl']):.2f} | "
            f"{float(row.get('diagnostic_profit_factor') or 0.0):.3f} | "
            f"{row.get('validation_trades') or ''} | "
            f"{'' if row.get('validation_total_pnl') is None else f'{float(row.get('validation_total_pnl') or 0.0):.2f}'} | "
            f"{'' if row.get('validation_profit_factor') is None else f'{float(row.get('validation_profit_factor') or 0.0):.3f}'} | "
            f"`{row['relation']}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    for item in payload.get("interpretation", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Outputs", ""])
    for key, value in payload.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    attempt_ids = [item.strip() for item in str(args.attempt_ids or "").split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for attempt_id in attempt_ids:
        rows, attempt_blockers = enrich_attempt_trades(args.search_dir, attempt_id)
        all_rows.extend(rows)
        blockers.extend(attempt_blockers)
    sessions = session_summary(all_rows)
    buckets = bucket_summary(all_rows)
    negative_buckets = top_negative_diagnostic_buckets(buckets)
    diagnostic_negative_sessions = [
        row
        for row in sessions
        if row.get("split") == "diagnostic_test" and float(row.get("total_pnl") or 0.0) < 0.0
    ]
    outputs = {
        "selected_trade_context_csv": str(args.out_dir / "selected_trade_context.csv"),
        "session_summary_csv": str(args.out_dir / "session_summary.csv"),
        "bucket_summary_csv": str(args.out_dir / "bucket_summary.csv"),
        "top_negative_diagnostic_buckets_csv": str(args.out_dir / "top_negative_diagnostic_buckets.csv"),
        "diagnostic_negative_sessions_csv": str(args.out_dir / "diagnostic_negative_sessions.csv"),
        "summary_json": str(args.out_dir / "summary.json"),
        "report_md": str(args.out_dir / "report.md"),
    }
    write_csv(Path(outputs["selected_trade_context_csv"]), all_rows)
    write_csv(Path(outputs["session_summary_csv"]), sessions)
    write_csv(Path(outputs["bucket_summary_csv"]), buckets)
    write_csv(Path(outputs["top_negative_diagnostic_buckets_csv"]), negative_buckets)
    write_csv(Path(outputs["diagnostic_negative_sessions_csv"]), diagnostic_negative_sessions)
    interpretation = [
        "This packet explains failed put-near attempts after the fact; it is not a training or threshold-selection run.",
        "A causal gate should only be tested if losing diagnostic sessions share a live-observable context pattern that is also absent or profitable in validation.",
    ]
    if diagnostic_negative_sessions:
        worst = min(diagnostic_negative_sessions, key=lambda row: float(row.get("total_pnl") or 0.0))
        interpretation.append(
            f"Worst diagnostic session is {worst.get('session')} for {worst.get('attempt_id')} with stressed PnL {float(worst.get('total_pnl') or 0.0):.2f}."
        )
    split_unstable = [
        row
        for row in negative_buckets
        if row.get("relation") == "diagnostic_negative_validation_positive"
    ]
    if split_unstable:
        interpretation.append(
            "Some large diagnostic-loss buckets are positive in validation; treat those as split-instability evidence rather than safe causal guards."
        )
    payload = {
        "schema_version": "Protocol101FairContractPutNearRegimeDiagnosticV1",
        "status": "pass" if not blockers else "partial",
        "attempts_inspected": len(attempt_ids),
        "trade_rows": len(all_rows),
        "blockers": blockers,
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_downloaded": False,
        "session_summary": sessions,
        "top_negative_diagnostic_buckets": negative_buckets,
        "interpretation": interpretation,
        "outputs": outputs,
    }
    Path(outputs["summary_json"]).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    Path(outputs["report_md"]).write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "attempts_inspected": payload["attempts_inspected"],
                "trade_rows": payload["trade_rows"],
                "report": outputs["report_md"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
