"""Compare fair-contract candidate stability across attempts and splits.

This offline diagnostic reads strict selected-candidate replay artifacts from
completed fair-contract model-search attempts and asks a narrow question:
which causal slices, if any, are profitable in both validation and diagnostic
splits after stress?

It does not score models, train, tune thresholds, contact brokers/vendors,
download data, change defaults, or promote anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from v4.scripts.run_protocol101_fair_contract_failure_diagnostic import (
    offset_bucket,
    premium_bucket,
    time_bucket,
)
from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION,
)


DEFAULT_SEARCH_DIRS = [
    Path("v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_sep2025_64_q1_export_v4_account_recheck"),
    Path("v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_sep2025_64_q1_decision_balanced"),
    Path("v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_sep2025_64_q1_tail20_calibration"),
]
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_stability_diagnostic"
)
BUCKET_FIELDS = (
    "overall",
    "time_bucket",
    "right",
    "offset_bucket",
    "premium_bucket",
    "right_time_bucket",
    "right_offset_bucket",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dirs", nargs="+", type=Path, default=DEFAULT_SEARCH_DIRS)
    parser.add_argument(
        "--attempt-ids",
        default="",
        help="Optional comma-separated attempt ids. Empty scans all attempts under search dirs.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-bucket-trades", type=int, default=4)
    parser.add_argument("--min-profit-factor", type=float, default=1.25)
    parser.add_argument(
        "--allow-stale-export-version",
        action="store_true",
        help="Include attempts whose registry does not match the current selected-candidate export implementation.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_registry(search_dir: Path) -> dict[str, dict[str, Any]]:
    registry = search_dir / "experiment_registry.jsonl"
    out: dict[str, dict[str, Any]] = {}
    if not registry.exists():
        return out
    for line in registry.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        attempt_id = str(row.get("attempt_id") or "")
        if attempt_id:
            out[attempt_id] = row
    return out


def selected_attempt_dirs(
    search_dirs: list[Path],
    attempt_ids: set[str],
    *,
    require_current_export_version: bool = True,
) -> tuple[list[tuple[Path, Path, dict[str, Any]]], list[dict[str, Any]]]:
    out: list[tuple[Path, Path, dict[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for search_dir in search_dirs:
        registry = load_registry(search_dir)
        attempts_root = search_dir / "attempts"
        if not attempts_root.exists():
            continue
        for attempt_dir in sorted(path for path in attempts_root.iterdir() if path.is_dir()):
            attempt_id = attempt_dir.name
            if attempt_ids and attempt_id not in attempt_ids:
                continue
            key = (str(search_dir), attempt_id)
            if key in seen:
                continue
            seen.add(key)
            entry = registry.get(attempt_id) or {}
            export_version = str(
                ((entry.get("implementation_versions") or {}).get("selected_candidate_export"))
                or ""
            )
            if require_current_export_version and export_version != SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION:
                skipped.append(
                    {
                        "search_dir": str(search_dir),
                        "attempt_id": attempt_id,
                        "reason": "stale_or_missing_selected_candidate_export_version",
                        "selected_candidate_export_version": export_version,
                    }
                )
                continue
            out.append((search_dir, attempt_dir, entry))
    return out, skipped


def pnl_metrics(rows: list[dict[str, Any]], *, pnl_key: str = "stressed_pnl") -> dict[str, Any]:
    pnl_values = [safe_float(row.get(pnl_key)) for row in rows]
    pnl = np.asarray([value for value in pnl_values if value is not None], dtype=float)
    if len(pnl) == 0:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = abs(float(losses.sum()))
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    drawdowns = cumulative - peak
    return {
        "trades": int(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else (float("inf") if wins.sum() > 0 else 0.0),
        "max_drawdown": float(drawdowns.min()) if len(drawdowns) else 0.0,
    }


def enrich_trade(row: dict[str, Any], *, search_dir: Path, attempt_id: str, registry: dict[str, Any]) -> dict[str, Any]:
    config = registry.get("config") or {}
    enriched = dict(row)
    enriched["search_dir"] = str(search_dir)
    enriched["attempt_id"] = attempt_id
    enriched["target_mode"] = str(config.get("target_mode") or "")
    enriched["fit_mode"] = str(config.get("fit_mode") or "")
    enriched["sample_weight_mode"] = str(config.get("sample_weight_mode") or "")
    enriched["entry_filter"] = str(config.get("entry_filter") or "")
    enriched["max_trades_per_session"] = int(config.get("max_trades_per_session") or 0)
    enriched["time_bucket"] = time_bucket(row.get("decision_time"))
    enriched["offset_bucket"] = offset_bucket(row.get("offset"))
    enriched["premium_bucket"] = premium_bucket(row.get("entry_ask"))
    enriched["right_time_bucket"] = f"{row.get('right')}_{enriched['time_bucket']}"
    enriched["right_offset_bucket"] = f"{row.get('right')}_{enriched['offset_bucket']}"
    return enriched


def load_attempt_trades(search_dir: Path, attempt_dir: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    trades_path = attempt_dir / "selected_candidate_replay_gate" / "strict_replay_trades.csv"
    rows = read_csv_rows(trades_path)
    return [
        enrich_trade(row, search_dir=search_dir, attempt_id=attempt_dir.name, registry=registry)
        for row in rows
    ]


def build_bucket_metrics(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in trades:
        for field in BUCKET_FIELDS:
            bucket = "all" if field == "overall" else str(row.get(field) or "unknown")
            key = (str(row["attempt_id"]), str(row.get("split") or ""), field, bucket)
            grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (attempt_id, split, dimension, bucket), rows in sorted(grouped.items()):
        metrics = pnl_metrics(rows)
        example = rows[0]
        out.append(
            {
                "attempt_id": attempt_id,
                "split": split,
                "dimension": dimension,
                "bucket": bucket,
                "target_mode": example.get("target_mode", ""),
                "fit_mode": example.get("fit_mode", ""),
                "sample_weight_mode": example.get("sample_weight_mode", ""),
                **metrics,
            }
        )
    return out


def build_stability_pairs(
    bucket_metrics: list[dict[str, Any]],
    *,
    min_bucket_trades: int,
    min_profit_factor: float,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in bucket_metrics:
        key = (str(row["attempt_id"]), str(row["dimension"]), str(row["bucket"]))
        by_key.setdefault(key, {})[str(row["split"])] = row
    out: list[dict[str, Any]] = []
    for (attempt_id, dimension, bucket), splits in sorted(by_key.items()):
        validation = splits.get("validation")
        diagnostic = splits.get("diagnostic_test")
        if not validation or not diagnostic:
            continue
        val_trades = int(validation.get("trades") or 0)
        diag_trades = int(diagnostic.get("trades") or 0)
        val_pnl = float(validation.get("total_pnl") or 0.0)
        diag_pnl = float(diagnostic.get("total_pnl") or 0.0)
        val_pf = float(validation.get("profit_factor") or 0.0)
        diag_pf = float(diagnostic.get("profit_factor") or 0.0)
        if (
            val_trades >= min_bucket_trades
            and diag_trades >= min_bucket_trades
            and val_pnl > 0.0
            and diag_pnl > 0.0
            and val_pf >= min_profit_factor
            and diag_pf >= min_profit_factor
        ):
            status = "stable_positive_candidate"
        elif val_pnl * diag_pnl < 0.0:
            status = "split_sign_flip"
        elif val_trades < min_bucket_trades or diag_trades < min_bucket_trades:
            status = "under_sampled"
        elif val_pnl <= 0.0 and diag_pnl <= 0.0:
            status = "both_negative"
        else:
            status = "weak_or_mixed"
        out.append(
            {
                "attempt_id": attempt_id,
                "dimension": dimension,
                "bucket": bucket,
                "status": status,
                "validation_trades": val_trades,
                "diagnostic_trades": diag_trades,
                "validation_pnl": val_pnl,
                "diagnostic_pnl": diag_pnl,
                "validation_profit_factor": val_pf,
                "diagnostic_profit_factor": diag_pf,
                "validation_max_drawdown": float(validation.get("max_drawdown") or 0.0),
                "diagnostic_max_drawdown": float(diagnostic.get("max_drawdown") or 0.0),
                "target_mode": validation.get("target_mode", ""),
                "fit_mode": validation.get("fit_mode", ""),
                "sample_weight_mode": validation.get("sample_weight_mode", ""),
            }
        )
    return out


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Stability Diagnostic",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Attempts inspected: `{payload['attempts_inspected']}`",
        f"- Attempts skipped as stale/missing export version: `{payload.get('attempts_skipped', 0)}`",
        f"- Stable positive buckets: `{payload['stable_positive_bucket_count']}`",
        f"- Required export version: `{payload.get('required_selected_candidate_export_version')}`",
        f"- Broker endpoint called: `false`",
        f"- Paper-submit allowed: `false`",
        "",
        "## Overall Attempt Metrics",
        "",
        "| Attempt | Validation Trades | Validation PnL | Validation PF | Diagnostic Trades | Diagnostic PnL | Diagnostic PF | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("overall_pairs") or []:
        lines.append(
            f"| `{row['attempt_id']}` | {row['validation_trades']} | {row['validation_pnl']:.2f} | "
            f"{row['validation_profit_factor']:.3f} | {row['diagnostic_trades']} | "
            f"{row['diagnostic_pnl']:.2f} | {row['diagnostic_profit_factor']:.3f} | `{row['status']}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    if payload["stable_positive_bucket_count"]:
        lines.append("- At least one causal bucket was positive and met PF/count thresholds in both splits; inspect `stable_bucket_candidates.csv` before proposing a gate.")
    else:
        lines.append("- No causal bucket met the minimum count/PF stability threshold in both validation and diagnostic; this argues against adding a simple side/time/offset/premium gate yet.")
    lines.append("- Split sign flips indicate bucket-specific overfit or regime dependence; these are not safe live gates without further forward evidence.")
    lines.extend(["", "## Outputs", ""])
    for key, value in (payload.get("outputs") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    requested = {item.strip() for item in str(args.attempt_ids or "").split(",") if item.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_trades: list[dict[str, Any]] = []
    attempts, skipped_attempts = selected_attempt_dirs(
        list(args.search_dirs),
        requested,
        require_current_export_version=not bool(args.allow_stale_export_version),
    )
    for search_dir, attempt_dir, registry in attempts:
        all_trades.extend(load_attempt_trades(search_dir, attempt_dir, registry))

    bucket_metrics = build_bucket_metrics(all_trades)
    stability_pairs = build_stability_pairs(
        bucket_metrics,
        min_bucket_trades=int(args.min_bucket_trades),
        min_profit_factor=float(args.min_profit_factor),
    )
    stable = [row for row in stability_pairs if row["status"] == "stable_positive_candidate"]
    overall_pairs = [
        row for row in stability_pairs if row["dimension"] == "overall" and row["bucket"] == "all"
    ]
    outputs = {
        "all_trades_csv": str(args.out_dir / "all_strict_replay_trades_enriched.csv"),
        "bucket_metrics_csv": str(args.out_dir / "bucket_metrics.csv"),
        "stability_pairs_csv": str(args.out_dir / "stability_pairs.csv"),
        "stable_bucket_candidates_csv": str(args.out_dir / "stable_bucket_candidates.csv"),
        "summary_json": str(args.out_dir / "summary.json"),
        "report_md": str(args.out_dir / "report.md"),
    }
    write_csv(Path(outputs["all_trades_csv"]), all_trades)
    write_csv(Path(outputs["bucket_metrics_csv"]), bucket_metrics)
    write_csv(Path(outputs["stability_pairs_csv"]), stability_pairs)
    write_csv(Path(outputs["stable_bucket_candidates_csv"]), stable)

    payload = {
        "schema_version": "Protocol101FairContractStabilityDiagnosticV1",
        "status": "pass",
        "feature_contract": "protocol101-live-v1",
        "attempts_inspected": len({row["attempt_id"] for row in all_trades}),
        "attempts_skipped": len(skipped_attempts),
        "skipped_attempts": skipped_attempts,
        "trade_rows": len(all_trades),
        "required_selected_candidate_export_version": (
            None
            if bool(args.allow_stale_export_version)
            else SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION
        ),
        "min_bucket_trades": int(args.min_bucket_trades),
        "min_profit_factor": float(args.min_profit_factor),
        "stable_positive_bucket_count": len(stable),
        "overall_pairs": overall_pairs,
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_downloaded": False,
        "outputs": outputs,
    }
    Path(outputs["summary_json"]).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    Path(outputs["report_md"]).write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "attempts_inspected": payload["attempts_inspected"],
                "stable_positive_bucket_count": payload["stable_positive_bucket_count"],
                "report": outputs["report_md"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
