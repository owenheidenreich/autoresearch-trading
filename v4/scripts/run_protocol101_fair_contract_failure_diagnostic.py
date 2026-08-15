"""Diagnose why fair-contract candidates fail diagnostic generalization.

This is an offline attribution packet. It loads one completed model-search
attempt, scores validation and diagnostic candidates from the approved manifest
splits, and compares selected trades against score distributions, missed
profitable candidates, side/time/moneyness/premium/spread buckets, and simple
context regimes. It does not train, tune, contact brokers/vendors, download
data, change defaults, or promote a candidate.
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

from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    candidate_records_from_row,
    load_json_optional,
    load_model,
    load_rows,
    score_candidate_records,
)


DEFAULT_SEARCH_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_model_search"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_failure_diagnostic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--attempt-id", default="attempt_003_policy0_risk_adjusted_h96_s42")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "diagnostic_test"),
        default=["validation", "diagnostic_test"],
    )
    parser.add_argument(
        "--profitable-label-threshold",
        type=float,
        default=20.0,
        help="Dollar PnL threshold for missed-profitable-candidate diagnostics.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def time_bucket(value: Any) -> str:
    dt = parse_time(value)
    if dt is None:
        return "unknown"
    local = dt.astimezone(ZoneInfo("America/New_York"))
    minute = local.hour * 60 + local.minute
    if minute < 10 * 60:
        return "open_0931_0959"
    if minute < 11 * 60 + 30:
        return "morning_1000_1129"
    if minute < 13 * 60 + 30:
        return "midday_1130_1329"
    if minute < 15 * 60:
        return "afternoon_1330_1459"
    return "late_1500_1530"


def offset_bucket(offset: Any) -> str:
    value = safe_float(offset)
    if value is None:
        return "unknown"
    abs_value = abs(value)
    if abs_value <= 5:
        return "atm_0_5"
    if abs_value <= 20:
        return "near_10_20"
    if abs_value <= 35:
        return "mid_25_35"
    return "far_40_plus"


def premium_bucket(ask: Any) -> str:
    value = safe_float(ask)
    if value is None:
        return "unknown"
    if value < 1.0:
        return "lt_1"
    if value < 3.0:
        return "1_to_3"
    if value < 7.5:
        return "3_to_7_5"
    if value < 15.0:
        return "7_5_to_15"
    return "gte_15"


def context_regime(row: dict[str, Any]) -> dict[str, Any]:
    window = np.asarray(row.get("market_window"), dtype=np.float32)
    if window.ndim != 2 or window.shape[0] == 0 or window.shape[1] < 7:
        return {
            "vwap_side": "unknown",
            "omar_side": "unknown",
            "momentum15_side": "unknown",
            "range_bucket": "unknown",
        }
    last = window[-1]
    spx_close = safe_float(last[0])
    spx_vwap = safe_float(last[2])
    omar = safe_float(last[3])
    session_range = safe_float(last[4])
    momentum15 = safe_float(last[6])
    if spx_close is None or spx_vwap is None:
        vwap_side = "unknown"
    elif spx_close > spx_vwap:
        vwap_side = "above_vwap"
    elif spx_close < spx_vwap:
        vwap_side = "below_vwap"
    else:
        vwap_side = "at_vwap"
    omar_side = "unknown" if omar is None else "omar_pos" if omar > 0 else "omar_neg" if omar < 0 else "omar_flat"
    momentum15_side = (
        "unknown"
        if momentum15 is None
        else "mom15_pos"
        if momentum15 > 0
        else "mom15_neg"
        if momentum15 < 0
        else "mom15_flat"
    )
    if session_range is None:
        range_bucket = "unknown"
    elif session_range < 20:
        range_bucket = "range_lt_20"
    elif session_range < 45:
        range_bucket = "range_20_45"
    elif session_range < 80:
        range_bucket = "range_45_80"
    else:
        range_bucket = "range_gte_80"
    return {
        "vwap_side": vwap_side,
        "omar_side": omar_side,
        "momentum15_side": momentum15_side,
        "range_bucket": range_bucket,
    }


def selected_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("split") or ""),
        str(row.get("session") or ""),
        str(row.get("decision_time") or ""),
        str(row.get("contract_id") or ""),
    )


def load_selected_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {selected_key(row) for row in csv.DictReader(handle)}


def score_split_candidates(
    *,
    split: str,
    paths: list[Path],
    loaded_model,
    selected_keys: set[tuple[str, str, str, str]],
    policy_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for path in paths:
        session = path.name.removesuffix(".pkl")
        for row in load_rows(path):
            decision_time = row.get("decision_time")
            candidates = candidate_records_from_row(
                row,
                session=session,
                split=split,
                policy_index=policy_index,
            )
            if not candidates:
                continue
            score_candidate_records(candidates, loaded_model)
            regime = context_regime(row)
            for item in candidates:
                item["time_bucket"] = time_bucket(item.get("decision_time"))
                item["offset_bucket"] = offset_bucket(item.get("offset"))
                item["premium_bucket"] = premium_bucket(item.get("entry_ask"))
                bid = safe_float(item.get("entry_bid"))
                ask = safe_float(item.get("entry_ask"))
                item["spread"] = None if bid is None or ask is None else float(ask - bid)
                item.update(regime)
            by_score = sorted(candidates, key=lambda item: float(item.get("score") or -1e18), reverse=True)
            by_label = sorted(candidates, key=lambda item: float(item.get("label_net_pnl") or -1e18), reverse=True)
            label_rank_by_contract = {
                str(item.get("contract_id")): rank for rank, item in enumerate(by_label, start=1)
            }
            score_rank_by_contract = {
                str(item.get("contract_id")): rank for rank, item in enumerate(by_score, start=1)
            }
            top_score = by_score[0]
            top_label = by_label[0]
            decision_selected = []
            for item in candidates:
                key = selected_key(item)
                is_selected = key in selected_keys
                item["selected_by_policy"] = bool(is_selected)
                item["score_rank"] = int(score_rank_by_contract.get(str(item.get("contract_id")), 0))
                item["label_rank"] = int(label_rank_by_contract.get(str(item.get("contract_id")), 0))
                item["decision_top_score"] = float(top_score.get("score") or 0.0)
                item["decision_top_score_label"] = float(top_score.get("label_net_pnl") or 0.0)
                item["decision_top_label"] = float(top_label.get("label_net_pnl") or 0.0)
                item["decision_top_label_score"] = float(top_label.get("score") or 0.0)
                item["decision_top_label_contract_id"] = str(top_label.get("contract_id") or "")
                if is_selected:
                    decision_selected.append(item)
                    selected_rows.append(item)
            decision_rows.append(
                {
                    "split": split,
                    "session": session,
                    "decision_time": str(decision_time.isoformat() if hasattr(decision_time, "isoformat") else decision_time),
                    "time_bucket": time_bucket(decision_time),
                    **regime,
                    "candidate_count": len(candidates),
                    "top_score": float(top_score.get("score") or 0.0),
                    "top_score_label_net_pnl": float(top_score.get("label_net_pnl") or 0.0),
                    "top_score_right": str(top_score.get("right") or ""),
                    "top_score_offset": safe_float(top_score.get("offset")),
                    "top_score_entry_ask": safe_float(top_score.get("entry_ask")),
                    "top_score_premium_bucket": premium_bucket(top_score.get("entry_ask")),
                    "top_label_net_pnl": float(top_label.get("label_net_pnl") or 0.0),
                    "top_label_score": float(top_label.get("score") or 0.0),
                    "top_label_right": str(top_label.get("right") or ""),
                    "top_label_offset": safe_float(top_label.get("offset")),
                    "top_label_entry_ask": safe_float(top_label.get("entry_ask")),
                    "top_label_premium_bucket": premium_bucket(top_label.get("entry_ask")),
                    "selected_count": len(decision_selected),
                    "selected_label_net_pnl": float(decision_selected[0].get("label_net_pnl") or 0.0) if decision_selected else None,
                    "selected_score": float(decision_selected[0].get("score") or 0.0) if decision_selected else None,
                    "selected_label_rank": int(decision_selected[0].get("label_rank") or 0) if decision_selected else None,
                    "selected_score_rank": int(decision_selected[0].get("score_rank") or 0) if decision_selected else None,
                }
            )
            candidate_rows.extend(candidates)
    return candidate_rows, decision_rows, selected_rows


def numeric(values: list[Any]) -> np.ndarray:
    cleaned = [float(value) for value in values if safe_float(value) is not None]
    return np.asarray(cleaned, dtype=float)


def distribution(values: list[Any]) -> dict[str, Any]:
    arr = numeric(values)
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def pnl_metrics(rows: list[dict[str, Any]], pnl_key: str = "label_net_pnl") -> dict[str, Any]:
    pnl = numeric([row.get(pnl_key) for row in rows])
    if len(pnl) == 0:
        return {"count": 0, "total_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = abs(float(losses.sum()))
    return {
        "count": int(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else (float("inf") if wins.sum() > 0 else 0.0),
    }


def group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return {
        name: pnl_metrics(items)
        for name, items in sorted(groups.items(), key=lambda item: item[0])
    }


def score_label_correlation(rows: list[dict[str, Any]]) -> float | None:
    scores = numeric([row.get("score") for row in rows])
    labels = numeric([row.get("label_net_pnl") for row in rows])
    if len(scores) != len(labels) or len(scores) < 2:
        return None
    corr = np.corrcoef(scores, labels)[0, 1]
    return float(corr) if math.isfinite(float(corr)) else None


def summarize_split(
    *,
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    profitable_label_threshold: float,
) -> dict[str, Any]:
    missed_profitable = [
        row
        for row in decisions
        if safe_float(row.get("top_label_net_pnl")) is not None
        and float(row["top_label_net_pnl"]) > profitable_label_threshold
        and int(row.get("selected_count") or 0) == 0
    ]
    selected_capture = [
        float(row.get("decision_top_label") or 0.0) - float(row.get("label_net_pnl") or 0.0)
        for row in selected
    ]
    return {
        "candidate_count": len(candidates),
        "decision_count": len(decisions),
        "selected_count": len(selected),
        "selected_metrics": pnl_metrics(selected),
        "selected_by_time_bucket": group_metrics(selected, "time_bucket"),
        "selected_by_right": group_metrics(selected, "right"),
        "selected_by_offset_bucket": group_metrics(selected, "offset_bucket"),
        "selected_by_premium_bucket": group_metrics(selected, "premium_bucket"),
        "selected_by_vwap_side": group_metrics(selected, "vwap_side"),
        "selected_by_omar_side": group_metrics(selected, "omar_side"),
        "selected_by_momentum15_side": group_metrics(selected, "momentum15_side"),
        "top_score_distribution": distribution([row.get("top_score") for row in decisions]),
        "top_label_distribution": distribution([row.get("top_label_net_pnl") for row in decisions]),
        "selected_score_distribution": distribution([row.get("score") for row in selected]),
        "selected_entry_ask_distribution": distribution([row.get("entry_ask") for row in selected]),
        "selected_spread_distribution": distribution([row.get("spread") for row in selected]),
        "score_label_correlation_all_candidates": score_label_correlation(candidates),
        "missed_profitable_decisions": len(missed_profitable),
        "missed_profitable_top_label_distribution": distribution(
            [row.get("top_label_net_pnl") for row in missed_profitable]
        ),
        "selected_missed_best_label_pnl_distribution": distribution(selected_capture),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Failure Diagnostic",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Attempt: `{payload['attempt_id']}`",
        f"- Broker endpoint called: `false`",
        f"- Paper-submit allowed: `false`",
        "",
        "## Split Summary",
        "",
        "| Split | Decisions | Candidates | Selected | Selected PnL | PF | Missed Profitable Decisions | Score/Label Corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, summary in (payload.get("split_summary") or {}).items():
        selected_metrics = summary.get("selected_metrics") or {}
        corr = summary.get("score_label_correlation_all_candidates")
        lines.append(
            f"| `{split}` | {summary.get('decision_count')} | {summary.get('candidate_count')} | "
            f"{summary.get('selected_count')} | {float(selected_metrics.get('total_pnl') or 0.0):.2f} | "
            f"{float(selected_metrics.get('profit_factor') or 0.0):.3f} | "
            f"{summary.get('missed_profitable_decisions')} | "
            f"{float(corr):.3f} |" if corr is not None else
            f"| `{split}` | {summary.get('decision_count')} | {summary.get('candidate_count')} | "
            f"{summary.get('selected_count')} | {float(selected_metrics.get('total_pnl') or 0.0):.2f} | "
            f"{float(selected_metrics.get('profit_factor') or 0.0):.3f} | "
            f"{summary.get('missed_profitable_decisions')} | n/a |"
        )
    lines.extend(["", "## Interpretation", ""])
    for item in payload.get("interpretation") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Outputs", ""])
    for key, value in (payload.get("outputs") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    attempt_dir = args.search_dir / "attempts" / args.attempt_id
    runner_plan = load_json_optional(attempt_dir / "training_runner" / "runner_plan.json")
    training_result = load_json_optional(attempt_dir / "training_runner" / "training_result.json")
    if not runner_plan or not training_result:
        raise SystemExit(f"missing training artifacts for attempt: {args.attempt_id}")
    loaded_model = load_model(training_result)
    selected_keys = load_selected_keys(
        attempt_dir / "selected_candidate_export" / "selected_candidates.csv"
    )
    split_files = runner_plan.get("split_files") or {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    for split in args.splits:
        candidates, decisions, selected = score_split_candidates(
            split=split,
            paths=[Path(path) for path in split_files.get(split, [])],
            loaded_model=loaded_model,
            selected_keys=selected_keys,
            policy_index=int(loaded_model.policy_index),
        )
        all_candidates.extend(candidates)
        all_decisions.extend(decisions)
        all_selected.extend(selected)
        split_summary[split] = summarize_split(
            candidates=candidates,
            decisions=decisions,
            selected=selected,
            profitable_label_threshold=float(args.profitable_label_threshold),
        )

    missed_profitable = [
        row
        for row in all_decisions
        if safe_float(row.get("top_label_net_pnl")) is not None
        and float(row["top_label_net_pnl"]) > float(args.profitable_label_threshold)
        and int(row.get("selected_count") or 0) == 0
    ]
    outputs = {
        "candidate_scores_csv": str(args.out_dir / "candidate_scores.csv"),
        "decision_summary_csv": str(args.out_dir / "decision_summary.csv"),
        "selected_trade_attribution_csv": str(args.out_dir / "selected_trade_attribution.csv"),
        "missed_profitable_decisions_csv": str(args.out_dir / "missed_profitable_decisions.csv"),
        "summary_json": str(args.out_dir / "summary.json"),
        "report_md": str(args.out_dir / "report.md"),
    }
    write_csv(Path(outputs["candidate_scores_csv"]), all_candidates)
    write_csv(Path(outputs["decision_summary_csv"]), all_decisions)
    write_csv(Path(outputs["selected_trade_attribution_csv"]), all_selected)
    write_csv(Path(outputs["missed_profitable_decisions_csv"]), missed_profitable)

    validation = split_summary.get("validation") or {}
    diagnostic = split_summary.get("diagnostic_test") or {}
    interpretation: list[str] = []
    if validation and diagnostic:
        validation_selected = (validation.get("selected_metrics") or {}).get("total_pnl", 0.0)
        diagnostic_selected = (diagnostic.get("selected_metrics") or {}).get("total_pnl", 0.0)
        interpretation.append(
            f"Selected raw-label PnL changes from {validation_selected:.2f} on validation to {diagnostic_selected:.2f} on diagnostic."
        )
        interpretation.append(
            "Compare `selected_by_time_bucket`, side, offset, premium, and regime buckets to see whether the model is selecting a different market slice out of sample."
        )
        interpretation.append(
            "Missed-profitable-decision counts measure opportunities where labels later showed a profitable candidate but the frozen attempt selected nothing at that timestamp."
        )
    payload = {
        "schema_version": "Protocol101FairContractFailureDiagnosticV1",
        "status": "pass",
        "attempt_id": args.attempt_id,
        "feature_contract": "protocol101-live-v1",
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_downloaded": False,
        "split_summary": split_summary,
        "interpretation": interpretation,
        "outputs": outputs,
    }
    Path(outputs["summary_json"]).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    Path(outputs["report_md"]).write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "attempt_id": args.attempt_id,
                "report": outputs["report_md"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
