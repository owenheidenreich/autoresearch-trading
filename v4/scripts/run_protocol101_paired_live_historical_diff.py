"""Offline Protocol101 live-vs-historical paired trace diff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.protocol101_paired_replay_diff import (
    PairedReplayDiffConfig,
    build_paired_replay_diff,
)
from v4.live.protocol101_synchronization import Protocol101ParityAttributionV2


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_paired_live_historical_diff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-traces", type=Path, required=True)
    parser.add_argument("--historical-traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--session", default=None)
    parser.add_argument("--mode", choices=("cross_vendor", "same_input"), default="cross_vendor")
    parser.add_argument("--score-abs-tolerance", type=float, default=1e-9)
    parser.add_argument("--threshold-adjacent-epsilon", type=float, default=0.02)
    parser.add_argument("--candidate-count-tolerance", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = PairedReplayDiffConfig(
        score_abs_tolerance=float(args.score_abs_tolerance),
        threshold_adjacent_epsilon=float(args.threshold_adjacent_epsilon),
        candidate_count_tolerance=int(args.candidate_count_tolerance),
        mode=str(args.mode),
    )
    live_rows = load_trace_rows(args.live_traces, session=args.session)
    historical_rows = load_trace_rows(args.historical_traces, session=args.session)
    result = build_paired_replay_diff(live_rows, historical_rows, config=config)
    result["inputs"] = {
        "live_traces": str(args.live_traces),
        "historical_traces": str(args.historical_traces),
        "session": args.session,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(result["row_results"]).to_csv(args.out_dir / "paired_diff.csv", index=False)
    attribution_path = args.out_dir / "parity_attribution_v2.jsonl"
    with attribution_path.open("w") as handle:
        for row in result["row_results"]:
            key = str(row.get("key") or "")
            pieces = key.split("|", 2)
            attribution = Protocol101ParityAttributionV2(
                session=pieces[0] if pieces else "UNKNOWN",
                decision_time=pieces[1] if len(pieces) > 1 else "UNKNOWN",
                action_match=row.get("live_action") == row.get("historical_action"),
                threshold_adjacent=bool(row.get("threshold_adjacent")),
                categories=list(row.get("mismatch_categories") or []),
                candidate_overlap=row.get("candidate_identity_overlap"),
                live_max_edge=None,
                historical_max_edge=None,
                edge_delta=None,
                selected_contract_match=(
                    row.get("live_contract_id") == row.get("historical_contract_id")
                    if row.get("live_contract_id") is not None
                    or row.get("historical_contract_id") is not None
                    else None
                ),
                explanation=(
                    "exact_match"
                    if not row.get("mismatch_categories")
                    else ",".join(row.get("mismatch_categories") or [])
                ),
            )
            handle.write(json.dumps(attribution.to_dict(), sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "decision_failures": result["decision_failures"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if str(result["status"]).startswith("pass") else 2


def load_trace_rows(path: Path, *, session: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if session and str(row.get("session", "")) != str(session):
            continue
        if not is_trace_like(row):
            continue
        rows.append(row)
    return rows


def is_trace_like(row: dict[str, Any]) -> bool:
    if row.get("decision_trace_schema_version") or row.get("schema_version") == "Protocol101DecisionTraceV1":
        return True
    if row.get("event_type") in {"candidate_set", "model_decision", "risk_gate"}:
        return True
    trace_keys = {"candidate_universe_hash", "feature_hash", "score_hash", "selected_action"}
    return any(key in row for key in trace_keys)


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Protocol101 Paired Live-Historical Diff",
        "",
        f"- Status: `{result['status']}`",
        f"- Mode: `{result['mode']}`",
        f"- Rows compared: `{result['rows']}`",
        f"- Exact matches: `{result['exact_matches']}`",
        f"- Threshold-adjacent reviews: `{result['threshold_adjacent_reviews']}`",
        f"- Decision failures: `{result['decision_failures']}`",
        f"- Action matches: `{result.get('action_matches')}`",
        f"- Action mismatches: `{result.get('action_mismatches')}`",
        f"- Selected-contract matches: `{result.get('selected_contract_matches')}`",
        f"- Selected-contract mismatches: `{result.get('selected_contract_mismatches')}`",
        f"- Category counts: `{result['category_counts']}`",
        "",
        "## Inputs",
        "",
    ]
    for key, value in result.get("inputs", {}).items():
        lines.append(f"- {key}: `{value}`")
    failures = [row for row in result.get("row_results", []) if row.get("row_status") != "exact_match"]
    if failures:
        lines.extend(["", "## First Mismatches", ""])
        for row in failures[:25]:
            lines.append(
                "- "
                f"`{row.get('key')}` status=`{row.get('row_status')}` categories=`{row.get('mismatch_categories')}` "
                f"live_action=`{row.get('live_action')}` historical_action=`{row.get('historical_action')}`"
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
