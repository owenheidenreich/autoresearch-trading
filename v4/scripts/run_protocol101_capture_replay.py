"""Finalize a recorder capture into canonical rows and exact replay evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.ibkr_market_capture import COLLECTION_GATE_VERSION, atomic_write_json
from v4.live.protocol101_capture_replay import (
    ReplayArtifacts,
    build_replay_inputs,
    deterministic_trace_hash,
    lifecycle_canonical_rows,
    replay_capture,
    replay_lifecycle,
)
from v4.live.protocol101_paired_replay_diff import PairedReplayDiffConfig, build_paired_replay_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--capture-id", default=None)
    parser.add_argument("--capture-root", type=Path, default=Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture")
    parser.add_argument("--surface-manifest", type=Path, required=True)
    parser.add_argument("--protocol101-manifest", type=Path, required=True)
    parser.add_argument("--protocol101-summary", type=Path, required=True)
    parser.add_argument("--lifecycle-manifest", type=Path, default=None)
    parser.add_argument("--historical-traces", type=Path, default=None)
    parser.add_argument("--gate-path", type=Path, default=None)
    parser.add_argument("--decision-start-et", default="09:31")
    parser.add_argument("--decision-end-et", default="15:55")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    args = parse_args()
    capture_id = args.capture_id or f"protocol101-recorder-{args.session}"
    out_dir = args.capture_root.expanduser() / args.session / capture_id
    events = out_dir / "market_events.jsonl"
    artifacts = ReplayArtifacts(
        surface_manifest=args.surface_manifest,
        protocol101_manifest=args.protocol101_manifest,
        protocol101_summary=args.protocol101_summary,
        lifecycle_manifest=args.lifecycle_manifest,
    )
    replay_inputs = build_replay_inputs(
        events,
        session=args.session,
        decision_start_et=str(args.decision_start_et),
        decision_end_et=str(args.decision_end_et),
    )
    canonical_a, traces_a = replay_capture(
        events,
        artifacts,
        session=args.session,
        run_id=f"ibkr-captured-{args.session}",
        decision_start_et=str(args.decision_start_et),
        decision_end_et=str(args.decision_end_et),
        replay_inputs=replay_inputs,
    )
    canonical_b, traces_b = replay_capture(
        events,
        artifacts,
        session=args.session,
        run_id=f"ibkr-captured-{args.session}",
        decision_start_et=str(args.decision_start_et),
        decision_end_et=str(args.decision_end_et),
        replay_inputs=replay_inputs,
    )
    hash_a = deterministic_trace_hash(traces_a)
    hash_b = deterministic_trace_hash(traces_b)
    exact = bool(traces_a) and hash_a == hash_b and len(traces_a) == len(traces_b)
    canonical_path = out_dir / "ibkr_canonical_minutes.parquet"
    traces_path = out_dir / "ibkr_protocol101_traces.jsonl"
    pd.DataFrame(canonical_a).to_parquet(canonical_path, index=False)
    write_jsonl(traces_path, traces_a)
    lifecycle_rows = replay_lifecycle(
        lifecycle_canonical_rows(replay_inputs, session=args.session),
        traces_a,
        args.lifecycle_manifest,
    )
    lifecycle_path = out_dir / "ibkr_protocol101_lifecycle_traces.jsonl"
    write_jsonl(lifecycle_path, lifecycle_rows)
    report = {
        "schema_version": "Protocol101SameInputReplayV1",
        "status": "pass" if exact else "fail",
        "session": args.session,
        "decision_rows": len(traces_a),
        "first_decision_ts": traces_a[0].get("decision_ts") if traces_a else None,
        "last_decision_ts": traces_a[-1].get("decision_ts") if traces_a else None,
        "trace_hash_run_a": hash_a,
        "trace_hash_run_b": hash_b,
        "same_input_exact": exact,
        "entry_actions": sum(1 for row in traces_a if row.get("selected_action") == "enter"),
        "lifecycle_rows": len(lifecycle_rows),
        "lifecycle_actions": {
            action: sum(1 for row in lifecycle_rows if row.get("action") == action)
            for action in ("hold", "exit", "stop", "forced_flat")
        },
        "lifecycle_quote_missing_rows": sum(
            1 for row in lifecycle_rows if row.get("quote_missing")
        ),
        "lifecycle_terminal_stale_quote_rows": sum(
            1 for row in lifecycle_rows if row.get("terminal_quote_stale")
        ),
        "broker_order_endpoint_called": False,
        "canonical_path": str(canonical_path),
        "traces_path": str(traces_path),
    }
    atomic_write_json(out_dir / "same_input_replay_summary.json", report)
    (out_dir / "same_input_replay_report.md").write_text(
        "# Protocol101 Same-Input Replay\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Decision rows: `{report['decision_rows']}`\n"
        f"- Exact deterministic replay: `{report['same_input_exact']}`\n"
        f"- Entry actions: `{report['entry_actions']}`\n"
        f"- Trace hash: `{hash_a}`\n"
        "- Broker order endpoint called: `false`\n"
    )
    paired_summary: dict[str, Any] = {
        "schema_version": "Protocol101PairedReplayDiffV1",
        "status": "pending_historical_data",
        "reason": "matching Databento/ThetaData traces were not supplied",
        "rows": 0,
    }
    paired_rows: list[dict[str, Any]] = []
    if args.historical_traces is not None and args.historical_traces.exists():
        historical_rows = [json.loads(line) for line in args.historical_traces.read_text().splitlines() if line.strip()]
        paired_summary = build_paired_replay_diff(
            traces_a,
            historical_rows,
            config=PairedReplayDiffConfig(mode="cross_vendor"),
        )
        paired_rows = list(paired_summary.get("row_results", []))
    atomic_write_json(out_dir / "paired_diff_summary.json", paired_summary)
    pd.DataFrame(paired_rows or [{"status": "pending_historical_data"}]).to_csv(out_dir / "paired_diff.csv", index=False)
    quality = {}
    try:
        quality = json.loads((out_dir / "ibkr_capture_quality.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    gate = {
        "schema_version": COLLECTION_GATE_VERSION,
        "session": args.session,
        "capture_id": capture_id,
        "capture_integrity": "pass" if quality.get("status") == "pass" else "fail",
        "opening_context": "pass" if quality.get("opening_context_ready") and quality.get("missing_opening_minutes") == 0 else "fail",
        "trace_extraction": "pass" if traces_a else "fail",
        "same_input_replay": "pass" if exact else "fail",
        "broker_order_endpoint_called": False,
    }
    gate["status"] = "pass" if all(gate[key] == "pass" for key in ("capture_integrity", "opening_context", "trace_extraction", "same_input_replay")) else "fail"
    gate_path = args.gate_path or (args.capture_root.expanduser() / f"collection_gate_{args.session}.json")
    atomic_write_json(gate_path, gate)
    quality_status = quality.get("status", "missing")
    (out_dir / "report.md").write_text(
        "# Protocol101 Recorder-First Parity Report\n\n"
        f"- Session: `{args.session}`\n"
        f"- Capture quality: `{quality_status}`\n"
        f"- Same-input replay: `{report['status']}`\n"
        f"- Decision rows: `{report['decision_rows']}`\n"
        f"- Entry actions: `{report['entry_actions']}`\n"
        f"- Lifecycle rows: `{report['lifecycle_rows']}`\n"
        f"- Cross-vendor paired diff: `{paired_summary.get('status')}`\n"
        f"- Collection gate: `{gate['status']}`\n"
        "- Broker order endpoint called: `false`\n"
    )
    print(json.dumps({"report": report, "gate": gate, "gate_path": str(gate_path)}, indent=2, sort_keys=True))
    return 0 if gate["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
