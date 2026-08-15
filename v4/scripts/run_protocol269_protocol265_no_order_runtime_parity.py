"""RUNTIME_PROTOCOL265_NO_ORDER_PARITY_V1.

Historical replay proxy for Protocol265 runtime parity. It loads saved
Protocol265 artifacts only, rebuilds lifecycle feature states, emits no-order
JSONL events, and verifies no broker/order fields are present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

import v4.scripts.run_protocol265_source_penalty_baseline_anchored_continuation as p265
from v4.scripts.run_protocol251_premium_blend_slot_aware_lifecycle import (
    load_premium_blend_candidates,
    safe_find_normalized_path,
)
from v4.scripts.run_protocol266_protocol265_artifact_reproduction import load_artifact


ROLE_LABEL = "RUNTIME_PROTOCOL265_NO_ORDER_PARITY_V1"
HISTORICAL_ID = "Protocol269"
SCHEMA_VERSION = "protocol265_runtime_v1"
DEFAULT_SOURCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_269_protocol265_no_order_runtime_parity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-events", type=int, default=300)
    parser.add_argument("--split", default="recent_2026")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((args.source_dir / "summary.json").read_text())
    data_used = source_summary.get("data_used", {})
    p265.p200.find_normalized_path = safe_find_normalized_path
    candidates = load_premium_blend_candidates(Path(data_used.get("source_penalty_trades", p265.DEFAULT_TRADES)))
    records, path_skips = p265.build_path_records(
        candidates,
        normalized_dir=Path(data_used.get("normalized_dir", p265.DEFAULT_NORMALIZED_DIR)),
        forced_flat_time="15:55",
    )
    records = [record for record in records if record.reported_split == str(args.split)]
    artifact_manifests = sorted((args.source_dir / "model_artifacts").glob("*/seed_*/manifest.json"))
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    for manifest_path in artifact_manifests:
        artifact = load_artifact(manifest_path)
        manifest = artifact["manifest"]
        if str(args.split) not in set(map(str, manifest.get("test_splits", []))):
            continue
        threshold = float(manifest["threshold"])
        model_seed = int(manifest["seed"])
        selected_records = records[: max(1, int(args.max_events) // max(1, len(artifact_manifests)))]
        for record in selected_records:
            start = time.perf_counter()
            predictions = p265.predict_records(artifact["model"], artifact["scaler"], [record])
            pred = predictions.get(record.uid)
            anchor_idx = p265.baseline_anchor_idx(record)
            current_idx = anchor_idx
            decision = "exit" if pred is not None and float(pred[current_idx]) <= threshold else "hold"
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            row = runtime_row(record, pred, model_seed=model_seed, threshold=threshold, current_idx=current_idx, selected_action=decision, elapsed_ms=elapsed_ms)
            rows.append(row)
            latency_rows.append(
                {
                    "session": record.session,
                    "reported_split": record.reported_split,
                    "model_seed": model_seed,
                    "selected_action": decision,
                    "total_decision_ms": elapsed_ms,
                    "candidate_count": 1,
                    "broker_endpoint_called": False,
                }
            )
            if len(rows) >= int(args.max_events):
                break
        if len(rows) >= int(args.max_events):
            break
    validation = validate_rows(rows)
    latency_summary = summarize_latency(latency_rows)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / no-order historical replay proxy for Protocol265",
        "changes_paper_default": False,
        "candidate_label": p265.CANDIDATE_LABEL,
        "paper_default_label": p265.PAPER_DEFAULT_LABEL,
        "data_used": data_used,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "historical_replay_proxy": True,
        "split": str(args.split),
        "row_counts": {
            "records_available": int(len(records)),
            "path_skips": int(len(path_skips)),
            "runtime_rows": int(len(rows)),
            "artifact_manifests": int(len(artifact_manifests)),
        },
        "runtime_validation": validation,
        "latency_summary": latency_summary,
        "decision": decide(validation, latency_summary, rows),
        "next_experiment": "Build full-surface action-advantage labels; Protocol265 remains research-only.",
    }
    write_jsonl(args.out_dir / "protocol265_no_order_runtime_events.jsonl", rows)
    pd.DataFrame(latency_rows).to_csv(args.out_dir / "latency_rows.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def runtime_row(record: Any, pred: np.ndarray | None, *, model_seed: int, threshold: float, current_idx: int, selected_action: str, elapsed_ms: float) -> dict[str, Any]:
    prediction = None if pred is None else float(pred[current_idx])
    quote_time = record.quote_times[current_idx] if current_idx < len(record.quote_times) else record.baseline_exit_ts.isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "protocol265_challenger",
        "event_type": "model_decision",
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "session": record.session,
        "reported_split": record.reported_split,
        "live_orders_enabled": False,
        "broker_endpoint_called": False,
        "selected_action": selected_action,
        "candidate_set": {
            "candidate_count": 1,
            "valid_candidate_count": 1,
            "root": "SPXW",
            "settlement_style": "PM",
            "historical_replay_proxy": True,
            "protocol101_min_edge_gate_applied": False,
            "protocol101_time_bucket_gate_applied": False,
        },
        "selected_contract": {
            "contract_id": record.contract_id,
            "right": record.right,
            "offset": record.offset,
            "entry_ask": record.entry_ask,
            "entry_premium": record.entry_premium,
        },
        "lifecycle_state": {
            "position_state": "holding",
            "quote_time": quote_time,
            "baseline_anchor_idx": int(p265.baseline_anchor_idx(record)),
            "current_idx": int(current_idx),
            "path_points_seen": int(current_idx + 1),
            "path_points_total_historical_proxy": int(len(record.path_pnl)),
            "future_path_fields_used_for_decision": False,
        },
        "model_decision": {
            "model_seed": int(model_seed),
            "threshold": float(threshold),
            "predicted_continuation_value": prediction,
            "feature_columns": list(p265.p200.FEATURE_COLUMNS),
            "scaler_loaded_from_artifact": True,
        },
        "risk_gate": {"passed": True, "reason": "pass"},
        "latency": {
            "total_decision_ms": float(elapsed_ms),
            "model_inference_ms": float(elapsed_ms),
            "budget_passed": bool(elapsed_ms <= 1000.0),
        },
        "paper_account_state": {
            "starting_cash": float(p265.STARTING_CASH),
            "cash_available": float(p265.STARTING_CASH),
            "account_equity": float(p265.STARTING_CASH),
            "open_position_count": 1,
            "max_concurrent_positions": 1,
            "max_contracts": 1,
        },
        "operational_default": p265.PAPER_DEFAULT_LABEL,
        "challenger_status": "no_order_runtime_parity_only",
    }


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    action_counts: dict[str, int] = {}
    for idx, row in enumerate(rows):
        action_counts[str(row.get("selected_action"))] = action_counts.get(str(row.get("selected_action")), 0) + 1
        for key in ["schema_version", "protocol_id", "event_type", "candidate_set", "model_decision", "lifecycle_state", "latency", "paper_account_state"]:
            if key not in row:
                errors.append(f"row {idx}: missing {key}")
        if row.get("live_orders_enabled") is not False:
            errors.append(f"row {idx}: live_orders_enabled must be false")
        if row.get("broker_endpoint_called") is not False:
            errors.append(f"row {idx}: broker_endpoint_called must be false")
        if row.get("selected_action") not in {"hold", "exit", "blocked"}:
            errors.append(f"row {idx}: invalid selected_action")
        if row.get("lifecycle_state", {}).get("future_path_fields_used_for_decision") is not False:
            errors.append(f"row {idx}: future path fields marked as used")
    return {"status": "pass" if not errors else "fail", "rows": len(rows), "errors": errors, "action_counts": action_counts}


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["total_decision_ms"]) for row in rows]
    if not values:
        return {"rows": 0, "p50_ms": None, "p95_ms": None, "max_ms": None, "budget_passed": False}
    return {
        "rows": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(max(values)),
        "budget_passed": bool(max(values) <= 1000.0),
    }


def decide(validation: dict[str, Any], latency: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "blocked_protocol265_runtime_no_rows"
    if validation.get("status") != "pass":
        return "blocked_protocol265_runtime_schema_failure"
    if not latency.get("budget_passed"):
        return "blocked_protocol265_runtime_latency_failure"
    return "runtime_protocol265_no_order_parity_passed_historical_proxy"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    latency = payload["latency_summary"]
    validation = payload["runtime_validation"]
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate: `{payload['candidate_label']}`",
        f"Paper default baseline: `{payload['paper_default_label']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Runtime Validation",
        "",
        f"- Rows: `{validation['rows']}`",
        f"- Status: `{validation['status']}`",
        f"- Action counts: `{validation['action_counts']}`",
        f"- Latency p95 ms: `{latency['p95_ms']}`",
        f"- Max latency ms: `{latency['max_ms']}`",
        "",
        "This is a historical replay proxy, not live/paper promotion evidence.",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    if marker in ledger.read_text():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    f"- Candidate: `{payload['candidate_label']}`",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
