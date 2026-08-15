"""RUNTIME_PREMIUM_LEANING_BLEND_NO_ORDER_PARITY_V1.

Historically Protocol241. This is a no-order runtime parity harness for the
frozen research challenger CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1.

It reuses the live-safe full-action candidate validation, feature tensor build,
model inference, latency logging, and historical selection match checks from the
Protocol217 runtime harness. It does not change PAPER_DEFAULT_PROTOCOL101, place
orders, call broker endpoints, or download paid data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import v4.scripts.run_protocol217_full_action_history_runtime_parity as p217


ROLE_LABEL = "RUNTIME_PREMIUM_LEANING_BLEND_NO_ORDER_PARITY_V1"
HISTORICAL_ID = "Protocol241"
SCHEMA_VERSION = "premium_leaning_blend_runtime_v1"
PROTOCOL_ID = "challenger_premium_leaning_blended_utility_v1"
DEFAULT_DATASET = p217.DEFAULT_DATASET
DEFAULT_ARTIFACT_MANIFEST = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_238_premium_leaning_blend_seed_stability/"
    "model_artifacts/fold4_train_2025_validate_q1_2026_test_recent/seed_1/manifest.json"
)
DEFAULT_HISTORICAL_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_240_premium_leaning_blend_five_seed_decision/five_seed_model_trades.csv"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_241_premium_blend_runtime_parity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    parser.add_argument("--historical-trades", type=Path, default=DEFAULT_HISTORICAL_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="recent_2026")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", default="fold4_train_2025_validate_q1_2026_test_recent")
    parser.add_argument("--max-events", type=int, default=750)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p217.ROLE_LABEL = ROLE_LABEL
    p217.HISTORICAL_ID = HISTORICAL_ID
    p217.SCHEMA_VERSION = SCHEMA_VERSION
    p217.PROTOCOL_ID = PROTOCOL_ID

    artifact = p217.load_artifact(args.artifact_manifest)
    feature_audit = p217.audit_feature_columns(artifact["feature_columns"])
    frame = p217.load_replay_frame(args.dataset, split=str(args.split), feature_columns=artifact["feature_columns"])
    events = p217.select_events(frame, max_events=int(args.max_events))
    rows, latency_rows, selected_trades = p217.run_replay(events, artifact=artifact, starting_cash=float(args.starting_cash))
    validation = p217.validate_runtime_stream(rows)
    latency_summary = p217.summarize_latency(latency_rows)
    historical_match = p217.compare_historical_trades(
        selected_trades,
        args.historical_trades,
        split=str(args.split),
        fold=str(args.fold),
        seed=int(args.seed),
        replayed_event_keys=p217.event_keys(events),
    )
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / no-order parity harness",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "other_baseline_label": "historical Protocol240 selected trades for the same fold/seed/split",
        "data_used": {
            "dataset": str(args.dataset),
            "artifact_manifest": str(args.artifact_manifest),
            "historical_trades": str(args.historical_trades),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "split": str(args.split),
        "seed": int(args.seed),
        "fold": str(args.fold),
        "events_requested": int(args.max_events),
        "events_replayed": int(len(events)),
        "runtime_validation": validation,
        "latency_summary": latency_summary,
        "feature_audit": feature_audit,
        "historical_trade_match": historical_match,
        "important_parity_note": (
            "The runtime selection path uses only candidate fields and model features available at decision time. "
            "candidate_exit_dt and candidate_pnl are used only after a hypothetical no-order entry to advance the "
            "historical replay clock and compare against frozen historical artifacts."
        ),
        "decision": p217.decide(validation, latency_summary, feature_audit, historical_match),
        "next_experiment": (
            "Run attribution versus PAPER_DEFAULT_PROTOCOL101, then require live-style/no-order feature parity before "
            "any paper-default replacement decision."
        ),
    }
    p217.write_jsonl(args.out_dir / "runtime_events.jsonl", rows)
    pd.DataFrame(latency_rows).to_csv(args.out_dir / "latency_rows.csv", index=False)
    pd.DataFrame(selected_trades).to_csv(args.out_dir / "selected_no_order_trades.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    p217.write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
