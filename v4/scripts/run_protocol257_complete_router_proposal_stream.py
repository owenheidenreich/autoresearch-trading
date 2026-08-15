"""DATA_COMPLETE_ROUTER_PROPOSAL_STREAM_V1.

Historically Protocol257. This materializes the missing proposal-stream bridge
for router experiments. It replays saved PAPER_DEFAULT_PROTOCOL101 artifacts and
saved CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1 artifacts over each
chronological fold's train/validation/test splits, then writes a fold-aware
proposal stream with `router_fold`.

This is not a model change. It does not download data or call a broker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol165_full_action_space_policy import FOLDS, STARTING_CASH, load_dataset
from v4.scripts.run_protocol172_full_action_value_policy import build_events as build_full_action_events
from v4.scripts.run_protocol097_sequential_event_policy import (
    EventPolicyConfig,
    EventSetPolicy,
    build_events as build_serial_events,
    predict_event_action,
)
import v4.scripts.run_protocol101_event_history_policy as p101
import v4.scripts.run_protocol249_entry_quality_calibrator as p249


ROLE_LABEL = "DATA_COMPLETE_ROUTER_PROPOSAL_STREAM_V1"
HISTORICAL_ID = "Protocol257"
DEFAULT_FULL_ACTION_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_SERIAL_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy/serial_opportunity_dataset.parquet")
DEFAULT_PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
DEFAULT_RECENT_ATTRIBUTION = Path("v4/audit/autoresearch/v4_aplus_hypothesis_242_premium_blend_vs_protocol101_attribution/enriched_policy_trades.csv")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_257_complete_router_proposal_stream")
MODEL_SEEDS = [1, 2, 3, 4, 5]
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-action-dataset", type=Path, default=DEFAULT_FULL_ACTION_DATASET)
    parser.add_argument("--serial-dataset", type=Path, default=DEFAULT_SERIAL_DATASET)
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--recent-attribution", type=Path, default=DEFAULT_RECENT_ATTRIBUTION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-smoke-sessions", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"stage": "load_full_action", "path": str(args.full_action_dataset)}), flush=True)
    full_action = load_dataset(args.full_action_dataset)
    p249.p183.set_active_feature_columns(p249.BASE_ARTIFACT_ROOTS and p249.load_base_artifact(FOLDS[0]["name"], int(args.seeds[0]), full_action).feature_columns, full_action)
    full_events = build_full_action_events(full_action, starting_cash=float(args.starting_cash))

    print(json.dumps({"stage": "load_serial", "path": str(args.serial_dataset)}), flush=True)
    serial = pd.read_parquet(args.serial_dataset)
    serial["decision_dt"] = pd.to_datetime(serial["decision_time"], utc=True)
    serial["candidate_exit_dt"] = pd.to_datetime(serial["candidate_exit_time"], utc=True)
    serial_events = build_serial_events(serial)
    p101.add_causal_history_features(serial_events)

    if args.smoke:
        full_events = smoke_events(full_events, int(args.max_smoke_sessions))
        serial_events = smoke_events(serial_events, int(args.max_smoke_sessions))

    protocol101_thresholds = load_protocol101_thresholds(args.protocol101_dir / "summary.json")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for fold in FOLDS:
        needed_splits = set(fold["train_splits"]) | {str(fold["validation_split"]), str(fold["test_split"])}
        if args.smoke:
            needed_splits = {full_events[0]["split"]} if full_events else needed_splits
        print(json.dumps({"stage": "fold_start", "fold": fold["name"], "splits": sorted(needed_splits)}), flush=True)
        fold_full_events = [event for event in full_events if event["split"] in needed_splits]
        fold_serial_events = [event for event in serial_events if event["split"] in needed_splits]
        for seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "fold": fold["name"], "seed": int(seed)}), flush=True)
            try:
                rows.extend(
                    premium_blend_rows(
                        fold,
                        int(seed),
                        fold_full_events,
                        full_action,
                        starting_cash=float(args.starting_cash),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report and keep other streams materializing.
                failures.append({"router_fold": fold["name"], "seed": int(seed), "policy": "challenger", "error": repr(exc)})
            try:
                rows.extend(
                    protocol101_rows(
                        fold,
                        int(seed),
                        fold_serial_events,
                        args.protocol101_dir,
                        protocol101_thresholds,
                        starting_cash=float(args.starting_cash),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"router_fold": fold["name"], "seed": int(seed), "policy": "protocol101", "error": repr(exc)})
            print(json.dumps({"stage": "seed_done", "fold": fold["name"], "seed": int(seed), "rows_so_far": len(rows)}), flush=True)

    if not args.smoke:
        rows.extend(load_recent_protocol101_rows(args.recent_attribution, router_fold="fold4_train_2025_validate_q1_2026_test_recent"))

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = normalize_output_frame(frame)
        frame.to_csv(args.out_dir / "complete_router_proposal_stream.csv", index=False)
    else:
        (args.out_dir / "complete_router_proposal_stream.csv").write_text("")
    summary = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "data / proposal-stream materialization",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "full_action_dataset": str(args.full_action_dataset),
        "serial_dataset": str(args.serial_dataset),
        "rows": int(len(frame)),
        "failures": failures,
        "summary": summarize(frame),
        "decision": "complete_router_proposal_stream_ready" if len(frame) and not failures else "complete_router_proposal_stream_has_failures",
        "next_experiment": "Run fold-aware history router on complete_router_proposal_stream.csv.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", summary)
    print(json.dumps({"decision": summary["decision"], "rows": int(len(frame)), "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def premium_blend_rows(
    fold: dict[str, Any],
    seed: int,
    events: list[dict[str, Any]],
    dataset: pd.DataFrame,
    *,
    starting_cash: float,
) -> list[dict[str, Any]]:
    if not events:
        return []
    base = p249.load_base_artifact(str(fold["name"]), int(seed), dataset)
    predictions = p249.predict_base_actions(events, base)
    sim = p249.simulate_base_from_predictions(
        events,
        predictions,
        base,
        slippage_per_side=0.0,
        starting_cash=starting_cash,
        strategy="premium_blend_complete_stream",
    )
    return [
        {
            **row,
            "router_fold": str(fold["name"]),
            "seed": int(seed),
            "reported_split": str(row.get("split", "")),
            "policy": "challenger",
            "source_artifact": str(base.artifact_dir),
        }
        for row in sim.trades
    ]


def protocol101_rows(
    fold: dict[str, Any],
    seed: int,
    events: list[dict[str, Any]],
    artifact_root: Path,
    thresholds: dict[tuple[str, int], float],
    *,
    starting_cash: float,
) -> list[dict[str, Any]]:
    if not events:
        return []
    artifact_dir = artifact_root / "model_artifacts" / str(fold["name"]) / f"seed_{seed}"
    threshold_fold = str(fold["name"])
    if not artifact_dir.exists() and str(fold["name"]) == "fold4_train_2025_validate_q1_2026_test_recent":
        # Protocol101 predates the recent-2026 fold. For fold4 proposal
        # materialization, use the latest saved paper-default artifact rather
        # than training or inventing a new Protocol101 model.
        threshold_fold = "fold3_train_q1_q2_q3_validate_q4_test_q1_2026"
        artifact_dir = artifact_root / "model_artifacts" / threshold_fold / f"seed_{seed}"
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    feature_columns = [str(column) for column in manifest["feature_columns"]]
    hidden_dim = int(manifest.get("config", {}).get("hidden_dim", 96))
    model = EventSetPolicy(input_dim=len(feature_columns), hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu"))
    model.eval()
    scaler = scaler_from_json(artifact_dir / "scaler.json")
    threshold = float(thresholds[(threshold_fold, int(seed))])

    out: list[dict[str, Any]] = []
    for _, session_events in group_serial_events([event for event in events if int(event.get("seed", seed)) == int(seed)]).items():
        equity = float(starting_cash)
        open_until: pd.Timestamp | None = None
        for event in sorted(session_events, key=lambda item: item["decision_dt"]):
            if open_until is not None and event["decision_dt"] < open_until:
                continue
            action, margin = predict_event_action(event, model, scaler, feature_columns=feature_columns)
            if action <= 0 or margin < threshold or action > len(event["candidates"]):
                continue
            row = event["candidates"].iloc[action - 1]
            entry_ask = finite(row.get("entry_ask"), 0.0)
            if entry_ask <= 0.0:
                continue
            entry_premium = entry_ask * CONTRACT_MULTIPLIER
            if entry_premium > equity:
                continue
            pnl = finite(row.get("candidate_pnl"), 0.0)
            trade = {
                "candidate_uid": str(row.get("candidate_uid", "")),
                "trade_uid": str(row.get("trade_uid", "")),
                "split": str(row.get("split", event["split"])),
                "session": str(row.get("session", event["session"])),
                "decision_time": pd.Timestamp(row.get("decision_dt", event["decision_dt"])).isoformat(),
                "exit_time": pd.Timestamp(row["candidate_exit_dt"]).isoformat(),
                "contract_id": str(row["contract_id"]),
                "right": str(row["right"]),
                "offset": float(row["offset"]),
                "score": float(margin),
                "threshold": float(threshold),
                "entry_bid": finite(row.get("entry_bid"), np.nan),
                "entry_ask": float(entry_ask),
                "entry_mid": finite(row.get("entry_mid"), np.nan),
                "entry_spread": finite(row.get("entry_spread"), np.nan),
                "entry_underlying": finite(row.get("entry_underlying_price"), np.nan),
                "entry_premium": float(entry_premium),
                "entry_premium_with_slippage": float(entry_premium),
                "account_equity_before": float(equity),
                "account_equity_after": float(equity + pnl),
                "pnl": float(pnl),
                "raw_candidate_pnl": float(pnl),
                "slippage_per_side": 0.0,
                "strategy": "protocol101_complete_stream",
                "exit_reason": str(row.get("candidate_exit_reason", "")),
                "label_source": str(row.get("label_source", "")),
                "router_fold": str(fold["name"]),
                "seed": int(seed),
                "reported_split": str(row.get("split", event["split"])),
                "policy": "protocol101",
                "source_artifact": str(artifact_dir),
            }
            out.append(trade)
            equity += pnl
            open_until = pd.Timestamp(row["candidate_exit_dt"])
    return out


def load_recent_protocol101_rows(path: Path, *, router_fold: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame[(frame["policy"] == "protocol101") & (frame["reported_split"] == "recent_2026")].copy()
    if frame.empty:
        return []
    frame["router_fold"] = router_fold
    frame["policy"] = "protocol101"
    frame["source_artifact"] = str(path)
    frame["entry_ask"] = coalesce(frame, "entry_ask", "entry_ask_live")
    frame["entry_bid"] = coalesce(frame, "entry_bid", "entry_bid_live")
    frame["entry_underlying"] = coalesce(frame, "entry_underlying", "entry_underlying_price")
    frame["entry_premium"] = frame["entry_ask"] * CONTRACT_MULTIPLIER
    frame["entry_premium_with_slippage"] = frame["entry_premium"]
    return frame.to_dict("records")


def normalize_output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["reported_split"] = out["reported_split"].fillna(out.get("split", ""))
    for column in ["entry_bid", "entry_ask", "entry_mid", "entry_spread", "entry_underlying", "entry_premium", "pnl", "raw_candidate_pnl", "score", "threshold", "offset"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["router_fold", "policy", "seed", "reported_split", "session", "decision_time", "contract_id"]).reset_index(drop=True)


def load_protocol101_thresholds(path: Path) -> dict[tuple[str, int], float]:
    payload = json.loads(path.read_text())
    out: dict[tuple[str, int], float] = {}
    for result in payload.get("fold_results", []):
        out[(str(result["fold"]), int(result["seed"]))] = float(result["threshold"])
    return out


def scaler_from_json(path: Path) -> FeatureScaler:
    payload = json.loads(path.read_text())
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def group_serial_events(events: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault((str(event["split"]), int(event["seed"]), str(event["session"])), []).append(event)
    return grouped


def smoke_events(events: list[dict[str, Any]], max_sessions: int) -> list[dict[str, Any]]:
    sessions = sorted({str(event["session"]) for event in events})[:max_sessions]
    return [event for event in events if str(event["session"]) in set(sessions)]


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    rows = []
    for (router_fold, policy, split), group in frame.groupby(["router_fold", "policy", "reported_split"], sort=True):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "router_fold": str(router_fold),
                "policy": str(policy),
                "reported_split": str(split),
                "rows": int(len(group)),
                "pnl": float(pnl.sum()),
                "win_rate": float((pnl > 0.0).mean()),
                "median_entry_premium": float(pd.to_numeric(group["entry_premium"], errors="coerce").median()),
            }
        )
    return {"by_fold_policy_split": rows}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: data / proposal-stream materialization",
        "Does it change the paper-trading default: no",
        "Candidate being tested: none",
        "Baseline it supports: PAPER_DEFAULT_PROTOCOL101 and CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1 proposal streams",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Rows: `{payload['rows']}`",
        "",
        "## Outputs",
        "",
        f"- Summary: `{path.parent / 'summary.json'}`",
        f"- Proposal stream: `{path.parent / 'complete_router_proposal_stream.csv'}`",
    ]
    if payload.get("failures"):
        lines.extend(["", "## Failures", "", json.dumps(payload["failures"], indent=2, sort_keys=True)])
    path.write_text("\n".join(lines) + "\n")


def coalesce(frame: pd.DataFrame, *columns: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        if column in frame.columns:
            out = out.combine_first(pd.to_numeric(frame[column], errors="coerce"))
    return out


def finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


if __name__ == "__main__":
    raise SystemExit(main())
