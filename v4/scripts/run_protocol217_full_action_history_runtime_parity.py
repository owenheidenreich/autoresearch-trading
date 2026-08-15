"""RUNTIME_FULL_ACTION_HISTORY_NO_ORDER_PARITY_V1.

Historically Protocol217. This harness checks whether the confirmed research
challenger can be evaluated in a live-style, no-order runtime path using the
same full-action candidate surface and repaired causal-history features.

It does not change PAPER_DEFAULT_PROTOCOL101. It does not download market data,
place orders, or call broker endpoints.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.live.protocol166_parity_contract import validate_candidate
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol165_full_action_space_policy import MAX_ACTION_CANDIDATES
from v4.scripts.run_protocol183_two_stage_full_action_policy import TwoStageFullActionPolicy


ROLE_LABEL = "RUNTIME_FULL_ACTION_HISTORY_NO_ORDER_PARITY_V1"
HISTORICAL_ID = "Protocol217"
SCHEMA_VERSION = "full_action_history_runtime_v1"
PROTOCOL_ID = "challenger_full_action_surface_edge_history_v1"
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_ARTIFACT_MANIFEST = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_213_full_action_surface_edge_history_policy_screen/"
    "model_artifacts/fold4_train_2025_validate_q1_2026_test_recent/seed_1/manifest.json"
)
DEFAULT_HISTORICAL_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_215_full_action_surface_edge_history_5seed_confirmation/model_trades_5seed.csv"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_217_full_action_history_runtime_parity")
CONTRACT_MULTIPLIER = 100.0
FORBIDDEN_FEATURE_TOKENS = ("candidate_pnl", "candidate_exit", "path_", "label_", "future_", "exit_bid", "exit_ask")


class RuntimeBudget:
    max_total_decision_ms = 1_000.0
    max_candidate_validation_ms = 250.0
    max_feature_build_ms = 250.0
    max_model_inference_ms = 250.0


BASE_COLUMNS = [
    "split",
    "session",
    "decision_time",
    "decision_dt",
    "candidate_uid",
    "contract_id",
    "root",
    "settlement_style",
    "right",
    "offset",
    "entry_quote_time",
    "entry_bid",
    "entry_ask",
    "entry_mid",
    "entry_spread",
    "entry_bid_size",
    "entry_ask_size",
    "entry_premium",
    "entry_delta",
    "entry_gamma",
    "entry_theta",
    "entry_iv",
    "entry_underlying_price",
    "entry_affordable_10k",
    "candidate_exit_dt",
    "candidate_pnl",
]


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
    artifact = load_artifact(args.artifact_manifest)
    feature_audit = audit_feature_columns(artifact["feature_columns"])
    frame = load_replay_frame(args.dataset, split=str(args.split), feature_columns=artifact["feature_columns"])
    events = select_events(frame, max_events=int(args.max_events))
    rows, latency_rows, selected_trades = run_replay(events, artifact=artifact, starting_cash=float(args.starting_cash))
    validation = validate_runtime_stream(rows)
    latency_summary = summarize_latency(latency_rows)
    historical_match = compare_historical_trades(
        selected_trades,
        args.historical_trades,
        split=str(args.split),
        fold=str(args.fold),
        seed=int(args.seed),
        replayed_event_keys=event_keys(events),
    )
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / no-order parity harness",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "other_baseline_label": "historical Protocol215 selected trades for the same fold/seed/split",
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
        "decision": decide(validation, latency_summary, feature_audit, historical_match),
        "next_experiment": (
            "Inspect selected-trade charts and resolve any runtime/historical mismatch before considering a paper-default replacement."
        ),
    }
    write_jsonl(args.out_dir / "runtime_events.jsonl", rows)
    pd.DataFrame(latency_rows).to_csv(args.out_dir / "latency_rows.csv", index=False)
    pd.DataFrame(selected_trades).to_csv(args.out_dir / "selected_no_order_trades.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_artifact(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    feature_columns = list(manifest["feature_columns"])
    config = manifest.get("config", {})
    model = TwoStageFullActionPolicy(input_dim=len(feature_columns), hidden_dim=int(config.get("hidden_dim", 96)))
    model.load_state_dict(torch.load(manifest_path.parent / "model.pt", map_location="cpu"))
    model.eval()
    scaler_payload = json.loads((manifest_path.parent / "scaler.json").read_text())
    scaler = FeatureScaler(
        fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
        mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    threshold = float(manifest.get("threshold_selection", {}).get("threshold", math.inf))
    return {"manifest": manifest, "model": model, "scaler": scaler, "feature_columns": feature_columns, "threshold": threshold}


def audit_feature_columns(feature_columns: list[str]) -> dict[str, Any]:
    forbidden = [column for column in feature_columns if any(token in column for token in FORBIDDEN_FEATURE_TOKENS)]
    required_history = [
        "hist_events_seen",
        "hist_minutes_since_prev_event",
        "hist_prev_candidate_count",
        "hist_prev_max_gamma",
        "hist_prev_mean_theta_burden",
        "hist_prev_min_spread_over_mid",
        "hist_prev_call_count",
        "hist_prev_put_count",
        "hist_prev_call_minus_put_edge",
        "hist_roll3_candidate_count_mean",
        "hist_roll3_max_gamma",
        "hist_roll3_mean_theta_burden",
        "hist_roll3_min_spread_over_mid",
        "hist_roll3_call_minus_put_edge",
    ]
    missing_history = [column for column in required_history if column not in feature_columns]
    return {
        "feature_count": int(len(feature_columns)),
        "forbidden_future_feature_columns": forbidden,
        "missing_required_history_features": missing_history,
        "status": "pass" if not forbidden and not missing_history else "fail",
    }


def load_replay_frame(path: Path, *, split: str, feature_columns: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(BASE_COLUMNS + feature_columns))
    try:
        frame = pd.read_parquet(path, columns=columns, filters=[("split", "==", split)])
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["split"].astype(str).eq(split)].copy()
    if frame.empty:
        raise ValueError(f"no rows found for split {split}")
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["candidate_exit_dt"] = pd.to_datetime(frame["candidate_exit_dt"], utc=True, errors="coerce")
    frame["entry_quote_dt"] = pd.to_datetime(frame["entry_quote_time"], utc=True, errors="coerce")
    return frame.sort_values(["session", "decision_dt", "contract_id"]).reset_index(drop=True)


def select_events(frame: pd.DataFrame, *, max_events: int) -> list[pd.DataFrame]:
    events = []
    for _, group in frame.groupby(["session", "decision_dt"], sort=True):
        events.append(group.head(MAX_ACTION_CANDIDATES).copy().reset_index(drop=True))
        if len(events) >= max_events:
            break
    return events


def run_replay(events: list[pd.DataFrame], *, artifact: dict[str, Any], starting_cash: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    selected_trades: list[dict[str, Any]] = []
    equity = float(starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    for event in events:
        session = str(event["session"].iloc[0])
        decision_dt = pd.Timestamp(event["decision_dt"].iloc[0])
        if open_until.get(session) is not None and decision_dt < open_until[session]:
            continue
        account = {
            "starting_cash": float(starting_cash),
            "account_equity": float(equity),
            "cash_available": float(equity),
            "open_position_count": 0,
            "max_concurrent_positions": 1,
            "max_contracts": 1,
        }
        perf_start = time.perf_counter()
        mask, validation_ms = timed_call(live_safe_candidate_mask, event, account_state=account)
        tensor, feature_ms = timed_call(build_feature_tensor, event, mask, artifact["feature_columns"], artifact["scaler"])
        output, inference_ms = timed_call(run_model, artifact["model"], tensor)
        total_ms = (time.perf_counter() - perf_start) * 1000.0
        candidate_idx, score = choose_candidate(output, mask)
        latency = {
            "candidate_validation_ms": float(validation_ms),
            "feature_build_ms": float(feature_ms),
            "model_inference_ms": float(inference_ms),
            "total_decision_ms": float(total_ms),
            "budget_passed": budget_passed(validation_ms, feature_ms, inference_ms, total_ms),
        }
        selected_action = "wait"
        selected_contract = None
        risk_gate = {"passed": True, "reason": "pass"}
        if not bool(mask.any()):
            selected_action = "blocked"
            risk_gate = {"passed": False, "reason": "no_valid_live_candidates"}
        elif not latency["budget_passed"]:
            selected_action = "blocked"
            risk_gate = {"passed": False, "reason": "latency_budget_failed"}
        elif candidate_idx is not None and score >= float(artifact["threshold"]):
            selected_row = event.iloc[int(candidate_idx)]
            selected_action = "enter"
            selected_contract = candidate_from_row(selected_row)
            pnl = finite(selected_row.get("candidate_pnl"))
            equity += pnl
            open_until[session] = pd.Timestamp(selected_row["candidate_exit_dt"])
            selected_trades.append(
                {
                    "session": session,
                    "decision_time": decision_dt.isoformat(),
                    "exit_time": pd.Timestamp(selected_row["candidate_exit_dt"]).isoformat(),
                    "contract_id": str(selected_row["contract_id"]),
                    "candidate_uid": str(selected_row.get("candidate_uid", "")),
                    "right": str(selected_row["right"]),
                    "offset": finite(selected_row.get("offset")),
                    "score": float(score),
                    "threshold": float(artifact["threshold"]),
                    "pnl": pnl,
                    "account_equity_after": float(equity),
                }
            )
        row = build_event(
            session=session,
            timestamp=decision_dt.isoformat(),
            selected_action=selected_action,
            candidate_set=candidate_set_summary(event, mask),
            model_decision={
                "score": float(score),
                "threshold": float(artifact["threshold"]),
                "candidate_index": None if candidate_idx is None else int(candidate_idx),
                "feature_count": int(len(artifact["feature_columns"])),
                "future_exit_fields_used_for_selection": False,
            },
            latency=latency,
            risk_gate=risk_gate,
            paper_account_state=account,
            selected_contract=selected_contract,
        )
        rows.append(row)
        latency_rows.append(
            {
                "session": session,
                "decision_time": decision_dt.isoformat(),
                "selected_action": selected_action,
                "candidate_count": len(event),
                "valid_candidate_count": int(mask.sum()),
                "score": float(score),
                "threshold": float(artifact["threshold"]),
                **latency,
            }
        )
    return rows, latency_rows, selected_trades


def live_safe_candidate_mask(event: pd.DataFrame, *, account_state: dict[str, Any]) -> np.ndarray:
    mask = []
    cash = finite(account_state.get("cash_available"))
    for _, row in event.iterrows():
        candidate = candidate_from_row(row)
        result = validate_candidate(candidate, account_state)
        affordable = finite(candidate.get("entry_premium"), math.inf) <= cash
        fixed_train_affordable = finite(row.get("entry_affordable_10k"), 0.0) >= 1.0
        valid = result["status"] == "pass" and affordable and fixed_train_affordable
        mask.append(bool(valid))
    return np.asarray(mask, dtype=bool)


def candidate_from_row(row: pd.Series) -> dict[str, Any]:
    bid = finite(row.get("entry_bid"), math.nan)
    ask = finite(row.get("entry_ask"), math.nan)
    mid = finite(row.get("entry_mid"), (bid + ask) / 2.0 if math.isfinite(bid) and math.isfinite(ask) else math.nan)
    spread = finite(row.get("entry_spread"), ask - bid if math.isfinite(bid) and math.isfinite(ask) else math.nan)
    return {
        "decision_time": str(row.get("decision_time", "")),
        "contract_id": str(row.get("contract_id", "")),
        "root": str(row.get("root", "")),
        "settlement_style": str(row.get("settlement_style", "")),
        "right": str(row.get("right", "")),
        "offset": finite(row.get("offset"), math.nan),
        "entry_bid": bid,
        "entry_ask": ask,
        "entry_mid": mid,
        "entry_spread": spread,
        "entry_bid_size": finite(row.get("entry_bid_size"), math.nan),
        "entry_ask_size": finite(row.get("entry_ask_size"), math.nan),
        "entry_premium": finite(row.get("entry_premium"), ask * CONTRACT_MULTIPLIER if math.isfinite(ask) else math.nan),
        "entry_delta": finite(row.get("entry_delta"), math.nan),
        "entry_gamma": finite(row.get("entry_gamma"), math.nan),
        "entry_theta": finite(row.get("entry_theta"), math.nan),
        "entry_iv": finite(row.get("entry_iv"), math.nan),
    }


def build_feature_tensor(event: pd.DataFrame, mask: np.ndarray, feature_columns: list[str], scaler: FeatureScaler) -> tuple[torch.Tensor, torch.Tensor]:
    raw = event.head(MAX_ACTION_CANDIDATES)[feature_columns].to_numpy(dtype=np.float32)
    n = len(raw)
    x = np.zeros((1, MAX_ACTION_CANDIDATES, len(feature_columns)), dtype=np.float32)
    m = np.zeros((1, MAX_ACTION_CANDIDATES), dtype=bool)
    x[0, :n, :] = scaler.transform(raw)
    m[0, : min(n, len(mask))] = mask[:n]
    return torch.from_numpy(x), torch.from_numpy(m)


def run_model(model: TwoStageFullActionPolicy, tensors: tuple[torch.Tensor, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    x, mask = tensors
    with torch.no_grad():
        event_logit, candidate_scores = model(x, mask)
    return event_logit.detach().cpu().numpy(), candidate_scores.detach().cpu().numpy()


def choose_candidate(model_output: tuple[np.ndarray, np.ndarray], mask: np.ndarray) -> tuple[int | None, float]:
    event_logit, candidate_scores = model_output
    score = float(event_logit[0])
    if not bool(mask.any()):
        return None, score
    scores = np.asarray(candidate_scores[0], dtype=float)
    padded = np.zeros(len(scores), dtype=bool)
    padded[: min(len(mask), len(scores))] = mask[: min(len(mask), len(scores))]
    valid_scores = np.where(padded, scores, -np.inf)
    idx = int(np.argmax(valid_scores))
    return (idx, score) if np.isfinite(valid_scores[idx]) else (None, score)


def candidate_set_summary(event: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    quote_age_ms = (pd.to_datetime(event["decision_dt"], utc=True) - pd.to_datetime(event["entry_quote_dt"], utc=True)).dt.total_seconds().mul(1000.0)
    return {
        "candidate_count": int(len(event)),
        "valid_candidate_count": int(mask.sum()),
        "root": "SPXW" if event["root"].astype(str).eq("SPXW").all() else "mixed",
        "settlement_style": "PM" if event["settlement_style"].astype(str).eq("PM").all() else "mixed",
        "max_abs_offset": finite(pd.to_numeric(event["offset"], errors="coerce").abs().max()),
        "call_count": int(event["right"].astype(str).eq("C").sum()),
        "put_count": int(event["right"].astype(str).eq("P").sum()),
        "max_option_quote_age_ms": finite(quote_age_ms.max()),
        "protocol101_min_edge_gate_applied": False,
        "protocol101_time_bucket_gate_applied": False,
        "historical_replay_proxy": True,
    }


def build_event(**kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "event_type": "model_decision",
        "live_orders_enabled": False,
        "broker_endpoint_called": False,
        "operational_default": "PAPER_DEFAULT_PROTOCOL101",
        "challenger_status": "no_order_runtime_parity_only",
        **kwargs,
    }


def validate_runtime_stream(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for idx, row in enumerate(rows):
        for key in ["schema_version", "protocol_id", "event_type", "session", "timestamp", "selected_action", "candidate_set", "model_decision", "latency", "risk_gate", "paper_account_state"]:
            if key not in row:
                errors.append(f"row {idx}: missing {key}")
        if row.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"row {idx}: bad schema_version")
        if row.get("protocol_id") != PROTOCOL_ID:
            errors.append(f"row {idx}: bad protocol_id")
        if row.get("live_orders_enabled") is not False:
            errors.append(f"row {idx}: live_orders_enabled must be false")
        if row.get("broker_endpoint_called") is not False:
            errors.append(f"row {idx}: broker_endpoint_called must be false")
        if row.get("selected_action") not in {"enter", "wait", "blocked"}:
            errors.append(f"row {idx}: invalid selected_action")
        if row.get("latency", {}).get("budget_passed") is not True:
            warnings.append(f"row {idx}: latency budget did not pass")
        candidate_set = row.get("candidate_set", {})
        if candidate_set.get("root") != "SPXW":
            errors.append(f"row {idx}: candidate root is not SPXW")
        if candidate_set.get("settlement_style") != "PM":
            errors.append(f"row {idx}: candidate settlement is not PM")
        if row.get("selected_action") == "enter" and not isinstance(row.get("selected_contract"), dict):
            errors.append(f"row {idx}: enter without selected_contract")
    counts = pd.Series([row.get("selected_action", "missing") for row in rows]).value_counts().to_dict() if rows else {}
    return {"status": "pass" if not errors else "fail", "rows": int(len(rows)), "errors": errors, "warnings": warnings, "action_counts": counts}


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {"rows": int(len(frame)), "action_counts": frame["selected_action"].value_counts().to_dict()}
    for column in ["candidate_validation_ms", "feature_build_ms", "model_inference_ms", "total_decision_ms"]:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        out[column] = {
            "p50": float(values.quantile(0.50)) if len(values) else None,
            "p95": float(values.quantile(0.95)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    out["budget_pass_fraction"] = float(frame["budget_passed"].mean()) if len(frame) else 0.0
    out["max_valid_candidates"] = int(pd.to_numeric(frame["valid_candidate_count"], errors="coerce").max()) if len(frame) else 0
    return out


def compare_historical_trades(
    selected: list[dict[str, Any]],
    path: Path,
    *,
    split: str,
    fold: str,
    seed: int,
    replayed_event_keys: set[str],
) -> dict[str, Any]:
    selected_frame = pd.DataFrame(selected)
    if selected_frame.empty:
        return {"status": "no_runtime_selected_trades", "runtime_trades": 0}
    selected_frame["key"] = selected_frame["session"].astype(str) + "|" + selected_frame["decision_time"].astype(str) + "|" + selected_frame["contract_id"].astype(str)
    historical = pd.read_csv(path)
    historical = historical[
        historical["reported_split"].astype(str).eq(split)
        & historical["fold"].astype(str).eq(fold)
        & (pd.to_numeric(historical["seed"], errors="coerce").fillna(-1).astype(int) == int(seed))
    ].copy()
    historical["event_key"] = historical["session"].astype(str) + "|" + pd.to_datetime(historical["decision_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    historical = historical[historical["event_key"].isin(replayed_event_keys)].copy()
    historical["key"] = historical["event_key"] + "|" + historical["contract_id"].astype(str)
    selected_keys = set(selected_frame["key"])
    historical_keys = set(historical["key"])
    common = selected_keys & historical_keys
    return {
        "status": "pass" if len(common) == len(selected_keys) == len(historical_keys) else "mismatch",
        "runtime_trades": int(len(selected_keys)),
        "historical_trades": int(len(historical_keys)),
        "replayed_event_count": int(len(replayed_event_keys)),
        "common_exact_trades": int(len(common)),
        "runtime_only_trades": int(len(selected_keys - historical_keys)),
        "historical_only_trades": int(len(historical_keys - selected_keys)),
        "match_fraction_vs_runtime": float(len(common) / max(len(selected_keys), 1)),
        "match_fraction_vs_historical": float(len(common) / max(len(historical_keys), 1)),
    }


def event_keys(events: list[pd.DataFrame]) -> set[str]:
    keys: set[str] = set()
    for event in events:
        session = str(event["session"].iloc[0])
        decision = pd.Timestamp(event["decision_dt"].iloc[0]).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        keys.add(f"{session}|{decision}")
    return keys


def budget_passed(validation_ms: float, feature_ms: float, inference_ms: float, total_ms: float) -> bool:
    return (
        validation_ms <= RuntimeBudget.max_candidate_validation_ms
        and feature_ms <= RuntimeBudget.max_feature_build_ms
        and inference_ms <= RuntimeBudget.max_model_inference_ms
        and total_ms <= RuntimeBudget.max_total_decision_ms
    )


def decide(validation: dict[str, Any], latency: dict[str, Any], feature_audit: dict[str, Any], historical_match: dict[str, Any]) -> str:
    if validation.get("status") != "pass":
        return "blocked_runtime_schema_validation_failed"
    if feature_audit.get("status") != "pass":
        return "blocked_feature_audit_failed"
    if float(latency.get("budget_pass_fraction", 0.0)) < 0.99:
        return "blocked_latency_budget_failed"
    if historical_match.get("status") != "pass":
        return "blocked_runtime_historical_selection_mismatch"
    return "runtime_parity_ready_no_order_protocol101_default_unchanged"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    latency = payload["latency_summary"]
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Other baseline: {payload['other_baseline_label']}",
        f"Data used: {payload['data_used']['dataset']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        f"Parity note: {payload['important_parity_note']}",
        "",
        "## Runtime Summary",
        "",
        f"- Events replayed: `{payload['events_replayed']}`",
        f"- Runtime validation: `{payload['runtime_validation']['status']}`",
        f"- Feature audit: `{payload['feature_audit']['status']}`",
        f"- Historical selection match: `{payload['historical_trade_match']['status']}`",
        f"- Action counts: `{latency.get('action_counts')}`",
        f"- Budget pass fraction: `{latency.get('budget_pass_fraction')}`",
        "",
        "## Latency",
        "",
        "| metric | p50 ms | p95 ms | max ms |",
        "|---|---:|---:|---:|",
    ]
    for metric in ["candidate_validation_ms", "feature_build_ms", "model_inference_ms", "total_decision_ms"]:
        item = latency.get(metric, {})
        lines.append(f"| {metric} | {fmt(item.get('p50'))} | {fmt(item.get('p95'))} | {fmt(item.get('max'))} |")
    lines.extend(
        [
            "",
            "## Historical Match",
            "",
            f"- Runtime trades: `{payload['historical_trade_match'].get('runtime_trades')}`",
            f"- Historical trades: `{payload['historical_trade_match'].get('historical_trades')}`",
            f"- Common exact trades: `{payload['historical_trade_match'].get('common_exact_trades')}`",
            f"- Runtime-only trades: `{payload['historical_trade_match'].get('runtime_only_trades')}`",
            f"- Historical-only trades: `{payload['historical_trade_match'].get('historical_only_trades')}`",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Runtime JSONL: `{path.parent / 'runtime_events.jsonl'}`",
            f"- Latency rows: `{path.parent / 'latency_rows.csv'}`",
            f"- Selected no-order trades: `{path.parent / 'selected_no_order_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000.0


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
