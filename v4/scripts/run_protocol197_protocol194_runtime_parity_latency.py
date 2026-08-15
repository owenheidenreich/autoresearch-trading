"""Protocol197: Protocol194 no-order runtime parity and latency harness.

Protocol101 remains the live-paper default. Protocol194 is the strongest
research challenger, so the next gate is not another entry-side model knob; it
is proving that a future runtime can evaluate the full-action candidate surface
quickly, safely, and without future-only fields.

This runner uses already-collected historical rows as a replay source. It does
not download market data, submit orders, or call broker endpoints.
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

from v4.live.protocol194_runtime import (
    Protocol194LatencyBudget,
    build_runtime_event,
    candidate_from_row,
    latency_passed,
    live_candidate_mask,
    now_iso,
    timed_call,
    validate_runtime_stream,
)
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol183_two_stage_full_action_policy import TwoStageFullActionPolicy
from v4.scripts.run_protocol165_full_action_space_policy import MAX_ACTION_CANDIDATES


LOOP_ID = "v4_aplus_hypothesis_197_protocol194_runtime_parity_latency"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_189_full_coverage_surface_edge_enrichment/"
    "protocol185_full_action_with_surface_edge.parquet"
)
DEFAULT_ARTIFACT = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_190_full_coverage_surface_edge_baseline_exit_screen/"
    "model_artifacts/fold4_train_2025_validate_q1_2026_test_recent/seed_1/manifest.json"
)
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
    "market_spx_close",
    "market_vix_close",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="recent_2026")
    parser.add_argument("--max-events", type=int, default=250)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_artifact(args.artifact_manifest)
    frame = load_replay_frame(args.dataset, split=str(args.split), feature_columns=artifact["feature_columns"])
    events = select_events(frame, max_events=int(args.max_events))
    rows, latency_rows = run_replay(events, artifact=artifact, starting_cash=float(args.starting_cash))
    validation = validate_runtime_stream(rows)
    latency_summary = summarize_latency(latency_rows)
    payload = {
        "protocol": "197_protocol194_runtime_parity_latency",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "operational_live_paper_default": "protocol101",
        "challenger": "protocol194",
        "source_dataset": str(args.dataset),
        "source_artifact_manifest": str(args.artifact_manifest),
        "split": str(args.split),
        "events_requested": int(args.max_events),
        "events_replayed": int(len(events)),
        "runtime_validation": validation,
        "latency_summary": latency_summary,
        "decision": decide(validation, latency_summary, len(events)),
        "important_parity_note": (
            "This harness uses a live-safe candidate mask. It does not use candidate_exit_dt, "
            "candidate_pnl, or any future path/exit label for runtime selection."
        ),
    }
    write_jsonl(args.out_dir / "protocol197_runtime_events.jsonl", rows)
    pd.DataFrame(latency_rows).to_csv(args.out_dir / "latency_rows.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_artifact(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    feature_columns = list(manifest["feature_columns"])
    config = manifest.get("config", {})
    model = TwoStageFullActionPolicy(input_dim=len(feature_columns), hidden_dim=int(config.get("hidden_dim", 96)))
    state_path = manifest_path.parent / "model.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    scaler_payload = json.loads((manifest_path.parent / "scaler.json").read_text())
    scaler = FeatureScaler(
        fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
        mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    threshold = float(manifest.get("threshold_selection", {}).get("threshold", math.inf))
    return {
        "manifest": manifest,
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "threshold": threshold,
    }


def load_replay_frame(path: Path, *, split: str, feature_columns: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(BASE_COLUMNS + feature_columns))
    try:
        frame = pd.read_parquet(path, columns=columns, filters=[("split", "==", split)])
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["split"].astype(str).eq(split)].copy()
    if frame.empty:
        raise ValueError(f"no rows found for split {split} in {path}")
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["entry_quote_dt"] = pd.to_datetime(frame["entry_quote_time"], utc=True, errors="coerce")
    return frame.sort_values(["session", "decision_dt", "contract_id"]).reset_index(drop=True)


def select_events(frame: pd.DataFrame, *, max_events: int) -> list[pd.DataFrame]:
    events = []
    for _, group in frame.groupby(["session", "decision_time"], sort=True):
        events.append(group.head(MAX_ACTION_CANDIDATES).copy())
        if len(events) >= max_events:
            break
    return events


def run_replay(events: list[pd.DataFrame], *, artifact: dict[str, Any], starting_cash: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    budget = Protocol194LatencyBudget()
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    account = {
        "starting_cash": float(starting_cash),
        "account_equity": float(starting_cash),
        "cash_available": float(starting_cash),
        "open_positions": 0,
        "open_position_count": 0,
        "max_concurrent_positions": 1,
        "max_contracts": 1,
    }
    for event in events:
        perf_start = time.perf_counter()
        mask, validation_ms = timed_call(live_candidate_mask, event, account_state=account)
        feature_tensor, feature_ms = timed_call(build_feature_tensor, event, mask, artifact["feature_columns"], artifact["scaler"])
        model_output, inference_ms = timed_call(run_model, artifact["model"], feature_tensor)
        total_ms = (time.perf_counter() - perf_start) * 1000.0
        candidate_idx, score = choose_candidate(model_output, mask)
        selected_action = "wait"
        selected_contract = None
        risk_gate = {"passed": True, "reason": "pass"}
        if not bool(mask.any()):
            selected_action = "blocked"
            risk_gate = {"passed": False, "reason": "no_valid_live_candidates"}
        elif not latency_passed(
            {
                "candidate_validation_ms": validation_ms,
                "model_inference_ms": inference_ms,
                "total_decision_ms": total_ms,
            },
            budget,
        ):
            selected_action = "blocked"
            risk_gate = {"passed": False, "reason": "latency_budget_failed"}
        elif score >= float(artifact["threshold"]) and candidate_idx is not None:
            selected_action = "enter"
            selected_contract = candidate_from_row(event.iloc[int(candidate_idx)])

        latency = {
            "candidate_validation_ms": float(validation_ms),
            "feature_build_ms": float(feature_ms),
            "model_inference_ms": float(inference_ms),
            "total_decision_ms": float(total_ms),
            "budget_passed": bool(
                latency_passed(
                    {
                        "candidate_validation_ms": validation_ms,
                        "model_inference_ms": inference_ms,
                        "total_decision_ms": total_ms,
                    },
                    budget,
                )
            ),
            "budget": budget.__dict__,
        }
        candidate_set = candidate_set_summary(event, mask)
        model_decision = {
            "score": float(score),
            "threshold": float(artifact["threshold"]),
            "candidate_index": None if candidate_idx is None else int(candidate_idx),
            "features_used": list(artifact["feature_columns"]),
            "live_safe_mask": True,
            "future_exit_fields_used": False,
        }
        row = build_runtime_event(
            session=str(event["session"].iloc[0]),
            timestamp=now_iso(),
            selected_action=selected_action,
            candidate_set=candidate_set,
            model_decision=model_decision,
            latency=latency,
            risk_gate=risk_gate,
            paper_account_state=account,
            selected_contract=selected_contract,
        )
        rows.append(row)
        latency_rows.append(
            {
                "session": str(event["session"].iloc[0]),
                "decision_time": str(event["decision_time"].iloc[0]),
                "selected_action": selected_action,
                "candidate_count": int(candidate_set["candidate_count"]),
                "valid_candidate_count": int(candidate_set["valid_candidate_count"]),
                "score": float(score),
                "threshold": float(artifact["threshold"]),
                **{key: latency[key] for key in ("candidate_validation_ms", "feature_build_ms", "model_inference_ms", "total_decision_ms", "budget_passed")},
            }
        )
    return rows, latency_rows


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
    padded_mask = np.zeros(len(scores), dtype=bool)
    padded_mask[: min(len(mask), len(scores))] = mask[: min(len(mask), len(scores))]
    valid_scores = np.where(padded_mask, scores, -np.inf)
    idx = int(np.argmax(valid_scores))
    if not np.isfinite(valid_scores[idx]):
        return None, score
    return idx, score


def candidate_set_summary(event: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    quote_age_ms = (
        (pd.to_datetime(event["decision_dt"], utc=True) - pd.to_datetime(event["entry_quote_dt"], utc=True))
        .dt.total_seconds()
        .mul(1000.0)
    )
    return {
        "candidate_count": int(len(event)),
        "valid_candidate_count": int(mask.sum()),
        "root": "SPXW" if event["root"].astype(str).eq("SPXW").all() else "mixed",
        "settlement_style": "PM" if event["settlement_style"].astype(str).eq("PM").all() else "mixed",
        "max_abs_offset": float(pd.to_numeric(event["offset"], errors="coerce").abs().max()),
        "call_count": int(event["right"].astype(str).eq("C").sum()),
        "put_count": int(event["right"].astype(str).eq("P").sum()),
        "max_option_quote_age_ms": float(quote_age_ms.max()) if len(quote_age_ms) else None,
        "max_context_age_ms": 0.0,
        "historical_replay_proxy": True,
        "protocol101_min_edge_gate_applied": False,
        "protocol101_time_bucket_gate_applied": False,
    }


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {"rows": int(len(frame))}
    for column in ("candidate_validation_ms", "feature_build_ms", "model_inference_ms", "total_decision_ms"):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        out[column] = {
            "p50": float(values.quantile(0.50)) if not values.empty else None,
            "p95": float(values.quantile(0.95)) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
        }
    out["budget_pass_fraction"] = float(frame["budget_passed"].mean()) if len(frame) else 0.0
    out["action_counts"] = frame["selected_action"].value_counts().to_dict()
    out["max_valid_candidates"] = int(frame["valid_candidate_count"].max()) if len(frame) else 0
    return out


def decide(validation: dict[str, Any], latency: dict[str, Any], events: int) -> str:
    if events <= 0:
        return "blocked_no_replay_events"
    if validation.get("status") != "pass":
        return "blocked_runtime_schema_validation_failed"
    total_p95 = latency.get("total_decision_ms", {}).get("p95")
    inference_p95 = latency.get("model_inference_ms", {}).get("p95")
    if total_p95 is None or inference_p95 is None:
        return "blocked_missing_latency_metrics"
    if float(total_p95) > Protocol194LatencyBudget().max_total_decision_ms:
        return "blocked_runtime_latency_budget_failed"
    return "runtime_parity_latency_harness_ready_protocol101_default_unchanged"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    latency = payload["latency_summary"]
    lines = [
        "# Protocol197 Protocol194 Runtime Parity And Latency",
        "",
        "Protocol101 remains the live-paper default. Protocol194 is evaluated here only as a no-order challenger.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Events replayed: `{payload['events_replayed']}`",
        f"- Runtime schema status: `{payload['runtime_validation']['status']}`",
        f"- Live orders: `{payload['live_orders']}`",
        f"- Broker endpoint called: `{payload['broker_endpoint_called']}`",
        f"- Parity note: {payload['important_parity_note']}",
        "",
        "## Latency",
        "",
        "| metric | p50 ms | p95 ms | max ms |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("candidate_validation_ms", "feature_build_ms", "model_inference_ms", "total_decision_ms"):
        row = latency.get(metric, {})
        lines.append(
            f"| {metric} | {fmt(row.get('p50'))} | {fmt(row.get('p95'))} | {fmt(row.get('max'))} |"
        )
    lines.extend(
        [
            "",
            "## Runtime Summary",
            "",
            f"- Budget pass fraction: `{latency.get('budget_pass_fraction')}`",
            f"- Action counts: `{latency.get('action_counts')}`",
            f"- Max valid candidates: `{latency.get('max_valid_candidates')}`",
            f"- Validation errors: `{len(payload['runtime_validation'].get('errors', []))}`",
            f"- Validation warnings: `{len(payload['runtime_validation'].get('warnings', []))}`",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Runtime JSONL: `{path.parent / 'protocol197_runtime_events.jsonl'}`",
            f"- Latency rows: `{path.parent / 'latency_rows.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
