"""Replay a fair-contract candidate over historical fair-contract decision rows.

This is the historical counterpart to
``run_protocol101_fair_contract_ibkr_capture_replay``. It scores existing
protocol101-live-v1 pickle rows with a frozen fair-contract candidate and emits
Protocol101DecisionTraceV1 rows suitable for paired replay diffs.

The replay is entry-intent only: labels may be present in the historical rows
and are preserved in selected-candidate output, but labels do not affect entry
decisions, account state, daily-loss state, or later entries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.ibkr_market_capture import clean_json, stable_hash
from v4.live.protocol101_decision_trace import normalize_decision_trace
from v4.live.protocol101_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
    feature_contract_model_transform,
    feature_contract_version,
)
from v4.scripts.run_protocol101_fair_contract_ibkr_capture_replay import (
    DEFAULT_TRAINING_RESULT,
    _annotate_affordability,
    _csv_row,
    _decision_trace_identity,
    _json_feature_vector,
    _loaded_for_feature_contract,
    _selected_output,
    _source_quote_time,
    _trace_candidate_filter_payload,
    _trace_candidate_payload,
    render_report as render_capture_style_report,
    write_jsonl,
)
from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    STRICT_REPLAY_CONTRACT_MULTIPLIER,
    STRICT_REPLAY_STARTING_CASH,
    candidate_allowed_by_entry_filter,
    candidate_records_from_row,
    load_json_optional,
    load_model,
    score_candidate_records,
    top_candidate_with_margin,
    write_csv,
    _safe_float,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_fair_contract_dataset_replay")
SCHEMA_VERSION = "Protocol101FairContractDatasetReplayV1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", type=Path, nargs="+", required=True)
    parser.add_argument("--training-result", type=Path, default=DEFAULT_TRAINING_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--session", default=None)
    parser.add_argument(
        "--feature-contract",
        choices=(FEATURE_CONTRACT_VERSION, FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED),
        default=FEATURE_CONTRACT_VERSION,
        help=(
            "expected model-facing replay contract. v2 preserves raw quotes for labels/fills "
            "but masks vendor-sensitive option microstructure before scoring."
        ),
    )
    parser.add_argument("--repeat-check", action="store_true")
    return parser.parse_args()


def load_decision_rows(paths: list[Path]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"{path} expected list, got {type(payload).__name__}")
        session = path.name.removesuffix(".pkl")
        for row in payload:
            if isinstance(row, dict):
                rows.append((session, row))
    rows.sort(key=lambda item: str(item[1].get("decision_time") or ""))
    return rows


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def replay_once(
    *,
    input_paths: list[Path],
    training_result: dict[str, Any],
    run_id: str,
    session_filter: str | None,
    feature_contract: str = FEATURE_CONTRACT_VERSION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    loaded = load_model(training_result)
    contract_version = feature_contract_version(feature_contract)
    scoring_loaded = _loaded_for_feature_contract(loaded, contract_version)
    model_artifact_feature_transform = str(getattr(loaded, "feature_transform", "none") or "none")
    effective_feature_transform = str(getattr(scoring_loaded, "feature_transform", "none") or "none")
    decision_rows = load_decision_rows(input_paths)
    traces: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    all_candidate_rows = 0
    scored_candidate_rows = 0
    threshold_waits = 0
    score_ceiling_waits = 0
    score_margin_waits = 0
    entry_filter_blocks = 0
    affordability_blocks = 0
    cooldown_blocks = 0
    session_trade_cap_blocks = 0
    rows_without_candidates = 0
    wrong_contract_rows = 0
    cash = float(STRICT_REPLAY_STARTING_CASH)
    next_time_by_session: dict[str, pd.Timestamp] = {}
    trades_by_session: dict[str, int] = {}
    entry_filter = str(getattr(loaded, "entry_filter", "none") or "none")
    min_score_margin = float(getattr(loaded, "min_score_margin", 0.0) or 0.0)
    max_score_ceiling = max(float(getattr(loaded, "max_score_ceiling", 0.0) or 0.0), 0.0)
    max_trades_per_session = max(int(getattr(loaded, "max_trades_per_session", 0) or 0), 0)

    for decision_index, (session, row) in enumerate(decision_rows):
        if session_filter and session != session_filter:
            continue
        if str(row.get("feature_contract_version") or "") != contract_version:
            wrong_contract_rows += 1
            continue
        decision_time = _timestamp(row.get("decision_time"))
        if decision_time is None:
            continue
        candidates = candidate_records_from_row(
            row,
            session=session,
            split="historical_dataset",
            policy_index=loaded.policy_index,
        )
        _annotate_affordability(candidates, cash=cash)
        all_candidate_rows += len(candidates)
        if not candidates:
            rows_without_candidates += 1
            block_reasons = ["no_candidates"]
            selected_action = "wait"
            selected_contract_id = None
            selected_score = None
        else:
            score_candidate_records(candidates, scoring_loaded)
            scored_candidate_rows += len(candidates)
            selected_action = "wait"
            selected_contract_id = None
            selected_score = None
            block_reasons = []
            next_time = next_time_by_session.get(session)
            if next_time is not None and decision_time < next_time:
                cooldown_blocks += 1
                block_reasons.append("cooldown")
            elif max_trades_per_session > 0 and trades_by_session.get(session, 0) >= max_trades_per_session:
                session_trade_cap_blocks += 1
                block_reasons.append("session_trade_cap")
            else:
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate_allowed_by_entry_filter(candidate, row, entry_filter)
                ]
                if not eligible:
                    entry_filter_blocks += 1
                    block_reasons.append("entry_filter")
                else:
                    affordable = []
                    for candidate in eligible:
                        ask = _safe_float(candidate.get("entry_ask"))
                        if ask is not None and ask > 0.0 and ask * STRICT_REPLAY_CONTRACT_MULTIPLIER <= cash + 1e-9:
                            affordable.append(candidate)
                    if not affordable:
                        affordability_blocks += 1
                        block_reasons.append("affordability")
                    else:
                        top = top_candidate_with_margin(affordable)
                        if top is None:
                            threshold_waits += 1
                            block_reasons.append("no_scored_candidate")
                        else:
                            best, score_margin = top
                            selected_score = _safe_float(best.get("score"))
                            if min_score_margin > 0.0 and score_margin < min_score_margin:
                                score_margin_waits += 1
                                block_reasons.append("score_margin")
                            elif (
                                max_score_ceiling > 0.0
                                and selected_score is not None
                                and selected_score >= max_score_ceiling
                            ):
                                score_ceiling_waits += 1
                                block_reasons.append("score_ceiling")
                            elif selected_score is None or selected_score < float(loaded.threshold):
                                threshold_waits += 1
                                block_reasons.append("threshold")
                            else:
                                selected_action = "enter"
                                selected_contract_id = str(best.get("contract_id") or "")
                                block_reasons = []
                                selected.append(
                                    _selected_output(
                                        best,
                                        loaded=loaded,
                                        score_margin=score_margin,
                                        split="historical_dataset",
                                    )
                                )
                                trades_by_session[session] = trades_by_session.get(session, 0) + 1
                                next_time_by_session[session] = decision_time + pd.Timedelta(minutes=int(loaded.cooldown_minutes))
        candidate_payload = _trace_candidate_payload(candidates)
        candidate_filter_payload = _trace_candidate_filter_payload(row)
        token_features = [
            {
                "contract_id": candidate.get("contract_id"),
                "feature_hash": candidate.get("model_feature_hash") or candidate.get("feature_hash"),
                "features": _json_feature_vector(candidate.get("_model_features", candidate.get("_features"))),
            }
            for candidate in candidates
        ]
        scores = [
            {
                "contract_id": candidate.get("contract_id"),
                "score": candidate.get("score"),
            }
            for candidate in candidates
        ]
        trace_input = {
            "protocol_id": "protocol101_fair_contract_candidate",
            "source": "historical_dataset_fair_contract_replay",
            "session": session,
            "run_id": run_id,
            "mode": "historical-dataset-fair-contract-replay",
            "decision_ts": decision_time.isoformat(),
            "decision_index": decision_index,
            "feature_contract_version": contract_version,
            "source_quote_ts": _source_quote_time(candidates) or _iso(row.get("source_quote_time")),
            "source_context_ts": _iso(row.get("source_context_time")),
            "candidate_count": len(candidates),
            "candidate_universe": candidate_payload,
            "candidate_universe_hash": stable_hash(candidate_payload),
            "candidate_filter_trace": candidate_filter_payload,
            "candidate_filter_trace_hash": stable_hash(candidate_filter_payload),
            "ladder_context": row.get("ladder_context") or {},
            "features": {
                "token_features": token_features,
                "entry_filter": entry_filter,
                "policy_index": loaded.policy_index,
                "model_scoring_feature_transform": effective_feature_transform,
            },
            "feature_hash": stable_hash(token_features),
            "model_scores": {
                "candidate_scores": scores,
                "threshold": float(loaded.threshold),
                "max_score_ceiling": max_score_ceiling,
            },
            "score_hash": stable_hash(scores),
            "selected_action": selected_action,
            "selected_contract_id": selected_contract_id,
            "selected_score": selected_score,
            "decision_threshold": float(loaded.threshold),
            "block_reasons": block_reasons,
            "risk_gate": {
                "passed": selected_action == "enter" or bool(block_reasons),
                "reason": ",".join(block_reasons) if block_reasons else "enter",
                "reasons": block_reasons,
            },
            "account_state": {
                "mode": "offline_flat_entry_intent",
                "cash": cash,
                "open_positions": 0,
                "trades_for_session": trades_by_session.get(session, 0),
            },
            "quote_freshness_ms": row.get("max_quote_age_ms"),
            "opening_context_ready": bool(row.get("context_ready")),
            "missing_opening_minutes": 0 if row.get("context_ready") else None,
            "broker_order_endpoint_called": False,
            "raw_row_hash": stable_hash(
                {
                    "decision_time": _iso(row.get("decision_time")),
                    "feature_contract_version": row.get("feature_contract_version"),
                    "candidate_mask": np.asarray(row.get("candidate_mask"), dtype=bool).astype(int).tolist(),
                    "contract_ids": np.asarray(row.get("contract_ids"), dtype=object).tolist(),
                }
            ),
        }
        trace = normalize_decision_trace(trace_input, source="historical_dataset_fair_contract_replay").to_dict()
        traces.append(clean_json(trace))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "session": session_filter or ",".join(sorted({session for session, _row in decision_rows})),
        "sessions": sorted({row.get("session") for row in traces}),
        "run_id": run_id,
        "feature_contract": contract_version,
        "training_result": str(training_result.get("training_result_path") or ""),
        "model_path": str(getattr(loaded, "model_path", "")),
        "model_family": str(getattr(loaded, "model_family", "mlp") or "mlp"),
        "policy_index": int(loaded.policy_index),
        "policy_name": str(loaded.policy_name),
        "threshold": float(loaded.threshold),
        "entry_filter": entry_filter,
        "min_score_margin": min_score_margin,
        "max_score_ceiling": max_score_ceiling,
        "feature_transform": effective_feature_transform,
        "model_artifact_feature_transform": model_artifact_feature_transform,
        "contract_forced_feature_transform": feature_contract_model_transform(contract_version),
        "max_trades_per_session": max_trades_per_session,
        "decision_rows": len(traces),
        "candidate_rows": all_candidate_rows,
        "scored_candidate_rows": scored_candidate_rows,
        "selected_entries": len(selected),
        "threshold_waits": threshold_waits,
        "score_ceiling_waits": score_ceiling_waits,
        "score_margin_waits": score_margin_waits,
        "entry_filter_blocks": entry_filter_blocks,
        "affordability_blocks": affordability_blocks,
        "cooldown_blocks": cooldown_blocks,
        "session_trade_cap_blocks": session_trade_cap_blocks,
        "rows_without_candidates": rows_without_candidates,
        "context_not_ready": 0,
        "wrong_contract_rows": wrong_contract_rows,
        "trace_identity_hash": stable_hash([_decision_trace_identity(row) for row in traces]),
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "paid_data_downloaded": False,
    }
    return traces, selected, summary


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_result = load_json_optional(args.training_result)
    if not training_result:
        raise FileNotFoundError(f"training result missing or invalid: {args.training_result}")
    training_result = dict(training_result)
    training_result["training_result_path"] = str(args.training_result)
    traces, selected, summary = replay_once(
        input_paths=list(args.input_pkl),
        training_result=training_result,
        run_id=f"{args.session or 'multi'}-attempt107-historical-dataset-replay",
        session_filter=args.session,
        feature_contract=str(args.feature_contract),
    )
    repeat_payload: dict[str, Any] = {}
    if args.repeat_check:
        traces_b, selected_b, summary_b = replay_once(
            input_paths=list(args.input_pkl),
            training_result=training_result,
            run_id=f"{args.session or 'multi'}-attempt107-historical-dataset-replay-repeat",
            session_filter=args.session,
            feature_contract=str(args.feature_contract),
        )
        repeat_payload = {
            "same_input_exact": summary["trace_identity_hash"] == summary_b["trace_identity_hash"]
            and stable_hash([_csv_row(row) for row in selected]) == stable_hash([_csv_row(row) for row in selected_b]),
            "first_trace_identity_hash": summary["trace_identity_hash"],
            "second_trace_identity_hash": summary_b["trace_identity_hash"],
            "first_selected_hash": stable_hash([_csv_row(row) for row in selected]),
            "second_selected_hash": stable_hash([_csv_row(row) for row in selected_b]),
            "second_selected_entries": len(selected_b),
            "second_decision_rows": len(traces_b),
        }
    traces_jsonl = args.out_dir / "decision_traces.jsonl"
    selected_jsonl = args.out_dir / "selected_candidates.jsonl"
    selected_csv = args.out_dir / "selected_candidates.csv"
    summary_json = args.out_dir / "summary.json"
    report_md = args.out_dir / "report.md"
    write_jsonl(traces_jsonl, traces)
    write_jsonl(selected_jsonl, [_csv_row(row) for row in selected])
    write_csv(selected_csv, [_csv_row(row) for row in selected])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "input_pkl": [str(path) for path in args.input_pkl],
        "summary": summary,
        "repeat_check": repeat_payload,
        "outputs": {
            "decision_traces_jsonl": str(traces_jsonl),
            "selected_candidates_csv": str(selected_csv),
            "selected_candidates_jsonl": str(selected_jsonl),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n")
    report_md.write_text(render_capture_style_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "sessions": summary["sessions"],
                "selected_entries": summary["selected_entries"],
                "same_input_exact": repeat_payload.get("same_input_exact") if repeat_payload else None,
                "report": str(report_md),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
