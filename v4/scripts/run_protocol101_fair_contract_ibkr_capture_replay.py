"""Replay a fair-contract candidate over immutable IBKR recorder captures.

This script is intentionally offline-only. It rebuilds protocol101-live-v1
decision rows from a captured IBKR recorder stream, loads a frozen fair-contract
training result, scores each candidate, and writes reconstructable decision
traces for synchronization work.

It does not contact IBKR, download vendor data, train, tune thresholds, submit
orders, change defaults, or promote a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.ibkr_market_capture import clean_json, stable_hash
from v4.live.protocol101_capture_replay import (
    NY,
    build_replay_inputs,
    canonical_quote,
    parse_timestamp,
)
from v4.live.protocol101_decision_trace import normalize_decision_trace
from v4.live.protocol101_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
    feature_contract_metadata,
    feature_contract_model_transform,
    feature_contract_version,
)
from v4.live.protocol101_live_entry import LiveIndexState, build_live_surface_row
from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    STRICT_REPLAY_CONTRACT_MULTIPLIER,
    STRICT_REPLAY_STARTING_CASH,
    candidate_allowed_by_entry_filter,
    candidate_records_from_row,
    feature_hash,
    load_json_optional,
    load_model,
    score_candidate_records,
    top_candidate_with_margin,
    write_csv,
    _safe_float,
)


DEFAULT_CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"
DEFAULT_TRAINING_RESULT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_vwap_pocket_plateau_threshold"
    "/attempts/attempt_107_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_dailyloss500_plateau_s42"
    "/training_runner/training_result.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_ibkr_capture_replay"
)
SCHEMA_VERSION = "Protocol101FairContractIBKRCaptureReplayV1"


@dataclass(frozen=True)
class ReplayPaths:
    capture_dir: Path
    events: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Session date, e.g. 2026-07-02.")
    parser.add_argument("--capture-id", default=None)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--training-result", type=Path, default=DEFAULT_TRAINING_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--decision-end-et", default="15:30")
    parser.add_argument(
        "--feature-contract",
        choices=(FEATURE_CONTRACT_VERSION, FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED),
        default=FEATURE_CONTRACT_VERSION,
        help=(
            "model-facing replay contract. v2 preserves raw quotes for filters/fills "
            "but masks vendor-sensitive option microstructure before scoring."
        ),
    )
    parser.add_argument("--repeat-check", action="store_true")
    return parser.parse_args()


def resolve_capture_paths(capture_root: Path, session: str, capture_id: str | None) -> ReplayPaths:
    capture_id = capture_id or f"protocol101-recorder-{session}"
    capture_dir = capture_root / session / capture_id
    events = capture_dir / "market_events.jsonl"
    if not events.exists():
        raise FileNotFoundError(f"capture events missing: {events}")
    return ReplayPaths(capture_dir=capture_dir, events=events)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _decision_trace_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_ts": row.get("decision_ts"),
        "candidate_universe_hash": row.get("candidate_universe_hash"),
        "feature_hash": row.get("feature_hash"),
        "score_hash": row.get("score_hash"),
        "selected_action": row.get("selected_action"),
        "selected_contract_id": row.get("selected_contract_id"),
        "selected_score": row.get("selected_score"),
        "decision_threshold": row.get("decision_threshold"),
        "block_reasons": row.get("block_reasons"),
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_") and key not in {"token_features", "candidate_scores"}
    }


def _trace_candidate_payload(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for candidate in candidates:
        payload.append(
            {
                "contract_id": candidate.get("contract_id"),
                "right": candidate.get("right"),
                "offset": candidate.get("offset"),
                "strike_idx": candidate.get("strike_idx"),
                "right_idx": candidate.get("right_idx"),
                "entry_bid": candidate.get("entry_bid"),
                "entry_ask": candidate.get("entry_ask"),
                "entry_mid": candidate.get("entry_mid"),
                "entry_spread": candidate.get("entry_spread"),
                "entry_spread_frac": candidate.get("entry_spread_frac"),
                "bid_size": candidate.get("bid_size"),
                "ask_size": candidate.get("ask_size"),
                "quote_age_ms": candidate.get("quote_age_ms"),
                "quote_age_source": candidate.get("quote_age_source"),
                "raw_quote_timestamp_utc": candidate.get("raw_quote_timestamp_utc"),
                "received_timestamp_utc": candidate.get("received_timestamp_utc"),
                "source_quote_ts": candidate.get("source_quote_ts"),
                "source_context_ts": candidate.get("source_context_ts"),
                "spx_for_ladder": candidate.get("spx_for_ladder"),
                "atm_strike": candidate.get("atm_strike"),
                "strike_step": candidate.get("strike_step"),
                "rounding_tie_policy": candidate.get("rounding_tie_policy"),
                "pre_filter_candidate": candidate.get("pre_filter_candidate"),
                "post_filter_candidate": candidate.get("post_filter_candidate"),
                "candidate_filter": candidate.get("candidate_filter"),
                "filter_reasons": candidate.get("filter_reasons"),
                "tradability_pass": candidate.get("tradability_pass"),
                "freshness_pass": candidate.get("freshness_pass"),
                "model_scoring_greek_pass": candidate.get("model_scoring_greek_pass"),
                "affordable_at_decision": candidate.get("affordable_at_decision"),
                "affordability_result": candidate.get("affordability_result"),
                "feature_hash": candidate.get("model_feature_hash") or candidate.get("feature_hash"),
                "score": candidate.get("score"),
            }
        )
    return payload


def _trace_candidate_filter_payload(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = row.get("candidate_filter_trace") or []
    if not isinstance(trace, list):
        return []
    return [
        {
            key: value
            for key, value in dict(item).items()
            if key not in {"raw_vendor_fields", "_features", "_model_features"}
        }
        for item in trace
        if isinstance(item, dict)
    ]


def _annotate_affordability(candidates: list[dict[str, Any]], *, cash: float) -> None:
    for candidate in candidates:
        ask = _safe_float(candidate.get("entry_ask"))
        required_cash = ask * STRICT_REPLAY_CONTRACT_MULTIPLIER if ask is not None else None
        affordable = (
            required_cash is not None
            and ask is not None
            and ask > 0.0
            and required_cash <= cash + 1e-9
        )
        candidate["affordable_at_decision"] = bool(affordable)
        candidate["affordability_result"] = {
            "cash": cash,
            "entry_ask": ask,
            "contract_multiplier": STRICT_REPLAY_CONTRACT_MULTIPLIER,
            "required_cash": required_cash,
            "passed": bool(affordable),
        }


def _json_feature_vector(vector: Any) -> list[float | None]:
    if vector is None:
        return []
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    out: list[float | None] = []
    for value in values:
        number = float(value)
        out.append(number if math.isfinite(number) else None)
    return out


def _source_quote_time(candidates: list[dict[str, Any]]) -> str | None:
    stamps = []
    for candidate in candidates:
        ts = parse_timestamp(candidate.get("source_quote_time"))
        if ts is not None:
            stamps.append(ts)
    if not stamps:
        return None
    return max(stamps).isoformat()


def _selected_output(
    best: dict[str, Any],
    *,
    loaded: Any,
    score_margin: float,
    split: str,
) -> dict[str, Any]:
    out = {key: value for key, value in best.items() if key != "_features"}
    out.update(
        {
            "selected_rank": 1,
            "threshold": loaded.threshold,
            "policy_index": loaded.policy_index,
            "policy_name": loaded.policy_name,
            "cooldown_minutes": loaded.cooldown_minutes,
            "target_mode": loaded.target_mode,
            "entry_filter": loaded.entry_filter,
            "min_score_margin": loaded.min_score_margin,
            "max_score_ceiling": loaded.max_score_ceiling,
            "max_trades_per_session": loaded.max_trades_per_session,
            "max_daily_loss": loaded.max_daily_loss,
            "score_margin": score_margin,
            "model_path": str(loaded.model_path),
            "model_family": str(getattr(loaded, "model_family", "mlp") or "mlp"),
            "split": split,
        }
    )
    return out


def _loaded_for_feature_contract(loaded: Any, feature_contract: str) -> Any:
    contract_transform = feature_contract_model_transform(feature_contract)
    if contract_transform == "none":
        return loaded
    return replace(loaded, feature_transform=contract_transform)


def replay_once(
    *,
    session: str,
    events_path: Path,
    training_result: dict[str, Any],
    decision_end_et: str,
    run_id: str,
    feature_contract: str = FEATURE_CONTRACT_VERSION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    loaded = load_model(training_result)
    contract_version = feature_contract_version(feature_contract)
    scoring_loaded = _loaded_for_feature_contract(loaded, contract_version)
    model_artifact_feature_transform = str(getattr(loaded, "feature_transform", "none") or "none")
    effective_feature_transform = str(getattr(scoring_loaded, "feature_transform", "none") or "none")
    checkpoints, raw_index_state = build_replay_inputs(
        events_path,
        session=session,
        decision_end_et=decision_end_et,
    )
    causal_state = LiveIndexState(prior_session_close=raw_index_state.prior_session_close)
    raw_rows = sorted(raw_index_state.rows, key=lambda item: item["timestamp"])
    raw_cursor = 0
    traces: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    all_candidate_rows = 0
    scored_candidate_rows = 0
    decision_rows = 0
    threshold_waits = 0
    score_ceiling_waits = 0
    score_margin_waits = 0
    entry_filter_blocks = 0
    affordability_blocks = 0
    cooldown_blocks = 0
    session_trade_cap_blocks = 0
    context_not_ready = 0
    rows_without_candidates = 0
    next_time: pd.Timestamp | None = None
    trades_for_session = 0
    cash = float(STRICT_REPLAY_STARTING_CASH)
    entry_filter = str(getattr(loaded, "entry_filter", "none") or "none")
    min_score_margin = float(getattr(loaded, "min_score_margin", 0.0) or 0.0)
    max_score_ceiling = max(float(getattr(loaded, "max_score_ceiling", 0.0) or 0.0), 0.0)
    max_trades_per_session = max(int(getattr(loaded, "max_trades_per_session", 0) or 0), 0)

    for decision_index, checkpoint in enumerate(checkpoints):
        decision_time = parse_timestamp(checkpoint.get("decision_time_et"))
        if decision_time is None:
            continue
        local = decision_time.tz_convert(NY)
        if local.time() < pd.Timestamp("09:31").time() or local.time() > pd.Timestamp(decision_end_et).time():
            continue
        decision_rows += 1
        while raw_cursor < len(raw_rows) and pd.Timestamp(raw_rows[raw_cursor]["timestamp"]) <= decision_time:
            item = raw_rows[raw_cursor]
            causal_state.add(timestamp=item["timestamp"], spx=float(item["spx"]), vix=float(item["vix"]))
            raw_cursor += 1
        completed_context = causal_state.frame(decision_time.floor("min") - pd.Timedelta(microseconds=1))
        spx_payload = checkpoint.get("spx") if isinstance(checkpoint.get("spx"), dict) else {}
        vix_payload = checkpoint.get("vix") if isinstance(checkpoint.get("vix"), dict) else {}
        spx = float(completed_context.iloc[-1]["spx"]) if not completed_context.empty else _safe_float(spx_payload.get("price"))
        vix = float(completed_context.iloc[-1]["vix"]) if not completed_context.empty else _safe_float(vix_payload.get("price"))
        if spx is None or vix is None or not math.isfinite(spx) or not math.isfinite(vix):
            context_not_ready += 1
            continue
        received = checkpoint.get("_received_timestamp_utc")
        quotes = [
            canonical_quote(item, spx=spx, decision_time=decision_time, received_time=received)
            for item in checkpoint.get("contracts", [])
            if isinstance(item, dict)
        ]
        row, _lookup = build_live_surface_row(
            decision_time=decision_time,
            spx=spx,
            vix=vix,
            option_quotes=quotes,
            index_state=causal_state,
            policy_count=max(int(loaded.policy_index) + 1, 3),
            feature_contract_name=contract_version,
        )
        row["feature_contract_version"] = contract_version
        row["feature_contract"] = feature_contract_metadata(contract_version)
        context_summary = causal_state.session_context_summary(decision_time)
        candidates = candidate_records_from_row(
            row,
            session=session,
            split="ibkr_capture",
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
            if next_time is not None and decision_time < next_time:
                cooldown_blocks += 1
                block_reasons.append("cooldown")
            elif max_trades_per_session > 0 and trades_for_session >= max_trades_per_session:
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
                                out = _selected_output(
                                    best,
                                    loaded=loaded,
                                    score_margin=score_margin,
                                    split="ibkr_capture",
                                )
                                selected.append(out)
                                trades_for_session += 1
                                next_time = decision_time + pd.Timedelta(minutes=int(loaded.cooldown_minutes))
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
            "source": "ibkr_capture_fair_contract_replay",
            "session": session,
            "run_id": run_id,
            "mode": "recorder-offline-fair-contract-replay",
            "decision_ts": decision_time.isoformat(),
            "decision_index": decision_index,
            "feature_contract_version": contract_version,
            "source_quote_ts": _source_quote_time(candidates) or received,
            "source_context_ts": context_summary.get("last_timestamp"),
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
                "trades_for_session": trades_for_session,
            },
            "quote_freshness_ms": row.get("max_quote_age_ms"),
            "opening_context_ready": context_summary.get("opening_context_ready"),
            "missing_opening_minutes": context_summary.get("missing_opening_minutes"),
            "broker_order_endpoint_called": False,
            "raw_checkpoint_hash": stable_hash(checkpoint),
        }
        trace = normalize_decision_trace(trace_input, source="ibkr_capture_fair_contract_replay").to_dict()
        traces.append(clean_json(trace))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "session": session,
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
        "decision_rows": decision_rows,
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
        "context_not_ready": context_not_ready,
        "rows_without_candidates": rows_without_candidates,
        "trace_identity_hash": stable_hash([_decision_trace_identity(row) for row in traces]),
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "paid_data_downloaded": False,
    }
    return traces, selected, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(clean_json(row), sort_keys=True) + "\n")


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    repeat = payload.get("repeat_check") or {}
    lines = [
        "# Protocol101 Fair-Contract IBKR Capture Replay",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Session: `{summary['session']}`",
        f"- Feature contract: `{summary['feature_contract']}`",
        f"- Selected entries: `{summary['selected_entries']}`",
        f"- Same-input exact repeat: `{str(repeat.get('same_input_exact', False)).lower()}`"
        if repeat
        else "- Same-input exact repeat: `not_run`",
        "- Broker endpoint called: `false`",
        "- Paper-submit allowed: `false`",
        "- Model training executed here: `false`",
        "- Threshold tuning executed here: `false`",
        "- Paid data downloaded: `false`",
        "",
        "## Replay Summary",
        "",
        f"- Decisions: `{summary['decision_rows']}`",
        f"- Candidates: `{summary['candidate_rows']}`",
        f"- Scored candidates: `{summary['scored_candidate_rows']}`",
        f"- Entry filter: `{summary['entry_filter']}`",
        f"- Threshold: `{summary['threshold']}`",
        f"- Max score ceiling: `{summary['max_score_ceiling']}`",
        f"- Threshold waits: `{summary['threshold_waits']}`",
        f"- Score-ceiling waits: `{summary['score_ceiling_waits']}`",
        f"- Entry-filter blocks: `{summary['entry_filter_blocks']}`",
        f"- Cooldown blocks: `{summary['cooldown_blocks']}`",
        f"- Session trade-cap blocks: `{summary['session_trade_cap_blocks']}`",
        f"- Context not ready rows: `{summary['context_not_ready']}`",
        f"- Rows without candidates: `{summary['rows_without_candidates']}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in payload.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_capture_paths(args.capture_root, args.session, args.capture_id)
    training_result = load_json_optional(args.training_result)
    if not training_result:
        raise FileNotFoundError(f"training result missing or invalid: {args.training_result}")
    training_result = dict(training_result)
    training_result["training_result_path"] = str(args.training_result)
    traces, selected, summary = replay_once(
        session=args.session,
        events_path=paths.events,
        training_result=training_result,
        decision_end_et=str(args.decision_end_et),
        run_id=f"{args.session}-attempt107-ibkr-capture-replay",
        feature_contract=str(args.feature_contract),
    )
    repeat_payload: dict[str, Any] = {}
    if args.repeat_check:
        traces_b, selected_b, summary_b = replay_once(
            session=args.session,
            events_path=paths.events,
            training_result=training_result,
            decision_end_et=str(args.decision_end_et),
            run_id=f"{args.session}-attempt107-ibkr-capture-replay-repeat",
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
        "capture_dir": str(paths.capture_dir),
        "capture_events": str(paths.events),
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
    report_md.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "session": args.session,
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
