"""Protocol 161: replay frozen Protocol101 on May 2026 historical data.

This runner answers one narrow live-parity question: if we replay the same
calendar sessions from normalized historical OPRA + official SPX/VIX context,
does the frozen Protocol101 entry stack produce any entry signals?

It does not train, tune thresholds, download data, call IBKR, or place orders.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.protocol051_surface_edge import load_surface_edge_artifact, score_surface_decisions
from v4.live.protocol101_entry import (
    Protocol101HistoryState,
    load_protocol101_entry_artifact,
    predict_protocol101_entry,
    protocol101_candidate_frame_from_surface,
    protocol101_candidate_gate_diagnostics,
)
from v4.live.protocol101_live_entry import scalar_feature_payload, scored_token_universe_payload
from v4.live.paper_trade_log import stable_json_hash
from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION, HISTORICAL_FEATURE_CONTRACT_VERSION
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    SurfaceVariant,
    registered_aplus_surface_variants,
    surface_decision_from_row,
)
from v4.scripts.run_protocol119_protocol101_live_readiness import (
    DEFAULT_PROTOCOL101_MANIFEST,
    DEFAULT_PROTOCOL101_SUMMARY,
)
from v4.scripts.run_protocol121_protocol101_entry_router_smoke import DEFAULT_SURFACE_MANIFEST


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_161_may2026_historical_replay")
DEFAULT_PROCESSED_DIR = Path("data/processed/spxw_0dte_neural_protocol161_may2026_official_context")
DEFAULT_SPX_DIR = Path("data/vendor/thetadata/index/spx_1m")
DEFAULT_VIX_DIR = Path("data/vendor/thetadata/index/vix_1m")
DEFAULT_LIVE_LOG_ROOT = Path("v4/logs/paper_trading")
DEFAULT_SESSIONS = ("2026-05-19", "2026-05-20")


def variant_for(name: str) -> SurfaceVariant:
    """Resolve the frozen surface variant without importing broker runtime code."""

    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise ValueError(f"no registered A+ surface variant named {name!r}")


@dataclass(frozen=True)
class ReplayDecision:
    session: str
    decision_time: str
    source_decision_time: str | None
    completed_minute_lag_minutes: int
    timestamp_alignment_mode: str
    time_bucket: str | None
    context_ready: bool | None
    context_minute_rows: int | None
    context_span_minutes: float | None
    action: str
    reason: str | None
    candidate_count: int
    above_min_edge_count: int
    max_edge: float | None
    best_call_edge: float | None
    best_put_edge: float | None
    margin: float | None
    threshold: float | None
    selected_contract_id: str | None
    selected_right: str | None
    selected_offset_points: float | None
    selected_edge: float | None
    selected_ask: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--protocol101-manifest", type=Path, default=DEFAULT_PROTOCOL101_MANIFEST)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--official-spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--official-vix-dir", type=Path, default=DEFAULT_VIX_DIR)
    parser.add_argument("--live-log-root", type=Path, default=DEFAULT_LIVE_LOG_ROOT)
    parser.add_argument("--min-edge", type=float, default=25.0)
    parser.add_argument("--max-rows", type=int, default=0, help="debug limit; 0 means all rows")
    parser.add_argument(
        "--scope-index-context-to-sessions",
        action="store_true",
        help=(
            "Build session-scoped SPX/VIX symlink directories under the output directory "
            "before replay so a small parity run does not read unrelated historical context files."
        ),
    )
    parser.add_argument(
        "--max-prior-context-gap-days",
        type=int,
        default=7,
        help="When scoping index context, include prior-session files only if they are within this many calendar days.",
    )
    parser.add_argument(
        "--completed-minute-lag-minutes",
        type=int,
        default=0,
        help=(
            "For live-parity rehearsals, shift the effective decision timestamp forward "
            "by this many minutes while preserving the source row features. A value of "
            "1 means the replay decision at T only sees the historical row completed at T-1."
        ),
    )
    parser.add_argument(
        "--market-context-lag-minutes",
        type=int,
        default=None,
        help="Index-bar context lag. Defaults to one minute for protocol101-live-v1 rows and zero otherwise.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    surface_artifact = load_surface_edge_artifact(args.surface_manifest)
    protocol101 = load_protocol101_entry_artifact(args.protocol101_manifest, args.protocol101_summary)
    variant = variant_for(surface_artifact.variant_name)
    official_spx_dir = args.official_spx_dir
    official_vix_dir = args.official_vix_dir
    scoped_index_context: dict[str, Any] = {"enabled": False}
    if args.scope_index_context_to_sessions:
        official_spx_dir, spx_context_files, spx_prior_gaps = build_session_scoped_index_dir(
            args.official_spx_dir,
            args.sessions,
            args.out_dir / "_session_scoped_index_context" / "spx_1m",
            max_prior_gap_days=int(args.max_prior_context_gap_days),
        )
        official_vix_dir, vix_context_files, vix_prior_gaps = build_session_scoped_index_dir(
            args.official_vix_dir,
            args.sessions,
            args.out_dir / "_session_scoped_index_context" / "vix_1m",
            max_prior_gap_days=int(args.max_prior_context_gap_days),
        )
        scoped_index_context = {
            "enabled": True,
            "max_prior_context_gap_days": int(args.max_prior_context_gap_days),
            "spx_dir": str(official_spx_dir),
            "vix_dir": str(official_vix_dir),
            "spx_files": [str(path) for path in spx_context_files],
            "vix_files": [str(path) for path in vix_context_files],
            "spx_prior_context_gaps": spx_prior_gaps,
            "vix_prior_context_gaps": vix_prior_gaps,
        }
    rows = load_neural_rows(args.processed_dir, args.sessions, max_rows=int(args.max_rows))
    market_context_lag_minutes = args.market_context_lag_minutes
    if market_context_lag_minutes is None:
        market_context_lag_minutes = 1 if any(
            row.get("feature_contract_version") == FEATURE_CONTRACT_VERSION for _, row in rows
        ) else 0
    market_cache = MarketStructureCache(
        source="index_bars",
        index_spx_dir=official_spx_dir,
        index_vix_dir=official_vix_dir,
        decision_context_lag_minutes=int(market_context_lag_minutes),
    )
    history = Protocol101HistoryState()
    replay_rows: list[ReplayDecision] = []
    surface_edge_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    completed_minute_lag_minutes = int(args.completed_minute_lag_minutes)
    if completed_minute_lag_minutes < 0:
        raise ValueError("--completed-minute-lag-minutes must be >= 0")
    timestamp_alignment_mode = (
        f"completed-minute-lag-{completed_minute_lag_minutes}m"
        if completed_minute_lag_minutes
        else "historical-row-timestamp"
    )

    for decision_index, (session, raw_row) in enumerate(rows):
        row, source_decision_time, effective_decision_time = apply_completed_minute_lag(
            raw_row,
            completed_minute_lag_minutes,
        )
        decision = surface_decision_from_row(
            session=session,
            row=row,
            policy_index=surface_artifact.policy_index,
            variant=variant,
            market_cache=market_cache,
        )
        if completed_minute_lag_minutes:
            decision.decision_time = effective_decision_time.to_pydatetime()
        surface_scores = score_surface_decisions(surface_artifact, [decision])[0]
        diagnostics = protocol101_candidate_gate_diagnostics(
            decision,
            surface_scores,
            min_edge=float(args.min_edge),
        )
        context_ready = row_context_ready(row)
        context_blocked = bool(diagnostics.get("allowed_time_bucket")) and context_ready is False
        if context_blocked:
            diagnostics.update(
                {
                    "filter_reason": "insufficient_historical_index_context",
                    "canonical_filter_reason": "insufficient_index_context",
                    "context_ready": False,
                    "context_required_minutes": row.get("context_required_minutes"),
                    "context_minute_rows": row.get("context_minute_rows"),
                    "context_span_minutes": row.get("context_span_minutes"),
                    "above_min_edge_count": 0,
                }
            )
            candidates = pd.DataFrame()
            prediction = {
                "action": "wait",
                "reason": "insufficient_historical_index_context",
                "margin": None,
                "threshold": protocol101.threshold,
                "selected": {},
            }
        else:
            diagnostics.update(
                {
                    "context_ready": context_ready,
                    "context_required_minutes": row.get("context_required_minutes"),
                    "context_minute_rows": row.get("context_minute_rows"),
                    "context_span_minutes": row.get("context_span_minutes"),
                }
            )
            candidates = protocol101_candidate_frame_from_surface(
                decision,
                surface_scores,
                history,
                min_edge=float(args.min_edge),
            )
            prediction = predict_protocol101_entry(protocol101, candidates)
            if not candidates.empty:
                history.update(candidates, pd.Timestamp(decision.decision_time))
        selected = prediction.get("selected") or {}
        contract_lookup = object_or_empty(row.get("contract_quote_metadata"))
        scored_tokens = scored_token_universe_payload(decision, surface_scores, contract_lookup)
        scalar_payload = scalar_feature_payload(decision)
        trace_features = {
            **scalar_payload,
            "token_features": [
                {
                    "contract_id": token.get("contract_id"),
                    "token_idx": token.get("token_idx"),
                    "token_feature_hash": token.get("token_feature_hash"),
                    "token_features": token.get("token_features"),
                }
                for token in scored_tokens
            ],
        }
        trace_scores = {
            "surface_scores": [finite_or_none(value) for value in list(surface_scores)],
            "protocol101_raw_logits": prediction.get("raw_logits", []),
            "protocol101_wait_logit": prediction.get("wait_logit"),
            "protocol101_candidate_logits": prediction.get("candidate_logits", []),
        }
        filter_reason = diagnostics.get("filter_reason")
        prediction_reason = prediction.get("reason")
        trace_rows.append(
            {
                "decision_trace_schema_version": "Protocol101DecisionTraceV1",
                "protocol_id": "protocol101",
                "source": "historical",
                "session": session,
                "run_id": "protocol161_historical_replay",
                "decision_ts": pd.Timestamp(decision.decision_time).isoformat(),
                "decision_index": decision_index,
                "source_decision_ts": source_decision_time.isoformat(),
                "source_quote_ts": iso_or_none(row.get("source_quote_time")),
                "source_context_ts": iso_or_none(row.get("source_context_time") or row.get("context_last_timestamp")),
                "feature_contract_version": row.get("feature_contract_version", HISTORICAL_FEATURE_CONTRACT_VERSION),
                "completed_minute_lag_minutes": completed_minute_lag_minutes,
                "timestamp_alignment_mode": timestamp_alignment_mode,
                "mode": "historical-replay",
                "selected_action": "enter" if prediction.get("action") == "enter" else "wait",
                "selected_contract": {"contract_id": selected.get("contract_id")} if selected.get("contract_id") else {},
                "selected_score": finite_or_none(prediction.get("margin")),
                "decision_threshold": finite_or_none(prediction.get("threshold")),
                "candidate_count": int(len(candidates)),
                "candidate_universe": scored_tokens,
                "candidate_universe_hash": stable_json_hash(scored_tokens),
                "features": trace_features,
                "feature_hash": stable_json_hash(trace_features),
                "model_scores": trace_scores,
                "score_hash": stable_json_hash(trace_scores),
                "quote_freshness_ms": finite_or_none(row.get("max_quote_age_ms")),
                "context_freshness_ms": 0,
                "metadata": {
                    "feature_contract": row.get("feature_contract", {}),
                    "source_quote_time": iso_or_none(row.get("source_quote_time")),
                    "source_context_time": iso_or_none(row.get("source_context_time") or row.get("context_last_timestamp")),
                    "timestamp_alignment_mode": timestamp_alignment_mode,
                    "completed_minute_lag_minutes": completed_minute_lag_minutes,
                },
                "risk_gate": {
                    "passed": True,
                    "reason": filter_reason,
                    "reasons": [] if filter_reason in {"candidates_available", "below_min_edge", "no_valid_surface_scores", None} else [filter_reason],
                },
                "block_reasons": [] if prediction.get("action") == "enter" or not prediction_reason else [prediction_reason],
            }
        )
        replay_rows.append(
            ReplayDecision(
                session=session,
                decision_time=pd.Timestamp(decision.decision_time).isoformat(),
                source_decision_time=source_decision_time.isoformat(),
                completed_minute_lag_minutes=completed_minute_lag_minutes,
                timestamp_alignment_mode=timestamp_alignment_mode,
                time_bucket=diagnostics.get("time_bucket"),
                context_ready=context_ready,
                context_minute_rows=int_or_none(row.get("context_minute_rows")),
                context_span_minutes=finite_or_none(row.get("context_span_minutes")),
                action="enter" if prediction.get("action") == "enter" else "wait",
                reason=prediction.get("reason"),
                candidate_count=int(len(candidates)),
                above_min_edge_count=int(diagnostics.get("above_min_edge_count") or 0),
                max_edge=finite_or_none(diagnostics.get("max_edge")),
                best_call_edge=finite_or_none(diagnostics.get("best_call_edge")),
                best_put_edge=finite_or_none(diagnostics.get("best_put_edge")),
                margin=finite_or_none(prediction.get("margin")),
                threshold=finite_or_none(prediction.get("threshold")),
                selected_contract_id=none_if_empty(selected.get("contract_id")),
                selected_right=none_if_empty(selected.get("right")),
                selected_offset_points=finite_or_none(selected.get("offset_points")),
                selected_edge=finite_or_none(selected.get("edge")),
                selected_ask=finite_or_none(selected.get("entry_ask")),
            )
        )
        surface_edge_rows.append(
            {
                "session": session,
                "decision_time": pd.Timestamp(decision.decision_time).isoformat(),
                "source_decision_time": source_decision_time.isoformat(),
                "completed_minute_lag_minutes": completed_minute_lag_minutes,
                "timestamp_alignment_mode": timestamp_alignment_mode,
                "time_bucket": diagnostics.get("time_bucket"),
                "filter_reason": diagnostics.get("filter_reason"),
                "flat_score": diagnostics.get("flat_score"),
                "max_edge": diagnostics.get("max_edge"),
                "best_call_edge": diagnostics.get("best_call_edge"),
                "best_put_edge": diagnostics.get("best_put_edge"),
                "above_min_edge_count": diagnostics.get("above_min_edge_count"),
                "eligible_token_count": diagnostics.get("eligible_token_count"),
                "context_ready": context_ready,
                "context_minute_rows": row.get("context_minute_rows"),
                "context_span_minutes": row.get("context_span_minutes"),
                "canonical_filter_reason": diagnostics.get("canonical_filter_reason"),
            }
        )

    replay_frame = pd.DataFrame([asdict(row) for row in replay_rows])
    surface_frame = pd.DataFrame(surface_edge_rows)
    live_summary = summarize_live_logs(args.live_log_root, args.sessions)
    payload = {
        "protocol": "161_may2026_historical_replay",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "surface_manifest": str(args.surface_manifest),
        "protocol101_manifest": str(args.protocol101_manifest),
        "processed_dir": str(args.processed_dir),
        "sessions": list(args.sessions),
        "official_spx_dir": str(official_spx_dir),
        "official_vix_dir": str(official_vix_dir),
        "scoped_index_context": scoped_index_context,
        "min_edge": float(args.min_edge),
        "timestamp_alignment": {
            "mode": timestamp_alignment_mode,
            "completed_minute_lag_minutes": completed_minute_lag_minutes,
            "meaning": (
                "decision_time is the effective live decision minute; "
                "source_decision_time is the historical row/features minute"
            ),
        },
        "summary": summarize_replay(replay_frame, surface_frame),
        "live_log_comparison": live_summary,
        "decision": replay_decision(replay_frame),
    }

    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    replay_frame.to_csv(args.out_dir / "replay_decisions.csv", index=False)
    surface_frame.to_csv(args.out_dir / "surface_gate_rows.csv", index=False)
    (args.out_dir / "decision_traces.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, default=str, allow_nan=False) + "\n" for row in trace_rows)
    )
    write_report(args.out_dir / "report.md", payload, replay_frame)
    print(json.dumps({"decision": payload["decision"], "summary": payload["summary"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_neural_rows(processed_dir: Path, sessions: list[str], *, max_rows: int) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for session in sessions:
        path = processed_dir / f"{session}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing processed neural rows: {path}")
        rows = pickle.loads(path.read_bytes())
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a list of decision rows")
        for row in rows:
            out.append((session, row))
            if max_rows > 0 and len(out) >= max_rows:
                return out
    return out


def build_session_scoped_index_dir(
    source_dir: Path,
    sessions: list[str],
    target_dir: Path,
    *,
    max_prior_gap_days: int,
) -> tuple[Path, list[Path], list[dict[str, Any]]]:
    suffixes = {".parquet", ".csv", ".jsonl", ".txt"}
    dated_files: dict[pd.Timestamp, Path] = {}
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            day = pd.Timestamp(path.stem).normalize()
        except ValueError:
            continue
        dated_files[day] = path
    if not dated_files:
        raise FileNotFoundError(f"no dated index context files found under {source_dir}")

    available_days = sorted(dated_files)
    needed_days: set[pd.Timestamp] = set()
    missing_sessions: list[str] = []
    prior_context_gaps: list[dict[str, Any]] = []
    for session in sessions:
        day = pd.Timestamp(session).normalize()
        if day not in dated_files:
            missing_sessions.append(session)
            continue
        needed_days.add(day)
        prior_days = [candidate for candidate in available_days if candidate < day]
        if prior_days:
            prior_day = prior_days[-1]
            gap_days = int((day - prior_day).days)
            if gap_days <= int(max_prior_gap_days):
                needed_days.add(prior_day)
            else:
                prior_context_gaps.append(
                    {
                        "session": session,
                        "prior_available_session": prior_day.date().isoformat(),
                        "gap_days": gap_days,
                        "action": "skipped_stale_prior_context",
                    }
                )
    if missing_sessions:
        raise FileNotFoundError(f"missing index context files for sessions: {missing_sessions}")

    target_dir.mkdir(parents=True, exist_ok=True)
    for existing in target_dir.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()
    linked_files: list[Path] = []
    for day in sorted(needed_days):
        source = dated_files[day]
        target = target_dir / source.name
        target.symlink_to(source.resolve())
        linked_files.append(target)
    return target_dir, linked_files, prior_context_gaps


def apply_completed_minute_lag(
    row: dict[str, Any],
    lag_minutes: int,
) -> tuple[dict[str, Any], pd.Timestamp, pd.Timestamp]:
    """Return a replay row whose effective decision time is shifted forward.

    Historical one-minute OPRA/context rows are stamped at their source minute.
    The live IBKR loop makes a decision at the top of a minute using the most
    recently completed market state. For live-parity rehearsals, a one-minute
    lag labels the decision at T while preserving the source features from T-1.
    """

    source_decision_time = pd.Timestamp(row.get("decision_time"))
    if source_decision_time.tzinfo is None:
        source_decision_time = source_decision_time.tz_localize("UTC")
    else:
        source_decision_time = source_decision_time.tz_convert("UTC")
    effective_decision_time = source_decision_time + pd.Timedelta(minutes=int(lag_minutes))
    if lag_minutes <= 0:
        return row, source_decision_time, effective_decision_time
    adjusted = dict(row)
    adjusted["source_decision_time"] = source_decision_time.to_pydatetime()
    adjusted["completed_minute_lag_minutes"] = int(lag_minutes)
    return adjusted, source_decision_time, effective_decision_time


def summarize_replay(replay: pd.DataFrame, surface: pd.DataFrame) -> dict[str, Any]:
    if replay.empty:
        return {
            "decision_rows": 0,
            "entry_signals": 0,
            "candidate_rows": 0,
            "events_with_candidates": 0,
            "max_edge": None,
            "best_call_edge_max": None,
            "best_put_edge_max": None,
            "by_session": {},
            "by_time_bucket": {},
        }
    by_session: dict[str, Any] = {}
    for session, group in replay.groupby("session"):
        by_session[str(session)] = summarize_group(group)
    by_bucket: dict[str, Any] = {}
    for bucket, group in replay.groupby("time_bucket", dropna=False):
        by_bucket[str(bucket)] = summarize_group(group)
    return {
        "decision_rows": int(len(replay)),
        "entry_signals": int((replay["action"] == "enter").sum()),
        "candidate_rows": int(pd.to_numeric(replay["candidate_count"], errors="coerce").fillna(0).sum()),
        "events_with_candidates": int((pd.to_numeric(replay["candidate_count"], errors="coerce").fillna(0) > 0).sum()),
        "events_above_min_edge": int((pd.to_numeric(replay["above_min_edge_count"], errors="coerce").fillna(0) > 0).sum()),
        "context_not_ready_rows": int((replay.get("context_ready") == False).sum()) if "context_ready" in replay else 0,
        "max_edge": numeric_max(surface.get("max_edge")),
        "best_call_edge_max": numeric_max(surface.get("best_call_edge")),
        "best_put_edge_max": numeric_max(surface.get("best_put_edge")),
        "by_session": by_session,
        "by_time_bucket": by_bucket,
    }


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "decision_rows": int(len(group)),
        "entry_signals": int((group["action"] == "enter").sum()),
        "events_with_candidates": int((pd.to_numeric(group["candidate_count"], errors="coerce").fillna(0) > 0).sum()),
        "candidate_rows": int(pd.to_numeric(group["candidate_count"], errors="coerce").fillna(0).sum()),
        "context_not_ready_rows": int((group.get("context_ready") == False).sum()) if "context_ready" in group else 0,
        "max_edge": numeric_max(group.get("max_edge")),
        "best_call_edge_max": numeric_max(group.get("best_call_edge")),
        "best_put_edge_max": numeric_max(group.get("best_put_edge")),
        "max_margin": numeric_max(group.get("margin")),
    }


def summarize_live_logs(live_log_root: Path, sessions: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for session in sessions:
        day_dir = live_log_root / session
        files = sorted(day_dir.glob("protocol101*.jsonl")) if day_dir.exists() else []
        if day_dir.exists() and not files:
            files = sorted(day_dir.glob("daily_paper_autopilot_*.jsonl"))
        rows: list[dict[str, Any]] = []
        for path in files:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event_type") == "candidate_set":
                    rows.append(row)
        gate_rows = [object_or_empty(row.get("candidate_gate_diagnostics")) for row in rows]
        timestamps = [pd.Timestamp(row.get("timestamp")) for row in rows if row.get("timestamp")]
        out[session] = {
            "files": [str(path) for path in files],
            "candidate_set_rows": len(rows),
            "first_candidate_set_utc": min(timestamps).isoformat() if timestamps else None,
            "last_candidate_set_utc": max(timestamps).isoformat() if timestamps else None,
            "first_candidate_set_et": min(timestamps).tz_convert("America/New_York").isoformat() if timestamps else None,
            "last_candidate_set_et": max(timestamps).tz_convert("America/New_York").isoformat() if timestamps else None,
            "max_edge": numeric_max([row.get("max_edge") for row in gate_rows]),
            "best_call_edge_max": numeric_max([row.get("best_call_edge") for row in gate_rows]),
            "best_put_edge_max": numeric_max([row.get("best_put_edge") for row in gate_rows]),
            "events_above_min_edge": int(sum((row.get("above_min_edge_count") or 0) > 0 for row in gate_rows)),
            "filter_reasons": count_values(row.get("filter_reason") for row in gate_rows),
        }
    return out


def replay_decision(replay: pd.DataFrame) -> str:
    if replay.empty:
        return "blocked_no_replay_rows"
    entries = int((replay["action"] == "enter").sum())
    if entries > 0:
        return "historical_replay_produced_protocol101_entries"
    candidates = int((pd.to_numeric(replay["candidate_count"], errors="coerce").fillna(0) > 0).sum())
    if candidates > 0:
        return "historical_replay_had_candidates_but_protocol101_waited"
    return "historical_replay_no_candidates_above_edge_gate"


def write_report(path: Path, payload: dict[str, Any], replay: pd.DataFrame) -> None:
    summary = payload["summary"]
    timestamp_alignment = payload.get("timestamp_alignment", {})
    lines = [
        "# Protocol 161: May 2026 Historical Replay",
        "",
        "Frozen Protocol101 was replayed on historical OPRA + official SPX/VIX context for May 19-20, 2026. No model was retrained and no broker endpoint was called.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Timestamp alignment: `{timestamp_alignment.get('mode', 'historical-row-timestamp')}`",
        f"- Completed-minute lag minutes: `{timestamp_alignment.get('completed_minute_lag_minutes', 0)}`",
        f"- Session-scoped index context: `{payload.get('scoped_index_context', {}).get('enabled', False)}`",
        f"- Decision rows: `{summary['decision_rows']}`",
        f"- Entry signals: `{summary['entry_signals']}`",
        f"- Events with candidates after edge gate: `{summary['events_with_candidates']}`",
        f"- Candidate rows after edge gate: `{summary['candidate_rows']}`",
        f"- Context-not-ready rows: `{summary.get('context_not_ready_rows', 0)}`",
        f"- Max surface edge: `{summary['max_edge']}`",
        f"- Max call edge: `{summary['best_call_edge_max']}`",
        f"- Max put edge: `{summary['best_put_edge_max']}`",
        "",
        "## By Session",
        "",
        "| session | rows | entries | context-not-ready | events with candidates | max edge | max call edge | max put edge |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for session, row in summary["by_session"].items():
        lines.append(
            f"| {session} | {row['decision_rows']} | {row['entry_signals']} | {row.get('context_not_ready_rows', 0)} | {row['events_with_candidates']} | "
            f"{fmt(row['max_edge'])} | {fmt(row['best_call_edge_max'])} | {fmt(row['best_put_edge_max'])} |"
        )
    lines.extend(["", "## Live Log Comparison", ""])
    lines.append("| session | first live candidates ET | live candidate rows | live max edge | live above gate events | live filter reasons |")
    lines.append("|---|---|---:|---:|---:|---|")
    for session, row in payload["live_log_comparison"].items():
        lines.append(
            f"| {session} | {row.get('first_candidate_set_et') or ''} | {row['candidate_set_rows']} | {fmt(row['max_edge'])} | "
            f"{row['events_above_min_edge']} | `{row['filter_reasons']}` |"
        )
    if not replay.empty and (replay["action"] == "enter").any():
        lines.extend(["", "## Entry Signals", ""])
        cols = [
            "session",
            "decision_time",
            "selected_contract_id",
            "selected_right",
            "selected_offset_points",
            "selected_edge",
            "margin",
            "selected_ask",
        ]
        for row in replay[replay["action"] == "enter"][cols].to_dict("records"):
            lines.append(
                f"- `{row['session']}` `{row['decision_time']}` {row['selected_contract_id']} "
                f"edge={fmt(row['selected_edge'])} margin={fmt(row['margin'])} ask={fmt(row['selected_ask'])}"
            )
    path.write_text("\n".join(lines) + "\n")


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def int_or_none(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number


def row_context_ready(row: dict[str, Any]) -> bool | None:
    if "context_ready" not in row:
        return None
    return bool(row.get("context_ready"))


def numeric_max(values: Any) -> float | None:
    if values is None:
        return None
    try:
        series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    except Exception:
        return None
    if series.empty:
        return None
    return float(series.max())


def none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        text = str(value)
        return text if text else None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def object_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def count_values(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def fmt(value: Any) -> str:
    number = finite_or_none(value)
    if number is None:
        return ""
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
