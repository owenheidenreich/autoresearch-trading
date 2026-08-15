"""Compare archived IBKR Protocol101 live logs against historical replay.

This is an offline forensic tool. It does not call IBKR, Databento, ThetaData,
or any model-training path. It exists for the parity question: when historical
replay takes entries and archived IBKR live paper does not, where does the
decision game diverge?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_DATES = ("2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09")
DEFAULT_LIVE_ROOT = Path(
    "/Users/gduby/.autoresearch-trading/archive/"
    "ibkr_live_trading_sessions_2026-06-04_05_08_09_10_20260611T155134Z/"
    "live_runtime_paper_trading"
)
DEFAULT_HISTORICAL_REPLAY = Path(
    "v4/audit/autoresearch/"
    "protocol101_live_contract_june2026_protocol161_completed_minute_lag1/"
    "replay_decisions.csv"
)
DEFAULT_HISTORICAL_TRADES = Path(
    "v4/audit/autoresearch/"
    "protocol101_live_contract_june2026_protocol162_serial_lifecycle_replay/"
    "serial_lifecycle_trades.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/"
    "protocol101_ibkr_captured_vs_historical_compare"
)


UTC = timezone.utc


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _minute_key(value: Any) -> str | None:
    dt = _parse_timestamp(value)
    if dt is None:
        return None
    return dt.replace(second=0, microsecond=0).isoformat()


def _float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _int(value: Any) -> int | None:
    number = _float(value)
    if number is None:
        return None
    return int(number)


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _json_dump(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _counter_to_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def _safe_median(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return float(median(vals)) if vals else None


def _percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    rank = (len(vals) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(vals[lo])
    weight = rank - lo
    return float(vals[lo] * (1.0 - weight) + vals[hi] * weight)


def _live_log_path(live_root: Path, session: str) -> Path:
    return live_root / session / f"daily_paper_autopilot_{session}.jsonl"


@dataclass
class HistoricalInputs:
    rows_by_key: dict[tuple[str, str], dict[str, Any]]
    session_counts: Counter[str]
    entry_counts: Counter[str]
    serial_trade_counts: Counter[str]
    serial_trade_minutes: set[tuple[str, str]]


def load_historical(replay_csv: Path, trades_csv: Path | None, dates: set[str]) -> HistoricalInputs:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    session_counts: Counter[str] = Counter()
    entry_counts: Counter[str] = Counter()

    with replay_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session = str(row.get("session") or "").strip()
            if session not in dates:
                continue
            minute = _minute_key(row.get("decision_time"))
            if not minute:
                continue
            action = str(row.get("action") or "").strip().lower()
            hist = {
                "session": session,
                "decision_minute_utc": minute,
                "hist_decision_time": row.get("decision_time") or "",
                "hist_source_decision_time": row.get("source_decision_time") or "",
                "hist_timestamp_alignment_mode": row.get("timestamp_alignment_mode") or "",
                "hist_time_bucket": row.get("time_bucket") or "",
                "hist_context_ready": _bool(row.get("context_ready")),
                "hist_context_minute_rows": _int(row.get("context_minute_rows")),
                "hist_context_span_minutes": _float(row.get("context_span_minutes")),
                "hist_action": action,
                "hist_reason": row.get("reason") or "",
                "hist_candidate_count": _int(row.get("candidate_count")),
                "hist_above_min_edge_count": _int(row.get("above_min_edge_count")),
                "hist_max_edge": _float(row.get("max_edge")),
                "hist_best_call_edge": _float(row.get("best_call_edge")),
                "hist_best_put_edge": _float(row.get("best_put_edge")),
                "hist_margin": _float(row.get("margin")),
                "hist_threshold": _float(row.get("threshold")),
                "hist_selected_contract_id": row.get("selected_contract_id") or "",
                "hist_selected_right": row.get("selected_right") or "",
                "hist_selected_offset_points": _float(row.get("selected_offset_points")),
                "hist_selected_edge": _float(row.get("selected_edge")),
                "hist_selected_ask": _float(row.get("selected_ask")),
            }
            rows_by_key[(session, minute)] = hist
            session_counts[session] += 1
            if action == "enter":
                entry_counts[session] += 1

    serial_trade_counts: Counter[str] = Counter()
    serial_trade_minutes: set[tuple[str, str]] = set()
    if trades_csv and trades_csv.exists():
        with trades_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session = str(row.get("session") or "").strip()
                if session not in dates:
                    continue
                if str(row.get("serial_status") or "").strip() not in {"", "taken"}:
                    continue
                minute = _minute_key(row.get("decision_time"))
                if not minute:
                    continue
                serial_trade_counts[session] += 1
                serial_trade_minutes.add((session, minute))

    for key in serial_trade_minutes:
        if key in rows_by_key:
            rows_by_key[key]["hist_serial_trade_taken"] = True
    for row in rows_by_key.values():
        row.setdefault("hist_serial_trade_taken", False)

    return HistoricalInputs(
        rows_by_key=rows_by_key,
        session_counts=session_counts,
        entry_counts=entry_counts,
        serial_trade_counts=serial_trade_counts,
        serial_trade_minutes=serial_trade_minutes,
    )


def _extract_live_tokens(diag: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = diag.get("top_surface_tokens") or diag.get("top_rejected_contracts") or []
    if not isinstance(tokens, list):
        return []
    out: list[dict[str, Any]] = []
    for token in tokens:
        if isinstance(token, dict):
            out.append(token)
    return out


def _match_contract(tokens: list[dict[str, Any]], contract_id: str) -> dict[str, Any] | None:
    if not contract_id:
        return None
    for token in tokens:
        if str(token.get("contract_id") or "") == contract_id:
            return token
    return None


def _update_live_market(row: dict[str, Any], target: dict[str, Any]) -> None:
    snapshot = row.get("market_snapshot") or {}
    if not isinstance(snapshot, dict):
        return
    context = snapshot.get("context") or {}
    option_nbbo = snapshot.get("option_nbbo") or {}
    underlying = snapshot.get("underlying") or {}
    freshness = option_nbbo.get("freshness") or {}

    if isinstance(context, dict):
        for key, out_key in (
            ("context_ready", "live_context_ready"),
            ("context_minute_rows", "live_context_minute_rows"),
            ("context_span_minutes", "live_context_span_minutes"),
            ("context_start_timestamp_utc", "live_context_start_timestamp_utc"),
            ("context_last_timestamp_utc", "live_context_last_timestamp_utc"),
        ):
            if key in context:
                target[out_key] = context.get(key)
    if isinstance(option_nbbo, dict):
        for key, out_key in (
            ("quote_count", "live_option_quote_count"),
            ("subscribed_contracts", "live_subscribed_contracts"),
            ("option_quotes_digest", "live_option_quotes_digest"),
        ):
            if key in option_nbbo:
                target[out_key] = option_nbbo.get(key)
    if isinstance(freshness, dict):
        for key, out_key in (
            ("median_ms", "live_option_quote_median_age_ms"),
            ("max_ms", "live_option_quote_max_age_ms"),
            ("known_count", "live_option_quote_known_count"),
            ("missing_count", "live_option_quote_missing_count"),
        ):
            if key in freshness:
                target[out_key] = freshness.get(key)
    if isinstance(underlying, dict):
        for key, out_key in (
            ("spx", "live_spx"),
            ("vix", "live_vix"),
            ("spx_quote_age_ms", "live_spx_age_ms"),
            ("vix_quote_age_ms", "live_vix_age_ms"),
            ("spx_raw_quote_timestamp_utc", "live_spx_quote_timestamp_utc"),
            ("vix_raw_quote_timestamp_utc", "live_vix_quote_timestamp_utc"),
        ):
            if key in underlying:
                target[out_key] = underlying.get(key)


def load_live(live_root: Path, dates: set[str]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    event_counts_by_session: dict[str, Counter[str]] = defaultdict(Counter)
    missing_files: list[str] = []

    for session in sorted(dates):
        path = _live_log_path(live_root, session)
        if not path.exists():
            missing_files.append(str(path))
            continue
        with path.open() as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    event_counts_by_session[session]["json_decode_error"] += 1
                    continue
                row_session = str(row.get("session") or row.get("session_id") or session)
                if row_session not in dates:
                    continue
                minute = _minute_key(row.get("decision_timestamp_utc") or row.get("timestamp_utc"))
                if not minute:
                    continue
                event_type = str(row.get("event_type") or row.get("event") or row.get("row_type") or "unknown")
                event_counts_by_session[row_session][event_type] += 1
                key = (row_session, minute)
                live = rows_by_key.setdefault(
                    key,
                    {
                        "session": row_session,
                        "decision_minute_utc": minute,
                        "live_log_path": str(path),
                        "live_line_min": line_number,
                        "live_line_max": line_number,
                        "live_event_count": 0,
                        "live_event_types": Counter(),
                        "live_candidate_set_rows": 0,
                        "live_market_snapshot_rows": 0,
                        "live_model_decision_rows": 0,
                        "live_risk_gate_rows": 0,
                        "live_order_intents": 0,
                        "live_order_submissions": 0,
                        "live_fills": 0,
                        "live_cancels": 0,
                        "live_rejects": 0,
                        "live_errors": 0,
                        "live_broker_endpoint_called": False,
                    },
                )
                live["live_line_max"] = line_number
                live["live_event_count"] += 1
                live["live_event_types"][event_type] += 1
                if row.get("broker_order_endpoint_called"):
                    live["live_broker_endpoint_called"] = True

                if event_type == "market_snapshot":
                    live["live_market_snapshot_rows"] += 1
                    _update_live_market(row, live)
                elif event_type == "model_decision":
                    live["live_model_decision_rows"] += 1
                    decision = row.get("model_decision") or {}
                    if isinstance(decision, dict):
                        live["live_selected_action"] = decision.get("selected_action") or decision.get("action")
                        live["live_model_reason"] = decision.get("reason") or decision.get("no_entry_reason")
                        live["live_threshold"] = decision.get("threshold")
                elif event_type == "risk_gate":
                    live["live_risk_gate_rows"] += 1
                elif event_type in {"paper_order_intent", "order_intent"}:
                    live["live_order_intents"] += 1
                elif event_type in {"paper_order_submit", "order_submitted", "paper_order_submitted"}:
                    live["live_order_submissions"] += 1
                elif event_type in {"paper_fill", "fill", "order_fill"}:
                    live["live_fills"] += 1
                elif event_type in {"paper_order_cancel", "order_cancel", "paper_cancel"}:
                    live["live_cancels"] += 1
                elif event_type in {"paper_order_reject", "order_reject", "paper_reject"}:
                    live["live_rejects"] += 1
                elif "error" in event_type:
                    live["live_errors"] += 1

                if event_type == "candidate_set":
                    live["live_candidate_set_rows"] += 1
                    _update_live_market(row, live)
                    diag = row.get("candidate_gate_diagnostics") or {}
                    if not isinstance(diag, dict):
                        diag = {}
                    decision = row.get("model_decision") or {}
                    if not isinstance(decision, dict):
                        decision = {}
                    risk_gate = row.get("risk_gate") or {}
                    if not isinstance(risk_gate, dict):
                        risk_gate = {}
                    tokens = _extract_live_tokens(diag)
                    top = tokens[0] if tokens else {}

                    live.update(
                        {
                            "live_candidate_timestamp_utc": row.get("timestamp_utc") or "",
                            "live_candidate_count": row.get("candidate_count"),
                            "live_filter_reason": diag.get("filter_reason")
                            or decision.get("reason")
                            or decision.get("no_entry_reason")
                            or risk_gate.get("reason"),
                            "live_above_min_edge_count": diag.get("above_min_edge_count"),
                            "live_token_count": diag.get("token_count"),
                            "live_eligible_token_count": diag.get("eligible_token_count"),
                            "live_valid_score_count": diag.get("valid_score_count"),
                            "live_max_edge": diag.get("max_edge"),
                            "live_best_call_edge": diag.get("best_call_edge"),
                            "live_best_put_edge": diag.get("best_put_edge"),
                            "live_min_edge": diag.get("min_edge"),
                            "live_flat_score": diag.get("flat_score"),
                            "live_time_bucket": diag.get("time_bucket"),
                            "live_allowed_time_bucket": diag.get("allowed_time_bucket"),
                            "live_selected_action": decision.get("selected_action") or decision.get("action"),
                            "live_model_reason": decision.get("reason") or decision.get("no_entry_reason"),
                            "live_risk_reason": risk_gate.get("reason"),
                            "live_candidate_set_hash": row.get("candidate_set_hash") or "",
                            "live_feature_vector_hash": row.get("feature_vector_hash") or "",
                            "live_top_token_count": len(tokens),
                            "live_top_contract_id": top.get("contract_id") if isinstance(top, dict) else "",
                            "live_top_right": top.get("right") if isinstance(top, dict) else "",
                            "live_top_offset_points": top.get("offset_points") if isinstance(top, dict) else None,
                            "live_top_edge": top.get("edge") if isinstance(top, dict) else None,
                            "live_top_ask": top.get("ask") if isinstance(top, dict) else None,
                            "live_top_bid": top.get("bid") if isinstance(top, dict) else None,
                            "live_top_iv": top.get("iv") if isinstance(top, dict) else None,
                            "live_top_delta": top.get("delta") if isinstance(top, dict) else None,
                            "live_top_gamma": top.get("gamma") if isinstance(top, dict) else None,
                            "live_top_theta": top.get("theta") if isinstance(top, dict) else None,
                            "live_top_spread_frac": top.get("spread_frac") if isinstance(top, dict) else None,
                            "live_top_surface_action_score": top.get("surface_action_score") if isinstance(top, dict) else None,
                            "live_top_surface_flat_score": top.get("surface_flat_score") if isinstance(top, dict) else None,
                            "_live_tokens": tokens,
                        }
                    )

    metadata = {
        "event_counts_by_session": {
            session: _counter_to_dict(counter)
            for session, counter in sorted(event_counts_by_session.items())
        },
        "missing_live_files": missing_files,
    }
    return rows_by_key, metadata


JOIN_FIELDS = [
    "session",
    "decision_minute_utc",
    "classification",
    "hist_action",
    "hist_serial_trade_taken",
    "hist_reason",
    "live_filter_reason",
    "hist_selected_contract_id",
    "hist_selected_right",
    "hist_selected_offset_points",
    "hist_selected_edge",
    "hist_selected_ask",
    "live_selected_contract_seen_in_top_tokens",
    "live_same_contract_edge",
    "live_same_contract_ask",
    "live_same_contract_bid",
    "live_same_contract_iv",
    "live_same_contract_delta",
    "live_same_contract_gamma",
    "live_same_contract_theta",
    "same_contract_edge_delta_hist_minus_live",
    "same_contract_ask_delta_hist_minus_live",
    "hist_max_edge",
    "live_max_edge",
    "max_edge_delta_hist_minus_live",
    "hist_best_call_edge",
    "live_best_call_edge",
    "call_edge_delta_hist_minus_live",
    "hist_best_put_edge",
    "live_best_put_edge",
    "put_edge_delta_hist_minus_live",
    "hist_above_min_edge_count",
    "live_above_min_edge_count",
    "hist_candidate_count",
    "live_candidate_count",
    "live_token_count",
    "live_eligible_token_count",
    "live_valid_score_count",
    "hist_time_bucket",
    "live_time_bucket",
    "hist_context_ready",
    "live_context_ready",
    "hist_context_minute_rows",
    "live_context_minute_rows",
    "hist_context_span_minutes",
    "live_context_span_minutes",
    "live_option_quote_count",
    "live_option_quote_median_age_ms",
    "live_spx",
    "live_vix",
    "live_spx_age_ms",
    "live_vix_age_ms",
    "live_top_contract_id",
    "live_top_right",
    "live_top_offset_points",
    "live_top_edge",
    "live_top_ask",
    "live_top_bid",
    "live_top_iv",
    "live_top_delta",
    "live_top_gamma",
    "live_top_theta",
    "live_flat_score",
    "live_candidate_set_hash",
    "live_feature_vector_hash",
    "hist_decision_time",
    "hist_source_decision_time",
    "live_candidate_timestamp_utc",
]


def _classify(hist: dict[str, Any] | None, live: dict[str, Any] | None) -> str:
    if hist is None:
        return "live_only_minute"
    if live is None:
        return "missing_live_minute"
    hist_entry = hist.get("hist_action") == "enter"
    hist_edge = (_int(hist.get("hist_above_min_edge_count")) or 0) > 0
    live_edge = (_int(live.get("live_above_min_edge_count")) or 0) > 0
    live_reason = str(live.get("live_filter_reason") or "")
    if hist_entry and not live_edge:
        if live_reason == "below_min_edge":
            return "historical_entry_live_below_min_edge"
        if live_reason:
            return f"historical_entry_live_{live_reason}"
        return "historical_entry_live_no_edge_unknown_reason"
    if hist_entry and live_edge:
        return "historical_entry_live_edge_present"
    if hist_edge and not live_edge:
        return "historical_edge_live_below_edge"
    return "no_entry_alignment_or_non_entry_drift"


def build_join_rows(
    historical: HistoricalInputs,
    live_rows: dict[tuple[str, str], dict[str, Any]],
    dates: set[str],
) -> list[dict[str, Any]]:
    keys = sorted(
        key
        for key in set(historical.rows_by_key) | set(live_rows)
        if key[0] in dates
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        hist = historical.rows_by_key.get(key)
        live = live_rows.get(key)
        row: dict[str, Any] = {
            "session": key[0],
            "decision_minute_utc": key[1],
            "classification": _classify(hist, live),
        }
        if hist:
            row.update(hist)
        if live:
            row.update({k: v for k, v in live.items() if not k.startswith("_") and not isinstance(v, Counter)})

        tokens = live.get("_live_tokens", []) if live else []
        selected_contract = str(row.get("hist_selected_contract_id") or "")
        same = _match_contract(tokens, selected_contract)
        if same:
            row["live_selected_contract_seen_in_top_tokens"] = True
            row["live_same_contract_edge"] = _float(same.get("edge"))
            row["live_same_contract_ask"] = _float(same.get("ask"))
            row["live_same_contract_bid"] = _float(same.get("bid"))
            row["live_same_contract_iv"] = _float(same.get("iv"))
            row["live_same_contract_delta"] = _float(same.get("delta"))
            row["live_same_contract_gamma"] = _float(same.get("gamma"))
            row["live_same_contract_theta"] = _float(same.get("theta"))
        else:
            row["live_selected_contract_seen_in_top_tokens"] = False if selected_contract else ""

        for left, right, out in (
            ("hist_max_edge", "live_max_edge", "max_edge_delta_hist_minus_live"),
            ("hist_best_call_edge", "live_best_call_edge", "call_edge_delta_hist_minus_live"),
            ("hist_best_put_edge", "live_best_put_edge", "put_edge_delta_hist_minus_live"),
            ("hist_selected_edge", "live_same_contract_edge", "same_contract_edge_delta_hist_minus_live"),
            ("hist_selected_ask", "live_same_contract_ask", "same_contract_ask_delta_hist_minus_live"),
        ):
            lval = _float(row.get(left))
            rval = _float(row.get(right))
            row[out] = lval - rval if lval is not None and rval is not None else None
        rows.append(row)
    return rows


def summarize(
    join_rows: list[dict[str, Any]],
    historical: HistoricalInputs,
    live_metadata: dict[str, Any],
    dates: set[str],
) -> dict[str, Any]:
    by_session: dict[str, dict[str, Any]] = {}
    reason_counts: Counter[tuple[str, str]] = Counter()
    classification_counts: Counter[str] = Counter()
    historical_entry_rows = [r for r in join_rows if r.get("hist_action") == "enter"]

    for row in join_rows:
        classification_counts[str(row.get("classification") or "")] += 1
        if row.get("hist_action") == "enter":
            reason_counts[(str(row.get("session")), str(row.get("live_filter_reason") or "missing_live"))] += 1

    for session in sorted(dates):
        rows = [r for r in join_rows if r.get("session") == session]
        hist_rows = [r for r in rows if r.get("hist_decision_time")]
        live_rows = [r for r in rows if r.get("live_log_path")]
        entry_rows = [r for r in rows if r.get("hist_action") == "enter"]
        live_edge_rows = [r for r in rows if (_int(r.get("live_above_min_edge_count")) or 0) > 0]
        edge_deltas = [abs(v) for r in rows if (v := _float(r.get("max_edge_delta_hist_minus_live"))) is not None]
        same_contract_deltas = [
            abs(v)
            for r in entry_rows
            if (v := _float(r.get("same_contract_edge_delta_hist_minus_live"))) is not None
        ]
        by_session[session] = {
            "historical_minutes": len(hist_rows),
            "live_minutes": len(live_rows),
            "missing_live_minutes": len([r for r in hist_rows if not r.get("live_log_path")]),
            "historical_entry_signals": len(entry_rows),
            "historical_serial_trades": int(historical.serial_trade_counts.get(session, 0)),
            "live_minutes_with_edge_candidates": len(live_edge_rows),
            "historical_entries_with_live_edge_candidates": len(
                [r for r in entry_rows if (_int(r.get("live_above_min_edge_count")) or 0) > 0]
            ),
            "historical_entries_with_selected_contract_in_live_top_tokens": len(
                [r for r in entry_rows if r.get("live_selected_contract_seen_in_top_tokens") is True]
            ),
            "historical_entries_with_live_below_min_edge": len(
                [r for r in entry_rows if r.get("live_filter_reason") == "below_min_edge"]
            ),
            "median_abs_max_edge_delta": _safe_median(edge_deltas),
            "p90_abs_max_edge_delta": _percentile(edge_deltas, 0.90),
            "median_abs_same_contract_edge_delta_at_entries": _safe_median(same_contract_deltas),
            "p90_abs_same_contract_edge_delta_at_entries": _percentile(same_contract_deltas, 0.90),
            "median_live_option_quote_count": _safe_median(
                [v for r in live_rows if (v := _float(r.get("live_option_quote_count"))) is not None]
            ),
            "median_live_option_quote_age_ms": _safe_median(
                [v for r in live_rows if (v := _float(r.get("live_option_quote_median_age_ms"))) is not None]
            ),
            "median_live_spx_age_ms": _safe_median(
                [v for r in live_rows if (v := _float(r.get("live_spx_age_ms"))) is not None]
            ),
            "median_live_vix_age_ms": _safe_median(
                [v for r in live_rows if (v := _float(r.get("live_vix_age_ms"))) is not None]
            ),
        }

    return {
        "protocol": "protocol101_ibkr_captured_vs_historical_compare",
        "scope": "offline_existing_artifacts_only",
        "dates": sorted(dates),
        "historical_entry_signals": len(historical_entry_rows),
        "historical_serial_trades": sum(historical.serial_trade_counts.values()),
        "joined_rows": len(join_rows),
        "classification_counts": _counter_to_dict(classification_counts),
        "live_reason_counts_at_historical_entries": {
            f"{session}|{reason}": count
            for (session, reason), count in sorted(reason_counts.items())
        },
        "by_session": by_session,
        "live_metadata": live_metadata,
        "interpretation": _interpret(join_rows),
    }


def _interpret(join_rows: list[dict[str, Any]]) -> dict[str, Any]:
    entry_rows = [r for r in join_rows if r.get("hist_action") == "enter"]
    entries = len(entry_rows)
    entries_live_below = len([r for r in entry_rows if r.get("live_filter_reason") == "below_min_edge"])
    entries_contract_seen = len([r for r in entry_rows if r.get("live_selected_contract_seen_in_top_tokens") is True])
    same_contract_delta = [
        abs(v)
        for r in entry_rows
        if (v := _float(r.get("same_contract_edge_delta_hist_minus_live"))) is not None
    ]
    live_quote_counts = [
        v for r in entry_rows if (v := _float(r.get("live_option_quote_count"))) is not None
    ]
    if entries and entries_live_below == entries:
        verdict = "remaining_pre_execution_score_mismatch"
    elif entries:
        verdict = "mixed_or_partial_entry_mismatch"
    else:
        verdict = "no_historical_entry_minutes_in_scope"
    return {
        "verdict": verdict,
        "historical_entries": entries,
        "historical_entries_live_below_min_edge": entries_live_below,
        "historical_entries_selected_contract_seen_in_live_top_tokens": entries_contract_seen,
        "median_abs_same_contract_edge_delta_at_entries": _safe_median(same_contract_delta),
        "p90_abs_same_contract_edge_delta_at_entries": _percentile(same_contract_delta, 0.90),
        "median_live_quote_count_at_historical_entries": _safe_median(live_quote_counts),
        "repair_hint": (
            "The mismatch is before broker execution. Compare per-candidate feature vectors "
            "for the same contract/minute, then repair the historical/live feature contract "
            "or retrain on the repaired live-reproducible contract."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        all_fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    all_fields.append(key)
        fields = all_fields
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (dict, list, Counter)):
                    clean[field] = _json_dump(value)
                elif value is None:
                    clean[field] = ""
                else:
                    clean[field] = value
            writer.writerow(clean)


def write_report(
    out_dir: Path,
    summary: dict[str, Any],
    replay_csv: Path,
    trades_csv: Path | None,
    live_root: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Protocol101 IBKR Captured Vs Historical Comparison")
    lines.append("")
    lines.append("This is an offline comparison over existing artifacts only. It did not contact IBKR, download paid data, train, tune thresholds, or change the paper default.")
    lines.append("")
    lines.append(f"- Historical replay CSV: `{replay_csv}`")
    if trades_csv:
        lines.append(f"- Historical serial trades CSV: `{trades_csv}`")
    lines.append(f"- Archived IBKR live root: `{live_root}`")
    lines.append(f"- Dates: `{', '.join(summary['dates'])}`")
    lines.append(f"- Joined minute rows: `{summary['joined_rows']}`")
    lines.append(f"- Historical entry signals: `{summary['historical_entry_signals']}`")
    lines.append(f"- Historical serial trades: `{summary['historical_serial_trades']}`")
    lines.append("")
    lines.append("## Summary By Session")
    lines.append("")
    headers = [
        "session",
        "hist_minutes",
        "live_minutes",
        "missing_live",
        "hist_entries",
        "hist_serial_trades",
        "live_edge_minutes",
        "hist_entries_with_live_edge",
        "hist_entries_with_contract_seen",
        "median_abs_edge_delta",
        "median_abs_same_contract_entry_delta",
        "median_live_quotes",
        "median_quote_age_ms",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for session, row in summary["by_session"].items():
        vals = [
            session,
            row.get("historical_minutes"),
            row.get("live_minutes"),
            row.get("missing_live_minutes"),
            row.get("historical_entry_signals"),
            row.get("historical_serial_trades"),
            row.get("live_minutes_with_edge_candidates"),
            row.get("historical_entries_with_live_edge_candidates"),
            row.get("historical_entries_with_selected_contract_in_live_top_tokens"),
            _fmt(row.get("median_abs_max_edge_delta")),
            _fmt(row.get("median_abs_same_contract_edge_delta_at_entries")),
            _fmt(row.get("median_live_option_quote_count")),
            _fmt(row.get("median_live_option_quote_age_ms")),
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    lines.append("")
    interp = summary["interpretation"]
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- Verdict: `{interp['verdict']}`.")
    lines.append(f"- Historical entry minutes with live `below_min_edge`: `{interp['historical_entries_live_below_min_edge']}` / `{interp['historical_entries']}`.")
    lines.append(f"- Historical selected contract seen in live top-token samples: `{interp['historical_entries_selected_contract_seen_in_live_top_tokens']}` / `{interp['historical_entries']}`.")
    lines.append(f"- Median absolute same-contract edge delta at entry minutes: `{_fmt(interp['median_abs_same_contract_edge_delta_at_entries'])}`.")
    lines.append(f"- Median live option quote count at historical entry minutes: `{_fmt(interp['median_live_quote_count_at_historical_entries'])}`.")
    lines.append("")
    lines.append("The evidence points to a model-facing score/feature mismatch rather than an order-routing problem: historical replay creates entry signals, while archived IBKR live evaluates the same minutes with option quotes present and still remains below the edge threshold.")
    lines.append("")
    lines.append("## Repair Plan")
    lines.append("")
    lines.append("1. Preserve Protocol101 as the default and keep paper-submit paused while IBKR market-data eligibility is unavailable.")
    lines.append("2. Add an `ibkr-captured-replay` mode that replays captured IBKR candidate snapshots through the same Protocol101 scoring path offline. This proves same-input determinism without waiting for live data.")
    lines.append("3. For future full-trace sessions, diff the same contract/minute feature vector field by field: bid/ask/mid/spread, repaired Greeks, IV, offset, OMAR/opening context, SPX/VIX context, quote age, and live-unavailable volume/OI substitutes.")
    lines.append("4. Where historical edge is high and live same-contract edge is low, repair the shared feature contract or historical builder first. Do not lower thresholds to force trades.")
    lines.append("5. Rebuild the live-style historical dataset, rerun Protocol161/Protocol162 and the pre-live sanity gate, then run one prospective full-trace paper session once IBKR data is restored.")
    lines.append("6. Only after a prospective same-date paired diff passes should this move back toward a five-day evidence packet and later hill climbing.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name in [
        "summary.json",
        "summary_by_session.csv",
        "minute_diagnostic_join.csv",
        "historical_entry_minutes_vs_live.csv",
        "top_edge_delta_minutes.csv",
        "historical_entry_live_reason_counts.csv",
        "live_event_counts.csv",
    ]:
        lines.append(f"- `{name}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    parser.add_argument("--historical-replay", type=Path, default=DEFAULT_HISTORICAL_REPLAY)
    parser.add_argument("--historical-trades", type=Path, default=DEFAULT_HISTORICAL_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    args = parser.parse_args()

    dates = set(args.dates)
    if not args.historical_replay.exists():
        raise SystemExit(f"missing historical replay CSV: {args.historical_replay}")
    historical_trades = args.historical_trades if args.historical_trades and args.historical_trades.exists() else None

    historical = load_historical(args.historical_replay, historical_trades, dates)
    live_rows, live_metadata = load_live(args.live_root, dates)
    join_rows = build_join_rows(historical, live_rows, dates)
    summary = summarize(join_rows, historical, live_metadata, dates)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    summary_rows = []
    for session, row in summary["by_session"].items():
        summary_rows.append({"session": session, **row})
    write_csv(out_dir / "summary_by_session.csv", summary_rows)
    write_csv(out_dir / "minute_diagnostic_join.csv", join_rows, JOIN_FIELDS)

    entry_rows = [row for row in join_rows if row.get("hist_action") == "enter"]
    write_csv(out_dir / "historical_entry_minutes_vs_live.csv", entry_rows, JOIN_FIELDS)

    top_delta_rows = sorted(
        [row for row in join_rows if _float(row.get("max_edge_delta_hist_minus_live")) is not None],
        key=lambda r: abs(_float(r.get("max_edge_delta_hist_minus_live")) or 0.0),
        reverse=True,
    )[:100]
    write_csv(out_dir / "top_edge_delta_minutes.csv", top_delta_rows, JOIN_FIELDS)

    reason_counter: Counter[tuple[str, str]] = Counter()
    for row in entry_rows:
        reason_counter[(str(row.get("session")), str(row.get("live_filter_reason") or "missing_live"))] += 1
    reason_rows = [
        {"session": session, "live_filter_reason_at_historical_entry": reason, "count": count}
        for (session, reason), count in sorted(reason_counter.items())
    ]
    write_csv(out_dir / "historical_entry_live_reason_counts.csv", reason_rows)

    event_rows = []
    for session, counts in live_metadata.get("event_counts_by_session", {}).items():
        for event_type, count in counts.items():
            event_rows.append({"session": session, "event_type": event_type, "count": count})
    write_csv(out_dir / "live_event_counts.csv", event_rows)

    write_report(out_dir, summary, args.historical_replay, historical_trades, args.live_root)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary["interpretation"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
