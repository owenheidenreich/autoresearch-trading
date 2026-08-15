"""Decompose Protocol101 historical-vs-archived-IBKR feature/score drift.

This tool compares historical Protocol101 decision traces against archived IBKR
candidate-set diagnostics for the same session/minute. The archived June 2026
IBKR logs only contain the top live surface tokens, so this report is a
forensic decomposition, not a full candidate-universe parity proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
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
DEFAULT_HISTORICAL_TRACES = Path(
    "v4/audit/autoresearch/"
    "protocol101_live_contract_june2026_protocol161_completed_minute_lag1/"
    "decision_traces.jsonl"
)
DEFAULT_HISTORICAL_REPLAY = Path(
    "v4/audit/autoresearch/"
    "protocol101_live_contract_june2026_protocol161_completed_minute_lag1/"
    "replay_decisions.csv"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/"
    "protocol101_ibkr_historical_feature_decomposition"
)

UTC = timezone.utc
COMPARE_FIELDS = (
    "edge",
    "surface_action_score",
    "surface_flat_score",
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_frac",
    "iv",
    "delta",
    "gamma",
    "theta",
)


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


def _token_list(gate: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = gate.get("top_surface_tokens") or gate.get("top_rejected_contracts") or []
    if not isinstance(tokens, list):
        return []
    return [dict(token) for token in tokens if isinstance(token, dict)]


def _ensure_derived_quote_fields(token: dict[str, Any]) -> dict[str, Any]:
    out = dict(token)
    bid = _float(out.get("bid"))
    ask = _float(out.get("ask"))
    mid = _float(out.get("mid"))
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
        out["mid"] = mid
    if out.get("spread") is None and bid is not None and ask is not None:
        out["spread"] = ask - bid
    if out.get("spread_frac") is None:
        spread = _float(out.get("spread"))
        if spread is not None and mid is not None and mid > 0:
            out["spread_frac"] = spread / mid
    return out


def load_live_top_tokens(live_root: Path, dates: set[str]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    by_minute: dict[tuple[str, str], dict[str, Any]] = {}
    token_depths: Counter[int] = Counter()
    reasons: Counter[str] = Counter()
    missing_files: list[str] = []
    for session in sorted(dates):
        path = _live_log_path(live_root, session)
        if not path.exists():
            missing_files.append(str(path))
            continue
        with path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event_type") != "candidate_set":
                    continue
                minute = _minute_key(row.get("decision_timestamp_utc") or row.get("timestamp_utc"))
                if not minute:
                    continue
                gate = row.get("candidate_gate_diagnostics") or {}
                if not isinstance(gate, dict):
                    gate = {}
                tokens = [_ensure_derived_quote_fields(token) for token in _token_list(gate)]
                token_depths[len(tokens)] += 1
                reasons[str(gate.get("filter_reason") or "")] += 1
                snapshot = row.get("market_snapshot") or {}
                context = snapshot.get("context") or {} if isinstance(snapshot, dict) else {}
                option_nbbo = snapshot.get("option_nbbo") or {} if isinstance(snapshot, dict) else {}
                underlying = snapshot.get("underlying") or {} if isinstance(snapshot, dict) else {}
                freshness = option_nbbo.get("freshness") or {} if isinstance(option_nbbo, dict) else {}
                by_minute[(session, minute)] = {
                    "session": session,
                    "decision_minute_utc": minute,
                    "live_filter_reason": gate.get("filter_reason"),
                    "live_above_min_edge_count": gate.get("above_min_edge_count"),
                    "live_max_edge": gate.get("max_edge"),
                    "live_best_call_edge": gate.get("best_call_edge"),
                    "live_best_put_edge": gate.get("best_put_edge"),
                    "live_flat_score": gate.get("flat_score"),
                    "live_token_count": gate.get("token_count"),
                    "live_eligible_token_count": gate.get("eligible_token_count"),
                    "live_valid_score_count": gate.get("valid_score_count"),
                    "live_top_token_depth": len(tokens),
                    "live_context_ready": context.get("context_ready") if isinstance(context, dict) else None,
                    "live_context_minute_rows": context.get("context_minute_rows") if isinstance(context, dict) else None,
                    "live_context_span_minutes": context.get("context_span_minutes") if isinstance(context, dict) else None,
                    "live_option_quote_count": option_nbbo.get("quote_count") if isinstance(option_nbbo, dict) else None,
                    "live_option_quote_median_age_ms": freshness.get("median_ms") if isinstance(freshness, dict) else None,
                    "live_spx": underlying.get("spx") if isinstance(underlying, dict) else None,
                    "live_vix": underlying.get("vix") if isinstance(underlying, dict) else None,
                    "live_spx_age_ms": underlying.get("spx_quote_age_ms") if isinstance(underlying, dict) else None,
                    "live_vix_age_ms": underlying.get("vix_quote_age_ms") if isinstance(underlying, dict) else None,
                    "live_tokens": tokens,
                }
    return by_minute, {
        "missing_live_files": missing_files,
        "live_top_token_depths": {str(k): int(v) for k, v in sorted(token_depths.items())},
        "live_filter_reasons": {str(k): int(v) for k, v in sorted(reasons.items())},
    }


def load_historical_traces(path: Path, dates: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    by_minute: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            session = str(row.get("session") or "")
            if session not in dates:
                continue
            minute = _minute_key(row.get("decision_ts"))
            if not minute:
                continue
            universe = row.get("candidate_universe") or []
            if not isinstance(universe, list):
                universe = []
            candidates: dict[str, dict[str, Any]] = {}
            ranked = []
            for token in universe:
                if not isinstance(token, dict):
                    continue
                contract_id = str(token.get("contract_id") or "")
                if not contract_id:
                    continue
                clean = _ensure_derived_quote_fields(token)
                candidates[contract_id] = clean
                ranked.append(clean)
            ranked.sort(key=lambda item: _float(item.get("edge")) if _float(item.get("edge")) is not None else -math.inf, reverse=True)
            ranks = {str(token.get("contract_id")): idx + 1 for idx, token in enumerate(ranked)}
            selected = row.get("selected_contract") or {}
            selected_contract_id = ""
            if isinstance(selected, dict):
                selected_contract_id = str(selected.get("contract_id") or selected.get("candidate_id") or "")
            by_minute[(session, minute)] = {
                "session": session,
                "decision_minute_utc": minute,
                "hist_selected_action": row.get("selected_action"),
                "hist_selected_contract_id": selected_contract_id,
                "hist_block_reasons": row.get("block_reasons") or [],
                "hist_candidate_count": row.get("candidate_count"),
                "hist_feature_contract_version": row.get("feature_contract_version"),
                "hist_source_decision_ts": row.get("source_decision_ts"),
                "hist_candidate_universe_hash": row.get("candidate_universe_hash"),
                "hist_feature_hash": row.get("feature_hash"),
                "hist_score_hash": row.get("score_hash"),
                "hist_candidates": candidates,
                "hist_ranks": ranks,
            }
    return by_minute


def load_historical_entries(path: Path, dates: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session = str(row.get("session") or "")
            if session not in dates:
                continue
            if str(row.get("action") or "").lower() != "enter":
                continue
            minute = _minute_key(row.get("decision_time"))
            if not minute:
                continue
            entries[(session, minute)] = {
                "hist_entry_action": row.get("action"),
                "hist_entry_reason": row.get("reason"),
                "hist_entry_contract_id": row.get("selected_contract_id") or "",
                "hist_entry_right": row.get("selected_right") or "",
                "hist_entry_offset_points": row.get("selected_offset_points") or "",
                "hist_entry_edge": row.get("selected_edge") or "",
                "hist_entry_ask": row.get("selected_ask") or "",
                "hist_entry_max_edge": row.get("max_edge") or "",
                "hist_entry_best_call_edge": row.get("best_call_edge") or "",
                "hist_entry_best_put_edge": row.get("best_put_edge") or "",
                "hist_entry_margin": row.get("margin") or "",
                "hist_entry_threshold": row.get("threshold") or "",
            }
    return entries


def _delta_row(prefix: str, hist: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in COMPARE_FIELDS:
        h = _float(hist.get(field))
        l = _float(live.get(field))
        out[f"{prefix}{field}_hist"] = h
        out[f"{prefix}{field}_live"] = l
        out[f"{prefix}{field}_delta_hist_minus_live"] = h - l if h is not None and l is not None else None
    return out


def build_decomposition(
    historical: dict[tuple[str, str], dict[str, Any]],
    historical_entries: dict[tuple[str, str], dict[str, Any]],
    live: dict[tuple[str, str], dict[str, Any]],
    dates: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matched_top: list[dict[str, Any]] = []
    entry_rows: list[dict[str, Any]] = []
    unmatched_live_tokens: list[dict[str, Any]] = []

    for key in sorted(set(historical) | set(live)):
        session, minute = key
        if session not in dates:
            continue
        hist = historical.get(key)
        live_minute = live.get(key)
        entry = historical_entries.get(key)
        if not hist or not live_minute:
            if entry:
                entry_rows.append(
                    {
                        "session": session,
                        "decision_minute_utc": minute,
                        **entry,
                        "classification": "missing_live_or_historical_trace",
                    }
                )
            continue
        hist_candidates = hist["hist_candidates"]
        hist_ranks = hist["hist_ranks"]
        live_tokens = live_minute.get("live_tokens") or []
        live_ids = {str(token.get("contract_id") or "") for token in live_tokens}

        for live_rank, token in enumerate(live_tokens, start=1):
            contract_id = str(token.get("contract_id") or "")
            hist_token = hist_candidates.get(contract_id)
            if not hist_token:
                unmatched_live_tokens.append(
                    {
                        "session": session,
                        "decision_minute_utc": minute,
                        "live_rank": live_rank,
                        "contract_id": contract_id,
                        "live_edge": token.get("edge"),
                        "live_right": token.get("right"),
                        "live_offset_points": token.get("offset_points"),
                        "reason": "live_top_token_missing_from_historical_candidate_universe",
                    }
                )
                continue
            row = {
                "session": session,
                "decision_minute_utc": minute,
                "contract_id": contract_id,
                "live_rank": live_rank,
                "hist_rank_by_edge": hist_ranks.get(contract_id),
                "rank_delta_hist_minus_live": (hist_ranks.get(contract_id) - live_rank) if hist_ranks.get(contract_id) is not None else None,
                "hist_selected_action": hist.get("hist_selected_action"),
                "hist_selected_contract_id": hist.get("hist_selected_contract_id"),
                "is_historical_selected_contract": contract_id == hist.get("hist_selected_contract_id"),
                "is_historical_entry_minute": key in historical_entries,
                "live_filter_reason": live_minute.get("live_filter_reason"),
                "live_above_min_edge_count": live_minute.get("live_above_min_edge_count"),
                "live_option_quote_count": live_minute.get("live_option_quote_count"),
                "live_option_quote_median_age_ms": live_minute.get("live_option_quote_median_age_ms"),
                "live_spx": live_minute.get("live_spx"),
                "live_vix": live_minute.get("live_vix"),
                "live_spx_age_ms": live_minute.get("live_spx_age_ms"),
                "live_vix_age_ms": live_minute.get("live_vix_age_ms"),
                "hist_right": hist_token.get("right"),
                "live_right": token.get("right"),
                "hist_offset_points": hist_token.get("offset_points"),
                "live_offset_points": token.get("offset_points"),
                "hist_token_feature_hash": hist_token.get("token_feature_hash"),
                "live_has_token_feature_hash": bool(token.get("token_feature_hash")),
            }
            row.update(_delta_row("", hist_token, token))
            matched_top.append(row)

        if entry:
            selected_id = str(entry.get("hist_entry_contract_id") or hist.get("hist_selected_contract_id") or "")
            live_token = next((token for token in live_tokens if str(token.get("contract_id") or "") == selected_id), None)
            hist_token = hist_candidates.get(selected_id)
            row = {
                "session": session,
                "decision_minute_utc": minute,
                **entry,
                "classification": "selected_contract_seen_in_live_top_tokens" if live_token else "selected_contract_not_in_live_top5_tokens",
                "live_filter_reason": live_minute.get("live_filter_reason"),
                "live_top_token_depth": live_minute.get("live_top_token_depth"),
                "live_token_count": live_minute.get("live_token_count"),
                "live_option_quote_count": live_minute.get("live_option_quote_count"),
                "live_option_quote_median_age_ms": live_minute.get("live_option_quote_median_age_ms"),
                "hist_selected_contract_in_historical_universe": bool(hist_token),
                "hist_rank_by_edge": hist_ranks.get(selected_id),
                "live_rank_if_seen": None,
                "unknown_reason_if_not_seen": "" if live_token else "archived_live_log_only_kept_top5_tokens",
            }
            if live_token:
                row["live_rank_if_seen"] = next(
                    idx
                    for idx, token in enumerate(live_tokens, start=1)
                    if str(token.get("contract_id") or "") == selected_id
                )
                row.update(_delta_row("selected_", hist_token or {}, live_token))
            entry_rows.append(row)

    return matched_top, entry_rows, unmatched_live_tokens


def summarize(matched: list[dict[str, Any]], entries: list[dict[str, Any]], unmatched: list[dict[str, Any]], live_meta: dict[str, Any], dates: set[str]) -> dict[str, Any]:
    by_session: dict[str, dict[str, Any]] = {}
    for session in sorted(dates):
        session_matched = [row for row in matched if row.get("session") == session]
        session_entries = [row for row in entries if row.get("session") == session]
        entry_seen = [row for row in session_entries if row.get("classification") == "selected_contract_seen_in_live_top_tokens"]
        by_session[session] = {
            "matched_live_top_tokens": len(session_matched),
            "historical_entry_minutes": len(session_entries),
            "entry_selected_contract_seen_in_live_top5": len(entry_seen),
            "entry_selected_contract_not_in_live_top5": len(session_entries) - len(entry_seen),
            "median_abs_edge_delta_on_matched_live_top_tokens": _safe_median(
                [abs(v) for row in session_matched if (v := _float(row.get("edge_delta_hist_minus_live"))) is not None]
            ),
            "median_abs_ask_delta_on_matched_live_top_tokens": _safe_median(
                [abs(v) for row in session_matched if (v := _float(row.get("ask_delta_hist_minus_live"))) is not None]
            ),
            "median_abs_iv_delta_on_matched_live_top_tokens": _safe_median(
                [abs(v) for row in session_matched if (v := _float(row.get("iv_delta_hist_minus_live"))) is not None]
            ),
            "median_abs_delta_delta_on_matched_live_top_tokens": _safe_median(
                [abs(v) for row in session_matched if (v := _float(row.get("delta_delta_hist_minus_live"))) is not None]
            ),
            "median_abs_theta_delta_on_matched_live_top_tokens": _safe_median(
                [abs(v) for row in session_matched if (v := _float(row.get("theta_delta_hist_minus_live"))) is not None]
            ),
            "median_abs_selected_contract_edge_delta_when_seen": _safe_median(
                [abs(v) for row in entry_seen if (v := _float(row.get("selected_edge_delta_hist_minus_live"))) is not None]
            ),
        }
    field_summaries = []
    for field in COMPARE_FIELDS:
        deltas = [
            abs(v)
            for row in matched
            if (v := _float(row.get(f"{field}_delta_hist_minus_live"))) is not None
        ]
        field_summaries.append(
            {
                "field": field,
                "matched_rows": len(deltas),
                "median_abs_delta": _safe_median(deltas),
                "p90_abs_delta": _percentile(deltas, 0.90),
                "max_abs_delta": max(deltas) if deltas else None,
            }
        )
    return {
        "protocol": "protocol101_ibkr_historical_feature_decomposition",
        "scope": "offline_existing_artifacts_only",
        "dates": sorted(dates),
        "limitation": "archived_live_logs_keep_only_top5_surface_tokens_and_not_full_token_feature_vectors",
        "matched_live_top_tokens": len(matched),
        "historical_entry_minutes": len(entries),
        "historical_entry_selected_contract_seen_in_live_top5": len([row for row in entries if row.get("classification") == "selected_contract_seen_in_live_top_tokens"]),
        "historical_entry_selected_contract_not_in_live_top5": len([row for row in entries if row.get("classification") == "selected_contract_not_in_live_top5_tokens"]),
        "unmatched_live_top_tokens": len(unmatched),
        "by_session": by_session,
        "field_delta_summary": field_summaries,
        "live_metadata": live_meta,
        "repair_decision": {
            "primary": "capture_or_reconstruct_full_per_candidate_features_before_changing_thresholds",
            "why": "historical and live disagree at surface edge scoring before Protocol101 entry and before broker execution",
            "next_code_target": "make archived/captured IBKR rows replayable into Protocol101DecisionTraceV1 with full token features",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (dict, list)):
                    clean[field] = json.dumps(value, sort_keys=True)
                elif value is None:
                    clean[field] = ""
                else:
                    clean[field] = value
            writer.writerow(clean)


def write_report(out_dir: Path, summary: dict[str, Any], historical_traces: Path, historical_replay: Path, live_root: Path) -> None:
    lines = [
        "# Protocol101 IBKR/Historical Feature Decomposition",
        "",
        "This is an offline artifact-only decomposition. It did not call IBKR, download paid data, train, tune thresholds, or change the paper default.",
        "",
        f"- Historical traces: `{historical_traces}`",
        f"- Historical replay: `{historical_replay}`",
        f"- Archived IBKR live root: `{live_root}`",
        f"- Limitation: `{summary['limitation']}`",
        "",
        "## Headline",
        "",
        f"- Matched live top-token rows: `{summary['matched_live_top_tokens']}`",
        f"- Historical entry minutes: `{summary['historical_entry_minutes']}`",
        f"- Historical selected contract seen in live top 5: `{summary['historical_entry_selected_contract_seen_in_live_top5']}`",
        f"- Historical selected contract not in live top 5: `{summary['historical_entry_selected_contract_not_in_live_top5']}`",
        "",
        "## By Session",
        "",
        "| session | matched live top tokens | hist entries | selected seen in live top5 | selected not in live top5 | median abs matched edge delta | median abs matched IV delta | median abs selected edge delta when seen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for session, row in summary["by_session"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    session,
                    str(row["matched_live_top_tokens"]),
                    str(row["historical_entry_minutes"]),
                    str(row["entry_selected_contract_seen_in_live_top5"]),
                    str(row["entry_selected_contract_not_in_live_top5"]),
                    _fmt(row["median_abs_edge_delta_on_matched_live_top_tokens"]),
                    _fmt(row["median_abs_iv_delta_on_matched_live_top_tokens"]),
                    _fmt(row["median_abs_selected_contract_edge_delta_when_seen"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Field Delta Summary", ""])
    lines.append("| field | matched rows | median abs delta | p90 abs delta | max abs delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary["field_delta_summary"]:
        lines.append(
            f"| {row['field']} | {row['matched_rows']} | {_fmt(row['median_abs_delta'])} | {_fmt(row['p90_abs_delta'])} | {_fmt(row['max_abs_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The old IBKR logs do not contain the full live token feature vectors, so the precise root feature cannot be proven from these days alone. But the decomposition confirms that the disagreement is at the surface-edge scoring layer: live top tokens and historical tokens for the same contract/minute can differ materially in edge, IV, delta, theta, bid/ask, and rank.",
            "",
            "The next fix should not be a threshold change. It should make captured IBKR rows replayable into `Protocol101DecisionTraceV1` with full candidate features, then use same-input replay to determine whether the remaining issue is live feature construction, historical transformation, or model calibration on the repaired contract.",
            "",
            "## Repair Plan",
            "",
            "1. Extend live/captured candidate logging to persist full scored token universe, not just top 5 diagnostics.",
            "2. Add an offline `ibkr-captured-replay` path that reconstructs `SurfaceDecision`/`Protocol101DecisionTraceV1` from captured IBKR quotes and context.",
            "3. Diff exact feature vectors field-by-field for historical selected contracts and live top-ranked contracts.",
            "4. Repair the shared feature contract where semantics differ, especially IV/Greek repair, quote timing, strike ladder membership, and OMAR/index context.",
            "5. Rebuild live-style historical data and rerun Protocol161/Protocol162 plus the pre-live sanity gate before paper-submit resumes.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`",
            "- `matched_live_top_token_feature_deltas.csv`",
            "- `historical_entry_selected_contract_decomposition.csv`",
            "- `field_delta_summary.csv`",
            "- `unmatched_live_top_tokens.csv`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    parser.add_argument("--historical-traces", type=Path, default=DEFAULT_HISTORICAL_TRACES)
    parser.add_argument("--historical-replay", type=Path, default=DEFAULT_HISTORICAL_REPLAY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    args = parser.parse_args()

    dates = set(args.dates)
    if not args.historical_traces.exists():
        raise SystemExit(f"missing historical traces: {args.historical_traces}")
    if not args.historical_replay.exists():
        raise SystemExit(f"missing historical replay: {args.historical_replay}")

    live, live_meta = load_live_top_tokens(args.live_root, dates)
    historical = load_historical_traces(args.historical_traces, dates)
    entries = load_historical_entries(args.historical_replay, dates)
    matched, entry_rows, unmatched = build_decomposition(historical, entries, live, dates)
    summary = summarize(matched, entry_rows, unmatched, live_meta, dates)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(out_dir / "matched_live_top_token_feature_deltas.csv", matched)
    write_csv(out_dir / "historical_entry_selected_contract_decomposition.csv", entry_rows)
    write_csv(out_dir / "unmatched_live_top_tokens.csv", unmatched)
    write_csv(out_dir / "field_delta_summary.csv", summary["field_delta_summary"])
    write_report(out_dir, summary, args.historical_traces, args.historical_replay, args.live_root)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary["repair_decision"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
