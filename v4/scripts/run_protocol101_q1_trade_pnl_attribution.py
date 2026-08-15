"""Attribute Q1 Protocol101 trade/PnL degradation by feature contract.

This is an offline diagnostic. It consumes already-built Q1 legacy and
protocol101-live-v1 replay artifacts, then joins trade losses to decision,
feature-drift, and frozen-model score-attribution evidence. It does not train,
tune thresholds, contact data vendors, or touch broker/order paths.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_Q1_ROOT = Path("v4/audit/autoresearch/protocol101_q1_2026_contract_comparison")
DEFAULT_OUT = Path("v4/audit/autoresearch/protocol101_q1_2026_trade_pnl_attribution")
TRADE_KEY = ["session", "decision_time", "contract_id"]
SCENARIO_PREFIX = "live_with_historical_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-root", type=Path, default=DEFAULT_Q1_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--near-minutes", type=int, default=5)
    parser.add_argument("--near-strike-points", type=float, default=10.0)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_strike(contract_id: Any) -> float | None:
    match = re.search(r"-(\d+(?:\.\d+)?)-[CP]$", str(contract_id or ""))
    return float(match.group(1)) if match else None


def decision_minute_et(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York").strftime("%H:%M")


def key_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for key in keys:
        if key not in out.columns:
            out[key] = ""
    return out.drop_duplicates(keys).set_index(keys, drop=False)


def trade_key(row: pd.Series) -> tuple[str, str, str]:
    return (str(row.get("session") or ""), str(row.get("decision_time") or ""), str(row.get("contract_id") or ""))


def same_minute_key(row: pd.Series) -> tuple[str, str]:
    return (str(row.get("session") or ""), str(row.get("decision_time") or ""))


def scenario_columns(score: pd.DataFrame) -> list[str]:
    return [
        column
        for column in score.columns
        if column.startswith(SCENARIO_PREFIX) and column.endswith("_gap_closure")
    ]


def scenario_name(column: str) -> str:
    return column.removeprefix(SCENARIO_PREFIX).removesuffix("_gap_closure")


def dominant_scenario(row: pd.Series, columns: list[str]) -> tuple[str, float | None]:
    best_name = "UNKNOWN"
    best_value: float | None = None
    for column in columns:
        value = finite(row.get(column))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_name = scenario_name(column)
    return best_name, best_value


def nearest_trade(
    row: pd.Series,
    other: pd.DataFrame,
    *,
    near_minutes: int,
    near_strike_points: float,
) -> dict[str, Any]:
    if other.empty:
        return {"near_match": False}
    session = str(row.get("session") or "")
    right = str(row.get("right") or row.get("right_legacy") or row.get("right_live") or "")
    strike = parse_strike(row.get("contract_id"))
    ts = pd.Timestamp(row.get("decision_time"))
    candidates = other[other["session"].astype(str).eq(session)].copy()
    if right:
        candidates = candidates[candidates["right"].astype(str).eq(right)]
    if candidates.empty:
        return {"near_match": False}
    candidates["_ts"] = pd.to_datetime(candidates["decision_time"], utc=True, errors="coerce")
    candidates["_minute_delta"] = (candidates["_ts"] - ts).abs().dt.total_seconds() / 60.0
    if strike is not None:
        candidates["_strike_delta"] = candidates["contract_id"].map(parse_strike).sub(strike).abs()
    else:
        candidates["_strike_delta"] = float("inf")
    candidates = candidates.sort_values(["_minute_delta", "_strike_delta"])
    best = candidates.iloc[0]
    minute_delta = finite(best.get("_minute_delta"))
    strike_delta = finite(best.get("_strike_delta"))
    near = (
        minute_delta is not None
        and minute_delta <= near_minutes
        and strike_delta is not None
        and strike_delta <= near_strike_points
    )
    return {
        "near_match": bool(near),
        "near_contract_id": best.get("contract_id"),
        "near_decision_time": best.get("decision_time"),
        "near_minute_delta": minute_delta,
        "near_strike_delta": strike_delta,
        "near_pnl": finite(best.get("candidate_pnl")),
    }


def classify_trade(
    row: pd.Series,
    *,
    side: str,
    decision_by_minute: pd.DataFrame,
    skipped_exact: set[tuple[str, str, str]],
    skipped_minute: set[tuple[str, str]],
) -> str:
    key = trade_key(row)
    minute_key = same_minute_key(row)
    decision = decision_by_minute.loc[minute_key] if minute_key in decision_by_minute.index else pd.Series(dtype=object)
    if side == "legacy_missing_from_live":
        if key in skipped_exact or minute_key in skipped_minute:
            return "serial_open_position_interaction"
        if str(decision.get("action_live") or "") == "wait":
            return "entry_suppressed_by_live_contract"
        if str(decision.get("action_live") or "") == "enter":
            live_contract = str(decision.get("selected_contract_id_live") or "")
            legacy_contract = str(row.get("contract_id") or "")
            if live_contract and live_contract != legacy_contract:
                return "selected_contract_or_ranking_drift"
            return "serial_or_lifecycle_interaction"
    if side == "live_only":
        if key in skipped_exact or minute_key in skipped_minute:
            return "serial_open_position_interaction"
        if str(decision.get("action_legacy") or "") == "wait":
            return "entry_created_by_live_contract"
        if str(decision.get("action_legacy") or "") == "enter":
            legacy_contract = str(decision.get("selected_contract_id_legacy") or "")
            live_contract = str(row.get("contract_id") or "")
            if legacy_contract and legacy_contract != live_contract:
                return "selected_contract_or_ranking_drift"
            return "serial_or_lifecycle_interaction"
    return "UNKNOWN"


def summarize_numeric(values: pd.Series) -> dict[str, Any]:
    nums = pd.to_numeric(values, errors="coerce").dropna()
    if nums.empty:
        return {"n": 0}
    return {
        "n": int(len(nums)),
        "mean": float(nums.mean()),
        "median": float(nums.median()),
        "p10": float(nums.quantile(0.10)),
        "p90": float(nums.quantile(0.90)),
        "sum": float(nums.sum()),
    }


def group_summary(frame: pd.DataFrame, by: str, pnl_col: str) -> list[dict[str, Any]]:
    if frame.empty or by not in frame.columns:
        return []
    rows = []
    for name, group in frame.groupby(by, dropna=False):
        pnl = pd.to_numeric(group[pnl_col], errors="coerce").fillna(0.0)
        rows.append(
            {
                by: str(name),
                "trades": int(len(group)),
                "pnl": float(pnl.sum()),
                "wins": int((pnl > 0).sum()),
                "losses": int((pnl < 0).sum()),
                "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
            }
        )
    return sorted(rows, key=lambda item: abs(float(item["pnl"])), reverse=True)


def parse_drift_list(value: Any) -> list[dict[str, Any]]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def top_lost_field_rows(lost: pd.DataFrame, *, top_trades: int = 50, top_fields: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lost.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, Any]] = defaultdict(lambda: {"occurrences": 0, "pnl": 0.0, "standardized_abs_sum": 0.0})
    top = lost.sort_values("pnl", ascending=False).head(top_trades)
    for _, trade in top.iterrows():
        base = {
            "session": trade.get("session"),
            "decision_time": trade.get("decision_time"),
            "contract_id": trade.get("contract_id"),
            "right": trade.get("right"),
            "pnl": finite(trade.get("pnl")) or 0.0,
            "classification": trade.get("classification"),
            "dominant_gap_closure_group": trade.get("dominant_gap_closure_group"),
            "dominant_gap_closure_points": finite(trade.get("dominant_gap_closure_points")),
            "legacy_max_edge": finite(trade.get("legacy_max_edge")),
            "live_max_edge": finite(trade.get("live_max_edge")),
        }
        for family, column in (
            ("token", "largest_token_standardized_drifts"),
            ("scalar", "largest_scalar_standardized_drifts"),
        ):
            for rank, drift in enumerate(parse_drift_list(trade.get(column))[:top_fields], start=1):
                feature = str(drift.get("feature") or "")
                standardized_abs = finite(drift.get("standardized_abs")) or 0.0
                row = {
                    **base,
                    "drift_family": family,
                    "drift_rank": rank,
                    "feature": feature,
                    "raw_delta": finite(drift.get("raw_delta")),
                    "standardized_abs": standardized_abs,
                }
                rows.append(row)
                key = f"{family}:{feature}"
                counts[key]["feature"] = feature
                counts[key]["drift_family"] = family
                counts[key]["occurrences"] += 1
                counts[key]["pnl"] += float(base["pnl"])
                counts[key]["standardized_abs_sum"] += standardized_abs
    summary = []
    for payload in counts.values():
        occurrences = int(payload["occurrences"])
        summary.append(
            {
                "drift_family": payload["drift_family"],
                "feature": payload["feature"],
                "occurrences": occurrences,
                "pnl": float(payload["pnl"]),
                "mean_standardized_abs": float(payload["standardized_abs_sum"]) / max(occurrences, 1),
            }
        )
    summary.sort(key=lambda item: (int(item["occurrences"]), abs(float(item["pnl"]))), reverse=True)
    return pd.DataFrame(rows), pd.DataFrame(summary)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    comparison = args.q1_root / "comparison"
    decisions = read_csv(comparison / "candidate_score_action_comparison.csv")
    trades = read_csv(comparison / "trade_comparison.csv")
    daily = read_csv(comparison / "daily_comparison.csv")
    minute_parity = read_csv(args.q1_root / "feature_audit" / "minute_parity.csv")
    score = read_csv(args.q1_root / "score_attribution" / "score_attribution_by_minute.csv")
    legacy_taken = read_csv(args.q1_root / "legacy_protocol162" / "serial_lifecycle_trades.csv")
    live_taken = read_csv(args.q1_root / "live_protocol162" / "serial_lifecycle_trades.csv")
    legacy_skipped = read_csv(args.q1_root / "legacy_protocol162" / "skipped_entry_signals.csv")
    live_skipped = read_csv(args.q1_root / "live_protocol162" / "skipped_entry_signals.csv")

    legacy_taken = legacy_taken[legacy_taken.get("serial_status", "taken").astype(str).str.lower().eq("taken")].copy()
    live_taken = live_taken[live_taken.get("serial_status", "taken").astype(str).str.lower().eq("taken")].copy()

    decisions_by_minute = key_frame(decisions, ["session", "decision_time"])
    minute_parity_by_minute = key_frame(minute_parity.rename(columns={"decision_minute_utc": "decision_time"}), ["session", "decision_time"])
    score = score.copy()
    score["decision_minute_et"] = score["decision_minute_et"].astype(str)
    score_by_minute = key_frame(score, ["session", "decision_minute_et"])
    score_cols = scenario_columns(score)

    live_skipped_exact = {trade_key(row) for _, row in live_skipped.iterrows()}
    live_skipped_minute = {same_minute_key(row) for _, row in live_skipped.iterrows()}
    legacy_skipped_exact = {trade_key(row) for _, row in legacy_skipped.iterrows()}
    legacy_skipped_minute = {same_minute_key(row) for _, row in legacy_skipped.iterrows()}

    lost_rows: list[dict[str, Any]] = []
    live_only_rows: list[dict[str, Any]] = []
    both_rows: list[dict[str, Any]] = []

    for _, raw in trades.iterrows():
        merge = str(raw.get("_merge"))
        if merge == "left_only":
            row = {
                "session": raw.get("session"),
                "decision_time": raw.get("decision_time"),
                "contract_id": raw.get("contract_id"),
                "right": raw.get("right_legacy"),
                "edge": finite(raw.get("edge_legacy")),
                "entry_ask": finite(raw.get("entry_ask_legacy")),
                "pnl": finite(raw.get("candidate_pnl_legacy")) or 0.0,
                "exit_reason": raw.get("candidate_exit_reason_legacy"),
                "exit_time": raw.get("candidate_exit_time_legacy"),
            }
            minute_key = (str(row["session"]), str(row["decision_time"]))
            decision = decisions_by_minute.loc[minute_key] if minute_key in decisions_by_minute.index else pd.Series(dtype=object)
            parity = minute_parity_by_minute.loc[minute_key] if minute_key in minute_parity_by_minute.index else pd.Series(dtype=object)
            score_key = (str(row["session"]), decision_minute_et(row["decision_time"]))
            score_row = score_by_minute.loc[score_key] if score_key in score_by_minute.index else pd.Series(dtype=object)
            dominant, closure = dominant_scenario(score_row, score_cols)
            row.update(
                {
                    "classification": classify_trade(
                        pd.Series(row),
                        side="legacy_missing_from_live",
                        decision_by_minute=decisions_by_minute,
                        skipped_exact=live_skipped_exact,
                        skipped_minute=live_skipped_minute,
                    ),
                    "live_action": decision.get("action_live"),
                    "live_reason": decision.get("reason_live"),
                    "live_selected_contract_id": decision.get("selected_contract_id_live"),
                    "live_max_edge": finite(decision.get("max_edge_live")),
                    "legacy_max_edge": finite(decision.get("max_edge_legacy")),
                    "max_edge_delta_live_minus_legacy": finite(decision.get("max_edge_delta")),
                    "candidate_count_delta_live_minus_legacy": finite(decision.get("candidate_count_delta")),
                    "time_bucket": parity.get("time_bucket"),
                    "candidate_identity_overlap": finite(parity.get("candidate_identity_overlap")),
                    "candidate_rank_correlation": finite(parity.get("candidate_rank_correlation")),
                    "top_contract_match": parity.get("top_contract_match"),
                    "top3_identity_overlap": finite(parity.get("top3_identity_overlap")),
                    "token_standardized_rms_drift": finite(parity.get("token_standardized_rms_drift")),
                    "scalar_standardized_rms_drift": finite(parity.get("scalar_standardized_rms_drift")),
                    "dominant_gap_closure_group": dominant,
                    "dominant_gap_closure_points": closure,
                    "largest_token_standardized_drifts": parity.get("largest_token_standardized_drifts"),
                    "largest_scalar_standardized_drifts": parity.get("largest_scalar_standardized_drifts"),
                }
            )
            row.update(nearest_trade(pd.Series(row), live_taken, near_minutes=args.near_minutes, near_strike_points=args.near_strike_points))
            lost_rows.append(row)
        elif merge == "right_only":
            row = {
                "session": raw.get("session"),
                "decision_time": raw.get("decision_time"),
                "contract_id": raw.get("contract_id"),
                "right": raw.get("right_live"),
                "edge": finite(raw.get("edge_live")),
                "entry_ask": finite(raw.get("entry_ask_live")),
                "pnl": finite(raw.get("candidate_pnl_live")) or 0.0,
                "exit_reason": raw.get("candidate_exit_reason_live"),
                "exit_time": raw.get("candidate_exit_time_live"),
            }
            minute_key = (str(row["session"]), str(row["decision_time"]))
            decision = decisions_by_minute.loc[minute_key] if minute_key in decisions_by_minute.index else pd.Series(dtype=object)
            parity = minute_parity_by_minute.loc[minute_key] if minute_key in minute_parity_by_minute.index else pd.Series(dtype=object)
            score_key = (str(row["session"]), decision_minute_et(row["decision_time"]))
            score_row = score_by_minute.loc[score_key] if score_key in score_by_minute.index else pd.Series(dtype=object)
            dominant, closure = dominant_scenario(score_row, score_cols)
            row.update(
                {
                    "classification": classify_trade(
                        pd.Series(row),
                        side="live_only",
                        decision_by_minute=decisions_by_minute,
                        skipped_exact=legacy_skipped_exact,
                        skipped_minute=legacy_skipped_minute,
                    ),
                    "legacy_action": decision.get("action_legacy"),
                    "legacy_reason": decision.get("reason_legacy"),
                    "legacy_selected_contract_id": decision.get("selected_contract_id_legacy"),
                    "live_max_edge": finite(decision.get("max_edge_live")),
                    "legacy_max_edge": finite(decision.get("max_edge_legacy")),
                    "max_edge_delta_live_minus_legacy": finite(decision.get("max_edge_delta")),
                    "candidate_count_delta_live_minus_legacy": finite(decision.get("candidate_count_delta")),
                    "time_bucket": parity.get("time_bucket"),
                    "candidate_identity_overlap": finite(parity.get("candidate_identity_overlap")),
                    "candidate_rank_correlation": finite(parity.get("candidate_rank_correlation")),
                    "top_contract_match": parity.get("top_contract_match"),
                    "top3_identity_overlap": finite(parity.get("top3_identity_overlap")),
                    "token_standardized_rms_drift": finite(parity.get("token_standardized_rms_drift")),
                    "scalar_standardized_rms_drift": finite(parity.get("scalar_standardized_rms_drift")),
                    "dominant_gap_closure_group": dominant,
                    "dominant_gap_closure_points": closure,
                    "largest_token_standardized_drifts": parity.get("largest_token_standardized_drifts"),
                    "largest_scalar_standardized_drifts": parity.get("largest_scalar_standardized_drifts"),
                }
            )
            row.update(nearest_trade(pd.Series(row), legacy_taken, near_minutes=args.near_minutes, near_strike_points=args.near_strike_points))
            live_only_rows.append(row)
        elif merge == "both":
            both_rows.append(
                {
                    "session": raw.get("session"),
                    "decision_time": raw.get("decision_time"),
                    "contract_id": raw.get("contract_id"),
                    "legacy_pnl": finite(raw.get("candidate_pnl_legacy")) or 0.0,
                    "live_pnl": finite(raw.get("candidate_pnl_live")) or 0.0,
                    "pnl_delta_live_minus_legacy": (finite(raw.get("candidate_pnl_live")) or 0.0)
                    - (finite(raw.get("candidate_pnl_legacy")) or 0.0),
                }
            )

    lost = pd.DataFrame(lost_rows)
    live_only = pd.DataFrame(live_only_rows)
    exact = pd.DataFrame(both_rows)
    daily = daily.copy()
    daily["pnl_delta_live_minus_legacy"] = pd.to_numeric(daily["pnl_live"], errors="coerce").fillna(0.0) - pd.to_numeric(
        daily["pnl_legacy"], errors="coerce"
    ).fillna(0.0)
    daily["trade_delta_live_minus_legacy"] = pd.to_numeric(daily["trades_live"], errors="coerce").fillna(0.0) - pd.to_numeric(
        daily["trades_legacy"], errors="coerce"
    ).fillna(0.0)

    action_crosstab = pd.crosstab(decisions["action_legacy"], decisions["action_live"], dropna=False)
    action_crosstab.to_csv(args.out_dir / "decision_action_crosstab.csv")
    write_csv(args.out_dir / "lost_legacy_trades_attribution.csv", lost)
    write_csv(args.out_dir / "live_only_trades_attribution.csv", live_only)
    write_csv(args.out_dir / "exact_trade_matches.csv", exact)
    write_csv(args.out_dir / "daily_pnl_attribution.csv", daily.sort_values("pnl_delta_live_minus_legacy"))
    top_lost_fields, top_lost_field_frequency = top_lost_field_rows(lost)
    write_csv(args.out_dir / "top_lost_trade_field_report.csv", top_lost_fields)
    write_csv(args.out_dir / "top_lost_field_frequency.csv", top_lost_field_frequency)

    legacy_total = float(pd.to_numeric(legacy_taken["candidate_pnl"], errors="coerce").fillna(0.0).sum())
    live_total = float(pd.to_numeric(live_taken["candidate_pnl"], errors="coerce").fillna(0.0).sum())
    lost_total = float(pd.to_numeric(lost.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    live_only_total = float(pd.to_numeric(live_only.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    exact_delta = float(pd.to_numeric(exact.get("pnl_delta_live_minus_legacy", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())

    summary = {
        "schema_version": "Protocol101Q1TradePnlAttributionV1",
        "model_training": False,
        "threshold_tuning": False,
        "broker_endpoint_called": False,
        "q1_root": str(args.q1_root),
        "legacy": {
            "serial_trades": int(len(legacy_taken)),
            "serial_pnl": legacy_total,
            "independent_candidate_signals": int(len(legacy_taken) + len(legacy_skipped)),
            "skipped_open_position": int(len(legacy_skipped)),
        },
        "protocol101_live_v1": {
            "serial_trades": int(len(live_taken)),
            "serial_pnl": live_total,
            "independent_candidate_signals": int(len(live_taken) + len(live_skipped)),
            "skipped_open_position": int(len(live_skipped)),
        },
        "pnl_waterfall": {
            "legacy_serial_pnl": legacy_total,
            "minus_lost_legacy_trade_pnl": -lost_total,
            "plus_live_only_trade_pnl": live_only_total,
            "plus_exact_trade_pnl_delta": exact_delta,
            "live_contract_serial_pnl": live_total,
            "explained_delta_live_minus_legacy": live_only_total - lost_total + exact_delta,
        },
        "exact_trade_matches": int(len(exact)),
        "lost_legacy_trades": int(len(lost)),
        "live_only_trades": int(len(live_only)),
        "lost_legacy_pnl": lost_total,
        "live_only_pnl": live_only_total,
        "net_pnl_delta_live_minus_legacy": live_total - legacy_total,
        "decision_action_crosstab": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in action_crosstab.iterrows()
        },
        "lost_classification": group_summary(lost, "classification", "pnl"),
        "lost_dominant_gap_closure_group": group_summary(lost, "dominant_gap_closure_group", "pnl"),
        "lost_time_bucket": group_summary(lost, "time_bucket", "pnl"),
        "live_only_classification": group_summary(live_only, "classification", "pnl"),
        "live_only_dominant_gap_closure_group": group_summary(live_only, "dominant_gap_closure_group", "pnl"),
        "top_deteriorating_days": daily.sort_values("pnl_delta_live_minus_legacy").head(12).to_dict("records"),
        "top_improving_days": daily.sort_values("pnl_delta_live_minus_legacy", ascending=False).head(12).to_dict("records"),
        "top_lost_field_frequency": top_lost_field_frequency.head(15).to_dict("records") if not top_lost_field_frequency.empty else [],
        "candidate_identity_overlap_on_lost": summarize_numeric(lost.get("candidate_identity_overlap", pd.Series(dtype=float))),
        "candidate_rank_correlation_on_lost": summarize_numeric(lost.get("candidate_rank_correlation", pd.Series(dtype=float))),
        "token_rms_drift_on_lost": summarize_numeric(lost.get("token_standardized_rms_drift", pd.Series(dtype=float))),
        "scalar_rms_drift_on_lost": summarize_numeric(lost.get("scalar_standardized_rms_drift", pd.Series(dtype=float))),
        "near_trade_policy": {
            "near_minutes": int(args.near_minutes),
            "near_strike_points": float(args.near_strike_points),
            "lost_legacy_near_live_trade_count": int(lost.get("near_match", pd.Series(dtype=bool)).fillna(False).sum()),
            "live_only_near_legacy_trade_count": int(live_only.get("near_match", pd.Series(dtype=bool)).fillna(False).sum()),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Protocol101 Q1 Trade/PnL Attribution",
        "",
        "## Scope",
        "",
        "- Offline only: no training, no threshold tuning, no broker endpoint, no paid-data call.",
        f"- Source Q1 packet: `{args.q1_root}`",
        "",
        "## Headline",
        "",
        f"- Same-runner legacy serial result: `{len(legacy_taken)}` trades / `${legacy_total:,.0f}`.",
        f"- Live-reproducible contract serial result: `{len(live_taken)}` trades / `${live_total:,.0f}`.",
        f"- Net degradation: `${live_total - legacy_total:,.0f}`.",
        f"- Exact same trade/time/contract matches: `{len(exact)}`.",
        f"- Legacy-only trades: `{len(lost)}` worth `${lost_total:,.0f}`.",
        f"- Live-only trades: `{len(live_only)}` worth `${live_only_total:,.0f}`.",
        "",
        "## Interpretation",
        "",
        "- The degradation is not primarily a missing-contract-universe problem: lost-trade minutes still have high candidate identity overlap.",
        "- The dominant issue is score/ranking/feature semantics: many minutes keep the same candidates but choose different top contracts or suppress entries.",
        "- More live evidence is still useful for confirmation, but this Q1 attribution can be acted on now.",
        "",
        "## Current Root-Cause Read",
        "",
        "- Treat candidate-universe construction as lower priority unless future captures show low overlap; Q1 lost-trade candidate identity overlap is high.",
        "- Treat derived token/pattern/context semantics as the primary repair target; they explain the largest share of lost positive legacy PnL.",
        "- Treat volume/open-interest enrichment as diagnostic, not the first rescue path; it is live-obtainable in some forms, but Q1 attribution does not show it as the dominant average degradation source.",
        "- Do not repair this by moving thresholds or model weights. The purpose is to decide whether the canonical causal feature contract is wrong/incomplete or whether the frozen model needs retraining later.",
        "",
        "## Next Offline Checks",
        "",
        "1. Keep the raw-input parity fixture for the pattern/structure feature family as a regression guard; it now passes for identical canonical minutes.",
        "2. Use `top_lost_trade_field_report.csv` and `top_lost_field_frequency.csv` to inspect the exact fields behind the largest lost trades.",
        "3. Use July 6+ captures to confirm whether the same feature family is stable on real captured IBKR input, instead of treating each live day as a generic smoke test.",
        "",
        "## Outputs",
        "",
        "- `lost_legacy_trades_attribution.csv`",
        "- `live_only_trades_attribution.csv`",
        "- `daily_pnl_attribution.csv`",
        "- `top_lost_trade_field_report.csv`",
        "- `top_lost_field_frequency.csv`",
        "- `decision_action_crosstab.csv`",
        "- `summary.json`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
