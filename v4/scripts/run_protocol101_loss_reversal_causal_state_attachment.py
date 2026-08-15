"""Protocol101 loss-reversal direct causal-state attachment.

The strategy-selection packet identified loss-reversal as the first hypothesis,
but the first diagnostic showed a label mismatch: final losing trades are not
necessarily losing at the later blocked-signal time. This packet attaches direct
quote-path state at that blocked signal for as many rows as local normalized
quotes allow. It is still research-only and does not train a model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.scripts.run_protocol101_loss_reversal_exit_gate_diagnostic import load_loss_reversal_candidates  # noqa: E402
from v4.scripts.run_protocol101_strategy_forensics_packet import _fmt_money  # noqa: E402


ROLE_LABEL = "AUDIT_PROTOCOL101_LOSS_REVERSAL_CAUSAL_STATE_ATTACHMENT_V1"
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_LOSS_REVERSAL_CAUSAL_STATE_ATTACHMENT_V1.md"
)
DEFAULT_SLOT_COST_DIR = Path("v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1")
DEFAULT_NORMALIZED_DIRS = (Path("v4/normalized_official_context"), Path("v4/normalized"))
CONTRACT_MULTIPLIER = 100.0


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def candidate_paths(normalized_dirs: list[Path], session: str) -> list[Path]:
    paths: list[Path] = []
    for normalized_dir in normalized_dirs:
        preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
        fallback = sorted(
            path
            for path in normalized_dir.glob(f"*{session}*.parquet")
            if path not in preferred and "derived_context" not in path.name
        )
        paths.extend(preferred + fallback)
    return paths


def read_session_quotes(
    normalized_dirs: list[Path],
    session: str,
    contract_ids: set[str],
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    for path in candidate_paths(normalized_dirs, session):
        try:
            frame = pd.read_parquet(path, columns=["quote_time", "contract_id", "bid", "ask", "underlying_price"])
        except Exception as first_error:
            try:
                frame = pd.read_parquet(path, columns=["quote_time", "contract_id", "bid", "ask"])
                frame["underlying_price"] = np.nan
            except Exception as second_error:
                errors.append(
                    {
                        "session": session,
                        "path": str(path),
                        "error": f"{type(second_error).__name__}: {str(second_error)[:180]}",
                        "first_error": f"{type(first_error).__name__}: {str(first_error)[:180]}",
                    }
                )
                continue
        frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True, errors="coerce")
        frame["contract_id"] = frame["contract_id"].astype(str)
        frame = frame[frame["contract_id"].isin(contract_ids)].copy()
        if frame.empty:
            errors.append({"session": session, "path": str(path), "error": "no_requested_contract_ids"})
            continue
        for column in ["bid", "ask", "underlying_price"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame[
            frame["quote_time"].notna()
            & frame["bid"].notna()
            & frame["ask"].notna()
            & (frame["bid"] >= 0.0)
            & (frame["ask"] > 0.0)
            & (frame["ask"] >= frame["bid"])
        ].copy()
        if frame.empty:
            errors.append({"session": session, "path": str(path), "error": "requested_contract_quotes_invalid"})
            continue
        return frame, str(path), errors
    return pd.DataFrame(), "", errors


def last_at_or_before(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    rows = frame[frame["quote_time"] <= timestamp]
    if rows.empty:
        return None
    return rows.iloc[-1]


def quote_state_for_candidate(row: pd.Series, quotes: pd.DataFrame | None) -> dict[str, Any]:
    base = {
        "direct_quote_state_status": "missing",
        "direct_quote_state_skip_reason": "",
        "direct_quote_source_path": row.get("_quote_source_path", ""),
    }
    if quotes is None or quotes.empty:
        base["direct_quote_state_skip_reason"] = "missing_contract_quotes"
        return base
    entry_time = pd.Timestamp(row["open_trade_entry_dt"])
    blocked_time = pd.Timestamp(row["best_blocked_decision_dt"])
    exit_time = pd.Timestamp(row["open_trade_exit_dt"])
    entry_ask = _finite(row.get("open_premium_paid"), math.nan) / CONTRACT_MULTIPLIER
    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        base["direct_quote_state_skip_reason"] = "invalid_entry_ask_from_open_premium_paid"
        return base
    current_quote = last_at_or_before(quotes, blocked_time)
    if current_quote is None:
        base["direct_quote_state_skip_reason"] = "missing_quote_at_or_before_blocked_signal"
        return base
    current_quote_time = pd.Timestamp(current_quote["quote_time"])
    path_to_signal = quotes[(quotes["quote_time"] >= entry_time) & (quotes["quote_time"] <= current_quote_time)].copy()
    if path_to_signal.empty:
        base["direct_quote_state_skip_reason"] = "missing_entry_to_signal_path"
        return base
    path_to_exit = quotes[(quotes["quote_time"] >= current_quote_time) & (quotes["quote_time"] <= exit_time)].copy()
    if path_to_exit.empty:
        path_to_exit = quotes[quotes["quote_time"] >= current_quote_time].head(1).copy()
    path_to_signal["pnl"] = (pd.to_numeric(path_to_signal["bid"], errors="coerce") - entry_ask) * CONTRACT_MULTIPLIER
    path_to_exit["pnl"] = (pd.to_numeric(path_to_exit["bid"], errors="coerce") - entry_ask) * CONTRACT_MULTIPLIER
    current_pnl = (_finite(current_quote.get("bid"), math.nan) - entry_ask) * CONTRACT_MULTIPLIER
    pnl_path = pd.to_numeric(path_to_signal["pnl"], errors="coerce").dropna()
    exit_path = pd.to_numeric(path_to_exit["pnl"], errors="coerce").dropna()
    if pnl_path.empty or not math.isfinite(current_pnl):
        base["direct_quote_state_skip_reason"] = "invalid_signal_path_pnl"
        return base
    mfe = float(pnl_path.max())
    mae = float(pnl_path.min())
    giveback = float(mfe - current_pnl)
    current_idx = int(path_to_signal.index[-1])
    best_idx = int(path_to_signal["pnl"].idxmax()) if not path_to_signal.empty else current_idx
    quote_index = path_to_signal.index.to_list()
    current_pos = len(quote_index) - 1

    def velocity(steps: int) -> float:
        prior_pos = current_pos - int(steps)
        if prior_pos < 0:
            return 0.0
        prior_idx = quote_index[prior_pos]
        prior_pnl = _finite(path_to_signal.loc[prior_idx, "pnl"], math.nan)
        return float((current_pnl - prior_pnl) / float(steps)) if math.isfinite(prior_pnl) else 0.0

    next_quote = quotes[quotes["quote_time"] > current_quote_time].head(1)
    if next_quote.empty:
        one_step_delta = 0.0
        next_quote_time = ""
    else:
        next_bid = _finite(next_quote.iloc[0].get("bid"), math.nan)
        one_step_delta = float((next_bid - _finite(current_quote.get("bid"), math.nan)) * CONTRACT_MULTIPLIER)
        next_quote_time = pd.Timestamp(next_quote.iloc[0]["quote_time"]).isoformat()
    actual_exit_pnl = _finite(row.get("open_trade_pnl"), current_pnl)
    future_best = float(exit_path.max()) if not exit_path.empty else current_pnl
    future_worst = float(exit_path.min()) if not exit_path.empty else current_pnl
    return {
        "direct_quote_state_status": "matched",
        "direct_quote_state_skip_reason": "",
        "direct_quote_source_path": row.get("_quote_source_path", ""),
        "blocked_signal_time": blocked_time.isoformat(),
        "current_quote_time": current_quote_time.isoformat(),
        "quote_lag_seconds": float((blocked_time - current_quote_time).total_seconds()),
        "next_quote_time": next_quote_time,
        "entry_ask_from_open_premium": float(entry_ask),
        "current_bid": _finite(current_quote.get("bid")),
        "current_ask": _finite(current_quote.get("ask")),
        "current_mid": float((_finite(current_quote.get("bid")) + _finite(current_quote.get("ask"))) / 2.0),
        "current_spread": float(_finite(current_quote.get("ask")) - _finite(current_quote.get("bid"))),
        "current_underlying_price": _finite(current_quote.get("underlying_price"), math.nan),
        "current_pnl_at_signal": float(current_pnl),
        "mfe_to_signal": mfe,
        "mae_to_signal": mae,
        "giveback_from_mfe_to_signal": giveback,
        "giveback_fraction_to_signal": float(giveback / abs(mfe)) if abs(mfe) > 1e-9 else 0.0,
        "pnl_velocity_1_to_signal": velocity(1),
        "pnl_velocity_3_to_signal": velocity(3),
        "pnl_velocity_5_to_signal": velocity(5),
        "time_since_mfe_quote_steps": float(max(current_idx - best_idx, 0)),
        "one_step_pnl_delta_after_signal": one_step_delta,
        "future_best_until_protocol101_exit_pnl": future_best,
        "future_worst_until_protocol101_exit_pnl": future_worst,
        "future_best_until_exit_minus_current": float(future_best - current_pnl),
        "future_worst_until_exit_minus_current": float(future_worst - current_pnl),
        "protocol101_exit_minus_current": float(actual_exit_pnl - current_pnl),
        "current_loss_at_signal": bool(current_pnl < 0.0),
        "current_positive_at_signal": bool(current_pnl >= 0.0),
        "one_step_hold_negative_at_signal": bool(one_step_delta < 0.0),
        "protocol101_hold_to_exit_deteriorates_after_signal": bool(actual_exit_pnl - current_pnl < 0.0),
    }


def attach_direct_quote_state(candidates: pd.DataFrame, normalized_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for session, group in candidates.groupby("session", sort=True):
        contracts = set(group["open_trade_contract_id"].astype(str).unique())
        quotes, source_path, read_errors = read_session_quotes(normalized_dirs, str(session), contracts)
        skips.extend(read_errors)
        by_contract = {
            str(contract_id): part.sort_values("quote_time").reset_index(drop=True)
            for contract_id, part in quotes.groupby("contract_id", sort=False)
        } if not quotes.empty else {}
        if not source_path:
            for _, row in group.iterrows():
                out = row.to_dict()
                out.update(
                    {
                        "direct_quote_state_status": "missing",
                        "direct_quote_state_skip_reason": "no_readable_session_quotes",
                        "direct_quote_source_path": "",
                    }
                )
                rows.append(out)
            continue
        for _, row in group.iterrows():
            work = row.copy()
            work["_quote_source_path"] = source_path
            state = quote_state_for_candidate(work, by_contract.get(str(row["open_trade_contract_id"])))
            out = row.to_dict()
            out.update(state)
            rows.append(out)
    joined = pd.DataFrame(rows)
    joined["live_test_bucket"] = joined.apply(classify_live_test_bucket, axis=1)
    return joined, pd.DataFrame(skips)


def classify_live_test_bucket(row: pd.Series) -> str:
    if str(row.get("direct_quote_state_status")) != "matched":
        return f"blocked_{row.get('direct_quote_state_skip_reason', 'missing_quote_state')}"
    relation = str(row.get("best_blocked_relation", ""))
    current_loss = bool(row.get("current_loss_at_signal", False))
    one_step_negative = bool(row.get("one_step_hold_negative_at_signal", False))
    deteriorates = bool(row.get("protocol101_hold_to_exit_deteriorates_after_signal", False))
    if current_loss and relation == "opposite_side" and one_step_negative and deteriorates:
        return "priority_1_current_loss_opposite_side_negative_next_and_exit_deteriorates"
    if current_loss and relation == "opposite_side":
        return "priority_2_current_loss_opposite_side"
    if current_loss and relation in {"same_contract", "same_side_different_contract"}:
        return "priority_3_current_loss_same_side"
    if not current_loss and one_step_negative and deteriorates:
        return "priority_4_not_yet_loss_but_negative_next_and_exit_deteriorates"
    return "priority_5_final_loss_but_live_state_not_stale_enough"


def summary_row(prefix: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    slot = pd.to_numeric(group["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0)
    matched = group["direct_quote_state_status"].astype(str).eq("matched")
    mg = group[matched]
    current = pd.to_numeric(mg.get("current_pnl_at_signal", pd.Series(dtype=float)), errors="coerce")
    row = dict(prefix)
    row.update(
        {
            "rows": int(len(group)),
            "positive_slot_cost_total": float(slot.clip(lower=0.0).sum()),
            "median_slot_cost": float(slot.median()) if len(group) else 0.0,
            "matched_quote_state_rows": int(matched.sum()),
            "quote_state_match_rate": float(matched.mean()) if len(group) else 0.0,
            "current_loss_rows": int(mg["current_loss_at_signal"].sum()) if not mg.empty else 0,
            "current_positive_rows": int(mg["current_positive_at_signal"].sum()) if not mg.empty else 0,
            "one_step_hold_negative_rows": int(mg["one_step_hold_negative_at_signal"].sum()) if not mg.empty else 0,
            "hold_to_exit_deteriorates_rows": int(mg["protocol101_hold_to_exit_deteriorates_after_signal"].sum())
            if not mg.empty
            else 0,
            "median_current_pnl_at_signal": float(current.median()) if current.notna().any() else 0.0,
            "median_protocol101_exit_minus_current": float(
                pd.to_numeric(mg.get("protocol101_exit_minus_current", pd.Series(dtype=float)), errors="coerce").median()
            )
            if not mg.empty
            else 0.0,
        }
    )
    return row


def build_direct_state_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = [summary_row({"summary_bucket": "all_loss_reversal_candidates"}, joined)]
    for bucket, group in joined.groupby("loss_reversal_subcase", dropna=False):
        rows.append(summary_row({"summary_bucket": f"subcase:{bucket}"}, group))
    for bucket, group in joined.groupby("live_test_bucket", dropna=False):
        rows.append(summary_row({"summary_bucket": f"live_test:{bucket}"}, group))
    for split, group in joined.groupby("reported_split", dropna=False):
        rows.append(summary_row({"summary_bucket": f"split:{split}"}, group))
    return pd.DataFrame(rows)


def build_manual_review_rows(joined: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    priority = {
        "priority_1_current_loss_opposite_side_negative_next_and_exit_deteriorates": 1,
        "priority_2_current_loss_opposite_side": 2,
        "priority_3_current_loss_same_side": 3,
        "priority_4_not_yet_loss_but_negative_next_and_exit_deteriorates": 4,
        "priority_5_final_loss_but_live_state_not_stale_enough": 5,
    }
    frame = joined.copy()
    frame["review_priority"] = frame["live_test_bucket"].map(priority).fillna(99).astype(int)
    cols = [
        "review_priority",
        "live_test_bucket",
        "reported_split",
        "seed",
        "session",
        "open_trade_entry_dt",
        "open_trade_exit_dt",
        "best_blocked_decision_dt",
        "current_quote_time",
        "quote_lag_seconds",
        "open_trade_right",
        "best_blocked_right",
        "best_blocked_relation",
        "open_trade_exit_reason",
        "open_trade_pnl",
        "current_pnl_at_signal",
        "mfe_to_signal",
        "mae_to_signal",
        "giveback_from_mfe_to_signal",
        "one_step_pnl_delta_after_signal",
        "protocol101_exit_minus_current",
        "future_best_until_exit_minus_current",
        "future_worst_until_exit_minus_current",
        "best_blocked_candidate_pnl",
        "best_blocked_minus_open_pnl",
        "current_bid",
        "current_ask",
        "current_spread",
        "best_blocked_entry_ask",
        "best_blocked_entry_spread",
        "direct_quote_source_path",
        "direct_quote_state_status",
        "direct_quote_state_skip_reason",
    ]
    available = [col for col in cols if col in frame.columns]
    return frame.sort_values(["review_priority", "best_blocked_minus_open_pnl"], ascending=[True, False], kind="stable").head(limit)[
        available
    ]


def build_summary(joined: pd.DataFrame, state_summary: pd.DataFrame, skips: pd.DataFrame) -> dict[str, Any]:
    matched = joined["direct_quote_state_status"].astype(str).eq("matched")
    mg = joined[matched]
    current_loss = mg["current_loss_at_signal"].astype(bool) if not mg.empty else pd.Series(dtype=bool)
    opposite = mg["best_blocked_relation"].astype(str).eq("opposite_side") if not mg.empty else pd.Series(dtype=bool)
    one_step_negative = mg["one_step_hold_negative_at_signal"].astype(bool) if not mg.empty else pd.Series(dtype=bool)
    deteriorates = mg["protocol101_hold_to_exit_deteriorates_after_signal"].astype(bool) if not mg.empty else pd.Series(dtype=bool)
    p1 = current_loss & opposite & one_step_negative & deteriorates
    slot = pd.to_numeric(joined["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0)
    matched_slot = pd.to_numeric(mg.get("best_blocked_minus_open_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "direct quote-path causal-state attachment for selected Protocol101 loss-reversal hypothesis",
        "decision": "protocol101_loss_reversal_causal_state_attachment_partial_quote_coverage_replay_scaffold_ready_training_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "challenge_allowed": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "counts": {
            "loss_reversal_candidates": int(len(joined)),
            "matched_direct_quote_state_rows": int(matched.sum()),
            "missing_direct_quote_state_rows": int((~matched).sum()),
            "quote_read_error_rows": int(len(skips)),
        },
        "coverage": {
            "direct_quote_state_match_rate": float(matched.mean()) if len(joined) else 0.0,
        },
        "slot_cost": {
            "total_positive_slot_cost": float(slot.clip(lower=0.0).sum()),
            "matched_positive_slot_cost": float(matched_slot.clip(lower=0.0).sum()),
        },
        "live_state": {
            "current_loss_rows": int(current_loss.sum()) if len(current_loss) else 0,
            "current_positive_rows": int((~current_loss).sum()) if len(current_loss) else 0,
            "current_loss_opposite_side_rows": int((current_loss & opposite).sum()) if len(current_loss) else 0,
            "priority_1_rows": int(p1.sum()) if len(current_loss) else 0,
            "priority_1_positive_slot_cost": float(matched_slot[p1].clip(lower=0.0).sum()) if len(current_loss) else 0.0,
        },
        "foundational_truth": (
            "Full direct quote attachment improves coverage substantially, but the trainable hypothesis must be "
            "current-state loss/staleness at the blocked signal, not final losing trade status."
        ),
        "blockers": [
            "missing_quote_state_for_some_sessions",
            "mutually_exclusive_keep_hold_vs_exit_switch_replay_not_built",
            "switching_cost_latency_fill_and_quote_freshness_not_calibrated",
            "diagnostic_splits_are_research_exposed",
            "neural_training_forbidden_until_replay_defines_live_observable_labels",
        ],
    }


def write_report(output_dir: Path, summary: dict[str, Any], state_summary: pd.DataFrame) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: direct quote-path causal-state attachment for `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "Direct quote-path attachment substantially improves the selected hypothesis from a final-PnL story into a live-state diagnostic. It still does not authorize neural training.",
        "",
        f"- Loss-reversal candidates: `{summary['counts']['loss_reversal_candidates']}`",
        f"- Direct quote state matched: `{summary['counts']['matched_direct_quote_state_rows']}`",
        f"- Match rate: `{summary['coverage']['direct_quote_state_match_rate']:.3f}`",
        f"- Matched positive slot-cost: `{_fmt_money(summary['slot_cost']['matched_positive_slot_cost'])}`",
        f"- Current-loss rows at blocked signal: `{summary['live_state']['current_loss_rows']}`",
        f"- Current-loss plus opposite-side rows: `{summary['live_state']['current_loss_opposite_side_rows']}`",
        f"- Priority-1 rows: `{summary['live_state']['priority_1_rows']}`",
        f"- Priority-1 positive slot-cost: `{_fmt_money(summary['live_state']['priority_1_positive_slot_cost'])}`",
        "",
        "The next valid experiment is a mutually exclusive replay, not a neural net. For each matched row, compare the actual keep-holding path against exit-now and exit/switch-to-later-signal, charging spread, latency, fill uncertainty, and single-slot opportunity cost.",
        "",
        "## Direct State Summary",
        "",
        "| bucket | rows | slot-cost total | matched | match rate | current loss | current positive | one-step negative | hold-to-exit worse | median current PnL | median exit-current |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in state_summary.iterrows():
        lines.append(
            f"| {row['summary_bucket']} | {int(row['rows'])} | {_fmt_money(row['positive_slot_cost_total'])} | "
            f"{int(row['matched_quote_state_rows'])} | {row['quote_state_match_rate']:.3f} | "
            f"{int(row['current_loss_rows'])} | {int(row['current_positive_rows'])} | "
            f"{int(row['one_step_hold_negative_rows'])} | {int(row['hold_to_exit_deteriorates_rows'])} | "
            f"{_fmt_money(row['median_current_pnl_at_signal'])} | {_fmt_money(row['median_protocol101_exit_minus_current'])} |"
        )
    lines.extend(
        [
            "",
            "## Stopping Point",
            "",
            "This packet reaches the next Track A stopping point. We now know which rows are live-state candidates, but we have not priced the action. The next artifact must be `PROTOCOL101_LOSS_REVERSAL_MUTUALLY_EXCLUSIVE_REPLAY_V1`.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Direct state join: `{output_dir / 'direct_causal_state_join.csv'}`",
            f"- Direct state summary: `{output_dir / 'direct_causal_state_summary.csv'}`",
            f"- Manual review rows: `{output_dir / 'manual_review_priority_rows.csv'}`",
            f"- Quote read issues: `{output_dir / 'quote_read_issues.csv'}`",
            f"- Report: `{output_dir / 'report.md'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dirs = [Path(path) for path in args.normalized_dir]
    candidates = load_loss_reversal_candidates(Path(args.slot_cost_dir))
    joined, quote_issues = attach_direct_quote_state(candidates, normalized_dirs)
    state_summary = build_direct_state_summary(joined)
    review_rows = build_manual_review_rows(joined)
    summary = build_summary(joined, state_summary, quote_issues)
    summary["inputs"] = {
        "slot_cost_dir": str(args.slot_cost_dir),
        "normalized_dirs": [str(path) for path in normalized_dirs],
    }
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "direct_causal_state_join": str(output_dir / "direct_causal_state_join.csv"),
        "direct_causal_state_summary": str(output_dir / "direct_causal_state_summary.csv"),
        "manual_review_priority_rows": str(output_dir / "manual_review_priority_rows.csv"),
        "quote_read_issues": str(output_dir / "quote_read_issues.csv"),
    }
    joined.to_csv(output_dir / "direct_causal_state_join.csv", index=False)
    state_summary.to_csv(output_dir / "direct_causal_state_summary.csv", index=False)
    review_rows.to_csv(output_dir / "manual_review_priority_rows.csv", index=False)
    quote_issues.to_csv(output_dir / "quote_read_issues.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    report = write_report(output_dir, summary, state_summary)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--slot-cost-dir", type=Path, default=DEFAULT_SLOT_COST_DIR)
    parser.add_argument("--normalized-dir", type=Path, action="append", default=list(DEFAULT_NORMALIZED_DIRS))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
