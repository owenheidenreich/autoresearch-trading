"""DATASET_FULL_SURFACE_ACTION_ADVANTAGE_V1.

Build the next-source training labels for the unified 0DTE trading game:
flat-state wait/enter advantages over the full live-compatible SPXW candidate
surface. This is a dataset runner, not a model and not a promotion decision.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.unified_serial_game import (
    UnifiedSerialGameConfig,
    build_flat_action_advantage_labels,
    default_unified_game_config,
)
from v4.scripts.run_protocol164_full_action_space_dataset import FULL_ACTION_FEATURE_COLUMNS


ROLE_LABEL = "DATASET_FULL_SURFACE_ACTION_ADVANTAGE_V1"
HISTORICAL_ID = "Protocol270"
DEFAULT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset")
CONTRACT_MULTIPLIER = 100.0
REQUIRED_COLUMNS = [
    "split",
    "session",
    "decision_time",
    "decision_dt",
    "candidate_uid",
    "trade_uid",
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
    "candidate_exit_time",
    "candidate_exit_dt",
    "candidate_pnl",
    "candidate_exit_reason",
    "path_status",
    "label_source",
]
OPTIONAL_HISTORY_COLUMNS = [
    "edge",
    "surface_edge",
    "surface_action_score",
    "surface_flat_score",
    "surface_best_edge",
    "surface_mean_edge",
    "surface_best_call_edge",
    "surface_best_put_edge",
    "surface_call_minus_put_best_edge",
    "surface_above25_count",
    "surface_roll3_best_edge",
    "surface_roll3_mean_edge",
    "surface_prev_best_edge",
    "surface_prev_mean_edge",
    "hist_events_seen",
    "hist_minutes_since_prev_event",
    "hist_prev_candidate_count",
    "hist_prev_max_edge",
    "hist_prev_mean_edge",
    "hist_prev_max_gamma",
    "hist_prev_mean_theta_burden",
    "hist_prev_min_spread_over_mid",
    "hist_prev_call_count",
    "hist_prev_put_count",
    "hist_prev_call_minus_put_edge",
    "hist_roll3_candidate_count_mean",
    "hist_roll3_max_edge",
    "hist_roll3_mean_edge",
    "hist_roll3_max_gamma",
    "hist_roll3_mean_theta_burden",
    "hist_roll3_min_spread_over_mid",
    "hist_roll3_call_minus_put_edge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-events", type=int, default=0, help="Optional event cap for smoke runs; 0 means all events.")
    parser.add_argument("--max-sessions-per-split", type=int, default=0, help="Optional per-split cap for development runs.")
    parser.add_argument("--slippage-per-side", type=float, default=0.0)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_source(args.dataset)
    filtered, rejection = filter_live_compatible(frame, default_unified_game_config())
    filtered = limit_dataset(filtered, max_events=int(args.max_events), max_sessions_per_split=int(args.max_sessions_per_split))
    labels = build_flat_action_advantage_labels(filtered, slippage_per_side=float(args.slippage_per_side))
    labels = add_scope_columns(labels)
    out_path = args.out_dir / "full_surface_action_advantage.parquet"
    labels.to_parquet(out_path, index=False)
    split_summary = summarize_labels(labels)
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "dataset / full-surface serial action-advantage labels",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1_INPUT",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.dataset),
        "output_dataset": str(out_path),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "slippage_per_side": float(args.slippage_per_side),
        "game_contract": default_unified_game_config().to_dict(),
        "disallowed_gates_applied": {
            "protocol101_min_edge_gate": False,
            "protocol101_allowed_time_bucket_gate": False,
            "protocol101_selected_candidate_dependency": False,
        },
        "row_counts": {
            "source_rows": int(len(frame)),
            "filtered_rows": int(len(filtered)),
            "label_rows": int(len(labels)),
            "source_decision_events": int(frame[["split", "session", "decision_dt"]].drop_duplicates().shape[0]),
            "label_decision_events": int(labels[["split", "session", "decision_dt"]].drop_duplicates().shape[0]) if not labels.empty else 0,
        },
        "rejection_counts": rejection,
        "split_summary": split_summary.to_dict("records"),
        "decision": decide(labels),
        "next_experiment": "Train CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1 against these labels, then compare to PAPER_DEFAULT_PROTOCOL101.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "dataset": str(out_path), "rows": len(labels)}, indent=2, sort_keys=True))
    return 0


def load_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    columns = list(dict.fromkeys([*REQUIRED_COLUMNS, *FULL_ACTION_FEATURE_COLUMNS, *OPTIONAL_HISTORY_COLUMNS]))
    frame = pd.read_parquet(path, columns=columns)
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["candidate_exit_dt"] = pd.to_datetime(frame["candidate_exit_dt"], utc=True, errors="coerce")
    for column in [
        "offset",
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
        "candidate_pnl",
        *FULL_ACTION_FEATURE_COLUMNS,
        *OPTIONAL_HISTORY_COLUMNS,
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def filter_live_compatible(frame: pd.DataFrame, config: UnifiedSerialGameConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    working = frame.copy()
    rejection: dict[str, int] = {}

    def keep(mask: pd.Series, reason: str) -> None:
        nonlocal working
        rejection[reason] = int((~mask).sum())
        working = working[mask].copy()

    keep(working["decision_dt"].notna() & working["candidate_exit_dt"].notna(), "missing_timestamps")
    keep(working["root"].astype(str).eq(config.symbol_root), "wrong_root")
    keep(working["settlement_style"].astype(str).eq(config.settlement_style), "wrong_settlement_style")
    keep(working["right"].astype(str).isin(["C", "P"]), "invalid_right")
    offset = pd.to_numeric(working["offset"], errors="coerce")
    strike_aligned = np.isclose(np.mod(np.abs(offset), config.strike_spacing), 0.0, atol=1e-6)
    keep(offset.notna() & offset.abs().le(config.strike_window) & pd.Series(strike_aligned, index=working.index), "invalid_strike_offset")
    bid = pd.to_numeric(working["entry_bid"], errors="coerce")
    ask = pd.to_numeric(working["entry_ask"], errors="coerce")
    mid = pd.to_numeric(working["entry_mid"], errors="coerce")
    spread = pd.to_numeric(working["entry_spread"], errors="coerce")
    keep(bid.gt(0) & ask.gt(0) & mid.gt(0) & ask.ge(bid) & spread.ge(0), "invalid_nbbo")
    bid_size = pd.to_numeric(working["entry_bid_size"], errors="coerce")
    ask_size = pd.to_numeric(working["entry_ask_size"], errors="coerce")
    keep(bid_size.gt(0) & ask_size.gt(0), "missing_quote_size")
    greek_mask = (
        pd.to_numeric(working["entry_delta"], errors="coerce").map(math.isfinite)
        & pd.to_numeric(working["entry_gamma"], errors="coerce").map(math.isfinite)
        & pd.to_numeric(working["entry_theta"], errors="coerce").map(math.isfinite)
        & pd.to_numeric(working["entry_iv"], errors="coerce").map(math.isfinite)
    )
    keep(greek_mask, "invalid_unrepaired_greeks")
    premium = pd.to_numeric(working["entry_premium"], errors="coerce")
    expected = pd.to_numeric(working["entry_ask"], errors="coerce") * CONTRACT_MULTIPLIER
    keep(premium.gt(0) & premium.le(config.starting_cash) & np.isclose(premium, expected, atol=1e-6), "unaffordable_or_bad_premium")
    keep(working["candidate_exit_dt"].gt(working["decision_dt"]) & pd.to_numeric(working["candidate_pnl"], errors="coerce").map(math.isfinite), "invalid_candidate_path")
    keep(working["path_status"].astype(str).eq("ok"), "path_not_ok")
    local = working["decision_dt"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    keep(minute.lt(config.no_new_entries_after_minute_et), "after_no_new_entries_cutoff")
    return working.sort_values(["split", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True), rejection


def limit_dataset(frame: pd.DataFrame, *, max_events: int, max_sessions_per_split: int) -> pd.DataFrame:
    working = frame
    if max_sessions_per_split > 0:
        pieces = []
        for split, group in working.groupby("split", sort=True):
            sessions = sorted(group["session"].astype(str).unique())[:max_sessions_per_split]
            pieces.append(group[group["session"].astype(str).isin(sessions)])
        working = pd.concat(pieces, ignore_index=True) if pieces else working.iloc[0:0]
    if max_events > 0 and not working.empty:
        events = working[["split", "session", "decision_dt"]].drop_duplicates().sort_values(["split", "session", "decision_dt"]).head(max_events)
        working = working.merge(events.assign(_keep_event=1), on=["split", "session", "decision_dt"], how="inner").drop(columns=["_keep_event"])
    return working.reset_index(drop=True)


def add_scope_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["runtime_feature_scope"] = "entry_causal_only"
    out["future_path_columns_used_as_features"] = False
    out["paper_account_starting_cash"] = 10_000.0
    out["max_concurrent_positions"] = 1
    out["max_contracts"] = 1
    return out


def summarize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=["split", "rows", "decision_events", "sessions", "oracle_enter_events", "oracle_wait_events", "mean_best_enter_advantage", "median_best_enter_advantage"])
    grouped_events = frame.groupby(["split", "session", "decision_dt"], sort=True)
    event_best = grouped_events["a_enter"].max().rename("best_a_enter").reset_index()
    oracle_enter = frame[frame["oracle_action"].astype(str).eq("enter")][["split", "session", "decision_dt"]].drop_duplicates()
    for split, group in frame.groupby("split", sort=True):
        split_events = event_best[event_best["split"].astype(str).eq(str(split))]
        enter_events = oracle_enter[oracle_enter["split"].astype(str).eq(str(split))]
        rows.append(
            {
                "split": str(split),
                "rows": int(len(group)),
                "decision_events": int(split_events.shape[0]),
                "sessions": int(group["session"].nunique()),
                "oracle_enter_events": int(enter_events.shape[0]),
                "oracle_wait_events": int(split_events.shape[0] - enter_events.shape[0]),
                "mean_best_enter_advantage": float(split_events["best_a_enter"].mean()) if len(split_events) else 0.0,
                "median_best_enter_advantage": float(split_events["best_a_enter"].median()) if len(split_events) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def decide(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "blocked_action_advantage_dataset_empty"
    splits = set(map(str, frame["split"].unique()))
    required = {"q3_2025", "q4_2025", "q1_2026", "recent_2026"}
    if not required.issubset(splits):
        return "action_advantage_dataset_ready_partial_split_coverage"
    return "action_advantage_dataset_ready_for_unified_policy_training"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being prepared: `{payload['candidate_label']}`",
        f"Paper default baseline: `{payload['paper_default_label']}`",
        f"Data used: `{payload['data_used']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Scope",
        "",
        "- Full SPXW PM 0DTE ATM +/- $50 surface.",
        "- No Protocol101 min-edge, time-bucket, or selected-candidate gate.",
        "- Labels are serial one-account action advantages: `Q_wait`, `Q_enter`, and `A_enter`.",
        "- Holding labels included here are entry-path proxies; full hold/exit training remains the next modeling step.",
        "",
        "## Split Summary",
        "",
        "| split | rows | events | sessions | oracle enter events | median best A_enter |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['split']} | {row['rows']} | {row['decision_events']} | {row['sessions']} | "
            f"{row['oracle_enter_events']} | {row['median_best_enter_advantage']:.2f} |"
        )
    lines.extend(["", "## Outputs", "", f"- Dataset: `{payload['output_dataset']}`", f"- Summary: `{path.parent / 'summary.json'}`"])
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    text = ledger.read_text()
    if marker in text:
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    f"- Candidate: `{payload['candidate_label']}`",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
