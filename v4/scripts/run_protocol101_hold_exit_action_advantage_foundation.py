from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.unified_serial_game import CONTRACT_MULTIPLIER, hold_exit_opportunity_advantage_path  # noqa: E402
from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import (  # noqa: E402
    DEFAULT_NORMALIZED_DIR,
    finite_sum,
    money,
    normalize_trade_frame,
    pct,
)
from v4.scripts.run_protocol199_lifecycle_full_path_oracle import load_session_quotes  # noqa: E402


ROLE_LABEL = "FOUNDATION_PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_V1"
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv")
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_FOUNDATION_V1.md"
)
NY = ZoneInfo("America/New_York")

LABEL_ONLY_COLUMNS = {
    "q_exit",
    "q_hold",
    "a_hold",
    "a_exit",
    "a_switch",
    "a_hold_one_step_realized",
    "future_best_hold_value",
    "future_best_pnl",
    "future_worst_pnl",
    "future_worst_before_best_pnl",
    "future_best_time",
    "oracle_holding_action",
    "oracle_one_step_action",
}


def load_protocol101_selected_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"no Protocol101 selected trades found in {path}")
    if "segment" not in raw.columns:
        raise ValueError("Protocol101 selected trades must include segment")
    raw = raw.rename(columns={"segment": "reported_split"}).copy()
    raw["fold"] = "protocol101_seed1_selected"
    return normalize_trade_frame(raw)


def forced_flat_timestamp(session: str, forced_flat_time: str) -> pd.Timestamp:
    hour, minute = [int(part) for part in forced_flat_time.split(":", 1)]
    return pd.Timestamp(session).replace(hour=hour, minute=minute, tzinfo=NY).tz_convert("UTC")


def build_hold_exit_rows(
    trades: pd.DataFrame,
    *,
    normalized_dir: Path,
    forced_flat_time: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for session, session_trades in trades.groupby("session", sort=True):
        contracts = set(session_trades["contract_id"].astype(str).unique())
        quotes = load_session_quotes(normalized_dir, str(session), contracts)
        if quotes.empty:
            skips.extend(base_skip(trade, "missing_session_or_contract_quotes") for _, trade in session_trades.iterrows())
            continue
        by_contract = {
            str(contract_id): part.sort_values("quote_time").reset_index(drop=True)
            for contract_id, part in quotes.groupby("contract_id", sort=False)
        }
        forced_flat = forced_flat_timestamp(str(session), forced_flat_time)
        for _, trade in session_trades.iterrows():
            trade_rows, skip = labels_for_trade(trade, by_contract.get(str(trade["contract_id"])), forced_flat)
            if skip:
                skips.append(skip)
            else:
                rows.extend(trade_rows)
    return rows, skips


def labels_for_trade(
    trade: pd.Series,
    quotes: pd.DataFrame | None,
    forced_flat: pd.Timestamp,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if quotes is None or quotes.empty:
        return [], base_skip(trade, "missing_contract_quotes")
    entry_time = pd.Timestamp(trade["decision_ts"])
    exit_time = pd.Timestamp(trade["exit_ts"])
    entry_ask = finite(trade.get("entry_ask"))
    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        return [], base_skip(trade, "invalid_entry_ask")

    path = quotes[(quotes["quote_time"] >= entry_time) & (quotes["quote_time"] <= forced_flat)].copy()
    path = path.sort_values("quote_time").reset_index(drop=True)
    if path.empty:
        return [], base_skip(trade, "missing_post_entry_path")
    path["bid"] = pd.to_numeric(path["bid"], errors="coerce")
    path["ask"] = pd.to_numeric(path["ask"], errors="coerce")
    path = path[path["bid"].notna() & path["ask"].notna()].reset_index(drop=True)
    if path.empty:
        return [], base_skip(trade, "invalid_path_bid_ask")

    labels = hold_exit_opportunity_advantage_path(path["bid"].to_numpy(dtype=float), entry_ask, None)
    if labels.empty:
        return [], base_skip(trade, "empty_hold_exit_labels")
    path = pd.concat([path.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    path["a_exit"] = path["q_exit"] - path["q_hold"]
    path["future_best_pnl"] = np.maximum.accumulate(path["current_pnl"].to_numpy(dtype=float)[::-1])[::-1]
    path["future_worst_pnl"] = np.minimum.accumulate(path["current_pnl"].to_numpy(dtype=float)[::-1])[::-1]
    path["next_pnl"] = path["current_pnl"].shift(-1).fillna(path["current_pnl"])
    path["a_hold_one_step_realized"] = path["next_pnl"] - path["current_pnl"]
    path["oracle_one_step_action"] = np.where(path["a_hold_one_step_realized"] > 0.0, "hold", "exit")
    path["mfe_to_now"] = path["current_pnl"].cummax()
    path["mae_to_now"] = path["current_pnl"].cummin()
    path["giveback_from_mfe"] = path["mfe_to_now"] - path["current_pnl"]
    path["giveback_fraction"] = np.where(path["mfe_to_now"].abs() > 1e-9, path["giveback_from_mfe"] / path["mfe_to_now"].abs(), 0.0)
    path["pnl_velocity_1"] = path["current_pnl"].diff(1).fillna(0.0)
    path["pnl_velocity_3"] = path["current_pnl"].diff(3).fillna(0.0) / 3.0
    path["pnl_velocity_5"] = path["current_pnl"].diff(5).fillna(0.0) / 5.0

    exit_index = first_index_at_or_after(path["quote_time"], exit_time)
    if exit_index is None:
        exit_index = int(len(path) - 1)
    best_seen_idx = 0
    rows: list[dict[str, Any]] = []
    for idx, row in path.iterrows():
        current_pnl = finite(row.get("current_pnl"))
        if current_pnl >= finite(path.loc[best_seen_idx, "current_pnl"]):
            best_seen_idx = int(idx)
        future_best_idx = first_future_best_index(path, int(idx))
        quote_time = pd.Timestamp(row["quote_time"])
        rows.append(
            {
                "reported_split": str(trade["reported_split"]),
                "fold": str(trade.get("fold", "protocol101_seed1_selected")),
                "seed": int(trade.get("seed", 0)),
                "session": str(trade["session"]),
                "candidate_uid": str(trade.get("candidate_uid", "")),
                "contract_id": str(trade["contract_id"]),
                "right": str(trade["right"]),
                "offset": finite(trade.get("offset")),
                "entry_time": entry_time.isoformat(),
                "protocol101_exit_time": exit_time.isoformat(),
                "forced_flat_time": forced_flat.isoformat(),
                "state_time": quote_time.isoformat(),
                "state_index": int(idx),
                "minutes_since_entry": float((quote_time - entry_time).total_seconds() / 60.0),
                "minutes_to_protocol101_exit": float((exit_time - quote_time).total_seconds() / 60.0),
                "minutes_to_forced_flat": float((forced_flat - quote_time).total_seconds() / 60.0),
                "is_protocol101_exit_state": bool(int(idx) == int(exit_index)),
                "is_after_protocol101_exit": bool(quote_time > exit_time),
                "time_bucket": str(trade.get("time_bucket", "")),
                "entry_ask": entry_ask,
                "entry_bid": finite(trade.get("entry_bid")),
                "entry_premium": finite(trade.get("entry_ask")) * CONTRACT_MULTIPLIER,
                "score": finite(trade.get("score")),
                "threshold": finite(trade.get("threshold")),
                "score_margin": finite(trade.get("score")) - finite(trade.get("threshold")),
                "exit_reason": str(trade.get("exit_reason", "")),
                "bid": finite(row.get("bid")),
                "ask": finite(row.get("ask")),
                "mid": (finite(row.get("bid")) + finite(row.get("ask"))) / 2.0,
                "spread": finite(row.get("ask")) - finite(row.get("bid")),
                "underlying_price": finite(row.get("underlying_price")),
                "current_pnl": current_pnl,
                "mfe_to_now": finite(row.get("mfe_to_now")),
                "mae_to_now": finite(row.get("mae_to_now")),
                "giveback_from_mfe": finite(row.get("giveback_from_mfe")),
                "giveback_fraction": finite(row.get("giveback_fraction")),
                "pnl_velocity_1": finite(row.get("pnl_velocity_1")),
                "pnl_velocity_3": finite(row.get("pnl_velocity_3")),
                "pnl_velocity_5": finite(row.get("pnl_velocity_5")),
                "time_since_mfe_minutes": float(int(idx) - int(best_seen_idx)),
                "q_exit": finite(row.get("q_exit")),
                "q_hold": finite(row.get("q_hold")),
                "a_hold": finite(row.get("a_hold")),
                "a_exit": finite(row.get("a_exit")),
                "a_switch": finite(row.get("a_switch")),
                "a_hold_one_step_realized": finite(row.get("a_hold_one_step_realized")),
                "future_best_hold_value": finite(row.get("future_best_hold_value")),
                "future_best_pnl": finite(row.get("future_best_pnl")),
                "future_worst_pnl": finite(row.get("future_worst_pnl")),
                "future_worst_before_best_pnl": future_worst_before_index(path, int(idx), future_best_idx),
                "future_best_time": pd.Timestamp(path.loc[future_best_idx, "quote_time"]).isoformat(),
                "oracle_holding_action": str(row.get("oracle_holding_action")),
                "oracle_one_step_action": str(row.get("oracle_one_step_action")),
                "label_source": "protocol101_hold_exit_action_advantage_no_slot_v1",
                "label_limitations": "no_future_flat_slot_opportunity_value_no_fill_model_hindsight_labels",
                "future_path_columns_used_as_features": False,
            }
        )
    return rows, None


def first_index_at_or_after(times: pd.Series, target: pd.Timestamp) -> int | None:
    parsed = pd.to_datetime(times, utc=True, errors="coerce")
    mask = parsed >= target
    if not bool(mask.any()):
        return None
    return int(np.flatnonzero(mask.to_numpy())[0])


def first_future_best_index(path: pd.DataFrame, idx: int) -> int:
    current = path.iloc[idx:].reset_index()
    max_value = float(pd.to_numeric(current["current_pnl"], errors="coerce").max())
    match = current[pd.to_numeric(current["current_pnl"], errors="coerce").eq(max_value)]
    if match.empty:
        return idx
    return int(match.iloc[0]["index"])


def future_worst_before_index(path: pd.DataFrame, idx: int, best_idx: int) -> float:
    lo = min(idx, best_idx)
    hi = max(idx, best_idx)
    window = pd.to_numeric(path.iloc[lo : hi + 1]["current_pnl"], errors="coerce")
    if window.empty:
        return finite(path.iloc[idx].get("current_pnl"))
    return float(window.min())


def build_split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for split, group in frame.groupby("reported_split", sort=True):
        rows.append(summary_row({"reported_split": split}, group))
    return pd.DataFrame(rows).sort_values("reported_split", kind="stable")


def build_exit_state_audit(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    exits = frame[frame["is_protocol101_exit_state"].astype(bool)].copy()
    if exits.empty:
        return exits
    sort_cols = [col for col in ["reported_split", "session", "state_time", "contract_id"] if col in exits.columns]
    if not sort_cols:
        return exits.reset_index(drop=True)
    return exits.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def build_exit_reason_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(["reported_split", "exit_reason"], dropna=False, observed=False):
        rows.append(summary_row({"reported_split": key_values[0], "exit_reason": key_values[1]}, group))
    return pd.DataFrame(rows).sort_values(["reported_split", "exit_reason"], kind="stable")


def build_protocol101_exit_decision_summary(exit_audit: pd.DataFrame) -> pd.DataFrame:
    if exit_audit.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key_values, group in exit_audit.groupby(["reported_split", "exit_reason", "oracle_holding_action"], dropna=False, observed=False):
        rows.append(summary_row({"reported_split": key_values[0], "exit_reason": key_values[1], "oracle_holding_action": key_values[2]}, group))
    return pd.DataFrame(rows).sort_values(["reported_split", "exit_reason", "oracle_holding_action"], kind="stable")


def summary_row(prefix: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    a_hold = pd.to_numeric(group["a_hold"], errors="coerce").fillna(0.0)
    current = pd.to_numeric(group["current_pnl"], errors="coerce").fillna(0.0)
    out = dict(prefix)
    out.update(
        {
            "rows": int(len(group)),
            "trades": int(group["candidate_uid"].nunique()) if "candidate_uid" in group else 0,
            "hold_fraction_oracle_best": float(group["oracle_holding_action"].astype(str).eq("hold").mean()),
            "one_step_hold_fraction": float(group["oracle_one_step_action"].astype(str).eq("hold").mean()),
            "median_a_hold": float(a_hold.median()),
            "mean_a_hold": float(a_hold.mean()),
            "median_current_pnl": float(current.median()),
            "median_mfe_to_now": float(pd.to_numeric(group["mfe_to_now"], errors="coerce").median()),
            "median_giveback_from_mfe": float(pd.to_numeric(group["giveback_from_mfe"], errors="coerce").median()),
            "median_future_worst_before_best": float(pd.to_numeric(group["future_worst_before_best_pnl"], errors="coerce").median()),
        }
    )
    return out


def build_feature_contract() -> dict[str, Any]:
    causal_feature_columns = [
        "reported_split",
        "session",
        "state_time",
        "right",
        "offset",
        "minutes_since_entry",
        "minutes_to_protocol101_exit",
        "minutes_to_forced_flat",
        "time_bucket",
        "entry_ask",
        "entry_bid",
        "entry_premium",
        "score",
        "threshold",
        "score_margin",
        "bid",
        "ask",
        "mid",
        "spread",
        "underlying_price",
        "current_pnl",
        "mfe_to_now",
        "mae_to_now",
        "giveback_from_mfe",
        "giveback_fraction",
        "pnl_velocity_1",
        "pnl_velocity_3",
        "pnl_velocity_5",
        "time_since_mfe_minutes",
    ]
    forbidden_overlap = sorted(set(causal_feature_columns) & LABEL_ONLY_COLUMNS)
    return {
        "causal_feature_columns": causal_feature_columns,
        "label_only_columns": sorted(LABEL_ONLY_COLUMNS),
        "forbidden_overlap": forbidden_overlap,
        "status": "pass" if not forbidden_overlap else "fail",
    }


def build_summary(
    frame: pd.DataFrame,
    skips: pd.DataFrame,
    *,
    source_trades: int,
    output_dataset: Path,
    feature_contract: dict[str, Any],
) -> dict[str, Any]:
    exit_audit = build_exit_state_audit(frame)
    hold_at_exit = float(exit_audit["oracle_holding_action"].astype(str).eq("hold").mean()) if not exit_audit.empty else 0.0
    one_step_hold_at_exit = (
        float(exit_audit["oracle_one_step_action"].astype(str).eq("hold").mean())
        if not exit_audit.empty and "oracle_one_step_action" in exit_audit.columns
        else 0.0
    )
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "foundation / Protocol101 holding-state hold-vs-exit action-advantage labels",
        "decision": "protocol101_hold_exit_action_advantage_foundation_complete_training_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "output_dataset": str(output_dataset),
        "row_counts": {
            "source_trades": int(source_trades),
            "labeled_state_rows": int(len(frame)),
            "covered_trades": int(frame["candidate_uid"].nunique()) if not frame.empty else 0,
            "path_skips": int(len(skips)),
        },
        "label_semantics": {
            "q_exit": "current executable bid PnL only; future flat-slot value is intentionally omitted in this Protocol101 foundation packet",
            "q_hold": "best later executable bid PnL after holding at least one more quote step; hindsight upper-bound label",
            "a_hold": "q_hold - q_exit",
            "a_exit": "q_exit - q_hold",
            "a_hold_one_step_realized": "next quote-step PnL minus current PnL; realized diagnostic target",
        },
        "limitations": [
            "no_counterfactual_flat_slot_opportunity_cost",
            "no_explicit_switching_cost",
            "no_calibrated_fill_model",
            "hindsight_future_best_label_is_not_a_deployed_policy",
            "no_distributional_risk_or_uncertainty_target_yet",
            "current_splits_are_research_exposed",
        ],
        "engineer_response_alignment": {
            "implemented": [
                "hold_exit_advantage_label_skeleton",
                "current_executable_exit_bid_accounting",
                "causal_post_entry_path_features",
                "one_step_hold_exit_diagnostic_target",
                "label_only_future_path_columns_kept_out_of_feature_contract",
            ],
            "still_missing_before_training": [
                "counterfactual_flat_slot_opportunity_cost",
                "explicit_switching_cost_for_exit_reentry_churn",
                "calibrated_fill_latency_uncertainty",
                "distributional_outcome_targets",
                "same_lifecycle_policy_candidate_set_train_live_parity",
                "untouched_validation_and_formal_overfit_controls",
            ],
        },
        "feature_contract": feature_contract,
        "exit_state_audit": {
            "rows": int(len(exit_audit)),
            "oracle_hold_fraction_at_protocol101_exit": hold_at_exit,
            "one_step_hold_fraction_at_protocol101_exit": one_step_hold_at_exit,
            "median_a_hold_at_protocol101_exit": float(pd.to_numeric(exit_audit.get("a_hold", pd.Series(dtype=float)), errors="coerce").median())
            if not exit_audit.empty
            else 0.0,
        },
        "challenge_allowed": False,
        "training_allowed": False,
    }


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    split_summary: pd.DataFrame,
    exit_decision_summary: pd.DataFrame,
) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: Protocol101 hold/exit action-advantage foundation",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "This packet implements the engineer-response lifecycle framing: hold/exit should be an action-advantage problem, not a blanket hold-longer rule. The dataset labels `A_hold = Q(hold) - Q(exit now at bid)` for Protocol101 selected trades, while explicitly blocking training because slot opportunity cost and fill realism are not yet included.",
        "",
        "## Coverage",
        "",
        f"- Source trades: `{summary['row_counts']['source_trades']}`",
        f"- Covered trades: `{summary['row_counts']['covered_trades']}`",
        f"- Labeled holding-state rows: `{summary['row_counts']['labeled_state_rows']}`",
        f"- Path skips: `{summary['row_counts']['path_skips']}`",
        "",
        "## Exit-State Audit",
        "",
        f"- Protocol101 exit states audited: `{summary['exit_state_audit']['rows']}`",
        f"- Oracle-best hold fraction at Protocol101 exit: `{summary['exit_state_audit']['oracle_hold_fraction_at_protocol101_exit']:.3f}`",
        f"- One-step hold fraction at Protocol101 exit: `{summary['exit_state_audit']['one_step_hold_fraction_at_protocol101_exit']:.3f}`",
        f"- Median `A_hold` at Protocol101 exit: `{summary['exit_state_audit']['median_a_hold_at_protocol101_exit']:.2f}`",
        "",
        "## Interpretation",
        "",
        "The high oracle-best hold fraction is not a recommendation to hold every trade longer. It says many Protocol101 exits had a later better bid somewhere before forced flat, using hindsight labels. The one-step diagnostic and the earlier exact-runner audit are the counterweight: continuation must be learned as a causal state decision with giveback, switching cost, fill uncertainty, and slot opportunity cost, not imposed as a blanket duration rule.",
        "",
        "The actionable question remains: when does `A_hold` stay positive after subtracting the cost of keeping the only trade slot occupied and the cost of exiting/re-entering under live execution uncertainty?",
        "",
        "## Engineer-Response Alignment",
        "",
        "Implemented in this packet:",
        "",
        "- `A_hold = Q(hold) - Q(exit now at bid)` label skeleton.",
        "- Current executable bid is used for `Q(exit)`; no midpoint exit fantasy is introduced.",
        "- Causal post-entry state features include PnL path, MFE/MAE, giveback, velocities, spread, time, and score context.",
        "- Future/path columns are marked label-only and excluded from the feature contract.",
        "",
        "Still missing before training:",
        "",
        "- Counterfactual flat-slot opportunity cost.",
        "- Explicit switching cost for exit/re-entry churn.",
        "- Calibrated fill, latency, and quote-age uncertainty.",
        "- Distributional risk/uncertainty targets, not just mean or future-best labels.",
        "- Train/live parity for the exact lifecycle policy and candidate set.",
        "- Untouched validation and formal overfit controls.",
        "",
        "## Split Summary",
        "",
        "| split | rows | trades | hold frac | one-step hold frac | median A_hold | median current PnL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in split_summary.iterrows():
        lines.append(
            f"| {row['reported_split']} | {int(row['rows'])} | {int(row['trades'])} | "
            f"{row['hold_fraction_oracle_best']:.3f} | {row['one_step_hold_fraction']:.3f} | "
            f"{row['median_a_hold']:.2f} | {row['median_current_pnl']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol101 Exit Decision Summary",
            "",
            "| split | exit reason | oracle action | rows | median A_hold | median current PnL |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in exit_decision_summary.iterrows():
        lines.append(
            f"| {row['reported_split']} | {row['exit_reason']} | {row['oracle_holding_action']} | {int(row['rows'])} | "
            f"{row['median_a_hold']:.2f} | {row['median_current_pnl']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            "- Slot opportunity cost is not included because counterfactual-flat Protocol101 actions are still missing.",
            "- Fill realism is not calibrated.",
            "- Future-best labels are label-only diagnostics, not runtime features.",
            "- No model should be trained from this packet until those blockers are resolved or deliberately scoped.",
            "",
            "## Outputs",
            "",
            f"- Dataset: `{output_dir / 'protocol101_hold_exit_action_advantage.parquet'}`",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Split summary: `{output_dir / 'split_summary.csv'}`",
            f"- Exit-state audit: `{output_dir / 'protocol101_exit_state_audit.csv'}`",
            f"- Exit decision summary: `{output_dir / 'protocol101_exit_decision_summary.csv'}`",
            f"- Path skips: `{output_dir / 'path_skips.csv'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def base_skip(trade: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "reported_split": str(trade.get("reported_split", "")),
        "seed": int(trade.get("seed", 0)),
        "session": str(trade.get("session", "")),
        "candidate_uid": str(trade.get("candidate_uid", "")),
        "contract_id": str(trade.get("contract_id", "")),
        "right": str(trade.get("right", "")),
        "skip_reason": reason,
    }


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = load_protocol101_selected_trades(Path(args.trades))
    rows, skips = build_hold_exit_rows(
        trades,
        normalized_dir=Path(args.normalized_dir),
        forced_flat_time=str(args.forced_flat_time),
    )
    frame = pd.DataFrame(rows)
    skip_frame = pd.DataFrame(skips)
    out_path = output_dir / "protocol101_hold_exit_action_advantage.parquet"
    frame.to_parquet(out_path, index=False)
    skip_frame.to_csv(output_dir / "path_skips.csv", index=False)
    split_summary = build_split_summary(frame)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    exit_audit = build_exit_state_audit(frame)
    exit_audit.to_csv(output_dir / "protocol101_exit_state_audit.csv", index=False)
    exit_reason_summary = build_exit_reason_summary(frame)
    exit_reason_summary.to_csv(output_dir / "exit_reason_summary.csv", index=False)
    exit_decision_summary = build_protocol101_exit_decision_summary(exit_audit)
    exit_decision_summary.to_csv(output_dir / "protocol101_exit_decision_summary.csv", index=False)
    feature_contract = build_feature_contract()
    (output_dir / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2, sort_keys=True) + "\n")
    summary = build_summary(frame, skip_frame, source_trades=len(trades), output_dataset=out_path, feature_contract=feature_contract)
    summary["source_trades"] = str(args.trades)
    summary["normalized_dir"] = str(args.normalized_dir)
    summary["forced_flat_time"] = str(args.forced_flat_time)
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "dataset": str(out_path),
        "split_summary": str(output_dir / "split_summary.csv"),
        "exit_reason_summary": str(output_dir / "exit_reason_summary.csv"),
        "protocol101_exit_state_audit": str(output_dir / "protocol101_exit_state_audit.csv"),
        "protocol101_exit_decision_summary": str(output_dir / "protocol101_exit_decision_summary.csv"),
        "feature_contract": str(output_dir / "feature_contract.json"),
        "path_skips": str(output_dir / "path_skips.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = write_report(output_dir, summary, split_summary, exit_decision_summary)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--forced-flat-time", default="15:55")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
