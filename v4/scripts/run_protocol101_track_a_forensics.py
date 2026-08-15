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

from v4.scripts.run_protocol101_strategy_forensics_packet import (  # noqa: E402
    _fmt_money,
    _profit_factor,
    enrich_protocol101_trades,
    group_summary,
)


ROLE_LABEL = "AUDIT_PROTOCOL101_TRACK_A_FORENSICS_V1"
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_track_a_forensics_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_TRACK_A_FORENSICS_V1.md"
)
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv")
DEFAULT_ALL_SEED_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/research_all_seed_trades.csv")
DEFAULT_FULL_PATH_ORACLE = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_199_lifecycle_full_path_oracle/full_path_oracle_rows.csv"
)
DEFAULT_EXACT_SELECTED_RUNNER_PATHS = Path(
    "v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1/exact_selected_runner_paths.csv"
)
DEFAULT_HOLD_EXIT_FOUNDATION_SUMMARY = Path(
    "v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/summary.json"
)
DEFAULT_INTERNAL_SLOT_COUNTERFACTUAL_SUMMARY = Path(
    "v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/summary.json"
)
DEFAULT_SLOT_COST_ARCHETYPE_SUMMARY = Path(
    "v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1/summary.json"
)
DEFAULT_BASELINE_ACTIONS = Path(
    "v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet"
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def classify_hard_stop_mechanism(row: pd.Series) -> tuple[str, str]:
    pnl = _finite(row.get("pnl"))
    mfe = _finite(row.get("path_mfe"))
    duration = _finite(row.get("duration_minutes"), default=float("nan"))
    score_margin = _finite(row.get("score_margin"), default=float("nan"))
    spread = _finite(row.get("entry_spread_dollars"), default=float("nan"))
    spread_over_mid = _finite(row.get("spread_over_mid"), default=float("nan"))
    quote_gap = _finite(row.get("quote_gap_seconds"), default=0.0)

    if quote_gap > 0 or (math.isfinite(spread) and spread > 50.0) or (math.isfinite(spread_over_mid) and spread_over_mid > 0.03):
        return (
            "execution_or_quote_fragility_candidate",
            "Do not model-train yet; first test a quote freshness/spread rejection gate on matched non-hard-stop controls.",
        )
    if pnl < 0 and mfe >= 300.0:
        return (
            "path_management_candidate",
            "Test a post-entry early-MFE giveback/loss guard; this is lifecycle state evidence, not a new entry model.",
        )
    if math.isfinite(score_margin) and score_margin < 0.25:
        return (
            "low_margin_entry_candidate",
            "Test whether low-margin Protocol101 entries have negative tail utility before adding an entry rejection gate.",
        )
    if math.isfinite(duration) and duration <= 8.0:
        return (
            "fast_adverse_move_candidate",
            "Inspect pre-entry exhaustion/reversal context; if not separable, treat as unavoidable cost of the playbook.",
        )
    return (
        "unclassified_or_unavoidable_candidate",
        "No model should act on this row without manual chart/path review and matched controls.",
    )


def build_hard_stop_classification(trades: pd.DataFrame) -> pd.DataFrame:
    hard = trades[trades["exit_reason"].astype(str).eq("hard_stop")].copy()
    if hard.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "session",
                "decision_time",
                "exit_time",
                "right",
                "moneyness",
                "premium_paid",
                "score_margin",
                "entry_spread_dollars",
                "duration_minutes",
                "pnl",
                "path_mfe",
                "path_mae",
                "mechanism",
                "next_test",
                "candidate_uid",
            ]
        )
    classifications = hard.apply(classify_hard_stop_mechanism, axis=1)
    hard["mechanism"] = [item[0] for item in classifications]
    hard["next_test"] = [item[1] for item in classifications]
    cols = [
        "segment",
        "session",
        "decision_time",
        "exit_time",
        "right",
        "moneyness",
        "premium_paid",
        "score_margin",
        "entry_spread_dollars",
        "duration_minutes",
        "pnl",
        "path_mfe",
        "path_mae",
        "mechanism",
        "next_test",
        "contract_id",
        "candidate_uid",
    ]
    return hard[cols].sort_values(["mechanism", "pnl"], kind="stable")


def build_hard_stop_mechanism_summary(classified: pd.DataFrame) -> pd.DataFrame:
    if classified.empty:
        return pd.DataFrame(columns=["mechanism", "trades", "pnl", "median_mfe", "median_duration_minutes"])
    rows: list[dict[str, Any]] = []
    for mechanism, group in classified.groupby("mechanism", dropna=False):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "mechanism": mechanism,
                "trades": int(len(group)),
                "pnl": float(pnl.sum()),
                "avg_pnl": float(pnl.mean()) if len(group) else 0.0,
                "median_mfe": float(pd.to_numeric(group["path_mfe"], errors="coerce").median()),
                "median_mae": float(pd.to_numeric(group["path_mae"], errors="coerce").median()),
                "median_duration_minutes": float(pd.to_numeric(group["duration_minutes"], errors="coerce").median()),
                "next_test": str(group["next_test"].mode().iloc[0]) if not group["next_test"].mode().empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["pnl", "trades"], ascending=[True, False], kind="stable")


def classify_losing_day(group: pd.DataFrame) -> str:
    pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
    total_loss = abs(float(pnl.sum()))
    if total_loss <= 0:
        return "not_losing_day"
    hard_loss = abs(float(group.loc[group["exit_reason"].astype(str).eq("hard_stop"), "pnl"].sum()))
    early_mfe_loss = abs(float(group.loc[group["early_mfe_then_loss"], "pnl"].sum()))
    largest_loss = abs(float(pnl.min()))
    loss_trades = int((pnl < 0).sum())
    if hard_loss / total_loss >= 0.50:
        return "hard_stop_dominated"
    if early_mfe_loss / total_loss >= 0.50:
        return "path_management_dominated"
    if largest_loss / total_loss >= 0.50:
        return "single_trade_concentration"
    if loss_trades >= 3:
        return "multi_trade_churn_or_chop"
    return "mixed_small_sample"


def build_losing_day_mechanisms(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (segment, day), group in trades.groupby(["segment", "day"], dropna=False):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        total = float(pnl.sum())
        if total >= 0:
            continue
        hard = group[group["exit_reason"].astype(str).eq("hard_stop")]
        early = group[group["early_mfe_then_loss"]]
        rows.append(
            {
                "segment": segment,
                "day": day,
                "mechanism": classify_losing_day(group),
                "trades": int(len(group)),
                "loss_trades": int((pnl < 0).sum()),
                "pnl": total,
                "win_rate": float((pnl > 0).mean()) if len(group) else 0.0,
                "hard_stop_trades": int(len(hard)),
                "hard_stop_pnl": float(pd.to_numeric(hard["pnl"], errors="coerce").fillna(0.0).sum()) if not hard.empty else 0.0,
                "early_mfe_loss_trades": int(len(early)),
                "early_mfe_loss_pnl": float(pd.to_numeric(early["pnl"], errors="coerce").fillna(0.0).sum()) if not early.empty else 0.0,
                "worst_trade_pnl": float(pnl.min()),
                "dominant_side": str(group["right"].mode().iloc[0]) if not group["right"].mode().empty else "",
                "dominant_exit_reason": str(group["exit_reason"].mode().iloc[0])
                if not group["exit_reason"].mode().empty
                else "",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("pnl", kind="stable")


def classify_runner_state(row: pd.Series) -> str:
    post_best = _finite(row.get("post_frozen_best_minus_frozen_pnl"))
    forced_delta = _finite(row.get("forced_flat_minus_frozen_pnl"))
    oracle_delta = _finite(row.get("oracle_minus_frozen_pnl"))
    if post_best >= 500.0 and forced_delta >= 0.0:
        return "runner_extension_candidate"
    if post_best >= 500.0 and forced_delta < 0.0:
        return "giveback_guard_required"
    if oracle_delta <= 100.0:
        return "do_not_extend_or_already_right"
    if forced_delta <= -500.0:
        return "extension_destroys_value"
    return "ambiguous_runner_state"


def build_runner_extension_audit(full_path: pd.DataFrame) -> pd.DataFrame:
    if full_path.empty:
        return pd.DataFrame()
    frame = full_path.copy()
    frame["runner_state"] = frame.apply(classify_runner_state, axis=1)
    frame["post_exit_positive"] = pd.to_numeric(frame["post_frozen_best_minus_frozen_pnl"], errors="coerce").fillna(0.0) > 0
    rows: list[dict[str, Any]] = []
    keys = ["reported_split", "runner_state", "frozen_exit_reason", "right", "time_bucket"]
    for key_values, group in frame.groupby(keys, dropna=False, observed=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        frozen = pd.to_numeric(group["frozen_pnl"], errors="coerce").fillna(0.0)
        post_best = pd.to_numeric(group["post_frozen_best_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        forced = pd.to_numeric(group["forced_flat_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        oracle = pd.to_numeric(group["oracle_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "rows": int(len(group)),
                "frozen_pnl": float(frozen.sum()),
                "post_exit_best_delta": float(post_best.sum()),
                "oracle_delta": float(oracle.sum()),
                "forced_flat_delta": float(forced.sum()),
                "post_exit_positive_rate": float((post_best > 0).mean()) if len(group) else 0.0,
                "forced_flat_positive_rate": float((forced > 0).mean()) if len(group) else 0.0,
                "median_post_exit_best_delta": float(post_best.median()),
                "median_forced_flat_delta": float(forced.median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["post_exit_best_delta", "rows"], ascending=[False, False], kind="stable")


def build_runner_extension_examples(full_path: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    if full_path.empty:
        return pd.DataFrame()
    frame = full_path.copy()
    frame["runner_state"] = frame.apply(classify_runner_state, axis=1)
    cols = [
        "reported_split",
        "session",
        "decision_time",
        "frozen_exit_time",
        "post_frozen_best_time",
        "forced_flat_time",
        "contract_id",
        "right",
        "entry_ask",
        "frozen_exit_reason",
        "frozen_pnl",
        "post_frozen_best_minus_frozen_pnl",
        "forced_flat_minus_frozen_pnl",
        "oracle_minus_frozen_pnl",
        "runner_state",
    ]
    available = [col for col in cols if col in frame.columns]
    return frame[available].sort_values("post_frozen_best_minus_frozen_pnl", ascending=False, kind="stable").head(limit)


def build_selected_trade_runner_join(trades: pd.DataFrame, full_path: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if trades.empty or full_path.empty:
        return pd.DataFrame(), {
            "selected_trades": int(len(trades)),
            "matched_rows": 0,
            "matched_unique_trade_keys": 0,
            "match_rate": 0.0,
            "status": "blocked_missing_selected_trade_post_exit_paths",
        }
    left = trades.copy()
    right = full_path.copy()
    left["decision_dt_join"] = pd.to_datetime(left["decision_time"], utc=True, errors="coerce").dt.tz_convert(None)
    right["decision_dt_join"] = pd.to_datetime(right["decision_time"], utc=True, errors="coerce").dt.tz_convert(None)
    join_keys = ["seed", "session", "decision_dt_join", "contract_id", "right"]
    duplicate_oracle_matches = int(right.duplicated(join_keys, keep=False).sum())
    if not right.empty:
        right = right.sort_values(["oracle_minus_frozen_pnl", "post_frozen_best_minus_frozen_pnl"], ascending=False, kind="stable")
        right = right.drop_duplicates(join_keys, keep="first")
    merged = left.merge(right, on=join_keys, how="left", suffixes=("_trade", "_oracle"), indicator=True)
    merged["runner_state"] = merged.apply(classify_runner_state, axis=1)
    merged["selected_runner_match_status"] = np.where(merged["_merge"].astype(str).eq("both"), "matched", "missing")
    unique_key_cols = ["seed", "session", "decision_dt_join", "contract_id", "right"]
    matched = merged[merged["_merge"].astype(str).eq("both")]
    matched_unique = int(matched[unique_key_cols].drop_duplicates().shape[0]) if not matched.empty else 0
    selected_unique = int(left[unique_key_cols].drop_duplicates().shape[0])
    summary = {
        "selected_trades": int(len(left)),
        "matched_rows": int(len(matched)),
        "matched_unique_trade_keys": matched_unique,
        "match_rate": float(matched_unique / selected_unique) if selected_unique else 0.0,
        "ambiguous_oracle_match_rows_before_dedup": duplicate_oracle_matches,
        "status": "partial_selected_trade_post_exit_path_coverage"
        if matched_unique > 0
        else "blocked_missing_selected_trade_post_exit_paths",
    }
    cols = [
        "segment",
        "session",
        "decision_time_trade",
        "exit_time",
        "contract_id",
        "right",
        "pnl",
        "path_mfe",
        "path_mae",
        "frozen_pnl",
        "post_frozen_best_minus_frozen_pnl",
        "forced_flat_minus_frozen_pnl",
        "oracle_minus_frozen_pnl",
        "runner_state",
        "selected_runner_match_status",
        "candidate_uid",
    ]
    available = [col for col in cols if col in merged.columns]
    return merged[available].sort_values(["selected_runner_match_status", "segment", "decision_time_trade"], kind="stable"), summary


def build_exact_runner_state_summary(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame(
            columns=[
                "runner_state",
                "rows",
                "frozen_pnl",
                "post_exit_best_delta",
                "forced_flat_delta",
                "material_continuation_rate",
                "forced_flat_positive_rate",
            ]
        )
    frame = paths.copy()
    if "runner_state" not in frame.columns:
        frame["runner_state"] = frame.apply(classify_runner_state, axis=1)
    rows: list[dict[str, Any]] = []
    for state, group in frame.groupby("runner_state", dropna=False):
        post_delta = pd.to_numeric(group["post_frozen_best_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        forced_delta = pd.to_numeric(group["forced_flat_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        material = pd.to_numeric(group.get("material_continuation_after_frozen_exit", pd.Series(False, index=group.index)), errors="coerce").fillna(0.0)
        rows.append(
            {
                "runner_state": state,
                "rows": int(len(group)),
                "frozen_pnl": float(pd.to_numeric(group["frozen_pnl"], errors="coerce").fillna(0.0).sum()),
                "post_exit_best_delta": float(post_delta.sum()),
                "forced_flat_delta": float(forced_delta.sum()),
                "material_continuation_rate": float(material.mean()) if len(group) else 0.0,
                "post_exit_positive_rate": float((post_delta > 0).mean()) if len(group) else 0.0,
                "forced_flat_positive_rate": float((forced_delta > 0).mean()) if len(group) else 0.0,
                "median_post_exit_best_delta": float(post_delta.median()),
                "median_forced_flat_delta": float(forced_delta.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("post_exit_best_delta", ascending=False, kind="stable")


def summarize_exact_runner_paths(paths: pd.DataFrame, *, selected_trades: int) -> dict[str, Any]:
    if paths.empty:
        return {
            "status": "missing_exact_selected_runner_paths",
            "exact_path_rows": 0,
            "selected_trades": int(selected_trades),
            "coverage": 0.0,
        }
    post_delta = pd.to_numeric(paths["post_frozen_best_minus_frozen_pnl"], errors="coerce").fillna(0.0)
    forced_delta = pd.to_numeric(paths["forced_flat_minus_frozen_pnl"], errors="coerce").fillna(0.0)
    runner_state = paths.get("runner_state", pd.Series("", index=paths.index)).astype(str)
    return {
        "status": "exact_selected_runner_paths_available",
        "exact_path_rows": int(len(paths)),
        "selected_trades": int(selected_trades),
        "coverage": float(len(paths) / selected_trades) if selected_trades else 0.0,
        "runner_or_giveback_guard_rows": int(runner_state.isin(["runner_extension_candidate", "giveback_guard_required"]).sum()),
        "post_exit_best_delta": float(post_delta.sum()),
        "forced_flat_delta": float(forced_delta.sum()),
        "forced_flat_positive_rate": float((forced_delta > 0.0).mean()) if len(paths) else 0.0,
    }


def build_internal_slot_cost_proxy(event_actions: pd.DataFrame, all_seed_trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if event_actions.empty:
        return pd.DataFrame(
            [
                {
                    "split": "all",
                    "status": "blocked_missing_protocol101_baseline_event_actions",
                    "rows": 0,
                    "holding_rows": 0,
                    "enter_rows": 0,
                    "observable_blocked_enter_rows": 0,
                    "observable_blocked_pnl": 0.0,
                    "why_blocked": "No baseline event-action attachment exists.",
                }
            ]
        )
    actions = event_actions.copy()
    action_col = "protocol101_action"
    for split, group in actions.groupby("split", dropna=False):
        holding = group[group[action_col].astype(str).eq("holding")]
        enters = group[group[action_col].astype(str).eq("enter")]
        rows.append(
            {
                "split": split,
                "status": "blocked_missing_counterfactual_flat_protocol101_actions",
                "rows": int(len(group)),
                "holding_rows": int(len(holding)),
                "enter_rows": int(len(enters)),
                "observable_blocked_enter_rows": 0,
                "observable_blocked_pnl": 0.0,
                "same_seed_independent_overlap_rows": int(count_same_seed_overlaps(all_seed_trades, split=split)),
                "why_blocked": (
                    "The deployed action stream marks rows as holding during open intervals, so it provides a lower bound of zero "
                    "rather than the counterfactual-flat Protocol101 entries needed to price internal slot cost."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("split", kind="stable")


def count_same_seed_overlaps(all_seed_trades: pd.DataFrame, *, split: Any | None = None) -> int:
    if all_seed_trades.empty:
        return 0
    frame = all_seed_trades.copy()
    if split is not None and "segment" in frame.columns:
        frame = frame[frame["segment"].astype(str).eq(str(split))]
    if frame.empty:
        return 0
    frame["entry_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["exit_dt"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["entry_dt", "exit_dt"]).sort_values(["seed", "entry_dt", "exit_dt"])
    overlaps = 0
    for _, group in frame.groupby("seed", dropna=False):
        entries = group["entry_dt"].to_numpy(dtype="datetime64[ns]")
        exits = group["exit_dt"].to_numpy(dtype="datetime64[ns]")
        for idx, exit_dt in enumerate(exits):
            later_entries = entries[idx + 1 :]
            overlaps += int((later_entries < exit_dt).sum())
    return overlaps


def build_side_strategy_audit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return group_summary(trades, ["right", "time_bucket", "moneyness", "exit_reason"])


def build_score_reliability_deep(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["right", "time_bucket", "score_margin_bucket"]
    for key_values, group in trades.groupby(keys, dropna=False, observed=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        score = pd.to_numeric(group["score_margin"], errors="coerce")
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "trades": int(len(group)),
                "pnl": float(pnl.sum()),
                "avg_pnl": float(pnl.mean()) if len(group) else 0.0,
                "win_rate": float((pnl > 0).mean()) if len(group) else 0.0,
                "profit_factor": _profit_factor(pnl),
                "median_mfe": float(pd.to_numeric(group["path_mfe"], errors="coerce").median()),
                "median_mae": float(pd.to_numeric(group["path_mae"], errors="coerce").median()),
                "hard_stop_rate": float(group["exit_reason"].astype(str).eq("hard_stop").mean()),
                "spearman_score_margin_vs_pnl": float(score.corr(pnl, method="spearman")) if score.notna().sum() >= 2 else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["right", "time_bucket", "score_margin_bucket"], kind="stable")


def build_next_track_a_actions(
    hard_summary: pd.DataFrame,
    runner_audit: pd.DataFrame,
    slot_proxy: pd.DataFrame,
    selected_runner_summary: dict[str, Any] | None = None,
    exact_runner_summary: dict[str, Any] | None = None,
    hold_exit_summary: dict[str, Any] | None = None,
    internal_slot_counterfactual_summary: dict[str, Any] | None = None,
    slot_cost_archetype_summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    path_management_rows = 0
    if not hard_summary.empty:
        match = hard_summary[hard_summary["mechanism"].astype(str).eq("path_management_candidate")]
        path_management_rows = int(match["trades"].sum()) if not match.empty else 0
    runner_rows = 0
    if not runner_audit.empty:
        runner_rows = int(
            runner_audit[
                runner_audit["runner_state"].astype(str).isin(["runner_extension_candidate", "giveback_guard_required"])
            ]["rows"].sum()
        )
    slot_blocked = bool(not slot_proxy.empty and slot_proxy["status"].astype(str).str.startswith("blocked").any())
    selected_runner_summary = selected_runner_summary or {}
    selected_match_rate = float(selected_runner_summary.get("match_rate", 0.0))
    exact_runner_summary = exact_runner_summary or {}
    exact_rows = int(exact_runner_summary.get("exact_path_rows", 0))
    exact_coverage = float(exact_runner_summary.get("coverage", 0.0))
    exact_status = (
        "exact_path_diagnostic_ready_no_training"
        if exact_rows > 0
        else "diagnostic_ready_from_full_path_oracle_not_promotion"
    )
    exact_finding = (
        f"{exact_rows} exact selected-trade paths attached ({exact_coverage:.3f} coverage); "
        f"{int(exact_runner_summary.get('runner_or_giveback_guard_rows', runner_rows))} rows show runner/giveback-guard evidence."
        if exact_rows > 0
        else (
            f"{runner_rows} full-path-oracle rows show runner or giveback-guard evidence; exact selected-trade "
            f"post-exit path match rate is {selected_match_rate:.3f}."
        )
    )
    exact_next = (
        "Design a causal confirmed-MFE runner/giveback diagnostic using only in-trade state available before extension."
        if exact_rows > 0
        else "Attach exact Protocol101 selected-trade post-exit paths for the full selected trade log, then test runner transition with giveback guard."
    )
    hold_exit_summary = hold_exit_summary or {}
    hold_exit_rows = int(hold_exit_summary.get("row_counts", {}).get("labeled_state_rows", 0))
    exit_audit = hold_exit_summary.get("exit_state_audit", {})
    hold_exit_status = (
        "foundation_complete_training_blocked" if hold_exit_rows > 0 else "blocked_missing_hold_exit_action_advantage_foundation"
    )
    hold_exit_finding = (
        f"{hold_exit_rows} hold/exit state rows; oracle-best exit hold fraction "
        f"{float(exit_audit.get('oracle_hold_fraction_at_protocol101_exit', 0.0)):.3f}, one-step exit hold fraction "
        f"{float(exit_audit.get('one_step_hold_fraction_at_protocol101_exit', 0.0)):.3f}."
        if hold_exit_rows > 0
        else "Hold/exit action-advantage foundation has not been generated."
    )
    internal_slot_counterfactual_summary = internal_slot_counterfactual_summary or {}
    internal_slot_blocked = int(internal_slot_counterfactual_summary.get("counts", {}).get("blocked_entry_events", 0))
    internal_slot_status = (
        "counterfactual_flat_diagnostic_ready_no_training" if internal_slot_blocked > 0 else "blocked_counterfactual_flat_required"
    )
    internal_slot_finding = (
        f"Flat-mode Protocol101 replay found {internal_slot_blocked} model-approved entries blocked by actual open positions; "
        f"best-blocked-minus-open total is "
        f"{float(internal_slot_counterfactual_summary.get('aggregate', {}).get('best_blocked_minus_open_total', 0.0)):.0f}."
        if internal_slot_blocked > 0
        else "Current deployed action stream cannot reveal Protocol101 entries that would have fired if the account were flat."
    )
    internal_slot_next = (
        "Decompose blocked entries by open-trade archetype and add mutually exclusive replay plus fill/latency realism before any learned defer gate."
        if internal_slot_blocked > 0
        else "Build counterfactual-flat Protocol101 event replay and attribute later feasible entries inside actual open intervals."
    )
    slot_cost_archetype_summary = slot_cost_archetype_summary or {}
    slot_archetype_rows = int(slot_cost_archetype_summary.get("counts", {}).get("open_archetype_rows", 0))
    slot_archetype_status = (
        "archetype_decomposition_complete_manual_review_required"
        if slot_archetype_rows > 0
        else "blocked_missing_slot_cost_archetype_decomposition"
    )
    top_theses = slot_cost_archetype_summary.get("top_thesis_tests", [])
    top_thesis = str(top_theses[0].get("thesis", "")) if top_theses else ""
    slot_archetype_finding = (
        f"{slot_archetype_rows} slot-cost archetype rows; top thesis is {top_thesis}."
        if slot_archetype_rows > 0
        else "Slot-cost archetypes have not been decomposed."
    )
    rows = [
        {
            "priority": 1,
            "track_a_item": "PROTOCOL101_HARD_STOP_AUTOPSY_V1",
            "status": "diagnostic_ready_no_training",
            "finding": f"{path_management_rows} hard-stop rows are path-management candidates under current causal proxy rules.",
            "next_action": "Manually review hard-stop rows, then build matched-control rejection/lifecycle tests before any model.",
        },
        {
            "priority": 2,
            "track_a_item": "PROTOCOL101_CONFIRMED_MFE_RUNNER_AUDIT_V1",
            "status": exact_status,
            "finding": exact_finding,
            "next_action": exact_next,
        },
        {
            "priority": 3,
            "track_a_item": "PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_V1",
            "status": hold_exit_status,
            "finding": hold_exit_finding,
            "next_action": "Add counterfactual flat-slot opportunity cost, switching cost, fill/latency penalties, and distributional risk targets before lifecycle training.",
        },
        {
            "priority": 4,
            "track_a_item": "PROTOCOL101_INTERNAL_SLOT_COST_V1",
            "status": internal_slot_status if slot_blocked else "ready",
            "finding": internal_slot_finding,
            "next_action": internal_slot_next,
        },
        {
            "priority": 5,
            "track_a_item": "PROTOCOL101_SLOT_COST_ARCHETYPE_DECOMPOSITION_V1",
            "status": slot_archetype_status,
            "finding": slot_archetype_finding,
            "next_action": "Stop automated model work here; manually review top rows and choose exactly one named strategy hypothesis.",
        },
        {
            "priority": 6,
            "track_a_item": "PROTOCOL101_SIDE_SPECIFIC_STRATEGY_AUDIT_V1",
            "status": "diagnostic_ready_no_training",
            "finding": "Side-specific summaries are now emitted for call/put mechanism review.",
            "next_action": "Use side audit to decide whether puts and calls need separate playbook notes.",
        },
        {
            "priority": 7,
            "track_a_item": "PROTOCOL101_SCORE_RELIABILITY_V1",
            "status": "selected_trade_proxy_only",
            "finding": "Score reliability remains selected-trade-only until rejected candidates are joined.",
            "next_action": "Join rejected candidates for true calibration; do not treat score margin as utility yet.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_packet(
    trades: pd.DataFrame,
    hard_classification: pd.DataFrame,
    runner_audit: pd.DataFrame,
    slot_proxy: pd.DataFrame,
    selected_runner_summary: dict[str, Any] | None = None,
    exact_runner_summary: dict[str, Any] | None = None,
    hold_exit_summary: dict[str, Any] | None = None,
    internal_slot_counterfactual_summary: dict[str, Any] | None = None,
    slot_cost_archetype_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    hard_pnl = float(pd.to_numeric(hard_classification.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    runner_rows = (
        int(
            runner_audit[
                runner_audit["runner_state"].astype(str).isin(["runner_extension_candidate", "giveback_guard_required"])
            ]["rows"].sum()
        )
        if not runner_audit.empty
        else 0
    )
    exact_runner_summary = exact_runner_summary or {}
    hold_exit_summary = hold_exit_summary or {}
    internal_slot_counterfactual_summary = internal_slot_counterfactual_summary or {}
    slot_cost_archetype_summary = slot_cost_archetype_summary or {}
    exact_rows = int(exact_runner_summary.get("exact_path_rows", 0))
    hold_exit_rows = int(hold_exit_summary.get("row_counts", {}).get("labeled_state_rows", 0))
    internal_slot_blocked = int(internal_slot_counterfactual_summary.get("counts", {}).get("blocked_entry_events", 0))
    slot_archetype_rows = int(slot_cost_archetype_summary.get("counts", {}).get("open_archetype_rows", 0))
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "Track A Protocol101 forensics / hard-stop, runner, slot-cost, side, and score diagnostics",
        "decision": "protocol101_track_a_forensics_partial_complete_training_still_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "counts": {
            "protocol101_seed1_trades": int(len(trades)),
            "hard_stop_rows": int(len(hard_classification)),
            "hard_stop_pnl": hard_pnl,
            "runner_or_giveback_guard_rows": runner_rows,
            "exact_selected_runner_path_rows": exact_rows,
            "hold_exit_action_advantage_rows": hold_exit_rows,
            "internal_slot_counterfactual_blocked_events": internal_slot_blocked,
            "slot_cost_archetype_rows": slot_archetype_rows,
            "slot_proxy_splits": int(len(slot_proxy)),
        },
        "selected_trade_runner_join": selected_runner_summary or {},
        "exact_selected_runner_paths": exact_runner_summary,
        "hold_exit_action_advantage_foundation": hold_exit_summary,
        "internal_slot_cost_counterfactual": internal_slot_counterfactual_summary,
        "slot_cost_archetype_decomposition": slot_cost_archetype_summary,
        "headline": {
            "pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()) if len(trades) else 0.0,
            "profit_factor": _profit_factor(pnl),
            "hard_stop_pnl": hard_pnl,
        },
        "blockers": [
            "counterfactual_flat_protocol101_actions_missing"
            if internal_slot_blocked <= 0
            else "internal_slot_counterfactual_is_upper_bound_not_additive_replay",
            "slot_cost_archetypes_require_manual_strategy_hypothesis_selection"
            if slot_archetype_rows > 0
            else "slot_cost_archetype_decomposition_missing",
            "exact_selected_trade_post_exit_paths_partial_coverage"
            if exact_rows > 0
            else "exact_selected_trade_post_exit_paths_not_attached",
            "causal_runner_giveback_policy_not_defined",
            "hold_exit_slot_switching_fill_distributional_terms_missing",
            "fill_and_latency_realism_still_unresolved",
            "formal_strategy_matrix_and_untouched_holdout_still_required",
        ],
        "challenge_allowed": False,
    }


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    hard_summary: pd.DataFrame,
    losing_days: pd.DataFrame,
    runner_audit: pd.DataFrame,
    selected_runner_summary: dict[str, Any],
    exact_runner_state_summary: pd.DataFrame,
    exact_runner_summary: dict[str, Any],
    hold_exit_summary: dict[str, Any],
    internal_slot_counterfactual_summary: dict[str, Any],
    slot_cost_archetype_summary: dict[str, Any],
    slot_proxy: pd.DataFrame,
    next_actions: pd.DataFrame,
) -> str:
    headline = summary["headline"]
    hold_exit_rows = int(hold_exit_summary.get("row_counts", {}).get("labeled_state_rows", 0))
    hold_exit_exit = hold_exit_summary.get("exit_state_audit", {})
    internal_slot_counts = internal_slot_counterfactual_summary.get("counts", {})
    internal_slot_aggregate = internal_slot_counterfactual_summary.get("aggregate", {})
    internal_slot_splits = internal_slot_counterfactual_summary.get("split_summary", [])
    slot_archetype_counts = slot_cost_archetype_summary.get("counts", {})
    slot_archetype_aggregate = slot_cost_archetype_summary.get("aggregate", {})
    slot_archetype_top = slot_cost_archetype_summary.get("top_thesis_tests", [])
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: Track A Protocol101 forensics packet",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "Track A now has executable diagnostics for the Protocol101 weak points we can inspect from existing artifacts. The packet does not authorize a new model. It narrows the next work to manual hard-stop review, exact post-exit runner paths, counterfactual-flat Protocol101 slot-cost replay, and side/score interpretation.",
        "",
        "## Headline",
        "",
        f"- Seed-1 Protocol101 trades: `{summary['counts']['protocol101_seed1_trades']}`",
        f"- Seed-1 Protocol101 PnL: `{_fmt_money(headline['pnl'])}`",
        f"- Win rate: `{headline['win_rate']:.3f}`",
        f"- Profit factor: `{headline['profit_factor']:.3f}`",
        f"- Hard-stop rows: `{summary['counts']['hard_stop_rows']}` for `{_fmt_money(headline['hard_stop_pnl'])}`",
        f"- Runner/giveback evidence rows from full-path oracle: `{summary['counts']['runner_or_giveback_guard_rows']}`",
        f"- Exact selected-trade runner path rows: `{summary['counts'].get('exact_selected_runner_path_rows', 0)}`",
        f"- Exact selected-trade runner path coverage: `{exact_runner_summary.get('coverage', 0.0):.3f}`",
        f"- Exact selected-trade post-exit path match rate: `{selected_runner_summary.get('match_rate', 0.0):.3f}`",
        f"- Hold/exit action-advantage state rows: `{hold_exit_rows}`",
        f"- Internal slot-cost blocked events: `{int(internal_slot_counts.get('blocked_entry_events', 0))}`",
        f"- Slot-cost archetype rows: `{int(slot_archetype_counts.get('open_archetype_rows', 0))}`",
        "",
        "## Hard-Stop Mechanisms",
        "",
        "| mechanism | trades | pnl | median MFE | median MAE | next test |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in hard_summary.iterrows():
        lines.append(
            f"| {row['mechanism']} | {int(row['trades'])} | {_fmt_money(row['pnl'])} | "
            f"{_fmt_money(row['median_mfe'])} | {_fmt_money(row['median_mae'])} | {row['next_test']} |"
        )
    lines.extend(
        [
            "",
            "## Losing-Day Mechanisms",
            "",
            "| mechanism | days | pnl | trades | hard-stop pnl | early-MFE loss pnl |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not losing_days.empty:
        grouped = (
            losing_days.groupby("mechanism", dropna=False)
            .agg(
                days=("day", "count"),
                pnl=("pnl", "sum"),
                trades=("trades", "sum"),
                hard_stop_pnl=("hard_stop_pnl", "sum"),
                early_mfe_loss_pnl=("early_mfe_loss_pnl", "sum"),
            )
            .reset_index()
            .sort_values("pnl", kind="stable")
        )
        for _, row in grouped.iterrows():
            lines.append(
                f"| {row['mechanism']} | {int(row['days'])} | {_fmt_money(row['pnl'])} | {int(row['trades'])} | "
                f"{_fmt_money(row['hard_stop_pnl'])} | {_fmt_money(row['early_mfe_loss_pnl'])} |"
            )
    lines.extend(
        [
            "",
            "## Runner/Giveback Evidence",
            "",
            "The runner audit now includes an exact selected-trade path packet when available. It remains a hindsight diagnostic, but it is much closer to the actual Protocol101 trade log than the broad Protocol199 oracle.",
            f"Exact selected-trade path coverage is `{exact_runner_summary.get('exact_path_rows', 0)}` of `{exact_runner_summary.get('selected_trades', 0)}` selected trades.",
            "",
            "### Exact Selected-Trade Paths",
            "",
            "| runner state | rows | post-exit best delta | forced-flat delta | post-positive rate | forced-flat positive rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not exact_runner_state_summary.empty:
        for _, row in exact_runner_state_summary.iterrows():
            lines.append(
                f"| {row['runner_state']} | {int(row['rows'])} | {_fmt_money(row['post_exit_best_delta'])} | "
                f"{_fmt_money(row['forced_flat_delta'])} | {row['post_exit_positive_rate']:.3f} | "
                f"{row['forced_flat_positive_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "### Broad Full-Path Oracle Context",
            "",
            f"Legacy broad-oracle exact selected-trade join coverage is `{selected_runner_summary.get('matched_unique_trade_keys', 0)}` of `{selected_runner_summary.get('selected_trades', 0)}` selected trade keys.",
            "",
            "| runner state | rows | post-exit best delta | forced-flat delta | post-positive rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if not runner_audit.empty:
        grouped_runner = (
            runner_audit.groupby("runner_state", dropna=False)
            .agg(
                rows=("rows", "sum"),
                post_exit_best_delta=("post_exit_best_delta", "sum"),
                forced_flat_delta=("forced_flat_delta", "sum"),
                post_exit_positive_rate=("post_exit_positive_rate", "mean"),
            )
            .reset_index()
            .sort_values("post_exit_best_delta", ascending=False, kind="stable")
        )
        for _, row in grouped_runner.iterrows():
            lines.append(
                f"| {row['runner_state']} | {int(row['rows'])} | {_fmt_money(row['post_exit_best_delta'])} | "
                f"{_fmt_money(row['forced_flat_delta'])} | {row['post_exit_positive_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Hold/Exit Action-Advantage Foundation",
            "",
            "The lifecycle direction from `engineer-response.md` is now represented as a foundation artifact: `A_hold = Q(hold) - Q(exit now at bid)`. This still does not authorize training. Its purpose is to stop framing runner work as `hold longer` and instead frame it as a causal continuation-advantage decision with explicit missing costs.",
            "",
            f"- Labeled holding-state rows: `{hold_exit_rows}`",
            f"- Covered Protocol101 selected trades: `{hold_exit_summary.get('row_counts', {}).get('covered_trades', 0)}`",
            f"- Oracle-best hold fraction at Protocol101 exit: `{float(hold_exit_exit.get('oracle_hold_fraction_at_protocol101_exit', 0.0)):.3f}`",
            f"- One-step hold fraction at Protocol101 exit: `{float(hold_exit_exit.get('one_step_hold_fraction_at_protocol101_exit', 0.0)):.3f}`",
            "",
            "Interpretation: the high future-best hold fraction says later upside often existed, but the low one-step hold fraction and negative forced-flat runner audit show why naive extension is dangerous. The next lifecycle question is whether continuation advantage survives slot opportunity cost, switching cost, fill/latency uncertainty, and distributional tail risk.",
            "",
            "## Internal Slot-Cost Status",
            "",
            "A counterfactual-flat Protocol101 event replay is now available. It scores the frozen Protocol101 event policy at every event as if the account were flat, then attributes model-approved entries that actual strict-serial Protocol101 blocked by already holding one contract.",
            "",
            f"- Hypothetical flat entries: `{int(internal_slot_counts.get('hypothetical_flat_entries', 0))}`",
            f"- Blocked entry events: `{int(internal_slot_counts.get('blocked_entry_events', 0))}`",
            f"- Open trades with blocked entries: `{int(internal_slot_counts.get('open_trades_with_blocked_entries', 0))}`",
            f"- Open trades where best blocked entry beats open trade: `{int(internal_slot_counts.get('open_trades_where_best_blocked_beats_open', 0))}`",
            f"- Best-blocked-minus-open total: `{_fmt_money(float(internal_slot_aggregate.get('best_blocked_minus_open_total', 0.0)))}`",
            "",
            "This is still not a tradable replay: multiple blocked entries inside one open interval are mutually exclusive, and fill/latency realism is not calibrated.",
            "",
            "| split | blocked events | open trades w/ blocked | best-blocked beats open | best-blocked-minus-open |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in internal_slot_splits:
        lines.append(
            f"| {row['reported_split']} | {int(row['blocked_entry_events'])} | "
            f"{int(row['open_trades_with_blocked_entries'])} | "
            f"{int(row['open_trades_where_best_blocked_beats_open'])} | "
            f"{_fmt_money(float(row['best_blocked_minus_open_total']))} |"
        )
    lines.extend(
        [
            "",
            "## Slot-Cost Archetype Decomposition",
            "",
            "The slot-cost branch has reached its useful automated stopping point. The decomposition shows concrete weakness hypotheses, but the next step is manual strategy selection, not another model run.",
            "",
            f"- Open trades with blocked entries: `{int(slot_archetype_counts.get('open_trades_with_blocked_entries', 0))}`",
            f"- Positive slot-cost open trades: `{int(slot_archetype_counts.get('positive_slot_cost_open_trades', 0))}`",
            f"- Best-blocked-minus-open total: `{_fmt_money(float(slot_archetype_aggregate.get('best_blocked_minus_open_total', 0.0)))}`",
            f"- Positive slot-cost total: `{_fmt_money(float(slot_archetype_aggregate.get('positive_slot_cost_total', 0.0)))}`",
            "",
            "| thesis | open trades | slot-cost total | median slot cost | next action |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in slot_archetype_top[:5]:
        lines.append(
            f"| {row['thesis']} | {int(row['open_trades'])} | "
            f"{_fmt_money(float(row['best_blocked_minus_open_total']))} | "
            f"{_fmt_money(float(row['median_slot_cost']))} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "### Deployed-State Action Log",
            "",
            "The older deployed-state action stream is still included as a sanity check. It records open intervals as `holding`, so by itself it cannot reveal counterfactual-flat entries.",
            "",
            "| split | status | holding rows | enter rows | observable blocked enters |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in slot_proxy.iterrows():
        lines.append(
            f"| {row['split']} | {row['status']} | {int(row['holding_rows'])} | {int(row['enter_rows'])} | "
            f"{int(row['observable_blocked_enter_rows'])} |"
        )
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "| priority | item | status | next action |",
            "|---:|---|---|---|",
        ]
    )
    for _, row in next_actions.iterrows():
        lines.append(f"| {int(row['priority'])} | {row['track_a_item']} | {row['status']} | {row['next_action']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Hard-stop classification: `{output_dir / 'hard_stop_classification.csv'}`",
            f"- Hard-stop mechanism summary: `{output_dir / 'hard_stop_mechanism_summary.csv'}`",
            f"- Losing-day mechanisms: `{output_dir / 'losing_day_mechanisms.csv'}`",
            f"- Runner extension audit: `{output_dir / 'runner_extension_audit.csv'}`",
            f"- Runner extension examples: `{output_dir / 'runner_extension_examples.csv'}`",
            f"- Selected-trade runner join: `{output_dir / 'selected_trade_runner_join.csv'}`",
            f"- Exact selected runner state summary: `{output_dir / 'exact_selected_runner_state_summary.csv'}`",
            f"- Internal slot counterfactual summary: `{summary['outputs'].get('internal_slot_counterfactual_summary', '')}`",
            f"- Slot-cost archetype summary: `{summary['outputs'].get('slot_cost_archetype_summary', '')}`",
            f"- Internal slot-cost proxy: `{output_dir / 'internal_slot_cost_proxy.csv'}`",
            f"- Side strategy audit: `{output_dir / 'side_strategy_audit.csv'}`",
            f"- Score reliability deep dive: `{output_dir / 'score_reliability_deep.csv'}`",
            f"- Next actions: `{output_dir / 'track_a_next_actions.csv'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_trades = _read_csv(Path(args.trades))
    if raw_trades.empty:
        raise FileNotFoundError(f"Protocol101 trades missing or empty: {args.trades}")
    trades = enrich_protocol101_trades(raw_trades)
    all_seed_trades = _read_csv(Path(args.all_seed_trades))
    full_path = _read_csv(Path(args.full_path_oracle))
    exact_selected_runner_paths = _read_csv(Path(args.exact_selected_runner_paths))
    hold_exit_summary = _read_json(Path(args.hold_exit_foundation_summary))
    internal_slot_counterfactual_summary = _read_json(Path(args.internal_slot_counterfactual_summary))
    slot_cost_archetype_summary = _read_json(Path(args.slot_cost_archetype_summary))
    event_actions = _read_parquet(Path(args.baseline_actions))

    hard_classification = build_hard_stop_classification(trades)
    hard_summary = build_hard_stop_mechanism_summary(hard_classification)
    losing_days = build_losing_day_mechanisms(trades)
    runner_audit = build_runner_extension_audit(full_path)
    runner_examples = build_runner_extension_examples(full_path)
    selected_runner_join, selected_runner_summary = build_selected_trade_runner_join(trades, full_path)
    exact_runner_state_summary = build_exact_runner_state_summary(exact_selected_runner_paths)
    exact_runner_summary = summarize_exact_runner_paths(exact_selected_runner_paths, selected_trades=len(trades))
    slot_proxy = build_internal_slot_cost_proxy(event_actions, all_seed_trades)
    side_audit = build_side_strategy_audit(trades)
    score_reliability = build_score_reliability_deep(trades)
    next_actions = build_next_track_a_actions(
        hard_summary,
        runner_audit,
        slot_proxy,
        selected_runner_summary,
        exact_runner_summary,
        hold_exit_summary,
        internal_slot_counterfactual_summary,
        slot_cost_archetype_summary,
    )
    summary = summarize_packet(
        trades,
        hard_classification,
        runner_audit,
        slot_proxy,
        selected_runner_summary,
        exact_runner_summary,
        hold_exit_summary,
        internal_slot_counterfactual_summary,
        slot_cost_archetype_summary,
    )

    hard_classification.to_csv(output_dir / "hard_stop_classification.csv", index=False)
    hard_summary.to_csv(output_dir / "hard_stop_mechanism_summary.csv", index=False)
    losing_days.to_csv(output_dir / "losing_day_mechanisms.csv", index=False)
    runner_audit.to_csv(output_dir / "runner_extension_audit.csv", index=False)
    runner_examples.to_csv(output_dir / "runner_extension_examples.csv", index=False)
    selected_runner_join.to_csv(output_dir / "selected_trade_runner_join.csv", index=False)
    exact_runner_state_summary.to_csv(output_dir / "exact_selected_runner_state_summary.csv", index=False)
    slot_proxy.to_csv(output_dir / "internal_slot_cost_proxy.csv", index=False)
    side_audit.to_csv(output_dir / "side_strategy_audit.csv", index=False)
    score_reliability.to_csv(output_dir / "score_reliability_deep.csv", index=False)
    next_actions.to_csv(output_dir / "track_a_next_actions.csv", index=False)

    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "hard_stop_classification": str(output_dir / "hard_stop_classification.csv"),
        "hard_stop_mechanism_summary": str(output_dir / "hard_stop_mechanism_summary.csv"),
        "losing_day_mechanisms": str(output_dir / "losing_day_mechanisms.csv"),
        "runner_extension_audit": str(output_dir / "runner_extension_audit.csv"),
        "runner_extension_examples": str(output_dir / "runner_extension_examples.csv"),
        "selected_trade_runner_join": str(output_dir / "selected_trade_runner_join.csv"),
        "exact_selected_runner_state_summary": str(output_dir / "exact_selected_runner_state_summary.csv"),
        "hold_exit_action_advantage_foundation_summary": str(args.hold_exit_foundation_summary),
        "internal_slot_counterfactual_summary": str(args.internal_slot_counterfactual_summary),
        "slot_cost_archetype_summary": str(args.slot_cost_archetype_summary),
        "slot_cost_archetype_report": str(Path(args.slot_cost_archetype_summary).with_name("report.md")),
        "internal_slot_cost_proxy": str(output_dir / "internal_slot_cost_proxy.csv"),
        "side_strategy_audit": str(output_dir / "side_strategy_audit.csv"),
        "score_reliability_deep": str(output_dir / "score_reliability_deep.csv"),
        "track_a_next_actions": str(output_dir / "track_a_next_actions.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = write_report(
        output_dir,
        summary,
        hard_summary,
        losing_days,
        runner_audit,
        selected_runner_summary,
        exact_runner_state_summary,
        exact_runner_summary,
        hold_exit_summary,
        internal_slot_counterfactual_summary,
        slot_cost_archetype_summary,
        slot_proxy,
        next_actions,
    )
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--all-seed-trades", type=Path, default=DEFAULT_ALL_SEED_TRADES)
    parser.add_argument("--full-path-oracle", type=Path, default=DEFAULT_FULL_PATH_ORACLE)
    parser.add_argument("--exact-selected-runner-paths", type=Path, default=DEFAULT_EXACT_SELECTED_RUNNER_PATHS)
    parser.add_argument("--hold-exit-foundation-summary", type=Path, default=DEFAULT_HOLD_EXIT_FOUNDATION_SUMMARY)
    parser.add_argument("--internal-slot-counterfactual-summary", type=Path, default=DEFAULT_INTERNAL_SLOT_COUNTERFACTUAL_SUMMARY)
    parser.add_argument("--slot-cost-archetype-summary", type=Path, default=DEFAULT_SLOT_COST_ARCHETYPE_SUMMARY)
    parser.add_argument("--baseline-actions", type=Path, default=DEFAULT_BASELINE_ACTIONS)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
