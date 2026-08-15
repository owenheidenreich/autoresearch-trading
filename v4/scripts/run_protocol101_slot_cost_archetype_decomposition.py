"""Protocol101 internal slot-cost archetype decomposition.

This Track A packet decomposes the counterfactual-flat Protocol101 slot-cost
audit by the open trade that consumed the single account slot and by the
blocked Protocol101 signal that could not be taken while that slot was occupied.
It is diagnostic only: no model is trained and Protocol101 remains the paper
default.
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

from v4.scripts.run_protocol101_strategy_forensics_packet import _fmt_money, enrich_protocol101_trades  # noqa: E402


ROLE_LABEL = "AUDIT_PROTOCOL101_SLOT_COST_ARCHETYPE_DECOMPOSITION_V1"
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_slot_cost_archetype_decomposition_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_SLOT_COST_ARCHETYPE_DECOMPOSITION_V1.md"
)
DEFAULT_OPEN_SLOT_SUMMARY = Path(
    "v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/open_trade_slot_summary.csv"
)
DEFAULT_BLOCKED_EVENTS = Path(
    "v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/blocked_protocol101_internal_slot_events.csv"
)
DEFAULT_ALL_SEED_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/research_all_seed_trades.csv"
)


def split_for_join(value: Any) -> str:
    value_s = str(value)
    return "q1_2026" if value_s == "march_2026" else value_s


def load_enriched_actual_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"empty actual trade file: {path}")
    enriched = enrich_protocol101_trades(raw)
    enriched["join_split"] = enriched["segment"].map(split_for_join)
    return enriched


def enrich_open_slots(open_slots: pd.DataFrame, actual_trades: pd.DataFrame) -> pd.DataFrame:
    slots = open_slots.copy()
    slots["join_split"] = slots["reported_split"].map(split_for_join)
    trade_cols = [
        "join_split",
        "seed",
        "session",
        "candidate_uid",
        "contract_id",
        "decision_time",
        "exit_time",
        "right",
        "offset",
        "time_bucket",
        "moneyness",
        "premium_paid",
        "premium_bucket",
        "duration_minutes",
        "duration_bucket",
        "entry_spread_dollars",
        "spread_bucket",
        "score",
        "threshold",
        "score_margin",
        "score_margin_bucket",
        "path_mfe",
        "path_mae",
        "giveback_from_mfe",
        "mfe_capture",
        "early_mfe_then_loss",
        "large_uncaptured_mfe",
        "pnl",
    ]
    trades = actual_trades[trade_cols].rename(
        columns={
            "candidate_uid": "open_trade_candidate_uid",
            "contract_id": "open_trade_contract_id",
            "decision_time": "open_trade_decision_time_source",
            "exit_time": "open_trade_exit_time_source",
            "right": "open_trade_right_source",
            "offset": "open_trade_offset",
            "time_bucket": "open_time_bucket",
            "moneyness": "open_moneyness",
            "premium_paid": "open_premium_paid",
            "premium_bucket": "open_premium_bucket",
            "duration_minutes": "open_duration_minutes",
            "duration_bucket": "open_duration_bucket",
            "entry_spread_dollars": "open_entry_spread_dollars",
            "spread_bucket": "open_spread_bucket",
            "score": "open_score",
            "threshold": "open_threshold",
            "score_margin": "open_score_margin",
            "score_margin_bucket": "open_score_margin_bucket",
            "path_mfe": "open_path_mfe",
            "path_mae": "open_path_mae",
            "giveback_from_mfe": "open_giveback_from_mfe",
            "mfe_capture": "open_mfe_capture",
            "early_mfe_then_loss": "open_early_mfe_then_loss",
            "large_uncaptured_mfe": "open_large_uncaptured_mfe",
            "pnl": "open_trade_pnl_source",
        }
    )
    out = slots.merge(
        trades,
        on=["join_split", "seed", "session", "open_trade_candidate_uid", "open_trade_contract_id"],
        how="left",
    )
    out["best_blocked_minus_open_pnl"] = -pd.to_numeric(out["open_minus_best_blocked_pnl"], errors="coerce").fillna(0.0)
    out["slot_cost_bucket"] = out["best_blocked_minus_open_pnl"].map(slot_cost_bucket)
    out["open_outcome_bucket"] = out["open_trade_pnl"].map(open_outcome_bucket)
    out["open_mfe_bucket"] = out["open_path_mfe"].map(mfe_bucket)
    out["open_giveback_bucket"] = out["open_giveback_from_mfe"].map(giveback_bucket)
    out["open_trade_join_status"] = np.where(out["open_duration_minutes"].notna(), "matched_actual_trade", "missing_actual_trade")
    return out


def enrich_blocked_events(blocked: pd.DataFrame, actual_trades: pd.DataFrame) -> pd.DataFrame:
    events = blocked.copy()
    events["join_split"] = events["reported_split"].map(split_for_join)
    trade_cols = [
        "join_split",
        "seed",
        "session",
        "candidate_uid",
        "contract_id",
        "offset",
        "time_bucket",
        "moneyness",
        "premium_paid",
        "premium_bucket",
        "duration_minutes",
        "duration_bucket",
        "entry_spread_dollars",
        "spread_bucket",
        "score_margin",
        "score_margin_bucket",
        "path_mfe",
        "path_mae",
        "giveback_from_mfe",
        "mfe_capture",
        "early_mfe_then_loss",
        "large_uncaptured_mfe",
    ]
    trades = actual_trades[trade_cols].rename(
        columns={
            "candidate_uid": "open_trade_candidate_uid",
            "contract_id": "open_trade_contract_id",
            "offset": "open_trade_offset",
            "time_bucket": "open_time_bucket",
            "moneyness": "open_moneyness",
            "premium_paid": "open_premium_paid",
            "premium_bucket": "open_premium_bucket",
            "duration_minutes": "open_duration_minutes",
            "duration_bucket": "open_duration_bucket",
            "entry_spread_dollars": "open_entry_spread_dollars",
            "spread_bucket": "open_spread_bucket",
            "score_margin": "open_score_margin",
            "score_margin_bucket": "open_score_margin_bucket",
            "path_mfe": "open_path_mfe",
            "path_mae": "open_path_mae",
            "giveback_from_mfe": "open_giveback_from_mfe",
            "mfe_capture": "open_mfe_capture",
            "early_mfe_then_loss": "open_early_mfe_then_loss",
            "large_uncaptured_mfe": "open_large_uncaptured_mfe",
        }
    )
    out = events.merge(
        trades,
        on=["join_split", "seed", "session", "open_trade_candidate_uid", "open_trade_contract_id"],
        how="left",
    )
    out["blocked_relation"] = [
        blocked_relation(open_right, blocked_right, open_contract, blocked_contract)
        for open_right, blocked_right, open_contract, blocked_contract in zip(
            out["open_trade_right"],
            out["blocked_right"],
            out["open_trade_contract_id"],
            out["blocked_contract_id"],
        )
    ]
    out["blocked_pnl_bucket"] = out["blocked_candidate_pnl"].map(blocked_pnl_bucket)
    out["blocked_premium_bucket"] = (pd.to_numeric(out["blocked_entry_ask"], errors="coerce") * 100.0).map(premium_bucket_local)
    out["blocked_spread_bucket"] = (pd.to_numeric(out["blocked_entry_spread"], errors="coerce") * 100.0).map(spread_bucket_local)
    out["blocked_minutes_after_open_bucket"] = out["minutes_after_open_entry"].map(minutes_after_open_bucket)
    out["blocked_minutes_before_exit_bucket"] = out["minutes_before_open_exit"].map(minutes_before_exit_bucket)
    out["open_trade_join_status"] = np.where(out["open_duration_minutes"].notna(), "matched_actual_trade", "missing_actual_trade")
    return out


def blocked_relation(open_right: Any, blocked_right: Any, open_contract: Any, blocked_contract: Any) -> str:
    if str(open_contract) == str(blocked_contract):
        return "same_contract"
    if str(open_right) == str(blocked_right):
        return "same_side_different_contract"
    return "opposite_side"


def slot_cost_bucket(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x <= 0:
        return "le_0"
    if x <= 250:
        return "1_250"
    if x <= 500:
        return "251_500"
    if x <= 1000:
        return "501_1000"
    return "gt_1000"


def open_outcome_bucket(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x < -500:
        return "loss_lt_-500"
    if x < 0:
        return "loss_-500_0"
    if x < 250:
        return "win_0_250"
    if x < 750:
        return "win_250_750"
    if x < 1500:
        return "win_750_1500"
    return "win_gt_1500"


def mfe_bucket(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x < 250:
        return "mfe_lt_250"
    if x < 750:
        return "mfe_250_750"
    if x < 1500:
        return "mfe_750_1500"
    return "mfe_gt_1500"


def giveback_bucket(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x <= 100:
        return "giveback_le_100"
    if x <= 500:
        return "giveback_101_500"
    if x <= 1000:
        return "giveback_501_1000"
    return "giveback_gt_1000"


def blocked_pnl_bucket(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x < -500:
        return "lt_-500"
    if x < 0:
        return "-500_0"
    if x < 250:
        return "0_250"
    if x < 750:
        return "250_750"
    if x < 1500:
        return "750_1500"
    return "gt_1500"


def premium_bucket_local(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x < 1000:
        return "lt_1000"
    if x < 2000:
        return "1000_2000"
    if x < 3000:
        return "2000_3000"
    return "gte_3000"


def spread_bucket_local(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x <= 10:
        return "le_10"
    if x <= 25:
        return "11_25"
    if x <= 50:
        return "26_50"
    return "gt_50"


def minutes_after_open_bucket(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x <= 2:
        return "le_2m_after_open"
    if x <= 5:
        return "3_5m_after_open"
    if x <= 10:
        return "6_10m_after_open"
    return "gt_10m_after_open"


def minutes_before_exit_bucket(value: Any) -> str:
    x = finite(value, default=float("nan"))
    if not math.isfinite(x):
        return "unknown"
    if x <= 2:
        return "le_2m_before_exit"
    if x <= 5:
        return "3_5m_before_exit"
    if x <= 10:
        return "6_10m_before_exit"
    return "gt_10m_before_exit"


def summarize_open_archetypes(open_slots: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "reported_split",
        "open_trade_right",
        "open_time_bucket",
        "open_moneyness",
        "open_premium_bucket",
        "open_duration_bucket",
        "open_trade_exit_reason",
        "open_score_margin_bucket",
    ]
    rows: list[dict[str, Any]] = []
    for key_values, group in open_slots.groupby(keys, dropna=False, observed=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        rows.append(open_summary_row(dict(zip(keys, key_values)), group))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["best_blocked_minus_open_total", "open_trades"], ascending=[False, False], kind="stable")


def open_summary_row(prefix: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    open_pnl = pd.to_numeric(group["open_trade_pnl"], errors="coerce").fillna(0.0)
    best_blocked = pd.to_numeric(group["best_blocked_candidate_pnl"], errors="coerce").fillna(0.0)
    slot_cost = pd.to_numeric(group["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0)
    out = dict(prefix)
    out.update(
        {
            "open_trades": int(len(group)),
            "open_pnl": float(open_pnl.sum()),
            "best_blocked_pnl": float(best_blocked.sum()),
            "best_blocked_minus_open_total": float(slot_cost.sum()),
            "positive_slot_cost_trades": int((slot_cost > 0.0).sum()),
            "positive_slot_cost_rate": float((slot_cost > 0.0).mean()) if len(group) else 0.0,
            "median_slot_cost": float(slot_cost.median()) if len(group) else 0.0,
            "median_open_pnl": float(open_pnl.median()) if len(group) else 0.0,
            "median_best_blocked_pnl": float(best_blocked.median()) if len(group) else 0.0,
            "median_open_mfe": median_numeric(group.get("open_path_mfe", pd.Series(dtype=float))),
            "median_open_giveback": median_numeric(group.get("open_giveback_from_mfe", pd.Series(dtype=float))),
            "early_mfe_loss_rate": mean_bool(group.get("open_early_mfe_then_loss", pd.Series(dtype=bool))),
            "large_uncaptured_mfe_rate": mean_bool(group.get("open_large_uncaptured_mfe", pd.Series(dtype=bool))),
        }
    )
    return out


def summarize_blocked_event_archetypes(blocked: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "reported_split",
        "open_trade_right",
        "blocked_right",
        "blocked_relation",
        "open_trade_exit_reason",
        "open_time_bucket",
        "blocked_time_bucket",
        "blocked_minutes_after_open_bucket",
        "blocked_minutes_before_exit_bucket",
    ]
    rows: list[dict[str, Any]] = []
    for key_values, group in blocked.groupby(keys, dropna=False, observed=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        pnl = pd.to_numeric(group["blocked_candidate_pnl"], errors="coerce").fillna(0.0)
        open_minus = pd.to_numeric(group["open_minus_blocked_pnl"], errors="coerce").fillna(0.0)
        row = dict(zip(keys, key_values))
        row.update(
            {
                "blocked_events": int(len(group)),
                "blocked_candidate_pnl_non_additive": float(pnl.sum()),
                "positive_blocked_rate": float((pnl > 0.0).mean()) if len(group) else 0.0,
                "median_blocked_candidate_pnl": float(pnl.median()) if len(group) else 0.0,
                "median_open_minus_blocked_pnl": float(open_minus.median()) if len(group) else 0.0,
                "open_minus_blocked_total_non_additive": float(open_minus.sum()),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["blocked_candidate_pnl_non_additive", "blocked_events"], ascending=[False, False], kind="stable")


def build_best_blocked_event_rows(open_slots: pd.DataFrame, blocked: pd.DataFrame) -> pd.DataFrame:
    if open_slots.empty or blocked.empty:
        return pd.DataFrame()
    best = open_slots[
        [
            "reported_split",
            "fold",
            "seed",
            "session",
            "open_trade_candidate_uid",
            "best_blocked_candidate_uid",
            "best_blocked_decision_dt",
        ]
    ].copy()
    out = best.merge(
        blocked,
        left_on=[
            "reported_split",
            "fold",
            "seed",
            "session",
            "open_trade_candidate_uid",
            "best_blocked_candidate_uid",
            "best_blocked_decision_dt",
        ],
        right_on=[
            "reported_split",
            "fold",
            "seed",
            "session",
            "open_trade_candidate_uid",
            "blocked_candidate_uid",
            "blocked_decision_dt",
        ],
        how="left",
        suffixes=("", "_blocked"),
    )
    return out


def build_thesis_tests(open_slots: pd.DataFrame, best_events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, mask: pd.Series, mechanism: str, next_action: str) -> None:
        group = open_slots[mask.fillna(False)].copy()
        slot_cost = pd.to_numeric(group.get("best_blocked_minus_open_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        rows.append(
            {
                "thesis": name,
                "mechanism": mechanism,
                "open_trades": int(len(group)),
                "best_blocked_minus_open_total": float(slot_cost.sum()),
                "positive_slot_cost_rate": float((slot_cost > 0.0).mean()) if len(group) else 0.0,
                "median_slot_cost": float(slot_cost.median()) if len(group) else 0.0,
                "next_action": next_action,
            }
        )

    slot_positive = pd.to_numeric(open_slots["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0) > 0.0
    add(
        "losing_open_trade_blocks_positive_later_signal",
        (pd.to_numeric(open_slots["open_trade_pnl"], errors="coerce").fillna(0.0) < 0.0)
        & (pd.to_numeric(open_slots["best_blocked_candidate_pnl"], errors="coerce").fillna(0.0) > 0.0),
        "early entry or lifecycle loss may be spending the slot before a later Protocol101 signal",
        "Audit these rows first for pre-entry avoidability versus early exit/loss-cut evidence.",
    )
    add(
        "small_winner_blocks_larger_later_signal",
        (pd.to_numeric(open_slots["open_trade_pnl"], errors="coerce").fillna(0.0).between(0.0, 500.0))
        & slot_positive,
        "capital-efficient scalp may be correct locally but poor versus the later single-slot opportunity",
        "Test a learned defer / earlier exit rule only in this declared low-open-PnL archetype.",
    )
    add(
        "fallback_exit_slot_cost",
        open_slots["open_trade_exit_reason"].astype(str).eq("protocol054_fallback") & slot_positive,
        "fallback-held positions may block better Protocol101 events",
        "Prioritize lifecycle improvement here; this is more specific than generic hold-longer.",
    )
    add(
        "sequence_residual_slot_cost",
        open_slots["open_trade_exit_reason"].astype(str).eq("sequence_residual_override") & slot_positive,
        "main Protocol101 exit path may still sometimes hold a weaker position through later signals",
        "Segment by side/time/premium before proposing any gate.",
    )
    add(
        "high_mfe_giveback_blocks_later_signal",
        open_slots.get("open_large_uncaptured_mfe", pd.Series(False, index=open_slots.index)).astype(bool) & slot_positive,
        "runner/giveback management interacts with slot opportunity cost",
        "Pair with hold/exit action-advantage labels and require a giveback guard.",
    )
    add(
        "long_duration_slot_cost",
        pd.to_numeric(open_slots.get("open_duration_minutes", pd.Series(dtype=float)), errors="coerce").fillna(0.0).gt(15.0)
        & slot_positive,
        "longer Protocol101 holds are the most plausible internal slot-cost source",
        "Build mutually exclusive replay for only long-duration positive-slot-cost rows.",
    )

    if not best_events.empty and "blocked_relation" in best_events:
        relation_by_open = best_events.set_index(
            ["reported_split", "fold", "seed", "session", "open_trade_candidate_uid"]
        )["blocked_relation"]
        keys = list(
            zip(
                open_slots["reported_split"],
                open_slots["fold"],
                open_slots["seed"],
                open_slots["session"],
                open_slots["open_trade_candidate_uid"],
            )
        )
        relation = pd.Series([relation_by_open.get(key, "unknown") for key in keys], index=open_slots.index)
        add(
            "same_side_later_signal_blocked",
            relation.isin(["same_contract", "same_side_different_contract"]) & slot_positive,
            "Protocol101 may be exiting/re-entering or staying in the wrong same-direction contract",
            "This is the cleanest candidate for a same-side runner/switching-cost lifecycle diagnostic.",
        )
        add(
            "opposite_side_later_signal_blocked",
            relation.eq("opposite_side") & slot_positive,
            "market may have flipped while Protocol101 still occupied the slot",
            "Treat as a regime-reversal/exit question, not a runner-extension question.",
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("best_blocked_minus_open_total", ascending=False, kind="stable")


def build_top_open_trades(open_slots: pd.DataFrame, best_events: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    cols = [
        "reported_split",
        "seed",
        "session",
        "open_trade_entry_dt",
        "open_trade_exit_dt",
        "open_trade_right",
        "open_trade_contract_id",
        "open_trade_exit_reason",
        "open_time_bucket",
        "open_moneyness",
        "open_premium_bucket",
        "open_duration_minutes",
        "open_score_margin",
        "open_trade_pnl",
        "open_path_mfe",
        "open_path_mae",
        "open_giveback_from_mfe",
        "best_blocked_candidate_pnl",
        "best_blocked_minus_open_pnl",
        "best_blocked_candidate_uid",
        "best_blocked_decision_dt",
    ]
    available = [col for col in cols if col in open_slots.columns]
    out = open_slots.sort_values("best_blocked_minus_open_pnl", ascending=False, kind="stable").head(int(limit))[available].copy()
    if best_events.empty:
        return out
    best_cols = [
        "reported_split",
        "fold",
        "seed",
        "session",
        "open_trade_candidate_uid",
        "blocked_candidate_uid",
        "blocked_right",
        "blocked_contract_id",
        "blocked_relation",
        "blocked_entry_ask",
        "blocked_entry_spread",
        "blocked_time_bucket",
        "minutes_after_open_entry",
        "minutes_before_open_exit",
    ]
    best_small = best_events[[col for col in best_cols if col in best_events.columns]].rename(
        columns={"blocked_candidate_uid": "best_blocked_candidate_uid"}
    )
    return out.merge(
        best_small,
        on=["reported_split", "seed", "session", "best_blocked_candidate_uid"],
        how="left",
    )


def build_summary(
    open_slots: pd.DataFrame,
    blocked: pd.DataFrame,
    open_archetypes: pd.DataFrame,
    thesis_tests: pd.DataFrame,
) -> dict[str, Any]:
    positive = open_slots[pd.to_numeric(open_slots["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0) > 0.0]
    top_theses = thesis_tests.head(5).to_dict(orient="records") if not thesis_tests.empty else []
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "Track A Protocol101 internal slot-cost archetype decomposition",
        "decision": "protocol101_slot_cost_archetype_decomposition_complete_model_design_ready_training_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "counts": {
            "open_trades_with_blocked_entries": int(len(open_slots)),
            "positive_slot_cost_open_trades": int(len(positive)),
            "blocked_entry_events": int(len(blocked)),
            "open_archetype_rows": int(len(open_archetypes)),
            "actual_trade_join_failures": int(open_slots["open_trade_join_status"].astype(str).ne("matched_actual_trade").sum()),
        },
        "aggregate": {
            "best_blocked_minus_open_total": finite_sum(open_slots["best_blocked_minus_open_pnl"]),
            "positive_slot_cost_total": finite_sum(positive["best_blocked_minus_open_pnl"]),
            "negative_or_zero_slot_cost_total": finite_sum(
                open_slots[pd.to_numeric(open_slots["best_blocked_minus_open_pnl"], errors="coerce").fillna(0.0) <= 0.0][
                    "best_blocked_minus_open_pnl"
                ]
            ),
            "blocked_candidate_pnl_non_additive": finite_sum(blocked["blocked_candidate_pnl"]),
        },
        "top_thesis_tests": top_theses,
        "blockers": [
            "counterfactual_slot_cost_is_upper_bound_not_mutually_exclusive_replay",
            "fill_latency_and_quote_freshness_not_calibrated",
            "no_causal_rule_or_model_defined_from_archetypes_yet",
            "untouched_holdout_not_scored",
        ],
        "recommended_next_step": "manual_review_top_slot_cost_rows_then_define_one_named_strategy_hypothesis",
        "challenge_allowed": False,
        "training_allowed": False,
    }


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    open_archetypes: pd.DataFrame,
    blocked_archetypes: pd.DataFrame,
    thesis_tests: pd.DataFrame,
) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: Protocol101 internal slot-cost archetype decomposition",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "Protocol101's internal slot-cost weakness is real enough to study, but not yet real enough to train against. The counterfactual-flat audit found many later Protocol101 signals blocked by current open positions; this packet shows where those blocked opportunities concentrate.",
        "",
        "The strongest next research direction is not a broad neural model. It is a named, narrow hypothesis around same-side continuation/switching cost, long-duration/fallback holds, or small-winner/loser entries that block stronger later Protocol101 signals.",
        "",
        "## Headline",
        "",
        f"- Open trades with blocked entries: `{summary['counts']['open_trades_with_blocked_entries']}`",
        f"- Positive slot-cost open trades: `{summary['counts']['positive_slot_cost_open_trades']}`",
        f"- Blocked entry events: `{summary['counts']['blocked_entry_events']}`",
        f"- Best-blocked-minus-open total: `{_fmt_money(summary['aggregate']['best_blocked_minus_open_total'])}`",
        f"- Positive slot-cost total: `{_fmt_money(summary['aggregate']['positive_slot_cost_total'])}`",
        f"- Actual-trade join failures: `{summary['counts']['actual_trade_join_failures']}`",
        "",
        "## Thesis Tests",
        "",
        "| thesis | open trades | slot-cost total | positive rate | median slot cost | next action |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in thesis_tests.iterrows():
        lines.append(
            f"| {row['thesis']} | {int(row['open_trades'])} | {_fmt_money(row['best_blocked_minus_open_total'])} | "
            f"{row['positive_slot_cost_rate']:.3f} | {_fmt_money(row['median_slot_cost'])} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Top Open-Trade Archetypes By Slot Cost",
            "",
            "| split | side | time | moneyness | premium | duration | exit | score margin | open trades | slot-cost total | positive rate | median MFE | median giveback |",
            "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in open_archetypes.head(20).iterrows():
        lines.append(
            f"| {row['reported_split']} | {row['open_trade_right']} | {row['open_time_bucket']} | {row['open_moneyness']} | "
            f"{row['open_premium_bucket']} | {row['open_duration_bucket']} | {row['open_trade_exit_reason']} | "
            f"{row['open_score_margin_bucket']} | {int(row['open_trades'])} | {_fmt_money(row['best_blocked_minus_open_total'])} | "
            f"{row['positive_slot_cost_rate']:.3f} | {_fmt_money(row['median_open_mfe'])} | {_fmt_money(row['median_open_giveback'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Blocked-Event Archetypes",
            "",
            "| split | open side | blocked side | relation | open exit | open time | blocked time | after open | before exit | events | blocked PnL | positive rate |",
            "|---|---|---|---|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in blocked_archetypes.head(20).iterrows():
        lines.append(
            f"| {row['reported_split']} | {row['open_trade_right']} | {row['blocked_right']} | {row['blocked_relation']} | "
            f"{row['open_trade_exit_reason']} | {row['open_time_bucket']} | {row['blocked_time_bucket']} | "
            f"{row['blocked_minutes_after_open_bucket']} | {row['blocked_minutes_before_exit_bucket']} | "
            f"{int(row['blocked_events'])} | {_fmt_money(row['blocked_candidate_pnl_non_additive'])} | "
            f"{row['positive_blocked_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Stopping Rule",
            "",
            "This Track A branch should stop here before model work. The diagnostics have isolated where Protocol101 may be weak, but the next step requires human trading judgment: review the top rows and choose exactly one strategy hypothesis to test. The current evidence is not a training label because it is counterfactual, non-additive, research-exposed, and not fill-calibrated.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Enriched open slots: `{output_dir / 'enriched_open_trade_slot_summary.csv'}`",
            f"- Enriched blocked events: `{output_dir / 'enriched_blocked_slot_events.csv'}`",
            f"- Open archetypes: `{output_dir / 'slot_cost_open_trade_archetypes.csv'}`",
            f"- Blocked event archetypes: `{output_dir / 'slot_cost_blocked_event_archetypes.csv'}`",
            f"- Thesis tests: `{output_dir / 'slot_cost_thesis_tests.csv'}`",
            f"- Top open trades: `{output_dir / 'slot_cost_top_open_trades.csv'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def median_numeric(values: Any) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        numeric = numeric.dropna()
        return float(numeric.median()) if not numeric.empty else 0.0
    return float(numeric) if pd.notna(numeric) else 0.0


def mean_bool(values: Any) -> float:
    if values is None:
        return 0.0
    series = pd.Series(values).fillna(False).astype(bool)
    return float(series.mean()) if len(series) else 0.0


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_sum(values: Any) -> float:
    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    open_slots_raw = pd.read_csv(args.open_slot_summary)
    blocked_raw = pd.read_csv(args.blocked_events)
    actual = load_enriched_actual_trades(Path(args.all_seed_trades))
    open_slots = enrich_open_slots(open_slots_raw, actual)
    blocked = enrich_blocked_events(blocked_raw, actual)
    best_events = build_best_blocked_event_rows(open_slots, blocked)
    open_archetypes = summarize_open_archetypes(open_slots)
    blocked_archetypes = summarize_blocked_event_archetypes(blocked)
    thesis_tests = build_thesis_tests(open_slots, best_events)
    top_open = build_top_open_trades(open_slots, best_events)
    summary = build_summary(open_slots, blocked, open_archetypes, thesis_tests)
    summary["inputs"] = {
        "open_slot_summary": str(args.open_slot_summary),
        "blocked_events": str(args.blocked_events),
        "all_seed_trades": str(args.all_seed_trades),
    }
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "enriched_open_slots": str(output_dir / "enriched_open_trade_slot_summary.csv"),
        "enriched_blocked_events": str(output_dir / "enriched_blocked_slot_events.csv"),
        "open_archetypes": str(output_dir / "slot_cost_open_trade_archetypes.csv"),
        "blocked_event_archetypes": str(output_dir / "slot_cost_blocked_event_archetypes.csv"),
        "thesis_tests": str(output_dir / "slot_cost_thesis_tests.csv"),
        "top_open_trades": str(output_dir / "slot_cost_top_open_trades.csv"),
    }
    open_slots.to_csv(output_dir / "enriched_open_trade_slot_summary.csv", index=False)
    blocked.to_csv(output_dir / "enriched_blocked_slot_events.csv", index=False)
    open_archetypes.to_csv(output_dir / "slot_cost_open_trade_archetypes.csv", index=False)
    blocked_archetypes.to_csv(output_dir / "slot_cost_blocked_event_archetypes.csv", index=False)
    thesis_tests.to_csv(output_dir / "slot_cost_thesis_tests.csv", index=False)
    top_open.to_csv(output_dir / "slot_cost_top_open_trades.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    report = write_report(output_dir, summary, open_archetypes, blocked_archetypes, thesis_tests)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--open-slot-summary", type=Path, default=DEFAULT_OPEN_SLOT_SUMMARY)
    parser.add_argument("--blocked-events", type=Path, default=DEFAULT_BLOCKED_EVENTS)
    parser.add_argument("--all-seed-trades", type=Path, default=DEFAULT_ALL_SEED_TRADES)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
