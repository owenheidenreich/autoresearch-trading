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

from v4.model.environment_diagnostics import time_bucket


ROLE_LABEL = "AUDIT_PROTOCOL101_STRATEGY_FORENSICS_PACKET_V1"
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_strategy_forensics_packet_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_STRATEGY_FORENSICS_PACKET_V1.md"
)
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv")
DEFAULT_ALL_SEED_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/research_all_seed_trades.csv")
DEFAULT_DELAY_ROWS = Path("v4/audit/autoresearch/v4_aplus_hypothesis_114_protocol101_skeptical_falsification/delay_stress_rows.csv")
DEFAULT_BASELINE_ACTIONS = Path(
    "v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet"
)
DEFAULT_CHALLENGER_COMPARISON = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_248_challenger_failure_surface/scoped_seed_trades.csv"
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt_money(value: float) -> str:
    return f"${value:,.0f}"


def _profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses < 0:
        return float(wins / abs(losses))
    return float("inf") if wins > 0 else 0.0


def _median_numeric(values: Any) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        numeric = numeric.dropna()
        return float(numeric.median()) if not numeric.empty else float("nan")
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def moneyness_bucket(right: Any, offset: Any) -> str:
    right_s = str(right).upper()
    off = _finite(offset, default=float("nan"))
    if not math.isfinite(off):
        return "unknown"
    if abs(off) < 1e-9:
        return "ATM"
    if (right_s == "C" and off < 0) or (right_s == "CALL" and off < 0):
        return "ITM"
    if (right_s == "P" and off > 0) or (right_s == "PUT" and off > 0):
        return "ITM"
    return "OTM"


def premium_bucket(premium: Any) -> str:
    value = _finite(premium, default=float("nan"))
    if not math.isfinite(value):
        return "unknown"
    if value < 1_000:
        return "lt_1000"
    if value < 1_500:
        return "1000_1500"
    if value < 2_000:
        return "1500_2000"
    if value < 2_500:
        return "2000_2500"
    if value < 3_000:
        return "2500_3000"
    if value < 3_500:
        return "3000_3500"
    return "gte_3500"


def duration_bucket(minutes: Any) -> str:
    value = _finite(minutes, default=float("nan"))
    if not math.isfinite(value):
        return "unknown"
    if value <= 5:
        return "le_5m"
    if value <= 10:
        return "6_10m"
    if value <= 15:
        return "11_15m"
    if value <= 20:
        return "16_20m"
    if value <= 25:
        return "21_25m"
    return "gt_25m"


def spread_bucket(spread_dollars: Any) -> str:
    value = _finite(spread_dollars, default=float("nan"))
    if not math.isfinite(value):
        return "unknown"
    if value <= 10:
        return "le_10"
    if value <= 25:
        return "11_25"
    if value <= 50:
        return "26_50"
    if value <= 100:
        return "51_100"
    return "gt_100"


def score_margin_bucket(margin: Any) -> str:
    value = _finite(margin, default=float("nan"))
    if not math.isfinite(value):
        return "unknown"
    if value < 0:
        return "below_threshold"
    if value < 0.25:
        return "0_0.25"
    if value < 0.50:
        return "0.25_0.50"
    if value < 1.0:
        return "0.50_1.00"
    return "gte_1.00"


def enrich_protocol101_trades(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["decision_dt"] = pd.to_datetime(out["decision_time"], utc=True)
    out["exit_dt"] = pd.to_datetime(out["exit_time"], utc=True)
    out["duration_minutes"] = (out["exit_dt"] - out["decision_dt"]).dt.total_seconds() / 60.0
    out["day"] = out["decision_dt"].dt.date.astype(str)
    out["time_bucket"] = out["decision_dt"].map(lambda dt: time_bucket(dt.to_pydatetime()))
    out["moneyness"] = [moneyness_bucket(r, o) for r, o in zip(out["right"], out["offset"])]
    out["premium_bucket"] = out["premium_paid"].map(premium_bucket)
    out["duration_bucket"] = out["duration_minutes"].map(duration_bucket)
    out["entry_spread_dollars"] = (pd.to_numeric(out["entry_ask"], errors="coerce") - pd.to_numeric(out["entry_bid"], errors="coerce")) * 100.0
    out["entry_mid"] = (pd.to_numeric(out["entry_ask"], errors="coerce") + pd.to_numeric(out["entry_bid"], errors="coerce")) / 2.0
    out["spread_over_mid"] = np.where(out["entry_mid"] > 0, (out["entry_ask"] - out["entry_bid"]) / out["entry_mid"], np.nan)
    out["spread_bucket"] = out["entry_spread_dollars"].map(spread_bucket)
    out["score_margin"] = pd.to_numeric(out["score"], errors="coerce") - pd.to_numeric(out["threshold"], errors="coerce")
    out["score_margin_bucket"] = out["score_margin"].map(score_margin_bucket)
    out["mfe_capture"] = np.where(pd.to_numeric(out["path_mfe"], errors="coerce") > 0, out["pnl"] / out["path_mfe"], np.nan)
    out["giveback_from_mfe"] = pd.to_numeric(out["path_mfe"], errors="coerce") - pd.to_numeric(out["pnl"], errors="coerce")
    out["early_mfe_then_loss"] = (out["pnl"] < 0) & (out["path_mfe"] > 0)
    out["large_uncaptured_mfe"] = (out["path_mfe"] >= 1_000) & (out["giveback_from_mfe"] >= 500)
    out["trade_archetype"] = (
        out["right"].astype(str)
        + "|"
        + out["time_bucket"].astype(str)
        + "|"
        + out["moneyness"].astype(str)
        + "|"
        + out["premium_bucket"].astype(str)
    )
    return out


def group_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=keys + ["trades", "pnl", "win_rate", "avg_pnl", "profit_factor"])
    for key_values, group in frame.groupby(keys, dropna=False, observed=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "trades": int(len(group)),
                "pnl": float(pnl.sum()),
                "win_rate": float((pnl > 0).mean()) if len(group) else 0.0,
                "avg_pnl": float(pnl.mean()) if len(group) else 0.0,
                "profit_factor": _profit_factor(pnl),
                "median_premium": _median_numeric(group.get("premium_paid", pd.Series(dtype=float))),
                "median_duration_minutes": _median_numeric(group.get("duration_minutes", pd.Series(dtype=float))),
                "median_mfe": _median_numeric(group.get("path_mfe", pd.Series(dtype=float))),
                "median_mae": _median_numeric(group.get("path_mae", pd.Series(dtype=float))),
                "median_mfe_capture": _median_numeric(group.get("mfe_capture", pd.Series(dtype=float))),
                "hard_stop_rate": float(group.get("exit_reason", pd.Series(dtype=object)).eq("hard_stop").mean()),
                "early_mfe_loss_rate": float(group.get("early_mfe_then_loss", pd.Series(dtype=bool)).mean()),
                "large_uncaptured_mfe_rate": float(group.get("large_uncaptured_mfe", pd.Series(dtype=bool)).mean()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["pnl", "trades"], ascending=[False, False], kind="stable")


def build_trade_atlas(trades: pd.DataFrame) -> pd.DataFrame:
    keys = ["segment", "right", "time_bucket", "moneyness", "premium_bucket", "exit_reason"]
    return group_summary(trades, keys)


def build_hard_stop_autopsy(trades: pd.DataFrame) -> pd.DataFrame:
    hard = trades[trades["exit_reason"].eq("hard_stop")].copy()
    if hard.empty:
        return hard
    cols = [
        "segment",
        "session",
        "decision_time",
        "exit_time",
        "right",
        "offset",
        "moneyness",
        "premium_paid",
        "entry_spread_dollars",
        "score",
        "threshold",
        "score_margin",
        "duration_minutes",
        "pnl",
        "path_mfe",
        "path_mae",
        "giveback_from_mfe",
        "mfe_capture",
        "early_mfe_then_loss",
        "quote_gap_seconds",
        "contract_id",
        "candidate_uid",
    ]
    return hard[cols].sort_values("pnl", kind="stable")


def build_losing_day_autopsy(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (segment, day), group in trades.groupby(["segment", "day"], dropna=False):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        total = float(pnl.sum())
        if total >= 0:
            continue
        rows.append(
            {
                "segment": segment,
                "day": day,
                "trades": int(len(group)),
                "pnl": total,
                "win_rate": float((pnl > 0).mean()),
                "hard_stop_trades": int(group["exit_reason"].eq("hard_stop").sum()),
                "hard_stop_pnl": float(group.loc[group["exit_reason"].eq("hard_stop"), "pnl"].sum()),
                "early_mfe_loss_trades": int(group["early_mfe_then_loss"].sum()),
                "early_mfe_loss_pnl": float(group.loc[group["early_mfe_then_loss"], "pnl"].sum()),
                "worst_trade_pnl": float(pnl.min()),
                "median_premium": float(pd.to_numeric(group["premium_paid"], errors="coerce").median()),
                "dominant_side": str(group["right"].mode().iloc[0]) if not group["right"].mode().empty else "",
                "dominant_exit_reason": str(group["exit_reason"].mode().iloc[0]) if not group["exit_reason"].mode().empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values("pnl", kind="stable")


def classify_path(row: pd.Series) -> str:
    pnl = _finite(row.get("pnl"))
    mfe = _finite(row.get("path_mfe"))
    capture = _finite(row.get("mfe_capture"), default=float("nan"))
    giveback = _finite(row.get("giveback_from_mfe"))
    if pnl < 0 and mfe >= 300:
        return "early_mfe_then_loss"
    if pnl < 0:
        return "low_or_no_mfe_loss"
    if mfe >= 1_000 and giveback >= 500:
        return "large_mfe_giveback_winner"
    if math.isfinite(capture) and capture >= 0.80:
        return "clean_capture"
    return "partial_capture"


def build_runner_giveback_audit(trades: pd.DataFrame) -> pd.DataFrame:
    temp = trades.copy()
    temp["path_archetype"] = temp.apply(classify_path, axis=1)
    return group_summary(temp, ["path_archetype", "exit_reason", "right", "time_bucket", "premium_bucket"])


def build_score_calibration(trades: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    temp = trades.dropna(subset=["score_margin"]).copy()
    if temp.empty:
        return pd.DataFrame()
    try:
        temp["score_margin_decile"] = pd.qcut(temp["score_margin"], q=bins, duplicates="drop")
    except ValueError:
        temp["score_margin_decile"] = "all"
    corr_pnl = float(temp["score_margin"].corr(temp["pnl"], method="spearman"))
    corr_mfe = float(temp["score_margin"].corr(temp["path_mfe"], method="spearman"))
    rows: list[dict[str, Any]] = []
    for decile, group in temp.groupby("score_margin_decile", dropna=False, observed=False):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "score_margin_decile": str(decile),
                "trades": int(len(group)),
                "pnl": float(pnl.sum()),
                "win_rate": float((pnl > 0).mean()) if len(group) else 0.0,
                "avg_pnl": float(pnl.mean()) if len(group) else 0.0,
                "profit_factor": _profit_factor(pnl),
                "median_premium": _median_numeric(group["premium_paid"]),
                "median_duration_minutes": _median_numeric(group["duration_minutes"]),
                "median_mfe": _median_numeric(group["path_mfe"]),
                "median_mae": _median_numeric(group["path_mae"]),
                "median_mfe_capture": _median_numeric(group["mfe_capture"]),
                "hard_stop_rate": float(group["exit_reason"].eq("hard_stop").mean()),
                "early_mfe_loss_rate": float(group["early_mfe_then_loss"].mean()),
                "large_uncaptured_mfe_rate": float(group["large_uncaptured_mfe"].mean()),
                "min_score_margin": float(group["score_margin"].min()),
                "max_score_margin": float(group["score_margin"].max()),
                "spearman_score_margin_vs_pnl": corr_pnl,
                "spearman_score_margin_vs_mfe": corr_mfe,
            }
        )
    return pd.DataFrame(rows).sort_values("min_score_margin", kind="stable")


def build_delay_fragility_by_archetype(all_seed_trades: pd.DataFrame, delay_rows: pd.DataFrame) -> pd.DataFrame:
    if all_seed_trades.empty or delay_rows.empty:
        return pd.DataFrame()
    trades = enrich_protocol101_trades(all_seed_trades)
    join_cols = ["candidate_uid", "seed", "session"]
    merged = delay_rows.merge(
        trades[
            join_cols
            + [
                "segment",
                "right",
                "time_bucket",
                "moneyness",
                "premium_bucket",
                "exit_reason",
                "score_margin_bucket",
            ]
        ],
        on=join_cols,
        how="left",
        suffixes=("", "_trade"),
    )
    rows: list[dict[str, Any]] = []
    keys = ["reported_split", "right", "time_bucket", "moneyness", "premium_bucket", "exit_reason"]
    for key_values, group in merged.groupby(keys, dropna=False, observed=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        original = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        entry = pd.to_numeric(group["entry_delay_pnl"], errors="coerce").fillna(0.0)
        both = pd.to_numeric(group["both_delay_pnl"], errors="coerce").fillna(0.0)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "rows": int(len(group)),
                "original_pnl": float(original.sum()),
                "entry_delay_pnl": float(entry.sum()),
                "entry_delay_delta": float(entry.sum() - original.sum()),
                "both_delay_pnl": float(both.sum()),
                "both_delay_delta": float(both.sum() - original.sum()),
                "entry_delay_loss_rate": float((entry < original).mean()) if len(group) else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_delay_delta", kind="stable")


def build_policy_quality_comparison(challenger_comparison: pd.DataFrame) -> pd.DataFrame:
    if challenger_comparison.empty or "policy" not in challenger_comparison.columns:
        return pd.DataFrame()
    temp = challenger_comparison.copy()
    if "premium_paid" not in temp.columns and "entry_premium" in temp.columns:
        temp["premium_paid"] = temp["entry_premium"]
    if "duration_minutes" not in temp.columns:
        temp["decision_dt"] = pd.to_datetime(temp["decision_time"], utc=True)
        temp["exit_dt"] = pd.to_datetime(temp["exit_time"], utc=True)
        temp["duration_minutes"] = (temp["exit_dt"] - temp["decision_dt"]).dt.total_seconds() / 60.0
    if "moneyness" not in temp.columns:
        temp["moneyness"] = [moneyness_bucket(r, o) for r, o in zip(temp["right"], temp["offset"])]
    if "premium_bucket" not in temp.columns:
        temp["premium_bucket"] = temp["premium_paid"].map(premium_bucket)
    if "mfe_capture" not in temp.columns:
        temp["mfe_capture"] = np.where(temp["path_mfe"] > 0, temp["pnl"] / temp["path_mfe"], np.nan)
    temp["early_mfe_then_loss"] = (temp["pnl"] < 0) & (temp["path_mfe"] > 0)
    temp["large_uncaptured_mfe"] = (temp["path_mfe"] >= 1_000) & ((temp["path_mfe"] - temp["pnl"]) >= 500)
    return group_summary(temp, ["policy", "reported_split", "right", "moneyness", "premium_bucket"])


def build_slot_opportunity_readiness(event_actions: pd.DataFrame) -> dict[str, Any]:
    if event_actions.empty:
        return {
            "status": "blocked_missing_protocol101_baseline_event_actions",
            "reason": "No baseline action attachment was available.",
        }
    action_col = "protocol101_action"
    counts = event_actions[action_col].value_counts(dropna=False).to_dict()
    holding = event_actions[event_actions[action_col].eq("holding")]
    enter_while_holding = int(holding[action_col].eq("enter").sum())
    return {
        "status": "blocked_missing_counterfactual_flat_protocol101_actions",
        "reason": (
            "The available baseline attachment records deployed Protocol101 state. During open intervals rows are marked holding, "
            "so it cannot reveal which Protocol101 entries would have fired if the account were flat."
        ),
        "rows": int(len(event_actions)),
        "action_counts": {str(k): int(v) for k, v in counts.items()},
        "holding_rows": int(len(holding)),
        "observable_enter_rows_while_holding": enter_while_holding,
        "next_experiment": "Build a counterfactual-flat Protocol101 event replay that scores every event as if no position were open, then compare later feasible entries inside actual open intervals.",
    }


def build_experiment_backlog(slot_status: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "experiment": "PROTOCOL101_EXECUTION_REALISM_BY_ARCHETYPE_V1",
            "question": "Does Protocol101 edge survive latency, quote freshness, spread, and fills by trade archetype?",
            "implementation": "Join no-order/paper fill observations to trade atlas and replay latency distributions instead of only fixed delays.",
            "current_status": "blocked_until_live_no_order_and_fill_observations_exist",
            "promotion_effect": "blocks all challenger and paper-default expansion claims",
        },
        {
            "priority": 2,
            "experiment": "PROTOCOL101_HARD_STOP_AUTOPSY_V1",
            "question": "Are hard-stop losses pre-entry identifiable or only path-identifiable after early MFE?",
            "implementation": "Use hard_stop_autopsy.csv plus quote/path features to classify late/exhausted/quote-failure/early-MFE loss modes.",
            "current_status": "ready_from_current_trade_log_proxy",
            "promotion_effect": "may justify a conservative rejection gate or lifecycle patch",
        },
        {
            "priority": 3,
            "experiment": "PROTOCOL101_CONFIRMED_MFE_RUNNER_AUDIT_V1",
            "question": "Which Protocol101 exits should become runner states and which should not be extended?",
            "implementation": "Replay post-exit bid paths to forced flat; current packet only computes in-trade MFE/giveback proxy.",
            "current_status": "blocked_until_post_exit_path_rows_are_attached",
            "promotion_effect": "may justify a narrow runner overlay instead of new entry model",
        },
        {
            "priority": 4,
            "experiment": "PROTOCOL101_INTERNAL_SLOT_COST_V1",
            "question": "Do Protocol101 entries block better later Protocol101 opportunities?",
            "implementation": slot_status.get("next_experiment", "Build counterfactual-flat Protocol101 event replay."),
            "current_status": slot_status.get("status", "unknown"),
            "promotion_effect": "may justify a Protocol101 defer overlay",
        },
        {
            "priority": 5,
            "experiment": "PROTOCOL101_MISSED_WINNER_ABSTENTION_AUDIT_V1",
            "question": "Is Protocol101 too narrow, or correctly rejecting hindsight bait?",
            "implementation": "Use full-surface candidates and Protocol101-like similarity rules; require causal separability from rejected losers.",
            "current_status": "partially_ready_from_challenger_comparison_proxy",
            "promotion_effect": "decides whether to widen Protocol101 or define a separate playbook",
        },
        {
            "priority": 6,
            "experiment": "PROTOCOL101_SCORE_RELIABILITY_V1",
            "question": "Does Protocol101 score margin calibrate PnL, MFE, and risk?",
            "implementation": "Use score_calibration.csv by side/time/regime, then ablate short-history features against strict baseline.",
            "current_status": "ready_from_current_trade_log_proxy",
            "promotion_effect": "decides whether score can support defer/risk logic",
        },
    ]
    return pd.DataFrame(rows)


def build_response_claim_analysis(summary: dict[str, Any], slot_status: dict[str, Any]) -> pd.DataFrame:
    headline = summary["headline"]
    rows = [
        {
            "claim": "Protocol101 is a trading hypothesis, not the final model.",
            "verdict": "supported",
            "answer_from_current_evidence": "Protocol101 is deployable because it encodes a conservative A+/surface-edge, high-premium, mostly ITM, wait-first belief set; exits are inherited rather than owned by Protocol101.",
            "current_evidence": "Protocol101 strict serial gate; Protocol113 trade log; live entry bridge.",
            "next_test_or_experiment": "Keep Protocol101 as paper default while running strategy-forensics diagnostics before new model training.",
        },
        {
            "claim": "Execution realism is the first gate.",
            "verdict": "supported_blocker",
            "answer_from_current_evidence": "Historical delay stress is severe and fill calibration remains absent; this blocks promotion-grade claims.",
            "current_evidence": "Protocol114/126 delay stress; Protocol117 high-res validation; Protocol272 found zero fill observations.",
            "next_test_or_experiment": "PROTOCOL101_EXECUTION_REALISM_BY_ARCHETYPE_V1 plus stratified paper/no-order fill collection.",
        },
        {
            "claim": "Protocol101 strategy identity is blurry: scalp, runner, or hybrid.",
            "verdict": "partially_answered",
            "answer_from_current_evidence": f"Current seed-1 replay looks like high-premium ITM quick capture: median duration {headline['median_duration_minutes']:.1f} minutes, median premium {_fmt_money(headline['median_premium'])}, and sequence_residual_override dominates PnL.",
            "current_evidence": "trade_archetype_cube.csv; runner_giveback_proxy.csv.",
            "next_test_or_experiment": "Attach post-exit paths and classify scalp, runner, failed runner, giveback winner, and giveback loser states.",
        },
        {
            "claim": "Hard-stop losses are a high-EV diagnostic target.",
            "verdict": "supported",
            "answer_from_current_evidence": f"Hard stops are only {headline['hard_stop_trades']} seed-1 rows but lost {_fmt_money(headline['hard_stop_pnl'])}; most show some positive MFE before loss.",
            "current_evidence": "hard_stop_autopsy.csv; losing_day_autopsy.csv.",
            "next_test_or_experiment": "PROTOCOL101_HARD_STOP_AUTOPSY_V1 to separate pre-entry rejection signatures from path-management failures.",
        },
        {
            "claim": "Protocol101 exits may be early, late, or accidentally right.",
            "verdict": "proxy_only",
            "answer_from_current_evidence": "In-trade MFE/giveback proxy is available, but post-exit opportunity cannot be measured from the selected trade log alone.",
            "current_evidence": "runner_giveback_proxy.csv.",
            "next_test_or_experiment": "PROTOCOL101_CONFIRMED_MFE_RUNNER_AUDIT_V1 using bid paths from actual exit to forced flat.",
        },
        {
            "claim": "Protocol101 may be too narrow.",
            "verdict": "plausible_not_proven",
            "answer_from_current_evidence": "Broader challengers found more exposed-split PnL, but the clean Protocol248 comparison showed lower win rate, lower PF, worse drawdown, and worse MFE capture.",
            "current_evidence": "protocol101_narrowness_proxy.csv; Protocol248 failure surface.",
            "next_test_or_experiment": "Matched rejected-candidate audit using full_surface_action_advantage.parquet joined to Protocol101 baseline actions, with no future/path columns used for similarity.",
        },
        {
            "claim": "Calls and puts may be separate strategies.",
            "verdict": "supported_for_audit",
            "answer_from_current_evidence": "Calls are more frequent, puts have higher average PnL in seed-1 replay, and hard-stop losses are mostly puts.",
            "current_evidence": "trade_archetype_cube.csv; hard_stop_autopsy.csv.",
            "next_test_or_experiment": "Side-specific strategy audit by side/time/premium/moneyness/exit/path archetype.",
        },
        {
            "claim": "Protocol101 may block better later Protocol101 opportunities.",
            "verdict": "unanswered_blocked",
            "answer_from_current_evidence": slot_status["reason"],
            "current_evidence": "protocol101_baseline_event_actions_training_scope.parquet records deployed holding state, not counterfactual-flat actions.",
            "next_test_or_experiment": slot_status["next_experiment"],
        },
        {
            "claim": "Protocol101 score may not calibrate realized value.",
            "verdict": "concerning_proxy",
            "answer_from_current_evidence": f"Selected-trade score margin has weak negative Spearman correlation with PnL ({summary['score_calibration']['spearman_score_margin_vs_pnl']:.4f}) and MFE ({summary['score_calibration']['spearman_score_margin_vs_mfe']:.4f}). This is not a full rejected-candidate calibration.",
            "current_evidence": "score_calibration.csv.",
            "next_test_or_experiment": "PROTOCOL101_SCORE_RELIABILITY_V1 with rejected candidates and same-event rankings.",
        },
        {
            "claim": "Current diagnostic splits are not final promotion evidence.",
            "verdict": "supported_blocker",
            "answer_from_current_evidence": "Protocol273 and learned-defer validation both say Q3/Q4/Q1/March/recent are research-exposed and formal strategy-matrix/PBO controls are missing.",
            "current_evidence": "Protocol273 split exposure; learned-defer validation packet.",
            "next_test_or_experiment": "Build strategy matrix, CSCV/PBO diagnostics, and freeze a packet before untouched holdout scoring.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_headline(trades: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    return {
        "trades": int(len(trades)),
        "pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()) if len(trades) else 0.0,
        "profit_factor": _profit_factor(pnl),
        "median_premium": float(pd.to_numeric(trades["premium_paid"], errors="coerce").median()),
        "median_duration_minutes": float(pd.to_numeric(trades["duration_minutes"], errors="coerce").median()),
        "hard_stop_trades": int(trades["exit_reason"].eq("hard_stop").sum()),
        "hard_stop_pnl": float(trades.loc[trades["exit_reason"].eq("hard_stop"), "pnl"].sum()),
        "early_mfe_loss_trades": int(trades["early_mfe_then_loss"].sum()),
    }


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    trade_atlas: pd.DataFrame,
    hard_stop: pd.DataFrame,
    losing_days: pd.DataFrame,
    score_calibration: pd.DataFrame,
    slot_status: dict[str, Any],
    response_claim_analysis: pd.DataFrame,
) -> str:
    headline = summary["headline"]
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: research-only Protocol101 strategy-forensics packet",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Headline",
        "",
        f"- Trades: `{headline['trades']}`",
        f"- PnL: `{_fmt_money(headline['pnl'])}`",
        f"- Win rate: `{headline['win_rate']:.3f}`",
        f"- Profit factor: `{headline['profit_factor']:.3f}`",
        f"- Median premium: `{_fmt_money(headline['median_premium'])}`",
        f"- Median duration: `{headline['median_duration_minutes']:.1f}` minutes",
        f"- Hard-stop trades: `{headline['hard_stop_trades']}` for `{_fmt_money(headline['hard_stop_pnl'])}`",
        f"- Early-MFE-then-loss trades: `{headline['early_mfe_loss_trades']}`",
        "",
        "## Response Analysis",
        "",
        "The pasted response is directionally right: Protocol101 should be treated as an interrogated trading hypothesis, not a final model. This packet turns the memo into executable diagnostics where current artifacts are sufficient and explicit blockers where they are not.",
        "",
        "## Top Trade Archetypes",
        "",
        "| segment | right | time | moneyness | premium | exit | trades | pnl | win rate | PF |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in trade_atlas.head(12).iterrows():
        lines.append(
            f"| {row['segment']} | {row['right']} | {row['time_bucket']} | {row['moneyness']} | {row['premium_bucket']} | "
            f"{row['exit_reason']} | {int(row['trades'])} | {_fmt_money(row['pnl'])} | {row['win_rate']:.3f} | {row['profit_factor']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Hard-Stop Finding",
            "",
            f"Hard-stop rows are small in the seed-1 paper replay but severe: `{len(hard_stop)}` rows for `{_fmt_money(float(hard_stop['pnl'].sum()) if not hard_stop.empty else 0.0)}`. They are high-priority because they are interpretable and may point to either pre-entry rejection or path-based lifecycle management.",
            "",
            "Worst hard-stop rows:",
            "",
            "| session | decision_time | side | premium | pnl | MFE | MAE | early MFE loss |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in hard_stop.head(8).iterrows():
        lines.append(
            f"| {row['session']} | {row['decision_time']} | {row['right']} | {_fmt_money(row['premium_paid'])} | "
            f"{_fmt_money(row['pnl'])} | {_fmt_money(row['path_mfe'])} | {_fmt_money(row['path_mae'])} | {bool(row['early_mfe_then_loss'])} |"
        )
    lines.extend(
        [
            "",
            "## Losing-Day Finding",
            "",
            "| day | segment | trades | pnl | win rate | hard stops | early-MFE losses |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in losing_days.head(10).iterrows():
        lines.append(
            f"| {row['day']} | {row['segment']} | {int(row['trades'])} | {_fmt_money(row['pnl'])} | "
            f"{row['win_rate']:.3f} | {int(row['hard_stop_trades'])} | {int(row['early_mfe_loss_trades'])} |"
        )
    lines.extend(
        [
            "",
            "## Score Calibration Proxy",
            "",
            f"Spearman score-margin vs PnL: `{summary['score_calibration']['spearman_score_margin_vs_pnl']:.4f}`",
            f"Spearman score-margin vs MFE: `{summary['score_calibration']['spearman_score_margin_vs_mfe']:.4f}`",
            "",
            "This is only a selected-trade calibration proxy. A full calibration test needs rejected candidates and same-event candidate rankings.",
            "",
            "## Claim-By-Claim Verdict",
            "",
            "| claim | verdict | next test |",
            "|---|---|---|",
        ]
    )
    for _, row in response_claim_analysis.iterrows():
        lines.append(f"| {row['claim']} | `{row['verdict']}` | {row['next_test_or_experiment']} |")
    lines.extend(
        [
            "",
            "## Internal Slot-Cost Status",
            "",
            f"Status: `{slot_status['status']}`",
            "",
            slot_status["reason"],
            "",
            "## Generated Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Trade atlas: `{output_dir / 'trade_archetype_cube.csv'}`",
            f"- Hard-stop autopsy: `{output_dir / 'hard_stop_autopsy.csv'}`",
            f"- Losing-day autopsy: `{output_dir / 'losing_day_autopsy.csv'}`",
            f"- Runner/giveback proxy: `{output_dir / 'runner_giveback_proxy.csv'}`",
            f"- Score calibration: `{output_dir / 'score_calibration.csv'}`",
            f"- Timing fragility by archetype: `{output_dir / 'execution_fragility_by_archetype.csv'}`",
            f"- Narrowness proxy: `{output_dir / 'protocol101_narrowness_proxy.csv'}`",
            f"- Experiment backlog: `{output_dir / 'experiment_backlog.csv'}`",
            f"- Response claim analysis: `{output_dir / 'response_claim_analysis.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = enrich_protocol101_trades(pd.read_csv(args.trades))
    all_seed_trades = pd.read_csv(args.all_seed_trades)
    delay_rows = pd.read_csv(args.delay_rows)
    event_actions = pd.read_parquet(args.baseline_actions) if Path(args.baseline_actions).exists() else pd.DataFrame()
    challenger_comparison = pd.read_csv(args.challenger_comparison, low_memory=False) if Path(args.challenger_comparison).exists() else pd.DataFrame()

    trade_atlas = build_trade_atlas(trades)
    hard_stop = build_hard_stop_autopsy(trades)
    losing_days = build_losing_day_autopsy(trades)
    runner_giveback = build_runner_giveback_audit(trades)
    score_calibration = build_score_calibration(trades)
    delay_fragility = build_delay_fragility_by_archetype(all_seed_trades, delay_rows)
    narrowness_proxy = build_policy_quality_comparison(challenger_comparison)
    slot_status = build_slot_opportunity_readiness(event_actions)
    experiment_backlog = build_experiment_backlog(slot_status)

    score_corr_pnl = float(score_calibration["spearman_score_margin_vs_pnl"].iloc[0]) if not score_calibration.empty else float("nan")
    score_corr_mfe = float(score_calibration["spearman_score_margin_vs_mfe"].iloc[0]) if not score_calibration.empty else float("nan")
    summary = {
        "role_label": ROLE_LABEL,
        "what_is_this": "research-only Protocol101 strategy-forensics packet",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "untouched_holdout_scored": False,
        "decision": "protocol101_strategy_forensics_packet_complete_model_training_still_blocked",
        "source_paths": {
            "trades": str(args.trades),
            "all_seed_trades": str(args.all_seed_trades),
            "delay_rows": str(args.delay_rows),
            "baseline_actions": str(args.baseline_actions),
            "challenger_comparison": str(args.challenger_comparison),
        },
        "headline": summarize_headline(trades),
        "diagnostic_status": {
            "trade_atlas": "complete_from_protocol113_trade_log",
            "hard_stop_autopsy": "ready_for_manual_review",
            "runner_giveback": "proxy_only_post_exit_path_missing",
            "score_calibration": "selected_trade_proxy_only_rejected_candidates_missing",
            "execution_fragility": "complete_from_protocol114_delay_rows",
            "narrowness": "proxy_from_protocol248_challenger_comparison",
            "internal_slot_cost": slot_status["status"],
            "claim_analysis": "complete_from_packet_and_prior_audits",
        },
        "score_calibration": {
            "spearman_score_margin_vs_pnl": score_corr_pnl,
            "spearman_score_margin_vs_mfe": score_corr_mfe,
        },
        "slot_opportunity_status": slot_status,
    }
    response_claim_analysis = build_response_claim_analysis(summary, slot_status)

    trade_atlas.to_csv(output_dir / "trade_archetype_cube.csv", index=False)
    hard_stop.to_csv(output_dir / "hard_stop_autopsy.csv", index=False)
    losing_days.to_csv(output_dir / "losing_day_autopsy.csv", index=False)
    runner_giveback.to_csv(output_dir / "runner_giveback_proxy.csv", index=False)
    score_calibration.to_csv(output_dir / "score_calibration.csv", index=False)
    delay_fragility.to_csv(output_dir / "execution_fragility_by_archetype.csv", index=False)
    narrowness_proxy.to_csv(output_dir / "protocol101_narrowness_proxy.csv", index=False)
    experiment_backlog.to_csv(output_dir / "experiment_backlog.csv", index=False)
    response_claim_analysis.to_csv(output_dir / "response_claim_analysis.csv", index=False)

    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "trade_archetype_cube": str(output_dir / "trade_archetype_cube.csv"),
        "hard_stop_autopsy": str(output_dir / "hard_stop_autopsy.csv"),
        "losing_day_autopsy": str(output_dir / "losing_day_autopsy.csv"),
        "runner_giveback_proxy": str(output_dir / "runner_giveback_proxy.csv"),
        "score_calibration": str(output_dir / "score_calibration.csv"),
        "execution_fragility_by_archetype": str(output_dir / "execution_fragility_by_archetype.csv"),
        "protocol101_narrowness_proxy": str(output_dir / "protocol101_narrowness_proxy.csv"),
        "experiment_backlog": str(output_dir / "experiment_backlog.csv"),
        "response_claim_analysis": str(output_dir / "response_claim_analysis.csv"),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = write_report(
        output_dir,
        summary,
        trade_atlas,
        hard_stop,
        losing_days,
        score_calibration,
        slot_status,
        response_claim_analysis,
    )
    (output_dir / "report.md").write_text(report)
    Path(args.doc).write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--all-seed-trades", type=Path, default=DEFAULT_ALL_SEED_TRADES)
    parser.add_argument("--delay-rows", type=Path, default=DEFAULT_DELAY_ROWS)
    parser.add_argument("--baseline-actions", type=Path, default=DEFAULT_BASELINE_ACTIONS)
    parser.add_argument("--challenger-comparison", type=Path, default=DEFAULT_CHALLENGER_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
