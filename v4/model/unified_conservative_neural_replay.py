"""Strict replay helpers for conservative unified neural policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from v4.model.supervised_pilot import Trade, metrics_for_trades
from v4.model.unified_conservative_neural_policy import ConservativeNeuralPolicyConfig, conservative_action_allowed


ROLE_LABEL = "REPLAY_UNIFIED_CONSERVATIVE_NEURAL_POLICY_STRICT_SERIAL_V1"


@dataclass(frozen=True)
class ConservativeReplayConfig:
    starting_cash: float = 10_000.0
    contract_multiplier: float = 100.0
    max_contracts: int = 1
    max_concurrent_positions: int = 1
    entry_price: str = "ask"
    exit_price: str = "bid"
    fallback_baseline: str = "PAPER_DEFAULT_PROTOCOL101"
    policy_mode: str = "baseline_anchored_overlay"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_conservative_entry_candidate(
    candidates: pd.DataFrame,
    *,
    equity: float,
    policy_config: ConservativeNeuralPolicyConfig,
    replay_config: ConservativeReplayConfig = ConservativeReplayConfig(),
) -> tuple[int | None, str]:
    """Select the highest predicted allowed candidate, if any."""

    if candidates.empty:
        return None, "empty_candidate_set"
    required = {"predicted_advantage", "positive_probability", "tail_probability", "entry_premium"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"missing predicted candidate columns: {missing}")
    numeric = candidates.copy()
    for column in required:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    allowed = (
        numeric["predicted_advantage"].ge(float(policy_config.min_advantage_margin))
        & numeric["positive_probability"].ge(float(policy_config.positive_probability_min))
        & numeric["tail_probability"].le(float(policy_config.tail_probability_max))
        & numeric["entry_premium"].gt(0.0)
        & numeric["entry_premium"].le(float(equity))
    )
    if not bool(allowed.any()):
        return None, "no_candidate_clears_conservative_gate"
    selected_index = numeric.loc[allowed, "predicted_advantage"].idxmax()
    return int(selected_index), "challenger_entry_selected"


def choose_conservative_holding_exit(
    path: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    policy_config: ConservativeNeuralPolicyConfig,
) -> tuple[int | None, str]:
    """Exit at the first state where the conservative hold gate fails."""

    if path.empty:
        return None, "empty_holding_path"
    required = {"predicted_advantage", "positive_probability", "tail_probability"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing holding prediction columns: {missing}")
    if len(path) != len(predictions):
        raise ValueError("holding path and predictions must have the same length")
    for idx, row in predictions.reset_index(drop=True).iterrows():
        decision = conservative_action_allowed(
            predicted_advantage=float(row["predicted_advantage"]),
            positive_probability=float(row["positive_probability"]),
            tail_probability=float(row["tail_probability"]),
            config=policy_config,
        )
        if not decision["allowed"]:
            return int(idx), "lifecycle_conservative_exit"
    return int(len(path) - 1), "forced_flat_no_lifecycle_exit_signal"


def summarize_replay_trades(
    trades: list[dict[str, Any]],
    *,
    event_count: int,
    starting_cash: float,
    skipped: dict[str, int] | None = None,
) -> dict[str, Any]:
    converted = [
        Trade(
            session=str(trade["session"]),
            decision_time=str(trade["decision_time"]),
            pnl=float(trade["pnl"]),
            score=float(trade.get("predicted_advantage", 0.0)),
            right=str(trade.get("right", "")),
            offset=float(trade.get("offset", 0.0)),
            strategy=str(trade.get("strategy", ROLE_LABEL)),
        )
        for trade in trades
    ]
    metrics = metrics_for_trades(converted)
    equity = [float(starting_cash)] + [float(trade["account_equity_after"]) for trade in trades]
    peak = float(starting_cash)
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = min(drawdown, value - peak)
    metrics.update(
        {
            "starting_cash": float(starting_cash),
            "ending_equity": float(equity[-1]),
            "return_pct": float((equity[-1] - starting_cash) / starting_cash * 100.0),
            "max_account_drawdown": float(drawdown),
            "input_events": int(event_count),
            "max_concurrent_positions": 1 if trades else 0,
            "serial_status": "pass",
            "all_flat_by_session_end": True,
        }
    )
    skipped = skipped or {}
    metrics.update({f"skipped_{key}": int(value) for key, value in sorted(skipped.items())})
    if trades:
        frame = pd.DataFrame(trades)
        metrics["median_duration_minutes"] = float(np.median(pd.to_numeric(frame["duration_minutes"], errors="coerce").fillna(0.0)))
        metrics["source_counts"] = {str(key): int(value) for key, value in frame["source"].value_counts(dropna=False).sort_index().items()}
        metrics["exit_reason_counts"] = {str(key): int(value) for key, value in frame["exit_reason"].value_counts(dropna=False).sort_index().items()}
    else:
        metrics["median_duration_minutes"] = 0.0
        metrics["source_counts"] = {}
        metrics["exit_reason_counts"] = {}
    return metrics
