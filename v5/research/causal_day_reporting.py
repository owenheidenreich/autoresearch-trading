"""Economic/risk summaries for completed causal-day simulation results.

This module performs no model fitting and makes no strategy decision.  It is
the single metric contract used by future shallow/neural comparisons and by
known-answer simulator fixtures.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.causal_day_simulator import SimulationResult


class ReportingError(RuntimeError):
    """Simulation results cannot support a complete economic summary."""


BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 0


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _maximum_drawdown(session_pnl: pd.Series, starting_equity: float) -> float:
    equity = starting_equity + session_pnl.cumsum()
    peak = pd.concat(
        [pd.Series([starting_equity]), equity], ignore_index=True
    ).cummax().iloc[1:].to_numpy(float)
    return float(np.max(peak - equity.to_numpy(float))) if len(equity) else 0.0


def _corrected_session_interval(
    values: np.ndarray, *, family_size: int, reps: int = BOOTSTRAP_REPS
) -> dict[str, Any]:
    if family_size < 1:
        raise ReportingError("family_size must be at least one")
    if reps < 1:
        raise ReportingError("bootstrap reps must be positive")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[draws].mean(axis=1)
    alpha = 0.05 / family_size
    return {
        "method": "session bootstrap with Bonferroni family correction",
        "level": 0.95,
        "family_size": family_size,
        "session_unit": True,
        "reps": reps,
        "seed": BOOTSTRAP_SEED,
        "low": float(np.quantile(means, alpha / 2.0)),
        "high": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def summarize_results(
    results: Sequence[SimulationResult], *, family_size: int = 1
) -> dict[str, Any]:
    """Return the fixed $10,000 trader metric contract for one family cell."""

    if not results:
        raise ReportingError("at least one simulation result is required")
    sessions = [result.session for result in results]
    if len(set(sessions)) != len(sessions):
        raise ReportingError("one family cell may contain only one result per session")
    if any(result.blocked_terminal_position for result in results):
        raise ReportingError("blocked terminal positions prevent complete economic scoring")
    risk_modes = {result.risk_mode for result in results}
    trade_caps = {result.trade_cap for result in results}
    starting = {result.starting_equity_usd for result in results}
    if len(risk_modes) != 1 or len(trade_caps) != 1 or len(starting) != 1:
        raise ReportingError("risk mode, trade cap and starting equity must be homogeneous")

    ordered = sorted(results, key=lambda result: result.session)
    trade_blocks = [result.trades.assign(_session=result.session) for result in ordered]
    trades = pd.concat(trade_blocks, ignore_index=True) if trade_blocks else pd.DataFrame()
    pnl = pd.Series(
        [result.realised_pnl_usd for result in ordered],
        index=[result.session for result in ordered],
        dtype=float,
    )
    if not trades.empty and not np.isclose(
        float(trades["net_pnl_usd"].sum()), float(pnl.sum()), atol=1e-8, rtol=0.0
    ):
        raise ReportingError("trade ledger P&L does not equal session realised P&L")

    event_blocks = [result.events.assign(_session=result.session) for result in ordered]
    events = pd.concat(event_blocks, ignore_index=True)
    flat_statuses = {
        "flat_no_action",
        "filled_ask",
        "rejected_sell_while_flat",
        "rejected_outside_entry_clock",
        "rejected_trade_cap",
        "rejected_daily_breaker",
        "rejected_missing_contract",
        "rejected_ineligible_contract",
        "rejected_buying_power",
    }
    flat = events[events["status"].isin(flat_statuses)]
    abstention_rate = (
        float(flat["status"].eq("flat_no_action").mean()) if len(flat) else float("nan")
    )

    positive = trades[trades["net_pnl_usd"].gt(0.0)] if not trades.empty else trades
    negative = trades[trades["net_pnl_usd"].lt(0.0)] if not trades.empty else trades
    average_winner = float(positive["net_pnl_usd"].mean()) if len(positive) else float("nan")
    average_loser = float(-negative["net_pnl_usd"].mean()) if len(negative) else float("nan")
    payoff = (
        average_winner / average_loser
        if np.isfinite(average_winner) and np.isfinite(average_loser) and average_loser > 0.0
        else float("nan")
    )
    start = float(next(iter(starting)))

    report: dict[str, Any] = {
        "sessions": len(ordered),
        "trades": int(len(trades)),
        "days_traded": int(trades["_session"].nunique()) if len(trades) else 0,
        "trades_per_day": float(len(trades) / len(ordered)),
        "abstention_rate": _number(abstention_rate),
        "p_win": float(trades["net_pnl_usd"].gt(0.0).mean()) if len(trades) else None,
        "average_winner_usd": _number(average_winner),
        "average_loser_abs_usd": _number(average_loser),
        "payoff_ratio": _number(payoff),
        "mean_net_usd_per_trade": float(trades["net_pnl_usd"].mean()) if len(trades) else None,
        "median_net_usd_per_trade": float(trades["net_pnl_usd"].median()) if len(trades) else None,
        "net_usd_total": float(pnl.sum()),
        "net_usd_per_session": float(pnl.mean()),
        "return_per_session": float(pnl.mean() / start),
        "net_usd_per_session_corrected_interval": _corrected_session_interval(
            pnl.to_numpy(float), family_size=family_size
        ),
        "maximum_drawdown_usd": _maximum_drawdown(pnl, start),
        "worst_trade_usd": float(trades["net_pnl_usd"].min()) if len(trades) else None,
        "starting_equity_usd": start,
        "risk_mode": next(iter(risk_modes)),
        "trade_cap": int(next(iter(trade_caps))),
        "premium_at_risk_mean_usd": float(trades["entry_ask_usd"].mean()) if len(trades) else None,
        "premium_at_risk_max_usd": float(trades["entry_ask_usd"].max()) if len(trades) else None,
        "capital_utilisation_mean": float(trades["entry_ask_usd"].mean() / start) if len(trades) else None,
        "capital_utilisation_max": float(trades["entry_ask_usd"].max() / start) if len(trades) else None,
        "spread_paid_mean_usd": (
            float(trades["total_spread_paid_usd"].mean())
            if len(trades) and "total_spread_paid_usd" in trades
            else None
        ),
        "fees_total_usd": (
            float(trades["fees_usd"].sum())
            if len(trades) and "fees_usd" in trades
            else None
        ),
        "holding_time_mean_minutes": float(trades["minutes_held"].mean()) if len(trades) else None,
        "holding_time_median_minutes": float(trades["minutes_held"].median()) if len(trades) else None,
        "entry_otm_depth_mean_points": (
            float(trades["entry_otm_depth_points"].mean()) if len(trades) else None
        ),
        "entry_otm_depth_median_points": (
            float(trades["entry_otm_depth_points"].median()) if len(trades) else None
        ),
        "maximum_itm_depth_mean_points": (
            float(trades["maximum_itm_depth_points"].mean()) if len(trades) else None
        ),
        "final_itm_depth_mean_points": (
            float(trades["final_itm_depth_points"].mean()) if len(trades) else None
        ),
        "otm_to_itm_conversion_rate": (
            float(trades["otm_to_itm_conversion"].astype(bool).mean()) if len(trades) else None
        ),
        "time_to_cross_mean_minutes": (
            _number(float(trades["time_to_cross_minutes"].mean())) if len(trades) else None
        ),
        "underlying_mfe_mean_points": (
            float(trades["underlying_mfe_points"].mean()) if len(trades) else None
        ),
        "underlying_mae_mean_points": (
            float(trades["underlying_mae_points"].mean()) if len(trades) else None
        ),
        "maximum_favourable_excursion_mean_usd": (
            float(trades["maximum_unrealised_usd"].replace([np.inf, -np.inf], np.nan).mean())
            if len(trades)
            else None
        ),
        "maximum_adverse_excursion_mean_usd": (
            float(trades["minimum_unrealised_usd"].replace([np.inf, -np.inf], np.nan).mean())
            if len(trades)
            else None
        ),
        "calls": int(trades["right"].eq("C").sum()) if len(trades) else 0,
        "puts": int(trades["right"].eq("P").sum()) if len(trades) else 0,
        "morning_origin_trades": int(trades["origin_regime"].eq("morning").sum()) if len(trades) else 0,
        "afternoon_origin_trades": int(trades["origin_regime"].eq("afternoon").sum()) if len(trades) else 0,
        "cash_settled_trades": int(trades["exit_type"].eq("validated_cash_settlement").sum()) if len(trades) else 0,
        "zero_recovery_sensitivity_trades": int(trades["exit_type"].eq("zero_recovery_sensitivity").sum()) if len(trades) else 0,
        "planned_loss_usd": None,
        "planned_loss_status": "UNKNOWN_UNLESS_POLICY_DIAGNOSTICS_DECLARE_IT",
    }
    return report
