"""Offline position-sizing account simulator for frozen Protocol 101.

The simulator is intentionally separate from live paper trading. It answers a
research question: if the frozen one-contract trade stream were replayed with a
quantity rule, how would cash, drawdown, daily loss, and premium exposure evolve?
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


CONTRACT_MULTIPLIER = 100.0


@dataclass(frozen=True)
class PositionSizingPolicy:
    name: str
    starting_cash: float = 10_000.0
    max_contracts: int = 1
    equity_for_two_contracts: float = math.inf
    equity_for_three_contracts: float = math.inf
    premium_exposure_fraction: float = 1.0
    max_premium_dollars: float = math.inf
    daily_new_entry_stop_loss: float | None = None
    max_drawdown_for_scaling: float = 1.0
    require_recent_positive_for_scaling: bool = False
    recent_window: int = 10
    initial_contracts: int = 1
    min_score_margin_for_two: float | None = None
    min_score_margin_for_three: float | None = None
    min_profit_for_scaling: float = 0.0
    profit_cushion_multiplier: float = 0.0
    scale_only_if_daily_pnl_nonnegative: bool = False
    max_one_contract_premium_for_scaling: float = math.inf


@dataclass
class AccountState:
    cash: float
    peak_equity: float
    daily_realized_pnl: dict[str, float]
    recent_trade_pnls: list[float]

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.cash) / self.peak_equity)


def baseline_one_contract_policy() -> PositionSizingPolicy:
    return PositionSizingPolicy(
        name="one_contract_baseline",
        max_contracts=1,
        premium_exposure_fraction=1.0,
        max_premium_dollars=math.inf,
        daily_new_entry_stop_loss=None,
    )


def conservative_profit_ladder_policy() -> PositionSizingPolicy:
    return PositionSizingPolicy(
        name="conservative_profit_ladder",
        max_contracts=3,
        equity_for_two_contracts=20_000.0,
        equity_for_three_contracts=50_000.0,
        premium_exposure_fraction=0.40,
        max_premium_dollars=12_000.0,
        daily_new_entry_stop_loss=-750.0,
        max_drawdown_for_scaling=0.05,
        require_recent_positive_for_scaling=True,
    )


def strict_exposure_ladder_policy() -> PositionSizingPolicy:
    return PositionSizingPolicy(
        name="strict_exposure_ladder",
        max_contracts=3,
        equity_for_two_contracts=20_000.0,
        equity_for_three_contracts=50_000.0,
        premium_exposure_fraction=0.20,
        max_premium_dollars=8_000.0,
        daily_new_entry_stop_loss=-750.0,
        max_drawdown_for_scaling=0.05,
        require_recent_positive_for_scaling=True,
    )


def recovery_lock_policy() -> PositionSizingPolicy:
    return PositionSizingPolicy(
        name="recovery_lock_ladder",
        max_contracts=3,
        equity_for_two_contracts=30_000.0,
        equity_for_three_contracts=75_000.0,
        premium_exposure_fraction=0.30,
        max_premium_dollars=10_000.0,
        daily_new_entry_stop_loss=-750.0,
        max_drawdown_for_scaling=0.03,
        require_recent_positive_for_scaling=True,
    )


def confidence_ladder_policy() -> PositionSizingPolicy:
    return PositionSizingPolicy(
        name="confidence_ladder",
        max_contracts=3,
        equity_for_two_contracts=25_000.0,
        equity_for_three_contracts=75_000.0,
        premium_exposure_fraction=0.30,
        max_premium_dollars=10_000.0,
        daily_new_entry_stop_loss=-750.0,
        max_drawdown_for_scaling=0.03,
        require_recent_positive_for_scaling=True,
        min_score_margin_for_two=2.0,
        min_score_margin_for_three=2.9,
        min_profit_for_scaling=10_000.0,
        profit_cushion_multiplier=2.0,
        scale_only_if_daily_pnl_nonnegative=True,
    )


def high_conviction_profit_cushion_policy() -> PositionSizingPolicy:
    return PositionSizingPolicy(
        name="high_conviction_profit_cushion",
        max_contracts=3,
        equity_for_two_contracts=30_000.0,
        equity_for_three_contracts=100_000.0,
        premium_exposure_fraction=0.25,
        max_premium_dollars=8_000.0,
        daily_new_entry_stop_loss=-750.0,
        max_drawdown_for_scaling=0.025,
        require_recent_positive_for_scaling=True,
        min_score_margin_for_two=2.4,
        min_score_margin_for_three=3.2,
        min_profit_for_scaling=20_000.0,
        profit_cushion_multiplier=3.0,
        scale_only_if_daily_pnl_nonnegative=True,
        max_one_contract_premium_for_scaling=3_000.0,
    )


def slow_growth_two_contract_policy() -> PositionSizingPolicy:
    return PositionSizingPolicy(
        name="slow_growth_two_contract",
        max_contracts=2,
        equity_for_two_contracts=40_000.0,
        equity_for_three_contracts=math.inf,
        premium_exposure_fraction=0.20,
        max_premium_dollars=6_000.0,
        daily_new_entry_stop_loss=-750.0,
        max_drawdown_for_scaling=0.02,
        require_recent_positive_for_scaling=True,
        min_score_margin_for_two=2.2,
        min_profit_for_scaling=30_000.0,
        profit_cushion_multiplier=4.0,
        scale_only_if_daily_pnl_nonnegative=True,
        max_one_contract_premium_for_scaling=2_750.0,
    )


def default_position_sizing_policies() -> tuple[PositionSizingPolicy, ...]:
    return (
        baseline_one_contract_policy(),
        strict_exposure_ladder_policy(),
        conservative_profit_ladder_policy(),
        recovery_lock_policy(),
        confidence_ladder_policy(),
        high_conviction_profit_cushion_policy(),
        slow_growth_two_contract_policy(),
    )


def simulate_position_sizing(trades: list[dict[str, Any]], policy: PositionSizingPolicy) -> dict[str, Any]:
    state = AccountState(
        cash=float(policy.starting_cash),
        peak_equity=float(policy.starting_cash),
        daily_realized_pnl={},
        recent_trade_pnls=[],
    )
    rows: list[dict[str, Any]] = []
    for trade in sorted(trades, key=_trade_sort_key):
        session = str(trade["session"])
        cash_before = state.cash
        quantity, reason = choose_quantity(trade, state, policy)
        pnl = float(trade["pnl"]) * quantity
        state.cash += pnl
        state.peak_equity = max(state.peak_equity, state.cash)
        if quantity > 0:
            state.daily_realized_pnl[session] = state.daily_realized_pnl.get(session, 0.0) + pnl
            state.recent_trade_pnls.append(pnl)
            state.recent_trade_pnls = state.recent_trade_pnls[-max(1, int(policy.recent_window)) :]
        rows.append(
            {
                "policy": policy.name,
                "trade_number": int(trade.get("trade_number", len(rows) + 1)),
                "session": session,
                "stage": str(trade.get("stage", "")),
                "segment": str(trade.get("segment", "")),
                "decision_time": str(trade["decision_time"]),
                "exit_time": str(trade["exit_time"]),
                "contract_id": str(trade["contract_id"]),
                "side": str(trade.get("side", "")),
                "score": number(trade.get("score")),
                "threshold": number(trade.get("threshold")),
                "score_margin": score_margin(trade),
                "one_contract_premium": premium_dollars(trade),
                "one_contract_pnl": float(trade["pnl"]),
                "quantity": int(quantity),
                "premium_exposure": premium_dollars(trade) * quantity if premium_dollars(trade) is not None else None,
                "cash_before": round(cash_before, 6),
                "cash_after": round(state.cash, 6),
                "peak_equity_after": round(state.peak_equity, 6),
                "drawdown_pct_before": round(drawdown_pct(cash_before, state.peak_equity), 6),
                "daily_pnl_before": round(state.daily_realized_pnl.get(session, 0.0) - pnl, 6),
                "realized_pnl": round(pnl, 6),
                "skip_reason": reason,
            }
        )
    summary = summarize_position_sizing(rows, policy)
    return {"policy": policy, "rows": rows, "summary": summary, "daily": daily_summary(rows, policy.name)}


def choose_quantity(
    trade: dict[str, Any],
    state: AccountState,
    policy: PositionSizingPolicy,
) -> tuple[int, str]:
    session = str(trade["session"])
    premium = premium_dollars(trade)
    if premium is None or premium <= 0:
        return 0, "missing_premium"
    daily = state.daily_realized_pnl.get(session, 0.0)
    if policy.daily_new_entry_stop_loss is not None and daily <= policy.daily_new_entry_stop_loss:
        return 0, "daily_loss_stop"

    max_by_equity = max_contracts_by_equity(state.cash, policy)
    if state.cash - policy.starting_cash < policy.min_profit_for_scaling:
        max_by_equity = min(max_by_equity, policy.initial_contracts)
    if state.drawdown_pct > policy.max_drawdown_for_scaling:
        max_by_equity = min(max_by_equity, policy.initial_contracts)
    if policy.require_recent_positive_for_scaling and max_by_equity > policy.initial_contracts:
        recent_total = sum(state.recent_trade_pnls[-max(1, int(policy.recent_window)) :])
        if recent_total <= 0:
            max_by_equity = policy.initial_contracts
    if policy.scale_only_if_daily_pnl_nonnegative and daily < 0:
        max_by_equity = min(max_by_equity, policy.initial_contracts)
    if premium > policy.max_one_contract_premium_for_scaling:
        max_by_equity = min(max_by_equity, policy.initial_contracts)
    margin = score_margin(trade)
    if max_by_equity >= 3 and policy.min_score_margin_for_three is not None:
        if margin is None or margin < policy.min_score_margin_for_three:
            max_by_equity = 2
    if max_by_equity >= 2 and policy.min_score_margin_for_two is not None:
        if margin is None or margin < policy.min_score_margin_for_two:
            max_by_equity = 1
    if max_by_equity > 1 and policy.profit_cushion_multiplier > 0:
        profit = max(0.0, state.cash - policy.starting_cash)
        affordable_extra_by_profit = math.floor(profit / max(premium * policy.profit_cushion_multiplier, 1e-9))
        max_by_equity = min(max_by_equity, 1 + int(affordable_extra_by_profit))

    max_by_cash = math.floor(state.cash / premium)
    exposure_cap = min(policy.max_premium_dollars, state.cash * policy.premium_exposure_fraction)
    max_by_exposure = math.floor(exposure_cap / premium)
    quantity = int(min(policy.max_contracts, max_by_equity, max_by_cash, max_by_exposure))
    if quantity <= 0:
        if max_by_cash <= 0:
            return 0, "insufficient_cash"
        return 0, "premium_exposure_cap"
    return quantity, ""


def max_contracts_by_equity(equity: float, policy: PositionSizingPolicy) -> int:
    if equity >= policy.equity_for_three_contracts:
        return min(3, policy.max_contracts)
    if equity >= policy.equity_for_two_contracts:
        return min(2, policy.max_contracts)
    return min(policy.initial_contracts, policy.max_contracts)


def summarize_position_sizing(rows: list[dict[str, Any]], policy: PositionSizingPolicy) -> dict[str, Any]:
    cash_path = [float(policy.starting_cash)] + [float(row["cash_after"]) for row in rows]
    peak = float(policy.starting_cash)
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    longest_recovery = 0
    underwater_start: int | None = None
    for index, value in enumerate(cash_path):
        if value >= peak:
            if underwater_start is not None:
                longest_recovery = max(longest_recovery, index - underwater_start)
                underwater_start = None
            peak = value
        else:
            if underwater_start is None:
                underwater_start = index
        drawdown = value - peak
        max_drawdown = min(max_drawdown, drawdown)
        max_drawdown_pct = min(max_drawdown_pct, drawdown / peak if peak else 0.0)
    if underwater_start is not None:
        longest_recovery = max(longest_recovery, len(cash_path) - underwater_start)

    daily = daily_summary(rows, policy.name)
    skip_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("skip_reason") or "")
        if reason:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
    ending = cash_path[-1] if cash_path else float(policy.starting_cash)
    return {
        "policy": policy.name,
        "starting_cash": round(float(policy.starting_cash), 2),
        "ending_cash": round(ending, 2),
        "total_pnl": round(ending - float(policy.starting_cash), 2),
        "return_on_starting_cash": round((ending - float(policy.starting_cash)) / float(policy.starting_cash), 6),
        "candidate_trades": len(rows),
        "taken_trades": sum(1 for row in rows if int(row["quantity"]) > 0),
        "skipped_trades": sum(1 for row in rows if int(row["quantity"]) <= 0),
        "skip_counts": dict(sorted(skip_counts.items())),
        "total_contracts": sum(int(row["quantity"]) for row in rows),
        "max_quantity": max([int(row["quantity"]) for row in rows] or [0]),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 6),
        "worst_day_pnl": round(min([row["daily_pnl"] for row in daily] or [0.0]), 2),
        "best_day_pnl": round(max([row["daily_pnl"] for row in daily] or [0.0]), 2),
        "min_cash": round(min(cash_path), 2),
        "risk_of_ruin": any(value <= 0 for value in cash_path),
        "longest_recovery_trades": int(longest_recovery),
    }


def daily_summary(rows: list[dict[str, Any]], policy_name: str) -> list[dict[str, Any]]:
    daily: dict[str, float] = {}
    for row in rows:
        if int(row["quantity"]) <= 0:
            continue
        session = str(row["session"])
        daily[session] = daily.get(session, 0.0) + float(row["realized_pnl"])
    return [
        {"policy": policy_name, "session": session, "daily_pnl": round(value, 6)}
        for session, value in sorted(daily.items())
    ]


def premium_dollars(trade: dict[str, Any]) -> float | None:
    premium = number(trade.get("premium_paid"))
    if premium is not None:
        return premium
    ask = number(trade.get("entry_ask"))
    return None if ask is None else ask * CONTRACT_MULTIPLIER


def score_margin(trade: dict[str, Any]) -> float | None:
    score = number(trade.get("score"))
    threshold = number(trade.get("threshold"))
    if score is None or threshold is None:
        return None
    return score - threshold


def drawdown_pct(cash: float, peak: float) -> float:
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - cash) / peak)


def _trade_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    decision = pd.Timestamp(row["decision_time"])
    exit_time = pd.Timestamp(row["exit_time"])
    return (int(decision.value // 1_000_000), int(exit_time.value // 1_000_000), str(row.get("candidate_uid", "")))


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
