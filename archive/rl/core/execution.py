"""Execution math for v3's physical-only environment."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution and action-bound settings."""

    starting_equity: float = 10_000.0
    contract_multiplier: int = 100
    commission_per_contract: float = 1.30
    max_risk_budget_frac: float = 0.05
    min_stop_frac: float = 0.02
    max_stop_frac: float = 0.95
    min_target_frac: float = 0.05
    max_target_frac: float = 3.00
    min_time_stop_frac: float = 1.0 / 390.0
    max_time_stop_frac: float = 1.0
    max_scale_fraction: float = 1.0
    size_slippage_alpha: float = 0.20
    size_slippage_power: float = 0.50
    minimum_tick_under_3: float = 0.05
    minimum_tick_over_3: float = 0.10
    max_affordable_qty_cap: int = 250

    def to_dict(self) -> dict:
        return asdict(self)


def squash_to_range(raw: float, low: float, high: float) -> float:
    """Map tanh-squashed action in [-1, 1] into [low, high]."""

    raw = max(-1.0, min(1.0, float(raw)))
    return low + (raw + 1.0) * 0.5 * (high - low)


def range_to_squashed(value: float, low: float, high: float) -> float:
    """Inverse of squash_to_range for baseline/test actions."""

    if high <= low:
        return 0.0
    value = max(low, min(high, float(value)))
    return ((value - low) / (high - low)) * 2.0 - 1.0


def size_aware_slippage_frac(
    *,
    mid: float,
    bid: float | None,
    ask: float | None,
    qty: int,
    config: ExecutionConfig,
) -> float:
    """Half-turn slippage fraction for a buy or sell fill."""

    if mid <= 0 or qty <= 0:
        return 1.0
    spread = 0.0
    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        spread = (ask - bid) / max(mid, 1e-6)
    min_tick = config.minimum_tick_under_3 if mid < 3.0 else config.minimum_tick_over_3
    tick_frac = min_tick / max(mid, 1e-6)
    half_spread = max(spread * 0.5, tick_frac)
    size_mult = 1.0 + config.size_slippage_alpha * (max(qty, 1) - 1) ** config.size_slippage_power
    return half_spread * size_mult


def buy_fill_price(mid: float, bid: float | None, ask: float | None, qty: int, config: ExecutionConfig) -> float:
    return max(0.01, mid * (1.0 + size_aware_slippage_frac(mid=mid, bid=bid, ask=ask, qty=qty, config=config)))


def sell_fill_price(mid: float, bid: float | None, ask: float | None, qty: int, config: ExecutionConfig) -> float:
    return max(0.01, mid * (1.0 - size_aware_slippage_frac(mid=mid, bid=bid, ask=ask, qty=qty, config=config)))


def commission_dollars(qty: int, config: ExecutionConfig) -> float:
    return float(qty) * config.commission_per_contract


def map_entry_fracs(
    *,
    risk_budget_raw: float,
    stop_raw: float,
    target_raw: float,
    time_stop_raw: float,
    config: ExecutionConfig,
) -> tuple[float, float, float, float]:
    """Convert model actions into environment parameters."""

    risk_budget = squash_to_range(risk_budget_raw, 0.0, config.max_risk_budget_frac)
    stop_frac = squash_to_range(stop_raw, config.min_stop_frac, config.max_stop_frac)
    target_frac = squash_to_range(target_raw, config.min_target_frac, config.max_target_frac)
    time_stop_frac = squash_to_range(time_stop_raw, config.min_time_stop_frac, config.max_time_stop_frac)
    return risk_budget, stop_frac, target_frac, time_stop_frac


def time_stop_to_bar(
    *,
    current_bar: int,
    time_stop_frac: float,
    bars_per_day: int,
) -> int:
    remaining = max(1, bars_per_day - current_bar - 1)
    hold_bars = max(1, int(math.ceil(time_stop_frac * remaining)))
    return min(bars_per_day - 1, current_bar + hold_bars)


def worst_case_loss_per_contract(
    *,
    entry_mid: float,
    bid: float | None,
    ask: float | None,
    stop_frac: float,
    qty: int,
    config: ExecutionConfig,
) -> float:
    """Worst-case dollar loss per contract at the chosen stop."""

    if entry_mid <= 0:
        return float("inf")
    entry_fill = buy_fill_price(entry_mid, bid, ask, qty, config)
    stop_mid = max(0.01, entry_fill * (1.0 - stop_frac))
    stop_fill = sell_fill_price(stop_mid, bid, ask, qty, config)
    gross_loss = max(0.0, entry_fill - stop_fill) * config.contract_multiplier
    commissions = 2.0 * commission_dollars(1, config)
    return gross_loss + commissions


def affordable_qty(*, cash: float, entry_mid: float, bid: float | None, ask: float | None, config: ExecutionConfig) -> int:
    """Maximum integer quantity the account can afford."""

    if cash <= 0 or entry_mid <= 0:
        return 0
    qty = 0
    for candidate in range(1, config.max_affordable_qty_cap + 1):
        fill = buy_fill_price(entry_mid, bid, ask, candidate, config)
        cost = fill * config.contract_multiplier * candidate + commission_dollars(candidate, config)
        if cost > cash:
            break
        qty = candidate
    return qty


def risk_budget_to_qty(
    *,
    risk_budget_frac: float,
    equity: float,
    cash: float,
    entry_mid: float,
    bid: float | None,
    ask: float | None,
    stop_frac: float,
    config: ExecutionConfig,
) -> int:
    """Convert a risk budget into an executable integer position size."""

    if equity <= 0 or cash <= 0 or risk_budget_frac <= 0 or entry_mid <= 0:
        return 0
    risk_budget_dollars = equity * risk_budget_frac
    max_affordable = affordable_qty(cash=cash, entry_mid=entry_mid, bid=bid, ask=ask, config=config)
    if max_affordable <= 0:
        return 0
    qty = min(
        max_affordable,
        max(1, int(risk_budget_dollars / max(worst_case_loss_per_contract(
            entry_mid=entry_mid,
            bid=bid,
            ask=ask,
            stop_frac=stop_frac,
            qty=1,
            config=config,
        ), 1e-6))),
    )
    while qty > 0:
        total_risk = qty * worst_case_loss_per_contract(
            entry_mid=entry_mid,
            bid=bid,
            ask=ask,
            stop_frac=stop_frac,
            qty=qty,
            config=config,
        )
        if total_risk <= risk_budget_dollars and qty <= max_affordable:
            return qty
        qty -= 1
    return 0
