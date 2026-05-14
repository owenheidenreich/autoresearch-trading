"""Deterministic paper-account risk gate for frozen Protocol 101."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CONTRACT_MULTIPLIER = 100.0


@dataclass(frozen=True)
class Protocol101RiskConfig:
    starting_cash: float = 10_000.0
    ibkr_access_reserve: float = 500.0
    max_contracts_initial: int = 1
    max_concurrent_positions: int = 1
    max_premium_dollars: float = 4_000.0
    max_premium_fraction_of_equity: float = 0.40
    daily_new_entry_stop_loss: float = -750.0
    max_option_quote_age_ms: int = 1500
    max_context_age_ms: int = 5000
    max_entry_ask_move: float = 0.25
    require_spxw_pm: bool = True


@dataclass(frozen=True)
class AccountState:
    cash: float
    equity: float
    realized_daily_pnl: float = 0.0
    open_positions: int = 0


def premium_cap(config: Protocol101RiskConfig, equity: float) -> float:
    return min(float(config.max_premium_dollars), float(config.max_premium_fraction_of_equity) * float(equity))


def evaluate_entry_risk_gate(
    *,
    contract: dict[str, Any],
    quote: dict[str, Any],
    context: dict[str, Any],
    account: AccountState,
    config: Protocol101RiskConfig = Protocol101RiskConfig(),
) -> dict[str, Any]:
    """Return pass/fail and exact reasons for a potential one-contract entry."""

    reasons: list[str] = []
    bid = _number(quote.get("bid"))
    ask = _number(quote.get("ask"))
    quote_age = _number(quote.get("quote_age_ms"))
    context_age = _number(context.get("context_age_ms"))
    entry_reference_ask = _number(quote.get("reference_ask"))
    root = str(contract.get("root") or _root_from_contract_id(contract.get("contract_id")))
    settlement = str(contract.get("settlement_style") or "")
    quantity = int(_number(contract.get("quantity")) or 1)

    if config.require_spxw_pm and root != "SPXW":
        reasons.append("wrong_root")
    if config.require_spxw_pm and settlement and settlement != "PM":
        reasons.append("wrong_settlement")
    if quantity != config.max_contracts_initial:
        reasons.append("position_size_not_initial_one_contract")
    if account.open_positions >= config.max_concurrent_positions:
        reasons.append("max_concurrent_position_reached")
    if account.realized_daily_pnl <= config.daily_new_entry_stop_loss:
        reasons.append("daily_loss_stop")
    if bid is None or ask is None:
        reasons.append("missing_bid_ask")
    elif bid <= 0 or ask <= 0:
        reasons.append("zero_or_negative_bid_ask")
    elif ask <= bid:
        reasons.append("locked_or_crossed_quote")
    if quote_age is None:
        reasons.append("missing_quote_age")
    elif quote_age > config.max_option_quote_age_ms:
        reasons.append("stale_option_quote")
    if context_age is None:
        reasons.append("missing_context_age")
    elif context_age > config.max_context_age_ms:
        reasons.append("stale_context")

    premium = ask * CONTRACT_MULTIPLIER * quantity if ask is not None else math.inf
    cap = premium_cap(config, account.equity)
    if premium > account.cash + 1e-9:
        reasons.append("insufficient_cash")
    if premium > cap + 1e-9:
        reasons.append("premium_cap_exceeded")
    if entry_reference_ask is not None and ask is not None and ask - entry_reference_ask > config.max_entry_ask_move + 1e-9:
        reasons.append("entry_ask_moved_beyond_budget")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "reason": "pass" if not reasons else ",".join(reasons),
        "premium_required": None if not math.isfinite(premium) else round(premium, 6),
        "premium_cap": round(cap, 6),
        "quantity": quantity,
        "cash": round(float(account.cash), 6),
        "equity": round(float(account.equity), 6),
        "daily_pnl": round(float(account.realized_daily_pnl), 6),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _root_from_contract_id(value: Any) -> str:
    text = str(value or "")
    return text.split("-", 1)[0] if text else ""
