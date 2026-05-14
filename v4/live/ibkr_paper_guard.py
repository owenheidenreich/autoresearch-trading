"""IBKR paper-order guardrails for Protocol 101.

This module deliberately separates paper-order permission from model research.
It never reads credentials and never enables orders by default.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any


CONTRACT_MULTIPLIER = 100.0
PAPER_PERMISSION_ENV = "V4_ALLOW_IBKR_PAPER_ORDERS"


@dataclass(frozen=True)
class PaperOrderGuardConfig:
    starting_paper_cash: float = 10_000.0
    ibkr_access_reserve: float = 500.0
    max_concurrent_positions: int = 1
    require_paper_account_prefix: bool = True
    paper_account_prefixes: tuple[str, ...] = ("DU",)
    max_limit_offset: float = 0.25
    max_option_quote_age_ms: int = 1500
    max_context_age_ms: int = 5000


@dataclass(frozen=True)
class PaperOrderIntent:
    action: str
    symbol: str
    expiry: str
    strike: float
    right: str
    quantity: int
    limit_price: float
    trading_class: str = "SPXW"
    exchange: str = "SMART"
    currency: str = "USD"

    @property
    def premium_required(self) -> float:
        return float(self.quantity) * float(self.limit_price) * CONTRACT_MULTIPLIER


def paper_order_permission(
    *,
    enable_paper_orders: bool,
    acknowledge_paper_loss: bool,
    account_id: str | None,
    config: PaperOrderGuardConfig = PaperOrderGuardConfig(),
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic permission result for broker-connected paper orders."""

    env = os.environ if environ is None else environ
    reasons: list[str] = []
    if not enable_paper_orders:
        reasons.append("enable_paper_orders_flag_missing")
    if not acknowledge_paper_loss:
        reasons.append("acknowledge_paper_loss_flag_missing")
    if str(env.get(PAPER_PERMISSION_ENV, "")).strip().upper() != "YES":
        reasons.append("paper_order_env_not_set")
    clean_account = str(account_id or "").strip()
    if not clean_account:
        reasons.append("missing_account_id")
    elif config.require_paper_account_prefix and not clean_account.startswith(config.paper_account_prefixes):
        reasons.append("account_not_recognized_as_paper")
    return {
        "passed": not reasons,
        "reason": "pass" if not reasons else ",".join(reasons),
        "reasons": reasons,
        "account_id_redacted": redact_account_id(clean_account),
        "required_env": f"{PAPER_PERMISSION_ENV}=YES",
    }


def validate_order_intent(
    intent: PaperOrderIntent,
    *,
    account_cash: float,
    open_positions: int,
    quote: dict[str, Any],
    context: dict[str, Any],
    config: PaperOrderGuardConfig = PaperOrderGuardConfig(),
) -> dict[str, Any]:
    """Validate a paper order intent before any broker endpoint is called."""

    reasons: list[str] = []
    action = str(intent.action).upper()
    right = str(intent.right).upper()
    if action not in {"BUY", "SELL"}:
        reasons.append("invalid_action")
    if intent.symbol != "SPX":
        reasons.append("invalid_underlying_symbol")
    if intent.trading_class != "SPXW":
        reasons.append("wrong_trading_class")
    if right not in {"C", "P"}:
        reasons.append("invalid_right")
    if int(intent.quantity) <= 0:
        reasons.append("nonpositive_quantity")
    if int(open_positions) >= int(config.max_concurrent_positions) and action == "BUY":
        reasons.append("max_concurrent_position_reached")

    bid = _number(quote.get("bid"))
    ask = _number(quote.get("ask"))
    reference_ask = _number(quote.get("reference_ask", ask))
    quote_age_ms = _number(quote.get("quote_age_ms"))
    context_age_ms = _number(context.get("context_age_ms"))
    if bid is None or ask is None:
        reasons.append("missing_bid_ask")
    elif bid <= 0 or ask <= 0:
        reasons.append("zero_or_negative_bid_ask")
    elif ask < bid:
        reasons.append("crossed_quote")
    if quote_age_ms is None:
        reasons.append("missing_quote_age")
    elif quote_age_ms > config.max_option_quote_age_ms:
        reasons.append("stale_option_quote")
    if context_age_ms is None:
        reasons.append("missing_context_age")
    elif context_age_ms > config.max_context_age_ms:
        reasons.append("stale_context")
    if not math.isfinite(float(intent.limit_price)) or float(intent.limit_price) <= 0:
        reasons.append("invalid_limit_price")
    if ask is not None and reference_ask is not None and ask - reference_ask > config.max_limit_offset + 1e-9:
        reasons.append("ask_moved_beyond_budget")
    if action == "BUY" and intent.premium_required > float(account_cash) + 1e-9:
        reasons.append("insufficient_paper_cash")

    return {
        "passed": not reasons,
        "reason": "pass" if not reasons else ",".join(reasons),
        "reasons": reasons,
        "premium_required": round(intent.premium_required, 6),
        "account_cash": round(float(account_cash), 6),
        "open_positions": int(open_positions),
    }


def redact_account_id(account_id: str | None) -> str | None:
    if not account_id:
        return None
    text = str(account_id)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
