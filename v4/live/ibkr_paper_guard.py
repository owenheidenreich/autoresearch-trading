"""IBKR paper-order guardrails for Protocol 101.

This module deliberately separates paper-order permission from model research.
It never reads credentials and never enables orders by default.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any


CONTRACT_MULTIPLIER = 100.0
PAPER_PERMISSION_ENV = "V4_ALLOW_IBKR_PAPER_ORDERS"


@dataclass(frozen=True)
class PaperOrderGuardConfig:
    starting_paper_cash: float = 10_000.0
    ibkr_access_reserve: float = 500.0
    max_order_quantity: int = 1
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
    enable_flag_present = bool(enable_paper_orders)
    acknowledge_flag_present = bool(acknowledge_paper_loss)
    env_present = str(env.get(PAPER_PERMISSION_ENV, "")).strip().upper() == "YES"
    if not enable_flag_present:
        reasons.append("enable_paper_orders_flag_missing")
    if not acknowledge_flag_present:
        reasons.append("acknowledge_paper_loss_flag_missing")
    if not env_present:
        reasons.append("paper_order_env_not_set")
    clean_account = str(account_id or "").strip()
    account_prefix_ok = True
    if not clean_account:
        account_prefix_ok = False
        reasons.append("missing_account_id")
    elif config.require_paper_account_prefix:
        account_prefix_ok = clean_account.startswith(config.paper_account_prefixes)
        if not account_prefix_ok:
            reasons.append("account_not_recognized_as_paper")
    return {
        "passed": not reasons,
        "reason": "pass" if not reasons else ",".join(reasons),
        "reasons": reasons,
        "guard_passed": not reasons,
        "guard_block_reasons": reasons,
        "permission_enable_flag_present": enable_flag_present,
        "permission_ack_flag_present": acknowledge_flag_present,
        "permission_env_present": env_present,
        "account_prefix_ok": account_prefix_ok,
        "paper_account_confirmed": bool(clean_account and account_prefix_ok),
        "real_money_false_confirmed": True,
        "account_id_redacted": redact_account_id(clean_account),
        "required_env": f"{PAPER_PERMISSION_ENV}=YES",
        "guard_config_digest": guard_config_digest(config),
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
    quantity_ok = True
    one_open_position_ok = True
    quote_freshness_ok = True
    context_freshness_ok = True
    affordability_ok = True
    if action not in {"BUY", "SELL"}:
        reasons.append("invalid_action")
    if intent.symbol != "SPX":
        reasons.append("invalid_underlying_symbol")
    if intent.trading_class != "SPXW":
        reasons.append("wrong_trading_class")
    if right not in {"C", "P"}:
        reasons.append("invalid_right")
    if int(intent.quantity) <= 0:
        quantity_ok = False
        reasons.append("nonpositive_quantity")
    elif int(intent.quantity) > int(config.max_order_quantity):
        quantity_ok = False
        reasons.append("paper_quantity_exceeds_one_contract_limit")
    if int(open_positions) >= int(config.max_concurrent_positions) and action == "BUY":
        one_open_position_ok = False
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
        quote_freshness_ok = False
        reasons.append("missing_quote_age")
    elif quote_age_ms > config.max_option_quote_age_ms:
        quote_freshness_ok = False
        reasons.append("stale_option_quote")
    if context_age_ms is None:
        context_freshness_ok = False
        reasons.append("missing_context_age")
    elif context_age_ms > config.max_context_age_ms:
        context_freshness_ok = False
        reasons.append("stale_context")
    if not math.isfinite(float(intent.limit_price)) or float(intent.limit_price) <= 0:
        reasons.append("invalid_limit_price")
    if ask is not None and reference_ask is not None and ask - reference_ask > config.max_limit_offset + 1e-9:
        reasons.append("ask_moved_beyond_budget")
    if action == "BUY" and intent.premium_required > float(account_cash) + 1e-9:
        affordability_ok = False
        reasons.append("insufficient_paper_cash")

    return {
        "passed": not reasons,
        "reason": "pass" if not reasons else ",".join(reasons),
        "reasons": reasons,
        "guard_passed": not reasons,
        "guard_block_reasons": reasons,
        "quantity_ok": quantity_ok,
        "one_open_position_ok": one_open_position_ok,
        "quote_freshness_ok": quote_freshness_ok,
        "context_freshness_ok": context_freshness_ok,
        "affordability_ok": affordability_ok,
        "real_money_false_confirmed": True,
        "guard_config_digest": guard_config_digest(config),
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


def guard_config_digest(config: PaperOrderGuardConfig) -> str:
    payload = {
        "starting_paper_cash": config.starting_paper_cash,
        "ibkr_access_reserve": config.ibkr_access_reserve,
        "max_order_quantity": config.max_order_quantity,
        "max_concurrent_positions": config.max_concurrent_positions,
        "require_paper_account_prefix": config.require_paper_account_prefix,
        "paper_account_prefixes": list(config.paper_account_prefixes),
        "max_limit_offset": config.max_limit_offset,
        "max_option_quote_age_ms": config.max_option_quote_age_ms,
        "max_context_age_ms": config.max_context_age_ms,
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
