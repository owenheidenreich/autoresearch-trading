"""Broker adapter for guarded IBKR paper option orders.

The executor is intentionally thin: it converts an already-approved paper order
intent into an IBKR contract/order pair and calls ``placeOrder`` only after the
paper guard has passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v4.live.ibkr_paper_guard import (
    PaperOrderGuardConfig,
    PaperOrderIntent,
    paper_order_permission,
    validate_order_intent,
)
from v4.live.paper_trade_log import append_trade_event, executor_result_event, trade_log_path


@dataclass(frozen=True)
class PaperExecutionConfig:
    guard: PaperOrderGuardConfig = PaperOrderGuardConfig()
    order_type: str = "LMT"
    tif: str = "DAY"
    outside_rth: bool = False


def execute_guarded_paper_order(
    *,
    ib: Any,
    option_cls: Any,
    order_cls: Any,
    intent: PaperOrderIntent,
    account_id: str | None,
    account_cash: float,
    open_positions: int,
    quote: dict[str, Any],
    context: dict[str, Any],
    enable_paper_orders: bool,
    acknowledge_paper_loss: bool,
    dry_run: bool = True,
    config: PaperExecutionConfig = PaperExecutionConfig(),
    environ: dict[str, str] | None = None,
    trade_log_root: Any | None = None,
    trade_log_run_id: str | None = None,
    trade_uid: str | None = None,
) -> dict[str, Any]:
    """Validate, optionally submit, and return a broker-safe execution record."""

    permission = paper_order_permission(
        enable_paper_orders=enable_paper_orders,
        acknowledge_paper_loss=acknowledge_paper_loss,
        account_id=account_id,
        config=config.guard,
        environ=environ,
    )
    validation = validate_order_intent(
        intent,
        account_cash=account_cash,
        open_positions=open_positions,
        quote=quote,
        context=context,
        config=config.guard,
    )
    base = {
        "permission": permission,
        "validation": validation,
        "dry_run": bool(dry_run),
        "broker_order_endpoint_called": False,
        "paper_order_submitted": False,
        "intent": intent.__dict__,
        "quote": quote,
        "context": context,
        "account": {
            "account_id_redacted": permission.get("account_id_redacted"),
            "cash": validation.get("account_cash"),
            "open_positions": validation.get("open_positions"),
        },
    }
    if not permission["passed"] or not validation["passed"]:
        result = {**base, "status": "blocked", "reason": blocked_reason(permission, validation)}
        maybe_log_executor_result(result, trade_log_root=trade_log_root, trade_log_run_id=trade_log_run_id, trade_uid=trade_uid)
        return result

    contract = build_option_contract(option_cls, intent)
    order = build_limit_order(order_cls, intent, config=config)
    if dry_run:
        result = {
            **base,
            "status": "dry_run_pass",
            "reason": "paper_order_validated_not_submitted",
            "qualified_contracts": 0,
            "contract_preview": contract_preview(contract),
            "order_preview": order_preview(order),
        }
        maybe_log_executor_result(result, trade_log_root=trade_log_root, trade_log_run_id=trade_log_run_id, trade_uid=trade_uid)
        return result

    qualified_contracts = []
    if hasattr(ib, "qualifyContracts"):
        qualified_contracts = list(ib.qualifyContracts(contract) or [])
        if qualified_contracts:
            contract = qualified_contracts[0]

    trade = ib.placeOrder(contract, order)
    result = {
        **base,
        "status": "submitted",
        "reason": "paper_order_submitted",
        "broker_order_endpoint_called": True,
        "paper_order_submitted": True,
        "qualified_contracts": len(qualified_contracts),
        "contract_preview": contract_preview(contract),
        "order_preview": order_preview(order),
        "trade_preview": trade_preview(trade),
    }
    maybe_log_executor_result(result, trade_log_root=trade_log_root, trade_log_run_id=trade_log_run_id, trade_uid=trade_uid)
    return result


def build_option_contract(option_cls: Any, intent: PaperOrderIntent) -> Any:
    return option_cls(
        intent.symbol,
        intent.expiry,
        float(intent.strike),
        intent.right,
        intent.exchange,
        currency=intent.currency,
        tradingClass=intent.trading_class,
    )


def build_limit_order(order_cls: Any, intent: PaperOrderIntent, *, config: PaperExecutionConfig) -> Any:
    order = order_cls(
        str(intent.action).upper(),
        int(intent.quantity),
        round(float(intent.limit_price), 2),
        tif=config.tif,
        outsideRth=bool(config.outside_rth),
    )
    return order


def blocked_reason(permission: dict[str, Any], validation: dict[str, Any]) -> str:
    reasons = []
    if not permission.get("passed"):
        reasons.extend(permission.get("reasons", []))
    if not validation.get("passed"):
        reasons.extend(validation.get("reasons", []))
    return ",".join(str(reason) for reason in reasons) or "blocked"


def contract_preview(contract: Any) -> dict[str, Any]:
    return {
        "symbol": getattr(contract, "symbol", None),
        "lastTradeDateOrContractMonth": getattr(contract, "lastTradeDateOrContractMonth", None),
        "strike": getattr(contract, "strike", None),
        "right": getattr(contract, "right", None),
        "exchange": getattr(contract, "exchange", None),
        "currency": getattr(contract, "currency", None),
        "tradingClass": getattr(contract, "tradingClass", None),
    }


def order_preview(order: Any) -> dict[str, Any]:
    return {
        "action": getattr(order, "action", None),
        "totalQuantity": getattr(order, "totalQuantity", None),
        "lmtPrice": getattr(order, "lmtPrice", None),
        "tif": getattr(order, "tif", None),
        "outsideRth": getattr(order, "outsideRth", None),
    }


def trade_preview(trade: Any) -> dict[str, Any]:
    order = getattr(trade, "order", None)
    contract = getattr(trade, "contract", None)
    return {
        "contract": contract_preview(contract) if contract is not None else None,
        "order": order_preview(order) if order is not None else None,
    }


def maybe_log_executor_result(
    result: dict[str, Any],
    *,
    trade_log_root: Any | None,
    trade_log_run_id: str | None,
    trade_uid: str | None,
) -> None:
    if trade_log_root is None:
        return
    run_id = trade_log_run_id or "protocol101_paper"
    event = executor_result_event(
        result=result,
        run_id=run_id,
        mode="paper_executor",
        trade_uid=trade_uid,
    )
    path = trade_log_path(root=trade_log_root, session=str(event["session"]), run_id=run_id)
    append_trade_event(path, event)
