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
from v4.live.paper_trade_log import append_trade_event, executor_result_event, make_intent_id, trade_log_path


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
    artifact_ids: dict[str, Any] | None = None,
    runtime_flag_digest: str | None = None,
    wait_for_fill_seconds: float = 0.0,
    cancel_unfilled: bool = True,
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
    intent_payload = intent.__dict__
    intent_id = make_intent_id(
        {
            "run_id": trade_log_run_id or "protocol101_paper",
            "trade_uid": trade_uid or "",
            "selected_contract": intent_payload,
            "order": intent_payload,
        }
    )
    base = {
        "permission": permission,
        "validation": validation,
        "dry_run": bool(dry_run),
        "broker_order_endpoint_called": False,
        "paper_order_submitted": False,
        "intent": intent_payload,
        "intent_id": intent_id,
        "quote": quote,
        "context": context,
        "guard_config_digest": validation.get("guard_config_digest") or permission.get("guard_config_digest"),
        "artifact_ids": artifact_ids or {},
        "runtime_flag_digest": runtime_flag_digest or "",
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
    fill_summary = wait_for_order_resolution(
        ib=ib,
        trade=trade,
        order=order,
        seconds=float(wait_for_fill_seconds),
        cancel_unfilled=bool(cancel_unfilled),
    )
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
        "fill_summary": fill_summary,
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
        "order_id": getattr(order, "orderId", None),
        "perm_id": getattr(order, "permId", None),
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
        "order_id": getattr(order, "orderId", None) if order is not None else None,
        "perm_id": getattr(order, "permId", None) if order is not None else None,
    }


def wait_for_order_resolution(
    *,
    ib: Any,
    trade: Any,
    order: Any,
    seconds: float,
    cancel_unfilled: bool,
) -> dict[str, Any]:
    deadline = max(0.0, float(seconds))
    if deadline > 0.0:
        import time

        end = time.monotonic() + deadline
        while time.monotonic() < end:
            summary = order_status_summary(trade)
            if summary["filled"]:
                return summary
            if str(summary.get("status") or "").lower() in {"cancelled", "inactive", "apicancelled"}:
                return summary
            if hasattr(ib, "sleep"):
                ib.sleep(0.25)
            else:
                time.sleep(0.25)
    summary = order_status_summary(trade)
    if not summary["filled"] and cancel_unfilled and hasattr(ib, "cancelOrder"):
        try:
            ib.cancelOrder(order)
            summary = {**summary, "cancel_requested": True}
        except Exception as exc:
            summary = {**summary, "cancel_requested": False, "cancel_error": str(exc)}
    return summary


def order_status_summary(trade: Any) -> dict[str, Any]:
    status_obj = getattr(trade, "orderStatus", None)
    status = getattr(status_obj, "status", None)
    filled = _float_or_zero(getattr(status_obj, "filled", 0.0))
    remaining = _float_or_zero(getattr(status_obj, "remaining", 0.0))
    avg_fill_price = _float_or_none(getattr(status_obj, "avgFillPrice", None))
    fills = list(getattr(trade, "fills", []) or [])
    if fills and (avg_fill_price is None or avg_fill_price <= 0):
        prices = []
        sizes = []
        for fill in fills:
            execution = getattr(fill, "execution", None)
            price = _float_or_none(getattr(execution, "price", None))
            shares = _float_or_none(getattr(execution, "shares", None))
            if price is not None and shares is not None and shares > 0:
                prices.append(price * shares)
                sizes.append(shares)
        if sizes:
            filled = max(filled, sum(sizes))
            avg_fill_price = sum(prices) / sum(sizes)
    return {
        "status": status,
        "filled_quantity": filled,
        "remaining_quantity": remaining,
        "avg_fill_price": avg_fill_price,
        "fill_count": len(fills),
        "filled": bool(str(status).lower() == "filled" or filled > 0),
        "cancel_requested": False,
    }


def _float_or_zero(value: Any) -> float:
    out = _float_or_none(value)
    return 0.0 if out is None else out


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


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
