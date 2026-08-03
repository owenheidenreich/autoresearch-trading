"""Read-only IBKR paper compatibility adapter for the Path-D candidate.

This module has qualification and local order-construction capability only.  It
does not expose a submission operation and accepts paper accounts exclusively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v4.path_d.contracts import BrokerStateSnapshotV1, ExecutionIntentV1
from v4.path_d.risk.governor import DeterministicGovernor, FeedHealthV1


class IBKRPaperDryRunError(RuntimeError):
    """A paper/read-only compatibility invariant failed."""


@dataclass(frozen=True)
class IBKRPreviewV1:
    status: str
    account_id_redacted: str
    intent_id: str
    governor_disposition: str
    governor_reason_codes: tuple[str, ...]
    qualified_contract_count: int
    contract_preview: dict[str, Any]
    order_preview: dict[str, Any]
    broker_submit_endpoint_called: bool = False
    paper_order_submitted: bool = False


def _redact(account_id: str) -> str:
    return account_id[:2] + "…" + account_id[-2:]


def _require_paper_account(account_id: str) -> None:
    if not isinstance(account_id, str) or not account_id.startswith("DU") or len(account_id) < 4:
        raise IBKRPaperDryRunError("Path-D dry-run requires an authenticated DU paper account")


def _contract(option_cls: Any, intent: ExecutionIntentV1) -> Any:
    identity = intent.contract
    return option_cls(
        "SPX",
        identity.expiry.replace("-", ""),
        identity.strike_milli / 1_000.0,
        identity.right,
        "SMART",
        multiplier="100",
        currency="USD",
        tradingClass="SPXW",
    )


def _order(order_cls: Any, intent: ExecutionIntentV1, account_id: str) -> Any:
    return order_cls(
        intent.decision.side,
        intent.decision.quantity,
        intent.price_budget.hard_limit_micros / 1_000_000.0,
        tif="DAY",
        outsideRth=False,
        account=account_id,
    )


def _preview(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: getattr(value, name, None) for name in names}


def qualify_and_preview(
    *,
    ib: Any,
    option_cls: Any,
    order_cls: Any,
    account_id: str,
    intent: ExecutionIntentV1,
    broker_state: BrokerStateSnapshotV1,
    feed_health: FeedHealthV1,
    now_utc: str,
    governor: DeterministicGovernor | None = None,
) -> IBKRPreviewV1:
    """Qualify one exact SPXW contract and build a governed local order preview."""

    _require_paper_account(account_id)
    if broker_state.source != "IBKR" or broker_state.connectivity != "CONNECTED":
        raise IBKRPaperDryRunError("connected IBKR broker state is required")
    decision = (governor or DeterministicGovernor()).evaluate(
        intent, broker_state=broker_state, feed_health=feed_health, now_utc=now_utc
    )
    contract = _contract(option_cls, intent)
    qualified = list(ib.qualifyContracts(contract) or [])
    if len(qualified) != 1:
        raise IBKRPaperDryRunError(
            f"exactly one IBKR SPXW qualification required, observed {len(qualified)}"
        )
    contract = qualified[0]
    local_symbol = str(getattr(contract, "localSymbol", "") or "")
    if local_symbol and local_symbol != intent.contract.osi_symbol:
        raise IBKRPaperDryRunError("qualified IBKR localSymbol does not match raw OSI identity")
    order = _order(order_cls, intent, account_id)
    contract_preview = _preview(
        contract,
        (
            "conId", "symbol", "lastTradeDateOrContractMonth", "strike", "right",
            "exchange", "currency", "tradingClass", "localSymbol", "multiplier",
        ),
    )
    order_preview = _preview(
        order, ("action", "totalQuantity", "orderType", "lmtPrice", "tif", "outsideRth", "account")
    )
    if order_preview["action"] != intent.decision.side:
        raise IBKRPaperDryRunError("IBKR order preview side drift")
    if int(order_preview["totalQuantity"]) != intent.decision.quantity:
        raise IBKRPaperDryRunError("IBKR order preview quantity drift")
    expected_limit = intent.price_budget.hard_limit_micros / 1_000_000.0
    if abs(float(order_preview["lmtPrice"]) - expected_limit) > 1e-9:
        raise IBKRPaperDryRunError("IBKR order preview limit drift")
    return IBKRPreviewV1(
        status="DRY_RUN_PASS" if decision.disposition == "ALLOW" else "GOVERNOR_BLOCKED",
        account_id_redacted=_redact(account_id),
        intent_id=intent.intent_id,
        governor_disposition=decision.disposition,
        governor_reason_codes=decision.reason_codes,
        qualified_contract_count=1,
        contract_preview=contract_preview,
        order_preview=order_preview,
    )


__all__ = [
    "IBKRPaperDryRunError",
    "IBKRPreviewV1",
    "qualify_and_preview",
]
