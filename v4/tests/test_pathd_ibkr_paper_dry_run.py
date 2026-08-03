from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

from v4.path_d.contracts import BrokerStateSnapshotV1, ExecutionIntentV1
from v4.path_d.risk.governor import FeedHealthV1
from v4.path_d.runtime import ibkr_paper_dry_run as adapter


FIXTURES = Path("v4/path_d/contracts/fixtures")


class Option:
    def __init__(
        self, symbol, expiry, strike, right, exchange, *, multiplier, currency, tradingClass
    ):
        self.conId = 123
        self.symbol = symbol
        self.lastTradeDateOrContractMonth = expiry
        self.strike = strike
        self.right = right
        self.exchange = exchange
        self.multiplier = multiplier
        self.currency = currency
        self.tradingClass = tradingClass
        self.localSymbol = "SPXW  260630C07490000"


class LimitOrder:
    def __init__(self, action, quantity, price, **kwargs):
        self.action = action
        self.totalQuantity = quantity
        self.orderType = "LMT"
        self.lmtPrice = price
        for key, value in kwargs.items():
            setattr(self, key, value)


class PaperReadOnlyIB:
    def __init__(self):
        self.qualified = 0

    def qualifyContracts(self, contract):
        self.qualified += 1
        return [contract]


def _inputs():
    intent = ExecutionIntentV1.from_dict(
        json.loads((FIXTURES / "execution_intent_v1.json").read_text())
    )
    base = BrokerStateSnapshotV1.from_dict(
        json.loads((FIXTURES / "broker_state_snapshot_v1.json").read_text())
    )
    state = replace(base, source="IBKR", account_id_redacted="DU…01")
    feed = FeedHealthV1(
        option_received_timestamp_utc="2026-06-30T13:30:02Z",
        spx_received_timestamp_utc="2026-06-30T13:30:01Z",
    )
    return intent, state, feed


def test_dry_run_qualifies_and_builds_preview_without_submit_surface() -> None:
    intent, state, feed = _inputs()
    ib = PaperReadOnlyIB()
    result = adapter.qualify_and_preview(
        ib=ib,
        option_cls=Option,
        order_cls=LimitOrder,
        account_id="DU123401",
        intent=intent,
        broker_state=state,
        feed_health=feed,
        now_utc="2026-06-30T13:30:02Z",
    )
    assert result.status == "DRY_RUN_PASS"
    assert result.order_preview["action"] == "SELL"
    assert result.order_preview["lmtPrice"] == 1.9
    assert result.broker_submit_endpoint_called is False
    assert result.paper_order_submitted is False
    assert ib.qualified == 1
    forbidden = "place" + "Order"
    assert forbidden not in inspect.getsource(adapter)


def test_dry_run_rejects_nonpaper_account() -> None:
    intent, state, feed = _inputs()
    with pytest.raises(adapter.IBKRPaperDryRunError, match="DU paper"):
        adapter.qualify_and_preview(
            ib=PaperReadOnlyIB(),
            option_cls=Option,
            order_cls=LimitOrder,
            account_id="U123401",
            intent=intent,
            broker_state=state,
            feed_health=feed,
            now_utc="2026-06-30T13:30:02Z",
        )
