from __future__ import annotations

from dataclasses import dataclass

from v4.live.ibkr_paper_executor import execute_guarded_paper_order
from v4.live.ibkr_paper_guard import PaperOrderIntent
from v4.scripts.run_protocol142_ibkr_paper_executor_smoke import decide


@dataclass
class FakeOption:
    symbol: str
    lastTradeDateOrContractMonth: str
    strike: float
    right: str
    exchange: str
    currency: str = "USD"
    tradingClass: str = "SPXW"


@dataclass
class FakeLimitOrder:
    action: str
    totalQuantity: int
    lmtPrice: float
    tif: str = "DAY"
    outsideRth: bool = False


@dataclass
class FakeTrade:
    contract: FakeOption
    order: FakeLimitOrder


class FakeIB:
    def __init__(self) -> None:
        self.placed = []

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        return FakeTrade(contract, order)


def _intent() -> PaperOrderIntent:
    return PaperOrderIntent(
        action="BUY",
        symbol="SPX",
        expiry="20260320",
        strike=6700.0,
        right="C",
        quantity=1,
        limit_price=10.0,
    )


def test_executor_dry_run_validates_without_calling_place_order() -> None:
    ib = FakeIB()
    result = execute_guarded_paper_order(
        ib=ib,
        option_cls=FakeOption,
        order_cls=FakeLimitOrder,
        intent=_intent(),
        account_id="DU12345",
        account_cash=10_000.0,
        open_positions=0,
        quote={"bid": 9.9, "ask": 10.0, "reference_ask": 10.0, "quote_age_ms": 100},
        context={"context_age_ms": 100},
        enable_paper_orders=True,
        acknowledge_paper_loss=True,
        environ={"V4_ALLOW_IBKR_PAPER_ORDERS": "YES"},
        dry_run=True,
    )

    assert result["status"] == "dry_run_pass"
    assert result["broker_order_endpoint_called"] is False
    assert ib.placed == []
    assert decide("dry-run", result) == "pass_paper_executor_validates_without_order_submission"


def test_executor_writes_trade_log_when_log_root_is_supplied(tmp_path) -> None:
    ib = FakeIB()
    result = execute_guarded_paper_order(
        ib=ib,
        option_cls=FakeOption,
        order_cls=FakeLimitOrder,
        intent=_intent(),
        account_id="DU12345",
        account_cash=10_000.0,
        open_positions=0,
        quote={"bid": 9.9, "ask": 10.0, "reference_ask": 10.0, "quote_age_ms": 100},
        context={"context_age_ms": 100},
        enable_paper_orders=True,
        acknowledge_paper_loss=True,
        environ={"V4_ALLOW_IBKR_PAPER_ORDERS": "YES"},
        dry_run=True,
        trade_log_root=tmp_path,
        trade_log_run_id="test_executor",
    )

    assert result["status"] == "dry_run_pass"
    assert list(tmp_path.glob("*/test_executor.jsonl"))


def test_executor_blocks_without_paper_permission() -> None:
    ib = FakeIB()
    result = execute_guarded_paper_order(
        ib=ib,
        option_cls=FakeOption,
        order_cls=FakeLimitOrder,
        intent=_intent(),
        account_id="DU12345",
        account_cash=10_000.0,
        open_positions=0,
        quote={"bid": 9.9, "ask": 10.0, "reference_ask": 10.0, "quote_age_ms": 100},
        context={"context_age_ms": 100},
        enable_paper_orders=False,
        acknowledge_paper_loss=False,
        environ={},
        dry_run=False,
    )

    assert result["status"] == "blocked"
    assert result["broker_order_endpoint_called"] is False
    assert ib.placed == []


def test_executor_submit_calls_place_order_only_after_guard_passes() -> None:
    ib = FakeIB()
    result = execute_guarded_paper_order(
        ib=ib,
        option_cls=FakeOption,
        order_cls=FakeLimitOrder,
        intent=_intent(),
        account_id="DU12345",
        account_cash=10_000.0,
        open_positions=0,
        quote={"bid": 9.9, "ask": 10.0, "reference_ask": 10.0, "quote_age_ms": 100},
        context={"context_age_ms": 100},
        enable_paper_orders=True,
        acknowledge_paper_loss=True,
        environ={"V4_ALLOW_IBKR_PAPER_ORDERS": "YES"},
        dry_run=False,
    )

    assert result["status"] == "submitted"
    assert result["broker_order_endpoint_called"] is True
    assert len(ib.placed) == 1
