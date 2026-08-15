from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from v4.foundation.execution_observation_contract import PASS, validate_observation
from v4.live.paper_trade_log import (
    load_trade_log,
    stable_json_hash,
    trade_log_path,
    validate_observability_contract,
    validate_trade_log,
)
from v4.scripts.run_tuesday_protocol101_paper_fill_observation import decide, execute_probe


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
    orderId: int | None = None
    permId: int | None = None


class FakeTicker:
    def __init__(self, *, bid: float, ask: float, bid_size: int = 12, ask_size: int = 11) -> None:
        self.bid = bid
        self.ask = ask
        self.bidSize = bid_size
        self.askSize = ask_size

    @property
    def time(self) -> datetime:
        return datetime.now(tz=timezone.utc)


class FilledPaperIB:
    def __init__(self) -> None:
        self.placed: list[tuple[FakeOption, FakeLimitOrder]] = []
        self._next_order_id = 1001

    def qualifyContracts(self, contract: FakeOption) -> list[FakeOption]:
        return [contract]

    def placeOrder(self, contract: FakeOption, order: FakeLimitOrder) -> SimpleNamespace:
        order.orderId = self._next_order_id
        order.permId = self._next_order_id + 10_000
        self._next_order_id += 1
        self.placed.append((contract, order))
        status = SimpleNamespace(
            status="Filled",
            filled=order.totalQuantity,
            remaining=0,
            avgFillPrice=order.lmtPrice,
        )
        return SimpleNamespace(contract=contract, order=order, orderStatus=status, fills=())

    def sleep(self, seconds: float) -> None:
        return None

    def cancelOrder(self, order: FakeLimitOrder) -> None:
        self.cancelled = order


def test_tuesday_paper_fill_observation_fake_broker_round_trip_passes_contracts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("V4_ALLOW_IBKR_PAPER_ORDERS", "YES")
    monkeypatch.setenv("TUESDAY_PAPER_FILL_OBSERVATIONS_APPROVED", "YES")
    session = datetime.now(tz=timezone.utc).date().isoformat()
    run_id = "fake_tuesday_paper_fill_e2e"
    trade_log_root = tmp_path / "paper_trading"
    trade_log = trade_log_path(root=trade_log_root, session=session, run_id=run_id)
    observation_log = tmp_path / "audit" / "execution_observations.jsonl"
    decision_time = datetime(2026, 5, 26, 14, 35, tzinfo=timezone.utc)
    contract_id = "SPXW-20260526-06700.000-C"
    quote_timestamp = "2026-05-26T14:34:59.900000+00:00"
    received_timestamp = decision_time.isoformat()
    quote = {
        "contract_id": contract_id,
        "symbol": "SPX",
        "expiry": "20260526",
        "strike": 6700.0,
        "right": "C",
        "trading_class": "SPXW",
        "settlement": "PM",
        "exchange": "SMART",
        "currency": "USD",
        "bid": 9.90,
        "ask": 10.00,
        "spread": 0.10,
        "bid_size": 12,
        "ask_size": 11,
        "quote_age_ms": 100,
        "quote_timestamp": quote_timestamp,
        "raw_quote_timestamp_utc": quote_timestamp,
        "received_timestamp": received_timestamp,
        "received_timestamp_utc": received_timestamp,
        "decision_timestamp": received_timestamp,
        "decision_timestamp_utc": received_timestamp,
    }
    candidate = pd.Series(
        {
            "contract_id": contract_id,
            "score": 42.0,
            "threshold": 0.0,
            "right": "C",
            "offset_points": 0.0,
        }
    )
    args = SimpleNamespace(
        quantity=1,
        paper_cash=10_000.0,
        enable_paper_orders=True,
        acknowledge_paper_loss=True,
        trade_log_root=trade_log_root,
        entry_timeout_seconds=0.0,
        exit_timeout_seconds=0.0,
        exit_force_offset=0.0,
    )
    artifact_ids = {"surface_manifest": "fake_surface", "protocol101_manifest": "fake_protocol101"}
    runtime_flag = {"passed": True, "source": "fake_broker_e2e"}

    observation = execute_probe(
        args=args,
        ib=FilledPaperIB(),
        option_cls=FakeOption,
        order_cls=FakeLimitOrder,
        account_id="DU12345",
        trade_log=trade_log,
        observation_log=observation_log,
        session=session,
        run_id=run_id,
        candidate=candidate,
        quote=quote,
        selection_reason="fake_broker_for_midday_e2e_validation",
        spx=6700.0,
        vix=16.0,
        decision_time=decision_time,
        probe_index=0,
        option_subscriptions=[
            (
                FakeOption("SPX", "20260526", 6700.0, "C", "SMART"),
                FakeTicker(bid=10.25, ask=10.35),
            )
        ],
        artifact_ids=artifact_ids,
        runtime_flag=runtime_flag,
        candidate_set_hash=stable_json_hash([{"contract_id": contract_id}]),
        feature_vector_hash=stable_json_hash([{"score": 42.0}]),
    )

    observation_rows = [json.loads(line) for line in observation_log.read_text().splitlines()]
    trade_rows = load_trade_log(trade_log)

    assert observation["fill_status"] == "filled"
    assert observation["exit_fill_status"] == "filled"
    assert observation["broker_order_endpoint_called"] is True
    assert observation["paper_order_submitted"] is True
    assert observation["open_position_risk"] is False
    assert decide([observation], broker_order_endpoint_called=True) == "paper_fill_observations_collected_for_execution_truth_review"
    assert len(observation_rows) == 1
    assert validate_observation(observation)["status"] == PASS
    assert validate_trade_log(trade_rows)["status"] == "pass"
    assert validate_observability_contract(trade_rows)["status"] == "pass"
