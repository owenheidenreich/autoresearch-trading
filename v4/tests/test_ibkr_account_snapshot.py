from __future__ import annotations

import json
from dataclasses import dataclass

from v4.ops.ibkr.ibkr_account_snapshot import fetch_ibkr_account_snapshot, parse_account_summary_rows


@dataclass
class AccountValue:
    tag: str
    value: str
    currency: str = "USD"
    account: str = "DU123456"


def test_parse_account_summary_rows_redacts_account_and_extracts_values() -> None:
    snapshot = parse_account_summary_rows(
        [
            AccountValue("NetLiquidation", "26745.12"),
            AccountValue("TotalCashValue", "25100.25"),
            AccountValue("AvailableFunds", "24500.00"),
            AccountValue("BuyingPower", "98000.00"),
            AccountValue("RealizedPnL", "-30.00"),
            AccountValue("UnrealizedPnL", "12.50"),
        ],
        account_id="DU123456",
        checked_at_utc="2026-05-26T20:05:00+00:00",
    )

    assert snapshot["status"] == "pass"
    assert snapshot["account_id_redacted"] == "DU***56"
    assert snapshot["paper_account_confirmed"] is True
    assert snapshot["broker_order_endpoint_called"] is False
    assert snapshot["real_money_trading"] is False
    assert snapshot["raw_account_id_logged"] is False
    assert snapshot["values"]["net_liquidation"] == 26745.12
    assert snapshot["values"]["cash"] == 25100.25
    assert snapshot["values"]["realized_pnl"] == -30.0
    assert "DU123456" not in json.dumps(snapshot)


def test_fetch_ibkr_account_snapshot_uses_fake_ib_without_order_endpoint() -> None:
    class FakeIB:
        def __init__(self) -> None:
            self.connected = False

        def connect(self, host: str, port: int, *, clientId: int, timeout: float) -> None:
            self.connected = True

        def managedAccounts(self) -> list[str]:
            return ["DU123456"]

        def accountSummary(self, *, account: str) -> list[AccountValue]:
            assert account == "DU123456"
            return [
                AccountValue("NetLiquidation", "26,745.12"),
                AccountValue("TotalCashValue", "25,100.25"),
                AccountValue("RealizedPnL", "-30.00"),
            ]

        def accountValues(self, *, account: str) -> list[AccountValue]:
            return []

        def isConnected(self) -> bool:
            return self.connected

        def disconnect(self) -> None:
            self.connected = False

    snapshot = fetch_ibkr_account_snapshot(
        host="127.0.0.1",
        ports=[4002],
        client_id=257,
        ib_factory=FakeIB,
        require_socket=False,
    )

    assert snapshot["status"] == "pass"
    assert snapshot["port"] == 4002
    assert snapshot["values"]["net_liquidation"] == 26745.12
    assert snapshot["broker_order_endpoint_called"] is False
    assert "DU123456" not in json.dumps(snapshot)
