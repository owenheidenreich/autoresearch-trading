#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from ib_insync import IB


DEFAULT_TAGS = [
    "AccountType",
    "NetLiquidation",
    "TotalCashValue",
    "BuyingPower",
    "ExcessLiquidity",
    "AvailableFunds",
    "GrossPositionValue",
    "DayTradesRemainingT+0",
    "DayTradesRemainingT+1",
    "DayTradesRemainingT+2",
    "DayTradesRemainingT+3",
]


def _summary_dict(ib: IB, account: str) -> dict[str, Any]:
    rows = ib.accountSummary(account=account)
    out: dict[str, Any] = {}
    for row in rows:
        if row.tag not in DEFAULT_TAGS:
            continue
        key = row.tag if not row.currency else f"{row.tag}:{row.currency}"
        out[key] = row.value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch safe IBKR paper-account summary snapshot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=192)
    parser.add_argument("--account", default=None, help="Optional explicit account id (e.g. DUPxxxxxx)")
    args = parser.parse_args()

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
        managed = list(ib.managedAccounts() or [])
        account = args.account or (managed[0] if managed else None)
        payload: dict[str, Any] = {
            "connected": ib.isConnected(),
            "host": args.host,
            "port": args.port,
            "managed_accounts": managed,
            "account": account,
            "summary": {},
        }
        if account:
            payload["summary"] = _summary_dict(ib, account)
        print(json.dumps(payload, indent=2))
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()

