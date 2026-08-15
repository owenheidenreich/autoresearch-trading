"""No-order IBKR live market-data entitlement check for SPXW 0DTE.

This script is intentionally narrower than the trading/router code. It only
checks whether IBKR can deliver fresh live data needed by the Protocol 081
shadow router:

* SPX index L1
* VIX index L1
* SPXW 0DTE option NBBO for a small ATM-centered universe

It never constructs, stages, or submits broker orders.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from v4.scripts.run_protocol081_live_shadow_router import (
    IbkrErrorLog,
    _discover_spxw_0dte_contracts,
    _is_regular_market_hours,
    _market_data_type_name,
    _option_quote,
    _request_index_ticker,
    _ticker_market_data_type,
    _ticker_price,
    _wait_for_price,
    _connect_ibkr,
    _json_sanitize,
)


_NY = ZoneInfo("America/New_York")
DEFAULT_OUT_DIR = Path("v4/audit/ibkr_live_data_entitlements")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=7497)
    parser.add_argument("--ibkr-auto-ports", default="4002,7497,7496,4001")
    parser.add_argument("--ibkr-client-id", type=int, default=91)
    parser.add_argument("--strikes-around-atm", type=int, default=1)
    parser.add_argument("--wait-seconds", type=float, default=8.0)
    parser.add_argument("--allow-delayed-plumbing", action="store_true")
    return parser.parse_args()


def _status_for_index(ticker: Any, price: float | None) -> dict[str, Any]:
    market_data_type = _market_data_type_name(_ticker_market_data_type(ticker))
    return {
        "price": price,
        "market_data_type": market_data_type,
        "live_price_available": price is not None and market_data_type == "live",
    }


def _required_actions() -> list[dict[str, str]]:
    return [
        {
            "item": "Market Data API Acknowledgement",
            "why": "IBKR can reject API market data until this Client Portal acknowledgement is enabled.",
            "source": "https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/",
        },
        {
            "item": "Cboe Streaming Market Indexes",
            "why": "Needed for live SPX/VIX index L1 context.",
            "source": "https://www.interactivebrokers.com/en/pricing/market-data-pricing.php",
        },
        {
            "item": "OPRA Top of Book (L1)(US Option Exchanges)",
            "why": "Needed for live SPXW option NBBO.",
            "source": "https://www.interactivebrokers.com/en/pricing/market-data-pricing.php",
        },
        {
            "item": "Underlying/index plus derivative data",
            "why": "IBKR documents that options Greeks need both underlying and derivative subscriptions.",
            "source": "https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/",
        },
    ]


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# IBKR SPXW Live Market-Data Entitlement Check",
        "",
        "No orders were created or submitted.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Blocked reason: `{payload.get('blocked_reason')}`",
        f"- IBKR connected: `{payload.get('ibkr_connected')}`",
        f"- IBKR port: `{payload.get('ibkr_port')}`",
        f"- Regular market hours: `{payload.get('regular_market_hours')}`",
        "",
        "## Feed Status",
        "",
        "```json",
        json.dumps(_json_sanitize(payload.get("feed_status", {})), indent=2, sort_keys=True),
        "```",
        "",
        "## Required Actions",
        "",
    ]
    for action in payload["required_actions"]:
        lines.append(f"- `{action['item']}`: {action['why']} Source: {action['source']}")
    if payload.get("subscription_errors"):
        lines.extend(
            [
                "",
                "## Subscription Errors",
                "",
                "```json",
                json.dumps(_json_sanitize(payload["subscription_errors"]), indent=2, sort_keys=True),
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=_NY)
    base = {
        "checked_at": now.isoformat(),
        "regular_market_hours": _is_regular_market_hours(now),
        "decision": "blocked",
        "blocked_reason": None,
        "ibkr_connected": False,
        "ibkr_port": None,
        "required_actions": _required_actions(),
        "no_order_guarantee": {
            "broker_order_endpoint_called": False,
            "order_intent_non_null_rows": 0,
        },
    }
    try:
        from ib_insync import IB, Index, Option  # type: ignore
    except ImportError:
        payload = {**base, "blocked_reason": "missing_ib_insync"}
        _write_outputs(args.out_dir, payload)
        return 1

    ib = None
    error_log = IbkrErrorLog()
    subscribed: list[Any] = []
    try:
        ib, port, attempts = _connect_ibkr(args, IB)
        if ib is None:
            payload = {
                **base,
                "blocked_reason": "ibkr_connection_failed",
                "connection_attempts": attempts,
            }
            _write_outputs(args.out_dir, payload)
            return 1
        ib.errorEvent += error_log.handler
        ib.reqMarketDataType(3 if args.allow_delayed_plumbing else 1)

        spx_contract, spx_ticker = _request_index_ticker(ib, Index, "SPX")
        vix_contract, vix_ticker = _request_index_ticker(ib, Index, "VIX")
        subscribed.extend([spx_contract, vix_contract])
        spx_price = _wait_for_price(ib, spx_ticker, seconds=args.wait_seconds)
        vix_value = _wait_for_price(ib, vix_ticker, seconds=args.wait_seconds)

        feed_status: dict[str, Any] = {
            "spx": _status_for_index(spx_ticker, spx_price),
            "vix": _status_for_index(vix_ticker, vix_value),
            "spxw_options": {
                "contracts_requested": 0,
                "contracts_qualified": 0,
                "live_nbbo_rows": 0,
                "delayed_nbbo_rows": 0,
                "market_data_type_counts": {},
            },
        }

        chain_meta: dict[str, Any] = {}
        if spx_price is not None:
            contracts, chain_meta = _discover_spxw_0dte_contracts(
                ib=ib,
                option_cls=Option,
                spx_contract=spx_contract,
                spx_price=float(spx_price),
                now=now,
                strikes_around_atm=args.strikes_around_atm,
            )
            feed_status["spxw_options"]["contracts_requested"] = chain_meta.get("requested_contracts", 0)
            feed_status["spxw_options"]["contracts_qualified"] = len(contracts)
            option_tickers = []
            for contract in contracts:
                ticker = ib.reqMktData(contract, "", False, False)
                option_tickers.append(ticker)
                subscribed.append(contract)
            if option_tickers:
                ib.sleep(args.wait_seconds)
            type_counts: dict[str, int] = {}
            live_nbbo = 0
            delayed_nbbo = 0
            for ticker in option_tickers:
                type_name = _market_data_type_name(_ticker_market_data_type(ticker))
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                if _option_quote(ticker) is None:
                    continue
                if type_name == "live":
                    live_nbbo += 1
                elif "delayed" in type_name:
                    delayed_nbbo += 1
            feed_status["spxw_options"].update(
                {
                    "live_nbbo_rows": live_nbbo,
                    "delayed_nbbo_rows": delayed_nbbo,
                    "market_data_type_counts": type_counts,
                }
            )

        live_ready = (
            feed_status["spx"]["live_price_available"]
            and feed_status["vix"]["live_price_available"]
            and feed_status["spxw_options"]["live_nbbo_rows"] > 0
            and not error_log.subscription_errors
        )
        blocked_reason = None if live_ready else "missing_live_market_data_entitlements"
        payload = {
            **base,
            "decision": "pass" if live_ready else "blocked",
            "blocked_reason": blocked_reason,
            "ibkr_connected": True,
            "ibkr_port": port,
            "connection_attempts": attempts,
            "feed_status": feed_status,
            "chain_meta": chain_meta,
            "ibkr_errors": error_log.events[-30:],
            "subscription_errors": error_log.subscription_errors[-30:],
        }
        _write_outputs(args.out_dir, payload)
        print(
            json.dumps(
                {
                    "decision": payload["decision"],
                    "blocked_reason": payload["blocked_reason"],
                    "ibkr_port": payload["ibkr_port"],
                    "feed_status": payload["feed_status"],
                },
                sort_keys=True,
                default=str,
            )
        )
        return 0 if live_ready else 1
    finally:
        if ib is not None and ib.isConnected():
            for contract in subscribed:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass
            ib.disconnect()


def _write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    summary_path.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False, sort_keys=True) + "\n")
    _write_report(report_path, payload)
    print(report_path)


if __name__ == "__main__":
    raise SystemExit(main())
