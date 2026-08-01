"""Independent recorder for Protocol101 IBKR parity evidence.

This process deliberately imports only the standard library plus ib_insync.
It does not import Protocol101, torch, training code, or any order executor.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, time as clock_time, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

from v4.live.ibkr_market_capture import CapturePaths, CaptureWriter, clean_json, iso_utc, option_snapshot_delta


NY = ZoneInfo("America/New_York")
UTC = timezone.utc
MARKET_OPEN = clock_time(9, 30)
MARKET_CLOSE = clock_time(16, 0)
MARKET_DATA_TYPES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed_frozen"}
SUBSCRIPTION_ERROR_CODES = {354, 10089, 10167, 10168}
STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--capture-id", default=None)
    parser.add_argument("--capture-root", type=Path, default=Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--client-id", type=int, default=159)
    parser.add_argument("--strikes-around-atm", type=int, default=10)
    parser.add_argument(
        "--max-option-subscriptions",
        type=int,
        default=90,
        help=(
            "Retain contracts from prior ladder revisions up to this cap so an "
            "offline lifecycle path does not disappear as ATM moves."
        ),
    )
    parser.add_argument(
        "--option-generic-ticks",
        default="100,101",
        help=(
            "Diagnostic option generic ticks. IBKR 100/101 expose daily option "
            "volume/open interest and are preserved raw; they are not treated as "
            "equivalent to Databento one-minute OHLCV/statistics."
        ),
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--ladder-refresh-seconds", type=float, default=30.0)
    parser.add_argument("--reconnect-max-seconds", type=float, default=30.0)
    parser.add_argument("--stop-time-et", default="16:05")
    parser.add_argument("--max-runtime-seconds", type=float, default=None)
    parser.add_argument("--preflight-out", type=Path, default=None)
    return parser.parse_args()


def _signal_handler(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def source_timestamp(ticker: Any) -> str | None:
    for name in ("time", "rtTime", "timestamp"):
        value = getattr(ticker, name, None)
        if isinstance(value, datetime):
            return iso_utc(value)
        if value:
            return str(value)
    return None


def market_data_type(ticker: Any) -> dict[str, Any]:
    value = getattr(ticker, "marketDataType", None)
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = None
    return {"market_data_type": number, "market_data_type_name": MARKET_DATA_TYPES.get(number, "unknown")}


def ticker_price(ticker: Any) -> float | None:
    for name in ("last", "close"):
        value = finite(getattr(ticker, name, None))
        if value is not None and value > 0:
            return value
    bid = finite(getattr(ticker, "bid", None))
    ask = finite(getattr(ticker, "ask", None))
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    try:
        value = finite(ticker.marketPrice())
    except Exception:
        value = None
    return value if value is not None and value > 0 else None


def greek_payload(greeks: Any) -> dict[str, Any] | None:
    if greeks is None:
        return None
    return {
        "implied_vol": finite(getattr(greeks, "impliedVol", None)),
        "delta": finite(getattr(greeks, "delta", None)),
        "gamma": finite(getattr(greeks, "gamma", None)),
        "vega": finite(getattr(greeks, "vega", None)),
        "theta": finite(getattr(greeks, "theta", None)),
        "option_price": finite(getattr(greeks, "optPrice", None)),
        "underlying_price": finite(getattr(greeks, "undPrice", None)),
    }


def index_snapshot(symbol: str, ticker: Any) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "price": ticker_price(ticker),
        "bid": finite(getattr(ticker, "bid", None)),
        "ask": finite(getattr(ticker, "ask", None)),
        "last": finite(getattr(ticker, "last", None)),
        "close": finite(getattr(ticker, "close", None)),
        "source_timestamp_utc": source_timestamp(ticker),
        **market_data_type(ticker),
    }


def contract_payload(contract: Any) -> dict[str, Any]:
    return {
        "contract_id": contract_id(contract),
        "conid": int(getattr(contract, "conId", 0) or 0),
        "symbol": str(getattr(contract, "symbol", "SPX")),
        "local_symbol": str(getattr(contract, "localSymbol", "")),
        "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", "")),
        "strike": finite(getattr(contract, "strike", None)),
        "right": str(getattr(contract, "right", "")),
        "exchange": str(getattr(contract, "exchange", "SMART")),
        "primary_exchange": str(getattr(contract, "primaryExchange", "")),
        "currency": str(getattr(contract, "currency", "USD")),
        "trading_class": str(getattr(contract, "tradingClass", "SPXW")),
        "multiplier": str(getattr(contract, "multiplier", "100")),
        "settlement": "PM",
    }


def option_snapshot(contract: Any, ticker: Any) -> dict[str, Any]:
    bid = finite(getattr(ticker, "bid", None))
    ask = finite(getattr(ticker, "ask", None))
    return {
        **contract_payload(contract),
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0 if bid is not None and ask is not None and ask >= bid else None,
        "bid_size": finite(getattr(ticker, "bidSize", None)),
        "ask_size": finite(getattr(ticker, "askSize", None)),
        "last": finite(getattr(ticker, "last", None)),
        "last_size": finite(getattr(ticker, "lastSize", None)),
        "volume": finite(getattr(ticker, "volume", None)),
        "open_interest_call": finite(getattr(ticker, "callOpenInterest", None)),
        "open_interest_put": finite(getattr(ticker, "putOpenInterest", None)),
        "source_timestamp_utc": source_timestamp(ticker),
        "bid_greeks": greek_payload(getattr(ticker, "bidGreeks", None)),
        "ask_greeks": greek_payload(getattr(ticker, "askGreeks", None)),
        "last_greeks": greek_payload(getattr(ticker, "lastGreeks", None)),
        "model_greeks": greek_payload(getattr(ticker, "modelGreeks", None)),
        **market_data_type(ticker),
    }


def contract_id(contract: Any) -> str:
    expiry = str(getattr(contract, "lastTradeDateOrContractMonth", ""))[:8]
    strike = finite(getattr(contract, "strike", 0.0)) or 0.0
    right = str(getattr(contract, "right", ""))
    return f"SPXW-{expiry}-{strike:09.3f}-{right}"


def round_to_5(value: float) -> int:
    return int(round(value / 5.0) * 5)


def connect(IB: Any, args: argparse.Namespace) -> tuple[Any | None, int | None, list[dict[str, Any]]]:
    """Connect without ib_insync's account/order synchronization.

    The recorder needs contract discovery and market data only.  ``IB.connect``
    also starts account, position, order, and execution synchronization; rapid
    watchdog recovery can leave those server-side requests outstanding and hit
    IBKR's account-summary limit.  Connecting the underlying client preserves
    the market-data API while avoiding that unrelated account traffic.
    """
    attempts: list[dict[str, Any]] = []
    for text in str(args.ports).split(","):
        try:
            port = int(text.strip())
        except ValueError:
            continue
        ib = IB()
        try:
            client_id = int(args.client_id)
            ib.wrapper.clientId = client_id
            ib.run(ib.client.connectAsync(args.host, port, client_id, timeout=8))
            attempts.append({"port": port, "status": "connected", "connection_mode": "market_data_only"})
            return ib, port, attempts
        except Exception as exc:
            attempts.append({"port": port, "status": "failed", "error": str(exc)})
            if ib.isConnected():
                ib.disconnect()
    return None, None, attempts


def request_index(ib: Any, Index: Any, symbol: str, *, timeout_seconds: float = 15.0) -> tuple[Any, Any]:
    contract = Index(symbol, "CBOE", "USD")
    qualified = ib.run(asyncio.wait_for(ib.qualifyContractsAsync(contract), timeout=timeout_seconds))
    contract = qualified[0] if qualified else contract
    return contract, ib.reqMktData(contract, "", False, False)


def discover_contracts(
    ib: Any,
    Option: Any,
    spx_contract: Any,
    spx: float,
    session: str,
    width: int,
    *,
    timeout_seconds: float = 15.0,
) -> tuple[list[Any], dict[str, Any]]:
    expiry = session.replace("-", "")
    chain = None
    chain_discovery_mode = "secdef_option_chain"
    try:
        chains = ib.run(
            asyncio.wait_for(
                ib.reqSecDefOptParamsAsync("SPX", "", "IND", int(spx_contract.conId)),
                timeout=timeout_seconds,
            )
        )
        matching = [
            item for item in chains
            if str(getattr(item, "tradingClass", "")) == "SPXW"
            and expiry in set(getattr(item, "expirations", ()))
        ]
        if not matching:
            return [], {
                "expiry": expiry,
                "blocked_reason": "no_spxw_0dte_option_chain",
                "chains_returned": len(chains),
            }
        chain = sorted(
            matching,
            key=lambda item: (
                0 if getattr(item, "exchange", "") == "SMART" else 1,
                str(getattr(item, "exchange", "")),
            ),
        )[0]
    except asyncio.TimeoutError:
        # Direct qualification preserves the exact static ladder when IBKR's
        # option-chain metadata service stalls but contract details remain live.
        chain_discovery_mode = "direct_contract_fallback_after_secdef_timeout"

    atm = round_to_5(spx)
    wanted = {float(atm + offset * 5) for offset in range(-width, width + 1)}
    available = (
        {float(value) for value in getattr(chain, "strikes", ())}
        if chain is not None
        else wanted
    )
    strikes = sorted(wanted & available)
    requested = [Option("SPX", expiry, strike, right, "SMART", currency="USD", tradingClass="SPXW") for strike in strikes for right in ("C", "P")]
    qualified = list(
        ib.run(asyncio.wait_for(ib.qualifyContractsAsync(*requested), timeout=timeout_seconds))
    ) if requested else []
    return qualified, {
        "expiry": expiry,
        "atm_strike": atm,
        "strikes": strikes,
        "requested_contracts": len(requested),
        "qualified_contracts": len(qualified),
        "chain_exchange": str(getattr(chain, "exchange", "")) if chain is not None else "SMART",
        "chain_discovery_mode": chain_discovery_mode,
    }


def wait_for_price(ib: Any, ticker: Any, seconds: float = 20.0) -> float | None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = ticker_price(ticker)
        if value is not None:
            return value
        ib.sleep(0.25)
    return ticker_price(ticker)


def stop_deadline(args: argparse.Namespace) -> datetime:
    hour, minute = (int(value) for value in str(args.stop_time_et).split(":", 1))
    day = datetime.strptime(args.session, "%Y-%m-%d").date()
    return datetime.combine(day, clock_time(hour, minute), tzinfo=NY)


def in_regular_session(now: datetime) -> bool:
    local = now.astimezone(NY)
    return MARKET_OPEN <= local.time().replace(tzinfo=None) < MARKET_CLOSE


def completed_minute_label(now: datetime) -> str | None:
    local = now.astimezone(NY)
    if local.time().replace(tzinfo=None) <= MARKET_OPEN:
        return None
    completed = local.replace(second=0, microsecond=0)
    completed = completed - timedelta(minutes=1)
    if completed.time().replace(tzinfo=None) < MARKET_OPEN or completed.time().replace(tzinfo=None) >= MARKET_CLOSE:
        return None
    return completed.isoformat()


def run_preflight(args: argparse.Namespace, IB: Any, Index: Any, Option: Any) -> int:
    payload: dict[str, Any] = {
        "schema_version": "Protocol101RecorderPreflightV1",
        "session": args.session,
        "checked_at_utc": iso_utc(),
        "requested_market_data_type": "live",
        "broker_order_endpoint_called": False,
        "real_money_trading": False,
    }
    ib, port, attempts = connect(IB, args)
    payload["connection_attempts"] = attempts
    payload["port"] = port
    if ib is None:
        payload.update({"status": "fail", "reason": "ibkr_api_connection_failed"})
    else:
        errors: list[dict[str, Any]] = []
        ib.errorEvent += lambda req_id, code, message, contract=None: errors.append({"request_id": req_id, "code": code, "message": message})
        try:
            ib.reqMarketDataType(1)
            spx_contract, spx_ticker = request_index(ib, Index, "SPX")
            _, vix_ticker = request_index(ib, Index, "VIX")
            ib.sleep(3.0)
            spx = wait_for_price(ib, spx_ticker, seconds=10.0)
            contracts, chain = discover_contracts(ib, Option, spx_contract, spx or 0.0, args.session, int(args.strikes_around_atm)) if spx else ([], {})
            vix = ticker_price(vix_ticker)
            feed_types = {market_data_type(spx_ticker)["market_data_type_name"], market_data_type(vix_ticker)["market_data_type_name"]}
            payload.update({
                "status": "pass" if spx and vix and contracts and feed_types == {"live"} and not any(int(item.get("code", 0)) in SUBSCRIPTION_ERROR_CODES for item in errors) else "fail",
                "spx": index_snapshot("SPX", spx_ticker),
                "vix": index_snapshot("VIX", vix_ticker),
                "chain": chain,
                "qualified_contracts": len(contracts),
                "errors": errors,
            })
            if payload["status"] != "pass":
                payload["reason"] = "live_entitlement_or_chain_preflight_failed"
        finally:
            ib.disconnect()
    assert args.preflight_out is not None
    args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_out.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(clean_json(payload), indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


def run_recorder(args: argparse.Namespace, IB: Any, Index: Any, Option: Any) -> int:
    capture_id = args.capture_id or f"protocol101-recorder-{args.session}"
    paths = CapturePaths.for_capture(args.capture_root, args.session, capture_id)
    started = time.monotonic()
    stop_at = stop_deadline(args)
    reconnect_sleep = 1.0
    last_heartbeat = 0.0
    last_ladder_refresh = 0.0
    last_checkpoint: str | None = None
    ib = None
    option_tickers: dict[str, tuple[Any, Any]] = {}
    option_received: dict[str, str] = {}
    option_last: dict[str, dict[str, Any]] = {}
    spx_contract = spx_ticker = vix_ticker = None
    current_atm: int | None = None
    live_feed_confirmed = False
    subscription_errors = 0

    with CaptureWriter(paths, session=args.session, capture_id=capture_id) as writer:
        writer.append(
            "recorder_started",
            {
                "client_id": args.client_id,
                "requested_market_data_type": "live",
                "option_generic_ticks": str(args.option_generic_ticks),
                "option_generic_tick_semantics": "diagnostic_daily_aggregate_not_databento_minute_equivalent",
                "orders_enabled": False,
            },
            flush_to_disk=True,
        )

        def record_error(req_id: Any, code: Any, message: Any, contract: Any = None) -> None:
            nonlocal subscription_errors
            try:
                code_number = int(code)
            except (TypeError, ValueError):
                code_number = 0
            if code_number in SUBSCRIPTION_ERROR_CODES:
                subscription_errors += 1
            writer.append("ibkr_error", {"request_id": req_id, "code": code_number, "message": str(message), "contract": contract_payload(contract) if contract is not None and hasattr(contract, "secType") else str(contract) if contract else None})

        def disconnect() -> None:
            nonlocal ib, option_tickers, option_received, option_last, spx_contract, spx_ticker, vix_ticker
            if ib is not None and ib.isConnected():
                for _, (contract, _) in option_tickers.items():
                    try:
                        ib.cancelMktData(contract)
                    except Exception:
                        pass
                ib.disconnect()
            ib = None
            option_tickers = {}
            option_received = {}
            option_last = {}
            spx_contract = spx_ticker = vix_ticker = None

        def install_connection() -> bool:
            nonlocal ib, spx_contract, spx_ticker, vix_ticker, reconnect_sleep
            ib, port, attempts = connect(IB, args)
            if ib is None:
                writer.append("connection_failed", {"attempts": attempts})
                return False
            writer.new_connection_epoch()
            writer.append("connection", {"status": "connected", "port": port, "attempts": attempts}, flush_to_disk=True)
            ib.errorEvent += record_error
            ib.disconnectedEvent += lambda: writer.append("disconnect", {"reason": "ibkr_disconnected_event"}, flush_to_disk=True)
            ib.reqMarketDataType(1)
            spx_contract, spx_ticker = request_index(ib, Index, "SPX")
            _, vix_ticker = request_index(ib, Index, "VIX")
            spx_ticker.updateEvent += lambda ticker: writer.append("index_update", index_snapshot("SPX", ticker), event_timestamp_utc=source_timestamp(ticker))
            vix_ticker.updateEvent += lambda ticker: writer.append("index_update", index_snapshot("VIX", ticker), event_timestamp_utc=source_timestamp(ticker))
            writer.append("subscription", {"kind": "index", "symbols": ["SPX", "VIX"], "requested_market_data_type": "live"})
            reconnect_sleep = 1.0
            return True

        def refresh_ladder(spx: float) -> bool:
            nonlocal option_tickers, option_received, option_last, current_atm, last_ladder_refresh
            contracts, metadata = discover_contracts(ib, Option, spx_contract, spx, args.session, int(args.strikes_around_atm))
            if not contracts:
                writer.append("ladder_error", metadata, flush_to_disk=True)
                return False
            definitions = [contract_payload(contract) for contract in contracts]
            writer.append(
                "ladder_definition",
                {
                    **metadata,
                    "contracts": definitions,
                    "option_generic_ticks": str(args.option_generic_ticks),
                },
                flush_to_disk=True,
            )
            new_tickers: dict[str, tuple[Any, Any]] = dict(option_tickers)
            for contract in contracts:
                key = contract_id(contract)
                if key in option_tickers:
                    new_tickers[key] = option_tickers[key]
                    continue
                ticker = ib.reqMktData(contract, str(args.option_generic_ticks), False, False)
                def record_option_update(updated: Any, selected: Any = contract, selected_key: str = key) -> None:
                    snapshot = option_snapshot(selected, updated)
                    delta = option_snapshot_delta(option_last.get(selected_key), snapshot)
                    changes = delta.get("changes")
                    if isinstance(changes, dict) and not changes:
                        return
                    written = writer.append(
                        "option_update" if selected_key not in option_last else "option_delta",
                        delta,
                        event_timestamp_utc=source_timestamp(updated),
                    )
                    option_last[selected_key] = snapshot
                    option_received[selected_key] = str(written["received_timestamp_utc"])
                ticker.updateEvent += record_option_update
                new_tickers[key] = (contract, ticker)
            if len(new_tickers) > int(args.max_option_subscriptions):
                keep = set(
                    sorted(
                        new_tickers,
                        key=lambda key: abs(
                            float(getattr(new_tickers[key][0], "strike", current_atm or 0.0))
                            - float(round_to_5(spx))
                        ),
                    )[: int(args.max_option_subscriptions)]
                )
            else:
                keep = set(new_tickers)
            for key, (contract, _) in list(new_tickers.items()):
                if key not in keep:
                    try:
                        ib.cancelMktData(contract)
                    except Exception:
                        pass
                    del new_tickers[key]
            option_tickers = new_tickers
            option_received = {key: value for key, value in option_received.items() if key in new_tickers}
            option_last = {key: value for key, value in option_last.items() if key in new_tickers}
            current_atm = round_to_5(spx)
            last_ladder_refresh = time.monotonic()
            return True

        try:
            while not STOP_REQUESTED:
                now = datetime.now(UTC)
                if now.astimezone(NY) >= stop_at:
                    break
                if args.max_runtime_seconds is not None and time.monotonic() - started >= float(args.max_runtime_seconds):
                    break
                if ib is None or not ib.isConnected():
                    disconnect()
                    if not install_connection():
                        writer.write_state(status="reconnecting", subscription_errors=subscription_errors)
                        time.sleep(reconnect_sleep)
                        reconnect_sleep = min(float(args.reconnect_max_seconds), reconnect_sleep * 2.0)
                        continue
                assert ib is not None
                ib.sleep(0.2)
                spx = ticker_price(spx_ticker)
                vix = ticker_price(vix_ticker)
                if spx is not None and vix is not None:
                    types = {market_data_type(spx_ticker)["market_data_type_name"], market_data_type(vix_ticker)["market_data_type_name"]}
                    live_feed_confirmed = live_feed_confirmed or types == {"live"}
                if spx is not None and (
                    not option_tickers
                    or (
                        current_atm != round_to_5(spx)
                        and time.monotonic() - last_ladder_refresh >= float(args.ladder_refresh_seconds)
                    )
                ):
                    refresh_ladder(spx)

                checkpoint = completed_minute_label(now)
                if checkpoint is not None and checkpoint != last_checkpoint:
                    quotes = []
                    for key, (contract, ticker) in option_tickers.items():
                        quote = option_snapshot(contract, ticker)
                        quote["last_received_timestamp_utc"] = option_received.get(key)
                        quotes.append(quote)
                    writer.append(
                        "ladder_checkpoint",
                        {
                            "completed_minute_et": checkpoint,
                            "decision_time_et": (datetime.fromisoformat(checkpoint) + timedelta(minutes=1)).isoformat(),
                            "spx": index_snapshot("SPX", spx_ticker),
                            "vix": index_snapshot("VIX", vix_ticker),
                            "atm_strike": current_atm,
                            "contracts": quotes,
                            "contract_count": len(quotes),
                            "opening_context_policy": "09:30_completed_candle_feeds_09:31_decision",
                        },
                        flush_to_disk=True,
                    )
                    last_checkpoint = checkpoint

                if time.monotonic() - last_heartbeat >= float(args.heartbeat_seconds):
                    last_heartbeat = time.monotonic()
                    writer.append(
                        "heartbeat",
                        {
                            "connected": bool(ib.isConnected()),
                            "spx": spx,
                            "vix": vix,
                            "subscribed_contracts": len(option_tickers),
                            "max_option_subscriptions": int(args.max_option_subscriptions),
                            "ladder_atm": current_atm,
                            "live_feed_confirmed": live_feed_confirmed,
                            "subscription_errors": subscription_errors,
                            "market_open": in_regular_session(now),
                        },
                        flush_to_disk=True,
                    )
                    writer.write_state(
                        connected=bool(ib.isConnected()),
                        spx=spx,
                        vix=vix,
                        subscribed_contracts=len(option_tickers),
                        ladder_atm=current_atm,
                        live_feed_confirmed=live_feed_confirmed,
                        subscription_errors=subscription_errors,
                        last_checkpoint_minute_et=last_checkpoint,
                    )
        except Exception as exc:
            writer.append("recorder_error", {"error_type": type(exc).__name__, "error": str(exc)}, flush_to_disk=True)
            writer.write_state(status="failed", error=str(exc), subscription_errors=subscription_errors)
            disconnect()
            return 2
        finally:
            disconnect()
        writer.append("recorder_stopped", {"reason": "signal" if STOP_REQUESTED else "scheduled_stop", "live_feed_confirmed": live_feed_confirmed}, flush_to_disk=True)
        writer.write_state(status="stopped", live_feed_confirmed=live_feed_confirmed, subscription_errors=subscription_errors, last_checkpoint_minute_et=last_checkpoint)
    return 0


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        from ib_insync import IB, Index, Option  # type: ignore
    except Exception as exc:
        print(json.dumps({"status": "fail", "reason": "ib_insync_import_failed", "error": str(exc)}), file=sys.stderr)
        return 2
    if args.preflight_out is not None:
        return run_preflight(args, IB, Index, Option)
    return run_recorder(args, IB, Index, Option)


if __name__ == "__main__":
    raise SystemExit(main())
