#!/usr/bin/env python3
"""
IB Gateway connectivity probe.

Tests which market data is available via IBKR:
  1. ES mini futures (continuous front-month) — 5-min bars
  2. I:SPX cash index — 5-min bars
  3. SPXW 0DTE option chains — contract details

Usage:
  python3 ib_probe.py                  # default localhost:4001
  python3 ib_probe.py --port 4002      # paper trading port
  python3 ib_probe.py --host 192.168.1.5 --port 7496  # remote TWS
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

try:
    from ib_insync import IB, Contract, Future, Index, util
except ImportError:
    print("ERROR: pip install ib_insync")
    sys.exit(1)


def probe(host: str = "127.0.0.1", port: int = 4001, client_id: int = 99):
    ib = IB()
    ib_errors: list[tuple[int, str]] = []

    def _on_error(reqId, errorCode, errorString, contract):
        msg = str(errorString or "")
        ib_errors.append((int(errorCode), msg))

    ib.errorEvent += _on_error

    # --- Connect ---
    print(f"Connecting to IB Gateway at {host}:{port} ...")
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
    except Exception as e:
        print(f"FAILED: {e}")
        print("\nMake sure IB Gateway or TWS is running.")
        print("  Gateway live: port 4001")
        print("  Gateway paper: port 4002")
        print("  TWS live: port 7496")
        print("  TWS paper: port 7497")
        return False

    print(f"Connected: {ib.managedAccounts()}")
    results = {}

    # --- Test 1: ES mini futures ---
    print("\n--- Test 1: ES Mini Futures ---")
    try:
        es = Future("ES", exchange="CME", currency="USD")
        contracts = ib.reqContractDetails(es)
        if contracts:
            front = contracts[0].contract
            print(f"  Contract: {front.localSymbol} (expiry {front.lastTradeDateOrContractMonth})")
            bars = ib.reqHistoricalData(
                front,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=True,
            )
            if bars:
                print(f"  Got {len(bars)} 5-min bars")
                print(f"  Last bar: {bars[-1].date} O={bars[-1].open} H={bars[-1].high} L={bars[-1].low} C={bars[-1].close}")
                results["es"] = True
            else:
                print("  No bars returned (market data subscription may be needed)")
                results["es"] = False
        else:
            print("  No ES contracts found")
            results["es"] = False
    except Exception as e:
        print(f"  FAILED: {e}")
        results["es"] = False

    # --- Test 2: SPX Cash Index ---
    print("\n--- Test 2: SPX Cash Index ---")
    try:
        spx = Index("SPX", exchange="CBOE", currency="USD")
        ib.qualifyContracts(spx)
        bars = ib.reqHistoricalData(
            spx,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=True,
        )
        if bars:
            print(f"  Got {len(bars)} 5-min bars")
            print(f"  Last bar: {bars[-1].date} O={bars[-1].open} C={bars[-1].close}")
            results["spx"] = True
        else:
            print("  No bars returned")
            results["spx"] = False
    except Exception as e:
        print(f"  FAILED: {e}")
        results["spx"] = False

    # --- Test 3: SPXW 0DTE Option Chain ---
    print("\n--- Test 3: SPXW 0DTE Options ---")
    try:
        spx_for_opts = Index("SPX", exchange="CBOE", currency="USD")
        ib.qualifyContracts(spx_for_opts)
        chains = ib.reqSecDefOptParams(spx_for_opts.symbol, "", spx_for_opts.secType, spx_for_opts.conId)
        if chains:
            # Find SPXW (weekly) chain
            spxw_chain = [c for c in chains if "SPXW" in (c.tradingClass or "")]
            if spxw_chain:
                chain = spxw_chain[0]
                today = datetime.now().strftime("%Y%m%d")
                has_today = today in chain.expirations
                print(f"  SPXW chain found: {chain.exchange}")
                print(f"  Expirations: {len(chain.expirations)} dates")
                print(f"  Strikes: {len(chain.strikes)} strikes")
                print(f"  Today ({today}) available: {has_today}")
                results["spxw"] = True
            else:
                # Check non-weekly SPX
                spx_chain = [c for c in chains if c.tradingClass == "SPX"]
                if spx_chain:
                    print(f"  SPX (monthly) chain found, but no SPXW (weekly)")
                    results["spxw"] = False
                else:
                    print(f"  No SPX option chains found")
                    results["spxw"] = False
        else:
            print("  No option chain data")
            results["spxw"] = False
    except Exception as e:
        print(f"  FAILED: {e}")
        results["spxw"] = False

    # --- Summary ---
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for name, ok in results.items():
        status = "OK" if ok else "UNAVAILABLE"
        print(f"  {name:10s}: {status}")

    has_ip_conflict = any(
        code == 162 and "different IP address" in msg
        for code, msg in ib_errors
    )
    has_subscription_error = any(code in (354, 10089, 10090) for code, _ in ib_errors)

    if has_ip_conflict:
        print("\nDetected IBKR session/IP conflict (Error 162).")
        print("Historical requests are blocked because another trading session is active on a different IP.")
        print("Fix: log out all other IBKR sessions/devices, restart Gateway, reconnect from one machine/network.")
    elif has_subscription_error:
        print("\nDetected market-data subscription/API entitlement errors.")
        print("Confirm SPY/SPX/VIX/SPXW packages + Market Data API Acknowledgement are active.")
    elif results.get("es"):
        print("\nRecommendation: Use ES mini futures as primary data source.")
    elif results.get("spx"):
        print("\nRecommendation: Use I:SPX cash index (ES not available).")
    else:
        print("\nNo SPX-level data available. Re-run during market hours and re-check entitlements.")

    ib.disconnect()
    try:
        ib.errorEvent -= _on_error
    except Exception:
        pass
    return all(results.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IB Gateway connectivity probe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--client-id", type=int, default=99)
    args = parser.parse_args()

    ok = probe(args.host, args.port, args.client_id)
    sys.exit(0 if ok else 1)
