from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

from ib_insync import IB, Index, Option, Stock

ET_TZ = ZoneInfo("America/New_York")


@dataclass
class SymbolEntitlement:
    symbol: str
    market_data_type: int
    has_bid_ask: bool
    has_last: bool
    has_option_greeks: bool
    delayed_tick_detected: bool
    stale: bool
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.market_data_type != 1:
            return False
        if self.symbol.startswith("SPXW_"):
            return self.has_option_greeks and (self.has_bid_ask or self.has_last)
        return self.has_bid_ask or self.has_last


@dataclass
class EntitlementReport:
    passed: bool
    created_at: str
    host: str
    port: int
    account: str | None
    symbols: dict[str, SymbolEntitlement]
    warnings: list[str] = field(default_factory=list)


def _is_finite(v: Any) -> bool:
    try:
        return v is not None and not math.isnan(float(v))
    except Exception:
        return False


def _sample_spx_price(ib: IB, spx: Index) -> float:
    t = ib.reqMktData(spx, "", False, False)
    ib.sleep(2.0)
    px = t.marketPrice()
    ib.cancelMktData(spx)
    return float(px) if _is_finite(px) else 5000.0


def probe_entitlements(
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 91,
    stale_seconds: int = 20,
    require_paper_account: bool = True,
    max_active_tickers_warn: int = 90,
) -> EntitlementReport:
    """Probe required IBKR market-data entitlements for live paper session.

    Fail-closed behavior:
      - marketDataType must be 1 (live, non-delayed)
      - options need top-of-book and Greeks
    """
    ib = IB()
    warnings: list[str] = []
    symbols: dict[str, SymbolEntitlement] = {}

    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
        ib.reqMarketDataType(1)  # force live request

        account = None
        managed = list(ib.managedAccounts() or [])
        if managed:
            account = managed[0]
            if require_paper_account and not str(account).startswith("DU"):
                warnings.append(f"Connected account {account} does not look like paper (DU*)")

        spx = Index("SPX", "CBOE", "USD")
        vix = Index("VIX", "CBOE", "USD")
        spy = Stock("SPY", "ARCA", "USD")
        ib.qualifyContracts(spx, vix, spy)

        spx_px = _sample_spx_price(ib, spx)
        expiry = dt.datetime.now(dt.timezone.utc).astimezone(ET_TZ).strftime("%Y%m%d")
        atm = round(spx_px / 5.0) * 5.0
        spxw_call = Option(
            symbol="SPX",
            lastTradeDateOrContractMonth=expiry,
            strike=float(atm),
            right="C",
            exchange="SMART",
            currency="USD",
            tradingClass="SPXW",
        )
        ib.qualifyContracts(spxw_call)

        probes = {
            "SPY": (spy, ""),
            "SPX": (spx, ""),
            "VIX": (vix, ""),
            "SPXW_ATM_CALL": (spxw_call, "100,101,104,106"),
        }

        ticker_by_name = {}
        for name, (contract, ticks) in probes.items():
            ticker_by_name[name] = ib.reqMktData(contract, ticks, False, False)
        ib.sleep(4.0)

        now = dt.datetime.now(dt.timezone.utc)
        for name, ticker in ticker_by_name.items():
            md_type = int(getattr(ticker, "marketDataType", 0) or 0)
            has_bid_ask = _is_finite(ticker.bid) and _is_finite(ticker.ask)
            has_last = _is_finite(ticker.last) or _is_finite(ticker.close)
            has_greeks = bool(
                ticker.bidGreeks is not None
                or ticker.askGreeks is not None
                or ticker.lastGreeks is not None
                or ticker.modelGreeks is not None
            )
            delayed_tick = any(66 <= int(getattr(t, "tickType", -1)) <= 76 for t in ticker.ticks)
            tick_time = getattr(ticker, "time", None)
            stale = True
            if tick_time is not None:
                if tick_time.tzinfo is None:
                    tick_time = tick_time.replace(tzinfo=dt.timezone.utc)
                stale = (now - tick_time).total_seconds() > stale_seconds

            symbols[name] = SymbolEntitlement(
                symbol=name,
                market_data_type=md_type,
                has_bid_ask=has_bid_ask,
                has_last=has_last,
                has_option_greeks=has_greeks,
                delayed_tick_detected=delayed_tick,
                stale=stale,
                notes=[],
            )

        for _, (contract, _) in probes.items():
            ib.cancelMktData(contract)

        active = len(getattr(ib.wrapper, "tickers", {}))
        if active > max_active_tickers_warn:
            warnings.append(
                f"High active ticker count ({active}) may approach market-data line limits"
            )

        passed = all(v.passed for v in symbols.values())
        return EntitlementReport(
            passed=passed,
            created_at=dt.datetime.utcnow().isoformat(),
            host=host,
            port=port,
            account=managed[0] if managed else None,
            symbols=symbols,
            warnings=warnings,
        )
    finally:
        if ib.isConnected():
            ib.disconnect()
