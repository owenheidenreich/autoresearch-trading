from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

from ib_insync import IB, Option

from training.prepare import (
    ACTION_BUY_CALL_ATM,
    ACTION_BUY_CALL_OTM5,
    ACTION_BUY_CALL_OTM10,
    ACTION_BUY_PUT_ATM,
    ACTION_BUY_PUT_OTM5,
    ACTION_BUY_PUT_OTM10,
)


@dataclass(frozen=True)
class ActionSpec:
    label: str
    right: str
    strike_offset: int


ACTION_TO_SPEC = {
    ACTION_BUY_CALL_ATM: ActionSpec("CALL_ATM", "C", 0),
    ACTION_BUY_CALL_OTM5: ActionSpec("CALL_OTM5", "C", +5),
    ACTION_BUY_CALL_OTM10: ActionSpec("CALL_OTM10", "C", +10),
    ACTION_BUY_PUT_ATM: ActionSpec("PUT_ATM", "P", 0),
    ACTION_BUY_PUT_OTM5: ActionSpec("PUT_OTM5", "P", -5),
    ACTION_BUY_PUT_OTM10: ActionSpec("PUT_OTM10", "P", -10),
}


class SPXWContractResolver:
    def __init__(self, ib: IB | None = None, auto_qualify: bool = True) -> None:
        self.ib = ib
        self.auto_qualify = auto_qualify

    def _expiry(self, when: dt.datetime | None = None) -> str:
        if when is None:
            when = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=-5)))
        return when.strftime("%Y%m%d")

    @staticmethod
    def _atm_strike(spx_price: float) -> float:
        return float(round(spx_price / 5.0) * 5.0)

    def resolve(self, action: int, spx_price: float, when: dt.datetime | None = None) -> Option:
        spec = ACTION_TO_SPEC.get(action)
        if spec is None:
            raise ValueError(f"Action {action} is not a tradeable SPXW action")
        strike = self._atm_strike(spx_price) + float(spec.strike_offset)
        contract = Option(
            symbol="SPX",
            lastTradeDateOrContractMonth=self._expiry(when),
            strike=float(strike),
            right=spec.right,
            exchange="SMART",
            currency="USD",
            tradingClass="SPXW",
        )
        if self.ib and self.ib.isConnected() and self.auto_qualify:
            self.ib.qualifyContracts(contract)
        return contract

    def resolve_all(self, spx_price: float, when: dt.datetime | None = None) -> dict[int, Option]:
        return {action: self.resolve(action, spx_price, when=when) for action in ACTION_TO_SPEC}

    def quote_mid(self, contract: Option, timeout_s: float = 2.0) -> float | None:
        if not self.ib or not self.ib.isConnected():
            return None
        ticker = self.ib.reqMktData(contract, "100,101,104,106", False, False)
        self.ib.sleep(timeout_s)
        bid = float(ticker.bid) if _finite(ticker.bid) else math.nan
        ask = float(ticker.ask) if _finite(ticker.ask) else math.nan
        last = float(ticker.last) if _finite(ticker.last) else math.nan
        self.ib.cancelMktData(contract)
        if _finite(bid) and _finite(ask) and ask >= bid:
            return (bid + ask) / 2.0
        if _finite(last):
            return last
        return None


def _finite(v: Any) -> bool:
    try:
        return v is not None and not math.isnan(float(v))
    except Exception:
        return False

