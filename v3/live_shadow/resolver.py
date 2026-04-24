from __future__ import annotations

from dataclasses import replace
from typing import Any

from v3.live_shadow.schema import IBKRContractSpec


class ShadowContractResolver:
    """SPX/SPXW contract resolver for intent-only shadow sessions.

    The resolver returns contract specifications that can be qualified by an
    IBKR client, but this module never places orders.
    """

    def __init__(
        self,
        *,
        exchange: str = "SMART",
        trading_class: str = "SPXW",
        multiplier: str = "100",
    ) -> None:
        self.exchange = exchange
        self.trading_class = trading_class
        self.multiplier = multiplier

    def spec(self, *, expiry_yyyymmdd: str, strike: float, right: str) -> IBKRContractSpec:
        right_norm = str(right).upper()
        if right_norm not in {"C", "P"}:
            raise ValueError(f"SPX option right must be 'C' or 'P', got {right!r}")
        return IBKRContractSpec(
            exchange=self.exchange,
            trading_class=self.trading_class,
            multiplier=self.multiplier,
            last_trade_date_or_contract_month=str(expiry_yyyymmdd),
            strike=float(strike),
            right=right_norm,
        )

    def with_exchange(self, exchange: str) -> "ShadowContractResolver":
        return ShadowContractResolver(
            exchange=exchange,
            trading_class=self.trading_class,
            multiplier=self.multiplier,
        )

    def secdef_request(self, underlying_con_id: int) -> dict[str, Any]:
        return {
            "symbol": "SPX",
            "fut_fop_exchange": "",
            "sec_type": "IND",
            "underlying_con_id": int(underlying_con_id),
            "expected_trading_class": self.trading_class,
            "expected_multiplier": self.multiplier,
        }


def to_ib_insync_option(spec: IBKRContractSpec) -> Any:
    """Convert a shadow spec to ib_insync.Option if ib_insync is installed."""
    try:
        from ib_insync import Option
    except Exception as exc:  # pragma: no cover - depends on local IBKR install
        raise RuntimeError("ib_insync is required to build an IBKR Option object") from exc
    option = Option(
        symbol=spec.symbol,
        lastTradeDateOrContractMonth=spec.last_trade_date_or_contract_month,
        strike=float(spec.strike),
        right=spec.right,
        exchange=spec.exchange,
        currency=spec.currency,
        multiplier=spec.multiplier,
    )
    option.tradingClass = spec.trading_class
    return option


def normalize_spec(spec: IBKRContractSpec, **updates: Any) -> IBKRContractSpec:
    return replace(spec, **updates)

