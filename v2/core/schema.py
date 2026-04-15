"""TradeIntent and related exact-contract runtime contracts."""
from __future__ import annotations

import dataclasses
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any


ORDER_STYLES = ("MKT", "LMT", "ADAPTIVE")
TIF_VALUES = ("DAY", "IOC")
EXIT_POLICIES = ("STOP_TP_TIME", "TRAILING", "MODEL_EXIT")
RIGHTS = ("C", "P")

BARS_PER_DAY = 390
STRIKE_GRID = 5.0
_YYYYMMDD = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class TradeIntent:
    """Atomic trading decision used by replay, artifacts, and later live trading."""

    trade: bool

    expiry: str | None = None
    strike: float | None = None
    right: str | None = None

    qty: int = 0

    entry_ref_price: float | None = None
    order_style: str = "MKT"
    limit_price: float | None = None
    tif: str = "DAY"

    stop_price: float = 0.0
    take_profit_price: float = 0.0
    max_hold_bars: int = 0
    exit_policy: str = "STOP_TP_TIME"

    confidence: float = 0.0
    reason_codes: tuple[str, ...] = ()
    bar_index: int = 0
    timestamp: str = ""
    intent_id: str = ""

    bid_at_decision: float | None = None
    ask_at_decision: float | None = None
    underlying_price: float | None = None

    decision_day: str = ""
    snapshot_row: int = -1
    contract_index: int = -1
    contract_score: float = 0.0
    no_trade_score: float = 0.0

    policy_version: str = "v4.0.0"

    def __post_init__(self) -> None:
        if self.trade and self.qty < 1:
            raise ValueError(
                f"TradeIntent with trade=True must have qty >= 1, got qty={self.qty}. "
                f"This is a bug in the caller — every trade intent must specify a position size."
            )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.trade:
            return errors

        if self.expiry is None or not _YYYYMMDD.match(self.expiry):
            errors.append(f"trade=True requires YYYYMMDD expiry, got {self.expiry!r}")
        if self.strike is None or self.strike <= 0:
            errors.append(f"trade=True requires positive strike, got {self.strike}")
        if self.strike is not None and self.strike % STRIKE_GRID != 0:
            errors.append(f"strike must be on {STRIKE_GRID}-point grid, got {self.strike}")
        if self.right not in RIGHTS:
            errors.append(f"right must be one of {RIGHTS}, got {self.right!r}")
        if self.qty < 1:
            errors.append(f"qty must be >= 1, got {self.qty}")
        if self.entry_ref_price is None or self.entry_ref_price <= 0:
            errors.append(f"entry_ref_price must be positive, got {self.entry_ref_price}")
        if self.stop_price <= 0:
            errors.append(f"stop_price must be positive, got {self.stop_price}")
        if self.entry_ref_price is not None and self.stop_price >= self.entry_ref_price:
            errors.append("stop_price must be below entry_ref_price")
        if self.entry_ref_price is not None and self.take_profit_price <= self.entry_ref_price:
            errors.append("take_profit_price must be above entry_ref_price")
        if not 1 <= self.max_hold_bars <= BARS_PER_DAY:
            errors.append(f"max_hold_bars must be 1-{BARS_PER_DAY}, got {self.max_hold_bars}")
        if self.order_style not in ORDER_STYLES:
            errors.append(f"order_style must be one of {ORDER_STYLES}, got {self.order_style!r}")
        if self.tif not in TIF_VALUES:
            errors.append(f"tif must be one of {TIF_VALUES}, got {self.tif!r}")
        if self.exit_policy not in EXIT_POLICIES:
            errors.append(f"exit_policy must be one of {EXIT_POLICIES}, got {self.exit_policy!r}")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append(f"confidence must be 0-1, got {self.confidence}")
        if not 0 <= self.bar_index < BARS_PER_DAY:
            errors.append(f"bar_index must be 0-{BARS_PER_DAY - 1}, got {self.bar_index}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradeIntent":
        d = dict(d)
        if isinstance(d.get("reason_codes"), list):
            d["reason_codes"] = tuple(d["reason_codes"])
        return cls(**d)

    @classmethod
    def no_trade(
        cls,
        *,
        bar_index: int = 0,
        timestamp: str = "",
        reason_codes: tuple[str, ...] = (),
        no_trade_score: float = 0.0,
    ) -> "TradeIntent":
        return cls(
            trade=False,
            bar_index=bar_index,
            timestamp=timestamp,
            reason_codes=reason_codes,
            no_trade_score=no_trade_score,
            intent_id=str(uuid.uuid4()),
        )


@dataclass(frozen=True)
class RiskAdjustment:
    intent_id: str
    new_stop_price: float | None = None
    new_take_profit_price: float | None = None
    reason_codes: tuple[str, ...] = ()
    adjustment_id: str = ""
    timestamp: str = ""


@dataclass
class SimulatedTrade:
    intent: TradeIntent
    entry_bar: int = 0
    entry_price: float = 0.0
    entry_fill_bar: int = 0
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    raw_pnl_pct: float = 0.0
    spread_cost_pct: float = 0.0
    net_pnl_pct: float = 0.0
    bars_held: int = 0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    trade_date: str = ""
    underlying_at_entry: float = 0.0
    underlying_at_exit: float = 0.0
    vix_regime_at_entry: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["intent"] = self.intent.to_dict()
        return d
