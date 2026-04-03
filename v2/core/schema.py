"""TradeIntent and related data contracts.

The central contract of the v2 system. See docs/v2/contracts.md.

TradeIntent is what:
- replay scores (simulates the trade, measures P&L)
- live executes (places IBKR orders)
- training learns to emit (model output -> TradeIntent)
"""
from __future__ import annotations

import dataclasses
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any


# Valid enum values
ORDER_STYLES = ("MKT", "LMT", "ADAPTIVE")
TIF_VALUES = ("DAY", "IOC")
EXIT_POLICIES = ("STOP_TP_TIME", "TRAILING", "MODEL_EXIT")
RIGHTS = ("C", "P")

# Constants
BARS_PER_DAY = 390
STRIKE_GRID = 5.0  # SPX 0DTE options are on 5-point grid
_YYYYMMDD = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class TradeIntent:
    """The atomic unit of the v2 trading system.

    Replay scores this. Live executes this. Training learns to emit this.
    Frozen (immutable). Risk updates use RiskAdjustment.
    """

    # --- Decision ---
    trade: bool

    # --- Contract identity (None when trade=False) ---
    expiry: str | None = None
    strike: float | None = None
    right: str | None = None

    # --- Sizing ---
    qty: int = 0

    # --- Entry ---
    entry_ref_price: float | None = None
    order_style: str = "MKT"
    limit_price: float | None = None
    tif: str = "DAY"

    # --- Risk management ---
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    max_hold_bars: int = 0
    exit_policy: str = "STOP_TP_TIME"

    # --- Metadata ---
    confidence: float = 0.0
    reason_codes: tuple[str, ...] = ()
    bar_index: int = 0
    timestamp: str = ""
    intent_id: str = ""

    # --- Quote provenance ---
    bid_at_decision: float | None = None
    ask_at_decision: float | None = None
    underlying_price: float | None = None

    # --- Versioning ---
    policy_version: str = "v2.0.0"

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list = valid."""
        errors: list[str] = []

        if self.trade:
            # Contract identity required
            if self.expiry is None or not _YYYYMMDD.match(self.expiry):
                errors.append(f"trade=True requires valid YYYYMMDD expiry, got {self.expiry!r}")
            if self.strike is None or self.strike <= 0:
                errors.append(f"trade=True requires positive strike, got {self.strike}")
            if self.strike is not None and self.strike % STRIKE_GRID != 0:
                errors.append(f"strike must be on {STRIKE_GRID}-point grid, got {self.strike}")
            if self.right not in RIGHTS:
                errors.append(f"right must be 'C' or 'P', got {self.right!r}")

            # Sizing
            if self.qty < 1:
                errors.append(f"trade=True requires qty >= 1, got {self.qty}")

            # Entry
            if self.entry_ref_price is None or self.entry_ref_price <= 0:
                errors.append(f"trade=True requires positive entry_ref_price, got {self.entry_ref_price}")

            # Risk
            if self.stop_price <= 0:
                errors.append(f"stop_price must be positive, got {self.stop_price}")
            if self.entry_ref_price is not None and self.stop_price >= self.entry_ref_price:
                errors.append(f"stop_price ({self.stop_price}) must be < entry_ref_price ({self.entry_ref_price})")
            if self.entry_ref_price is not None and self.take_profit_price <= self.entry_ref_price:
                errors.append(f"take_profit_price ({self.take_profit_price}) must be > entry_ref_price ({self.entry_ref_price})")
            if not 1 <= self.max_hold_bars <= BARS_PER_DAY:
                errors.append(f"max_hold_bars must be 1-{BARS_PER_DAY}, got {self.max_hold_bars}")

            # Enums
            if self.order_style not in ORDER_STYLES:
                errors.append(f"order_style must be one of {ORDER_STYLES}, got {self.order_style!r}")
            if self.tif not in TIF_VALUES:
                errors.append(f"tif must be one of {TIF_VALUES}, got {self.tif!r}")
            if self.exit_policy not in EXIT_POLICIES:
                errors.append(f"exit_policy must be one of {EXIT_POLICIES}, got {self.exit_policy!r}")

            # Metadata
            if not 0.0 <= self.confidence <= 1.0:
                errors.append(f"confidence must be 0-1, got {self.confidence}")
            if not 0 <= self.bar_index < BARS_PER_DAY:
                errors.append(f"bar_index must be 0-{BARS_PER_DAY - 1}, got {self.bar_index}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Deterministic dict for serialization."""
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        """Deterministic JSON string (sorted keys for checksums)."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TradeIntent:
        """Reconstruct from dict. Converts reason_codes list to tuple."""
        d = dict(d)
        if isinstance(d.get("reason_codes"), list):
            d["reason_codes"] = tuple(d["reason_codes"])
        return cls(**d)

    @classmethod
    def no_trade(cls, bar_index: int = 0, timestamp: str = "",
                 reason_codes: tuple[str, ...] = ()) -> TradeIntent:
        """Convenience: create a no-trade intent."""
        return cls(
            trade=False,
            bar_index=bar_index,
            timestamp=timestamp,
            reason_codes=reason_codes,
            intent_id=str(uuid.uuid4()),
        )


@dataclass(frozen=True)
class RiskAdjustment:
    """Adjusts stop/TP on an open position. Immutable."""

    intent_id: str                       # references the original TradeIntent
    new_stop_price: float | None = None
    new_take_profit_price: float | None = None
    reason_codes: tuple[str, ...] = ()
    adjustment_id: str = ""
    timestamp: str = ""


@dataclass
class SimulatedTrade:
    """Result of simulating a TradeIntent against historical data."""

    intent: TradeIntent

    # Entry
    entry_bar: int = 0
    entry_price: float = 0.0
    entry_fill_bar: int = 0

    # Exit
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""                # STOP_LOSS, TAKE_PROFIT, TRAILING_STOP,
                                         # MODEL_EXIT, MAX_HOLD, EOD

    # P&L
    raw_pnl_pct: float = 0.0            # before spread cost
    spread_cost_pct: float = 0.0        # round-trip spread
    net_pnl_pct: float = 0.0            # raw - spread

    # Duration
    bars_held: int = 0

    # Excursions
    mfe_pct: float = 0.0                # max favorable excursion
    mae_pct: float = 0.0                # max adverse excursion

    # Context
    underlying_at_entry: float = 0.0
    underlying_at_exit: float = 0.0
    vix_regime_at_entry: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["intent"] = self.intent.to_dict()
        return d
