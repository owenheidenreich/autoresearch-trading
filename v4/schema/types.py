"""Shared schema primitives.

Enumerated string types for vendor sources, IV/Greek sources, option rights,
and contract roots. These are used as discriminators across all layers.
"""
from __future__ import annotations

from enum import Enum


class VendorSource(str, Enum):
    """Where a row originated."""

    OPTIONSDX = "OPTIONSDX"
    DATABENTO_OPRA = "DATABENTO_OPRA"
    IBKR_LIVE = "IBKR_LIVE"
    IBKR_PAPER = "IBKR_PAPER"
    OPTIONSDEPTH = "OPTIONSDEPTH"
    SYNTHETIC = "SYNTHETIC"  # for tests only

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


class GreekSource(str, Enum):
    """Provenance of IV / Greeks values."""

    OPTIONSDX = "optionsdx"
    BLACK_SCHOLES_V4 = "black_scholes_v4"
    IBKR_MODEL = "ibkr_model"
    DATABENTO = "databento"  # only set when Databento ever provides Greeks; not today

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


class OptionRight(str, Enum):
    CALL = "C"
    PUT = "P"


class ContractRoot(str, Enum):
    """SPX-family roots. SPXW is the daily-expiration weekly root.

    Tuesday and Thursday SPXW expirations launched April 18 and May 11 2022;
    pre-2022 0DTE coverage uses Mon/Wed/Fri SPXW only. See protocol Section 5
    Phase 2A regime cohorts.
    """

    SPX = "SPX"      # standard monthly third-Friday AM-settled
    SPXW = "SPXW"    # weekly / daily PM-settled (modern 0DTE)
    SPY = "SPY"
    VIX = "VIX"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


class TimestampPrecision(str, Enum):
    MILLI = "ms"
    MICRO = "us"
    NANO = "ns"


SCHEMA_VERSION = "v1.0.1"
"""Bumped only via documented schema change. CI compares deterministic-rebuild
hash across runs; mismatch without a schema bump is a critical error."""
