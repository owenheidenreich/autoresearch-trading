"""v3 configuration — symbol-parameterized so an XSP/SPY switch is a config change.

Two frozen dataclasses:
- SymbolConfig: product-level facts (ticker prefix, strike grid, multiplier)
- GuardrailConfig: research-framing risk rails (equity, caps, floors)

Defaults match the SPX/$25k research framing as of 2026-04-20. XSP values are
kept for reference even though the XSP path is deferred until a data vendor is
restored.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str = "SPX"
    polygon_root_prefixes: tuple[str, ...] = ("SPX", "SPXW")
    strike_grid: float = 5.0
    contract_multiplier: int = 100
    cache_dirname: str = "spxw_full_chain"

    @classmethod
    def for_xsp(cls) -> SymbolConfig:
        return cls(
            symbol="XSP",
            polygon_root_prefixes=("XSP",),
            strike_grid=1.0,
            contract_multiplier=100,
            cache_dirname="xsp_full_chain",
        )


@dataclass(frozen=True)
class GuardrailConfig:
    """Hard-risk envelope — Layer 0 of the v3 architecture.

    Philosophy (see `v3/reference/decay_analysis_2026_04_20.md`): the cap is
    a worst-case single-trade ceiling, NOT a risk-model assumption. We do not
    bake a stop-loss fraction into the guardrail because empirical 0DTE price
    paths show 5-9% of -60%-drawdown contracts recover to positive — a fixed
    stop would cut them all. Exit discipline is Layer 2/3's problem; Layer 0
    just prevents account blow-up by capping one trade's worst case and
    halting the day on cumulative damage.

    Defaults are the "Moderate" tier from the decay analysis:
    - per-trade cap fits morning 0.20-0.30 delta SPX 0DTE contracts
    - daily brake halts after a catastrophic loss + meaningful follow-on
    - 2-loser rule is a redundant circuit breaker for exact-full-premium wipes
    """

    starting_equity: float = 25_000.0

    premium_cap_abs: float = 1_200.0
    premium_cap_pct_equity: float = 0.048
    premium_floor: float = 100.0

    delta_floor: float = 0.12
    spread_fraction_cap: float = 0.10

    daily_loss_cap_abs: float = 1_500.0
    daily_loss_cap_pct_equity: float = 0.06
    max_full_premium_losers_per_day: int = 2

    max_open_positions: int = 1

    def per_trade_premium_cap(self, equity: float) -> float:
        return min(self.premium_cap_abs, self.premium_cap_pct_equity * equity)

    def daily_loss_cap(self, equity: float) -> float:
        return min(self.daily_loss_cap_abs, self.daily_loss_cap_pct_equity * equity)

    def fingerprint(self) -> str:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]
