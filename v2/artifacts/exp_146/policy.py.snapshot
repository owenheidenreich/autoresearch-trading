"""Shared trading policy for the frozen v4 exact-chain harness."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecisionPolicy:
    """All parameters that materially shape trading decisions."""

    # Entry gate
    gate_threshold: float = -100.0
    label_gate_min_pnl: float = 0.04

    # Executability filters
    min_contract_mid: float = 0.50
    max_spread_fraction: float = 0.18
    require_volume_or_transactions: bool = True

    # Fixed risk policy for the first frozen exact-chain harness
    stop_pct: float = 0.30
    target_pct: float = 0.50
    max_hold_bars: int = 120
    exit_policy: str = "TRAILING"
    breakeven_trigger_pct: float = 0.15
    extra_trailing_tiers: tuple[tuple[float, float], ...] = ((0.25, 0.08),)

    # Position management
    cooldown_bars: int = 3
    max_concurrent: int = 1
    qty: int = 1

    # Session restrictions — model has edge only in early morning window (bars 60-105)
    no_trade_before_bar: int = 60
    no_trade_after_bar: int = 105

    # Execution
    order_style: str = "MKT"
    tif: str = "DAY"

    # Risk limits
    daily_loss_cap_pct: float = 0.05

    # Account
    starting_equity: float = 10_000.0
    contract_multiplier: int = 100

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionPolicy":
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        # Ensure extra_trailing_tiers is tuple-of-tuples (JSON gives lists)
        if "extra_trailing_tiers" in filtered:
            filtered["extra_trailing_tiers"] = tuple(
                tuple(t) for t in filtered["extra_trailing_tiers"]
            )
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> "DecisionPolicy":
        return cls.from_dict(json.loads(s))

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


DEFAULT_POLICY = DecisionPolicy()
