"""Shared trading policy: single source of truth for all trade-shaping parameters.

Used identically by train.py (output squashing), replay.py (intent construction),
and live/decision.py (live inference). The AI researcher can mutate this file
alongside train.py as one of the two mutable research surfaces.

This file is part of the artifact bundle. Every kept experiment saves a snapshot
of the policy that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class DecisionPolicy:
    """All parameters that materially shape trading decisions.

    Frozen so it can be hashed, serialized, and compared across experiments.
    """

    # --- Gate ---
    gate_threshold: float = 0.50  # trade only when model predicts positive P&L

    # --- Risk output ranges (clamped in model_to_intent) ---
    stop_range: tuple[float, float] = (0.10, 0.50)
    target_range: tuple[float, float] = (0.15, 1.65)
    max_hold_range: tuple[int, int] = (10, 250)

    # --- Position management ---
    cooldown_bars: int = 5
    max_concurrent: int = 1
    qty: int = 1

    # --- Time blocks ---
    no_trade_before_bar: int = 30
    no_trade_after_bar: int = 330

    # --- Order execution ---
    order_style: str = "MKT"
    exit_policy: str = "TRAILING"

    # --- Risk limits ---
    daily_loss_cap_pct: float = 0.05  # 5% of equity, hard stop for the day

    # --- Account ---
    starting_equity: float = 10_000.0
    contract_multiplier: int = 100     # SPX option multiplier

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        d = asdict(self)
        # tuples become lists in asdict, convert back for clarity
        d['stop_range'] = list(d['stop_range'])
        d['target_range'] = list(d['target_range'])
        d['max_hold_range'] = list(d['max_hold_range'])
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> DecisionPolicy:
        d = dict(d)
        d['stop_range'] = tuple(d['stop_range'])
        d['target_range'] = tuple(d['target_range'])
        d['max_hold_range'] = tuple(int(x) for x in d['max_hold_range'])
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> DecisionPolicy:
        return cls.from_dict(json.loads(s))

    def fingerprint(self) -> str:
        """SHA-256 fingerprint. Changes mean a new policy version."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# Default policy used when no override is provided
DEFAULT_POLICY = DecisionPolicy()
