"""Layer 0: hard-risk envelope.

Two responsibilities:

1. `filter_contract` — per-contract pass/fail check against the hard rails.
   Returns a `ContractCheck` carrying per-gate flags (not a single ordered
   rejection reason) so the opportunity-surface logger can compute counter-
   factuals like "would this contract have passed if the premium cap were
   relaxed?" directly, without a second recomputation pass. This is load-
   bearing for `premium_cap_bind_rate` and `low_delta_forcing_rate` in the
   SPX/$25k feasibility diagnostics.

2. `DailyBudget` — per-day state machine that tracks cumulative realized
   loss, open positions, and *full-premium losers* (trades that lost at
   least `FULL_PREMIUM_LOSS_THRESHOLD` of the premium paid). The plan's
   "3 full-premium losers" rule is a catastrophic-loss circuit breaker,
   not a generic "3 losing trades" rule — a -30% stop-out is not a full-
   premium loser. The dollar loss cap handles accumulated small losses
   separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from v3.config import GuardrailConfig


# A trade counts as a "full-premium loser" when its realized loss is at least
# this fraction of the premium paid. Captures expired-worthless and near-zero
# exits; excludes normal -30% / -50% stop-outs.
FULL_PREMIUM_LOSS_THRESHOLD = 0.95


@dataclass(frozen=True)
class ContractCheck:
    """Per-gate flags for one contract, plus raw values for logging.

    `passed` is the conjunction of all gates. The `would_pass_without_*`
    properties exist specifically to support the Stage 1 feasibility
    diagnostics and must not be removed without updating the logger.
    """

    contract_valid: bool
    premium_cap_ok: bool
    premium_floor_ok: bool
    delta_floor_ok: bool
    spread_cap_ok: bool
    premium: float
    abs_delta: float
    spread_fraction: float

    @property
    def passed(self) -> bool:
        return (
            self.contract_valid
            and self.premium_cap_ok
            and self.premium_floor_ok
            and self.delta_floor_ok
            and self.spread_cap_ok
        )

    @property
    def would_pass_without_premium_cap(self) -> bool:
        return (
            self.contract_valid
            and self.premium_floor_ok
            and self.delta_floor_ok
            and self.spread_cap_ok
        )

    @property
    def would_pass_without_delta_floor(self) -> bool:
        return (
            self.contract_valid
            and self.premium_cap_ok
            and self.premium_floor_ok
            and self.spread_cap_ok
        )

    @property
    def blocked_solely_by_premium_cap(self) -> bool:
        return (not self.premium_cap_ok) and self.would_pass_without_premium_cap


def filter_contract(
    mid: float,
    delta: float,
    spread_fraction: float,
    contract_valid: bool,
    equity: float,
    cfg: GuardrailConfig,
) -> ContractCheck:
    """Run each gate independently and report all pass/fail flags.

    `delta` is passed raw; we absolute-value it here so put/call symmetry is
    enforced at a single point. Callers MUST NOT pre-flip put deltas.
    """
    abs_d = abs(delta)
    premium = mid * 100.0  # both SPX and XSP use $100 multiplier
    cap = cfg.per_trade_premium_cap(equity)

    return ContractCheck(
        contract_valid=contract_valid,
        premium_cap_ok=(premium <= cap),
        premium_floor_ok=(premium >= cfg.premium_floor),
        delta_floor_ok=(abs_d >= cfg.delta_floor),
        spread_cap_ok=(spread_fraction <= cfg.spread_fraction_cap),
        premium=premium,
        abs_delta=abs_d,
        spread_fraction=spread_fraction,
    )


@dataclass
class DailyBudget:
    """Per-day budget state. Caller constructs fresh instance each session.

    The halt triggers are:
    - `full_premium_losers >= cfg.max_full_premium_losers_per_day` (catastrophe)
    - `-realized_pnl >= cfg.daily_loss_cap(equity)` (accumulated dollar drawdown)
    - `open_positions >= cfg.max_open_positions` (blocks new opens, not a halt)

    A `full-premium loser` is a close with `pnl <= -FULL_PREMIUM_LOSS_THRESHOLD
    * premium_paid`. Ordinary -30% stop-outs do NOT count.
    """

    cfg: GuardrailConfig
    session_start_equity: float
    realized_pnl: float = 0.0
    full_premium_losers: int = 0
    open_positions: int = 0
    halted: bool = field(default=False, init=False)
    _pending_premium: float = field(default=0.0, init=False, repr=False)

    def can_open(self) -> bool:
        if self.halted:
            return False
        if self.open_positions >= self.cfg.max_open_positions:
            return False
        return True

    def record_open(self, premium_paid: float) -> None:
        self.open_positions += 1
        self._pending_premium = premium_paid

    def record_close(self, pnl: float) -> None:
        self.open_positions = max(0, self.open_positions - 1)
        self.realized_pnl += pnl
        if (
            self._pending_premium > 0.0
            and pnl <= -FULL_PREMIUM_LOSS_THRESHOLD * self._pending_premium
        ):
            self.full_premium_losers += 1
        self._pending_premium = 0.0
        if (
            self.full_premium_losers >= self.cfg.max_full_premium_losers_per_day
            or -self.realized_pnl >= self.cfg.daily_loss_cap(self.session_start_equity)
        ):
            self.halted = True
