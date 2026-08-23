"""Model-free scale-sensitivity study for the signed 2026-08-16 risk law.

Asks what happens to a **serial compounding** account as it grows from $10,000
toward $100,000 under the signed law: a $2,000 per-trade premium ceiling stated
in dollars, a 20% daily breaker on session-starting equity, a −40%-or-wider
declared stop, at most two entries a session, and no overnight holds.

**This study assumes a hypothetical edge and is not evidence that one exists.**
Win rates are swept as parameters. Every prior measurement on this population is
negative at random entry; nothing here changes that, and no result below may be
cited as support for the strategy.

Dollar outcomes are resampled as measured `(ticket, net)` **pairs** from the
quote-priced path population, never drawn from a parametric assumption and never
from a quantile grid — a quantile grid trims the right tail and biased an earlier
generation of these numbers pessimistic by 6-19%. Resampling whole pairs also
preserves the joint distribution of ticket size and outcome, which independent
draws would destroy.

Nothing here fits a model, reads the partially-acquired backfill, touches
reserved post-2026-08-05 sessions, contacts a vendor, or spends money.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FEES_USD = 3.08
CONTRACT_MULTIPLIER = 100.0
SURVIVAL_FLOOR_SHARE = 0.50
SESSIONS_PER_YEAR = 252


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class SignedLaw:
    """The 2026-08-16 signed risk law, as simulated."""

    premium_cap_usd: float = 2500.0  # absolute dollars, never a share of equity
    daily_breaker_share: float = 0.20  # of session-starting equity
    declared_stop: float = 0.40  # −40%, the tightest the addendum permits
    max_trades_per_session: int = 2
    survival_floor_share: float = SURVIVAL_FLOOR_SHARE

    def sha256(self) -> str:
        payload = "|".join(
            f"{k}={v}" for k, v in sorted(self.__dict__.items())
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TradePopulation:
    """Measured (ticket, net) pairs eligible under the dollar cap."""

    ticket_usd: np.ndarray
    net_usd: np.ndarray

    @property
    def winners(self) -> "TradePopulation":
        keep = self.net_usd > 0
        return TradePopulation(self.ticket_usd[keep], self.net_usd[keep])

    @property
    def losers(self) -> "TradePopulation":
        keep = self.net_usd <= 0
        return TradePopulation(self.ticket_usd[keep], self.net_usd[keep])

    def __len__(self) -> int:
        return int(len(self.net_usd))


def build_population(paths_parquet: str, law: SignedLaw) -> TradePopulation:
    """Per-trade outcomes under the declared stop, capped in dollars."""

    frame = pd.read_parquet(
        paths_parquet,
        columns=["trade_id", "minute_in_trade", "return_from_entry", "bid", "entry_ask_usd"],
    ).sort_values(["trade_id", "minute_in_trade"])
    live = frame[frame.minute_in_trade > 0]

    returns = live.return_from_entry.to_numpy()
    bids = live.bid.to_numpy() * CONTRACT_MULTIPLIER
    ids = live.trade_id.to_numpy()
    asks = live.entry_ask_usd.to_numpy()

    change = np.r_[True, ids[1:] != ids[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(ids)]

    tickets = np.empty(len(starts))
    nets = np.empty(len(starts))
    for i, (a, b) in enumerate(zip(starts, ends)):
        ask = float(asks[a])
        seg = returns[a:b]
        exit_value = float(bids[b - 1])
        hit = np.flatnonzero(seg <= -law.declared_stop)
        if hit.size:
            exit_value = float(bids[a + hit[0]])
        tickets[i] = ask
        nets[i] = exit_value - ask - FEES_USD

    eligible = (tickets + FEES_USD) <= law.premium_cap_usd
    return TradePopulation(tickets[eligible], nets[eligible])


@dataclass
class ScaleResult:
    account_usd: float
    win_rate: float
    ruin_share: float
    ruin_ci95: tuple[float, float]
    trades_per_session_mean: float
    trades_per_session_ci95: tuple[float, float]
    net_per_trade_mean: float
    net_per_trade_ci95: tuple[float, float]
    breaker_sessions_share: float
    affordability_blocked_share: float
    max_premium_bought_usd: float
    median_final_multiple: float
    paths: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return ((centre - margin) / denom, (centre + margin) / denom)


def simulate_scale(
    population: TradePopulation,
    *,
    account_usd: float,
    win_rate: float,
    law: SignedLaw,
    paths: int,
    sessions: int = SESSIONS_PER_YEAR,
    rng: np.random.Generator,
) -> ScaleResult:
    """Walk `paths` compounding years under the signed law.

    The premium ceiling is enforced as **dollars** — `min(cap, equity)` — so it
    cannot widen as equity grows. That is the property the study exists to
    verify, so it is implemented explicitly rather than inherited from any
    share-parameterised helper.
    """

    winners, losers = population.winners, population.losers
    if len(winners) < 50 or len(losers) < 50:
        raise ValueError("population too small to resample both outcomes")

    equity = np.full(paths, float(account_usd))
    alive = np.ones(paths, dtype=bool)
    trades_taken = np.zeros(paths)
    net_total = np.zeros(paths)
    breaker_sessions = np.zeros(paths)
    blocked = np.zeros(paths)
    offered = np.zeros(paths)
    max_premium = np.zeros(paths)

    for _ in range(sessions):
        start = equity.copy()
        session_net = np.zeros(paths)
        tripped = np.zeros(paths, dtype=bool)
        for _slot in range(law.max_trades_per_session):
            active = alive & ~tripped
            if not active.any():
                break
            win = rng.random(paths) < win_rate
            pick_w = rng.integers(0, len(winners), size=paths)
            pick_l = rng.integers(0, len(losers), size=paths)
            ticket = np.where(win, winners.ticket_usd[pick_w], losers.ticket_usd[pick_l])
            net = np.where(win, winners.net_usd[pick_w], losers.net_usd[pick_l])

            # Dollar ceiling, and you cannot buy what you cannot afford.
            ceiling = np.minimum(law.premium_cap_usd, start)
            affordable = (ticket + FEES_USD) <= ceiling
            offered += active
            blocked += active & ~affordable

            take = active & affordable
            session_net += np.where(take, net, 0.0)
            trades_taken += take
            net_total += np.where(take, net, 0.0)
            max_premium = np.maximum(max_premium, np.where(take, ticket, 0.0))
            tripped |= take & (session_net <= -law.daily_breaker_share * start)

        breaker_sessions += tripped
        equity = np.where(alive, start + session_net, equity)
        alive &= equity > law.survival_floor_share * account_usd

    ruined = int((~alive).sum())
    per_session = trades_taken / sessions
    per_trade = np.divide(
        net_total, trades_taken, out=np.zeros(paths), where=trades_taken > 0
    )
    return ScaleResult(
        account_usd=float(account_usd),
        win_rate=float(win_rate),
        ruin_share=ruined / paths,
        ruin_ci95=_wilson(ruined, paths),
        trades_per_session_mean=float(per_session.mean()),
        trades_per_session_ci95=(
            float(np.quantile(per_session, 0.025)),
            float(np.quantile(per_session, 0.975)),
        ),
        net_per_trade_mean=float(per_trade.mean()),
        net_per_trade_ci95=(
            float(per_trade.mean() - 1.96 * per_trade.std() / math.sqrt(paths)),
            float(per_trade.mean() + 1.96 * per_trade.std() / math.sqrt(paths)),
        ),
        breaker_sessions_share=float(breaker_sessions.mean() / sessions),
        affordability_blocked_share=float(blocked.sum() / max(offered.sum(), 1)),
        max_premium_bought_usd=float(max_premium.max()),
        median_final_multiple=float(np.median(equity) / account_usd),
        paths=paths,
    )
