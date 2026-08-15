"""Deterministic arithmetic for the decision-grade MES strategic pivot.

This module does not score a strategy.  It translates already measured ES
spread, public MES contract/fee terms and the project's measured power table
into the smallest honest owner decision after the long-SPXW branch closed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


MES_MULTIPLIER_USD_PER_POINT = 5.0
MES_TICK_POINTS = 0.25
IBKR_COMMISSION_USD_PER_SIDE = 0.25
IBKR_EXCHANGE_FEE_USD_PER_SIDE = 0.35
CONSERVATIVE_ROUND_TRIP_POINTS = 0.55


@dataclass(frozen=True)
class MesCostTranslation:
    spread_ticks: float
    spread_points: float
    spread_usd: float
    stated_fees_usd: float
    stated_fees_points: float
    translated_round_trip_usd: float
    translated_round_trip_points: float
    conservative_round_trip_usd: float
    conservative_round_trip_points: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def translate_es_spread_to_mes(spread_ticks: float) -> MesCostTranslation:
    """Translate an observed ES spread in ticks to one MES round trip.

    The tick count is a declared proxy until MES BBO is purchased and measured.
    One full displayed spread is charged over aggressive entry plus exit.
    """

    if not 0 < spread_ticks < 10:
        raise ValueError("spread_ticks must be finite and between zero and ten")
    spread_points = spread_ticks * MES_TICK_POINTS
    spread_usd = spread_points * MES_MULTIPLIER_USD_PER_POINT
    stated_fees_usd = 2 * (IBKR_COMMISSION_USD_PER_SIDE + IBKR_EXCHANGE_FEE_USD_PER_SIDE)
    stated_fees_points = stated_fees_usd / MES_MULTIPLIER_USD_PER_POINT
    translated_points = spread_points + stated_fees_points
    translated_usd = translated_points * MES_MULTIPLIER_USD_PER_POINT
    return MesCostTranslation(
        spread_ticks=spread_ticks,
        spread_points=spread_points,
        spread_usd=spread_usd,
        stated_fees_usd=stated_fees_usd,
        stated_fees_points=stated_fees_points,
        translated_round_trip_usd=translated_usd,
        translated_round_trip_points=translated_points,
        conservative_round_trip_usd=(
            CONSERVATIVE_ROUND_TRIP_POINTS * MES_MULTIPLIER_USD_PER_POINT
        ),
        conservative_round_trip_points=CONSERVATIVE_ROUND_TRIP_POINTS,
    )


def target_stop_economics(target_points: float, stop_points: float) -> dict[str, float]:
    """Net one-MES target/stop arithmetic under the conservative cost ceiling."""

    if target_points <= 0 or stop_points <= 0:
        raise ValueError("target and stop points must be positive")
    cost = CONSERVATIVE_ROUND_TRIP_POINTS * MES_MULTIPLIER_USD_PER_POINT
    win = target_points * MES_MULTIPLIER_USD_PER_POINT - cost
    loss = -(stop_points * MES_MULTIPLIER_USD_PER_POINT + cost)
    return {
        "target_points": target_points,
        "stop_points": stop_points,
        "net_win_usd": win,
        "net_loss_usd": loss,
        "reward_to_risk": win / abs(loss),
        "breakeven_target_hit_rate": abs(loss) / (win + abs(loss)),
    }


def scaled_data_cost(
    sessions: int,
    *,
    ohlcv_cost_usd: float,
    ohlcv_sessions: int,
    bbo_cost_usd: float,
    bbo_sessions: int,
) -> dict[str, float | int]:
    """Scale prior exact vendor estimates without making a vendor request."""

    if min(sessions, ohlcv_sessions, bbo_sessions) <= 0:
        raise ValueError("session counts must be positive")
    ohlcv_per_session = ohlcv_cost_usd / ohlcv_sessions
    bbo_per_session = bbo_cost_usd / bbo_sessions
    estimate = sessions * (ohlcv_per_session + bbo_per_session)
    return {
        "sessions": sessions,
        "ohlcv_usd_per_session": ohlcv_per_session,
        "bbo_usd_per_session": bbo_per_session,
        "scaled_total_usd": estimate,
    }

