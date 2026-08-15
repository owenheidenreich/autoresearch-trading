"""Shared OPRA-only implied-spot, volatility, and Greek adapters.

Historical and live wrappers deliberately delegate to the same functions in
this file.  Official SPX is not accepted by any public feature API; it may only
be joined later by the Phase-0b receipt generator for validation measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Final, Iterable

import numpy as np
import pandas as pd

from v4.research.pathd_contract_clock import session_bounds


RISK_FREE_RATE: Final = 0.04
DIVIDEND_YIELD: Final = 0.0
MIN_PAIR_COUNT: Final = 5
PAIR_STRIKE_RADIUS_POINTS: Final = 150.0
PAIR_SPOT_RESIDUAL_POINTS: Final = 25.0
MIN_COMBINED_SPREAD: Final = 0.05
MIN_TIME_YEARS: Final = 1.0 / (365.0 * 24.0 * 60.0 * 60.0)
IV_LOWER: Final = 1e-4
IV_UPPER: Final = 5.0
IV_ITERATIONS: Final = 100


@dataclass(frozen=True)
class ParityPair:
    strike: float
    call_symbol: str
    put_symbol: str
    call_mid: float
    put_mid: float
    combined_spread: float
    implied_spot: float
    weight: float


@dataclass(frozen=True)
class ParitySnapshot:
    decision_time: pd.Timestamp
    implied_spot: float
    dispersion_bps: float
    selected_pairs: tuple[ParityPair, ...]
    total_weight: float

    @property
    def candidate_pair_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{pair.strike:.3f}|{pair.call_symbol}|{pair.put_symbol}"
            for pair in self.selected_pairs
        )


@dataclass(frozen=True)
class IVSnapshot:
    decision_time: pd.Timestamp
    implied_spot: float
    atm_iv: float
    put_skew: float
    call_skew: float
    smile_curvature: float
    straddle_bps: float


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _norm_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _time_to_expiry_years(decision_time: pd.Timestamp | datetime) -> float:
    timestamp = pd.Timestamp(decision_time)
    if timestamp.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    session = timestamp.tz_convert("America/New_York").date()
    _, closed, _ = session_bounds(session)
    seconds = (pd.Timestamp(closed).tz_convert("UTC") - timestamp.tz_convert("UTC")).total_seconds()
    if seconds <= 0.0:
        raise ValueError("decision_time must precede option expiration")
    return max(seconds / (365.0 * 24.0 * 60.0 * 60.0), MIN_TIME_YEARS)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = float(ordered_weights.sum()) / 2.0
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), cutoff)])


def parity_snapshot(rows: pd.DataFrame, *, decision_time: pd.Timestamp | datetime) -> ParitySnapshot:
    """Estimate spot from exact-strike call/put pairs without an index input."""

    required = {"strike", "right", "bid", "ask", "raw_symbol"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"parity rows missing columns: {sorted(missing)}")
    frame = rows.loc[:, sorted(required)].copy()
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    for name in ("bid", "ask"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame[
        np.isfinite(frame["strike"])
        & np.isfinite(frame["bid"])
        & np.isfinite(frame["ask"])
        & (frame["bid"] >= 0.0)
        & (frame["ask"] >= frame["bid"])
        & frame["right"].astype(str).isin(("C", "P"))
    ].copy()
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
    frame["spread"] = frame["ask"] - frame["bid"]
    calls = (
        frame[frame["right"].astype(str).eq("C")]
        .sort_values(["strike", "raw_symbol"])
        .drop_duplicates("strike", keep="last")
        .set_index("strike")
    )
    puts = (
        frame[frame["right"].astype(str).eq("P")]
        .sort_values(["strike", "raw_symbol"])
        .drop_duplicates("strike", keep="last")
        .set_index("strike")
    )
    strikes = calls.index.intersection(puts.index).to_numpy(float)
    if len(strikes) < MIN_PAIR_COUNT:
        raise ValueError("insufficient exact-strike call/put pairs")
    calls = calls.loc[strikes]
    puts = puts.loc[strikes]
    years = _time_to_expiry_years(pd.Timestamp(decision_time))
    discounted_strikes = strikes * math.exp(-RISK_FREE_RATE * years)
    pair_spots = calls["mid"].to_numpy(float) - puts["mid"].to_numpy(float) + discounted_strikes
    spreads = calls["spread"].to_numpy(float) + puts["spread"].to_numpy(float)
    weights = 1.0 / np.maximum(spreads, MIN_COMBINED_SPREAD)
    preliminary = _weighted_median(pair_spots, weights)
    selected = (
        (np.abs(strikes - preliminary) <= PAIR_STRIKE_RADIUS_POINTS)
        & (np.abs(pair_spots - preliminary) <= PAIR_SPOT_RESIDUAL_POINTS)
    )
    if int(selected.sum()) < MIN_PAIR_COUNT:
        raise ValueError("insufficient self-consistent parity pairs")
    strikes = strikes[selected]
    pair_spots = pair_spots[selected]
    spreads = spreads[selected]
    weights = weights[selected]
    calls = calls.iloc[np.flatnonzero(selected)]
    puts = puts.iloc[np.flatnonzero(selected)]
    spot = _weighted_median(pair_spots, weights)
    dispersion = float(np.quantile(pair_spots, 0.9) - np.quantile(pair_spots, 0.1))
    pairs = tuple(
        ParityPair(
            strike=float(strike),
            call_symbol=str(call_symbol),
            put_symbol=str(put_symbol),
            call_mid=float(call_mid),
            put_mid=float(put_mid),
            combined_spread=float(spread),
            implied_spot=float(pair_spot),
            weight=float(weight),
        )
        for strike, call_symbol, put_symbol, call_mid, put_mid, spread, pair_spot, weight in zip(
            strikes,
            calls["raw_symbol"].astype(str),
            puts["raw_symbol"].astype(str),
            calls["mid"].to_numpy(float),
            puts["mid"].to_numpy(float),
            spreads,
            pair_spots,
            weights,
            strict=True,
        )
    )
    return ParitySnapshot(
        decision_time=pd.Timestamp(decision_time),
        implied_spot=spot,
        dispersion_bps=dispersion / spot * 10_000.0,
        selected_pairs=pairs,
        total_weight=float(weights.sum()),
    )


def historical_parity_snapshot(rows: pd.DataFrame, *, decision_time: pd.Timestamp | datetime) -> ParitySnapshot:
    return parity_snapshot(rows, decision_time=decision_time)


def live_parity_snapshot(rows: pd.DataFrame, *, decision_time: pd.Timestamp | datetime) -> ParitySnapshot:
    return parity_snapshot(rows, decision_time=decision_time)


def implied_spot_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Build the complete eight-field causal implied-spot family."""

    if "event_time" not in rows:
        raise ValueError("implied-spot frame requires event_time")
    snapshots: list[ParitySnapshot | None] = []
    decision_times: list[pd.Timestamp] = []
    for decision_time, group in rows.groupby("event_time", sort=True):
        decision_times.append(pd.Timestamp(decision_time))
        try:
            snapshots.append(parity_snapshot(group, decision_time=pd.Timestamp(decision_time)))
        except ValueError:
            # Missing/invalid ladders are explicit nonfinite rows.  No prior
            # implied spot is carried into the failed decision boundary.
            snapshots.append(None)
    frame = pd.DataFrame(
        {
            "event_time": decision_times,
            "opra_implied_spot": [snapshot.implied_spot if snapshot else np.nan for snapshot in snapshots],
            "opra_implied_spot_dispersion_bps": [snapshot.dispersion_bps if snapshot else np.nan for snapshot in snapshots],
            "_weight": [snapshot.total_weight if snapshot else np.nan for snapshot in snapshots],
        }
    ).sort_values("event_time").reset_index(drop=True)
    spot = frame["opra_implied_spot"]
    indexed_spot = pd.Series(spot.to_numpy(float), index=pd.DatetimeIndex(frame["event_time"]))
    for minutes in (1, 5, 15):
        exact_prior = indexed_spot.reindex(
            indexed_spot.index - pd.Timedelta(minutes=minutes)
        ).to_numpy(float)
        frame[f"opra_spot_return_{minutes}m_bps"] = (spot.to_numpy(float) / exact_prior - 1.0) * 10_000.0
    cumulative_weight = frame["_weight"].cumsum()
    weighted_session_mean = (spot * frame["_weight"]).cumsum() / cumulative_weight
    frame["opra_spot_vwap_gap_bps"] = (spot / weighted_session_mean - 1.0) * 10_000.0
    running_range = spot.cummax() - spot.cummin()
    frame["opra_spot_session_range_bps"] = running_range / spot * 10_000.0
    denominator = frame["opra_spot_session_range_bps"].replace(0.0, np.nan)
    frame["opra_spot_omar_clipped"] = (
        frame["opra_spot_vwap_gap_bps"] / denominator
    ).clip(-3.0, 3.0).fillna(0.0)
    return frame.drop(columns=["_weight"])


def bsm_price(
    *, spot: float, strike: float, years: float, rate: float, dividend: float,
    volatility: float, right: str,
) -> float:
    if min(spot, strike, years, volatility) <= 0.0 or right not in {"C", "P"}:
        raise ValueError("invalid Black-Scholes inputs")
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * years) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    if right == "C":
        return spot * math.exp(-dividend * years) * _norm_cdf(d1) - strike * math.exp(-rate * years) * _norm_cdf(d2)
    return strike * math.exp(-rate * years) * _norm_cdf(-d2) - spot * math.exp(-dividend * years) * _norm_cdf(-d1)


def solve_implied_volatility(
    *, price: float, spot: float, strike: float, years: float, right: str,
    rate: float = RISK_FREE_RATE, dividend: float = DIVIDEND_YIELD,
) -> float:
    if not all(math.isfinite(value) for value in (price, spot, strike, years, rate, dividend)):
        raise ValueError("nonfinite implied-volatility input")
    discounted_spot = spot * math.exp(-dividend * years)
    discounted_strike = strike * math.exp(-rate * years)
    intrinsic = max(0.0, discounted_spot - discounted_strike) if right == "C" else max(0.0, discounted_strike - discounted_spot)
    upper = discounted_spot if right == "C" else discounted_strike
    if price < intrinsic - 1e-10 or price >= upper:
        raise ValueError("option price outside no-arbitrage bounds")
    low, high = IV_LOWER, IV_UPPER
    if bsm_price(spot=spot, strike=strike, years=years, rate=rate, dividend=dividend, volatility=high, right=right) < price:
        raise ValueError("implied volatility above frozen solver domain")
    for _ in range(IV_ITERATIONS):
        mid = (low + high) / 2.0
        value = bsm_price(spot=spot, strike=strike, years=years, rate=rate, dividend=dividend, volatility=mid, right=right)
        if value < price:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def iv_snapshot(rows: pd.DataFrame, *, decision_time: pd.Timestamp | datetime) -> IVSnapshot:
    parity = parity_snapshot(rows, decision_time=decision_time)
    years = _time_to_expiry_years(pd.Timestamp(decision_time))
    solved: list[tuple[ParityPair, float]] = []
    for pair in parity.selected_pairs:
        # Use the OTM leg away from ATM.  Deep-ITM mids can sit a few cents
        # below theoretical intrinsic inside the quoted spread and are an
        # unstable numerical source; the paired OTM leg carries the same IV.
        legs = (
            ((pair.put_mid, "P"),)
            if pair.strike < parity.implied_spot - 2.5
            else ((pair.call_mid, "C"),)
            if pair.strike > parity.implied_spot + 2.5
            else ((pair.call_mid, "C"), (pair.put_mid, "P"))
        )
        values = []
        for price, right in legs:
            try:
                values.append(
                    solve_implied_volatility(
                        price=price,
                        spot=parity.implied_spot,
                        strike=pair.strike,
                        years=years,
                        right=right,
                    )
                )
            except ValueError:
                continue
        if not values:
            continue
        solved.append((pair, float(sum(values) / len(values))))
    if len(solved) < 3:
        raise ValueError("insufficient numerically stable IV pairs")
    atm_pair, atm_iv = min(solved, key=lambda item: (abs(item[0].strike - parity.implied_spot), item[0].strike))
    put_candidates = [item for item in solved if item[0].strike <= parity.implied_spot - 20.0]
    call_candidates = [item for item in solved if item[0].strike >= parity.implied_spot + 20.0]
    if not put_candidates or not call_candidates:
        raise ValueError("missing fixed-wing IV pairs")
    put_pair, put_iv = max(put_candidates, key=lambda item: item[0].strike)
    call_pair, call_iv = min(call_candidates, key=lambda item: item[0].strike)
    return IVSnapshot(
        decision_time=pd.Timestamp(decision_time),
        implied_spot=parity.implied_spot,
        atm_iv=float(atm_iv),
        put_skew=float(put_iv - atm_iv),
        call_skew=float(call_iv - atm_iv),
        smile_curvature=float((put_iv + call_iv) / 2.0 - atm_iv),
        straddle_bps=float((atm_pair.call_mid + atm_pair.put_mid) / parity.implied_spot * 10_000.0),
    )


def historical_iv_snapshot(rows: pd.DataFrame, *, decision_time: pd.Timestamp | datetime) -> IVSnapshot:
    return iv_snapshot(rows, decision_time=decision_time)


def live_iv_snapshot(rows: pd.DataFrame, *, decision_time: pd.Timestamp | datetime) -> IVSnapshot:
    return iv_snapshot(rows, decision_time=decision_time)


def implied_volatility_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    snapshots: list[IVSnapshot | None] = []
    decision_times: list[pd.Timestamp] = []
    for decision_time, group in rows.groupby("event_time", sort=True):
        decision_times.append(pd.Timestamp(decision_time))
        try:
            snapshots.append(iv_snapshot(group, decision_time=pd.Timestamp(decision_time)))
        except ValueError:
            snapshots.append(None)
    frame = pd.DataFrame(
        {
            "event_time": decision_times,
            "opra_atm_iv": [snapshot.atm_iv if snapshot else np.nan for snapshot in snapshots],
            "opra_put_skew": [snapshot.put_skew if snapshot else np.nan for snapshot in snapshots],
            "opra_call_skew": [snapshot.call_skew if snapshot else np.nan for snapshot in snapshots],
            "opra_smile_curvature": [snapshot.smile_curvature if snapshot else np.nan for snapshot in snapshots],
            "opra_straddle_bps": [snapshot.straddle_bps if snapshot else np.nan for snapshot in snapshots],
        }
    ).sort_values("event_time").reset_index(drop=True)
    indexed_iv = pd.Series(
        frame["opra_atm_iv"].to_numpy(float),
        index=pd.DatetimeIndex(frame["event_time"]),
    )
    exact_prior = indexed_iv.reindex(
        indexed_iv.index - pd.Timedelta(minutes=5)
    ).to_numpy(float)
    frame["opra_atm_iv_change_5m"] = frame["opra_atm_iv"].to_numpy(float) - exact_prior
    return frame


def self_computed_greeks(
    *, option_price: float, spot: float, strike: float, years: float, right: str,
    rate: float = RISK_FREE_RATE, dividend: float = DIVIDEND_YIELD,
) -> dict[str, float]:
    volatility = solve_implied_volatility(
        price=option_price, spot=spot, strike=strike, years=years,
        right=right, rate=rate, dividend=dividend,
    )
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * years) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    discount_q = math.exp(-dividend * years)
    discount_r = math.exp(-rate * years)
    delta = discount_q * _norm_cdf(d1) if right == "C" else discount_q * (_norm_cdf(d1) - 1.0)
    gamma = discount_q * _norm_pdf(d1) / (spot * volatility * root_t)
    vega = spot * discount_q * _norm_pdf(d1) * root_t
    common_theta = -spot * discount_q * _norm_pdf(d1) * volatility / (2.0 * root_t)
    if right == "C":
        theta = common_theta - rate * strike * discount_r * _norm_cdf(d2) + dividend * spot * discount_q * _norm_cdf(d1)
    else:
        theta = common_theta + rate * strike * discount_r * _norm_cdf(-d2) - dividend * spot * discount_q * _norm_cdf(-d1)
    return {
        "bs_delta": float(delta),
        "bs_gamma": float(gamma),
        "E.bs.delta": float(delta),
        "E.bs.gamma": float(gamma),
        "bs_theta": float(theta),
        "bs_vega": float(vega),
        "self_iv": float(volatility),
    }


def historical_self_computed_greeks(**inputs: float | str) -> dict[str, float]:
    return self_computed_greeks(**inputs)  # type: ignore[arg-type]


def live_self_computed_greeks(**inputs: float | str) -> dict[str, float]:
    return self_computed_greeks(**inputs)  # type: ignore[arg-type]
