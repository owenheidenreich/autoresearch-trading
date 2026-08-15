"""Exact model-facing contract for scoped Protocol101 Stage-1 research.

The processed corpus retains many raw/vendor fields for guards, labels, fills,
PnL, and audit. This adapter is the only allowed bridge from those rows to the
new Stage-1 alpha matrix. It deliberately exposes exactly 17 synchronized
features and never returns the generic market/option arrays.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model.protocol101_regimen_repair import (
    FORBIDDEN_MODEL_FIELDS,
    assert_alpha_feature_names,
)

CONTRACT_ID = "protocol101-scoped-canonical-stage1-v1"
OPTION_TICK = 0.05
RISK_FREE_RATE = 0.05
DIVIDEND_YIELD = 0.0
NEAR_ATM_ABS_OFFSET = 19.0
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

CONTEXT_FEATURES = (
    "spx_vwap_gap_points",
    "spx_vwap_gap_bps",
    "spx_vwap_gap_over_session_range",
    "session_range_bps",
    "momentum_5m_bps",
    "momentum_15m_bps",
    "momentum_5m_over_session_range",
    "momentum_15m_over_session_range",
    "omar_clipped_neg3_pos3",
    "vwap_side_alignment_flag",
    "omar_side_alignment_flag",
    "momentum15_side_alignment_flag",
)
D_FEATURES = (
    "D.near_atm.straddle_mid_spot_bps",
    "D.near_atm.put_call_mid_ratio",
    "D.near_atm.side_smile_slope_bps_per_5pt",
)
E_FEATURES = (
    "E.bs.delta",
    "E.bs.gamma",
)
FEATURE_NAMES = (*CONTEXT_FEATURES, *D_FEATURES, *E_FEATURES)

HYPOTHESES = {
    "H0": CONTEXT_FEATURES,
    "H1": (*CONTEXT_FEATURES, *D_FEATURES),
    "H2": (*CONTEXT_FEATURES, *E_FEATURES),
    "H3": FEATURE_NAMES,
}

QUARANTINED_ALPHA_TOKENS = (
    "C.mid.",
    "E.bs.iv",
    "vix_change",
    "raw_bid",
    "raw_ask",
    "raw_spread",
    "bid_size",
    "ask_size",
    "quote_age",
    "volume",
    "open_interest",
    "vendor_greek",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantized_mid(value: Any) -> float:
    number = _finite(value)
    if number is None or number <= 0.0:
        return float("nan")
    return float(round(number / OPTION_TICK) * OPTION_TICK)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _bs_price(spot: float, strike: float, tte: float, vol: float, right: str) -> float:
    if min(spot, strike, tte, vol) <= 0.0:
        return float("nan")
    sqrt_t = math.sqrt(tte)
    d1 = (
        math.log(spot / strike)
        + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * vol * vol) * tte
    ) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df_r = math.exp(-RISK_FREE_RATE * tte)
    df_q = math.exp(-DIVIDEND_YIELD * tte)
    if right == "C":
        return spot * df_q * _normal_cdf(d1) - strike * df_r * _normal_cdf(d2)
    return strike * df_r * _normal_cdf(-d2) - spot * df_q * _normal_cdf(-d1)


def _implied_vol(mid: float, spot: float, strike: float, tte: float, right: str) -> float | None:
    if (
        not math.isfinite(mid)
        or mid < 2.0 * OPTION_TICK
        or min(spot, strike, tte) <= 0.0
        or right not in {"C", "P"}
    ):
        return None
    intrinsic = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
    if mid < max(0.0, intrinsic - 1.0):
        return None
    lo, hi = 1e-4, 5.0
    if _bs_price(spot, strike, tte, hi, right) < mid:
        return None
    for _ in range(45):
        trial = (lo + hi) / 2.0
        price = _bs_price(spot, strike, tte, trial, right)
        if not math.isfinite(price):
            return None
        if price < mid:
            lo = trial
        else:
            hi = trial
    return (lo + hi) / 2.0


def _bs_delta_gamma(
    spot: float, strike: float, tte: float, vol: float, right: str
) -> tuple[float, float]:
    sqrt_t = math.sqrt(max(tte, 1e-12))
    d1 = (
        math.log(spot / strike)
        + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * vol * vol) * tte
    ) / (vol * sqrt_t)
    df_q = math.exp(-DIVIDEND_YIELD * tte)
    delta = df_q * _normal_cdf(d1)
    if right == "P":
        delta -= df_q
    gamma = df_q * _normal_pdf(d1) / (spot * vol * sqrt_t)
    return float(delta), float(gamma)


def _tte_years(value: Any) -> float:
    decision = pd.Timestamp(value)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    decision_dt = decision.to_pydatetime()
    local_date = decision_dt.astimezone(NY).date()
    expiry = datetime(
        local_date.year, local_date.month, local_date.day, 16, 0, tzinfo=NY
    ).astimezone(UTC)
    seconds = max((expiry - decision_dt.astimezone(UTC)).total_seconds(), 60.0)
    return seconds / (365.0 * 24.0 * 60.0 * 60.0)


def _required_indexes(row: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    option_names = tuple(row.get("feature_names") or ())
    market_names = tuple(row.get("market_feature_names") or ())
    required_option = {"mid"}
    required_market = {
        "spx_close",
        "spx_vwap",
        "omar",
        "session_range",
        "momentum_5m",
        "momentum_15m",
    }
    missing_option = sorted(required_option - set(option_names))
    missing_market = sorted(required_market - set(market_names))
    if missing_option or missing_market:
        raise ValueError(
            f"canonical feature inputs missing: option={missing_option}, market={missing_market}"
        )
    return (
        {name: option_names.index(name) for name in required_option},
        {name: market_names.index(name) for name in required_market},
    )


def _side_slope(offsets: np.ndarray, values: np.ndarray, position: int) -> float:
    finite = np.isfinite(offsets) & np.isfinite(values)
    valid_positions = np.flatnonzero(finite)
    if position not in valid_positions or len(valid_positions) < 2:
        return float("nan")
    rank = int(np.flatnonzero(valid_positions == position)[0])
    if 0 < rank < len(valid_positions) - 1:
        left, right = valid_positions[rank - 1], valid_positions[rank + 1]
    elif rank == 0:
        left, right = valid_positions[0], valid_positions[1]
    else:
        left, right = valid_positions[-2], valid_positions[-1]
    dx = float(offsets[right] - offsets[left])
    if dx == 0.0:
        return float("nan")
    return float((values[right] - values[left]) / dx * 5.0)


def feature_matrix(row: dict[str, Any]) -> np.ndarray:
    """Return a ``(strike, right, 17)`` model-facing feature tensor."""
    assert_model_alpha_firewall(FEATURE_NAMES)
    option_idx, market_idx = _required_indexes(row)
    ladder = np.asarray(row.get("option_ladder"), dtype=float)
    offsets = np.asarray(row.get("strike_offsets"), dtype=float)
    market_window = np.asarray(row.get("market_window"), dtype=float)
    rights = tuple(str(value) for value in (row.get("rights") or ("C", "P")))
    if ladder.ndim != 3 or ladder.shape[:2] != (len(offsets), len(rights)):
        raise ValueError(f"unexpected option ladder shape: {ladder.shape}")
    if market_window.ndim != 2 or len(market_window) == 0:
        raise ValueError("market_window must contain completed-minute context")

    market = market_window[-1]
    spx = float(market[market_idx["spx_close"]])
    vwap = float(market[market_idx["spx_vwap"]])
    omar = float(market[market_idx["omar"]])
    session_range = float(market[market_idx["session_range"]])
    momentum_5m = float(market[market_idx["momentum_5m"]])
    momentum_15m = float(market[market_idx["momentum_15m"]])
    spx_denom = max(abs(spx), 1.0)
    range_denom = max(abs(session_range), 1.0)
    gap = spx - vwap

    mids = np.vectorize(_quantized_mid)(
        ladder[:, :, option_idx["mid"]]
    ).astype(float)
    near = np.abs(offsets) <= NEAR_ATM_ABS_OFFSET
    out = np.full((len(offsets), len(rights), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    decision_tte = _tte_years(row.get("decision_time"))
    atm = float(row.get("atm_strike"))
    strikes = atm + offsets

    for strike_idx, offset in enumerate(offsets):
        call_mid = mids[strike_idx, rights.index("C")] if "C" in rights else float("nan")
        put_mid = mids[strike_idx, rights.index("P")] if "P" in rights else float("nan")
        for right_idx, right in enumerate(rights):
            values = {
                "spx_vwap_gap_points": gap,
                "spx_vwap_gap_bps": gap / spx_denom * 10_000.0,
                "spx_vwap_gap_over_session_range": gap / range_denom,
                "session_range_bps": session_range / spx_denom * 10_000.0,
                "momentum_5m_bps": momentum_5m / spx_denom * 10_000.0,
                "momentum_15m_bps": momentum_15m / spx_denom * 10_000.0,
                "momentum_5m_over_session_range": momentum_5m / range_denom,
                "momentum_15m_over_session_range": momentum_15m / range_denom,
                "omar_clipped_neg3_pos3": min(max(omar, -3.0), 3.0),
                "vwap_side_alignment_flag": float(
                    (right == "C" and gap > 0.0) or (right == "P" and gap < 0.0)
                ),
                "omar_side_alignment_flag": float(
                    (right == "C" and omar > 0.0) or (right == "P" and omar < 0.0)
                ),
                "momentum15_side_alignment_flag": float(
                    (right == "C" and momentum_15m > 0.0)
                    or (right == "P" and momentum_15m < 0.0)
                ),
            }
            if near[strike_idx]:
                if math.isfinite(call_mid) and math.isfinite(put_mid):
                    values["D.near_atm.straddle_mid_spot_bps"] = (
                        (call_mid + put_mid) / spx_denom * 10_000.0
                    )
                    if call_mid > 0.0:
                        values["D.near_atm.put_call_mid_ratio"] = put_mid / call_mid
                side_values = np.where(near, mids[:, right_idx] / spx_denom * 10_000.0, np.nan)
                values["D.near_atm.side_smile_slope_bps_per_5pt"] = _side_slope(
                    offsets, side_values, strike_idx
                )

            mid = float(mids[strike_idx, right_idx])
            vol = _implied_vol(mid, spx, float(strikes[strike_idx]), decision_tte, right)
            if vol is not None:
                delta, gamma = _bs_delta_gamma(
                    spx, float(strikes[strike_idx]), decision_tte, vol, right
                )
                values["E.bs.delta"] = delta
                values["E.bs.gamma"] = gamma

            for name, value in values.items():
                out[strike_idx, right_idx, feature_index[name]] = float(value)
    return out


def hypothesis_matrix(row: dict[str, Any], hypothesis: str) -> np.ndarray:
    names = HYPOTHESES.get(str(hypothesis))
    if names is None:
        raise ValueError(f"unknown canonical Stage-1 hypothesis: {hypothesis}")
    assert_model_alpha_firewall(names)
    full = feature_matrix(row)
    indexes = [FEATURE_NAMES.index(name) for name in names]
    return full[:, :, indexes]


def boundary_stable_mask(
    row: dict[str, Any],
    margins: dict[str, float],
    *,
    starting_cash: float = 10_000.0,
    contract_multiplier: float = 100.0,
) -> np.ndarray:
    """Return the historical side of the frozen boundary-stable guard."""
    ladder = np.asarray(row.get("option_ladder"), dtype=float)
    option_names = tuple(row.get("feature_names") or ())
    ids = np.asarray(row.get("contract_ids"), dtype=object)
    metadata = row.get("contract_quote_metadata") or {}
    out = np.zeros(ladder.shape[:2], dtype=bool)
    indexes = {name: option_names.index(name) for name in ("bid", "ask", "mid")}
    for strike_idx in range(ladder.shape[0]):
        for right_idx in range(ladder.shape[1]):
            item = metadata.get(str(ids[strike_idx, right_idx])) or {}
            bid = _finite(item.get("bid"))
            ask = _finite(item.get("ask"))
            mid = _finite(item.get("mid"))
            if bid is None:
                bid = _finite(ladder[strike_idx, right_idx, indexes["bid"]])
            if ask is None:
                ask = _finite(ladder[strike_idx, right_idx, indexes["ask"]])
            if mid is None:
                mid = _finite(ladder[strike_idx, right_idx, indexes["mid"]])
            if None in (bid, ask, mid):
                continue
            spread = _finite(item.get("spread"))
            if spread is None:
                spread = float(ask - bid)
            spread_frac = _finite(item.get("spread_frac"))
            if spread_frac is None and mid:
                spread_frac = float(spread / abs(mid))
            quote_age = _finite(item.get("quote_age_ms"))
            bid_size = _finite(item.get("bid_size"))
            ask_size = _finite(item.get("ask_size"))
            utilization = ask * contract_multiplier / starting_cash
            out[strike_idx, right_idx] = bool(
                mid >= margins["stable_min_mid"]
                and mid <= margins["stable_max_mid"]
                and spread is not None
                and spread <= margins["stable_max_spread_abs"]
                and spread_frac is not None
                and spread_frac <= margins["stable_max_spread_frac"]
                and quote_age is not None
                and quote_age <= margins["stable_max_quote_age_ms"]
                and bid_size is not None
                and bid_size >= margins["stable_min_bid_size"]
                and ask_size is not None
                and ask_size >= margins["stable_min_ask_size"]
                and utilization <= margins["max_affordability_utilization"]
            )
    return out & np.asarray(row.get("candidate_mask"), dtype=bool)


def assert_model_alpha_firewall(feature_names: tuple[str, ...] | list[str]) -> None:
    """Reject wildcard, future/path, exit, quote-age, and PnL feature inputs."""

    assert_alpha_feature_names(
        feature_names,
        allowed_feature_sets=HYPOTHESES.values(),
        boundary="Protocol101 canonical Stage-1 model matrix construction",
    )


def assert_contract() -> None:
    if len(FEATURE_NAMES) != 17 or len(set(FEATURE_NAMES)) != 17:
        raise AssertionError("scoped canonical Stage-1 feature list must contain 17 unique names")
    for name in FEATURE_NAMES:
        if any(token in name.lower() for token in QUARANTINED_ALPHA_TOKENS):
            raise AssertionError(f"quarantined alpha leaked into contract: {name}")
    if set(HYPOTHESES) != {"H0", "H1", "H2", "H3"}:
        raise AssertionError("unexpected Stage-1 hypothesis set")
    for names in HYPOTHESES.values():
        assert_model_alpha_firewall(names)
    if not {
        "labels_net_pnl",
        "labels_mid_pnl",
        "label_exit_quote_age_ms",
        "label_realized_exit_time_ns",
        "label_source_exit_quote_time_ns",
    } <= FORBIDDEN_MODEL_FIELDS:
        raise AssertionError("signed exit/PnL fields missing from alpha firewall")


assert_contract()
