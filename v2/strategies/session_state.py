"""Per-session state for Fork A1's SPX-only proxy of Pickles' Row 1.

Each session resets at RTH open (bar_of_day == 0). All quantities are computed
**point-in-time** — a decision at bar_of_day B can only consult session state
through bar_of_day B-1 (running VWAP / σ-band / ovn proxy use prior-session
closes; first-15m stats become available only once bar_of_day ≥ 15).

This module intentionally does **not** import ``v2/core/policy.py`` constants
(``NO_TRADE_BEFORE_BAR``, ``NO_TRADE_AFTER_BAR``, ``MIN_HOLD_BARS``, etc).
Fork A1 defines its own window and lifecycle — inheriting the default policy
would silently redefine the strategy (see plan: "Do not inherit
``DEFAULT_POLICY``").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Fork-A1 trading window (bar-of-day indices, RTH-only, 0 = 09:30 ET).
# Window intentionally differs from ``DEFAULT_POLICY`` — we need bars 15-29
# (09:45-10:00 ET) which the default mask blocks.
WINDOW_FIRST_BAR = 15   # 09:45 ET — first 15m just closed
WINDOW_LAST_BAR = 150   # 12:00 ET — pre-lunch cutoff

FIRST15_BARS = 15
BARS_PER_DAY = 390

# σ-band lookback — rolling stdev of (spot - vwap)/spot across prior sessions.
SIGMA_LOOKBACK_SESSIONS = 30


@dataclass
class SessionState:
    """Mutable per-session state consumed by ``pickles_row1.decide``.

    Fields are populated as the session unfolds. Consumers must respect the
    bar-of-day ordering: e.g. ``first15_close_position`` is **None** until
    bar-of-day 14 has closed (i.e. consulted only at bar_of_day >= 15).
    """

    date: str
    # Session open values (set at bar 0).
    session_open_spot: float = 0.0
    prior_session_close_spot: float = 0.0
    # Running session VWAP (recomputed each bar from cumulative spot*volume).
    session_vwap: float = 0.0
    # First-15m cache (populated at bar_of_day == 15).
    first15_high: float = 0.0
    first15_low: float = 0.0
    first15_close: float = 0.0
    first15_open: float = 0.0
    first15_close_position: float | None = None
    first15_ready: bool = False
    # σ-band scalar for this session (precomputed from prior sessions).
    vwap_sigma_frac: float = 0.0
    # OVN-proxy direction (precomputed at session open from prior-close vs
    # current-open). +1 / 0 / -1 sign. Labeled "proxy" because it is NOT
    # Pickles' real OVN inventory classification (that requires ES/ETH data).
    ovn_proxy_direction: int = 0
    # Per-day trade lifecycle.
    trades_taken: int = 0
    # Rolling buffer of the prior ``VWAP_RALLY_LOOKBACK_BARS`` session bars'
    # vwap_dist (fractional signed) — used by the revised entry trigger which
    # fires when the session has been > +10 bps above VWAP within the lookback,
    # not just on the immediately prior bar. Populated by the runner.
    recent_vwap_dist_buffer: list[float] = field(default_factory=list)
    recent_window_ready: bool = False

    def reset_for_day(self, date: str, prior_session_close_spot: float) -> None:
        self.date = date
        self.prior_session_close_spot = float(prior_session_close_spot)
        self.session_open_spot = 0.0
        self.session_vwap = 0.0
        self.first15_high = 0.0
        self.first15_low = 0.0
        self.first15_close = 0.0
        self.first15_open = 0.0
        self.first15_close_position = None
        self.first15_ready = False
        self.ovn_proxy_direction = 0
        self.trades_taken = 0
        self.recent_vwap_dist_buffer = []
        self.recent_window_ready = False
        # vwap_sigma_frac is set by the runner (depends on prior-session
        # history; computed once per day).


# -------------------------------------------------------------------------
# Running-VWAP computation (point-in-time, no look-ahead).
# -------------------------------------------------------------------------

def compute_session_vwap(spot_prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Cumulative volume-weighted mean price from session open through each bar.

    ``spot_prices`` and ``volumes`` are 1-D arrays for a single RTH session
    (length ``BARS_PER_DAY``). Returns an array of the same length where
    element ``k`` is the session VWAP at the close of bar ``k``.

    This matches the computation in ``v2/pipeline/compute_features.py:311-318``
    for the ``vwap_dist`` feature and is safe to call per day in a backtest.
    """
    vol = np.maximum(volumes.astype(np.float64), 0.0)
    cum_pv = np.cumsum(spot_prices.astype(np.float64) * vol)
    cum_v = np.cumsum(vol)
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-9), spot_prices.astype(np.float64))
    return vwap


# -------------------------------------------------------------------------
# σ-band precompute (rolling stdev of intraday VWAP deviations across prior
# sessions). One scalar per trading day.
# -------------------------------------------------------------------------

def compute_daily_vwap_sigma(
    spot_by_day: list[np.ndarray],
    vol_by_day: list[np.ndarray],
    lookback: int = SIGMA_LOOKBACK_SESSIONS,
) -> np.ndarray:
    """Rolling per-day σ of ``(spot - vwap) / spot`` over prior ``lookback`` sessions.

    ``spot_by_day[i]`` is the spot array for session i; ``vol_by_day[i]`` is
    the matching volume array. Returns ``np.array`` of length ``len(spot_by_day)``
    where element ``i`` is the σ applicable to **session i** (computed from
    prior sessions only — point-in-time).

    σ for session i is NaN if fewer than 5 prior sessions are available.
    """
    n_days = len(spot_by_day)
    sigmas = np.full(n_days, np.nan, dtype=np.float64)

    # Collect per-session intraday deviation samples, then use a rolling window
    # of the most recent ``lookback`` sessions' samples.
    prior_samples: list[np.ndarray] = []
    for i in range(n_days):
        if len(prior_samples) >= 5:
            # Take the last ``lookback`` sessions' samples, pooled.
            pool = prior_samples[-lookback:]
            flat = np.concatenate(pool) if pool else np.zeros(0)
            if flat.size >= 5:
                sigmas[i] = float(np.nanstd(flat))
        # Now append this session's samples for use by future sessions.
        spot = np.asarray(spot_by_day[i], dtype=np.float64)
        vol = np.asarray(vol_by_day[i], dtype=np.float64)
        if spot.size == 0:
            prior_samples.append(np.zeros(0))
            continue
        vwap = compute_session_vwap(spot, vol)
        with np.errstate(invalid="ignore", divide="ignore"):
            dev = np.where(spot > 0, (spot - vwap) / spot, np.nan)
        prior_samples.append(dev[np.isfinite(dev)])

    return sigmas


# -------------------------------------------------------------------------
# Half-hour / news cool-down gate (clock-only approximation).
# -------------------------------------------------------------------------

# bars are 1 minute wide; RTH starts at bar 0 = 09:30 ET. A :00 or :30 clock
# mark falls at any bar whose minute-of-RTH is congruent to 0 or 30 modulo 30.
# minute_of_rth = bar_of_day (since bars are minute-aligned). A bar at minute m
# is <= 5 minutes before the next :00 or :30 iff ``(30 - m % 30) <= 5``.

COOLDOWN_WINDOW_BARS = 5  # ±5 min window around :00 / :30


def in_halfhour_cooldown(bar_of_day: int, window: int = COOLDOWN_WINDOW_BARS) -> bool:
    """Return True if the bar is within ``window`` minutes before a :00 or :30
    clock mark (and therefore inside Pickles' stated no-entry gate)."""
    # RTH open is 09:30; bar 0 = 09:30 (clock minute 30 of hour 9).
    # We care about clock time, not bar index. minute-of-hour at bar_of_day k:
    # (30 + k) % 60.
    minute_of_hour = (30 + int(bar_of_day)) % 60
    # Distance to next :00 or :30 boundary (in minutes, forward):
    dist_to_next_half = (30 - (minute_of_hour % 30)) % 30
    # Special case: exactly on the boundary → dist 0, definitely a "blackout"
    # minute for the "few minutes before" framing. Treat 0 as in-cooldown.
    if dist_to_next_half == 0:
        return True
    return dist_to_next_half <= window
