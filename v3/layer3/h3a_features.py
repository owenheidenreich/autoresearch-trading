"""H3a candidate features for L3 oracle trade_state.

Each function is a pure, causal function of pnl history up to bar t — i.e.
f(pnls, t) depends ONLY on pnls[0..t]. Look-ahead audited by
[scripts/look_ahead_audit_h3.py](../../scripts/look_ahead_audit_h3.py) which
proves bit-exact invariance under random mutation of pnls[t+1:].

Discipline anchor: 2026-04-25 oracle-gate label-leakage retraction. Every new
trade_state feature must pass mutate-future-bars audit before integration.

Returns are RAW (unnormalized). Callers (e.g. _build_trade_data) may divide by
a per-trade `denom` (entry premium dollars) for scale-invariance — that's a
constant factor and preserves causality.
"""
from __future__ import annotations

import numpy as np


def realized_vol_10bar(pnls: np.ndarray, t: int) -> float:
    """Rolling std of pnls over last 10 bars (or fewer if early in trade)."""
    start = max(0, t - 9)
    window = pnls[start : t + 1]
    if window.size < 2:
        return 0.0
    return float(np.std(window, ddof=0))


def pnl_velocity_5bar(pnls: np.ndarray, t: int) -> float:
    """Avg per-bar pnl change over last 5 bars: (pnls[t] - pnls[t-5]) / 5,
    clipped to actual lookback when t < 5."""
    lookback = min(5, t)
    if lookback == 0:
        return 0.0
    return float((pnls[t] - pnls[t - lookback]) / lookback)


def mfe_decay_rate(pnls: np.ndarray, t: int) -> float:
    """(current_pnl - mfe_so_far) / max(1, mfe_bar_age).

    Rate of give-back from running peak. 0 if no give-back or no peak yet."""
    history = pnls[: t + 1]
    if history.size == 0:
        return 0.0
    mfe = float(np.max(history))
    mfe_idx = int(np.argmax(history))
    mfe_bar_age = t - mfe_idx
    return float((pnls[t] - mfe) / max(1, mfe_bar_age))


FEATURES = {
    "realized_vol_10bar": realized_vol_10bar,
    "pnl_velocity_5bar": pnl_velocity_5bar,
    "mfe_decay_rate": mfe_decay_rate,
}
