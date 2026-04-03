"""Replay scoring: computes performance metrics from simulated trades.

See docs/v2/evaluator.md for the promotion score formula.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np

from v2.core.schema import SimulatedTrade


@dataclass
class ReplayMetrics:
    """Aggregate metrics from a set of simulated trades."""

    # Core
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    trades_per_day: float = 0.0
    num_days: int = 0

    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    sharpe: float = 0.0

    # Direction balance
    call_count: int = 0
    put_count: int = 0
    call_pct: float = 0.0
    put_pct: float = 0.0

    # Exit reasons
    stop_loss_count: int = 0
    take_profit_count: int = 0
    trailing_stop_count: int = 0
    eod_count: int = 0
    max_hold_count: int = 0

    # Duration
    avg_bars_held: float = 0.0

    # Excursions
    avg_mfe: float = 0.0
    avg_mae: float = 0.0

    # Promotion
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(trades: list[SimulatedTrade], num_days: int = 1) -> ReplayMetrics:
    """Compute aggregate metrics from a list of simulated trades."""
    m = ReplayMetrics()
    m.num_days = max(num_days, 1)

    if not trades:
        return m

    m.total_trades = len(trades)
    m.trades_per_day = m.total_trades / m.num_days

    pnls = [t.net_pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    m.gross_profit = sum(wins) if wins else 0.0
    m.gross_loss = abs(sum(losses)) if losses else 0.0
    m.net_pnl = sum(pnls)

    m.win_rate = len(wins) / m.total_trades if m.total_trades > 0 else 0.0
    m.avg_win = np.mean(wins) if wins else 0.0
    m.avg_loss = np.mean(losses) if losses else 0.0

    # Profit factor
    if m.gross_loss > 0:
        m.profit_factor = m.gross_profit / m.gross_loss
    elif m.gross_profit > 0:
        m.profit_factor = 10.0  # cap at 10 when no losses
    else:
        m.profit_factor = 0.0

    # Max drawdown
    equity = [1.0]
    for p in pnls:
        equity.append(equity[-1] * (1.0 + p))
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    m.max_drawdown = max_dd

    # Sharpe (annualized from per-trade returns)
    if len(pnls) > 1:
        arr = np.array(pnls)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1))
        if std_ret > 0:
            # Annualize: assume ~1.5 trades/day, 252 days/year
            ann_factor = np.sqrt(1.5 * 252)
            m.sharpe = (mean_ret / std_ret) * ann_factor

    # Direction balance
    for t in trades:
        if t.intent.right == "C":
            m.call_count += 1
        elif t.intent.right == "P":
            m.put_count += 1
    m.call_pct = m.call_count / m.total_trades if m.total_trades > 0 else 0.0
    m.put_pct = m.put_count / m.total_trades if m.total_trades > 0 else 0.0

    # Exit reasons
    for t in trades:
        r = t.exit_reason
        if r == "STOP_LOSS":
            m.stop_loss_count += 1
        elif r == "TAKE_PROFIT":
            m.take_profit_count += 1
        elif r == "TRAILING_STOP":
            m.trailing_stop_count += 1
        elif r == "EOD":
            m.eod_count += 1
        elif r == "MAX_HOLD":
            m.max_hold_count += 1

    # Duration and excursions
    m.avg_bars_held = np.mean([t.bars_held for t in trades])
    m.avg_mfe = np.mean([t.mfe_pct for t in trades])
    m.avg_mae = np.mean([t.mae_pct for t in trades])

    # Score
    m.score = compute_score(m)

    return m


def compute_score(metrics: ReplayMetrics) -> float:
    """Compute the promotion score. See docs/v2/evaluator.md.

    Primary: profit factor. Adjusted by win rate, frequency, drawdown, direction balance.
    """
    pf = metrics.profit_factor
    wr = metrics.win_rate
    tpd = metrics.trades_per_day
    dd = metrics.max_drawdown

    # Direction balance: ratio of minority to majority direction
    call_pct = metrics.call_pct
    put_pct = metrics.put_pct
    dir_majority = max(call_pct, put_pct, 0.01)
    dir_minority = min(call_pct, put_pct)
    dir_balance = dir_minority / dir_majority

    # Base: profit factor (must be > 1.0 to be profitable)
    base = max(0.0, pf - 1.0)

    # Win rate bonus: reward consistent winners
    wr_bonus = max(0.0, wr - 0.45) * 2.0

    # Frequency penalty: too few or too many trades
    freq_penalty = 1.0 - min(1.0, abs(tpd - 1.5) / 2.5)

    # Drawdown penalty: severe drawdowns kill the score
    if dd < 0.15:
        dd_penalty = 1.0
    else:
        dd_penalty = max(0.0, 1.0 - (dd - 0.15) * 4.0)

    # Direction collapse penalty
    if dir_balance > 0.2:
        dir_penalty = 1.0
    else:
        dir_penalty = dir_balance / 0.2

    score = base * (1.0 + wr_bonus) * freq_penalty * dd_penalty * dir_penalty

    # Floor: unprofitable strategies get negative scores
    if pf < 1.0:
        score = -(1.0 - pf)

    return round(score, 6)


# ---------------------------------------------------------------------------
# Score config fingerprint
# ---------------------------------------------------------------------------

_SCORE_CONFIG = {
    "base": "pf - 1.0",
    "wr_bonus": "max(0, wr - 0.45) * 2.0",
    "freq_center": 1.5,
    "freq_width": 2.5,
    "dd_threshold": 0.15,
    "dd_slope": 4.0,
    "dir_balance_threshold": 0.2,
    "pf_floor": True,
}


def score_config_fingerprint() -> str:
    """SHA-256 of score config. Changes reset best_score."""
    payload = json.dumps(_SCORE_CONFIG, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
