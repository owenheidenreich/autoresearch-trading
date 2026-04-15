"""Replay scoring: computes performance metrics from simulated trades.

Score = (0.5*sortino + 0.5*dollar_pf) * positive_day_rate * dd_mult

All P&L metrics are DOLLAR-WEIGHTED (entry_price * contract_multiplier * qty).
This was fixed 2026-04-15 after discovering that percentage-weighted PF
masked an 81.5% portfolio loss as near-breakeven (PF 0.974 vs real 0.830).

See v2/docs/incidents/2026-04-15-pf-metric-bug.md for the full diagnosis.
See v2/docs/evaluator.md for the promotion score formula.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np

from v2.core.schema import SimulatedTrade


@dataclass
class ReplayMetrics:
    """Aggregate metrics from a set of simulated trades."""

    # Core (dollar-weighted — the primary economic truth)
    profit_factor: float = 0.0          # dollar-weighted PF (PRIMARY)
    win_rate: float = 0.0
    total_trades: int = 0
    trades_per_day: float = 0.0
    num_days: int = 0

    # Dollar P&L (primary)
    gross_profit: float = 0.0           # sum of winning dollar P&Ls
    gross_loss: float = 0.0             # abs(sum of losing dollar P&Ls)
    net_pnl: float = 0.0               # net dollar P&L

    # Percentage P&L (secondary diagnostic)
    pct_profit_factor: float = 0.0      # unweighted percentage PF (diagnostic only)
    pct_gross_profit: float = 0.0
    pct_gross_loss: float = 0.0
    pct_net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Account curve (dollar-based)
    starting_equity: float = 10_000.0
    net_pnl_dollars: float = 0.0
    daily_returns: list[float] = field(default_factory=list)
    positive_day_rate: float = 0.0
    daily_sortino: float = 0.0
    max_account_drawdown: float = 0.0
    traded_days: int = 0

    # Risk
    max_drawdown: float = 0.0  # per-trade equity curve drawdown (legacy)
    sharpe: float = 0.0

    # Direction balance
    call_count: int = 0
    put_count: int = 0
    call_pct: float = 0.0
    put_pct: float = 0.0
    minority_side_share: float = 0.0
    direction_balance: float = 0.0

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

    # Dollar-weighted metrics (each trade weighted by notional)
    dollar_weighted_win_rate: float = 0.0
    dollar_weighted_profit_factor: float = 0.0

    # Promotion
    score: float = 0.0
    gate_failure: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def compute_metrics(
    trades: list[SimulatedTrade],
    num_days: int = 1,
    starting_equity: float = 10_000.0,
    contract_multiplier: int = 100,
) -> ReplayMetrics:
    """Compute aggregate metrics from a list of simulated trades."""
    m = ReplayMetrics()
    m.num_days = max(num_days, 1)
    m.starting_equity = starting_equity

    if not trades:
        return m

    m.total_trades = len(trades)
    m.trades_per_day = m.total_trades / m.num_days

    # --- Validate qty on every trade (prevent silent zeroing) ---
    for i, t in enumerate(trades):
        if t.intent.qty < 1:
            import warnings
            warnings.warn(
                f"Trade {i} has intent.qty={t.intent.qty} — using qty=1 as fallback. "
                f"This is a bug in the caller; fix the TradeIntent construction.",
                stacklevel=2,
            )

    # --- Dollar P&L (PRIMARY — the economic truth) ---
    dollar_pnls = [
        t.net_pnl_pct * t.entry_price * contract_multiplier * max(t.intent.qty, 1)
        for t in trades
    ]
    dollar_wins = [p for p in dollar_pnls if p > 0]
    dollar_losses = [p for p in dollar_pnls if p <= 0]

    m.gross_profit = sum(dollar_wins) if dollar_wins else 0.0
    m.gross_loss = abs(sum(dollar_losses)) if dollar_losses else 0.0
    m.net_pnl = sum(dollar_pnls)

    if m.gross_loss > 0:
        m.profit_factor = m.gross_profit / m.gross_loss
    elif m.gross_profit > 0:
        m.profit_factor = 10.0
    else:
        m.profit_factor = 0.0

    # --- Percentage P&L (secondary diagnostic) ---
    pct_pnls = [t.net_pnl_pct for t in trades]
    pct_wins = [p for p in pct_pnls if p > 0]
    pct_losses = [p for p in pct_pnls if p <= 0]

    m.pct_gross_profit = sum(pct_wins) if pct_wins else 0.0
    m.pct_gross_loss = abs(sum(pct_losses)) if pct_losses else 0.0
    m.pct_net_pnl = sum(pct_pnls)

    if m.pct_gross_loss > 0:
        m.pct_profit_factor = m.pct_gross_profit / m.pct_gross_loss
    elif m.pct_gross_profit > 0:
        m.pct_profit_factor = 10.0
    else:
        m.pct_profit_factor = 0.0

    m.win_rate = len(pct_wins) / m.total_trades if m.total_trades > 0 else 0.0
    m.avg_win = float(np.mean(pct_wins)) if pct_wins else 0.0
    m.avg_loss = float(np.mean(pct_losses)) if pct_losses else 0.0

    # Per-trade equity curve drawdown (legacy, uses percentage returns)
    equity = [1.0]
    for p in pct_pnls:
        equity.append(equity[-1] * (1.0 + p))
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    m.max_drawdown = max_dd

    # Sharpe (annualized from per-trade percentage returns)
    if len(pct_pnls) > 1:
        arr = np.array(pct_pnls)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1))
        if std_ret > 0:
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
    m.minority_side_share = min(m.call_pct, m.put_pct) if m.total_trades > 0 else 0.0
    dir_majority = max(m.call_pct, m.put_pct, 0.01)
    m.direction_balance = m.minority_side_share / dir_majority if m.total_trades > 0 else 0.0

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
    m.avg_bars_held = float(np.mean([t.bars_held for t in trades]))
    m.avg_mfe = float(np.mean([t.mfe_pct for t in trades]))
    m.avg_mae = float(np.mean([t.mae_pct for t in trades]))

    # Dollar-weighted metrics (now consistent with primary PF above)
    notionals = np.array([t.entry_price * contract_multiplier * max(t.intent.qty, 1) for t in trades])
    dw_pnls = np.array([t.net_pnl_pct * n for t, n in zip(trades, notionals)])
    total_notional = notionals.sum()
    if total_notional > 0:
        win_notional = sum(n for t, n in zip(trades, notionals) if t.net_pnl_pct > 0)
        m.dollar_weighted_win_rate = win_notional / total_notional
    m.dollar_weighted_profit_factor = m.profit_factor  # now identical to primary

    # --- Account curve (dollar-based) ---
    _compute_account_curve(m, trades, starting_equity, contract_multiplier)

    # Score
    m.score = compute_score(m)

    return m


def _compute_account_curve(
    m: ReplayMetrics,
    trades: list[SimulatedTrade],
    starting_equity: float,
    contract_multiplier: int,
) -> None:
    """Build daily equity curve and compute account-level metrics.

    Groups trades by trade_date, computes dollar P&L per day,
    builds equity curve, derives Sortino and drawdown.
    """
    # Group trades by date
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        date = t.trade_date
        if not date:
            continue
        # Dollar P&L = pnl_pct * entry_price * contract_multiplier * qty
        dollar_pnl = t.net_pnl_pct * t.entry_price * contract_multiplier * max(t.intent.qty, 1)
        daily_pnl[date] += dollar_pnl

    if not daily_pnl:
        return

    # Sort by date
    sorted_dates = sorted(daily_pnl.keys())
    m.traded_days = len(sorted_dates)

    # Build equity curve
    equity = [starting_equity]
    daily_dollar_pnls = []
    for date in sorted_dates:
        dpnl = daily_pnl[date]
        daily_dollar_pnls.append(dpnl)
        equity.append(equity[-1] + dpnl)

    m.net_pnl_dollars = equity[-1] - starting_equity

    # Daily returns (as fraction of starting equity for Sortino)
    m.daily_returns = [dpnl / starting_equity for dpnl in daily_dollar_pnls]

    # Positive day rate
    positive_days = sum(1 for r in m.daily_returns if r > 0)
    m.positive_day_rate = positive_days / m.traded_days if m.traded_days > 0 else 0.0

    # Max account drawdown (peak-to-trough on equity curve)
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
    m.max_account_drawdown = max_dd

    # Daily Sortino ratio (annualized)
    if len(m.daily_returns) > 1:
        arr = np.array(m.daily_returns)
        mean_daily = float(np.mean(arr))
        # Downside deviation: std of returns below zero only
        downside = arr[arr < 0]
        if len(downside) > 0:
            downside_std = float(np.std(downside, ddof=1))
        else:
            # No negative days: perfect but cap Sortino
            downside_std = 0.0

        if downside_std > 0:
            # Annualize: sqrt(252 trading days)
            m.daily_sortino = (mean_daily / downside_std) * np.sqrt(252)
        elif mean_daily > 0:
            m.daily_sortino = 6.0  # cap when no downside days
        else:
            m.daily_sortino = 0.0


def compute_score(metrics: ReplayMetrics) -> float:
    """Compute the promotion score.

    score = (0.5 * sortino_term + 0.5 * pf_term) * positive_day_rate * dd_mult

    Composite of sortino (risk-adjusted consistency) and profit factor
    (magnitude of winners vs losers). This rewards both steady returns AND
    big wins, instead of penalizing profitable volatility.

    Hard gates return negative scores on failure.
    """
    # --- Hard gates ---
    if metrics.total_trades < 30:
        metrics.gate_failure = f"too_few_trades ({metrics.total_trades} < 30)"
        return -1.0

    if metrics.traded_days < 15:
        metrics.gate_failure = f"too_few_traded_days ({metrics.traded_days} < 15)"
        return -0.5

    if metrics.max_account_drawdown > 0.25:
        metrics.gate_failure = f"excessive_drawdown ({metrics.max_account_drawdown:.1%} > 25%)"
        return -0.2

    metrics.gate_failure = None

    # --- Ranking ---
    sortino_term = min(metrics.daily_sortino, 10.0)
    pf_term = min(metrics.profit_factor, 4.0)
    pdr = metrics.positive_day_rate

    # DD multiplier: 1.0 at <= 12%, linear decay to 0.0 at 25%
    dd = metrics.max_account_drawdown
    if dd <= 0.12:
        dd_mult = 1.0
    else:
        dd_mult = max(0.0, 1.0 - (dd - 0.12) / 0.13)

    score = (0.5 * sortino_term + 0.5 * pf_term) * pdr * dd_mult
    return round(score, 6)


# ---------------------------------------------------------------------------
# Score config fingerprint
# ---------------------------------------------------------------------------

_SCORE_CONFIG = {
    "version": "v4.0_dollar_weighted_pf",
    "primary": "(0.5*sortino + 0.5*dollar_pf) * positive_day_rate * dd_mult",
    "sortino_cap": 10.0,
    "pf_cap": 4.0,
    "starting_equity": 10_000,
    "contract_multiplier": 100,
    "gate_min_trades": 30,
    "gate_min_traded_days": 15,
    "gate_max_drawdown": 0.25,
    "dd_mult_free_below": 0.12,
    "dd_mult_zero_above": 0.25,
}


def score_config_fingerprint() -> str:
    """SHA-256 of score config. Changes reset best_score."""
    payload = json.dumps(_SCORE_CONFIG, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
