"""Oracle labeler: computes training labels from historical option data.

See docs/v2/labeling.md for the full specification.

At each decision bar, searches candidate trades x risk parameter grid
to find the best executable trade under the evaluator's rules.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from v2.core.schema import TradeIntent
from v2.core.features import (
    BARS_PER_DAY, DYNAMIC_STOP_MIN, DYNAMIC_STOP_MAX,
    NO_TRADE_BEFORE_BAR, NO_TRADE_AFTER_BAR,
    SPREAD_COST_PCT, _FEAT_IDX,
    compute_adaptive_spread_bps,
)


# ---------------------------------------------------------------------------
# Risk parameter search grids
# ---------------------------------------------------------------------------

# Tier 1 (fast): fixed risk
TIER1_STOPS = [0.30]
TIER1_TARGETS = [0.50]
TIER1_MAX_HOLDS = [120]

# Tier 2 (medium): small grid
TIER2_STOPS = [0.20, 0.30, 0.45]
TIER2_TARGETS = [0.30, 0.50, 0.80]
TIER2_MAX_HOLDS = [60, 120]

# Tier 3 (full): large grid
TIER3_STOPS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TIER3_TARGETS = [0.20, 0.30, 0.50, 0.80, 1.20]
TIER3_MAX_HOLDS = [30, 60, 120, 240, 390]


@dataclass
class OracleLabel:
    """Oracle label for a single bar."""
    trade: bool
    right: str | None = None        # "C" or "P"
    strike_offset: int = 0          # offset from ATM in points
    stop_pct: float = 0.0           # optimal stop as fraction of premium
    target_pct: float = 0.0         # optimal target as fraction of premium
    max_hold: int = 0
    best_pnl: float = 0.0           # P&L of the best trade found
    confidence: float = 0.0         # |best_pnl| as proxy for conviction
    num_candidates_searched: int = 0
    num_profitable: int = 0


def _simulate_forward(
    entry_bar: int,
    option_prices: np.ndarray,
    stop_pct: float,
    target_pct: float,
    max_hold: int,
    spread_cost: float,
    dates: list[str],
) -> float:
    """Fast forward simulation of a single trade. Returns net P&L pct."""
    N = len(option_prices)
    fill_bar = entry_bar + 1
    if fill_bar >= N:
        return float('-inf')

    entry_px = float(option_prices[fill_bar])
    if np.isnan(entry_px) or entry_px <= 0:
        return float('-inf')

    entry_day = dates[entry_bar]
    last_valid_px = entry_px

    for k in range(1, min(max_hold + 1, N - fill_bar)):
        check = fill_bar + k
        if check >= N or dates[check] != entry_day:
            break

        px = float(option_prices[check])
        if np.isnan(px) or px <= 0:
            continue

        last_valid_px = px
        unrealized = (px - entry_px) / entry_px

        # Stop loss (checked first)
        if unrealized <= -stop_pct:
            return -stop_pct - spread_cost

        # Take profit
        if unrealized >= target_pct:
            return target_pct - spread_cost

    # EOD / max hold
    raw_pnl = (last_valid_px - entry_px) / entry_px
    return raw_pnl - spread_cost


def compute_oracle_labels(
    features: np.ndarray,
    option_prices: dict[str, np.ndarray],
    dates: list[str],
    bar_of_day: np.ndarray,
    spot_prices: np.ndarray,
    tier: int = 1,
) -> list[OracleLabel]:
    """Compute oracle labels for all bars.

    Args:
        features: (N, 39) feature array
        option_prices: dict mapping price keys to (N,) arrays
            Keys: 'atm_call_prices', 'atm_put_prices',
                  'otm5_call_prices', 'otm5_put_prices', etc.
        dates: date string per bar
        bar_of_day: (N,) bar-of-day indices
        spot_prices: (N,) SPX close prices
        tier: labeling tier (1=fast, 2=medium, 3=full)

    Returns:
        list of OracleLabel, one per bar
    """
    N = len(features)

    # Select search grid
    if tier == 1:
        stops, targets, holds = TIER1_STOPS, TIER1_TARGETS, TIER1_MAX_HOLDS
    elif tier == 2:
        stops, targets, holds = TIER2_STOPS, TIER2_TARGETS, TIER2_MAX_HOLDS
    else:
        stops, targets, holds = TIER3_STOPS, TIER3_TARGETS, TIER3_MAX_HOLDS

    # Map option price keys to (offset, right) pairs
    price_keys = []
    for key, arr in option_prices.items():
        if 'call' in key:
            right = 'C'
        elif 'put' in key:
            right = 'P'
        else:
            continue
        if 'atm' in key:
            offset = 0
        elif 'otm5' in key:
            offset = 5 if right == 'C' else -5
        elif 'otm10' in key:
            offset = 10 if right == 'C' else -10
        elif 'otm15' in key:
            offset = 15 if right == 'C' else -15
        elif 'otm20' in key:
            offset = 20 if right == 'C' else -20
        elif 'otm25' in key:
            offset = 25 if right == 'C' else -25
        elif 'otm30' in key:
            offset = 30 if right == 'C' else -30
        else:
            continue
        price_keys.append((key, offset, right))

    # For tier 1, only use ATM call and put
    if tier == 1:
        price_keys = [(k, o, r) for k, o, r in price_keys if o == 0]

    vix_idx = _FEAT_IDX.get('vix_regime', 18)

    labels: list[OracleLabel] = []

    for i in range(N):
        bod = int(bar_of_day[i])

        # Time block: no trading outside allowed window
        if bod < NO_TRADE_BEFORE_BAR or bod >= NO_TRADE_AFTER_BAR:
            labels.append(OracleLabel(trade=False))
            continue

        # Search for best trade at this bar
        best_pnl = 0.0  # must be positive to trade
        best_right = None
        best_offset = 0
        best_stop = 0.0
        best_target = 0.0
        best_hold = 0
        num_searched = 0
        num_profitable = 0

        # VIX regime for spread cost
        vix_regime = float(features[i, vix_idx]) if i < len(features) else 0.0
        mtc = BARS_PER_DAY - bod

        for key, offset, right in price_keys:
            arr = option_prices[key]
            if i >= len(arr):
                continue
            px = float(arr[i])
            if np.isnan(px) or px <= 0:
                continue

            is_otm = offset != 0
            # Estimate spread cost for this candidate
            entry_spread = compute_adaptive_spread_bps(mtc, vix_regime, is_otm)
            # Rough exit spread (assume exit ~30 bars later)
            exit_mtc = max(mtc - 30, 10)
            exit_spread = compute_adaptive_spread_bps(exit_mtc, vix_regime, is_otm)
            spread_cost = (entry_spread + exit_spread) / 10000.0

            for stop in stops:
                for target in targets:
                    for hold in holds:
                        num_searched += 1
                        pnl = _simulate_forward(
                            i, arr, stop, target, hold, spread_cost, dates,
                        )
                        if pnl > 0:
                            num_profitable += 1
                        if pnl > best_pnl:
                            best_pnl = pnl
                            best_right = right
                            best_offset = offset
                            best_stop = stop
                            best_target = target
                            best_hold = hold

        if best_right is not None and best_pnl > 0:
            labels.append(OracleLabel(
                trade=True,
                right=best_right,
                strike_offset=best_offset,
                stop_pct=best_stop,
                target_pct=best_target,
                max_hold=best_hold,
                best_pnl=best_pnl,
                confidence=min(1.0, best_pnl * 2.0),  # scale P&L to [0,1]
                num_candidates_searched=num_searched,
                num_profitable=num_profitable,
            ))
        else:
            labels.append(OracleLabel(
                trade=False,
                num_candidates_searched=num_searched,
            ))

    return labels


def labels_to_tensors(
    labels: list[OracleLabel],
) -> dict[str, np.ndarray]:
    """Convert oracle labels to numpy arrays for training.

    Returns dict with:
        oracle_trade: (N,) bool
        oracle_right: (N,) int (0=call, 1=put, -1=no trade)
        oracle_strike_offset: (N,) int
        oracle_stop_pct: (N,) float
        oracle_target_pct: (N,) float
        oracle_max_hold: (N,) int
        oracle_confidence: (N,) float
        oracle_pnl: (N,) float
    """
    N = len(labels)
    out = {
        'oracle_trade': np.zeros(N, dtype=bool),
        'oracle_right': np.full(N, -1, dtype=np.int32),
        'oracle_strike_offset': np.zeros(N, dtype=np.int32),
        'oracle_stop_pct': np.zeros(N, dtype=np.float32),
        'oracle_target_pct': np.zeros(N, dtype=np.float32),
        'oracle_max_hold': np.zeros(N, dtype=np.int32),
        'oracle_confidence': np.zeros(N, dtype=np.float32),
        'oracle_pnl': np.zeros(N, dtype=np.float32),
    }

    for i, lab in enumerate(labels):
        out['oracle_trade'][i] = lab.trade
        if lab.trade:
            out['oracle_right'][i] = 0 if lab.right == 'C' else 1
            out['oracle_strike_offset'][i] = lab.strike_offset
            out['oracle_stop_pct'][i] = lab.stop_pct
            out['oracle_target_pct'][i] = lab.target_pct
            out['oracle_max_hold'][i] = lab.max_hold
        out['oracle_confidence'][i] = lab.confidence
        out['oracle_pnl'][i] = lab.best_pnl

    return out


def label_quality_report(labels: list[OracleLabel]) -> dict:
    """Generate quality metrics for oracle labels. See labeling.md."""
    N = len(labels)
    if N == 0:
        return {}

    trade_labels = [l for l in labels if l.trade]
    no_trade_rate = 1.0 - len(trade_labels) / N
    call_count = sum(1 for l in trade_labels if l.right == 'C')
    put_count = sum(1 for l in trade_labels if l.right == 'P')

    return {
        'total_bars': N,
        'trade_bars': len(trade_labels),
        'no_trade_rate': round(no_trade_rate, 4),
        'call_count': call_count,
        'put_count': put_count,
        'direction_balance': round(min(call_count, put_count) / max(call_count, put_count, 1), 4),
        'avg_pnl': round(float(np.mean([l.best_pnl for l in trade_labels])), 6) if trade_labels else 0.0,
        'avg_stop': round(float(np.mean([l.stop_pct for l in trade_labels])), 4) if trade_labels else 0.0,
        'avg_target': round(float(np.mean([l.target_pct for l in trade_labels])), 4) if trade_labels else 0.0,
        'avg_hold': round(float(np.mean([l.max_hold for l in trade_labels])), 1) if trade_labels else 0.0,
    }
