"""Trade simulation engine: executes TradeIntents against historical data.

See v2/docs/evaluator.md for the complete simulation rules.
Single source of truth for trade P&L computation.
"""
from __future__ import annotations

import numpy as np

from v2.core.schema import TradeIntent, SimulatedTrade
from v2.core.features import (
    BARS_PER_DAY, STOP_COOLDOWN_BARS, MIN_HOLD_BARS,
    NO_TRADE_BEFORE_BAR, NO_TRADE_AFTER_BAR,
    _FEAT_IDX, compute_adaptive_spread_bps,
)


# Trailing stop tiers: (unrealized_pct_threshold, lock_pct)
# Used by labels.py for oracle label computation — do NOT change without
# also recomputing training labels.
TRAILING_TIERS = [
    (1.20, 0.80),  # +120% unrealized -> lock +80%
    (0.80, 0.50),  # +80% -> lock +50%
    (0.50, 0.25),  # +50% -> lock +25%
    (0.30, 0.00),  # +30% -> lock breakeven
]


def _build_trailing_tiers(
    breakeven_trigger_pct: float,
    extra_tiers: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Build trailing tiers with a custom breakeven trigger threshold.

    The upper tiers (120%->80%, 80%->50%, 50%->25%) are fixed.
    The lowest tier (breakeven lock) uses the provided threshold.
    Extra tiers are inserted and the list is sorted descending by threshold.
    """
    tiers = [
        (1.20, 0.80),
        (0.80, 0.50),
        (0.50, 0.25),
        (breakeven_trigger_pct, 0.00),
    ]
    if extra_tiers:
        tiers.extend(extra_tiers)
        tiers.sort(key=lambda t: -t[0])
    return tiers


def _compute_spread_cost(
    entry_bar_of_day: int,
    exit_bar_of_day: int,
    vix_regime_entry: float,
    vix_regime_exit: float,
    is_otm: bool,
    entry_px: float | None = None,
) -> float:
    """Compute round-trip spread cost as a fraction (not bps).

    Enforces a minimum-tick dollar floor: SPX options have $0.05 minimum
    tick for options under $3.00. The BPS model alone severely understates
    spread costs on cheap options.
    """
    mtc_entry = BARS_PER_DAY - entry_bar_of_day
    mtc_exit = BARS_PER_DAY - exit_bar_of_day
    entry_spread_frac = compute_adaptive_spread_bps(mtc_entry, vix_regime_entry, is_otm) / 10000.0
    exit_spread_frac = compute_adaptive_spread_bps(mtc_exit, vix_regime_exit, is_otm) / 10000.0

    # Floor: minimum tick is $0.05 per side for options under $3.00
    if entry_px is not None and entry_px > 0:
        min_tick = 0.05 if entry_px < 3.00 else 0.10
        min_spread_frac = min_tick / entry_px
        entry_spread_frac = max(entry_spread_frac, min_spread_frac)
        exit_spread_frac = max(exit_spread_frac, min_spread_frac)

    return entry_spread_frac + exit_spread_frac


def simulate_trade(
    intent: TradeIntent,
    option_prices: np.ndarray,
    features: np.ndarray,
    bar_of_day: np.ndarray,
    dates: list[str],
    global_entry_bar: int,
    breakeven_trigger_pct: float | None = None,
    extra_trailing_tiers: tuple[tuple[float, float], ...] = (),
) -> SimulatedTrade | None:
    """Simulate a single TradeIntent against historical price data.

    Args:
        intent: the trade to simulate
        option_prices: (N,) array of option mid-prices for this contract
        features: (N, 39) feature array (for VIX regime lookup)
        bar_of_day: (N,) array of bar-of-day indices (0-389)
        dates: list of date strings per bar
        global_entry_bar: global index where intent was emitted
        breakeven_trigger_pct: override the lowest trailing tier threshold
            (default None uses TRAILING_TIERS as-is, i.e. 0.30)
        extra_trailing_tiers: additional (threshold, lock_pct) tiers to insert

    Returns:
        SimulatedTrade or None if entry fill fails
    """
    if not intent.trade:
        return None

    N = len(option_prices)
    fill_bar = global_entry_bar + 1  # fill at next bar
    if fill_bar >= N:
        return None

    entry_day = dates[global_entry_bar]

    # Entry fill: use next bar's price (MKT fills at ask ~ mid for simulation)
    entry_px = float(option_prices[fill_bar])
    if np.isnan(entry_px) or entry_px <= 0:
        return None

    # Skip penny options: below $0.50 mid, fills are unreliable
    MIN_ENTRY_PRICE = 0.50
    if entry_px < MIN_ENTRY_PRICE:
        return None

    # Compute stop/TP as percentages of entry premium
    stop_pct = (entry_px - intent.stop_price) / entry_px
    tp_pct = (intent.take_profit_price - entry_px) / entry_px

    # Guard: if price moved through stop or TP before fill, skip trade
    if stop_pct <= 0 or tp_pct <= 0:
        return None

    # Track position
    entry_bod = int(bar_of_day[fill_bar])
    is_otm = intent.strike is not None and intent.right is not None and (
        abs(intent.strike - (intent.underlying_price or 0)) > 2.5
    )

    vix_idx = _FEAT_IDX['vix_regime']  # fail loudly if feature index missing
    vix_entry = float(features[fill_bar, vix_idx]) if fill_bar < len(features) else 0.0

    # Track MFE/MAE
    mfe = 0.0
    mae = 0.0
    trailing_stop = -float('inf')  # no trailing stop initially
    exit_bar = fill_bar
    exit_price = entry_px
    exit_reason = "EOD"
    last_valid_px = entry_px

    bars_in_trade = 0
    max_bars = min(intent.max_hold_bars, BARS_PER_DAY)

    for k in range(1, max_bars + 1):
        check = fill_bar + k
        if check >= N or dates[check] != entry_day:
            # End of day
            exit_bar = min(check - 1, N - 1)
            exit_price = last_valid_px
            exit_reason = "EOD"
            break

        px = float(option_prices[check])
        if np.isnan(px) or px <= 0:
            continue

        last_valid_px = px
        bars_in_trade = k
        unrealized = (px - entry_px) / entry_px

        # Track excursions
        mfe = max(mfe, unrealized)
        mae = min(mae, unrealized)

        # Enforce minimum hold period before any exit checks
        if k < MIN_HOLD_BARS:
            continue

        # 1. Stop loss (checked first - highest priority)
        if unrealized <= -stop_pct:
            exit_bar = check
            exit_price = entry_px * (1.0 - stop_pct)  # fill at stop price
            exit_reason = "STOP_LOSS"
            break

        # 2. Take profit
        if unrealized >= tp_pct:
            exit_bar = check
            exit_price = entry_px * (1.0 + tp_pct)  # fill at TP price
            exit_reason = "TAKE_PROFIT"
            break

        # 3. Trailing stop (if exit_policy is TRAILING)
        if intent.exit_policy == "TRAILING":
            _tiers = _build_trailing_tiers(breakeven_trigger_pct, list(extra_trailing_tiers) or None) if breakeven_trigger_pct is not None else TRAILING_TIERS
            for tier_threshold, lock_pct in _tiers:
                if unrealized >= tier_threshold:
                    new_floor = lock_pct
                    if new_floor > trailing_stop:
                        trailing_stop = new_floor
                    break
            if trailing_stop > -float('inf') and unrealized <= trailing_stop:
                exit_bar = check
                exit_price = entry_px * (1.0 + trailing_stop)
                exit_reason = "TRAILING_STOP"
                break

        # 4. Max hold
        if k >= max_bars:
            exit_bar = check
            exit_price = px
            exit_reason = "MAX_HOLD"
            break

        # 5. Last bar of day (bar 389)
        if int(bar_of_day[check]) >= BARS_PER_DAY - 1:
            exit_bar = check
            exit_price = px
            exit_reason = "EOD"
            break
    else:
        exit_bar = fill_bar + bars_in_trade if bars_in_trade > 0 else fill_bar
        exit_price = last_valid_px
        exit_reason = "EOD"

    # Compute P&L
    raw_pnl = (exit_price - entry_px) / entry_px
    exit_bod = int(bar_of_day[min(exit_bar, N - 1)])
    vix_exit = float(features[min(exit_bar, len(features) - 1), vix_idx])
    spread_cost = _compute_spread_cost(entry_bod, exit_bod, vix_entry, vix_exit, is_otm, entry_px=entry_px)
    # Commission: $0.65/leg, 2 legs per round-trip, as fraction of entry premium
    commission_frac = (2 * 0.65) / (entry_px * 100)
    net_pnl = raw_pnl - spread_cost - commission_frac

    underlying_entry = float(features[fill_bar, _FEAT_IDX.get('ret_6', 0)]) if fill_bar < len(features) else 0.0
    underlying_exit = float(features[min(exit_bar, len(features) - 1), _FEAT_IDX.get('ret_6', 0)])

    return SimulatedTrade(
        intent=intent,
        entry_bar=global_entry_bar,
        entry_price=entry_px,
        entry_fill_bar=fill_bar,
        exit_bar=exit_bar,
        exit_price=exit_price,
        exit_reason=exit_reason,
        raw_pnl_pct=raw_pnl,
        spread_cost_pct=spread_cost,
        net_pnl_pct=net_pnl,
        bars_held=exit_bar - fill_bar,
        mfe_pct=mfe,
        mae_pct=mae,
        trade_date=entry_day,
        vix_regime_at_entry=vix_entry,
    )


def simulate_day(
    intents: list[tuple[int, TradeIntent]],
    option_prices_by_intent: dict[str, np.ndarray],
    features: np.ndarray,
    bar_of_day: np.ndarray,
    dates: list[str],
    daily_loss_cap_pct: float = 0.05,
    starting_equity: float = 10_000.0,
    contract_multiplier: int = 100,
    breakeven_trigger_pct: float | None = None,
    extra_trailing_tiers: tuple[tuple[float, float], ...] = (),
) -> list[SimulatedTrade]:
    """Simulate a day of trading from a list of (bar_index, TradeIntent) pairs.

    Enforces:
    - Max 1 concurrent position
    - Cooldown after stop loss
    - Time block restrictions
    - Daily loss cap (skip new entries when cumulative loss exceeds cap)

    Args:
        intents: list of (global_bar_index, TradeIntent), sorted by bar
        option_prices_by_intent: dict mapping intent_id -> option price array
        features: (N, F) feature array
        bar_of_day: (N,) bar-of-day indices
        dates: date strings per bar
        daily_loss_cap_pct: max daily loss as fraction of starting_equity
        starting_equity: account size for loss cap computation
        contract_multiplier: option multiplier (100 for SPX)
        extra_trailing_tiers: additional (threshold, lock_pct) tiers to insert
    """
    trades: list[SimulatedTrade] = []
    in_position = False
    position_exit_bar = -1
    last_stop_bar = -STOP_COOLDOWN_BARS - 1
    daily_dollar_pnl = 0.0

    for global_bar, intent in intents:
        if not intent.trade:
            continue

        # Check time blocks
        bod = int(bar_of_day[global_bar]) if global_bar < len(bar_of_day) else 0
        if bod < NO_TRADE_BEFORE_BAR:
            continue
        if bod >= NO_TRADE_AFTER_BAR:
            continue

        # Check position overlap
        if in_position:
            if global_bar <= position_exit_bar:
                continue
            else:
                in_position = False

        # Check cooldown
        if global_bar - last_stop_bar < STOP_COOLDOWN_BARS:
            continue

        # Check daily loss cap
        if daily_dollar_pnl < 0 and abs(daily_dollar_pnl) / starting_equity >= daily_loss_cap_pct:
            continue

        # Get option prices for this intent
        prices_key = intent.intent_id
        if prices_key not in option_prices_by_intent:
            continue
        option_prices = option_prices_by_intent[prices_key]

        trade = simulate_trade(
            intent=intent,
            option_prices=option_prices,
            features=features,
            bar_of_day=bar_of_day,
            dates=dates,
            global_entry_bar=global_bar,
            breakeven_trigger_pct=breakeven_trigger_pct,
            extra_trailing_tiers=extra_trailing_tiers,
        )

        if trade is None:
            continue

        trades.append(trade)
        in_position = True
        position_exit_bar = trade.exit_bar

        # Accumulate daily dollar P&L for loss cap
        dollar_pnl = trade.net_pnl_pct * trade.entry_price * contract_multiplier * trade.intent.qty
        daily_dollar_pnl += dollar_pnl

        if trade.exit_reason == "STOP_LOSS":
            last_stop_bar = trade.exit_bar

    return trades
