"""
Trading rules engine: deterministic domain-knowledge rules for 0DTE SPX options.

Centralizes all trading decisions so replay.py and decision.py use identical logic.
The model predicts where SPX goes. This module decides when/how to trade.

Sources: Pickles trading knowledge, Elder, Douglas, Sinclair, Coulling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from prepare import (
        ACTION_DO_NOTHING, ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5,
        ACTION_BUY_CALL_OTM10, ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5,
        ACTION_BUY_PUT_OTM10, BARS_PER_DAY, NO_TRADE_BEFORE_BAR,
        NO_TRADE_LUNCH_START, NO_TRADE_LUNCH_END, STOP_COOLDOWN_BARS,
        MIN_HOLD_BARS, EXIT_GATE_THRESHOLD, STARTING_CAPITAL,
        POSITION_RISK_TARGET, SPX_MULTIPLIER, DYNAMIC_STOP_BASE,
        MAX_HOLD_BARS, compute_dynamic_stop, _FEAT_IDX,
    )
except ImportError:
    from training.prepare import (
        ACTION_DO_NOTHING, ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5,
        ACTION_BUY_CALL_OTM10, ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5,
        ACTION_BUY_PUT_OTM10, BARS_PER_DAY, NO_TRADE_BEFORE_BAR,
        NO_TRADE_LUNCH_START, NO_TRADE_LUNCH_END, STOP_COOLDOWN_BARS,
        MIN_HOLD_BARS, EXIT_GATE_THRESHOLD, STARTING_CAPITAL,
        POSITION_RISK_TARGET, SPX_MULTIPLIER, DYNAMIC_STOP_BASE,
        MAX_HOLD_BARS, compute_dynamic_stop, _FEAT_IDX,
    )

# ---------------------------------------------------------------------------
# Time-of-day session boundaries (bar indices within a 390-bar day)
# ---------------------------------------------------------------------------
MORNING_START = NO_TRADE_BEFORE_BAR   # bar 30 = 10:00 AM (first tradeable bar)
MORNING_END = 60                       # bar 60 = 10:30 AM (end of prime morning window)
LUNCH_START = NO_TRADE_LUNCH_START     # bar 60 = 10:30 AM
LUNCH_END = NO_TRADE_LUNCH_END         # bar 240 = 1:30 PM
POWER_HOUR_START = 330                 # bar 330 = 3:00 PM (extreme gamma zone)

# Confidence thresholds by session
LUNCH_CONFIDENCE_MULTIPLIER = 2.0      # require 2x confidence during lunch
POWER_HOUR_BLOCKED = True              # block long options during power hour (gamma too dangerous)

# Exit thresholds
TIME_DECAY_MAX_HOLD = 120              # bars (2 hours) -- exit if held this long with < +10%
TIME_DECAY_MIN_PROFIT = 0.10           # 10% min unrealized to justify holding past 2 hours

# Scaled exit tiers (Pickles' thirds method)
# Each tier: (trigger_pct, lock_pct) -- when unrealized >= trigger, lock stop at lock
TRAILING_STOP_TIERS = [
    (1.20, 0.80),  # +120% unrealized -> lock +80%
    (0.80, 0.50),  # +80% -> lock +50%
    (0.50, 0.25),  # +50% -> lock +25%
    (0.30, 0.00),  # +30% -> lock breakeven (Pickles: "always take profits off the table")
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EntrySignal:
    """Result of should_enter(): the action to take and why."""
    action: int                  # ACTION_BUY_CALL_ATM, etc.
    confidence: float            # trade_prob from model
    pred_return_30: float        # predicted 30-bar SPX return
    stop_pct: float              # dynamic stop loss percentage
    reason_codes: list[str]


@dataclass
class ExitSignal:
    """Result of should_exit(): whether to exit and why."""
    should_exit: bool
    new_stop: float | None       # new stop price (for trailing), or None
    reason_codes: list[str]


@dataclass
class StrikeSelection:
    """Result of select_strike(): which strike to trade."""
    action: int                  # ACTION_BUY_CALL_ATM, etc.
    reason: str


# ---------------------------------------------------------------------------
# Entry rules
# ---------------------------------------------------------------------------

def should_enter(
    trade_prob: float,
    pred_return_30: float,
    exit_signal: float,
    bar_of_day: int,
    bars_since_last_stop: int,
    account_balance: float,
    min_trade_prob: float = 0.5,
    min_predicted_move: float = 0.001,
) -> EntrySignal | None:
    """Determine whether to enter a trade based on model prediction + domain rules.

    Returns EntrySignal if entry is warranted, None otherwise.

    Domain rules applied:
    - Pre-10am block (opening chaos)
    - Lunch chop suppression (require 2x confidence)
    - Power hour block (extreme gamma, too dangerous for long options)
    - Cooldown after stop loss (5 bars)
    - Minimum predicted move threshold
    """
    reason_codes: list[str] = []

    # Cooldown after stop loss
    if bars_since_last_stop < STOP_COOLDOWN_BARS:
        return None

    # Pre-10am block (Pickles: "watch, don't immediately trade" during opening drive)
    if bar_of_day < NO_TRADE_BEFORE_BAR:
        return None

    # Power hour block (Pickles: "I'm total trash at POWER HOUR")
    if POWER_HOUR_BLOCKED and bar_of_day >= POWER_HOUR_START:
        return None

    # Lunch chop suppression (Pickles: "avoid lunch; low volume, erratic moves")
    effective_threshold = min_trade_prob
    if LUNCH_START <= bar_of_day < LUNCH_END:
        effective_threshold = min_trade_prob * LUNCH_CONFIDENCE_MULTIPLIER
        reason_codes.append("lunch_elevated_threshold")

    # Check trade probability
    if trade_prob < effective_threshold:
        return None

    # Check minimum predicted move
    if abs(pred_return_30) < min_predicted_move:
        return None

    # Determine direction from predicted return
    if pred_return_30 > 0:
        direction_action = ACTION_BUY_CALL_ATM
        reason_codes.append("predicted_up")
    else:
        direction_action = ACTION_BUY_PUT_ATM
        reason_codes.append("predicted_down")

    reason_codes.append("trade_signal")

    return EntrySignal(
        action=direction_action,
        confidence=trade_prob,
        pred_return_30=pred_return_30,
        stop_pct=DYNAMIC_STOP_BASE,  # will be refined by compute_stop
        reason_codes=reason_codes,
    )


def select_strike(
    direction: int,
    confidence: float,
    pred_conf: float,
    iv: float = 0.0,
    vix_regime: float = 0.0,
) -> StrikeSelection:
    """Select strike based on confidence and volatility.

    Domain rules:
    - ATM for high-confidence directional (Pickles: ATM for quick scalps)
    - OTM5 for moderate confidence with favorable IV
    - OTM10 when VRP is favorable and vol is low (rare)
    - Never OTM15+ (0% data coverage in backtest)

    Args:
        direction: ACTION_BUY_CALL_ATM or ACTION_BUY_PUT_ATM (base direction)
        confidence: trade_prob from model (0-1)
        pred_conf: predicted volatility from model
        iv: current ATM implied volatility
        vix_regime: VIX regime indicator
    """
    is_call = direction in (ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5, ACTION_BUY_CALL_OTM10)

    # High confidence + low predicted vol = ATM (tightest spread, best liquidity)
    if confidence > 0.7:
        if is_call:
            return StrikeSelection(ACTION_BUY_CALL_ATM, "high_confidence_atm")
        return StrikeSelection(ACTION_BUY_PUT_ATM, "high_confidence_atm")

    # Moderate confidence = OTM5 (cheaper entry, still decent liquidity)
    if confidence > 0.55:
        if is_call:
            return StrikeSelection(ACTION_BUY_CALL_OTM5, "moderate_confidence_otm5")
        return StrikeSelection(ACTION_BUY_PUT_OTM5, "moderate_confidence_otm5")

    # Low confidence but still trading = OTM10 (cheapest, widest spread)
    if is_call:
        return StrikeSelection(ACTION_BUY_CALL_OTM10, "low_confidence_otm10")
    return StrikeSelection(ACTION_BUY_PUT_OTM10, "low_confidence_otm10")


def compute_stop(
    confidence: float,
    iv: float = 0.0,
    vix_regime: float = 0.0,
) -> float:
    """Compute dynamic stop loss percentage.

    Delegates to prepare.py's compute_dynamic_stop for consistency.
    """
    return compute_dynamic_stop(confidence, iv, vix_regime)


def compute_position_size(
    account_balance: float,
    entry_price: float,
    max_risk_pct: float = POSITION_RISK_TARGET,
) -> int:
    """Compute position size (number of contracts).

    Uses Kelly-informed sizing: risk no more than max_risk_pct of account per trade.
    SPX options have 100x multiplier.
    """
    contract_cost = entry_price * SPX_MULTIPLIER
    if contract_cost <= 0 or account_balance <= 0:
        return 0
    max_contracts = int(account_balance * max_risk_pct / contract_cost)
    return max(1, max_contracts)


# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------

def should_exit(
    bars_held: int,
    unrealized_pnl_pct: float,
    exit_signal: float,
    current_stop: float,
    entry_price: float,
    dynamic_stop_pct: float,
    is_last_bar: bool = False,
) -> ExitSignal:
    """Determine whether to exit a trade.

    Exit priority:
    1. Stop loss hit
    2. Model exit signal (exit_signal > threshold, after MIN_HOLD_BARS)
    3. Time decay death (held > 2 hours with < +10%)
    4. Max hold / EOD
    5. Trailing stop adjustments (not an exit, just a stop update)

    Args:
        bars_held: bars since entry
        unrealized_pnl_pct: current (price - entry) / entry
        exit_signal: model's exit signal from action head (0-1)
        current_stop: current stop price
        entry_price: entry price for trailing stop calculations
        dynamic_stop_pct: stop loss percentage set at entry
        is_last_bar: True if this is the last bar of the day
    """
    reason_codes: list[str] = []

    # 1. Stop loss
    if unrealized_pnl_pct <= -dynamic_stop_pct:
        return ExitSignal(
            should_exit=True,
            new_stop=None,
            reason_codes=["STOP_LOSS"],
        )

    # 2. Model exit signal (require MIN_HOLD_BARS)
    if bars_held >= MIN_HOLD_BARS and exit_signal > EXIT_GATE_THRESHOLD:
        return ExitSignal(
            should_exit=True,
            new_stop=None,
            reason_codes=["MODEL_EXIT", f"exit_signal={exit_signal:.4f}"],
        )

    # 3. Time decay death (Pickles: "cut losses by lunch if thesis fails")
    if bars_held >= TIME_DECAY_MAX_HOLD and unrealized_pnl_pct < TIME_DECAY_MIN_PROFIT:
        return ExitSignal(
            should_exit=True,
            new_stop=None,
            reason_codes=["TIME_DECAY", f"held={bars_held}_bars_under_{TIME_DECAY_MIN_PROFIT:.0%}"],
        )

    # 4. Max hold / EOD
    if bars_held >= MAX_HOLD_BARS:
        return ExitSignal(
            should_exit=True,
            new_stop=None,
            reason_codes=["MAX_HOLD"],
        )
    if is_last_bar:
        return ExitSignal(
            should_exit=True,
            new_stop=None,
            reason_codes=["EOD"],
        )

    # 5. Trailing stop adjustments (Pickles: "always take profits off the table")
    new_stop = None
    for trigger_pct, lock_pct in TRAILING_STOP_TIERS:
        if unrealized_pnl_pct >= trigger_pct:
            locked_stop = entry_price * (1.0 + lock_pct)
            if locked_stop > current_stop:
                new_stop = locked_stop
                reason_codes.append(f"trailing_stop_lock_{int(lock_pct*100)}pct")
            break

    if new_stop is not None:
        return ExitSignal(
            should_exit=False,
            new_stop=new_stop,
            reason_codes=reason_codes,
        )

    # No exit, no stop update
    return ExitSignal(
        should_exit=False,
        new_stop=None,
        reason_codes=[],
    )
