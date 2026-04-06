"""ART² v2 Replay: evaluate a trained model by simulating trades on validation data.

Usage:
    python -m v2.replay [--model v2/model.pt] [--data v2/data.pt] [--mask promote]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict

import numpy as np
import torch

from v2.core.schema import TradeIntent
from v2.core.features import (
    BARS_PER_DAY, NUM_FEATURES, _FEAT_IDX,
    NO_TRADE_BEFORE_BAR, NO_TRADE_AFTER_BAR,
    STOP_COOLDOWN_BARS,
    compute_adaptive_spread_bps,
)
from v2.core.policy import DecisionPolicy, DEFAULT_POLICY
from v2.core.simulator import simulate_trade
from v2.core.metrics import compute_metrics, ReplayMetrics
from v2.train import TradingModel, LOOKBACK, STRIKE_OFFSETS, STRIKE_OFFSET_TO_IDX

BATCH_SIZE = 4096
BASELINE_CACHE_PATH = "v2/.baseline_cache.json"


def load_model(path: str, device: str = "cpu") -> TradingModel:
    """Load a trained model from checkpoint.

    TODO(Phase 3b): Replace with artifact bundle loading that reconstructs
    from saved spec and hard-fails on mismatch.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TradingModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def model_to_intent(
    outputs: dict[str, torch.Tensor],
    bar_idx: int,
    bar_of_day: int,
    spot_price: float,
    option_mid: float,
    expiry: str,
    policy: DecisionPolicy = DEFAULT_POLICY,
    timestamp: str = "",
) -> TradeIntent:
    """Convert model outputs to a TradeIntent using the shared policy."""
    gate_prob = torch.sigmoid(outputs['gate']).item()

    if gate_prob < policy.gate_threshold:
        return TradeIntent.no_trade(
            bar_index=bar_of_day,
            timestamp=timestamp,
            reason_codes=('gate_below_threshold',),
        )

    # Direction: call or put
    dir_probs = torch.softmax(outputs['direction'], dim=-1)
    dir_class = dir_probs.argmax(dim=-1).item()
    right = "C" if dir_class == 0 else "P"

    # Strike offset
    strike_probs = torch.softmax(outputs['strike'], dim=-1)
    strike_class = strike_probs.argmax(dim=-1).item()
    strike_offset = STRIKE_OFFSETS[min(strike_class, len(STRIKE_OFFSETS) - 1)]

    # Round spot to nearest 5 and apply offset
    atm = round(spot_price / 5.0) * 5.0
    strike = atm + strike_offset

    # Risk parameters: clamp raw outputs to policy ranges (no sigmoid)
    risk = outputs['risk']
    stop_lo, stop_hi = policy.stop_range
    target_lo, target_hi = policy.target_range
    hold_lo, hold_hi = policy.max_hold_range

    stop_pct = torch.clamp(risk[0], stop_lo, stop_hi).item()
    target_pct = torch.clamp(risk[1], target_lo, target_hi).item()
    hold_raw = torch.clamp(risk[2], 0.0, 1.0).item()
    max_hold = max(hold_lo, int(hold_raw * hold_hi))

    # Convert to prices
    stop_price = option_mid * (1.0 - stop_pct)
    tp_price = option_mid * (1.0 + target_pct)

    confidence = torch.sigmoid(outputs['confidence']).item()

    return TradeIntent(
        trade=True,
        expiry=expiry,
        strike=strike,
        right=right,
        qty=policy.qty,
        entry_ref_price=option_mid,
        order_style=policy.order_style,
        tif="DAY",
        stop_price=max(0.01, stop_price),
        take_profit_price=tp_price,
        max_hold_bars=max_hold,
        exit_policy=policy.exit_policy,
        confidence=confidence,
        reason_codes=(f"gate={gate_prob:.2f}", f"dir={right}", f"offset={strike_offset}"),
        bar_index=bar_of_day,
        timestamp=timestamp,
        intent_id=str(uuid.uuid4()),
        underlying_price=spot_price,
    )


def _build_day_index(dates) -> dict[str, list[int]]:
    """Build a dict mapping date string -> list of global bar indices."""
    day_to_bars: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        day_to_bars[d].append(i)
    return day_to_bars


def replay_validation(
    model: TradingModel,
    data: dict,
    mask_key: str = "promote_mask",
    max_days: int | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
    device: str = "cpu",
) -> tuple[ReplayMetrics, list]:
    """Run replay on data selected by mask_key.

    Args:
        model: trained TradingModel
        data: loaded dataset dict
        mask_key: which mask to use ("val_mask", "promote_mask", "shadow_mask")
        max_days: limit number of evaluation days
        policy: DecisionPolicy controlling gate threshold, risk ranges, etc.
        device: torch device

    Returns (metrics, trades_list).
    """
    features = data['X'].numpy()
    mask = data[mask_key].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    # Find evaluation day boundaries
    mask_indices = np.where(mask)[0]
    if len(mask_indices) == 0:
        return ReplayMetrics(), []

    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days is not None:
        eval_dates = eval_dates[:max_days]

    model = model.to(device)
    model.eval()

    # Precompute day index
    day_to_bars = _build_day_index(dates)

    # --- Pass 1: collect all eligible bar indices ---
    t_start = time.time()
    eligible_bars = []  # list of (day, bar_idx, bar_of_day_value)

    for day in eval_dates:
        day_bars = day_to_bars.get(day, [])
        if len(day_bars) < LOOKBACK + 10:
            continue
        for bar_idx in day_bars:
            if bar_idx < LOOKBACK:
                continue
            bod = int(bar_of_day[bar_idx])
            if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
                continue
            eligible_bars.append((day, bar_idx, bod))

    # --- Pass 2: build lookback windows and batch inference ---
    n_bars = len(eligible_bars)
    if n_bars == 0:
        return ReplayMetrics(), []

    # Build all windows at once using numpy for speed
    window_indices = np.array([b[1] for b in eligible_bars])
    # Create index arrays for gathering: for each bar, we need [bar_idx-LOOKBACK : bar_idx]
    # Shape: (n_bars, LOOKBACK)
    offsets = np.arange(-LOOKBACK, 0).reshape(1, -1)  # (1, LOOKBACK)
    gather_idx = window_indices.reshape(-1, 1) + offsets  # (n_bars, LOOKBACK)
    # Clip to valid range (shouldn't be needed given bar_idx >= LOOKBACK check, but safety)
    gather_idx = np.clip(gather_idx, 0, len(features) - 1)

    # Gather all windows: (n_bars, LOOKBACK, NUM_FEATURES)
    all_windows = features[gather_idx]

    # Batched inference
    all_outputs_list = []
    t_inf_start = time.time()
    with torch.no_grad():
        for start in range(0, n_bars, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_bars)
            batch = torch.from_numpy(all_windows[start:end]).to(device)
            batch_out = model(batch)
            # Move to CPU and store
            all_outputs_list.append({k: v.cpu() for k, v in batch_out.items()})

    # Concatenate all batch outputs
    all_outputs = {}
    if all_outputs_list:
        keys = all_outputs_list[0].keys()
        for k in keys:
            all_outputs[k] = torch.cat([o[k] for o in all_outputs_list], dim=0)

    t_inf = time.time() - t_inf_start
    print(f"  Inference: {t_inf:.1f}s ({n_bars} bars)")

    # --- Pass 3: sequential trade simulation using pre-computed model outputs ---
    t_sim_start = time.time()

    all_trades = []
    num_days = 0
    current_day = None

    # Per-day state
    in_trade = False
    trade_exit_bar = -1
    last_stop_bar = -1000

    for i, (day, bar_idx, bod) in enumerate(eligible_bars):
        # Reset state on day boundary
        if day != current_day:
            current_day = day
            num_days += 1
            in_trade = False
            trade_exit_bar = -1
            last_stop_bar = -policy.cooldown_bars - 1

        # Skip if in position
        if in_trade:
            if bar_idx <= trade_exit_bar:
                continue
            else:
                in_trade = False

        if bar_idx - last_stop_bar < policy.cooldown_bars:
            continue

        # Extract this bar's outputs from the batched result
        outputs = {k: v[i] for k, v in all_outputs.items()}

        spot = float(spot_prices[bar_idx])
        if spot <= 0 or np.isnan(spot):
            continue

        # Get ATM option price for entry reference
        atm_call_px = float(data.get('atm_call_prices', torch.zeros(1))[bar_idx]) if 'atm_call_prices' in data else 0
        atm_put_px = float(data.get('atm_put_prices', torch.zeros(1))[bar_idx]) if 'atm_put_prices' in data else 0
        option_mid = max(atm_call_px, atm_put_px)
        if option_mid <= 0:
            continue  # no valid ATM price, skip bar

        expiry = day.replace("-", "")
        intent = model_to_intent(
            outputs, bar_idx, bod, spot, option_mid, expiry,
            policy=policy,
        )

        if not intent.trade:
            continue

        # Find the option price array for this intent
        price_key = _intent_to_price_key(intent)
        if price_key not in data:
            continue

        option_prices = data[price_key].numpy().astype(np.float32)

        trade = simulate_trade(
            intent=intent,
            option_prices=option_prices,
            features=features,
            bar_of_day=bar_of_day,
            dates=dates,
            global_entry_bar=bar_idx,
        )

        if trade is None:
            continue

        all_trades.append(trade)
        in_trade = True
        trade_exit_bar = trade.exit_bar

        if trade.exit_reason == "STOP_LOSS":
            last_stop_bar = trade.exit_bar

    t_sim = time.time() - t_sim_start
    print(f"  Simulation: {t_sim:.1f}s ({len(all_trades)} trades)")

    metrics = compute_metrics(
        all_trades,
        num_days=max(num_days, 1),
        starting_equity=policy.starting_equity,
        contract_multiplier=policy.contract_multiplier,
    )
    return metrics, all_trades


def _intent_to_price_key(intent: TradeIntent) -> str:
    """Map a TradeIntent to the v1 option price array key."""
    if intent.right == "C":
        side = "call"
    else:
        side = "put"

    offset = abs(intent.strike - (intent.underlying_price or 0))
    # Round to nearest known offset
    if offset < 2.5:
        prefix = "atm"
    elif offset < 7.5:
        prefix = "otm5"
    elif offset < 12.5:
        prefix = "otm10"
    elif offset < 17.5:
        prefix = "otm15"
    elif offset < 22.5:
        prefix = "otm20"
    elif offset < 27.5:
        prefix = "otm25"
    else:
        prefix = "otm30"

    return f"{prefix}_{side}_prices"


# ---------------------------------------------------------------------------
# Baseline caching
# ---------------------------------------------------------------------------

def _baseline_cache_key(data: dict, mask_key: str, policy: DecisionPolicy, max_days: int | None) -> str:
    """Generate a fingerprint for the dataset + policy + mask combo."""
    X = data['X']
    parts = [
        str(X.shape),
        str(int(data[mask_key].sum().item())),
        str(float(X[0, 0].item())) if X.numel() > 0 else "0",
        str(float(X[-1, -1].item())) if X.numel() > 0 else "0",
        mask_key,
        policy.fingerprint(),  # covers ALL policy fields
        str(max_days),
    ]
    fingerprint = "|".join(parts)
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _load_cached_baselines(cache_key: str) -> dict | None:
    """Load baselines from cache if fingerprint matches."""
    if not os.path.exists(BASELINE_CACHE_PATH):
        return None
    try:
        with open(BASELINE_CACHE_PATH, 'r') as f:
            cache = json.load(f)
        if cache.get("cache_key") == cache_key:
            return cache.get("baselines")
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None


def _save_baseline_cache(cache_key: str, baselines: dict):
    """Save baselines to cache file."""
    cache = {
        "cache_key": cache_key,
        "baselines": baselines,
    }
    try:
        with open(BASELINE_CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass  # non-fatal


def _dict_to_replay_metrics(d: dict) -> ReplayMetrics:
    """Reconstruct a ReplayMetrics from a dict (cache loading)."""
    m = ReplayMetrics()
    for k, v in d.items():
        if hasattr(m, k):
            setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# Baselines (computed on the same mask as the model for fair comparison)
# ---------------------------------------------------------------------------

def compute_baseline_random(
    data: dict,
    mask_key: str = "promote_mask",
    n_seeds: int = 5,
    max_days: int | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
    day_to_bars: dict[str, list[int]] | None = None,
) -> ReplayMetrics:
    """Random baseline: 2% chance to trade at each bar, random candidate.

    Aggregates all trades across seeds into one pool then computes metrics
    once, so the score (Sortino, drawdown, etc.) is properly computed.
    """
    features = data['X'].numpy()
    mask = data[mask_key].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    if day_to_bars is None:
        day_to_bars = _build_day_index(dates)

    option_keys = [k for k in data.keys() if k.endswith('_prices') and k != 'spot_prices']
    option_keys = [k for k in option_keys if 'call_prices' in k or 'put_prices' in k]

    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days:
        eval_dates = eval_dates[:max_days]

    all_trades_combined = []
    total_num_days = 0

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        num_days = 0

        for day in eval_dates:
            day_bars = day_to_bars.get(day, [])
            if len(day_bars) < 50:
                continue
            num_days += 1
            expiry = day.replace("-", "")

            last_exit = -policy.cooldown_bars - 1

            for bar_idx in day_bars:
                bod = int(bar_of_day[bar_idx])
                if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
                    continue
                if bar_idx - last_exit < policy.cooldown_bars:
                    continue
                if rng.random() > 0.02:
                    continue

                key = rng.choice(option_keys)
                arr = data[key].numpy().astype(np.float32)
                px = float(arr[bar_idx])
                if np.isnan(px) or px <= 0:
                    continue

                spot = float(spot_prices[bar_idx])
                if np.isnan(spot) or spot <= 0:
                    continue
                right = "C" if "call" in key else "P"
                atm = round(spot / 5.0) * 5.0

                intent = TradeIntent(
                    trade=True, expiry=expiry, strike=atm, right=right,
                    qty=policy.qty,
                    entry_ref_price=px, order_style="MKT", tif="DAY",
                    stop_price=px * 0.70, take_profit_price=px * 1.50,
                    max_hold_bars=120, exit_policy="STOP_TP_TIME",
                    confidence=0.5, reason_codes=("random",),
                    bar_index=bod, intent_id=str(uuid.uuid4()),
                    underlying_price=spot,
                )

                trade = simulate_trade(intent, arr, features, bar_of_day, dates, bar_idx)
                if trade:
                    all_trades_combined.append(trade)
                    last_exit = trade.exit_bar

        total_num_days += num_days

    # Average the day count across seeds for a fair per-day metric
    avg_num_days = max(total_num_days // max(n_seeds, 1), 1)

    return compute_metrics(
        all_trades_combined,
        num_days=avg_num_days,
        starting_equity=policy.starting_equity,
        contract_multiplier=policy.contract_multiplier,
    )


def compute_baseline_atm_always(
    data: dict,
    mask_key: str = "promote_mask",
    max_days: int | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
    day_to_bars: dict[str, list[int]] | None = None,
) -> ReplayMetrics:
    """ATM-always baseline: buy ATM call at bar 30 every day."""
    features = data['X'].numpy()
    mask = data[mask_key].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    if day_to_bars is None:
        day_to_bars = _build_day_index(dates)

    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days:
        eval_dates = eval_dates[:max_days]

    all_trades = []
    num_days = 0

    for day in eval_dates:
        day_bars = day_to_bars.get(day, [])
        if len(day_bars) < 50:
            continue
        num_days += 1
        expiry = day.replace("-", "")

        entry_bar = None
        for b in day_bars:
            if int(bar_of_day[b]) == 30:
                entry_bar = b
                break
        if entry_bar is None:
            continue

        if 'atm_call_prices' not in data:
            continue
        arr = data['atm_call_prices'].numpy().astype(np.float32)
        px = float(arr[entry_bar])
        if np.isnan(px) or px <= 0:
            continue

        spot = float(spot_prices[entry_bar])
        atm = round(spot / 5.0) * 5.0

        intent = TradeIntent(
            trade=True, expiry=expiry, strike=atm, right="C",
            qty=policy.qty,
            entry_ref_price=px, order_style="MKT", tif="DAY",
            stop_price=px * 0.70, take_profit_price=px * 1.50,
            max_hold_bars=120, exit_policy="STOP_TP_TIME",
            confidence=0.5, reason_codes=("atm_always",),
            bar_index=30, intent_id=str(uuid.uuid4()),
            underlying_price=spot,
        )

        trade = simulate_trade(intent, arr, features, bar_of_day, dates, entry_bar)
        if trade:
            all_trades.append(trade)

    return compute_metrics(
        all_trades, num_days=max(num_days, 1),
        starting_equity=policy.starting_equity,
        contract_multiplier=policy.contract_multiplier,
    )


def compute_baseline_simple_rules(
    data: dict,
    mask_key: str = "promote_mask",
    max_days: int | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
    day_to_bars: dict[str, list[int]] | None = None,
) -> ReplayMetrics:
    """Simple rules: buy call on +momentum, put on -momentum."""
    features = data['X'].numpy()
    mask = data[mask_key].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()
    ret_idx = _FEAT_IDX.get('ret_6', 0)

    if day_to_bars is None:
        day_to_bars = _build_day_index(dates)

    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days:
        eval_dates = eval_dates[:max_days]

    all_trades = []
    num_days = 0

    for day in eval_dates:
        day_bars = day_to_bars.get(day, [])
        if len(day_bars) < 50:
            continue
        num_days += 1
        expiry = day.replace("-", "")

        last_exit = -10 - 1

        for bar_idx in day_bars:
            bod = int(bar_of_day[bar_idx])
            if bod < 30 or bod >= 300:
                continue
            if bar_idx - last_exit < 10:
                continue

            momentum = float(features[bar_idx, ret_idx])

            if abs(momentum) < 0.005:
                continue

            right = "C" if momentum > 0 else "P"
            key = "atm_call_prices" if right == "C" else "atm_put_prices"
            if key not in data:
                continue

            arr = data[key].numpy().astype(np.float32)
            px = float(arr[bar_idx])
            if np.isnan(px) or px <= 0:
                continue

            spot = float(spot_prices[bar_idx])
            atm = round(spot / 5.0) * 5.0

            intent = TradeIntent(
                trade=True, expiry=expiry, strike=atm, right=right,
                qty=policy.qty,
                entry_ref_price=px, order_style="MKT", tif="DAY",
                stop_price=px * 0.75, take_profit_price=px * 1.40,
                max_hold_bars=60, exit_policy="STOP_TP_TIME",
                confidence=0.5, reason_codes=("simple_rules",),
                bar_index=bod, intent_id=str(uuid.uuid4()),
                underlying_price=spot,
            )

            trade = simulate_trade(intent, arr, features, bar_of_day, dates, bar_idx)
            if trade:
                all_trades.append(trade)
                last_exit = trade.exit_bar

    return compute_metrics(
        all_trades, num_days=max(num_days, 1),
        starting_equity=policy.starting_equity,
        contract_multiplier=policy.contract_multiplier,
    )


def compute_baseline_atm_trailing(
    data: dict,
    mask_key: str = "promote_mask",
    max_days: int | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
    day_to_bars: dict[str, list[int]] | None = None,
) -> ReplayMetrics:
    """ATM-always with TRAILING exits: isolates neural net value from exit strategy.

    Same as ATM-always but uses the model's TRAILING exit policy and risk params,
    so any difference between this and the model is attributable to the neural net.
    """
    features = data['X'].numpy()
    mask = data[mask_key].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    if day_to_bars is None:
        day_to_bars = _build_day_index(dates)

    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days:
        eval_dates = eval_dates[:max_days]

    all_trades = []
    num_days = 0

    # Use midpoint of policy risk ranges for a fair baseline
    stop_pct = (policy.stop_range[0] + policy.stop_range[1]) / 2.0
    target_pct = (policy.target_range[0] + policy.target_range[1]) / 2.0
    max_hold = (policy.max_hold_range[0] + policy.max_hold_range[1]) // 2

    for day in eval_dates:
        day_bars = day_to_bars.get(day, [])
        if len(day_bars) < 50:
            continue
        num_days += 1
        expiry = day.replace("-", "")

        entry_bar = None
        for b in day_bars:
            if int(bar_of_day[b]) == 30:
                entry_bar = b
                break
        if entry_bar is None:
            continue

        if 'atm_call_prices' not in data:
            continue
        arr = data['atm_call_prices'].numpy().astype(np.float32)
        px = float(arr[entry_bar])
        if np.isnan(px) or px <= 0:
            continue

        spot = float(spot_prices[entry_bar])
        atm = round(spot / 5.0) * 5.0

        intent = TradeIntent(
            trade=True, expiry=expiry, strike=atm, right="C",
            qty=policy.qty,
            entry_ref_price=px, order_style="MKT", tif="DAY",
            stop_price=px * (1.0 - stop_pct),
            take_profit_price=px * (1.0 + target_pct),
            max_hold_bars=max_hold,
            exit_policy="TRAILING",
            confidence=0.5, reason_codes=("atm_trailing",),
            bar_index=30, intent_id=str(uuid.uuid4()),
            underlying_price=spot,
        )

        trade = simulate_trade(intent, arr, features, bar_of_day, dates, entry_bar)
        if trade:
            all_trades.append(trade)

    return compute_metrics(
        all_trades, num_days=max(num_days, 1),
        starting_equity=policy.starting_equity,
        contract_multiplier=policy.contract_multiplier,
    )


def print_metrics(name: str, m: ReplayMetrics):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Score={m.score:.4f}  PF={m.profit_factor:.3f}  WR={m.win_rate:.1%}")
    print(f"  Trades={m.total_trades}  TPD={m.trades_per_day:.2f}  Days={m.num_days}  Traded={m.traded_days}")
    print(f"  Sortino={m.daily_sortino:.2f}  +DayRate={m.positive_day_rate:.1%}  AcctDD={m.max_account_drawdown:.1%}")
    print(f"  Net P&L=${m.net_pnl_dollars:,.2f}  Equity=${m.starting_equity + m.net_pnl_dollars:,.2f}")
    print(f"  C={m.call_count} P={m.put_count}  "
          f"SL={m.stop_loss_count} TP={m.take_profit_count} EOD={m.eod_count}")
    print(f"  Avg hold={m.avg_bars_held:.0f} bars  MFE={m.avg_mfe:.1%}  MAE={m.avg_mae:.1%}")
    if m.gate_failure:
        print(f"  GATE FAILURE: {m.gate_failure}")


def _compute_all_baselines(data, mask_key, max_days, policy, day_to_bars):
    """Compute all four baselines, using cache when available."""
    cache_key = _baseline_cache_key(data, mask_key, policy, max_days)
    cached = _load_cached_baselines(cache_key)
    if cached is not None and "atm_trailing" in cached:
        print("  (baselines loaded from cache)")
        return (
            _dict_to_replay_metrics(cached["random"]),
            _dict_to_replay_metrics(cached["atm_always"]),
            _dict_to_replay_metrics(cached["simple_rules"]),
            _dict_to_replay_metrics(cached["atm_trailing"]),
        )

    t0 = time.time()
    b_random = compute_baseline_random(data, mask_key=mask_key, max_days=max_days, policy=policy, day_to_bars=day_to_bars)
    b_atm = compute_baseline_atm_always(data, mask_key=mask_key, max_days=max_days, policy=policy, day_to_bars=day_to_bars)
    b_rules = compute_baseline_simple_rules(data, mask_key=mask_key, max_days=max_days, policy=policy, day_to_bars=day_to_bars)
    b_trailing = compute_baseline_atm_trailing(data, mask_key=mask_key, max_days=max_days, policy=policy, day_to_bars=day_to_bars)
    t_bl = time.time() - t0
    print(f"  Baselines: {t_bl:.1f}s")

    # Save to cache
    _save_baseline_cache(cache_key, {
        "random": b_random.to_dict(),
        "atm_always": b_atm.to_dict(),
        "simple_rules": b_rules.to_dict(),
        "atm_trailing": b_trailing.to_dict(),
    })

    return b_random, b_atm, b_rules, b_trailing


def main():
    parser = argparse.ArgumentParser(description="v2 replay evaluation")
    parser.add_argument("--model", type=str, default="v2/model.pt")
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--mask", type=str, default="promote",
                        choices=["val", "promote", "shadow"],
                        help="Which mask to evaluate on (default: promote)")
    parser.add_argument("--baselines", action="store_true", help="Compute baselines only")
    parser.add_argument("--gate", type=float, default=None,
                        help="Override gate threshold from policy")
    args = parser.parse_args()

    mask_key = f"{args.mask}_mask"

    data = torch.load(args.data, map_location="cpu", weights_only=False)

    # Check mask exists (backwards compat with old data.pt)
    if mask_key not in data:
        print(f"WARNING: {mask_key} not found in dataset. Available masks:")
        masks = [k for k in data.keys() if k.endswith('_mask')]
        print(f"  {masks}")
        if 'val_mask' in data and mask_key == 'promote_mask':
            print(f"  WARNING: Falling back to val_mask -- scores may be inflated (evaluating on validation data)")
            mask_key = 'val_mask'
        else:
            return

    policy = DEFAULT_POLICY
    if args.gate is not None:
        # Create a new policy with overridden gate threshold
        policy = DecisionPolicy(gate_threshold=args.gate)

    # Precompute day index once for all uses
    dates = data['dates']
    day_to_bars = _build_day_index(dates)

    if args.baselines:
        print(f"\n--- COMPUTING BASELINES (on {mask_key}) ---")
        b_random, b_atm, b_rules, b_trailing = _compute_all_baselines(data, mask_key, args.days, policy, day_to_bars)
        print_metrics("Random Baseline", b_random)
        print_metrics("ATM-Always Baseline", b_atm)
        print_metrics("Simple-Rules Baseline", b_rules)
        print_metrics("ATM-Trailing Baseline", b_trailing)
        return

    if not os.path.exists(args.model):
        print(f"No model at {args.model}. Train first: python -m v2.train")
        return

    model = load_model(args.model)
    print(f"Model loaded from {args.model}")
    print(f"Evaluating on {mask_key}")

    metrics, trades = replay_validation(
        model, data, mask_key=mask_key, max_days=args.days, policy=policy,
    )
    print_metrics("Model Replay", metrics)

    # Compare with baselines
    print(f"\n--- BASELINES (on {mask_key}) ---")
    b_random, b_atm, b_rules, b_trailing = _compute_all_baselines(data, mask_key, args.days, policy, day_to_bars)
    print_metrics("Random", b_random)
    print_metrics("ATM-Always", b_atm)
    print_metrics("Simple-Rules", b_rules)
    print_metrics("ATM-Trailing", b_trailing)

    print(f"\n--- COMPARISON ---")
    print(f"  Model Score={metrics.score:.4f} vs Random={b_random.score:.4f} "
          f"ATM={b_atm.score:.4f} Rules={b_rules.score:.4f} Trailing={b_trailing.score:.4f}")
    beats_random = metrics.score > b_random.score
    beats_atm = metrics.score > b_atm.score
    beats_rules = metrics.score > b_rules.score
    beats_trailing = metrics.score > b_trailing.score
    print(f"  Beats random: {'YES' if beats_random else 'NO'}")
    print(f"  Beats ATM-always: {'YES' if beats_atm else 'NO'}")
    print(f"  Beats simple-rules: {'YES' if beats_rules else 'NO'}")
    print(f"  Beats ATM-trailing: {'YES' if beats_trailing else 'NO'}")


if __name__ == "__main__":
    main()
