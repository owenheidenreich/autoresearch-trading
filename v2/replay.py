"""ART² v2 Replay: evaluate a trained model by simulating trades on validation data.

Usage:
    python -m v2.replay [--model v2/model.pt] [--data v2/data.pt] [--mask promote]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid

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

    # Risk parameters (sigmoid squashed to policy ranges)
    risk = outputs['risk']
    stop_lo, stop_hi = policy.stop_range
    target_lo, target_hi = policy.target_range
    hold_lo, hold_hi = policy.max_hold_range

    stop_pct = torch.sigmoid(risk[0]).item() * (stop_hi - stop_lo) + stop_lo
    target_pct = torch.sigmoid(risk[1]).item() * (target_hi - target_lo) + target_lo
    hold_frac = torch.sigmoid(risk[2]).item()
    max_hold = max(hold_lo, int(hold_frac * hold_hi))

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

    all_trades = []
    num_days = 0

    for day in eval_dates:
        # Get bars for this day
        day_bars = [i for i in range(len(dates)) if dates[i] == day]
        if len(day_bars) < LOOKBACK + 10:
            continue

        num_days += 1
        expiry = day.replace("-", "")

        # State for this day
        in_trade = False
        trade_exit_bar = -1
        last_stop_bar = -policy.cooldown_bars - 1

        for bar_idx in day_bars:
            bod = int(bar_of_day[bar_idx])

            # Skip if can't form lookback window
            if bar_idx < LOOKBACK:
                continue

            # Skip if outside trading window (from policy)
            if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
                continue

            # Skip if in position or cooldown
            if in_trade and bar_idx <= trade_exit_bar:
                continue
            in_trade = False

            if bar_idx - last_stop_bar < policy.cooldown_bars:
                continue

            # Get feature window
            window = features[bar_idx - LOOKBACK:bar_idx]
            if window.shape[0] != LOOKBACK:
                continue

            x = torch.from_numpy(window).unsqueeze(0).to(device)

            # Model inference
            with torch.no_grad():
                outputs = model(x)
                outputs = {k: v.squeeze(0) for k, v in outputs.items()}

            spot = float(spot_prices[bar_idx])
            if spot <= 0 or np.isnan(spot):
                continue

            # Get ATM option price for entry reference
            atm_call_px = float(data.get('atm_call_prices', torch.zeros(1))[bar_idx]) if 'atm_call_prices' in data else 0
            atm_put_px = float(data.get('atm_put_prices', torch.zeros(1))[bar_idx]) if 'atm_put_prices' in data else 0
            option_mid = max(atm_call_px, atm_put_px, 0.5)  # fallback

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
            in_trade = False

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
# Baselines (computed on the same mask as the model for fair comparison)
# ---------------------------------------------------------------------------

def compute_baseline_random(
    data: dict,
    mask_key: str = "promote_mask",
    n_seeds: int = 20,
    max_days: int | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
) -> ReplayMetrics:
    """Random baseline: 50% chance to trade at each bar, random candidate."""
    features = data['X'].numpy()
    mask = data[mask_key].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    all_pfs = []
    all_wrs = []
    all_trades_count = []

    option_keys = [k for k in data.keys() if k.endswith('_prices') and k != 'spot_prices']
    option_keys = [k for k in option_keys if 'call_prices' in k or 'put_prices' in k]

    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days:
        eval_dates = eval_dates[:max_days]

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        all_trades = []
        num_days = 0

        for day in eval_dates:
            day_bars = [i for i in range(len(dates)) if dates[i] == day]
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
                    all_trades.append(trade)
                    last_exit = trade.exit_bar

        m = compute_metrics(
            all_trades, num_days=max(num_days, 1),
            starting_equity=policy.starting_equity,
            contract_multiplier=policy.contract_multiplier,
        )
        all_pfs.append(m.profit_factor)
        all_wrs.append(m.win_rate)
        all_trades_count.append(m.total_trades)

    avg = ReplayMetrics()
    avg.profit_factor = float(np.mean(all_pfs)) if all_pfs else 0
    avg.win_rate = float(np.mean(all_wrs)) if all_wrs else 0
    avg.total_trades = int(np.mean(all_trades_count)) if all_trades_count else 0
    avg.num_days = len(eval_dates)
    avg.trades_per_day = avg.total_trades / max(avg.num_days, 1)
    return avg


def compute_baseline_atm_always(
    data: dict,
    mask_key: str = "promote_mask",
    max_days: int | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
) -> ReplayMetrics:
    """ATM-always baseline: buy ATM call at bar 30 every day."""
    features = data['X'].numpy()
    mask = data[mask_key].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days:
        eval_dates = eval_dates[:max_days]

    all_trades = []
    num_days = 0

    for day in eval_dates:
        day_bars = [i for i in range(len(dates)) if dates[i] == day]
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
) -> ReplayMetrics:
    """Simple rules: buy call on +momentum, put on -momentum."""
    features = data['X'].numpy()
    mask = data[mask_key].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()
    ret_idx = _FEAT_IDX.get('ret_6', 0)

    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))
    if max_days:
        eval_dates = eval_dates[:max_days]

    all_trades = []
    num_days = 0

    for day in eval_dates:
        day_bars = [i for i in range(len(dates)) if dates[i] == day]
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
            print(f"  Falling back to val_mask (old 2-way split)")
            mask_key = 'val_mask'
        else:
            return

    policy = DEFAULT_POLICY
    if args.gate is not None:
        # Create a new policy with overridden gate threshold
        policy = DecisionPolicy(gate_threshold=args.gate)

    if args.baselines:
        print(f"\n--- COMPUTING BASELINES (on {mask_key}) ---")
        b_random = compute_baseline_random(data, mask_key=mask_key, max_days=args.days, policy=policy)
        print_metrics("Random Baseline", b_random)

        b_atm = compute_baseline_atm_always(data, mask_key=mask_key, max_days=args.days, policy=policy)
        print_metrics("ATM-Always Baseline", b_atm)

        b_rules = compute_baseline_simple_rules(data, mask_key=mask_key, max_days=args.days, policy=policy)
        print_metrics("Simple-Rules Baseline", b_rules)
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
    b_random = compute_baseline_random(data, mask_key=mask_key, max_days=args.days, policy=policy)
    print_metrics("Random", b_random)

    b_atm = compute_baseline_atm_always(data, mask_key=mask_key, max_days=args.days, policy=policy)
    print_metrics("ATM-Always", b_atm)

    b_rules = compute_baseline_simple_rules(data, mask_key=mask_key, max_days=args.days, policy=policy)
    print_metrics("Simple-Rules", b_rules)

    print(f"\n--- COMPARISON ---")
    print(f"  Model Score={metrics.score:.4f} vs Random={b_random.score:.4f} "
          f"ATM={b_atm.score:.4f} Rules={b_rules.score:.4f}")
    beats_random = metrics.score > b_random.score
    beats_atm = metrics.score > b_atm.score
    beats_rules = metrics.score > b_rules.score
    print(f"  Beats random: {'YES' if beats_random else 'NO'}")
    print(f"  Beats ATM-always: {'YES' if beats_atm else 'NO'}")
    print(f"  Beats simple-rules: {'YES' if beats_rules else 'NO'}")


if __name__ == "__main__":
    main()
