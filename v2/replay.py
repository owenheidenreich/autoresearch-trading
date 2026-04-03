"""ART² v2 Replay: evaluate a trained model by simulating trades on validation data.

Usage:
    python -m v2.replay [--model v2/model.pt] [--data v2/data.pt] [--days N]
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
from v2.core.simulator import simulate_trade
from v2.core.metrics import compute_metrics, ReplayMetrics
from v2.train import TradingModel, LOOKBACK, STRIKE_OFFSETS, STRIKE_OFFSET_TO_IDX


def load_model(path: str, device: str = "cpu") -> TradingModel:
    """Load a trained model."""
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
    timestamp: str = "",
    min_gate_prob: float = 0.5,
) -> TradeIntent:
    """Convert model outputs to a TradeIntent."""
    gate_prob = torch.sigmoid(outputs['gate']).item()

    if gate_prob < min_gate_prob:
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

    # Risk parameters (sigmoid/softplus to constrain ranges)
    risk = outputs['risk']
    stop_pct = torch.sigmoid(risk[0]).item() * 0.55 + 0.10      # [0.10, 0.65]
    target_pct = torch.sigmoid(risk[1]).item() * 1.5 + 0.15      # [0.15, 1.65]
    hold_frac = torch.sigmoid(risk[2]).item()
    max_hold = max(10, int(hold_frac * BARS_PER_DAY))

    # Convert to prices
    stop_price = option_mid * (1.0 - stop_pct)
    tp_price = option_mid * (1.0 + target_pct)

    confidence = torch.sigmoid(outputs['confidence']).item()

    return TradeIntent(
        trade=True,
        expiry=expiry,
        strike=strike,
        right=right,
        qty=1,
        entry_ref_price=option_mid,
        order_style="MKT",
        tif="DAY",
        stop_price=max(0.01, stop_price),
        take_profit_price=tp_price,
        max_hold_bars=max_hold,
        exit_policy="STOP_TP_TIME",
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
    max_days: int | None = None,
    min_gate_prob: float = 0.5,
    device: str = "cpu",
) -> tuple[ReplayMetrics, list]:
    """Run replay on validation data.

    Returns (metrics, trades_list).
    """
    features = data['X'].numpy()
    val_mask = data['val_mask'].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    # Get option price arrays
    option_keys = [k for k in data.keys() if k.endswith('_prices') and k != 'spot_prices']

    # Find validation day boundaries
    val_indices = np.where(val_mask)[0]
    if len(val_indices) == 0:
        return ReplayMetrics(), []

    val_dates = sorted(set(dates[i] for i in val_indices))
    if max_days is not None:
        val_dates = val_dates[:max_days]

    model = model.to(device)
    model.eval()

    all_trades = []
    num_days = 0

    for day in val_dates:
        # Get bars for this day
        day_bars = [i for i in range(len(dates)) if dates[i] == day]
        if len(day_bars) < LOOKBACK + 10:
            continue

        num_days += 1
        expiry = day.replace("-", "")

        # State for this day
        in_trade = False
        trade_exit_bar = -1
        last_stop_bar = -STOP_COOLDOWN_BARS - 1

        for bar_idx in day_bars:
            bod = int(bar_of_day[bar_idx])

            # Skip if can't form lookback window
            if bar_idx < LOOKBACK:
                continue

            # Skip if outside trading window
            if bod < NO_TRADE_BEFORE_BAR or bod >= NO_TRADE_AFTER_BAR:
                continue

            # Skip if in position or cooldown
            if in_trade and bar_idx <= trade_exit_bar:
                continue
            in_trade = False

            if bar_idx - last_stop_bar < STOP_COOLDOWN_BARS:
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
                min_gate_prob=min_gate_prob,
            )

            if not intent.trade:
                continue

            # Find the option price array for this intent
            # Map intent to closest available price key
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

    metrics = compute_metrics(all_trades, num_days=max(num_days, 1))
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


def compute_baseline_random(data: dict, n_seeds: int = 20, max_days: int | None = None) -> ReplayMetrics:
    """Random baseline: 50% chance to trade at each bar, random candidate."""
    features = data['X'].numpy()
    val_mask = data['val_mask'].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    all_pfs = []
    all_wrs = []
    all_trades_count = []

    option_keys = [k for k in data.keys() if k.endswith('_prices') and k != 'spot_prices']
    # Filter to call/put price keys
    option_keys = [k for k in option_keys if 'call_prices' in k or 'put_prices' in k]

    val_indices = np.where(val_mask)[0]
    val_dates = sorted(set(dates[i] for i in val_indices))
    if max_days:
        val_dates = val_dates[:max_days]

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        all_trades = []
        num_days = 0

        for day in val_dates:
            day_bars = [i for i in range(len(dates)) if dates[i] == day]
            if len(day_bars) < 50:
                continue
            num_days += 1
            expiry = day.replace("-", "")

            last_exit = -STOP_COOLDOWN_BARS - 1

            for bar_idx in day_bars:
                bod = int(bar_of_day[bar_idx])
                if bod < NO_TRADE_BEFORE_BAR or bod >= NO_TRADE_AFTER_BAR:
                    continue
                if bar_idx - last_exit < STOP_COOLDOWN_BARS:
                    continue
                if rng.random() > 0.02:  # ~2% chance per bar to trade
                    continue

                # Random candidate
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
                    trade=True, expiry=expiry, strike=atm, right=right, qty=1,
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

        m = compute_metrics(all_trades, num_days=max(num_days, 1))
        all_pfs.append(m.profit_factor)
        all_wrs.append(m.win_rate)
        all_trades_count.append(m.total_trades)

    avg = ReplayMetrics()
    avg.profit_factor = float(np.mean(all_pfs)) if all_pfs else 0
    avg.win_rate = float(np.mean(all_wrs)) if all_wrs else 0
    avg.total_trades = int(np.mean(all_trades_count)) if all_trades_count else 0
    avg.num_days = len(val_dates)
    avg.trades_per_day = avg.total_trades / max(avg.num_days, 1)
    return avg


def compute_baseline_atm_always(data: dict, max_days: int | None = None) -> ReplayMetrics:
    """ATM-always baseline: buy ATM call at bar 30 every day."""
    features = data['X'].numpy()
    val_mask = data['val_mask'].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()

    val_indices = np.where(val_mask)[0]
    val_dates = sorted(set(dates[i] for i in val_indices))
    if max_days:
        val_dates = val_dates[:max_days]

    all_trades = []
    num_days = 0

    for day in val_dates:
        day_bars = [i for i in range(len(dates)) if dates[i] == day]
        if len(day_bars) < 50:
            continue
        num_days += 1
        expiry = day.replace("-", "")

        # Find bar 30 of this day
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
            trade=True, expiry=expiry, strike=atm, right="C", qty=1,
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

    return compute_metrics(all_trades, num_days=max(num_days, 1))


def compute_baseline_simple_rules(data: dict, max_days: int | None = None) -> ReplayMetrics:
    """Simple rules: buy call on +momentum, put on -momentum."""
    features = data['X'].numpy()
    val_mask = data['val_mask'].numpy()
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    spot_prices = data['spot_prices'].numpy()
    ret_idx = _FEAT_IDX.get('ret_6', 0)

    val_indices = np.where(val_mask)[0]
    val_dates = sorted(set(dates[i] for i in val_indices))
    if max_days:
        val_dates = val_dates[:max_days]

    all_trades = []
    num_days = 0

    for day in val_dates:
        day_bars = [i for i in range(len(dates)) if dates[i] == day]
        if len(day_bars) < 50:
            continue
        num_days += 1
        expiry = day.replace("-", "")

        last_exit = -10 - 1  # 10-bar cooldown for rules baseline

        for bar_idx in day_bars:
            bod = int(bar_of_day[bar_idx])
            if bod < 30 or bod >= 300:
                continue
            if bar_idx - last_exit < 10:
                continue

            # 5-bar momentum from ret_6 feature (raw, not normalized)
            momentum = float(features[bar_idx, ret_idx])

            if abs(momentum) < 0.005:  # 0.5% move threshold (raw feature scale)
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
                trade=True, expiry=expiry, strike=atm, right=right, qty=1,
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

    return compute_metrics(all_trades, num_days=max(num_days, 1))


def print_metrics(name: str, m: ReplayMetrics):
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  PF={m.profit_factor:.3f}  WR={m.win_rate:.1%}  Score={m.score:.4f}")
    print(f"  Trades={m.total_trades}  TPD={m.trades_per_day:.2f}  Days={m.num_days}")
    print(f"  DD={m.max_drawdown:.1%}  Sharpe={m.sharpe:.2f}")
    print(f"  C={m.call_count} P={m.put_count}  "
          f"SL={m.stop_loss_count} TP={m.take_profit_count} EOD={m.eod_count}")
    print(f"  Avg hold={m.avg_bars_held:.0f} bars  MFE={m.avg_mfe:.1%}  MAE={m.avg_mae:.1%}")


def main():
    parser = argparse.ArgumentParser(description="v2 replay evaluation")
    parser.add_argument("--model", type=str, default="v2/model.pt")
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--baselines", action="store_true", help="Compute baselines")
    parser.add_argument("--gate", type=float, default=0.5, help="Min gate probability")
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu", weights_only=False)

    if args.baselines:
        print("\n--- COMPUTING BASELINES ---")
        b_random = compute_baseline_random(data, max_days=args.days)
        print_metrics("Random Baseline", b_random)

        b_atm = compute_baseline_atm_always(data, max_days=args.days)
        print_metrics("ATM-Always Baseline", b_atm)

        b_rules = compute_baseline_simple_rules(data, max_days=args.days)
        print_metrics("Simple-Rules Baseline", b_rules)
        return

    if not os.path.exists(args.model):
        print(f"No model at {args.model}. Train first: python -m v2.train")
        return

    model = load_model(args.model)
    print(f"Model loaded from {args.model}")

    metrics, trades = replay_validation(
        model, data, max_days=args.days, min_gate_prob=args.gate,
    )
    print_metrics("Model Replay", metrics)

    # Compare with baselines
    print("\n--- BASELINES ---")
    b_random = compute_baseline_random(data, max_days=args.days)
    print_metrics("Random", b_random)

    b_atm = compute_baseline_atm_always(data, max_days=args.days)
    print_metrics("ATM-Always", b_atm)

    b_rules = compute_baseline_simple_rules(data, max_days=args.days)
    print_metrics("Simple-Rules", b_rules)

    print(f"\n--- COMPARISON ---")
    print(f"  Model PF={metrics.profit_factor:.3f} vs Random={b_random.profit_factor:.3f} "
          f"ATM={b_atm.profit_factor:.3f} Rules={b_rules.profit_factor:.3f}")
    beats_random = metrics.profit_factor > b_random.profit_factor
    beats_atm = metrics.profit_factor > b_atm.profit_factor
    beats_rules = metrics.profit_factor > b_rules.profit_factor
    print(f"  Beats random: {'YES' if beats_random else 'NO'}")
    print(f"  Beats ATM-always: {'YES' if beats_atm else 'NO'}")
    print(f"  Beats simple-rules: {'YES' if beats_rules else 'NO'}")


if __name__ == "__main__":
    main()
