"""Diagnostic: Oracle Replay — what score does PERFECT contract selection achieve?

This answers: if the model always picked the oracle's best contract, would it
pass the evaluation gates? If not, the gates are structurally impossible.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from v2.core.chain_data import (
    extract_contract_series,
    load_sidecar_cached,
    padded_snapshot,
)
from v2.core.metrics import ReplayMetrics, compute_metrics, compute_score
from v2.core.policy import DEFAULT_POLICY
from v2.core.schema import TradeIntent
from v2.core.simulator import simulate_trade
from v2.core.walkforward import generate_folds
from v2.train import LOOKBACK


def oracle_replay(data: dict, mask_key: str = "promote_mask", policy=DEFAULT_POLICY) -> tuple[ReplayMetrics, list]:
    """Replay using oracle's best contract selection (perfect foresight)."""
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    spot_prices = data["spot_prices"].numpy()
    sim_features = data["X_sim"].numpy() if "X_sim" in data else data["X"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    mask = data[mask_key].numpy()

    mask_indices = np.where(mask)[0]
    eval_dates = sorted(set(dates[i] for i in mask_indices))

    day_to_bars: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        day_to_bars.setdefault(d, []).append(i)

    trades = []
    current_day = None
    in_trade = False
    trade_exit_bar = -1
    last_stop_bar = -policy.cooldown_bars - 1
    daily_dollar_pnl = 0.0
    daily_loss_cap_hit = False
    cumulative_equity = policy.starting_equity
    num_days = 0
    total_bars = 0
    trade_bars = 0
    skipped_no_label = 0

    for day in eval_dates:
        sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))

        for bar_idx in day_to_bars.get(day, []):
            if bar_idx < LOOKBACK:
                continue
            bod = int(bar_of_day[bar_idx])
            if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
                continue

            if day != current_day:
                current_day = day
                num_days += 1
                in_trade = False
                trade_exit_bar = -1
                last_stop_bar = -policy.cooldown_bars - 1
                daily_dollar_pnl = 0.0
                daily_loss_cap_hit = False

            total_bars += 1

            if cumulative_equity <= 0 or daily_loss_cap_hit:
                continue
            if in_trade and bar_idx <= trade_exit_bar:
                continue
            if bar_idx - last_stop_bar < policy.cooldown_bars:
                continue

            # Oracle selection: use the sidecar's best contract
            if not bool(sidecar["bar_label_trade"][bod]):
                skipped_no_label += 1
                continue

            # bar_best_contract_idx is RELATIVE to the bar's snapshot, not absolute
            best_row_in_bar = int(sidecar["bar_best_contract_idx"][bod])
            if best_row_in_bar < 0:
                continue

            # Map relative index to absolute contract index
            ptr_start = int(sidecar["bar_ptrs"][bod])
            ptr_end = int(sidecar["bar_ptrs"][bod + 1])
            if ptr_start + best_row_in_bar >= ptr_end:
                continue
            best_contract_idx = int(sidecar["row_contract_idx"][ptr_start + best_row_in_bar])

            trade_bars += 1

            # Get contract info
            contract_strike = float(sidecar["contract_strike"][best_contract_idx])
            raw_right = int(sidecar["contract_right"][best_contract_idx])
            contract_right = "P" if raw_right == 1 else "C"
            series = extract_contract_series(sidecar, best_contract_idx)
            entry_mid = float(series["mid"][bod])

            if entry_mid <= 0 or not np.isfinite(entry_mid):
                continue

            intent = TradeIntent(
                trade=True,
                expiry=sidecar.get("date", day),
                strike=contract_strike,
                right=contract_right,
                qty=policy.qty,
                entry_ref_price=entry_mid,
                order_style=policy.order_style,
                tif=policy.tif,
                stop_price=max(0.01, entry_mid * (1.0 - policy.stop_pct)),
                take_profit_price=entry_mid * (1.0 + policy.target_pct),
                max_hold_bars=policy.max_hold_bars,
                exit_policy=policy.exit_policy,
                confidence=1.0,
                reason_codes=("oracle",),
                bar_index=bod,
                timestamp="",
                underlying_price=float(spot_prices[bar_idx]),
                decision_day=day,
                contract_index=best_contract_idx,
            )

            mid_series = series["mid"].astype(np.float32)
            trade = simulate_trade(
                intent=intent,
                option_prices=mid_series,
                features=sim_features[day_to_bars[day]],
                bar_of_day=np.arange(len(day_to_bars[day]), dtype=np.int32),
                dates=[day] * len(day_to_bars[day]),
                global_entry_bar=bod,
            )
            if trade is None:
                continue

            trade.trade_date = day
            trades.append(trade)
            in_trade = True
            trade_exit_bar = bar_idx + (trade.exit_bar - bod)
            dollar_pnl = trade.net_pnl_pct * trade.entry_price * policy.contract_multiplier * policy.qty
            daily_dollar_pnl += dollar_pnl
            cumulative_equity += dollar_pnl
            if daily_dollar_pnl < 0 and abs(daily_dollar_pnl) / policy.starting_equity >= policy.daily_loss_cap_pct:
                daily_loss_cap_hit = True
            if trade.exit_reason == "STOP_LOSS":
                last_stop_bar = trade_exit_bar

    metrics = compute_metrics(trades, num_days=max(num_days, 1), starting_equity=policy.starting_equity, contract_multiplier=policy.contract_multiplier)
    score = compute_score(metrics)

    print(f"\n{'='*60}")
    print(f"  ORACLE REPLAY DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"  Eval days:     {num_days}")
    print(f"  Eligible bars: {total_bars}")
    print(f"  Oracle trades: {trade_bars} (skipped {skipped_no_label} no-label bars)")
    print(f"  Executed:      {len(trades)} trades")
    print(f"  Win rate:      {metrics.win_rate:.1%}")
    print(f"  Profit factor: {metrics.profit_factor:.3f}")
    print(f"  Net P&L:       ${metrics.net_pnl_dollars:,.2f}")
    print(f"  Max DD:        {metrics.max_account_drawdown:.1%}")
    print(f"  Sortino:       {metrics.daily_sortino:.2f}")
    print(f"  +Day rate:     {metrics.positive_day_rate:.1%}")
    print(f"  Calls/Puts:    {metrics.call_count}C / {metrics.put_count}P")
    dir_balance = min(metrics.call_pct, metrics.put_pct) / max(metrics.call_pct, metrics.put_pct, 0.01)
    print(f"  Dir balance:   {dir_balance:.2f}")
    print(f"  Gate failure:  {metrics.gate_failure or 'NONE'}")
    print(f"  SCORE:         {score:.4f}")
    print(f"{'='*60}")

    if score > 0:
        print("  VERDICT: Oracle passes gates → model training is the bottleneck")
    elif metrics.gate_failure and "drawdown" in metrics.gate_failure:
        print("  VERDICT: Oracle FAILS on drawdown → 20% DD gate may be too tight")
    elif metrics.gate_failure and "direction" in metrics.gate_failure:
        print("  VERDICT: Oracle FAILS on direction → labels are one-sided")
    elif metrics.gate_failure and "trades" in metrics.gate_failure:
        print("  VERDICT: Oracle FAILS on trade count → not enough tradeable bars")
    else:
        print(f"  VERDICT: Oracle fails with score {score:.4f} → evaluation may be structurally broken")

    return metrics, trades


def main():
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    print("Loaded data.pt")
    print(f"Dataset: {len(data['dates'])} bars, fingerprint={data['metadata']['fingerprint']}")

    # Run on promote mask (same as 1-fold test window)
    print("\n--- Promote mask (out-of-sample test window) ---")
    oracle_replay(data, mask_key="promote_mask")

    # Also run on train mask to see oracle performance on training data
    print("\n--- Train mask (in-sample, for comparison) ---")
    oracle_replay(data, mask_key="train_mask")


if __name__ == "__main__":
    main()
