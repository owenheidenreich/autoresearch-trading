"""Centering ablation: counterfactual replay with 4 centering transforms.

Tests whether per-side centering is the blocker for SIDE_SEL_W's balanced raw scores.
Runs the same model checkpoint with different post-head centering on promote_mask bars,
simulating trades through the full replay pipeline.

Transforms:
  1. per_side_mean  — current production centering (per-side mean subtract)
  2. raw_plus_bias  — raw scores + learned put_bias only (no centering)
  3. global_mean    — global mean centering across all valid contracts
  4. per_side_zscore — per-side z-score: (score - side_mean) / side_std

Usage:
    python3 -m v2.analysis.centering_ablation --model v2/artifacts/exp_170e/model.pt
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
import warnings
from collections import defaultdict

import numpy as np
import torch

from v2.core.chain_data import (
    QUALITY_PARTIAL,
    describe_contract,
    extract_contract_series,
    load_sidecar_cached,
    padded_snapshot,
)
from v2.core.policy import DEFAULT_POLICY, DecisionPolicy
from v2.core.schema import TradeIntent
from v2.core.simulator import simulate_trade
from v2.replay import BATCH_SIZE, LOOKBACK, load_model_from_path, model_to_intent

warnings.filterwarnings("ignore")


def _build_day_index(dates):
    d2b = defaultdict(list)
    for i, d in enumerate(dates):
        d2b[d].append(i)
    return dict(d2b)


def apply_centering(
    call_raw: np.ndarray,
    put_raw: np.ndarray,
    is_put: np.ndarray,
    valid_mask: np.ndarray,
    put_bias: float,
    method: str,
) -> np.ndarray:
    """Apply centering transform to raw scores. Returns final contract_scores."""
    n_bars, n_contracts = call_raw.shape
    scores = np.full((n_bars, n_contracts), -1e9, dtype=np.float32)

    for i in range(n_bars):
        vm = valid_mask[i]
        ip = is_put[i]
        if not vm.any():
            continue

        if method == "per_side_mean":
            call_m = (~ip) & vm
            put_m = ip & vm
            cs = call_raw[i].copy()
            ps = put_raw[i].copy()
            if call_m.any():
                cs[call_m] -= cs[call_m].mean()
            if put_m.any():
                ps[put_m] -= ps[put_m].mean()
            scores[i] = np.where(ip, ps + put_bias, cs)
            scores[i][~vm] = -1e9

        elif method == "raw_plus_bias":
            raw = np.where(ip, put_raw[i] + put_bias, call_raw[i])
            raw[~vm] = -1e9
            scores[i] = raw

        elif method == "global_mean":
            raw = np.where(ip, put_raw[i] + put_bias, call_raw[i])
            raw[~vm] = -1e9
            valid_vals = raw[vm]
            if len(valid_vals) > 0:
                raw[vm] -= valid_vals.mean()
            scores[i] = raw

        elif method == "per_side_zscore":
            call_m = (~ip) & vm
            put_m = ip & vm
            cs = call_raw[i].copy()
            ps = put_raw[i].copy()
            if call_m.any():
                cm, cstd = cs[call_m].mean(), cs[call_m].std()
                cs[call_m] = (cs[call_m] - cm) / max(cstd, 1e-8)
            if put_m.any():
                pm, pstd = ps[put_m].mean(), ps[put_m].std()
                ps[put_m] = (ps[put_m] - pm) / max(pstd, 1e-8)
            scores[i] = np.where(ip, ps + put_bias, cs)
            scores[i][~vm] = -1e9

    return scores


def run_ablation(model_path: str):
    print(f"Centering Ablation — checkpoint: {model_path}")
    print("=" * 78)

    model = load_model_from_path(model_path)
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    policy = DecisionPolicy()

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    put_bias_val = float(sd["put_bias"].item()) if "put_bias" in sd else 0.0
    print(f"Learned put_bias: {put_bias_val:.6f}")

    features = data["X"].numpy()
    sim_features = data["X_sim"].numpy() if "X_sim" in data else features
    mask = data["promote_mask"].numpy()
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    spot_prices = data["spot_prices"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    max_contracts = int(data["metadata"]["max_contracts_per_bar"])

    day_to_bars = _build_day_index(dates)

    # Build eligible bars
    eligible = []
    snapshots = []
    for idx in range(len(mask)):
        if not mask[idx]:
            continue
        day = dates[idx]
        bod = int(bar_of_day[idx])
        if bod < policy.no_trade_before_bar or bod >= policy.no_trade_after_bar:
            continue
        if idx < LOOKBACK:
            continue
        sc = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
        snap = padded_snapshot(sc, bod, max_contracts)
        eligible.append((day, idx, bod))
        snapshots.append(snap)

    n_bars = len(eligible)
    print(f"Bars: {n_bars}")

    # Batch inference
    window_indices = np.array([bar_idx for _, bar_idx, _ in eligible], dtype=np.int32)
    offsets = np.arange(-LOOKBACK, 0).reshape(1, -1)
    gather_idx = window_indices.reshape(-1, 1) + offsets
    all_windows = features[gather_idx]
    all_contracts = np.stack([s[0] for s in snapshots]).astype(np.float32)
    all_labels = np.stack([s[1] for s in snapshots]).astype(np.float32)
    all_indices = np.stack([s[2] for s in snapshots]).astype(np.int32)

    model.eval()
    outputs_all = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_bars, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_bars)
            batch_x = torch.from_numpy(all_windows[start:end]).float()
            batch_c = torch.from_numpy(all_contracts[start:end]).float()
            batch_out = model(batch_x, batch_c)
            outputs_all.append({k: v.cpu() for k, v in batch_out.items()})

    all_out = {k: torch.cat([o[k] for o in outputs_all], dim=0) for k in outputs_all[0]}
    call_raw = all_out["call_scores_raw"].numpy()
    put_raw = all_out["put_scores_raw"].numpy()
    valid_mask_arr = all_out["valid_mask"].numpy().astype(bool)
    is_put_arr = all_out["is_put"].numpy().astype(bool)
    print(f"Inference: {time.time() - t0:.1f}s\n")

    methods = ["per_side_mean", "raw_plus_bias", "global_mean", "per_side_zscore"]

    for method in methods:
        centered = apply_centering(call_raw, put_raw, is_put_arr, valid_mask_arr, put_bias_val, method)

        # Full replay loop with trade simulation
        trades_list = []
        current_day = None
        in_trade = False
        trade_exit_bar = -1
        last_stop_bar = -policy.cooldown_bars - 1
        daily_dollar_pnl = 0.0
        daily_loss_cap_hit = False
        cumulative_equity = policy.starting_equity
        peak_equity = cumulative_equity
        max_dd = 0.0
        call_count = 0
        put_count = 0
        day_pnls = defaultdict(float)

        for i in range(n_bars):
            day, global_bar, bod = eligible[i]

            # Day reset
            if day != current_day:
                current_day = day
                in_trade = False
                trade_exit_bar = -1
                daily_dollar_pnl = 0.0
                daily_loss_cap_hit = False

            # Opportunity gate (same as production)
            opp = float(all_out["opportunity_logit"][i].item())
            if opp <= policy.gate_threshold:
                continue

            # Trade cooldown
            if in_trade or bod <= trade_exit_bar:
                continue
            if bod - last_stop_bar < policy.cooldown_bars:
                continue
            if daily_loss_cap_hit:
                continue

            # Pick best contract under this centering
            cs = centered[i].copy()
            quality = all_contracts[i][:, 14]
            cs[quality < QUALITY_PARTIAL] = -1e9
            cs[~valid_mask_arr[i]] = -1e9

            best = int(np.argmax(cs))
            if cs[best] <= -1e8:
                continue

            # Check label is finite (unexecutable filter)
            if not np.isfinite(all_labels[i][best]):
                continue

            # Build intent and simulate
            contract_idx = int(all_indices[i][best])
            sc = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
            spot = float(spot_prices[global_bar])

            contract = describe_contract(sc, contract_idx)
            series_dict = extract_contract_series(sc, contract_idx)
            entry_mid = float(series_dict["mid"][bod])
            if not np.isfinite(entry_mid) or entry_mid <= 0:
                continue

            bid_now = float(series_dict["bid"][bod]) if np.isfinite(series_dict["bid"][bod]) else None
            ask_now = float(series_dict["ask"][bod]) if np.isfinite(series_dict["ask"][bod]) else None

            intent = TradeIntent(
                trade=True,
                expiry=contract.expiry,
                strike=contract.strike,
                right=contract.right,
                qty=policy.qty,
                entry_ref_price=entry_mid,
                order_style=policy.order_style,
                tif=policy.tif,
                stop_price=max(0.01, entry_mid * (1.0 - policy.stop_pct)),
                take_profit_price=entry_mid * (1.0 + policy.target_pct),
                max_hold_bars=policy.max_hold_bars,
                exit_policy=policy.exit_policy,
                confidence=min(1.0, max(0.0, float(cs[best]) * 3.0)),
                reason_codes=(),
                bar_index=bod,
                timestamp="",
                intent_id=str(uuid.uuid4()),
                bid_at_decision=bid_now,
                ask_at_decision=ask_now,
                underlying_price=spot,
                decision_day=day,
                snapshot_row=best,
                contract_index=contract_idx,
                contract_score=float(cs[best]),
            )

            series = series_dict["mid"].astype(np.float32)
            bars_for_day = day_to_bars[day]
            trade = simulate_trade(
                intent=intent,
                option_prices=series,
                features=sim_features[bars_for_day],
                bar_of_day=np.arange(len(bars_for_day), dtype=np.int32),
                dates=[day] * len(bars_for_day),
                global_entry_bar=bod,
                breakeven_trigger_pct=policy.breakeven_trigger_pct,
                extra_trailing_tiers=policy.extra_trailing_tiers,
            )

            if trade is None:
                continue

            trades_list.append(trade)
            pnl_dollars = trade.net_pnl_pct * trade.entry_price * policy.contract_multiplier * policy.qty
            day_pnls[day] += pnl_dollars
            cumulative_equity += pnl_dollars
            peak_equity = max(peak_equity, cumulative_equity)
            dd = (peak_equity - cumulative_equity) / policy.starting_equity
            max_dd = max(max_dd, dd)

            if is_put_arr[i][best]:
                put_count += 1
            else:
                call_count += 1

            trade_exit_bar = trade.exit_bar
            in_trade = bod < trade_exit_bar
            if trade.exit_reason == "stop_loss":
                last_stop_bar = trade.exit_bar

            daily_dollar_pnl += pnl_dollars
            if daily_dollar_pnl < -policy.daily_loss_cap_pct * policy.starting_equity:
                daily_loss_cap_hit = True

        # Compute metrics
        n_trades = len(trades_list)
        if n_trades == 0:
            print(f"  {method:<20}  NO TRADES")
            continue

        def _dpnl(t):
            return t.net_pnl_pct * t.entry_price * policy.contract_multiplier * policy.qty
        gross_win = sum(_dpnl(t) for t in trades_list if _dpnl(t) > 0)
        gross_loss = abs(sum(_dpnl(t) for t in trades_list if _dpnl(t) <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = sum(1 for t in trades_list if _dpnl(t) > 0) / n_trades
        net_pnl = sum(_dpnl(t) for t in trades_list)
        call_pct = 100 * call_count / n_trades if n_trades > 0 else 0
        traded_days = len(day_pnls)
        pos_days = sum(1 for v in day_pnls.values() if v > 0)
        pos_day_rate = pos_days / traded_days if traded_days > 0 else 0

        print(f"  {method:<20}  PF={pf:.3f}  WR={wr:.1%}  DD={max_dd:.1%}  "
              f"Trades={n_trades}  Call%={call_pct:.0f}%  ({call_count}C/{put_count}P)  "
              f"+Day={pos_day_rate:.1%}  NetPnL=${net_pnl:.0f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v2/artifacts/exp_170e/model.pt")
    args = parser.parse_args()
    run_ablation(args.model)
