"""Relabel data.pt with Tier 3 grid search for variable risk labels.

Reads existing data.pt, runs grid search over stop/target/hold for each
signal bar using raw option prices, and saves updated data.pt with variable
risk labels (stop_pct, target_pct, max_hold) that have actual variance.

Usage:
    python -m v2.pipeline.relabel_tier3
    python -m v2.pipeline.relabel_tier3 --tier 2  # medium grid (faster)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from v2.core.dataset_fingerprint import compute_dataset_fingerprint
from v2.core.features import (
    BARS_PER_DAY, MIN_HOLD_BARS, _FEAT_IDX,
    compute_adaptive_spread_bps,
)
from v2.core.simulator import TRAILING_TIERS
from v2.core.policy import DEFAULT_POLICY

MIN_ENTRY_PRICE = 0.50
COMMISSION_PER_CONTRACT = 0.65

# Search grids
GRIDS = {
    2: {
        'stops': [0.20, 0.30, 0.45],
        'targets': [0.30, 0.50, 0.80],
        'holds': [60, 120],
    },
    3: {
        'stops': [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        'targets': [0.20, 0.30, 0.50, 0.80, 1.20],
        'holds': [30, 60, 120, 240, 390],
    },
}


def _sim_one(entry_px, prices, stop, target, hold, entry_bod, vix_regime, is_otm):
    """Simulate a single trade with given risk params. Returns net P&L pct."""
    if entry_px is None or entry_px < MIN_ENTRY_PRICE:
        return float('-inf')

    last_px = entry_px
    trailing_stop = -float('inf')
    exit_k = min(hold, len(prices) - 1)  # track actual exit bar offset

    for k in range(1, min(hold + 1, len(prices))):
        px = prices[k]
        if np.isnan(px) or px <= 0:
            continue
        last_px = px
        unr = (px - entry_px) / entry_px

        if k < MIN_HOLD_BARS:
            continue

        if unr <= -stop:
            exit_px = entry_px * (1.0 - stop)
            exit_k = k
            break
        if unr >= target:
            exit_px = entry_px * (1.0 + target)
            exit_k = k
            break

        for tier_thr, lock_pct in TRAILING_TIERS:
            if unr >= tier_thr:
                if lock_pct > trailing_stop:
                    trailing_stop = lock_pct
                break
        if trailing_stop > -float('inf') and unr <= trailing_stop:
            exit_px = entry_px * (1.0 + trailing_stop)
            exit_k = k
            break
    else:
        exit_px = last_px

    raw_pnl = (exit_px - entry_px) / entry_px

    # Cost model: use ACTUAL exit bar timing, not max hold
    mtc = max(BARS_PER_DAY - entry_bod, 10)
    actual_exit_bod = min(entry_bod + exit_k, 389)
    exit_mtc = max(BARS_PER_DAY - actual_exit_bod, 1)
    entry_spread = compute_adaptive_spread_bps(mtc, vix_regime, is_otm) / 10000.0
    exit_spread = compute_adaptive_spread_bps(exit_mtc, vix_regime, is_otm) / 10000.0
    min_tick = 0.05 if entry_px < 3.00 else 0.10
    min_frac = min_tick / entry_px
    entry_spread = max(entry_spread, min_frac)
    exit_spread = max(exit_spread, min_frac)
    commission_frac = (2 * COMMISSION_PER_CONTRACT) / (entry_px * 100)

    return raw_pnl - entry_spread - exit_spread - commission_frac


def relabel(data_path: str = "v2/data.pt", tier: int = 3):
    """Relabel data.pt with variable risk params from grid search."""
    print(f"Loading {data_path}...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    N = len(data['X'])
    dates = data['dates']
    bar_of_day = data['bar_of_day'].numpy()
    features = data['X'].numpy()

    # Use nearest (dynamic ATM) prices for labeling -- these track the
    # actual nearest ATM strike per bar, not the stale session-open ATM.
    # After the Step 2 rebuild, atm_*_prices == nearest_*_close by
    # construction, but we use nearest_* here to be explicit.
    call_px = data['nearest_call_close'].numpy()
    put_px = data['nearest_put_close'].numpy()

    vix_idx = _FEAT_IDX.get('vix_regime', 0)
    grid = GRIDS[tier]
    stops = grid['stops']
    targets = grid['targets']
    holds = grid['holds']
    n_combos = len(stops) * len(targets) * len(holds)
    print(f"Tier {tier}: {len(stops)} stops x {len(targets)} targets x {len(holds)} holds = {n_combos} combos per direction")

    # Current labels (we'll update risk params in-place)
    label_direction = data['label_direction'].numpy()
    old_call_pnl = data['label_call_pnl'].numpy()
    old_put_pnl = data['label_put_pnl'].numpy()

    # New label arrays
    new_call_pnl = np.zeros(N, dtype=np.float32)
    new_put_pnl = np.zeros(N, dtype=np.float32)
    new_stop = np.zeros(N, dtype=np.float32)
    new_target = np.zeros(N, dtype=np.float32)
    new_hold = np.zeros(N, dtype=np.int32)
    new_confidence = np.zeros(N, dtype=np.float32)
    new_trade = np.zeros(N, dtype=bool)
    new_direction = np.full(N, -1, dtype=np.int32)
    new_pnl = np.zeros(N, dtype=np.float32)

    # Build day boundaries
    day_end = {}
    for i, d in enumerate(dates):
        day_end[d] = i

    t0 = time.time()
    n_signal = 0
    n_updated = 0
    max_forward = max(holds) + 2

    for i in range(N):
        if label_direction[i] < 0:
            continue

        bod = int(bar_of_day[i])
        if bod < 30 or bod >= 270:
            continue

        n_signal += 1

        # Get forward prices (same day only)
        d = dates[i]
        end = min(i + max_forward + 1, day_end.get(d, i) + 1)
        if end <= i + 1:
            continue

        call_fwd = call_px[i:end]
        put_fwd = put_px[i:end]

        if len(call_fwd) < 3:
            continue

        c_entry = call_fwd[1] if len(call_fwd) > 1 else np.nan
        p_entry = put_fwd[1] if len(put_fwd) > 1 else np.nan

        vix_regime = float(features[i, vix_idx])

        # Grid search for BEST call and BEST put risk params
        best_call_pnl = float('-inf')
        best_call_params = (0.30, 0.50, 30)
        best_put_pnl = float('-inf')
        best_put_params = (0.30, 0.50, 30)

        for stop in stops:
            for target in targets:
                for hold in holds:
                    if not np.isnan(c_entry) and c_entry >= MIN_ENTRY_PRICE:
                        cpnl = _sim_one(c_entry, call_fwd[1:], stop, target, hold, bod, vix_regime, False)
                        if cpnl > best_call_pnl:
                            best_call_pnl = cpnl
                            best_call_params = (stop, target, hold)

                    if not np.isnan(p_entry) and p_entry >= MIN_ENTRY_PRICE:
                        ppnl = _sim_one(p_entry, put_fwd[1:], stop, target, hold, bod, vix_regime, False)
                        if ppnl > best_put_pnl:
                            best_put_pnl = ppnl
                            best_put_params = (stop, target, hold)

        # Use best P&L from grid search (uncapped by fixed target)
        if best_call_pnl == float('-inf'):
            best_call_pnl = 0.0
        if best_put_pnl == float('-inf'):
            best_put_pnl = 0.0

        new_call_pnl[i] = best_call_pnl
        new_put_pnl[i] = best_put_pnl

        # Direction = whichever side had higher P&L
        best_pnl = max(best_call_pnl, best_put_pnl)
        if best_call_pnl >= best_put_pnl:
            direction = 0
            best_params = best_call_params
        else:
            direction = 1
            best_params = best_put_params

        is_trade = best_pnl > DEFAULT_POLICY.label_gate_min_pnl

        new_direction[i] = direction
        new_pnl[i] = best_pnl
        new_trade[i] = is_trade
        new_stop[i] = best_params[0]
        new_target[i] = best_params[1]
        new_hold[i] = best_params[2]
        new_confidence[i] = max(0.0, min(1.0, abs(best_call_pnl - best_put_pnl) * 3.0))
        n_updated += 1

        if n_signal % 10000 == 0:
            elapsed = time.time() - t0
            rate = n_signal / elapsed if elapsed > 0 else 0
            print(f"  {n_signal:,} bars, {rate:.0f} bars/s, "
                  f"eta {(N - i) / max(rate, 1) / 60:.1f} min")

    elapsed = time.time() - t0
    print(f"\nRelabeled {n_updated:,} bars in {elapsed:.1f}s ({n_signal:,} signal bars)")

    # Stats
    valid = new_direction >= 0
    if valid.any():
        s = new_stop[valid]
        t = new_target[valid]
        h = new_hold[valid]
        cp = new_call_pnl[valid]
        pp = new_put_pnl[valid]
        bp = np.maximum(cp, pp)
        print(f"\n=== New Label Stats ===")
        print(f"Stop:   mean={s.mean():.3f} std={s.std():.3f} min={s.min():.3f} max={s.max():.3f}")
        print(f"Target: mean={t.mean():.3f} std={t.std():.3f} min={t.min():.3f} max={t.max():.3f}")
        print(f"Hold:   mean={h.mean():.1f} std={h.std():.1f} min={h.min()} max={h.max()}")
        print(f"Call P&L: mean={cp.mean():.4f} max={cp.max():.4f}")
        print(f"Put P&L:  mean={pp.mean():.4f} max={pp.max():.4f}")
        print(f"Best P&L: mean={bp.mean():.4f} max={bp.max():.4f}")
        print(f"Big wins (>50%): {(bp > 0.50).sum()} ({100*(bp > 0.50).mean():.1f}%)")
        print(f"Big wins (>100%): {(bp > 1.0).sum()} ({100*(bp > 1.0).mean():.1f}%)")
        print(f"Trade rate: {new_trade.sum()}/{valid.sum()} ({100*new_trade[valid].mean():.1f}%)")

    # Update data dict
    data['label_call_pnl'] = torch.from_numpy(new_call_pnl)
    data['label_put_pnl'] = torch.from_numpy(new_put_pnl)
    data['label_stop_pct'] = torch.from_numpy(new_stop)
    data['label_target_pct'] = torch.from_numpy(new_target)
    data['label_max_hold'] = torch.from_numpy(new_hold).to(torch.int32)
    data['label_confidence'] = torch.from_numpy(new_confidence)
    data['label_trade'] = torch.from_numpy(new_trade)
    data['label_direction'] = torch.from_numpy(new_direction).to(torch.int32)
    data['label_pnl'] = torch.from_numpy(new_pnl)

    # Update metadata to match actual tensor contents (Step 5: metadata truthfulness)
    actual_trades = int(new_trade.sum())
    actual_gate_rate = new_trade[valid].mean() if valid.any() else 0.0
    pnls_trade = new_pnl[new_trade]
    actual_mean_pnl = float(pnls_trade.mean()) if len(pnls_trade) > 0 else 0.0

    meta = data.get('metadata', {})
    meta['label_scheme'] = f'dual_direction_pnl_tier{tier}'
    meta['label_tier'] = tier
    meta['label_grid'] = f"stops={stops} targets={targets} holds={holds}"
    meta['total_trades'] = actual_trades
    meta['gate_true_rate'] = float(actual_gate_rate)
    meta['mean_pnl_trade'] = actual_mean_pnl
    meta['fixed_stop'] = 'VARIABLE'
    meta['fixed_target'] = 'VARIABLE'
    meta['fixed_hold'] = 'VARIABLE'
    data['metadata'] = meta
    data['metadata']['fingerprint'] = compute_dataset_fingerprint(data)

    # Save
    out_path = data_path
    print(f"\nSaving to {out_path}...")
    torch.save(data, out_path)
    print(f"Done. {out_path} updated with Tier {tier} labels.")
    print(f"Fingerprint: {data['metadata']['fingerprint']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--tier", type=int, default=3)
    args = parser.parse_args()
    relabel(args.data, args.tier)
