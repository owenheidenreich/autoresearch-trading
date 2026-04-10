"""Quantify structural put dominance on full 999-day Polygon dataset.

Maps when puts win, when calls win (if ever), and by how much.
Runs locally on CPU. No GPU needed.

Usage:
    python -m v2.research.direction_edge_full
"""
import numpy as np
import torch
import sys
from collections import defaultdict


DATA_PATH = "v2/data.pt"

# Match build_v2_dataset.py labeling config
FIXED_STOP = 0.30
FIXED_TARGET = 0.50
HOLD_DURATIONS = [10, 30, 60, 120]

# Direction features (same as build_v2_dataset.py)
DIR_FEATURES = ['session_range_pct', 'realized_vol', 'option_spread_width']

# Time-of-day buckets
TOD_BUCKETS = {
    'morning': (30, 90),    # 10:00-11:00
    'midday': (90, 210),    # 11:00-13:00
    'afternoon': (210, 330), # 13:00-15:00
}


def pf(pnls):
    """Profit factor from list of P&L values."""
    if not pnls:
        return 0.0
    w = sum(p for p in pnls if p > 0)
    l = abs(sum(p for p in pnls if p < 0))
    if l == 0:
        return 999.0 if w > 0 else 0.0
    return w / l


def wr(pnls):
    """Win rate from list of P&L values."""
    if not pnls:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls) * 100


def simulate_trade(prices, entry_bar, hold, stop, target):
    """Simulate one trade. Returns P&L fraction or None if no valid entry."""
    fill_bar = entry_bar + 1
    if fill_bar >= len(prices):
        return None
    entry_px = prices[fill_bar]
    if entry_px <= 0 or np.isnan(entry_px):
        return None

    last_px = entry_px
    for k in range(1, hold + 1):
        check = fill_bar + k
        if check >= len(prices):
            break
        px = prices[check]
        if px <= 0 or np.isnan(px):
            continue
        last_px = px
        unrealized = (px - entry_px) / entry_px

        if unrealized <= -stop:
            return -stop
        if unrealized >= target:
            return target

    return (last_px - entry_px) / entry_px


def main():
    print("Loading data...")
    d = torch.load(DATA_PATH, map_location='cpu', weights_only=False)

    X = d['X'].numpy()
    feature_names = d['feature_names']
    call_prices = d['nearest_call_close'].numpy()
    put_prices = d['nearest_put_close'].numpy()
    bar_of_day = d['bar_of_day'].numpy()
    dates = np.array(d['dates'])
    unique_dates = sorted(set(dates))

    # Split masks
    splits = {}
    for name in ['train', 'val', 'promote', 'shadow']:
        splits[name] = d[f'{name}_mask'].numpy().astype(bool)

    # Feature indices for direction voting
    dir_indices = {}
    for fname in DIR_FEATURES:
        if fname in feature_names:
            dir_indices[fname] = feature_names.index(fname)
    print(f"Direction features: {list(dir_indices.keys())}")

    # Compute per-feature medians on TRAIN split only (matching build_v2_dataset.py)
    tradeable_train = (bar_of_day >= 30) & (bar_of_day < 270) & splits['train']
    dir_medians = {}
    for fname, fidx in dir_indices.items():
        vals = X[tradeable_train, fidx]
        vals = vals[~np.isnan(vals)]
        med = float(np.median(vals)) if len(vals) > 0 else 0.0
        dir_medians[fname] = med
        print(f"  {fname}: median={med:.6f} (n={len(vals)})")

    # Build day boundaries for trade simulation
    print("\nBuilding day index...")
    day_to_indices = defaultdict(list)
    for i, dt in enumerate(dates):
        day_to_indices[dt].append(i)

    # Assign quarter labels
    def quarter(dt):
        y = int(str(dt)[:4])
        m = int(str(dt)[5:7])
        return f"{y}Q{(m-1)//3+1}"

    # Run simulations
    print("\nSimulating trades...")

    # Results structure: key -> list of pnl
    # Keys: (strategy, hold, split, quarter, tod)
    results = defaultdict(list)

    entry_bars_per_day = list(range(30, 271, 10))  # every 10 bars, bar 30-270

    n_days = len(unique_dates)
    for day_idx, day in enumerate(unique_dates):
        if (day_idx + 1) % 200 == 0:
            print(f"  {day_idx+1}/{n_days} days...")

        indices = day_to_indices[day]
        if len(indices) < 100:
            continue

        day_call = call_prices[indices]
        day_put = put_prices[indices]
        day_bod = bar_of_day[indices]
        day_X = X[indices]

        # Determine split
        first_gi = indices[0]
        if splits['train'][first_gi]:
            split = 'train'
        elif splits['val'][first_gi]:
            split = 'val'
        elif splits['promote'][first_gi]:
            split = 'promote'
        elif splits['shadow'][first_gi]:
            split = 'shadow'
        else:
            continue

        q = quarter(day)

        for local_entry in entry_bars_per_day:
            if local_entry >= len(indices):
                continue
            gi = indices[local_entry]
            bod = int(day_bod[local_entry])

            # Check valid prices
            if call_prices[gi] <= 0 or put_prices[gi] <= 0:
                continue

            # Determine time-of-day bucket
            tod = None
            for bucket_name, (lo, hi) in TOD_BUCKETS.items():
                if lo <= bod < hi:
                    tod = bucket_name
                    break
            if tod is None:
                continue

            # Compute regime direction (median-split majority vote)
            call_votes = 0
            put_votes = 0
            for fname, fidx in dir_indices.items():
                val = day_X[local_entry, fidx]
                if np.isnan(val):
                    continue
                if val > dir_medians[fname]:
                    call_votes += 1
                else:
                    put_votes += 1
            regime_dir = 'call' if call_votes > put_votes else 'put'

            for hold in HOLD_DURATIONS:
                # Simulate call trade
                call_pnl = simulate_trade(
                    day_call, local_entry, hold, FIXED_STOP, FIXED_TARGET
                )
                # Simulate put trade
                put_pnl = simulate_trade(
                    day_put, local_entry, hold, FIXED_STOP, FIXED_TARGET
                )

                if call_pnl is not None:
                    results[('always_call', hold, split, q, tod)].append(call_pnl)
                    results[('always_call', hold, split, 'ALL', tod)].append(call_pnl)
                    results[('always_call', hold, split, q, 'ALL')].append(call_pnl)
                    results[('always_call', hold, split, 'ALL', 'ALL')].append(call_pnl)

                if put_pnl is not None:
                    results[('always_put', hold, split, q, tod)].append(put_pnl)
                    results[('always_put', hold, split, 'ALL', tod)].append(put_pnl)
                    results[('always_put', hold, split, q, 'ALL')].append(put_pnl)
                    results[('always_put', hold, split, 'ALL', 'ALL')].append(put_pnl)

                # Regime strategy
                if regime_dir == 'call' and call_pnl is not None:
                    results[('regime', hold, split, q, tod)].append(call_pnl)
                    results[('regime', hold, split, 'ALL', tod)].append(call_pnl)
                    results[('regime', hold, split, q, 'ALL')].append(call_pnl)
                    results[('regime', hold, split, 'ALL', 'ALL')].append(call_pnl)
                elif regime_dir == 'put' and put_pnl is not None:
                    results[('regime', hold, split, q, tod)].append(put_pnl)
                    results[('regime', hold, split, 'ALL', tod)].append(put_pnl)
                    results[('regime', hold, split, q, 'ALL')].append(put_pnl)
                    results[('regime', hold, split, 'ALL', 'ALL')].append(put_pnl)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DIRECTION EDGE ANALYSIS - FULL DATASET")
    print("=" * 80)

    # 1. Overall by hold duration
    print("\n--- 1. OVERALL BY HOLD DURATION (all splits combined) ---")
    print(f"{'Hold':>6} | {'Always-Call':>20} | {'Always-Put':>20} | {'Regime':>20}")
    print("-" * 75)
    for hold in HOLD_DURATIONS:
        for strat in ['always_call', 'always_put', 'regime']:
            # Combine all splits
            combined = []
            for split in ['train', 'val', 'promote', 'shadow']:
                combined.extend(results.get((strat, hold, split, 'ALL', 'ALL'), []))
            results[(strat, hold, 'ALL', 'ALL', 'ALL')] = combined

        ac = results[('always_call', hold, 'ALL', 'ALL', 'ALL')]
        ap = results[('always_put', hold, 'ALL', 'ALL', 'ALL')]
        rg = results[('regime', hold, 'ALL', 'ALL', 'ALL')]
        print(f"h{hold:>4}  | PF={pf(ac):>5.3f} WR={wr(ac):>4.1f}% n={len(ac):>5} "
              f"| PF={pf(ap):>5.3f} WR={wr(ap):>4.1f}% n={len(ap):>5} "
              f"| PF={pf(rg):>5.3f} WR={wr(rg):>4.1f}% n={len(rg):>5}")

    # 2. By split
    print("\n--- 2. BY SPLIT (h30 only, matching labeler) ---")
    print(f"{'Split':>8} | {'Always-Call':>20} | {'Always-Put':>20} | {'Regime':>20}")
    print("-" * 78)
    for split in ['train', 'val', 'promote', 'shadow']:
        ac = results.get(('always_call', 30, split, 'ALL', 'ALL'), [])
        ap = results.get(('always_put', 30, split, 'ALL', 'ALL'), [])
        rg = results.get(('regime', 30, split, 'ALL', 'ALL'), [])
        if not ac:
            continue
        print(f"{split:>8} | PF={pf(ac):>5.3f} WR={wr(ac):>4.1f}% n={len(ac):>5} "
              f"| PF={pf(ap):>5.3f} WR={wr(ap):>4.1f}% n={len(ap):>5} "
              f"| PF={pf(rg):>5.3f} WR={wr(rg):>4.1f}% n={len(rg):>5}")

    # 3. By quarter (h30)
    print("\n--- 3. BY QUARTER (h30, all splits) ---")
    quarters = sorted(set(
        k[3] for k in results.keys() if k[3] not in ('ALL',)
    ))
    print(f"{'Quarter':>8} | {'Always-Call':>20} | {'Always-Put':>20} | {'Regime':>20}")
    print("-" * 78)
    for q in quarters:
        ac = []
        ap = []
        rg = []
        for split in ['train', 'val', 'promote', 'shadow']:
            ac.extend(results.get(('always_call', 30, split, q, 'ALL'), []))
            ap.extend(results.get(('always_put', 30, split, q, 'ALL'), []))
            rg.extend(results.get(('regime', 30, split, q, 'ALL'), []))
        if not ac:
            continue
        regime_better = "<<" if pf(rg) > pf(ap) else ""
        print(f"{q:>8} | PF={pf(ac):>5.3f} WR={wr(ac):>4.1f}% n={len(ac):>5} "
              f"| PF={pf(ap):>5.3f} WR={wr(ap):>4.1f}% n={len(ap):>5} "
              f"| PF={pf(rg):>5.3f} WR={wr(rg):>4.1f}% n={len(rg):>5} {regime_better}")

    # 4. By time-of-day (h30)
    print("\n--- 4. BY TIME OF DAY (h30, all splits) ---")
    print(f"{'TOD':>10} | {'Always-Call':>20} | {'Always-Put':>20} | {'Regime':>20}")
    print("-" * 80)
    for tod in ['morning', 'midday', 'afternoon']:
        ac = []
        ap = []
        rg = []
        for split in ['train', 'val', 'promote', 'shadow']:
            ac.extend(results.get(('always_call', 30, split, 'ALL', tod), []))
            ap.extend(results.get(('always_put', 30, split, 'ALL', tod), []))
            rg.extend(results.get(('regime', 30, split, 'ALL', tod), []))
        if not ac:
            continue
        regime_better = "<<" if pf(rg) > pf(ap) else ""
        print(f"{tod:>10} | PF={pf(ac):>5.3f} WR={wr(ac):>4.1f}% n={len(ac):>5} "
              f"| PF={pf(ap):>5.3f} WR={wr(ap):>4.1f}% n={len(ap):>5} "
              f"| PF={pf(rg):>5.3f} WR={wr(rg):>4.1f}% n={len(rg):>5} {regime_better}")

    # 5. By time-of-day x hold (regime vs always-put delta)
    print("\n--- 5. REGIME EDGE OVER ALWAYS-PUT (PF difference) ---")
    print(f"{'':>10}", end="")
    for hold in HOLD_DURATIONS:
        print(f" | h{hold:>3}", end="")
    print()
    print("-" * 60)
    for tod in ['morning', 'midday', 'afternoon', 'ALL']:
        print(f"{tod:>10}", end="")
        for hold in HOLD_DURATIONS:
            ap = []
            rg = []
            for split in ['train', 'val', 'promote', 'shadow']:
                ap.extend(results.get(('always_put', hold, split, 'ALL', tod), []))
                rg.extend(results.get(('regime', hold, split, 'ALL', tod), []))
            delta = pf(rg) - pf(ap)
            marker = "+" if delta > 0 else ""
            print(f" | {marker}{delta:>5.3f}", end="")
        print()

    # 6. Direction label balance check
    print("\n--- 6. DIRECTION LABEL BALANCE ---")
    total_call = 0
    total_put = 0
    for day in unique_dates:
        indices = day_to_indices[day]
        for local_entry in entry_bars_per_day:
            if local_entry >= len(indices):
                continue
            gi = indices[local_entry]
            day_X_row = X[gi]
            call_votes = 0
            put_votes = 0
            for fname, fidx in dir_indices.items():
                val = day_X_row[fidx]
                if np.isnan(val):
                    continue
                if val > dir_medians[fname]:
                    call_votes += 1
                else:
                    put_votes += 1
            if call_votes > put_votes:
                total_call += 1
            else:
                total_put += 1
    total = total_call + total_put
    print(f"  Call labels: {total_call} ({total_call/total*100:.1f}%)")
    print(f"  Put labels:  {total_put} ({total_put/total*100:.1f}%)")
    print(f"  Median split should be ~50/50. If skewed, direction features are correlated.")

    print("\n" + "=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
