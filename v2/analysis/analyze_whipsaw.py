"""Deep analysis of the 4 losing whipsaw days.

What features distinguish them from winning days?
Focus on: session_range_pct, realized_vol, option_spread_width, bar_range.
"""
import torch
import numpy as np
from collections import defaultdict
from v2.replay import load_best_model, replay_validation
from v2.core.features import FEATURE_NAMES

data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
model, policy, manifest = load_best_model()
features = data['X'].numpy()
dates = data['dates']
bar_of_day = data['bar_of_day'].numpy()

metrics, trades = replay_validation(model, data, mask_key="promote_mask", policy=policy)

def trade_dollar_pnl(t):
    return t.net_pnl_pct * t.entry_price * 100 * t.intent.qty

# Group by day
day_trades = defaultdict(list)
for t in trades:
    day_trades[t.trade_date].append(t)

daily_pnl = {}
for day, ts in sorted(day_trades.items()):
    daily_pnl[day] = sum(trade_dollar_pnl(t) for t in ts)

losing_days = sorted([d for d, p in daily_pnl.items() if p < 0])
winning_days = sorted([d for d, p in daily_pnl.items() if p > 0])

print(f"Losing days: {losing_days}")
print(f"Winning days: {len(winning_days)}")

# Find feature indices for key signals
print(f"\nFeature names ({len(FEATURE_NAMES)}):")
key_features = {}
for i, name in enumerate(FEATURE_NAMES):
    if name in ('session_range_pct', 'rsi_7', 'bollinger_position',
                'atm_iv', 'atm_gamma', 'option_spread_width',
                'vix_level', 'volume_zero_flag', 'call_put_flow_ratio',
                'intraday_drift_pct', 'theta_acceleration'):
        key_features[name] = i
        print(f"  [{i}] {name}")

# Build day->bar index
day_to_bars = defaultdict(list)
for i, d in enumerate(dates):
    day_to_bars[d].append(i)

# For each losing/winning day, compute average feature values
print(f"\n{'='*80}")
print(f"  FEATURE COMPARISON: LOSING vs WINNING DAYS")
print(f"{'='*80}")

def day_feature_stats(day):
    bars = day_to_bars.get(day, [])
    if not bars:
        return {}
    stats = {}
    for name, idx in key_features.items():
        vals = [features[b, idx] for b in bars if not np.isnan(features[b, idx])]
        if vals:
            stats[name] = {
                'mean': np.mean(vals),
                'max': np.max(vals),
                'std': np.std(vals),
            }
    return stats

print(f"\n--- LOSING DAYS ---")
losing_stats = defaultdict(list)
for day in losing_days:
    stats = day_feature_stats(day)
    pnl = daily_pnl[day]
    print(f"\n{day} (P&L: ${pnl:.0f}):")
    for name, s in sorted(stats.items()):
        print(f"  {name:30s} mean={s['mean']:.4f}  max={s['max']:.4f}  std={s['std']:.4f}")
        losing_stats[name].append(s['mean'])

print(f"\n--- WINNING DAYS (averages) ---")
winning_stats = defaultdict(list)
for day in winning_days:
    stats = day_feature_stats(day)
    for name, s in sorted(stats.items()):
        winning_stats[name].append(s['mean'])

print(f"\n{'Feature':30s} {'Losing avg':>12s} {'Winning avg':>12s} {'Diff':>10s}")
print("-" * 70)
for name in sorted(key_features.keys()):
    l_avg = np.mean(losing_stats[name]) if losing_stats[name] else 0
    w_avg = np.mean(winning_stats[name]) if winning_stats[name] else 0
    diff = l_avg - w_avg
    marker = " ***" if abs(diff) > 0.1 * max(abs(l_avg), abs(w_avg), 0.001) else ""
    print(f"{name:30s} {l_avg:12.4f} {w_avg:12.4f} {diff:+10.4f}{marker}")

# Also look at the trade-level features at entry
print(f"\n{'='*80}")
print(f"  ENTRY-BAR FEATURES FOR LOSING TRADES")
print(f"{'='*80}")

for day in losing_days:
    ts = day_trades[day]
    print(f"\n{day}:")
    for t in ts:
        entry_bar = t.entry_bar
        dpnl = trade_dollar_pnl(t)
        print(f"  {t.intent.right} bar={entry_bar} hold={t.bars_held} exit={t.exit_reason} P&L=${dpnl:.0f}")
        for name, idx in sorted(key_features.items()):
            val = features[entry_bar, idx]
            print(f"    {name:30s} = {val:.4f}")

# Check: are losing days HIGH volatility (whipsaw) or LOW (theta decay)?
print(f"\n{'='*80}")
print(f"  VOLATILITY REGIME ON LOSING DAYS")
print(f"{'='*80}")
if 'session_range_pct' in key_features:
    idx = key_features['session_range_pct']
    for day in losing_days:
        bars = day_to_bars.get(day, [])
        if bars:
            # Use last bar's session_range as the day's final range
            last_bar_val = features[bars[-1], idx]
            print(f"  {day}: session_range_pct (last bar) = {last_bar_val:.4f}")
    print()
    all_promote_days = sorted(set(dates[i] for i in np.where(data['promote_mask'].numpy())[0]))
    all_ranges = []
    for day in all_promote_days:
        bars = day_to_bars.get(day, [])
        if bars:
            all_ranges.append(features[bars[-1], idx])
    print(f"  Promote period range stats: mean={np.mean(all_ranges):.4f}, "
          f"median={np.median(all_ranges):.4f}, "
          f"p25={np.percentile(all_ranges, 25):.4f}, "
          f"p75={np.percentile(all_ranges, 75):.4f}")
