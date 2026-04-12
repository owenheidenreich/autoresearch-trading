"""Analyze losing days in replay to understand what causes negative days."""
import torch
import numpy as np
from collections import defaultdict
from v2.replay import load_best_model, replay_validation

data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
model, policy, manifest = load_best_model()

metrics, trades, _ = replay_validation(model, data, mask_key="promote_mask", policy=policy)

# Group trades by day
day_trades = defaultdict(list)
for t in trades:
    day_trades[t.trade_date].append(t)

# Compute daily P&L using net_pnl_pct * entry_price * 100
print(f"\n{'='*70}")
print(f"  LOSING DAYS ANALYSIS")
print(f"{'='*70}")
print(f"Total trades: {len(trades)}")
print(f"Total days: {len(day_trades)}")

def trade_dollar_pnl(t):
    return t.net_pnl_pct * t.entry_price * 100 * t.intent.qty

daily_pnl = {}
for day, ts in sorted(day_trades.items()):
    pnl = sum(trade_dollar_pnl(t) for t in ts)
    daily_pnl[day] = pnl

losing_days = {d: p for d, p in daily_pnl.items() if p < 0}
print(f"Losing days: {len(losing_days)}")
print()

for day, pnl in sorted(losing_days.items()):
    ts = day_trades[day]
    print(f"\n--- {day} (net P&L: ${pnl:.2f}) ---")
    print(f"  Trades: {len(ts)}")
    for t in ts:
        dpnl = trade_dollar_pnl(t)
        print(f"    {t.intent.right} strike={t.intent.strike} bar={t.entry_bar} "
              f"exit={t.exit_reason} hold={t.bars_held} bars "
              f"P&L=${dpnl:.2f} ({t.net_pnl_pct*100:+.1f}%) "
              f"entry=${t.entry_price:.2f} exit=${t.exit_price:.2f}")

    n_sl = sum(1 for t in ts if t.exit_reason == "STOP_LOSS")
    n_tp = sum(1 for t in ts if t.exit_reason == "TAKE_PROFIT")
    n_eod = sum(1 for t in ts if "EOD" in t.exit_reason or "END" in t.exit_reason)
    n_calls = sum(1 for t in ts if t.intent.right == "C")
    n_puts = sum(1 for t in ts if t.intent.right == "P")
    print(f"  SL={n_sl} TP={n_tp} EOD={n_eod} | C={n_calls} P={n_puts}")

# Direction analysis
print(f"\n{'='*70}")
print(f"  DIRECTION ANALYSIS")
print(f"{'='*70}")
calls = [t for t in trades if t.intent.right == "C"]
puts = [t for t in trades if t.intent.right == "P"]
call_wins = sum(1 for t in calls if t.net_pnl_pct > 0)
put_wins = sum(1 for t in puts if t.net_pnl_pct > 0)
print(f"Calls: {len(calls)} trades, WR={call_wins/max(len(calls),1)*100:.1f}%, "
      f"avg=${np.mean([trade_dollar_pnl(t) for t in calls]) if calls else 0:.2f}")
print(f"Puts:  {len(puts)} trades, WR={put_wins/max(len(puts),1)*100:.1f}%, "
      f"avg=${np.mean([trade_dollar_pnl(t) for t in puts]) if puts else 0:.2f}")
print(f"Total call P&L: ${sum(trade_dollar_pnl(t) for t in calls):.2f}")
print(f"Total put P&L: ${sum(trade_dollar_pnl(t) for t in puts):.2f}")

# Exit reason breakdown
print(f"\n{'='*70}")
print(f"  EXIT REASON ANALYSIS")
print(f"{'='*70}")
reasons = defaultdict(list)
for t in trades:
    reasons[t.exit_reason].append(t)
for reason, ts in sorted(reasons.items()):
    wr = sum(1 for t in ts if t.net_pnl_pct > 0) / len(ts) * 100
    avg = np.mean([trade_dollar_pnl(t) for t in ts])
    print(f"  {reason}: {len(ts)} trades, WR={wr:.1f}%, avg=${avg:.2f}")
