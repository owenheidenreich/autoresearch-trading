#!/usr/bin/env python3
"""Export trades.jsonl to CSV for comparison against IBKR paper trade history.

Usage:
  python3 tools/export_trades_csv.py                          # all trades
  python3 tools/export_trades_csv.py --date 2026-03-26        # single day
  python3 tools/export_trades_csv.py --output trades.csv      # custom output
"""
import argparse
import csv
import json
import os
import sys

TRADES_PATH = os.path.join("results", "live", "trades.jsonl")
DEFAULT_OUTPUT = os.path.join("results", "live", "trades.csv")

COLUMNS = [
    "date",
    "entry_ts",
    "exit_ts",
    "direction",
    "strike",
    "right",
    "contract",
    "qty",
    "entry_price",
    "exit_price",
    "pnl_pct",
    "pnl_dollar",
    "bars_held",
    "exit_reason",
    "spx_at_entry",
    "spx_at_exit",
    "stop_price",
    "tp_price",
    "gate_confidence",
    "session_id",
    "position_id",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trades to CSV")
    parser.add_argument("--input", default=TRADES_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--date", default=None, help="Filter to specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"No trades file at {args.input}")
        sys.exit(1)

    trades = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if args.date and record.get("date") != args.date:
                continue
            trades.append(record)

    if not trades:
        print(f"No trades found" + (f" for {args.date}" if args.date else ""))
        sys.exit(0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for t in trades:
            writer.writerow(t)

    print(f"Exported {len(trades)} trades to {args.output}")
    # Print summary
    total_pnl = sum(t.get("pnl_dollar", 0) for t in trades)
    winners = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Win rate: {winners}/{len(trades)} ({winners/max(len(trades),1)*100:.0f}%)")


if __name__ == "__main__":
    main()
