"""Isolated risk overlay tests with causal diagnostics.

Runs 7 replay tests against the exp_165 model, each testing a single
overlay hypothesis. Reports per-test metrics, day-type classification,
blocked-trade outcome analysis, and post-stop behavior patterns.

Usage:
    python3 v2/analysis/overlay_diagnostic.py
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict

import numpy as np
import torch

from v2.core.policy import DecisionPolicy
from v2.replay import load_best_model, replay_validation


# ── Test configurations ──────────────────────────────────────────────
# Each test modifies exactly ONE overlay field from defaults.
TESTS = [
    ("Baseline",        {}),
    ("Control-20%",     {"random_skip_pct": 0.20}),
    ("A1: MaxTrades=4", {"max_daily_trades": 4}),
    ("A2: MaxTrades=6", {"max_daily_trades": 6}),
    ("B1: ConsStops=1", {"max_consecutive_stops": 1}),
    ("B2: ConsStops=2", {"max_consecutive_stops": 2}),
    ("C1: GateTight=0.1", {"gate_tighten_after_loss": 0.1}),
]

DEFAULT_POLICY = DecisionPolicy()


def trade_dollar_pnl(t, policy: DecisionPolicy) -> float:
    return t.net_pnl_pct * t.entry_price * policy.contract_multiplier * policy.qty


def classify_day(trade_pnls: list[float]) -> str:
    """Classify a trading day based on cumulative P&L path.

    green:     cumulative P&L never goes negative
    red:       cumulative P&L never goes positive
    recovered: goes negative then ends positive (or breaks even)
    collapsed: goes positive then ends negative
    chop:      multiple sign changes (3+)
    """
    if not trade_pnls:
        return "no_trades"
    cum = np.cumsum(trade_pnls)
    ever_neg = np.any(cum < -1e-6)
    ever_pos = np.any(cum > 1e-6)
    final = cum[-1]

    if not ever_neg and not ever_pos:
        return "chop"
    if not ever_neg:
        return "green"
    if not ever_pos:
        return "red"

    # Both positive and negative excursions — classify by path shape
    sign_changes = np.sum(np.diff(np.sign(cum)) != 0)
    if sign_changes >= 3:
        return "chop"
    if final >= 0:
        return "recovered"
    return "collapsed"


def compute_post_stop_stats(day_trades: list, policy: DecisionPolicy) -> dict:
    """Compute P&L statistics segmented by position in loss sequence."""
    first_of_day = []
    after_first_stop = []
    after_second_stop = []
    loss_after_first_loss = []  # P&L of trades after the first losing trade

    for day, ts in day_trades.items():
        stop_count = 0
        first_loss_seen = False
        day_loss_before_first = 0.0
        day_loss_after_first = 0.0

        for idx, t in enumerate(ts):
            pnl = trade_dollar_pnl(t, policy)
            if idx == 0:
                first_of_day.append(pnl)
            if stop_count == 0:
                pass  # before any stop
            elif stop_count == 1:
                after_first_stop.append(pnl)
            elif stop_count >= 2:
                after_second_stop.append(pnl)

            if first_loss_seen:
                loss_after_first_loss.append(pnl)
                day_loss_after_first += pnl
            else:
                if pnl < 0:
                    first_loss_seen = True

            if t.exit_reason == "STOP_LOSS":
                stop_count += 1

    return {
        "avg_pnl_first_of_day": np.mean(first_of_day) if first_of_day else 0.0,
        "avg_pnl_after_first_stop": np.mean(after_first_stop) if after_first_stop else 0.0,
        "avg_pnl_after_second_stop": np.mean(after_second_stop) if after_second_stop else 0.0,
        "n_after_first_stop": len(after_first_stop),
        "n_after_second_stop": len(after_second_stop),
        "pct_daily_loss_after_first_loser": _pct_loss_after_first(day_trades, policy),
    }


def _pct_loss_after_first(day_trades: dict, policy: DecisionPolicy) -> float:
    """What percent of daily loss comes after the first losing trade?"""
    total_daily_loss = 0.0
    loss_after_first = 0.0
    for day, ts in day_trades.items():
        day_pnl = sum(trade_dollar_pnl(t, policy) for t in ts)
        if day_pnl >= 0:
            continue  # only count losing days
        first_loss_seen = False
        for t in ts:
            pnl = trade_dollar_pnl(t, policy)
            if first_loss_seen:
                if pnl < 0:
                    loss_after_first += abs(pnl)
            else:
                if pnl < 0:
                    first_loss_seen = True
        total_daily_loss += abs(day_pnl)
    return (loss_after_first / total_daily_loss * 100) if total_daily_loss > 0 else 0.0


def _pct_bad_days_rescued(day_trades: dict, policy: DecisionPolicy) -> float:
    """What percent of days that go red early end up recovering?"""
    red_early = 0
    rescued = 0
    for day, ts in day_trades.items():
        pnls = [trade_dollar_pnl(t, policy) for t in ts]
        if len(pnls) < 2:
            continue
        if pnls[0] < 0:  # red early
            red_early += 1
            if sum(pnls) >= 0:  # recovered
                rescued += 1
    return (rescued / red_early * 100) if red_early > 0 else 0.0


def run_single_test(
    name: str,
    overrides: dict,
    model,
    data: dict,
    base_policy: DecisionPolicy,
) -> dict:
    """Run one isolated overlay test and compute all diagnostics."""
    policy = dataclasses.replace(base_policy, **overrides) if overrides else base_policy
    metrics, trades, _ = replay_validation(model, data, mask_key="promote_mask", policy=policy)

    # Group trades by day
    day_trades = defaultdict(list)
    for t in trades:
        day_trades[t.trade_date].append(t)

    # Day-type classification
    day_types = defaultdict(int)
    for day, ts in day_trades.items():
        pnls = [trade_dollar_pnl(t, policy) for t in ts]
        day_types[classify_day(pnls)] += 1

    # Post-stop analysis
    post_stop = compute_post_stop_stats(day_trades, policy)

    # Blocked trade analysis
    blocked = getattr(metrics, "blocked_trades", [])
    n_blocked = len(blocked)
    blocked_wins = sum(1 for b in blocked if b.get("would_have_won", False))
    blocked_losses = sum(1 for b in blocked if not b.get("would_have_won", True) and np.isfinite(b.get("oracle_pnl", float("nan"))))
    blocked_pnl = [b["oracle_pnl"] for b in blocked if np.isfinite(b.get("oracle_pnl", float("nan")))]

    return {
        "name": name,
        "overrides": overrides,
        "trades": len(trades),
        "traded_days": metrics.traded_days,
        "tpd": metrics.trades_per_day,
        "pf": metrics.profit_factor,
        "dd": metrics.max_account_drawdown,
        "net_pnl": metrics.net_pnl_dollars,
        "wr": metrics.win_rate,
        "avg_pnl_per_trade": metrics.net_pnl_dollars / len(trades) if trades else 0,
        "positive_day_rate": metrics.positive_day_rate,
        # Post-stop
        **post_stop,
        # Day types
        "day_types": dict(day_types),
        # Blocked trades
        "blocked_total": n_blocked,
        "blocked_wins": blocked_wins,
        "blocked_losses": blocked_losses,
        "blocked_net_pnl": sum(blocked_pnl) if blocked_pnl else 0.0,
        "pct_bad_days_rescued": _pct_bad_days_rescued(day_trades, policy),
    }


def print_summary_table(results: list[dict]):
    """Print compact comparison table across all tests."""
    print("\n" + "=" * 110)
    print("  OVERLAY DIAGNOSTIC — SUMMARY TABLE")
    print("=" * 110)
    hdr = f"{'Test':<22} {'Trades':>6} {'TPD':>5} {'PF':>6} {'DD%':>6} {'WR%':>5} {'NetP&L':>9} {'AvgP&L':>8} {'+Day%':>5} {'Blk':>4} {'BlkW':>4} {'BlkL':>4}"
    print(hdr)
    print("-" * 110)
    for r in results:
        print(f"  {r['name']:<20} {r['trades']:>6} {r['tpd']:>5.1f} {r['pf']:>6.3f} "
              f"{r['dd']*100:>5.1f}% {r['wr']*100:>5.1f} ${r['net_pnl']:>8.0f} "
              f"${r['avg_pnl_per_trade']:>7.1f} {r['positive_day_rate']*100:>5.1f} "
              f"{r['blocked_total']:>4} {r['blocked_wins']:>4} {r['blocked_losses']:>4}")


def print_post_stop_table(results: list[dict]):
    """Print post-stop behavior comparison."""
    print("\n" + "=" * 110)
    print("  POST-STOP BEHAVIOR")
    print("=" * 110)
    hdr = (f"{'Test':<22} {'1stOfDay':>9} {'After1stSL':>11} {'n':>4} "
           f"{'After2ndSL':>11} {'n':>4} {'%LossAfter1st':>14} {'%Rescued':>9}")
    print(hdr)
    print("-" * 110)
    for r in results:
        print(f"  {r['name']:<20} ${r['avg_pnl_first_of_day']:>7.1f} "
              f"${r['avg_pnl_after_first_stop']:>9.1f} {r['n_after_first_stop']:>4} "
              f"${r['avg_pnl_after_second_stop']:>9.1f} {r['n_after_second_stop']:>4} "
              f"{r['pct_daily_loss_after_first_loser']:>13.1f}% "
              f"{r['pct_bad_days_rescued']:>8.1f}%")


def print_day_type_table(results: list[dict]):
    """Print day-type classification comparison."""
    all_types = ["green", "red", "recovered", "collapsed", "chop", "no_trades"]
    print("\n" + "=" * 90)
    print("  DAY-TYPE CLASSIFICATION")
    print("=" * 90)
    hdr = f"{'Test':<22}" + "".join(f" {t:>10}" for t in all_types)
    print(hdr)
    print("-" * 90)
    for r in results:
        dt = r["day_types"]
        vals = "".join(f" {dt.get(t, 0):>10}" for t in all_types)
        print(f"  {r['name']:<20}{vals}")


def print_blocked_detail(results: list[dict]):
    """Print blocked trade outcome analysis."""
    print("\n" + "=" * 90)
    print("  BLOCKED TRADE ANALYSIS")
    print("=" * 90)
    for r in results:
        if r["blocked_total"] == 0:
            continue
        bl = r["blocked_total"]
        bw = r["blocked_wins"]
        bloss = r["blocked_losses"]
        bunk = bl - bw - bloss
        print(f"\n  {r['name']}: {bl} blocked entries")
        print(f"    Would have won:  {bw:>4} ({bw/bl*100:.1f}%)")
        print(f"    Would have lost: {bloss:>4} ({bloss/bl*100:.1f}%)")
        if bunk > 0:
            print(f"    Unknown (NaN):   {bunk:>4} ({bunk/bl*100:.1f}%)")
        pnl = r["blocked_net_pnl"]
        print(f"    Net oracle P&L of blocked: {pnl:+.4f} ({'avoided losses' if pnl < 0 else 'lost gains'})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    print("Loading model and data...")
    if args.model:
        from v2.replay import load_model_from_path
        model = load_model_from_path(args.model)
        policy = DEFAULT_POLICY
        manifest = {"experiment_id": args.model}
    else:
        model, policy, manifest = load_best_model()
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    print(f"  Model: {manifest.get('experiment_id', 'unknown')}")
    print(f"  Policy: gate_threshold={policy.gate_threshold}, "
          f"cooldown={policy.cooldown_bars}")

    results = []
    for name, overrides in TESTS:
        print(f"\n{'─'*60}")
        print(f"  Running: {name}")
        print(f"  Overrides: {overrides or '(none)'}")
        print(f"{'─'*60}")
        r = run_single_test(name, overrides, model, data, policy)
        results.append(r)

    # Print all diagnostic tables
    print_summary_table(results)
    print_post_stop_table(results)
    print_day_type_table(results)
    print_blocked_detail(results)

    # Interpretation guide
    print("\n" + "=" * 90)
    print("  INTERPRETATION GUIDE")
    print("=" * 90)
    baseline = results[0]
    control = results[1]
    print(f"\n  Baseline: PF={baseline['pf']:.3f}, DD={baseline['dd']*100:.1f}%, "
          f"Trades={baseline['trades']}")
    print(f"  Control (random 20% skip): PF={control['pf']:.3f}, DD={control['dd']*100:.1f}%, "
          f"Trades={control['trades']}")
    print()
    print("  If entry cap (A) helps most → volume problem, not session-awareness")
    print("  If consecutive-stop (B) helps most → bad-day persistence, need session-state")
    print("  If gate tightening (C) helps most → model has exploitable confidence structure")
    print("  If control matches overlays → mechanical variance reduction, not real signal")
    print("  If none help → entry model too weak, consider RL (Path 3)")


if __name__ == "__main__":
    main()
