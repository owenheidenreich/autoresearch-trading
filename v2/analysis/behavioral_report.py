"""Behavioral report for sequential agent replay.

Prints session-level diagnostics: trades/day distribution, hold rate,
call/put balance, side flips, intraday phase breakdown, and post-stop
behavior analysis (same-side re-entry vs opposite-side reversal vs stand-down).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

# Action constants (mirror env.py)
ACT_HOLD = 0
ACT_ENTER_CALL = 1
ACT_ENTER_PUT = 2
ACT_EXIT = 3

# Intraday phase boundaries (local bar indices)
EARLY_END = 74
MID_END = 89


def _phase_label(bar: int) -> str:
    if bar <= EARLY_END:
        return "early"
    elif bar <= MID_END:
        return "mid"
    return "late"


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  0.0%"
    return f"{100 * n / total:5.1f}%"


def print_behavioral_report(
    episode_summaries: list[dict],
    trades: list,
    metrics,
) -> None:
    """Print structured behavioral diagnostics for sequential replay."""
    if not episode_summaries:
        print("\n  (no episodes to report)")
        return

    n_days = len(episode_summaries)
    print("\n" + "=" * 60)
    print("  BEHAVIORAL REPORT")
    print("=" * 60)

    # --- 1. Trades/day distribution ---
    trades_per_day = [s["trades"] for s in episode_summaries]
    tpd = np.array(trades_per_day, dtype=float)
    print(f"\n  Trades/Day Distribution (n={n_days} days)")
    print(f"    min={int(tpd.min()):3d}  p25={np.percentile(tpd, 25):4.1f}  "
          f"median={np.median(tpd):4.1f}  p75={np.percentile(tpd, 75):4.1f}  "
          f"max={int(tpd.max()):3d}")

    zero_trade_days = sum(1 for t in trades_per_day if t == 0)
    single_entry_days = sum(1 for t in trades_per_day if t == 1)
    print(f"    Zero-trade days: {zero_trade_days}/{n_days} ({100*zero_trade_days/n_days:.1f}%)")
    print(f"    Single-entry days: {single_entry_days}/{n_days} ({100*single_entry_days/n_days:.1f}%)")

    # --- 2. Hold rate ---
    hold_rates = [s["hold_rate"] for s in episode_summaries]
    hr = np.array(hold_rates)
    print(f"\n  Hold Rate")
    print(f"    mean={hr.mean():.3f}  std={hr.std():.3f}  min={hr.min():.3f}  max={hr.max():.3f}")

    # --- 3. Call/put balance ---
    total_calls = sum(s["actions"].get(ACT_ENTER_CALL, 0) for s in episode_summaries)
    total_puts = sum(s["actions"].get(ACT_ENTER_PUT, 0) for s in episode_summaries)
    total_entries = total_calls + total_puts
    print(f"\n  Call/Put Balance")
    print(f"    Calls: {total_calls}  Puts: {total_puts}  "
          f"Split: {_pct(total_calls, total_entries)} / {_pct(total_puts, total_entries)}")

    # --- 4. Side flips ---
    flips = [s["side_flips"] for s in episode_summaries]
    flip_arr = np.array(flips, dtype=float)
    flip_days = sum(1 for f in flips if f >= 1)
    high_flip_days = [(s["day"], s["side_flips"]) for s in episode_summaries if s["side_flips"] > 2]
    print(f"\n  Side Flips")
    print(f"    total={int(flip_arr.sum())}  mean={flip_arr.mean():.2f}/day  max={int(flip_arr.max())}")
    print(f"    Days with >= 1 flip: {flip_days}/{n_days} ({100*flip_days/n_days:.1f}%)")
    if high_flip_days:
        top = sorted(high_flip_days, key=lambda x: -x[1])[:10]
        print(f"    High-flip days (>2): {', '.join(f'{d}({f})' for d, f in top)}")

    # --- 5. Exits ---
    total_agent_exits = sum(s["actions"].get(ACT_EXIT, 0) for s in episode_summaries)
    all_exit_reasons = Counter()
    for s in episode_summaries:
        for reason, count in s.get("exit_reasons", {}).items():
            all_exit_reasons[reason] += count
    print(f"\n  Exits")
    print(f"    Agent EXIT actions: {total_agent_exits}")
    if all_exit_reasons:
        for reason, count in all_exit_reasons.most_common():
            print(f"    {reason}: {count}")

    # --- 6. Action counts by intraday phase ---
    phase_actions = {"early": Counter(), "mid": Counter(), "late": Counter()}
    for s in episode_summaries:
        actions_seq = s.get("actions_sequence", [])
        bars = s.get("bars", [])
        for act, bar in zip(actions_seq, bars):
            phase = _phase_label(bar)
            phase_actions[phase][act] += 1

    print(f"\n  Actions by Intraday Phase")
    print(f"    {'Phase':>6s}  {'HOLD':>6s}  {'CALL':>6s}  {'PUT':>6s}  {'EXIT':>6s}  {'Total':>6s}")
    for phase in ["early", "mid", "late"]:
        c = phase_actions[phase]
        total = sum(c.values())
        print(f"    {phase:>6s}  {c[ACT_HOLD]:6d}  {c[ACT_ENTER_CALL]:6d}  "
              f"{c[ACT_ENTER_PUT]:6d}  {c[ACT_EXIT]:6d}  {total:6d}")

    # --- 7. Post-stop behavior ---
    _print_post_stop_behavior(episode_summaries)

    print("=" * 60)


def _print_post_stop_behavior(episode_summaries: list[dict]) -> None:
    """Analyze what the agent does after stop-loss events.

    For each stop-loss, find the next entry action and classify:
    - same_side_reentry: re-enters the same direction (thesis persistence)
    - opposite_side_reversal: flips direction after stop (panic reversal)
    - stand_down: no re-entry within the day (caution)
    """
    same_side = 0
    opposite_side = 0
    stand_down = 0
    total_stops = 0

    for s in episode_summaries:
        exit_events = s.get("exit_events", [])  # [(step_idx, reason), ...]
        actions_seq = s.get("actions_sequence", [])
        if not exit_events or not actions_seq:
            continue

        stop_steps = [step for step, reason in exit_events if reason == "stop_loss"]
        if not stop_steps:
            continue

        # Build entry index: list of (step_idx, side) for all entries
        entries = []
        for i, act in enumerate(actions_seq):
            if act == ACT_ENTER_CALL:
                entries.append((i, "call"))
            elif act == ACT_ENTER_PUT:
                entries.append((i, "put"))

        for stop_step in stop_steps:
            total_stops += 1

            # What side was the stopped position?
            # Find the most recent entry before (or at) the stop step
            stopped_side = None
            for eidx, eside in reversed(entries):
                if eidx <= stop_step:
                    stopped_side = eside
                    break

            if stopped_side is None:
                stand_down += 1
                continue

            # Find the next entry after the stop
            next_entry = None
            for eidx, eside in entries:
                if eidx > stop_step:
                    next_entry = (eidx, eside)
                    break

            if next_entry is None:
                stand_down += 1
                continue

            _, next_side = next_entry
            if next_side == stopped_side:
                same_side += 1
            else:
                opposite_side += 1

    print(f"\n  Post-Stop Behavior (n={total_stops} stop-loss events)")
    if total_stops > 0:
        print(f"    Same-side re-entry:      {same_side:4d} ({_pct(same_side, total_stops)})  -- thesis persistence")
        print(f"    Opposite-side reversal:   {opposite_side:4d} ({_pct(opposite_side, total_stops)})  -- panic reversal")
        print(f"    Stand-down (no re-entry): {stand_down:4d} ({_pct(stand_down, total_stops)})  -- caution")
    else:
        print("    No stop-loss events observed")
