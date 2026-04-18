"""Flip-day forensic study for the sequential agent.

Purpose: answer "what is the dominant cause of thesis thrash, and what
architectural or training change would address it?"

Categories:
- oracle_ambiguous: oracle itself changes direction within the day
- post_stop_panic: flips follow stop-loss events
- state_insensitive: agent ignores session state, reacts to bar features as if stateless
- genuine_reversal: large intraday move with legitimate reversal
- noise: no clear pattern

Not a taxonomy exercise. Keep it actionable.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import torch

from v2.core.chain_data import load_sidecar_cached, padded_snapshot
from v2.core.env import (
    TradingEnv,
    ACT_HOLD,
    ACT_ENTER_CALL,
    ACT_ENTER_PUT,
    ACT_EXIT,
)
from v2.core.features import _FEAT_IDX
from v2.core.policy import DEFAULT_POLICY
from v2.replay import load_model_from_path, replay_sequential
from v2.seq_agent import SequentialAgent
from v2.train import TradingModel, LOOKBACK, D_MODEL

# Feature indices
IDX_VIX_REGIME = _FEAT_IDX["vix_regime"]
IDX_TREND = _FEAT_IDX["trend_5min"]
IDX_RET_6 = _FEAT_IDX["ret_6"]
IDX_SESSION_RANGE = _FEAT_IDX["session_range_pct"]


def _compute_oracle_sides(env: TradingEnv, day: str) -> list[int]:
    """Compute oracle entry side for each eligible bar (1=call, -1=put, 0=no trade)."""
    from v2.train import _compute_strict_opportunity

    obs = env.reset(day)
    if env._done:
        return []

    sidecar = env._sidecar
    sides = []
    for step_idx in range(len(env._eligible_bars)):
        _, local_bar = env._eligible_bars[step_idx]
        strict_pos = _compute_strict_opportunity(sidecar, local_bar)

        oracle_side = 0
        if strict_pos:
            bar_ptrs = sidecar["bar_ptrs"]
            start = int(bar_ptrs[local_bar])
            end = int(bar_ptrs[local_bar + 1])
            if end > start:
                best_idx = int(sidecar["bar_best_contract_idx"][local_bar])
                if best_idx >= 0:
                    cidx = int(sidecar["row_contract_idx"][start + best_idx])
                    is_put = int(sidecar["contract_right"][cidx]) == 1
                    oracle_side = -1 if is_put else 1

        sides.append(oracle_side)

        # Step env to keep consistent (oracle action doesn't matter for side analysis)
        obs, _, done, _ = env.step(ACT_HOLD)
        if done:
            break

    return sides


def _count_oracle_side_changes(oracle_sides: list[int]) -> int:
    """Count direction changes in oracle entry signals."""
    last_dir = None
    changes = 0
    for s in oracle_sides:
        if s != 0:
            if last_dir is not None and s != last_dir:
                changes += 1
            last_dir = s
    return changes


def _classify_flip_day(
    summary: dict,
    oracle_sides: list[int],
    day_features: np.ndarray,
    eligible_bar_indices: list[int],
) -> dict:
    """Classify a high-flip day into its dominant cause.

    Returns dict with classification and supporting evidence.
    """
    actions_seq = summary["actions_sequence"]
    exit_events = summary.get("exit_events", [])
    n_flips = summary["side_flips"]

    # Oracle ambiguity
    oracle_changes = _count_oracle_side_changes(oracle_sides)
    oracle_entries = sum(1 for s in oracle_sides if s != 0)

    # Regime features (average over eligible bars)
    bar_features = []
    for bi in eligible_bar_indices:
        if bi < len(day_features):
            bar_features.append(day_features[bi])
    if bar_features:
        feat_mean = np.mean(bar_features, axis=0)
        vix_regime = float(feat_mean[IDX_VIX_REGIME])
        trend = float(feat_mean[IDX_TREND])
        session_range = float(feat_mean[IDX_SESSION_RANGE])
    else:
        vix_regime = trend = session_range = 0.0

    # Find flip positions in action sequence
    flip_positions = []
    last_entry_side = None
    for i, act in enumerate(actions_seq):
        if act == ACT_ENTER_CALL:
            if last_entry_side == "put":
                flip_positions.append(i)
            last_entry_side = "call"
        elif act == ACT_ENTER_PUT:
            if last_entry_side == "call":
                flip_positions.append(i)
            last_entry_side = "put"

    # Check if flips follow stops
    stop_steps = {step for step, reason in exit_events if reason == "stop_loss"}
    flips_after_stop = 0
    for fp in flip_positions:
        # Check if any stop occurred in the 3 steps before this flip
        if any(s in stop_steps for s in range(max(0, fp - 3), fp)):
            flips_after_stop += 1

    # Intraday phase of flips
    bars = summary.get("bars", [])
    flip_phases = Counter()
    for fp in flip_positions:
        if fp < len(bars):
            bar = bars[fp]
            if bar <= 74:
                flip_phases["early"] += 1
            elif bar <= 89:
                flip_phases["mid"] += 1
            else:
                flip_phases["late"] += 1

    # State-insensitivity check: if the agent has session state indicating
    # recent activity (num_trades > 0, bars_since_last_trade small) but still
    # flips, it may be ignoring that state. Heuristic: if most flips happen
    # when the agent just had a trade (within 2 steps of previous entry/exit),
    # and oracle is NOT ambiguous, it's state-insensitive.
    rapid_flips = 0
    entry_or_exit_steps = set()
    for i, act in enumerate(actions_seq):
        if act in (ACT_ENTER_CALL, ACT_ENTER_PUT, ACT_EXIT):
            entry_or_exit_steps.add(i)
    for fp in flip_positions:
        if any(s in entry_or_exit_steps for s in range(max(0, fp - 2), fp)):
            rapid_flips += 1

    # --- Classification ---
    cause = "noise"
    evidence = {}

    if oracle_changes >= n_flips * 0.5 and oracle_changes >= 2:
        cause = "oracle_ambiguous"
        evidence["oracle_side_changes"] = oracle_changes
    elif flips_after_stop >= n_flips * 0.5:
        cause = "post_stop_panic"
        evidence["flips_after_stop"] = flips_after_stop
    elif rapid_flips >= n_flips * 0.6 and oracle_changes < 2:
        cause = "state_insensitive"
        evidence["rapid_flips"] = rapid_flips
    elif session_range > 0.5 and abs(trend) > 0.3:
        cause = "genuine_reversal"
        evidence["session_range"] = f"{session_range:.2f}"
        evidence["trend"] = f"{trend:.2f}"

    return {
        "day": summary["day"],
        "flips": n_flips,
        "cause": cause,
        "oracle_changes": oracle_changes,
        "oracle_entries": oracle_entries,
        "flips_after_stop": flips_after_stop,
        "rapid_flips": rapid_flips,
        "vix_regime": vix_regime,
        "trend": trend,
        "flip_phases": dict(flip_phases),
        "evidence": evidence,
    }


def run_flip_study(
    data_path: str = "v2/data.pt",
    model_path: str = "v2/models/model.pt",
    agent_path: str = "v2/models/seq_agent.pt",
    min_flips: int = 2,
    mask: str = "promote",
) -> None:
    """Run the flip-day forensic study."""
    print("Loading data and models...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    # Load encoder
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {model_path}")
    encoder = load_model_from_path(model_path)
    print(f"  Encoder loaded from {model_path}")
    encoder.eval()

    # Load agent
    agent = SequentialAgent(encoder, context_dim=D_MODEL, freeze_encoder=True)
    if os.path.exists(agent_path):
        seq_ckpt = torch.load(agent_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(seq_ckpt["agent_state_dict"], strict=False)
        print(f"  Agent loaded from {agent_path}")
    else:
        print(f"  WARNING: No agent at {agent_path}, using untrained")

    # Get test days
    mask_key = f"{mask}_mask"
    all_days = sorted(set(data["dates"]))
    if mask_key in data:
        mask_arr = data[mask_key]
        dates = data["dates"]
        test_day_set = set()
        for i in range(len(dates)):
            if mask_arr[i]:
                test_day_set.add(dates[i])
        test_days = sorted(test_day_set)
    else:
        test_days = all_days[-60:]

    # Run replay to get episode summaries
    print(f"Running sequential replay on {len(test_days)} days...")
    metrics, trades, episode_summaries = replay_sequential(
        agent, data, test_days, deterministic=True,
    )

    # Filter to high-flip days
    high_flip_summaries = [s for s in episode_summaries if s["side_flips"] > min_flips]
    if not high_flip_summaries:
        print(f"\nNo days with > {min_flips} side flips. Nothing to analyze.")
        return

    print(f"\n{'=' * 70}")
    print(f"  FLIP-DAY FORENSIC STUDY")
    print(f"  {len(high_flip_summaries)} days with > {min_flips} side flips "
          f"(of {len(episode_summaries)} total)")
    print(f"{'=' * 70}")

    # Build day index for feature lookup
    dates = data["dates"]
    bar_of_day = data["bar_of_day"]
    features = data["X"]

    day_bar_ranges = {}
    for i in range(len(dates)):
        d = dates[i]
        if d not in day_bar_ranges:
            day_bar_ranges[d] = [i, i]
        day_bar_ranges[d][1] = i

    # Create env for oracle analysis
    env = TradingEnv(data, DEFAULT_POLICY, "v2/data_sidecars", encoder, "cpu", LOOKBACK)

    # Analyze each high-flip day
    results = []
    for s in sorted(high_flip_summaries, key=lambda x: -x["side_flips"]):
        day = s["day"]

        # Get oracle sides
        oracle_sides = _compute_oracle_sides(env, day)

        # Get day features
        if day in day_bar_ranges:
            start_gi, end_gi = day_bar_ranges[day]
            day_features = features[start_gi:end_gi + 1].numpy()
            eligible_bars = [int(bar_of_day[start_gi + j]) for j in range(end_gi - start_gi + 1)]
        else:
            day_features = np.zeros((1, features.shape[1]))
            eligible_bars = []

        result = _classify_flip_day(s, oracle_sides, day_features, eligible_bars)
        results.append(result)

    # Print results table
    print(f"\n  {'Day':<12s} {'Flips':>5s} {'Cause':<20s} {'Oracle':>6s} "
          f"{'PostStop':>8s} {'Rapid':>5s} {'VIX':>5s} {'Trend':>6s} {'Phases'}")
    print(f"  {'-' * 90}")
    for r in results:
        phases = r["flip_phases"]
        phase_str = "/".join(f"{p}:{n}" for p, n in sorted(phases.items()))
        print(f"  {r['day']:<12s} {r['flips']:5d} {r['cause']:<20s} "
              f"{r['oracle_changes']:6d} {r['flips_after_stop']:8d} "
              f"{r['rapid_flips']:5d} {r['vix_regime']:5.2f} {r['trend']:6.2f} "
              f"{phase_str}")

    # Aggregate classification
    cause_counts = Counter(r["cause"] for r in results)
    cause_flips = {}
    for r in results:
        cause_flips[r["cause"]] = cause_flips.get(r["cause"], 0) + r["flips"]

    total_flips_analyzed = sum(r["flips"] for r in results)
    print(f"\n  Classification Summary")
    print(f"  {'Cause':<20s} {'Days':>5s} {'Flips':>6s} {'% of Flips':>10s}")
    print(f"  {'-' * 45}")
    for cause, count in cause_counts.most_common():
        flips = cause_flips[cause]
        print(f"  {cause:<20s} {count:5d} {flips:6d} {100*flips/total_flips_analyzed:9.1f}%")

    # Actionable summary
    dominant = cause_counts.most_common(1)[0][0] if cause_counts else "unknown"
    print(f"\n  Dominant cause: {dominant}")
    if dominant == "oracle_ambiguous":
        print("  -> Oracle itself is indecisive on these days.")
        print("     Consider: oracle smoothing, side-commitment logic, or")
        print("     filtering ambiguous oracle days from BC training.")
    elif dominant == "post_stop_panic":
        print("  -> Agent reverses direction after stop-loss events.")
        print("     Consider: cooldown period in oracle after stops, or")
        print("     explicit post-stop stand-down reward in RL phase.")
    elif dominant == "state_insensitive":
        print("  -> Agent ignores session history, reacts to bar features alone.")
        print("     Consider: increasing session_proj capacity, adding attention")
        print("     over session history, or curriculum that rewards state use.")
    elif dominant == "genuine_reversal":
        print("  -> High-vol reversal days with legitimate side changes.")
        print("     These may be acceptable. Consider stand-down logic for")
        print("     extreme session_range days instead of penalizing flips.")
    else:
        print("  -> No clear dominant pattern. May need richer features or")
        print("     more training data to disambiguate.")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Flip-day forensic study")
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--model", type=str, default="v2/models/model.pt")
    parser.add_argument("--agent", type=str, default="v2/models/seq_agent.pt")
    parser.add_argument("--min-flips", type=int, default=2)
    parser.add_argument("--mask", type=str, default="promote")
    args = parser.parse_args()

    run_flip_study(
        data_path=args.data,
        model_path=args.model,
        agent_path=args.agent,
        min_flips=args.min_flips,
        mask=args.mask,
    )


if __name__ == "__main__":
    main()
