"""H3a look-ahead audit: prove candidate features depend only on past+current bars.

Discipline anchor: the 2026-04-25 oracle-gate label-leakage retraction. Every
new feature in trade_state must be causal — its value at bar t must be a
function of bars[0..t], NOT bars[t+1..].

Audit protocol per feature:
  1. Define the feature as a pure function f(pnls[0..t], ...) -> scalar
  2. Load real per-trade pnl trajectories (fw_trade_trajectories.pkl)
  3. For each trade, for each bar t (1 <= t < len-1):
     a. Compute feature_orig = f(trajectory, t)
     b. Mutate trajectory: replace pnls[t+1:] with random values
     c. Compute feature_mut = f(mutated_trajectory, t)
     d. Assert feature_orig == feature_mut (bit-exact)
  4. PASS only if every (trade, bar) passes; FAIL otherwise

Three candidate features for H3a:
  - realized_vol_10bar: rolling std of pnl over last min(10, t+1) bars
  - pnl_velocity_5bar: (current_pnl - pnl[max(0, t-5)]) / max(1, lookback)
  - mfe_decay_rate: (current_pnl - mfe_so_far) / max(1, mfe_bar_age)

These three feature functions are the SAME implementations that will be
plumbed into v3/layer3/common.py:_build_trade_data. After this audit passes,
the v3/layer3 implementation must be byte-equivalent to these definitions.
"""
from __future__ import annotations

import pickle

import numpy as np


# ---------------------------------------------------------------------------
# Candidate feature implementations (must be pure functions of bars[0..t]).
# ---------------------------------------------------------------------------


def realized_vol_10bar(pnls: np.ndarray, t: int) -> float:
    """Rolling std of pnls over last 10 bars (or fewer if early in trade)."""
    start = max(0, t - 9)
    window = pnls[start : t + 1]
    if window.size < 2:
        return 0.0
    return float(np.std(window, ddof=0))


def pnl_velocity_5bar(pnls: np.ndarray, t: int) -> float:
    """Avg per-bar pnl change over last 5 bars: (pnls[t] - pnls[t-5]) / 5,
    clipped to actual lookback when t < 5."""
    lookback = min(5, t)
    if lookback == 0:
        return 0.0
    return float((pnls[t] - pnls[t - lookback]) / lookback)


def mfe_decay_rate(pnls: np.ndarray, t: int) -> float:
    """(current_pnl - mfe_so_far) / max(1, mfe_bar_age).

    Rate of give-back from running peak. 0 if no give-back or no peak yet."""
    history = pnls[: t + 1]
    if history.size == 0:
        return 0.0
    mfe = float(np.max(history))
    mfe_idx = int(np.argmax(history))
    mfe_bar_age = t - mfe_idx
    return float((pnls[t] - mfe) / max(1, mfe_bar_age))


FEATURES = {
    "realized_vol_10bar": realized_vol_10bar,
    "pnl_velocity_5bar": pnl_velocity_5bar,
    "mfe_decay_rate": mfe_decay_rate,
}


# ---------------------------------------------------------------------------
# Audit driver.
# ---------------------------------------------------------------------------


def audit_feature(
    name: str,
    fn,
    trajectories: list[dict],
    rng: np.random.Generator,
    max_trades: int = 50,
) -> tuple[int, int, list[str]]:
    """Run mutate-future test for a single feature. Returns (passes, fails, errors)."""
    passes = 0
    fails = 0
    errors: list[str] = []

    sample = trajectories[:max_trades] if len(trajectories) > max_trades else trajectories

    for traj in sample:
        bars = traj.get("pnl_per_bar", [])
        if len(bars) < 3:
            continue
        pnls = np.array([b["pnl"] for b in bars], dtype=np.float64)
        n = len(pnls)

        # Test every bar except the first and last (where future is empty)
        for t in range(0, n - 1):
            # Original feature value
            try:
                v_orig = fn(pnls, t)
            except Exception as e:
                errors.append(f"{name} trade={traj.get('trade_id','?')} t={t} orig raised {e!r}")
                fails += 1
                continue

            # Mutate future bars: replace pnls[t+1:] with random values in plausible range
            mutated = pnls.copy()
            future_n = n - (t + 1)
            mutated[t + 1 :] = rng.uniform(-1e4, 1e4, size=future_n)

            # Recompute on mutated trajectory
            try:
                v_mut = fn(mutated, t)
            except Exception as e:
                errors.append(f"{name} trade={traj.get('trade_id','?')} t={t} mut raised {e!r}")
                fails += 1
                continue

            # Bit-exact: a causal feature must produce identical output
            if v_orig == v_mut:
                passes += 1
            else:
                fails += 1
                if len(errors) < 5:
                    errors.append(
                        f"{name} trade={traj.get('trade_id','?')} t={t}: "
                        f"orig={v_orig!r} mut={v_mut!r} (LEAK)"
                    )

    return passes, fails, errors


def main():
    print("=== H3a Look-ahead Audit ===")
    print("Testing 3 candidate features for causality (mutate-future-bars test)\n")

    with open("v3/artifacts/research/fw_trade_trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trade trajectories from fw_trade_trajectories.pkl\n")

    rng = np.random.default_rng(seed=42)

    all_pass = True
    for name, fn in FEATURES.items():
        passes, fails, errors = audit_feature(name, fn, trajectories, rng, max_trades=50)
        status = "PASS" if fails == 0 else "FAIL"
        print(f"  {name:25} {status}  ({passes:5d} bars audited, {fails} leaks)")
        for err in errors[:5]:
            print(f"    {err}")
        if fails > 0:
            all_pass = False

    print()
    if all_pass:
        print("✓ AUDIT PASSED. All 3 features are causal. Safe to integrate into v3/layer3/common.py.")
        print("  Remember: the v3/layer3 implementation MUST be byte-equivalent to these definitions.")
        return 0
    else:
        print("✗ AUDIT FAILED. Do NOT integrate any feature with leaks. Fix the implementation.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
