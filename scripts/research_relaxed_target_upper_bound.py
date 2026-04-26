"""Upper-bound analysis for the H1 relaxed-target oracle.

We can't run the new oracle's predictions without rebuilding (~50 min/seed).
But we can simulate the BEST CASE: what would a perfect oracle trained
on `current >= 0.85 * suffix_max` produce on these same trades?

The "perfect relaxed oracle" picks the FIRST bar where current_pnl >=
0.85 * future_peak (computed from actual trajectory). This is the
training target's perfect-information optimum. It's an UPPER BOUND
because real models don't predict perfectly.

If this upper bound shows big PF lift, H1 is worth pursuing.
If this upper bound shows little lift, H1 won't help even with perfect prediction.

Notes:
  - Compute per-bar pnl trajectory using the same formula as
    _time_stop_pnl (entry_ask, exit_bid, commission).
  - Sweep target threshold values (0.70, 0.85, 0.95, 1.0) to see how
    much the threshold affects PF.
  - Compare to current oracle's reported PF (1.89 on FW).
"""
from __future__ import annotations

import pickle

import numpy as np


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def perfect_relaxed_oracle_exit(traj, threshold):
    """Pick first bar where current_pnl >= threshold * future_peak.
    Future_peak is the max from current bar to end.

    threshold=1.0: original oracle (exit at every local-max-equiv).
    threshold<1.0: relaxed oracle (exit when within threshold of future peak)."""
    bars = traj["pnl_per_bar"]
    if not bars: return 0.0, 0

    pnls = np.array([b["pnl"] for b in bars])
    # suffix_max[i] = max of pnls[i+1:]
    suffix_max = np.full(len(pnls), -np.inf)
    for i in range(len(pnls) - 2, -1, -1):
        suffix_max[i] = max(pnls[i+1], suffix_max[i+1])
    suffix_max[-1] = pnls[-1]  # last bar exits regardless

    for i in range(len(pnls)):
        smax = suffix_max[i]
        if smax > 0:
            if pnls[i] >= threshold * smax:
                return float(pnls[i]), int(bars[i]["bar_offset"])
        else:
            # If future is non-positive, fall back: exit when current >= future
            if pnls[i] >= smax:
                return float(pnls[i]), int(bars[i]["bar_offset"])
    return float(pnls[-1]), int(bars[-1]["bar_offset"])


def main():
    with open("v3/artifacts/research/fw_trade_trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} FW trajectories\n")

    print("=== Upper bound: perfect oracle at different relaxation thresholds ===")
    print(f"{'threshold':>10} {'PF':>7} {'sum':>10} {'mean':>8} {'mean_offset':>13}")
    for thr in [0.50, 0.70, 0.85, 0.90, 0.95, 1.00]:
        pnls = []
        offsets = []
        for t in trajectories:
            p, off = perfect_relaxed_oracle_exit(t, thr)
            pnls.append(p)
            offsets.append(off)
        pnls = np.array(pnls)
        print(f"  {thr:.2f}     {pf(pnls):>7.3f} {pnls.sum():>10.0f} {pnls.mean():>8.0f} {np.mean(offsets):>13.1f}")
    print()

    # Compare to "perfect peak" (oracle that exits at the maximum pnl bar)
    perfect_peaks = []
    for t in trajectories:
        if t["pnl_per_bar"]:
            perfect_peaks.append(max(b["pnl"] for b in t["pnl_per_bar"]))
    print(f"Per-trade peak (true ceiling): PF {pf(perfect_peaks):.3f}, sum ${sum(perfect_peaks):.0f}")
    print()

    # Compare to fixed-bar
    pnls_180 = []
    for t in trajectories:
        for b in t["pnl_per_bar"]:
            if b["bar_offset"] >= 180:
                pnls_180.append(b["pnl"])
                break
        else:
            if t["pnl_per_bar"]: pnls_180.append(t["pnl_per_bar"][-1]["pnl"])
    print(f"Fixed bar+180:                 PF {pf(pnls_180):.3f}, sum ${sum(pnls_180):.0f}")
    print()

    # Per-trade comparison: does relaxed beat strict on each trade?
    print("=== Per-trade: relaxed (0.85) vs strict (1.00) exit pnl ===")
    same = 0; relax_better = 0; strict_better = 0
    for t in trajectories:
        p_relax, _ = perfect_relaxed_oracle_exit(t, 0.85)
        p_strict, _ = perfect_relaxed_oracle_exit(t, 1.00)
        if abs(p_relax - p_strict) < 0.5: same += 1
        elif p_relax > p_strict + 0.5: relax_better += 1
        else: strict_better += 1
    print(f"  same:           {same}/{len(trajectories)}")
    print(f"  relaxed better: {relax_better}/{len(trajectories)}")
    print(f"  strict better:  {strict_better}/{len(trajectories)}")


if __name__ == "__main__":
    main()
