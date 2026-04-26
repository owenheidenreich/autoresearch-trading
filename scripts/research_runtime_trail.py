"""Runtime trail-stop research: extend oracle's exit using only in-trade pnl.

The L3 oracle exits at predicted bar X. If at bar X pnl is still rising
(current pnl is the running peak), we may want to hold longer. This is a
LEAK-FREE rule: only uses pnl observed up to current bar, no future info.

Rules to test:
  R0 (baseline): exit at oracle's predicted bar regardless.
  R1 (extend if peak): if at oracle's bar, current pnl == running peak,
      extend by N bars (then re-evaluate).
  R2 (trail-stop after oracle): from oracle's bar onward, trail-stop
      with width W (exit when pnl drops W below running peak).
  R3 (peak-and-hold-positive): never exit while pnl > some threshold X
      AND still rising; cut on reversal.
  R4 (peak-time-decay): if running peak hasn't moved up in N bars, exit.

All rules use ONLY:
  - pnl[entry .. current_bar] — in-trade observed
  - oracle's predicted exit bar (computed at entry)
  - constants

NO future info. NO labels. NO classifier.

Output: comparison table of rule performance on FW trajectories.
"""
from __future__ import annotations

import pickle

import numpy as np


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def exit_at_first_offset_at_or_after(traj, target_offset):
    """Returns the pnl at the first bar offset >= target_offset, or last bar pnl."""
    for b in traj["pnl_per_bar"]:
        if b["bar_offset"] >= target_offset:
            return b["pnl"], b["bar_offset"]
    if traj["pnl_per_bar"]:
        return traj["pnl_per_bar"][-1]["pnl"], traj["pnl_per_bar"][-1]["bar_offset"]
    return 0.0, 0


def rule_R1_extend_if_peak(traj, oracle_offset, extend_by_n=10, max_extends=5):
    """Exit at oracle's bar UNLESS current pnl == running peak, then extend
    N bars at a time, up to max_extends."""
    bars = traj["pnl_per_bar"]
    if not bars: return 0.0, 0
    # Find oracle bar position
    pnls = []
    target = oracle_offset
    for b in bars:
        pnls.append(b["pnl"])
        if b["bar_offset"] >= target:
            running_peak = max(pnls)
            extends_used = 0
            while b["pnl"] >= running_peak * 0.99 and extends_used < max_extends:
                target += extend_by_n
                extends_used += 1
            if b["bar_offset"] >= target:
                return b["pnl"], b["bar_offset"]
    if bars: return bars[-1]["pnl"], bars[-1]["bar_offset"]
    return 0.0, 0


def rule_R2_trail_after_oracle(traj, oracle_offset, trail_width=300):
    """From oracle's bar onward, exit when pnl drops trail_width below
    running peak. Until oracle's bar, just track peak."""
    bars = traj["pnl_per_bar"]
    if not bars: return 0.0, 0
    running_peak = -np.inf
    triggered = False
    for b in bars:
        running_peak = max(running_peak, b["pnl"])
        if b["bar_offset"] >= oracle_offset:
            triggered = True
        if triggered and (running_peak - b["pnl"]) >= trail_width:
            return b["pnl"], b["bar_offset"]
    return bars[-1]["pnl"], bars[-1]["bar_offset"]


def rule_R3_peak_and_positive(traj, oracle_offset, hold_threshold=200):
    """If at oracle's bar pnl >= hold_threshold AND pnl == peak, hold;
    else exit at oracle's bar."""
    bars = traj["pnl_per_bar"]
    if not bars: return 0.0, 0
    pnls = []
    for b in bars:
        pnls.append(b["pnl"])
        if b["bar_offset"] >= oracle_offset:
            running_peak = max(pnls)
            if b["pnl"] >= hold_threshold and b["pnl"] >= running_peak * 0.99:
                # Hold 1 more bar (very simple)
                continue
            return b["pnl"], b["bar_offset"]
    return bars[-1]["pnl"], bars[-1]["bar_offset"]


def rule_R4_peak_decay(traj, oracle_offset, no_new_peak_bars=15):
    """Exit when peak hasn't moved up for N bars (after oracle's bar)."""
    bars = traj["pnl_per_bar"]
    if not bars: return 0.0, 0
    running_peak = -np.inf
    bars_since_peak = 0
    triggered = False
    for b in bars:
        if b["pnl"] > running_peak:
            running_peak = b["pnl"]
            bars_since_peak = 0
        else:
            bars_since_peak += 1
        if b["bar_offset"] >= oracle_offset:
            triggered = True
        if triggered and bars_since_peak >= no_new_peak_bars:
            return b["pnl"], b["bar_offset"]
    return bars[-1]["pnl"], bars[-1]["bar_offset"]


def main():
    with open("v3/artifacts/research/fw_trade_trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} forward-walk trajectories")

    import numpy as np
    seeds = [42, 43, 44, 45, 46]
    oracles = {}
    for s in seeds:
        try:
            o = np.load(
                f"v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{s}_balanced_fresh.npz",
                allow_pickle=True,
            )
            oracles[s] = (o["l3_exit_pnl"], o["l3_exit_bar"])
        except Exception as e:
            print(f"  oracle for seed {s} not loadable: {e}")

    # Bundle for joining
    from v3.layer2.common import load_export_bundle
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    key_to_row = (rows[["day","bar_index","__row__"]].drop_duplicates(subset=["day","bar_index"])
                                                      .set_index(["day","bar_index"])["__row__"]
                                                      .to_dict())
    # Each FW trade was generated by ONE seed; we don't know which.
    # Use the earliest seed that has a finite prediction.
    # Add oracle_offset to each trajectory.
    enriched = []
    for t in trajectories:
        key = (t["day"], t["bar_index"])
        r = key_to_row.get(key)
        if r is None: continue
        # We need the chosen action_id. We don't have it in trajectories — would
        # need to re-look up from forward_walk_chosen_seedX.pkl. Take first seed
        # with a finite prediction for any side in the trade's risk band.
        oracle_pnl_pred = None
        oracle_exit_offset = None
        # Take from the trade's seed (we have it)
        s = t["seed"]
        if s in oracles:
            l3_pnl, l3_bar = oracles[s]
            # Try each action_id from 1 to 24, take any finite prediction
            for a in range(1, 25):
                if np.isfinite(l3_pnl[r, a]) and l3_bar[r, a] >= 0:
                    oracle_pnl_pred = float(l3_pnl[r, a])
                    oracle_exit_bar = int(l3_bar[r, a])
                    oracle_exit_offset = oracle_exit_bar - t["bar_index"]
                    break
        if oracle_exit_offset is None: continue
        if oracle_exit_offset < 5: oracle_exit_offset = 5  # sanity floor
        t = dict(t)
        t["oracle_offset"] = oracle_exit_offset
        t["oracle_pnl_pred"] = oracle_pnl_pred
        enriched.append(t)
    print(f"Enriched {len(enriched)} trajectories with oracle exit_offset (median offset {np.median([t['oracle_offset'] for t in enriched])})")
    print()

    # Apply rules
    print(f"{'rule':>30} {'n':>4} {'PF':>7} {'sum':>9} {'mean':>7} {'median':>8} {'mean_offset':>13}")

    # R0 baseline: exit at oracle's predicted bar
    rule_results = {}
    for label, fn in [
        ("R0: exit at oracle bar (baseline)", lambda t: exit_at_first_offset_at_or_after(t, t["oracle_offset"])),
        ("R1: extend by 10×5 if peak", lambda t: rule_R1_extend_if_peak(t, t["oracle_offset"], 10, 5)),
        ("R1: extend by 20×5 if peak", lambda t: rule_R1_extend_if_peak(t, t["oracle_offset"], 20, 5)),
        ("R2: trail $300 from oracle", lambda t: rule_R2_trail_after_oracle(t, t["oracle_offset"], 300)),
        ("R2: trail $500 from oracle", lambda t: rule_R2_trail_after_oracle(t, t["oracle_offset"], 500)),
        ("R2: trail $1000 from oracle", lambda t: rule_R2_trail_after_oracle(t, t["oracle_offset"], 1000)),
        ("R3: peak+positive ($200)", lambda t: rule_R3_peak_and_positive(t, t["oracle_offset"], 200)),
        ("R4: no_new_peak 10 bars", lambda t: rule_R4_peak_decay(t, t["oracle_offset"], 10)),
        ("R4: no_new_peak 15 bars", lambda t: rule_R4_peak_decay(t, t["oracle_offset"], 15)),
        ("R4: no_new_peak 20 bars", lambda t: rule_R4_peak_decay(t, t["oracle_offset"], 20)),
    ]:
        pnls = []
        offsets = []
        for t in enriched:
            p, off = fn(t)
            pnls.append(p); offsets.append(off)
        pnls = np.asarray(pnls); offsets = np.asarray(offsets)
        rule_results[label] = pnls
        print(f"{label:>30} {len(pnls):>4} {pf(pnls):>7.3f} {pnls.sum():>9.0f} {pnls.mean():>7.0f} {np.median(pnls):>8.0f} {offsets.mean():>13.0f}")
    print()

    # Reported labels for comparison
    print("=== Reported labels (FW eval) ===")
    of = [t.get("fwd_pnl_hybrid_with_oracle", 0) for t in enriched]
    ts = [t.get("fwd_pnl_time_stop", 0) for t in enriched]
    print(f"  fwd_pnl_hybrid_with_oracle: PF {pf(of):.3f}, sum ${sum(of):.0f}")
    print(f"  fwd_pnl_time_stop:          PF {pf(ts):.3f}, sum ${sum(ts):.0f}")


if __name__ == "__main__":
    main()
