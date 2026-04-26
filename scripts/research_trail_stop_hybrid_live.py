"""Trail-stop research with proper hybrid_live_utility scoring.

Earlier trail-stop research used raw mid-based pnl which differed from
the canonical reported pnls. This version applies hybrid_live_utility to
the chosen exit bar's raw pnl, matching deployment scoring.

For each FW trade:
  - Build per-bar raw pnl trajectory from chain data.
  - Apply each candidate exit rule (oracle bar, fixed bar, trail rules,
    extension rules) to pick exit bar.
  - Score the chosen bar's raw pnl through hybrid_live_utility.

This gives apples-to-apples comparison with reported labels.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def score_pnl(pnl_raw: float, entry_features: dict, exit_bar: int) -> float:
    """Apply hybrid_live_utility for production-equivalent scoring."""
    return hybrid_live_utility(
        pnl_raw if np.isfinite(pnl_raw) else None,
        entry_mid=entry_features["entry_mid"],
        spread_fraction=entry_features["entry_sf"] if np.isfinite(entry_features["entry_sf"]) else 0.0,
        stopout_risk=entry_features["stopout_risk"] if np.isfinite(entry_features["stopout_risk"]) else 0.0,
        entry_bar=int(entry_features["entry_bar"]),
        exit_bar=int(exit_bar) if exit_bar >= 0 else None,
        session_end_bar=375,
    )


def main():
    with open("v3/artifacts/research/fw_trade_trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} FW trajectories")

    # Need entry_features per trajectory
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]].drop_duplicates(subset=["day","bar_index"])
                  .set_index(["day","bar_index"])["__row__"].to_dict())

    # Need oracle predictions per traj
    seeds = [42, 43, 44, 45, 46]
    oracles = {}
    for s in seeds:
        try:
            o = np.load(f"v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{s}_balanced_fresh.npz", allow_pickle=True)
            oracles[s] = (o["l3_exit_pnl"], o["l3_exit_bar"])
        except: continue

    # FW chosen trades for action_id lookup
    fw_chosen = {}
    for s in seeds:
        try:
            df = pd.read_pickle(f"v3/artifacts/forward_walk/forward_walk_chosen_seed{s}.pkl")
            fw_chosen[s] = df.set_index(["day","bar_index"])
        except: continue

    enriched = []
    for t in trajectories:
        s = t["seed"]
        if s not in oracles or s not in fw_chosen: continue
        key = (t["day"], t["bar_index"])
        r = key_to_row.get(key)
        if r is None: continue
        try:
            chosen_row = fw_chosen[s].loc[key]
        except KeyError:
            continue
        if isinstance(chosen_row, pd.DataFrame): chosen_row = chosen_row.iloc[0]
        a = int(chosen_row.get("chosen_action_id", 0))
        if a <= 0: continue

        l3_pnl, l3_bar = oracles[s]
        if not (np.isfinite(l3_pnl[r,a]) and l3_bar[r,a] >= 0): continue
        oracle_pnl_pred = float(l3_pnl[r,a])
        oracle_exit_bar = int(l3_bar[r,a])
        oracle_offset = oracle_exit_bar - t["bar_index"]

        em = float(al["entry_fill_mid"][r,a])
        ef = float(al["entry_spread_fraction"][r,a])
        eb = float(al["entry_fill_bar"][r,a])
        so = float(al["stopout_risk"][r,a])
        if not np.isfinite(em) or not np.isfinite(eb): continue

        t = dict(t)
        t["action_id"] = a
        t["oracle_pnl_pred"] = oracle_pnl_pred
        t["oracle_offset"] = oracle_offset
        t["oracle_exit_bar"] = oracle_exit_bar
        t["entry_features"] = {
            "entry_mid": em, "entry_sf": ef, "entry_bar": eb, "stopout_risk": so,
        }
        enriched.append(t)
    print(f"Enriched: {len(enriched)} trajectories\n")

    # Helper: pick exit pnl & bar at a given offset from entry
    def exit_at_offset(traj, offset):
        for b in traj["pnl_per_bar"]:
            if b["bar_offset"] >= offset:
                return b["pnl"], b["bar"]
        if traj["pnl_per_bar"]:
            return traj["pnl_per_bar"][-1]["pnl"], traj["pnl_per_bar"][-1]["bar"]
        return 0.0, traj["bar_index"]

    # Helper: trail-stop
    def trail_stop_exit(traj, trail_pct, min_offset=30, peak_floor=50):
        bars = traj["pnl_per_bar"]
        if not bars: return 0.0, traj["bar_index"]
        running_peak = -np.inf
        for b in bars:
            running_peak = max(running_peak, b["pnl"])
            if b["bar_offset"] < min_offset: continue
            if running_peak >= peak_floor and b["pnl"] < running_peak * (1 - trail_pct):
                return b["pnl"], b["bar"]
        return bars[-1]["pnl"], bars[-1]["bar"]

    print(f"=== Strategies scored with hybrid_live_utility ===")
    print(f"{'rule':>40} {'PF':>7} {'sum':>9} {'mean':>7}")

    # 1. Oracle baseline
    oracle_scores = []
    for t in enriched:
        scored = score_pnl(t["oracle_pnl_pred"], t["entry_features"], t["oracle_exit_bar"])
        if np.isfinite(scored): oracle_scores.append(scored)
    print(f"{'R0: oracle (baseline reproduction)':>40} {pf(oracle_scores):>7.3f} {sum(oracle_scores):>9.0f} {np.mean(oracle_scores):>7.0f}")

    # 2. Fixed bar offsets
    for offset in [60, 90, 120, 150, 180, 210]:
        scores = []
        for t in enriched:
            pnl, bar = exit_at_offset(t, offset)
            scored = score_pnl(pnl, t["entry_features"], bar)
            if np.isfinite(scored): scores.append(scored)
        print(f"{'fixed bar+'+str(offset):>40} {pf(scores):>7.3f} {sum(scores):>9.0f} {np.mean(scores):>7.0f}")

    # 3. Trail-stop variants (now scored with hybrid_live)
    for trail_pct in [0.15, 0.25, 0.40]:
        scores = []
        for t in enriched:
            pnl, bar = trail_stop_exit(t, trail_pct)
            scored = score_pnl(pnl, t["entry_features"], bar)
            if np.isfinite(scored): scores.append(scored)
        print(f"{'trail '+str(int(trail_pct*100))+'% (peak>=50, min+30)':>40} {pf(scores):>7.3f} {sum(scores):>9.0f} {np.mean(scores):>7.0f}")

    # 4. Hybrid: take MAX of oracle's choice vs trail-stop
    for trail_pct in [0.25, 0.40]:
        scores = []
        for t in enriched:
            tp, tb = trail_stop_exit(t, trail_pct)
            op = t["oracle_pnl_pred"]
            ob = t["oracle_exit_bar"]
            # Pick whichever has higher hybrid_live score
            sa = score_pnl(op, t["entry_features"], ob)
            sb = score_pnl(tp, t["entry_features"], tb)
            scored = max(sa, sb) if np.isfinite(sa) and np.isfinite(sb) else (sa if np.isfinite(sa) else sb)
            if np.isfinite(scored): scores.append(scored)
        print(f"{'max(oracle, trail '+str(int(trail_pct*100))+'%)':>40} {pf(scores):>7.3f} {sum(scores):>9.0f} {np.mean(scores):>7.0f}")
    print()

    # 5. Reported labels for sanity
    of = [t.get("fwd_pnl_hybrid_with_oracle", 0) for t in enriched]
    print(f"=== Sanity check: reported fwd_pnl_hybrid_with_oracle ===")
    print(f"  PF {pf(of):.3f}, sum ${sum(of):.0f}")


if __name__ == "__main__":
    main()
