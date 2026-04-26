"""Iteration 5: combine K=2 consensus filter + gate.

The K-of-5 side consensus filter (from earlier sprint) lifts forward-walk PF
by filtering bars where seeds disagree on side. The gate (Iter 4) lifts PF by
using no_oracle exit on flagged trades. Test whether they compose:

    1. Apply K=2 consensus to each seed's forward-walk trades.
    2. For surviving trades, apply gate to decide oracle vs no_oracle exit.
    3. Compare to K=2 alone, gate alone, and unfiltered.

This is the strongest candidate for a deployment recipe.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from v3.live_shadow.oracle_gate import OracleGate, FEATURES


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def main():
    seeds = [42, 43, 44, 45, 46]

    # Load all seed forward-walk trades
    by_seed = {}
    for s in seeds:
        df = pd.read_pickle(f"v3/artifacts/forward_walk/forward_walk_chosen_seed{s}.pkl")
        if df.empty:
            continue
        df = df.copy()
        df["seed"] = s
        finite = np.isfinite(df["fwd_pnl_hybrid_with_oracle"]) & np.isfinite(df["fwd_pnl_time_stop"])
        df = df[finite].reset_index(drop=True)
        by_seed[s] = df
    all_df = pd.concat(by_seed.values(), ignore_index=True)

    # Build K-consensus structure
    grp = all_df.groupby(["day", "bar_index"]).agg(
        n_call=("chosen_side", lambda x: int((x == "call").sum())),
        n_put=("chosen_side", lambda x: int((x == "put").sum())),
    ).reset_index()
    grp["max_count"] = np.maximum(grp["n_call"], grp["n_put"])
    grp_index = grp.set_index(["day", "bar_index"])

    # Load gate
    gate = OracleGate.load()
    print(f"Loaded gate (threshold {gate.threshold})")
    print()

    # Function: per-seed K-consensus filter + per-trade gate
    def apply_recipe(K, use_gate):
        results = {}
        for s, df in by_seed.items():
            # K-consensus filter: only keep trades where >= K seeds agree on side
            keep = []
            for _, row in df.iterrows():
                key = (row["day"], row["bar_index"])
                if key not in grp_index.index:
                    keep.append(False); continue
                side = row["chosen_side"]
                cnt = grp_index.loc[key]["n_call"] if side == "call" else grp_index.loc[key]["n_put"]
                keep.append(cnt >= K)
            kept = df[pd.Series(keep, index=df.index)]
            if kept.empty:
                results[s] = {"n": 0, "pnl": np.array([])}
                continue
            # Apply gate or not
            if use_gate:
                X = kept[gate.feature_names].fillna(0).values
                probs = gate.classifier.predict_proba(X)[:, 1]
                use_oracle = probs > gate.threshold
                pnl = np.where(use_oracle, kept["fwd_pnl_hybrid_with_oracle"].astype(float).values,
                                          kept["fwd_pnl_time_stop"].astype(float).values)
                n_oracle = int(use_oracle.sum())
            else:
                pnl = kept["fwd_pnl_hybrid_with_oracle"].astype(float).values
                n_oracle = len(pnl)
            results[s] = {
                "n": len(kept), "pnl": pnl, "n_oracle": n_oracle,
            }
        return results

    print(f"=== Forward-walk recipes (per-seed metrics) ===")
    print()
    recipes = [
        ("K=1 (no consensus filter), no gate", 1, False),
        ("K=1, with gate", 1, True),
        ("K=2 consensus, no gate", 2, False),
        ("K=2 consensus, with gate", 2, True),
        ("K=3 consensus, no gate", 3, False),
        ("K=3 consensus, with gate", 3, True),
    ]
    for label, K, use_gate in recipes:
        r = apply_recipe(K, use_gate)
        # cross-seed: per-seed PF, mean
        seed_pfs = []
        seed_ns = []
        all_pnls = []
        for s in seeds:
            if r[s]["n"] == 0: continue
            seed_pfs.append(pf(r[s]["pnl"]))
            seed_ns.append(r[s]["n"])
            all_pnls.extend(r[s]["pnl"].tolist())
        if not all_pnls:
            print(f"{label:50s}: no trades")
            continue
        all_pnls = np.array(all_pnls)
        print(f"{label:50s}: agg PF {pf(all_pnls):.3f}, sum ${all_pnls.sum():.0f}, "
              f"mean per-seed PF {np.mean(seed_pfs):.3f}, "
              f"mean trades {np.mean(seed_ns):.1f}")
    print()

    # Detail on K=2 + gate
    print("=== Detail: K=2 consensus + gate ===")
    r = apply_recipe(K=2, use_gate=True)
    for s in seeds:
        if r[s]["n"] == 0: continue
        seed_pf = pf(r[s]["pnl"])
        sum_p = r[s]["pnl"].sum()
        print(f"  seed {s}: n={r[s]['n']}, oracle={r[s]['n_oracle']}/{r[s]['n']}, PF {seed_pf:.3f}, sum ${sum_p:.0f}")

    # Detail on K=3 + gate
    print()
    print("=== Detail: K=3 consensus + gate ===")
    r = apply_recipe(K=3, use_gate=True)
    for s in seeds:
        if r[s]["n"] == 0: continue
        seed_pf = pf(r[s]["pnl"])
        sum_p = r[s]["pnl"].sum()
        print(f"  seed {s}: n={r[s]['n']}, oracle={r[s]['n_oracle']}/{r[s]['n']}, PF {seed_pf:.3f}, sum ${sum_p:.0f}")


if __name__ == "__main__":
    main()
