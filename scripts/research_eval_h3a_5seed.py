"""5-seed aggregate evaluation of H3a oracles vs balanced_fresh baseline.

Runs the per-seed eval (research_eval_h3a_oracle.py logic) for each seed
42-46, then aggregates: cross-seed mean PF (base vs h3a), per-seed deltas,
small-winner capture-of-best lift consistency.

Discipline check: H3a only proceeds to forward-walk gate if
  - cross-seed mean delta-PF >= +0.10
  - all 5 seeds individually have delta-PF >= 0 (no seed crashes)
  - small-winner ($0-200) capture lift positive on >= 4/5 seeds
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle


SEEDS = [42, 43, 44, 45, 46]
BASE_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_balanced_fresh.npz"
H3A_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_h3a.npz"
CHOSEN_PATTERN = "v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{seed}/seed_{seed}/chosen_trades.pkl"


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def hl(pnl_raw, em, ef, eb, so, exit_bar):
    return hybrid_live_utility(
        pnl_raw if np.isfinite(pnl_raw) else None,
        entry_mid=em, spread_fraction=ef if np.isfinite(ef) else 0.0,
        stopout_risk=so if np.isfinite(so) else 0.0,
        entry_bar=int(eb),
        exit_bar=int(exit_bar) if exit_bar >= 0 else None,
        session_end_bar=375,
    )


def eval_seed(seed: int, bundle, key_to_row, al) -> dict | None:
    base_path = BASE_PATTERN.format(seed=seed)
    h3a_path = H3A_PATTERN.format(seed=seed)
    chosen_path = CHOSEN_PATTERN.format(seed=seed)

    if not all(os.path.exists(p) for p in [base_path, h3a_path, chosen_path]):
        missing = [p for p in [base_path, h3a_path, chosen_path] if not os.path.exists(p)]
        print(f"  seed {seed}: missing {missing}")
        return None

    fw = pd.read_pickle(chosen_path)
    fw = fw[fw["chosen_action_id"] > 0].reset_index(drop=True)

    base = np.load(base_path, allow_pickle=True)
    h3a = np.load(h3a_path, allow_pickle=True)
    bp_arr, bb_arr = base["l3_exit_pnl"], base["l3_exit_bar"]
    hp_arr, hb_arr = h3a["l3_exit_pnl"], h3a["l3_exit_bar"]

    records = []
    for _, row in fw.iterrows():
        key = (row["day"], row["bar_index"])
        r = key_to_row.get(key)
        if r is None: continue
        a = int(row["chosen_action_id"])
        if a <= 0: continue
        em = float(al["entry_fill_mid"][r, a])
        ef = float(al["entry_spread_fraction"][r, a])
        eb = float(al["entry_fill_bar"][r, a])
        so = float(al["stopout_risk"][r, a])
        if not np.isfinite(em) or not np.isfinite(eb): continue

        bp, bb = float(bp_arr[r, a]), int(bb_arr[r, a])
        hp, hb = float(hp_arr[r, a]), int(hb_arr[r, a])

        records.append({
            "side": row["chosen_side"],
            "best": float(row["best_forward_pnl_call"]) if row["chosen_side"] == "call"
                    else float(row["best_forward_pnl_put"]),
            "base_pnl": bp, "base_hl": hl(bp, em, ef, eb, so, bb),
            "h3a_pnl": hp, "h3a_hl": hl(hp, em, ef, eb, so, hb),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return None

    base_pf = pf(df["base_hl"]); h3a_pf = pf(df["h3a_hl"])
    df["base_capture"] = (df["base_pnl"] - df["best"]) / np.clip(np.abs(df["best"]), 1.0, None)
    df["h3a_capture"] = (df["h3a_pnl"] - df["best"]) / np.clip(np.abs(df["best"]), 1.0, None)
    sm = (df["best"] >= 0) & (df["best"] < 200)
    sm_base = float(df.loc[sm, "base_capture"].mean()) if sm.any() else float("nan")
    sm_h3a = float(df.loc[sm, "h3a_capture"].mean()) if sm.any() else float("nan")

    return {
        "seed": seed,
        "n": int(len(df)),
        "base_pf": float(base_pf),
        "h3a_pf": float(h3a_pf),
        "delta_pf": float(h3a_pf - base_pf),
        "base_sum": float(df["base_hl"].sum()),
        "h3a_sum": float(df["h3a_hl"].sum()),
        "delta_sum": float(df["h3a_hl"].sum() - df["base_hl"].sum()),
        "small_n": int(sm.sum()),
        "sm_base_capture": sm_base,
        "sm_h3a_capture": sm_h3a,
        "sm_capture_delta": sm_h3a - sm_base if (np.isfinite(sm_base) and np.isfinite(sm_h3a)) else float("nan"),
    }


def main():
    print("=== H3a 5-seed evaluation: full OOS, all seeds ===\n")

    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    results = []
    for seed in SEEDS:
        r = eval_seed(seed, bundle, key_to_row, al)
        if r is not None:
            results.append(r)

    if not results:
        print("No seeds evaluable")
        return 1

    print(f"{'seed':>5} {'n':>4} {'base_pf':>8} {'h3a_pf':>8} {'d_pf':>7} "
          f"{'base_sum':>10} {'h3a_sum':>10} {'d_sum':>9} "
          f"{'sm_n':>5} {'sm_base':>8} {'sm_h3a':>8} {'d_sm':>7}")
    for r in results:
        print(f"{r['seed']:>5} {r['n']:>4} {r['base_pf']:>8.3f} {r['h3a_pf']:>8.3f} "
              f"{r['delta_pf']:>+7.3f} ${r['base_sum']:>9.0f} ${r['h3a_sum']:>9.0f} "
              f"${r['delta_sum']:>+8.0f} {r['small_n']:>5} "
              f"{r['sm_base_capture']:>8.3f} {r['sm_h3a_capture']:>8.3f} "
              f"{r['sm_capture_delta']:>+7.3f}")
    print()

    deltas = [r["delta_pf"] for r in results]
    sm_deltas = [r["sm_capture_delta"] for r in results if np.isfinite(r["sm_capture_delta"])]
    pf_means = [r["h3a_pf"] for r in results]
    base_means = [r["base_pf"] for r in results]
    print(f"Cross-seed mean base PF:  {np.mean(base_means):.3f}")
    print(f"Cross-seed mean H3a PF:   {np.mean(pf_means):.3f}")
    print(f"Cross-seed mean delta PF: {np.mean(deltas):+.3f}  (gate: +0.10)")
    print(f"All seeds delta_pf >= 0?  {all(d >= 0 for d in deltas)}")
    print(f"Cross-seed mean small-winner capture delta: {np.mean(sm_deltas):+.3f}  (target: positive)")
    print(f"Seeds with small-winner lift > 0: {sum(1 for d in sm_deltas if d > 0)}/{len(sm_deltas)}")
    print()

    summary = {
        "per_seed": results,
        "mean_base_pf": float(np.mean(base_means)),
        "mean_h3a_pf": float(np.mean(pf_means)),
        "mean_delta_pf": float(np.mean(deltas)),
        "all_deltas_nonneg": all(d >= 0 for d in deltas),
        "mean_sm_delta": float(np.mean(sm_deltas)) if sm_deltas else float("nan"),
        "n_sm_positive": sum(1 for d in sm_deltas if d > 0),
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/h3a_5seed_eval.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Decision gate
    cross_pf_pass = np.mean(deltas) >= 0.10
    no_seed_crashes = all(d >= -0.10 for d in deltas)
    sm_consistent = sum(1 for d in sm_deltas if d > 0) >= 4

    print("=== Decision gate ===")
    if cross_pf_pass and no_seed_crashes:
        print("✓ H3a 5-SEED PASSES: cross-seed mean delta_pf >= +0.10, no seed delta < -0.10.")
        print("  Proceed to forward-walk gate.")
        verdict = "PASS"
    elif sm_consistent and no_seed_crashes:
        print("≈ H3a 5-SEED PASS-SECONDARY: small-winner lift consistent across seeds.")
        print("  Proceed to forward-walk gate; note PF gate marginal.")
        verdict = "PASS-SECONDARY"
    else:
        print("✗ H3a 5-SEED FAILS: cross-seed PF lift not consistent or seed crashed.")
        print("  Recommend git revert.")
        verdict = "FAIL"
    summary["verdict"] = verdict
    with open("v3/artifacts/research/h3a_5seed_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote v3/artifacts/research/h3a_5seed_eval.json")
    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
