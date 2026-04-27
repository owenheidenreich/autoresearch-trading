"""Phase 2a sensitivity: drop-only vs rescan-after-veto offline comparison.

Phase 1 evaluated the avoid-union on chosen_trades.pkl (post-_select_daily_trades),
which is drop-only semantics. The runtime applies the avoid mask inside
_select_daily_trades, which is rescan-after-veto: vetoed bars are removed
from eligibility BEFORE the daily-best pick. The replacement bar (if any)
re-enters the kept distribution.

This script re-evaluates the avoid-union under rescan semantics using the
per-bar oof_predictions.pkl frame, then compares to Phase 1's drop-only
numbers. The forward-walk evaluation in Phase 2b is the authoritative
rescan-mode test; this is a fast in-sample sanity check.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from v3.layer2.post_filters_v0 import (
    AVOID_RULES,
    FILTER_SET_ID,
    attach_buckets,
)


SEEDS = [42, 43, 44, 45, 46]
PRED_PATTERN = (
    "v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{seed}"
    "/seed_{seed}/oof_predictions.pkl"
)
TOP_K_CONTRACTS = 7  # per side; from manifest


def attach_features_for_rules(pred: pd.DataFrame) -> pd.DataFrame:
    """oof_predictions.pkl already has sigma_pos, iv_percentile,
    decision_margin, etc. We need to derive `side`, `cell`, `sigma_b`, `iv_b`
    using the same frozen tertiles as Phase 0.
    """
    out = pred.copy()
    # Derive side from chosen_action_id: 0=flat, 1..K=call, K+1..2K=put
    cs = out["chosen_side"].fillna("flat")
    out["side"] = cs
    # Recompute sigma_pos column if absent — present in oof_predictions
    if "sigma_pos" not in out.columns:
        # Fallback: vwap-relative computation
        out["sigma_pos"] = (out["underlying_close"] - out["vwap"]) / (
            out["vwap"] * out["atm_iv"].replace(0, np.nan) / np.sqrt(252)
        )
    out = attach_buckets(out)
    return out


def vectorized_avoid_mask(pred: pd.DataFrame) -> np.ndarray:
    """Boolean mask over rows: True = matches at least one avoid rule."""
    mask = np.zeros(len(pred), dtype=bool)
    for rule in AVOID_RULES:
        m = pred.apply(rule.predicate, axis=1).astype(bool).values
        mask |= m
    return mask


def select_daily_winner(pred: pd.DataFrame, eligible_mask: np.ndarray) -> pd.DataFrame:
    """Mirror _select_daily_trades: keep eligible rows, then pick the row with
    the highest best_nonflat_score per day."""
    if eligible_mask.sum() == 0:
        return pred.iloc[0:0].copy()
    elig = pred[eligible_mask & np.isfinite(pred["best_nonflat_score"])].copy()
    if elig.empty:
        return elig
    idx = elig.groupby("day")["best_nonflat_score"].idxmax()
    return elig.loc[idx].sort_values(["day", "bar_index"]).reset_index(drop=True)


def chosen_with_hl(chosen: pd.DataFrame, hl_lookup: dict) -> pd.DataFrame:
    """Attach hl from the Phase 1 sample where (seed, day, bar_index, side, strike) matches."""
    out = chosen.copy()
    keys = list(zip(out["seed"], out["day"].astype(str), out["bar_index"], out["chosen_side"], out["chosen_strike"]))
    hls = [hl_lookup.get(k, np.nan) for k in keys]
    out["hl"] = hls
    return out


def profit_factor(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    pos = arr[arr > 0].sum()
    neg = arr[arr < 0].sum()
    if neg < 0:
        return float(pos / abs(neg))
    return float("inf") if pos > 0 else float("nan")


def main():
    print(f"=== Phase 2a sensitivity: rescan vs drop-only ({FILTER_SET_ID}) ===\n")

    # Load Phase 1 hl lookup
    phase1_csv = pd.read_csv("v3/artifacts/research/phase_c_trade_review.csv")
    hl_lookup = {}
    for _, r in phase1_csv.iterrows():
        key = (int(r["seed"]), str(r["day"])[:10], int(r["bar_index"]),
               r["side"], float(r["strike"]))
        hl_lookup[key] = float(r["hl"])
    print(f"Loaded hl lookup: {len(hl_lookup)} (seed, day, bar, side, strike) keys")

    rescan_results = []
    drop_only_results = []

    for seed in SEEDS:
        path = PRED_PATTERN.format(seed=seed)
        if not os.path.exists(path):
            print(f"[skip] seed {seed}: no oof_predictions")
            continue
        pred = pd.read_pickle(path)
        pred["seed"] = seed
        pred = attach_features_for_rules(pred)

        # Existing eligibility gate
        elig_base = (
            (pred["chosen_action_id"] > 0)
            & np.isfinite(pred["best_nonflat_score"])
            & (pred["decision_margin"] >= 0.0)
            & (pred["pred_win_prob"].fillna(0.0) >= 0.0)
            & (pred["pred_stopout_risk"].fillna(1.0) <= 1.0)
        ).values

        # Baseline (no avoid)
        chosen_base = select_daily_winner(pred, elig_base)
        chosen_base["seed"] = seed
        chosen_base = chosen_with_hl(chosen_base, hl_lookup)
        base_n = len(chosen_base)
        base_n_with_hl = chosen_base["hl"].notna().sum()
        base_pf = profit_factor(chosen_base["hl"].dropna().values)

        # Rescan-after-veto
        avoid_mask = vectorized_avoid_mask(pred)
        elig_rescan = elig_base & ~avoid_mask
        chosen_rescan = select_daily_winner(pred, elig_rescan)
        chosen_rescan["seed"] = seed
        chosen_rescan = chosen_with_hl(chosen_rescan, hl_lookup)
        rescan_n = len(chosen_rescan)
        rescan_n_with_hl = chosen_rescan["hl"].notna().sum()
        rescan_pf = profit_factor(chosen_rescan["hl"].dropna().values)

        # Drop-only (vetoes applied to chosen_base)
        chosen_drop = chosen_base.copy()
        # Re-apply avoid filters to the chosen_base frame
        chosen_drop = attach_features_for_rules(chosen_drop)
        drop_mask = np.zeros(len(chosen_drop), dtype=bool)
        for rule in AVOID_RULES:
            m = chosen_drop.apply(rule.predicate, axis=1).astype(bool).values
            drop_mask |= m
        chosen_drop = chosen_drop[~drop_mask]
        drop_n = len(chosen_drop)
        drop_n_with_hl = chosen_drop["hl"].notna().sum()
        drop_pf = profit_factor(chosen_drop["hl"].dropna().values)

        # How many "winning" bars get replaced (i.e. bar_index differs between base and rescan)?
        base_keys = chosen_base.set_index("day")[["bar_index"]].rename(columns={"bar_index": "base_bar"})
        rescan_keys = chosen_rescan.set_index("day")[["bar_index"]].rename(columns={"bar_index": "rescan_bar"})
        merged = base_keys.join(rescan_keys, how="outer")
        n_replaced = (merged["base_bar"] != merged["rescan_bar"]).fillna(True).sum()
        n_dropped_no_replacement = (merged["rescan_bar"].isna() & merged["base_bar"].notna()).sum()

        rescan_results.append({
            "seed": seed,
            "base_n": base_n,
            "base_n_with_hl": int(base_n_with_hl),
            "base_pf": base_pf,
            "rescan_n": rescan_n,
            "rescan_n_with_hl": int(rescan_n_with_hl),
            "rescan_pf": rescan_pf,
            "drop_n": drop_n,
            "drop_n_with_hl": int(drop_n_with_hl),
            "drop_pf": drop_pf,
            "n_days_replaced": int(n_replaced),
            "n_days_dropped_no_replacement": int(n_dropped_no_replacement),
        })

        print(f"  seed {seed}:")
        print(f"    baseline:     n={base_n} (n_with_hl={base_n_with_hl}), PF={base_pf:.3f}")
        print(f"    drop-only:    n={drop_n} (n_with_hl={drop_n_with_hl}), PF={drop_pf:.3f}, "
              f"ΔPF={drop_pf - base_pf:+.3f}")
        print(f"    rescan:       n={rescan_n} (n_with_hl={rescan_n_with_hl}), PF={rescan_pf:.3f}, "
              f"ΔPF={rescan_pf - base_pf:+.3f}")
        print(f"    days replaced: {n_replaced}, days dropped (no replacement): {n_dropped_no_replacement}")

    print("\n=== Aggregate across seeds ===")
    df = pd.DataFrame(rescan_results)
    print(df.to_string(index=False))

    print("\n=== Summary ===")
    mean_drop_dpf = float((df["drop_pf"] - df["base_pf"]).mean())
    mean_rescan_dpf = float((df["rescan_pf"] - df["base_pf"]).mean())
    print(f"Mean ΔPF drop-only (in-sample, per seed): {mean_drop_dpf:+.3f}")
    print(f"Mean ΔPF rescan    (in-sample, per seed): {mean_rescan_dpf:+.3f}")
    print(f"Mean days replaced per seed: {df['n_days_replaced'].mean():.1f}")
    print(f"Mean days dropped (no replacement) per seed: {df['n_days_dropped_no_replacement'].mean():.1f}")

    out_dir = "v3/artifacts/research"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/phase2a_rescan_sensitivity.csv", index=False)
    print(f"\nWrote {out_dir}/phase2a_rescan_sensitivity.csv")


if __name__ == "__main__":
    main()
