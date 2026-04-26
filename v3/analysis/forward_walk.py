"""Forward-walk inference: run the 5 saved seed models on dataset rows
that were never used in any OOS window (post-2026-02-24), then apply
the K=2 consensus + cal_pf>4 deployment recipe.

This is the strongest available offline proof short of live trading.
The action_surface dataset already contains rows through 2026-04-01;
training/OOS evaluation only covered through 2026-02-24, so any rows
on or after 2026-02-25 are completely unseen by the model.

For each seed (42-46):
  1. Load window_12/model.pkl (the latest trained model in the rolling
     window stack — tightest fit to recent data).
  2. Load window_12/calibration.json (the threshold the model picked).
  3. Run predict() on forward-walk rows.
  4. Apply the saved abstention policy (decision_margin, min_win_prob,
     max_stopout_prob).
  5. Build chosen_trades_forward_walk for the seed.

Then apply the deployment recipe (K=2 consensus + cal_pf>4 guard) and
compute PF/DD on three PnL realizations:
  - hybrid_live_with_oracle: matches the training utility target
    (L3 oracle exits + spread/penalty adjustments).
  - hybrid_live_no_oracle: dataset's hybrid_live_utility label (uses
    time_stop as base; honest "live realizable" without oracle exits).
  - time_stop: naive hold-to-bar-120 PnL (most conservative).

Usage:

    .venv/bin/python -m v3.analysis.forward_walk \
        --champion-dir v3/artifacts \
        --champion-name spx_combined_3seed_001 \
        --seeds 42 43 44 45 46 \
        --dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \
        --oracle-pattern v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed%s_balanced.npz \
        --forward-walk-after 2026-02-24 \
        --out v3/artifacts/forward_walk/spx_combined_3seed_001.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch

from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle
from v3.layer2.train_unified_policy import (
    _prediction_frame,
    _select_daily_trades,
    _slice_inputs,
)


def _profit_factor(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def _max_drawdown_pct(pnl: np.ndarray, capital_base: float = 25_000.0) -> float:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        return 0.0
    eq = capital_base
    pk = capital_base
    mx = 0.0
    for p in pnl:
        eq += float(p)
        if eq > pk:
            pk = eq
        if pk > 0:
            dd = (pk - eq) / pk * 100.0
            if dd > mx:
                mx = dd
    return mx


def _summarize(pnl: np.ndarray) -> dict[str, float]:
    pnl = np.asarray(pnl, dtype=float)
    finite = pnl[np.isfinite(pnl)]
    if len(finite) == 0:
        return {"n": 0, "pf": 0.0, "dd_pct": 0.0, "sum_pnl": 0.0, "mean_pnl": 0.0, "win_rate": 0.0}
    return {
        "n": int(len(finite)),
        "pf": _profit_factor(finite),
        "dd_pct": _max_drawdown_pct(finite),
        "sum_pnl": float(finite.sum()),
        "mean_pnl": float(finite.mean()),
        "win_rate": float((finite > 0).mean()),
    }


def _compute_hybrid_with_oracle(
    rows_idx: np.ndarray,  # global row indices
    action_idx: np.ndarray,  # per-row action choice
    bundle: dict,
    oracle: dict,
) -> np.ndarray:
    """Recompute hybrid_live_utility using L3 oracle's predicted exit."""
    al = bundle["action_labels"]
    out = np.full(len(rows_idx), np.nan, dtype=np.float32)
    for i, (r, a) in enumerate(zip(rows_idx, action_idx)):
        if a <= 0:
            continue
        l3_pnl = float(oracle["l3_exit_pnl"][r, a])
        l3_exit_bar = int(oracle["l3_exit_bar"][r, a])
        entry_bar = float(al["entry_fill_bar"][r, a])
        entry_mid = float(al["entry_fill_mid"][r, a])
        spread_frac = float(al["entry_spread_fraction"][r, a])
        stopout = float(al["stopout_risk"][r, a])
        if not np.isfinite(entry_mid) or not np.isfinite(entry_bar):
            continue
        out[i] = hybrid_live_utility(
            l3_pnl if np.isfinite(l3_pnl) else None,
            entry_mid=entry_mid,
            spread_fraction=spread_frac if np.isfinite(spread_frac) else 0.0,
            stopout_risk=stopout if np.isfinite(stopout) else 0.0,
            entry_bar=int(entry_bar),
            exit_bar=l3_exit_bar if l3_exit_bar >= 0 else None,
            session_end_bar=375,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-dir", default="v3/artifacts")
    parser.add_argument("--champion-name", default="spx_combined_3seed_001")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--dataset", default="v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl")
    parser.add_argument(
        "--oracle-pattern",
        default="v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed%s_balanced.npz",
        help="%%s placeholder for seed",
    )
    parser.add_argument("--forward-walk-after", default="2026-02-24")
    parser.add_argument("--window", type=int, default=12, help="Which trained window's model to use")
    parser.add_argument("--out", default="v3/artifacts/forward_walk/spx_combined_3seed_001.json")
    parser.add_argument("--K", type=int, default=2, help="Consensus K")
    parser.add_argument("--cal-pf-guard", type=float, default=4.0)
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset}")
    bundle = load_export_bundle(args.dataset)
    rows = bundle["rows"]
    meta = bundle["meta"]
    top_k_contracts = int(meta.get("top_k_contracts", 12))
    contract_feature_names = list(meta.get("contract_feature_names", ()))

    forward_mask = (rows["day"] > args.forward_walk_after).to_numpy()
    forward_indices = np.where(forward_mask)[0]
    forward_days = sorted(rows.loc[forward_mask, "day"].unique())
    print(f"Forward-walk rows: {forward_mask.sum()}, unique days: {len(forward_days)} "
          f"({forward_days[0] if forward_days else 'none'} -> {forward_days[-1] if forward_days else 'none'})")

    seed_chosen: dict[int, pd.DataFrame] = {}
    seed_calibrations: dict[int, dict] = {}
    seed_window_models_used: dict[int, int] = {}
    for seed in args.seeds:
        # Load oracle for this seed
        oracle_path = args.oracle_pattern % seed
        print(f"\n--- Seed {seed} ---")
        oracle = np.load(oracle_path, allow_pickle=True)

        # Build subset for forward-walk rows using hybrid_live target with this seed's oracle
        l3_pnl = oracle["l3_exit_pnl"]
        l3_exit_bar = oracle["l3_exit_bar"]
        subset = _slice_inputs(
            forward_mask,
            bundle,
            utility_target="hybrid_live",
            simulated_l3_pnl=l3_pnl,
            simulated_l3_exit_bar=l3_exit_bar,
        )
        print(f"  subset rows: {len(subset['rows'])}")

        # Pick the latest available window's model for this seed
        seed_dir = os.path.join(
            args.champion_dir,
            f"layer2_unified_policy_{args.champion_name}_seed{seed}",
            f"seed_{seed}",
        )
        chosen_window = args.window
        model_path = os.path.join(seed_dir, f"window_{chosen_window:02d}", "model.pkl")
        cal_path = os.path.join(seed_dir, f"window_{chosen_window:02d}", "calibration.json")
        if not os.path.exists(model_path):
            # fallback to the latest existing window
            avail = sorted([
                int(d.split("_")[1]) for d in os.listdir(seed_dir)
                if d.startswith("window_") and os.path.exists(os.path.join(seed_dir, d, "model.pkl"))
            ])
            if not avail:
                raise RuntimeError(f"No model.pkl for seed {seed}")
            chosen_window = avail[-1]
            model_path = os.path.join(seed_dir, f"window_{chosen_window:02d}", "model.pkl")
            cal_path = os.path.join(seed_dir, f"window_{chosen_window:02d}", "calibration.json")
        print(f"  using window {chosen_window} model: {model_path}")
        with open(model_path, "rb") as f:
            predictor = pickle.load(f)
        with open(cal_path) as f:
            calibration = json.load(f)
        seed_calibrations[seed] = calibration
        seed_window_models_used[seed] = chosen_window

        # Run inference
        pred = predictor.predict(
            scalar=subset["scalar"],
            seq=subset["seq"],
            seq_mask=subset["seq_mask"],
            contracts=subset["contracts"],
            contract_mask=subset["contract_mask"],
            batch_size=1024,
        )

        # Build prediction frame (uses subset["utility_raw"] as the realized PnL,
        # which for hybrid_live target IS hybrid_live with this seed's L3 oracle).
        pred_df = _prediction_frame(
            subset,
            pred,
            top_k_contracts=top_k_contracts,
            contract_feature_names=contract_feature_names,
        )

        # Now also compute time_stop_pnl and dataset's hybrid_live_no_oracle
        chosen_action_id = pred_df["chosen_action_id"].to_numpy()
        chosen_global_rows = forward_indices  # forward_indices aligns with subset rows

        # time_stop label is action_labels["utility_raw"]
        ts_arr = bundle["action_labels"]["utility_raw"]
        # dataset's hybrid_live_utility label
        hl_no_oracle_arr = bundle["action_labels"]["hybrid_live_utility"]
        time_stop_pnl = np.full(len(pred_df), np.nan, dtype=np.float32)
        hybrid_no_oracle = np.full(len(pred_df), np.nan, dtype=np.float32)
        for i, (r, a) in enumerate(zip(chosen_global_rows, chosen_action_id)):
            if a <= 0:
                continue
            time_stop_pnl[i] = ts_arr[r, a]
            hybrid_no_oracle[i] = hl_no_oracle_arr[r, a]
        pred_df["fwd_pnl_time_stop"] = time_stop_pnl
        pred_df["fwd_pnl_hybrid_no_oracle"] = hybrid_no_oracle
        pred_df["fwd_pnl_hybrid_with_oracle"] = pred_df["chosen_objective_pnl"]

        # Apply the saved abstention policy
        chosen = _select_daily_trades(pred_df, abstention_policy=calibration).copy()
        chosen["window_idx"] = chosen_window
        chosen["seed"] = seed
        seed_chosen[seed] = chosen
        print(f"  forward-walk chosen trades: {len(chosen)} "
              f"(calls={(chosen['chosen_side']=='call').sum()}, puts={(chosen['chosen_side']=='put').sum()})")

    # Per-seed metrics
    print("\n=== Per-seed forward-walk metrics ===")
    print(f"{'seed':>5} {'n':>5} {'pf_oracle':>10} {'dd_oracle':>10} {'pf_no_oracle':>13} {'dd_no_oracle':>13} {'pf_tstop':>10} {'dd_tstop':>10} {'cal_pf':>8}")
    per_seed_metrics: dict[int, dict] = {}
    for seed in args.seeds:
        df = seed_chosen[seed]
        cal_pf = seed_calibrations[seed].get("objective_pf")
        if df.empty:
            per_seed_metrics[seed] = {"n": 0}
            continue
        s_oracle = _summarize(df["fwd_pnl_hybrid_with_oracle"].values)
        s_no_oracle = _summarize(df["fwd_pnl_hybrid_no_oracle"].values)
        s_tstop = _summarize(df["fwd_pnl_time_stop"].values)
        per_seed_metrics[seed] = {
            "n_trades": s_oracle["n"],
            "n_calls": int((df["chosen_side"] == "call").sum()),
            "n_puts": int((df["chosen_side"] == "put").sum()),
            "with_oracle": s_oracle,
            "no_oracle_dataset_label": s_no_oracle,
            "time_stop_naive": s_tstop,
            "calibration_window_used": seed_window_models_used[seed],
            "calibration_pf": cal_pf,
            "calibration_decision_margin": seed_calibrations[seed].get("decision_margin"),
        }
        print(
            f"{seed:>5} {s_oracle['n']:>5} {s_oracle['pf']:>10.3f} {s_oracle['dd_pct']:>10.2f} "
            f"{s_no_oracle['pf']:>13.3f} {s_no_oracle['dd_pct']:>13.2f} "
            f"{s_tstop['pf']:>10.3f} {s_tstop['dd_pct']:>10.2f} "
            f"{cal_pf if cal_pf is not None else 0:>8.3f}"
        )

    # Cross-seed mean
    pfs_o = [m["with_oracle"]["pf"] for m in per_seed_metrics.values() if m.get("n_trades", 0) > 0]
    pfs_n = [m["no_oracle_dataset_label"]["pf"] for m in per_seed_metrics.values() if m.get("n_trades", 0) > 0]
    pfs_t = [m["time_stop_naive"]["pf"] for m in per_seed_metrics.values() if m.get("n_trades", 0) > 0]
    dds_o = [m["with_oracle"]["dd_pct"] for m in per_seed_metrics.values() if m.get("n_trades", 0) > 0]
    cross_seed = {
        "mean_pf_with_oracle": float(np.mean(pfs_o)) if pfs_o else 0.0,
        "min_pf_with_oracle": float(np.min(pfs_o)) if pfs_o else 0.0,
        "mean_pf_no_oracle": float(np.mean(pfs_n)) if pfs_n else 0.0,
        "mean_pf_time_stop": float(np.mean(pfs_t)) if pfs_t else 0.0,
        "mean_dd_with_oracle": float(np.mean(dds_o)) if dds_o else 0.0,
        "max_dd_with_oracle": float(np.max(dds_o)) if dds_o else 0.0,
    }
    print("\n=== Cross-seed forward-walk (no consensus filter) ===")
    print(f"  mean PF with oracle:   {cross_seed['mean_pf_with_oracle']:.3f}")
    print(f"  min PF with oracle:    {cross_seed['min_pf_with_oracle']:.3f}")
    print(f"  mean PF no oracle:     {cross_seed['mean_pf_no_oracle']:.3f}")
    print(f"  mean PF time_stop:     {cross_seed['mean_pf_time_stop']:.3f}")
    print(f"  mean DD with oracle:   {cross_seed['mean_dd_with_oracle']:.2f}%")
    print(f"  max DD with oracle:    {cross_seed['max_dd_with_oracle']:.2f}%")

    # K=2 consensus + cal_pf>4 guard on forward-walk trades
    all_chosen = pd.concat(seed_chosen.values(), ignore_index=True)
    if len(all_chosen) > 0:
        # Determine which seeds are guard-flagged
        guarded_seeds = {s for s, m in per_seed_metrics.items()
                         if m.get("calibration_pf") is not None and m["calibration_pf"] > args.cal_pf_guard}
        print(f"\nSeeds guarded by cal_pf > {args.cal_pf_guard}: {sorted(guarded_seeds)}")

        # Per-bar consensus build (across non-guarded seeds, OR all seeds — try both)
        for guard_label, active_seeds in [("no_guard", set(args.seeds)), ("with_cal_guard", set(args.seeds) - guarded_seeds)]:
            active_chosen = all_chosen[all_chosen["seed"].isin(active_seeds)]
            if len(active_chosen) == 0:
                print(f"\n[{guard_label}] no trades after guard")
                continue
            grp = active_chosen.groupby(["day", "bar_index"]).agg(
                n_call=("chosen_side", lambda s: int((s == "call").sum())),
                n_put=("chosen_side", lambda s: int((s == "put").sum())),
            ).reset_index()
            grp["max_count"] = np.maximum(grp["n_call"], grp["n_put"])
            grp_index = grp.set_index(["day", "bar_index"])

            print(f"\n=== K={args.K} consensus, {guard_label}, active seeds {sorted(active_seeds)} ===")
            print(f"  unique bars: {len(grp)}, K>={args.K} agreement: {(grp['max_count']>=args.K).sum()}")

            # Per-seed K-consensus (deploy 1 seed, vote on others)
            agg_pfs_o = []
            agg_pfs_n = []
            agg_pfs_t = []
            agg_dds_o = []
            agg_ns = []
            for seed in active_seeds:
                df = seed_chosen[seed]
                if df.empty:
                    continue
                keep = []
                for _, row in df.iterrows():
                    key = (row["day"], row["bar_index"])
                    if key not in grp_index.index:
                        keep.append(False); continue
                    side = row["chosen_side"]
                    cnt = grp_index.loc[key]["n_call"] if side == "call" else grp_index.loc[key]["n_put"]
                    keep.append(cnt >= args.K)
                kept = df[pd.Series(keep, index=df.index)]
                if kept.empty:
                    continue
                so = _summarize(kept["fwd_pnl_hybrid_with_oracle"].values)
                sn = _summarize(kept["fwd_pnl_hybrid_no_oracle"].values)
                st = _summarize(kept["fwd_pnl_time_stop"].values)
                agg_pfs_o.append(so["pf"]); agg_pfs_n.append(sn["pf"]); agg_pfs_t.append(st["pf"])
                agg_dds_o.append(so["dd_pct"]); agg_ns.append(so["n"])
                print(f"  seed {seed}: n={so['n']}, pf_oracle={so['pf']:.3f}, dd={so['dd_pct']:.2f}%, "
                      f"pf_no_oracle={sn['pf']:.3f}, pf_tstop={st['pf']:.3f}")
            if agg_pfs_o:
                print(f"  mean across {len(agg_pfs_o)} active seeds:")
                print(f"    pf_with_oracle:   mean={np.mean(agg_pfs_o):.3f}, min={np.min(agg_pfs_o):.3f}")
                print(f"    pf_no_oracle:     mean={np.mean(agg_pfs_n):.3f}")
                print(f"    pf_time_stop:     mean={np.mean(agg_pfs_t):.3f}")
                print(f"    dd_with_oracle:   mean={np.mean(agg_dds_o):.2f}%, max={np.max(agg_dds_o):.2f}%")
                print(f"    n_trades:         mean={np.mean(agg_ns):.0f}, min={np.min(agg_ns)}")

    # Save chosen_trades + summary
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    chosen_dir = os.path.dirname(args.out)
    for seed, df in seed_chosen.items():
        path = os.path.join(chosen_dir, f"forward_walk_chosen_seed{seed}.pkl")
        df.to_pickle(path)

    out = {
        "champion": args.champion_name,
        "seeds": args.seeds,
        "forward_walk_after": args.forward_walk_after,
        "n_forward_walk_rows": int(forward_mask.sum()),
        "forward_walk_unique_days": len(forward_days),
        "forward_walk_first_day": forward_days[0] if forward_days else None,
        "forward_walk_last_day": forward_days[-1] if forward_days else None,
        "K": args.K,
        "cal_pf_guard": args.cal_pf_guard,
        "per_seed": per_seed_metrics,
        "cross_seed_no_filter": cross_seed,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    print(f"Per-seed forward-walk chosen_trades saved under {chosen_dir}/")


if __name__ == "__main__":
    main()
