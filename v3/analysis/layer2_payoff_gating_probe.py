"""Layer-2 payoff-sufficiency gating probe.

The fallback-routing line is exhausted: blunt teacher+put still beats both
put-or-flat and full call/put/flat learned routers on the detach-side
chosen bars. The remaining hypothesis worth a cheap test is that some
context feature carves the non-teacher universe into a low-payoff bucket
where fallback puts are not worth taking.

This probe is descriptive first, simulation second. It does NOT train a
regressor on PnL means (that failure mode is what put_or_flat_model fell
into). Instead it bins the per-fold non-teacher TRAINING universe into
deciles on each candidate feature, reports the distribution of
time_stop_pnl_put across deciles, and flags features with monotonic
separation. For any flagged feature, it then simulates "suppress fallback
when the chosen-bar feature value falls in the bottom-2 deciles" on the
exact 176 fallback bars the detach-side baseline already chose, and
reports PF/DD/TPD vs baseline (always-put fallback).

Disqualifying signal:
- No candidate feature shows monotonic decile separation
  (bottom-2-decile mean put PnL < $0 AND top-2-decile > $300 AND Spearman
   |rho| over decile means > 0.50), OR
- The best simulated suppression drops PF below 1.35 or hurts fold-0,
- Then the regime/payoff-gating thesis is also falsified and the routing
  workstream ends.

Run:
    python -m v3.analysis.layer2_payoff_gating_probe \
        --baseline-run-dir v3/artifacts/layer2_shared_enc_fixedq_detach
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    load_export_bundle,
    replay_metrics_from_pnls,
    teacher_direction_hint_from_row,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer2_payoff_gating_probe")

# Candidate features hypothesised to discriminate payoff regime for puts.
# Kept short on purpose: anything else is multi-feature modeling, which is
# out of scope for this cheap gate.
CANDIDATE_FEATURES = [
    "atm_iv",
    "iv_percentile",
    "vix",
    "first15_range_pct",
    "omar_range_pct",
    "sigma_pos",
    "abs_sigma_pos",
    "last10_range_over_omar",
    "atm_iv_over_first15_range",  # derived: IV cost vs realized intraday range
]

N_DECILES = 10
SUPPRESS_BOTTOM_K_DEFAULT = 2  # bottom 2 deciles -> suppress
PF_FAIL_THRESHOLD = 1.35       # below this = thesis falsified
PF_PASS_THRESHOLD = 1.55       # above this with no fold regression = pass
SEP_GAP_MIN = 80.0             # |top2_mean - bottom2_mean| in $ per put trade
SEP_RHO_MIN = 0.50             # |Spearman rho| of decile means vs decile rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-2 payoff-sufficiency gating probe.")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument(
        "--suppress-k",
        type=int,
        default=SUPPRESS_BOTTOM_K_DEFAULT,
        help="Number of bottom-decile bins to suppress in simulation (default 2).",
    )
    return p.parse_args()


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "abs_sigma_pos" not in out.columns and "sigma_pos" in out.columns:
        out["abs_sigma_pos"] = out["sigma_pos"].abs()
    if "first15_range_pct" in out.columns and "atm_iv" in out.columns:
        denom = out["first15_range_pct"].replace(0.0, np.nan)
        out["atm_iv_over_first15_range"] = out["atm_iv"] / denom
    return out


def _eligible_universe(rows: pd.DataFrame) -> pd.DataFrame:
    """Non-teacher rows where a put is actually available + has finite PnL."""
    mask = (
        (rows["teacher_any_triggered"] <= 0.5)
        & (rows["has_passing_put"] > 0.5)
        & rows["time_stop_pnl_put"].notna()
        & np.isfinite(rows["time_stop_pnl_put"].astype(float))
    )
    return rows[mask].copy()


def _decile_edges(values: np.ndarray, n: int = N_DECILES) -> np.ndarray:
    """Return n+1 edges (-inf, q1, q2, ..., +inf) for binning."""
    qs = np.linspace(0.0, 1.0, n + 1)[1:-1]  # interior cut points
    edges = np.quantile(values, qs)
    return np.concatenate([[-np.inf], edges, [np.inf]])


def _bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return integer bin in [0, n_bins-1] for each value."""
    # np.searchsorted returns insertion index; subtract 1 so first bin is 0
    bins = np.searchsorted(edges, values, side="right") - 1
    return np.clip(bins, 0, len(edges) - 2)


def _per_feature_separation(
    train_df: pd.DataFrame,
    feature: str,
) -> dict[str, Any] | None:
    """For one feature, bin train universe into deciles and report
    distribution of put PnL by decile."""
    if feature not in train_df.columns:
        return None
    sub = train_df[[feature, "time_stop_pnl_put"]].dropna()
    if len(sub) < 200:
        return None
    edges = _decile_edges(sub[feature].to_numpy(dtype=float))
    bins = _bin(sub[feature].to_numpy(dtype=float), edges)
    pnls = sub["time_stop_pnl_put"].to_numpy(dtype=float)

    decile_stats = []
    for d in range(N_DECILES):
        m = bins == d
        if m.sum() == 0:
            decile_stats.append({"n": 0, "mean": float("nan"), "median": float("nan"),
                                 "p25": float("nan"), "p75": float("nan"),
                                 "hit_rate": float("nan")})
            continue
        d_pnls = pnls[m]
        decile_stats.append({
            "n": int(m.sum()),
            "mean": float(np.mean(d_pnls)),
            "median": float(np.median(d_pnls)),
            "p25": float(np.percentile(d_pnls, 25)),
            "p75": float(np.percentile(d_pnls, 75)),
            "hit_rate": float(np.mean(d_pnls > 0)),
        })

    means = np.array([s["mean"] for s in decile_stats], dtype=float)
    finite = np.isfinite(means)
    ranks = np.arange(N_DECILES, dtype=float)
    if finite.sum() < 3:
        spearman = float("nan")
    else:
        x = ranks[finite]
        y = means[finite]
        # Spearman: rank correlation of (rank-of-x, rank-of-y).
        x_r = pd.Series(x).rank().to_numpy()
        y_r = pd.Series(y).rank().to_numpy()
        if np.std(x_r) == 0 or np.std(y_r) == 0:
            spearman = float("nan")
        else:
            spearman = float(np.corrcoef(x_r, y_r)[0, 1])

    bottom_means = means[: 2]
    top_means = means[-2:]
    bottom_mean = float(np.nanmean(bottom_means)) if np.any(np.isfinite(bottom_means)) else float("nan")
    top_mean = float(np.nanmean(top_means)) if np.any(np.isfinite(top_means)) else float("nan")

    flagged = bool(
        np.isfinite(bottom_mean)
        and np.isfinite(top_mean)
        and np.isfinite(spearman)
        and abs(top_mean - bottom_mean) >= SEP_GAP_MIN
        and abs(spearman) >= SEP_RHO_MIN
    )

    return {
        "feature": feature,
        "n_train": int(len(sub)),
        "edges": edges.tolist(),
        "deciles": decile_stats,
        "bottom2_mean": bottom_mean,
        "top2_mean": top_mean,
        "spearman_rho": spearman,
        "flagged": flagged,
    }


def _per_fold_separation(
    rows: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
) -> dict[str, Any]:
    """For each fold, compute separation of each feature on train + val
    days (the universe the model would have learned from)."""
    universe = _eligible_universe(rows)
    out: dict[str, Any] = {"per_fold": {}, "all_folds": {}}

    # Per-fold (using train+val as the bin-edge source)
    pooled_means_per_feature: dict[str, list[float]] = {f: [] for f in features}
    for fold in folds:
        fold_idx = int(fold["fold_idx"])
        pretest_days = set(fold["train_days"]) | set(fold["val_days"])
        train_df = universe[universe["day"].isin(pretest_days)]
        per_feat = {}
        for feat in features:
            sep = _per_feature_separation(train_df, feat)
            if sep is None:
                continue
            per_feat[feat] = sep
            decile_means = [d["mean"] for d in sep["deciles"]]
            # Track for pooled summary
            if all(np.isfinite(decile_means)):
                pooled_means_per_feature[feat].append(decile_means)
        out["per_fold"][fold_idx] = per_feat

    # Pooled "all folds together" view: bin on full universe (still no test
    # leakage because we'll only use train+val rows from any fold).
    pooled_train_days: set[str] = set()
    for fold in folds:
        pooled_train_days.update(fold["train_days"])
        pooled_train_days.update(fold["val_days"])
    # Note: pooled view is descriptive only (helps eyeball stability across
    # folds); the simulation below uses per-fold edges to stay leakage-free.
    pooled_train = universe[universe["day"].isin(pooled_train_days)]
    for feat in features:
        sep = _per_feature_separation(pooled_train, feat)
        if sep is not None:
            out["all_folds"][feat] = sep
    return out


def _simulate_suppression(
    chosen_fallback: pd.DataFrame,
    rows: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature: str,
    suppress_k: int,
    equity: float,
    suppress_side: str = "bottom",
) -> dict[str, Any]:
    """Simulate: on chosen fallback bars, suppress to flat when
    feature(bar) falls in the bottom-`suppress_k` (or top-`suppress_k`)
    deciles defined by the fold's training universe.

    `suppress_side` is 'bottom' (drop the lowest-K deciles) or 'top'
    (drop the highest-K deciles). For features with negative rho between
    decile rank and put PnL, the low-payoff bars sit at the TOP, so we
    suppress 'top'.

    Baseline action is teacher+put on every fallback bar (that's what the
    current detach-side run does).
    """
    if suppress_side not in {"bottom", "top"}:
        raise ValueError(f"suppress_side must be 'bottom' or 'top', got {suppress_side!r}")
    universe = _eligible_universe(rows)
    fold_edges: dict[int, np.ndarray] = {}
    for fold in folds:
        fold_idx = int(fold["fold_idx"])
        pretest_days = set(fold["train_days"]) | set(fold["val_days"])
        train_df = universe[universe["day"].isin(pretest_days)]
        if feature not in train_df.columns:
            continue
        vals = train_df[feature].dropna().to_numpy(dtype=float)
        if len(vals) < 200:
            continue
        fold_edges[fold_idx] = _decile_edges(vals)

    simulated = []
    suppressed_count = 0
    kept_count = 0
    for _, row in chosen_fallback.iterrows():
        fold_idx = int(row["fold_idx"])
        pnl = float(row["time_stop_pnl_put"]) if pd.notna(row["time_stop_pnl_put"]) else None
        if pnl is None:
            continue
        edges = fold_edges.get(fold_idx)
        if edges is None or pd.isna(row.get(feature, np.nan)):
            # No edges -> can't gate; default to keep (teacher+put behavior)
            simulated.append({"day": row["day"], "fold_idx": fold_idx,
                              "bar_index": int(row["bar_index"]),
                              "feature_val": float(row.get(feature, np.nan)) if pd.notna(row.get(feature, np.nan)) else None,
                              "decile": None, "kept": True,
                              "pnl_kept": pnl, "pnl_after": pnl})
            kept_count += 1
            continue
        decile = int(_bin(np.array([float(row[feature])]), edges)[0])
        if suppress_side == "bottom":
            kept = decile >= suppress_k
        else:
            kept = decile < (N_DECILES - suppress_k)
        pnl_after = pnl if kept else 0.0
        if kept:
            kept_count += 1
        else:
            suppressed_count += 1
        simulated.append({"day": row["day"], "fold_idx": fold_idx,
                          "bar_index": int(row["bar_index"]),
                          "feature_val": float(row[feature]),
                          "decile": decile, "kept": bool(kept),
                          "pnl_kept": pnl, "pnl_after": pnl_after})

    sim_df = pd.DataFrame(simulated)

    # Now compose with the teacher trades (which are unchanged) and report
    # full-baseline PF/DD using kept fallback PnLs only (we drop suppressed
    # bars from the trade list rather than booking a $0 trade).
    return {
        "feature": feature,
        "suppress_k": suppress_k,
        "n_chosen_fallback": int(len(chosen_fallback)),
        "n_kept_fallback": int(kept_count),
        "n_suppressed_fallback": int(suppressed_count),
        "kept_mean_pnl": float(sim_df.loc[sim_df["kept"], "pnl_kept"].mean()) if kept_count else float("nan"),
        "suppressed_mean_pnl_avoided": float(sim_df.loc[~sim_df["kept"], "pnl_kept"].mean()) if suppressed_count else float("nan"),
        "simulated": sim_df,
    }


def _compose_full_run_metrics(
    teacher_trades: pd.DataFrame,
    sim_df: pd.DataFrame,
    total_days: int,
    equity: float,
) -> dict[str, Any]:
    """Build PF/DD on combined teacher trades + kept fallback trades.

    Trades are ordered chronologically by (day, bar_index) so the DD
    sequence matches the canonical replay equity curve.
    """
    teacher_view = teacher_trades[["day", "bar_index", "fold_idx", "pnl"]].copy()
    teacher_view["pnl"] = teacher_view["pnl"].astype(float)
    fb_kept = sim_df[sim_df["kept"]][["day", "bar_index", "fold_idx", "pnl_kept"]].copy()
    fb_kept = fb_kept.rename(columns={"pnl_kept": "pnl"})
    fb_kept["pnl"] = fb_kept["pnl"].astype(float)
    combined = pd.concat([teacher_view, fb_kept], ignore_index=True)
    combined = combined.sort_values(["day", "bar_index"]).reset_index(drop=True)

    overall = replay_metrics_from_pnls(combined["pnl"].tolist(), equity)
    overall["trades"] = float(len(combined))
    overall["trades_per_day"] = float(len(combined) / max(total_days, 1))

    per_fold: dict[int, dict[str, float]] = {}
    for fold_idx, group in combined.groupby("fold_idx"):
        m = replay_metrics_from_pnls(group["pnl"].tolist(), equity)
        m["trades"] = float(len(group))
        per_fold[int(fold_idx)] = m
    return {"overall": overall, "per_fold": per_fold}


def _load_chosen_with_features(
    dataset_path: str,
    baseline_run_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    bundle = load_export_bundle(dataset_path)
    rows: pd.DataFrame = bundle["rows"].copy()
    rows = _add_derived_features(rows)
    folds = list(bundle["meta"]["folds"])

    trades = pd.read_csv(os.path.join(baseline_run_dir, "layer2_trades.csv"))
    keep_cols = list(dict.fromkeys([
        "day", "bar_index", "fold_id",
        "teacher_any_triggered",
        "orc_buy_call", "orc_buy_put",
        "failed_break_buy_call", "failed_break_buy_put",
        "time_stop_pnl_call", "time_stop_pnl_put",
        "has_passing_call", "has_passing_put",
        *[f for f in CANDIDATE_FEATURES if f in rows.columns],
    ]))
    chosen = trades.merge(
        rows[keep_cols],
        left_on=["day", "bar_index", "fold_idx"],
        right_on=["day", "bar_index", "fold_id"],
        how="left",
        validate="one_to_one",
    )
    chosen["teacher_direction"] = chosen.apply(teacher_direction_hint_from_row, axis=1)
    chosen["route_source"] = np.where(chosen["teacher_direction"] != "", "teacher", "fallback")
    return chosen, rows, folds


def _print_separation_table(per_feature: dict[str, dict[str, Any]]) -> None:
    print()
    print("=" * 100)
    print("Pooled (all-folds train+val) separation table for fallback-eligible non-teacher universe")
    print("=" * 100)
    print(f"{'feature':<32}{'n':>8}{'bot2_mean':>12}{'top2_mean':>12}{'rho':>8}{'flag':>6}")
    rows_sorted = sorted(
        per_feature.values(),
        key=lambda r: (-(r.get("flagged") or False), -abs(r.get("spearman_rho") or 0.0)),
    )
    for r in rows_sorted:
        flag = " *" if r["flagged"] else ""
        print(
            f"{r['feature']:<32}{r['n_train']:>8}"
            f"{r['bottom2_mean']:>12.1f}{r['top2_mean']:>12.1f}"
            f"{r['spearman_rho']:>8.3f}{flag:>6}"
        )

    # Print decile means for the top 3 features by |rho|
    rows_top = sorted(
        per_feature.values(),
        key=lambda r: -abs(r.get("spearman_rho") or 0.0),
    )[:3]
    for r in rows_top:
        print()
        print(f"-- decile means for {r['feature']} (rho={r['spearman_rho']:+.3f}, flagged={r['flagged']}) --")
        for i, d in enumerate(r["deciles"]):
            if d["n"] == 0:
                continue
            print(f"  d{i}: n={d['n']:>5}  mean={d['mean']:>+8.1f}  median={d['median']:>+8.1f}  hit={d['hit_rate']:>4.2f}")


def _print_simulation(label: str, baseline: dict[str, Any], sim: dict[str, Any], compose: dict[str, Any]) -> None:
    o = compose["overall"]
    bo = baseline["overall"]
    print(
        f"  {label:<40s} kept={sim['n_kept_fallback']:>3d} suppressed={sim['n_suppressed_fallback']:>3d}  "
        f"PF={o['pf']:.3f} (Δ={o['pf']-bo['pf']:+.3f})  DD={o['max_dd_pct']:.1f}% (Δ={o['max_dd_pct']-bo['max_dd_pct']:+.1f})  "
        f"TPD={o['trades_per_day']:.3f} (Δ={o['trades_per_day']-bo['trades_per_day']:+.3f})  "
        f"avoided_mean={sim['suppressed_mean_pnl_avoided']:>+7.1f}$"
    )


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chosen, rows, folds = _load_chosen_with_features(args.dataset, args.baseline_run_dir)
    teacher_trades = chosen[chosen["route_source"] == "teacher"].copy()
    fallback_trades = chosen[chosen["route_source"] == "fallback"].copy()
    total_days = sum(len(f["test_days"]) for f in folds)

    print(f"Loaded chosen={len(chosen)} (teacher={len(teacher_trades)}, fallback={len(fallback_trades)})")
    print(f"Fold count: {len(folds)}; total test days: {total_days}")

    # --- Baseline metrics (always-put fallback) -----------------------------
    baseline_sim = pd.DataFrame({
        "day": fallback_trades["day"].values,
        "bar_index": fallback_trades["bar_index"].values,
        "fold_idx": fallback_trades["fold_idx"].values,
        "kept": [True] * len(fallback_trades),
        "pnl_kept": fallback_trades["pnl"].astype(float).values,
    })
    baseline_compose = _compose_full_run_metrics(teacher_trades, baseline_sim, total_days, args.equity)
    print()
    print("Baseline (teacher+put on all fallback bars):")
    bo = baseline_compose["overall"]
    print(f"  PF={bo['pf']:.3f}  DD={bo['max_dd_pct']:.1f}%  TPD={bo['trades_per_day']:.3f}  trades={int(bo['trades'])}")
    for fi in sorted(baseline_compose["per_fold"]):
        m = baseline_compose["per_fold"][fi]
        print(f"    fold {fi}: PF={m['pf']:.3f} DD={m['max_dd_pct']:.1f}% trades={int(m['trades'])}")

    # --- Separation analysis on full pooled train+val universe --------------
    sep_per_feature = _per_fold_separation(rows, folds, CANDIDATE_FEATURES)
    pooled = sep_per_feature["all_folds"]
    _print_separation_table(pooled)

    # --- Suppression simulation for any flagged or near-flagged feature -----
    print()
    print("=" * 100)
    print(f"Suppression simulation: drop fallback bar if feature in bottom {args.suppress_k} deciles "
          f"(per-fold edges from train+val)")
    print("=" * 100)

    # Always test the top-K by |rho|, regardless of flagged status, so the
    # gate doesn't pivot on a single hard threshold. Mark each as flagged
    # vs near for the verdict.
    candidates: list[tuple[str, float, str]] = []
    flagged_feats = {feat for feat, r in pooled.items() if r["flagged"]}
    ranked = sorted(pooled.values(), key=lambda x: -abs(x.get("spearman_rho") or 0.0))
    for r in ranked[:5]:
        why = "flagged" if r["feature"] in flagged_feats else "near"
        candidates.append((r["feature"], r["spearman_rho"], why))

    sims_out: dict[str, Any] = {}
    for feat, rho, why in candidates:
        # Suppress the LOW-PAYOFF tail. With rho > 0, low payoff sits at
        # the bottom of the feature distribution; with rho < 0, it sits at
        # the top.
        suppress_side = "bottom" if (rho is not None and rho >= 0) else "top"
        sim = _simulate_suppression(
            fallback_trades, rows, folds, feat, args.suppress_k, args.equity,
            suppress_side=suppress_side,
        )
        compose = _compose_full_run_metrics(teacher_trades, sim["simulated"], total_days, args.equity)
        per_fold_pf = {fi: compose["per_fold"][fi]["pf"] for fi in compose["per_fold"]}
        sims_out[feat] = {
            "spearman_rho": rho,
            "reason": why,
            "suppress_side": suppress_side,
            "n_kept": sim["n_kept_fallback"],
            "n_suppressed": sim["n_suppressed_fallback"],
            "suppressed_mean_pnl_avoided": sim["suppressed_mean_pnl_avoided"],
            "kept_mean_pnl": sim["kept_mean_pnl"],
            "overall": compose["overall"],
            "per_fold": compose["per_fold"],
            "fold_pf": per_fold_pf,
        }
        _print_simulation(f"{feat} ({why}, rho={rho:+.3f}, suppress {suppress_side})",
                          baseline_compose, sim, compose)

    # --- Verdict ------------------------------------------------------------
    print()
    print("=" * 100)
    print("Verdict criteria")
    print("=" * 100)
    flagged_count = sum(1 for r in pooled.values() if r["flagged"])
    best_feat = None
    best_pf = -math.inf
    for feat, s in sims_out.items():
        if s["overall"]["pf"] > best_pf:
            best_pf = float(s["overall"]["pf"])
            best_feat = feat

    fold0_baseline = float(baseline_compose["per_fold"].get(0, {}).get("pf", float("nan")))
    fold0_best = float(sims_out.get(best_feat, {}).get("per_fold", {}).get(0, {}).get("pf", float("nan"))) if best_feat else float("nan")

    if flagged_count == 0:
        print("  ⚠ No candidate feature shows monotonic separation per criteria (bot2<$0, top2>$300, |rho|>=0.50).")
        if best_feat is None:
            verdict = "FALSIFIED — no separation, no suppression candidate"
        elif best_pf < PF_FAIL_THRESHOLD:
            verdict = f"FALSIFIED — best near-candidate {best_feat!r} suppression PF={best_pf:.3f} < {PF_FAIL_THRESHOLD}"
        elif best_pf >= PF_PASS_THRESHOLD and fold0_best >= 1.0:
            verdict = f"WEAK PASS — {best_feat!r} suppression PF={best_pf:.3f} but no flagged feature; treat as exploratory"
        else:
            verdict = f"INCONCLUSIVE — best near-candidate {best_feat!r} PF={best_pf:.3f} (fold0={fold0_best:.3f})"
    else:
        if best_pf < PF_FAIL_THRESHOLD:
            verdict = f"FALSIFIED — best flagged feature {best_feat!r} suppression PF={best_pf:.3f} < {PF_FAIL_THRESHOLD}"
        elif best_pf >= PF_PASS_THRESHOLD and fold0_best >= 1.0:
            verdict = f"PASS — {best_feat!r} suppression PF={best_pf:.3f} (fold0={fold0_best:.3f}); regime gate is real"
        else:
            verdict = (f"SOFT — {best_feat!r} suppression PF={best_pf:.3f} (baseline {bo['pf']:.3f}, "
                      f"fold0 baseline={fold0_baseline:.3f} -> {fold0_best:.3f}); inspect DD/TPD trade-off")

    print(f"  Best feature: {best_feat}  best PF: {best_pf:.3f}  baseline PF: {bo['pf']:.3f}")
    print(f"  Fold-0 baseline PF: {fold0_baseline:.3f}  Fold-0 best PF: {fold0_best:.3f}")
    print(f"  VERDICT: {verdict}")

    # --- Save artifact ------------------------------------------------------
    serializable = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "dataset": args.dataset,
            "n_chosen": int(len(chosen)),
            "n_teacher": int(len(teacher_trades)),
            "n_fallback": int(len(fallback_trades)),
            "suppress_k": int(args.suppress_k),
            "candidate_features": CANDIDATE_FEATURES,
        },
        "baseline": baseline_compose,
        "separation_pooled": pooled,
        "suppression_sims": {k: {kk: vv for kk, vv in v.items() if kk != "simulated"} for k, v in sims_out.items()},
        "verdict": verdict,
    }
    out_json = os.path.join(args.out_dir, "payoff_gating_probe.json")
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print()
    print(f"Saved results: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
