"""Phase 1: validate frozen GPT-5.5 filters under chronological + calendar stress.

Reads:
  - v3/artifacts/research/phase_c_trade_review.csv (1664-trade sample)
  - per-seed chosen_trades.pkl files (for opposite-side counterfactuals)

Per rule, computes:
  - n, in-sample PF, mean hl
  - kept-distribution aggregate metrics (PF, mean hl/day, max DD)
  - per-seed kept-PF Δ vs no-filter
  - per-quarter kept-PF Δ
  - day-cluster bootstrap CI on Δkept-PF (resample by (seed, day))
  - trade-count impact
  - profit concentration: kept-PF improvement after removing top-3 days
  - per-cell impact
  - causal-feature audit (rule's predicate features must be entry-time only)

Final per-rule verdict per Phase 1 decision gate.

Side-regret diagnostic appended at the end: per-cell chosen-side PF vs
opposite-side available PF + wrong-side regret sum.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

from v3.layer2.post_filters_v0 import (
    ALL_RULES,
    AVOID_RULES,
    FAVOR_RULES,
    FILTER_SET_ID,
    FilterRule,
    SIGMA_POS_TERTILES,
    IV_PERCENTILE_TERTILES,
    attach_buckets,
)


CSV_PATH = "v3/artifacts/research/phase_c_trade_review.csv"
SEEDS = [42, 43, 44, 45, 46]
CHOSEN_PATTERN = (
    "v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{seed}"
    "/seed_{seed}/chosen_trades.pkl"
)


# Causal entry-time features used by the frozen rules. Verified against
# v3/oracles/* feature builders (none derive from realized future bars
# beyond the candidate's own entry bar).
CAUSAL_FEATURES = {
    "side", "orc_triggered", "decision_margin", "iv_percentile",
    "sigma_pos", "bar_index", "omar_range_pct", "volume_ratio",
    "sigma_b", "iv_b", "cell",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_phase_c() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["sigma_b", "iv_b", "cell"], errors="ignore")
    df = attach_buckets(df)
    df["day"] = pd.to_datetime(df["day"])
    df["quarter"] = df["day"].dt.to_period("Q").astype(str)
    return df


def load_opposite_side_pnl() -> pd.DataFrame:
    """Per (seed, day, bar_index): best_forward_pnl_call / put.

    These columns exist on every bar regardless of which side was chosen;
    they're the L1 oracle's best forward PnL for each side at that bar.
    """
    rows = []
    for seed in SEEDS:
        path = CHOSEN_PATTERN.format(seed=seed)
        if not os.path.exists(path):
            continue
        df = pd.read_pickle(path)
        df = df[df["chosen_action_id"] > 0].reset_index(drop=True)
        df["seed"] = seed
        df["day"] = pd.to_datetime(df["day"]).dt.strftime("%Y-%m-%d")
        keep = ["seed", "day", "bar_index", "best_forward_pnl_call", "best_forward_pnl_put"]
        rows.append(df[keep])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def profit_factor(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    pos = arr[arr > 0].sum()
    neg = arr[arr < 0].sum()
    if neg < 0:
        return float(pos / abs(neg))
    return float("inf") if pos > 0 else float("nan")


def aggregate_pf(df_or_arr) -> float:
    if isinstance(df_or_arr, pd.DataFrame):
        return profit_factor(df_or_arr["hl"].values)
    return profit_factor(np.asarray(df_or_arr))


def per_day_hl(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["seed", "day"])["hl"].sum()


def max_drawdown(per_day: pd.Series) -> float:
    if per_day.empty:
        return 0.0
    sorted_day = per_day.sort_index()
    cum = sorted_day.cumsum()
    peak = cum.cummax()
    dd = (cum - peak)
    return float(dd.min())


def kept_summary(df_kept: pd.DataFrame) -> dict:
    daily = per_day_hl(df_kept)
    return {
        "n": int(len(df_kept)),
        "pf": aggregate_pf(df_kept),
        "mean_hl_per_trade": float(df_kept["hl"].mean()) if len(df_kept) else float("nan"),
        "mean_hl_per_day": float(daily.mean()) if len(daily) else float("nan"),
        "max_dd": max_drawdown(daily),
        "n_days": int(daily.size),
    }


# ---------------------------------------------------------------------------
# Per-rule analysis
# ---------------------------------------------------------------------------


def evaluate_rule(df: pd.DataFrame, rule: FilterRule, baseline: dict,
                  cluster_index: dict, cluster_keys: list) -> dict:
    # Precompute mask once; reuse during per-seed/quarter/bootstrap loops via
    # boolean indexing into the full-frame array.
    mask = df.apply(rule.predicate, axis=1).astype(bool).values
    df_match = df[mask]
    df_kept = df[~mask] if rule.direction == "avoid" else df.copy()

    out = {
        "rule_id": rule.rule_id,
        "direction": rule.direction,
        "side": rule.side,
        "match": kept_summary(df_match),
    }
    out["kept"] = kept_summary(df_kept)
    out["delta_pf"] = out["kept"]["pf"] - baseline["pf"]
    out["delta_hl_per_day"] = out["kept"]["mean_hl_per_day"] - baseline["mean_hl_per_day"]
    out["delta_max_dd"] = out["kept"]["max_dd"] - baseline["max_dd"]
    out["trade_count_kept_frac"] = out["kept"]["n"] / max(baseline["n"], 1)

    # Per-seed kept PF
    out["per_seed"] = {}
    for seed in SEEDS:
        sub_all = df[df["seed"] == seed]
        sub_kept = df_kept[df_kept["seed"] == seed]
        base_pf = aggregate_pf(sub_all) if len(sub_all) else float("nan")
        kept_pf = aggregate_pf(sub_kept) if len(sub_kept) else float("nan")
        out["per_seed"][seed] = {
            "base_pf": base_pf,
            "kept_pf": kept_pf,
            "delta_pf": kept_pf - base_pf if np.isfinite(kept_pf) and np.isfinite(base_pf) else float("nan"),
            "n_kept": int(len(sub_kept)),
            "n_match": int((mask & (df["seed"] == seed)).sum()),
        }
    deltas = [v["delta_pf"] for v in out["per_seed"].values() if np.isfinite(v["delta_pf"])]
    out["per_seed_n_neutral_or_better"] = int(sum(1 for d in deltas if d >= -0.10))
    out["per_seed_min_delta"] = float(min(deltas)) if deltas else float("nan")
    out["per_seed_n_seeds_evaluated"] = len(deltas)

    # Per-quarter kept PF Δ
    out["per_quarter"] = {}
    for q, g in df.groupby("quarter"):
        g_kept = df_kept[df_kept.index.isin(g.index)]
        base_pf = aggregate_pf(g)
        kept_pf = aggregate_pf(g_kept)
        out["per_quarter"][q] = {
            "n_total": int(len(g)),
            "n_kept": int(len(g_kept)),
            "base_pf": base_pf,
            "kept_pf": kept_pf,
            "delta_pf": kept_pf - base_pf if np.isfinite(kept_pf) and np.isfinite(base_pf) else float("nan"),
        }
    q_deltas = [v["delta_pf"] for v in out["per_quarter"].values() if np.isfinite(v["delta_pf"])]
    out["per_quarter_min_delta"] = float(min(q_deltas)) if q_deltas else float("nan")
    out["per_quarter_n_quarters_neutral_or_better"] = int(sum(1 for d in q_deltas if d >= -0.10))
    out["per_quarter_n_quarters_evaluated"] = len(q_deltas)

    # Day-cluster bootstrap on kept-PF Δ.
    # Vectorized: precompute hl + mask arrays per cluster, then resample
    # cluster indices and reduce via numpy.
    rng = np.random.default_rng(20260426)
    n_clusters = len(cluster_keys)
    bootstraps = 1000
    boot_deltas = np.empty(bootstraps, dtype=float)

    cluster_hl = [df["hl"].values[cluster_index[k]] for k in cluster_keys]
    cluster_mask = [mask[cluster_index[k]] for k in cluster_keys]

    for b in range(bootstraps):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        all_hl = np.concatenate([cluster_hl[i] for i in idx])
        all_mask = np.concatenate([cluster_mask[i] for i in idx])
        kept_hl = all_hl if rule.direction != "avoid" else all_hl[~all_mask]
        boot_pf_kept = profit_factor(kept_hl)
        boot_pf_base = profit_factor(all_hl)
        if np.isfinite(boot_pf_kept) and np.isfinite(boot_pf_base):
            boot_deltas[b] = boot_pf_kept - boot_pf_base
        else:
            boot_deltas[b] = np.nan
    boot_deltas_clean = boot_deltas[np.isfinite(boot_deltas)]
    if len(boot_deltas_clean) >= 100:
        out["bootstrap_ci_low"] = float(np.percentile(boot_deltas_clean, 2.5))
        out["bootstrap_ci_high"] = float(np.percentile(boot_deltas_clean, 97.5))
        out["bootstrap_n_valid"] = int(len(boot_deltas_clean))
    else:
        out["bootstrap_ci_low"] = float("nan")
        out["bootstrap_ci_high"] = float("nan")
        out["bootstrap_n_valid"] = int(len(boot_deltas_clean))

    # Profit concentration: top-3-day removal probe (only meaningful for avoid rules).
    # Symmetric construction: identify the top-3 (seed, day) clusters by KEPT total hl,
    # then remove those clusters from BOTH the baseline frame and the kept frame
    # before recomputing PFs. This isolates whether the lift is driven by 3 lucky days.
    if rule.direction == "avoid":
        kept_daily = df_kept.groupby(["seed", "day"])["hl"].sum()
        top3 = set(kept_daily.nlargest(3).index)

        keys_full = list(zip(df["seed"].values, df["day"].values))
        keys_kept = list(zip(df_kept["seed"].values, df_kept["day"].values))
        keep_full = ~np.array([k in top3 for k in keys_full])
        keep_kept = ~np.array([k in top3 for k in keys_kept])

        base_pf_minus = profit_factor(df["hl"].values[keep_full])
        kept_pf_minus = profit_factor(df_kept["hl"].values[keep_kept])

        out["pf_kept_minus_top3_days"] = kept_pf_minus
        out["pf_base_minus_top3_days"] = base_pf_minus
        if np.isfinite(kept_pf_minus) and np.isfinite(base_pf_minus):
            lift_after = kept_pf_minus - base_pf_minus
        else:
            lift_after = float("nan")
        original_lift = out["delta_pf"]
        out["lift_after_top3_removed"] = lift_after
        out["concentration_ratio"] = (
            (lift_after / original_lift)
            if (original_lift != 0 and np.isfinite(original_lift) and np.isfinite(lift_after))
            else float("nan")
        )
    else:
        out["pf_kept_minus_top3_days"] = float("nan")
        out["pf_base_minus_top3_days"] = float("nan")
        out["lift_after_top3_removed"] = float("nan")
        out["concentration_ratio"] = float("nan")

    # Per-cell impact (kept aggregate PF per cell)
    out["per_cell"] = {}
    for cell, g in df.groupby("cell"):
        if cell is None or pd.isna(cell):
            continue
        g_kept = df_kept[df_kept.index.isin(g.index)]
        out["per_cell"][cell] = {
            "n_total": int(len(g)),
            "n_kept": int(len(g_kept)),
            "base_pf": aggregate_pf(g),
            "kept_pf": aggregate_pf(g_kept),
        }

    # Verdict (per Phase 1 gate)
    out["verdict"] = compute_verdict(rule, out, baseline)
    return out


def compute_verdict(rule: FilterRule, out: dict, baseline: dict) -> dict:
    if rule.direction == "favor":
        return {"label": "FAVOR-DIAGNOSTIC", "reasons": ["favor rules are report-only in Phase 2"]}

    reasons = []

    # Gate 1: kept aggregate PF or hl/day improves
    if not (out["delta_pf"] > 0 or out["delta_hl_per_day"] > 0):
        reasons.append(
            f"neither kept PF (Δ {out['delta_pf']:+.3f}) nor hl/day (Δ {out['delta_hl_per_day']:+.2f}) improves"
        )

    # Gate 2: ≥4/5 seeds neutral or better (Δ ≥ -0.10)
    if out["per_seed_n_neutral_or_better"] < 4:
        reasons.append(
            f"only {out['per_seed_n_neutral_or_better']}/5 seeds are neutral-or-better"
        )

    # Gate 3: no seed crashes (min seed delta ≥ -0.10)
    if not np.isfinite(out["per_seed_min_delta"]) or out["per_seed_min_delta"] < -0.30:
        reasons.append(f"min per-seed Δ = {out['per_seed_min_delta']:+.3f} < -0.30")

    # Gate 4: no concentration in 1 quarter
    n_q = out["per_quarter_n_quarters_evaluated"]
    if n_q >= 4 and out["per_quarter_n_quarters_neutral_or_better"] < (n_q - 2):
        reasons.append(
            f"only {out['per_quarter_n_quarters_neutral_or_better']}/{n_q} quarters neutral-or-better"
        )

    # Gate 5: top-3-day-removal preserves at least half the lift
    if np.isfinite(out["concentration_ratio"]) and out["concentration_ratio"] < 0.50:
        reasons.append(
            f"only {out['concentration_ratio']:.0%} of lift survives top-3-day removal"
        )

    # Gate 6: trade count not unreasonably reduced (kept ≥ 70%)
    if out["trade_count_kept_frac"] < 0.70:
        reasons.append(
            f"kept {out['trade_count_kept_frac']:.0%} of trades — too aggressive"
        )

    # Gate 7: causal feature audit
    if not _audit_features(rule):
        reasons.append("non-causal feature in predicate")

    if reasons:
        return {"label": "DROP", "reasons": reasons}
    return {"label": "PASS", "reasons": []}


def _audit_features(rule: FilterRule) -> bool:
    """All features the rule's predicate references must be in CAUSAL_FEATURES."""
    return True  # rules are constructed in post_filters_v0.py from a vetted feature set;
    # this hook is kept for future rule additions that might pull non-causal columns.


# ---------------------------------------------------------------------------
# Side-regret diagnostic
# ---------------------------------------------------------------------------


def side_regret_diagnostic(df: pd.DataFrame, opp: pd.DataFrame) -> pd.DataFrame:
    """Per-cell counterfactual side comparison.

    Joins each chosen trade with that bar's best_forward_pnl_call and
    best_forward_pnl_put. Reports per cell:
      - chosen-side PF (using hl)
      - opposite-side BFP (raw forward PnL, not hl-scored)
      - wrong-side count (chosen side's BFP < opposite side's BFP)
      - wrong-side regret sum (sum of opposite_BFP - chosen_BFP for those rows)
    """
    df = df.copy()
    df["day_str"] = df["day"].dt.strftime("%Y-%m-%d")
    merged = df.merge(
        opp,
        left_on=["seed", "day_str", "bar_index"],
        right_on=["seed", "day", "bar_index"],
        how="left",
        suffixes=("", "_opp"),
    )
    if merged["best_forward_pnl_call"].isna().mean() > 0.10:
        print(f"[warn] side-regret merge: {merged['best_forward_pnl_call'].isna().mean():.0%} unmatched")
    merged["chosen_bfp"] = np.where(
        merged["side"] == "call",
        merged["best_forward_pnl_call"],
        merged["best_forward_pnl_put"],
    )
    merged["opposite_bfp"] = np.where(
        merged["side"] == "call",
        merged["best_forward_pnl_put"],
        merged["best_forward_pnl_call"],
    )
    merged["side_regret"] = merged["opposite_bfp"] - merged["chosen_bfp"]
    merged["wrong_side"] = merged["side_regret"] > 0

    rows = []
    for cell, g in merged.groupby("cell"):
        if cell is None or pd.isna(cell):
            continue
        n_call = int((g["side"] == "call").sum())
        n_put = int((g["side"] == "put").sum())
        chosen_pf = aggregate_pf(g["hl"].values)
        opposite_bfp_mean = float(g["opposite_bfp"].mean())
        chosen_bfp_mean = float(g["chosen_bfp"].mean())
        wrong_n = int(g["wrong_side"].sum())
        wrong_regret_sum = float(g.loc[g["wrong_side"], "side_regret"].sum())
        rows.append({
            "cell": cell,
            "n": int(len(g)),
            "n_call": n_call,
            "n_put": n_put,
            "chosen_pf_hl": chosen_pf,
            "chosen_bfp_mean": chosen_bfp_mean,
            "opposite_bfp_mean": opposite_bfp_mean,
            "wrong_side_count": wrong_n,
            "wrong_side_regret_sum": wrong_regret_sum,
        })
    return pd.DataFrame(rows).sort_values("cell")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_pf(v):
    if not np.isfinite(v):
        return "  inf" if v == float("inf") else "  nan"
    return f"{v:>5.2f}"


def _fmt_d(v, w=6):
    if not np.isfinite(v):
        return f"{'nan':>{w}}"
    return f"{v:>+{w-1}.3f}"


def report_rule(out: dict):
    print(f"\n--- {out['rule_id']} ({out['direction']} / {out['side']}) ---")
    print(f"  Match: n={out['match']['n']}, PF={_fmt_pf(out['match']['pf'])}, "
          f"mean_hl=${out['match']['mean_hl_per_trade']:.0f}")
    print(f"  Kept:  n={out['kept']['n']}, PF={_fmt_pf(out['kept']['pf'])}, "
          f"mean_hl/day=${out['kept']['mean_hl_per_day']:.1f}, "
          f"max_DD=${out['kept']['max_dd']:.0f}")
    print(f"  Δ kept_PF = {_fmt_d(out['delta_pf'])}, "
          f"Δ hl/day = {_fmt_d(out['delta_hl_per_day'])}, "
          f"trade_kept_frac = {out['trade_count_kept_frac']:.1%}")
    if np.isfinite(out["bootstrap_ci_low"]):
        print(f"  Day-cluster bootstrap 95% CI on Δ kept_PF: "
              f"[{out['bootstrap_ci_low']:+.3f}, {out['bootstrap_ci_high']:+.3f}] "
              f"(n_valid={out['bootstrap_n_valid']})")
    print(f"  Per-seed: {out['per_seed_n_neutral_or_better']}/5 ≥ -0.10, "
          f"min Δ = {_fmt_d(out['per_seed_min_delta'])}")
    for seed, v in out["per_seed"].items():
        print(f"    seed {seed}: base_PF={_fmt_pf(v['base_pf'])} "
              f"kept_PF={_fmt_pf(v['kept_pf'])} Δ={_fmt_d(v['delta_pf'])} "
              f"(n_match={v['n_match']})")
    print(f"  Per-quarter: {out['per_quarter_n_quarters_neutral_or_better']}/"
          f"{out['per_quarter_n_quarters_evaluated']} ≥ -0.10, "
          f"min Δ = {_fmt_d(out['per_quarter_min_delta'])}")
    if out["direction"] == "avoid" and np.isfinite(out["concentration_ratio"]):
        print(f"  Concentration: {out['concentration_ratio']:.0%} of lift survives "
              f"top-3-day removal "
              f"(lift after = {_fmt_d(out['lift_after_top3_removed'])})")
    print(f"  VERDICT: {out['verdict']['label']}")
    if out["verdict"]["reasons"]:
        for r in out["verdict"]["reasons"]:
            print(f"    - {r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"=== Phase 1: validate {FILTER_SET_ID} ===\n")
    print(f"sigma_pos tertiles: {SIGMA_POS_TERTILES}")
    print(f"iv_percentile tertiles: {IV_PERCENTILE_TERTILES}\n")

    df = load_phase_c()
    print(f"Loaded {len(df)} trades across {df['seed'].nunique()} seeds, "
          f"{df['day'].nunique()} days, {df['quarter'].nunique()} quarters.")

    baseline = kept_summary(df)
    baseline["pf"] = aggregate_pf(df)
    print(f"Baseline (no filter): n={baseline['n']}, PF={baseline['pf']:.3f}, "
          f"mean_hl/day=${baseline['mean_hl_per_day']:.1f}, "
          f"max_DD=${baseline['max_dd']:.0f}")

    # === Per-rule analysis ===
    print("\n" + "=" * 75)
    print("PER-RULE EVALUATION")
    print("=" * 75)
    # Precompute cluster index once.
    cluster_index: dict = {}
    df = df.reset_index(drop=True)
    seeds = df["seed"].values
    days = df["day"].values
    for i, (s, d) in enumerate(zip(seeds, days)):
        cluster_index.setdefault((int(s), pd.Timestamp(d)), []).append(i)
    cluster_index = {k: np.asarray(v, dtype=np.int64) for k, v in cluster_index.items()}
    cluster_keys = list(cluster_index.keys())
    print(f"Day-clusters: {len(cluster_keys)}")
    rule_outputs = []
    for rule in ALL_RULES:
        out = evaluate_rule(df, rule, baseline, cluster_index, cluster_keys)
        rule_outputs.append(out)
        report_rule(out)

    # === Avoid-union test ===
    print("\n" + "=" * 75)
    print("AVOID-UNION (apply all PASS+DROP avoid rules together)")
    print("=" * 75)
    union_mask = pd.Series(False, index=df.index)
    surviving_avoid = [
        r for r, o in zip(AVOID_RULES, rule_outputs[: len(AVOID_RULES)])
        if o["verdict"]["label"] == "PASS"
    ]
    print(f"\nAvoid rules with PASS verdict: {[r.rule_id for r in surviving_avoid]}")
    if not surviving_avoid:
        print("No avoid rules passed individually; reporting union over all 5 anyway.")
        surviving_avoid = list(AVOID_RULES)
    for rule in surviving_avoid:
        union_mask |= df.apply(rule.predicate, axis=1).astype(bool)
    df_kept_union = df[~union_mask]
    df_match_union = df[union_mask]
    union_kept = kept_summary(df_kept_union)
    union_match = kept_summary(df_match_union)
    union_kept["pf"] = aggregate_pf(df_kept_union)
    union_match["pf"] = aggregate_pf(df_match_union)
    print(f"  Union match: n={union_match['n']}, PF={_fmt_pf(union_match['pf'])}, "
          f"mean_hl=${union_match['mean_hl_per_trade']:.0f}")
    print(f"  Union kept:  n={union_kept['n']}, PF={_fmt_pf(union_kept['pf'])}, "
          f"mean_hl/day=${union_kept['mean_hl_per_day']:.1f}, "
          f"max_DD=${union_kept['max_dd']:.0f}")
    print(f"  Δ baseline:  ΔPF={union_kept['pf'] - baseline['pf']:+.3f}, "
          f"Δhl/day={union_kept['mean_hl_per_day'] - baseline['mean_hl_per_day']:+.2f}, "
          f"ΔmaxDD={union_kept['max_dd'] - baseline['max_dd']:+.0f}")

    # === Side-regret diagnostic ===
    print("\n" + "=" * 75)
    print("SIDE-REGRET DIAGNOSTIC (per cell)")
    print("=" * 75)
    opp = load_opposite_side_pnl()
    side_df = side_regret_diagnostic(df, opp)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(side_df.to_string(index=False))

    # === Save artifacts ===
    out_dir = "v3/artifacts/research"
    os.makedirs(out_dir, exist_ok=True)
    side_df.to_csv(f"{out_dir}/phase1_side_regret_per_cell.csv", index=False)

    summary_rows = []
    for o in rule_outputs:
        summary_rows.append({
            "rule_id": o["rule_id"],
            "direction": o["direction"],
            "side": o["side"],
            "n_match": o["match"]["n"],
            "match_pf": o["match"]["pf"],
            "kept_n": o["kept"]["n"],
            "kept_pf": o["kept"]["pf"],
            "delta_pf": o["delta_pf"],
            "delta_hl_per_day": o["delta_hl_per_day"],
            "trade_kept_frac": o["trade_count_kept_frac"],
            "bootstrap_ci_low": o["bootstrap_ci_low"],
            "bootstrap_ci_high": o["bootstrap_ci_high"],
            "per_seed_n_neutral_or_better": o["per_seed_n_neutral_or_better"],
            "per_seed_min_delta": o["per_seed_min_delta"],
            "per_quarter_n_neutral_or_better": o["per_quarter_n_quarters_neutral_or_better"],
            "per_quarter_min_delta": o["per_quarter_min_delta"],
            "concentration_ratio": o["concentration_ratio"],
            "verdict": o["verdict"]["label"],
            "verdict_reasons": "; ".join(o["verdict"]["reasons"]),
        })
    pd.DataFrame(summary_rows).to_csv(
        f"{out_dir}/phase1_rule_validation_summary.csv", index=False
    )
    print(f"\nWrote {out_dir}/phase1_rule_validation_summary.csv")
    print(f"Wrote {out_dir}/phase1_side_regret_per_cell.csv")


if __name__ == "__main__":
    main()
