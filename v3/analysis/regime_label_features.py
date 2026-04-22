"""Phase 3A — Regime label definition + per-day feature engineering.

Goal: build a labeled dataset where each row is one chosen trading day
with (a) a label encoding whether V0's directional choice beat V1's
"always_put", and (b) per-day features available before any per-bar
trade decision (extracted at bar 14 = end of first 15 minutes).

If we can later train a classifier P(V0_favorable | features) with
meaningful AUC, we can switch between V0 and V1 per day intelligently.

Label: L1 with smoothing
  label = 1 (V0-favorable) if V0_pnl > V1_pnl + $50
  label = 0 (V1-favorable or tie) otherwise
Per the plan; the $50 buffer prevents single-trade noise from flipping
the label on near-equal days.

Per-day features (extracted from V2Dataset at bar 14):
  - opening_gap_pct          (32) overnight gap
  - session_open_dist         (33)
  - vwap_dist                 (6)
  - first15_range_pct         (34)
  - first15_close_position    (35)
  - vix_roc                   (12)
  - atm_iv                    (56)
  - iv_percentile             (58)
  - realized_vol              (4)
  - vix_regime                (14) categorical
  - sigma_pos                 (48)
  - omar_range_pct            (50)
  - omar_mid_pos_units        (54)

Sources:
  - In-sample V0: layer2_trades.csv (275 trades)
  - In-sample V1: in_sample_trades_V1.csv (281 trades)
  - OOS V0:       oos_trades_V0.csv (20 trades)
  - OOS V1:       oos_trades_V1.csv (20 trades)

Output: v3/artifacts/regime_labels/regime_labels.csv with columns
[day, fold_idx, v0_pnl, v1_pnl, label, ...features...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from v3.harness.v2_adapter import V2Dataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_VARIANTS_DIR = os.path.join("v3", "artifacts", "layer2_directional_variants")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "regime_labels")
LABEL_SMOOTHING_USD = 50.0
FEATURE_BAR = 14    # bar index where we extract features (end of first 15 min)

PER_DAY_FEATURE_NAMES = [
    "opening_gap_pct",
    "session_open_dist",
    "vwap_dist",
    "first15_range_pct",
    "first15_close_position",
    "vix_roc",
    "atm_iv",
    "iv_percentile",
    "realized_vol",
    "vix_regime",
    "sigma_pos",
    "omar_range_pct",
    "omar_mid_pos_units",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--variants-dir", default=DEFAULT_VARIANTS_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--label-smoothing-usd", type=float, default=LABEL_SMOOTHING_USD)
    p.add_argument("--feature-bar", type=int, default=FEATURE_BAR)
    return p.parse_args()


def load_v0_trades(baseline_run_dir: str, variants_dir: str) -> pd.DataFrame:
    """Combine in-sample (layer2_trades.csv) + OOS (oos_trades_V0.csv) V0."""
    is_path = os.path.join(baseline_run_dir, "layer2_trades.csv")
    oos_path = os.path.join(variants_dir, "oos_trades_V0.csv")
    is_df = pd.read_csv(is_path)
    is_df["origin"] = "in_sample"
    oos_df = pd.read_csv(oos_path)
    oos_df["origin"] = "oos"
    keep_cols = ["day", "bar_index", "direction", "pnl", "fold_idx", "origin"]
    is_df = is_df[[c for c in keep_cols if c in is_df.columns]]
    oos_df = oos_df[[c for c in keep_cols if c in oos_df.columns]]
    return pd.concat([is_df, oos_df], ignore_index=True)


def load_v1_trades(variants_dir: str) -> pd.DataFrame:
    is_path = os.path.join(variants_dir, "in_sample_trades_V1.csv")
    oos_path = os.path.join(variants_dir, "oos_trades_V1.csv")
    is_df = pd.read_csv(is_path)
    is_df["origin"] = "in_sample"
    oos_df = pd.read_csv(oos_path)
    oos_df["origin"] = "oos"
    keep_cols = ["day", "bar_index", "direction", "pnl", "fold_idx", "origin"]
    is_df = is_df[[c for c in keep_cols if c in is_df.columns]]
    oos_df = oos_df[[c for c in keep_cols if c in oos_df.columns]]
    return pd.concat([is_df, oos_df], ignore_index=True)


def extract_day_features(
    ds: V2Dataset, day: str, feature_bar: int, feature_indices: list[int],
    feature_names: list[str],
) -> dict:
    """Extract per-day features at the given bar (relative to session start)."""
    try:
        start, end = ds.day_bar_range(day)
    except Exception:
        return {n: float("nan") for n in feature_names}
    if end - start <= feature_bar:
        return {n: float("nan") for n in feature_names}
    abs_bar = start + feature_bar
    feats = ds.X_sim[abs_bar]
    return {n: float(feats[i]) for i, n in zip(feature_indices, feature_names)}


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset...", flush=True)
    ds = V2Dataset.load()
    feature_indices = [ds.feature_names.index(n) for n in PER_DAY_FEATURE_NAMES]
    print(f"  per-day features (extracted at bar {args.feature_bar}):", flush=True)
    for i, n in zip(feature_indices, PER_DAY_FEATURE_NAMES):
        print(f"    {i:>3}: {n}", flush=True)

    # Load V0 + V1 trade tables
    print(flush=True)
    print("Loading trade tables...", flush=True)
    v0 = load_v0_trades(args.baseline_run_dir, args.variants_dir)
    v1 = load_v1_trades(args.variants_dir)
    print(f"  V0 (in-sample + OOS): {len(v0)} trades", flush=True)
    print(f"  V1 (in-sample + OOS): {len(v1)} trades", flush=True)

    # Day-level join: v0 has bar_index unique per day (per_day_choice picks one)
    # but to be safe, group by day and aggregate
    v0_day = v0.groupby(["day", "origin"], as_index=False).agg(
        v0_pnl=("pnl", "sum"),
        v0_n_trades=("pnl", "count"),
        v0_direction=("direction", "first"),
        v0_bar=("bar_index", "first"),
    )
    v1_day = v1.groupby(["day", "origin"], as_index=False).agg(
        v1_pnl=("pnl", "sum"),
        v1_n_trades=("pnl", "count"),
        v1_direction=("direction", "first"),
        v1_bar=("bar_index", "first"),
        fold_idx=("fold_idx", "first"),
    )

    # Outer join: keep all V1 days (since V1 = V0 chose-or-abstain forced put)
    df = v1_day.merge(v0_day, on=["day", "origin"], how="outer")
    print(f"  Joined day-level table: {len(df)} day-rows "
          f"({(df['origin']=='in_sample').sum()} in-sample + "
          f"{(df['origin']=='oos').sum()} OOS)", flush=True)

    # Compute label
    df["v0_pnl"] = df["v0_pnl"].fillna(0.0)
    df["v1_pnl"] = df["v1_pnl"].fillna(0.0)
    df["pnl_delta"] = df["v0_pnl"] - df["v1_pnl"]
    df["label"] = (df["pnl_delta"] > args.label_smoothing_usd).astype(int)
    df["v0_chose"] = df["v0_n_trades"].fillna(0).astype(int) > 0
    df["v1_chose"] = df["v1_n_trades"].fillna(0).astype(int) > 0

    # Extract per-day features at bar_feature
    print(flush=True)
    print(f"Extracting per-day features at bar {args.feature_bar}...", flush=True)
    feat_rows = []
    for _, row in df.iterrows():
        feats = extract_day_features(
            ds, str(row["day"]), args.feature_bar, feature_indices, PER_DAY_FEATURE_NAMES,
        )
        feats["day"] = str(row["day"])
        feats["origin"] = row["origin"]
        feat_rows.append(feats)
    feat_df = pd.DataFrame(feat_rows)
    df = df.merge(feat_df, on=["day", "origin"], how="left")

    # Drop days with NaN features (rare)
    n_before = len(df)
    df = df.dropna(subset=PER_DAY_FEATURE_NAMES)
    n_after = len(df)
    if n_after < n_before:
        print(f"  Dropped {n_before - n_after} days with missing features", flush=True)

    # === Summary ===
    print(flush=True)
    print("=" * 100)
    print("Phase 3A label + features summary")
    print("=" * 100, flush=True)
    print(f"  Total day-rows:       {len(df)}", flush=True)
    print(f"  In-sample:            {(df['origin']=='in_sample').sum()}", flush=True)
    print(f"  OOS:                  {(df['origin']=='oos').sum()}", flush=True)
    print(flush=True)
    print(f"  Label distribution (V0-favorable = 1):", flush=True)
    is_df = df[df["origin"] == "in_sample"]
    oos_df = df[df["origin"] == "oos"]
    print(f"    In-sample: {is_df['label'].sum()}/{len(is_df)} = {is_df['label'].mean():.3f}", flush=True)
    print(f"    OOS:       {oos_df['label'].sum()}/{len(oos_df)} = {oos_df['label'].mean():.3f}", flush=True)

    # Where V0 vs V1 disagree (the days that matter for switching)
    print(flush=True)
    print(f"  Where V0 chose to trade AND label=1 (V0-favorable):", flush=True)
    v0_chose_df = df[df["v0_chose"]]
    v0_chose_is = v0_chose_df[v0_chose_df["origin"] == "in_sample"]
    v0_chose_oos = v0_chose_df[v0_chose_df["origin"] == "oos"]
    print(f"    In-sample: {v0_chose_is['label'].sum()}/{len(v0_chose_is)} = "
          f"{v0_chose_is['label'].mean():.3f}", flush=True)
    print(f"    OOS:       {v0_chose_oos['label'].sum()}/{len(v0_chose_oos)} = "
          f"{v0_chose_oos['label'].mean():.3f}", flush=True)

    # Per-fold label balance
    print(flush=True)
    print(f"  Per-fold label balance (in-sample only):", flush=True)
    print(f"    {'fold':<6}{'days':>8}{'V0_pos':>10}{'rate':>10}", flush=True)
    for fi in sorted(is_df["fold_idx"].dropna().unique()):
        sub = is_df[is_df["fold_idx"] == fi]
        print(f"    {int(fi):<6}{len(sub):>8}{sub['label'].sum():>10}{sub['label'].mean():>10.3f}", flush=True)

    # Save
    out_csv = os.path.join(args.out_dir, "regime_labels.csv")
    df.to_csv(out_csv, index=False)
    print(flush=True)
    print(f"Saved: {out_csv}", flush=True)

    # JSON summary for downstream consumption
    summary = {
        "meta": {
            "label_smoothing_usd": float(args.label_smoothing_usd),
            "feature_bar": int(args.feature_bar),
            "per_day_features": PER_DAY_FEATURE_NAMES,
            "feature_indices": feature_indices,
            "n_total": int(len(df)),
            "n_in_sample": int((df["origin"] == "in_sample").sum()),
            "n_oos": int((df["origin"] == "oos").sum()),
        },
        "label_balance": {
            "in_sample_pos": int(is_df["label"].sum()),
            "in_sample_total": int(len(is_df)),
            "in_sample_pos_rate": float(is_df["label"].mean()) if len(is_df) else 0.0,
            "oos_pos": int(oos_df["label"].sum()),
            "oos_total": int(len(oos_df)),
            "oos_pos_rate": float(oos_df["label"].mean()) if len(oos_df) else 0.0,
        },
        "v0_chose_label_balance": {
            "in_sample_pos": int(v0_chose_is["label"].sum()),
            "in_sample_total": int(len(v0_chose_is)),
            "in_sample_pos_rate": float(v0_chose_is["label"].mean()) if len(v0_chose_is) else 0.0,
            "oos_pos": int(v0_chose_oos["label"].sum()),
            "oos_total": int(len(v0_chose_oos)),
            "oos_pos_rate": float(v0_chose_oos["label"].mean()) if len(v0_chose_oos) else 0.0,
        },
        "per_fold_label_balance": {
            int(fi): {
                "n_days": int((is_df["fold_idx"] == fi).sum()),
                "n_pos": int(is_df[is_df["fold_idx"] == fi]["label"].sum()),
                "pos_rate": float(is_df[is_df["fold_idx"] == fi]["label"].mean())
                            if (is_df["fold_idx"] == fi).sum() > 0 else 0.0,
            }
            for fi in sorted(is_df["fold_idx"].dropna().unique())
        },
    }
    summary_path = os.path.join(args.out_dir, "regime_labels_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            summary, f, indent=2, sort_keys=True,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )
    print(f"Saved: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
