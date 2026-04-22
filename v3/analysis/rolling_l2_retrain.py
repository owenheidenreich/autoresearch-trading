"""Phase R3 — Retrain Layer-2 per rolling window with augmented features.

For each of 13 rolling windows:
  1. Build augmented feature matrix: existing 42 L2 features + 9 new
     intraday-developing features (6 X_sim passthrough + 3 derived)
  2. Train entry + side HistGradientBoostingRegressor on train_days
  3. Calibrate per-window thresholds on val_days (last 40 train days)
  4. Predict on oos_days
  5. Save per-window: models + calibration + oof predictions

Output: v3/artifacts/rolling_l2/window_<idx>/{entry_model.pkl,
side_model.pkl, calibration.json, oos_predictions.pkl}
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from v3.analysis.intraday_feature_audit import (
    CATEGORY_A_ADDITIONS,
    CATEGORY_B_SPECS,
)
from v3.harness.rolling_windows import (
    generate_rolling_windows,
    verify_windows,
    print_window_summary,
)
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    calibrate_thresholds,
    effective_direction,
    load_export_bundle,
    save_json,
    selected_value_for_direction_mode,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "rolling_l2")
DEFAULT_SEED = 42

CAT_B_FEATURE_NAMES = tuple(s["name"] for s in CATEGORY_B_SPECS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--entry-target", default="time_stop_value_rank")
    p.add_argument("--side-target", default="time_stop_margin_raw")
    p.add_argument("--direction-mode", default="teacher_if_triggered_else_put")
    p.add_argument("--score-mode", default="product")
    p.add_argument("--side-score-weight", type=float, default=0.15)
    p.add_argument("--calibration-mode", default="search_mean_time_stop")
    p.add_argument("--entry-quantile", type=float, default=0.60)
    p.add_argument("--side-quantile", type=float, default=0.10)
    return p.parse_args()


def _hist_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_depth=3,
        min_samples_leaf=50,
        random_state=seed,
    )


def compute_augmented_features_per_day(
    ds: V2Dataset, day: str, feature_name_to_idx: dict[str, int],
    close_by_bar: dict[int, float] | None = None,
) -> pd.DataFrame:
    """For one day, compute all augmented features (Cat A + Cat B) per bar.

    close_by_bar: optional {bar_index: underlying_close} from L2 bundle.
    If None, falls back to using X_sim vwap_dist + reconstruction.
    """
    start, end = ds.day_bar_range(day)
    n = end - start
    X_day = ds.X_sim[start:end]
    bar_of_day = np.asarray([int(ds.bar_of_day[i]) for i in range(start, end)])

    out = {"day": [day] * n, "bar_index": bar_of_day.tolist()}

    # Category A: direct passthrough
    for name in CATEGORY_A_ADDITIONS:
        col = feature_name_to_idx[name]
        out[name] = X_day[:, col].astype(np.float64)

    # Category B: derived
    vwap_dist_col = feature_name_to_idx["vwap_dist"]
    cum_delta_col = feature_name_to_idx["session_cum_delta"]
    vwap_dist = X_day[:, vwap_dist_col]
    cum_delta = X_day[:, cum_delta_col]

    # Close prices from L2 bundle if available, else reconstruction proxy
    if close_by_bar is not None:
        close = np.array([close_by_bar.get(int(b), np.nan) for b in bar_of_day])
        if np.isnan(close).any():
            mask = ~np.isnan(close)
            if mask.sum() == 0:
                close = np.cumsum(vwap_dist - vwap_dist.mean())
            else:
                close = np.where(mask, close, np.nanmean(close[mask]))
    else:
        close = np.cumsum(vwap_dist - vwap_dist.mean())

    # vwap_dist_last10_mean
    vwap_dist_l10 = np.zeros(n)
    cumsum = np.cumsum(np.concatenate([[0.0], vwap_dist]))
    for i in range(n):
        lo = max(0, i - 10)
        count = i - lo + 1
        vwap_dist_l10[i] = (cumsum[i + 1] - cumsum[lo]) / count

    # cum_delta_slope_30 via linregress
    cum_delta_slope = np.zeros(n)
    for i in range(n):
        lo = max(0, i - 30)
        y = cum_delta[lo:i + 1]
        if len(y) < 3:
            cum_delta_slope[i] = 0.0
            continue
        x = np.arange(len(y), dtype=np.float64)
        x_mean = x.mean()
        y_mean = float(y.mean())
        num = float(((x - x_mean) * (y - y_mean)).sum())
        den = float(((x - x_mean) ** 2).sum())
        cum_delta_slope[i] = num / den if den > 0 else 0.0

    # high_low_skew_since_open: (close - midpoint) / (span/2)
    hi_lo_skew = np.zeros(n)
    running_hi = close[0]
    running_lo = close[0]
    for i in range(n):
        running_hi = max(running_hi, close[i])
        running_lo = min(running_lo, close[i])
        span = running_hi - running_lo
        if span < 1e-9:
            hi_lo_skew[i] = 0.0
        else:
            midpoint = (running_hi + running_lo) / 2.0
            hi_lo_skew[i] = (close[i] - midpoint) / (span / 2.0)

    out["vwap_dist_last10_mean"] = vwap_dist_l10.astype(np.float64)
    out["cum_delta_slope_30"] = cum_delta_slope.astype(np.float64)
    out["high_low_skew_since_open"] = hi_lo_skew.astype(np.float64)

    return pd.DataFrame(out)


def build_augmented_dataframe(
    df_existing: pd.DataFrame, ds: V2Dataset,
) -> pd.DataFrame:
    """Compute the 9 new features for every (day, bar_index) in df_existing
    and merge. Vectorized per day."""
    feature_name_to_idx = {n: i for i, n in enumerate(ds.feature_names)}
    # Build close-by-bar lookup per day from L2 bundle's underlying_close column
    close_lookups_by_day: dict[str, dict[int, float]] = {}
    for day, sub in df_existing.groupby("day"):
        close_lookups_by_day[str(day)] = {
            int(b): float(c) for b, c in zip(sub["bar_index"], sub["underlying_close"])
        }
    all_days = sorted(df_existing["day"].unique().tolist())
    aug_frames = []
    for day in all_days:
        aug = compute_augmented_features_per_day(
            ds, day, feature_name_to_idx, close_by_bar=close_lookups_by_day.get(str(day)),
        )
        aug_frames.append(aug)
    aug_df = pd.concat(aug_frames, ignore_index=True)
    merged = df_existing.merge(aug_df, on=["day", "bar_index"], how="left")
    return merged


def _prepare_predictions(
    df: pd.DataFrame, entry_pred: np.ndarray, side_pred: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    out["entry_score"] = entry_pred
    out["side_score"] = side_pred
    out["side_conf"] = np.abs(side_pred)
    out["predicted_direction"] = np.where(side_pred > 0, "call", "put")
    both_available = (out["has_passing_call"] > 0.5) & (out["has_passing_put"] > 0.5)
    only_call = (out["has_passing_call"] > 0.5) & (out["has_passing_put"] <= 0.5)
    only_put = (out["has_passing_put"] > 0.5) & (out["has_passing_call"] <= 0.5)
    out.loc[~both_available, "side_score"] = 0.0
    out.loc[only_call, "predicted_direction"] = "call"
    out.loc[only_put, "predicted_direction"] = "put"
    out.loc[only_call | only_put, "side_conf"] = 1.0
    out.loc[(out["has_passing_call"] <= 0.5) & (out["has_passing_put"] <= 0.5),
            "predicted_direction"] = ""
    return out


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset + L2 bundle...", flush=True)
    ds = V2Dataset.load()
    bundle = load_export_bundle(args.dataset)
    df_existing: pd.DataFrame = bundle["rows"]
    orig_feature_names = list(bundle["meta"]["feature_names"])
    print(f"  Rows: {len(df_existing)}; original L2 features: {len(orig_feature_names)}", flush=True)

    # === Build augmented feature dataframe ===
    print(flush=True)
    print(f"Augmenting with {len(CATEGORY_A_ADDITIONS)} Cat A + {len(CAT_B_FEATURE_NAMES)} Cat B = {len(CATEGORY_A_ADDITIONS) + len(CAT_B_FEATURE_NAMES)} new features...",
          flush=True)
    t0 = time.time()
    df = build_augmented_dataframe(df_existing, ds)
    t1 = time.time()
    print(f"  Augmented {len(df)} rows in {t1 - t0:.1f}s", flush=True)

    augmented_feature_names = (
        list(orig_feature_names)
        + list(CATEGORY_A_ADDITIONS)
        + list(CAT_B_FEATURE_NAMES)
    )
    # Verify all feature columns exist and have no NaNs (intermediate check)
    for fn in augmented_feature_names:
        assert fn in df.columns, f"missing feature column {fn}"
    na_counts = df[augmented_feature_names].isna().sum()
    if na_counts.sum() > 0:
        print(f"  WARNING: NaN in {int((na_counts > 0).sum())} feature columns", flush=True)
        print(f"    worst: {na_counts.sort_values(ascending=False).head()}", flush=True)

    print(f"  Total features for training: {len(augmented_feature_names)}", flush=True)

    # === Generate rolling windows ===
    print(flush=True)
    all_days = sorted(set(ds.dates))
    windows = generate_rolling_windows(all_days)
    verify_windows(windows)
    print_window_summary(windows)

    # === Per-window training ===
    print(flush=True)
    print("=" * 100)
    print("Per-window training + calibration + OOS prediction")
    print("=" * 100, flush=True)
    print(f"{'idx':<5}{'train_n':>10}{'val_n':>8}{'oos_n':>7}{'seconds':>10}"
          f"{'entry_thr':>12}{'side_thr':>12}", flush=True)

    per_window_results: list[dict[str, Any]] = []
    total_t0 = time.time()

    for w in windows:
        window_t0 = time.time()
        window_dir = os.path.join(args.out_dir, f"window_{w.window_idx:02d}")
        os.makedirs(window_dir, exist_ok=True)

        train_df = df[df["day"].isin(w.train_days)].copy()
        val_df = df[df["day"].isin(w.val_days)].copy()
        oos_df = df[df["day"].isin(w.oos_days)].copy()

        X_train = train_df[augmented_feature_names].to_numpy(dtype=np.float32)
        X_val = val_df[augmented_feature_names].to_numpy(dtype=np.float32)
        X_oos = oos_df[augmented_feature_names].to_numpy(dtype=np.float32)

        # Entry model
        entry_train_mask = train_df[args.entry_target].notna().to_numpy()
        if not entry_train_mask.any():
            print(f"  window {w.window_idx}: no entry labels; skipping", flush=True)
            continue
        y_entry = train_df.loc[entry_train_mask, args.entry_target].to_numpy(dtype=np.float32)
        entry_weight = train_df.loc[entry_train_mask, "time_stop_value_rank"].fillna(0.5).to_numpy(dtype=np.float32)
        entry_weight = 0.5 + entry_weight
        entry_model = _hist_regressor(args.seed + w.window_idx)
        entry_model.fit(X_train[entry_train_mask], y_entry, sample_weight=entry_weight)

        # Side model (requires both call/put passable)
        side_train_mask = (
            train_df[args.side_target].notna().to_numpy()
            & (train_df["has_passing_call"].to_numpy(dtype=float) > 0.5)
            & (train_df["has_passing_put"].to_numpy(dtype=float) > 0.5)
        )
        if not side_train_mask.any():
            print(f"  window {w.window_idx}: no side labels; skipping", flush=True)
            continue
        side_model = _hist_regressor(args.seed + w.window_idx + 500)
        side_weight = np.abs(train_df.loc[side_train_mask, args.side_target].to_numpy(dtype=np.float32))
        side_weight = np.clip(
            side_weight, 1.0,
            np.percentile(side_weight, 95) if len(side_weight) > 5 else np.max(side_weight),
        )
        y_side = np.arcsinh(
            train_df.loc[side_train_mask, args.side_target].to_numpy(dtype=np.float32) / 100.0
        )
        side_model.fit(X_train[side_train_mask], y_side, sample_weight=side_weight)

        # Predict + calibrate on val
        val_pred = _prepare_predictions(
            val_df, entry_model.predict(X_val), side_model.predict(X_val),
        )
        thresholds = calibrate_thresholds(
            val_pred,
            direction_mode=args.direction_mode,
            calibration_mode=args.calibration_mode,
            score_mode=args.score_mode,
            side_score_weight=args.side_score_weight,
            entry_quantile=args.entry_quantile,
            side_quantile=args.side_quantile,
        )

        # Predict on OOS
        oos_pred = _prepare_predictions(
            oos_df, entry_model.predict(X_oos), side_model.predict(X_oos),
        )
        oos_pred["effective_direction"] = oos_pred.apply(
            effective_direction, axis=1, direction_mode=args.direction_mode,
        )
        oos_pred["window_idx"] = w.window_idx
        oos_pred["selected_time_stop_value"] = oos_pred.apply(
            selected_value_for_direction_mode,
            axis=1,
            direction_mode=args.direction_mode,
            call_col="time_stop_pnl_call",
            put_col="time_stop_pnl_put",
        )

        # Save
        with open(os.path.join(window_dir, "entry_model.pkl"), "wb") as f:
            pickle.dump(entry_model, f)
        with open(os.path.join(window_dir, "side_model.pkl"), "wb") as f:
            pickle.dump(side_model, f)
        save_json(os.path.join(window_dir, "calibration.json"), thresholds)
        with open(os.path.join(window_dir, "oos_predictions.pkl"), "wb") as f:
            pickle.dump(oos_pred, f)

        window_t = time.time() - window_t0
        print(
            f"{w.window_idx:<5}{len(train_df):>10}{len(val_df):>8}{len(oos_df):>7}"
            f"{window_t:>10.1f}{thresholds.get('entry_threshold', 0):>12.4f}"
            f"{thresholds.get('side_threshold', 0):>12.4f}",
            flush=True,
        )

        per_window_results.append({
            "window_idx": w.window_idx,
            "window_id": w.window_id,
            "train_n": w.train_n, "val_n": w.val_n, "oos_n": w.oos_n,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "oos_rows": int(len(oos_df)),
            "seconds": window_t,
            "thresholds": thresholds,
            "train_start": w.train_days[0], "train_end": w.train_days[-1],
            "oos_start": w.oos_days[0], "oos_end": w.oos_days[-1],
        })

    total_t = time.time() - total_t0
    print(flush=True)
    print(f"All {len(per_window_results)} windows trained in {total_t / 60:.1f} minutes",
          flush=True)

    # === Save aggregated manifest ===
    manifest = {
        "dataset": args.dataset,
        "n_windows": len(per_window_results),
        "augmented_feature_names": augmented_feature_names,
        "n_features": len(augmented_feature_names),
        "direction_mode": args.direction_mode,
        "score_mode": args.score_mode,
        "side_score_weight": float(args.side_score_weight),
        "calibration_mode": args.calibration_mode,
        "entry_target": args.entry_target,
        "side_target": args.side_target,
        "windows": per_window_results,
        "total_seconds": total_t,
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    save_json(manifest_path, manifest)
    print(f"Saved: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
