"""Phase R2 — Intraday-developing feature audit + engineering.

Two parts:

**R2a — Audit.** L2's 42 input features vs V2Dataset X_sim's 89
features. Identifies X_sim features that carry intraday-developing
directional signal but are NOT currently in L2's input vector.

**R2b — New derived features.** Defines entry-bar-relative derived
features that capture intraday *development* — specifically cross-bar
aggregates that can't be read from a single X_sim row.

Output: v3/artifacts/intraday_feature_audit/audit.json with
  - full L2 ↔ X_sim mapping
  - list of Category A (X_sim passthrough) adds
  - list of Category B (derived) adds + definitions
  - spot-check values on 3 sample days
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    W2A_FEATURE_NAMES,
    load_export_bundle,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "intraday_feature_audit")


CATEGORY_A_ADDITIONS = (
    "session_cum_delta",
    "vwap_dist",
    "trend_5min",
    "force_index_2",
    "ret_6",
    "ret_12",
)


CATEGORY_B_SPECS = [
    {
        "name": "vwap_dist_last10_mean",
        "definition": "mean(vwap_dist[bar-10:bar+1])",
        "rationale": "Smoothed recent VWAP position; reduces tick noise vs raw vwap_dist.",
        "lookback": 10,
    },
    {
        "name": "cum_delta_slope_30",
        "definition": "linear regression slope of session_cum_delta[bar-30:bar+1]",
        "rationale": "Is buying/selling accelerating? A directional development signal.",
        "lookback": 30,
    },
    {
        "name": "high_low_skew_since_open",
        "definition": "(close - session_high_so_far) / (session_high_so_far - session_low_so_far)",
        "rationale": "Where is price within the developing session range? Near high = bullish, near low = bearish.",
        "lookback": "session",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args()


def compute_cat_b_features(
    ds: V2Dataset, day: str, feature_name_to_idx: dict[str, int],
) -> pd.DataFrame:
    """Compute Category B features for every bar of the given day."""
    start, end = ds.day_bar_range(day)
    n = end - start
    bars = list(range(n))

    # Read per-bar values for needed X_sim features
    vwap_dist_col = feature_name_to_idx["vwap_dist"]
    cum_delta_col = feature_name_to_idx["session_cum_delta"]
    close_col = feature_name_to_idx.get("underlying_close",
                feature_name_to_idx.get("ret_6"))  # fallback

    X_day = ds.X_sim[start:end]

    vwap_dist = X_day[:, vwap_dist_col]
    cum_delta = X_day[:, cum_delta_col]

    # vwap_dist_last10_mean
    vwap_dist_l10 = np.zeros(n)
    for i in range(n):
        lo = max(0, i - 10)
        vwap_dist_l10[i] = float(vwap_dist[lo:i+1].mean())

    # cum_delta_slope_30 (linear regression slope over last 30 bars)
    cum_delta_slope = np.zeros(n)
    for i in range(n):
        lo = max(0, i - 30)
        y = cum_delta[lo:i+1]
        if len(y) < 3:
            cum_delta_slope[i] = 0.0
            continue
        x = np.arange(len(y), dtype=np.float64)
        x_mean = x.mean()
        y_mean = float(y.mean())
        num = float(((x - x_mean) * (y - y_mean)).sum())
        den = float(((x - x_mean) ** 2).sum())
        cum_delta_slope[i] = num / den if den > 0 else 0.0

    # high_low_skew_since_open — use underlying close values
    underlying_col = feature_name_to_idx.get("underlying_close")
    if underlying_col is not None:
        close = X_day[:, underlying_col]
    else:
        # Fallback: approximate via sign of vwap_dist swings (not ideal)
        close = np.cumsum(vwap_dist - vwap_dist.mean())
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
            hi_lo_skew[i] = (close[i] - running_hi) / span  # -1 = at hi, 0 = at hi, range [-1, 0] actually
    # Normalize to [-1, 1]: -1 = at low, +1 = at high
    hi_lo_skew_norm = np.zeros(n)
    running_hi = close[0]
    running_lo = close[0]
    for i in range(n):
        running_hi = max(running_hi, close[i])
        running_lo = min(running_lo, close[i])
        span = running_hi - running_lo
        if span < 1e-9:
            hi_lo_skew_norm[i] = 0.0
        else:
            midpoint = (running_hi + running_lo) / 2.0
            hi_lo_skew_norm[i] = (close[i] - midpoint) / (span / 2.0)

    return pd.DataFrame({
        "bar": bars,
        "vwap_dist_last10_mean": vwap_dist_l10,
        "cum_delta_slope_30": cum_delta_slope,
        "high_low_skew_since_open": hi_lo_skew_norm,
    })


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset + L2 bundle...", flush=True)
    ds = V2Dataset.load()
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    l2_feature_names = list(bundle["meta"]["feature_names"])
    xsim_feature_names = list(ds.feature_names)
    print(f"  L2: {len(l2_feature_names)} features", flush=True)
    print(f"  X_sim: {len(xsim_feature_names)} features", flush=True)
    print(f"  Current W2A adds to L2: {W2A_FEATURE_NAMES}", flush=True)

    # === R2a — Audit ===
    print(flush=True)
    print("=" * 100)
    print("R2a — Audit: X_sim features NOT in L2's input")
    print("=" * 100, flush=True)
    l2_set = set(l2_feature_names)
    w2a_set = set(W2A_FEATURE_NAMES)
    audit_rows = []
    for i, xname in enumerate(xsim_feature_names):
        in_l2 = xname in l2_set
        in_w2a = xname in w2a_set
        audit_rows.append({
            "xsim_idx": i,
            "xsim_name": xname,
            "in_l2": in_l2,
            "in_w2a": in_w2a,
            "category": "in_l2" if in_l2 else "x_sim_only",
        })
    audit_df = pd.DataFrame(audit_rows)
    missing_from_l2 = audit_df[~audit_df["in_l2"]]
    print(f"  {len(missing_from_l2)} of {len(xsim_feature_names)} X_sim features are NOT in L2",
          flush=True)

    print(flush=True)
    print("  Proposed Category A additions (X_sim passthrough):", flush=True)
    cat_a_status = []
    for name in CATEGORY_A_ADDITIONS:
        in_xsim = name in xsim_feature_names
        in_l2 = name in l2_set
        if in_xsim and not in_l2:
            status = "ADD (in X_sim, not in L2)"
        elif in_xsim and in_l2:
            status = "SKIP (already in L2)"
        else:
            status = "MISSING (not in X_sim!)"
        print(f"    {name:<30} {status}", flush=True)
        cat_a_status.append({
            "name": name, "in_xsim": in_xsim, "in_l2": in_l2, "status": status,
        })

    # === R2b — New derived features ===
    print(flush=True)
    print("=" * 100)
    print("R2b — New derived entry-bar-relative features")
    print("=" * 100, flush=True)

    feature_name_to_idx = {n: i for i, n in enumerate(xsim_feature_names)}
    for spec in CATEGORY_B_SPECS:
        print(f"  {spec['name']:<30} {spec['definition']}", flush=True)
        print(f"    rationale: {spec['rationale']}", flush=True)

    print(flush=True)
    print("  Spot-check on 3 sample days (first, middle, last OOS chronologically):", flush=True)

    # Pick 3 sample days from 2026-03 (our OOS window)
    sample_days = ["2026-03-05", "2026-03-18", "2026-04-01"]
    spot_checks = {}
    for day in sample_days:
        try:
            feats = compute_cat_b_features(ds, day, feature_name_to_idx)
        except Exception as e:
            print(f"    {day}: FAILED: {e}", flush=True)
            spot_checks[day] = {"error": str(e)}
            continue
        # Sample at bars 30, 60, 120 for diagnostic
        samples = {}
        for bar in (30, 60, 120):
            if bar < len(feats):
                row = feats.iloc[bar]
                samples[f"bar_{bar}"] = {
                    "vwap_dist_last10_mean": float(row["vwap_dist_last10_mean"]),
                    "cum_delta_slope_30": float(row["cum_delta_slope_30"]),
                    "high_low_skew_since_open": float(row["high_low_skew_since_open"]),
                }
        spot_checks[day] = samples
        print(f"    {day}:", flush=True)
        for bar_key, vals in samples.items():
            print(f"      {bar_key}: "
                  f"vwap_dist_l10={vals['vwap_dist_last10_mean']:+.4f}  "
                  f"cum_delta_slope={vals['cum_delta_slope_30']:+.4f}  "
                  f"hl_skew={vals['high_low_skew_since_open']:+.4f}",
                  flush=True)

    # === NaN check on full 986 days ===
    print(flush=True)
    print("  Full-dataset NaN check...", flush=True)
    all_days = sorted(set(ds.dates))
    nan_counts = {s["name"]: 0 for s in CATEGORY_B_SPECS}
    total_bars = 0
    for day in all_days:
        try:
            feats = compute_cat_b_features(ds, day, feature_name_to_idx)
        except Exception:
            continue
        total_bars += len(feats)
        for name in nan_counts:
            nan_counts[name] += int(feats[name].isna().sum())
    for name, cnt in nan_counts.items():
        pct = 100.0 * cnt / max(total_bars, 1)
        print(f"    {name:<30} NaN: {cnt}/{total_bars} ({pct:.2f}%)", flush=True)

    # === Save ===
    payload = {
        "meta": {
            "n_l2_features": len(l2_feature_names),
            "n_xsim_features": len(xsim_feature_names),
            "current_w2a_features": list(W2A_FEATURE_NAMES),
        },
        "audit_summary": {
            "missing_from_l2_count": int(len(missing_from_l2)),
            "missing_from_l2_names": missing_from_l2["xsim_name"].tolist(),
        },
        "category_a_additions": cat_a_status,
        "category_b_specs": CATEGORY_B_SPECS,
        "spot_check_values": spot_checks,
        "nan_counts_full_dataset": {
            n: {"nan_count": int(c), "total_bars": int(total_bars),
                "nan_pct": 100.0 * c / max(total_bars, 1)}
            for n, c in nan_counts.items()
        },
    }
    out = os.path.join(args.out_dir, "audit.json")
    with open(out, "w") as f:
        json.dump(
            payload, f, indent=2, sort_keys=True,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )
    audit_df.to_csv(os.path.join(args.out_dir, "l2_xsim_mapping.csv"), index=False)
    print(flush=True)
    print(f"Saved: {out}", flush=True)
    print(f"Saved: {os.path.join(args.out_dir, 'l2_xsim_mapping.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
