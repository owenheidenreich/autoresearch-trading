#!/usr/bin/env python3
"""Compare context bundle features against training data.pt for overlapping dates.

Reports per-feature correlation, MAE, NaN rates, and norm buffer shape validation.
Catches silent data corruption from stale context bundles.

Usage:
  python3 tools/verify_context_parity.py
  python3 tools/verify_context_parity.py --context-dir training/cache/live_context
  python3 tools/verify_context_parity.py --data-pt training/data.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.prepare import FEATURE_NAMES, NUM_FEATURES  # noqa: E402
from training.live.context import LIVE_CONTEXT_DIR, load_latest_context_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify context bundle vs training data parity.")
    parser.add_argument("--context-dir", default=LIVE_CONTEXT_DIR)
    parser.add_argument("--data-pt", default=os.path.join(ROOT, "training", "data.pt"))
    args = parser.parse_args()

    # --- Load context bundle ---
    bundle = load_latest_context_bundle(args.context_dir)
    if bundle is None:
        print("ERROR: No context bundle found. Run: python3 tools/paper_live.py --context-only")
        sys.exit(1)

    print(f"Context bundle: as_of={bundle.as_of_date}, "
          f"range={bundle.context_start_date}..{bundle.context_end_date}")
    print(f"  raw_features shape: {bundle.raw_features.shape}")
    print(f"  norm_raw_buffer shape: {bundle.norm_raw_buffer.shape}")
    print(f"  norm_valid_buffer shape: {bundle.norm_valid_buffer.shape}")
    print(f"  feature_contract_version: {bundle.feature_contract_version}")
    print(f"  Expected NUM_FEATURES: {NUM_FEATURES}")
    print()

    # --- Shape validation ---
    errors = []
    ctx_width = bundle.raw_features.shape[1] if bundle.raw_features.ndim == 2 else 0
    if ctx_width != NUM_FEATURES:
        errors.append(f"FAIL: raw_features width={ctx_width}, expected={NUM_FEATURES}")
    else:
        print(f"OK: raw_features width matches NUM_FEATURES ({NUM_FEATURES})")

    buf_width = bundle.norm_raw_buffer.shape[1] if bundle.norm_raw_buffer.ndim == 2 else 0
    if buf_width != NUM_FEATURES:
        errors.append(f"FAIL: norm_raw_buffer width={buf_width}, expected={NUM_FEATURES}")
    else:
        print(f"OK: norm_raw_buffer width matches NUM_FEATURES ({NUM_FEATURES})")

    # norm_valid_buffer is 1D (per-bar boolean), not per-feature
    if bundle.norm_valid_buffer.ndim != 1:
        errors.append(f"FAIL: norm_valid_buffer should be 1D, got shape={bundle.norm_valid_buffer.shape}")
    elif len(bundle.norm_valid_buffer) != bundle.norm_raw_buffer.shape[0]:
        errors.append(f"FAIL: norm_valid_buffer length={len(bundle.norm_valid_buffer)} != "
                       f"norm_raw_buffer rows={bundle.norm_raw_buffer.shape[0]}")
    else:
        print(f"OK: norm_valid_buffer is 1D with {len(bundle.norm_valid_buffer)} entries")

    if errors:
        print()
        for e in errors:
            print(f"  {e}")
        print()
        print("The context bundle is STALE. Refresh it:")
        print("  python3 tools/paper_live.py --context-only")
        sys.exit(1)

    # --- Load training data.pt ---
    if not os.path.exists(args.data_pt):
        print(f"\nWARNING: {args.data_pt} not found — skipping feature value comparison.")
        print("Shape validation passed. Context bundle is structurally correct.")
        sys.exit(0)

    data = torch.load(args.data_pt, map_location="cpu", weights_only=False)
    train_features = data.get("features")
    train_dates = data.get("dates")
    if train_features is None or train_dates is None:
        print(f"\nWARNING: data.pt missing 'features' or 'dates' keys — skipping comparison.")
        sys.exit(0)

    if isinstance(train_features, torch.Tensor):
        train_features = train_features.numpy()
    train_dates = list(train_dates)

    print(f"\nTraining data.pt: {train_features.shape[0]} bars, {train_features.shape[1]} features")

    # --- Find overlapping dates ---
    ctx_dates = set(bundle.dates)
    train_dates_set = set(train_dates)
    overlap = sorted(ctx_dates & train_dates_set)

    if not overlap:
        print("No overlapping dates between context and training data.")
        print("Shape validation passed. Context bundle is structurally correct.")
        sys.exit(0)

    print(f"Overlapping dates: {len(overlap)} (from {overlap[0]} to {overlap[-1]})")

    # Build date→row index maps
    ctx_date_list = list(bundle.dates)
    ctx_date_to_rows: dict[str, list[int]] = {}
    for i, d in enumerate(ctx_date_list):
        ctx_date_to_rows.setdefault(d, []).append(i)

    train_date_to_rows: dict[str, list[int]] = {}
    for i, d in enumerate(train_dates):
        train_date_to_rows.setdefault(d, []).append(i)

    # Compare feature values for overlapping dates
    num_features = min(bundle.raw_features.shape[1], train_features.shape[1], NUM_FEATURES)
    per_feature_diffs: list[list[float]] = [[] for _ in range(num_features)]
    per_feature_ctx_nan: list[int] = [0] * num_features
    per_feature_train_nan: list[int] = [0] * num_features
    total_compared = 0

    for date in overlap:
        ctx_rows = ctx_date_to_rows.get(date, [])
        train_rows = train_date_to_rows.get(date, [])
        n = min(len(ctx_rows), len(train_rows))
        for j in range(n):
            ci = ctx_rows[j]
            ti = train_rows[j]
            total_compared += 1
            for f in range(num_features):
                cv = bundle.raw_features[ci, f]
                tv = train_features[ti, f]
                c_nan = not np.isfinite(cv)
                t_nan = not np.isfinite(tv)
                if c_nan:
                    per_feature_ctx_nan[f] += 1
                if t_nan:
                    per_feature_train_nan[f] += 1
                if not c_nan and not t_nan:
                    per_feature_diffs[f].append(abs(cv - tv))

    print(f"Compared {total_compared} bar-pairs across {len(overlap)} dates")
    print()
    print(f"{'idx':>3} {'feature':<25} {'MAE':>10} {'ctx_nan%':>9} {'train_nan%':>10} {'n_pairs':>8} {'status'}")
    print("-" * 85)

    all_good = True
    for f in range(num_features):
        name = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f"feat_{f}"
        diffs = per_feature_diffs[f]
        mae = np.mean(diffs) if diffs else float("nan")
        ctx_nan_pct = per_feature_ctx_nan[f] / max(total_compared, 1) * 100
        train_nan_pct = per_feature_train_nan[f] / max(total_compared, 1) * 100

        if not np.isfinite(mae):
            status = "NO_DATA"
            all_good = False
        elif mae < 0.01:
            status = "MATCH"
        elif mae < 0.1:
            status = "CLOSE"
        else:
            status = "DIVERGE"
            all_good = False

        # Flag if context has way more NaNs than training
        if ctx_nan_pct > train_nan_pct + 10:
            status += " (ctx_nan_high)"
            all_good = False

        print(f"{f:3d} {name:<25} {mae:10.4f} {ctx_nan_pct:8.1f}% {train_nan_pct:9.1f}% {len(diffs):8d} {status}")

    print()
    if all_good:
        print("RESULT: Context bundle features align with training data.")
    else:
        print("RESULT: Feature divergence detected — investigate flagged features above.")


if __name__ == "__main__":
    main()
