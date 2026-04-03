"""Dataset construction pipeline for v2.

Loads features and option prices from v1's data.pt, computes oracle labels
using v2/core/labels.py, and produces a v2-format dataset.

v2 uses a 4-way date-only split:
  - train_mask: all dates except the last 140 trading days
  - val_mask: next 60 trading days (checkpoint selection during training)
  - promote_mask: next 60 trading days (keep/revert scoring, never seen during training)
  - shadow_mask: last 20 trading days (live-readiness eval only)

Usage:
    python -m v2.pipeline.build_dataset [--tier 1|2|3] [--output v2/data.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import torch

from v2.core.features import NUM_FEATURES, BARS_PER_DAY, validate_feature_shape
from v2.core.labels import (
    compute_oracle_labels, labels_to_tensors, label_quality_report,
    TIER1_STOPS, TIER1_TARGETS, TIER1_MAX_HOLDS,
    TIER2_STOPS, TIER2_TARGETS, TIER2_MAX_HOLDS,
    TIER3_STOPS, TIER3_TARGETS, TIER3_MAX_HOLDS,
)
from v2.core.metrics import score_config_fingerprint


V1_DATA_PATH = os.path.join("training", "data.pt")
V2_DATA_PATH = os.path.join("v2", "data.pt")

# Split sizes (in trading days, counted from the end of the dataset)
VAL_DAYS = 60
PROMOTE_DAYS = 60
SHADOW_DAYS = 20
EVAL_DAYS_TOTAL = VAL_DAYS + PROMOTE_DAYS + SHADOW_DAYS  # 140

# Option price keys available in v1 data.pt
OPTION_PRICE_KEYS = [
    'atm_call_prices', 'atm_put_prices',
    'otm5_call_prices', 'otm5_put_prices',
    'otm10_call_prices', 'otm10_put_prices',
    'otm15_call_prices', 'otm15_put_prices',
    'otm20_call_prices', 'otm20_put_prices',
    'otm25_call_prices', 'otm25_put_prices',
    'otm30_call_prices', 'otm30_put_prices',
]


def load_v1_data(path: str = V1_DATA_PATH) -> dict:
    """Load v1 data.pt and extract what v2 needs."""
    print(f"Loading v1 data from {path}...")
    d = torch.load(path, map_location="cpu", weights_only=False)

    features = d['features'].numpy()
    assert validate_feature_shape(features), f"Bad feature shape: {features.shape}"

    dates = d['dates']
    valid = d['valid_mask'].numpy().astype(bool)

    # Build bar_of_day from dates
    bar_of_day = np.zeros(len(dates), dtype=np.int32)
    current_date = None
    bar_count = 0
    for i, date in enumerate(dates):
        if date != current_date:
            current_date = date
            bar_count = 0
        bar_of_day[i] = bar_count
        bar_count += 1

    # SPX close prices (use atm_strikes as proxy for SPX spot)
    spot_prices = d['atm_strikes'].numpy().astype(np.float32)

    # Option prices
    option_prices = {}
    for key in OPTION_PRICE_KEYS:
        if key in d:
            option_prices[key] = d[key].numpy().astype(np.float32)

    # Prediction labels (auxiliary, for market head if used)
    aux_labels = {}
    for key in ['pred_return_15', 'pred_return_30', 'pred_return_60']:
        if key in d:
            aux_labels[key] = d[key].numpy().astype(np.float32)

    # Count days
    unique_dates = sorted(set(dates))
    num_days = len(unique_dates)

    print(f"  Bars: {len(dates):,}")
    print(f"  Days: {num_days}")
    print(f"  Features: {features.shape}")
    print(f"  Option price keys: {len(option_prices)}")

    return {
        'features': features,
        'dates': dates,
        'valid': valid,
        'bar_of_day': bar_of_day,
        'spot_prices': spot_prices,
        'option_prices': option_prices,
        'aux_labels': aux_labels,
        'num_days': num_days,
    }


def compute_4way_split(
    dates: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute 4-way date-only split masks.

    Returns (train_mask, val_mask, promote_mask, shadow_mask, split_info).
    """
    N = len(dates)
    unique_dates = sorted(set(dates))
    num_days = len(unique_dates)

    if num_days < EVAL_DAYS_TOTAL + 60:
        raise ValueError(
            f"Not enough trading days for 4-way split. "
            f"Need at least {EVAL_DAYS_TOTAL + 60}, got {num_days}."
        )

    # Split from the end
    shadow_start_day = num_days - SHADOW_DAYS
    promote_start_day = shadow_start_day - PROMOTE_DAYS
    val_start_day = promote_start_day - VAL_DAYS
    # train: everything before val

    shadow_dates = set(unique_dates[shadow_start_day:])
    promote_dates = set(unique_dates[promote_start_day:shadow_start_day])
    val_dates = set(unique_dates[val_start_day:promote_start_day])
    train_dates = set(unique_dates[:val_start_day])

    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    promote_mask = np.zeros(N, dtype=bool)
    shadow_mask = np.zeros(N, dtype=bool)

    for i, date in enumerate(dates):
        if date in train_dates:
            train_mask[i] = True
        elif date in val_dates:
            val_mask[i] = True
        elif date in promote_dates:
            promote_mask[i] = True
        elif date in shadow_dates:
            shadow_mask[i] = True

    # Verify no overlap and full coverage
    total_assigned = train_mask.sum() + val_mask.sum() + promote_mask.sum() + shadow_mask.sum()
    assert (train_mask & val_mask).sum() == 0, "train/val overlap"
    assert (train_mask & promote_mask).sum() == 0, "train/promote overlap"
    assert (train_mask & shadow_mask).sum() == 0, "train/shadow overlap"
    assert (val_mask & promote_mask).sum() == 0, "val/promote overlap"
    assert (val_mask & shadow_mask).sum() == 0, "val/shadow overlap"
    assert (promote_mask & shadow_mask).sum() == 0, "promote/shadow overlap"
    assert total_assigned == N, f"Unassigned bars: {N - total_assigned}"

    split_info = {
        'train_days': len(train_dates),
        'val_days': VAL_DAYS,
        'promote_days': PROMOTE_DAYS,
        'shadow_days': SHADOW_DAYS,
        'train_date_range': [min(train_dates), max(train_dates)],
        'val_date_range': [min(val_dates), max(val_dates)],
        'promote_date_range': [min(promote_dates), max(promote_dates)],
        'shadow_date_range': [min(shadow_dates), max(shadow_dates)],
        'train_bars': int(train_mask.sum()),
        'val_bars': int(val_mask.sum()),
        'promote_bars': int(promote_mask.sum()),
        'shadow_bars': int(shadow_mask.sum()),
    }

    return train_mask, val_mask, promote_mask, shadow_mask, split_info


def validate_label_diversity(label_tensors: dict, tier: int) -> None:
    """Fail the build if Tier 2/3 positive labels are degenerate.

    Tier 1 is expected to be uniform (fixed risk). Tier 2/3 must show
    diversity across strike, stop, target, and hold.
    """
    if tier == 1:
        return  # Tier 1 is intentionally uniform

    positive = label_tensors['oracle_trade'].astype(bool)
    n_positive = positive.sum()
    if n_positive == 0:
        raise ValueError("No positive labels found. Oracle labeler produced zero trades.")

    checks = {
        'oracle_strike_offset': 3,
        'oracle_stop_pct': 3,
        'oracle_target_pct': 3,
        'oracle_max_hold': 3,
    }

    for key, min_distinct in checks.items():
        values = label_tensors[key][positive]
        n_distinct = len(set(values.tolist()))
        if n_distinct < min_distinct:
            raise ValueError(
                f"Label diversity gate FAILED: {key} has only {n_distinct} "
                f"distinct values in {n_positive} positive labels "
                f"(need >= {min_distinct} for tier {tier}). "
                f"Values found: {sorted(set(values.tolist()))}"
            )
        print(f"  {key}: {n_distinct} distinct values -- OK")


def build_dataset(
    v1_data: dict,
    tier: int = 3,
    output_path: str = V2_DATA_PATH,
) -> dict:
    """Build v2 dataset with oracle labels and 4-way split.

    Args:
        v1_data: output of load_v1_data()
        tier: oracle labeling tier (1=fast, 2=medium, 3=full)
        output_path: where to save the dataset
    """
    features = v1_data['features']
    dates = v1_data['dates']
    bar_of_day = v1_data['bar_of_day']
    spot_prices = v1_data['spot_prices']
    option_prices = v1_data['option_prices']
    N = len(dates)

    # Compute 4-way split
    print("\nComputing 4-way date split...")
    train_mask, val_mask, promote_mask, shadow_mask, split_info = compute_4way_split(dates)
    print(f"  Train: {split_info['train_days']} days, {split_info['train_bars']:,} bars "
          f"({split_info['train_date_range'][0]} to {split_info['train_date_range'][1]})")
    print(f"  Val: {split_info['val_days']} days, {split_info['val_bars']:,} bars "
          f"({split_info['val_date_range'][0]} to {split_info['val_date_range'][1]})")
    print(f"  Promote: {split_info['promote_days']} days, {split_info['promote_bars']:,} bars "
          f"({split_info['promote_date_range'][0]} to {split_info['promote_date_range'][1]})")
    print(f"  Shadow: {split_info['shadow_days']} days, {split_info['shadow_bars']:,} bars "
          f"({split_info['shadow_date_range'][0]} to {split_info['shadow_date_range'][1]})")

    # Compute oracle labels
    print(f"\nComputing oracle labels (tier {tier})...")
    t0 = time.time()
    oracle_labels = compute_oracle_labels(
        features=features,
        option_prices=option_prices,
        dates=dates,
        bar_of_day=bar_of_day,
        spot_prices=spot_prices,
        tier=tier,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({N / elapsed:.0f} bars/sec)")

    # Quality report
    report = label_quality_report(oracle_labels)
    print("\nLabel Quality Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Convert to tensors
    label_tensors = labels_to_tensors(oracle_labels)

    # Validate label diversity (Tier 2/3 must show variation)
    print("\nLabel Diversity Check:")
    validate_label_diversity(label_tensors, tier)

    # Oracle search grid config (for metadata)
    if tier == 1:
        grid_config = {'stops': TIER1_STOPS, 'targets': TIER1_TARGETS, 'holds': TIER1_MAX_HOLDS}
    elif tier == 2:
        grid_config = {'stops': TIER2_STOPS, 'targets': TIER2_TARGETS, 'holds': TIER2_MAX_HOLDS}
    else:
        grid_config = {'stops': TIER3_STOPS, 'targets': TIER3_TARGETS, 'holds': TIER3_MAX_HOLDS}

    # Fingerprint
    fp_data = (
        f"features:{features.shape}:tier:{tier}:"
        f"dates:{dates[0]}:{dates[-1]}:"
        f"split:{split_info['train_days']}/{split_info['val_days']}/"
        f"{split_info['promote_days']}/{split_info['shadow_days']}:"
        f"grid:{grid_config}"
    )
    fingerprint = hashlib.sha256(fp_data.encode()).hexdigest()[:16]

    # Candidate universe description
    candidate_universe = {
        'strikes': 'ATM +/- 30 in 5-point steps',
        'sides': ['call', 'put'],
        'max_qty': 1,
        'option_price_keys': list(option_prices.keys()),
    }

    # Build output dict
    dataset = {
        # Features (from v1, identical)
        'X': torch.from_numpy(features),
        'feature_names': list(features_module_names()),

        # Oracle labels
        **{k: torch.from_numpy(v) for k, v in label_tensors.items()},

        # Auxiliary prediction labels
        **{k: torch.from_numpy(v) for k, v in v1_data['aux_labels'].items()},

        # 4-way split masks
        'dates': dates,
        'bar_of_day': torch.from_numpy(bar_of_day),
        'train_mask': torch.from_numpy(train_mask),
        'val_mask': torch.from_numpy(val_mask),
        'promote_mask': torch.from_numpy(promote_mask),
        'shadow_mask': torch.from_numpy(shadow_mask),

        # Option prices (needed for replay)
        **{k: torch.from_numpy(v) for k, v in option_prices.items()},
        'spot_prices': torch.from_numpy(spot_prices),

        # Provenance
        'metadata': {
            'feature_version': 'v2.0',
            'label_version': f'oracle_tier{tier}',
            'data_dates': [dates[0], dates[-1]],
            'num_bars': N,
            'split': split_info,
            'label_tier': tier,
            'oracle_search_grid': grid_config,
            'evaluator_fingerprint': score_config_fingerprint(),
            'candidate_universe': candidate_universe,
            'build_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'fingerprint': fingerprint,
            'label_quality': report,
        },
    }

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"\nSaving to {output_path}...")
    torch.save(dataset, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Fingerprint: {fingerprint}")

    return dataset


def features_module_names() -> list[str]:
    """Import-free access to feature names."""
    from v2.core.features import FEATURE_NAMES
    return FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Build v2 dataset")
    parser.add_argument("--tier", type=int, default=3, choices=[1, 2, 3],
                        help="Oracle labeling tier (1=fast, 2=medium, 3=full). Default: 3")
    parser.add_argument("--input", type=str, default=V1_DATA_PATH,
                        help="Path to v1 data.pt")
    parser.add_argument("--output", type=str, default=V2_DATA_PATH,
                        help="Output path for v2 dataset")
    args = parser.parse_args()

    v1_data = load_v1_data(args.input)
    build_dataset(v1_data, tier=args.tier, output_path=args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
