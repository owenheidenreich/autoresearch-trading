"""Dataset construction pipeline for v2.

Loads features and option prices from v1's data.pt, computes oracle labels
using v2/core/labels.py, and produces a v2-format dataset.

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
from v2.core.labels import compute_oracle_labels, labels_to_tensors, label_quality_report


V1_DATA_PATH = os.path.join("training", "data.pt")
V2_DATA_PATH = os.path.join("v2", "data.pt")

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

    # SPX close prices (reconstruct from features if not stored directly)
    # Use atm_strikes as proxy for SPX spot
    spot_prices = d['atm_strikes'].numpy().astype(np.float32)

    # Option prices
    option_prices = {}
    for key in OPTION_PRICE_KEYS:
        if key in d:
            option_prices[key] = d[key].numpy().astype(np.float32)

    # Train/val split info
    train_end_idx = int(d.get('train_end_idx', len(dates) - 1))
    val_start_idx = int(d.get('val_start_idx', train_end_idx + 1))

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
    print(f"  Train end: {train_end_idx} ({dates[train_end_idx]})")
    print(f"  Val start: {val_start_idx} ({dates[val_start_idx]})")

    return {
        'features': features,
        'dates': dates,
        'valid': valid,
        'bar_of_day': bar_of_day,
        'spot_prices': spot_prices,
        'option_prices': option_prices,
        'train_end_idx': train_end_idx,
        'val_start_idx': val_start_idx,
        'aux_labels': aux_labels,
        'num_days': num_days,
    }


def build_dataset(
    v1_data: dict,
    tier: int = 1,
    output_path: str = V2_DATA_PATH,
) -> dict:
    """Build v2 dataset with oracle labels.

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
    train_end_idx = v1_data['train_end_idx']
    val_start_idx = v1_data['val_start_idx']
    N = len(dates)

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

    # Build train/val masks
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    train_mask[:train_end_idx + 1] = True
    val_mask[val_start_idx:] = True

    # Fingerprint
    fp_data = f"features:{features.shape}:tier:{tier}:dates:{dates[0]}:{dates[-1]}"
    fingerprint = hashlib.sha256(fp_data.encode()).hexdigest()[:16]

    # Build output dict
    dataset = {
        # Features (from v1, identical)
        'X': torch.from_numpy(features),
        'feature_names': list(features_module_names()),

        # Oracle labels
        **{k: torch.from_numpy(v) for k, v in label_tensors.items()},

        # Auxiliary prediction labels
        **{k: torch.from_numpy(v) for k, v in v1_data['aux_labels'].items()},

        # Metadata
        'dates': dates,
        'bar_of_day': torch.from_numpy(bar_of_day),
        'train_mask': torch.from_numpy(train_mask),
        'val_mask': torch.from_numpy(val_mask),

        # Option prices (needed for replay)
        **{k: torch.from_numpy(v) for k, v in option_prices.items()},
        'spot_prices': torch.from_numpy(spot_prices),

        # Provenance
        'metadata': {
            'feature_version': 'v2.0',
            'label_version': f'oracle_tier{tier}',
            'data_dates': [dates[0], dates[-1]],
            'num_bars': N,
            'num_train_bars': int(train_mask.sum()),
            'num_val_bars': int(val_mask.sum()),
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

    return dataset


def features_module_names() -> list[str]:
    """Import-free access to feature names."""
    from v2.core.features import FEATURE_NAMES
    return FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Build v2 dataset")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="Oracle labeling tier (1=fast, 2=medium, 3=full)")
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
