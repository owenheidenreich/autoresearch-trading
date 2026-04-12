"""Data integrity validation layer.

Validates the full data chain before training: manifest (data.pt),
sidecars (per-day files), and normalized features. Catches garbage
inputs before they corrupt model training.

Usage:
    python -m v2.core.data_integrity [--data v2/data.pt] [--sample N]
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from v2.core.chain_data import (
    QUALITY_VALID, QUALITY_PARTIAL, QUALITY_CORRUPT,
    contract_snapshot, load_sidecar_cached,
)
from v2.core.features import FEATURE_NAMES, NUM_FEATURES, _NO_NORMALIZE


@dataclass
class DataQualityReport:
    """Result of a validation check."""

    source: str = ""              # what was validated
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.errors:
            self.passed = False

    def print_report(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"  [{status}] {self.source}")
        for e in self.errors:
            print(f"    ERROR: {e}")
        for w in self.warnings:
            print(f"    WARN:  {w}")
        if self.stats:
            for k, v in sorted(self.stats.items()):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ["X", "dates", "bar_of_day", "spot_prices", "metadata"]


def validate_manifest(data: dict) -> DataQualityReport:
    """Validate the data.pt manifest structure and basic sanity."""
    r = DataQualityReport(source="manifest (data.pt)")

    # Required keys
    for key in _REQUIRED_KEYS:
        if key not in data:
            r.errors.append(f"missing required key: {key}")
    if r.errors:
        return r

    X = data["X"]
    n_bars = X.shape[0]
    r.stats["n_bars"] = n_bars

    # Shape checks
    if X.ndim != 2:
        r.errors.append(f"X should be 2D, got {X.ndim}D")
    elif X.shape[1] != NUM_FEATURES:
        r.errors.append(f"X has {X.shape[1]} features, expected {NUM_FEATURES}")

    if "X_sim" in data:
        if data["X_sim"].shape[0] != n_bars:
            r.errors.append(f"X_sim has {data['X_sim'].shape[0]} bars, X has {n_bars}")

    # Dates
    dates = data["dates"]
    if len(dates) != n_bars:
        r.errors.append(f"dates has {len(dates)} entries, X has {n_bars} bars")

    unique_days = sorted(set(dates))
    r.stats["n_unique_days"] = len(unique_days)

    # bar_of_day bounds
    bod = data["bar_of_day"]
    if hasattr(bod, "numpy"):
        bod = bod.numpy()
    bod_min, bod_max = int(bod.min()), int(bod.max())
    if bod_min < 0 or bod_max > 389:
        r.errors.append(f"bar_of_day out of range: [{bod_min}, {bod_max}], expected [0, 389]")

    # spot_prices
    sp = data["spot_prices"]
    if hasattr(sp, "numpy"):
        sp = sp.numpy()
    n_nan = int(np.isnan(sp).sum())
    n_neg = int((sp <= 0).sum())
    if n_nan > 0:
        r.errors.append(f"spot_prices has {n_nan} NaN values")
    if n_neg > 0:
        r.errors.append(f"spot_prices has {n_neg} non-positive values")
    r.stats["spot_min"] = float(np.nanmin(sp))
    r.stats["spot_max"] = float(np.nanmax(sp))

    # Metadata
    meta = data.get("metadata", {})
    if "chain_sidecar_dir" not in meta:
        r.warnings.append("metadata missing chain_sidecar_dir")
    if "max_contracts_per_bar" not in meta:
        r.warnings.append("metadata missing max_contracts_per_bar")
    if "fingerprint" not in meta:
        r.warnings.append("metadata missing fingerprint")

    # Mask keys
    for mask_name in ["train_mask", "val_mask", "promote_mask"]:
        if mask_name in data:
            mask = data[mask_name]
            if hasattr(mask, "numpy"):
                mask = mask.numpy()
            n_true = int(mask.sum())
            r.stats[f"{mask_name}_bars"] = n_true
            if n_true == 0:
                r.warnings.append(f"{mask_name} has zero True values")
        else:
            r.warnings.append(f"missing mask: {mask_name}")

    return r


# ---------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------

def validate_features(X: np.ndarray) -> DataQualityReport:
    """Validate normalized feature matrix for quality issues."""
    r = DataQualityReport(source="feature matrix")

    if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
        r.errors.append(f"unexpected shape {X.shape}, expected (N, {NUM_FEATURES})")
        return r

    n_bars = X.shape[0]

    for j in range(NUM_FEATURES):
        name = FEATURE_NAMES[j]
        col = X[:, j].astype(np.float64)

        # NaN rate
        nan_rate = float(np.isnan(col).sum()) / n_bars
        if nan_rate > 0.05:
            r.warnings.append(f"feature '{name}' has {nan_rate:.1%} NaN rate (> 5%)")

        # For z-scored features, check clip saturation
        if name not in _NO_NORMALIZE:
            at_pos_clip = float((col >= 4.99).sum()) / n_bars
            at_neg_clip = float((col <= -4.99).sum()) / n_bars
            total_clip = at_pos_clip + at_neg_clip
            if total_clip > 0.10:
                r.warnings.append(
                    f"feature '{name}' has {total_clip:.1%} values at clip boundary (> 10%)"
                )

            # Dead feature check
            finite = col[np.isfinite(col)]
            if len(finite) > 100:
                std = float(np.std(finite))
                if std < 0.01:
                    r.warnings.append(f"feature '{name}' appears dead (std={std:.4f} < 0.01)")

                # Distribution sanity for z-scored
                mean = float(np.mean(finite))
                r.stats[f"feat_{name}_mean"] = mean
                r.stats[f"feat_{name}_std"] = std

    return r


# ---------------------------------------------------------------------------
# Sidecar validation
# ---------------------------------------------------------------------------

def validate_sidecar(sidecar: dict, date: str) -> DataQualityReport:
    """Validate a single day's sidecar file."""
    r = DataQualityReport(source=f"sidecar ({date})")

    # Shape integrity
    if "bar_ptrs" not in sidecar:
        r.errors.append("missing bar_ptrs")
        return r

    ptrs = sidecar["bar_ptrs"]
    if len(ptrs) != 391:
        r.errors.append(f"bar_ptrs has {len(ptrs)} entries, expected 391")

    if "contract_strike" not in sidecar or "contract_mid" not in sidecar:
        r.errors.append("missing contract_strike or contract_mid")
        return r

    n_contracts = len(sidecar["contract_strike"])
    mid_shape = sidecar["contract_mid"].shape
    if mid_shape[0] != n_contracts:
        r.errors.append(f"contract_mid has {mid_shape[0]} contracts, strike has {n_contracts}")

    r.stats["n_contracts"] = n_contracts

    # Price sanity (supervised window bars 30-270)
    bid = sidecar.get("contract_bid")
    mid = sidecar["contract_mid"]
    ask = sidecar.get("contract_ask")
    quality = sidecar.get("contract_quality")

    price_violations = 0
    ordering_violations = 0

    for c_idx in range(min(n_contracts, 50)):  # sample up to 50 contracts
        for bar in range(30, min(270, mid_shape[1] if mid.ndim > 1 else 390)):
            m = float(mid[c_idx, bar]) if mid.ndim > 1 else 0.0
            q = int(quality[c_idx, bar]) if quality is not None and quality.ndim > 1 else 0

            if q < QUALITY_PARTIAL:
                continue

            if not np.isfinite(m) or m <= 0:
                price_violations += 1
                continue

            if bid is not None and ask is not None and bid.ndim > 1:
                b = float(bid[c_idx, bar])
                a = float(ask[c_idx, bar])
                if np.isfinite(b) and np.isfinite(a):
                    if b > m + 0.01 or m > a + 0.01:
                        ordering_violations += 1

    if price_violations > 100:
        r.warnings.append(f"{price_violations} price violations (non-positive mid for valid quality)")
    if ordering_violations > 50:
        r.warnings.append(f"{ordering_violations} bid/mid/ask ordering violations")

    r.stats["price_violations"] = price_violations
    r.stats["ordering_violations"] = ordering_violations

    # Chain completeness: count executable contracts per bar
    thin_bars = 0
    total_checked = 0
    for bar in range(30, 270):
        feats, labels, _ = contract_snapshot(sidecar, bar)
        n_exec = int((feats[:, 0] > 0.5).sum()) if len(feats) > 0 else 0
        total_checked += 1
        if n_exec < 5:
            thin_bars += 1

    thin_pct = thin_bars / max(total_checked, 1)
    r.stats["thin_bar_pct"] = thin_pct
    if thin_pct > 0.50:
        r.warnings.append(f"{thin_pct:.0%} of supervised bars have < 5 executable contracts")

    # Label consistency
    if "bar_best_contract_idx" in sidecar and "bar_labelable" in sidecar:
        labelable = sidecar["bar_labelable"]
        best_idx = sidecar["bar_best_contract_idx"]
        label_issues = 0
        n_labelable = 0
        for bar in range(30, 270):
            if not labelable[bar]:
                continue
            n_labelable += 1
            idx = int(best_idx[bar])
            if idx < 0 or idx >= n_contracts:
                label_issues += 1

        r.stats["n_labelable_bars"] = n_labelable
        if label_issues > 0:
            r.warnings.append(f"{label_issues} bars with invalid best_contract_idx")

    # NaN in row_labels for labelable bars
    if "row_labels" in sidecar:
        rl = sidecar["row_labels"]
        nan_rate = float(np.isnan(rl).sum()) / max(len(rl), 1)
        r.stats["row_labels_nan_rate"] = nan_rate
        if nan_rate > 0.30:
            r.warnings.append(f"row_labels NaN rate is {nan_rate:.0%} (> 30%)")

    return r


# ---------------------------------------------------------------------------
# Label quality scoring
# ---------------------------------------------------------------------------

def compute_label_quality_scores(sidecar: dict) -> np.ndarray:
    """Compute per-bar label quality (margin between top and second-best).

    Returns (390,) array. High margin = clear oracle winner. Low margin = noise.
    """
    quality = np.zeros(390, dtype=np.float32)

    for bar in range(390):
        feats, labels, _ = contract_snapshot(sidecar, bar)
        if len(labels) == 0:
            continue

        valid = np.isfinite(labels) & (feats[:, 0] > 0.5)
        if valid.sum() < 1:
            continue

        valid_labels = labels[valid]
        if len(valid_labels) < 2:
            quality[bar] = float(valid_labels[0]) if len(valid_labels) == 1 else 0.0
            continue

        sorted_desc = np.sort(valid_labels)[::-1]
        quality[bar] = float(sorted_desc[0] - sorted_desc[1])

    return quality


# ---------------------------------------------------------------------------
# Full pipeline validation
# ---------------------------------------------------------------------------

def run_full_validation(
    data_path: str = "v2/data.pt",
    sidecar_sample: int = 10,
    seed: int = 42,
) -> list[DataQualityReport]:
    """Run all validation checks. Returns list of reports."""
    reports = []

    # 1. Manifest
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    manifest_report = validate_manifest(data)
    reports.append(manifest_report)

    # 2. Features
    X = data["X"]
    if hasattr(X, "numpy"):
        X = X.numpy()
    feat_report = validate_features(X)
    reports.append(feat_report)

    # 3. Sample sidecars
    sidecar_dir = data.get("metadata", {}).get("chain_sidecar_dir", "v2/data_sidecars")
    dates = sorted(set(data["dates"]))
    rng = random.Random(seed)
    sample_dates = rng.sample(dates, min(sidecar_sample, len(dates)))

    for day in sample_dates:
        path = os.path.join(sidecar_dir, f"{day}.pt")
        if not os.path.exists(path):
            r = DataQualityReport(source=f"sidecar ({day})")
            r.errors.append(f"sidecar file not found: {path}")
            reports.append(r)
            continue
        sidecar = torch.load(path, map_location="cpu", weights_only=False)
        sc_report = validate_sidecar(sidecar, day)
        reports.append(sc_report)

    return reports


def print_validation_summary(reports: list[DataQualityReport]) -> None:
    """Print a summary of all validation reports."""
    print(f"\n{'=' * 60}")
    print(f"  DATA INTEGRITY REPORT")
    print(f"{'=' * 60}")

    n_pass = sum(1 for r in reports if r.passed)
    n_fail = sum(1 for r in reports if not r.passed)
    n_warn = sum(len(r.warnings) for r in reports)

    print(f"  Checks: {len(reports)} total, {n_pass} passed, {n_fail} failed, {n_warn} warnings")
    print()

    for r in reports:
        r.print_report()

    overall = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n  Overall: {overall}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data integrity validation")
    parser.add_argument("--data", type=str, default="v2/data.pt")
    parser.add_argument("--sample", type=int, default=10, help="Number of sidecars to sample")
    args = parser.parse_args()

    reports = run_full_validation(data_path=args.data, sidecar_sample=args.sample)
    print_validation_summary(reports)

    if any(not r.passed for r in reports):
        exit(1)
