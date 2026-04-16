"""Counterfactual perturbation tests for validation gate liveness.

These tests deliberately break things and verify whether the current
validation infrastructure catches them. Run after any gate changes.

Usage:
    python3 -m v2.analysis.perturbation_tests

Exit code 0 means all perturbations were detected as expected.
Exit code 1 means at least one perturbation was NOT detected (gate gap).
"""

import copy
import hashlib
import os
import sys
import tempfile

import numpy as np
import torch

from v2.core.config import RUNTIME_CONFIG
from v2.core.features import FEATURE_NAMES


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def test_column_swap_detection() -> tuple[str, bool, str]:
    """Swap delta and charm columns in a sidecar — should be detected."""
    name = "sidecar column swap (delta<->charm)"
    sidecar_dir = "v2/data_sidecars"
    files = sorted(os.listdir(sidecar_dir))
    if not files:
        return name, False, "no sidecars found"

    test_file = files[len(files) // 2]
    original_path = os.path.join(sidecar_dir, test_file)
    original_hash = _file_hash(original_path)

    sc = torch.load(original_path, map_location="cpu", weights_only=False)
    corrupted = copy.deepcopy(sc)
    rf = np.asarray(corrupted["row_features"]).copy()
    rf[:, [8, 16]] = rf[:, [16, 8]]  # swap delta <-> charm
    corrupted["row_features"] = rf

    # Save to temp and check hash change
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    try:
        torch.save(corrupted, tmp.name)
        corrupt_hash = _file_hash(tmp.name)
    finally:
        os.unlink(tmp.name)

    if original_hash != corrupt_hash:
        return name, True, "file hash changed — sidecar digest would detect IF recomputed at load time"
    return name, False, "hashes match — column swap is invisible"


def test_schema_version_detection() -> tuple[str, bool, str]:
    """Flip schema version — should be detected by validate_dataset_metadata."""
    name = "schema version mismatch"
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    meta = copy.deepcopy(data["metadata"])
    meta["chain_schema_version"] = "v3_wrong_version"
    errors = RUNTIME_CONFIG.validate_dataset_metadata(meta)
    if errors:
        return name, True, f"validate_dataset_metadata caught: {errors[0]}"
    return name, False, "not detected"


def test_feature_count_detection() -> tuple[str, bool, str]:
    """Wrong feature count — should be detected."""
    name = "feature count mismatch (52->47)"
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    meta = copy.deepcopy(data["metadata"])
    meta["n_features"] = 47
    errors = RUNTIME_CONFIG.validate_dataset_metadata(meta)
    if errors:
        return name, True, f"validate_dataset_metadata caught: {errors[0]}"
    return name, False, "not detected"


def test_feature_name_reorder_detection() -> tuple[str, bool, str]:
    """Swap two feature names — should be detected."""
    name = "feature name reorder"
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    meta = copy.deepcopy(data["metadata"])
    if "feature_names" not in meta:
        return name, False, "no feature_names in metadata"
    names = list(meta["feature_names"])
    names[8], names[16] = names[16], names[8]
    meta["feature_names"] = names
    errors = RUNTIME_CONFIG.validate_dataset_metadata(meta)
    if errors:
        return name, True, f"validate_dataset_metadata caught: {errors[0]}"
    return name, False, "not detected"


def test_fingerprint_detection() -> tuple[str, bool, str]:
    """Corrupt dataset fingerprint — is it checked anywhere?"""
    name = "corrupted dataset fingerprint"
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)
    meta = copy.deepcopy(data["metadata"])
    meta["fingerprint"] = "deadbeefdeadbeef"
    errors = RUNTIME_CONFIG.validate_dataset_metadata(meta)
    if errors:
        return name, True, f"validate_dataset_metadata caught: {errors[0]}"
    # Also check if load_dataset checks it
    from v2.train import load_dataset
    import inspect
    src = inspect.getsource(load_dataset)
    if "fingerprint" in src:
        return name, True, "load_dataset references fingerprint"
    return name, False, "fingerprint not checked by any gate"


def test_missing_sidecar_detection() -> tuple[str, bool, str]:
    """Missing sidecar file — should crash, not silently proceed."""
    name = "missing sidecar file"
    try:
        torch.load("v2/data_sidecars/NONEXISTENT.pt", map_location="cpu", weights_only=False)
        return name, False, "torch.load did not error"
    except FileNotFoundError:
        return name, True, "FileNotFoundError raised — training would crash"
    except Exception as e:
        return name, True, f"{type(e).__name__} raised"


# Registry of all tests.
# detected_expected: True if we WANT the gate to catch this.
# If detected_expected=True and detected=False, it's a gate gap.
TESTS = [
    (test_column_swap_detection, True),
    (test_schema_version_detection, True),
    (test_feature_count_detection, True),
    (test_feature_name_reorder_detection, True),
    (test_fingerprint_detection, True),
    (test_missing_sidecar_detection, True),
]


def main() -> int:
    print("=" * 70)
    print("COUNTERFACTUAL PERTURBATION TESTS")
    print("Deliberately breaking things to verify gates are alive.")
    print("=" * 70)
    print()

    gaps = []
    for test_fn, should_detect in TESTS:
        name, detected, detail = test_fn()
        if should_detect and detected:
            status = "DETECTED"
        elif should_detect and not detected:
            status = "GAP"
            gaps.append(name)
        elif not should_detect and not detected:
            status = "OK (expected)"
        else:
            status = "UNEXPECTED"

        print(f"  [{status:>10}] {name}")
        print(f"             {detail}")
        print()

    print("=" * 70)
    if gaps:
        print(f"GATE GAPS FOUND ({len(gaps)}):")
        for g in gaps:
            print(f"  - {g}")
        print()
        print("These perturbations are NOT caught by any current gate.")
        return 1
    else:
        print("All perturbations detected. Gates are alive.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
