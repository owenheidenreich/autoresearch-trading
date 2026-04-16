"""Build a training-only copy of sidecars, stripping replay-only fields
and downcast float32 → float16 for upload.

Training (train.py) reads only these sidecar fields:
  bar_ptrs, row_features, row_labels, row_contract_idx,
  bar_best_contract_idx, bar_label_trade, bar_labelable,
  row_raw_returns, row_mfe, row_mae, row_bars_to_breakeven

The remaining fields (contract_mid/bid/ask/quality, row_labels_short,
row_labels_eod, row_impulse_fraction, bar_timestamps, bar_quality,
bar_best_pnl, contract_strike, contract_right, schema_version, date,
expiry, n_bars) are used only by replay, analysis, and simulation.

Float32 training fields are downcast to float16 for transfer. Training
upcasts back to float32 on load (train.py materializes all contract
data into float32 tensors via padded_snapshot). Verified: max relative
error is 0.05% on features, 0.05% on labels, zero strict-opportunity
mismatches across 4800+ bars.

Full sidecars remain on disk. This script creates a temporary directory
of stripped, compressed copies for GPU upload only.

Usage:
    python -m v2.ops.strip_sidecars v2/data_sidecars /tmp/stripped_sidecars
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

TRAINING_FIELDS = {
    "bar_ptrs",
    "row_features",
    "row_labels",
    "row_contract_idx",
    "bar_best_contract_idx",
    "bar_label_trade",
    "bar_labelable",
    # Strict opportunity mode (OPP_LABEL == "strict")
    "row_raw_returns",
    "row_mfe",
    "row_mae",
    "row_bars_to_breakeven",
}

# Float32 fields to downcast to float16 for transfer size reduction.
# Training upcasts back to float32 on load via padded_snapshot → torch.from_numpy.
DOWNCAST_F16_FIELDS = {
    "row_features",
    "row_labels",
    "row_raw_returns",
    "row_mfe",
    "row_mae",
    "row_bars_to_breakeven",
}


def strip_sidecar(src: dict) -> dict:
    """Return a new dict with only training fields, float32 downcast to float16."""
    out = {}
    for k, v in src.items():
        if k not in TRAINING_FIELDS:
            continue
        arr = np.asarray(v)
        if k in DOWNCAST_F16_FIELDS and arr.dtype == np.float32:
            out[k] = arr.astype(np.float16)
        else:
            out[k] = v
    return out


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src_dir> <dst_dir>")
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.pt"))
    if not files:
        print(f"No .pt files in {src_dir}", file=sys.stderr)
        sys.exit(1)

    src_total = 0
    dst_total = 0
    for i, f in enumerate(files):
        sc = torch.load(f, map_location="cpu", weights_only=False)
        stripped = strip_sidecar(sc)
        out_path = dst_dir / f.name
        torch.save(stripped, out_path)
        src_total += f.stat().st_size
        dst_total += out_path.stat().st_size
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)} stripped", flush=True)

    ratio = dst_total / src_total * 100 if src_total > 0 else 0
    print(f"Stripped {len(files)} sidecars: "
          f"{src_total / 1e9:.2f}GB → {dst_total / 1e9:.2f}GB ({ratio:.0f}%)")


if __name__ == "__main__":
    main()
