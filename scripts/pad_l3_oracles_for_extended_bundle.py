"""Pad each per-seed L3 oracle npz with NaN rows to match the extended
action_surface bundle's row count.

The oracle was built on the old bundle (89,692 rows). The new bundle is
91,148 rows (+1,456 forward-walk rows). The forward-walk script passes
the oracle to _slice_inputs which expects oracle.shape[0] == bundle rows.
The oracle's predictions on forward-walk rows are NaN by construction
(oracle was trained on training-cohort rows only); padding with NaN is
the correct extension.

Output: writes new files with `_extended` suffix; original files
preserved.
"""
from __future__ import annotations

import json
import os

import numpy as np


SEEDS = [42, 43, 44, 45, 46]
TEMPLATE = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_balanced.npz"
OUT_TEMPLATE = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_balanced_extended.npz"
NEW_N_ROWS = 91148  # from extended bundle


def main() -> None:
    for seed in SEEDS:
        in_path = TEMPLATE.format(seed=seed)
        out_path = OUT_TEMPLATE.format(seed=seed)
        npz = np.load(in_path, allow_pickle=True)
        keys = list(npz.keys())
        n_old = npz["l3_exit_pnl"].shape[0]
        n_actions = npz["l3_exit_pnl"].shape[1]
        n_pad = NEW_N_ROWS - n_old
        if n_pad < 0:
            raise RuntimeError(f"new bundle has fewer rows than oracle for seed {seed}")
        if n_pad == 0:
            print(f"seed {seed}: already extended")
            continue
        out: dict[str, np.ndarray] = {}
        for k in keys:
            v = npz[k]
            if k == "meta_json":
                meta = json.loads(str(v))
                meta["n_rows_extended"] = NEW_N_ROWS
                meta["original_n_rows"] = n_old
                out[k] = np.array(json.dumps(meta), dtype=object)
                continue
            if v.ndim == 2 and v.shape[0] == n_old:
                pad = np.full((n_pad, n_actions), np.nan if v.dtype.kind == "f" else 0, dtype=v.dtype)
                if v.dtype == np.int32 or v.dtype == np.int8:
                    pad.fill(-1)
                out[k] = np.concatenate([v, pad], axis=0)
            else:
                out[k] = v
        np.savez_compressed(out_path, **out)
        print(f"seed {seed}: padded from {n_old} -> {NEW_N_ROWS} ({n_pad} new rows)")
    print("\nDone. Use --oracle-pattern with new _extended suffix in forward_walk.")


if __name__ == "__main__":
    main()
