"""Build a mixture simulated-L3 oracle by blending two oracles per cell.

For each (row, action) cell, the mixture oracle's l3_exit_pnl is
`alpha * oracle_a + (1 - alpha) * oracle_b`. Cells where either oracle
is non-simulated (trigger=-1 / NaN) fall back to the other; cells where
both are non-simulated remain NaN.

Intended use: build a blend of the chosen-trained-L3 oracle and the
candidate-trained-L3 oracle per seed, to teach the entry policy a
signal that is neither pure champion nor pure candidate.

Usage:

  .venv/bin/python -m v3.layer2.build_mixture_oracle \\
    --oracle-a v3/artifacts/simulated_l3_oracle_seed42_fp.npz \\
    --oracle-b v3/artifacts/simulated_l3_oracle_seed42_candidate_l3_mpd4_fp.npz \\
    --alpha 0.5 \\
    --output v3/artifacts/simulated_l3_oracle_seed42_mix50_fp.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-a", required=True)
    p.add_argument("--oracle-b", required=True)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight on oracle A. 1.0 = pure A, 0.0 = pure B.")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    za = np.load(args.oracle_a, allow_pickle=True)
    zb = np.load(args.oracle_b, allow_pickle=True)
    pnl_a = za["l3_exit_pnl"]
    pnl_b = zb["l3_exit_pnl"]
    trig_a = za["l3_exit_trigger"]
    trig_b = zb["l3_exit_trigger"]
    bar_a = za["l3_exit_bar"]

    if pnl_a.shape != pnl_b.shape:
        raise RuntimeError(f"shape mismatch: {pnl_a.shape} vs {pnl_b.shape}")

    alpha = float(args.alpha)
    finite_a = np.isfinite(pnl_a)
    finite_b = np.isfinite(pnl_b)
    both = finite_a & finite_b
    only_a = finite_a & ~finite_b
    only_b = finite_b & ~finite_a

    mix = np.full_like(pnl_a, np.nan, dtype=np.float32)
    mix[both] = (alpha * pnl_a[both] + (1.0 - alpha) * pnl_b[both]).astype(np.float32)
    mix[only_a] = pnl_a[only_a]
    mix[only_b] = pnl_b[only_b]

    # Trigger: keep A's where both are simulated; otherwise the one that is.
    # For trainer purposes only trigger=0,1,2,3 vs -1 matters (flat / model /
    # time_stop / fold0 vs non_tradeable).
    mix_trig = trig_a.copy()
    mix_trig[only_b] = trig_b[only_b]

    flat_mask = (trig_a == 0) | (trig_b == 0)
    mix[flat_mask] = 0.0
    mix_trig[flat_mask] = 0

    meta = {
        "alpha": alpha,
        "oracle_a": args.oracle_a,
        "oracle_b": args.oracle_b,
        "cells_both_simulated": int(both.sum()),
        "cells_only_a": int(only_a.sum()),
        "cells_only_b": int(only_b.sum()),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        l3_exit_pnl=mix.astype(np.float32),
        l3_exit_bar=bar_a.astype(np.int32),  # bars from A (for reference only)
        l3_exit_trigger=mix_trig.astype(np.int8),
        meta_json=np.array(json.dumps(meta), dtype=object),
    )
    print(
        f"Saved mixture oracle alpha={alpha}: {args.output}\n"
        f"  cells both_simulated={both.sum()}  only_a={only_a.sum()}  only_b={only_b.sum()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
