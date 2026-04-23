from __future__ import annotations

import argparse
import time

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import save_pickle
from v3.layer2.surface_dataset import (
    DEFAULT_HISTORY_BARS,
    DEFAULT_SURFACE_DATASET_PATH,
    DEFAULT_TOP_K_CONTRACTS,
    build_surface_export_bundle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the W1 Layer-2 surface-aware dataset.")
    p.add_argument("--output", default=DEFAULT_SURFACE_DATASET_PATH)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--history-bars", type=int, default=DEFAULT_HISTORY_BARS)
    p.add_argument("--top-k-contracts", type=int, default=DEFAULT_TOP_K_CONTRACTS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    bundle = build_surface_export_bundle(
        ds,
        cfg,
        args.equity,
        history_bars=args.history_bars,
        top_k_contracts=args.top_k_contracts,
    )
    save_pickle(args.output, bundle)

    meta = bundle["meta"]
    print(f"Wrote {args.output}")
    print(f"Rows: {len(bundle['rows']):,} across {meta['kept_days']} days")
    print(
        f"Scalar/sequence/contract features: "
        f"{len(meta['scalar_feature_names'])}/"
        f"{len(meta['sequence_feature_names'])}/"
        f"{len(meta['contract_feature_names'])}"
    )
    print(
        f"History bars={meta['history_bars']} top_k_contracts={meta['top_k_contracts']} "
        f"elapsed={time.time() - t0:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
