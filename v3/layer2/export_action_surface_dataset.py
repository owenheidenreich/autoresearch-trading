from __future__ import annotations

import argparse
import time

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.action_surface_dataset import (
    DEFAULT_ACTION_SURFACE_DATASET_PATH,
    DEFAULT_CLEAN_MAE_FLOOR_PCT,
    DEFAULT_EXECUTION_END_BAR,
    DEFAULT_EXECUTION_START_BAR,
    DEFAULT_HISTORY_BARS,
    DEFAULT_STOPOUT_HORIZON_BARS,
    DEFAULT_STOPOUT_TARGET_PCT,
    DEFAULT_TOP_K_CONTRACTS,
    build_action_surface_bundle,
)
from v3.layer2.common import save_pickle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the canonical Layer-2 unified action surface dataset.")
    p.add_argument("--output", default=DEFAULT_ACTION_SURFACE_DATASET_PATH)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--history-bars", type=int, default=DEFAULT_HISTORY_BARS)
    p.add_argument("--top-k-contracts", type=int, default=DEFAULT_TOP_K_CONTRACTS)
    p.add_argument("--execution-start-bar", type=int, default=DEFAULT_EXECUTION_START_BAR)
    p.add_argument("--execution-end-bar", type=int, default=DEFAULT_EXECUTION_END_BAR)
    p.add_argument("--clean-mae-floor-pct", type=float, default=DEFAULT_CLEAN_MAE_FLOOR_PCT)
    p.add_argument("--stopout-target-pct", type=float, default=DEFAULT_STOPOUT_TARGET_PCT)
    p.add_argument("--stopout-horizon-bars", type=int, default=DEFAULT_STOPOUT_HORIZON_BARS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    bundle = build_action_surface_bundle(
        ds,
        cfg,
        args.equity,
        history_bars=args.history_bars,
        top_k_contracts=args.top_k_contracts,
        execution_start_bar=args.execution_start_bar,
        execution_end_bar=args.execution_end_bar,
        mae_floor_pct=args.clean_mae_floor_pct,
        stopout_target_pct=args.stopout_target_pct,
        stopout_horizon_bars=args.stopout_horizon_bars,
    )
    save_pickle(args.output, bundle)

    meta = bundle["meta"]
    print(f"Wrote {args.output}")
    print(f"Rows: {len(bundle['rows']):,} across {meta['kept_days']} days")
    print(
        "Scalar/sequence/contract/action-label features: "
        f"{len(meta['scalar_feature_names'])}/"
        f"{len(meta['sequence_feature_names'])}/"
        f"{len(meta['contract_feature_names'])}/"
        f"{len(meta['action_label_names'])}"
    )
    print(
        f"History bars={meta['history_bars']} top_k_contracts_per_side={meta['top_k_contracts_per_side']} "
        f"execution={meta['execution_window']['label']} elapsed={time.time() - t0:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
