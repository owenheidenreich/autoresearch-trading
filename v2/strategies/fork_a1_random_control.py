"""Random-entry control for Fork A1 Stage 1.

For each session, pick one random bar in the 09:45-12:00 ET window
(bars 15-150) and measure forward MFE/MAE at 5/10/20/30 minutes on SPX spot
and on a delta-targeted call contract picked at that bar. No preconditions,
no entry trigger — pure "what happens if you just pick a random moment?"

Purpose: provide a minimal control against which Stage 1's candidate-bar
hit rates can be read. If Stage 1 doesn't beat random entries, the narrated
Pickles-qualifier trigger carries no information *relative to picking a
random moment in the morning window*.

Output shape mirrors Stage 1's summary JSON so side-by-side comparison is
trivial.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v2.core.chain_data import load_sidecar_cached
from v2.core.provenance import build_provenance, write_provenance
from v2.strategies.fork_a1_stage1 import _group_by_day, _compute_forward_metrics
from v2.strategies.pickles_row1 import (
    DELTA_TOLERANCE,
    MIN_CONTRACT_MID,
    assert_feature_layout,
    pick_delta_targeted_call,
)
from v2.strategies.session_state import WINDOW_FIRST_BAR, WINDOW_LAST_BAR


def run_random_control(
    data_path: str,
    delta_grid: tuple[float, ...],
    out_dir: str,
    seed: int = 7,
) -> dict[str, Any]:
    t0 = time.time()
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    feature_names = data["feature_names"]
    assert_feature_layout(feature_names)

    bar_of_day = data["bar_of_day"].numpy()
    dates = list(data["dates"])
    spot_prices = data["spot_prices"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    dataset_fingerprint = str(data["metadata"].get("fingerprint", "unknown"))
    sidecar_schema_version = str(data["metadata"].get("chain_schema_version", "unknown"))

    sessions = _group_by_day(dates, bar_of_day)
    rng = np.random.default_rng(seed)

    records: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    for si, (day, day_start, day_end) in enumerate(sessions):
        session_length = day_end - day_start + 1
        # Window: bars WINDOW_FIRST_BAR..min(WINDOW_LAST_BAR, session_length - 1)
        last_bar = min(WINDOW_LAST_BAR, session_length - 1)
        if last_bar < WINDOW_FIRST_BAR:
            continue
        local_bar = int(rng.integers(WINDOW_FIRST_BAR, last_bar + 1))
        global_bar = day_start + local_bar
        sidecar_path = os.path.join(sidecar_dir, f"{day}.pt")
        if not os.path.exists(sidecar_path):
            continue
        sidecar = load_sidecar_cached(sidecar_path)

        entry_spot = float(spot_prices[global_bar])
        for td in delta_grid:
            pick, skip = pick_delta_targeted_call(sidecar, local_bar, td)
            rec = {
                "date": day, "bar_of_day": local_bar, "global_bar": global_bar,
                "target_delta": td, "spx_spot": entry_spot,
                "skip_reason": "" if pick is not None else (skip or "no_delta_match"),
                "contract_idx": pick.contract_idx if pick else -1,
                "contract_strike": pick.strike if pick else 0.0,
                "contract_delta": pick.delta if pick else 0.0,
                "contract_mid_at_entry": pick.mid_at_entry if pick else 0.0,
            }
            records.append(rec)
            if pick is None:
                continue
            opt_mid = np.asarray(sidecar["contract_mid"][pick.contract_idx], dtype=np.float64)
            fwd = _compute_forward_metrics(
                entry_global_bar=global_bar, day_last_bar_global=day_end,
                spot_prices=spot_prices, entry_spot=entry_spot,
                option_mid=opt_mid, local_bar=local_bar,
            )
            for fm in fwd:
                forward_rows.append({
                    "date": day, "bar_of_day": local_bar, "global_bar": global_bar,
                    "target_delta": td, "horizon": fm.horizon,
                    "spot_mfe": fm.spot_mfe, "spot_mae": fm.spot_mae,
                    "spot_end_ret": fm.spot_end_ret,
                    "opt_mfe": fm.opt_mfe, "opt_mae": fm.opt_mae,
                    "opt_end_ret": fm.opt_end_ret,
                    "n_valid_spot_bars": fm.n_valid_spot_bars,
                    "n_valid_opt_bars": fm.n_valid_opt_bars,
                })

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cand_df = pd.DataFrame(records)
    fwd_df = pd.DataFrame(forward_rows)
    cand_df.to_csv(os.path.join(out_dir, "random_control_candidates.csv"), index=False)
    fwd_df.to_csv(os.path.join(out_dir, "random_control_forward.csv"), index=False)

    # Summary — same shape as Stage 1's hit_rate_table but one "cell": random.
    rows: list[dict[str, Any]] = []
    if not fwd_df.empty:
        for (delta, h), g in fwd_df.groupby(["target_delta", "horizon"]):
            n = int(len(g))
            def frac(s): return float(s.mean()) if n else float("nan")
            rows.append({
                "ablation_cell": "RANDOM-CONTROL",
                "target_delta": float(delta),
                "horizon": int(h),
                "n": n,
                "spot_mfe_ge_30bps": frac(g["spot_mfe"] >= 3e-3),
                "spot_mfe_ge_50bps": frac(g["spot_mfe"] >= 5e-3),
                "spot_mfe_ge_100bps": frac(g["spot_mfe"] >= 1e-2),
                "spot_mae_le_neg30bps": frac(g["spot_mae"] <= -3e-3),
                "spot_end_ret_pos": frac(g["spot_end_ret"] > 0),
                "opt_end_ret_pos": frac(g["opt_end_ret"] > 0),
                "spot_mfe_median": float(g["spot_mfe"].median()),
                "spot_mae_median": float(g["spot_mae"].median()),
                "opt_mfe_median": float(g["opt_mfe"].median()),
                "opt_mae_median": float(g["opt_mae"].median()),
                "opt_end_ret_median": float(g["opt_end_ret"].median()),
            })

    summary = {
        "n_candidate_records": int(len(cand_df)),
        "n_forward_records": int(len(fwd_df)),
        "seed": seed,
        "hit_rate_table": rows,
    }
    with open(os.path.join(out_dir, "random_control_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    prov = build_provenance(
        data_path=data_path, feature_names=list(feature_names),
        label_mode="fork_a1_random_control",
        label_thresholds={"seed": seed, "delta_tolerance": DELTA_TOLERANCE,
                          "min_contract_mid": MIN_CONTRACT_MID},
        screen_mode="full", n_folds=0, research_tier=False,
        dataset_fingerprint=dataset_fingerprint,
        sidecar_schema_version=sidecar_schema_version,
    )
    write_provenance(os.path.join(out_dir, "random_control_provenance.json"), prov,
                     extra={"delta_grid": list(delta_grid),
                            "n_sessions": len(sessions),
                            "elapsed_sec": round(time.time() - t0, 1)})
    print(f"Random control: {len(cand_df)} candidates, {len(fwd_df)} forward rows, "
          f"elapsed={time.time() - t0:.1f}s")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default="v2/artifacts/fork_a1_stage1")
    ap.add_argument("--deltas", default="0.40,0.50")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    grid = tuple(float(x.strip()) for x in args.deltas.split(",") if x.strip())
    run_random_control(args.data, grid, args.out_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
