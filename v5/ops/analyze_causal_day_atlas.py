"""Summarize every day around 10:00 and 13:30 without fitting a policy.

The atlas distinguishes three different quantities that must not be conflated:

* underlying opportunity -- did the future path move 10/20/30 points?;
* contract opportunity -- did any *currently eligible* OTM contract reach the
  declared ITM depth?; and
* executable clock economics -- what happened to all eligible contracts at a
  fixed clock, plus an explicitly labelled hindsight best-contract ceiling.

None is a trading policy.  Quiet days and minutes with zero eligible contracts
remain rows with false opportunity indicators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import HORIZONS, ITM_DEPTHS, canonical_json, file_sha256


NAMED = {"10:00": "magic_time", "13:30": "algo"}
FAMILY_SIZE = len(NAMED) * len(HORIZONS) * len(ITM_DEPTHS)
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 0


class AtlasError(RuntimeError):
    """The derived dataset cannot support the declared atlas."""


def aggregate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per day/minute; no-contract minutes are added by the caller."""

    rows: list[dict[str, Any]] = []
    for (session, minute), group in candidates.groupby(["session", "entry_minute"], sort=False):
        row: dict[str, Any] = {
            "session": session,
            "minute": minute,
            "eligible_contracts": len(group),
            "eligible_calls": int(group["is_call"].sum()),
            "eligible_puts": int((~group["is_call"]).sum()),
            "mean_entry_ask_usd": float(group["entry_ask_usd"].mean()),
            "minimum_entry_ask_usd": float(group["entry_ask_usd"].min()),
            "maximum_entry_ask_usd": float(group["entry_ask_usd"].max()),
        }
        for horizon in HORIZONS:
            status = group[f"clock_exit_status_{horizon}m"]
            net = group[f"net_bid_{horizon}m_usd"]
            row[f"blocked_exit_contracts_{horizon}m"] = int(
                status.eq("blocked_no_executable_bid").sum()
            )
            row[f"blocked_exit_share_{horizon}m"] = float(
                status.eq("blocked_no_executable_bid").mean()
            )
            row[f"mean_clock_net_bid_{horizon}m_usd"] = float(net.mean())
            row[f"median_clock_net_bid_{horizon}m_usd"] = float(net.median())
            # Hindsight best contract is an opportunity ceiling, never an action.
            row[f"oracle_best_contract_net_bid_{horizon}m_usd"] = float(net.max())
            row[f"mean_option_mfe_{horizon}m_usd"] = float(
                group[f"option_mfe_{horizon}m_usd"].mean()
            )
            row[f"oracle_best_option_mfe_{horizon}m_usd"] = float(
                group[f"option_mfe_{horizon}m_usd"].max()
            )
            for depth in ITM_DEPTHS:
                name = "cross" if depth == 0 else f"{depth}_itm"
                values = group[f"reached_{name}_{horizon}m"]
                row[f"any_contract_reached_{name}_{horizon}m"] = bool(values.any())
                row[f"share_contracts_reached_{name}_{horizon}m"] = float(values.mean())
                row[f"any_call_reached_{name}_{horizon}m"] = bool(
                    values[group["is_call"]].any()
                )
                row[f"any_put_reached_{name}_{horizon}m"] = bool(
                    values[~group["is_call"]].any()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def complete_minute_atlas(
    atlas: pd.DataFrame, minutes: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Left join candidates so abstention/no-contract minutes cannot vanish."""

    aggregated = aggregate_candidates(candidates)
    causal_columns = [
        "session",
        "minute",
        "history_minutes",
        "regime",
        "spx_snapshot",
        "move_from_open_points",
        "session_range_points",
        "range_position",
        "live_ladder_contracts",
        "eligible_entry_contracts",
        "median_spread_usd",
        "median_mid_usd",
        "displayed_size_imbalance",
        "call_iv_median",
        "put_iv_median",
        "call_minus_put_iv",
    ]
    missing = sorted(set(causal_columns) - set(minutes.columns))
    if missing:
        raise AtlasError(f"minute state columns missing: {missing}")
    result = atlas.merge(
        minutes[causal_columns], on=["session", "minute", "regime"], how="left", validate="one_to_one"
    ).merge(aggregated, on=["session", "minute"], how="left", validate="one_to_one")

    result["eligible_contracts"] = result["eligible_contracts"].fillna(0).astype(int)
    for column in result.columns:
        if column.startswith("any_contract_reached_") or column.startswith("any_call_reached_") or column.startswith("any_put_reached_"):
            result[column] = result[column].fillna(False).astype(bool)
        elif column.startswith("share_contracts_reached_") or column.startswith("blocked_exit_share_"):
            result[column] = result[column].fillna(0.0)
    return result


def bootstrap_difference(values: np.ndarray) -> dict[str, float | int]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not clean.size:
        return {"n_sessions": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clean), size=(BOOTSTRAP_REPS, len(clean)))
    means = clean[draws].mean(axis=1)
    alpha = 0.05 / FAMILY_SIZE
    return {
        "n_sessions": len(clean),
        "mean": float(np.mean(clean)),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def named_comparisons(full: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for minute, band_name in NAMED.items():
        exact = full[full["minute"].eq(minute)].set_index("session")
        local = full[full["key_band"].eq(band_name) & ~full["minute"].eq(minute)]
        local_means = local.groupby("session")
        for horizon in HORIZONS:
            for depth in ITM_DEPTHS:
                name = "cross" if depth == 0 else f"{depth}_itm"
                metric = f"any_contract_reached_{name}_{horizon}m"
                neighbor = local_means[metric].mean()
                joined = exact[[metric]].join(neighbor.rename("local_mean"), how="inner")
                diff = joined[metric].astype(float) - joined["local_mean"]
                rows.append(
                    {
                        "named_minute": minute,
                        "band": band_name,
                        "horizon_minutes": horizon,
                        "itm_depth_points": depth,
                        "named_share": float(joined[metric].mean()),
                        "local_neighbor_share": float(joined["local_mean"].mean()),
                        "named_minus_local": bootstrap_difference(diff.to_numpy(float)),
                    }
                )
    return rows


def time_surface(full: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for minute, group in full.groupby("minute", sort=True):
        row: dict[str, Any] = {
            "minute": minute,
            "regime": group["regime"].iloc[0],
            "sessions": group["session"].nunique(),
            "sessions_with_eligible_contract": int(group["eligible_contracts"].gt(0).sum()),
            "mean_eligible_contracts": float(group["eligible_contracts"].mean()),
            "mean_entry_ask_usd": float(group["mean_entry_ask_usd"].mean()),
        }
        for horizon in HORIZONS:
            row[f"mean_up_excursion_{horizon}m_points"] = float(
                group[f"up_excursion_{horizon}m_points"].mean()
            )
            row[f"mean_down_excursion_{horizon}m_points"] = float(
                group[f"down_excursion_{horizon}m_points"].mean()
            )
            row[f"mean_clock_net_bid_{horizon}m_usd"] = float(
                group[f"mean_clock_net_bid_{horizon}m_usd"].mean()
            )
            row[f"oracle_best_contract_net_bid_{horizon}m_usd"] = float(
                group[f"oracle_best_contract_net_bid_{horizon}m_usd"].mean()
            )
            for depth in ITM_DEPTHS:
                name = "cross" if depth == 0 else f"{depth}_itm"
                metric = f"any_contract_reached_{name}_{horizon}m"
                row[f"session_share_{metric}"] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _status_counts(candidates: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        f"{horizon}m": {
            str(key): int(value)
            for key, value in candidates[f"clock_exit_status_{horizon}m"].value_counts().items()
        }
        for horizon in HORIZONS
    }


def run(dataset_dir: Path, declaration: Path, out_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    if out_dir.exists() or evidence_dir.exists():
        raise AtlasError("refusing to overwrite atlas output or evidence")
    atlas = pd.read_parquet(dataset_dir / "atlas.parquet")
    minutes = pd.read_parquet(dataset_dir / "minutes.parquet")
    candidates = pd.read_parquet(dataset_dir / "candidates.parquet")
    full = complete_minute_atlas(atlas, minutes, candidates)
    key = full[full["key_band"].notna()].copy()
    surface = time_surface(full)
    comparisons = named_comparisons(full)
    named_contracts = candidates[candidates["entry_minute"].isin(NAMED)].copy()

    out_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "all_minutes": out_dir / "all_minutes.parquet",
        "every_day_key_time": out_dir / "every_day_key_time.parquet",
        "time_surface": out_dir / "time_surface.csv",
        "named_contracts": out_dir / "named_minute_contracts.parquet",
        "named_comparisons": out_dir / "named_comparisons.json",
    }
    full.to_parquet(artifacts["all_minutes"], index=False)
    key.to_parquet(artifacts["every_day_key_time"], index=False)
    surface.to_csv(artifacts["time_surface"], index=False)
    named_contracts.to_parquet(artifacts["named_contracts"], index=False)
    artifacts["named_comparisons"].write_text(json.dumps(comparisons, indent=2, sort_keys=True) + "\n")

    named_summary: dict[str, Any] = {}
    for minute, label in NAMED.items():
        block = full[full["minute"].eq(minute)]
        named_summary[label] = {
            "minute": minute,
            "sessions": int(block["session"].nunique()),
            "sessions_with_eligible_contract": int(block["eligible_contracts"].gt(0).sum()),
            "mean_eligible_contracts": float(block["eligible_contracts"].mean()),
            "mean_entry_ask_usd": float(block["mean_entry_ask_usd"].mean()),
            "opportunity": {
                f"any_contract_reached_{name}_{horizon}m": float(
                    block[f"any_contract_reached_{name}_{horizon}m"].mean()
                )
                for horizon in HORIZONS
                for _, name in ((0, "cross"), (10, "10_itm"), (20, "20_itm"), (30, "30_itm"))
            },
            "mean_clock_net_bid_usd": {
                f"{horizon}m": float(block[f"mean_clock_net_bid_{horizon}m_usd"].mean())
                for horizon in HORIZONS
            },
            "oracle_best_contract_net_bid_usd": {
                f"{horizon}m": float(
                    block[f"oracle_best_contract_net_bid_{horizon}m_usd"].mean()
                )
                for horizon in HORIZONS
            },
        }

    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-atlas.v1",
        "created_on": "2026-08-14",
        "purpose": "model-free all-day and key-time opportunity atlas",
        "declaration_sha256": file_sha256(declaration),
        "sessions": int(full["session"].nunique()),
        "rows": {
            "all_minutes": len(full),
            "every_day_key_time": len(key),
            "named_contracts": len(named_contracts),
        },
        "named_summary": named_summary,
        "named_comparisons": comparisons,
        "comparison_family_size": FAMILY_SIZE,
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "session_unit": True},
        "clock_exit_status_all_candidates": _status_counts(candidates),
        "interpretation_limits": [
            "any-contract opportunity and hindsight best-contract net are ceilings, not causal actions",
            "mean clock net buys every eligible contract and is instrument characterization, not a feasible one-position policy",
            "no model, threshold, entry policy, or exit policy is fitted",
            "quiet and no-eligible-contract minutes remain in every denominator",
            "terminal no-bid rows remain blocked because official cash settlement is not validated",
        ],
        "artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for name, path in artifacts.items()
        },
        "model_fit": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    (evidence_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(evidence_dir / "receipt.json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v1")
    )
    parser.add_argument(
        "--declaration", type=Path, default=Path("v5/work/entry-exit-attribution/DECLARATION.json")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_dir, args.declaration, args.out_dir, args.evidence_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
