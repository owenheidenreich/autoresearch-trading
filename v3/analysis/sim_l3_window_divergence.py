"""Diagnose local divergence between simulated-L3 entry stacks.

The motivating case is seed 42, window 6, where the iteration-2
simulated-L3 branch raised mean PF but weakened the cross-seed floor.
This script compares saved Layer-2 chosen-trade artifacts and their
calibrated Layer-3 replays without retraining anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.layer2.common import replay_metrics_from_pnls
from v3.layer3.common import agg_exit_metrics


DEFAULT_BRANCHES = [
    (
        "champion",
        "v3/artifacts/layer2_unified_policy_simL3_objfix_seed42/seed_42",
        "v3/artifacts/layer3_unified_cpu_simL3_objfix_seed42",
    ),
    (
        "iter2",
        "v3/artifacts/layer2_unified_policy_simL3_iter2_seed42/seed_42",
        "v3/artifacts/layer3_unified_cpu_simL3_iter2_seed42",
    ),
    (
        "iter2_margin005",
        "v3/artifacts/layer2_unified_policy_simL3_iter2_margin005_seed42/seed_42",
        "v3/artifacts/layer3_unified_cpu_simL3_iter2_margin005_seed42",
    ),
]

DEFAULT_OUT_JSON = "v3/artifacts/analysis/sim_l3_iter2_w6_diagnostic.json"
DEFAULT_OUT_CSV = "v3/artifacts/analysis/sim_l3_iter2_w6_day_diff.csv"

ENTRY_DETAIL_COLS = [
    "day",
    "bar_index",
    "chosen_side",
    "chosen_strike",
    "decision_margin",
    "flat_score",
    "best_nonflat_score",
    "chosen_objective_pnl",
    "chosen_time_stop_pnl",
    "pred_clean_entry_prob",
    "pred_stopout_risk",
    "chosen_time_bucket",
    "teacher_any_triggered",
    "teacher_n_triggered",
    "sigma_pos",
    "omar_mid_pos_units",
    "last10_break_state",
]

EXIT_DETAIL_COLS = [
    "entry_bar",
    "exit_bar",
    "exit_pnl",
    "trigger",
    "bars_held",
    "max_exit_prob",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument(
        "--calibration-suffix",
        default="robust_90",
        help="Suffix used by v3.layer3.calibrate_threshold output files.",
    )
    p.add_argument(
        "--branch",
        action="append",
        nargs=3,
        metavar=("NAME", "L2_SEED_DIR", "L3_RUN_DIR"),
        help=(
            "Branch to include. May be repeated. If omitted, compares the "
            "seed-42 promoted champion, iter2, and iter2 +0.05 margin control."
        ),
    )
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    return p.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _pf(pnls: pd.Series | np.ndarray | list[float]) -> float:
    arr = np.asarray(pnls, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    wins = float(arr[arr > 0].sum())
    losses = float(-arr[arr <= 0].sum())
    return wins / losses if losses > 0 else float("inf")


def _pnl_metrics(pnls: pd.Series | np.ndarray | list[float], equity: float) -> dict[str, Any]:
    arr = np.asarray(pnls, dtype=np.float64)
    metrics = replay_metrics_from_pnls(arr.tolist(), equity)
    return {
        "trades": int(arr.size),
        "pnl": float(arr.sum()) if arr.size else 0.0,
        "pf": float(metrics["pf"]) if arr.size else 0.0,
        "max_dd_pct": float(metrics["max_dd_pct"]),
        "mean_pnl": float(metrics["mean_pnl"]),
        "win_rate": float((arr > 0).mean()) if arr.size else 0.0,
        "gross_profit": float(arr[arr > 0].sum()) if arr.size else 0.0,
        "gross_loss": float(-arr[arr <= 0].sum()) if arr.size else 0.0,
    }


def _value_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()}


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_branch(
    name: str,
    seed_dir: str,
    l3_dir: str,
    suffix: str,
) -> dict[str, Any]:
    chosen_path = os.path.join(seed_dir, "chosen_trades.pkl")
    l3_csv_path = os.path.join(l3_dir, f"layer3_trades_calibrated_{suffix}.csv")
    l3_cal_path = os.path.join(l3_dir, f"rolling_layer3_calibrated_{suffix}.json")
    l3_report_path = os.path.join(l3_dir, "rolling_layer3_report.json")

    for path in (chosen_path, l3_csv_path, l3_cal_path, l3_report_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    chosen = pd.read_pickle(chosen_path).copy()
    l3 = pd.read_csv(l3_csv_path)
    l3_cal = _load_json(l3_cal_path)
    l3_report = _load_json(l3_report_path)

    return {
        "name": name,
        "seed_dir": seed_dir,
        "l3_dir": l3_dir,
        "chosen": chosen,
        "l3": l3,
        "l3_cal": l3_cal,
        "l3_report": l3_report,
    }


def _train_report_for(branch: dict[str, Any], window: int) -> dict[str, Any] | None:
    for row in branch["l3_report"].get("train_reports", []):
        if int(row.get("window_idx", -1)) == int(window):
            return row
    return None


def _calibration_for(branch: dict[str, Any], window: int) -> dict[str, Any] | None:
    for row in branch["l3_cal"].get("per_window_calibration", []):
        if int(row.get("window_idx", -1)) == int(window):
            return row
    return None


def _entry_exit_join(branch: dict[str, Any], window: int) -> pd.DataFrame:
    chosen = branch["chosen"]
    l3 = branch["l3"]
    w_chosen = chosen.loc[chosen["window_idx"].astype(int) == int(window)].copy()
    w_l3 = l3.loc[l3["window_idx"].astype(int) == int(window)].copy()
    if w_chosen.empty:
        return w_chosen
    return w_chosen.merge(
        w_l3[["day", *EXIT_DETAIL_COLS]],
        left_on=["day", "bar_index"],
        right_on=["day", "entry_bar"],
        how="left",
    )


def _branch_window_summary(branch: dict[str, Any], window: int, equity: float) -> dict[str, Any]:
    joined = _entry_exit_join(branch, window)
    chosen = branch["chosen"].loc[branch["chosen"]["window_idx"].astype(int) == int(window)]
    l3 = branch["l3"].loc[branch["l3"]["window_idx"].astype(int) == int(window)]
    l3_metrics = agg_exit_metrics(l3, equity) if not l3.empty else {
        "pf": 0.0,
        "max_dd_pct": 0.0,
        "mean_pnl": 0.0,
        "trades": 0.0,
        "mean_bars_held": 0.0,
    }

    out: dict[str, Any] = {
        "branch": branch["name"],
        "window_idx": int(window),
        "entry_objective": _pnl_metrics(chosen.get("chosen_objective_pnl", pd.Series(dtype=float)), equity),
        "entry_time_stop_reference": _pnl_metrics(
            chosen.get("chosen_time_stop_pnl", pd.Series(dtype=float)),
            equity,
        ),
        "layer3": {
            **l3_metrics,
            "pnl": float(l3["exit_pnl"].sum()) if not l3.empty else 0.0,
            "win_rate": float((l3["exit_pnl"] > 0).mean()) if not l3.empty else 0.0,
        },
        "side_counts": _value_counts(chosen.get("chosen_side", pd.Series(dtype=object))),
        "time_bucket_counts": _value_counts(chosen.get("chosen_time_bucket", pd.Series(dtype=object))),
        "teacher_counts": _value_counts(chosen.get("teacher_any_triggered", pd.Series(dtype=object))),
        "trigger_counts": _value_counts(l3.get("trigger", pd.Series(dtype=object))),
        "mean_entry_bar": float(chosen["bar_index"].mean()) if not chosen.empty else None,
        "mean_decision_margin": float(chosen["decision_margin"].mean()) if not chosen.empty else None,
        "mean_clean_prob": float(chosen["pred_clean_entry_prob"].mean()) if not chosen.empty else None,
        "mean_stopout_risk": float(chosen["pred_stopout_risk"].mean()) if not chosen.empty else None,
        "l3_train_report": _train_report_for(branch, window),
        "l3_calibration": _calibration_for(branch, window),
    }
    if not joined.empty:
        out["worst_trades"] = _records(
            joined.sort_values("exit_pnl", na_position="last").head(10),
            [*ENTRY_DETAIL_COLS, *EXIT_DETAIL_COLS],
        )
    else:
        out["worst_trades"] = []
    return out


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    present = [c for c in columns if c in df.columns]
    return [_jsonable(row) for row in df[present].to_dict(orient="records")]


def _side_by_side_diff(
    left: dict[str, Any],
    right: dict[str, Any],
    window: int,
    equity: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    left_joined = _entry_exit_join(left, window)
    right_joined = _entry_exit_join(right, window)
    left_days = set(left_joined["day"].astype(str)) if not left_joined.empty else set()
    right_days = set(right_joined["day"].astype(str)) if not right_joined.empty else set()
    common_days = sorted(left_days & right_days)
    left_only_days = sorted(left_days - right_days)
    right_only_days = sorted(right_days - left_days)

    right_only = right_joined[right_joined["day"].astype(str).isin(right_only_days)].copy()
    left_only = left_joined[left_joined["day"].astype(str).isin(left_only_days)].copy()
    common_left = left_joined[left_joined["day"].astype(str).isin(common_days)].copy()
    common_right = right_joined[right_joined["day"].astype(str).isin(common_days)].copy()

    common_compare = pd.DataFrame()
    if common_days:
        common_compare = common_left.set_index("day").join(
            common_right.set_index("day"),
            lsuffix=f"_{left['name']}",
            rsuffix=f"_{right['name']}",
            how="inner",
        )
        common_compare["delta_exit_pnl"] = (
            common_compare[f"exit_pnl_{right['name']}"]
            - common_compare[f"exit_pnl_{left['name']}"]
        )
        common_compare = common_compare.reset_index()

    diff_rows: list[pd.DataFrame] = []
    if not right_only.empty:
        diff_rows.append(_only_rows(right_only, right["name"], f"{right['name']}_only"))
    if not left_only.empty:
        diff_rows.append(_only_rows(left_only, left["name"], f"{left['name']}_only"))
    if not common_compare.empty:
        diff_rows.append(_common_rows(common_compare, left["name"], right["name"]))
    diff_df = pd.concat(diff_rows, ignore_index=True, sort=False) if diff_rows else pd.DataFrame()

    right_only_losers = right_only[right_only.get("exit_pnl", pd.Series(dtype=float)) <= 0]
    summary = {
        "left_branch": left["name"],
        "right_branch": right["name"],
        "window_idx": int(window),
        "left_days": int(len(left_days)),
        "right_days": int(len(right_days)),
        "common_days": int(len(common_days)),
        "left_only_days": int(len(left_only_days)),
        "right_only_days": int(len(right_only_days)),
        "left_only_layer3": _pnl_metrics(left_only.get("exit_pnl", pd.Series(dtype=float)), equity),
        "right_only_layer3": _pnl_metrics(right_only.get("exit_pnl", pd.Series(dtype=float)), equity),
        "common_left_layer3": _pnl_metrics(common_left.get("exit_pnl", pd.Series(dtype=float)), equity),
        "common_right_layer3": _pnl_metrics(common_right.get("exit_pnl", pd.Series(dtype=float)), equity),
        "common_delta_pnl": (
            float(common_compare["delta_exit_pnl"].sum()) if not common_compare.empty else 0.0
        ),
        "right_only_worst": _records(
            right_only.sort_values("exit_pnl", na_position="last").head(15),
            [*ENTRY_DETAIL_COLS, *EXIT_DETAIL_COLS],
        ),
        "common_worst_deltas": _records(
            common_compare.sort_values("delta_exit_pnl").head(15)
            if not common_compare.empty else common_compare,
            [
                "day",
                f"bar_index_{left['name']}",
                f"bar_index_{right['name']}",
                f"chosen_side_{left['name']}",
                f"chosen_side_{right['name']}",
                f"chosen_strike_{left['name']}",
                f"chosen_strike_{right['name']}",
                f"decision_margin_{left['name']}",
                f"decision_margin_{right['name']}",
                f"exit_pnl_{left['name']}",
                f"exit_pnl_{right['name']}",
                "delta_exit_pnl",
                f"trigger_{left['name']}",
                f"trigger_{right['name']}",
                f"bars_held_{left['name']}",
                f"bars_held_{right['name']}",
                f"chosen_time_bucket_{left['name']}",
                f"chosen_time_bucket_{right['name']}",
            ],
        ),
        "right_only_loser_profile": _numeric_profile(
            right_only_losers,
            [
                "bar_index",
                "decision_margin",
                "pred_clean_entry_prob",
                "pred_stopout_risk",
                "sigma_pos",
                "omar_mid_pos_units",
                "chosen_time_stop_pnl",
                "chosen_objective_pnl",
                "bars_held",
            ],
        ),
    }
    return summary, diff_df


def _only_rows(df: pd.DataFrame, branch_name: str, diff_type: str) -> pd.DataFrame:
    cols = [
        "day",
        "bar_index",
        "chosen_side",
        "chosen_strike",
        "decision_margin",
        "chosen_objective_pnl",
        "chosen_time_stop_pnl",
        "exit_pnl",
        "trigger",
        "bars_held",
        "pred_clean_entry_prob",
        "pred_stopout_risk",
        "chosen_time_bucket",
        "teacher_any_triggered",
        "sigma_pos",
        "omar_mid_pos_units",
        "last10_break_state",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out.insert(0, "diff_type", diff_type)
    out.insert(1, "branch", branch_name)
    return out


def _common_rows(df: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "diff_type": "common_day_delta",
                "branch": f"{right_name}_minus_{left_name}",
                "day": row["day"],
                f"{left_name}_bar_index": row.get(f"bar_index_{left_name}"),
                f"{right_name}_bar_index": row.get(f"bar_index_{right_name}"),
                f"{left_name}_side": row.get(f"chosen_side_{left_name}"),
                f"{right_name}_side": row.get(f"chosen_side_{right_name}"),
                f"{left_name}_strike": row.get(f"chosen_strike_{left_name}"),
                f"{right_name}_strike": row.get(f"chosen_strike_{right_name}"),
                f"{left_name}_decision_margin": row.get(f"decision_margin_{left_name}"),
                f"{right_name}_decision_margin": row.get(f"decision_margin_{right_name}"),
                f"{left_name}_exit_pnl": row.get(f"exit_pnl_{left_name}"),
                f"{right_name}_exit_pnl": row.get(f"exit_pnl_{right_name}"),
                "delta_exit_pnl": row.get("delta_exit_pnl"),
                f"{left_name}_trigger": row.get(f"trigger_{left_name}"),
                f"{right_name}_trigger": row.get(f"trigger_{right_name}"),
                f"{left_name}_bars_held": row.get(f"bars_held_{left_name}"),
                f"{right_name}_bars_held": row.get(f"bars_held_{right_name}"),
                f"{left_name}_bucket": row.get(f"chosen_time_bucket_{left_name}"),
                f"{right_name}_bucket": row.get(f"chosen_time_bucket_{right_name}"),
            }
        )
    return pd.DataFrame(rows)


def _numeric_profile(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if df.empty:
        return out
    for col in columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        out[col] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def main() -> int:
    args = parse_args()
    branch_specs = args.branch if args.branch else DEFAULT_BRANCHES
    if len(branch_specs) < 2:
        raise ValueError("Need at least two --branch entries to compare.")

    branches = [
        _load_branch(
            str(name),
            os.path.abspath(seed_dir),
            os.path.abspath(l3_dir),
            args.calibration_suffix,
        )
        for name, seed_dir, l3_dir in branch_specs
    ]
    summaries = [
        _branch_window_summary(branch, args.window, args.equity)
        for branch in branches
    ]
    diff_summary, diff_df = _side_by_side_diff(
        branches[0],
        branches[1],
        args.window,
        args.equity,
    )

    payload = {
        "meta": {
            "window_idx": int(args.window),
            "equity": float(args.equity),
            "calibration_suffix": args.calibration_suffix,
            "branches": [
                {
                    "name": branch["name"],
                    "seed_dir": branch["seed_dir"],
                    "l3_dir": branch["l3_dir"],
                }
                for branch in branches
            ],
        },
        "branch_summaries": summaries,
        "first_two_branch_diff": diff_summary,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")

    if not diff_df.empty:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        diff_df.to_csv(args.out_csv, index=False)

    print(f"Window {args.window} divergence diagnostic", flush=True)
    for summary in summaries:
        l3 = summary["layer3"]
        train = summary.get("l3_train_report") or {}
        print(
            f"  {summary['branch']}: trades={int(l3['trades'])} "
            f"PF={float(l3['pf']):.3f} PnL=${float(l3['pnl']):+.0f} "
            f"L3mode={train.get('mode', 'unknown')}",
            flush=True,
        )
    print(
        f"  {branches[1]['name']} only: trades={diff_summary['right_only_layer3']['trades']} "
        f"PF={diff_summary['right_only_layer3']['pf']:.3f} "
        f"PnL=${diff_summary['right_only_layer3']['pnl']:+.0f}",
        flush=True,
    )
    print(
        f"  common-day delta ({branches[1]['name']} - {branches[0]['name']}): "
        f"${diff_summary['common_delta_pnl']:+.0f}",
        flush=True,
    )
    print(f"Saved JSON: {args.out_json}", flush=True)
    if not diff_df.empty:
        print(f"Saved CSV:  {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
