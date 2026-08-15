"""EXP_2026_05_22_FULL_ACTION_HISTORY_FEATURE_REPAIR_V1.

Historically Protocol211. This is a dataset repair/enrichment experiment, not a
model promotion.

It adds the missing PAPER_DEFAULT_PROTOCOL101-style causal short-history
features to the full-action surface-edge dataset. These features are computed
only from earlier decision events in the same session.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol101_event_history_policy import HISTORY_FEATURE_COLUMNS


ROLE_LABEL = "EXP_2026_05_22_FULL_ACTION_HISTORY_FEATURE_REPAIR_V1"
HISTORICAL_ID = "Protocol211"
DEFAULT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_189_full_coverage_surface_edge_enrichment/protocol185_full_action_with_surface_edge.parquet")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset)
    if dataset.empty:
        raise ValueError(f"empty dataset: {args.dataset}")
    enriched = add_history_features(dataset)
    out_path = args.out_dir / "full_action_surface_edge_with_history.parquet"
    enriched.to_parquet(out_path, index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / dataset repair",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "other_baseline_label": "CHALLENGER_FULL_ACTION_SURFACE_EDGE_V1",
        "data_used": str(args.dataset),
        "enriched_dataset": str(out_path),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "rows": int(len(enriched)),
        "history_feature_columns": HISTORY_FEATURE_COLUMNS,
        "history_nan_counts": {column: int(pd.to_numeric(enriched[column], errors="coerce").isna().sum()) for column in HISTORY_FEATURE_COLUMNS},
        "edge_alias_added": "edge" in enriched.columns,
        "decision": "ready_for_full_action_history_policy_test",
        "next_experiment": "Rerun the two-stage full-action policy with surface-edge plus repaired causal history features.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "rows": payload["rows"], "dataset": str(out_path)}, indent=2, sort_keys=True))
    return 0


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["decision_dt"] = pd.to_datetime(out["decision_dt"], utc=True, errors="coerce")
    out["right"] = out["right"].astype(str)
    if "edge" not in out.columns:
        out["edge"] = pd.to_numeric(out.get("surface_edge"), errors="coerce")
    out["edge"] = pd.to_numeric(out["edge"], errors="coerce").fillna(0.0)
    for column in ["entry_gamma", "entry_theta_burden", "entry_spread_over_mid"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    event_summary = build_event_summary(out)
    history = build_history(event_summary)
    out = out.merge(history, on=["split", "session", "decision_dt"], how="left", validate="many_to_one")
    for column in HISTORY_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default_history_value(column))
    return out


def build_event_summary(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame[["split", "session", "decision_dt", "right", "edge", "entry_gamma", "entry_theta_burden", "entry_spread_over_mid"]].copy()
    working["is_call"] = (working["right"] == "C").astype(float)
    working["is_put"] = (working["right"] == "P").astype(float)
    working["call_edge"] = np.where(working["right"].eq("C"), working["edge"], np.nan)
    working["put_edge"] = np.where(working["right"].eq("P"), working["edge"], np.nan)
    grouped = working.groupby(["split", "session", "decision_dt"], sort=True)
    summary = grouped.agg(
        candidate_count=("edge", "size"),
        max_edge=("edge", "max"),
        mean_edge=("edge", "mean"),
        max_gamma=("entry_gamma", "max"),
        mean_theta_burden=("entry_theta_burden", "mean"),
        min_spread_over_mid=("entry_spread_over_mid", "min"),
        call_count=("is_call", "sum"),
        put_count=("is_put", "sum"),
        max_call_edge=("call_edge", "max"),
        max_put_edge=("put_edge", "max"),
    ).reset_index()
    summary["max_call_edge"] = summary["max_call_edge"].fillna(0.0)
    summary["max_put_edge"] = summary["max_put_edge"].fillna(0.0)
    summary["call_minus_put_edge"] = summary["max_call_edge"] - summary["max_put_edge"]
    return summary.sort_values(["split", "session", "decision_dt"]).reset_index(drop=True)


def build_history(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = [
        "candidate_count",
        "max_edge",
        "mean_edge",
        "max_gamma",
        "mean_theta_burden",
        "min_spread_over_mid",
        "call_count",
        "put_count",
        "call_minus_put_edge",
    ]
    for (_, _), group in summary.groupby(["split", "session"], sort=True):
        group = group.sort_values("decision_dt").reset_index(drop=True)
        previous = group[metric_columns].shift(1)
        previous = previous.fillna(empty_summary())
        shifted = group[metric_columns].shift(1)
        rolling_mean = shifted.rolling(3, min_periods=1).mean().fillna(empty_summary())
        rolling_max = shifted.rolling(3, min_periods=1).max().fillna(empty_summary())
        rolling_min = shifted.rolling(3, min_periods=1).min().fillna(empty_summary())
        minutes_since_prev = group["decision_dt"].diff().dt.total_seconds().div(60.0).fillna(999.0)
        item = pd.DataFrame(
            {
                "split": group["split"],
                "session": group["session"],
                "decision_dt": group["decision_dt"],
                "hist_events_seen": np.minimum(np.arange(len(group), dtype=float), 50.0),
                "hist_minutes_since_prev_event": minutes_since_prev.astype(float),
                "hist_prev_candidate_count": previous["candidate_count"].astype(float),
                "hist_prev_max_edge": previous["max_edge"].astype(float),
                "hist_prev_mean_edge": previous["mean_edge"].astype(float),
                "hist_prev_max_gamma": previous["max_gamma"].astype(float),
                "hist_prev_mean_theta_burden": previous["mean_theta_burden"].astype(float),
                "hist_prev_min_spread_over_mid": previous["min_spread_over_mid"].astype(float),
                "hist_prev_call_count": previous["call_count"].astype(float),
                "hist_prev_put_count": previous["put_count"].astype(float),
                "hist_prev_call_minus_put_edge": previous["call_minus_put_edge"].astype(float),
                "hist_roll3_candidate_count_mean": rolling_mean["candidate_count"].astype(float),
                "hist_roll3_max_edge": rolling_max["max_edge"].astype(float),
                "hist_roll3_mean_edge": rolling_mean["mean_edge"].astype(float),
                "hist_roll3_max_gamma": rolling_max["max_gamma"].astype(float),
                "hist_roll3_mean_theta_burden": rolling_mean["mean_theta_burden"].astype(float),
                "hist_roll3_min_spread_over_mid": rolling_min["min_spread_over_mid"].astype(float),
                "hist_roll3_call_minus_put_edge": rolling_mean["call_minus_put_edge"].astype(float),
            }
        )
        rows.append(item)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["split", "session", "decision_dt", *HISTORY_FEATURE_COLUMNS])


def empty_summary() -> dict[str, float]:
    return {
        "candidate_count": 0.0,
        "max_edge": 0.0,
        "mean_edge": 0.0,
        "max_gamma": 0.0,
        "mean_theta_burden": 0.0,
        "min_spread_over_mid": 0.0,
        "call_count": 0.0,
        "put_count": 0.0,
        "call_minus_put_edge": 0.0,
    }


def default_history_value(column: str) -> float:
    if column == "hist_minutes_since_prev_event":
        return 999.0
    return 0.0


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        f"Does it change the paper-trading default: {'yes' if payload['changes_paper_default'] else 'no'}",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Other baseline: {payload['other_baseline_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Outputs",
        "",
        f"- Enriched dataset: `{payload['enriched_dataset']}`",
        f"- Summary: `{path.parent / 'summary.json'}`",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

