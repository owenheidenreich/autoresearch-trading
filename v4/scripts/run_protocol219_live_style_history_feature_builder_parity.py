"""RUNTIME_FULL_ACTION_HISTORY_FEATURE_BUILDER_PARITY_V1.

Historically Protocol219. This runtime/parity harness verifies that the
causal short-history features used by CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1
can be built from a live-style decision stream.

It does not train a model, download paid data, call a broker endpoint, or change
PAPER_DEFAULT_PROTOCOL101.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol101_event_history_policy import HISTORY_FEATURE_COLUMNS


ROLE_LABEL = "RUNTIME_FULL_ACTION_HISTORY_FEATURE_BUILDER_PARITY_V1"
HISTORICAL_ID = "Protocol219"
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_219_live_style_history_feature_builder_parity"
)

BASE_COLUMNS = [
    "split",
    "session",
    "decision_dt",
    "right",
    "edge",
    "surface_edge",
    "entry_gamma",
    "entry_theta_burden",
    "entry_spread_over_mid",
]
METRIC_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-events", type=int, default=0, help="0 means all events after optional split filter.")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame = load_dataset(args.dataset, splits=args.split)
    expected = expected_history_by_event(frame)
    if args.max_events and args.max_events > 0:
        keep = expected.sort_values(["split", "session", "decision_dt"]).head(int(args.max_events))
        frame = frame.merge(keep[["split", "session", "decision_dt"]], on=["split", "session", "decision_dt"], how="inner")
        expected = expected.merge(keep[["split", "session", "decision_dt"]], on=["split", "session", "decision_dt"], how="inner")

    built = build_live_style_history(frame)
    compared = compare_history(expected, built, tolerance=float(args.tolerance))
    mismatches = compared[compared["mismatch_count"] > 0].copy()
    mismatch_path = args.out_dir / "mismatch_rows.csv"
    mismatches.head(1000).to_csv(mismatch_path, index=False)
    compared.head(5000).to_csv(args.out_dir / "parity_sample.csv", index=False)

    elapsed = time.perf_counter() - started
    split_rows = compared.groupby("split", sort=True).agg(
        events=("decision_dt", "size"),
        mismatch_events=("mismatch_count", lambda values: int((values > 0).sum())),
        max_abs_diff=("max_abs_diff", "max"),
    ).reset_index()
    split_rows["pass"] = split_rows["mismatch_events"].eq(0)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / parity harness",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline_label": "historical Protocol211 parquet feature build",
        "data_used": str(args.dataset),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "events_checked": int(len(compared)),
        "candidate_rows_checked": int(len(frame)),
        "mismatch_events": int((compared["mismatch_count"] > 0).sum()),
        "mismatch_columns": mismatch_columns(compared),
        "max_abs_diff": float(compared["max_abs_diff"].max()) if len(compared) else 0.0,
        "elapsed_seconds": round(float(elapsed), 6),
        "events_per_second": round(float(len(compared) / elapsed), 6) if elapsed > 0 else None,
        "splits": split_rows.to_dict("records"),
        "history_feature_columns": HISTORY_FEATURE_COLUMNS,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "mismatch_rows": str(mismatch_path),
            "parity_sample": str(args.out_dir / "parity_sample.csv"),
        },
        "decision": "live_style_history_feature_builder_parity_passed"
        if int((compared["mismatch_count"] > 0).sum()) == 0
        else "live_style_history_feature_builder_parity_failed",
        "next_experiment": (
            "Wire this stream-safe feature builder into a no-order challenger shadow path; "
            "keep PAPER_DEFAULT_PROTOCOL101 unchanged until that live-safe path passes."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "events": payload["events_checked"], "mismatches": payload["mismatch_events"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0 if payload["decision"].endswith("_passed") else 1


def load_dataset(path: Path, *, splits: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    columns = list(dict.fromkeys([*BASE_COLUMNS, *HISTORY_FEATURE_COLUMNS]))
    frame = pd.read_parquet(path, columns=columns)
    if "edge" not in frame.columns:
        frame["edge"] = pd.to_numeric(frame["surface_edge"], errors="coerce")
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["split"] = frame["split"].astype(str)
    frame["session"] = frame["session"].astype(str)
    frame["right"] = frame["right"].astype(str)
    if splits:
        allowed = {str(split) for split in splits}
        frame = frame[frame["split"].isin(allowed)].copy()
    if frame.empty:
        raise ValueError("empty frame after split filter")
    for column in ["edge", "entry_gamma", "entry_theta_burden", "entry_spread_over_mid"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in HISTORY_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default_history_value(column))
    return frame


def expected_history_by_event(frame: pd.DataFrame) -> pd.DataFrame:
    expected = (
        frame[["split", "session", "decision_dt", *HISTORY_FEATURE_COLUMNS]]
        .groupby(["split", "session", "decision_dt"], sort=True)
        .first()
        .reset_index()
        .sort_values(["split", "session", "decision_dt"])
        .reset_index(drop=True)
    )
    return expected


@dataclass
class LiveHistoryState:
    seen: int = 0
    previous_time: pd.Timestamp | None = None
    previous_metrics: dict[str, float] = field(default_factory=lambda: empty_summary())
    recent_metrics: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=3))

    def emit(self, decision_time: pd.Timestamp) -> dict[str, float]:
        if self.previous_time is None:
            minutes_since_prev = 999.0
        else:
            minutes_since_prev = float((decision_time - self.previous_time).total_seconds() / 60.0)
        roll_mean = rolling_metric(self.recent_metrics, "mean")
        roll_max = rolling_metric(self.recent_metrics, "max")
        roll_min = rolling_metric(self.recent_metrics, "min")
        return {
            "hist_events_seen": float(min(self.seen, 50)),
            "hist_minutes_since_prev_event": minutes_since_prev,
            "hist_prev_candidate_count": self.previous_metrics["candidate_count"],
            "hist_prev_max_edge": self.previous_metrics["max_edge"],
            "hist_prev_mean_edge": self.previous_metrics["mean_edge"],
            "hist_prev_max_gamma": self.previous_metrics["max_gamma"],
            "hist_prev_mean_theta_burden": self.previous_metrics["mean_theta_burden"],
            "hist_prev_min_spread_over_mid": self.previous_metrics["min_spread_over_mid"],
            "hist_prev_call_count": self.previous_metrics["call_count"],
            "hist_prev_put_count": self.previous_metrics["put_count"],
            "hist_prev_call_minus_put_edge": self.previous_metrics["call_minus_put_edge"],
            "hist_roll3_candidate_count_mean": roll_mean["candidate_count"],
            "hist_roll3_max_edge": roll_max["max_edge"],
            "hist_roll3_mean_edge": roll_mean["mean_edge"],
            "hist_roll3_max_gamma": roll_max["max_gamma"],
            "hist_roll3_mean_theta_burden": roll_mean["mean_theta_burden"],
            "hist_roll3_min_spread_over_mid": roll_min["min_spread_over_mid"],
            "hist_roll3_call_minus_put_edge": roll_mean["call_minus_put_edge"],
        }

    def observe(self, decision_time: pd.Timestamp, metrics: dict[str, float]) -> None:
        self.seen += 1
        self.previous_time = decision_time
        self.previous_metrics = dict(metrics)
        self.recent_metrics.append(dict(metrics))


def build_live_style_history(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sort_columns = ["split", "session", "decision_dt"]
    for (split, session), session_rows in frame.sort_values(sort_columns).groupby(["split", "session"], sort=True):
        state = LiveHistoryState()
        for decision_time, event in session_rows.groupby("decision_dt", sort=True):
            decision_time = pd.Timestamp(decision_time)
            emitted = state.emit(decision_time)
            emitted.update({"split": split, "session": session, "decision_dt": decision_time})
            rows.append(emitted)
            state.observe(decision_time, summarize_event(event))
    return pd.DataFrame(rows).sort_values(["split", "session", "decision_dt"]).reset_index(drop=True)


def summarize_event(event: pd.DataFrame) -> dict[str, float]:
    edge = pd.to_numeric(event["edge"], errors="coerce").fillna(0.0)
    gamma = pd.to_numeric(event["entry_gamma"], errors="coerce").fillna(0.0)
    theta = pd.to_numeric(event["entry_theta_burden"], errors="coerce").fillna(0.0)
    spread = pd.to_numeric(event["entry_spread_over_mid"], errors="coerce").fillna(0.0)
    right = event["right"].astype(str)
    call_mask = right.eq("C")
    put_mask = right.eq("P")
    max_call_edge = float(edge[call_mask].max()) if bool(call_mask.any()) else 0.0
    max_put_edge = float(edge[put_mask].max()) if bool(put_mask.any()) else 0.0
    return {
        "candidate_count": float(len(event)),
        "max_edge": float(edge.max()) if len(edge) else 0.0,
        "mean_edge": float(edge.mean()) if len(edge) else 0.0,
        "max_gamma": float(gamma.max()) if len(gamma) else 0.0,
        "mean_theta_burden": float(theta.mean()) if len(theta) else 0.0,
        "min_spread_over_mid": float(spread.min()) if len(spread) else 0.0,
        "call_count": float(call_mask.sum()),
        "put_count": float(put_mask.sum()),
        "call_minus_put_edge": max_call_edge - max_put_edge,
    }


def rolling_metric(recent: deque[dict[str, float]], mode: str) -> dict[str, float]:
    if not recent:
        return empty_summary()
    out: dict[str, float] = {}
    for column in METRIC_COLUMNS:
        values = [float(item[column]) for item in recent]
        if mode == "mean":
            out[column] = float(np.mean(values))
        elif mode == "max":
            out[column] = float(np.max(values))
        elif mode == "min":
            out[column] = float(np.min(values))
        else:
            raise ValueError(mode)
    return out


def compare_history(expected: pd.DataFrame, built: pd.DataFrame, *, tolerance: float) -> pd.DataFrame:
    merged = expected.merge(
        built,
        on=["split", "session", "decision_dt"],
        how="outer",
        suffixes=("_expected", "_live"),
        indicator=True,
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        item = {
            "split": row["split"],
            "session": row["session"],
            "decision_dt": row["decision_dt"],
            "merge_status": row["_merge"],
            "mismatch_columns": "",
            "mismatch_count": 0,
            "max_abs_diff": 0.0,
        }
        mismatched: list[str] = []
        max_diff = 0.0
        if item["merge_status"] != "both":
            mismatched = list(HISTORY_FEATURE_COLUMNS)
            max_diff = float("inf")
        else:
            for column in HISTORY_FEATURE_COLUMNS:
                expected_value = float(row[f"{column}_expected"])
                live_value = float(row[f"{column}_live"])
                diff = abs(expected_value - live_value)
                max_diff = max(max_diff, diff)
                if not np.isclose(expected_value, live_value, atol=tolerance, rtol=tolerance):
                    mismatched.append(column)
        item["mismatch_columns"] = ",".join(mismatched)
        item["mismatch_count"] = len(mismatched)
        item["max_abs_diff"] = max_diff
        rows.append(item)
    return pd.DataFrame(rows)


def mismatch_columns(compared: pd.DataFrame) -> dict[str, int]:
    counts = {column: 0 for column in HISTORY_FEATURE_COLUMNS}
    for value in compared.loc[compared["mismatch_count"] > 0, "mismatch_columns"].astype(str):
        for column in [part for part in value.split(",") if part]:
            counts[column] = counts.get(column, 0) + 1
    return {column: int(count) for column, count in counts.items() if count}


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
    return 999.0 if column == "hist_minutes_since_prev_event" else 0.0


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Baseline being compared: {payload['baseline_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Parity Result",
        "",
        f"- Events checked: `{payload['events_checked']}`",
        f"- Candidate rows checked: `{payload['candidate_rows_checked']}`",
        f"- Mismatch events: `{payload['mismatch_events']}`",
        f"- Max absolute diff: `{payload['max_abs_diff']}`",
        f"- Builder throughput: `{payload['events_per_second']}` events/sec",
        "",
        "## Split Coverage",
        "",
        "| split | events | mismatch_events | max_abs_diff | pass |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["splits"]:
        lines.append(
            f"| {row['split']} | {int(row['events'])} | {int(row['mismatch_events'])} | "
            f"{float(row['max_abs_diff']):.12g} | {bool(row['pass'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Mismatch rows: `{payload['outputs']['mismatch_rows']}`",
            f"- Parity sample: `{payload['outputs']['parity_sample']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
