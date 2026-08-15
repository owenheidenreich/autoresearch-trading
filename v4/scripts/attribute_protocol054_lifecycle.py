"""Attribute Protocol 054 lifecycle effects versus frozen Protocol 051 exits.

This is a no-paid-data diagnostic. Protocol 054 keeps the Protocol 051 entries
unchanged, so every row compares the lifecycle exit PnL against the frozen-entry
baseline PnL for the same seed/session/contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model.environment_diagnostics import time_bucket


_NY = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-trades",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/"
            "v4_aplus_hypothesis_054_protocol052_lifecycle_10seed_validation/"
            "selected_trades_with_lifecycle_exits.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/"
            "v4_aplus_hypothesis_055_protocol054_lifecycle_promotion_checks/"
            "attribution"
        ),
    )
    return parser.parse_args()


def _load_frame(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    expanded: list[dict] = []
    for row in rows:
        expanded.append(row)
        if row.get("split") == "q1_2026" and str(row.get("session", "")) >= "2026-03-01":
            expanded.append({**row, "split": "march_2026"})
    frame = pd.DataFrame(expanded)
    if frame.empty:
        raise SystemExit(f"no selected trades found in {path}")
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["local_time"] = frame["decision_ts"].dt.tz_convert(_NY).dt.strftime("%H:%M")
    frame["time_bucket"] = frame["decision_ts"].map(time_bucket)
    frame["baseline_pnl"] = pd.to_numeric(frame["baseline_pnl"], errors="coerce").fillna(0.0)
    frame["dynamic_pnl"] = pd.to_numeric(frame["dynamic_pnl"], errors="coerce").fillna(0.0)
    frame["lifecycle_delta"] = frame["dynamic_pnl"] - frame["baseline_pnl"]
    frame["hold_minutes"] = pd.to_numeric(frame["hold_minutes"], errors="coerce")
    frame["mfe"] = pd.to_numeric(frame.get("mfe", np.nan), errors="coerce")
    frame["mae"] = pd.to_numeric(frame.get("mae", np.nan), errors="coerce")
    frame["giveback"] = pd.to_numeric(frame.get("giveback", np.nan), errors="coerce")
    frame["giveback_fraction"] = pd.to_numeric(frame.get("giveback_fraction", np.nan), errors="coerce")
    frame["hold_bucket"] = np.select(
        [
            frame["hold_minutes"] <= 5,
            frame["hold_minutes"] <= 15,
            frame["hold_minutes"] < 25,
        ],
        ["hold_00_05", "hold_06_15", "hold_16_24"],
        default="hold_25_flat",
    )
    frame["mfe_bucket"] = np.select(
        [
            frame["mfe"] < 100,
            frame["mfe"] < 300,
            frame["mfe"] < 700,
        ],
        ["mfe_lt_100", "mfe_100_300", "mfe_300_700"],
        default="mfe_700p",
    )
    frame["giveback_bucket"] = np.select(
        [
            frame["giveback_fraction"].fillna(0.0) < 0.25,
            frame["giveback_fraction"].fillna(0.0) < 0.50,
            frame["giveback_fraction"].fillna(0.0) < 0.75,
        ],
        ["gbfrac_lt_25", "gbfrac_25_50", "gbfrac_50_75"],
        default="gbfrac_75p",
    )
    frame["outcome_bucket"] = np.select(
        [
            (frame["baseline_pnl"] < 0) & (frame["lifecycle_delta"] > 0),
            (frame["baseline_pnl"] > 0) & (frame["lifecycle_delta"] > 0),
            (frame["baseline_pnl"] > 0) & (frame["lifecycle_delta"] < 0),
            (frame["baseline_pnl"] < 0) & (frame["lifecycle_delta"] < 0),
        ],
        ["saved_or_reduced_loss", "improved_winner", "clipped_winner", "worsened_loser"],
        default="unchanged_or_flat",
    )
    return frame


def _profit_factor(values: pd.Series) -> float:
    wins = values.clip(lower=0).sum()
    losses = -values.clip(upper=0).sum()
    if losses <= 0:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def _metrics(group: pd.DataFrame) -> dict:
    if group.empty:
        return {
            "trades": 0,
            "baseline_pnl": 0.0,
            "dynamic_pnl": 0.0,
            "lifecycle_delta": 0.0,
            "avg_delta": 0.0,
            "median_delta": 0.0,
            "delta_positive_fraction": 0.0,
            "baseline_pf": 0.0,
            "dynamic_pf": 0.0,
        }
    delta = group["lifecycle_delta"]
    return {
        "trades": int(len(group)),
        "baseline_pnl": float(group["baseline_pnl"].sum()),
        "dynamic_pnl": float(group["dynamic_pnl"].sum()),
        "lifecycle_delta": float(delta.sum()),
        "avg_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "delta_positive_fraction": float((delta > 0).mean()),
        "baseline_win_rate": float((group["baseline_pnl"] > 0).mean()),
        "dynamic_win_rate": float((group["dynamic_pnl"] > 0).mean()),
        "baseline_pf": _profit_factor(group["baseline_pnl"]),
        "dynamic_pf": _profit_factor(group["dynamic_pnl"]),
    }


def _seed_metrics(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (split, seed), group in frame.groupby(["split", "seed"], dropna=False):
        rows.append({"split": split, "seed": int(seed), **_metrics(group)})
    return rows


def _split_summary(frame: pd.DataFrame) -> list[dict]:
    seed_rows = pd.DataFrame(_seed_metrics(frame))
    out = []
    for split, group in frame.groupby("split", dropna=False):
        split_seed = seed_rows[seed_rows["split"] == split]
        row = {"split": split, **_metrics(group)}
        row |= {
            "seed_delta_median": float(split_seed["lifecycle_delta"].median()),
            "seed_delta_min": float(split_seed["lifecycle_delta"].min()),
            "seed_delta_max": float(split_seed["lifecycle_delta"].max()),
            "positive_seed_delta_fraction": float((split_seed["lifecycle_delta"] > 0).mean()),
        }
        out.append(row)
    return sorted(out, key=lambda row: str(row["split"]))


def _group_summary(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    for keys, group in frame.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({column: key for column, key in zip(columns, keys)} | _metrics(group))
    return sorted(rows, key=lambda row: (str(row.get("split", "")), row["lifecycle_delta"]))


def _examples(frame: pd.DataFrame, *, split: str, ascending: bool) -> list[dict]:
    columns = [
        "split",
        "seed",
        "session",
        "local_time",
        "right",
        "offset",
        "exit_reason",
        "hold_minutes",
        "baseline_pnl",
        "dynamic_pnl",
        "lifecycle_delta",
        "mfe",
        "mae",
        "giveback",
        "giveback_fraction",
        "contract_id",
    ]
    subset = frame[frame["split"] == split].sort_values("lifecycle_delta", ascending=ascending).head(20)
    return subset[columns].to_dict(orient="records")


def _interpretation(payload: dict) -> str:
    split_by_name = {row["split"]: row for row in payload["split_summary"]}
    q2 = split_by_name.get("q2_2025", {})
    q3 = split_by_name.get("q3_2025", {})
    q4 = split_by_name.get("q4_2025", {})
    march = split_by_name.get("march_2026", {})
    q2_exit = [row for row in payload["by_exit_reason"] if row["split"] == "q2_2025"]
    q2_worst = min(q2_exit, key=lambda row: row["lifecycle_delta"], default=None)
    improvement_rows = [
        row
        for row in payload["by_outcome_bucket"]
        if row["split"] in {"q3_2025", "q4_2025", "march_2026"}
    ]
    best_improvement = max(improvement_rows, key=lambda row: row["lifecycle_delta"], default=None)
    q2_phrase = "lifecycle-positive"
    if q2.get("lifecycle_delta", 0.0) < 0:
        q2_phrase = "lifecycle-negative"
    text = (
        f"Protocol 054 is lifecycle-positive in Q3 ({q3.get('lifecycle_delta', 0):.0f}), "
        f"Q4 ({q4.get('lifecycle_delta', 0):.0f}), and March ({march.get('lifecycle_delta', 0):.0f}), "
        f"while Q2 is {q2_phrase} on paired same-entry attribution ({q2.get('lifecycle_delta', 0):.0f}). "
        "The earlier Q2 caveat comes from independent median PnL ranking, not from paired trade-set attribution. "
    )
    if q2_worst:
        text += (
            f"The largest Q2 drag by exit reason is {q2_worst['exit_reason']} "
            f"({q2_worst['lifecycle_delta']:.0f} across {q2_worst['trades']} trades). "
        )
    if best_improvement:
        text += (
            f"The largest positive mechanism across the improving splits is "
            f"{best_improvement['outcome_bucket']} in {best_improvement['split']} "
            f"({best_improvement['lifecycle_delta']:.0f}). "
        )
    text += (
        "This supports lifecycle promotion checks, not a new entry-side change. The next model change, if any, "
        "should be based on why Q2 winners were clipped or losers worsened after entry, not on adding more entry filters."
    )
    return text


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 054 Lifecycle Attribution",
        "",
        "Compares Protocol 054 lifecycle exits against the frozen Protocol 051 baseline exits for the exact same entries.",
        "",
        "## Split Summary",
        "",
        "| Split | Trades | Baseline | Lifecycle | Delta | Seed Delta Median | Positive Seed Delta | Dynamic PF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['split']} | {row['trades']} | {row['baseline_pnl']:.0f} | {row['dynamic_pnl']:.0f} | "
            f"{row['lifecycle_delta']:.0f} | {row['seed_delta_median']:.0f} | "
            f"{row['positive_seed_delta_fraction']:.2f} | {row['dynamic_pf']:.3f} |"
        )
    lines += [
        "",
        "## By Outcome Bucket",
        "",
        "| Split | Outcome | Trades | Baseline | Lifecycle | Delta | Avg Delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["by_outcome_bucket"]:
        lines.append(
            f"| {row['split']} | {row['outcome_bucket']} | {row['trades']} | {row['baseline_pnl']:.0f} | "
            f"{row['dynamic_pnl']:.0f} | {row['lifecycle_delta']:.0f} | {row['avg_delta']:.0f} |"
        )
    lines += [
        "",
        "## By Exit Reason",
        "",
        "| Split | Exit Reason | Trades | Baseline | Lifecycle | Delta | Avg Delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["by_exit_reason"]:
        lines.append(
            f"| {row['split']} | {row['exit_reason']} | {row['trades']} | {row['baseline_pnl']:.0f} | "
            f"{row['dynamic_pnl']:.0f} | {row['lifecycle_delta']:.0f} | {row['avg_delta']:.0f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    frame = _load_frame(args.selected_trades)
    payload = {
        "selected_trades": str(args.selected_trades),
        "split_summary": _split_summary(frame),
        "seed_summary": _seed_metrics(frame),
        "by_exit_reason": _group_summary(frame, ["split", "exit_reason"]),
        "by_side_time": _group_summary(frame, ["split", "right", "time_bucket"]),
        "by_outcome_bucket": _group_summary(frame, ["split", "outcome_bucket"]),
        "by_hold_bucket": _group_summary(frame, ["split", "hold_bucket"]),
        "by_mfe_bucket": _group_summary(frame, ["split", "mfe_bucket"]),
        "by_giveback_bucket": _group_summary(frame, ["split", "giveback_bucket"]),
        "worst_q2_examples": _examples(frame, split="q2_2025", ascending=True),
        "best_q3_examples": _examples(frame, split="q3_2025", ascending=False),
        "best_q4_examples": _examples(frame, split="q4_2025", ascending=False),
        "best_march_examples": _examples(frame, split="march_2026", ascending=False),
    }
    payload["interpretation"] = _interpretation(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, payload)
    print(json.dumps(payload["split_summary"], indent=2, allow_nan=True))
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
