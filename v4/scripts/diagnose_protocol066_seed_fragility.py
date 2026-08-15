"""Diagnose Protocol 066 seed-level fragility.

Protocol 066 passed the 10-seed median gate, but Q3 had one seed with negative
delta versus Protocol 054 and March had two. This diagnostic does not train,
tune, or select anything; it attributes those negative-delta seeds by side,
time bucket, exit reason, and residual override path features.
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
_DEFAULT_SELECTED = Path(
    "v4/audit/autoresearch/"
    "v4_aplus_hypothesis_066_protocol065_10seed_validation/"
    "selected_trades_sequence_exits.json"
)
_DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/"
    "v4_aplus_hypothesis_067_protocol066_seed_fragility_diagnostic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-trades", type=Path, default=_DEFAULT_SELECTED)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    return parser.parse_args()


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _load_selected(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no selected rows found in {path}")
    frame["session"] = frame["session"].astype(str)
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["local_time"] = frame["decision_ts"].dt.tz_convert(_NY).dt.strftime("%H:%M")
    frame["time_bucket"] = frame["decision_ts"].map(time_bucket)
    _numeric(
        frame,
        [
            "seed",
            "entry_seed",
            "offset",
            "candidate_pnl",
            "protocol054_pnl",
            "delta_vs_protocol054",
            "candidate_exit_step",
            "protocol054_exit_step",
            "predicted_continuation_value",
            "override_threshold",
            "predicted_recovery_probability",
            "predicted_decay_probability",
            "current_pnl_at_exit",
            "mfe_to_exit",
            "mae_to_exit",
            "future_max_delta_at_exit",
            "future_min_delta_at_exit",
        ],
    )
    expanded = [frame.assign(eval_split=frame["split"])]
    march = frame[frame["split"].eq("q1_2026") & frame["session"].ge("2026-03-01")].copy()
    if not march.empty:
        march["eval_split"] = "march_2026"
        expanded.append(march)
    out = pd.concat(expanded, ignore_index=True)
    out["is_override"] = out["candidate_exit_reason"].eq("sequence_residual_override")
    out["exit_step_bucket"] = pd.cut(
        out["candidate_exit_step"].fillna(-1),
        bins=[-2, 0, 5, 12, 999],
        labels=["step_00", "step_01_05", "step_06_12", "step_13p"],
    ).astype(str)
    return out


def _profit_factor(values: pd.Series) -> float:
    wins = values.clip(lower=0).sum()
    losses = -values.clip(upper=0).sum()
    if losses <= 0:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def _seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, seed), group in frame.groupby(["eval_split", "seed"], dropna=False):
        candidate = group["candidate_pnl"].fillna(0.0)
        p054 = group["protocol054_pnl"].fillna(0.0)
        delta = candidate - p054
        rows.append(
            {
                "eval_split": split,
                "seed": int(seed),
                "rows": int(len(group)),
                "candidate_pnl": float(candidate.sum()),
                "protocol054_pnl": float(p054.sum()),
                "delta": float(delta.sum()),
                "candidate_pf": _profit_factor(candidate),
                "protocol054_pf": _profit_factor(p054),
                "override_fraction": float(group["is_override"].mean()),
                "avg_candidate_exit_step": float(group["candidate_exit_step"].mean()),
                "median_candidate_exit_step": float(group["candidate_exit_step"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["eval_split", "seed"]).reset_index(drop=True)


def _metrics(group: pd.DataFrame) -> dict:
    candidate = group["candidate_pnl"].fillna(0.0)
    p054 = group["protocol054_pnl"].fillna(0.0)
    delta = candidate - p054
    return {
        "rows": int(len(group)),
        "candidate_pnl": float(candidate.sum()),
        "protocol054_pnl": float(p054.sum()),
        "delta": float(delta.sum()),
        "median_delta": float(delta.median()) if len(delta) else 0.0,
        "override_fraction": float(group["is_override"].mean()) if len(group) else 0.0,
        "median_candidate_exit_step": float(group["candidate_exit_step"].median()) if len(group) else 0.0,
        "median_protocol054_exit_step": float(group["protocol054_exit_step"].median()) if len(group) else 0.0,
        "median_future_max_delta_at_exit": float(group["future_max_delta_at_exit"].median()) if len(group) else 0.0,
        "median_future_min_delta_at_exit": float(group["future_min_delta_at_exit"].median()) if len(group) else 0.0,
    }


def _group_summary(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    for keys, group in frame.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({column: str(key) for column, key in zip(columns, keys)} | _metrics(group))
    return sorted(rows, key=lambda row: (str(row.get("eval_split", "")), row["delta"]))


def _feature_contrast(frame: pd.DataFrame, negative_keys: set[tuple[str, int]]) -> list[dict]:
    features = [
        "candidate_exit_step",
        "protocol054_exit_step",
        "predicted_continuation_value",
        "override_threshold",
        "predicted_recovery_probability",
        "predicted_decay_probability",
        "current_pnl_at_exit",
        "mfe_to_exit",
        "mae_to_exit",
        "future_max_delta_at_exit",
        "future_min_delta_at_exit",
    ]
    rows = []
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        neg = frame[frame.apply(lambda row: (row["eval_split"], int(row["seed"])) in negative_keys, axis=1)]
        neg = neg[neg["eval_split"].eq(split)]
        pos = frame[frame["eval_split"].eq(split) & ~frame.apply(lambda row: (row["eval_split"], int(row["seed"])) in negative_keys, axis=1)]
        if neg.empty or pos.empty:
            continue
        for feature in features:
            if neg[feature].notna().sum() == 0 or pos[feature].notna().sum() == 0:
                continue
            rows.append(
                {
                    "eval_split": split,
                    "feature": feature,
                    "negative_seed_median": float(neg[feature].median()),
                    "positive_seed_median": float(pos[feature].median()),
                    "median_diff": float(neg[feature].median() - pos[feature].median()),
                }
            )
    return sorted(rows, key=lambda row: (row["eval_split"], -abs(row["median_diff"])))


def _worst_examples(frame: pd.DataFrame, negative_keys: set[tuple[str, int]], count: int = 30) -> list[dict]:
    columns = [
        "eval_split",
        "seed",
        "entry_seed",
        "session",
        "local_time",
        "time_bucket",
        "right",
        "offset",
        "contract_id",
        "candidate_exit_reason",
        "candidate_exit_step",
        "protocol054_exit_reason",
        "protocol054_exit_step",
        "candidate_pnl",
        "protocol054_pnl",
        "delta_vs_protocol054",
        "predicted_continuation_value",
        "override_threshold",
        "current_pnl_at_exit",
        "future_max_delta_at_exit",
        "future_min_delta_at_exit",
    ]
    mask = frame.apply(lambda row: (row["eval_split"], int(row["seed"])) in negative_keys, axis=1)
    subset = frame[mask].sort_values("delta_vs_protocol054").head(count)
    return subset[columns].to_dict(orient="records")


def _markdown_table(rows: list[dict], columns: list[str], max_rows: int = 20) -> str:
    rows = rows[:max_rows]
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.0f}" if abs(value) >= 100 else f"{value:.3f}"
            cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body])


def _interpret(seed_rows: pd.DataFrame) -> str:
    negative = seed_rows[seed_rows["delta"] < 0.0]
    if negative.empty:
        return "No negative-delta seeds were found versus Protocol 054."
    parts = []
    for _, row in negative.iterrows():
        parts.append(f"{row['eval_split']} seed {int(row['seed'])}: {row['delta']:.0f}")
    return (
        "Protocol 066 is robust on medians, but not seed-perfect. Negative-delta seeds were "
        + ", ".join(parts)
        + ". This argues for another diagnostic gate before objective changes or promotion work."
    )


def main() -> int:
    args = parse_args()
    frame = _load_selected(args.selected_trades)
    seed_rows = _seed_metrics(frame)
    negative_seed_rows = seed_rows[seed_rows["delta"] < 0.0]
    negative_keys = {(row["eval_split"], int(row["seed"])) for _, row in negative_seed_rows.iterrows()}
    mask_negative = frame.apply(lambda row: (row["eval_split"], int(row["seed"])) in negative_keys, axis=1)
    negative_frame = frame[mask_negative].copy()

    payload = {
        "protocol": "067_protocol066_seed_fragility_diagnostic",
        "data_used": {
            "selected_trades": str(args.selected_trades),
            "paid_data_downloaded": False,
        },
        "interpretation": _interpret(seed_rows),
        "seed_metrics": seed_rows.to_dict(orient="records"),
        "negative_seed_metrics": negative_seed_rows.to_dict(orient="records"),
        "negative_by_split_side": _group_summary(negative_frame, ["eval_split", "right"]),
        "negative_by_split_time_bucket": _group_summary(negative_frame, ["eval_split", "time_bucket"]),
        "negative_by_split_candidate_reason": _group_summary(negative_frame, ["eval_split", "candidate_exit_reason"]),
        "negative_by_split_protocol054_reason": _group_summary(negative_frame, ["eval_split", "protocol054_exit_reason"]),
        "negative_by_split_exit_step_bucket": _group_summary(negative_frame, ["eval_split", "exit_step_bucket"]),
        "feature_contrast_negative_vs_positive_seeds": _feature_contrast(frame, negative_keys),
        "worst_negative_seed_examples": _worst_examples(frame, negative_keys),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    md = [
        "# Protocol 067 Seed-Fragility Diagnostic",
        "",
        payload["interpretation"],
        "",
        "## Negative Seed Metrics",
        "",
        _markdown_table(
            payload["negative_seed_metrics"],
            [
                "eval_split",
                "seed",
                "rows",
                "candidate_pnl",
                "protocol054_pnl",
                "delta",
                "override_fraction",
                "median_candidate_exit_step",
            ],
            max_rows=20,
        ),
        "",
        "## Negative Seeds By Time Bucket",
        "",
        _markdown_table(
            payload["negative_by_split_time_bucket"],
            ["eval_split", "time_bucket", "rows", "delta", "median_delta", "override_fraction", "median_candidate_exit_step"],
            max_rows=20,
        ),
        "",
        "## Negative Seeds By Protocol 054 Exit Reason",
        "",
        _markdown_table(
            payload["negative_by_split_protocol054_reason"],
            [
                "eval_split",
                "protocol054_exit_reason",
                "rows",
                "delta",
                "median_delta",
                "override_fraction",
                "median_candidate_exit_step",
            ],
            max_rows=20,
        ),
        "",
        "## Decision",
        "",
        "Diagnostic only. Keep Protocol 066 as challenger, but do not promote until the remaining "
        "negative-delta seed patterns are either explained as acceptable variance or reduced without "
        "hurting the robust folds.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(md) + "\n")
    print(payload["interpretation"])
    print(args.out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
