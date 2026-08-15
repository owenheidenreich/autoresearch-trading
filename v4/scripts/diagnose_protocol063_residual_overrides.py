"""Diagnose Protocol 063 residual sequence override failures.

This is a no-paid-data, no-model-change diagnostic. Protocol 063 improved Q3,
Q4, and total Q1 versus Protocol 054, but failed the March preservation gate.
The goal here is to compare March false residual overrides against successful
overrides in Q3/Q4 and Jan-Feb before adding another sequence-model knob.
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
    "v4_aplus_hypothesis_063_calibrated_residual_sequence_screen/"
    "selected_trades_sequence_exits.json"
)
_DEFAULT_SEQUENCE_DIR = Path(
    "v4/audit/autoresearch/"
    "v4_aplus_hypothesis_060_lifecycle_sequence_dataset"
)
_DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/"
    "v4_aplus_hypothesis_064_protocol063_march_override_diagnostic"
)
_OVERRIDE_REASON = "sequence_residual_override"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-trades", type=Path, default=_DEFAULT_SELECTED)
    parser.add_argument("--sequence-dir", type=Path, default=_DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--protocol-label", default="Protocol 063")
    return parser.parse_args()


def _finite(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_selected(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no selected trades found in {path}")
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
    frame["candidate_exit_step_int"] = frame["candidate_exit_step"].round().astype("Int64")
    frame["is_override"] = frame["candidate_exit_reason"].eq(_OVERRIDE_REASON)
    frame["is_march"] = frame["split"].eq("q1_2026") & frame["session"].ge("2026-03-01")
    frame["is_janfeb"] = frame["split"].eq("q1_2026") & frame["session"].lt("2026-03-01")
    frame["is_q3q4"] = frame["split"].isin(["q3_2025", "q4_2025"])
    frame["override_outcome"] = np.select(
        [
            ~frame["is_override"],
            frame["delta_vs_protocol054"] > 0,
            frame["delta_vs_protocol054"] < 0,
        ],
        ["non_override", "positive", "negative"],
        default="flat",
    )
    frame["cohort"] = frame.apply(_cohort_name, axis=1)
    frame["exit_step_bucket"] = pd.cut(
        frame["candidate_exit_step"].fillna(-1),
        bins=[-2, 0, 2, 5, 999],
        labels=["step_00", "step_01_02", "step_03_05", "step_06p"],
    ).astype(str)
    frame["pnl_at_exit_bucket"] = pd.cut(
        frame["current_pnl_at_exit"].fillna(0.0),
        bins=[-99999, -200, 0, 100, 300, 99999],
        labels=["loss_lt_200", "loss_200_0", "win_0_100", "win_100_300", "win_300p"],
    ).astype(str)
    return frame


def _cohort_name(row: pd.Series) -> str:
    if not bool(row.get("is_override", False)):
        return "non_override"
    outcome = str(row.get("override_outcome", "flat"))
    if bool(row.get("is_march", False)):
        return f"march_{outcome}_override"
    if bool(row.get("is_janfeb", False)):
        return f"janfeb_{outcome}_override"
    if bool(row.get("is_q3q4", False)):
        return f"q3q4_{outcome}_override"
    split = str(row.get("split", "unknown"))
    return f"{split}_{outcome}_override"


def _load_steps(sequence_dir: Path) -> pd.DataFrame:
    path = sequence_dir / "protocol054_lifecycle_steps.parquet"
    frame = pd.read_parquet(path)
    if frame.empty:
        raise SystemExit(f"no lifecycle steps found in {path}")
    frame["trade_uid"] = frame["trade_uid"].astype(str)
    frame["step_idx"] = pd.to_numeric(frame["step_idx"], errors="coerce").astype("Int64")
    _numeric(frame, _step_feature_columns(frame))
    return frame


def _step_feature_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "minutes_since_entry",
        "minutes_to_deadline",
        "minutes_to_forced_flat",
        "bid",
        "ask",
        "mid",
        "spread",
        "spread_frac",
        "bid_size",
        "ask_size",
        "quote_gap_seconds",
        "option_ohlcv_volume",
        "stat_open_interest",
        "underlying_price",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "current_pnl",
        "mfe_to_now",
        "mae_to_now",
        "giveback_from_mfe",
        "giveback_fraction",
        "time_since_mfe_minutes",
        "pnl_velocity_1",
        "pnl_velocity_3",
        "pnl_velocity_5",
        "realized_pnl_vol_5",
        "realized_pnl_vol_10",
        "bid_over_entry_ask",
        "mid_over_entry_ask",
        "theta_over_mid",
        "gamma_theta_ratio",
        "time_theta_burden",
        "entry_edge",
        "entry_offset",
        "entry_ask",
        "entry_mid",
        "entry_delta",
        "entry_gamma",
        "entry_theta",
        "entry_iv",
        "future_max_pnl",
        "future_min_pnl",
        "future_final_pnl",
        "future_max_delta",
        "future_min_delta",
        "future_recovery_100",
        "future_recovery_200",
        "future_decay_100",
        "future_decay_200",
        "exit_now_regret_to_baseline",
    ]
    return [column for column in preferred if column in frame.columns]


def _merge_candidate_exit_features(selected: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    step_columns = ["trade_uid", "step_idx"] + _step_feature_columns(steps)
    step_slice = steps[step_columns].copy()
    merged = selected.merge(
        step_slice,
        left_on=["trade_uid", "candidate_exit_step_int"],
        right_on=["trade_uid", "step_idx"],
        how="left",
        suffixes=("", "_step"),
    )
    merged["step_feature_match"] = merged["step_idx"].notna()
    merged["trade_uid"] = merged["trade_uid"].astype(str)
    merged["canonical_entry_uid"] = merged["canonical_entry_uid"].astype(str)
    merged["right"] = merged["right"].astype(str)
    return merged


def _profit_factor(values: pd.Series) -> float:
    wins = values.clip(lower=0).sum()
    losses = -values.clip(upper=0).sum()
    if losses <= 0:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def _metrics(group: pd.DataFrame) -> dict:
    if group.empty:
        return {
            "rows": 0,
            "unique_trades": 0,
            "unique_canonical_entries": 0,
            "candidate_pnl_sum": 0.0,
            "protocol054_pnl_sum": 0.0,
            "delta_sum": 0.0,
            "median_delta": 0.0,
            "avg_delta": 0.0,
            "positive_delta_fraction": 0.0,
            "candidate_profit_factor": 0.0,
            "protocol054_profit_factor": 0.0,
        }
    delta = group["delta_vs_protocol054"].fillna(0.0)
    candidate = group["candidate_pnl"].fillna(0.0)
    protocol = group["protocol054_pnl"].fillna(0.0)
    return {
        "rows": int(len(group)),
        "unique_trades": int(group["trade_uid"].nunique()),
        "unique_canonical_entries": int(group["canonical_entry_uid"].nunique()),
        "candidate_pnl_sum": float(candidate.sum()),
        "protocol054_pnl_sum": float(protocol.sum()),
        "delta_sum": float(delta.sum()),
        "median_delta": float(delta.median()),
        "avg_delta": float(delta.mean()),
        "positive_delta_fraction": float((delta > 0).mean()),
        "candidate_profit_factor": _profit_factor(candidate),
        "protocol054_profit_factor": _profit_factor(protocol),
        "avg_candidate_exit_step": float(group["candidate_exit_step"].mean()),
        "avg_protocol054_exit_step": float(group["protocol054_exit_step"].mean()),
        "median_candidate_exit_step": float(group["candidate_exit_step"].median()),
        "median_protocol054_exit_step": float(group["protocol054_exit_step"].median()),
        "median_predicted_continuation_value": float(group["predicted_continuation_value"].median()),
        "median_override_threshold": float(group["override_threshold"].median()),
        "median_recovery_probability": float(group["predicted_recovery_probability"].median()),
        "median_decay_probability": float(group["predicted_decay_probability"].median()),
        "median_current_pnl_at_exit": float(group["current_pnl_at_exit"].median()),
        "median_future_max_delta_at_exit": float(group["future_max_delta_at_exit"].median()),
        "median_future_min_delta_at_exit": float(group["future_min_delta_at_exit"].median()),
        "step_feature_match_fraction": float(group["step_feature_match"].mean()),
    }


def _group_summary(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    for keys, group in frame.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({column: str(key) for column, key in zip(columns, keys)} | _metrics(group))
    return sorted(rows, key=lambda row: (str(row.get("cohort", "")), str(row.get("split", "")), row["delta_sum"]))


def _canonical_summary(frame: pd.DataFrame) -> list[dict]:
    """Summarize after collapsing repeated sequence-model seed rows."""
    overrides = frame[frame["is_override"]].copy()
    if overrides.empty:
        return []
    collapsed = (
        overrides.groupby(["cohort", "trade_uid"], dropna=False)
        .agg(
            split=("split", "first"),
            session=("session", "first"),
            right=("right", "first"),
            time_bucket=("time_bucket", "first"),
            canonical_entry_uid=("canonical_entry_uid", "first"),
            delta_vs_protocol054=("delta_vs_protocol054", "median"),
            candidate_pnl=("candidate_pnl", "median"),
            protocol054_pnl=("protocol054_pnl", "median"),
            candidate_exit_step=("candidate_exit_step", "median"),
            protocol054_exit_step=("protocol054_exit_step", "median"),
            predicted_continuation_value=("predicted_continuation_value", "median"),
            override_threshold=("override_threshold", "median"),
            predicted_recovery_probability=("predicted_recovery_probability", "median"),
            predicted_decay_probability=("predicted_decay_probability", "median"),
            current_pnl_at_exit=("current_pnl_at_exit", "median"),
            future_max_delta_at_exit=("future_max_delta_at_exit", "median"),
            future_min_delta_at_exit=("future_min_delta_at_exit", "median"),
            step_feature_match=("step_feature_match", "max"),
        )
        .reset_index()
    )
    return _group_summary(collapsed, ["cohort"])


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
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
        "minutes_since_entry",
        "minutes_to_deadline",
        "minutes_to_forced_flat",
        "bid",
        "ask",
        "mid",
        "spread",
        "spread_frac",
        "bid_size",
        "ask_size",
        "quote_gap_seconds",
        "option_ohlcv_volume",
        "stat_open_interest",
        "underlying_price",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "current_pnl",
        "mfe_to_now",
        "mae_to_now",
        "giveback_from_mfe",
        "giveback_fraction",
        "time_since_mfe_minutes",
        "pnl_velocity_1",
        "pnl_velocity_3",
        "pnl_velocity_5",
        "realized_pnl_vol_5",
        "realized_pnl_vol_10",
        "bid_over_entry_ask",
        "mid_over_entry_ask",
        "theta_over_mid",
        "gamma_theta_ratio",
        "time_theta_burden",
        "entry_edge",
        "entry_offset",
        "entry_ask",
        "entry_mid",
        "entry_delta",
        "entry_gamma",
        "entry_theta",
        "entry_iv",
        "future_max_pnl",
        "future_min_pnl",
        "future_final_pnl",
        "future_max_delta",
        "future_min_delta",
        "future_recovery_100",
        "future_recovery_200",
        "future_decay_100",
        "future_decay_200",
        "exit_now_regret_to_baseline",
    ]
    return [column for column in candidates if column in frame.columns]


def _feature_medians(frame: pd.DataFrame) -> list[dict]:
    rows = []
    features = _feature_columns(frame)
    for cohort, group in frame.groupby("cohort", dropna=False):
        row = {"cohort": str(cohort), "rows": int(len(group))}
        for feature in features:
            row[feature] = float(group[feature].median()) if group[feature].notna().any() else None
        rows.append(row)
    return sorted(rows, key=lambda row: row["cohort"])


def _feature_contrast(frame: pd.DataFrame, left: str, right: str) -> list[dict]:
    features = _feature_columns(frame)
    left_frame = frame[frame["cohort"].eq(left)]
    right_frame = frame[frame["cohort"].eq(right)]
    rows = []
    for feature in features:
        if left_frame[feature].notna().sum() == 0 or right_frame[feature].notna().sum() == 0:
            continue
        left_median = float(left_frame[feature].median())
        right_median = float(right_frame[feature].median())
        pooled = frame[frame["cohort"].isin([left, right])][feature].dropna().astype(float)
        scale = float((pooled - pooled.median()).abs().median())
        if scale <= 1e-9:
            scale = float(pooled.std(ddof=0))
        if scale <= 1e-9 or not np.isfinite(scale):
            scale = 1.0
        rows.append(
            {
                "feature": feature,
                f"{left}_median": left_median,
                f"{right}_median": right_median,
                "median_diff": left_median - right_median,
                "robust_z": (left_median - right_median) / scale,
            }
        )
    return sorted(rows, key=lambda row: abs(row["robust_z"]), reverse=True)


def _worst_examples(frame: pd.DataFrame, cohort: str, count: int = 20) -> list[dict]:
    columns = [
        "split",
        "seed",
        "entry_seed",
        "session",
        "local_time",
        "time_bucket",
        "right",
        "offset",
        "contract_id",
        "candidate_exit_step",
        "protocol054_exit_step",
        "candidate_pnl",
        "protocol054_pnl",
        "delta_vs_protocol054",
        "predicted_continuation_value",
        "override_threshold",
        "predicted_recovery_probability",
        "predicted_decay_probability",
        "current_pnl_at_exit",
        "mfe_to_exit",
        "mae_to_exit",
        "future_max_delta_at_exit",
        "future_min_delta_at_exit",
        "gamma",
        "theta",
        "theta_over_mid",
        "gamma_theta_ratio",
        "time_theta_burden",
    ]
    available = [column for column in columns if column in frame.columns]
    subset = frame[frame["cohort"].eq(cohort)].sort_values("delta_vs_protocol054").head(count)
    return subset[available].to_dict(orient="records")


def _top_group(group_rows: list[dict], *, cohort: str, column: str = "delta_sum") -> dict | None:
    rows = [row for row in group_rows if row.get("cohort") == cohort]
    if not rows:
        return None
    return min(rows, key=lambda row: _finite(row.get(column)))


def _interpretation(payload: dict) -> str:
    protocol_label = str(payload.get("protocol_label", "Selected residual sequence run"))
    cohorts = {row["cohort"]: row for row in payload["cohort_summary"]}
    march_false = cohorts.get("march_negative_override", {})
    q3q4_success = cohorts.get("q3q4_positive_override", {})
    janfeb_success = cohorts.get("janfeb_positive_override", {})
    worst_side = _top_group(payload["by_cohort_side"], cohort="march_negative_override")
    worst_time = _top_group(payload["by_cohort_time_bucket"], cohort="march_negative_override")
    step = march_false.get("median_candidate_exit_step", 0.0)
    p054_step = march_false.get("median_protocol054_exit_step", 0.0)
    recovery = march_false.get("median_future_max_delta_at_exit", 0.0)
    q3q4_step = q3q4_success.get("median_candidate_exit_step", 0.0)
    janfeb_step = janfeb_success.get("median_candidate_exit_step", 0.0)
    side_phrase = ""
    if worst_side:
        side_phrase = (
            f" The worst March side bucket was {worst_side.get('right')} "
            f"with delta {worst_side.get('delta_sum', 0.0):.0f}."
        )
    time_phrase = ""
    if worst_time:
        time_phrase = (
            f" The worst March time bucket was {worst_time.get('time_bucket')} "
            f"with delta {worst_time.get('delta_sum', 0.0):.0f}."
        )
    return (
        f"{protocol_label} is not failing because sequence overrides are useless; the same mechanism is "
        f"positive in Q3/Q4 ({q3q4_success.get('delta_sum', 0.0):.0f}) and positive in Jan-Feb "
        f"({janfeb_success.get('delta_sum', 0.0):.0f}). March false overrides are the key failure "
        f"mode: they exit at median step {step:.1f} while Protocol 054 would exit around step "
        f"{p054_step:.1f}, and their median post-exit recovery headroom is {recovery:.0f}. "
        f"Successful Q3/Q4 overrides exit around step {q3q4_step:.1f}; successful Jan-Feb overrides "
        f"exit around step {janfeb_step:.1f}."
        f"{side_phrase}{time_phrase} The next hypothesis should not add another entry-side knob. "
        "It should make the residual sequence objective explicitly penalize false early overrides "
        "when the causal state still resembles a recoverable convex pullback."
    )


def _markdown_table(rows: list[dict], columns: list[str], max_rows: int = 12) -> str:
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
                if abs(value) >= 100:
                    value = f"{value:.0f}"
                else:
                    value = f"{value:.3f}"
            cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body])


def _write_report(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    cohorts = [
        row
        for row in payload["cohort_summary"]
        if row["cohort"]
        in {
            "march_negative_override",
            "march_positive_override",
            "janfeb_negative_override",
            "janfeb_positive_override",
            "q3q4_negative_override",
            "q3q4_positive_override",
        }
    ]
    md = [
        f"# {payload.get('protocol_label', 'Residual Sequence Run')} Override Diagnostic",
        "",
        payload["interpretation"],
        "",
        "## Cohort Summary",
        "",
        _markdown_table(
            cohorts,
            [
                "cohort",
                "rows",
                "unique_trades",
                "delta_sum",
                "median_delta",
                "median_candidate_exit_step",
                "median_protocol054_exit_step",
                "median_future_max_delta_at_exit",
                "median_future_min_delta_at_exit",
            ],
            max_rows=20,
        ),
        "",
        "## March Negative Overrides By Side",
        "",
        _markdown_table(
            [row for row in payload["by_cohort_side"] if row.get("cohort") == "march_negative_override"],
            ["cohort", "right", "rows", "delta_sum", "median_delta", "median_candidate_exit_step"],
            max_rows=10,
        ),
        "",
        "## March Negative Overrides By Time Bucket",
        "",
        _markdown_table(
            [
                row
                for row in payload["by_cohort_time_bucket"]
                if row.get("cohort") == "march_negative_override"
            ],
            ["cohort", "time_bucket", "rows", "delta_sum", "median_delta", "median_candidate_exit_step"],
            max_rows=12,
        ),
        "",
        "## Top March False-vs-Q3/Q4 Success Feature Contrasts",
        "",
        _markdown_table(
            payload["march_false_vs_q3q4_success_feature_contrast"],
            [
                "feature",
                "march_negative_override_median",
                "q3q4_positive_override_median",
                "median_diff",
                "robust_z",
            ],
            max_rows=16,
        ),
        "",
        "## Decision",
        "",
        f"{payload.get('protocol_label', 'This residual sequence run')} remains diagnostic unless it passes "
        "the frozen promotion gate against Protocol 054. This diagnostic does not change the baseline. "
        "The pre-registered next hypothesis "
        "is a lifecycle-only residual sequence objective with asymmetric false-early-exit penalty, trained only "
        "on allowed training folds and evaluated against the same frozen holdouts.",
    ]
    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    (out_dir / "DECISION.md").write_text(
        "\n".join(
            [
                "# Protocol 064 Decision",
                "",
                "Decision: Diagnostic only. Keep Protocol 054 frozen as the baseline.",
                "",
                "Reason: Protocol 063 is still a serious challenger, but March negative residual overrides "
                "show that the sequence model can mistake recoverable convex pullbacks for decay and exit too early.",
                "",
                "Next Gate: Test one lifecycle-only residual objective that penalizes false early exits when "
                "post-entry state still has recovery/convexity evidence. No entry-side knobs and no paid data.",
            ]
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    selected = _load_selected(args.selected_trades)
    steps = _load_steps(args.sequence_dir)
    frame = _merge_candidate_exit_features(selected, steps)
    overrides = frame[frame["is_override"]].copy()

    payload = {
        "protocol": "064_protocol063_march_override_diagnostic",
        "protocol_label": args.protocol_label,
        "data_used": {
            "selected_trades": str(args.selected_trades),
            "sequence_dir": str(args.sequence_dir),
            "paid_data_downloaded": False,
        },
        "row_counts": {
            "selected_rows": int(len(frame)),
            "override_rows": int(len(overrides)),
            "step_feature_match_fraction": float(frame["step_feature_match"].mean()),
            "override_step_feature_match_fraction": float(overrides["step_feature_match"].mean()),
        },
        "cohort_summary": _group_summary(frame, ["cohort"]),
        "canonical_override_summary": _canonical_summary(frame),
        "by_cohort_side": _group_summary(overrides, ["cohort", "right"]),
        "by_cohort_time_bucket": _group_summary(overrides, ["cohort", "time_bucket"]),
        "by_cohort_exit_step_bucket": _group_summary(overrides, ["cohort", "exit_step_bucket"]),
        "by_cohort_pnl_at_exit_bucket": _group_summary(overrides, ["cohort", "pnl_at_exit_bucket"]),
        "by_cohort_protocol054_exit_reason": _group_summary(
            overrides, ["cohort", "protocol054_exit_reason"]
        ),
        "feature_medians_by_cohort": _feature_medians(overrides),
        "march_false_vs_q3q4_success_feature_contrast": _feature_contrast(
            overrides, "march_negative_override", "q3q4_positive_override"
        ),
        "march_false_vs_janfeb_success_feature_contrast": _feature_contrast(
            overrides, "march_negative_override", "janfeb_positive_override"
        ),
        "worst_march_false_override_examples": _worst_examples(frame, "march_negative_override"),
    }
    payload["interpretation"] = _interpretation(payload)
    _write_report(payload, args.out_dir)
    print(json.dumps(payload["row_counts"], indent=2))
    print(payload["interpretation"])
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
