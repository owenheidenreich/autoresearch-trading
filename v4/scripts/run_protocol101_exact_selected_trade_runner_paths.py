from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.scripts.run_protocol101_track_a_forensics import classify_runner_state  # noqa: E402
from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import (  # noqa: E402
    DEFAULT_NORMALIZED_DIR,
    finite_sum,
    money,
    normalize_trade_frame,
    pct,
)
from v4.scripts.run_protocol199_lifecycle_full_path_oracle import build_full_path_oracle_rows  # noqa: E402


ROLE_LABEL = "AUDIT_PROTOCOL101_EXACT_SELECTED_TRADE_RUNNER_PATHS_V1"
DEFAULT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv")
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_exact_selected_trade_runner_paths_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/execution/PROTOCOL101_EXACT_SELECTED_TRADE_RUNNER_PATHS_V1.md"
)


def load_protocol101_selected_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"no Protocol101 selected trades found in {path}")
    if "segment" not in raw.columns:
        raise ValueError("Protocol101 selected trades must include segment")
    raw = raw.rename(columns={"segment": "reported_split"}).copy()
    raw["fold"] = "protocol101_seed1_selected"
    return normalize_trade_frame(raw)


def attach_source_trade_fields(rows: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    source = trades.copy()
    source["decision_time"] = source["decision_ts"].map(lambda ts: pd.Timestamp(ts).isoformat())
    source_key = ["reported_split", "seed", "session", "decision_time", "contract_id", "right"]
    keep_cols = source_key + [
        "candidate_uid",
        "score",
        "threshold",
        "offset",
        "entry_bid",
        "entry_ask",
        "pnl",
        "path_mfe",
        "path_mae",
        "exit_reason",
    ]
    available = [column for column in keep_cols if column in source.columns]
    out = rows.merge(source[available], on=source_key, how="left", suffixes=("", "_source"))
    if "pnl_source" in out.columns:
        out = out.rename(columns={"pnl_source": "source_trade_pnl"})
    out["runner_state"] = out.apply(classify_runner_state, axis=1)
    out["post_exit_positive"] = pd.to_numeric(out["post_frozen_best_minus_frozen_pnl"], errors="coerce").fillna(0.0) > 0.0
    out["forced_flat_positive"] = pd.to_numeric(out["forced_flat_minus_frozen_pnl"], errors="coerce").fillna(0.0) > 0.0
    out["naive_forced_flat_safe"] = out["forced_flat_positive"]
    out["giveback_guard_needed"] = out["runner_state"].astype(str).isin(["giveback_guard_required", "extension_destroys_value"])
    return out


def summarize_runner_states(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "runner_state",
                "rows",
                "frozen_pnl",
                "post_exit_best_delta",
                "forced_flat_delta",
                "material_continuation_rate",
                "forced_flat_positive_rate",
            ]
        )
    rows: list[dict[str, Any]] = []
    for state, group in frame.groupby("runner_state", dropna=False):
        post_delta = pd.to_numeric(group["post_frozen_best_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        forced_delta = pd.to_numeric(group["forced_flat_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "runner_state": state,
                "rows": int(len(group)),
                "frozen_pnl": finite_sum(group["frozen_pnl"]),
                "oracle_delta": finite_sum(group["oracle_minus_frozen_pnl"]),
                "post_exit_best_delta": finite_sum(post_delta),
                "forced_flat_delta": finite_sum(forced_delta),
                "material_continuation_rate": float(group["material_continuation_after_frozen_exit"].mean()),
                "post_exit_positive_rate": float((post_delta > 0).mean()) if len(group) else 0.0,
                "forced_flat_positive_rate": float((forced_delta > 0).mean()) if len(group) else 0.0,
                "median_post_exit_best_delta": float(post_delta.median()),
                "median_forced_flat_delta": float(forced_delta.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("post_exit_best_delta", ascending=False, kind="stable")


def summarize_by_split_state(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["reported_split", "runner_state", "right", "time_bucket", "exit_reason"]
    for key_values, group in frame.groupby(keys, dropna=False, observed=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        post_delta = pd.to_numeric(group["post_frozen_best_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        forced_delta = pd.to_numeric(group["forced_flat_minus_frozen_pnl"], errors="coerce").fillna(0.0)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "rows": int(len(group)),
                "frozen_pnl": finite_sum(group["frozen_pnl"]),
                "oracle_delta": finite_sum(group["oracle_minus_frozen_pnl"]),
                "post_exit_best_delta": finite_sum(post_delta),
                "forced_flat_delta": finite_sum(forced_delta),
                "material_continuation_rate": float(group["material_continuation_after_frozen_exit"].mean()),
                "forced_flat_positive_rate": float((forced_delta > 0).mean()) if len(group) else 0.0,
                "median_post_exit_best_delta": float(post_delta.median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("post_exit_best_delta", ascending=False, kind="stable")


def build_runner_examples(frame: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = [
        "reported_split",
        "session",
        "decision_time",
        "frozen_exit_time",
        "post_frozen_best_time",
        "forced_flat_time",
        "contract_id",
        "right",
        "exit_reason",
        "frozen_pnl",
        "post_frozen_best_pnl",
        "post_frozen_best_minus_frozen_pnl",
        "forced_flat_minus_frozen_pnl",
        "oracle_minus_frozen_pnl",
        "runner_state",
        "candidate_uid",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].sort_values("post_frozen_best_minus_frozen_pnl", ascending=False, kind="stable").head(limit)


def summarize_path_skips(skips: list[dict[str, Any]], trades: pd.DataFrame) -> pd.DataFrame:
    if not skips:
        return pd.DataFrame(columns=["skip_reason", "rows"])
    frame = pd.DataFrame(skips)
    rows: list[dict[str, Any]] = []
    for reason, group in frame.groupby("skip_reason", dropna=False):
        rows.append({"skip_reason": reason, "rows": int(len(group))})
    return pd.DataFrame(rows).sort_values("rows", ascending=False, kind="stable")


def build_summary(trades: pd.DataFrame, paths: pd.DataFrame, skips: list[dict[str, Any]]) -> dict[str, Any]:
    selected_count = int(len(trades))
    path_count = int(len(paths))
    skip_count = int(len(skips))
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "exact selected-trade Protocol101 post-exit runner/giveback path audit",
        "decision": "protocol101_exact_selected_runner_paths_complete_training_still_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "source_trades": str(DEFAULT_TRADES),
        "coverage": {
            "selected_trades": selected_count,
            "exact_path_rows": path_count,
            "path_skips": skip_count,
            "exact_path_coverage": float(path_count / selected_count) if selected_count else 0.0,
        },
        "headline": {
            "frozen_pnl_with_paths": finite_sum(paths["frozen_pnl"]) if not paths.empty else 0.0,
            "post_exit_best_delta": finite_sum(paths["post_frozen_best_minus_frozen_pnl"]) if not paths.empty else 0.0,
            "forced_flat_delta": finite_sum(paths["forced_flat_minus_frozen_pnl"]) if not paths.empty else 0.0,
            "material_continuation_rate": float(paths["material_continuation_after_frozen_exit"].mean()) if not paths.empty else 0.0,
            "forced_flat_positive_rate": float(paths["forced_flat_positive"].mean()) if "forced_flat_positive" in paths else 0.0,
        },
        "blockers": [
            "runner_policy_not_trained",
            "runner_result_is_hindsight_upper_bound",
            "path_coverage_not_complete",
            "giveback_guard_required_before_extension",
            "fill_and_latency_realism_still_unresolved",
        ],
        "challenge_allowed": False,
    }


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    state_summary: pd.DataFrame,
    skip_summary: pd.DataFrame,
) -> str:
    coverage = summary["coverage"]
    headline = summary["headline"]
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: exact selected-trade Protocol101 post-exit runner/giveback path audit",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "This packet attaches actual normalized quote paths to Protocol101 selected trades and measures what happened after the frozen exit. It is a hindsight diagnostic for asking better runner/giveback questions, not a tradable runner policy.",
        "",
        "## Coverage",
        "",
        f"- Selected trades: `{coverage['selected_trades']}`",
        f"- Exact path rows: `{coverage['exact_path_rows']}`",
        f"- Path skips: `{coverage['path_skips']}`",
        f"- Exact path coverage: `{coverage['exact_path_coverage']:.3f}`",
        "",
        "## Headline",
        "",
        f"- Frozen PnL with paths: `{money(headline['frozen_pnl_with_paths'])}`",
        f"- Hindsight post-exit best delta: `{money(headline['post_exit_best_delta'])}`",
        f"- Naive forced-flat delta: `{money(headline['forced_flat_delta'])}`",
        f"- Material continuation rate: `{pct(headline['material_continuation_rate'])}`",
        f"- Forced-flat positive rate: `{pct(headline['forced_flat_positive_rate'])}`",
        "",
        "## Runner States",
        "",
        "| runner state | rows | frozen pnl | post-exit best delta | forced-flat delta | material continuation | forced-flat positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in state_summary.iterrows():
        lines.append(
            f"| {row['runner_state']} | {int(row['rows'])} | {money(row['frozen_pnl'])} | "
            f"{money(row['post_exit_best_delta'])} | {money(row['forced_flat_delta'])} | "
            f"{pct(row['material_continuation_rate'])} | {pct(row['forced_flat_positive_rate'])} |"
        )
    lines.extend(["", "## Path Skips", "", "| reason | rows |", "|---|---:|"])
    for _, row in skip_summary.iterrows():
        lines.append(f"| {row['skip_reason']} | {int(row['rows'])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive post-exit best delta means there was later bid-path opportunity after Protocol101 exited. A negative forced-flat delta means naive hold-to-close would have destroyed value. Runner research should therefore focus on confirmed-MFE state transitions with giveback guards, not a blanket hold-longer rule.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Exact selected runner paths: `{output_dir / 'exact_selected_runner_paths.csv'}`",
            f"- Runner state summary: `{output_dir / 'runner_state_summary.csv'}`",
            f"- Runner split/state summary: `{output_dir / 'runner_split_state_summary.csv'}`",
            f"- Runner examples: `{output_dir / 'runner_candidate_examples.csv'}`",
            f"- Path skips: `{output_dir / 'path_skips.csv'}`",
            f"- Path skip summary: `{output_dir / 'path_skip_summary.csv'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = load_protocol101_selected_trades(Path(args.trades))
    rows, skips = build_full_path_oracle_rows(
        trades,
        normalized_dir=Path(args.normalized_dir),
        forced_flat_time=str(args.forced_flat_time),
        material_delta=float(args.material_delta),
    )
    paths = attach_source_trade_fields(pd.DataFrame(rows), trades)
    state_summary = summarize_runner_states(paths)
    split_state_summary = summarize_by_split_state(paths)
    examples = build_runner_examples(paths)
    skip_summary = summarize_path_skips(skips, trades)
    summary = build_summary(trades, paths, skips)
    summary["source_trades"] = str(args.trades)
    summary["normalized_dir"] = str(args.normalized_dir)
    summary["forced_flat_time"] = str(args.forced_flat_time)
    summary["material_delta"] = float(args.material_delta)
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "exact_selected_runner_paths": str(output_dir / "exact_selected_runner_paths.csv"),
        "runner_state_summary": str(output_dir / "runner_state_summary.csv"),
        "runner_split_state_summary": str(output_dir / "runner_split_state_summary.csv"),
        "runner_candidate_examples": str(output_dir / "runner_candidate_examples.csv"),
        "path_skips": str(output_dir / "path_skips.csv"),
        "path_skip_summary": str(output_dir / "path_skip_summary.csv"),
    }

    paths.to_csv(output_dir / "exact_selected_runner_paths.csv", index=False)
    state_summary.to_csv(output_dir / "runner_state_summary.csv", index=False)
    split_state_summary.to_csv(output_dir / "runner_split_state_summary.csv", index=False)
    examples.to_csv(output_dir / "runner_candidate_examples.csv", index=False)
    pd.DataFrame(skips).to_csv(output_dir / "path_skips.csv", index=False)
    skip_summary.to_csv(output_dir / "path_skip_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = write_report(output_dir, summary, state_summary, skip_summary)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--material-delta", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
