"""Protocol199: full-path lifecycle oracle for frozen Protocol194 entries.

This is a model-design diagnostic. Protocol198 showed that same-side churn can
be harmful in some regimes. Protocol199 asks the next question: if the bot had
kept managing the original contract after the frozen Protocol081 exit, was
there material continuation value available later in the same executable quote
path?

The oracle is hindsight and is not tradable. Its purpose is to decide whether a
hold/exit neural policy has enough label signal to justify training.

No paid data is downloaded. No live broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import (
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_TRADES,
    LOOP_ID as PROTOCOL198_LOOP_ID,
    CONTRACT_MULTIPLIER,
    find_normalized_path,
    finite_float,
    finite_sum,
    load_trades,
    money,
    pct,
)


LOOP_ID = "v4_aplus_hypothesis_199_lifecycle_full_path_oracle"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
NY = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--material-delta", type=float, default=100.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = load_trades(args.trades)
    rows, skips = build_full_path_oracle_rows(
        trades,
        normalized_dir=args.normalized_dir,
        forced_flat_time=str(args.forced_flat_time),
        material_delta=float(args.material_delta),
    )
    frame = pd.DataFrame(rows)
    split_summary = summarize(frame, ["reported_split"])
    side_summary = summarize(frame, ["reported_split", "right"])
    time_summary = summarize(frame, ["reported_split", "time_bucket"])
    exit_summary = summarize(frame, ["reported_split", "exit_reason"])
    payload = {
        "protocol": "199_lifecycle_full_path_oracle",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_trades": str(args.trades),
        "normalized_dir": str(args.normalized_dir),
        "forced_flat_time": str(args.forced_flat_time),
        "material_delta": float(args.material_delta),
        "row_counts": {
            "source_trades": int(len(trades)),
            "oracle_rows": int(len(frame)),
            "path_skips": int(len(skips)),
        },
        "split_summary": split_summary,
        "side_summary": side_summary,
        "time_bucket_summary": time_summary,
        "exit_reason_summary": exit_summary,
        "path_skip_counts": count_by(skips, "skip_reason"),
        "decision": decide(frame),
        "interpretation": (
            "This is a hindsight upper-bound diagnostic. It is not live-tradable evidence. "
            "If continuation value after the frozen exit is common, the next model should learn "
            "holding-state continuation/exit decisions from causal post-entry features."
        ),
        "upstream_churn_audit": f"v4/audit/autoresearch/{PROTOCOL198_LOOP_ID}/report.md",
    }
    frame.to_csv(args.out_dir / "full_path_oracle_rows.csv", index=False)
    pd.DataFrame(skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    pd.DataFrame(split_summary).to_csv(args.out_dir / "split_summary.csv", index=False)
    pd.DataFrame(side_summary).to_csv(args.out_dir / "side_summary.csv", index=False)
    pd.DataFrame(time_summary).to_csv(args.out_dir / "time_bucket_summary.csv", index=False)
    pd.DataFrame(exit_summary).to_csv(args.out_dir / "exit_reason_summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload, frame)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def build_full_path_oracle_rows(
    trades: pd.DataFrame,
    *,
    normalized_dir: Path,
    forced_flat_time: str,
    material_delta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for session, session_trades in trades.groupby("session", sort=True):
        contracts = set(session_trades["contract_id"].astype(str).unique().tolist())
        quotes = load_session_quotes(normalized_dir, str(session), contracts)
        if quotes.empty:
            skips.extend(base_skip(trade, "missing_session_or_contract_quotes") for _, trade in session_trades.iterrows())
            continue
        by_contract = {
            str(contract_id): part.sort_values("quote_time").reset_index(drop=True)
            for contract_id, part in quotes.groupby("contract_id", sort=False)
        }
        forced_flat = forced_flat_timestamp(str(session), forced_flat_time)
        for _, trade in session_trades.iterrows():
            result, skip = oracle_for_trade(trade, by_contract.get(str(trade["contract_id"])), forced_flat, material_delta)
            if skip:
                skips.append(skip)
            else:
                rows.append(result)
    return rows, skips


def load_session_quotes(normalized_dir: Path, session: str, contract_ids: set[str]) -> pd.DataFrame:
    path = find_normalized_path(normalized_dir, session)
    if path is None or not contract_ids:
        return pd.DataFrame(columns=["quote_time", "contract_id", "bid", "ask", "underlying_price"])
    try:
        frame = pd.read_parquet(path, columns=["quote_time", "contract_id", "bid", "ask", "underlying_price"])
    except Exception:
        try:
            frame = pd.read_parquet(path, columns=["quote_time", "contract_id", "bid", "ask"])
        except Exception:
            return pd.DataFrame(columns=["quote_time", "contract_id", "bid", "ask", "underlying_price"])
        frame["underlying_price"] = np.nan
    frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True, errors="coerce")
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame = frame[frame["contract_id"].isin(contract_ids)].copy()
    for column in ["bid", "ask", "underlying_price"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        frame["quote_time"].notna()
        & frame["bid"].notna()
        & frame["ask"].notna()
        & (frame["bid"] >= 0.0)
        & (frame["ask"] > 0.0)
        & (frame["ask"] >= frame["bid"])
    ].copy()


def forced_flat_timestamp(session: str, forced_flat_time: str) -> pd.Timestamp:
    hour, minute = [int(part) for part in forced_flat_time.split(":", 1)]
    return pd.Timestamp(session).replace(hour=hour, minute=minute, tzinfo=NY).tz_convert("UTC")


def oracle_for_trade(
    trade: pd.Series,
    quotes: pd.DataFrame | None,
    forced_flat: pd.Timestamp,
    material_delta: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if quotes is None or quotes.empty:
        return {}, base_skip(trade, "missing_contract_quotes")
    entry_ask = finite_float(trade.get("entry_ask"), math.nan)
    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        return {}, base_skip(trade, "invalid_entry_ask")
    decision_ts = pd.Timestamp(trade["decision_ts"])
    frozen_exit_ts = pd.Timestamp(trade["exit_ts"])
    path = quotes[(quotes["quote_time"] >= decision_ts) & (quotes["quote_time"] <= forced_flat)].copy()
    if path.empty:
        return {}, base_skip(trade, "missing_full_path")
    path["path_pnl"] = (pd.to_numeric(path["bid"], errors="coerce") - entry_ask) * CONTRACT_MULTIPLIER
    path = path[path["path_pnl"].notna()].reset_index(drop=True)
    if path.empty:
        return {}, base_skip(trade, "invalid_path_pnl")
    oracle_idx = int(path["path_pnl"].idxmax())
    oracle_row = path.iloc[oracle_idx]
    forced_row = path.iloc[-1]
    post_exit = path[path["quote_time"] >= frozen_exit_ts].copy()
    if post_exit.empty:
        post_exit = path.iloc[[-1]].copy()
    post_idx = int(post_exit["path_pnl"].idxmax())
    post_row = post_exit.loc[post_idx]
    frozen_pnl = finite_float(trade.get("pnl"), 0.0)
    oracle_pnl = finite_float(oracle_row.get("path_pnl"), 0.0)
    forced_pnl = finite_float(forced_row.get("path_pnl"), 0.0)
    post_best_pnl = finite_float(post_row.get("path_pnl"), 0.0)
    post_exit_delta = post_best_pnl - frozen_pnl
    underlying_entry = finite_float(path.iloc[0].get("underlying_price"), math.nan)
    underlying_oracle = finite_float(oracle_row.get("underlying_price"), math.nan)
    underlying_exit = finite_float(path.iloc[-1].get("underlying_price"), math.nan)
    return {
        "reported_split": str(trade["reported_split"]),
        "fold": str(trade["fold"]),
        "seed": int(trade["seed"]),
        "session": str(trade["session"]),
        "time_bucket": str(trade.get("time_bucket", "")),
        "decision_time": decision_ts.isoformat(),
        "frozen_exit_time": frozen_exit_ts.isoformat(),
        "forced_flat_time": forced_flat.isoformat(),
        "contract_id": str(trade["contract_id"]),
        "right": str(trade["right"]),
        "entry_ask": entry_ask,
        "frozen_exit_reason": str(trade.get("candidate_exit_reason", trade.get("exit_reason", ""))),
        "exit_reason": str(trade.get("candidate_exit_reason", trade.get("exit_reason", ""))),
        "frozen_pnl": frozen_pnl,
        "oracle_pnl": oracle_pnl,
        "oracle_exit_time": pd.Timestamp(oracle_row["quote_time"]).isoformat(),
        "oracle_minus_frozen_pnl": float(oracle_pnl - frozen_pnl),
        "post_frozen_best_pnl": post_best_pnl,
        "post_frozen_best_time": pd.Timestamp(post_row["quote_time"]).isoformat(),
        "post_frozen_best_minus_frozen_pnl": float(post_exit_delta),
        "forced_flat_pnl": forced_pnl,
        "forced_flat_minus_frozen_pnl": float(forced_pnl - frozen_pnl),
        "path_mfe_pnl": finite_float(path["path_pnl"].max(), math.nan),
        "path_mae_pnl": finite_float(path["path_pnl"].min(), math.nan),
        "path_points": int(len(path)),
        "exited_before_oracle": bool(pd.Timestamp(oracle_row["quote_time"]) > frozen_exit_ts),
        "material_continuation_after_frozen_exit": bool(post_exit_delta >= material_delta),
        "underlying_entry": underlying_entry,
        "underlying_at_oracle": underlying_oracle,
        "underlying_at_forced_flat": underlying_exit,
        "directional_underlying_to_oracle": directional_underlying_move(str(trade["right"]), underlying_oracle - underlying_entry),
    }, None


def base_skip(trade: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "reported_split": str(trade.get("reported_split", "")),
        "fold": str(trade.get("fold", "")),
        "seed": int(trade.get("seed", 0)),
        "session": str(trade.get("session", "")),
        "contract_id": str(trade.get("contract_id", "")),
        "right": str(trade.get("right", "")),
        "skip_reason": reason,
    }


def directional_underlying_move(right: str, underlying_move: float) -> float:
    if not math.isfinite(underlying_move):
        return math.nan
    return float(-underlying_move if right == "P" else underlying_move)


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: str(value) for column, value in zip(group_cols, key)}
        row.update(
            {
                "rows": int(len(group)),
                "frozen_pnl": finite_sum(group["frozen_pnl"]),
                "oracle_pnl": finite_sum(group["oracle_pnl"]),
                "oracle_minus_frozen_pnl": finite_sum(group["oracle_minus_frozen_pnl"]),
                "post_frozen_best_minus_frozen_pnl": finite_sum(group["post_frozen_best_minus_frozen_pnl"]),
                "forced_flat_minus_frozen_pnl": finite_sum(group["forced_flat_minus_frozen_pnl"]),
                "median_oracle_minus_frozen_pnl": finite_float(group["oracle_minus_frozen_pnl"].median(), 0.0),
                "exited_before_oracle_fraction": float(group["exited_before_oracle"].mean()),
                "material_continuation_fraction": float(group["material_continuation_after_frozen_exit"].mean()),
                "median_directional_underlying_to_oracle": finite_float(group["directional_underlying_to_oracle"].median(), math.nan),
            }
        )
        rows.append(row)
    return rows


def decide(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "blocked_no_executable_full_path_rows"
    continuation_fraction = float(frame["material_continuation_after_frozen_exit"].mean())
    post_delta = finite_sum(frame["post_frozen_best_minus_frozen_pnl"])
    forced_delta = finite_sum(frame["forced_flat_minus_frozen_pnl"])
    if continuation_fraction >= 0.35 and post_delta > 0.0:
        return "strong_lifecycle_training_signal"
    if post_delta > 0.0 or forced_delta > 0.0:
        return "mixed_lifecycle_training_signal"
    return "weak_lifecycle_training_signal"


def count_by(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, ""))
        out[key] = out.get(key, 0) + 1
    return out


def write_report(path: Path, payload: dict[str, Any], frame: pd.DataFrame) -> None:
    lines = [
        "# Protocol199 Lifecycle Full-Path Oracle",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Interpretation: {payload['interpretation']}",
        "",
        "## Split Summary",
        "",
        "| split | rows | frozen | hindsight oracle | oracle - frozen | post-exit best delta | material continuation | exited before oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['reported_split']} | {row['rows']} | {money(row['frozen_pnl'])} | "
            f"{money(row['oracle_pnl'])} | {money(row['oracle_minus_frozen_pnl'])} | "
            f"{money(row['post_frozen_best_minus_frozen_pnl'])} | {pct(row['material_continuation_fraction'])} | "
            f"{pct(row['exited_before_oracle_fraction'])} |"
        )
    lines.extend(["", "## Largest Continuation Examples", ""])
    if not frame.empty:
        examples = frame.sort_values("post_frozen_best_minus_frozen_pnl", ascending=False).head(12)
        lines.extend(
            [
                "| split | session | side | time | frozen | post-exit best | delta | reason | contract |",
                "|---|---|---|---|---:|---:|---:|---|---|",
            ]
        )
        for _, row in examples.iterrows():
            lines.append(
                f"| {row['reported_split']} | {row['session']} | {row['right']} | {row['time_bucket']} | "
                f"{money(row['frozen_pnl'])} | {money(row['post_frozen_best_pnl'])} | "
                f"{money(row['post_frozen_best_minus_frozen_pnl'])} | {row['exit_reason']} | {row['contract_id']} |"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Oracle rows: `{path.parent / 'full_path_oracle_rows.csv'}`",
            f"- Split summary: `{path.parent / 'split_summary.csv'}`",
            f"- Path skips: `{path.parent / 'path_skips.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
