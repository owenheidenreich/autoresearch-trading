"""AUDIT_2026_05_22_POSITION_ACTION_DP_ORACLE_V1.

Historically Protocol229. The previous lifecycle labels were path-local and did
not produce useful hold/exit behavior. This audit builds the next training
target: a dynamic-programming oracle over position actions for each selected
account-aware trade.

Actions while holding:
* hold the current quantity
* reduce part of the position
* exit all remaining contracts

This is hindsight supervision for dataset design, not a tradable strategy.
No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol224_scale_in_out_path_audit import (
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_SIZED_TRADES,
    forced_flat_timestamp,
    load_session_quotes,
)


ROLE_LABEL = "AUDIT_2026_05_22_POSITION_ACTION_DP_ORACLE_V1"
HISTORICAL_ID = "Protocol229"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_229_position_action_dp_oracle")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sized-trades", type=Path, default=DEFAULT_SIZED_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--extra-slippage-per-side", type=float, default=0.0)
    parser.add_argument("--forced-flat-time", default="15:30")
    parser.add_argument("--max-quantity-state", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = load_sized_trades(args.sized_trades, float(args.starting_cash), float(args.extra_slippage_per_side))
    rows, labels, skips = build_dp_rows(
        trades,
        normalized_dir=args.normalized_dir,
        forced_flat_time=str(args.forced_flat_time),
        max_quantity_state=int(args.max_quantity_state),
    )
    detail = pd.DataFrame(rows)
    label_frame = pd.DataFrame(labels)
    skip_frame = pd.DataFrame(skips)
    detail.to_csv(args.out_dir / "position_dp_oracle_trades.csv", index=False)
    label_frame.to_csv(args.out_dir / "position_dp_action_labels.csv", index=False)
    skip_frame.to_csv(args.out_dir / "path_skips.csv", index=False)
    split_summary = summarize(detail, label_frame)
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / position-action dynamic-programming oracle",
        "changes_paper_default": False,
        "candidate_label": "ACCOUNT_AWARE_POSITION_ACTION_ORACLE_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.sized_trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "starting_cash": float(args.starting_cash),
        "extra_slippage_per_side": float(args.extra_slippage_per_side),
        "max_quantity_state": int(args.max_quantity_state),
        "row_counts": {
            "input_trades": int(len(trades)),
            "oracle_trades": int(len(detail)),
            "label_rows": int(len(label_frame)),
            "path_skips": int(len(skip_frame)),
        },
        "path_skip_counts": count_by(skip_frame, "skip_reason"),
        "aggregate": split_summary.to_dict("records"),
        "decision": decide(split_summary),
        "next_experiment": (
            "If reduce/exit actions materially improve the oracle without relying only on full hindsight, train a causal "
            "action classifier over these labels and validate it under serial account replay."
        ),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "oracle_trades": str(args.out_dir / "position_dp_oracle_trades.csv"),
            "action_labels": str(args.out_dir / "position_dp_action_labels.csv"),
            "split_summary": str(args.out_dir / "split_summary.csv"),
            "path_skips": str(args.out_dir / "path_skips.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_sized_trades(path: Path, starting_cash: float, extra_slippage_per_side: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["starting_cash"].astype(float).eq(starting_cash)
        & frame["extra_slippage_per_side"].astype(float).eq(extra_slippage_per_side)
        & (pd.to_numeric(frame["aa_contracts"], errors="coerce") > 0)
        & ~frame["reported_split"].astype(str).eq("march_2026")
    ].copy()
    for column in ["decision_time", "exit_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in ["entry_ask", "entry_premium", "aa_contracts", "aa_pnl", "pnl"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame.dropna(subset=["reported_split", "session", "decision_time", "exit_time", "contract_id", "entry_ask", "aa_contracts"]).reset_index(drop=True)


def build_dp_rows(
    trades: pd.DataFrame,
    *,
    normalized_dir: Path,
    forced_flat_time: str,
    max_quantity_state: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for session, group in trades.groupby("session", sort=True):
        contracts = set(group["contract_id"].astype(str).unique())
        quotes = load_session_quotes(normalized_dir, str(session), contracts)
        if quotes.empty:
            skips.extend(base_skip(row, "missing_session_or_contract_quotes") for _, row in group.iterrows())
            continue
        by_contract = {
            str(contract_id): part.sort_values("quote_time").reset_index(drop=True)
            for contract_id, part in quotes.groupby("contract_id", sort=False)
        }
        forced_flat = forced_flat_timestamp(str(session), forced_flat_time)
        for _, trade in group.iterrows():
            result, label_rows, skip = dp_for_trade(trade, by_contract.get(str(trade["contract_id"])), forced_flat, max_quantity_state)
            if skip:
                skips.append(skip)
            else:
                rows.append(result)
                labels.extend(label_rows)
    return rows, labels, skips


def dp_for_trade(
    trade: pd.Series,
    quotes: pd.DataFrame | None,
    forced_flat: pd.Timestamp,
    max_quantity_state: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    if quotes is None or quotes.empty:
        return None, [], base_skip(trade, "missing_contract_quotes")
    decision_ts = pd.Timestamp(trade["decision_time"])
    entry_ask = finite_float(trade.get("entry_ask"), math.nan)
    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        return None, [], base_skip(trade, "invalid_entry_ask")
    path = quotes[(quotes["quote_time"] >= decision_ts) & (quotes["quote_time"] <= forced_flat)].sort_values("quote_time").reset_index(drop=True)
    if path.empty:
        return None, [], base_skip(trade, "missing_post_entry_path")
    bid = pd.to_numeric(path["bid"], errors="coerce").to_numpy(dtype=float)
    unit_pnl = (bid - entry_ask) * CONTRACT_MULTIPLIER
    valid = np.isfinite(unit_pnl)
    if not valid.any():
        return None, [], base_skip(trade, "invalid_unit_pnl_path")
    path = path[valid].reset_index(drop=True)
    unit_pnl = unit_pnl[valid]
    quote_times = pd.to_datetime(path["quote_time"], utc=True)
    qty_actual = int(max(1, finite_float(trade.get("aa_contracts"), 1.0)))
    qty_state = int(min(qty_actual, max_quantity_state))
    scale = qty_actual / qty_state
    value, action, reduce_qty = solve_position_dp(unit_pnl, qty_state)
    oracle_value_scaled = float(value[0, qty_state] * scale)
    baseline_unit = finite_float(trade.get("pnl"), 0.0)
    baseline_position = baseline_unit * qty_actual
    actions = reconstruct_actions(action, reduce_qty, unit_pnl, quote_times, qty_state)
    label_rows = []
    for item in actions:
        label_rows.append(
            {
                "reported_split": str(trade.get("reported_split")),
                "session": str(trade.get("session")),
                "decision_time": pd.Timestamp(trade["decision_time"]).isoformat(),
                "contract_id": str(trade.get("contract_id")),
                "right": str(trade.get("right")),
                "quantity_actual": qty_actual,
                "quantity_state": qty_state,
                "state_time": item["time"],
                "state_index": item["idx"],
                "state_quantity": item["qty"],
                "oracle_action": item["action"],
                "reduce_quantity_state": item["reduce_qty"],
                "unit_pnl": item["unit_pnl"],
                "position_value_scaled": item["position_value"] * scale,
            }
        )
    first_reduce = next((item for item in actions if item["action"] == "reduce"), None)
    first_exit = next((item for item in actions if item["action"] == "exit"), None)
    result = {
        "reported_split": str(trade.get("reported_split")),
        "session": str(trade.get("session")),
        "decision_time": pd.Timestamp(trade["decision_time"]).isoformat(),
        "contract_id": str(trade.get("contract_id")),
        "right": str(trade.get("right")),
        "offset": finite_float(trade.get("offset"), math.nan),
        "quantity_actual": qty_actual,
        "quantity_state": qty_state,
        "baseline_position_pnl": baseline_position,
        "oracle_position_pnl": oracle_value_scaled,
        "oracle_delta_vs_baseline": oracle_value_scaled - baseline_position,
        "path_mfe_position_pnl": float(np.nanmax(unit_pnl) * qty_actual),
        "path_mae_position_pnl": float(np.nanmin(unit_pnl) * qty_actual),
        "oracle_action_count": len(actions),
        "oracle_reduce_count": sum(1 for item in actions if item["action"] == "reduce"),
        "oracle_exits_immediately": bool(actions and actions[0]["action"] == "exit"),
        "first_reduce_time": first_reduce["time"] if first_reduce else "",
        "first_reduce_unit_pnl": first_reduce["unit_pnl"] if first_reduce else math.nan,
        "first_reduce_was_winner": bool(first_reduce and first_reduce["unit_pnl"] > 0.0),
        "first_exit_time": first_exit["time"] if first_exit else "",
        "first_exit_unit_pnl": first_exit["unit_pnl"] if first_exit else math.nan,
        "path_points": int(len(unit_pnl)),
    }
    return result, label_rows, None


def solve_position_dp(unit_pnl: np.ndarray, max_qty: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(unit_pnl)
    value = np.zeros((n, max_qty + 1), dtype=float)
    action = np.full((n, max_qty + 1), "hold", dtype=object)
    reduce_qty = np.zeros((n, max_qty + 1), dtype=int)
    for qty in range(1, max_qty + 1):
        value[n - 1, qty] = qty * unit_pnl[n - 1]
        action[n - 1, qty] = "exit"
    for idx in range(n - 2, -1, -1):
        for qty in range(1, max_qty + 1):
            best_value = value[idx + 1, qty]
            best_action = "hold"
            best_reduce = 0
            exit_value = qty * unit_pnl[idx]
            if exit_value > best_value:
                best_value = exit_value
                best_action = "exit"
            for rq in reduce_candidates(qty):
                reduce_value = rq * unit_pnl[idx] + value[idx + 1, qty - rq]
                if reduce_value > best_value:
                    best_value = reduce_value
                    best_action = "reduce"
                    best_reduce = rq
            value[idx, qty] = best_value
            action[idx, qty] = best_action
            reduce_qty[idx, qty] = best_reduce
    return value, action, reduce_qty


def reduce_candidates(qty: int) -> list[int]:
    if qty <= 1:
        return []
    candidates = {1, max(1, qty // 2), qty - 1}
    return sorted(candidate for candidate in candidates if 0 < candidate < qty)


def reconstruct_actions(
    action: np.ndarray,
    reduce_qty: np.ndarray,
    unit_pnl: np.ndarray,
    quote_times: pd.Series,
    qty: int,
) -> list[dict[str, Any]]:
    out = []
    idx = 0
    while idx < len(unit_pnl) and qty > 0:
        act = str(action[idx, qty])
        rq = int(reduce_qty[idx, qty]) if act == "reduce" else 0
        out.append(
            {
                "idx": int(idx),
                "time": pd.Timestamp(quote_times.iloc[idx]).isoformat(),
                "qty": int(qty),
                "action": act,
                "reduce_qty": rq,
                "unit_pnl": float(unit_pnl[idx]),
                "position_value": float(unit_pnl[idx] * (rq if act == "reduce" else qty)),
            }
        )
        if act == "exit":
            break
        if act == "reduce":
            qty -= rq
        idx += 1
    return out


def summarize(detail: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for split, group in detail.groupby("reported_split", sort=True):
        label_group = labels[labels["reported_split"].astype(str).eq(str(split))]
        rows.append(
            {
                "reported_split": split,
                "trades": int(len(group)),
                "baseline_position_pnl": float(group["baseline_position_pnl"].sum()),
                "oracle_position_pnl": float(group["oracle_position_pnl"].sum()),
                "oracle_delta_vs_baseline": float(group["oracle_delta_vs_baseline"].sum()),
                "oracle_reduce_trade_fraction": float((group["oracle_reduce_count"] > 0).mean()),
                "oracle_immediate_exit_fraction": float(group["oracle_exits_immediately"].mean()),
                "first_reduce_winner_fraction": float(group.loc[group["oracle_reduce_count"] > 0, "first_reduce_was_winner"].mean()) if (group["oracle_reduce_count"] > 0).any() else 0.0,
                "label_hold_fraction": float(label_group["oracle_action"].eq("hold").mean()) if not label_group.empty else 0.0,
                "label_reduce_fraction": float(label_group["oracle_action"].eq("reduce").mean()) if not label_group.empty else 0.0,
                "label_exit_fraction": float(label_group["oracle_action"].eq("exit").mean()) if not label_group.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "position_action_dp_blocked_no_rows"
    if (summary["oracle_delta_vs_baseline"] > 0.0).all() and (summary["oracle_reduce_trade_fraction"] > 0.10).any():
        return "position_action_oracle_supports_reduce_exit_training"
    if (summary["oracle_delta_vs_baseline"] > 0.0).all():
        return "position_action_oracle_supports_exit_training_only"
    return "position_action_oracle_weak_or_inconsistent"


def count_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def base_skip(row: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "reported_split": str(row.get("reported_split", "")),
        "session": str(row.get("session", "")),
        "decision_time": str(row.get("decision_time", "")),
        "contract_id": str(row.get("contract_id", "")),
        "skip_reason": reason,
    }


def finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Split Summary",
        "",
        "| split | trades | baseline pnl | oracle pnl | delta | reduce trade % | immediate exit % | first reduce winner % | hold labels | reduce labels | exit labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['reported_split']} | {row['trades']} | {money(row['baseline_position_pnl'])} | "
            f"{money(row['oracle_position_pnl'])} | {money(row['oracle_delta_vs_baseline'])} | "
            f"{pct(row['oracle_reduce_trade_fraction'])} | {pct(row['oracle_immediate_exit_fraction'])} | "
            f"{pct(row['first_reduce_winner_fraction'])} | {pct(row['label_hold_fraction'])} | "
            f"{pct(row['label_reduce_fraction'])} | {pct(row['label_exit_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Oracle trades: `{payload['outputs']['oracle_trades']}`",
            f"- Action labels: `{payload['outputs']['action_labels']}`",
            f"- Split summary: `{payload['outputs']['split_summary']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def money(value: Any) -> str:
    number = finite_float(value, 0.0)
    sign = "-" if number < 0.0 else ""
    return f"{sign}${abs(number):,.0f}"


def pct(value: Any) -> str:
    return f"{finite_float(value, 0.0) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
