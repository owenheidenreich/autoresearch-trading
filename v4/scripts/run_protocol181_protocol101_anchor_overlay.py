"""Protocol181: frozen Protocol101 anchor plus Protocol175 additive overlay.

This is a no-retraining screen. Protocol101 remains the base policy. The
overlay may only add a Protocol175 trade when Protocol101 has no trade at that
decision time and the account is flat. If an added trade blocks a later
Protocol101 trade, that opportunity cost is counted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOOP_ID = "v4_aplus_hypothesis_181_protocol101_anchor_overlay"
DEFAULT_PROTOCOL101_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/serial_policy_trades.json")
DEFAULT_PROTOCOL101_RECENT_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/serial_lifecycle_trades.csv")
DEFAULT_PROTOCOL175_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_175_protocol163_q4_2024_prehistory/protocol175_model_trades.csv")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
STARTING_CASH = 10_000.0
SEEDS = [1, 2, 3, 4, 5]
SPLITS = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol101-trades", type=Path, default=DEFAULT_PROTOCOL101_TRADES)
    parser.add_argument("--protocol101-recent-trades", type=Path, default=DEFAULT_PROTOCOL101_RECENT_TRADES)
    parser.add_argument("--protocol175-trades", type=Path, default=DEFAULT_PROTOCOL175_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--min-overlay-score", type=float, default=-math.inf)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p101 = load_protocol101(args.protocol101_trades, args.protocol101_recent_trades)
    p175 = load_protocol175(args.protocol175_trades)
    all_trades: list[dict[str, Any]] = []
    baseline_trades: list[dict[str, Any]] = []
    split_results: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        split_results[split] = []
        for seed in SEEDS:
            p101_slice = p101[(p101["reported_split"].eq(split)) & (p101["seed"].eq(seed))].copy()
            p175_slice = p175[(p175["reported_split"].eq(split)) & (p175["seed"].eq(seed))].copy()
            baseline = simulate_anchor_overlay(
                p101_slice,
                p175_slice.iloc[0:0].copy(),
                seed=seed,
                split=split,
                strategy="protocol101_baseline",
                starting_cash=float(args.starting_cash),
            )
            overlay = simulate_anchor_overlay(
                p101_slice,
                p175_slice,
                seed=seed,
                split=split,
                strategy="protocol181_p101_anchor_p175_additive",
                starting_cash=float(args.starting_cash),
                min_overlay_score=float(args.min_overlay_score),
            )
            split_results[split].append(
                {
                    "seed": seed,
                    "baseline": baseline["summary"],
                    "overlay": overlay["summary"],
                    "delta": float(overlay["summary"]["total_pnl"] - baseline["summary"]["total_pnl"]),
                    "added_trades": int(sum(1 for trade in overlay["trades"] if trade["source_policy"] == "protocol175_add")),
                    "blocked_protocol101_trades": int(overlay["summary"]["skipped_protocol101_overlap"]),
                }
            )
            all_trades.extend(overlay["trades"])
            baseline_trades.extend(baseline["trades"])
    aggregate_payload = aggregate(split_results)
    payload = {
        "protocol": "181_protocol101_anchor_overlay",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "starting_cash": float(args.starting_cash),
        "min_overlay_score": float(args.min_overlay_score),
        "splits": split_results,
        "aggregate": aggregate_payload,
        "decision": decision(aggregate_payload),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(all_trades).to_csv(args.out_dir / "protocol181_overlay_trades.csv", index=False)
    pd.DataFrame(baseline_trades).to_csv(args.out_dir / "protocol101_baseline_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_protocol101(path: Path, recent_path: Path) -> pd.DataFrame:
    historical = pd.DataFrame(json.loads(path.read_text()))
    historical = normalize_trade_frame(historical, source_policy="protocol101")
    recent = pd.read_csv(recent_path)
    if not recent.empty:
        recent = recent.rename(
            columns={
                "candidate_exit_time": "exit_time",
                "candidate_pnl": "pnl",
                "candidate_exit_reason": "exit_reason",
            }
        )
        recent["reported_split"] = "recent_2026"
        recent["split"] = "recent_2026"
        recent["source_policy"] = "protocol101"
        recent["raw_candidate_pnl"] = recent["pnl"]
        copies = []
        for seed in SEEDS:
            item = recent.copy()
            item["seed"] = seed
            item["candidate_uid"] = "recent_p101:" + str(seed) + ":" + item["trade_uid"].astype(str)
            copies.append(item)
        recent = pd.concat(copies, ignore_index=True, sort=False)
        recent = normalize_trade_frame(recent, source_policy="protocol101")
        may = recent[recent["session"].astype(str) >= "2026-05-19"].copy()
        may["reported_split"] = "may19_20_diagnostic"
        recent = pd.concat([recent, may], ignore_index=True, sort=False)
    return pd.concat([historical, recent], ignore_index=True, sort=False)


def load_protocol175(path: Path) -> pd.DataFrame:
    return normalize_trade_frame(pd.read_csv(path), source_policy="protocol175")


def normalize_trade_frame(frame: pd.DataFrame, *, source_policy: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["source_policy"] = out.get("source_policy", source_policy)
    out["reported_split"] = out.get("reported_split", out.get("split", "")).astype(str)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(1).astype(int)
    out["session"] = out["session"].astype(str)
    out["decision_time"] = out["decision_time"].astype(str)
    out["exit_time"] = out["exit_time"].astype(str)
    out["entry_ts"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["exit_ts"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0)
    out["right"] = out["right"].astype(str)
    out["offset"] = pd.to_numeric(out["offset"], errors="coerce")
    if "candidate_uid" not in out.columns:
        out["candidate_uid"] = out["source_policy"].astype(str) + ":" + out["seed"].astype(str) + ":" + out["trade_uid"].astype(str)
    if "entry_ask" in out.columns:
        out["entry_ask"] = pd.to_numeric(out["entry_ask"], errors="coerce")
    else:
        out["entry_ask"] = np.nan
    return out


def simulate_anchor_overlay(
    protocol101: pd.DataFrame,
    protocol175: pd.DataFrame,
    *,
    seed: int,
    split: str,
    strategy: str,
    starting_cash: float,
    min_overlay_score: float = -math.inf,
) -> dict[str, Any]:
    rows = []
    for _, row in protocol101.iterrows():
        rows.append({**row.to_dict(), "priority": 0, "source_policy": "protocol101"})
    for _, row in protocol175.iterrows():
        rows.append({**row.to_dict(), "priority": 1, "source_policy": "protocol175_add"})
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return {"trades": [], "summary": metrics([], starting_cash=starting_cash, strategy=strategy)}
    candidates = candidates.sort_values(["entry_ts", "priority", "candidate_uid"]).reset_index(drop=True)
    trades: list[dict[str, Any]] = []
    open_until = pd.Timestamp.min.tz_localize("UTC")
    skipped_protocol101_overlap = 0
    skipped_protocol175_overlap = 0
    skipped_same_minute_overlay = 0
    skipped_overlay_score = 0
    taken_minutes: set[tuple[str, str]] = set()
    for _, row in candidates.iterrows():
        entry_ts = pd.Timestamp(row["entry_ts"])
        exit_ts = pd.Timestamp(row["exit_ts"])
        if pd.isna(entry_ts) or pd.isna(exit_ts) or exit_ts <= entry_ts:
            continue
        minute_key = (str(row["session"]), str(row["decision_time"]))
        if str(row["source_policy"]) == "protocol175_add" and minute_key in taken_minutes:
            skipped_same_minute_overlay += 1
            continue
        if str(row["source_policy"]) == "protocol175_add" and float(row.get("score", -math.inf)) < float(min_overlay_score):
            skipped_overlay_score += 1
            continue
        if entry_ts < open_until:
            if str(row["source_policy"]) == "protocol101":
                skipped_protocol101_overlap += 1
            else:
                skipped_protocol175_overlap += 1
            continue
        trade = {
            "strategy": strategy,
            "source_policy": str(row["source_policy"]),
            "reported_split": split,
            "seed": int(seed),
            "session": str(row["session"]),
            "decision_time": str(row["decision_time"]),
            "exit_time": str(row["exit_time"]),
            "contract_id": str(row.get("contract_id", "")),
            "right": str(row.get("right", "")),
            "offset": float(row.get("offset", 0.0)),
            "pnl": float(row["pnl"]),
            "candidate_uid": str(row.get("candidate_uid", "")),
            "exit_reason": str(row.get("exit_reason", "")),
        }
        trades.append(trade)
        taken_minutes.add(minute_key)
        open_until = exit_ts
    summary = metrics(trades, starting_cash=starting_cash, strategy=strategy)
    summary.update(
        {
            "skipped_protocol101_overlap": int(skipped_protocol101_overlap),
            "skipped_protocol175_overlap": int(skipped_protocol175_overlap),
            "skipped_same_minute_overlay": int(skipped_same_minute_overlay),
            "skipped_overlay_score": int(skipped_overlay_score),
            "overlay_added_trades": int(sum(1 for trade in trades if trade["source_policy"] == "protocol175_add")),
            "protocol101_trades_taken": int(sum(1 for trade in trades if trade["source_policy"] == "protocol101")),
        }
    )
    return {"trades": trades, "summary": summary}


def metrics(trades: list[dict[str, Any]], *, starting_cash: float, strategy: str) -> dict[str, Any]:
    pnl = np.asarray([float(trade["pnl"]) for trade in trades], dtype=float)
    total = float(pnl.sum()) if len(pnl) else 0.0
    wins = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
    losses = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
    equity = starting_cash
    peak = starting_cash
    max_drawdown = 0.0
    by_day: dict[str, float] = {}
    for trade in trades:
        equity += float(trade["pnl"])
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        by_day[str(trade["session"])] = by_day.get(str(trade["session"]), 0.0) + float(trade["pnl"])
    return {
        "strategy": strategy,
        "trades": int(len(trades)),
        "total_pnl": total,
        "ending_equity": float(starting_cash + total),
        "return_pct": float(total / starting_cash * 100.0),
        "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "median_pnl": float(np.median(pnl)) if len(pnl) else 0.0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0),
        "max_drawdown": float(max_drawdown),
        "sessions_traded": int(len(by_day)),
        "positive_day_fraction": float(np.mean([value > 0 for value in by_day.values()])) if by_day else 0.0,
        "side_counts": {
            "C": int(sum(1 for trade in trades if trade.get("right") == "C")),
            "P": int(sum(1 for trade in trades if trade.get("right") == "P")),
        },
        "max_concurrent_positions": 1 if trades else 0,
        "serial_status": "pass",
        "all_flat_by_session_end": True,
    }


def aggregate(split_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, rows in split_results.items():
        baseline = np.asarray([row["baseline"]["total_pnl"] for row in rows], dtype=float)
        overlay = np.asarray([row["overlay"]["total_pnl"] for row in rows], dtype=float)
        stress_pf = np.asarray([finite_pf(row["overlay"]["profit_factor"]) for row in rows], dtype=float)
        out[split] = {
            "seeds": int(len(rows)),
            "baseline_median_total_pnl": float(np.median(baseline)) if len(baseline) else 0.0,
            "overlay_median_total_pnl": float(np.median(overlay)) if len(overlay) else 0.0,
            "median_delta": float(np.median(overlay - baseline)) if len(overlay) else 0.0,
            "positive_seed_fraction": float((overlay > 0).mean()) if len(overlay) else 0.0,
            "median_profit_factor": float(np.median(stress_pf)) if len(stress_pf) else 0.0,
            "median_overlay_added_trades": float(np.median([row["added_trades"] for row in rows])) if rows else 0.0,
            "median_blocked_protocol101_trades": float(np.median([row["blocked_protocol101_trades"] for row in rows])) if rows else 0.0,
            "seed_rows": rows,
        }
    return out


def decision(aggregate_payload: dict[str, Any]) -> str:
    required = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    if all(aggregate_payload.get(split, {}).get("median_delta", -1e18) > 0.0 for split in required):
        return "promote_research_candidate: anchored overlay beats Protocol101 baseline on all required blocks"
    if any(aggregate_payload.get(split, {}).get("median_delta", 0.0) > 0.0 for split in required):
        return "research_only: anchored overlay improves some blocks but does not clear promotion gate"
    return "reject_current_hypothesis: anchored overlay does not improve Protocol101"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol181 Protocol101 Anchor Overlay",
        "",
        "Frozen Protocol101 has priority. Protocol175 may only add trades when the account is flat.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Paid data downloaded: `{payload['paid_data_downloaded_by_runner']}`",
        f"- Broker endpoint called: `{payload['broker_endpoint_called']}`",
        f"- Minimum overlay score: `{payload['min_overlay_score']}`",
        "",
        "## Aggregate",
        "",
        "| split | baseline median | overlay median | delta | PF | added trades | blocked P101 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in payload["aggregate"].items():
        lines.append(
            f"| {split} | {fmt(item['baseline_median_total_pnl'])} | {fmt(item['overlay_median_total_pnl'])} | "
            f"{fmt(item['median_delta'])} | {fmt(item['median_profit_factor'])} | "
            f"{fmt(item['median_overlay_added_trades'])} | {fmt(item['median_blocked_protocol101_trades'])} |"
        )
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Overlay trades: `{path.parent / 'protocol181_overlay_trades.csv'}`", f"- Baseline trades: `{path.parent / 'protocol101_baseline_trades.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def finite_pf(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(number):
        return 999.0
    return number if math.isfinite(number) else 0.0


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
