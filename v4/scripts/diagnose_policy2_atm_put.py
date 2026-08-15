"""Diagnose the March ATM-put signal under the 45-minute policy."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v4.model.action_pilot import (
    ActionDecision,
    load_action_decisions,
    simulate_action_baseline,
)
from v4.model.supervised_pilot import metrics_for_trades, session_from_path, split_name
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--policy-index", type=int, default=2, choices=sorted(POLICY_META))
    p.add_argument("--out", type=Path, default=Path("v4/audit/policy2_atm_put_diagnostic.json"))
    p.add_argument("--md-out", type=Path, default=Path("v4/audit/policy2_atm_put_diagnostic.md"))
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _time_bucket(ts: pd.Timestamp) -> str:
    local = ts.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    if minute < 10 * 60 + 30:
        return "09:30-10:29"
    if minute < 12 * 60:
        return "10:30-11:59"
    if minute < 14 * 60:
        return "12:00-13:59"
    return "14:00-15:30"


def _trade_records(decisions: list[ActionDecision], *, kind: str, cooldown: int) -> list[dict]:
    trades = simulate_action_baseline(decisions, kind=kind, cooldown_minutes=cooldown)
    by_key = {(t.session, t.decision_time, t.right): t for t in trades}
    records = []
    for decision in decisions:
        for action, right in ((1, "C"), (2, "P")):
            key = (decision.session, decision.decision_time.isoformat(), right)
            trade = by_key.get(key)
            if trade is None:
                continue
            ts = pd.Timestamp(decision.decision_time)
            market = decision.market_last
            records.append(
                {
                    "session": decision.session,
                    "decision_time": decision.decision_time.isoformat(),
                    "right": right,
                    "pnl": float(trade.pnl),
                    "offset": float(decision.offsets[action]),
                    "time_bucket": _time_bucket(ts),
                    "spx_close": float(market[0]),
                    "vix_proxy": float(market[1]),
                    "spx_vwap": float(market[2]),
                    "omar": float(market[3]),
                    "session_range": float(market[4]),
                    "momentum_5m": float(market[5]),
                    "momentum_15m": float(market[6]),
                    "above_vwap": bool(float(market[0]) > float(market[2])),
                    "omar_positive": bool(float(market[3]) > 0),
                    "momentum_15m_positive": bool(float(market[6]) > 0),
                }
            )
    return records


def _group_summary(records: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(float(record["pnl"]))
    out = []
    for value, pnls in sorted(grouped.items()):
        arr = np.asarray(pnls, dtype=float)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        gross_loss = abs(float(losses.sum()))
        out.append(
            {
                key: value,
                "trades": int(len(arr)),
                "total_pnl": float(arr.sum()),
                "avg_pnl": float(arr.mean()),
                "win_rate": float((arr > 0).mean()),
                "profit_factor": float(wins.sum() / gross_loss) if gross_loss else float("inf"),
            }
        )
    return out


def _daily_summary(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["session"]].append(float(record["pnl"]))
    rows = []
    for session, pnls in sorted(grouped.items()):
        arr = np.asarray(pnls, dtype=float)
        rows.append(
            {
                "session": session,
                "trades": int(len(arr)),
                "total_pnl": float(arr.sum()),
                "avg_pnl": float(arr.mean()),
                "win_rate": float((arr > 0).mean()),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    policy_name, cooldown = POLICY_META[args.policy_index]
    paths = _paths_by_split(args.data_dir)
    decisions = {
        split: load_action_decisions(files, policy_index=args.policy_index)
        for split, files in paths.items()
    }
    payload = {
        "policy": policy_name,
        "cooldown_minutes": cooldown,
        "regime_caveat": (
            "Treat Jan-Mar 2026 as real market truth, but only one environment slice. "
            "These diagnostics identify hypotheses, not generalizable rules."
        ),
        "splits": {},
    }
    for split, split_decisions in decisions.items():
        split_payload = {}
        for kind in ("atm_call", "atm_put", "vwap_omar"):
            trades = simulate_action_baseline(split_decisions, kind=kind, cooldown_minutes=cooldown)
            records = _trade_records(split_decisions, kind=kind, cooldown=cooldown)
            split_payload[kind] = {
                "metrics": metrics_for_trades(trades),
                "by_day": _daily_summary(records),
                "by_time_bucket": _group_summary(records, "time_bucket"),
                "by_above_vwap": _group_summary(records, "above_vwap"),
                "by_omar_positive": _group_summary(records, "omar_positive"),
                "by_momentum_15m_positive": _group_summary(records, "momentum_15m_positive"),
            }
        payload["splits"][split] = split_payload

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    test_put = payload["splits"]["test"]["atm_put"]
    lines = [
        "# Policy 2 ATM Put Diagnostic",
        "",
        f"Policy: `{policy_name}`",
        "",
        "Environment caveat: Jan-Mar 2026 is real market truth, but only one environment slice. "
        "These diagnostics identify hypotheses, not generalizable rules.",
        "",
        "## March Holdout ATM Put",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in test_put["metrics"].items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.3f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines += ["", "## By Time Bucket", "", "| Bucket | Trades | PnL | Avg | Win Rate | PF |", "|---|---:|---:|---:|---:|---:|"]
    for row in test_put["by_time_bucket"]:
        lines.append(
            f"| {row['time_bucket']} | {row['trades']} | {row['total_pnl']:.0f} | "
            f"{row['avg_pnl']:.1f} | {row['win_rate']:.3f} | {row['profit_factor']:.3f} |"
        )
    lines += ["", "## By Day", "", "| Session | Trades | PnL | Avg | Win Rate |", "|---|---:|---:|---:|---:|"]
    for row in test_put["by_day"]:
        lines.append(
            f"| {row['session']} | {row['trades']} | {row['total_pnl']:.0f} | "
            f"{row['avg_pnl']:.1f} | {row['win_rate']:.3f} |"
        )
    args.md_out.write_text("\n".join(lines) + "\n")
    print(args.out)
    print(args.md_out)
    print(json.dumps(test_put["metrics"], indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
