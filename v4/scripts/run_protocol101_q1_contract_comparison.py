"""Compare frozen Protocol101 under legacy and live-reproducible Q1 contracts.

This reporter consumes completed Protocol161 entry replays and Protocol162
serial lifecycle replays. It does not build data, train models, tune thresholds,
or contact a broker or data vendor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DECISION_KEYS = ["session", "decision_time"]
TRADE_KEYS = ["session", "decision_time", "contract_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-protocol161-dir", type=Path, required=True)
    parser.add_argument("--live-protocol161-dir", type=Path, required=True)
    parser.add_argument("--legacy-protocol162-dir", type=Path, required=True)
    parser.add_argument("--live-protocol162-dir", type=Path, required=True)
    parser.add_argument("--feature-audit-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _decision_comparison(legacy: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *DECISION_KEYS,
        "action",
        "reason",
        "candidate_count",
        "above_min_edge_count",
        "max_edge",
        "best_call_edge",
        "best_put_edge",
        "selected_contract_id",
        "selected_right",
        "selected_edge",
    ]
    left = legacy[[column for column in columns if column in legacy.columns]].copy()
    right = live[[column for column in columns if column in live.columns]].copy()
    out = left.merge(right, on=DECISION_KEYS, how="outer", suffixes=("_legacy", "_live"), indicator=True)
    out["action_match"] = out.get("action_legacy").fillna("") == out.get("action_live").fillna("")
    out["selected_contract_match"] = (
        out.get("selected_contract_id_legacy").fillna("")
        == out.get("selected_contract_id_live").fillna("")
    )
    if "max_edge_legacy" in out and "max_edge_live" in out:
        out["max_edge_delta"] = out["max_edge_live"] - out["max_edge_legacy"]
    if "candidate_count_legacy" in out and "candidate_count_live" in out:
        out["candidate_count_delta"] = out["candidate_count_live"] - out["candidate_count_legacy"]
    return out.sort_values(DECISION_KEYS).reset_index(drop=True)


def _taken_trades(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "serial_status" in out.columns:
        out = out[out["serial_status"].fillna("").astype(str).str.lower().eq("taken")]
    return out.sort_values([column for column in ("session", "decision_time") if column in out.columns]).reset_index(drop=True)


def _pnl_column(frame: pd.DataFrame) -> str:
    for column in ("candidate_pnl", "pnl", "baseline_path_pnl"):
        if column in frame.columns:
            return column
    raise ValueError("serial lifecycle trades do not contain a recognized PnL column")


def _trade_metrics(frame: pd.DataFrame, *, starting_equity: float) -> dict[str, Any]:
    trades = _taken_trades(frame)
    pnl_column = _pnl_column(trades)
    pnl = pd.to_numeric(trades[pnl_column], errors="coerce").fillna(0.0)
    equity = float(starting_equity) + pnl.cumsum()
    running_peak = pd.concat([pd.Series([float(starting_equity)]), equity], ignore_index=True).cummax().iloc[1:]
    drawdown = equity.reset_index(drop=True) - running_peak.reset_index(drop=True)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    premium = pd.Series(np.zeros(len(trades)), index=trades.index, dtype=float)
    if "entry_ask" in trades.columns:
        premium = pd.to_numeric(trades["entry_ask"], errors="coerce").fillna(0.0) * 100.0
    elif "protocol101_selected_ask" in trades.columns:
        premium = pd.to_numeric(trades["protocol101_selected_ask"], errors="coerce").fillna(0.0) * 100.0
    by_day = pnl.groupby(trades["session"].astype(str)).sum() if "session" in trades.columns else pd.Series(dtype=float)
    positive_total = float(pnl.sum())
    top_days = by_day.sort_values(ascending=False)
    top_trades = pnl.sort_values(ascending=False)
    return {
        "trades": int(len(trades)),
        "sessions_with_trades": int(by_day.size),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "total_pnl": positive_total,
        "starting_equity": float(starting_equity),
        "ending_equity": float(starting_equity + positive_total),
        "max_drawdown_dollars": float(drawdown.min()) if len(drawdown) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "premium_deployed": float(premium.sum()),
        "return_on_premium": positive_total / float(premium.sum()) if float(premium.sum()) > 0 else None,
        "top_1_day_pnl_share": float(top_days.head(1).sum() / positive_total) if positive_total > 0 else None,
        "top_5_day_pnl_share": float(top_days.head(5).sum() / positive_total) if positive_total > 0 else None,
        "top_10_trade_pnl_share": float(top_trades.head(10).sum() / positive_total) if positive_total > 0 else None,
    }


def _trade_comparison(legacy: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    left = _taken_trades(legacy)
    right = _taken_trades(live)
    columns = [*TRADE_KEYS, "right", "edge", "entry_ask", "candidate_pnl", "candidate_exit_reason", "candidate_exit_time"]
    left = left[[column for column in columns if column in left.columns]]
    right = right[[column for column in columns if column in right.columns]]
    return left.merge(right, on=TRADE_KEYS, how="outer", suffixes=("_legacy", "_live"), indicator=True)


def _daily_comparison(legacy: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    def daily(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        trades = _taken_trades(frame)
        pnl = _pnl_column(trades)
        out = trades.assign(_pnl=pd.to_numeric(trades[pnl], errors="coerce").fillna(0.0)).groupby("session").agg(
            **{f"trades_{suffix}": ("_pnl", "size"), f"pnl_{suffix}": ("_pnl", "sum")}
        )
        return out.reset_index()

    return daily(legacy, "legacy").merge(daily(live, "live"), on="session", how="outer").fillna(0).sort_values("session")


def _write_equity(path: Path, legacy: pd.DataFrame, live: pd.DataFrame, *, starting_equity: float) -> None:
    traces = []
    for name, frame in (("historical-default", legacy), ("protocol101-live-v1", live)):
        trades = _taken_trades(frame)
        pnl = pd.to_numeric(trades[_pnl_column(trades)], errors="coerce").fillna(0.0)
        x = trades.get("candidate_exit_time", trades.get("decision_time", pd.Series(range(len(trades)))))
        traces.append(
            {
                "x": [str(value) for value in x.tolist()],
                "y": [float(value) for value in (float(starting_equity) + pnl.cumsum()).tolist()],
                "mode": "lines",
                "name": name,
            }
        )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Protocol101 Q1 Feature-Contract Equity Comparison</title>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        "<style>html,body,#chart{width:100%;height:100%;margin:0;background:#fff;font-family:system-ui}</style>"
        "</head><body><div id='chart'></div><script>"
        f"const traces={json.dumps(traces)};"
        "Plotly.newPlot('chart',traces,{title:'Protocol101 Q1 Feature-Contract Equity Comparison',"
        "xaxis:{title:'Exit'},yaxis:{title:'Equity ($)'},hovermode:'x unified'},"
        "{responsive:true});</script></body></html>\n"
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    legacy_decisions = _read_csv(args.legacy_protocol161_dir / "replay_decisions.csv")
    live_decisions = _read_csv(args.live_protocol161_dir / "replay_decisions.csv")
    legacy_trades = _read_csv(args.legacy_protocol162_dir / "serial_lifecycle_trades.csv")
    live_trades = _read_csv(args.live_protocol162_dir / "serial_lifecycle_trades.csv")

    decisions = _decision_comparison(legacy_decisions, live_decisions)
    trades = _trade_comparison(legacy_trades, live_trades)
    daily = _daily_comparison(legacy_trades, live_trades)
    decisions.to_csv(args.out_dir / "candidate_score_action_comparison.csv", index=False)
    trades.to_csv(args.out_dir / "trade_comparison.csv", index=False)
    daily.to_csv(args.out_dir / "daily_comparison.csv", index=False)
    _write_equity(args.out_dir / "equity.html", legacy_trades, live_trades, starting_equity=args.starting_equity)

    feature_audit = None
    if args.feature_audit_summary is not None and args.feature_audit_summary.exists():
        feature_audit = json.loads(args.feature_audit_summary.read_text())
    action_mismatches = int((~decisions["action_match"]).sum())
    payload = {
        "schema_version": "Protocol101Q1FeatureContractComparisonV1",
        "status": "review",
        "model_training": False,
        "threshold_tuning": False,
        "broker_endpoint_called": False,
        "decision_rows": int(len(decisions)),
        "action_mismatches": action_mismatches,
        "selected_contract_matches": int(decisions["selected_contract_match"].sum()),
        "max_edge_delta": {
            "median_abs": float(decisions["max_edge_delta"].abs().median()) if "max_edge_delta" in decisions else None,
            "p95_abs": float(decisions["max_edge_delta"].abs().quantile(0.95)) if "max_edge_delta" in decisions else None,
            "max_abs": float(decisions["max_edge_delta"].abs().max()) if "max_edge_delta" in decisions else None,
        },
        "legacy": _trade_metrics(legacy_trades, starting_equity=args.starting_equity),
        "protocol101_live_v1": _trade_metrics(live_trades, starting_equity=args.starting_equity),
        "feature_audit": feature_audit,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = [
        "# Protocol101 Q1 Feature-Contract Comparison",
        "",
        f"- Decision rows: `{payload['decision_rows']}`",
        f"- Action mismatches: `{action_mismatches}`",
        f"- Legacy trades / PnL: `{payload['legacy']['trades']}` / `{payload['legacy']['total_pnl']:.2f}`",
        f"- Live-contract trades / PnL: `{payload['protocol101_live_v1']['trades']}` / `{payload['protocol101_live_v1']['total_pnl']:.2f}`",
        f"- Legacy max drawdown: `{payload['legacy']['max_drawdown_dollars']:.2f}`",
        f"- Live-contract max drawdown: `{payload['protocol101_live_v1']['max_drawdown_dollars']:.2f}`",
        "- Model training: `false`",
        "- Threshold tuning: `false`",
        "- Broker endpoint called: `false`",
        "",
        "## Outputs",
        "",
        "- `candidate_score_action_comparison.csv`",
        "- `trade_comparison.csv`",
        "- `daily_comparison.csv`",
        "- `equity.html`",
        "- `summary.json`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
