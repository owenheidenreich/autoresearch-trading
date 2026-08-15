"""Protocol194: five-seed confirmation for the full-action surface-edge policy.

This runner consolidates the Protocol081 replays of the Protocol190/192 entry
policies. It does not retrain, download data, or touch broker endpoints. Its
job is to answer one narrower question:

Did the full-action entry policy remain better than the frozen strict serial
Protocol101 baseline after all five seeds were replayed through the frozen
Protocol081 lifecycle stack under one-account serial rules?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOOP_ID = "v4_aplus_hypothesis_194_full_action_surface_edge_5seed_confirmation"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_REPLAY_DIRS = [
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_191_protocol081_replay_of_protocol190_entries"),
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_193_protocol081_replay_of_protocol192_seed45_entries"),
]
DEFAULT_BASELINE_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_190_full_coverage_surface_edge_baseline_exit_screen/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", action="append", type=Path, default=None)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay_dirs = args.replay_dir or DEFAULT_REPLAY_DIRS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [_load_summary(path) for path in replay_dirs]
    trades = _load_trades(replay_dirs)
    baseline = _load_protocol101_baselines(args.baseline_summary)
    aggregate = _combine_aggregates(summaries, baseline)
    invariants = _serial_invariants(trades)
    concentration = _concentration(trades)

    payload = {
        "protocol": "194_full_action_surface_edge_5seed_confirmation",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "source_replay_dirs": [str(path) for path in replay_dirs],
        "baseline_summary": str(args.baseline_summary),
        "trade_rows": int(len(trades)),
        "aggregate": aggregate,
        "invariants": invariants,
        "concentration": concentration,
        "decision": _decision(aggregate, invariants),
        "caveat": (
            "Protocol190/192 were trained with fast executable baseline-exit labels. "
            "Protocol194 confirms their selected entries survive frozen Protocol081 "
            "serial replay, but it does not replace live-paper timing validation."
        ),
    }

    trades.to_csv(args.out_dir / "protocol194_protocol081_5seed_serial_trades.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def _load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text())


def _load_trades(replay_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for path in replay_dirs:
        trades_path = path / "protocol191_protocol081_serial_trades.csv"
        if not trades_path.exists():
            raise FileNotFoundError(trades_path)
        frame = pd.read_csv(trades_path)
        frame["source_replay_dir"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["decision_time", "candidate_exit_time", "exit_time"]:
        if col in out:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out.sort_values(["fold", "seed", "reported_split", "session", "decision_time", "contract_id"]).reset_index(drop=True)


def _load_protocol101_baselines(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    baselines = payload.get("frozen_protocol101_baselines", {})
    if isinstance(baselines, dict):
        return {str(k): _finite_float(v, 0.0) for k, v in baselines.items()}
    return {}


def _combine_aggregates(summaries: list[dict[str, Any]], baseline: dict[str, float]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        for split, row in summary.get("aggregate", {}).items():
            by_split.setdefault(str(split), []).extend(row.get("seed_rows", []))

    out: dict[str, Any] = {}
    for split, seed_rows in sorted(by_split.items()):
        seed_rows = sorted(seed_rows, key=lambda row: int(row["seed"]))
        pnl_values = [_finite_float(row.get("total_pnl"), 0.0) for row in seed_rows]
        baseline_pnl = baseline.get(split)
        out[split] = {
            "seeds": len(seed_rows),
            "seed_ids": [int(row["seed"]) for row in seed_rows],
            "median_total_pnl": _median(seed_rows, "total_pnl"),
            "median_profit_factor": _median(seed_rows, "profit_factor"),
            "median_trades": _median(seed_rows, "trades"),
            "median_stress_0_10_total_pnl": _median(seed_rows, "stress_0_10_total_pnl"),
            "median_stress_0_25_total_pnl": _median(seed_rows, "stress_0_25_total_pnl"),
            "positive_seed_fraction": float(np.mean([pnl > 0.0 for pnl in pnl_values])) if pnl_values else 0.0,
            "frozen_protocol101_total_pnl": baseline_pnl,
            "median_delta_vs_frozen_protocol101": (
                _median(seed_rows, "total_pnl") - baseline_pnl if baseline_pnl is not None else None
            ),
            "beats_frozen_protocol101": (
                bool(_median(seed_rows, "total_pnl") > baseline_pnl) if baseline_pnl is not None else None
            ),
            "seed_rows": seed_rows,
        }
    return out


def _serial_invariants(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "overlap_violations": 0,
            "unaffordable_violations": 0,
            "non_positive_premium_rows": 0,
            "nan_time_rows": 0,
        }

    overlap_violations = 0
    for _, group in trades.groupby(["fold", "seed", "reported_split", "session"], sort=False):
        ordered = group.sort_values("decision_time")
        previous_exit = None
        for row in ordered.itertuples(index=False):
            decision_time = getattr(row, "decision_time")
            exit_time = getattr(row, "candidate_exit_time", pd.NaT)
            if previous_exit is not None and pd.notna(decision_time) and decision_time < previous_exit:
                overlap_violations += 1
            if pd.notna(exit_time):
                previous_exit = exit_time

    premium = pd.to_numeric(trades.get("entry_premium"), errors="coerce")
    equity_before = pd.to_numeric(trades.get("account_equity_before"), errors="coerce")
    return {
        "overlap_violations": int(overlap_violations),
        "unaffordable_violations": int(((premium > equity_before) | premium.isna() | equity_before.isna()).sum()),
        "non_positive_premium_rows": int((premium <= 0.0).sum()),
        "nan_time_rows": int(trades["decision_time"].isna().sum() + trades["candidate_exit_time"].isna().sum()),
    }


def _concentration(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {}
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    total = float(pnl.sum())
    by_trade = pnl.sort_values(ascending=False).to_numpy()
    by_day = trades.assign(pnl=pnl).groupby("session")["pnl"].sum().sort_values(ascending=False).to_numpy()
    return {
        "total_pnl_all_seeds": total,
        "top_5_trades_pnl": float(by_trade[:5].sum()) if len(by_trade) else 0.0,
        "top_10_trades_pnl": float(by_trade[:10].sum()) if len(by_trade) else 0.0,
        "top_5_days_pnl": float(by_day[:5].sum()) if len(by_day) else 0.0,
        "top_5_trades_fraction_of_total": _safe_ratio(float(by_trade[:5].sum()), total),
        "top_10_trades_fraction_of_total": _safe_ratio(float(by_trade[:10].sum()), total),
        "top_5_days_fraction_of_total": _safe_ratio(float(by_day[:5].sum()), total),
    }


def _decision(aggregate: dict[str, Any], invariants: dict[str, Any]) -> str:
    required = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    checks = []
    for split in required:
        row = aggregate.get(split, {})
        checks.append(int(row.get("seeds", 0)) >= 5)
        checks.append(_finite_float(row.get("median_total_pnl"), 0.0) > 0.0)
        checks.append(_finite_float(row.get("median_profit_factor"), 0.0) >= 1.15)
        checks.append(_finite_float(row.get("median_stress_0_10_total_pnl"), 0.0) > 0.0)
        checks.append(_finite_float(row.get("positive_seed_fraction"), 0.0) >= 0.80)
        if row.get("beats_frozen_protocol101") is not None:
            checks.append(bool(row.get("beats_frozen_protocol101")))
    checks.append(int(invariants.get("overlap_violations", 1)) == 0)
    checks.append(int(invariants.get("unaffordable_violations", 1)) == 0)
    checks.append(int(invariants.get("non_positive_premium_rows", 1)) == 0)
    checks.append(int(invariants.get("nan_time_rows", 1)) == 0)
    if all(checks):
        return "keep_research_candidate: 5-seed full-action policy survives Protocol081 serial replay"
    return "do_not_promote: 5-seed confirmation failed one or more serial research gates"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol194 Full-Action Surface-Edge 5-Seed Confirmation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Trade rows: `{payload['trade_rows']}`",
        f"- Paid data downloaded: `{payload['paid_data_downloaded_by_runner']}`",
        f"- Live orders: `{payload['live_orders']}`",
        f"- Caveat: {payload['caveat']}",
        "",
        "## Aggregate",
        "",
        "| split | seeds | median PnL | frozen Protocol101 | delta | PF | stress 0.10 | stress 0.25 | trades | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in payload["aggregate"].items():
        lines.append(
            "| {split} | {seeds} | {pnl:.0f} | {baseline} | {delta} | {pf:.2f} | {stress10:.0f} | {stress25:.0f} | {trades:.0f} | {positive:.2f} |".format(
                split=split,
                seeds=int(row["seeds"]),
                pnl=_finite_float(row["median_total_pnl"], 0.0),
                baseline=_format_optional(row.get("frozen_protocol101_total_pnl")),
                delta=_format_optional(row.get("median_delta_vs_frozen_protocol101")),
                pf=_finite_float(row["median_profit_factor"], 0.0),
                stress10=_finite_float(row["median_stress_0_10_total_pnl"], 0.0),
                stress25=_finite_float(row["median_stress_0_25_total_pnl"], 0.0),
                trades=_finite_float(row["median_trades"], 0.0),
                positive=_finite_float(row["positive_seed_fraction"], 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Serial Invariants",
            "",
            f"- Overlap violations: `{payload['invariants']['overlap_violations']}`",
            f"- Unaffordable violations: `{payload['invariants']['unaffordable_violations']}`",
            f"- Non-positive premium rows: `{payload['invariants']['non_positive_premium_rows']}`",
            f"- NaN time rows: `{payload['invariants']['nan_time_rows']}`",
            "",
            "## Concentration",
            "",
        ]
    )
    for key, value in payload["concentration"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Serial trades: `{path.parent / 'protocol194_protocol081_5seed_serial_trades.csv'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = [_finite_float(row.get(key), np.nan) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(np.median(values)) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return None
    return float(numerator / denominator)


def _format_optional(value: Any) -> str:
    if value is None:
        return ""
    return f"{_finite_float(value, 0.0):.0f}"


if __name__ == "__main__":
    raise SystemExit(main())
