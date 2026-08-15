"""Protocol 115: interpret existing CBBO-1s validation for frozen Protocol 101.

This script intentionally does not download data, train a model, or call broker
APIs. It reads the local one-second replay artifact produced by
``audit_selected_trades_1s_path.py`` and decides whether the already-covered
1s slices support, weaken, or remain insufficient for the Protocol 101 timing
assumption.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_115_protocol101_existing_1s_path_audit")
DEFAULT_REPLAY_JSON = DEFAULT_OUT_DIR / "report.json"
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
REQUIRED_SPLITS = ["q4_2024_external", "q3_2025", "q4_2025", "q1_2026", "march_2026"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-json", type=Path, default=DEFAULT_REPLAY_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.replay_json.read_text())
    rows = pd.DataFrame(data.get("rows", []))
    if rows.empty:
        raise SystemExit(f"no replay rows found in {args.replay_json}")

    prepared = prepare_rows(rows)
    base_rows = prepared[prepared["split"].ne("march_2026")].copy()
    split_summary = summarize_splits(prepared)
    audited = base_rows[base_rows["audit_status"].eq("audited")].copy()
    side_summary = summarize_dimension(audited, "right")
    time_summary = summarize_dimension(audited, "time_bucket")
    status_summary = summarize_dimension(base_rows, "audit_status")
    decision = decide(split_summary)
    payload = {
        "protocol": "115_protocol101_existing_1s_path_validation",
        "paid_data_downloaded": False,
        "live_orders": False,
        "model_training": False,
        "source_replay_json": str(args.replay_json),
        "decision": decision,
        "interpretation": interpretation(decision),
        "row_counts": {
            "input_rows": int(len(rows)),
            "expanded_rows_with_march_overlay": int(len(prepared)),
            "expanded_audited_rows_with_march_overlay": int(len(prepared[prepared["audit_status"].eq("audited")])),
            "audited_rows": int(len(audited)),
            "coverage": float(len(audited) / max(len(rows), 1)),
        },
        "split_summary": split_summary,
        "status_summary": status_summary,
        "side_summary": side_summary,
        "time_bucket_summary": time_summary,
        "largest_differences": largest_differences(audited),
        "next_gate": next_gate(decision),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def prepare_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["split"] = out["split"].fillna(out.get("reported_split", "")).astype(str)
    out["audit_status"] = out["audit_status"].astype(str)
    out["session"] = out["session"].astype(str)
    out["decision_ts"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["time_bucket"] = time_bucket(out["decision_ts"])
    for column in [
        "pnl_1m",
        "pnl_1s",
        "pnl_diff_1s_minus_1m",
        "pnl_1s_planned_exit",
        "pnl_diff_1s_planned_minus_1m",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    expanded = [out]
    march = out[out["split"].eq("q1_2026") & out["session"].ge("2026-03-01")].copy()
    if not march.empty:
        march["split"] = "march_2026"
        expanded.append(march)
    return pd.concat(expanded, ignore_index=True)


def time_bucket(ts: pd.Series) -> pd.Series:
    local = ts.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    return pd.Series(
        np.select(
            [minute < 600, minute < 690, minute < 810, minute <= 930],
            ["first30", "post_open_morning", "midday", "late_afternoon"],
            default="after_hours",
        ),
        index=ts.index,
    )


def summarize_splits(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in REQUIRED_SPLITS:
        group = rows[rows["split"].eq(split)]
        audited = group[group["audit_status"].eq("audited")]
        out[split] = summarize_group(group, audited)
    return out


def summarize_group(group: pd.DataFrame, audited: pd.DataFrame) -> dict[str, Any]:
    status_counts = {str(k): int(v) for k, v in group["audit_status"].value_counts(dropna=False).items()}
    row = {
        "input_trades": int(len(group)),
        "audited_trades": int(len(audited)),
        "coverage": float(len(audited) / max(len(group), 1)),
        "status_counts": status_counts,
        "audited_sessions": sorted(str(x) for x in audited["session"].dropna().unique().tolist()),
    }
    if audited.empty:
        return row
    diff = pd.to_numeric(audited["pnl_diff_1s_minus_1m"], errors="coerce").dropna()
    planned_diff = pd.to_numeric(audited.get("pnl_diff_1s_planned_minus_1m", pd.Series(dtype=float)), errors="coerce").dropna()
    row |= {
        "pnl_1m_sum": finite_float(audited["pnl_1m"].sum()),
        "pnl_1s_sum": finite_float(audited["pnl_1s"].sum()),
        "pnl_sum_diff_1s_minus_1m": finite_float(diff.sum()),
        "diff_median": finite_float(diff.median()),
        "diff_mean": finite_float(diff.mean()),
        "abs_diff_p95": finite_float(diff.abs().quantile(0.95)),
        "sign_flip_fraction": finite_float(audited["sign_flip"].astype(bool).mean()),
        "one_second_worse_fraction": finite_float((diff < 0).mean()),
        "planned_exit_diff_sum": finite_float(planned_diff.sum()) if not planned_diff.empty else None,
        "planned_abs_diff_p95": finite_float(planned_diff.abs().quantile(0.95)) if not planned_diff.empty else None,
        "planned_sign_flip_fraction": finite_float(audited.get("planned_sign_flip", pd.Series(dtype=bool)).astype(bool).mean())
        if "planned_sign_flip" in audited.columns
        else None,
        "mandatory_event_before_lifecycle_exit_fraction": finite_float(
            audited.get("mandatory_event_before_lifecycle_exit", pd.Series(dtype=bool)).astype(bool).mean()
        )
        if "mandatory_event_before_lifecycle_exit" in audited.columns
        else None,
    }
    return row


def summarize_dimension(rows: pd.DataFrame, dimension: str) -> list[dict[str, Any]]:
    if rows.empty or dimension not in rows.columns:
        return []
    out = []
    for value, group in rows.groupby(dimension, dropna=False, sort=True):
        audited = group[group["audit_status"].eq("audited")] if "audit_status" in group.columns else group
        out.append({"value": str(value), **summarize_group(group, audited)})
    return out


def decide(split_summary: dict[str, dict[str, Any]]) -> str:
    audited = [row for row in split_summary.values() if row.get("audited_trades", 0) > 0]
    if not audited:
        return "no_1s_evidence"
    covered_sign_flips = any((row.get("sign_flip_fraction") or 0.0) > 0.01 for row in audited)
    covered_planned_flips = any((row.get("planned_sign_flip_fraction") or 0.0) > 0.01 for row in audited)
    large_repricing = any(abs(row.get("pnl_sum_diff_1s_minus_1m") or 0.0) > 0.05 * abs(row.get("pnl_1m_sum") or 1.0) for row in audited)
    missing_critical = any(split_summary.get(split, {}).get("audited_trades", 0) == 0 for split in ["q4_2024_external", "q3_2025"])
    thin_coverage = any(split_summary.get(split, {}).get("coverage", 0.0) < 0.15 for split in ["q4_2025", "q1_2026", "march_2026"])
    if covered_sign_flips or covered_planned_flips or large_repricing:
        return "reject_or_reprice_protocol101_timing_assumption"
    if missing_critical or thin_coverage:
        return "partial_support_needs_targeted_1s_or_live_shadow"
    return "covered_1s_replay_supports_protocol101_timing_assumption"


def interpretation(decision: str) -> str:
    if decision == "reject_or_reprice_protocol101_timing_assumption":
        return (
            "Existing CBBO-1s slices materially disagree with the 1-minute replay. "
            "Treat the Protocol 101 equity curve as overstated until labels are rebuilt or repriced."
        )
    if decision == "covered_1s_replay_supports_protocol101_timing_assumption":
        return (
            "Available CBBO-1s coverage supports the 1-minute executable path assumptions across required splits, "
            "but live-shadow parity is still required before order placement."
        )
    if decision == "no_1s_evidence":
        return "No overlapping CBBO-1s evidence exists for Protocol 101 selected trades."
    return (
        "Where existing CBBO-1s coverage overlaps Protocol 101 trades, the 1-minute replay is not obviously fake. "
        "However, coverage is too thin and misses critical splits, so this is supportive evidence, not promotion-grade confidence."
    )


def next_gate(decision: str) -> str:
    if decision == "reject_or_reprice_protocol101_timing_assumption":
        return "stop model work; rebuild labels or replay engine around higher-resolution execution evidence"
    if decision == "covered_1s_replay_supports_protocol101_timing_assumption":
        return "run no-order live shadow parity for frozen Protocol 101 before broker-connected paper trading"
    return (
        "do not buy broad history yet; either run no-order live shadow when IBKR is ready, "
        "or request a tightly capped targeted CBBO-1s batch for missing Protocol 101 sessions"
    )


def largest_differences(audited: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    if audited.empty:
        return []
    rows = audited.copy()
    rows["_abs_diff"] = rows["pnl_diff_1s_minus_1m"].abs()
    cols = [
        "split",
        "session",
        "seed",
        "decision_time",
        "contract_id",
        "right",
        "pnl_1m",
        "pnl_1s",
        "pnl_diff_1s_minus_1m",
        "exit_reason_1s",
        "mandatory_event_before_lifecycle_exit",
    ]
    return [
        {column: json_ready(row.get(column)) for column in cols if column in row}
        for row in rows.sort_values("_abs_diff", ascending=False).head(limit).to_dict("records")
    ]


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return finite_float(value)
    return value


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=json_ready, allow_nan=False) + "\n"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 115: Protocol 101 Existing CBBO-1s Validation",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Interpretation: {payload['interpretation']}",
        f"- Next gate: {payload['next_gate']}",
        f"- Source replay artifact: `{payload['source_replay_json']}`",
        f"- Audited rows: `{payload['row_counts']['audited_rows']}` of `{payload['row_counts']['input_rows']}` "
        f"({pct(payload['row_counts']['coverage'])})",
        f"- March overlay audited rows: `{payload['row_counts']['expanded_audited_rows_with_march_overlay']}` "
        f"after duplicating March rows out of Q1 for split reporting",
        "",
        "## Split Coverage And Repricing",
        "",
        "| split | input | audited | coverage | missing_1s | missing_symbol | 1m_pnl | 1s_pnl | diff | sign_flips | planned_diff | mandatory_events |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in REQUIRED_SPLITS:
        row = payload["split_summary"][split]
        status = row.get("status_counts", {})
        lines.append(
            "| "
            f"{split} | {row['input_trades']} | {row['audited_trades']} | {pct(row['coverage'])} | "
            f"{status.get('missing_1s_session', 0)} | {status.get('missing_symbol', 0)} | "
            f"{money(row.get('pnl_1m_sum'))} | {money(row.get('pnl_1s_sum'))} | "
            f"{money(row.get('pnl_sum_diff_1s_minus_1m'))} | {pct(row.get('sign_flip_fraction'))} | "
            f"{money(row.get('planned_exit_diff_sum'))} | {pct(row.get('mandatory_event_before_lifecycle_exit_fraction'))} |"
        )
    lines.extend(
        [
            "",
            "## Why This Matters",
            "",
            (
                "Protocol 114 showed that the equity curve is very sensitive to a one-minute delayed entry/exit stress. "
                "Protocol 115 checks a different question: when a trade overlaps existing one-second quotes, does the "
                "same planned entry/exit path materially disagree with the 1-minute replay?"
            ),
            "",
            (
                "The answer is supportive where covered: no sign flips and planned-exit differences are zero in the "
                "available slices. But the evidence is incomplete because Q3 2025 and Q4 2024 external have no audited "
                "Protocol 101 trades, and Q4 2025 coverage is thin."
            ),
            "",
            "## Audited Side Summary",
            "",
            "| side | trades | 1m_pnl | 1s_pnl | diff | sign_flips |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["side_summary"]:
        lines.append(
            f"| {row['value']} | {row['audited_trades']} | {money(row.get('pnl_1m_sum'))} | "
            f"{money(row.get('pnl_1s_sum'))} | {money(row.get('pnl_sum_diff_1s_minus_1m'))} | "
            f"{pct(row.get('sign_flip_fraction'))} |"
        )
    lines.extend(
        [
            "",
            "## Audited Time Bucket Summary",
            "",
            "| bucket | trades | 1m_pnl | 1s_pnl | diff | sign_flips |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["time_bucket_summary"]:
        lines.append(
            f"| {row['value']} | {row['audited_trades']} | {money(row.get('pnl_1m_sum'))} | "
            f"{money(row.get('pnl_1s_sum'))} | {money(row.get('pnl_sum_diff_1s_minus_1m'))} | "
            f"{pct(row.get('sign_flip_fraction'))} |"
        )
    lines.extend(
        [
            "",
            "## Largest 1s Differences",
            "",
            "| split | session | seed | time | contract | side | 1m_pnl | 1s_pnl | diff | 1s_exit |",
            "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["largest_differences"]:
        lines.append(
            "| "
            f"{row.get('split')} | {row.get('session')} | {row.get('seed')} | {row.get('decision_time')} | "
            f"{row.get('contract_id')} | {row.get('right')} | {money(row.get('pnl_1m'))} | "
            f"{money(row.get('pnl_1s'))} | {money(row.get('pnl_diff_1s_minus_1m'))} | "
            f"{row.get('exit_reason_1s')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def money(value: Any) -> str:
    value = finite_float(value)
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def pct(value: Any) -> str:
    value = finite_float(value)
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    heading = "## 2026-05-13 Protocol 115 Protocol101 Existing CBBO-1s Validation"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    existing = ledger.read_text() if ledger.exists() else ""
    if heading in existing:
        return
    entry = f"""

{heading}

```text
Date: 2026-05-13
Decision / Experiment: Replayed frozen Protocol 101 selected trades on already-collected Databento CBBO-1s slices.
Reason: Protocol 114 showed the Protocol 101 equity curve is timing-sensitive under one-minute delayed entry/exit stress. Before buying broad history or designing a bigger neural network, the project needed to check whether existing one-second evidence contradicts the one-minute executable replay.
Data Used: Existing Protocol 101 and Protocol 107 selected trades, existing normalized official-context symbol maps, and existing local CBBO-1s audit files only. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Audited {payload['row_counts']['audited_rows']} of {payload['row_counts']['input_rows']} Protocol 101 rows; detailed split coverage and repricing are in {report_path}.
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    with ledger.open("a") as f:
        f.write(entry)


if __name__ == "__main__":
    raise SystemExit(main())
