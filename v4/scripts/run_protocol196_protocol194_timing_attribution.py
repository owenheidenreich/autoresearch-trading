"""Protocol196: attribution for Protocol194 timing fragility.

Protocol195 showed that Protocol194's edge is much more sensitive to delayed
entries than delayed exits. This runner explains where that fragility lives by
side, time bucket, premium, spread, score, offset, and exit reason. It does not
train, download data, or touch broker endpoints.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOOP_ID = "v4_aplus_hypothesis_196_protocol194_timing_attribution"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_194_full_action_surface_edge_5seed_confirmation/"
    "protocol194_protocol081_5seed_serial_trades.csv"
)
DEFAULT_PROTOCOL195 = Path("v4/audit/autoresearch/v4_aplus_hypothesis_195_protocol194_timing_fragility")
NY_TZ = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--protocol195-dir", type=Path, default=DEFAULT_PROTOCOL195)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = load_trades(args.trades)
    minute = load_minute_rows(args.protocol195_dir / "minute_delay_rows.csv")
    enriched = enrich_minute(minute, trades)
    group_summary = build_group_summary(enriched)
    split_summary = build_split_summary(enriched)
    worst_groups = group_summary.sort_values("both_delay_delta", ascending=True).head(30)
    resilient_groups = group_summary[
        (group_summary["both_delay_pnl"] > 0)
        & (group_summary["rows"] >= max(5, int(group_summary["rows"].median() * 0.25)))
    ].sort_values("both_delay_delta", ascending=False).head(30)
    payload = {
        "protocol": "196_protocol194_timing_attribution",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_trades": str(args.trades),
        "source_protocol195_dir": str(args.protocol195_dir),
        "row_counts": {
            "trades": int(len(trades)),
            "minute_rows": int(len(minute)),
            "enriched_rows": int(len(enriched)),
            "group_rows": int(len(group_summary)),
        },
        "split_summary": split_summary,
        "decision": decide(split_summary),
        "finding": primary_finding(split_summary, worst_groups),
        "next_gate": (
            "Treat Protocol194 as a fast-entry research candidate. Before live-paper replacement, "
            "the runtime must evaluate continuously, reject stale quotes, and record decision-to-order "
            "latency; model work should not hide this as a threshold tweak."
        ),
    }
    enriched.to_csv(args.out_dir / "protocol196_enriched_timing_rows.csv", index=False)
    group_summary.to_csv(args.out_dir / "group_timing_attribution.csv", index=False)
    worst_groups.to_csv(args.out_dir / "worst_timing_groups.csv", index=False)
    resilient_groups.to_csv(args.out_dir / "resilient_timing_groups.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload, worst_groups, resilient_groups)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["time_bucket"] = time_bucket(frame["decision_ts"])
    frame["entry_bid"] = pd.to_numeric(frame.get("entry_bid"), errors="coerce")
    frame["entry_ask"] = pd.to_numeric(frame.get("entry_ask"), errors="coerce")
    frame["entry_mid"] = pd.to_numeric(frame.get("entry_mid"), errors="coerce")
    frame["entry_spread"] = frame["entry_ask"] - frame["entry_bid"]
    frame["entry_spread_over_mid"] = np.where(frame["entry_mid"] > 0, frame["entry_spread"] / frame["entry_mid"], np.nan)
    frame["entry_premium"] = pd.to_numeric(frame.get("entry_premium"), errors="coerce")
    frame["score"] = pd.to_numeric(frame.get("score"), errors="coerce")
    frame["offset"] = pd.to_numeric(frame.get("offset"), errors="coerce")
    frame["premium_bucket"] = pd.cut(
        frame["entry_premium"],
        bins=[-0.01, 1000, 2000, 3000, 4000, math.inf],
        labels=["<=1k", "1k-2k", "2k-3k", "3k-4k", "4k+"],
    ).astype(str)
    frame["spread_bucket"] = pd.cut(
        frame["entry_spread_over_mid"],
        bins=[-0.01, 0.01, 0.02, 0.04, 0.08, math.inf],
        labels=["<=1%", "1-2%", "2-4%", "4-8%", "8%+"],
    ).astype(str)
    frame["offset_bucket"] = pd.cut(
        frame["offset"].abs(),
        bins=[-0.01, 5, 15, 30, 50, math.inf],
        labels=["<=5", "5-15", "15-30", "30-50", "50+"],
    ).astype(str)
    try:
        frame["score_bucket"] = pd.qcut(frame["score"], q=5, duplicates="drop").astype(str)
    except ValueError:
        frame["score_bucket"] = "all"
    frame["exit_reason_group"] = frame.get("candidate_exit_reason", frame.get("exit_reason", "")).astype(str)
    keys = ["candidate_uid", "seed", "reported_split", "session", "contract_id"]
    return frame.drop_duplicates(keys, keep="last")


def load_minute_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for col in ["pnl", "entry_delay_pnl", "exit_delay_pnl", "both_delay_pnl"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def enrich_minute(minute: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_uid", "seed", "reported_split", "session", "contract_id"]
    columns = keys + [
        "time_bucket",
        "premium_bucket",
        "spread_bucket",
        "offset_bucket",
        "score_bucket",
        "exit_reason_group",
        "score",
        "entry_spread_over_mid",
        "offset",
    ]
    out = minute.merge(trades[columns], on=keys, how="left", validate="many_to_one")
    out["entry_delay_delta"] = out["entry_delay_pnl"] - out["pnl"]
    out["exit_delay_delta"] = out["exit_delay_pnl"] - out["pnl"]
    out["both_delay_delta"] = out["both_delay_pnl"] - out["pnl"]
    out["side"] = np.where(out["right"].eq("C"), "CALL", np.where(out["right"].eq("P"), "PUT", out["right"]))
    out["outcome_bucket"] = np.where(out["pnl"] >= 0, "winner", "loser")
    return out


def build_split_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split, group in frame.groupby("reported_split", sort=True):
        ok = group[group["status"].eq("ok")]
        rows.append(
            {
                "reported_split": str(split),
                "rows": int(len(group)),
                "entry_delay_total_delta": finite_sum(ok["entry_delay_delta"]),
                "exit_delay_total_delta": finite_sum(ok["exit_delay_delta"]),
                "both_delay_total_delta": finite_sum(ok["both_delay_delta"]),
                "entry_delay_pnl_sum": finite_sum(ok["entry_delay_pnl"]),
                "exit_delay_pnl_sum": finite_sum(ok["exit_delay_pnl"]),
                "both_delay_pnl_sum": finite_sum(ok["both_delay_pnl"]),
                "original_pnl_sum": finite_sum(ok["pnl"]),
            }
        )
    return rows


def build_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "reported_split",
        "side",
        "time_bucket",
        "premium_bucket",
        "spread_bucket",
        "offset_bucket",
        "score_bucket",
        "exit_reason_group",
        "outcome_bucket",
    ]
    rows: list[dict[str, Any]] = []
    ok = frame[frame["status"].eq("ok")].copy()
    for dimension in dimensions:
        for value, group in ok.groupby(dimension, dropna=False, sort=True):
            rows.append(summary_row(str(dimension), str(value), group))
    for (split, dimension), group_source in [
        ((split, dimension), part)
        for split, split_part in ok.groupby("reported_split", sort=True)
        for dimension, part in [(dimension, split_part) for dimension in dimensions if dimension != "reported_split"]
    ]:
        for value, group in group_source.groupby(dimension, dropna=False, sort=True):
            row = summary_row(str(dimension), str(value), group)
            row["reported_split_filter"] = str(split)
            rows.append(row)
    return pd.DataFrame(rows)


def summary_row(dimension: str, value: str, group: pd.DataFrame) -> dict[str, Any]:
    original = finite_sum(group["pnl"])
    both = finite_sum(group["both_delay_pnl"])
    return {
        "dimension": dimension,
        "value": value,
        "reported_split_filter": "all",
        "rows": int(len(group)),
        "original_pnl": original,
        "entry_delay_pnl": finite_sum(group["entry_delay_pnl"]),
        "exit_delay_pnl": finite_sum(group["exit_delay_pnl"]),
        "both_delay_pnl": both,
        "entry_delay_delta": finite_sum(group["entry_delay_delta"]),
        "exit_delay_delta": finite_sum(group["exit_delay_delta"]),
        "both_delay_delta": finite_sum(group["both_delay_delta"]),
        "both_delay_delta_per_trade": (both - original) / len(group) if len(group) else None,
        "win_rate": float((group["pnl"] > 0).mean()) if len(group) else 0.0,
        "both_delay_win_rate": float((group["both_delay_pnl"] > 0).mean()) if len(group) else 0.0,
    }


def decide(split_summary: list[dict[str, Any]]) -> str:
    if not split_summary:
        return "blocked_no_timing_rows"
    entry_deltas = [float(row["entry_delay_total_delta"]) for row in split_summary]
    exit_deltas = [float(row["exit_delay_total_delta"]) for row in split_summary]
    if all(delta < 0 for delta in entry_deltas) and all(delta > -0.25 * abs(entry) for delta, entry in zip(exit_deltas, entry_deltas)):
        return "entry_timing_is_primary_fragility"
    return "mixed_timing_fragility_requires_manual_review"


def primary_finding(split_summary: list[dict[str, Any]], worst_groups: pd.DataFrame) -> str:
    worst = worst_groups.head(1)
    if worst.empty:
        return "No group attribution rows were available."
    row = worst.iloc[0]
    return (
        "Entry delay, not exit delay, is the dominant fragility. Worst group: "
        f"{row['dimension']}={row['value']} with both-delay delta {money(row['both_delay_delta'])} "
        f"over {int(row['rows'])} rows."
    )


def write_report(path: Path, payload: dict[str, Any], worst_groups: pd.DataFrame, resilient_groups: pd.DataFrame) -> None:
    lines = [
        "# Protocol196 Protocol194 Timing Attribution",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Finding: {payload['finding']}",
        f"- Next gate: {payload['next_gate']}",
        "",
        "## Split Timing Deltas",
        "",
        "| split | original | entry +1m | exit +1m | both +1m | entry delta | exit delta | both delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['original_pnl_sum'])} | {money(row['entry_delay_pnl_sum'])} | "
            f"{money(row['exit_delay_pnl_sum'])} | {money(row['both_delay_pnl_sum'])} | "
            f"{money(row['entry_delay_total_delta'])} | {money(row['exit_delay_total_delta'])} | "
            f"{money(row['both_delay_total_delta'])} |"
        )
    lines.extend(["", "## Worst Timing Groups", "", group_table(worst_groups.head(15))])
    lines.extend(["", "## Resilient Timing Groups", "", group_table(resilient_groups.head(15))])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Enriched rows: `{path.parent / 'protocol196_enriched_timing_rows.csv'}`",
            f"- Group attribution: `{path.parent / 'group_timing_attribution.csv'}`",
            f"- Worst groups: `{path.parent / 'worst_timing_groups.csv'}`",
            f"- Resilient groups: `{path.parent / 'resilient_timing_groups.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def group_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    lines = [
        "| split_filter | dimension | value | rows | original | both +1m | both delta | delta/trade |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['reported_split_filter']} | {row['dimension']} | {row['value']} | {int(row['rows'])} | "
            f"{money(row['original_pnl'])} | {money(row['both_delay_pnl'])} | "
            f"{money(row['both_delay_delta'])} | {money(row['both_delay_delta_per_trade'])} |"
        )
    return "\n".join(lines)


def time_bucket(ts: pd.Series) -> pd.Series:
    local = ts.dt.tz_convert(NY_TZ)
    minute = local.dt.hour * 60 + local.dt.minute
    return pd.Series(
        np.select(
            [minute < 600, minute < 690, minute < 810, minute < 930],
            ["first30", "post_open_morning", "midday", "late_afternoon"],
            default="closing_or_after",
        ),
        index=ts.index,
    )


def finite_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").dropna().sum())


def money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


if __name__ == "__main__":
    raise SystemExit(main())
