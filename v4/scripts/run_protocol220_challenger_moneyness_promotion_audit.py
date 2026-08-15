"""AUDIT_CHALLENGER_MONEYNESS_AND_PROMOTION_BLOCKERS_V1.

Historically Protocol220. This diagnostic quantifies how often
CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1 trades ITM/ATM/OTM contracts and
summarizes why it remains a research challenger rather than the paper default.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from v4.scripts.export_protocol101_trade_charts import attach_spx_prices, load_spx_bars


ROLE_LABEL = "AUDIT_CHALLENGER_MONEYNESS_AND_PROMOTION_BLOCKERS_V1"
HISTORICAL_ID = "Protocol220"
DEFAULT_PAPER_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_218_challenger_full_action_history_trade_charts/trades.csv"
)
DEFAULT_ALL_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_215_full_action_surface_edge_history_5seed_confirmation/model_trades_5seed.csv"
)
DEFAULT_SPX_DIR = Path("data/vendor/thetadata/index/spx_1m")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_220_challenger_moneyness_promotion_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-trades", type=Path, default=DEFAULT_PAPER_TRADES)
    parser.add_argument("--all-trades", type=Path, default=DEFAULT_ALL_TRADES)
    parser.add_argument("--spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paper = enrich_trades(pd.read_csv(args.paper_trades), spx=None)
    all5 = pd.read_csv(args.all_trades)
    all5 = all5[~all5["reported_split"].astype(str).eq("march_2026")].copy()
    all5 = enrich_trades(all5, spx=load_spx_bars(args.spx_dir))

    write_dataset_outputs(args.out_dir, "paper_seed1", paper)
    write_dataset_outputs(args.out_dir, "all5_no_march_duplicate", all5)

    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / audit",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": {
            "paper_seed1": str(args.paper_trades),
            "all5": str(args.all_trades),
            "spx_dir": str(args.spx_dir),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "paper_seed1": summarize_payload(paper),
        "all5_no_march_duplicate": summarize_payload(all5),
        "promotion_status": {
            "current_status": "confirmed_research_challenger_not_paper_default",
            "why_not_promoted": [
                "No explicit freeze/promotion decision has changed PAPER_DEFAULT_PROTOCOL101.",
                "Runtime parity has only been replay/no-order so far; it has not yet run as a live no-order challenger beside the paper bot.",
                "The challenger uses a broader full-action candidate surface and causal history features, so the live path must prove it can build the same surface and features from streaming market data.",
                "The premium profile is materially different from Protocol101: many trades are ITM with $2k-$4k premium, so paper replacement needs explicit exposure monitoring and visual inspection.",
            ],
        },
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "paper_seed1_trades": str(args.out_dir / "paper_seed1_moneyness_trades.csv"),
            "all5_trades": str(args.out_dir / "all5_no_march_duplicate_moneyness_trades.csv"),
        },
        "decision": "moneyness_audit_complete_paper_default_unchanged",
        "next_experiment": (
            "Run a no-order live-shadow challenger using the stream-safe full-action/history feature path, "
            "then compare its live candidate surface and premium selection against this audit."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload, paper=paper, all5=all5)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def enrich_trades(frame: pd.DataFrame, *, spx: list[dict[str, Any]] | None) -> pd.DataFrame:
    out = frame.copy()
    out["strike"] = out["contract_id"].map(parse_strike)
    out["offset"] = pd.to_numeric(out.get("offset"), errors="coerce")
    if "entry_spx" not in out.columns or pd.to_numeric(out["entry_spx"], errors="coerce").isna().all():
        if spx is None and "offset" not in out.columns:
            raise ValueError("SPX bars are required when entry_spx is missing")
        if spx is not None:
            rows = []
            for row in out.to_dict("records"):
                item = dict(row)
                item["decision_time"] = iso_utc(item["decision_time"])
                item["exit_time"] = iso_utc(item["exit_time"])
                rows.append(item)
            out = pd.DataFrame(attach_spx_prices(rows, spx))
            out["strike"] = out["contract_id"].map(parse_strike)
        else:
            out["entry_spx"] = math.nan

    out["entry_spx"] = pd.to_numeric(out["entry_spx"], errors="coerce")
    out["entry_ask"] = pd.to_numeric(out.get("entry_ask"), errors="coerce")
    if "entry_premium" in out.columns:
        out["premium"] = pd.to_numeric(out["entry_premium"], errors="coerce")
    elif "premium_paid" in out.columns:
        out["premium"] = pd.to_numeric(out["premium_paid"], errors="coerce")
    else:
        out["premium"] = out["entry_ask"] * 100.0
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    out["right"] = out["right"].astype(str).str.upper()
    out["distance_spx_minus_strike"] = out["entry_spx"] - out["strike"]
    out["signed_moneyness_points"] = out.apply(signed_moneyness_points, axis=1)
    out["moneyness_source"] = out.apply(moneyness_source, axis=1)
    out["intrinsic_points"] = out["signed_moneyness_points"].map(lambda value: max(float(value), 0.0) if math.isfinite(float(value)) else math.nan)
    out["otm_points"] = out["signed_moneyness_points"].map(lambda value: max(-float(value), 0.0) if math.isfinite(float(value)) else math.nan)
    out["strict_moneyness"] = out["signed_moneyness_points"].map(strict_moneyness)
    out["ladder_moneyness"] = out["signed_moneyness_points"].map(ladder_moneyness)
    out["premium_bucket"] = pd.cut(
        out["premium"],
        bins=[-1, 500, 1000, 2000, 3000, 4000, 100_000],
        labels=["<=500", "500-1k", "1k-2k", "2k-3k", "3k-4k", ">4k"],
    )
    out["abs_moneyness_bucket"] = pd.cut(
        out["signed_moneyness_points"].abs(),
        bins=[-1, 2.5, 10, 20, 30, 40, 1000],
        labels=["ATM<=2.5", "2.5-10", "10-20", "20-30", "30-40", ">40"],
    )
    return out


def parse_strike(contract_id: Any) -> float:
    match = re.search(r"-(\d+\.\d+)-[CP]$", str(contract_id))
    return float(match.group(1)) if match else math.nan


def intrinsic_points(row: pd.Series) -> float:
    if row["right"] == "C":
        return max(float(row["entry_spx"]) - float(row["strike"]), 0.0)
    if row["right"] == "P":
        return max(float(row["strike"]) - float(row["entry_spx"]), 0.0)
    return math.nan


def otm_points(row: pd.Series) -> float:
    if row["right"] == "C":
        return max(float(row["strike"]) - float(row["entry_spx"]), 0.0)
    if row["right"] == "P":
        return max(float(row["entry_spx"]) - float(row["strike"]), 0.0)
    return math.nan


def signed_moneyness_points(row: pd.Series) -> float:
    entry_spx = float(row["entry_spx"]) if pd.notna(row["entry_spx"]) else math.nan
    strike = float(row["strike"]) if pd.notna(row["strike"]) else math.nan
    if math.isfinite(entry_spx) and math.isfinite(strike):
        if row["right"] == "C":
            return entry_spx - strike
        if row["right"] == "P":
            return strike - entry_spx
    offset = float(row["offset"]) if pd.notna(row.get("offset")) else math.nan
    if math.isfinite(offset):
        if row["right"] == "C":
            return -offset
        if row["right"] == "P":
            return offset
    return math.nan


def moneyness_source(row: pd.Series) -> str:
    entry_spx = float(row["entry_spx"]) if pd.notna(row["entry_spx"]) else math.nan
    strike = float(row["strike"]) if pd.notna(row["strike"]) else math.nan
    if math.isfinite(entry_spx) and math.isfinite(strike):
        return "spx_entry"
    offset = float(row["offset"]) if pd.notna(row.get("offset")) else math.nan
    if math.isfinite(offset):
        return "offset_fallback"
    return "missing"


def strict_moneyness(value: float) -> str:
    if not math.isfinite(float(value)):
        return "unknown"
    if value > 0:
        return "ITM"
    if value < 0:
        return "OTM"
    return "ATM"


def ladder_moneyness(value: float) -> str:
    if not math.isfinite(float(value)):
        return "unknown"
    if value > 2.5:
        return "ITM"
    if value < -2.5:
        return "OTM"
    return "ATM_band"


def summarize(frame: pd.DataFrame, group: str | list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group, dropna=False, observed=False)
    out = grouped.agg(
        trades=("pnl", "size"),
        pnl=("pnl", "sum"),
        win_rate=("pnl", lambda values: float((values > 0).mean())),
        avg_pnl=("pnl", "mean"),
        median_pnl=("pnl", "median"),
        avg_premium=("premium", "mean"),
        median_premium=("premium", "median"),
        avg_signed_moneyness=("signed_moneyness_points", "mean"),
        median_signed_moneyness=("signed_moneyness_points", "median"),
    ).reset_index()
    return out


def summarize_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "moneyness_source_counts": frame["moneyness_source"].value_counts(dropna=False).to_dict(),
        "strict_moneyness": summarize(frame, "strict_moneyness").to_dict("records"),
        "ladder_moneyness": summarize(frame, "ladder_moneyness").to_dict("records"),
        "ladder_by_side": summarize(frame, ["ladder_moneyness", "right"]).to_dict("records"),
        "premium_bucket": summarize(frame, "premium_bucket").to_dict("records"),
    }


def write_dataset_outputs(out_dir: Path, prefix: str, frame: pd.DataFrame) -> None:
    frame.to_csv(out_dir / f"{prefix}_moneyness_trades.csv", index=False)
    summarize(frame, "strict_moneyness").to_csv(out_dir / f"{prefix}_strict_moneyness_summary.csv", index=False)
    summarize(frame, "ladder_moneyness").to_csv(out_dir / f"{prefix}_ladder_moneyness_summary.csv", index=False)
    summarize(frame, ["ladder_moneyness", "right"]).to_csv(out_dir / f"{prefix}_ladder_by_side_summary.csv", index=False)
    split_column = "segment" if "segment" in frame.columns else "reported_split"
    summarize(frame, ["ladder_moneyness", split_column]).to_csv(out_dir / f"{prefix}_ladder_by_split_summary.csv", index=False)
    summarize(frame, "premium_bucket").to_csv(out_dir / f"{prefix}_premium_bucket_summary.csv", index=False)
    summarize(frame, "abs_moneyness_bucket").to_csv(out_dir / f"{prefix}_abs_moneyness_bucket_summary.csv", index=False)


def write_report(path: Path, payload: dict[str, Any], *, paper: pd.DataFrame, all5: pd.DataFrame) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: diagnostic / audit",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        "Paid data downloaded: False",
        "Broker endpoint called: False",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{HISTORICAL_ID}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Definitions",
        "",
        "- Strict moneyness: any positive intrinsic value is `ITM`; negative intrinsic value is `OTM`.",
        "- Trader ladder moneyness: `ATM_band` means within +/- $2.50 of SPX, because SPXW strikes are $5 spaced.",
        "- Positive signed moneyness means the option is ITM; negative means OTM.",
        "- If an SPX entry bar is missing, the audit falls back to the existing strike offset for classification and marks `moneyness_source = offset_fallback`.",
        "",
    ]
    append_section(lines, "Source-of-truth seed-1 serial replay", paper)
    append_section(lines, "All five research seeds, March duplicate removed", all5)
    lines.extend(
        [
            "## Why It Was Not Promoted Yet",
            "",
            "- It has not failed the historical research gate; it actually cleared the five-seed historical challenger gate.",
            "- It was not promoted because `PAPER_DEFAULT_PROTOCOL101` can only change through an explicit replacement decision packet.",
            "- The remaining gate is live/runtime parity: the bot must produce the same full-action candidate surface, causal history features, masks, and order-intent logs from live data.",
            "- The ITM premium profile is a real operational issue to inspect, not an automatic rejection. It means the challenger often pays more premium for higher delta/intrinsic exposure instead of seeking cheaper OTM convexity.",
            "- Before replacement, we should run it no-order beside Protocol101 and confirm live quote freshness, candidate coverage, premium distribution, selected strikes, and latency.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Seed-1 moneyness trades: `{payload['outputs']['paper_seed1_trades']}`",
            f"- All-seed moneyness trades: `{payload['outputs']['all5_trades']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_section(lines: list[str], title: str, frame: pd.DataFrame) -> None:
    source_counts = frame["moneyness_source"].value_counts(dropna=False).to_dict()
    lines.extend([f"## {title}", "", f"Moneyness source counts: `{source_counts}`", "", "Strict moneyness:", ""])
    lines.extend(markdown_table(summarize(frame, "strict_moneyness")))
    lines.extend(["", "Trader ladder moneyness:", ""])
    lines.extend(markdown_table(summarize(frame, "ladder_moneyness")))
    lines.extend(["", "Ladder moneyness by side:", ""])
    lines.extend(markdown_table(summarize(frame, ["ladder_moneyness", "right"])))
    lines.extend(["", "Premium bucket:", ""])
    lines.extend(markdown_table(summarize(frame, "premium_bucket")))
    lines.append("")


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict("records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def iso_utc(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
