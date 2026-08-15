"""Diagnose why stricter A+ entry throttling did or did not work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model.environment_diagnostics import time_bucket


_NY = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_champion_export_003"))
    parser.add_argument("--strict-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_strict_entry_stress_003b"))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_entry_throttle_diagnostic"))
    return parser.parse_args()


def _load_trades(root: Path, label: str) -> pd.DataFrame:
    rows = []
    for split in ("selection", "march", "q4"):
        path = root / f"selected_trades_{split}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            rows.append({"run": label, "split": split, **row})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["decision_time_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["local_time"] = frame["decision_time_ts"].dt.tz_convert(_NY).dt.strftime("%H:%M")
    frame["time_bucket"] = frame["decision_time_ts"].map(time_bucket)
    frame = frame.sort_values(["run", "split", "seed", "session", "decision_time_ts"])
    frame["daily_ordinal"] = frame.groupby(["run", "split", "seed", "session"]).cumcount() + 1
    frame["ordinal_bucket"] = np.where(frame["daily_ordinal"] == 1, "first_trade", "later_trade")
    steps = np.where(frame["right"].astype(str) == "C", frame["offset"].astype(float) / 5.0, -frame["offset"].astype(float) / 5.0)
    frame["moneyness_steps"] = steps
    frame["moneyness_bucket"] = np.select(
        [
            steps <= -2,
            steps < 0,
            steps == 0,
            steps <= 2,
        ],
        ["itm_2p", "itm_1", "atm", "otm_1_2"],
        default="otm_3p",
    )
    if "feature_obvious_overpay_flag" in frame.columns:
        frame["aplus_agreement"] = np.where(
            (pd.to_numeric(frame.get("pattern_present", 0.0), errors="coerce").fillna(0.0) > 0.0)
            & (pd.to_numeric(frame.get("feature_worth_spread_flag", 0.0), errors="coerce").fillna(0.0) > 0.0)
            & (pd.to_numeric(frame.get("feature_obvious_overpay_flag", 1.0), errors="coerce").fillna(1.0) <= 0.0),
            "pattern_and_value_agree",
            "agreement_missing",
        )
    else:
        frame["aplus_agreement"] = "unknown"
    return frame


def _metrics(group: pd.DataFrame) -> dict:
    pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
    gross_profit = pnl.clip(lower=0).sum()
    gross_loss = -pnl.clip(upper=0).sum()
    return {
        "trades": int(len(group)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "median_pnl": float(pnl.median()) if len(pnl) else 0.0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
    }


def _group_metrics(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    if frame.empty:
        return rows
    for keys, group in frame.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({col: key for col, key in zip(columns, keys)} | _metrics(group))
    rows.sort(key=lambda row: (str(row.get("split", "")), -row["total_pnl"]))
    return rows


def _score_bucket_metrics(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (run, split), group in frame.groupby(["run", "split"], dropna=False):
        work = group.copy()
        try:
            work["score_bucket"] = pd.qcut(work["score"], q=4, labels=["score_q1_low", "score_q2", "score_q3", "score_q4_high"], duplicates="drop")
        except ValueError:
            work["score_bucket"] = "score_unknown"
        for bucket, bucket_group in work.groupby("score_bucket", dropna=False, observed=True):
            rows.append({"run": run, "split": split, "score_bucket": str(bucket)} | _metrics(bucket_group))
    return rows


def _numeric_bucket_metrics(frame: pd.DataFrame, column: str, bucket_name: str) -> list[dict]:
    rows = []
    if column not in frame.columns:
        return rows
    for (run, split), group in frame.groupby(["run", "split"], dropna=False):
        work = group.copy()
        values = pd.to_numeric(work[column], errors="coerce")
        try:
            work[bucket_name] = pd.qcut(values, q=4, labels=[f"{bucket_name}_q1_low", f"{bucket_name}_q2", f"{bucket_name}_q3", f"{bucket_name}_q4_high"], duplicates="drop")
        except ValueError:
            work[bucket_name] = f"{bucket_name}_unknown"
        for bucket, bucket_group in work.groupby(bucket_name, dropna=False, observed=True):
            rows.append({"run": run, "split": split, bucket_name: str(bucket)} | _metrics(bucket_group))
    return rows


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# A+ Entry Throttle Diagnostic",
        "",
        "Compares the Protocol 003 champion export against the fixed one-trade-per-day stress.",
        "",
        "## Run Summary",
        "",
        "| Run | Split | Trades | PnL | PF | Win Rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["run_split"]:
        lines.append(
            f"| {row['run']} | {row['split']} | {row['trades']} | {row['total_pnl']:.0f} | {row['profit_factor']:.3f} | {row['win_rate']:.2f} |"
        )
    lines += [
        "",
        "## Champion By Daily Order",
        "",
        "| Split | Order | Trades | PnL | PF | Avg PnL | Win Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["champion_ordinal"]:
        lines.append(
            f"| {row['split']} | {row['ordinal_bucket']} | {row['trades']} | {row['total_pnl']:.0f} | "
            f"{row['profit_factor']:.3f} | {row['avg_pnl']:.0f} | {row['win_rate']:.2f} |"
        )
    lines += [
        "",
        "## Champion By Score Quartile",
        "",
        "| Split | Score Bucket | Trades | PnL | PF | Avg PnL | Win Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["champion_score_buckets"]:
        lines.append(
            f"| {row['split']} | {row['score_bucket']} | {row['trades']} | {row['total_pnl']:.0f} | "
            f"{row['profit_factor']:.3f} | {row['avg_pnl']:.0f} | {row['win_rate']:.2f} |"
        )
    lines += [
        "",
        "## Champion By A+ Agreement",
        "",
        "| Split | Agreement | Trades | PnL | PF | Avg PnL | Win Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["champion_agreement"]:
        lines.append(
            f"| {row['split']} | {row['aplus_agreement']} | {row['trades']} | {row['total_pnl']:.0f} | "
            f"{row['profit_factor']:.3f} | {row['avg_pnl']:.0f} | {row['win_rate']:.2f} |"
        )
    lines += [
        "",
        "## Champion By Contract Value Score",
        "",
        "| Split | Value Bucket | Trades | PnL | PF | Avg PnL | Win Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["champion_value_score_buckets"]:
        lines.append(
            f"| {row['split']} | {row['value_score_bucket']} | {row['trades']} | {row['total_pnl']:.0f} | "
            f"{row['profit_factor']:.3f} | {row['avg_pnl']:.0f} | {row['win_rate']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    champion = _load_trades(args.champion_dir, "champion_max4")
    strict = _load_trades(args.strict_dir, "strict_max1")
    all_trades = pd.concat([champion, strict], ignore_index=True)
    run_split = _group_metrics(all_trades, ["run", "split"])
    champion_ordinal = _group_metrics(champion, ["split", "ordinal_bucket"])
    champion_score = [row for row in _score_bucket_metrics(champion) if row["run"] == "champion_max4"]
    champion_agreement = _group_metrics(champion, ["split", "aplus_agreement"])
    champion_value_score = [
        row
        for row in _numeric_bucket_metrics(champion, "feature_contract_value_score", "value_score_bucket")
        if row["run"] == "champion_max4"
    ]
    march_first = next((row for row in champion_ordinal if row["split"] == "march" and row["ordinal_bucket"] == "first_trade"), None)
    march_later = next((row for row in champion_ordinal if row["split"] == "march" and row["ordinal_bucket"] == "later_trade"), None)
    interpretation = (
        "The fixed one-trade-per-day stress is a poor throttle if first trades underperform later trades. "
        "The next throttle should not simply take the first qualifying setup; it should require stronger "
        "agreement between score, A+ contract value, and timing state before the first trade is allowed."
    )
    if march_first and march_later:
        interpretation += (
            f" In March, champion first trades produced {march_first['total_pnl']:.0f} "
            f"versus {march_later['total_pnl']:.0f} for later trades."
        )
    payload = {
        "run_split": run_split,
        "champion_ordinal": champion_ordinal,
        "champion_score_buckets": champion_score,
        "champion_agreement": champion_agreement,
        "champion_value_score_buckets": champion_value_score,
        "interpretation": interpretation,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, payload)
    print(json.dumps(payload, indent=2, allow_nan=True))
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
