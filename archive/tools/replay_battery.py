#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _date_range(start: dt.date, end: dt.date, include_weekends: bool = False) -> list[str]:
    if end < start:
        raise ValueError(f"end date {end.isoformat()} is before start date {start.isoformat()}")
    out: list[str] = []
    cur = start
    while cur <= end:
        if include_weekends or cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    return out


def _parse_dates(
    *,
    dates_csv: str | None,
    dates_file: str | None,
    start_date: str | None,
    end_date: str | None,
    include_weekends: bool,
) -> list[str]:
    out: list[str] = []
    if dates_csv:
        for d in dates_csv.split(","):
            ds = d.strip()
            if ds:
                out.append(ds)
    if dates_file:
        for raw in Path(dates_file).read_text().splitlines():
            ds = raw.strip()
            if ds and not ds.startswith("#"):
                out.append(ds)
    if start_date and end_date:
        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date)
        out.extend(_date_range(start, end, include_weekends=include_weekends))
    # Deduplicate while preserving order.
    dedup: list[str] = []
    seen: set[str] = set()
    for ds in out:
        if ds in seen:
            continue
        dt.date.fromisoformat(ds)  # validate format
        seen.add(ds)
        dedup.append(ds)
    if not dedup:
        raise ValueError("No replay dates provided. Use --dates, --dates-file, or --start-date/--end-date.")
    return dedup


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_battery(
    *,
    dates: list[str],
    output_root: Path,
    replay_script: Path,
    python_bin: str,
    model: str | None,
    train_py: str | None,
    no_download: bool,
    ib_port: int,
    warmup_days: int,
    min_trade_prob: float,
    risk_mode: str,
    enforce_max_hold: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for replay_date in dates:
        out_csv = output_root / f"replay-{replay_date}.csv"
        cmd = [
            python_bin,
            str(replay_script),
            "--date",
            replay_date,
            "--output",
            str(out_csv),
            "--ib-port",
            str(ib_port),
            "--warmup-days",
            str(warmup_days),
            "--min-trade-prob",
            str(min_trade_prob),
            "--risk-mode",
            risk_mode,
        ]
        if model:
            cmd.extend(["--model", model])
        if train_py:
            cmd.extend(["--train-py", train_py])
        if no_download:
            cmd.append("--no-download")
        if enforce_max_hold:
            cmd.append("--enforce-max-hold")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        (logs_dir / f"replay-{replay_date}.stdout.log").write_text(proc.stdout or "")
        (logs_dir / f"replay-{replay_date}.stderr.log").write_text(proc.stderr or "")

        qa_path = output_root / f"replay-{replay_date}_qa.json"
        day_path = output_root / f"replay-{replay_date}_ledger_days.csv"
        qa = _read_json(qa_path)

        num_trades = None
        total_pnl_pct = None
        qa_critical_count = None
        qa_warning_count = None
        qa_passed = None
        if qa is not None:
            qa_passed = bool(qa.get("passed", False))
            qa_critical_count = int(qa.get("critical_count", 0) or 0)
            qa_warning_count = int(qa.get("warning_count", 0) or 0)

        if day_path.exists():
            try:
                with day_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    row = next(reader, None)
                    if row:
                        num_trades = int(float(row.get("num_trades", 0) or 0))
                        total_pnl_pct = float(row.get("total_pnl_pct", 0.0) or 0.0)
            except Exception:
                pass

        status = "pass"
        reason = ""
        if proc.returncode != 0:
            status = "fail"
            reason = "replay_process_failed"
        elif qa is None:
            status = "fail"
            reason = "missing_qa_artifact"
        elif not bool(qa.get("passed", False)):
            status = "fail"
            reason = "qa_failed"

        runs.append(
            {
                "date": replay_date,
                "status": status,
                "reason": reason,
                "returncode": int(proc.returncode),
                "qa_passed": qa_passed,
                "qa_critical_count": qa_critical_count,
                "qa_warning_count": qa_warning_count,
                "num_trades": num_trades,
                "total_pnl_pct": total_pnl_pct,
                "output_csv": str(out_csv),
                "qa_json": str(qa_path),
                "day_ledger_csv": str(day_path),
                "stdout_log": str(logs_dir / f"replay-{replay_date}.stdout.log"),
                "stderr_log": str(logs_dir / f"replay-{replay_date}.stderr.log"),
            }
        )

    passed = [r for r in runs if r["status"] == "pass"]
    failed = [r for r in runs if r["status"] != "pass"]
    release_gate_passed = len(failed) == 0
    pnl_values = [float(r["total_pnl_pct"]) for r in runs if r.get("total_pnl_pct") is not None]
    aggregate_stats = {
        "total_num_trades": int(sum(int(r["num_trades"] or 0) for r in runs if r.get("num_trades") is not None)),
        "total_pnl_pct": float(sum(pnl_values)) if pnl_values else 0.0,
        "avg_pnl_pct": (float(sum(pnl_values)) / len(pnl_values)) if pnl_values else None,
        "total_qa_critical_count": int(
            sum(int(r["qa_critical_count"] or 0) for r in runs if r.get("qa_critical_count") is not None)
        ),
        "total_qa_warning_count": int(
            sum(int(r["qa_warning_count"] or 0) for r in runs if r.get("qa_warning_count") is not None)
        ),
    }
    summary = {
        "schema_version": "replay_battery_v1",
        "generated_at": dt.datetime.utcnow().isoformat(),
        "replay_script": str(replay_script),
        "output_root": str(output_root),
        "dates": dates,
        "total_dates": len(dates),
        "passed_dates": len(passed),
        "failed_dates": len(failed),
        "release_gate_passed": release_gate_passed,
        "aggregate_stats": aggregate_stats,
        "config": {
            "model": model,
            "train_py": train_py,
            "no_download": no_download,
            "ib_port": ib_port,
            "warmup_days": warmup_days,
            "min_trade_prob": min_trade_prob,
            "risk_mode": risk_mode,
            "enforce_max_hold": enforce_max_hold,
        },
        "runs": runs,
    }

    results_csv = output_root / "battery_results.csv"
    with results_csv.open("w", newline="") as f:
        if runs:
            fieldnames = list(runs[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(runs)
        else:
            f.write("date,status,reason\n")

    summary_json = output_root / "battery_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    summary_md = output_root / "battery_summary.md"
    lines = [
        "# Replay Battery Summary",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- total_dates: `{summary['total_dates']}`",
        f"- passed_dates: `{summary['passed_dates']}`",
        f"- failed_dates: `{summary['failed_dates']}`",
        f"- release_gate_passed: `{summary['release_gate_passed']}`",
        f"- total_num_trades: `{summary['aggregate_stats']['total_num_trades']}`",
        f"- total_pnl_pct: `{summary['aggregate_stats']['total_pnl_pct']}`",
        f"- avg_pnl_pct: `{summary['aggregate_stats']['avg_pnl_pct']}`",
        f"- total_qa_critical_count: `{summary['aggregate_stats']['total_qa_critical_count']}`",
        f"- total_qa_warning_count: `{summary['aggregate_stats']['total_qa_warning_count']}`",
        "",
        "## Runs",
        "",
        "| date | status | reason | qa_passed | qa_critical | qa_warning | num_trades | total_pnl_pct |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(
            f"| {row['date']} | {row['status']} | {row['reason'] or '-'} | "
            f"{row['qa_passed']} | {row['qa_critical_count']} | {row['qa_warning_count']} | "
            f"{row['num_trades']} | {row['total_pnl_pct']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-day replay battery with canonical QA gate.")
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated replay dates (YYYY-MM-DD,YYYY-MM-DD)")
    parser.add_argument("--dates-file", type=str, default=None, help="File containing replay dates (one per line)")
    parser.add_argument("--start-date", type=str, default=None, help="Range start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Range end date (YYYY-MM-DD)")
    parser.add_argument("--include-weekends", action="store_true", help="Include weekend dates when using date range")
    parser.add_argument("--output-root", type=str, default="results/analysis/replay-battery")
    parser.add_argument("--replay-script", type=str, default=None, help="Path to training/replay.py override")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--train-py", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--ib-port", type=int, default=4002)
    parser.add_argument("--warmup-days", type=int, default=5)
    parser.add_argument("--min-trade-prob", type=float, default=0.55)
    parser.add_argument("--risk-mode", type=str, default="live_like", choices=["live_like", "training"])
    parser.add_argument("--enforce-max-hold", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    replay_script = Path(args.replay_script).resolve() if args.replay_script else (repo_root / "training" / "replay.py")
    output_root = Path(args.output_root).expanduser().resolve()
    try:
        dates = _parse_dates(
            dates_csv=args.dates,
            dates_file=args.dates_file,
            start_date=args.start_date,
            end_date=args.end_date,
            include_weekends=bool(args.include_weekends),
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    summary = run_battery(
        dates=dates,
        output_root=output_root,
        replay_script=replay_script,
        python_bin=args.python_bin,
        model=args.model,
        train_py=args.train_py,
        no_download=bool(args.no_download),
        ib_port=int(args.ib_port),
        warmup_days=int(args.warmup_days),
        min_trade_prob=float(args.min_trade_prob),
        risk_mode=str(args.risk_mode),
        enforce_max_hold=bool(args.enforce_max_hold),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary.get("release_gate_passed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
