"""Summarize fixed time-filter robustness from the broad purchase run."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from v4.scripts.evaluate_broad_data_purchase_signal import PurchaseGate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=Path("v4/audit/broad_data_purchase_signal/report.json"))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/broad_data_purchase_signal"))
    return p.parse_args()


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median([float(row[key]) for row in rows]))


def _passes_gate(summary: dict, gate: PurchaseGate) -> bool:
    return bool(
        summary["test_profit_factor_median"] >= gate.min_test_profit_factor_median
        and summary["test_pnl_median"] >= gate.min_test_pnl_median
        and summary["positive_seed_fraction"] >= gate.min_positive_seed_fraction
        and summary["positive_day_fraction_median"] >= gate.min_positive_day_fraction_median
        and summary["test_max_drawdown_median"] >= gate.max_drawdown_floor_median
        and summary["test_trades_median"] >= gate.min_test_trades_median
        and summary["top_day_profit_share_median"] <= gate.max_top_day_profit_share_median
    )


def summarize(report: dict, gate: PurchaseGate) -> list[dict]:
    out = []
    policies = sorted({row["policy_index"] for row in report["runs"]})
    for policy_index in policies:
        runs = [row for row in report["runs"] if row["policy_index"] == policy_index]
        filters = sorted(runs[0]["filter_metrics"]["test"])
        for filter_name in filters:
            metrics = [row["filter_metrics"]["test"][filter_name] for row in runs]
            summary = {
                "policy_index": policy_index,
                "policy_name": runs[0]["policy_name"],
                "filter": filter_name,
                "runs": len(runs),
                "test_trades_median": _median(metrics, "trades"),
                "test_pnl_median": _median(metrics, "total_pnl"),
                "test_profit_factor_median": _median(metrics, "profit_factor"),
                "test_max_drawdown_median": _median(metrics, "max_drawdown"),
                "positive_seed_fraction": sum(1 for row in metrics if row["total_pnl"] > 0) / len(metrics),
                "positive_day_fraction_median": _median(metrics, "positive_day_fraction"),
                "top_day_profit_share_median": _median(metrics, "top_day_profit_share"),
            }
            summary["passes_broad_purchase_gate"] = _passes_gate(summary, gate)
            out.append(summary)
    return out


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text())
    gate = PurchaseGate()
    rows = summarize(report, gate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "fixed_filter_summary.json"
    json_path.write_text(json.dumps({"gate": gate.__dict__, "rows": rows}, indent=2) + "\n")

    lines = [
        "# Fixed Time-Filter Robustness",
        "",
        (
            "This summary reuses the broad purchase run and scores each fixed time window "
            "across seeds on March holdout. No thresholds or filters are reselected here."
        ),
        "",
        "| Policy | Filter | Runs | Trades | Median PnL | Median PF | Median DD | Positive Seeds | Positive Days | Top-Day Share | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy_name']} | {row['filter']} | {row['runs']} | "
            f"{row['test_trades_median']:.0f} | {row['test_pnl_median']:.0f} | "
            f"{row['test_profit_factor_median']:.3f} | {row['test_max_drawdown_median']:.0f} | "
            f"{row['positive_seed_fraction']:.2f} | {row['positive_day_fraction_median']:.2f} | "
            f"{row['top_day_profit_share_median']:.2f} | {row['passes_broad_purchase_gate']} |"
        )
    md_path = args.out_dir / "fixed_filter_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
