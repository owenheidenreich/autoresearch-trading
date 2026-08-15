"""Build environment diagnostics and validation-selected rule probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v4.model.action_pilot import load_action_decisions
from v4.model.environment_diagnostics import (
    EnvironmentBinner,
    environment_table,
    rule_search,
    summarize_group,
    write_json,
)
from v4.model.supervised_pilot import session_from_path, split_name
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--policy-index", type=int, default=2, choices=sorted(POLICY_META))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/environment_diagnostics_policy2"))
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _records(frame, max_rows=20):
    return json.loads(frame.head(max_rows).to_json(orient="records"))


def main() -> int:
    args = parse_args()
    policy_name, cooldown = POLICY_META[args.policy_index]
    paths = _paths_by_split(args.data_dir)
    decisions = {
        split: load_action_decisions(files, policy_index=args.policy_index)
        for split, files in paths.items()
    }
    binner = EnvironmentBinner.fit(decisions["train"])
    table = environment_table(decisions, binner=binner)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / "environment_table.parquet"
    table.to_parquet(table_path, index=False)

    summaries = {
        "side_time_split": _records(
            summarize_group(table, ["split", "side", "time_bucket"]), max_rows=200
        ),
        "side_time_vwap_split": _records(
            summarize_group(table, ["split", "side", "time_bucket", "above_vwap"]),
            max_rows=200,
        ),
        "side_time_momentum_split": _records(
            summarize_group(table, ["split", "side", "time_bucket", "momentum_15m_sign"]),
            max_rows=200,
        ),
        "side_time_vol_split": _records(
            summarize_group(table, ["split", "side", "time_bucket", "vix_bucket"]),
            max_rows=200,
        ),
    }
    rules = rule_search(table, cooldown_minutes=cooldown, min_validation_trades=20)
    report = {
        "policy": policy_name,
        "cooldown_minutes": cooldown,
        "environment_framing": (
            "Jan-Mar 2026 is real market truth, but only one environment slice. "
            "The goal is to identify causal environment descriptors the model can learn from, "
            "then require validation-selected patterns to survive holdout scoring."
        ),
        "binner": binner.to_dict(),
        "rows": int(len(table)),
        "environment_table": str(table_path),
        "summaries": summaries,
        "validation_selected_rules": rules[:50],
        "best_test_rules": sorted(
            rules,
            key=lambda r: (r["test"]["total_pnl"], r["test"]["profit_factor"], r["test"]["trades"]),
            reverse=True,
        )[:20],
    }
    report_path = args.out_dir / "report.json"
    write_json(report_path, report)

    md_path = args.out_dir / "report.md"
    lines = [
        "# Environment Diagnostics",
        "",
        f"Policy: `{policy_name}`",
        "",
        report["environment_framing"],
        "",
        "## Validation-Selected Rules",
        "",
        "| Rule | Val Trades | Val PnL | Val PF | Test Trades | Test PnL | Test PF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rules[:20]:
        lines.append(
            f"| `{item['rule']}` | {item['validation']['trades']} | "
            f"{item['validation']['total_pnl']:.0f} | {item['validation']['profit_factor']:.3f} | "
            f"{item['test']['trades']} | {item['test']['total_pnl']:.0f} | {item['test']['profit_factor']:.3f} |"
        )
    lines += [
        "",
        "## Test-Ranked Survivors",
        "",
        "| Rule | Val PnL | Test Trades | Test PnL | Test PF |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["best_test_rules"][:20]:
        lines.append(
            f"| `{item['rule']}` | {item['validation']['total_pnl']:.0f} | "
            f"{item['test']['trades']} | {item['test']['total_pnl']:.0f} | "
            f"{item['test']['profit_factor']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(report_path)
    print(md_path)
    print(json.dumps({"rules": len(rules), "top_rule": rules[0] if rules else None}, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
