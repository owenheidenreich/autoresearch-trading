"""Summarize stress/random controls for Protocol 003 A+ neural reports."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_neural_protocol_003/report.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_neural_protocol_003"),
    )
    return parser.parse_args()


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _matching_rows(payload: dict, champion: dict) -> list[dict]:
    return [
        row
        for row in payload["rows"]
        if row["variant"]["name"] == champion["variant"]
        and int(row["policy_index"]) == int(champion["policy_index"])
        and row["trial"]["name"] == champion["trial"]
    ]


def _split_controls(rows: list[dict], split: str) -> dict:
    pnl = [float(row["metrics_by_split"][split]["total_pnl"]) for row in rows]
    pf = [float(row["metrics_by_split"][split]["profit_factor"]) for row in rows]
    trades = [float(row["metrics_by_split"][split]["trades"]) for row in rows]
    out = {
        "pnl_by_seed": pnl,
        "pnl_median": _median(pnl),
        "profit_factor_by_seed": pf,
        "profit_factor_median": _median(pf),
        "trades_by_seed": trades,
        "trades_median": _median(trades),
    }
    random_rows = [
        float(row["random_baseline_by_split"][split]["total_pnl_median"])
        for row in rows
        if split in row.get("random_baseline_by_split", {})
    ]
    if random_rows:
        out["random_pnl_median_by_seed"] = random_rows
        out["random_pnl_median"] = _median(random_rows)
        out["edge_vs_random_median"] = out["pnl_median"] - out["random_pnl_median"]
    return out


def _stress_controls(rows: list[dict], split: str) -> dict:
    out = {}
    for cost in ("25", "50", "100"):
        pnl = [float(row["slippage_stress_by_split"][split][cost]["total_pnl"]) for row in rows]
        pf = [float(row["slippage_stress_by_split"][split][cost]["profit_factor"]) for row in rows]
        out[cost] = {
            "pnl_by_seed": pnl,
            "pnl_median": _median(pnl),
            "profit_factor_by_seed": pf,
            "profit_factor_median": _median(pf),
            "survives": bool(_median(pnl) > 0.0 and _median(pf) >= 1.05),
        }
    return out


def _write_markdown(path: Path, summary: dict) -> None:
    champion = summary["champion"]
    lines = [
        "# Protocol 003 Controls",
        "",
        f"Champion: `{champion['variant']} / policy{champion['policy_index']} / {champion['trial']}`",
        "",
        "## Matched Random",
        "",
        "| Split | Neural Median PnL | Random Median PnL | Edge vs Random | Median PF | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("selection", "march", "q4"):
        row = summary["split_controls"][split]
        lines.append(
            f"| {split} | {row['pnl_median']:.0f} | {row.get('random_pnl_median', 0.0):.0f} | "
            f"{row.get('edge_vs_random_median', 0.0):.0f} | {row['profit_factor_median']:.3f} | "
            f"{row['trades_median']:.0f} |"
        )
    lines += [
        "",
        "## Extra Cost Stress",
        "",
        "| Split | Extra Cost | Median PnL | Median PF | Survives |",
        "|---|---:|---:|---:|---|",
    ]
    for split in ("march", "q4"):
        for cost, row in summary["stress_controls"][split].items():
            lines.append(
                f"| {split} | {cost} | {row['pnl_median']:.0f} | "
                f"{row['profit_factor_median']:.3f} | {row['survives']} |"
            )
    lines += [
        "",
        "## Control Decision",
        "",
        summary["control_decision"],
        "",
        summary["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    payload = json.loads(args.report.read_text())
    champion = payload["aggregate"]["generalization_champion"]
    if champion is None:
        raise SystemExit("report has no champion")
    rows = _matching_rows(payload, champion)
    if not rows:
        raise SystemExit("could not find champion rows")
    split_controls = {split: _split_controls(rows, split) for split in ("selection", "march", "q4")}
    stress_controls = {split: _stress_controls(rows, split) for split in ("march", "q4")}
    survives_random = all(split_controls[split].get("edge_vs_random_median", 0.0) > 0.0 for split in ("selection", "march", "q4"))
    survives_25 = all(stress_controls[split]["25"]["survives"] for split in ("march", "q4"))
    survives_50 = all(stress_controls[split]["50"]["survives"] for split in ("march", "q4"))
    control_decision = (
        "Narrow promotion: beats matched random and survives +$25/trade stress; does not survive +$50/trade stress."
        if survives_random and survives_25 and not survives_50
        else "Rejected by controls."
        if not (survives_random and survives_25)
        else "Strong promotion: survives matched random plus +$50/trade stress."
    )
    summary = {
        "champion": champion,
        "matched_seed_rows": len(rows),
        "split_controls": split_controls,
        "stress_controls": stress_controls,
        "survives_matched_random": survives_random,
        "survives_25_cost_stress": survives_25,
        "survives_50_cost_stress": survives_50,
        "control_decision": control_decision,
        "interpretation": (
            "Protocol 003 is evidence that A+ pattern/value features improve the neural policy, "
            "but it is still a research lead. The next gate should be stricter entry throttling, "
            "1s path validation around selected trades, and cached feature reproducibility before "
            "any live-risk discussion."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "controls.json"
    md_path = args.out_dir / "controls.md"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, summary)
    print(json_path)
    print(md_path)
    print(json.dumps({"control_decision": control_decision, "survives_50": survives_50}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
