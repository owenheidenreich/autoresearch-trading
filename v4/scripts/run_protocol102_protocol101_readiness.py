"""Protocol 102: promotion-readiness diagnostics for Protocol 101."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.supervised_pilot import Trade, metrics_for_trades


DEFAULT_PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
DEFAULT_PROTOCOL097_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_097_sequential_event_policy")
DEFAULT_PROTOCOL092_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_102_protocol101_readiness")
FREEZE_PATH = Path("v4/promotion/PROTOCOL_101_FREEZE.json")
PACKET_PATH = Path("v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--protocol097-dir", type=Path, default=DEFAULT_PROTOCOL097_DIR)
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary101 = json.loads((args.protocol101_dir / "summary.json").read_text())
    summary097 = json.loads((args.protocol097_dir / "summary.json").read_text())
    summary092 = json.loads((args.protocol092_dir / "summary.json").read_text())
    trades101 = _load_trades(args.protocol101_dir / "serial_policy_trades.json")
    trades097 = _load_trades(args.protocol097_dir / "serial_policy_trades.json")
    trades092 = _load_trades(args.protocol092_dir / "serial_policy_trades.json")
    baseline = _load_trades(args.protocol092_dir / "strict_serial_baseline_trades.json")

    diagnostics = {
        "split_gate": _split_gate(summary101),
        "split_vs_protocols": _split_comparison(summary101, summary097, summary092),
        "seed_fragility": _seed_fragility(summary101),
        "month_diagnostics": _month_diagnostics(trades101, baseline),
        "side_time_diagnostics": _side_time_diagnostics(trades101, baseline),
        "concentration": _concentration(trades101),
        "paper_trading_blockers": _paper_trading_blockers(),
    }
    freeze = _freeze_manifest(args.protocol101_dir, args.out_dir)
    payload = {
        "protocol": "102_protocol101_readiness",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol101_dir": str(args.protocol101_dir),
        "diagnostics": diagnostics,
        "freeze_manifest": str(FREEZE_PATH),
        "promotion_packet": str(PACKET_PATH),
        "decision": _decision(diagnostics),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    _write_report(args.out_dir / "report.md", payload, freeze)
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(_json_dumps(freeze))
    PACKET_PATH.write_text(_promotion_packet(payload, freeze))
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md"), "freeze": str(FREEZE_PATH)}, indent=2))
    return 0


def _load_trades(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(path.read_text()))
    frame["entry_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["local_ts"] = frame["entry_ts"].dt.tz_convert("America/New_York")
    frame["month"] = frame["session"].str.slice(0, 7)
    minutes = frame["local_ts"].dt.hour * 60 + frame["local_ts"].dt.minute
    frame["bucket"] = np.select(
        [minutes < 600, minutes < 690, minutes < 810, minutes <= 930],
        ["first30", "post_open", "midday", "late"],
        default="after",
    )
    return frame


def _split_gate(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        item = summary["aggregate_gate"][split]
        out[split] = {
            "median_total_pnl": item["median_total_pnl"],
            "median_profit_factor": item["median_profit_factor"],
            "median_trades": item["median_trades"],
            "positive_seed_fraction": item["positive_seed_fraction"],
            "stress_0_10": item["median_stress_0_10_total_pnl"],
            "stress_0_25": item["median_stress_0_25_total_pnl"],
            "strict_serial_baseline": item["strict_serial_baseline_median_total_pnl"],
            "beats_strict_serial_baseline": item["beats_strict_serial_baseline"],
            "margin_vs_baseline": item["median_total_pnl"] - item["strict_serial_baseline_median_total_pnl"],
        }
    return out


def _split_comparison(summary101: dict[str, Any], summary097: dict[str, Any], summary092: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        current = summary101["aggregate_gate"][split]
        p97 = summary097["aggregate_gate"][split]
        p92 = summary092["aggregate_gate"][split]
        out[split] = {
            "delta_vs_protocol097": current["median_total_pnl"] - p97["median_total_pnl"],
            "delta_vs_protocol092": current["median_total_pnl"] - p92["median_total_pnl"],
            "pf_delta_vs_protocol097": current["median_profit_factor"] - p97["median_profit_factor"],
            "pf_delta_vs_protocol092": current["median_profit_factor"] - p92["median_profit_factor"],
        }
    return out


def _seed_fragility(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        rows = summary["aggregate_gate"][split]["seed_rows"]
        margins = [row["model_total_pnl"] - row["strict_serial_baseline_total_pnl"] for row in rows]
        out[split] = {
            "min_seed_margin_vs_baseline": float(min(margins)),
            "max_seed_margin_vs_baseline": float(max(margins)),
            "positive_seed_margin_fraction": float(np.mean(np.asarray(margins) > 0.0)),
            "negative_seed_margins": [
                {
                    "seed": row["seed"],
                    "model_total_pnl": row["model_total_pnl"],
                    "strict_serial_baseline_total_pnl": row["strict_serial_baseline_total_pnl"],
                    "margin": row["model_total_pnl"] - row["strict_serial_baseline_total_pnl"],
                }
                for row in rows
                if row["model_total_pnl"] <= row["strict_serial_baseline_total_pnl"]
            ],
        }
    return out


def _month_diagnostics(trades: pd.DataFrame, baseline: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    source = trades[trades["reported_split"].isin(["q3_2025", "q4_2025", "q1_2026"])].copy()
    base_source = baseline[baseline["reported_split"].isin(["q3_2025", "q4_2025", "q1_2026"])].copy()
    for month in sorted(source["month"].unique()):
        left = source[source["month"] == month]
        right = base_source[base_source["month"] == month]
        metrics = _metrics(left)
        base_metrics = _metrics(right)
        rows.append(
            {
                "month": month,
                "pnl": metrics["total_pnl"],
                "profit_factor": metrics["profit_factor"],
                "trades": metrics["trades"],
                "baseline_pnl": base_metrics["total_pnl"],
                "delta_vs_baseline": metrics["total_pnl"] - base_metrics["total_pnl"],
                "positive_day_fraction": metrics["positive_day_fraction"],
                "top_day_profit_share": _top_day_profit_share(left),
            }
        )
    return rows


def _side_time_diagnostics(trades: pd.DataFrame, baseline: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    source = trades[trades["reported_split"].isin(["q3_2025", "q4_2025", "q1_2026"])].copy()
    base_source = baseline[baseline["reported_split"].isin(["q3_2025", "q4_2025", "q1_2026"])].copy()
    for side in ["C", "P"]:
        for bucket in ["first30", "post_open", "midday", "late"]:
            left = source[(source["right"] == side) & (source["bucket"] == bucket)]
            right = base_source[(base_source["right"] == side) & (base_source["bucket"] == bucket)]
            if left.empty and right.empty:
                continue
            rows.append(
                {
                    "side": side,
                    "bucket": bucket,
                    "pnl": float(left["pnl"].sum()),
                    "trades": int(len(left)),
                    "baseline_pnl": float(right["pnl"].sum()),
                    "baseline_trades": int(len(right)),
                    "delta_vs_baseline": float(left["pnl"].sum() - right["pnl"].sum()),
                }
            )
    return rows


def _concentration(trades: pd.DataFrame) -> dict[str, Any]:
    source = trades[trades["reported_split"].isin(["q3_2025", "q4_2025", "q1_2026"])].copy()
    by_day = source.groupby("session")["pnl"].sum().sort_values(ascending=False)
    positive = by_day[by_day > 0]
    total_positive = float(positive.sum())
    return {
        "top_day_profit_share": float(positive.iloc[0] / total_positive) if total_positive > 0 and len(positive) else 1.0,
        "top_5_day_profit_share": float(positive.head(5).sum() / total_positive) if total_positive > 0 and len(positive) else 1.0,
        "worst_days": [{"session": str(k), "pnl": float(v)} for k, v in by_day.sort_values().head(10).items()],
        "best_days": [{"session": str(k), "pnl": float(v)} for k, v in by_day.head(10).items()],
        "negative_days": int((by_day < 0).sum()),
        "positive_days": int((by_day > 0).sum()),
    }


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    trades = [
        Trade(
            session=str(row.session),
            decision_time=str(row.decision_time),
            pnl=float(row.pnl),
            score=float(row.score) if "score" in frame.columns and pd.notna(row.score) else None,
            right=str(row.right),
            offset=float(row.offset),
            strategy="protocol101_diag",
        )
        for row in frame.itertuples()
    ]
    return metrics_for_trades(trades)


def _top_day_profit_share(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 1.0
    by_day = frame.groupby("session")["pnl"].sum()
    positive = by_day[by_day > 0]
    total = float(positive.sum())
    return float(positive.max() / total) if total > 0 and len(positive) else 1.0


def _paper_trading_blockers() -> list[str]:
    return [
        "Protocol 101 has not been validated on broader locked historical periods beyond the current Q3/Q4/Q1/March gate.",
        "No 1s/tick path audit has been run for Protocol 101 selected trades.",
        "No Protocol 101 no-order live shadow capture exists.",
        "IBKR live market-data entitlement/cash-settlement blockers remain unresolved from prior live attempts.",
        "Protocol 101 Q4 margin over strict baseline is thin, so broader data is required before paper/live confidence.",
    ]


def _decision(diagnostics: dict[str, Any]) -> str:
    split_gate = diagnostics["split_gate"]
    all_splits = all(item["beats_strict_serial_baseline"] for item in split_gate.values())
    stress_ok = all(item["stress_0_25"] > 0.0 for item in split_gate.values())
    q4_margin = split_gate["q4_2025"]["margin_vs_baseline"]
    if all_splits and stress_ok and q4_margin > 0:
        return "research_promotion_candidate_needs_broader_validation"
    return "research_continue_only_not_ready_for_broader_validation"


def _freeze_manifest(protocol101_dir: Path, out_dir: Path) -> dict[str, Any]:
    paths = []
    for root in [protocol101_dir, out_dir]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                paths.append(path)
    script_paths = [
        Path("v4/scripts/run_protocol101_event_history_policy.py"),
        Path("v4/scripts/run_protocol097_sequential_event_policy.py"),
        Path("v4/model/serial_opportunity.py"),
        Path("v4/tests/test_serial_opportunity.py"),
    ]
    paths.extend(path for path in script_paths if path.exists())
    files = []
    for path in sorted(set(paths)):
        files.append({"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size})
    return {
        "protocol": "101_event_history_policy",
        "status": "research_promotion_candidate_not_paper_approved",
        "created_by": "Protocol 102 readiness diagnostics",
        "files": files,
        "file_count": len(files),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_report(path: Path, payload: dict[str, Any], freeze: dict[str, Any]) -> None:
    diagnostics = payload["diagnostics"]
    lines = [
        "# Protocol 102: Protocol 101 Readiness Diagnostics",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Freeze manifest: `{payload['freeze_manifest']}`",
        f"- Promotion packet: `{payload['promotion_packet']}`",
        f"- Frozen files: `{freeze['file_count']}`",
        "",
        "## Split Gate",
        "",
        _table(
            [{"split": split, **values} for split, values in diagnostics["split_gate"].items()],
            ["split", "median_total_pnl", "median_profit_factor", "median_trades", "strict_serial_baseline", "margin_vs_baseline", "stress_0_25", "beats_strict_serial_baseline"],
        ),
        "",
        "## Seed Fragility",
        "",
        "```json",
        json.dumps(diagnostics["seed_fragility"], indent=2, sort_keys=True),
        "```",
        "",
        "## Month Diagnostics",
        "",
        _table(diagnostics["month_diagnostics"], ["month", "pnl", "profit_factor", "trades", "baseline_pnl", "delta_vs_baseline", "positive_day_fraction", "top_day_profit_share"]),
        "",
        "## Concentration",
        "",
        "```json",
        json.dumps(diagnostics["concentration"], indent=2, sort_keys=True),
        "```",
        "",
        "## Paper Trading Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in diagnostics["paper_trading_blockers"])
    path.write_text("\n".join(lines) + "\n")


def _promotion_packet(payload: dict[str, Any], freeze: dict[str, Any]) -> str:
    diag = payload["diagnostics"]
    lines = [
        "# Protocol 101 Promotion-Readiness Packet",
        "",
        "## Candidate",
        "",
        "- Name: Protocol 101 sequential event policy with causal short-history features",
        "- Status: research promotion candidate, not paper/live approved",
        f"- Freeze manifest: `{payload['freeze_manifest']}`",
        f"- Protocol 102 report: `{DEFAULT_OUT_DIR / 'report.md'}`",
        "- Incremental paid data cost: `$0`",
        "",
        "## Evidence",
        "",
    ]
    for split, values in diag["split_gate"].items():
        lines.append(
            f"- {split}: median PnL `{values['median_total_pnl']:.0f}`, PF `{values['median_profit_factor']:.3f}`, "
            f"baseline margin `{values['margin_vs_baseline']:.0f}`, +0.25 stress `{values['stress_0_25']:.0f}`."
        )
    lines.extend(
        [
            "",
            "## Blockers Before Paper Trading",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in diag["paper_trading_blockers"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Protocol 101 is ready for broader locked historical validation. It is not ready for broker-connected paper trading.",
        ]
    )
    return "\n".join(lines) + "\n"


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
