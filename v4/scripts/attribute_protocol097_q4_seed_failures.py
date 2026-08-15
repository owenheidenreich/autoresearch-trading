"""Protocol 100: Q4 seed-level attribution for Protocol 097."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROTOCOL092_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
DEFAULT_PROTOCOL097_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_097_sequential_event_policy")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_100_protocol097_q4_seed_attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--protocol097-dir", type=Path, default=DEFAULT_PROTOCOL097_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol097 = _load_trades(args.protocol097_dir / "serial_policy_trades.json")
    protocol092 = _load_trades(args.protocol092_dir / "serial_policy_trades.json")
    baseline = _load_trades(args.protocol092_dir / "strict_serial_baseline_trades.json")
    protocol097 = protocol097[protocol097["reported_split"] == "q4_2025"].copy()
    protocol092 = protocol092[protocol092["reported_split"] == "q4_2025"].copy()
    baseline = baseline[baseline["reported_split"] == "q4_2025"].copy()

    seed_rows = []
    p97_only_rows = []
    baseline_only_rows = []
    p92_only_rows = []
    for seed in sorted(protocol097["seed"].unique()):
        p97_seed = protocol097[protocol097["seed"] == seed].copy()
        p92_seed = protocol092[protocol092["seed"] == seed].copy()
        base_seed = baseline[baseline["seed"] == seed].copy()
        seed_rows.append(_seed_summary(seed, p97_seed, p92_seed, base_seed))

        p97_vs_base_common = set(p97_seed["candidate_uid"]) & set(base_seed["candidate_uid"])
        p97_only = p97_seed[~p97_seed["candidate_uid"].isin(p97_vs_base_common)].copy()
        baseline_only = base_seed[~base_seed["candidate_uid"].isin(p97_vs_base_common)].copy()
        baseline_only["miss_category"] = baseline_only.apply(lambda row: _baseline_only_category(row, p97_seed), axis=1)
        p97_only["add_category"] = p97_only.apply(lambda row: _model_only_category(row, base_seed), axis=1)
        p97_only_rows.extend(_records(p97_only))
        baseline_only_rows.extend(_records(baseline_only))

        p97_vs_p92_common = set(p97_seed["candidate_uid"]) & set(p92_seed["candidate_uid"])
        p92_only = p92_seed[~p92_seed["candidate_uid"].isin(p97_vs_p92_common)].copy()
        p92_only_rows.extend(_records(p92_only))

    seed3_baseline_misses = pd.DataFrame(baseline_only_rows)
    seed3_p92_misses = pd.DataFrame(p92_only_rows)
    payload = {
        "protocol": "100_protocol097_q4_seed_attribution",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol097_dir": str(args.protocol097_dir),
        "source_protocol092_dir": str(args.protocol092_dir),
        "seed_summaries": seed_rows,
        "answer": _answer(seed_rows, seed3_baseline_misses, seed3_p92_misses),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    pd.DataFrame(p97_only_rows).to_json(args.out_dir / "protocol097_only_q4_trades.json", orient="records", indent=2)
    pd.DataFrame(baseline_only_rows).to_json(args.out_dir / "baseline_only_q4_trades.json", orient="records", indent=2)
    pd.DataFrame(p92_only_rows).to_json(args.out_dir / "protocol092_only_q4_trades.json", orient="records", indent=2)
    _write_report(args.out_dir / "report.md", payload, seed3_baseline_misses, seed3_p92_misses)
    print(json.dumps({"answer": payload["answer"], "report": str(args.out_dir / "report.md")}, indent=2))
    return 0


def _load_trades(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(path.read_text()))
    frame["entry_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["exit_ts"] = pd.to_datetime(frame["exit_time"], utc=True)
    local = frame["entry_ts"].dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    frame["bucket"] = np.select(
        [minutes < 600, minutes < 690, minutes < 810, minutes <= 930],
        ["first30", "post_open", "midday", "late"],
        default="after",
    )
    return frame


def _seed_summary(seed: int, p97: pd.DataFrame, p92: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    common_97_base = set(p97["candidate_uid"]) & set(baseline["candidate_uid"])
    p97_only = p97[~p97["candidate_uid"].isin(common_97_base)]
    baseline_only = baseline[~baseline["candidate_uid"].isin(common_97_base)]
    common_97_92 = set(p97["candidate_uid"]) & set(p92["candidate_uid"])
    p92_only = p92[~p92["candidate_uid"].isin(common_97_92)]
    return {
        "seed": int(seed),
        "protocol097_pnl": float(p97["pnl"].sum()),
        "protocol092_pnl": float(p92["pnl"].sum()),
        "strict_baseline_pnl": float(baseline["pnl"].sum()),
        "delta_097_vs_baseline": float(p97["pnl"].sum() - baseline["pnl"].sum()),
        "delta_097_vs_092": float(p97["pnl"].sum() - p92["pnl"].sum()),
        "protocol097_trades": int(len(p97)),
        "protocol092_trades": int(len(p92)),
        "strict_baseline_trades": int(len(baseline)),
        "common_097_baseline": int(len(common_97_base)),
        "protocol097_only_pnl": float(p97_only["pnl"].sum()),
        "baseline_only_pnl": float(baseline_only["pnl"].sum()),
        "protocol097_only_trades": int(len(p97_only)),
        "baseline_only_trades": int(len(baseline_only)),
        "protocol092_only_pnl_vs_097": float(p92_only["pnl"].sum()),
        "side_delta_vs_baseline": _group_delta(p97, baseline, "right"),
        "bucket_delta_vs_baseline": _group_delta(p97, baseline, "bucket"),
        "top_baseline_only_winners": _records(baseline_only[baseline_only["pnl"] > 0.0].sort_values("pnl", ascending=False).head(8)),
        "top_protocol092_only_winners": _records(p92_only[p92_only["pnl"] > 0.0].sort_values("pnl", ascending=False).head(8)),
    }


def _baseline_only_category(row: pd.Series, model: pd.DataFrame) -> str:
    same = model[(model["entry_ts"] == row["entry_ts"]) & (model["session"] == row["session"])]
    if not same.empty:
        return "same_minute_contract_swap"
    holding = model[(model["session"] == row["session"]) & (model["entry_ts"] <= row["entry_ts"]) & (row["entry_ts"] < model["exit_ts"])]
    if not holding.empty:
        return "model_was_holding"
    return "model_flat_threshold_skip"


def _model_only_category(row: pd.Series, baseline: pd.DataFrame) -> str:
    same = baseline[(baseline["entry_ts"] == row["entry_ts"]) & (baseline["session"] == row["session"])]
    if not same.empty:
        return "same_minute_contract_swap"
    holding = baseline[(baseline["session"] == row["session"]) & (baseline["entry_ts"] <= row["entry_ts"]) & (row["entry_ts"] < baseline["exit_ts"])]
    if not holding.empty:
        return "baseline_was_holding"
    return "baseline_flat_model_added"


def _group_delta(model: pd.DataFrame, baseline: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in sorted(set(model[column]) | set(baseline[column])):
        left = model[model[column] == key]
        right = baseline[baseline[column] == key]
        out[str(key)] = {
            "protocol097_pnl": float(left["pnl"].sum()),
            "baseline_pnl": float(right["pnl"].sum()),
            "delta": float(left["pnl"].sum() - right["pnl"].sum()),
            "protocol097_trades": int(len(left)),
            "baseline_trades": int(len(right)),
        }
    return out


def _answer(seed_rows: list[dict[str, Any]], baseline_only: pd.DataFrame, p92_only: pd.DataFrame) -> dict[str, Any]:
    worst = min(seed_rows, key=lambda row: row["delta_097_vs_baseline"])
    seed3_base = baseline_only[baseline_only["seed"] == 3] if not baseline_only.empty else pd.DataFrame()
    seed3_p92 = p92_only[p92_only["seed"] == 3] if not p92_only.empty else pd.DataFrame()
    return {
        "primary_failure": "Protocol 097's Q4 median gap is seed-concentrated, not broad. Seed 3 is the largest failure.",
        "worst_seed": int(worst["seed"]),
        "worst_seed_delta_vs_baseline": float(worst["delta_097_vs_baseline"]),
        "worst_seed_trade_count_delta": int(worst["protocol097_trades"] - worst["strict_baseline_trades"]),
        "seed3_baseline_only_positive_pnl": float(seed3_base[seed3_base.get("pnl", pd.Series(dtype=float)) > 0]["pnl"].sum()) if not seed3_base.empty else 0.0,
        "seed3_protocol092_only_positive_pnl": float(seed3_p92[seed3_p92.get("pnl", pd.Series(dtype=float)) > 0]["pnl"].sum()) if not seed3_p92.empty else 0.0,
        "next_hypothesis": "Add causal short-history/state features so the wait/take policy can distinguish clusters of post-open opportunities instead of treating each event as memoryless.",
    }


def _write_report(path: Path, payload: dict[str, Any], baseline_only: pd.DataFrame, p92_only: pd.DataFrame) -> None:
    lines = [
        "# Protocol 100: Protocol 097 Q4 Seed Attribution",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        "## Answer",
        "",
    ]
    for key, value in payload["answer"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Seed Summary", ""])
    lines.append(
        _table(
            payload["seed_summaries"],
            [
                "seed",
                "protocol097_pnl",
                "protocol092_pnl",
                "strict_baseline_pnl",
                "delta_097_vs_baseline",
                "delta_097_vs_092",
                "protocol097_trades",
                "strict_baseline_trades",
            ],
        )
    )
    lines.extend(["", "## Seed 3 Missed Winners", ""])
    if not baseline_only.empty:
        seed3 = baseline_only[(baseline_only["seed"] == 3) & (baseline_only["pnl"] > 0)].sort_values("pnl", ascending=False).head(12)
        lines.append(_table(_records(seed3), ["session", "decision_time", "right", "offset", "pnl", "miss_category", "bucket"]))
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "Q4 is not failing because the sequential architecture is globally worse. It is failing because one seed under-trades and misses a concentrated set of post-open winners. That supports testing causal short-history/state memory next."
    )
    path.write_text("\n".join(lines) + "\n")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_sanitize(row) for row in frame.to_dict(orient="records")]


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
