"""Protocol 108: attribute Protocol 107 Q4 2024 external stress.

Protocol 107 passed the median temporal stress, but seed-level margins were not
uniform. This attribution compares frozen Protocol 101 against the strict serial
baseline trade by trade, with special attention to seed 4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROTOCOL107_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_108_protocol107_q4_2024_external_attribution")
FEATURE_COLUMNS = [
    "edge",
    "entry_bid",
    "entry_ask",
    "entry_mid",
    "entry_spread",
    "entry_spread_frac",
    "entry_iv",
    "entry_abs_delta",
    "entry_gamma",
    "entry_theta",
    "entry_gamma_theta_ratio",
    "entry_theta_burden",
    "entry_spread_over_mid",
    "entry_gamma_per_premium",
    "entry_premium_over_underlying",
    "offset",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol107-dir", type=Path, default=DEFAULT_PROTOCOL107_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_trades(args.protocol107_dir / "serial_policy_trades.json")
    baseline = _load_trades(args.protocol107_dir / "strict_serial_baseline_trades.json")
    dataset = pd.read_parquet(args.protocol107_dir / "q4_2024_external_serial_opportunity_dataset.parquet")
    feature_lookup = dataset.set_index("candidate_uid")[FEATURE_COLUMNS]

    summary, baseline_only, model_only, swap_pairs = _attribute(model, baseline, feature_lookup)
    payload = {
        "protocol": "108_protocol107_q4_2024_external_attribution",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol107_dir": str(args.protocol107_dir),
        "summary": summary,
        "answer": _answer(summary),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    pd.DataFrame(baseline_only).to_json(args.out_dir / "baseline_only_trades.json", orient="records", indent=2)
    pd.DataFrame(model_only).to_json(args.out_dir / "model_only_trades.json", orient="records", indent=2)
    pd.DataFrame(swap_pairs).to_json(args.out_dir / "same_minute_swap_pairs.json", orient="records", indent=2)
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"answer": payload["answer"], "report": str(args.out_dir / "report.md")}, indent=2))
    return 0


def _load_trades(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(path.read_text()))
    frame["entry_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["exit_ts"] = pd.to_datetime(frame["exit_time"], utc=True)
    local = frame["entry_ts"].dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    frame["entry_minutes"] = minutes
    frame["time_bucket_attribution"] = np.select(
        [minutes < 600, minutes < 690, minutes < 810, minutes <= 930],
        ["first30", "post_open", "midday", "late"],
        default="after",
    )
    frame["time_key"] = list(zip(frame["seed"], frame["session"], frame["decision_time"]))
    return frame


def _attribute(
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    feature_lookup: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    common = set(model["candidate_uid"]) & set(baseline["candidate_uid"])
    model_only = model[~model["candidate_uid"].isin(common)].copy()
    baseline_only = baseline[~baseline["candidate_uid"].isin(common)].copy()
    baseline_only["miss_category"] = baseline_only.apply(lambda row: _baseline_only_category(row, model), axis=1)
    model_only["add_category"] = model_only.apply(lambda row: _model_only_category(row, baseline), axis=1)
    swap_pairs = _same_minute_pairs(model_only, baseline_only, feature_lookup)
    summary = {
        "model_total_pnl": float(model["pnl"].sum()),
        "baseline_total_pnl": float(baseline["pnl"].sum()),
        "total_delta": float(model["pnl"].sum() - baseline["pnl"].sum()),
        "model_trades": int(len(model)),
        "baseline_trades": int(len(baseline)),
        "common_trades": int(len(common)),
        "model_only_trades": int(len(model_only)),
        "baseline_only_trades": int(len(baseline_only)),
        "decomposition": _decomposition(model_only, baseline_only),
        "side_totals": _side_totals(model, baseline),
        "bucket_totals": _bucket_totals(model, baseline),
        "seed_deltas": _seed_deltas(model, baseline, model_only, baseline_only),
        "seed4_focus": _seed_focus(4, model, baseline, model_only, baseline_only, swap_pairs),
        "same_minute_swaps": _swap_summary(pd.DataFrame(swap_pairs)),
        "missed_convex_winners": _missed_convex_summary(baseline_only[baseline_only["pnl"] >= 1000.0].copy()),
        "top_missed_winners": _records(
            baseline_only[baseline_only["pnl"] > 0.0]
            .sort_values("pnl", ascending=False)
            .head(15),
            ["seed", "session", "decision_time", "right", "offset", "pnl", "miss_category", "time_bucket_attribution", "candidate_uid"],
        ),
        "top_seed4_missed_winners": _records(
            baseline_only[(baseline_only["seed"] == 4) & (baseline_only["pnl"] > 0.0)]
            .sort_values("pnl", ascending=False)
            .head(15),
            ["session", "decision_time", "right", "offset", "pnl", "miss_category", "time_bucket_attribution", "candidate_uid"],
        ),
    }
    return (
        summary,
        _records(baseline_only, list(baseline_only.columns)),
        _records(model_only, list(model_only.columns)),
        swap_pairs,
    )


def _baseline_only_category(row: pd.Series, model: pd.DataFrame) -> str:
    same = model[
        (model["seed"] == row["seed"])
        & (model["session"] == row["session"])
        & (model["entry_ts"] == row["entry_ts"])
    ]
    if not same.empty:
        return "same_minute_contract_swap"
    if _holding_at(model, int(row["seed"]), str(row["session"]), row["entry_ts"]) is not None:
        return "model_was_holding"
    return "model_flat_threshold_skip"


def _model_only_category(row: pd.Series, baseline: pd.DataFrame) -> str:
    same = baseline[
        (baseline["seed"] == row["seed"])
        & (baseline["session"] == row["session"])
        & (baseline["entry_ts"] == row["entry_ts"])
    ]
    if not same.empty:
        return "same_minute_contract_swap"
    if _holding_at(baseline, int(row["seed"]), str(row["session"]), row["entry_ts"]) is not None:
        return "baseline_was_holding"
    return "baseline_flat_model_added"


def _holding_at(frame: pd.DataFrame, seed: int, session: str, entry_ts: pd.Timestamp) -> pd.Series | None:
    sub = frame[(frame["seed"] == seed) & (frame["session"] == session)]
    hit = sub[(sub["entry_ts"] <= entry_ts) & (entry_ts < sub["exit_ts"])]
    if hit.empty:
        return None
    return hit.iloc[0]


def _same_minute_pairs(
    model_only: pd.DataFrame,
    baseline_only: pd.DataFrame,
    feature_lookup: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for time_key in sorted(set(model_only["time_key"]) & set(baseline_only["time_key"])):
        model_row = model_only[model_only["time_key"] == time_key].iloc[0]
        baseline_row = baseline_only[baseline_only["time_key"] == time_key].iloc[0]
        row = {
            "seed": int(model_row["seed"]),
            "session": str(model_row["session"]),
            "decision_time": str(model_row["decision_time"]),
            "model_candidate_uid": str(model_row["candidate_uid"]),
            "baseline_candidate_uid": str(baseline_row["candidate_uid"]),
            "model_right": str(model_row["right"]),
            "baseline_right": str(baseline_row["right"]),
            "model_offset": float(model_row["offset"]),
            "baseline_offset": float(baseline_row["offset"]),
            "model_pnl": float(model_row["pnl"]),
            "baseline_pnl": float(baseline_row["pnl"]),
            "delta": float(model_row["pnl"] - baseline_row["pnl"]),
            "time_bucket": str(model_row["time_bucket_attribution"]),
        }
        if model_row["candidate_uid"] in feature_lookup.index and baseline_row["candidate_uid"] in feature_lookup.index:
            for column in FEATURE_COLUMNS:
                model_value = float(feature_lookup.loc[model_row["candidate_uid"], column])
                baseline_value = float(feature_lookup.loc[baseline_row["candidate_uid"], column])
                row[f"model_{column}"] = model_value
                row[f"baseline_{column}"] = baseline_value
                row[f"delta_{column}"] = model_value - baseline_value
        rows.append(row)
    return rows


def _decomposition(model_only: pd.DataFrame, baseline_only: pd.DataFrame) -> dict[str, Any]:
    model_by_category = model_only.groupby("add_category")["pnl"].sum().to_dict() if not model_only.empty else {}
    baseline_by_category = baseline_only.groupby("miss_category")["pnl"].sum().to_dict() if not baseline_only.empty else {}
    return {
        "same_minute_contract_swap": float(model_by_category.get("same_minute_contract_swap", 0.0) - baseline_by_category.get("same_minute_contract_swap", 0.0)),
        "slot_occupancy": float(model_by_category.get("baseline_was_holding", 0.0) - baseline_by_category.get("model_was_holding", 0.0)),
        "flat_threshold_or_no_entry": float(model_by_category.get("baseline_flat_model_added", 0.0) - baseline_by_category.get("model_flat_threshold_skip", 0.0)),
        "model_only_by_category": {str(k): float(v) for k, v in model_by_category.items()},
        "baseline_only_by_category": {str(k): float(v) for k, v in baseline_by_category.items()},
    }


def _side_totals(model: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    rows = {}
    for side in sorted(set(model["right"]) | set(baseline["right"])):
        model_side = model[model["right"] == side]
        baseline_side = baseline[baseline["right"] == side]
        rows[str(side)] = {
            "model_pnl": float(model_side["pnl"].sum()),
            "baseline_pnl": float(baseline_side["pnl"].sum()),
            "delta": float(model_side["pnl"].sum() - baseline_side["pnl"].sum()),
            "model_trades": int(len(model_side)),
            "baseline_trades": int(len(baseline_side)),
        }
    return rows


def _bucket_totals(model: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    rows = {}
    for bucket in ["first30", "post_open", "midday", "late", "after"]:
        model_bucket = model[model["time_bucket_attribution"] == bucket]
        baseline_bucket = baseline[baseline["time_bucket_attribution"] == bucket]
        if model_bucket.empty and baseline_bucket.empty:
            continue
        rows[bucket] = {
            "model_pnl": float(model_bucket["pnl"].sum()),
            "baseline_pnl": float(baseline_bucket["pnl"].sum()),
            "delta": float(model_bucket["pnl"].sum() - baseline_bucket["pnl"].sum()),
            "model_trades": int(len(model_bucket)),
            "baseline_trades": int(len(baseline_bucket)),
        }
    return rows


def _seed_deltas(model: pd.DataFrame, baseline: pd.DataFrame, model_only: pd.DataFrame, baseline_only: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for seed in sorted(set(model["seed"]) | set(baseline["seed"])):
        model_seed = model[model["seed"] == seed]
        baseline_seed = baseline[baseline["seed"] == seed]
        model_only_seed = model_only[model_only["seed"] == seed]
        baseline_only_seed = baseline_only[baseline_only["seed"] == seed]
        rows.append(
            {
                "seed": int(seed),
                "model_pnl": float(model_seed["pnl"].sum()),
                "baseline_pnl": float(baseline_seed["pnl"].sum()),
                "delta": float(model_seed["pnl"].sum() - baseline_seed["pnl"].sum()),
                "model_trades": int(len(model_seed)),
                "baseline_trades": int(len(baseline_seed)),
                "model_only_pnl": float(model_only_seed["pnl"].sum()),
                "baseline_only_pnl": float(baseline_only_seed["pnl"].sum()),
                "baseline_only_positive_pnl": float(baseline_only_seed[baseline_only_seed["pnl"] > 0.0]["pnl"].sum()),
                "model_only_positive_pnl": float(model_only_seed[model_only_seed["pnl"] > 0.0]["pnl"].sum()),
            }
        )
    return rows


def _seed_focus(
    seed: int,
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    model_only: pd.DataFrame,
    baseline_only: pd.DataFrame,
    swap_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    model_seed = model[model["seed"] == seed]
    baseline_seed = baseline[baseline["seed"] == seed]
    model_only_seed = model_only[model_only["seed"] == seed]
    baseline_only_seed = baseline_only[baseline_only["seed"] == seed]
    swaps = pd.DataFrame([row for row in swap_pairs if int(row["seed"]) == seed])
    return {
        "model_pnl": float(model_seed["pnl"].sum()),
        "baseline_pnl": float(baseline_seed["pnl"].sum()),
        "delta": float(model_seed["pnl"].sum() - baseline_seed["pnl"].sum()),
        "model_trades": int(len(model_seed)),
        "baseline_trades": int(len(baseline_seed)),
        "decomposition": _decomposition(model_only_seed, baseline_only_seed),
        "side_totals": _side_totals(model_seed, baseline_seed),
        "bucket_totals": _bucket_totals(model_seed, baseline_seed),
        "same_minute_swaps": _swap_summary(swaps),
        "baseline_only_positive_pnl": float(baseline_only_seed[baseline_only_seed["pnl"] > 0.0]["pnl"].sum()),
        "baseline_only_positive_count": int((baseline_only_seed["pnl"] > 0.0).sum()),
    }


def _swap_summary(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {"pairs": 0}
    negative = pairs[pairs["delta"] < 0.0]
    positive = pairs[pairs["delta"] > 0.0]
    big_negative = pairs[pairs["delta"] <= -500.0]
    big_positive = pairs[pairs["delta"] >= 500.0]
    return {
        "pairs": int(len(pairs)),
        "net_delta": float(pairs["delta"].sum()),
        "negative_pairs": int(len(negative)),
        "negative_delta": float(negative["delta"].sum()),
        "positive_pairs": int(len(positive)),
        "positive_delta": float(positive["delta"].sum()),
        "big_negative_pairs": int(len(big_negative)),
        "big_negative_delta": float(big_negative["delta"].sum()),
        "big_positive_pairs": int(len(big_positive)),
        "big_positive_delta": float(big_positive["delta"].sum()),
        "side_transitions_negative": _count((negative["model_right"] + "->" + negative["baseline_right"]).tolist()),
        "side_transitions_positive": _count((positive["model_right"] + "->" + positive["baseline_right"]).tolist()),
        "negative_feature_delta_means": _feature_delta_means(negative),
        "positive_feature_delta_means": _feature_delta_means(positive),
    }


def _feature_delta_means(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    columns = [f"delta_{column}" for column in FEATURE_COLUMNS if f"delta_{column}" in frame.columns]
    return {column: float(frame[column].mean()) for column in columns}


def _missed_convex_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"count": 0, "pnl": 0.0, "by_category_side_bucket": []}
    grouped = frame.groupby(["miss_category", "right", "time_bucket_attribution"])["pnl"].agg(["size", "sum"])
    return {
        "count": int(len(frame)),
        "pnl": float(frame["pnl"].sum()),
        "by_category_side_bucket": [
            {
                "miss_category": str(index[0]),
                "right": str(index[1]),
                "bucket": str(index[2]),
                "count": int(row["size"]),
                "pnl": float(row["sum"]),
            }
            for index, row in grouped.iterrows()
        ],
    }


def _answer(summary: dict[str, Any]) -> dict[str, str]:
    seed_deltas = sorted(summary["seed_deltas"], key=lambda row: row["delta"])
    worst = seed_deltas[0]
    decomp = summary["decomposition"]
    seed4 = summary["seed4_focus"]
    seed4_decomp = seed4["decomposition"]
    return {
        "bottom_line": (
            "Protocol 101 passed Q4 2024 external stress on median PnL, but the edge is not seed-uniform. "
            f"The worst seed was {worst['seed']} with delta {worst['delta']:.0f} versus strict serial."
        ),
        "primary_overall_driver": _driver_sentence(decomp),
        "seed4_failure": _driver_sentence(seed4_decomp),
        "side_exposure": _side_sentence(summary["side_totals"], seed4["side_totals"]),
        "time_bucket": _bucket_sentence(summary["bucket_totals"], seed4["bucket_totals"]),
        "convex_winners": (
            f"Baseline-only winners >= 1000 totaled {summary['missed_convex_winners']['pnl']:.0f} "
            f"across {summary['missed_convex_winners']['count']} trades, so missed convex winners remain a real fragility."
        ),
        "next_hypothesis": (
            "Do not add another entry knob yet. The next paid-data gate should test whether this seed-level fragility "
            "persists in Q3 2024; if it does, the next architecture should use richer event-state memory or seed ensembling, "
            "not a hand-coded side/time filter."
        ),
    }


def _driver_sentence(decomp: dict[str, Any]) -> str:
    pieces = {
        "same-minute contract swaps": float(decomp["same_minute_contract_swap"]),
        "slot occupancy": float(decomp["slot_occupancy"]),
        "flat threshold/no-entry": float(decomp["flat_threshold_or_no_entry"]),
    }
    worst = min(pieces.items(), key=lambda item: item[1])
    best = max(pieces.items(), key=lambda item: item[1])
    return f"Largest drag was {worst[0]} ({worst[1]:.0f}); largest offset was {best[0]} ({best[1]:.0f})."


def _side_sentence(overall: dict[str, Any], seed4: dict[str, Any]) -> str:
    return (
        "Overall side deltas were "
        + ", ".join(f"{side}: {row['delta']:.0f}" for side, row in overall.items())
        + ". Seed 4 side deltas were "
        + ", ".join(f"{side}: {row['delta']:.0f}" for side, row in seed4.items())
        + "."
    )


def _bucket_sentence(overall: dict[str, Any], seed4: dict[str, Any]) -> str:
    return (
        "Overall time-bucket deltas were "
        + ", ".join(f"{bucket}: {row['delta']:.0f}" for bucket, row in overall.items())
        + ". Seed 4 time-bucket deltas were "
        + ", ".join(f"{bucket}: {row['delta']:.0f}" for bucket, row in seed4.items())
        + "."
    )


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    safe_columns = [column for column in columns if column in frame.columns]
    out = []
    for row in frame[safe_columns].to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                clean[key] = value.isoformat()
            elif isinstance(value, float) and not np.isfinite(value):
                clean[key] = None
            else:
                clean[key] = value
        out.append(clean)
    return out


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Protocol 108: Protocol 107 Q4 2024 External Attribution",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        "## Answer",
        "",
    ]
    for key, value in payload["answer"].items():
        lines.append(f"- `{key}`: {value}")
    lines += [
        "",
        "## Overall",
        "",
        f"- Model PnL / baseline PnL / delta: `{summary['model_total_pnl']:.0f}` / `{summary['baseline_total_pnl']:.0f}` / `{summary['total_delta']:.0f}`",
        f"- Common trades: `{summary['common_trades']}`; model-only: `{summary['model_only_trades']}`; baseline-only: `{summary['baseline_only_trades']}`",
        "",
        "Decomposition:",
        "",
        _table(
            [
                {"source": "same_minute_contract_swap", "delta": summary["decomposition"]["same_minute_contract_swap"]},
                {"source": "slot_occupancy", "delta": summary["decomposition"]["slot_occupancy"]},
                {"source": "flat_threshold_or_no_entry", "delta": summary["decomposition"]["flat_threshold_or_no_entry"]},
            ],
            ["source", "delta"],
        ),
        "",
        "Seed deltas:",
        "",
        _table(summary["seed_deltas"], ["seed", "model_pnl", "baseline_pnl", "delta", "model_trades", "baseline_trades", "baseline_only_positive_pnl"]),
        "",
        "Side totals:",
        "",
        _table([{"side": side, **values} for side, values in summary["side_totals"].items()], ["side", "model_pnl", "baseline_pnl", "delta", "model_trades", "baseline_trades"]),
        "",
        "Time bucket totals:",
        "",
        _table([{"bucket": bucket, **values} for bucket, values in summary["bucket_totals"].items()], ["bucket", "model_pnl", "baseline_pnl", "delta", "model_trades", "baseline_trades"]),
        "",
        "## Seed 4 Focus",
        "",
        f"- Seed 4 model / baseline / delta: `{summary['seed4_focus']['model_pnl']:.0f}` / `{summary['seed4_focus']['baseline_pnl']:.0f}` / `{summary['seed4_focus']['delta']:.0f}`",
        "",
        "Seed 4 decomposition:",
        "",
        _table(
            [
                {"source": "same_minute_contract_swap", "delta": summary["seed4_focus"]["decomposition"]["same_minute_contract_swap"]},
                {"source": "slot_occupancy", "delta": summary["seed4_focus"]["decomposition"]["slot_occupancy"]},
                {"source": "flat_threshold_or_no_entry", "delta": summary["seed4_focus"]["decomposition"]["flat_threshold_or_no_entry"]},
            ],
            ["source", "delta"],
        ),
        "",
        "Seed 4 missed winners:",
        "",
        _table(summary["top_seed4_missed_winners"], ["session", "decision_time", "right", "offset", "pnl", "miss_category", "time_bucket_attribution"]),
        "",
        "## Same-Minute Swaps",
        "",
        "```json",
        json.dumps(summary["same_minute_swaps"], indent=2, sort_keys=True),
        "```",
        "",
        "## Missed Convex Winners",
        "",
        "```json",
        json.dumps(summary["missed_convex_winners"], indent=2, sort_keys=True),
        "```",
        "",
        "Top missed winners:",
        "",
        _table(summary["top_missed_winners"], ["seed", "session", "decision_time", "right", "offset", "pnl", "miss_category", "time_bucket_attribution"]),
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}" if abs(value) < 10 else f"{value:.0f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
