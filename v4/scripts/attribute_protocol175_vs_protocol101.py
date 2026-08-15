"""Protocol177: attribute Protocol175 against frozen Protocol101.

Protocol175 nearly closed the Q3/Q4 gap but did not beat frozen Protocol101.
This script compares both serial trade sets trade by trade so the next model
change is based on mechanism rather than another blind knob.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOOP_ID = "v4_aplus_hypothesis_177_protocol175_vs101_attribution"
DEFAULT_PROTOCOL101_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/serial_policy_trades.json"
)
DEFAULT_PROTOCOL175_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_175_protocol163_q4_2024_prehistory/protocol175_model_trades.csv"
)
DEFAULT_PROTOCOL175_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_175_protocol163_q4_2024_prehistory/protocol175_serial_one_account_dataset.parquet"
)
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
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
    "entry_gamma_per_premium",
    "entry_spread_over_mid",
    "entry_premium_over_underlying",
    "offset",
]
SPLITS = ["q3_2025", "q4_2025"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol101-trades", type=Path, default=DEFAULT_PROTOCOL101_TRADES)
    parser.add_argument("--protocol175-trades", type=Path, default=DEFAULT_PROTOCOL175_TRADES)
    parser.add_argument("--protocol175-dataset", type=Path, default=DEFAULT_PROTOCOL175_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="*", default=SPLITS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol101 = load_trades(args.protocol101_trades, protocol="protocol101")
    protocol175 = load_trades(args.protocol175_trades, protocol="protocol175")
    feature_lookup = load_feature_lookup(args.protocol175_dataset)
    split_payloads: dict[str, Any] = {}
    detail_frames: list[pd.DataFrame] = []
    for split in args.splits:
        baseline = protocol101[protocol101["reported_split"].eq(split)].copy()
        candidate = protocol175[protocol175["reported_split"].eq(split)].copy()
        summary, details = attribute_split(
            split=split,
            baseline=baseline,
            candidate=candidate,
            feature_lookup=feature_lookup,
        )
        split_payloads[split] = summary
        detail_frames.append(details)
        details.to_csv(args.out_dir / f"{split}_trade_attribution.csv", index=False)
    combined_details = pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame()
    combined_details.to_csv(args.out_dir / "trade_attribution_details.csv", index=False)
    payload = {
        "protocol": "177_protocol175_vs101_attribution",
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "baseline": "frozen_protocol101",
        "candidate": "protocol175_q4_2024_prehistory",
        "splits": split_payloads,
        "answer": answer(split_payloads),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"answer": payload["answer"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_trades(path: Path, *, protocol: str) -> pd.DataFrame:
    if path.suffix == ".json":
        frame = pd.DataFrame(json.loads(path.read_text()))
    else:
        frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["protocol"] = protocol
    frame["entry_ts"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["exit_ts"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["reported_split"] = frame.get("reported_split", frame.get("split", "")).astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype(int)
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    frame["offset"] = pd.to_numeric(frame["offset"], errors="coerce")
    frame["right"] = frame["right"].astype(str)
    local = frame["entry_ts"].dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    frame["entry_minutes"] = minutes
    frame["time_bucket"] = np.select(
        [minutes < 600, minutes < 690, minutes < 810, minutes <= 930],
        ["first30", "post_open", "midday", "late"],
        default="after",
    )
    frame["exact_key"] = list(
        zip(
            frame["seed"],
            frame["reported_split"],
            frame["session"].astype(str),
            frame["decision_time"].astype(str),
            frame["contract_id"].astype(str),
        )
    )
    frame["minute_key"] = list(
        zip(
            frame["seed"],
            frame["reported_split"],
            frame["session"].astype(str),
            frame["decision_time"].astype(str),
        )
    )
    return frame


def load_feature_lookup(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    available = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if not available or "candidate_uid" not in frame.columns:
        return pd.DataFrame()
    lookup = frame.drop_duplicates("candidate_uid", keep="last").set_index("candidate_uid")[available]
    return lookup


def attribute_split(
    *,
    split: str,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    feature_lookup: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline_by_exact = baseline.drop_duplicates("exact_key").set_index("exact_key", drop=False)
    candidate_by_exact = candidate.drop_duplicates("exact_key").set_index("exact_key", drop=False)
    exact_keys = set(baseline_by_exact.index) & set(candidate_by_exact.index)
    exact_baseline = rows_for_keys(baseline_by_exact, exact_keys)
    exact_candidate = rows_for_keys(candidate_by_exact, exact_keys)

    baseline_only_seed = baseline[~baseline["exact_key"].isin(exact_keys)].copy()
    candidate_only_seed = candidate[~candidate["exact_key"].isin(exact_keys)].copy()
    baseline_only_seed["category"] = baseline_only_seed.apply(
        lambda row: baseline_only_category(row, candidate_only_seed, candidate),
        axis=1,
    )
    candidate_only_seed["category"] = candidate_only_seed.apply(
        lambda row: candidate_only_category(row, baseline_only_seed, baseline),
        axis=1,
    )
    baseline_details = enrich_details(
        baseline_only_seed,
        source="protocol101_only",
        feature_lookup=feature_lookup,
    )
    candidate_details = enrich_details(
        candidate_only_seed,
        source="protocol175_only",
        feature_lookup=feature_lookup,
    )
    exact_details = enrich_details(
        exact_baseline.assign(category="exact_overlap"),
        source="exact_overlap_protocol101",
        feature_lookup=feature_lookup,
    )
    details = pd.concat([baseline_details, candidate_details, exact_details], ignore_index=True, sort=False)

    decomposition = {
        "exact_overlap_delta": float(exact_candidate["pnl"].sum() - exact_baseline["pnl"].sum()),
        "same_minute_swap_delta": float(
            candidate_only_seed[candidate_only_seed["category"].eq("same_minute_contract_swap")]["pnl"].sum()
            - baseline_only_seed[baseline_only_seed["category"].eq("same_minute_contract_swap")]["pnl"].sum()
        ),
        "slot_occupancy_delta": float(
            candidate_only_seed[candidate_only_seed["category"].eq("baseline_was_holding")]["pnl"].sum()
            - baseline_only_seed[baseline_only_seed["category"].eq("candidate_was_holding")]["pnl"].sum()
        ),
        "flat_decision_delta": float(
            candidate_only_seed[candidate_only_seed["category"].eq("candidate_flat_added")]["pnl"].sum()
            - baseline_only_seed[baseline_only_seed["category"].eq("candidate_flat_skipped")]["pnl"].sum()
        ),
    }
    summary = {
        "split": split,
        "protocol101": metrics(baseline),
        "protocol175": metrics(candidate),
        "pnl_delta_protocol175_minus_protocol101": float(candidate["pnl"].sum() - baseline["pnl"].sum()),
        "exact_overlap": metrics(exact_baseline),
        "protocol101_only": metrics(baseline_only_seed),
        "protocol175_only": metrics(candidate_only_seed),
        "decomposition": decomposition,
        "by_side": side_summary(baseline, candidate),
        "by_time_bucket": bucket_summary(baseline, candidate, "time_bucket"),
        "by_seed": seed_summary(baseline, candidate),
        "protocol101_only_by_category": category_summary(baseline_only_seed),
        "protocol175_only_by_category": category_summary(candidate_only_seed),
        "missed_winners": category_metrics(baseline_only_seed[baseline_only_seed["pnl"] > 0.0]),
        "avoided_losers": category_metrics(baseline_only_seed[baseline_only_seed["pnl"] < 0.0]),
        "added_winners": category_metrics(candidate_only_seed[candidate_only_seed["pnl"] > 0.0]),
        "added_losers": category_metrics(candidate_only_seed[candidate_only_seed["pnl"] < 0.0]),
        "top_missed_winners": top_records(baseline_only_seed[baseline_only_seed["pnl"] > 0.0], ascending=False),
        "top_added_losers": top_records(candidate_only_seed[candidate_only_seed["pnl"] < 0.0], ascending=True),
        "same_minute_convex_misses": top_records(
            baseline_only_seed[
                baseline_only_seed["category"].eq("same_minute_contract_swap")
                & (baseline_only_seed["pnl"] >= 1000.0)
            ],
            ascending=False,
        ),
    }
    return summary, details


def rows_for_keys(frame: pd.DataFrame, keys: set[tuple]) -> pd.DataFrame:
    if not keys:
        return frame.iloc[0:0].copy()
    return frame.loc[list(keys)].copy().reset_index(drop=True)


def baseline_only_category(row: pd.Series, candidate_only: pd.DataFrame, candidate_all: pd.DataFrame) -> str:
    same = candidate_only[candidate_only["minute_key"].map(lambda key: key == row["minute_key"])]
    if not same.empty:
        return "same_minute_contract_swap"
    if holding_at(candidate_all, int(row["seed"]), str(row["reported_split"]), str(row["session"]), row["entry_ts"]) is not None:
        return "candidate_was_holding"
    return "candidate_flat_skipped"


def candidate_only_category(row: pd.Series, baseline_only: pd.DataFrame, baseline_all: pd.DataFrame) -> str:
    same = baseline_only[baseline_only["minute_key"].map(lambda key: key == row["minute_key"])]
    if not same.empty:
        return "same_minute_contract_swap"
    if holding_at(baseline_all, int(row["seed"]), str(row["reported_split"]), str(row["session"]), row["entry_ts"]) is not None:
        return "baseline_was_holding"
    return "candidate_flat_added"


def holding_at(frame: pd.DataFrame, seed: int, split: str, session: str, entry_ts: pd.Timestamp) -> pd.Series | None:
    sub = frame[
        frame["seed"].eq(seed)
        & frame["reported_split"].eq(split)
        & frame["session"].astype(str).eq(session)
    ]
    hit = sub[(sub["entry_ts"] <= entry_ts) & (entry_ts < sub["exit_ts"])]
    if hit.empty:
        return None
    return hit.iloc[0]


def enrich_details(frame: pd.DataFrame, *, source: str, feature_lookup: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["source"] = source
    if not feature_lookup.empty:
        for column in feature_lookup.columns:
            out[column] = out["candidate_uid"].map(feature_lookup[column])
    keep = [
        "source",
        "category",
        "protocol",
        "reported_split",
        "seed",
        "session",
        "decision_time",
        "exit_time",
        "contract_id",
        "right",
        "offset",
        "pnl",
        "score",
        "threshold",
        "time_bucket",
        "exit_reason",
        "candidate_uid",
    ] + [column for column in FEATURE_COLUMNS if column in out.columns]
    return out[[column for column in keep if column in out.columns]]


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "total_pnl": 0.0, "avg_pnl": 0.0, "median_pnl": 0.0, "win_rate": 0.0}
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0.0].sum()
    losses = -pnl[pnl < 0.0].sum()
    return {
        "trades": int(len(frame)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "win_rate": float((pnl > 0.0).mean()),
        "profit_factor": float(wins / losses) if losses > 0.0 else (999.0 if wins > 0.0 else 0.0),
    }


def category_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    out = metrics(frame)
    out["by_side"] = simple_group(frame, "right")
    out["by_time_bucket"] = simple_group(frame, "time_bucket")
    out["by_category"] = simple_group(frame, "category")
    return out


def category_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return simple_group(frame, "category")


def simple_group(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): metrics(group) for key, group in frame.groupby(column, dropna=False)}


def side_summary(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in sorted(set(baseline["right"]) | set(candidate["right"])):
        b = baseline[baseline["right"].eq(side)]
        c = candidate[candidate["right"].eq(side)]
        out[str(side)] = {
            "protocol101": metrics(b),
            "protocol175": metrics(c),
            "delta_pnl": float(c["pnl"].sum() - b["pnl"].sum()),
            "delta_trades": int(len(c) - len(b)),
        }
    return out


def bucket_summary(baseline: pd.DataFrame, candidate: pd.DataFrame, column: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bucket in ["first30", "post_open", "midday", "late", "after"]:
        b = baseline[baseline[column].eq(bucket)]
        c = candidate[candidate[column].eq(bucket)]
        if b.empty and c.empty:
            continue
        out[bucket] = {
            "protocol101": metrics(b),
            "protocol175": metrics(c),
            "delta_pnl": float(c["pnl"].sum() - b["pnl"].sum()),
            "delta_trades": int(len(c) - len(b)),
        }
    return out


def seed_summary(baseline: pd.DataFrame, candidate: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for seed in sorted(set(baseline["seed"]) | set(candidate["seed"])):
        b = baseline[baseline["seed"].eq(seed)]
        c = candidate[candidate["seed"].eq(seed)]
        rows.append(
            {
                "seed": int(seed),
                "protocol101_pnl": float(b["pnl"].sum()),
                "protocol175_pnl": float(c["pnl"].sum()),
                "delta_pnl": float(c["pnl"].sum() - b["pnl"].sum()),
                "protocol101_trades": int(len(b)),
                "protocol175_trades": int(len(c)),
                "delta_trades": int(len(c) - len(b)),
            }
        )
    return rows


def top_records(frame: pd.DataFrame, *, ascending: bool, n: int = 15) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    columns = [
        "seed",
        "session",
        "decision_time",
        "exit_time",
        "right",
        "offset",
        "pnl",
        "category",
        "time_bucket",
        "contract_id",
        "candidate_uid",
    ]
    return (
        frame.sort_values("pnl", ascending=ascending)
        .head(n)[[column for column in columns if column in frame.columns]]
        .to_dict("records")
    )


def answer(split_payloads: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, item in split_payloads.items():
        side = item["by_side"]
        buckets = item["by_time_bucket"]
        categories = item["protocol101_only_by_category"]
        missed = item["missed_winners"]
        added_losers = item["added_losers"]
        out[split] = {
            "total_delta": item["pnl_delta_protocol175_minus_protocol101"],
            "primary_loss_bucket": max(
                buckets.items(),
                key=lambda kv: abs(float(kv[1]["delta_pnl"])),
            )[0]
            if buckets
            else None,
            "call_delta": side.get("C", {}).get("delta_pnl", 0.0),
            "put_delta": side.get("P", {}).get("delta_pnl", 0.0),
            "missed_winner_pnl": missed["total_pnl"],
            "added_loser_pnl": added_losers["total_pnl"],
            "same_minute_swap_delta": item["decomposition"]["same_minute_swap_delta"],
            "slot_occupancy_delta": item["decomposition"]["slot_occupancy_delta"],
            "flat_decision_delta": item["decomposition"]["flat_decision_delta"],
            "candidate_flat_skipped_pnl": categories.get("candidate_flat_skipped", {}).get("total_pnl", 0.0),
            "candidate_was_holding_pnl": categories.get("candidate_was_holding", {}).get("total_pnl", 0.0),
        }
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol177 Protocol175 Vs Protocol101 Attribution",
        "",
        f"- Baseline: `{payload['baseline']}`",
        f"- Candidate: `{payload['candidate']}`",
        f"- Paid data downloaded: `{payload['paid_data_downloaded']}`",
        f"- Broker endpoint called: `{payload['broker_endpoint_called']}`",
        "",
        "## Summary",
        "",
        "| split | P101 PnL | P175 PnL | delta | exact overlap | swap delta | holding delta | flat-skip delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in payload["splits"].items():
        lines.append(
            f"| {split} | {fmt(item['protocol101']['total_pnl'])} | {fmt(item['protocol175']['total_pnl'])} | "
            f"{fmt(item['pnl_delta_protocol175_minus_protocol101'])} | {item['exact_overlap']['trades']} | "
            f"{fmt(item['decomposition']['same_minute_swap_delta'])} | "
            f"{fmt(item['decomposition']['slot_occupancy_delta'])} | "
            f"{fmt(item['decomposition']['flat_decision_delta'])} |"
        )
    lines.extend(["", "## Answers", ""])
    for split, item in payload["answer"].items():
        seed_rows = payload["splits"][split].get("by_seed", [])
        seed_text = ", ".join(
            f"seed {row['seed']}: {fmt(row['delta_pnl'])}"
            for row in seed_rows
        )
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Total delta: `{fmt(item['total_delta'])}`",
                f"- Call delta: `{fmt(item['call_delta'])}`",
                f"- Put delta: `{fmt(item['put_delta'])}`",
                f"- Primary time bucket by absolute delta: `{item['primary_loss_bucket']}`",
                f"- Missed winner PnL: `{fmt(item['missed_winner_pnl'])}`",
                f"- Added loser PnL: `{fmt(item['added_loser_pnl'])}`",
                f"- Same-minute swap delta: `{fmt(item['same_minute_swap_delta'])}`",
                f"- Slot-occupancy delta: `{fmt(item['slot_occupancy_delta'])}`",
                f"- Flat decision delta: `{fmt(item['flat_decision_delta'])}`",
                f"- Seed deltas: `{seed_text}`",
                "",
            ]
        )
    lines.extend(["## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Details CSV: `{path.parent / 'trade_attribution_details.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "inf" if value > 0 else "-inf"
    return value


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.0f}"


if __name__ == "__main__":
    raise SystemExit(main())
