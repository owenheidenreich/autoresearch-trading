"""Protocol 114: skeptical falsification audit for frozen Protocol 101.

This runner intentionally does not train, download data, or touch broker APIs.
It reads frozen local ledgers and candidate datasets, then tries to break the
Protocol 101 story through baseline, random, concentration, stress, delay, and
paper-account checks.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_114_protocol101_skeptical_falsification")
DEFAULT_PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
DEFAULT_PROTOCOL097_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_097_sequential_event_policy")
DEFAULT_PROTOCOL092_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
DEFAULT_PROTOCOL107_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress")
DEFAULT_PROTOCOL113_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DEFAULT_NORMALIZED_DIRS = (
    Path("v4/normalized_official_context"),
    Path("v4/normalized"),
    Path("v4/normalized_official_context_fix_smoke"),
    Path("v4/normalized_official_context_smoke"),
)
REQUIRED_SPLITS = ["q4_2024_external", "q3_2025", "q4_2025", "q1_2026", "march_2026"]
REGISTERED_SPLITS = ["q3_2025", "q4_2025", "q1_2026", "march_2026"]
HEADLINE_SPLITS = ["q4_2024_external", "q3_2025", "q4_2025", "q1_2026"]
STRESS_PER_SIDE = [0.10, 0.25, 0.50]
CONTRACT_MULTIPLIER = 100.0
STARTING_EQUITY = 10_000.0


@dataclass(frozen=True)
class PaperAccountCheck:
    seed: int
    trades: int
    skipped_unaffordable: int
    overlap_errors: int
    ending_equity: float
    min_cash_before_trade: float
    max_premium: float
    all_flat_by_session_end: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--protocol097-dir", type=Path, default=DEFAULT_PROTOCOL097_DIR)
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--protocol107-dir", type=Path, default=DEFAULT_PROTOCOL107_DIR)
    parser.add_argument("--protocol113-dir", type=Path, default=DEFAULT_PROTOCOL113_DIR)
    parser.add_argument("--normalized-dir", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--random-paths", type=int, default=64)
    parser.add_argument("--starting-equity", type=float, default=STARTING_EQUITY)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalized_dirs = tuple(args.normalized_dir or DEFAULT_NORMALIZED_DIRS)

    features = load_feature_frame(args.protocol092_dir, args.protocol107_dir)
    model = load_protocol101_trades(args.protocol101_dir, args.protocol107_dir)
    model = enrich_trades(model, features)
    baseline = load_baseline_trades(args.protocol092_dir, args.protocol107_dir)
    baseline = enrich_trades(baseline, features)
    protocol097 = enrich_trades(load_json_trades(args.protocol097_dir / "serial_policy_trades.json", "protocol097"), features)
    protocol092 = enrich_trades(load_json_trades(args.protocol092_dir / "serial_policy_trades.json", "protocol092"), features)

    headline = headline_trades(model)
    random_summary = matched_random_summary(
        model=model,
        features=features,
        paths=int(args.random_paths),
        seed=114,
    )
    delay_summary, delay_rows = delay_stress_summary(
        model,
        normalized_dirs=normalized_dirs,
    )
    split_summary = split_comparison_summary(
        model=model,
        baseline=baseline,
        protocol097=protocol097,
        protocol092=protocol092,
        random_summary=random_summary,
    )
    group_breakdown = build_group_breakdown(headline)
    concentration = concentration_summary(headline)
    paper_checks = paper_account_checks(headline, starting_equity=float(args.starting_equity))
    headline_integrity = chart_source_of_truth_checks(args.protocol113_dir)
    decision = decide(
        split_summary=split_summary,
        random_summary=random_summary,
        concentration=concentration,
        paper_checks=paper_checks,
        delay_summary=delay_summary,
        headline_integrity=headline_integrity,
    )
    next_hypothesis = next_hypothesis_from_failures(
        split_summary=split_summary,
        delay_summary=delay_summary,
        concentration=concentration,
        paper_checks=paper_checks,
    )

    payload = {
        "protocol": "114_protocol101_skeptical_falsification",
        "paid_data_downloaded": False,
        "live_orders": False,
        "model_training": False,
        "source_protocol101_dir": str(args.protocol101_dir),
        "source_protocol107_dir": str(args.protocol107_dir),
        "source_protocol092_dir": str(args.protocol092_dir),
        "source_protocol097_dir": str(args.protocol097_dir),
        "source_of_truth_equity": str(args.protocol113_dir / "equity.html"),
        "decision": decision,
        "next_hypothesis": next_hypothesis,
        "split_summary": split_summary,
        "matched_random": random_summary,
        "concentration": concentration,
        "paper_account_checks": [check.__dict__ for check in paper_checks],
        "delay_stress_summary": delay_summary,
        "headline_integrity": headline_integrity,
        "stress_per_side": STRESS_PER_SIDE,
        "row_counts": {
            "model_trades": int(len(model)),
            "headline_trades": int(len(headline)),
            "baseline_trades": int(len(baseline)),
            "feature_rows": int(len(features)),
        },
    }
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    group_breakdown.to_csv(args.out_dir / "group_breakdown.csv", index=False)
    pd.DataFrame(delay_rows).to_csv(args.out_dir / "delay_stress_rows.csv", index=False)
    write_report(args.out_dir / "report.md", payload, group_breakdown)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")

    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_feature_frame(protocol092_dir: Path, protocol107_dir: Path) -> pd.DataFrame:
    paths = [
        protocol092_dir / "serial_opportunity_dataset.parquet",
        protocol107_dir / "q4_2024_external_serial_opportunity_dataset.parquet",
    ]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "reported_split" not in frame.columns:
            frame["reported_split"] = frame["split"].astype(str)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no Protocol 092-compatible candidate datasets found")
    features = pd.concat(frames, ignore_index=True)
    features["candidate_uid"] = features["candidate_uid"].astype(str)
    features["decision_ts"] = pd.to_datetime(features["decision_time"], utc=True)
    features["exit_ts"] = pd.to_datetime(features["candidate_exit_time"], utc=True)
    return features.drop_duplicates("candidate_uid", keep="last").reset_index(drop=True)


def load_protocol101_trades(protocol101_dir: Path, protocol107_dir: Path) -> pd.DataFrame:
    registered = load_json_trades(protocol101_dir / "serial_policy_trades.json", "protocol101")
    external = load_json_trades(protocol107_dir / "serial_policy_trades.json", "protocol101_q4_2024_external")
    external["reported_split"] = "q4_2024_external"
    return pd.concat([external, registered], ignore_index=True)


def load_baseline_trades(protocol092_dir: Path, protocol107_dir: Path) -> pd.DataFrame:
    registered = load_json_trades(protocol092_dir / "strict_serial_baseline_trades.json", "strict_serial_baseline")
    external = load_json_trades(protocol107_dir / "strict_serial_baseline_trades.json", "strict_serial_baseline_q4_2024")
    external["reported_split"] = "q4_2024_external"
    return pd.concat([external, registered], ignore_index=True)


def load_json_trades(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = json.loads(path.read_text())
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["source"] = source
    if "reported_split" not in frame.columns:
        frame["reported_split"] = frame.get("split", "")
    frame["reported_split"] = frame["reported_split"].fillna(frame.get("split", "")).astype(str)
    frame["candidate_uid"] = frame["candidate_uid"].astype(str)
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["exit_ts"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype(int)
    frame["session"] = frame["session"].astype(str)
    frame["right"] = frame["right"].astype(str)
    return frame


def enrich_trades(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    feature_columns = [
        "candidate_uid",
        "time_bucket",
        "entry_quote_time",
        "entry_ask",
        "entry_bid",
        "entry_mid",
        "entry_spread",
        "entry_spread_frac",
        "entry_spread_over_mid",
        "entry_gamma",
        "entry_theta",
        "entry_gamma_theta_ratio",
        "entry_theta_burden",
        "entry_gamma_per_premium",
        "entry_premium_over_underlying",
        "edge",
        "entry_bid_size",
        "entry_ask_size",
        "candidate_pnl",
        "candidate_exit_time",
        "candidate_exit_reason",
    ]
    keep = [column for column in feature_columns if column in features.columns]
    merged = trades.merge(features[keep], on="candidate_uid", how="left", suffixes=("", "_feature"))
    for column in [
        "entry_ask",
        "entry_bid",
        "entry_mid",
        "entry_spread",
        "entry_spread_frac",
        "entry_spread_over_mid",
        "entry_gamma",
        "entry_theta",
        "entry_gamma_theta_ratio",
        "entry_theta_burden",
        "entry_gamma_per_premium",
        "entry_premium_over_underlying",
        "edge",
        "entry_bid_size",
        "entry_ask_size",
    ]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["entry_premium"] = merged["entry_ask"] * CONTRACT_MULTIPLIER
    merged["month"] = merged["session"].str.slice(0, 7)
    merged["week"] = merged["decision_ts"].dt.strftime("%G-W%V")
    merged["day"] = merged["session"]
    merged["side"] = np.where(merged["right"] == "C", "CALL", np.where(merged["right"] == "P", "PUT", merged["right"]))
    if "time_bucket" not in merged.columns:
        merged["time_bucket"] = time_bucket_from_ts(merged["decision_ts"])
    merged["time_bucket"] = merged["time_bucket"].fillna(time_bucket_from_ts(merged["decision_ts"]))
    merged["offset_bucket"] = pd.cut(
        pd.to_numeric(merged["offset"], errors="coerce").abs(),
        bins=[-0.01, 5, 15, 30, 50, math.inf],
        labels=["0-5", "10-15", "20-30", "35-50", "50+"],
    ).astype(str)
    merged["premium_bucket"] = pd.cut(
        merged["entry_premium"],
        bins=[-0.01, 1000, 2000, 3000, 4000, math.inf],
        labels=["<=1k", "1k-2k", "2k-3k", "3k-4k", "4k+"],
    ).astype(str)
    merged["spread_bucket"] = pd.cut(
        merged["entry_spread_over_mid"],
        bins=[-0.01, 0.01, 0.02, 0.04, 0.08, math.inf],
        labels=["<=1%", "1-2%", "2-4%", "4-8%", "8%+"],
    ).astype(str)
    merged["gamma_theta_bucket"] = pd.cut(
        merged["entry_gamma_theta_ratio"].abs(),
        bins=[-0.01, 0.00015, 0.00030, 0.00050, 0.001, math.inf],
        labels=["very_low", "low", "medium", "high", "extreme"],
    ).astype(str)
    if "exit_reason" in merged.columns:
        merged["entry_reason"] = merged["exit_reason"].astype(str)
    else:
        merged["entry_reason"] = "unknown"
    return merged


def time_bucket_from_ts(ts: pd.Series) -> pd.Series:
    local = ts.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    return pd.Series(
        np.select(
            [minute < 600, minute < 690, minute < 810, minute <= 930],
            ["first30", "post_open_morning", "midday", "late_afternoon"],
            default="after_hours",
        ),
        index=ts.index,
    )


def headline_trades(model: pd.DataFrame) -> pd.DataFrame:
    return model[model["reported_split"].isin(HEADLINE_SPLITS)].copy()


def gross_profit(pnls: pd.Series) -> float:
    return float(pnls[pnls > 0].sum())


def gross_loss(pnls: pd.Series) -> float:
    return float(pnls[pnls < 0].sum())


def profit_factor(pnls: pd.Series) -> float:
    loss = abs(gross_loss(pnls))
    if loss <= 1e-9:
        return float("inf") if gross_profit(pnls) > 0 else 0.0
    return gross_profit(pnls) / loss


def stressed_pnl(pnl: float, stress_per_side: float) -> float:
    return float(pnl) - float(stress_per_side) * 2.0 * CONTRACT_MULTIPLIER


def split_seed_totals(frame: pd.DataFrame, pnl_column: str = "pnl") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["reported_split", "seed", "total_pnl", "trades", "profit_factor"])
    rows = []
    for (split, seed), group in frame.groupby(["reported_split", "seed"], sort=True):
        rows.append(
            {
                "reported_split": split,
                "seed": int(seed),
                "total_pnl": float(group[pnl_column].sum()),
                "trades": int(len(group)),
                "profit_factor": profit_factor(group[pnl_column]),
            }
        )
    return pd.DataFrame(rows)


def split_comparison_summary(
    *,
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    protocol097: pd.DataFrame,
    protocol092: pd.DataFrame,
    random_summary: dict[str, Any],
) -> dict[str, Any]:
    out = {}
    model_seed = split_seed_totals(model)
    baseline_seed = split_seed_totals(baseline)
    p97_seed = split_seed_totals(protocol097)
    p92_seed = split_seed_totals(protocol092)
    for split in REQUIRED_SPLITS:
        model_rows = model[model["reported_split"] == split].copy()
        seed_rows = model_seed[model_seed["reported_split"] == split]
        baseline_rows = baseline_seed[baseline_seed["reported_split"] == split]
        p97_rows = p97_seed[p97_seed["reported_split"] == split]
        p92_rows = p92_seed[p92_seed["reported_split"] == split]
        stress = {}
        for amount in STRESS_PER_SIDE:
            stressed = model_rows.assign(stressed_pnl=model_rows["pnl"].map(lambda value: stressed_pnl(value, amount)))
            stressed_seed = split_seed_totals(stressed, "stressed_pnl")
            stress[f"{amount:.2f}"] = {
                "median_total_pnl": median_or_none(stressed_seed["total_pnl"]),
                "positive_seed_fraction": positive_fraction(stressed_seed["total_pnl"]),
            }
        baseline_median = median_or_none(baseline_rows["total_pnl"])
        out[split] = {
            "trades": int(len(model_rows)),
            "median_total_pnl": median_or_none(seed_rows["total_pnl"]),
            "median_profit_factor": median_or_none(seed_rows["profit_factor"]),
            "positive_seed_fraction": positive_fraction(seed_rows["total_pnl"]),
            "strict_baseline_median_total_pnl": baseline_median,
            "strict_baseline_delta": none_subtract(median_or_none(seed_rows["total_pnl"]), baseline_median),
            "protocol097_median_total_pnl": median_or_none(p97_rows["total_pnl"]),
            "protocol097_delta": none_subtract(median_or_none(seed_rows["total_pnl"]), median_or_none(p97_rows["total_pnl"])),
            "protocol092_median_total_pnl": median_or_none(p92_rows["total_pnl"]),
            "protocol092_delta": none_subtract(median_or_none(seed_rows["total_pnl"]), median_or_none(p92_rows["total_pnl"])),
            "stress": stress,
            "matched_random_median_total_pnl": random_summary.get(split, {}).get("median_total_pnl"),
            "matched_random_delta": none_subtract(
                median_or_none(seed_rows["total_pnl"]),
                random_summary.get(split, {}).get("median_total_pnl"),
            ),
        }
    return out


def median_or_none(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def positive_fraction(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return float((values > 0).mean())


def none_subtract(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def matched_random_summary(*, model: pd.DataFrame, features: pd.DataFrame, paths: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {}
    events = features[features["split"].isin(["q3_2025", "q4_2025", "q1_2026", "q4_2024_external"])].copy()
    events["reported_split"] = events["split"].astype(str)
    for split in REQUIRED_SPLITS:
        base_split = "q1_2026" if split == "march_2026" else split
        split_features = events[events["reported_split"] == base_split].copy()
        if split == "march_2026":
            split_features = split_features[split_features["session"] >= "2026-03-01"].copy()
        model_split = model[model["reported_split"] == split].copy()
        if split_features.empty or model_split.empty:
            continue
        path_totals = []
        path_trades = []
        for path_idx in range(paths):
            per_seed_totals = []
            per_seed_trades = []
            for seed_value in sorted(model_split["seed"].unique()):
                target = int(len(model_split[model_split["seed"] == seed_value]))
                source = split_features[split_features["seed"] == seed_value].copy()
                trades = simulate_random_serial(source, target_trades=target, rng=rng)
                per_seed_totals.append(sum(float(row["pnl"]) for row in trades))
                per_seed_trades.append(len(trades))
            path_totals.append(float(np.median(per_seed_totals)) if per_seed_totals else 0.0)
            path_trades.append(float(np.median(per_seed_trades)) if per_seed_trades else 0.0)
        summary[split] = {
            "paths": int(paths),
            "median_total_pnl": float(np.median(path_totals)),
            "p10_total_pnl": float(np.quantile(path_totals, 0.10)),
            "p90_total_pnl": float(np.quantile(path_totals, 0.90)),
            "median_trades": float(np.median(path_trades)),
        }
    return summary


def simulate_random_serial(source: pd.DataFrame, *, target_trades: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if source.empty or target_trades <= 0:
        return []
    source = source.sort_values(["session", "decision_ts", "entry_seed", "contract_id", "candidate_uid"]).copy()
    event_keys = list(source.groupby(["session", "decision_time"], sort=False).groups.keys())
    take_probability = min(1.0, target_trades / max(len(event_keys), 1))
    held_until = pd.Timestamp.min.tz_localize("UTC")
    trades: list[dict[str, Any]] = []
    for (session, decision_time), group in source.groupby(["session", "decision_time"], sort=False):
        decision_ts = pd.Timestamp(group["decision_ts"].iloc[0])
        if decision_ts < held_until:
            continue
        if len(trades) >= target_trades:
            break
        if rng.random() > take_probability:
            continue
        row = group.iloc[int(rng.integers(0, len(group)))]
        held_until = pd.Timestamp(row["candidate_exit_dt"])
        trades.append(
            {
                "candidate_uid": str(row["candidate_uid"]),
                "session": str(session),
                "decision_time": str(decision_time),
                "exit_time": str(row["candidate_exit_time"]),
                "pnl": float(row["candidate_pnl"]),
            }
        )
    return trades


def build_group_breakdown(headline: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "reported_split",
        "month",
        "week",
        "day",
        "side",
        "time_bucket",
        "offset_bucket",
        "premium_bucket",
        "spread_bucket",
        "gamma_theta_bucket",
        "entry_reason",
    ]
    rows = []
    for dimension in dimensions:
        for value, group in headline.groupby(dimension, dropna=False, sort=True):
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "trades": int(len(group)),
                    "pnl": float(group["pnl"].sum()),
                    "stress_0_25_pnl": float(group["pnl"].map(lambda v: stressed_pnl(v, 0.25)).sum()),
                    "win_rate": float((group["pnl"] >= 0).mean()) if len(group) else 0.0,
                    "profit_factor": profit_factor(group["pnl"]),
                }
            )
    return pd.DataFrame(rows)


def concentration_summary(headline: pd.DataFrame) -> dict[str, Any]:
    if headline.empty:
        return {}
    sorted_trades = headline.sort_values("pnl", ascending=False)
    total = float(headline["pnl"].sum())
    gross = gross_profit(headline["pnl"])
    day_pnl = headline.groupby("day")["pnl"].sum().sort_values(ascending=False)
    month_pnl = headline.groupby("month")["pnl"].sum().sort_values(ascending=False)
    week_pnl = headline.groupby("week")["pnl"].sum().sort_values(ascending=False)
    top = {}
    for n in [5, 10, 20]:
        value = float(sorted_trades.head(n)["pnl"].sum())
        top[f"top_{n}_trades_pnl"] = value
        top[f"top_{n}_trades_share_net"] = value / total if total else None
        top[f"top_{n}_trades_share_gross"] = value / gross if gross else None
    leave_one_month = []
    for month, pnl in month_pnl.items():
        leave_one_month.append({"month": str(month), "pnl_without_month": float(total - pnl), "month_pnl": float(pnl)})
    return {
        "total_pnl": total,
        "gross_profit": gross,
        **top,
        "top_day_share_net": float(day_pnl.iloc[0] / total) if total and not day_pnl.empty else None,
        "top_5_day_share_net": float(day_pnl.head(5).sum() / total) if total and not day_pnl.empty else None,
        "best_day": {"day": str(day_pnl.index[0]), "pnl": float(day_pnl.iloc[0])},
        "worst_day": {"day": str(day_pnl.index[-1]), "pnl": float(day_pnl.iloc[-1])},
        "best_week": {"week": str(week_pnl.index[0]), "pnl": float(week_pnl.iloc[0])},
        "worst_week": {"week": str(week_pnl.index[-1]), "pnl": float(week_pnl.iloc[-1])},
        "leave_one_month": leave_one_month,
        "single_trade_majority": bool(sorted_trades.iloc[0]["pnl"] > 0.5 * total) if total > 0 else True,
        "top20_majority": bool(top["top_20_trades_share_net"] is not None and top["top_20_trades_share_net"] > 0.5),
    }


def paper_account_checks(headline: pd.DataFrame, *, starting_equity: float) -> list[PaperAccountCheck]:
    checks = []
    for seed, rows in headline.groupby("seed", sort=True):
        rows = rows.sort_values(["decision_ts", "exit_ts", "candidate_uid"])
        cash = float(starting_equity)
        held_until_by_session: dict[str, pd.Timestamp] = {}
        skipped_unaffordable = 0
        overlap_errors = 0
        min_cash_before = cash
        max_premium = 0.0
        all_flat_by_session_end = True
        for _, row in rows.iterrows():
            premium = float(row.get("entry_premium", math.nan))
            if not math.isfinite(premium):
                skipped_unaffordable += 1
                continue
            min_cash_before = min(min_cash_before, cash)
            max_premium = max(max_premium, premium)
            session = str(row["session"])
            decision_ts = pd.Timestamp(row["decision_ts"])
            if session in held_until_by_session and decision_ts < held_until_by_session[session]:
                overlap_errors += 1
            if premium > cash + 1e-9:
                skipped_unaffordable += 1
                continue
            cash += float(row["pnl"])
            exit_ts = pd.Timestamp(row["exit_ts"])
            held_until_by_session[session] = max(held_until_by_session.get(session, pd.Timestamp.min.tz_localize("UTC")), exit_ts)
            exit_local = exit_ts.tz_convert("America/New_York")
            session_close = exit_local.normalize() + pd.Timedelta(hours=16)
            if exit_local > session_close:
                all_flat_by_session_end = False
        checks.append(
            PaperAccountCheck(
                seed=int(seed),
                trades=int(len(rows)),
                skipped_unaffordable=int(skipped_unaffordable),
                overlap_errors=int(overlap_errors),
                ending_equity=float(cash),
                min_cash_before_trade=float(min_cash_before),
                max_premium=float(max_premium),
                all_flat_by_session_end=all_flat_by_session_end,
            )
        )
    return checks


def find_normalized_path(session: str, normalized_dirs: tuple[Path, ...]) -> Path | None:
    for directory in normalized_dirs:
        if not directory.exists():
            continue
        matches = sorted(directory.glob(f"*{session}*.parquet"))
        if matches:
            return matches[0]
    return None


def delay_stress_summary(model: pd.DataFrame, *, normalized_dirs: tuple[Path, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for session, session_trades in model.groupby("session", sort=True):
        path = find_normalized_path(str(session), normalized_dirs)
        if path is None:
            for _, trade in session_trades.iterrows():
                rows.append(delay_missing_row(trade, "missing_normalized_session"))
            continue
        try:
            quotes = pd.read_parquet(path, columns=["quote_time", "contract_id", "bid", "ask"])
        except Exception:
            for _, trade in session_trades.iterrows():
                rows.append(delay_missing_row(trade, "unreadable_normalized_session"))
            continue
        quotes["quote_time"] = pd.to_datetime(quotes["quote_time"], utc=True)
        quotes["bid"] = pd.to_numeric(quotes["bid"], errors="coerce")
        quotes["ask"] = pd.to_numeric(quotes["ask"], errors="coerce")
        by_contract = {str(k): g.sort_values("quote_time").reset_index(drop=True) for k, g in quotes.groupby("contract_id")}
        for _, trade in session_trades.iterrows():
            contract_quotes = by_contract.get(str(trade["contract_id"]))
            if contract_quotes is None or contract_quotes.empty:
                rows.append(delay_missing_row(trade, "missing_contract_quotes"))
                continue
            rows.append(delay_stress_for_trade(trade.to_dict(), contract_quotes))
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    if frame.empty:
        return summary, rows
    for split, group in frame.groupby("reported_split", sort=True):
        ok_entry = group.dropna(subset=["entry_delay_pnl"])
        ok_exit = group.dropna(subset=["exit_delay_pnl"])
        ok_both = group.dropna(subset=["both_delay_pnl"])
        original_by_seed = group.groupby("seed")["pnl"].sum()
        entry_by_seed = ok_entry.groupby("seed")["entry_delay_pnl"].sum()
        exit_by_seed = ok_exit.groupby("seed")["exit_delay_pnl"].sum()
        both_by_seed = ok_both.groupby("seed")["both_delay_pnl"].sum()
        summary[str(split)] = {
            "rows": int(len(group)),
            "original_median_total_pnl": median_or_none(original_by_seed),
            "entry_delay_coverage": float(len(ok_entry) / len(group)) if len(group) else 0.0,
            "exit_delay_coverage": float(len(ok_exit) / len(group)) if len(group) else 0.0,
            "both_delay_coverage": float(len(ok_both) / len(group)) if len(group) else 0.0,
            "entry_delay_median_total_pnl": median_or_none(entry_by_seed),
            "exit_delay_median_total_pnl": median_or_none(exit_by_seed),
            "both_delay_median_total_pnl": median_or_none(both_by_seed),
            "entry_delay_delta": float((ok_entry["entry_delay_pnl"] - ok_entry["pnl"]).sum()) if not ok_entry.empty else None,
            "exit_delay_delta": float((ok_exit["exit_delay_pnl"] - ok_exit["pnl"]).sum()) if not ok_exit.empty else None,
            "both_delay_delta": float((ok_both["both_delay_pnl"] - ok_both["pnl"]).sum()) if not ok_both.empty else None,
        }
    return summary, rows


def delay_missing_row(trade: pd.Series, status: str) -> dict[str, Any]:
    return {
        "candidate_uid": str(trade["candidate_uid"]),
        "reported_split": str(trade["reported_split"]),
        "seed": int(trade["seed"]),
        "session": str(trade["session"]),
        "pnl": float(trade["pnl"]),
        "status": status,
        "entry_delay_pnl": None,
        "exit_delay_pnl": None,
        "both_delay_pnl": None,
    }


def delay_stress_for_trade(trade: dict[str, Any], quotes: pd.DataFrame) -> dict[str, Any]:
    decision = pd.Timestamp(trade["decision_ts"])
    exit_ts = pd.Timestamp(trade["exit_ts"])
    entry_ask = float(trade.get("entry_ask", math.nan))
    pnl = float(trade.get("pnl", 0.0))
    if not math.isfinite(entry_ask):
        return {**delay_missing_row(pd.Series(trade), "missing_entry_ask")}
    original_exit_bid = entry_ask + pnl / CONTRACT_MULTIPLIER
    entry_quote = first_valid_quote_at_or_after(quotes, decision + pd.Timedelta(minutes=1), "ask")
    exit_quote = first_valid_quote_at_or_after(quotes, exit_ts + pd.Timedelta(minutes=1), "bid")
    entry_delay_pnl = None
    exit_delay_pnl = None
    both_delay_pnl = None
    if entry_quote is not None and pd.Timestamp(entry_quote["quote_time"]) <= exit_ts:
        entry_delay_pnl = (original_exit_bid - float(entry_quote["ask"])) * CONTRACT_MULTIPLIER
    if exit_quote is not None:
        exit_delay_pnl = (float(exit_quote["bid"]) - entry_ask) * CONTRACT_MULTIPLIER
    if entry_quote is not None and exit_quote is not None and pd.Timestamp(entry_quote["quote_time"]) <= pd.Timestamp(exit_quote["quote_time"]):
        both_delay_pnl = (float(exit_quote["bid"]) - float(entry_quote["ask"])) * CONTRACT_MULTIPLIER
    return {
        "candidate_uid": str(trade["candidate_uid"]),
        "reported_split": str(trade["reported_split"]),
        "seed": int(trade["seed"]),
        "session": str(trade["session"]),
        "pnl": pnl,
        "status": "ok",
        "entry_delay_pnl": entry_delay_pnl,
        "exit_delay_pnl": exit_delay_pnl,
        "both_delay_pnl": both_delay_pnl,
    }


def first_valid_quote_at_or_after(quotes: pd.DataFrame, target: pd.Timestamp, price_column: str) -> pd.Series | None:
    valid = quotes[(quotes["quote_time"] >= target) & quotes[price_column].notna()]
    if price_column == "ask":
        valid = valid[valid["ask"] > 0]
    if price_column == "bid":
        valid = valid[valid["bid"] >= 0]
    if valid.empty:
        return None
    return valid.iloc[0]


def chart_source_of_truth_checks(protocol113_dir: Path) -> dict[str, Any]:
    files = sorted(path.name for path in protocol113_dir.glob("*") if path.is_file())
    equity_files = [name for name in files if name.endswith("equity.html")]
    trades_path = protocol113_dir / "trades.csv"
    train_validation_rows = 0
    if trades_path.exists():
        try:
            trades = pd.read_csv(trades_path)
            if "stage" in trades.columns:
                train_validation_rows = int(trades["stage"].isin(["train", "validation"]).sum())
        except Exception:
            train_validation_rows = -1
    return {
        "exists": protocol113_dir.exists(),
        "equity_files": equity_files,
        "single_equity_source": equity_files == ["equity.html"],
        "train_validation_rows_in_trades_csv": train_validation_rows,
        "no_train_validation_in_headline": train_validation_rows == 0,
    }


def decide(
    *,
    split_summary: dict[str, Any],
    random_summary: dict[str, Any],
    concentration: dict[str, Any],
    paper_checks: list[PaperAccountCheck],
    delay_summary: dict[str, Any],
    headline_integrity: dict[str, Any],
) -> str:
    core_positive = all((split_summary.get(split, {}).get("median_total_pnl") or -1.0) > 0 for split in REQUIRED_SPLITS)
    stress25_positive = all(
        (split_summary.get(split, {}).get("stress", {}).get("0.25", {}).get("median_total_pnl") or -1.0) > 0
        for split in REQUIRED_SPLITS
    )
    random_worse = all(
        (split_summary.get(split, {}).get("matched_random_delta") is not None)
        and split_summary[split]["matched_random_delta"] > 5_000.0
        for split in REQUIRED_SPLITS
        if split in random_summary
    )
    concentration_ok = not concentration.get("single_trade_majority", True) and not concentration.get("top20_majority", True)
    paper_ok = all(check.skipped_unaffordable == 0 and check.overlap_errors == 0 for check in paper_checks)
    chart_ok = bool(headline_integrity.get("single_equity_source")) and bool(headline_integrity.get("no_train_validation_in_headline"))
    if not (core_positive and stress25_positive and random_worse and concentration_ok and paper_ok and chart_ok):
        return "reject_edge_hypothesis"
    thin_baseline = any(
        (split_summary.get(split, {}).get("strict_baseline_delta") is not None)
        and split_summary[split]["strict_baseline_delta"] < 2_000.0
        for split in ["q4_2024_external", "q4_2025"]
    )
    delay_fragile = any(
        (row.get("entry_delay_median_total_pnl") is not None and row["entry_delay_median_total_pnl"] <= 0.0)
        or (row.get("exit_delay_median_total_pnl") is not None and row["exit_delay_median_total_pnl"] <= 0.0)
        or (row.get("both_delay_median_total_pnl") is not None and row["both_delay_median_total_pnl"] <= 0.0)
        or (row.get("entry_delay_delta") is not None and row["entry_delay_delta"] < -20_000.0)
        or (row.get("exit_delay_delta") is not None and row["exit_delay_delta"] < -20_000.0)
        for row in delay_summary.values()
    )
    if thin_baseline or delay_fragile:
        return "fragile_needs_more_data"
    return "survives_falsification"


def next_hypothesis_from_failures(
    *,
    split_summary: dict[str, Any],
    delay_summary: dict[str, Any],
    concentration: dict[str, Any],
    paper_checks: list[PaperAccountCheck],
) -> str:
    if any(check.skipped_unaffordable or check.overlap_errors for check in paper_checks):
        return "fix paper-account and serial-execution realism before changing the model"
    if any(
        (row.get("entry_delay_median_total_pnl") is not None and row["entry_delay_median_total_pnl"] <= 0.0)
        or (row.get("exit_delay_median_total_pnl") is not None and row["exit_delay_median_total_pnl"] <= 0.0)
        or (row.get("both_delay_median_total_pnl") is not None and row["both_delay_median_total_pnl"] <= 0.0)
        or (row.get("entry_delay_delta") is not None and row["entry_delay_delta"] < -20_000.0)
        or (row.get("exit_delay_delta") is not None and row["exit_delay_delta"] < -20_000.0)
        for row in delay_summary.values()
    ):
        return "prioritize 1s/tick/live-shadow validation because edge is timing-sensitive"
    if concentration.get("top20_majority"):
        return "pause model work; test whether performance is a small convex-winner sample artifact"
    q4_delta = split_summary.get("q4_2024_external", {}).get("strict_baseline_delta")
    q4_2025_delta = split_summary.get("q4_2025", {}).get("strict_baseline_delta")
    if (q4_delta is not None and q4_delta < 2_000.0) or (q4_2025_delta is not None and q4_2025_delta < 2_000.0):
        return "broader locked historical validation before model changes; if repeated, test longer-memory event policy"
    return "keep Protocol 101 frozen and wait for live shadow parity or broader locked data"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def write_report(path: Path, payload: dict[str, Any], group_breakdown: pd.DataFrame) -> None:
    lines = [
        "# Protocol 114: Skeptical Falsification Before More Model Design",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Next hypothesis: `{payload['next_hypothesis']}`",
        f"- Source of truth equity: `{payload['source_of_truth_equity']}`",
        f"- Model trades audited: `{payload['row_counts']['model_trades']}`",
        f"- Headline paper-account trades: `{payload['row_counts']['headline_trades']}`",
        "",
        "## Split Falsification",
        "",
        "| split | median_pnl | stress_0.25 | strict_baseline | baseline_delta | matched_random | random_delta | p097_delta | p092_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in REQUIRED_SPLITS:
        row = payload["split_summary"].get(split, {})
        lines.append(
            "| "
            f"{split} | "
            f"{money(row.get('median_total_pnl'))} | "
            f"{money(row.get('stress', {}).get('0.25', {}).get('median_total_pnl'))} | "
            f"{money(row.get('strict_baseline_median_total_pnl'))} | "
            f"{money(row.get('strict_baseline_delta'))} | "
            f"{money(row.get('matched_random_median_total_pnl'))} | "
            f"{money(row.get('matched_random_delta'))} | "
            f"{money(row.get('protocol097_delta'))} | "
            f"{money(row.get('protocol092_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Delay Stress",
            "",
            "| split | original_median | entry_delay_median | entry_delay_delta | exit_delay_median | exit_delay_delta | both_delay_median | both_delay_delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, row in sorted(payload["delay_stress_summary"].items()):
        lines.append(
            "| "
            f"{split} | {money(row.get('original_median_total_pnl'))} | "
            f"{money(row.get('entry_delay_median_total_pnl'))} | {money(row.get('entry_delay_delta'))} | "
            f"{money(row.get('exit_delay_median_total_pnl'))} | {money(row.get('exit_delay_delta'))} | "
            f"{money(row.get('both_delay_median_total_pnl'))} | {money(row.get('both_delay_delta'))} |"
        )
    c = payload["concentration"]
    lines.extend(
        [
            "",
            "## Concentration",
            "",
            f"- Total PnL: `{money(c.get('total_pnl'))}`",
            f"- Top 5 trades: `{money(c.get('top_5_trades_pnl'))}` ({pct(c.get('top_5_trades_share_net'))} of net)",
            f"- Top 10 trades: `{money(c.get('top_10_trades_pnl'))}` ({pct(c.get('top_10_trades_share_net'))} of net)",
            f"- Top 20 trades: `{money(c.get('top_20_trades_pnl'))}` ({pct(c.get('top_20_trades_share_net'))} of net)",
            f"- Best day: `{c.get('best_day', {}).get('day')}` `{money(c.get('best_day', {}).get('pnl'))}`",
            f"- Worst day: `{c.get('worst_day', {}).get('day')}` `{money(c.get('worst_day', {}).get('pnl'))}`",
            f"- Best week: `{c.get('best_week', {}).get('week')}` `{money(c.get('best_week', {}).get('pnl'))}`",
            f"- Worst week: `{c.get('worst_week', {}).get('week')}` `{money(c.get('worst_week', {}).get('pnl'))}`",
            "",
            "## Paper Account Checks",
            "",
            "| seed | trades | unaffordable | overlaps | ending_equity | min_cash_before | max_premium |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["paper_account_checks"]:
        lines.append(
            "| "
            f"{row['seed']} | {row['trades']} | {row['skipped_unaffordable']} | {row['overlap_errors']} | "
            f"{money(row['ending_equity'])} | {money(row['min_cash_before_trade'])} | {money(row['max_premium'])} |"
        )
    lines.extend(
        [
            "",
            "## Worst Group Breakdowns",
            "",
            "| dimension | value | trades | pnl | stress_0.25_pnl | win_rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    worst = group_breakdown.sort_values("pnl").head(18)
    for _, row in worst.iterrows():
        lines.append(
            "| "
            f"{row['dimension']} | {row['value']} | {int(row['trades'])} | "
            f"{money(row['pnl'])} | {money(row['stress_0_25_pnl'])} | {pct(row['win_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `summary.json`: machine-readable falsification result.",
            "- `group_breakdown.csv`: split/month/week/day/side/time/contract-quality breakdowns.",
            "- `delay_stress_rows.csv`: per-trade one-minute entry/exit delay replay diagnostics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def money(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    entry = f"""

## 2026-05-13 Protocol 114 Skeptical Falsification

```text
Date: 2026-05-13
Decision / Experiment: Ran a no-training skeptical falsification audit for frozen Protocol 101 before any further neural architecture work.
Reason: The equity curve is exciting enough to be dangerous. The project needed to try to disprove the Protocol 101 edge with baselines, matched random, concentration, paper-account realism, slippage, and one-minute quote-delay stress.
Data Used: Existing Protocol 101, Protocol 107, Protocol 092, Protocol 097, Protocol 113, and normalized official-context artifacts only. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Headline trades {payload['row_counts']['headline_trades']}; model trades audited {payload['row_counts']['model_trades']}; stressed ending and detailed split results are in {report_path}.
Next Gate: {payload['next_hypothesis']}
Owner: Codex
```
"""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(entry)


if __name__ == "__main__":
    raise SystemExit(main())
