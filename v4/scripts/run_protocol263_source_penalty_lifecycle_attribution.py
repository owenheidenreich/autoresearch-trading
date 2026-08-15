"""AUDIT_SOURCE_PENALTY_LIFECYCLE_ATTRIBUTION_V1.

Historically Protocol263. Protocol262 is the first lifecycle-style challenger
in the latest loop that beats PAPER_DEFAULT_PROTOCOL101 on the common
lifecycle-harness test splits, but it still trails its frozen Protocol261 base
stream on recent 2026. This diagnostic explains why before we add another model
change.

It compares:
* Protocol262 lifecycle trades,
* Protocol261 frozen source-penalty router base trades,
* PAPER_DEFAULT_PROTOCOL101 reference trades.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct
import v4.scripts.run_protocol216_full_action_history_vs_protocol101_attribution as p216
from v4.scripts.run_protocol207_lifecycle_context_calibration import PAPER_DEFAULT_TRADES


ROLE_LABEL = "AUDIT_SOURCE_PENALTY_LIFECYCLE_ATTRIBUTION_V1"
HISTORICAL_ID = "Protocol263"
DEFAULT_PROTOCOL262_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_262_source_penalty_slot_lifecycle")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_263_source_penalty_lifecycle_attribution")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
REQUIRED_SPLITS = ("q1_2026", "march_2026", "recent_2026")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol262-dir", type=Path, default=DEFAULT_PROTOCOL262_DIR)
    parser.add_argument("--paper-default-trades", type=Path, default=PAPER_DEFAULT_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reentry-gap-minutes", type=float, default=30.0)
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = normalize_lifecycle_trades(
        pd.read_csv(args.protocol262_dir / "premium_blend_slot_aware_lifecycle_trades.csv"),
        policy="protocol262_lifecycle",
    )
    base = normalize_lifecycle_trades(
        pd.read_csv(args.protocol262_dir / "premium_blend_baseline_serial_trades.csv"),
        policy="protocol261_base",
    )
    paper = normalize_lifecycle_trades(pd.read_csv(args.paper_default_trades), policy="protocol101")
    model = model[model["reported_split"].isin(REQUIRED_SPLITS)].copy()
    base = base[base["reported_split"].isin(REQUIRED_SPLITS)].copy()
    paper = paper[paper["reported_split"].isin(REQUIRED_SPLITS)].copy()

    matched = matched_model_base(model, base)
    base_expanded = expand_base_to_model_combos(base, model)
    model_only, base_only = exclusive_model_base(model, base_expanded)
    combined = pd.concat([model, base, paper], ignore_index=True)
    if args.skip_enrichment:
        enriched = combined.copy()
        for column in ["directional_underlying_move", "path_mfe", "path_mae", "mfe_capture", "entry_bid_live", "entry_ask_live", "exit_bid_live"]:
            if column not in enriched.columns:
                enriched[column] = np.nan
        quote_skips: list[dict[str, Any]] = []
    else:
        enriched, quote_skips = p216.enrich_from_normalized(combined, args.normalized_dir)

    headline = summarize_policy(enriched)
    split_summary = summarize_split(model, base_expanded, paper, matched, model_only, base_only)
    matched_summary = summarize_matched(matched, ["reported_split"])
    matched_by_side = summarize_matched(matched, ["reported_split", "right_model"])
    matched_by_exit = summarize_matched(matched, ["reported_split", "baseline_exit_reason_model", "exit_reason_model"])
    exclusive_summary = summarize_exclusive(model_only, base_only)
    side_summary = p216.summarize_group(enriched, ["reported_split", "policy", "right"])
    time_summary = p216.summarize_group(enriched, ["reported_split", "policy", "time_bucket"])
    move_summary = p216.summarize_directional_moves(enriched)
    churn_summary, churn_chains = p216.summarize_churn(enriched, gap_minutes=float(args.reentry_gap_minutes))
    hold_summary, hold_counterfactuals = p216.summarize_churn_hold_counterfactuals(enriched, churn_chains)
    day_diffs = summarize_day_differences(model, base, paper)
    top_matched_deltas = top_matched(matched)

    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / lifecycle attribution audit",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_SOURCE_PENALTY_ROUTER_SLOT_LIFECYCLE_V1",
        "base_challenger_label": "CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": {
            "protocol262_dir": str(args.protocol262_dir),
            "paper_default_trades": str(args.paper_default_trades),
            "normalized_dir": str(args.normalized_dir),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "reentry_gap_minutes": float(args.reentry_gap_minutes),
        "row_counts": {
            "model_rows": int(len(model)),
            "base_rows": int(len(base)),
            "base_expanded_combo_rows": int(len(base_expanded)),
            "paper_default_rows": int(len(paper)),
            "matched_model_base_rows": int(len(matched)),
            "model_only_rows": int(len(model_only)),
            "base_only_rows": int(len(base_only)),
            "enriched_rows": int(len(enriched)),
            "quote_skip_rows": int(len(quote_skips)),
            "churn_chains": int(len(churn_chains)),
            "hold_counterfactual_rows": int(len(hold_counterfactuals)),
        },
        "headline": headline,
        "split_summary": split_summary,
        "matched_summary": matched_summary,
        "matched_by_side": matched_by_side,
        "matched_by_exit_reason": matched_by_exit,
        "exclusive_summary": exclusive_summary,
        "side_summary": side_summary,
        "time_bucket_summary": time_summary,
        "directional_move_summary": move_summary,
        "churn_summary": churn_summary,
        "churn_hold_counterfactual_summary": hold_summary,
        "day_differences": day_diffs,
        "top_matched_deltas": top_matched_deltas,
        "quote_skip_counts": p216.count_by(quote_skips, "skip_reason"),
        "decision": "",
        "next_experiment": "",
    }
    payload["decision"] = decide(payload)
    payload["next_experiment"] = next_experiment(payload)

    write_csv(args.out_dir / "headline.csv", headline)
    write_csv(args.out_dir / "split_summary.csv", split_summary)
    write_csv(args.out_dir / "matched_summary.csv", matched_summary)
    write_csv(args.out_dir / "matched_by_side.csv", matched_by_side)
    write_csv(args.out_dir / "matched_by_exit_reason.csv", matched_by_exit)
    write_csv(args.out_dir / "exclusive_summary.csv", exclusive_summary)
    write_csv(args.out_dir / "side_summary.csv", side_summary)
    write_csv(args.out_dir / "time_bucket_summary.csv", time_summary)
    write_csv(args.out_dir / "directional_move_summary.csv", move_summary)
    write_csv(args.out_dir / "churn_summary.csv", churn_summary)
    write_csv(args.out_dir / "churn_hold_counterfactual_summary.csv", hold_summary)
    write_csv(args.out_dir / "day_differences.csv", day_diffs)
    write_csv(args.out_dir / "top_matched_deltas.csv", top_matched_deltas)
    matched.to_csv(args.out_dir / "matched_model_base_trades.csv", index=False)
    model_only.to_csv(args.out_dir / "model_only_trades.csv", index=False)
    base_only.to_csv(args.out_dir / "base_only_trades.csv", index=False)
    enriched.to_csv(args.out_dir / "enriched_policy_trades.csv", index=False)
    churn_chains.to_csv(args.out_dir / "churn_chains.csv", index=False)
    hold_counterfactuals.to_csv(args.out_dir / "churn_hold_counterfactuals.csv", index=False)
    pd.DataFrame(quote_skips).to_csv(args.out_dir / "quote_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def normalize_lifecycle_trades(frame: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    out = frame.copy()
    if "candidate_exit_time" in out.columns and "exit_time" not in out.columns:
        out["exit_time"] = out["candidate_exit_time"]
    if "candidate_pnl" in out.columns and "pnl" not in out.columns:
        out["pnl"] = out["candidate_pnl"]
    if "candidate_exit_reason" in out.columns and "exit_reason" not in out.columns:
        out["exit_reason"] = out["candidate_exit_reason"]
    if "seed" not in out.columns:
        if "combo_seed" in out.columns:
            out["seed"] = out["combo_seed"]
        elif "entry_seed" in out.columns:
            out["seed"] = out["entry_seed"]
        else:
            out["seed"] = 0
    if "entry_seed" not in out.columns:
        out["entry_seed"] = out["seed"]
    if "model_seed" not in out.columns:
        out["model_seed"] = np.nan
    if "combo_seed" not in out.columns:
        out["combo_seed"] = out["seed"]
    for column, default in {
        "fold": "unknown",
        "reported_split": "unknown",
        "session": "",
        "decision_time": "",
        "exit_time": "",
        "contract_id": "",
        "right": "",
        "exit_reason": "",
        "baseline_exit_reason": "",
        "strategy": "",
    }.items():
        if column not in out.columns:
            out[column] = default
    out["policy"] = policy
    out["decision_ts"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["exit_ts"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out = out[out["decision_ts"].notna() & out["exit_ts"].notna()].copy()
    out["duration_minutes"] = (out["exit_ts"] - out["decision_ts"]).dt.total_seconds() / 60.0
    for column in [
        "seed",
        "entry_seed",
        "model_seed",
        "combo_seed",
        "pnl",
        "entry_ask",
        "entry_bid",
        "entry_mid",
        "entry_premium",
        "offset",
        "score",
        "threshold",
        "baseline_pnl",
        "predicted_continuation_value",
    ]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["seed"] = out["seed"].fillna(0).astype(int)
    out["entry_seed"] = out["entry_seed"].fillna(out["seed"]).astype(int)
    out["combo_seed"] = out["combo_seed"].fillna(out["seed"]).astype(int)
    out["reported_split"] = out["reported_split"].astype(str)
    out["session"] = out["session"].astype(str)
    out["fold"] = out["fold"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)
    out["right"] = out["right"].astype(str)
    out["exit_reason"] = out["exit_reason"].astype(str)
    out["baseline_exit_reason"] = out["baseline_exit_reason"].astype(str)
    out["time_bucket"] = [p216.time_bucket(ts.to_pydatetime()) for ts in out["decision_ts"]]
    out["moneyness"] = [moneyness(right, offset) for right, offset in zip(out["right"], out["offset"])]
    out["premium_bucket"] = pd.cut(
        out["entry_premium"].fillna(-1.0),
        bins=[-1.0, 0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 1_000_000.0],
        labels=["missing", "0-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", "3000-4000", "4000+"],
    ).astype(str)
    return out.reset_index(drop=True)


def matched_model_base(model: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    keys = ["reported_split", "entry_seed", "session", "decision_time", "contract_id"]
    cols = [
        *keys,
        "model_seed",
        "combo_seed",
        "right",
        "time_bucket",
        "moneyness",
        "premium_bucket",
        "pnl",
        "exit_time",
        "exit_ts",
        "duration_minutes",
        "exit_reason",
        "baseline_exit_reason",
        "baseline_pnl",
        "entry_premium",
        "predicted_continuation_value",
    ]
    model_cols = [column for column in cols if column in model.columns]
    base_cols = [column for column in cols if column in base.columns]
    matched = model[model_cols].merge(base[base_cols], on=keys, how="inner", suffixes=("_model", "_base"))
    if matched.empty:
        return matched
    matched["pnl_delta"] = pd.to_numeric(matched["pnl_model"], errors="coerce") - pd.to_numeric(matched["pnl_base"], errors="coerce")
    matched["exit_delta_minutes"] = (
        pd.to_datetime(matched["exit_ts_model"], utc=True, errors="coerce") - pd.to_datetime(matched["exit_ts_base"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    matched["model_exited_earlier"] = matched["exit_delta_minutes"] < 0.0
    matched["model_exited_later"] = matched["exit_delta_minutes"] > 0.0
    return matched


def expand_base_to_model_combos(base: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["reported_split", "fold", "entry_seed"]
    seed_map = model[[*merge_keys, "model_seed", "combo_seed"]].drop_duplicates()
    expanded = base.merge(seed_map, on=merge_keys, how="inner", suffixes=("", "_modelmap"))
    if "model_seed_modelmap" in expanded.columns:
        expanded["model_seed"] = expanded["model_seed_modelmap"]
        expanded = expanded.drop(columns=["model_seed_modelmap"])
    if "combo_seed_modelmap" in expanded.columns:
        expanded["combo_seed"] = expanded["combo_seed_modelmap"]
        expanded = expanded.drop(columns=["combo_seed_modelmap"])
    return expanded.reset_index(drop=True)


def exclusive_model_base(model: pd.DataFrame, base_expanded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["combo_seed", "reported_split", "session", "decision_time", "contract_id"]
    model_keys = model[key_cols].drop_duplicates()
    base_keys = base_expanded[key_cols].drop_duplicates()
    model_only = model.merge(base_keys.assign(_in_base=True), on=key_cols, how="left")
    model_only = model_only[model_only["_in_base"].isna()].drop(columns=["_in_base"])
    base_only = base_expanded.merge(model_keys.assign(_in_model=True), on=key_cols, how="left")
    base_only = base_only[base_only["_in_model"].isna()].drop(columns=["_in_model"])
    return model_only.reset_index(drop=True), base_only.reset_index(drop=True)


def summarize_policy(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (split, policy), group in frame.groupby(["reported_split", "policy"], sort=True):
        item = metrics(group)
        rows.append({"reported_split": str(split), "policy": str(policy), **item})
    return rows


def summarize_split(
    model: pd.DataFrame,
    base_expanded: pd.DataFrame,
    paper: pd.DataFrame,
    matched: pd.DataFrame,
    model_only: pd.DataFrame,
    base_only: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for split in sorted(set(model["reported_split"]) | set(base_expanded["reported_split"]) | set(paper["reported_split"])):
        mg = model[model["reported_split"].eq(split)]
        bg = base_expanded[base_expanded["reported_split"].eq(split)]
        pg = paper[paper["reported_split"].eq(split)]
        mat = matched[matched["reported_split"].eq(split)] if not matched.empty else matched
        mo = model_only[model_only["reported_split"].eq(split)] if not model_only.empty else model_only
        bo = base_only[base_only["reported_split"].eq(split)] if not base_only.empty else base_only
        rows.append(
            {
                "reported_split": str(split),
                "model_median_pnl": seed_median(mg, "combo_seed", "pnl"),
                "base_median_pnl": seed_median(bg, "combo_seed", "pnl"),
                "paper_default_median_pnl": seed_median(pg, "seed", "pnl"),
                "model_median_trades": seed_median_count(mg, "combo_seed"),
                "base_median_trades": seed_median_count(bg, "combo_seed"),
                "paper_default_median_trades": seed_median_count(pg, "seed"),
                "model_raw_pnl": finite_sum(mg["pnl"]),
                "base_expanded_raw_pnl": finite_sum(bg["pnl"]),
                "paper_default_raw_pnl": finite_sum(pg["pnl"]),
                "model_rows": int(len(mg)),
                "base_expanded_rows": int(len(bg)),
                "paper_default_rows": int(len(pg)),
                "matched_rows": int(len(mat)),
                "matched_pnl_delta": finite_sum(mat["pnl_delta"]) if not mat.empty else 0.0,
                "model_only_rows": int(len(mo)),
                "model_only_pnl": finite_sum(mo["pnl"]) if not mo.empty else 0.0,
                "base_only_rows": int(len(bo)),
                "base_only_pnl": finite_sum(bo["pnl"]) if not bo.empty else 0.0,
                "mean_exit_delta_minutes": finite(mat["exit_delta_minutes"].mean()) if not mat.empty else 0.0,
                "model_earlier_exit_fraction": finite((mat["model_exited_earlier"]).mean()) if not mat.empty else 0.0,
                "model_later_exit_fraction": finite((mat["model_exited_later"]).mean()) if not mat.empty else 0.0,
            }
        )
    return rows


def summarize_matched(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: str(value) for column, value in zip(group_cols, key)}
        row.update(
            {
                "rows": int(len(group)),
                "pnl_delta": finite_sum(group["pnl_delta"]),
                "median_pnl_delta": finite(group["pnl_delta"].median()),
                "model_pnl": finite_sum(group["pnl_model"]),
                "base_pnl": finite_sum(group["pnl_base"]),
                "mean_exit_delta_minutes": finite(group["exit_delta_minutes"].mean()),
                "median_exit_delta_minutes": finite(group["exit_delta_minutes"].median()),
                "model_earlier_exit_fraction": finite(group["model_exited_earlier"].mean()),
                "model_later_exit_fraction": finite(group["model_exited_later"].mean()),
            }
        )
        rows.append(row)
    return rows


def summarize_exclusive(model_only: pd.DataFrame, base_only: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for label, frame in [("model_only", model_only), ("base_only", base_only)]:
        if frame.empty:
            continue
        for (split, right), group in frame.groupby(["reported_split", "right"], sort=True):
            rows.append({"bucket": label, "reported_split": str(split), "right": str(right), **metrics(group)})
    return rows


def summarize_day_differences(model: pd.DataFrame, base: pd.DataFrame, paper: pd.DataFrame) -> list[dict[str, Any]]:
    frames = []
    for label, frame in [("protocol262", model), ("protocol261_base", base), ("protocol101", paper)]:
        tmp = frame.groupby(["reported_split", "session"], sort=True)["pnl"].sum().reset_index()
        tmp["policy"] = label
        frames.append(tmp)
    day = pd.concat(frames, ignore_index=True)
    pivot = day.pivot_table(index=["reported_split", "session"], columns="policy", values="pnl", aggfunc="sum").reset_index().fillna(0.0)
    for column in ["protocol262", "protocol261_base", "protocol101"]:
        if column not in pivot.columns:
            pivot[column] = 0.0
    pivot["delta_vs_base"] = pivot["protocol262"] - pivot["protocol261_base"]
    pivot["delta_vs_protocol101"] = pivot["protocol262"] - pivot["protocol101"]
    best = pivot.sort_values("delta_vs_base", ascending=False).head(15).copy()
    worst = pivot.sort_values("delta_vs_base", ascending=True).head(15).copy()
    best["bucket"] = "protocol262_best_vs_base_days"
    worst["bucket"] = "protocol262_worst_vs_base_days"
    return pd.concat([best, worst], ignore_index=True).to_dict("records")


def top_matched(matched: pd.DataFrame) -> list[dict[str, Any]]:
    if matched.empty:
        return []
    cols = [
        "reported_split",
        "model_seed",
        "entry_seed",
        "session",
        "decision_time",
        "contract_id",
        "right_model",
        "pnl_model",
        "pnl_base",
        "pnl_delta",
        "exit_delta_minutes",
        "exit_reason_model",
        "exit_reason_base",
        "baseline_exit_reason_model",
    ]
    best = matched.sort_values("pnl_delta", ascending=False).head(20).copy()
    worst = matched.sort_values("pnl_delta", ascending=True).head(20).copy()
    best["bucket"] = "best_matched_lifecycle_deltas"
    worst["bucket"] = "worst_matched_lifecycle_deltas"
    out = pd.concat([best, worst], ignore_index=True)
    return out[["bucket", *[c for c in cols if c in out.columns]]].to_dict("records")


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_loss = -float(losses.sum())
    gross_win = float(wins.sum())
    pf = gross_win / gross_loss if gross_loss > 1e-9 else (float("inf") if gross_win > 0.0 else 0.0)
    return {
        "trades": int(len(frame)),
        "pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "win_rate": float((pnl > 0.0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(pf),
        "median_duration_minutes": finite(frame["duration_minutes"].median()) if "duration_minutes" in frame.columns else 0.0,
        "median_entry_premium": finite(frame["entry_premium"].median()) if "entry_premium" in frame.columns else 0.0,
    }


def seed_median(frame: pd.DataFrame, seed_col: str, value_col: str) -> float:
    if frame.empty or seed_col not in frame.columns or value_col not in frame.columns:
        return 0.0
    totals = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0).groupby(frame[seed_col]).sum()
    return finite(totals.median()) if len(totals) else 0.0


def seed_median_count(frame: pd.DataFrame, seed_col: str) -> float:
    if frame.empty or seed_col not in frame.columns:
        return 0.0
    counts = frame.groupby(seed_col).size()
    return finite(counts.median()) if len(counts) else 0.0


def decide(payload: dict[str, Any]) -> str:
    split = {row["reported_split"]: row for row in payload["split_summary"]}
    recent = split.get("recent_2026", {})
    q1 = split.get("q1_2026", {})
    march = split.get("march_2026", {})
    beats_paper = all(
        split.get(item, {}).get("model_median_pnl", 0.0) > split.get(item, {}).get("paper_default_median_pnl", 0.0)
        for item in REQUIRED_SPLITS
    )
    recent_trails_base = float(recent.get("model_median_pnl", 0.0)) < float(recent.get("base_median_pnl", 0.0))
    q1_beats_base = float(q1.get("model_median_pnl", 0.0)) > float(q1.get("base_median_pnl", 0.0))
    march_beats_base = float(march.get("model_median_pnl", 0.0)) > float(march.get("base_median_pnl", 0.0))
    if beats_paper and recent_trails_base and q1_beats_base and march_beats_base:
        return "lifecycle_beats_paper_but_recent_gap_is_base_stream_opportunity_loss"
    if beats_paper:
        return "lifecycle_beats_paper_attribution_requires_runtime_parity"
    return "lifecycle_attribution_rejects_challenger"


def next_experiment(payload: dict[str, Any]) -> str:
    split = {row["reported_split"]: row for row in payload["split_summary"]}
    recent = split.get("recent_2026", {})
    if float(recent.get("base_only_pnl", 0.0)) > abs(float(recent.get("matched_pnl_delta", 0.0))):
        return (
            "Test a recent-preserving lifecycle fallback: keep Protocol262 exits where matched lifecycle delta is strong, "
            "but add a validation-only guard against exits that free the slot into lower-quality replacement trades."
        )
    return (
        "Test a sequence architecture that jointly scores continuation and next-slot replacement value; do not add more "
        "entry stream routers until lifecycle slot attribution is resolved."
    )


def moneyness(right: Any, offset: Any) -> str:
    try:
        value = float(offset)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value):
        return "unknown"
    if abs(value) <= 2.5:
        return "ATM"
    side = str(right)
    if side == "C":
        return "ITM" if value < 0.0 else "OTM"
    if side == "P":
        return "ITM" if value > 0.0 else "OTM"
    return "unknown"


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def finite_sum(values: Any) -> float:
    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Base challenger: {payload['base_challenger_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Split Summary",
        "",
        "| split | model median | base median | Protocol101 median | matched delta | model-only | base-only | exit delta | earlier exits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_pnl'])} | {money(row['base_median_pnl'])} | "
            f"{money(row['paper_default_median_pnl'])} | {money(row['matched_pnl_delta'])} | "
            f"{money(row['model_only_pnl'])} | {money(row['base_only_pnl'])} | "
            f"{row['mean_exit_delta_minutes']:.1f}m | {pct(row['model_earlier_exit_fraction'])} |"
        )
    lines.extend(["", "## Row Counts", ""])
    for key, value in payload["row_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Matched trades: `{path.parent / 'matched_model_base_trades.csv'}`",
            f"- Model-only trades: `{path.parent / 'model_only_trades.csv'}`",
            f"- Base-only trades: `{path.parent / 'base_only_trades.csv'}`",
            f"- Enriched policy trades: `{path.parent / 'enriched_policy_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    f"## {HISTORICAL_ID} - {ROLE_LABEL}",
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
