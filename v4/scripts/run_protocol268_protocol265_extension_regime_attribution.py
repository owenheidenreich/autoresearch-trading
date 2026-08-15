"""AUDIT_PROTOCOL265_EXTENSION_REGIME_ATTRIBUTION_V1.

Explain where Protocol265's baseline-anchored extensions help or hurt. This is
an attribution audit, not a new model and not a paper-default change.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money
from v4.scripts.run_protocol207_lifecycle_context_calibration import PAPER_DEFAULT_TRADES
from v4.scripts.run_protocol251_premium_blend_slot_aware_lifecycle import safe_find_normalized_path


ROLE_LABEL = "AUDIT_PROTOCOL265_EXTENSION_REGIME_ATTRIBUTION_V1"
HISTORICAL_ID = "Protocol268"
DEFAULT_PROTOCOL265_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation")
DEFAULT_PROTOCOL261_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_261_router_source_penalty_calibration/model_trades.csv")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_268_protocol265_extension_regime_attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol265-dir", type=Path, default=DEFAULT_PROTOCOL265_DIR)
    parser.add_argument("--protocol261-trades", type=Path, default=DEFAULT_PROTOCOL261_TRADES)
    parser.add_argument("--paper-default-trades", type=Path, default=PAPER_DEFAULT_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = normalize(pd.read_csv(args.protocol265_dir / "source_penalty_baseline_anchored_continuation_trades.csv"), policy="protocol265")
    base = normalize(pd.read_csv(args.protocol265_dir / "source_penalty_baseline_serial_trades.csv"), policy="protocol261_base")
    paper = normalize(pd.read_csv(args.paper_default_trades), policy="protocol101")
    model = add_extension_columns(model)
    model = enrich_entry_context(model, args.normalized_dir)
    base = enrich_entry_context(base, args.normalized_dir)
    paper = enrich_entry_context(paper, args.normalized_dir)
    model = add_regime_buckets(model)
    base = add_regime_buckets(base)
    paper = add_regime_buckets(paper)

    extension_rows = model[model["extended_beyond_baseline"].astype(bool)].copy()
    non_extension_rows = model[~model["extended_beyond_baseline"].astype(bool)].copy()
    by_dimensions = {}
    dimensions = [
        "reported_split",
        "right",
        "time_bucket",
        "moneyness_bucket",
        "premium_bucket",
        "spread_bucket",
        "iv_bucket",
        "delta_bucket",
        "gamma_bucket",
        "theta_burden_bucket",
        "trend_bucket",
        "volatility_bucket",
        "baseline_exit_reason",
        "exit_reason",
        "churn_chain_bucket",
    ]
    for dim in dimensions:
        by_dimensions[dim] = summarize_groups(model, [dim])
    split_dimension_rows = []
    for dim in dimensions[1:]:
        split_dimension_rows.extend(summarize_groups(model, ["reported_split", dim]))
    split_dimension_frame = pd.DataFrame(split_dimension_rows)
    extension_summary = summarize_groups(model, ["reported_split", "extended_beyond_baseline"])
    policy_summary = summarize_policy(model, base, paper)
    top_bad = top_groups(split_dimension_frame, ascending=True)
    top_good = top_groups(split_dimension_frame, ascending=False)
    concentration = concentration_summary(model)
    decision = decide(extension_summary, top_bad, top_good)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / extension-regime attribution audit",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_SOURCE_PENALTY_BASELINE_ANCHORED_CONTINUATION_V1",
        "base_challenger_label": "CHALLENGER_ROUTER_SOURCE_PENALTY_CALIBRATED_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": {
            "protocol265_dir": str(args.protocol265_dir),
            "protocol261_trades": str(args.protocol261_trades),
            "paper_default_trades": str(args.paper_default_trades),
            "normalized_dir": str(args.normalized_dir),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "row_counts": {
            "protocol265_rows": int(len(model)),
            "extended_rows": int(len(extension_rows)),
            "non_extended_rows": int(len(non_extension_rows)),
            "protocol261_base_rows": int(len(base)),
            "paper_default_rows": int(len(paper)),
        },
        "policy_summary": policy_summary,
        "extension_summary": extension_summary,
        "top_bad_extension_regimes": top_bad,
        "top_good_extension_regimes": top_good,
        "concentration": concentration,
        "decision": decision,
        "next_experiment": next_experiment(decision),
    }
    model.to_csv(args.out_dir / "protocol265_enriched_extension_rows.csv", index=False)
    pd.DataFrame(extension_summary).to_csv(args.out_dir / "extension_summary.csv", index=False)
    split_dimension_frame.to_csv(args.out_dir / "split_dimension_extension_summary.csv", index=False)
    pd.DataFrame(top_bad).to_csv(args.out_dir / "top_bad_extension_regimes.csv", index=False)
    pd.DataFrame(top_good).to_csv(args.out_dir / "top_good_extension_regimes.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def normalize(frame: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    out = frame.copy()
    if "candidate_pnl" in out.columns and "pnl" not in out.columns:
        out["pnl"] = out["candidate_pnl"]
    if "candidate_exit_time" in out.columns and "exit_time" not in out.columns:
        out["exit_time"] = out["candidate_exit_time"]
    if "candidate_exit_reason" in out.columns and "exit_reason" not in out.columns:
        out["exit_reason"] = out["candidate_exit_reason"]
    if "seed" not in out.columns:
        out["seed"] = out.get("combo_seed", out.get("entry_seed", 0))
    if "combo_seed" not in out.columns:
        out["combo_seed"] = out["seed"]
    for column in ["reported_split", "session", "decision_time", "exit_time", "contract_id", "right", "exit_reason"]:
        if column not in out.columns:
            out[column] = ""
    for column in ["pnl", "entry_ask", "entry_premium", "offset", "baseline_pnl"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["policy"] = policy
    out["decision_ts"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["exit_ts"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out["duration_minutes"] = (out["exit_ts"] - out["decision_ts"]).dt.total_seconds() / 60.0
    return out[out["decision_ts"].notna()].copy()


def add_extension_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "extended_beyond_baseline" not in out.columns:
        out["extended_beyond_baseline"] = False
    out["extension_delta_vs_baseline"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0) - pd.to_numeric(out["baseline_pnl"], errors="coerce").fillna(0.0)
    out["extra_steps"] = pd.to_numeric(out.get("exit_step"), errors="coerce").fillna(0) - pd.to_numeric(out.get("baseline_anchor_idx"), errors="coerce").fillna(0)
    return out


def enrich_entry_context(frame: pd.DataFrame, normalized_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    for column in ["entry_bid_live", "entry_ask_live", "entry_mid_live", "spread_live", "spread_frac_live", "iv", "delta", "gamma", "theta", "underlying_price", "quote_gap_seconds"]:
        out[column] = np.nan
    for session, idx in out.groupby("session").groups.items():
        path = safe_find_normalized_path(normalized_dir, str(session))
        if path is None:
            continue
        contracts = set(out.loc[idx, "contract_id"].astype(str))
        columns = ["quote_time", "contract_id", "bid", "ask", "mid", "iv", "delta", "gamma", "theta", "underlying_price", "quote_gap_seconds"]
        try:
            quotes = pd.read_parquet(path, columns=columns)
        except Exception:
            continue
        quotes["quote_time"] = pd.to_datetime(quotes["quote_time"], utc=True, errors="coerce")
        quotes = quotes[quotes["contract_id"].astype(str).isin(contracts)].copy()
        if quotes.empty:
            continue
        by_contract = {str(cid): group.sort_values("quote_time") for cid, group in quotes.groupby("contract_id", sort=False)}
        for row_idx in idx:
            contract = str(out.at[row_idx, "contract_id"])
            part = by_contract.get(contract)
            if part is None or part.empty:
                continue
            ts = pd.Timestamp(out.at[row_idx, "decision_ts"])
            loc = part["quote_time"].searchsorted(ts, side="left")
            if loc >= len(part):
                loc = len(part) - 1
            quote = part.iloc[int(loc)]
            bid = finite(quote.get("bid"))
            ask = finite(quote.get("ask"))
            mid = finite(quote.get("mid"), (bid + ask) / 2.0 if math.isfinite(bid) and math.isfinite(ask) else math.nan)
            out.at[row_idx, "entry_bid_live"] = bid
            out.at[row_idx, "entry_ask_live"] = ask
            out.at[row_idx, "entry_mid_live"] = mid
            out.at[row_idx, "spread_live"] = ask - bid if math.isfinite(ask) and math.isfinite(bid) else np.nan
            out.at[row_idx, "spread_frac_live"] = (ask - bid) / mid if math.isfinite(mid) and abs(mid) > 1e-9 else np.nan
            for column in ["iv", "delta", "gamma", "theta", "underlying_price", "quote_gap_seconds"]:
                out.at[row_idx, column] = finite(quote.get(column))
    return out


def add_regime_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    local = out["decision_ts"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    out["time_bucket"] = np.select(
        [minute < 10 * 60, minute < 11 * 60 + 30, minute < 13 * 60 + 30],
        ["first_30", "post_open_morning", "midday"],
        default="late_afternoon",
    )
    out["moneyness_bucket"] = pd.cut(pd.to_numeric(out["offset"], errors="coerce").abs(), [-1, 5, 20, 50, 1e9], labels=["atm", "near", "wing", "outside"])
    out["premium_bucket"] = pd.cut(pd.to_numeric(out["entry_premium"], errors="coerce"), [-1, 750, 1500, 3000, 5000, 1e9], labels=["<=750", "750-1500", "1500-3000", "3000-5000", ">5000"])
    out["spread_bucket"] = pd.cut(pd.to_numeric(out["spread_frac_live"], errors="coerce"), [-1, 0.01, 0.03, 0.07, 1e9], labels=["tight", "normal", "wide", "very_wide"])
    out["iv_bucket"] = safe_qcut(pd.to_numeric(out["iv"], errors="coerce"), 4, "iv")
    out["delta_bucket"] = pd.cut(pd.to_numeric(out["delta"], errors="coerce").abs(), [-1, 0.25, 0.45, 0.65, 1.0], labels=["low_delta", "mid_delta", "high_delta", "very_high_delta"])
    out["gamma_bucket"] = safe_qcut(pd.to_numeric(out["gamma"], errors="coerce"), 4, "gamma")
    theta_burden = pd.to_numeric(out["theta"], errors="coerce").abs() / pd.to_numeric(out["entry_mid_live"], errors="coerce").abs().replace(0, np.nan)
    out["theta_burden_bucket"] = safe_qcut(theta_burden, 4, "theta")
    underlying = pd.to_numeric(out["underlying_price"], errors="coerce")
    out["trend_bucket"] = np.where(underlying.groupby(out["session"]).diff().fillna(0.0) >= 0.0, "uptick_or_flat", "downtick")
    out["volatility_bucket"] = safe_qcut(underlying.groupby(out["session"]).diff().abs().fillna(0.0), 4, "rv")
    out["churn_chain_bucket"] = churn_bucket(out)
    return out


def safe_qcut(values: pd.Series, bins: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 2 or numeric.nunique(dropna=True) < 2:
        return pd.Series("missing_or_constant", index=values.index, dtype=object)
    labels = [f"{prefix}_q{idx}" for idx in range(1, bins + 1)]
    try:
        cut = pd.qcut(numeric, bins, labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series("missing_or_constant", index=values.index, dtype=object)
    return cut.astype(object).where(cut.notna(), "missing")


def churn_bucket(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series("no_recent_same_side", index=frame.index, dtype=object)
    for _, group in frame.sort_values("decision_ts").groupby(["combo_seed", "session", "right"], sort=False):
        prev_exit = None
        for idx, row in group.iterrows():
            if prev_exit is not None:
                gap = (row["decision_ts"] - prev_exit).total_seconds() / 60.0
                if 0 <= gap <= 30:
                    out.at[idx] = "same_side_reentry_30m"
            prev_exit = row["exit_ts"]
    return out


def summarize_groups(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        delta = pd.to_numeric(group["extension_delta_vs_baseline"], errors="coerce").fillna(0.0)
        item = {column: str(value) for column, value in zip(group_cols, keys)}
        item.update(
            {
                "rows": int(len(group)),
                "extended_rows": int(group["extended_beyond_baseline"].astype(bool).sum()),
                "pnl": float(pnl.sum()),
                "extension_delta": float(delta.sum()),
                "extension_delta_per_row": float(delta.mean()) if len(delta) else 0.0,
                "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
            }
        )
        rows.append(item)
    return rows


def summarize_policy(model: pd.DataFrame, base: pd.DataFrame, paper: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for name, frame in [("protocol265", model), ("protocol261_base", base), ("protocol101", paper)]:
        for split, group in frame.groupby("reported_split", sort=True):
            pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
            rows.append({"policy": name, "reported_split": str(split), "rows": int(len(group)), "pnl": float(pnl.sum()), "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0})
    return rows


def top_groups(frame: pd.DataFrame, *, ascending: bool) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.sort_values("extension_delta", ascending=ascending).head(20).to_dict("records")


def concentration_summary(frame: pd.DataFrame) -> dict[str, Any]:
    delta = pd.to_numeric(frame["extension_delta_vs_baseline"], errors="coerce").fillna(0.0)
    abs_total = float(delta.abs().sum())
    if abs_total <= 0:
        return {"abs_extension_delta": 0.0, "top_10_abs_share": 0.0, "top_day_abs_share": 0.0}
    top10 = float(delta.abs().nlargest(10).sum()) / abs_total
    by_day = delta.abs().groupby(frame["session"]).sum()
    return {"abs_extension_delta": abs_total, "top_10_abs_share": top10, "top_day_abs_share": float(by_day.max()) / abs_total if len(by_day) else 0.0}


def decide(extension_summary: list[dict[str, Any]], top_bad: list[dict[str, Any]], top_good: list[dict[str, Any]]) -> str:
    by_split = {str(row.get("reported_split")): row for row in extension_summary if str(row.get("extended_beyond_baseline")) == "True"}
    recent = float(by_split.get("recent_2026", {}).get("extension_delta", 0.0))
    q1 = float(by_split.get("q1_2026", {}).get("extension_delta", 0.0))
    march = float(by_split.get("march_2026", {}).get("extension_delta", 0.0))
    has_cluster = any(abs(float(row.get("extension_delta", 0.0))) > 10_000 and int(row.get("rows", 0)) >= 10 for row in [*top_bad, *top_good])
    if recent > 0 and (q1 < 0 or march < 0) and has_cluster:
        return "extension_value_is_regime_dependent_and_learnable"
    if recent > 0 and q1 >= 0 and march >= 0:
        return "extension_value_broadly_positive"
    return "extension_value_mixed_or_not_clustered"


def next_experiment(decision: str) -> str:
    if decision == "extension_value_is_regime_dependent_and_learnable":
        return "Use attribution clusters to build action-advantage hold/exit labels; do not hardcode extension rules."
    if decision == "extension_value_broadly_positive":
        return "Proceed to Protocol265 no-order runtime parity before any replacement discussion."
    return "Do not build another extension knob; prioritize full-surface action-advantage labels."


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate: `{payload['candidate_label']}`",
        f"Paper default baseline: `{payload['paper_default_label']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Extension Summary",
        "",
        "| split | extended | rows | delta | delta/row | pnl |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["extension_summary"]:
        if row.get("reported_split") in {"q1_2026", "march_2026", "recent_2026"}:
            lines.append(
                f"| {row.get('reported_split')} | {row.get('extended_beyond_baseline')} | {row['rows']} | "
                f"{money(row['extension_delta'])} | {money(row['extension_delta_per_row'])} | {money(row['pnl'])} |"
            )
    lines.extend(["", "## Top Bad Extension Regimes", "", "| split/group | rows | delta |", "|---|---:|---:|"])
    for row in payload["top_bad_extension_regimes"][:10]:
        group = ", ".join(f"{k}={v}" for k, v in row.items() if k not in {"rows", "extended_rows", "pnl", "extension_delta", "extension_delta_per_row", "win_rate"})
        lines.append(f"| {group} | {row['rows']} | {money(row['extension_delta'])} |")
    lines.extend(["", "## Top Good Extension Regimes", "", "| split/group | rows | delta |", "|---|---:|---:|"])
    for row in payload["top_good_extension_regimes"][:10]:
        group = ", ".join(f"{k}={v}" for k, v in row.items() if k not in {"rows", "extended_rows", "pnl", "extension_delta", "extension_delta_per_row", "win_rate"})
        lines.append(f"| {group} | {row['rows']} | {money(row['extension_delta'])} |")
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    if marker in ledger.read_text():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    f"- Candidate: `{payload['candidate_label']}`",
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
