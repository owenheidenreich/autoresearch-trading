"""AUDIT_2026_05_22_SCALE_IN_OUT_PATH_OPPORTUNITY_V1.

Historically Protocol224. This audit prepares the scale-in / scale-out research
track without hardcoding a trading rule. It uses the account-aware sized trade
stream from Protocol223 and reconstructs each selected contract's post-entry
quote path.

The goal is to measure whether scale-in/out has learnable structure:
* Did the frozen exit leave continuation value?
* Would adding another contract have had positive future value?
* Did the best add opportunity occur while the original position was green or
  while it was losing money, i.e. would it require averaging down?
* What were option delta, gamma/theta, spread, and time-left conditions at those
  moments?

No paid data is downloaded. No broker endpoint is called. This is an audit, not
a paper-default model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.greeks.repair import compute_repaired_greeks


ROLE_LABEL = "AUDIT_2026_05_22_SCALE_IN_OUT_PATH_OPPORTUNITY_V1"
HISTORICAL_ID = "Protocol224"
DEFAULT_SIZED_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_223_account_aware_confidence_sizing/account_aware_sized_trades.csv"
)
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_224_scale_in_out_path_opportunity")
NY = ZoneInfo("America/New_York")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sized-trades", type=Path, default=DEFAULT_SIZED_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--extra-slippage-per-side", type=float, default=0.0)
    parser.add_argument("--forced-flat-time", type=str, default="15:30")
    parser.add_argument("--min-material-pnl", type=float, default=100.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = load_sized_trades(
        args.sized_trades,
        starting_cash=float(args.starting_cash),
        extra_slippage_per_side=float(args.extra_slippage_per_side),
    )
    rows, skips = audit_paths(
        trades,
        normalized_dir=args.normalized_dir,
        forced_flat_time=str(args.forced_flat_time),
        min_material_pnl=float(args.min_material_pnl),
    )
    detail = pd.DataFrame(rows)
    skip_frame = pd.DataFrame(skips)
    if not detail.empty:
        detail.to_csv(args.out_dir / "scale_path_opportunities.csv", index=False)
    else:
        (args.out_dir / "scale_path_opportunities.csv").write_text("")
    if not skip_frame.empty:
        skip_frame.to_csv(args.out_dir / "path_skips.csv", index=False)
    else:
        (args.out_dir / "path_skips.csv").write_text("")
    split_summary = summarize(detail)
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / scale-in scale-out path opportunity audit",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1_WITH_ACCOUNT_AWARE_SIZING",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.sized_trades),
        "normalized_dir": str(args.normalized_dir),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "starting_cash": float(args.starting_cash),
        "extra_slippage_per_side": float(args.extra_slippage_per_side),
        "rows": int(len(detail)),
        "path_skips": int(len(skip_frame)),
        "path_skip_counts": count_by(skip_frame, "skip_reason"),
        "greek_coverage": greek_coverage(detail),
        "decision": decide(split_summary),
        "aggregate": split_summary.to_dict("records"),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "path_opportunities": str(args.out_dir / "scale_path_opportunities.csv"),
            "split_summary": str(args.out_dir / "split_summary.csv"),
            "path_skips": str(args.out_dir / "path_skips.csv"),
        },
        "next_experiment": (
            "If continuation and add opportunities are material and mostly occur while positions are already working, "
            "build a supervised position-management dataset with actions hold/add/reduce/exit. If opportunities mostly "
            "require averaging down into losers, keep scale-in disabled and focus on exit/scale-out only."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload, split_summary)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_sized_trades(path: Path, *, starting_cash: float, extra_slippage_per_side: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["starting_cash"].astype(float).eq(float(starting_cash))
        & frame["extra_slippage_per_side"].astype(float).eq(float(extra_slippage_per_side))
        & (pd.to_numeric(frame["aa_contracts"], errors="coerce") > 0)
    ].copy()
    for column in ["decision_time", "exit_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in [
        "entry_ask",
        "entry_premium",
        "aa_contracts",
        "aa_pnl",
        "aa_confidence",
        "aa_equity_before",
        "aa_premium_frac_realized",
        "score",
        "threshold",
        "offset",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.dropna(subset=["reported_split", "session", "decision_time", "exit_time", "contract_id", "entry_ask"])
    return frame.sort_values(["session", "decision_time", "contract_id"]).reset_index(drop=True)


def audit_paths(
    trades: pd.DataFrame,
    *,
    normalized_dir: Path,
    forced_flat_time: str,
    min_material_pnl: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for session, group in trades.groupby("session", sort=True):
        contracts = set(group["contract_id"].astype(str).unique())
        quotes = load_session_quotes(normalized_dir, str(session), contracts)
        if quotes.empty:
            skips.extend(base_skip(row, "missing_session_or_contract_quotes") for _, row in group.iterrows())
            continue
        by_contract = {
            str(contract_id): part.sort_values("quote_time").reset_index(drop=True)
            for contract_id, part in quotes.groupby("contract_id", sort=False)
        }
        forced_flat = forced_flat_timestamp(str(session), forced_flat_time)
        for _, trade in group.iterrows():
            result, skip = audit_trade_path(trade, by_contract.get(str(trade["contract_id"])), forced_flat, min_material_pnl)
            if skip:
                skips.append(skip)
            else:
                rows.append(result)
    return rows, skips


def load_session_quotes(normalized_dir: Path, session: str, contract_ids: set[str]) -> pd.DataFrame:
    path = find_normalized_path(normalized_dir, session)
    if path is None or not contract_ids:
        return pd.DataFrame()
    columns = [
        "quote_time",
        "contract_id",
        "bid",
        "ask",
        "mid",
        "bid_size",
        "ask_size",
        "quote_gap_seconds",
        "underlying_price",
        "iv",
        "delta",
        "gamma",
        "theta",
    ]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        available = pd.read_parquet(path)
        for column in columns:
            if column not in available.columns:
                available[column] = np.nan
        frame = available[columns].copy()
    frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True, errors="coerce")
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame = frame[frame["contract_id"].isin(contract_ids)].copy()
    for column in columns:
        if column not in {"quote_time", "contract_id"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["mid"] = frame["mid"].where(frame["mid"].notna(), (frame["bid"] + frame["ask"]) / 2.0)
    return frame[
        frame["quote_time"].notna()
        & frame["bid"].notna()
        & frame["ask"].notna()
        & (frame["bid"] >= 0.0)
        & (frame["ask"] > 0.0)
        & (frame["ask"] >= frame["bid"])
    ].copy()


def find_normalized_path(normalized_dir: Path, session: str) -> Path | None:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    if preferred:
        return preferred[0]
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return fallback[0] if fallback else None


def forced_flat_timestamp(session: str, forced_flat_time: str) -> pd.Timestamp:
    hour, minute = [int(part) for part in forced_flat_time.split(":", 1)]
    return pd.Timestamp(session).replace(hour=hour, minute=minute, tzinfo=NY).tz_convert("UTC")


def audit_trade_path(
    trade: pd.Series,
    quotes: pd.DataFrame | None,
    forced_flat: pd.Timestamp,
    min_material_pnl: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if quotes is None or quotes.empty:
        return None, base_skip(trade, "missing_contract_quotes")
    decision_ts = pd.Timestamp(trade["decision_time"])
    baseline_exit_ts = min(pd.Timestamp(trade["exit_time"]), forced_flat)
    entry_ask = finite_float(trade.get("entry_ask"), math.nan)
    qty = int(finite_float(trade.get("aa_contracts"), 0.0))
    if not math.isfinite(entry_ask) or entry_ask <= 0.0 or qty <= 0:
        return None, base_skip(trade, "invalid_entry_or_quantity")
    path = quotes[(quotes["quote_time"] >= decision_ts) & (quotes["quote_time"] <= forced_flat)].copy()
    if path.empty:
        return None, base_skip(trade, "missing_post_entry_path")
    path = path.sort_values("quote_time").reset_index(drop=True)
    bid = pd.to_numeric(path["bid"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(path["ask"], errors="coerce").to_numpy(dtype=float)
    pnl_path = (bid - entry_ask) * CONTRACT_MULTIPLIER
    valid = np.isfinite(pnl_path) & np.isfinite(ask) & np.isfinite(bid)
    if not valid.any():
        return None, base_skip(trade, "invalid_path_prices")
    path = path[valid].reset_index(drop=True)
    bid = bid[valid]
    ask = ask[valid]
    pnl_path = pnl_path[valid]
    quote_times = pd.to_datetime(path["quote_time"], utc=True)
    before_exit = quote_times <= baseline_exit_ts
    after_exit = quote_times >= baseline_exit_ts
    if not before_exit.any():
        before_exit = np.ones(len(path), dtype=bool)
    baseline_unit_pnl = finite_float(trade.get("pnl"), finite_float(trade.get("raw_candidate_pnl"), 0.0))
    if baseline_unit_pnl == 0.0 and math.isfinite(finite_float(trade.get("aa_pnl"), math.nan)):
        baseline_unit_pnl = finite_float(trade.get("aa_pnl"), 0.0) / max(qty, 1)
    best_pre_idx = int(np.nanargmax(np.where(before_exit, pnl_path, -np.inf)))
    best_full_idx = int(np.nanargmax(pnl_path))
    post_best_idx = int(np.nanargmax(np.where(after_exit, pnl_path, -np.inf))) if after_exit.any() else best_full_idx
    future_max_bid = np.maximum.accumulate(bid[::-1])[::-1]
    add_unit_pnl = (future_max_bid - ask) * CONTRACT_MULTIPLIER
    add_mask = quote_times > decision_ts
    best_add_idx = int(np.nanargmax(np.where(add_mask, add_unit_pnl, -np.inf))) if add_mask.any() else 0
    current_unit_pnl_at_add = float(pnl_path[best_add_idx])
    best_add_unit_pnl = float(add_unit_pnl[best_add_idx])
    add_time = pd.Timestamp(quote_times.iloc[best_add_idx])
    entry_greeks = repaired_greek_metrics(path.iloc[0], str(trade.get("contract_id")), str(trade.get("right")), forced_flat)
    best_add_greeks = repaired_greek_metrics(path.iloc[best_add_idx], str(trade.get("contract_id")), str(trade.get("right")), forced_flat)
    best_pre_greeks = repaired_greek_metrics(path.iloc[best_pre_idx], str(trade.get("contract_id")), str(trade.get("right")), forced_flat)
    best_full_greeks = repaired_greek_metrics(path.iloc[best_full_idx], str(trade.get("contract_id")), str(trade.get("right")), forced_flat)
    entry_delta = entry_greeks["delta"]
    best_add_delta = best_add_greeks["delta"]
    best_pre_delta = best_pre_greeks["delta"]
    best_full_delta = best_full_greeks["delta"]
    result = {
        "reported_split": str(trade.get("reported_split")),
        "session": str(trade.get("session")),
        "decision_time": decision_ts.isoformat(),
        "baseline_exit_time": baseline_exit_ts.isoformat(),
        "contract_id": str(trade.get("contract_id")),
        "right": str(trade.get("right")),
        "offset": finite_float(trade.get("offset"), math.nan),
        "quantity": qty,
        "entry_ask": entry_ask,
        "entry_premium": finite_float(trade.get("entry_premium"), entry_ask * CONTRACT_MULTIPLIER),
        "entry_delta": entry_delta,
        "entry_abs_delta": abs(entry_delta) if math.isfinite(entry_delta) else math.nan,
        "entry_gamma_theta_ratio": entry_greeks["gamma_theta_ratio"],
        "entry_iv": entry_greeks["iv"],
        "entry_greek_source": entry_greeks["source"],
        "entry_spread": finite_float(path.iloc[0].get("ask"), math.nan) - finite_float(path.iloc[0].get("bid"), math.nan),
        "entry_ask_size": finite_float(path.iloc[0].get("ask_size"), math.nan),
        "score": finite_float(trade.get("score"), math.nan),
        "threshold": finite_float(trade.get("threshold"), math.nan),
        "confidence": finite_float(trade.get("aa_confidence"), math.nan),
        "account_equity_before": finite_float(trade.get("aa_equity_before"), math.nan),
        "premium_frac_realized": finite_float(trade.get("aa_premium_frac_realized"), math.nan),
        "baseline_unit_pnl": baseline_unit_pnl,
        "baseline_position_pnl": baseline_unit_pnl * qty,
        "best_pre_exit_unit_pnl": float(pnl_path[best_pre_idx]),
        "best_pre_exit_position_pnl": float(pnl_path[best_pre_idx] * qty),
        "best_pre_exit_delta_vs_baseline": float(pnl_path[best_pre_idx] - baseline_unit_pnl),
        "best_pre_exit_time": pd.Timestamp(quote_times.iloc[best_pre_idx]).isoformat(),
        "best_pre_exit_delta": best_pre_delta,
        "best_pre_exit_abs_delta": abs(best_pre_delta) if math.isfinite(best_pre_delta) else math.nan,
        "best_full_unit_pnl": float(pnl_path[best_full_idx]),
        "best_full_position_pnl": float(pnl_path[best_full_idx] * qty),
        "best_full_delta_vs_baseline": float(pnl_path[best_full_idx] - baseline_unit_pnl),
        "best_full_time": pd.Timestamp(quote_times.iloc[best_full_idx]).isoformat(),
        "best_full_delta": best_full_delta,
        "best_full_abs_delta": abs(best_full_delta) if math.isfinite(best_full_delta) else math.nan,
        "post_exit_best_unit_pnl": float(pnl_path[post_best_idx]),
        "post_exit_best_delta_vs_baseline": float(pnl_path[post_best_idx] - baseline_unit_pnl),
        "post_exit_best_time": pd.Timestamp(quote_times.iloc[post_best_idx]).isoformat(),
        "material_continuation_after_exit": bool((pnl_path[post_best_idx] - baseline_unit_pnl) >= min_material_pnl),
        "best_add_time": add_time.isoformat(),
        "best_add_minutes_after_entry": float((add_time - decision_ts).total_seconds() / 60.0),
        "best_add_unit_pnl_to_future_best": best_add_unit_pnl,
        "best_add_position_pnl_if_one_more": best_add_unit_pnl,
        "best_add_current_unit_pnl": current_unit_pnl_at_add,
        "best_add_was_average_down": bool(current_unit_pnl_at_add < 0.0),
        "best_add_was_adding_to_winner": bool(current_unit_pnl_at_add > 0.0),
        "best_add_delta": best_add_delta,
        "best_add_abs_delta": abs(best_add_delta) if math.isfinite(best_add_delta) else math.nan,
        "best_add_gamma_theta_ratio": best_add_greeks["gamma_theta_ratio"],
        "best_add_iv": best_add_greeks["iv"],
        "best_add_greek_source": best_add_greeks["source"],
        "best_add_spread": finite_float(path.iloc[best_add_idx].get("ask"), math.nan) - finite_float(path.iloc[best_add_idx].get("bid"), math.nan),
        "best_add_ask_size": finite_float(path.iloc[best_add_idx].get("ask_size"), math.nan),
        "path_points": int(len(path)),
        "path_mfe_unit_pnl": float(np.nanmax(pnl_path)),
        "path_mae_unit_pnl": float(np.nanmin(pnl_path)),
    }
    return result, None


def gamma_theta(row: pd.Series) -> float:
    gamma = abs(finite_float(row.get("gamma"), math.nan))
    theta = abs(finite_float(row.get("theta"), math.nan))
    if not math.isfinite(gamma) or not math.isfinite(theta) or theta <= 1e-12:
        return math.nan
    return gamma / theta


def repaired_greek_metrics(row: pd.Series, contract_id: str, right: str, forced_flat: pd.Timestamp) -> dict[str, Any]:
    delta = finite_float(row.get("delta"), math.nan)
    gamma = finite_float(row.get("gamma"), math.nan)
    theta = finite_float(row.get("theta"), math.nan)
    iv = finite_float(row.get("iv"), math.nan)
    source = "normalized"
    if not all(math.isfinite(value) for value in (delta, gamma, theta, iv)):
        estimate = compute_repaired_greeks(
            S=row.get("underlying_price"),
            K=parse_strike(contract_id),
            T=time_to_expiry_years(row.get("quote_time"), forced_flat),
            is_call=str(right).upper() == "C",
            mid=row.get("mid"),
            ask=row.get("ask"),
            bid=row.get("bid"),
        )
        if estimate is not None:
            delta = estimate.delta
            gamma = estimate.gamma
            theta = estimate.theta_per_day
            iv = estimate.iv
            source = estimate.source
        else:
            source = "unrepairable"
    ratio = abs(gamma) / abs(theta) if math.isfinite(gamma) and math.isfinite(theta) and abs(theta) > 1e-12 else math.nan
    return {
        "delta": float(delta) if math.isfinite(delta) else math.nan,
        "gamma": float(gamma) if math.isfinite(gamma) else math.nan,
        "theta": float(theta) if math.isfinite(theta) else math.nan,
        "iv": float(iv) if math.isfinite(iv) else math.nan,
        "gamma_theta_ratio": float(ratio) if math.isfinite(ratio) else math.nan,
        "source": source,
    }


def parse_strike(contract_id: str) -> float:
    try:
        return float(str(contract_id).split("-")[-2])
    except (IndexError, ValueError):
        return math.nan


def time_to_expiry_years(quote_time: Any, forced_flat: pd.Timestamp) -> float:
    quote_ts = pd.Timestamp(quote_time)
    if quote_ts.tzinfo is None:
        quote_ts = quote_ts.tz_localize("UTC")
    seconds = max((forced_flat - quote_ts).total_seconds(), 60.0)
    return seconds / (365.0 * 24.0 * 60.0 * 60.0)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for split, group in detail.groupby("reported_split", sort=True):
        material = group[group["material_continuation_after_exit"]]
        positive_add = group[group["best_add_unit_pnl_to_future_best"] > 0.0]
        add_winners = positive_add[positive_add["best_add_was_adding_to_winner"]]
        add_down = positive_add[positive_add["best_add_was_average_down"]]
        rows.append(
            {
                "reported_split": split,
                "trades": int(len(group)),
                "quantity_sum": int(group["quantity"].sum()),
                "baseline_position_pnl": float(group["baseline_position_pnl"].sum()),
                "best_pre_exit_position_pnl": float(group["best_pre_exit_position_pnl"].sum()),
                "best_full_position_pnl": float(group["best_full_position_pnl"].sum()),
                "best_pre_exit_delta_total": float((group["best_pre_exit_delta_vs_baseline"] * group["quantity"]).sum()),
                "best_full_delta_total": float((group["best_full_delta_vs_baseline"] * group["quantity"]).sum()),
                "post_exit_continuation_total": float((group["post_exit_best_delta_vs_baseline"] * group["quantity"]).sum()),
                "material_continuation_trades": int(len(material)),
                "material_continuation_fraction": float(len(material) / len(group)) if len(group) else 0.0,
                "positive_add_opportunity_trades": int(len(positive_add)),
                "positive_add_fraction": float(len(positive_add) / len(group)) if len(group) else 0.0,
                "positive_add_total_one_contract_pnl": float(positive_add["best_add_unit_pnl_to_future_best"].sum()),
                "positive_add_winner_fraction": float(len(add_winners) / len(positive_add)) if len(positive_add) else 0.0,
                "positive_add_average_down_fraction": float(len(add_down) / len(positive_add)) if len(positive_add) else 0.0,
                "median_best_add_minutes": float(positive_add["best_add_minutes_after_entry"].median()) if len(positive_add) else math.nan,
                "median_entry_abs_delta": float(group["entry_abs_delta"].median()),
                "median_best_add_abs_delta": float(positive_add["best_add_abs_delta"].median()) if len(positive_add) else math.nan,
                "median_best_add_spread": float(positive_add["best_add_spread"].median()) if len(positive_add) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "audit_blocked_no_path_rows"
    if (summary["positive_add_average_down_fraction"] > 0.50).any():
        return "scale_in_requires_caution_many_best_adds_are_average_down"
    if (summary["positive_add_fraction"] > 0.25).all() and (summary["positive_add_winner_fraction"] > 0.50).all():
        return "scale_in_out_opportunity_present_build_supervised_position_dataset"
    if (summary["material_continuation_fraction"] > 0.25).any():
        return "scale_out_or_hold_opportunity_present_but_scale_in_unclear"
    return "scale_in_out_opportunity_weak_on_current_trade_stream"


def count_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def greek_coverage(detail: pd.DataFrame) -> dict[str, Any]:
    if detail.empty:
        return {}
    out: dict[str, Any] = {}
    for column in ["entry_abs_delta", "best_add_abs_delta", "entry_gamma_theta_ratio", "best_add_gamma_theta_ratio"]:
        out[f"{column}_coverage"] = float(detail[column].notna().mean()) if column in detail.columns else 0.0
        out[f"{column}_median"] = float(detail[column].median()) if column in detail.columns and detail[column].notna().any() else None
    for column in ["entry_greek_source", "best_add_greek_source"]:
        out[f"{column}_counts"] = count_by(detail, column)
    return out


def base_skip(row: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "reported_split": str(row.get("reported_split", "")),
        "session": str(row.get("session", "")),
        "decision_time": str(row.get("decision_time", "")),
        "contract_id": str(row.get("contract_id", "")),
        "skip_reason": reason,
    }


def write_report(path: Path, payload: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Split Summary",
        "",
        "| split | trades | qty | baseline pnl | best pre-exit | best full path | pre-exit delta | full delta | post-exit continuation | material continuation | add opps | add winner % | add average-down % | median add min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['reported_split']} | {row['trades']} | {row['quantity_sum']} | {money(row['baseline_position_pnl'])} | "
            f"{money(row['best_pre_exit_position_pnl'])} | {money(row['best_full_position_pnl'])} | "
            f"{money(row['best_pre_exit_delta_total'])} | {money(row['best_full_delta_total'])} | "
            f"{money(row['post_exit_continuation_total'])} | {pct(row['material_continuation_fraction'])} | "
            f"{pct(row['positive_add_fraction'])} | {pct(row['positive_add_winner_fraction'])} | "
            f"{pct(row['positive_add_average_down_fraction'])} | {finite_float(row['median_best_add_minutes'], 0.0):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Greek Coverage",
            "",
            "```json",
            json.dumps(payload.get("greek_coverage", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Path opportunities: `{payload['outputs']['path_opportunities']}`",
            f"- Split summary: `{payload['outputs']['split_summary']}`",
            f"- Path skips: `{payload['outputs']['path_skips']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def finite_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def money(value: Any) -> str:
    number = finite_float(value, 0.0)
    sign = "-" if number < 0.0 else ""
    return f"{sign}${abs(number):,.0f}"


def pct(value: Any) -> str:
    return f"{finite_float(value, 0.0) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
