"""AUDIT_CHALLENGER_FULL_ACTION_HISTORY_VS_PROTOCOL101_V1.

Historically Protocol216. This is an attribution audit, not a model change.

It compares CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1 against
PAPER_DEFAULT_PROTOCOL101 under the same one-account serial replay framing:
side exposure, time buckets, exact trade overlap, same-side churn/re-entry,
and whether each policy captured larger directional moves.

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

from v4.model.environment_diagnostics import time_bucket


ROLE_LABEL = "AUDIT_CHALLENGER_FULL_ACTION_HISTORY_VS_PROTOCOL101_V1"
HISTORICAL_ID = "Protocol216"
DEFAULT_CHALLENGER_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_215_full_action_surface_edge_history_5seed_confirmation/model_trades_5seed.csv"
)
DEFAULT_PROTOCOL101_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/serial_policy_trades.json")
DEFAULT_RECENT_PROTOCOL101_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/serial_lifecycle_trades.csv"
)
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_216_full_action_history_vs_protocol101_attribution")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger-trades", type=Path, default=DEFAULT_CHALLENGER_TRADES)
    parser.add_argument("--protocol101-trades", type=Path, default=DEFAULT_PROTOCOL101_TRADES)
    parser.add_argument("--recent-protocol101-trades", type=Path, default=DEFAULT_RECENT_PROTOCOL101_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reentry-gap-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    challenger = load_challenger(args.challenger_trades)
    protocol101 = load_protocol101(args.protocol101_trades, args.recent_protocol101_trades)
    combined = pd.concat([challenger, protocol101], ignore_index=True)
    enriched, quote_skips = enrich_from_normalized(combined, args.normalized_dir)

    side_summary = summarize_group(enriched, ["reported_split", "policy", "right"])
    time_summary = summarize_group(enriched, ["reported_split", "policy", "time_bucket"])
    exit_summary = summarize_group(enriched, ["reported_split", "policy", "exit_reason"])
    move_summary = summarize_directional_moves(enriched)
    overlap_summary, trade_overlap = summarize_trade_overlap(enriched)
    churn_summary, churn_chains = summarize_churn(enriched, gap_minutes=float(args.reentry_gap_minutes))
    hold_summary, hold_counterfactuals = summarize_churn_hold_counterfactuals(enriched, churn_chains)
    top_differences = summarize_day_differences(enriched)

    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / attribution audit",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "other_baseline_label": "none",
        "data_used": {
            "challenger_trades": str(args.challenger_trades),
            "protocol101_trades": str(args.protocol101_trades),
            "recent_protocol101_trades": str(args.recent_protocol101_trades),
            "normalized_dir": str(args.normalized_dir),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "reentry_gap_minutes": float(args.reentry_gap_minutes),
        "row_counts": {
            "challenger_trades": int(len(challenger)),
            "protocol101_trades": int(len(protocol101)),
            "enriched_trades": int(len(enriched)),
            "quote_skip_rows": int(len(quote_skips)),
            "trade_overlap_rows": int(len(trade_overlap)),
            "churn_chains": int(len(churn_chains)),
            "hold_counterfactual_rows": int(len(hold_counterfactuals)),
        },
        "headline": headline(enriched),
        "side_summary": side_summary,
        "time_bucket_summary": time_summary,
        "exit_reason_summary": exit_summary,
        "directional_move_summary": move_summary,
        "trade_overlap_summary": overlap_summary,
        "churn_summary": churn_summary,
        "churn_hold_counterfactual_summary": hold_summary,
        "top_day_differences": top_differences,
        "quote_skip_counts": count_by(quote_skips, "skip_reason"),
        "decision": decide(enriched, churn_summary, hold_summary),
        "next_experiment": (
            "Build the no-order runtime parity path for CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1, "
            "then inspect selected trades on charts before any paper-default replacement."
        ),
    }

    write_csv(args.out_dir / "side_summary.csv", side_summary)
    write_csv(args.out_dir / "time_bucket_summary.csv", time_summary)
    write_csv(args.out_dir / "exit_reason_summary.csv", exit_summary)
    write_csv(args.out_dir / "directional_move_summary.csv", move_summary)
    write_csv(args.out_dir / "trade_overlap_summary.csv", overlap_summary)
    trade_overlap.to_csv(args.out_dir / "trade_overlap_rows.csv", index=False)
    write_csv(args.out_dir / "churn_summary.csv", churn_summary)
    churn_chains.to_csv(args.out_dir / "churn_chains.csv", index=False)
    write_csv(args.out_dir / "churn_hold_counterfactual_summary.csv", hold_summary)
    hold_counterfactuals.to_csv(args.out_dir / "churn_hold_counterfactuals.csv", index=False)
    write_csv(args.out_dir / "top_day_differences.csv", top_differences)
    enriched.to_csv(args.out_dir / "enriched_policy_trades.csv", index=False)
    pd.DataFrame(quote_skips).to_csv(args.out_dir / "quote_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_challenger(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize_trades(frame, policy="challenger")


def load_protocol101(path: Path, recent_path: Path) -> pd.DataFrame:
    historical = pd.DataFrame(json.loads(path.read_text()))
    historical = normalize_trades(historical, policy="protocol101")
    recent = pd.read_csv(recent_path)
    recent = recent.rename(
        columns={
            "candidate_exit_time": "exit_time",
            "candidate_pnl": "pnl",
            "candidate_exit_reason": "exit_reason",
        }
    )
    recent["reported_split"] = "recent_2026"
    recent_rows = []
    for seed in [1, 2, 3, 4, 5]:
        item = recent.copy()
        item["seed"] = seed
        item["fold"] = "fold4_train_2025_validate_q1_2026_test_recent"
        recent_rows.append(item)
    recent_all = normalize_trades(pd.concat(recent_rows, ignore_index=True), policy="protocol101")
    return pd.concat([historical, recent_all], ignore_index=True)


def normalize_trades(frame: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    out = frame.copy()
    if "candidate_exit_time" in out.columns and "exit_time" not in out.columns:
        out["exit_time"] = out["candidate_exit_time"]
    if "candidate_pnl" in out.columns and "pnl" not in out.columns:
        out["pnl"] = out["candidate_pnl"]
    if "candidate_exit_reason" in out.columns and "exit_reason" not in out.columns:
        out["exit_reason"] = out["candidate_exit_reason"]
    for column, default in {
        "fold": "unknown",
        "seed": 0,
        "reported_split": "unknown",
        "session": "",
        "contract_id": "",
        "right": "",
        "exit_reason": "",
        "candidate_uid": "",
        "trade_uid": "",
    }.items():
        if column not in out.columns:
            out[column] = default
    out["policy"] = policy
    out["decision_ts"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["exit_ts"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    out = out[out["decision_ts"].notna() & out["exit_ts"].notna()].copy()
    out["duration_minutes"] = (out["exit_ts"] - out["decision_ts"]).dt.total_seconds() / 60.0
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(0).astype(int)
    for column in [
        "pnl",
        "entry_ask",
        "entry_bid",
        "entry_mid",
        "entry_premium",
        "entry_premium_with_slippage",
        "raw_candidate_pnl",
        "offset",
        "score",
        "threshold",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = np.nan
    out["reported_split"] = out["reported_split"].astype(str)
    out["session"] = out["session"].astype(str)
    out["fold"] = out["fold"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)
    out["right"] = out["right"].astype(str)
    out["exit_reason"] = out["exit_reason"].astype(str)
    out["time_bucket"] = [time_bucket(ts.to_pydatetime()) for ts in out["decision_ts"]]
    out["trade_key"] = (
        out["reported_split"].astype(str)
        + "|seed"
        + out["seed"].astype(str)
        + "|"
        + out["session"].astype(str)
        + "|"
        + out["decision_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        + "|"
        + out["contract_id"].astype(str)
    )
    return out.reset_index(drop=True)


def enrich_from_normalized(frame: pd.DataFrame, normalized_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = frame.copy()
    for column in ["entry_bid_live", "entry_ask_live", "exit_bid_live", "exit_ask_live", "entry_underlying", "exit_underlying", "path_mfe", "path_mae"]:
        out[column] = np.nan
    skips: list[dict[str, Any]] = []
    for session, indices in out.groupby("session", sort=True).groups.items():
        quotes = load_session_quotes(normalized_dir, str(session))
        if quotes.empty:
            skips.extend({"session": str(session), "row_index": int(i), "skip_reason": "missing_session_quotes"} for i in indices)
            continue
        by_contract = {cid: g.sort_values("quote_time").reset_index(drop=True) for cid, g in quotes.groupby("contract_id", sort=False)}
        for idx in indices:
            row = out.loc[idx]
            q = by_contract.get(str(row["contract_id"]))
            if q is None or q.empty:
                skips.append({"session": str(session), "row_index": int(idx), "contract_id": str(row["contract_id"]), "skip_reason": "missing_contract_quotes"})
                continue
            entry = first_at_or_after(q, pd.Timestamp(row["decision_ts"]))
            exit_row = first_at_or_after(q, pd.Timestamp(row["exit_ts"]))
            if entry is None or exit_row is None:
                skips.append({"session": str(session), "row_index": int(idx), "contract_id": str(row["contract_id"]), "skip_reason": "missing_entry_or_exit_quote"})
                continue
            entry_ask = finite(row.get("entry_ask"))
            if not math.isfinite(entry_ask) or entry_ask <= 0.0:
                entry_ask = finite(entry.get("ask"))
            if not math.isfinite(entry_ask) or entry_ask <= 0.0:
                skips.append({"session": str(session), "row_index": int(idx), "contract_id": str(row["contract_id"]), "skip_reason": "invalid_entry_ask"})
                continue
            path = q[(q["quote_time"] >= pd.Timestamp(row["decision_ts"])) & (q["quote_time"] <= pd.Timestamp(row["exit_ts"]))].copy()
            path_pnl = (pd.to_numeric(path["bid"], errors="coerce") - entry_ask) * CONTRACT_MULTIPLIER
            out.at[idx, "entry_bid_live"] = finite(entry.get("bid"))
            out.at[idx, "entry_ask_live"] = finite(entry.get("ask"))
            out.at[idx, "exit_bid_live"] = finite(exit_row.get("bid"))
            out.at[idx, "exit_ask_live"] = finite(exit_row.get("ask"))
            out.at[idx, "entry_underlying"] = finite(entry.get("underlying_price"))
            out.at[idx, "exit_underlying"] = finite(exit_row.get("underlying_price"))
            out.at[idx, "path_mfe"] = finite(path_pnl.max())
            out.at[idx, "path_mae"] = finite(path_pnl.min())
            if not math.isfinite(finite(row.get("entry_ask"))):
                out.at[idx, "entry_ask"] = entry_ask
            if not math.isfinite(finite(row.get("entry_premium"))):
                out.at[idx, "entry_premium"] = entry_ask * CONTRACT_MULTIPLIER
    out["underlying_move"] = out["exit_underlying"] - out["entry_underlying"]
    out["directional_underlying_move"] = np.where(out["right"].eq("P"), -out["underlying_move"], out["underlying_move"])
    out["mfe_capture"] = np.where(pd.to_numeric(out["path_mfe"], errors="coerce") > 0, pd.to_numeric(out["pnl"], errors="coerce") / pd.to_numeric(out["path_mfe"], errors="coerce"), np.nan)
    return out, skips


def load_session_quotes(normalized_dir: Path, session: str) -> pd.DataFrame:
    path = find_normalized_path(normalized_dir, session)
    if path is None:
        return pd.DataFrame()
    columns = ["quote_time", "contract_id", "bid", "ask", "underlying_price"]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.DataFrame()
    frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True, errors="coerce")
    frame["contract_id"] = frame["contract_id"].astype(str)
    for column in ["bid", "ask", "underlying_price"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        frame["quote_time"].notna()
        & frame["contract_id"].notna()
        & (frame["bid"] >= 0.0)
        & (frame["ask"] > 0.0)
        & (frame["ask"] >= frame["bid"])
    ][columns].copy()


def find_normalized_path(normalized_dir: Path, session: str) -> Path | None:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    if preferred:
        return preferred[0]
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return fallback[0] if fallback else None


def first_at_or_after(quotes: pd.DataFrame, target: pd.Timestamp) -> pd.Series | None:
    valid = quotes[quotes["quote_time"] >= target]
    return None if valid.empty else valid.iloc[0]


def summarize_group(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: str(value) for column, value in zip(group_cols, key)}
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        row.update(
            {
                "trades": int(len(group)),
                "pnl": float(pnl.sum()),
                "median_pnl": float(pnl.median()) if len(pnl) else 0.0,
                "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
                "profit_factor": float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-9 else (float("inf") if wins.sum() > 0 else 0.0),
                "median_duration_minutes": finite(group["duration_minutes"].median()),
                "median_entry_premium": finite(group["entry_premium"].median()),
                "median_mfe_capture": finite(group["mfe_capture"].median()),
                "median_directional_underlying_move": finite(group["directional_underlying_move"].median()),
            }
        )
        rows.append(row)
    return rows


def summarize_directional_moves(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (split, policy), group in frame.groupby(["reported_split", "policy"], sort=True):
        for threshold in [5.0, 10.0, 20.0, 30.0]:
            moves = group[pd.to_numeric(group["directional_underlying_move"], errors="coerce") >= threshold]
            rows.append(
                {
                    "reported_split": split,
                    "policy": policy,
                    "directional_move_threshold_spx": threshold,
                    "trades": int(len(moves)),
                    "share_of_policy_trades": float(len(moves) / max(len(group), 1)),
                    "pnl": finite_sum(moves["pnl"]),
                    "median_duration_minutes": finite(moves["duration_minutes"].median()) if len(moves) else 0.0,
                    "median_mfe_capture": finite(moves["mfe_capture"].median()) if len(moves) else 0.0,
                }
            )
    return rows


def summarize_trade_overlap(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    pivot = frame.pivot_table(index=["reported_split", "seed", "session", "decision_time", "contract_id"], columns="policy", values="pnl", aggfunc="sum").reset_index()
    for column in ["challenger", "protocol101"]:
        if column not in pivot.columns:
            pivot[column] = np.nan
    pivot["membership"] = np.select(
        [pivot["challenger"].notna() & pivot["protocol101"].notna(), pivot["challenger"].notna(), pivot["protocol101"].notna()],
        ["common_exact_trade", "challenger_only", "protocol101_only"],
        default="none",
    )
    rows = []
    for (split, membership), group in pivot.groupby(["reported_split", "membership"], sort=True):
        rows.append(
            {
                "reported_split": split,
                "membership": membership,
                "rows": int(len(group)),
                "challenger_pnl": finite_sum(group["challenger"]),
                "protocol101_pnl": finite_sum(group["protocol101"]),
                "delta": finite_sum(group["challenger"]) - finite_sum(group["protocol101"]),
            }
        )
    return rows, pivot


def summarize_churn(frame: pd.DataFrame, *, gap_minutes: float) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    chains = []
    for (policy, split, seed, session), group in frame.sort_values(["decision_ts", "exit_ts"]).groupby(["policy", "reported_split", "seed", "session"], sort=True):
        current: list[int] = []
        for idx, row in group.iterrows():
            if not current:
                current = [idx]
                continue
            prev = frame.loc[current[-1]]
            gap = (pd.Timestamp(row["decision_ts"]) - pd.Timestamp(prev["exit_ts"])).total_seconds() / 60.0
            if str(row["right"]) == str(prev["right"]) and gap >= -1e-9 and gap <= gap_minutes:
                current.append(idx)
            else:
                if len(current) >= 2:
                    chains.append(make_chain_row(frame, current, gap_minutes))
                current = [idx]
        if len(current) >= 2:
            chains.append(make_chain_row(frame, current, gap_minutes))
    chain_frame = pd.DataFrame(chains)
    summary = []
    if not chain_frame.empty:
        for (split, policy, right), group in chain_frame.groupby(["reported_split", "policy", "right"], sort=True):
            summary.append(
                {
                    "reported_split": split,
                    "policy": policy,
                    "right": right,
                    "chains": int(len(group)),
                    "trades_in_chains": int(group["chain_length"].sum()),
                    "chain_pnl": finite_sum(group["sequence_pnl"]),
                    "median_chain_length": finite(group["chain_length"].median()),
                    "median_chain_duration_minutes": finite(group["chain_duration_minutes"].median()),
                }
            )
    return summary, chain_frame


def make_chain_row(frame: pd.DataFrame, indices: list[int], gap_minutes: float) -> dict[str, Any]:
    rows = frame.loc[indices].sort_values("decision_ts")
    first = rows.iloc[0]
    last = rows.iloc[-1]
    return {
        "chain_id": f"{first['policy']}|{first['reported_split']}|seed{int(first['seed'])}|{first['session']}|{first['right']}|{pd.Timestamp(first['decision_ts']).isoformat()}",
        "policy": str(first["policy"]),
        "reported_split": str(first["reported_split"]),
        "seed": int(first["seed"]),
        "session": str(first["session"]),
        "right": str(first["right"]),
        "gap_minutes": float(gap_minutes),
        "chain_length": int(len(rows)),
        "first_decision_time": pd.Timestamp(first["decision_ts"]).isoformat(),
        "last_exit_time": pd.Timestamp(last["exit_ts"]).isoformat(),
        "chain_duration_minutes": float((pd.Timestamp(last["exit_ts"]) - pd.Timestamp(first["decision_ts"])).total_seconds() / 60.0),
        "sequence_pnl": finite_sum(rows["pnl"]),
        "contract_count": int(rows["contract_id"].nunique()),
        "time_bucket": str(first["time_bucket"]),
        "first_contract_id": str(first["contract_id"]),
        "last_contract_id": str(last["contract_id"]),
    }


def summarize_churn_hold_counterfactuals(frame: pd.DataFrame, chains: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rows = []
    if chains.empty:
        return [], pd.DataFrame()
    by_key = {
        (row.policy, row.reported_split, int(row.seed), row.session, row.right, row.first_decision_time): row
        for row in chains.itertuples(index=False)
    }
    for key, chain in by_key.items():
        policy, split, seed, session, right, first_decision = key
        members = frame[
            (frame["policy"].eq(policy))
            & (frame["reported_split"].eq(split))
            & (frame["seed"].eq(seed))
            & (frame["session"].eq(session))
            & (frame["right"].eq(right))
            & (frame["decision_ts"] >= pd.Timestamp(first_decision))
            & (frame["exit_ts"] <= pd.Timestamp(chain.last_exit_time))
        ].sort_values("decision_ts")
        if len(members) < 2:
            continue
        first = members.iloc[0]
        entry_ask = finite(first.get("entry_ask"))
        exit_bid = finite(first.get("exit_bid_live"))
        # The first contract's bid at the chain's final exit is approximated by the path final only when
        # the final member uses the same contract. Otherwise this counterfactual is intentionally marked.
        same_contract = members["contract_id"].nunique() == 1
        if same_contract:
            exit_bid = finite(members.iloc[-1].get("exit_bid_live"))
        if not same_contract or not math.isfinite(entry_ask) or not math.isfinite(exit_bid):
            continue
        hold_pnl = (exit_bid - entry_ask) * CONTRACT_MULTIPLIER
        sequence_pnl = finite_sum(members["pnl"])
        rows.append(
            {
                "policy": policy,
                "reported_split": split,
                "seed": seed,
                "session": session,
                "right": right,
                "time_bucket": str(first["time_bucket"]),
                "chain_length": int(len(members)),
                "sequence_pnl": sequence_pnl,
                "same_contract_hold_pnl": float(hold_pnl),
                "hold_minus_sequence_pnl": float(hold_pnl - sequence_pnl),
            }
        )
    cf = pd.DataFrame(rows)
    summary = []
    if not cf.empty:
        for (split, policy, right), group in cf.groupby(["reported_split", "policy", "right"], sort=True):
            summary.append(
                {
                    "reported_split": split,
                    "policy": policy,
                    "right": right,
                    "chains": int(len(group)),
                    "sequence_pnl": finite_sum(group["sequence_pnl"]),
                    "same_contract_hold_pnl": finite_sum(group["same_contract_hold_pnl"]),
                    "hold_minus_sequence_pnl": finite_sum(group["hold_minus_sequence_pnl"]),
                    "hold_better_fraction": float((group["hold_minus_sequence_pnl"] > 0).mean()),
                }
            )
    return summary, cf


def summarize_day_differences(frame: pd.DataFrame) -> list[dict[str, Any]]:
    day = frame.groupby(["reported_split", "session", "policy"], sort=True)["pnl"].sum().reset_index()
    pivot = day.pivot_table(index=["reported_split", "session"], columns="policy", values="pnl", aggfunc="sum").reset_index().fillna(0.0)
    if "challenger" not in pivot.columns:
        pivot["challenger"] = 0.0
    if "protocol101" not in pivot.columns:
        pivot["protocol101"] = 0.0
    pivot["delta"] = pivot["challenger"] - pivot["protocol101"]
    best = pivot.sort_values("delta", ascending=False).head(12).copy()
    worst = pivot.sort_values("delta", ascending=True).head(12).copy()
    best["bucket"] = "challenger_best_delta_days"
    worst["bucket"] = "challenger_worst_delta_days"
    out = pd.concat([best, worst], ignore_index=True)
    return out[["bucket", "reported_split", "session", "challenger", "protocol101", "delta"]].to_dict("records")


def headline(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (split, policy), group in frame.groupby(["reported_split", "policy"], sort=True):
        rows.append(
            {
                "reported_split": split,
                "policy": policy,
                "trades": int(len(group)),
                "pnl": finite_sum(group["pnl"]),
                "median_duration_minutes": finite(group["duration_minutes"].median()),
                "call_trades": int(group["right"].eq("C").sum()),
                "put_trades": int(group["right"].eq("P").sum()),
                "median_mfe_capture": finite(group["mfe_capture"].median()),
            }
        )
    return rows


def decide(frame: pd.DataFrame, churn_summary: list[dict[str, Any]], hold_summary: list[dict[str, Any]]) -> str:
    challenger = frame[frame["policy"].eq("challenger")]
    protocol101 = frame[frame["policy"].eq("protocol101")]
    if challenger.empty or protocol101.empty:
        return "blocked_missing_policy_trades"
    challenger_pnl = challenger.groupby("reported_split")["pnl"].sum()
    protocol_pnl = protocol101.groupby("reported_split")["pnl"].sum()
    common = sorted(set(challenger_pnl.index) & set(protocol_pnl.index))
    if not common:
        return "blocked_no_common_splits"
    beats = all(float(challenger_pnl[split]) > float(protocol_pnl[split]) for split in common)
    return "attribution_supports_challenger_next_runtime_parity" if beats else "mixed_attribution_requires_failure_analysis"


def count_by(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, ""))
        out[key] = out.get(key, 0) + 1
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Other baseline: {payload['other_baseline_label']}",
        f"Data used: {payload['data_used']['challenger_trades']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Headline",
        "",
        "| split | policy | trades | PnL | calls | puts | median duration | median MFE capture |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["headline"]:
        lines.append(
            f"| {row['reported_split']} | {row['policy']} | {row['trades']} | {money(row['pnl'])} | "
            f"{row['call_trades']} | {row['put_trades']} | {row['median_duration_minutes']:.1f} | "
            f"{row['median_mfe_capture']:.2f} |"
        )
    lines.extend(["", "## Exact Trade Overlap", "", "| split | membership | rows | challenger PnL | Protocol101 PnL | delta |", "|---|---|---:|---:|---:|---:|"])
    for row in payload["trade_overlap_summary"]:
        lines.append(
            f"| {row['reported_split']} | {row['membership']} | {row['rows']} | "
            f"{money(row['challenger_pnl'])} | {money(row['protocol101_pnl'])} | {money(row['delta'])} |"
        )
    lines.extend(["", "## Churn / Re-Entry", "", "| split | policy | side | chains | trades in chains | chain PnL | median chain length | median duration |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in payload["churn_summary"]:
        lines.append(
            f"| {row['reported_split']} | {row['policy']} | {row['right']} | {row['chains']} | "
            f"{row['trades_in_chains']} | {money(row['chain_pnl'])} | {row['median_chain_length']:.1f} | "
            f"{row['median_chain_duration_minutes']:.1f} |"
        )
    lines.extend(["", "## Directional Move Capture", "", "| split | policy | move >= SPX pts | trades | share | PnL | median duration | median MFE capture |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["directional_move_summary"]:
        if float(row["directional_move_threshold_spx"]) in {10.0, 20.0, 30.0}:
            lines.append(
                f"| {row['reported_split']} | {row['policy']} | {row['directional_move_threshold_spx']:.0f} | "
                f"{row['trades']} | {pct(row['share_of_policy_trades'])} | {money(row['pnl'])} | "
                f"{row['median_duration_minutes']:.1f} | {row['median_mfe_capture']:.2f} |"
            )
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Enriched trades: `{path.parent / 'enriched_policy_trades.csv'}`", f"- Churn chains: `{path.parent / 'churn_chains.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def finite_sum(values: Any) -> float:
    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())


def money(value: Any) -> str:
    return f"{finite(value):,.0f}"


def pct(value: Any) -> str:
    return f"{finite(value) * 100.0:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
