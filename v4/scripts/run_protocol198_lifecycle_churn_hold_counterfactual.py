"""Protocol198: lifecycle churn and continuous-hold counterfactual audit.

Protocol198 is diagnostic infrastructure, not a new trading rule. It attacks a
specific failure mode the user observed: the policy may exit and then re-enter
the same directional idea instead of managing one position through the larger
move.

The audit keeps Protocol194/081 trades frozen, finds same-side re-entry chains,
and asks whether the first contract held through the last chain exit would have
outperformed the actual exit/re-entry sequence. This does not hardcode a minimum
hold time or a percentage exit. It only measures whether our current labels and
lifecycle stack are rewarding churn.

No paid data is downloaded. No live broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.environment_diagnostics import time_bucket


LOOP_ID = "v4_aplus_hypothesis_198_lifecycle_churn_hold_counterfactual"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_194_full_action_surface_edge_5seed_confirmation/"
    "protocol194_protocol081_5seed_serial_trades.csv"
)
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
CONTRACT_MULTIPLIER = 100.0
DEFAULT_HORIZONS_MINUTES = (5, 15, 30)


@dataclass(frozen=True)
class ReentryChain:
    chain_id: str
    gap_minutes: float
    fold: str
    seed: int
    reported_split: str
    session: str
    right: str
    trade_indices: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gap-minutes", type=float, action="append", default=None)
    parser.add_argument("--primary-gap-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(sorted(set(float(v) for v in (args.gap_minutes or DEFAULT_HORIZONS_MINUTES))))
    if float(args.primary_gap_minutes) not in horizons:
        horizons = tuple(sorted((*horizons, float(args.primary_gap_minutes))))

    trades = load_trades(args.trades)
    horizon_summary = summarize_horizons(trades, horizons)
    chains = build_same_side_reentry_chains(trades, gap_minutes=float(args.primary_gap_minutes))
    member_rows = chain_member_rows(trades, chains)
    counterfactuals, path_skips = build_hold_counterfactuals(
        trades,
        chains,
        normalized_dir=args.normalized_dir,
    )
    counterfactual_frame = pd.DataFrame(counterfactuals)
    split_summary = summarize_counterfactuals(counterfactual_frame, ["reported_split"])
    side_summary = summarize_counterfactuals(counterfactual_frame, ["reported_split", "right"])
    time_summary = summarize_counterfactuals(counterfactual_frame, ["reported_split", "time_bucket"])
    exit_summary = summarize_counterfactuals(counterfactual_frame, ["reported_split", "first_exit_reason"])
    payload = {
        "protocol": "198_lifecycle_churn_hold_counterfactual",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_trades": str(args.trades),
        "normalized_dir": str(args.normalized_dir),
        "primary_gap_minutes": float(args.primary_gap_minutes),
        "horizons_minutes": list(horizons),
        "row_counts": {
            "source_trades": int(len(trades)),
            "primary_chains": int(len(chains)),
            "chain_member_rows": int(len(member_rows)),
            "hold_counterfactual_rows": int(len(counterfactual_frame)),
            "path_skips": int(len(path_skips)),
        },
        "horizon_summary": horizon_summary,
        "split_summary": split_summary,
        "side_summary": side_summary,
        "time_bucket_summary": time_summary,
        "exit_reason_summary": exit_summary,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": decide(counterfactual_frame, chains, path_skips),
        "interpretation": (
            "This audit does not say fast profits are fake or that the bot should hold for a fixed "
            "time. It asks whether same-side exit/re-entry sequences were better than managing the "
            "first contract through the later exit. A harmful churn result points toward learned "
            "hold/exit continuation modeling, not a hardcoded hold rule."
        ),
        "next_model_hypothesis": next_hypothesis(counterfactual_frame),
    }

    pd.DataFrame(horizon_summary).to_csv(args.out_dir / "horizon_summary.csv", index=False)
    pd.DataFrame(member_rows).to_csv(args.out_dir / "chain_trade_members.csv", index=False)
    counterfactual_frame.to_csv(args.out_dir / "churn_hold_counterfactuals.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    pd.DataFrame(split_summary).to_csv(args.out_dir / "split_summary.csv", index=False)
    pd.DataFrame(side_summary).to_csv(args.out_dir / "side_summary.csv", index=False)
    pd.DataFrame(time_summary).to_csv(args.out_dir / "time_bucket_summary.csv", index=False)
    pd.DataFrame(exit_summary).to_csv(args.out_dir / "exit_reason_summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload, counterfactual_frame)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"no trades found in {path}")
    return normalize_trade_frame(frame)


def normalize_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "candidate_exit_time" in out.columns:
        exit_column = "candidate_exit_time"
    elif "exit_time" in out.columns:
        exit_column = "exit_time"
    else:
        raise ValueError("trades must include candidate_exit_time or exit_time")
    required = ["reported_split", "session", "decision_time", "contract_id", "right", "pnl", "entry_ask"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"missing required trade columns: {missing}")
    out["decision_ts"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["exit_ts"] = pd.to_datetime(out[exit_column], utc=True, errors="coerce")
    out = out[out["decision_ts"].notna() & out["exit_ts"].notna()].copy()
    if "fold" not in out.columns:
        out["fold"] = "unknown"
    if "seed" not in out.columns:
        out["seed"] = 0
    out["fold"] = out["fold"].astype(str)
    out["reported_split"] = out["reported_split"].astype(str)
    out["session"] = out["session"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)
    out["right"] = out["right"].astype(str)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(0).astype(int)
    for column in ["pnl", "entry_ask", "entry_bid", "entry_mid", "entry_premium", "score", "offset"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["time_bucket"] = [time_bucket(ts.to_pydatetime()) for ts in out["decision_ts"]]
    out["source_trade_index"] = np.arange(len(out), dtype=np.int64)
    return out.sort_values(
        ["fold", "seed", "reported_split", "session", "decision_ts", "contract_id"]
    ).reset_index(drop=True)


def build_same_side_reentry_chains(trades: pd.DataFrame, *, gap_minutes: float) -> list[ReentryChain]:
    chains: list[ReentryChain] = []
    group_cols = ["fold", "seed", "reported_split", "session"]
    for key, group in trades.groupby(group_cols, sort=True):
        current: list[int] = []
        chain_number = 0
        ordered = group.sort_values(["decision_ts", "exit_ts", "contract_id"])
        for idx, row in ordered.iterrows():
            if not current:
                current = [idx]
                continue
            prev = trades.loc[current[-1]]
            gap = (pd.Timestamp(row["decision_ts"]) - pd.Timestamp(prev["exit_ts"])).total_seconds() / 60.0
            same_side = str(row["right"]) == str(prev["right"])
            serial_ordered = gap >= -1e-9
            if same_side and serial_ordered and gap <= float(gap_minutes):
                current.append(idx)
                continue
            if len(current) >= 2:
                chains.append(make_chain(trades, current, key, gap_minutes, chain_number))
                chain_number += 1
            current = [idx]
        if len(current) >= 2:
            chains.append(make_chain(trades, current, key, gap_minutes, chain_number))
    return chains


def make_chain(
    trades: pd.DataFrame,
    indices: list[int],
    key: tuple[Any, Any, Any, Any],
    gap_minutes: float,
    chain_number: int,
) -> ReentryChain:
    fold, seed, reported_split, session = key
    first = trades.loc[indices[0]]
    chain_id = f"{fold}|seed{int(seed)}|{reported_split}|{session}|{first['right']}|g{gap_minutes:g}|{chain_number}"
    return ReentryChain(
        chain_id=chain_id,
        gap_minutes=float(gap_minutes),
        fold=str(fold),
        seed=int(seed),
        reported_split=str(reported_split),
        session=str(session),
        right=str(first["right"]),
        trade_indices=tuple(int(idx) for idx in indices),
    )


def summarize_horizons(trades: pd.DataFrame, horizons: Iterable[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        chains = build_same_side_reentry_chains(trades, gap_minutes=float(horizon))
        chain_trade_indices = [idx for chain in chains for idx in chain.trade_indices]
        chain_trades = trades.loc[chain_trade_indices] if chain_trade_indices else trades.iloc[0:0]
        rows.append(
            {
                "gap_minutes": float(horizon),
                "chains": int(len(chains)),
                "trades_in_chains": int(len(chain_trade_indices)),
                "source_trades": int(len(trades)),
                "share_of_trades_in_chains": float(len(chain_trade_indices) / max(len(trades), 1)),
                "actual_chain_pnl": finite_sum(chain_trades.get("pnl", pd.Series(dtype=float))),
                "call_chains": int(sum(chain.right == "C" for chain in chains)),
                "put_chains": int(sum(chain.right == "P" for chain in chains)),
            }
        )
    return rows


def chain_member_rows(trades: pd.DataFrame, chains: list[ReentryChain]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chain in chains:
        previous_exit: pd.Timestamp | None = None
        for member_order, idx in enumerate(chain.trade_indices):
            trade = trades.loc[idx]
            gap = None
            if previous_exit is not None:
                gap = (pd.Timestamp(trade["decision_ts"]) - previous_exit).total_seconds() / 60.0
            rows.append(
                {
                    "chain_id": chain.chain_id,
                    "member_order": int(member_order),
                    "gap_from_previous_exit_minutes": gap,
                    "reported_split": chain.reported_split,
                    "fold": chain.fold,
                    "seed": int(chain.seed),
                    "session": chain.session,
                    "right": chain.right,
                    "decision_time": pd.Timestamp(trade["decision_ts"]).isoformat(),
                    "exit_time": pd.Timestamp(trade["exit_ts"]).isoformat(),
                    "contract_id": str(trade["contract_id"]),
                    "pnl": finite_float(trade.get("pnl"), 0.0),
                    "entry_ask": finite_float(trade.get("entry_ask"), math.nan),
                    "exit_reason": str(trade.get("candidate_exit_reason", trade.get("exit_reason", ""))),
                    "score": finite_float(trade.get("score"), math.nan),
                    "offset": finite_float(trade.get("offset"), math.nan),
                }
            )
            previous_exit = pd.Timestamp(trade["exit_ts"])
    return rows


def build_hold_counterfactuals(
    trades: pd.DataFrame,
    chains: list[ReentryChain],
    *,
    normalized_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    if not chains:
        return rows, skips
    chain_frame = pd.DataFrame(
        {
            "chain_id": [chain.chain_id for chain in chains],
            "session": [chain.session for chain in chains],
            "first_idx": [chain.trade_indices[0] for chain in chains],
        }
    )
    chain_by_id = {chain.chain_id: chain for chain in chains}
    for session, session_chains in chain_frame.groupby("session", sort=True):
        first_contracts = {
            str(trades.loc[int(row.first_idx), "contract_id"]) for row in session_chains.itertuples(index=False)
        }
        quotes = load_session_quotes(normalized_dir, str(session), first_contracts)
        if quotes.empty:
            for row in session_chains.itertuples(index=False):
                skips.append({"chain_id": row.chain_id, "session": str(session), "skip_reason": "missing_session_or_contract_quotes"})
            continue
        by_contract = {
            str(contract_id): part.sort_values("quote_time").reset_index(drop=True)
            for contract_id, part in quotes.groupby("contract_id", sort=False)
        }
        for row in session_chains.itertuples(index=False):
            chain = chain_by_id[str(row.chain_id)]
            result, skip = hold_counterfactual_for_chain(trades, chain, by_contract)
            if skip:
                skips.append(skip)
            else:
                rows.append(result)
    return rows, skips


def load_session_quotes(normalized_dir: Path, session: str, contract_ids: set[str]) -> pd.DataFrame:
    path = find_normalized_path(normalized_dir, session)
    if path is None or not contract_ids:
        return pd.DataFrame(columns=["quote_time", "contract_id", "bid", "ask", "underlying_price"])
    columns = ["quote_time", "contract_id", "bid", "ask", "underlying_price"]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        try:
            frame = pd.read_parquet(path, columns=["quote_time", "contract_id", "bid", "ask"])
        except Exception:
            return pd.DataFrame(columns=columns)
        frame["underlying_price"] = np.nan
    frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True, errors="coerce")
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame = frame[frame["contract_id"].isin(contract_ids)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in ["bid", "ask", "underlying_price"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        frame["quote_time"].notna()
        & frame["bid"].notna()
        & frame["ask"].notna()
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


def hold_counterfactual_for_chain(
    trades: pd.DataFrame,
    chain: ReentryChain,
    quotes_by_contract: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    first = trades.loc[chain.trade_indices[0]]
    last = trades.loc[chain.trade_indices[-1]]
    contract_id = str(first["contract_id"])
    quotes = quotes_by_contract.get(contract_id)
    if quotes is None or quotes.empty:
        return {}, base_skip(chain, "missing_first_contract_quotes")
    entry_ask = finite_float(first.get("entry_ask"), math.nan)
    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        return {}, base_skip(chain, "invalid_entry_ask")
    start = pd.Timestamp(first["decision_ts"])
    final_exit = pd.Timestamp(last["exit_ts"])
    path = quotes[(quotes["quote_time"] >= start) & (quotes["quote_time"] <= final_exit)].copy()
    final_quote = first_at_or_after(quotes, final_exit, "bid")
    if final_quote is None:
        return {}, base_skip(chain, "missing_final_exit_quote")
    if path.empty:
        path = quotes[(quotes["quote_time"] >= start) & (quotes["quote_time"] <= pd.Timestamp(final_quote["quote_time"]))].copy()
    if path.empty:
        return {}, base_skip(chain, "missing_hold_path")
    hold_exit_bid = finite_float(final_quote.get("bid"), math.nan)
    if not math.isfinite(hold_exit_bid) or hold_exit_bid < 0.0:
        return {}, base_skip(chain, "invalid_hold_exit_bid")
    path_pnl = (pd.to_numeric(path["bid"], errors="coerce") - entry_ask) * CONTRACT_MULTIPLIER
    actual_sequence_pnl = finite_sum(trades.loc[list(chain.trade_indices), "pnl"])
    hold_pnl = float((hold_exit_bid - entry_ask) * CONTRACT_MULTIPLIER)
    underlying_entry = finite_float(path.iloc[0].get("underlying_price"), math.nan)
    underlying_exit = finite_float(final_quote.get("underlying_price"), math.nan)
    underlying_move = underlying_exit - underlying_entry if math.isfinite(underlying_entry) and math.isfinite(underlying_exit) else math.nan
    aligned_move = directional_underlying_move(chain.right, underlying_move)
    contract_ids = trades.loc[list(chain.trade_indices), "contract_id"].astype(str).tolist()
    exit_reasons = trades.loc[list(chain.trade_indices)].apply(
        lambda row: str(row.get("candidate_exit_reason", row.get("exit_reason", ""))),
        axis=1,
    ).tolist()
    gaps = reentry_gaps_minutes(trades, chain.trade_indices)
    first_trade_pnl = finite_float(first.get("pnl"), 0.0)
    return {
        "chain_id": chain.chain_id,
        "reported_split": chain.reported_split,
        "fold": chain.fold,
        "seed": int(chain.seed),
        "session": chain.session,
        "right": chain.right,
        "time_bucket": str(first.get("time_bucket", "")),
        "chain_length": int(len(chain.trade_indices)),
        "same_contract_all": bool(len(set(contract_ids)) == 1),
        "contract_count": int(len(set(contract_ids))),
        "first_contract_id": contract_id,
        "last_contract_id": str(last["contract_id"]),
        "first_decision_time": start.isoformat(),
        "first_exit_time": pd.Timestamp(first["exit_ts"]).isoformat(),
        "last_exit_time": final_exit.isoformat(),
        "counterfactual_exit_quote_time": pd.Timestamp(final_quote["quote_time"]).isoformat(),
        "chain_duration_minutes": float((final_exit - start).total_seconds() / 60.0),
        "avg_reentry_gap_minutes": float(np.mean(gaps)) if gaps else 0.0,
        "max_reentry_gap_minutes": float(np.max(gaps)) if gaps else 0.0,
        "entry_ask": entry_ask,
        "hold_exit_bid": hold_exit_bid,
        "actual_sequence_pnl": float(actual_sequence_pnl),
        "first_trade_pnl": float(first_trade_pnl),
        "counterfactual_hold_pnl": float(hold_pnl),
        "hold_minus_sequence_pnl": float(hold_pnl - actual_sequence_pnl),
        "hold_minus_first_trade_pnl": float(hold_pnl - first_trade_pnl),
        "path_mfe_pnl": finite_float(path_pnl.max(), math.nan),
        "path_mae_pnl": finite_float(path_pnl.min(), math.nan),
        "path_final_pnl": float(hold_pnl),
        "path_points": int(len(path)),
        "underlying_entry": underlying_entry,
        "underlying_exit": underlying_exit,
        "underlying_move": underlying_move,
        "directional_underlying_move": aligned_move,
        "first_exit_reason": exit_reasons[0] if exit_reasons else "",
        "last_exit_reason": exit_reasons[-1] if exit_reasons else "",
        "exit_reason_sequence": " -> ".join(exit_reasons),
        "classification": classify_churn(hold_pnl, actual_sequence_pnl),
    }, None


def base_skip(chain: ReentryChain, reason: str) -> dict[str, Any]:
    return {
        "chain_id": chain.chain_id,
        "reported_split": chain.reported_split,
        "fold": chain.fold,
        "seed": int(chain.seed),
        "session": chain.session,
        "right": chain.right,
        "skip_reason": reason,
    }


def first_at_or_after(quotes: pd.DataFrame, target: pd.Timestamp, column: str) -> pd.Series | None:
    valid = quotes[(quotes["quote_time"] >= target) & quotes[column].notna()]
    if column == "ask":
        valid = valid[valid["ask"] > 0]
    if column == "bid":
        valid = valid[valid["bid"] >= 0]
    return None if valid.empty else valid.iloc[0]


def reentry_gaps_minutes(trades: pd.DataFrame, indices: tuple[int, ...]) -> list[float]:
    gaps: list[float] = []
    for previous_idx, idx in zip(indices, indices[1:]):
        previous = trades.loc[previous_idx]
        current = trades.loc[idx]
        gaps.append(float((pd.Timestamp(current["decision_ts"]) - pd.Timestamp(previous["exit_ts"])).total_seconds() / 60.0))
    return gaps


def directional_underlying_move(right: str, underlying_move: float) -> float:
    if not math.isfinite(underlying_move):
        return math.nan
    if right == "P":
        return float(-underlying_move)
    return float(underlying_move)


def classify_churn(hold_pnl: float, sequence_pnl: float) -> str:
    if hold_pnl > sequence_pnl + 1e-9:
        return "counterfactual_hold_better"
    if sequence_pnl > hold_pnl + 1e-9:
        return "exit_reentry_sequence_better"
    return "neutral"


def summarize_counterfactuals(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: str(value) for column, value in zip(group_cols, key)}
        row.update(
            {
                "chains": int(len(group)),
                "chain_trades": int(pd.to_numeric(group["chain_length"], errors="coerce").fillna(0).sum()),
                "actual_sequence_pnl": finite_sum(group["actual_sequence_pnl"]),
                "counterfactual_hold_pnl": finite_sum(group["counterfactual_hold_pnl"]),
                "hold_minus_sequence_pnl": finite_sum(group["hold_minus_sequence_pnl"]),
                "median_hold_minus_sequence_pnl": finite_float(group["hold_minus_sequence_pnl"].median(), 0.0),
                "hold_better_fraction": float((group["hold_minus_sequence_pnl"] > 0.0).mean()),
                "sequence_better_fraction": float((group["hold_minus_sequence_pnl"] < 0.0).mean()),
                "same_contract_fraction": float(group["same_contract_all"].mean()),
                "median_directional_underlying_move": finite_float(group["directional_underlying_move"].median(), math.nan),
            }
        )
        rows.append(row)
    return rows


def decide(frame: pd.DataFrame, chains: list[ReentryChain], path_skips: list[dict[str, Any]]) -> str:
    if not chains:
        return "blocked_no_same_side_reentry_chains"
    coverage = len(frame) / max(len(chains), 1)
    if coverage < 0.75:
        return "blocked_insufficient_hold_counterfactual_coverage"
    total_delta = finite_sum(frame["hold_minus_sequence_pnl"])
    hold_better_fraction = float((frame["hold_minus_sequence_pnl"] > 0.0).mean()) if len(frame) else 0.0
    if total_delta > 0.0 and hold_better_fraction >= 0.50:
        return "lifecycle_churn_failure_mode_detected"
    if total_delta > 0.0:
        return "mixed_lifecycle_churn_needs_sequence_training"
    return "churn_not_primary_failure_mode_for_selected_trades"


def next_hypothesis(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "Build a lifecycle sequence dataset first; no churn conclusion is possible without executable hold paths."
    total_delta = finite_sum(frame["hold_minus_sequence_pnl"])
    hold_better_fraction = float((frame["hold_minus_sequence_pnl"] > 0.0).mean())
    if total_delta > 0.0:
        return (
            "Train/evaluate a unified entry-plus-lifecycle sequence policy: flat actions wait/enter call/enter put; "
            "holding actions hold/exit. Use post-entry causal state, MFE/MAE shape, PnL velocity, gamma/theta decay, "
            "spread/liquidity, and account state so the model learns continuation utility instead of repeated fresh entries."
        )
    if hold_better_fraction > 0.35:
        return (
            "Churn is mixed. Keep the same sequence-policy direction, but first attribute harmful chains by side, "
            "time bucket, and exit reason to determine whether exit labels or entry re-selection are the dominant source."
        )
    return (
        "For these selected trades, exit/re-entry usually beat holding the first contract. The next lifecycle work should "
        "focus on missed entries or candidate filtering rather than forcing longer holds."
    )


def count_by(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, ""))
        out[key] = out.get(key, 0) + 1
    return out


def write_report(path: Path, payload: dict[str, Any], counterfactuals: pd.DataFrame) -> None:
    lines = [
        "# Protocol198 Lifecycle Churn Hold Counterfactual",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Primary same-side re-entry horizon: `{payload['primary_gap_minutes']}` minutes",
        f"- Interpretation: {payload['interpretation']}",
        f"- Next hypothesis: {payload['next_model_hypothesis']}",
        "",
        "## Horizon Sensitivity",
        "",
        "| gap | chains | trades in chains | share of trades | actual chain PnL | calls | puts |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["horizon_summary"]:
        lines.append(
            f"| {row['gap_minutes']:.0f}m | {row['chains']} | {row['trades_in_chains']} | "
            f"{pct(row['share_of_trades_in_chains'])} | {money(row['actual_chain_pnl'])} | "
            f"{row['call_chains']} | {row['put_chains']} |"
        )
    lines.extend(["", "## Split Counterfactuals", ""])
    if payload["split_summary"]:
        lines.extend(
            [
                "| split | chains | actual sequence | continuous hold | hold - sequence | hold better | median delta |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["split_summary"]:
            lines.append(
                f"| {row['reported_split']} | {row['chains']} | {money(row['actual_sequence_pnl'])} | "
                f"{money(row['counterfactual_hold_pnl'])} | {money(row['hold_minus_sequence_pnl'])} | "
                f"{pct(row['hold_better_fraction'])} | {money(row['median_hold_minus_sequence_pnl'])} |"
            )
    else:
        lines.append("No executable hold counterfactual rows were available.")
    lines.extend(["", "## Worst Harmful Churn Chains", ""])
    if not counterfactuals.empty:
        worst = counterfactuals.sort_values("hold_minus_sequence_pnl", ascending=False).head(12)
        lines.extend(
            [
                "| split | session | side | time | trades | actual | hold | delta | first reason | contracts |",
                "|---|---|---|---|---:|---:|---:|---:|---|---:|",
            ]
        )
        for _, row in worst.iterrows():
            lines.append(
                f"| {row['reported_split']} | {row['session']} | {row['right']} | {row['time_bucket']} | "
                f"{int(row['chain_length'])} | {money(row['actual_sequence_pnl'])} | "
                f"{money(row['counterfactual_hold_pnl'])} | {money(row['hold_minus_sequence_pnl'])} | "
                f"{row['first_exit_reason']} | {int(row['contract_count'])} |"
            )
    else:
        lines.append("No chains with executable hold paths were available.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Chain members: `{path.parent / 'chain_trade_members.csv'}`",
            f"- Hold counterfactuals: `{path.parent / 'churn_hold_counterfactuals.csv'}`",
            f"- Split summary: `{path.parent / 'split_summary.csv'}`",
            f"- Path skips: `{path.parent / 'path_skips.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_sum(values: Any) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(np.nansum(numeric.to_numpy(dtype=float))) if len(numeric) else 0.0


def money(value: Any) -> str:
    number = finite_float(value, math.nan)
    if not math.isfinite(number):
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


def pct(value: Any) -> str:
    number = finite_float(value, math.nan)
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
