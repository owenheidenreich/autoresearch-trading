"""DATASET_POSITION_STATE_ACTION_ADVANTAGE_V1.

Build holding-state action-advantage labels for the unified SPXW 0DTE game.
The dataset answers: once the bot is in a selected contract, is it better to
exit now at bid and free the slot, or continue holding the current option?

This runner uses the Protocol270 full-surface flat oracle for future slot value
and normalized official-context quote paths for executable current-position
state. It is a dataset builder, not a model, not a promotion decision, and it
does not download paid data or call a broker.
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

from v4.model.unified_serial_game import CONTRACT_MULTIPLIER, hold_exit_opportunity_advantage_path


ROLE_LABEL = "DATASET_POSITION_STATE_ACTION_ADVANTAGE_V1"
HISTORICAL_ID = "Protocol274"
DEFAULT_ACTION_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset")
NY = ZoneInfo("America/New_York")
SESSION_OPEN_MINUTE = 9 * 60 + 30
FORCED_FLAT_MINUTE = 15 * 60 + 55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-dataset", type=Path, default=DEFAULT_ACTION_DATASET)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-trades", type=int, default=0)
    parser.add_argument("--max-sessions-per-split", type=int, default=0)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    actions = load_action_dataset(args.action_dataset)
    selected = select_oracle_entries(actions, max_trades=int(args.max_trades), max_sessions_per_split=int(args.max_sessions_per_split))
    flat_values = build_flat_value_maps(actions)
    rows, skips = build_position_state_rows(
        selected,
        flat_values,
        normalized_dir=args.normalized_dir,
        forced_flat_time=str(args.forced_flat_time),
    )
    frame = pd.DataFrame(rows)
    skip_frame = pd.DataFrame(skips)
    out_path = args.out_dir / "position_state_action_advantage.parquet"
    if not frame.empty:
        frame.to_parquet(out_path, index=False)
    else:
        pd.DataFrame().to_parquet(out_path, index=False)
    skip_frame.to_csv(args.out_dir / "path_skips.csv", index=False)
    split_summary = summarize(frame)
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "dataset / holding-state hold-vs-exit action-advantage labels",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V2_INPUT",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": {
            "action_advantage_dataset": str(args.action_dataset),
            "normalized_dir": str(args.normalized_dir),
        },
        "output_dataset": str(out_path),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "label_semantics": {
            "q_exit": "current executable bid PnL plus future flat-slot oracle value",
            "q_hold": "best later executable exit value plus future flat-slot oracle value after holding at least one more quote step",
            "a_hold": "q_hold minus q_exit",
            "a_switch": "q_exit minus q_hold",
        },
        "row_counts": {
            "action_dataset_rows": int(len(actions)),
            "oracle_entry_trades": int(len(selected)),
            "position_state_rows": int(len(frame)),
            "path_skips": int(len(skip_frame)),
        },
        "path_skip_counts": count_by(skip_frame, "skip_reason"),
        "split_summary": split_summary.to_dict("records"),
        "decision": decide(frame),
        "next_experiment": "Train the unified policy with both flat enter advantages and holding-state hold/exit advantages, then compare to PAPER_DEFAULT_PROTOCOL101.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "rows": int(len(frame)), "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_action_dataset(path: Path) -> pd.DataFrame:
    columns = [
        "split",
        "session",
        "decision_time",
        "decision_dt",
        "candidate_uid",
        "trade_uid",
        "contract_id",
        "right",
        "offset",
        "entry_ask",
        "entry_bid",
        "entry_mid",
        "entry_spread",
        "entry_premium",
        "entry_underlying_price",
        "entry_iv",
        "entry_delta",
        "entry_gamma",
        "entry_theta",
        "entry_gamma_theta_ratio",
        "entry_theta_burden",
        "candidate_exit_dt",
        "candidate_exit_time",
        "candidate_pnl",
        "candidate_exit_reason",
        "oracle_action",
        "oracle_action_uid",
        "q_wait",
        "q_enter",
        "a_enter",
        "session_oracle_value",
        "best_enter_value_at_decision",
        "runtime_feature_scope",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["candidate_exit_dt"] = pd.to_datetime(frame["candidate_exit_dt"], utc=True, errors="coerce")
    for column in [
        "offset",
        "entry_ask",
        "entry_bid",
        "entry_mid",
        "entry_spread",
        "entry_premium",
        "entry_underlying_price",
        "entry_iv",
        "entry_delta",
        "entry_gamma",
        "entry_theta",
        "entry_gamma_theta_ratio",
        "entry_theta_burden",
        "candidate_pnl",
        "q_wait",
        "q_enter",
        "a_enter",
        "session_oracle_value",
        "best_enter_value_at_decision",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["split", "session", "decision_dt", "contract_id", "entry_ask"]).reset_index(drop=True)


def select_oracle_entries(frame: pd.DataFrame, *, max_trades: int, max_sessions_per_split: int) -> pd.DataFrame:
    selected = frame[frame["oracle_action"].astype(str).eq("enter") & frame["candidate_uid"].astype(str).eq(frame["oracle_action_uid"].astype(str))].copy()
    selected = selected.sort_values(["split", "session", "decision_dt", "contract_id"]).reset_index(drop=True)
    if max_sessions_per_split > 0 and not selected.empty:
        pieces = []
        for split, group in selected.groupby("split", sort=True):
            sessions = sorted(group["session"].astype(str).unique())[:max_sessions_per_split]
            pieces.append(group[group["session"].astype(str).isin(sessions)])
        selected = pd.concat(pieces, ignore_index=True) if pieces else selected.iloc[0:0]
    if max_trades > 0:
        selected = selected.head(max_trades).copy()
    return selected.reset_index(drop=True)


def build_flat_value_maps(frame: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    values: dict[tuple[str, str], pd.DataFrame] = {}
    event_values = (
        frame[["split", "session", "decision_dt", "session_oracle_value"]]
        .dropna(subset=["decision_dt"])
        .drop_duplicates(["split", "session", "decision_dt"])
        .sort_values(["split", "session", "decision_dt"])
    )
    for key, group in event_values.groupby(["split", "session"], sort=False):
        values[(str(key[0]), str(key[1]))] = group[["decision_dt", "session_oracle_value"]].reset_index(drop=True)
    return values


def build_position_state_rows(
    selected: pd.DataFrame,
    flat_values: dict[tuple[str, str], pd.DataFrame],
    *,
    normalized_dir: Path,
    forced_flat_time: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for session, session_trades in selected.groupby("session", sort=True):
        contracts = set(session_trades["contract_id"].astype(str).unique())
        quotes = load_session_quotes(normalized_dir, str(session), contracts)
        if quotes.empty:
            skips.extend(base_skip(trade, "missing_session_or_contract_quotes") for _, trade in session_trades.iterrows())
            continue
        by_contract = {str(contract_id): part.sort_values("quote_time").reset_index(drop=True) for contract_id, part in quotes.groupby("contract_id", sort=False)}
        forced_flat = forced_flat_timestamp(str(session), forced_flat_time)
        for _, trade in session_trades.iterrows():
            key = (str(trade["split"]), str(trade["session"]))
            result, skip = labels_for_trade(trade, by_contract.get(str(trade["contract_id"])), flat_values.get(key), forced_flat)
            if skip:
                skips.append(skip)
            else:
                rows.extend(result)
    return rows, skips


def load_session_quotes(normalized_dir: Path, session: str, contract_ids: set[str]) -> pd.DataFrame:
    paths = find_normalized_paths(normalized_dir, session)
    if not paths:
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
        "vega",
    ]
    frame = pd.DataFrame()
    for path in paths:
        try:
            frame = pd.read_parquet(path, columns=columns)
            break
        except Exception:
            try:
                fallback = pd.read_parquet(path)
            except Exception:
                continue
            for column in columns:
                if column not in fallback.columns:
                    fallback[column] = np.nan
            frame = fallback[columns].copy()
            break
    if frame.empty:
        return pd.DataFrame()
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
        & frame["bid"].ge(0.0)
        & frame["ask"].gt(0.0)
        & frame["ask"].ge(frame["bid"])
    ].copy()


def labels_for_trade(
    trade: pd.Series,
    quotes: pd.DataFrame | None,
    flat_values: pd.DataFrame | None,
    forced_flat: pd.Timestamp,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if quotes is None or quotes.empty:
        return [], base_skip(trade, "missing_contract_quotes")
    if flat_values is None or flat_values.empty:
        return [], base_skip(trade, "missing_flat_value_map")
    entry_time = pd.Timestamp(trade["decision_dt"])
    entry_ask = finite(trade.get("entry_ask"))
    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        return [], base_skip(trade, "invalid_entry_ask")
    path = quotes[(quotes["quote_time"] >= entry_time) & (quotes["quote_time"] <= forced_flat)].sort_values("quote_time").reset_index(drop=True)
    if path.empty:
        return [], base_skip(trade, "missing_post_entry_path")
    bid = pd.to_numeric(path["bid"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(bid)
    if not valid.any():
        return [], base_skip(trade, "invalid_path_bid")
    path = path[valid].reset_index(drop=True)
    bid = bid[valid]
    future_flat = future_flat_values(path["quote_time"], flat_values)
    labels = hold_exit_opportunity_advantage_path(bid, entry_ask, future_flat)
    path = pd.concat([path.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    path["mfe_to_now"] = path["current_pnl"].cummax()
    path["mae_to_now"] = path["current_pnl"].cummin()
    path["giveback_from_mfe"] = path["mfe_to_now"] - path["current_pnl"]
    path["giveback_fraction"] = np.where(path["mfe_to_now"].abs() > 1e-9, path["giveback_from_mfe"] / path["mfe_to_now"].abs(), 0.0)
    path["pnl_velocity_1"] = path["current_pnl"].diff(1).fillna(0.0)
    path["pnl_velocity_3"] = path["current_pnl"].diff(3).fillna(0.0) / 3.0
    path["pnl_velocity_5"] = path["current_pnl"].diff(5).fillna(0.0) / 5.0
    mfe_idx = path["current_pnl"].cummax().groupby(path["current_pnl"].cummax()).cumcount()
    del mfe_idx
    rows = []
    best_seen_idx = 0
    for idx, row in path.iterrows():
        if float(row["current_pnl"]) >= float(path.loc[best_seen_idx, "current_pnl"]):
            best_seen_idx = int(idx)
        quote_time = pd.Timestamp(row["quote_time"])
        local = quote_time.tz_convert(NY)
        minute = local.hour * 60 + local.minute
        rows.append(
            {
                "split": str(trade["split"]),
                "session": str(trade["session"]),
                "candidate_uid": str(trade["candidate_uid"]),
                "trade_uid": str(trade["trade_uid"]),
                "contract_id": str(trade["contract_id"]),
                "right": str(trade["right"]),
                "offset": finite(trade.get("offset")),
                "entry_time": entry_time.isoformat(),
                "state_time": quote_time.isoformat(),
                "state_index": int(idx),
                "minutes_since_entry": float((quote_time - entry_time).total_seconds() / 60.0),
                "minutes_since_open": float(minute - SESSION_OPEN_MINUTE),
                "minutes_to_forced_flat": float((forced_flat - quote_time).total_seconds() / 60.0),
                "entry_ask": entry_ask,
                "entry_premium": finite(trade.get("entry_premium")),
                "entry_a_enter": finite(trade.get("a_enter")),
                "entry_q_wait": finite(trade.get("q_wait")),
                "entry_q_enter": finite(trade.get("q_enter")),
                "entry_underlying_price": finite(trade.get("entry_underlying_price")),
                "entry_iv": finite(trade.get("entry_iv")),
                "entry_delta": finite(trade.get("entry_delta")),
                "entry_gamma": finite(trade.get("entry_gamma")),
                "entry_theta": finite(trade.get("entry_theta")),
                "bid": finite(row.get("bid")),
                "ask": finite(row.get("ask")),
                "mid": finite(row.get("mid")),
                "spread": finite(row.get("ask")) - finite(row.get("bid")),
                "spread_frac": safe_div(finite(row.get("ask")) - finite(row.get("bid")), finite(row.get("mid"))),
                "bid_size": finite(row.get("bid_size")),
                "ask_size": finite(row.get("ask_size")),
                "quote_gap_seconds": finite(row.get("quote_gap_seconds")),
                "underlying_price": finite(row.get("underlying_price")),
                "iv": finite(row.get("iv")),
                "delta": finite(row.get("delta")),
                "gamma": finite(row.get("gamma")),
                "theta": finite(row.get("theta")),
                "vega": finite(row.get("vega")),
                "gamma_theta_ratio": safe_div(abs(finite(row.get("gamma"))), abs(finite(row.get("theta")))),
                "theta_over_mid": safe_div(abs(finite(row.get("theta"))), abs(finite(row.get("mid")))),
                "current_pnl": finite(row.get("current_pnl")),
                "mfe_to_now": finite(row.get("mfe_to_now")),
                "mae_to_now": finite(row.get("mae_to_now")),
                "giveback_from_mfe": finite(row.get("giveback_from_mfe")),
                "giveback_fraction": finite(row.get("giveback_fraction")),
                "pnl_velocity_1": finite(row.get("pnl_velocity_1")),
                "pnl_velocity_3": finite(row.get("pnl_velocity_3")),
                "pnl_velocity_5": finite(row.get("pnl_velocity_5")),
                "time_since_mfe_minutes": float(idx - best_seen_idx),
                "future_flat_value_after_exit": float(future_flat[int(idx)]),
                "q_exit": finite(row.get("q_exit")),
                "q_hold": finite(row.get("q_hold")),
                "a_hold": finite(row.get("a_hold")),
                "a_switch": finite(row.get("a_switch")),
                "oracle_holding_action": str(row.get("oracle_holding_action")),
                "label_source": "position_state_action_advantage_v1",
                "future_path_columns_used_as_features": False,
            }
        )
    return rows, None


def future_flat_values(times: pd.Series, flat_values: pd.DataFrame) -> np.ndarray:
    decision_times = pd.to_datetime(flat_values["decision_dt"], utc=True).astype("int64").to_numpy()
    values = pd.to_numeric(flat_values["session_oracle_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(len(times), dtype=float)
    for idx, ts in enumerate(pd.to_datetime(times, utc=True)):
        pos = int(np.searchsorted(decision_times, int(ts.value), side="right"))
        out[idx] = float(values[pos]) if pos < len(values) else 0.0
    return out


def find_normalized_paths(normalized_dir: Path, session: str) -> list[Path]:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return list(dict.fromkeys([*preferred, *fallback]))


def forced_flat_timestamp(session: str, forced_flat_time: str) -> pd.Timestamp:
    hour, minute = [int(part) for part in forced_flat_time.split(":", 1)]
    return pd.Timestamp(session).replace(hour=hour, minute=minute, tzinfo=NY).tz_convert("UTC")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["split", "rows", "trades", "sessions", "hold_fraction", "exit_fraction", "median_a_hold", "mean_a_hold", "positive_hold_advantage_rows"])
    rows = []
    for split, group in frame.groupby("split", sort=True):
        rows.append(
            {
                "split": str(split),
                "rows": int(len(group)),
                "trades": int(group["candidate_uid"].nunique()),
                "sessions": int(group["session"].nunique()),
                "hold_fraction": float(group["oracle_holding_action"].eq("hold").mean()),
                "exit_fraction": float(group["oracle_holding_action"].eq("exit").mean()),
                "median_a_hold": float(pd.to_numeric(group["a_hold"], errors="coerce").median()),
                "mean_a_hold": float(pd.to_numeric(group["a_hold"], errors="coerce").mean()),
                "positive_hold_advantage_rows": int((pd.to_numeric(group["a_hold"], errors="coerce") > 0.0).sum()),
                "median_current_pnl": float(pd.to_numeric(group["current_pnl"], errors="coerce").median()),
                "median_future_flat_value_after_exit": float(pd.to_numeric(group["future_flat_value_after_exit"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def decide(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "blocked_position_state_action_advantage_empty"
    hold_fraction = float(frame["oracle_holding_action"].eq("hold").mean())
    if 0.05 <= hold_fraction <= 0.95:
        return "position_state_action_advantage_dataset_ready_for_lifecycle_training"
    return "position_state_action_advantage_dataset_ready_but_label_balance_extreme"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being prepared: `{payload['candidate_label']}`",
        f"Paper default baseline: `{payload['paper_default_label']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Label Semantics",
        "",
        "- `q_exit`: current bid exit PnL plus the future flat-slot oracle value.",
        "- `q_hold`: best later bid exit after holding at least one more quote step plus future flat-slot value.",
        "- `a_hold`: `q_hold - q_exit`; positive means continuing the current position is better than freeing the slot now.",
        "- `a_switch`: `q_exit - q_hold`; positive means exit/free-slot is better.",
        "",
        "## Split Summary",
        "",
        "| split | rows | trades | sessions | hold frac | median A_hold | median current PnL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_summary"]:
        lines.append(
            f"| {row['split']} | {row['rows']} | {row['trades']} | {row['sessions']} | "
            f"{row['hold_fraction']:.3f} | {row['median_a_hold']:.2f} | {row['median_current_pnl']:.2f} |"
        )
    lines.extend(["", "## Outputs", "", f"- Dataset: `{payload['output_dataset']}`", f"- Summary: `{path.parent / 'summary.json'}`"])
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    text = ledger.read_text()
    if marker in text:
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


def count_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).items()}


def base_skip(trade: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "split": str(trade.get("split", "")),
        "session": str(trade.get("session", "")),
        "candidate_uid": str(trade.get("candidate_uid", "")),
        "contract_id": str(trade.get("contract_id", "")),
        "right": str(trade.get("right", "")),
        "skip_reason": reason,
    }


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_div(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-9:
        return 0.0
    return float(num / den)


if __name__ == "__main__":
    raise SystemExit(main())
