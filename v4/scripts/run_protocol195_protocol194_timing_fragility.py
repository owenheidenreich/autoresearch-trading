"""Protocol195: timing fragility audit for Protocol194 selected trades.

Protocol194 established a five-seed full-action research candidate under
strict serial one-account replay and frozen Protocol081 exits. This runner does
not change the model. It attacks the candidate's timing realism by repricing
the selected trades with delayed entry/exit quotes from already-collected
normalized one-minute data, and with existing one-second audit files where
coverage exists.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOOP_ID = "v4_aplus_hypothesis_195_protocol194_timing_fragility"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_194_full_action_surface_edge_5seed_confirmation/"
    "protocol194_protocol081_5seed_serial_trades.csv"
)
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_HIGHRES_ROOT = Path("data/raw/audit/protocol101_highres_opra/cbbo-1s")
CONTRACT_MULTIPLIER = 100.0
HIGHRES_DELAYS_SECONDS = (1, 5, 15, 30, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--highres-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--highres-delay-seconds", action="append", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    highres_delays = tuple(sorted(set(args.highres_delay_seconds or HIGHRES_DELAYS_SECONDS)))

    trades = load_trades(args.trades)
    minute_rows, raw_symbol_map = build_minute_delay_rows(trades, normalized_dir=args.normalized_dir)
    highres_rows = build_highres_delay_rows(
        trades,
        raw_symbol_map=raw_symbol_map,
        highres_root=args.highres_root,
        delays=highres_delays,
    )

    minute_frame = pd.DataFrame(minute_rows)
    highres_frame = pd.DataFrame(highres_rows)
    minute_summary = summarize_minute(minute_frame)
    highres_summary = summarize_highres(highres_frame)
    payload = {
        "protocol": "195_protocol194_timing_fragility",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_trades": str(args.trades),
        "normalized_dir": str(args.normalized_dir),
        "highres_root": str(args.highres_root),
        "row_counts": {
            "trades": int(len(trades)),
            "minute_delay_rows": int(len(minute_rows)),
            "highres_delay_rows": int(len(highres_rows)),
        },
        "minute_delay_summary": minute_summary,
        "highres_delay_summary": highres_summary,
        "decision": decide(minute_summary, highres_summary),
        "interpretation": (
            "One-minute delay stress is a conservative coarse timing attack using the same normalized "
            "quote source as the training/replay path. One-second rows are only an already-collected "
            "coverage check; low one-second coverage is a data gap, not a model pass."
        ),
    }
    minute_frame.to_csv(args.out_dir / "minute_delay_rows.csv", index=False)
    highres_frame.to_csv(args.out_dir / "highres_delay_rows.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"no trades found in {path}")
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    exit_col = "candidate_exit_time" if "candidate_exit_time" in frame.columns else "exit_time"
    frame["exit_ts"] = pd.to_datetime(frame[exit_col], utc=True, errors="coerce")
    frame["session"] = frame["session"].astype(str)
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int)
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    frame["entry_ask"] = pd.to_numeric(frame["entry_ask"], errors="coerce")
    return frame.sort_values(["reported_split", "seed", "session", "decision_ts", "contract_id"]).reset_index(drop=True)


def build_minute_delay_rows(trades: pd.DataFrame, *, normalized_dir: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    rows: list[dict[str, Any]] = []
    raw_symbol_map: dict[tuple[str, str], str] = {}
    for session, group in trades.groupby("session", sort=True):
        path = find_normalized_path(normalized_dir, str(session))
        if path is None:
            rows.extend(minute_missing_row(trade, "missing_normalized_session") for _, trade in group.iterrows())
            continue
        try:
            quotes = pd.read_parquet(path, columns=["quote_time", "contract_id", "raw_symbol", "bid", "ask"])
        except Exception as exc:
            rows.extend(minute_missing_row(trade, f"unreadable_normalized_session:{type(exc).__name__}") for _, trade in group.iterrows())
            continue
        quotes["quote_time"] = pd.to_datetime(quotes["quote_time"], utc=True, errors="coerce")
        quotes["contract_id"] = quotes["contract_id"].astype(str)
        quotes["raw_symbol"] = quotes["raw_symbol"].astype(str)
        quotes["bid"] = pd.to_numeric(quotes["bid"], errors="coerce")
        quotes["ask"] = pd.to_numeric(quotes["ask"], errors="coerce")
        for contract_id, raw_symbol in quotes.dropna(subset=["raw_symbol"]).groupby("contract_id")["raw_symbol"].first().items():
            raw_symbol_map[(str(session), str(contract_id))] = str(raw_symbol)
        by_contract = {
            str(contract_id): part.sort_values("quote_time").reset_index(drop=True)
            for contract_id, part in quotes.groupby("contract_id", sort=False)
        }
        for _, trade in group.iterrows():
            contract_quotes = by_contract.get(str(trade["contract_id"]))
            if contract_quotes is None or contract_quotes.empty:
                rows.append(minute_missing_row(trade, "missing_contract_quotes"))
                continue
            rows.append(minute_delay_for_trade(trade, contract_quotes))
    return rows, raw_symbol_map


def build_highres_delay_rows(
    trades: pd.DataFrame,
    *,
    raw_symbol_map: dict[tuple[str, str], str],
    highres_root: Path,
    delays: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not highres_root.exists():
        return [
            highres_missing_row(trade, delay, "missing_highres_root")
            for _, trade in trades.iterrows()
            for delay in delays
        ]
    for session, group in trades.groupby("session", sort=True):
        raw_symbols = {
            raw_symbol_map.get((str(session), str(contract_id)))
            for contract_id in group["contract_id"].astype(str).unique()
        }
        raw_symbols = {symbol for symbol in raw_symbols if symbol}
        quotes = load_highres_quotes(highres_root / f"{session}.cbbo-1s.parquet", raw_symbols)
        by_symbol = {
            str(symbol): part.sort_values("ts_recv").reset_index(drop=True)
            for symbol, part in quotes.groupby("symbol", sort=False)
        }
        for _, trade in group.iterrows():
            raw_symbol = raw_symbol_map.get((str(session), str(trade["contract_id"])))
            if not raw_symbol:
                for delay in delays:
                    rows.append(highres_missing_row(trade, delay, "missing_raw_symbol_mapping"))
                continue
            symbol_quotes = by_symbol.get(str(raw_symbol))
            for delay in delays:
                if symbol_quotes is None or symbol_quotes.empty:
                    rows.append(highres_missing_row(trade, delay, "missing_highres_contract_quotes"))
                else:
                    rows.append(highres_delay_for_trade(trade, str(raw_symbol), symbol_quotes, delay_seconds=delay))
    return rows


def find_normalized_path(normalized_dir: Path, session: str) -> Path | None:
    matches = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return matches[0] if matches else None


def load_highres_quotes(path: Path, raw_symbols: set[str]) -> pd.DataFrame:
    if not path.exists() or not raw_symbols:
        return pd.DataFrame(columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"])
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"])
    frame = frame.reset_index()
    if "ts_recv" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "ts_recv"})
    required = {"ts_recv", "symbol", "bid_px_00", "ask_px_00"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"])
    frame = frame[frame["symbol"].astype(str).isin(raw_symbols)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"])
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True, errors="coerce")
    frame["bid_px_00"] = pd.to_numeric(frame["bid_px_00"], errors="coerce")
    frame["ask_px_00"] = pd.to_numeric(frame["ask_px_00"], errors="coerce")
    return frame[
        frame["ts_recv"].notna()
        & frame["bid_px_00"].notna()
        & frame["ask_px_00"].notna()
        & (frame["bid_px_00"] >= 0)
        & (frame["ask_px_00"] > 0)
        & (frame["ask_px_00"] >= frame["bid_px_00"])
    ][["ts_recv", "symbol", "bid_px_00", "ask_px_00"]].copy()


def minute_delay_for_trade(trade: pd.Series, quotes: pd.DataFrame) -> dict[str, Any]:
    entry_ask = finite_float(trade.get("entry_ask"), math.nan)
    pnl = finite_float(trade.get("pnl"), 0.0)
    if not math.isfinite(entry_ask):
        return minute_missing_row(trade, "missing_entry_ask")
    decision = pd.Timestamp(trade["decision_ts"])
    exit_ts = pd.Timestamp(trade["exit_ts"])
    original_exit_bid = entry_ask + pnl / CONTRACT_MULTIPLIER
    entry_quote = first_at_or_after(quotes, decision + pd.Timedelta(minutes=1), "ask")
    exit_quote = first_at_or_after(quotes, exit_ts + pd.Timedelta(minutes=1), "bid")
    entry_delay_pnl = None
    exit_delay_pnl = None
    both_delay_pnl = None
    if entry_quote is not None and pd.Timestamp(entry_quote["quote_time"]) <= exit_ts:
        entry_delay_pnl = (float(entry_quote["ask"]) * -1.0 + original_exit_bid) * CONTRACT_MULTIPLIER
    if exit_quote is not None:
        exit_delay_pnl = (float(exit_quote["bid"]) - entry_ask) * CONTRACT_MULTIPLIER
    if entry_quote is not None and exit_quote is not None and pd.Timestamp(entry_quote["quote_time"]) <= pd.Timestamp(exit_quote["quote_time"]):
        both_delay_pnl = (float(exit_quote["bid"]) - float(entry_quote["ask"])) * CONTRACT_MULTIPLIER
    return {
        **base_trade_fields(trade),
        "status": "ok",
        "entry_delay_pnl": entry_delay_pnl,
        "exit_delay_pnl": exit_delay_pnl,
        "both_delay_pnl": both_delay_pnl,
    }


def highres_delay_for_trade(trade: pd.Series, raw_symbol: str, quotes: pd.DataFrame, *, delay_seconds: int) -> dict[str, Any]:
    entry_ask = finite_float(trade.get("entry_ask"), math.nan)
    pnl = finite_float(trade.get("pnl"), 0.0)
    if not math.isfinite(entry_ask):
        return highres_missing_row(trade, delay_seconds, "missing_entry_ask")
    decision = pd.Timestamp(trade["decision_ts"])
    exit_ts = pd.Timestamp(trade["exit_ts"])
    entry_quote = first_at_or_after_highres(quotes, decision + pd.Timedelta(seconds=delay_seconds), "ask_px_00")
    exit_quote = first_at_or_after_highres(quotes, exit_ts + pd.Timedelta(seconds=delay_seconds), "bid_px_00")
    if entry_quote is None:
        return highres_missing_row(trade, delay_seconds, "missing_delayed_entry")
    if exit_quote is None:
        return highres_missing_row(trade, delay_seconds, "missing_delayed_exit")
    if pd.Timestamp(entry_quote["ts_recv"]) > pd.Timestamp(exit_quote["ts_recv"]):
        return highres_missing_row(trade, delay_seconds, "entry_after_exit")
    delayed = (float(exit_quote["bid_px_00"]) - float(entry_quote["ask_px_00"])) * CONTRACT_MULTIPLIER
    return {
        **base_trade_fields(trade),
        "raw_symbol": raw_symbol,
        "delay_seconds": int(delay_seconds),
        "status": "ok",
        "entry_time": pd.Timestamp(entry_quote["ts_recv"]).isoformat(),
        "exit_time": pd.Timestamp(exit_quote["ts_recv"]).isoformat(),
        "delayed_pnl": float(delayed),
        "pnl_delta": float(delayed - pnl),
    }


def first_at_or_after(quotes: pd.DataFrame, target: pd.Timestamp, column: str) -> pd.Series | None:
    valid = quotes[(quotes["quote_time"] >= target) & quotes[column].notna()]
    if column == "ask":
        valid = valid[valid["ask"] > 0]
    if column == "bid":
        valid = valid[valid["bid"] >= 0]
    return None if valid.empty else valid.iloc[0]


def first_at_or_after_highres(quotes: pd.DataFrame, target: pd.Timestamp, column: str) -> pd.Series | None:
    valid = quotes[(quotes["ts_recv"] >= target) & quotes[column].notna()]
    if column == "ask_px_00":
        valid = valid[valid["ask_px_00"] > 0]
    if column == "bid_px_00":
        valid = valid[valid["bid_px_00"] >= 0]
    return None if valid.empty else valid.iloc[0]


def minute_missing_row(trade: pd.Series, status: str) -> dict[str, Any]:
    return {
        **base_trade_fields(trade),
        "status": status,
        "entry_delay_pnl": None,
        "exit_delay_pnl": None,
        "both_delay_pnl": None,
    }


def highres_missing_row(trade: pd.Series, delay_seconds: int, status: str) -> dict[str, Any]:
    return {
        **base_trade_fields(trade),
        "raw_symbol": None,
        "delay_seconds": int(delay_seconds),
        "status": status,
        "delayed_pnl": None,
        "pnl_delta": None,
    }


def base_trade_fields(trade: pd.Series) -> dict[str, Any]:
    return {
        "candidate_uid": str(trade.get("candidate_uid", "")),
        "reported_split": str(trade.get("reported_split", "")),
        "seed": int(trade.get("seed", 0)),
        "session": str(trade.get("session", "")),
        "decision_time": pd.Timestamp(trade.get("decision_ts")).isoformat(),
        "exit_time": pd.Timestamp(trade.get("exit_ts")).isoformat(),
        "contract_id": str(trade.get("contract_id", "")),
        "right": str(trade.get("right", "")),
        "pnl": finite_float(trade.get("pnl"), 0.0),
        "entry_ask": finite_float(trade.get("entry_ask"), math.nan),
        "entry_premium": finite_float(trade.get("entry_premium"), math.nan),
    }


def summarize_minute(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if frame.empty:
        return out
    for split, group in frame.groupby("reported_split", sort=True):
        ok = group[group["status"].eq("ok")]
        original_by_seed = group.groupby("seed")["pnl"].sum()
        entry = ok.dropna(subset=["entry_delay_pnl"]).groupby("seed")["entry_delay_pnl"].sum()
        exit_ = ok.dropna(subset=["exit_delay_pnl"]).groupby("seed")["exit_delay_pnl"].sum()
        both = ok.dropna(subset=["both_delay_pnl"]).groupby("seed")["both_delay_pnl"].sum()
        out[str(split)] = {
            "rows": int(len(group)),
            "status_counts": group["status"].value_counts().to_dict(),
            "original_median_total_pnl": median_or_none(original_by_seed),
            "entry_delay_coverage": float(ok["entry_delay_pnl"].notna().sum() / max(len(group), 1)),
            "exit_delay_coverage": float(ok["exit_delay_pnl"].notna().sum() / max(len(group), 1)),
            "both_delay_coverage": float(ok["both_delay_pnl"].notna().sum() / max(len(group), 1)),
            "entry_delay_median_total_pnl": median_or_none(entry),
            "exit_delay_median_total_pnl": median_or_none(exit_),
            "both_delay_median_total_pnl": median_or_none(both),
            "entry_delay_total_delta": sum_delta(ok, "entry_delay_pnl"),
            "exit_delay_total_delta": sum_delta(ok, "exit_delay_pnl"),
            "both_delay_total_delta": sum_delta(ok, "both_delay_pnl"),
        }
    return out


def summarize_highres(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if frame.empty:
        return out
    for (split, delay_seconds), group in frame.groupby(["reported_split", "delay_seconds"], sort=True):
        ok = group[group["status"].eq("ok")]
        original_by_seed = group.groupby("seed")["pnl"].sum()
        delayed_by_seed = ok.groupby("seed")["delayed_pnl"].sum()
        out[f"{split}|{delay_seconds}s"] = {
            "reported_split": str(split),
            "delay_seconds": int(delay_seconds),
            "rows": int(len(group)),
            "status_counts": group["status"].value_counts().to_dict(),
            "coverage": float(len(ok) / max(len(group), 1)),
            "original_median_total_pnl": median_or_none(original_by_seed),
            "delayed_median_total_pnl": median_or_none(delayed_by_seed),
            "total_delta": sum_delta(ok, "delayed_pnl"),
        }
    return out


def decide(minute_summary: dict[str, Any], highres_summary: dict[str, Any]) -> str:
    required = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    if any(split not in minute_summary for split in required):
        return "blocked_missing_minute_delay_split"
    if any(float(minute_summary[split].get("both_delay_coverage") or 0.0) < 0.95 for split in required):
        return "blocked_incomplete_minute_delay_coverage"
    if any(float(minute_summary[split].get("both_delay_median_total_pnl") or -1.0) <= 0.0 for split in required):
        return "timing_fragile: one-minute both-side delay breaks at least one split"
    highres_coverages = [float(row.get("coverage") or 0.0) for row in highres_summary.values()]
    if highres_coverages and max(highres_coverages) >= 0.50:
        return "survives_coarse_timing_stress_with_partial_highres_support"
    return "survives_coarse_timing_stress_highres_gap_remains"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol195 Protocol194 Timing Fragility",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source trades: `{payload['source_trades']}`",
        f"- Interpretation: {payload['interpretation']}",
        "",
        "## One-Minute Delay Summary",
        "",
        "| split | rows | original | entry +1m | exit +1m | both +1m | both coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in payload["minute_delay_summary"].items():
        lines.append(
            f"| {split} | {row['rows']} | {money(row['original_median_total_pnl'])} | "
            f"{money(row['entry_delay_median_total_pnl'])} | {money(row['exit_delay_median_total_pnl'])} | "
            f"{money(row['both_delay_median_total_pnl'])} | {pct(row['both_delay_coverage'])} |"
        )
    lines.extend(["", "## One-Second Coverage Summary", ""])
    if payload["highres_delay_summary"]:
        lines.extend(
            [
                "| split | delay | rows | coverage | original | delayed | delta |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["highres_delay_summary"].values():
            lines.append(
                f"| {row['reported_split']} | {row['delay_seconds']}s | {row['rows']} | "
                f"{pct(row['coverage'])} | {money(row['original_median_total_pnl'])} | "
                f"{money(row['delayed_median_total_pnl'])} | {money(row['total_delta'])} |"
            )
    else:
        lines.append("No one-second rows were available.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Minute delay rows: `{path.parent / 'minute_delay_rows.csv'}`",
            f"- High-resolution delay rows: `{path.parent / 'highres_delay_rows.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def median_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return finite_float(series.median(), None)  # type: ignore[arg-type]


def sum_delta(frame: pd.DataFrame, delayed_column: str) -> float | None:
    ok = frame.dropna(subset=[delayed_column])
    if ok.empty:
        return None
    return finite_float((pd.to_numeric(ok[delayed_column], errors="coerce") - pd.to_numeric(ok["pnl"], errors="coerce")).sum())


def money(value: Any) -> str:
    if value is None:
        return "n/a"
    number = finite_float(value, math.nan)
    if not math.isfinite(number):
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    number = finite_float(value, math.nan)
    return "n/a" if not math.isfinite(number) else f"{number * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
