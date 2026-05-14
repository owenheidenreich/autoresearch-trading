"""Protocol 126: timing-fragility hardening for frozen Protocol 101.

No paid data, broker calls, or model training. This uses already-collected
high-resolution quote slices plus the frozen Protocol 101 replay rows to test
how much PnL depends on fast entry/exit timing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HIGHRES_REPLAY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_117_protocol101_targeted_highres_path_audit/report.json"
)
DEFAULT_HIGHRES_ROOT = Path("data/raw/audit/protocol101_highres_opra")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DELAYS_SECONDS = (0, 1, 5, 15, 30, 60)
CONTRACT_MULTIPLIER = 100.0
NY_TZ = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--highres-replay", type=Path, default=DEFAULT_HIGHRES_REPLAY)
    parser.add_argument("--highres-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--delay-seconds", action="append", type=int, default=None)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    delays = tuple(sorted(set(args.delay_seconds or DELAYS_SECONDS)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    replay = json.loads(args.highres_replay.read_text())
    rows = pd.DataFrame(replay.get("rows", []))
    rows = prepare_rows(rows)
    if rows.empty:
        raise SystemExit(f"no audited rows found in {args.highres_replay}")

    delay_rows = build_delay_rows(rows, highres_root=args.highres_root, delays=delays)
    delay_frame = pd.DataFrame(delay_rows)
    split_summary = summarize(delay_frame, ["split", "delay_seconds"])
    group_summary = build_group_summary(delay_frame)
    expiry_rule = decision_expiry_rule()
    decision = decide(split_summary)

    delay_frame.to_csv(args.out_dir / "timing_delay_rows.csv", index=False)
    pd.DataFrame(split_summary).to_csv(args.out_dir / "split_delay_summary.csv", index=False)
    pd.DataFrame(group_summary).to_csv(args.out_dir / "group_delay_summary.csv", index=False)
    payload = {
        "protocol": "126_protocol101_timing_fragility_hardening",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_highres_replay": str(args.highres_replay),
        "source_highres_root": str(args.highres_root),
        "delay_seconds": list(delays),
        "row_counts": {
            "input_audited_rows": int(len(rows)),
            "delay_rows": int(len(delay_rows)),
        },
        "split_delay_summary": split_summary,
        "group_delay_summary_rows": len(group_summary),
        "decision_expiry_rule": expiry_rule,
        "next_gate": (
            "Use this expiry rule in no-order live shadow and paper-order rehearsal. "
            "Protocol 101 is not paper-order ready until live rows prove quote/context freshness inside budget."
        ),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "delay_rows": str(args.out_dir / "timing_delay_rows.csv"),
            "split_summary": str(args.out_dir / "split_delay_summary.csv"),
            "group_summary": str(args.out_dir / "group_delay_summary.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload)
    print(json.dumps({"decision": decision, "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def prepare_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows = rows[rows.get("audit_status", "").eq("audited")].copy()
    if rows.empty:
        return rows
    rows["split"] = rows["split"].astype(str)
    rows["session"] = rows["session"].astype(str)
    rows["decision_ts"] = pd.to_datetime(rows["decision_time"], utc=True, format="mixed")
    rows["exit_ts"] = pd.to_datetime(rows["mandatory_exit_time_1s"], utc=True, format="mixed")
    rows["time_bucket"] = time_bucket(rows["decision_ts"])
    rows["right"] = rows["right"].astype(str)
    rows["side"] = np.where(rows["right"].eq("C"), "CALL", np.where(rows["right"].eq("P"), "PUT", rows["right"]))
    for column in ("entry_bid_1s", "entry_ask_1s", "exit_bid_1s", "exit_ask_1s", "pnl_1s", "offset"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["entry_spread"] = rows["entry_ask_1s"] - rows["entry_bid_1s"]
    mid = (rows["entry_ask_1s"] + rows["entry_bid_1s"]) / 2.0
    rows["entry_spread_over_mid"] = np.where(mid > 0, rows["entry_spread"] / mid, np.nan)
    rows["entry_premium"] = rows["entry_ask_1s"] * CONTRACT_MULTIPLIER
    rows["premium_bucket"] = pd.cut(
        rows["entry_premium"],
        bins=[-0.01, 1000, 2000, 3000, 4000, math.inf],
        labels=["<=1k", "1k-2k", "2k-3k", "3k-4k", "4k+"],
    ).astype(str)
    rows["spread_bucket"] = pd.cut(
        rows["entry_spread_over_mid"],
        bins=[-0.01, 0.01, 0.02, 0.04, 0.08, math.inf],
        labels=["<=1%", "1-2%", "2-4%", "4-8%", "8%+"],
    ).astype(str)
    rows["outcome_bucket"] = np.where(rows["pnl_1s"] >= 0, "winner", "loser")
    return rows


def time_bucket(ts: pd.Series) -> pd.Series:
    local = ts.dt.tz_convert(NY_TZ)
    minute = local.dt.hour * 60 + local.dt.minute
    return pd.Series(
        np.select(
            [minute < 600, minute < 690, minute < 810, minute <= 930],
            ["first30", "post_open_morning", "midday", "late_afternoon"],
            default="after_hours",
        ),
        index=ts.index,
    )


def build_delay_rows(rows: pd.DataFrame, *, highres_root: Path, delays: tuple[int, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for session, group in rows.groupby("session", sort=True):
        quotes = load_session_quotes(highres_root, str(session), set(group["raw_symbol"].astype(str)))
        for _, trade in group.iterrows():
            for delay in delays:
                out.append(reprice_trade_delay(trade, quotes, delay_seconds=delay))
    return out


def load_session_quotes(highres_root: Path, session: str, symbols: set[str]) -> pd.DataFrame:
    frames = []
    for schema in ("cbbo-1s", "cmbp-1"):
        path = highres_root / schema / f"{session}.{schema}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path).reset_index()
        if "ts_recv" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "ts_recv"})
        if "symbol" not in frame.columns:
            continue
        frame = frame[frame["symbol"].astype(str).isin(symbols)].copy()
        if frame.empty:
            continue
        frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True)
        frame["bid_px_00"] = pd.to_numeric(frame["bid_px_00"], errors="coerce")
        frame["ask_px_00"] = pd.to_numeric(frame["ask_px_00"], errors="coerce")
        frame = frame[
            frame["bid_px_00"].notna()
            & frame["ask_px_00"].notna()
            & (frame["bid_px_00"] > 0)
            & (frame["ask_px_00"] > 0)
            & (frame["ask_px_00"] > frame["bid_px_00"])
        ].copy()
        frame["schema"] = schema
        frames.append(frame[["ts_recv", "symbol", "bid_px_00", "ask_px_00", "schema"]])
    if not frames:
        return pd.DataFrame(columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00", "schema"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["symbol", "ts_recv"], keep="last").sort_values(["symbol", "ts_recv"])


def reprice_trade_delay(trade: pd.Series, quotes: pd.DataFrame, *, delay_seconds: int) -> dict[str, Any]:
    base = {
        "split": str(trade["split"]),
        "seed": int(trade["seed"]),
        "session": str(trade["session"]),
        "candidate_uid": str(trade["candidate_uid"]),
        "contract_id": str(trade["contract_id"]),
        "raw_symbol": str(trade["raw_symbol"]),
        "right": str(trade["right"]),
        "side": str(trade["side"]),
        "time_bucket": str(trade["time_bucket"]),
        "premium_bucket": str(trade["premium_bucket"]),
        "spread_bucket": str(trade["spread_bucket"]),
        "exit_reason": str(trade.get("exit_reason_1s", trade.get("exit_reason", ""))),
        "outcome_bucket": str(trade["outcome_bucket"]),
        "delay_seconds": int(delay_seconds),
        "original_pnl": float(trade["pnl_1s"]),
        "entry_premium": float(trade["entry_premium"]),
    }
    if delay_seconds == 0:
        return {
            **base,
            "status": "ok",
            "entry_time": pd.Timestamp(trade["entry_time_1s"]).isoformat(),
            "exit_time": pd.Timestamp(trade["mandatory_exit_time_1s"]).isoformat(),
            "entry_ask": float(trade["entry_ask_1s"]),
            "exit_bid": float(trade["exit_bid_1s"]),
            "delayed_pnl": float(trade["pnl_1s"]),
            "pnl_delta": 0.0,
        }
    symbol_quotes = quotes[quotes["symbol"].astype(str).eq(str(trade["raw_symbol"]))].copy()
    if symbol_quotes.empty:
        return {**base, "status": "missing_quote_path", "delayed_pnl": None, "pnl_delta": None}
    entry_target = pd.Timestamp(trade["decision_ts"]) + pd.Timedelta(seconds=delay_seconds)
    exit_target = pd.Timestamp(trade["exit_ts"]) + pd.Timedelta(seconds=delay_seconds)
    entry = first_at_or_after(symbol_quotes, entry_target, "ask_px_00")
    exit_row = first_at_or_after(symbol_quotes, exit_target, "bid_px_00")
    if entry is None:
        return {**base, "status": "missing_delayed_entry", "delayed_pnl": None, "pnl_delta": None}
    if exit_row is None:
        return {**base, "status": "missing_delayed_exit", "delayed_pnl": None, "pnl_delta": None}
    if pd.Timestamp(entry["ts_recv"]) > pd.Timestamp(exit_row["ts_recv"]):
        return {**base, "status": "entry_after_exit", "delayed_pnl": None, "pnl_delta": None}
    delayed = (float(exit_row["bid_px_00"]) - float(entry["ask_px_00"])) * CONTRACT_MULTIPLIER
    return {
        **base,
        "status": "ok",
        "entry_time": pd.Timestamp(entry["ts_recv"]).isoformat(),
        "exit_time": pd.Timestamp(exit_row["ts_recv"]).isoformat(),
        "entry_ask": float(entry["ask_px_00"]),
        "exit_bid": float(exit_row["bid_px_00"]),
        "delayed_pnl": float(delayed),
        "pnl_delta": float(delayed - float(trade["pnl_1s"])),
    }


def first_at_or_after(quotes: pd.DataFrame, target: pd.Timestamp, column: str) -> pd.Series | None:
    eligible = quotes[(quotes["ts_recv"] >= target) & quotes[column].notna()]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    if frame.empty:
        return rows
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        ok = group[group["status"].eq("ok")]
        row = {col: json_ready(value) for col, value in zip(group_cols, key)}
        row.update(
            {
                "rows": int(len(group)),
                "coverage": float(len(ok) / max(len(group), 1)),
                "original_pnl": finite_float(ok["original_pnl"].sum()) if not ok.empty else None,
                "delayed_pnl": finite_float(ok["delayed_pnl"].sum()) if not ok.empty else None,
                "pnl_delta": finite_float(ok["pnl_delta"].sum()) if not ok.empty else None,
                "median_trade_delta": finite_float(ok["pnl_delta"].median()) if not ok.empty else None,
                "positive_trade_fraction": finite_float((ok["delayed_pnl"] > 0).mean()) if not ok.empty else None,
            }
        )
        rows.append(row)
    return rows


def build_group_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in ("side", "time_bucket", "premium_bucket", "spread_bucket", "exit_reason", "outcome_bucket"):
        grouped = summarize(frame, [dimension, "delay_seconds"])
        for row in grouped:
            row["dimension"] = dimension
            row["value"] = row.pop(dimension)
            rows.append(row)
    return rows


def decision_expiry_rule() -> dict[str, Any]:
    return {
        "max_option_quote_age_ms": 1500,
        "max_context_age_ms": 5000,
        "reject_missing_bid_ask": True,
        "reject_zero_or_negative_bid_ask": True,
        "reject_locked_or_crossed_quotes": True,
        "max_entry_ask_move": 0.25,
        "allowed_root": "SPXW",
        "allowed_settlement_style": "PM",
        "requires_live_shadow_proof_before_orders": True,
    }


def decide(split_summary: list[dict[str, Any]]) -> str:
    critical = [row for row in split_summary if int(row["delay_seconds"]) in {1, 5}]
    if not critical:
        return "blocked_no_timing_coverage"
    if any(float(row.get("coverage") or 0.0) < 0.95 for row in critical):
        return "blocked_incomplete_timing_coverage"
    if any(float(row.get("delayed_pnl") or -1.0) <= 0 for row in critical):
        return "timing_fragile_requires_live_shadow_before_paper"
    return "pass_historical_timing_hardening_live_shadow_required"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 126: Protocol101 Timing Fragility Hardening",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source replay: `{payload['source_highres_replay']}`",
        f"- Delay seconds: `{payload['delay_seconds']}`",
        "",
        "## Split Delay Summary",
        "",
        "| split | delay_s | coverage | original_pnl | delayed_pnl | pnl_delta | positive_trades |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["split_delay_summary"]:
        lines.append(
            "| "
            f"{row['split']} | {row['delay_seconds']} | {pct(row['coverage'])} | "
            f"{money(row['original_pnl'])} | {money(row['delayed_pnl'])} | "
            f"{money(row['pnl_delta'])} | {pct(row['positive_trade_fraction'])} |"
        )
    rule = payload["decision_expiry_rule"]
    lines.extend(
        [
            "",
            "## Future Decision Expiry Rule",
            "",
            f"- Max option quote age: `{rule['max_option_quote_age_ms']}ms`",
            f"- Max context age: `{rule['max_context_age_ms']}ms`",
            f"- Max entry ask move before entry: `${rule['max_entry_ask_move']:.2f}`",
            "- Reject missing, zero, locked, crossed, stale, non-SPXW, or non-PM-settled quotes.",
            "",
            "## Outputs",
            "",
            f"- Delay rows: `{payload['outputs']['delay_rows']}`",
            f"- Split summary: `{payload['outputs']['split_summary']}`",
            f"- Group summary: `{payload['outputs']['group_summary']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    marker = "## 2026-05-14 Protocol 126 Protocol101 Timing Fragility Hardening"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Repriced frozen Protocol101 selected trades under sub-minute entry/exit delay stress using already-collected high-resolution quote data.
Reason: Protocol 114 found one-minute timing fragility; the project needed a tighter timing budget and live-style expiry rule before paper trading.
Data Used: Existing Protocol 117 high-resolution replay and local high-resolution quote files only. No paid data was downloaded, no broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Delay rows {payload['row_counts']['delay_rows']}; report {payload['outputs']['report']}.
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    existing = path.read_text() if path.exists() else ""
    if marker not in existing:
        path.write_text(existing.rstrip() + entry + "\n")
        return
    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        path.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_float(value)
    return value


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def money(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


def pct(value: Any) -> str:
    number = finite_float(value)
    return "n/a" if number is None else f"{number * 100:.1f}%"


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
