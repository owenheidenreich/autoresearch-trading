"""Replay selected trades on available high-resolution top-of-book slices.

The v4 labels use CBBO-1m ask-entry / bid-exit paths. This script checks a
selected-trade file against the narrower CBBO-1s or CMBP-1 audit data where
available. It does not download data; missing sessions are reported as
unaudited.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


_NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ReplayPolicy:
    stop_loss_pct: float = 0.50
    take_profit_pct: float = 1.00
    max_hold_minutes: int = 25
    forced_flat_time: str = "15:55"
    contract_multiplier: float = 100.0
    fee_per_contract: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-trades",
        nargs="*",
        type=Path,
        default=[Path("v4/audit/autoresearch/v4_aplus_strict_entry_stress_003b/selected_trades_march.json")],
    )
    parser.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized"))
    parser.add_argument("--cbbo-1s-dir", type=Path, default=Path("data/raw/audit/opra_spxw_cbbo_1s"))
    parser.add_argument("--highres-root", type=Path, default=Path("data/raw/audit/protocol101_highres_opra"))
    parser.add_argument("--highres-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_strict_entry_stress_003b/one_second_path_audit"))
    parser.add_argument("--stop-loss-pct", type=float, default=0.50)
    parser.add_argument("--take-profit-pct", type=float, default=1.00)
    parser.add_argument("--max-hold-minutes", type=int, default=25)
    parser.add_argument("--fee-per-contract", type=float, default=0.0)
    parser.add_argument(
        "--replay-mode",
        choices=("fixed_policy", "lifecycle_exit"),
        default="fixed_policy",
        help="Audit either the fixed stop/target/deadline policy or selected lifecycle exits.",
    )
    return parser.parse_args()


def _load_selected(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        split = path.name.removeprefix("selected_trades_").removesuffix(".json")
        data = json.loads(path.read_text())
        for row in data:
            rows.append({"source_file": str(path), "source_split": split, **row})
    return rows


def _raw_symbol_map(normalized_dir: Path, session: str) -> dict[str, str]:
    candidates = (
        normalized_dir / f"databento_spxw_0dte_{session}_derived_context.parquet",
        normalized_dir / f"databento_spxw_0dte_{session}_official_context.parquet",
        normalized_dir / f"databento_spxw_0dte_{session}.parquet",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return {}
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["contract_id", "raw_symbol"])
    frame = frame.dropna().drop_duplicates("contract_id")
    return dict(zip(frame["contract_id"].astype(str), frame["raw_symbol"].astype(str)))


def _load_quotes(path: Path, symbols: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path).reset_index()
    if "ts_recv" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "ts_recv"})
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True)
    frame = frame[frame["symbol"].astype(str).isin(symbols)].copy()
    frame["bid_px_00"] = pd.to_numeric(frame["bid_px_00"], errors="coerce")
    frame["ask_px_00"] = pd.to_numeric(frame["ask_px_00"], errors="coerce")
    frame = frame[
        frame["bid_px_00"].notna()
        & frame["ask_px_00"].notna()
        & (frame["ask_px_00"] >= frame["bid_px_00"])
        & (frame["ask_px_00"] > 0)
    ].copy()
    return frame.sort_values(["symbol", "ts_recv"]).reset_index(drop=True)


def _schema_by_session(manifest_path: Path | None) -> dict[str, str]:
    if manifest_path is None:
        return {}
    manifest = json.loads(manifest_path.read_text())
    return {str(row["session"]): str(row["schema"]) for row in manifest.get("requests", [])}


def _quote_paths(
    *,
    session: str,
    schema: str,
    cbbo_1s_dir: Path,
    highres_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    if schema == "cbbo-1s":
        base = cbbo_1s_dir / f"{session}.cbbo-1s.parquet"
        if base.exists():
            paths.append(base)
    overlay = highres_root / schema / f"{session}.{schema}.parquet"
    if overlay.exists() and overlay not in paths:
        paths.append(overlay)
    return paths


def _load_session_quotes(paths: list[Path], symbols: set[str]) -> pd.DataFrame:
    frames = [_load_quotes(path, symbols) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["symbol", "ts_recv"], keep="last")
    return out.sort_values(["symbol", "ts_recv"]).reset_index(drop=True)


def _deadline(decision_time: pd.Timestamp, policy: ReplayPolicy) -> pd.Timestamp:
    max_hold = decision_time + pd.Timedelta(minutes=policy.max_hold_minutes)
    hour, minute = [int(x) for x in policy.forced_flat_time.split(":", 1)]
    local_day = decision_time.tz_convert(_NY).date()
    forced = pd.Timestamp(local_day).replace(hour=hour, minute=minute, tzinfo=_NY).tz_convert("UTC")
    return min(max_hold, forced)


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _pnl_1m_value(trade: dict) -> float:
    if "dynamic_pnl" in trade:
        return float(trade["dynamic_pnl"])
    if "candidate_pnl" in trade:
        return float(trade["candidate_pnl"])
    return float(trade["pnl"])


def _lifecycle_exit_time(trade: dict, decision_time: pd.Timestamp) -> pd.Timestamp | None:
    exit_time = trade.get("exit_time")
    if exit_time:
        return _utc_timestamp(exit_time)
    candidate_exit_time = trade.get("candidate_exit_time")
    if candidate_exit_time:
        return _utc_timestamp(candidate_exit_time)
    hold_minutes = trade.get("hold_minutes")
    if hold_minutes is None:
        return None
    return decision_time + pd.Timedelta(minutes=float(hold_minutes))


def _replay_one(
    trade: dict,
    quotes: pd.DataFrame,
    *,
    policy: ReplayPolicy,
) -> dict:
    decision_time = _utc_timestamp(trade["decision_time"])
    symbol = trade["raw_symbol"]
    symbol_quotes = quotes[quotes["symbol"] == symbol].copy()
    if symbol_quotes.empty:
        return {**trade, "audit_status": "missing_symbol"}
    entry_candidates = symbol_quotes[symbol_quotes["ts_recv"] <= decision_time]
    if entry_candidates.empty:
        return {**trade, "audit_status": "missing_entry_quote"}
    entry = entry_candidates.iloc[-1]
    entry_bid = float(entry["bid_px_00"])
    entry_ask = float(entry["ask_px_00"])
    end = _deadline(decision_time, policy)
    future = symbol_quotes[
        (symbol_quotes["ts_recv"] > decision_time)
        & (symbol_quotes["ts_recv"] <= end)
    ].copy()
    if future.empty:
        return {**trade, "audit_status": "missing_exit_path", "entry_ask_1s": entry_ask}

    stop_bid = entry_ask * (1.0 - policy.stop_loss_pct)
    target_bid = entry_ask * (1.0 + policy.take_profit_pct)
    exit_row = future.iloc[-1]
    exit_reason = "deadline"
    for _, row in future.iterrows():
        bid = float(row["bid_px_00"])
        if bid <= stop_bid:
            exit_row = row
            exit_reason = "stop"
            break
        if bid >= target_bid:
            exit_row = row
            exit_reason = "target"
            break
    exit_bid = float(exit_row["bid_px_00"])
    exit_ask = float(exit_row["ask_px_00"])
    net_1s = (exit_bid - entry_ask) * policy.contract_multiplier
    if policy.fee_per_contract:
        net_1s -= 2.0 * policy.fee_per_contract
    pnl_1m = _pnl_1m_value(trade)
    return {
        **trade,
        "audit_status": "audited",
        "entry_time_1s": pd.Timestamp(entry["ts_recv"]).isoformat(),
        "exit_time_1s": pd.Timestamp(exit_row["ts_recv"]).isoformat(),
        "exit_reason_1s": exit_reason,
        "entry_bid_1s": entry_bid,
        "entry_ask_1s": entry_ask,
        "exit_bid_1s": exit_bid,
        "exit_ask_1s": exit_ask,
        "pnl_1s": float(net_1s),
        "pnl_1m": pnl_1m,
        "pnl_diff_1s_minus_1m": float(net_1s - pnl_1m),
        "sign_flip": bool((net_1s > 0) != (pnl_1m > 0)),
    }


def _replay_lifecycle_exit_one(
    trade: dict,
    quotes: pd.DataFrame,
    *,
    policy: ReplayPolicy,
) -> dict:
    decision_time = _utc_timestamp(trade["decision_time"])
    requested_exit_time = _lifecycle_exit_time(trade, decision_time)
    if requested_exit_time is None:
        return {**trade, "audit_status": "missing_lifecycle_exit_time"}

    symbol = trade["raw_symbol"]
    symbol_quotes = quotes[quotes["symbol"] == symbol].copy()
    if symbol_quotes.empty:
        return {**trade, "audit_status": "missing_symbol"}
    entry_candidates = symbol_quotes[symbol_quotes["ts_recv"] <= decision_time]
    if entry_candidates.empty:
        return {**trade, "audit_status": "missing_entry_quote"}
    entry = entry_candidates.iloc[-1]
    entry_bid = float(entry["bid_px_00"])
    entry_ask = float(entry["ask_px_00"])
    future = symbol_quotes[
        (symbol_quotes["ts_recv"] > decision_time)
        & (symbol_quotes["ts_recv"] <= requested_exit_time)
    ].copy()
    if future.empty:
        return {**trade, "audit_status": "missing_exit_path", "entry_ask_1s": entry_ask}

    planned_exit = future.iloc[-1]
    planned_exit_bid = float(planned_exit["bid_px_00"])
    planned_exit_ask = float(planned_exit["ask_px_00"])
    planned_pnl = (planned_exit_bid - entry_ask) * policy.contract_multiplier

    stop_bid = entry_ask * (1.0 - policy.stop_loss_pct)
    target_bid = entry_ask * (1.0 + policy.take_profit_pct)
    mandatory_exit = planned_exit
    mandatory_reason = "lifecycle_exit"
    for _, row in future.iterrows():
        bid = float(row["bid_px_00"])
        if bid <= stop_bid:
            mandatory_exit = row
            mandatory_reason = "stop"
            break
        if bid >= target_bid:
            mandatory_exit = row
            mandatory_reason = "target"
            break

    mandatory_bid = float(mandatory_exit["bid_px_00"])
    mandatory_ask = float(mandatory_exit["ask_px_00"])
    mandatory_pnl = (mandatory_bid - entry_ask) * policy.contract_multiplier
    if policy.fee_per_contract:
        planned_pnl -= 2.0 * policy.fee_per_contract
        mandatory_pnl -= 2.0 * policy.fee_per_contract

    path_bid = pd.to_numeric(future["bid_px_00"], errors="coerce").dropna()
    mfe_1s = float((path_bid.max() - entry_ask) * policy.contract_multiplier) if len(path_bid) else None
    mae_1s = float((path_bid.min() - entry_ask) * policy.contract_multiplier) if len(path_bid) else None
    pnl_1m = _pnl_1m_value(trade)
    return {
        **trade,
        "audit_status": "audited",
        "replay_mode": "lifecycle_exit",
        "entry_time_1s": pd.Timestamp(entry["ts_recv"]).isoformat(),
        "lifecycle_exit_time_requested": requested_exit_time.isoformat(),
        "lifecycle_exit_time_1s": pd.Timestamp(planned_exit["ts_recv"]).isoformat(),
        "mandatory_exit_time_1s": pd.Timestamp(mandatory_exit["ts_recv"]).isoformat(),
        "exit_reason_1m": trade.get("exit_reason", trade.get("candidate_exit_reason")),
        "exit_reason_1s": mandatory_reason,
        "entry_bid_1s": entry_bid,
        "entry_ask_1s": entry_ask,
        "exit_bid_1s": mandatory_bid,
        "exit_ask_1s": mandatory_ask,
        "planned_exit_bid_1s": planned_exit_bid,
        "planned_exit_ask_1s": planned_exit_ask,
        "pnl_1s": float(mandatory_pnl),
        "pnl_1s_planned_exit": float(planned_pnl),
        "pnl_1m": pnl_1m,
        "pnl_diff_1s_minus_1m": float(mandatory_pnl - pnl_1m),
        "pnl_diff_1s_planned_minus_1m": float(planned_pnl - pnl_1m),
        "sign_flip": bool((mandatory_pnl > 0) != (pnl_1m > 0)),
        "planned_sign_flip": bool((planned_pnl > 0) != (pnl_1m > 0)),
        "mandatory_event_before_lifecycle_exit": bool(mandatory_reason != "lifecycle_exit"),
        "mfe_1s": mfe_1s,
        "mae_1s": mae_1s,
    }


def _summarize(rows: list[dict]) -> dict:
    audited = [row for row in rows if row.get("audit_status") == "audited"]
    if not audited:
        return {
            "input_trades": len(rows),
            "audited_trades": 0,
            "coverage": 0.0,
        }
    pnl_1m = np.asarray([row["pnl_1m"] for row in audited], dtype=float)
    pnl_1s = np.asarray([row["pnl_1s"] for row in audited], dtype=float)
    diff = pnl_1s - pnl_1m
    sign_flip = np.asarray([row["sign_flip"] for row in audited], dtype=bool)
    out = {
        "input_trades": len(rows),
        "audited_trades": len(audited),
        "coverage": float(len(audited) / max(len(rows), 1)),
        "unique_sessions_audited": sorted({row["session"] for row in audited}),
        "pnl_1m_sum": float(pnl_1m.sum()),
        "pnl_1s_sum": float(pnl_1s.sum()),
        "pnl_sum_diff_1s_minus_1m": float(diff.sum()),
        "pnl_1m_median": float(np.median(pnl_1m)),
        "pnl_1s_median": float(np.median(pnl_1s)),
        "diff_median": float(np.median(diff)),
        "diff_mean": float(np.mean(diff)),
        "abs_diff_p95": float(np.quantile(np.abs(diff), 0.95)),
        "sign_flip_fraction": float(sign_flip.mean()),
        "one_second_worse_fraction": float((diff < 0).mean()),
        "exit_reason_counts": {
            reason: int(sum(row["exit_reason_1s"] == reason for row in audited))
            for reason in sorted({row["exit_reason_1s"] for row in audited})
        },
    }
    if "pnl_1s_planned_exit" in audited[0]:
        planned = np.asarray([row["pnl_1s_planned_exit"] for row in audited], dtype=float)
        planned_diff = planned - pnl_1m
        planned_flip = np.asarray([row["planned_sign_flip"] for row in audited], dtype=bool)
        mandatory_events = np.asarray([row["mandatory_event_before_lifecycle_exit"] for row in audited], dtype=bool)
        out |= {
            "pnl_1s_planned_exit_sum": float(planned.sum()),
            "pnl_sum_diff_1s_planned_minus_1m": float(planned_diff.sum()),
            "planned_diff_median": float(np.median(planned_diff)),
            "planned_abs_diff_p95": float(np.quantile(np.abs(planned_diff), 0.95)),
            "planned_sign_flip_fraction": float(planned_flip.mean()),
            "mandatory_event_before_lifecycle_exit_count": int(mandatory_events.sum()),
            "mandatory_event_before_lifecycle_exit_fraction": float(mandatory_events.mean()),
        }
    return out


def _summarize_by_split(rows: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    for row in rows:
        expanded.append(row)
        if row.get("split") == "q1_2026" and str(row.get("session", "")) >= "2026-03-01":
            expanded.append({**row, "split": "march_2026"})
    split_order = ["q2_2025", "q3_2025", "q4_2025", "q1_2026", "march_2026"]
    out = []
    for split in split_order:
        group = [row for row in expanded if row.get("split") == split]
        if not group:
            continue
        out.append({"split": split, **_summarize(group)})
    return out


def _write_markdown(path: Path, summary: dict, rows: list[dict]) -> None:
    lines = [
        "# Selected Trades 1s Path Audit",
        "",
        "Replays selected trades on available Databento CBBO-1s audit slices using ask-entry and bid-exit stop/target/hold logic.",
        "",
        "## Summary",
        "",
        f"Input trades: `{summary['input_trades']}`",
        f"Audited trades: `{summary['audited_trades']}`",
        f"Coverage: `{summary['coverage']:.2f}`",
    ]
    if summary.get("audited_trades", 0):
        lines += [
            f"1m PnL sum: `{summary['pnl_1m_sum']:.0f}`",
            f"1s PnL sum: `{summary['pnl_1s_sum']:.0f}`",
            f"1s - 1m PnL sum: `{summary['pnl_sum_diff_1s_minus_1m']:.0f}`",
            f"Median diff: `{summary['diff_median']:.0f}`",
            f"p95 absolute diff: `{summary['abs_diff_p95']:.0f}`",
            f"Sign flip fraction: `{summary['sign_flip_fraction']:.2f}`",
            f"1s worse fraction: `{summary['one_second_worse_fraction']:.2f}`",
        ]
        if "pnl_1s_planned_exit_sum" in summary:
            lines += [
                f"1s planned-exit PnL sum: `{summary['pnl_1s_planned_exit_sum']:.0f}`",
                f"1s planned-exit - 1m PnL sum: `{summary['pnl_sum_diff_1s_planned_minus_1m']:.0f}`",
                f"Planned-exit median diff: `{summary['planned_diff_median']:.0f}`",
                f"Planned-exit p95 absolute diff: `{summary['planned_abs_diff_p95']:.0f}`",
                f"Planned-exit sign flip fraction: `{summary['planned_sign_flip_fraction']:.2f}`",
                f"Mandatory 1s stop/target before lifecycle exit: `{summary['mandatory_event_before_lifecycle_exit_count']}` "
                f"(`{summary['mandatory_event_before_lifecycle_exit_fraction']:.2f}`)",
            ]
        lines += [
            "",
            "## Audited Sessions",
            "",
            ", ".join(summary["unique_sessions_audited"]),
            "",
            "## Split Summary",
            "",
            "| Split | Input | Audited | Coverage | 1m PnL | 1s PnL | Diff | Planned Diff | Mandatory Events |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary.get("by_split", []):
            planned_diff = row.get("pnl_sum_diff_1s_planned_minus_1m", 0.0)
            mandatory_count = row.get("mandatory_event_before_lifecycle_exit_count", 0)
            lines.append(
                f"| {row['split']} | {row['input_trades']} | {row['audited_trades']} | {row['coverage']:.2f} | "
                f"{row.get('pnl_1m_sum', 0.0):.0f} | {row.get('pnl_1s_sum', 0.0):.0f} | "
                f"{row.get('pnl_sum_diff_1s_minus_1m', 0.0):.0f} | {planned_diff:.0f} | {mandatory_count} |"
            )
        lines += [
            "",
            "## Largest Differences",
            "",
            "| Session | Seed | Time | Contract | 1m PnL | 1s PnL | Diff | Exit |",
            "|---|---:|---|---|---:|---:|---:|---|",
        ]
        audited = [row for row in rows if row.get("audit_status") == "audited"]
        audited.sort(key=lambda row: abs(row["pnl_diff_1s_minus_1m"]), reverse=True)
        for row in audited[:20]:
            lines.append(
                f"| {row['session']} | {row['seed']} | {row['decision_time']} | {row['contract_id']} | "
                f"{row['pnl_1m']:.0f} | {row['pnl_1s']:.0f} | {row['pnl_diff_1s_minus_1m']:.0f} | {row['exit_reason_1s']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    policy = ReplayPolicy(
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_hold_minutes=args.max_hold_minutes,
        fee_per_contract=args.fee_per_contract,
    )
    selected = _load_selected(args.selected_trades)
    maps = {session: _raw_symbol_map(args.normalized_dir, session) for session in sorted({row["session"] for row in selected})}
    schema_by_session = _schema_by_session(args.highres_manifest)
    enriched = []
    for row in selected:
        raw_symbol = maps.get(row["session"], {}).get(str(row["contract_id"]))
        status = "candidate"
        if raw_symbol is None:
            status = "missing_raw_symbol_map"
        enriched.append({**row, "raw_symbol": raw_symbol, "audit_status": status})

    out_rows: list[dict] = []
    for session in sorted({row["session"] for row in enriched}):
        session_rows = [row for row in enriched if row["session"] == session]
        schema = schema_by_session.get(session, "cbbo-1s")
        quote_paths = _quote_paths(
            session=session,
            schema=schema,
            cbbo_1s_dir=args.cbbo_1s_dir,
            highres_root=args.highres_root,
        )
        if not quote_paths:
            out_rows.extend([{**row, "audit_status": f"missing_{schema}_session"} for row in session_rows])
            continue
        symbols = {row["raw_symbol"] for row in session_rows if row.get("raw_symbol")}
        if not symbols:
            out_rows.extend(session_rows)
            continue
        quotes = _load_session_quotes(quote_paths, symbols)
        for row in session_rows:
            if row.get("raw_symbol") is None:
                out_rows.append(row)
            else:
                if args.replay_mode == "lifecycle_exit":
                    out_rows.append(_replay_lifecycle_exit_one(row, quotes, policy=policy))
                else:
                    out_rows.append(_replay_one(row, quotes, policy=policy))

    summary = _summarize(out_rows)
    summary["by_split"] = _summarize_by_split(out_rows)
    payload = {
        "policy": policy.__dict__,
        "replay_mode": args.replay_mode,
        "selected_trade_files": [str(path) for path in args.selected_trades],
        "summary": summary,
        "rows": out_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, summary, out_rows)
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
