"""Export Protocol 101 trade inspection charts.

Creates standalone HTML files:

* trades.html: SPX historical chart with selected long-option entries/exits.
* equity.html: single paper-account equity/PnL curve.
* trades.csv: canonical non-overlapping inspection trade log.

No paid data is downloaded. This uses only local ThetaData SPX bars and existing
Protocol 101/107/112 replay artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts")
DEFAULT_PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
DEFAULT_PROTOCOL107_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress")

# Measured on IBKR paper 2026-08-04 (Path-D Phase-0b Track C): avgCost 81.54028
# on a 0.80 fill implies $1.54 per side, $3.08 per round trip, one SPXW
# contract. Evidence: v4/audit/autoresearch/
# pathd_phase0b_trackc_paper_transitions_2026_08_04/trackc_transition_evidence.json
#
# This chart previously charged NO commission at all -- gross PnL crossed the
# spread and stopped there -- so every equity curve it has ever produced is
# overstated by roughly $3.08 per round trip. The measured figure also
# corrected a $0.65/side number that was wrong by 2.4x across four prior
# documents, and it very nearly confirms the frozen Path-D FILL_LAW at
# $1.50/side (v4/research/pathd_entry_exit.py::fill_law).
MEASURED_COMMISSION_PER_SIDE_USD = 1.54
DEFAULT_PROTOCOL112_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_112_protocol101_money_breakdown")
DEFAULT_SPX_DIR = Path("data/vendor/thetadata/index/spx_1m")
DEFAULT_NORMALIZED_DIRS = (
    Path("v4/normalized_official_context"),
    Path("v4/normalized"),
    Path("v4/normalized_official_context_fix_smoke"),
    Path("v4/normalized_official_context_smoke"),
)
NY_TZ = "America/New_York"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--protocol107-dir", type=Path, default=DEFAULT_PROTOCOL107_DIR)
    parser.add_argument("--protocol112-dir", type=Path, default=DEFAULT_PROTOCOL112_DIR)
    parser.add_argument("--spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--normalized-dir", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument("--paper-seed", type=int, default=1)
    parser.add_argument(
        "--stress-per-side",
        type=float,
        default=0.10,
        help=(
            "Additional slippage per side in option PRICE POINTS (x100 multiplier). "
            "0.10 is one tick at or above $3.00. Applied on top of commission."
        ),
    )
    parser.add_argument(
        "--commission-per-side",
        type=float,
        default=MEASURED_COMMISSION_PER_SIDE_USD,
        help=(
            "Commission per side in DOLLARS (no x100 multiplier). Default is the "
            "IBKR-measured 1.54. Charged on the base equity curve, not only the "
            "stress line."
        ),
    )
    parser.add_argument("--include-train-validation", action="store_true")
    parser.add_argument("--skip-train-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_duplicate_outputs(args.out_dir)
    normalized_dirs = tuple(args.normalized_dir or DEFAULT_NORMALIZED_DIRS)
    spx = load_spx_bars(args.spx_dir)

    replay_bundle = build_replay_bundle(
        protocol101_dir=args.protocol101_dir,
        protocol107_dir=args.protocol107_dir,
        protocol112_dir=args.protocol112_dir if args.include_train_validation and not args.skip_train_validation else None,
        spx=spx,
        normalized_dirs=normalized_dirs,
        starting_equity=args.starting_equity,
        paper_seed=args.paper_seed,
    )
    if not replay_bundle["paper_trades"]:
        raise SystemExit("no trades found for chart export")

    include_train_validation = args.include_train_validation and not args.skip_train_validation
    research_trades = replay_bundle["research_trades"]
    paper_trades = replay_bundle["paper_trades"]
    skipped_trades = replay_bundle["skipped_trades"]
    paper_account_summary = replay_bundle["paper_account_summary"]
    if not paper_trades:
        raise SystemExit(f"no affordable paper trades found for seed {args.paper_seed}")

    trades_csv = args.out_dir / "trades.csv"
    skipped_csv = args.out_dir / "skipped_trades.csv"
    research_csv = args.out_dir / "research_all_seed_trades.csv"
    paper_account_summary_path = args.out_dir / "paper_account_summary.json"

    write_trades_csv(trades_csv, paper_trades)
    write_trades_csv(skipped_csv, skipped_trades)
    write_trades_csv(research_csv, research_trades)
    paper_account_summary_path.write_text(json.dumps(paper_account_summary, indent=2, sort_keys=True) + "\n")
    write_trades_html(
        args.out_dir / "trades.html",
        trades=paper_trades,
        spx=spx,
        starting_equity=args.starting_equity,
        skipped_trades=skipped_trades,
    )
    write_equity_html(
        args.out_dir / "equity.html",
        trades=paper_trades,
        starting_equity=args.starting_equity,
        skipped_trades=skipped_trades,
        stress_per_side=args.stress_per_side,
        commission_per_side=args.commission_per_side,
        chart_title="Protocol 101 Equity",
        subtitle=(
            "Single source-of-truth paper replay: one frozen seed, one affordable SPXW 0DTE "
            "contract at a time, with NBBO and slippage-stress equity on the same chart."
        ),
    )
    write_report(
        args.out_dir / "report.md",
        trades=paper_trades,
        research_trades=research_trades,
        skipped_trades=skipped_trades,
        spx=spx,
        starting_equity=args.starting_equity,
        paper_account_summary=paper_account_summary,
        include_train_validation=include_train_validation,
        stress_per_side=args.stress_per_side,
        commission_per_side=args.commission_per_side,
    )

    payload = {
        "paid_data_downloaded": False,
        "live_orders": False,
        "paper_seed": args.paper_seed,
        "starting_equity": args.starting_equity,
        "paper_trades": len(paper_trades),
        "skipped_trades": len(skipped_trades),
        "unaffordable_skipped_trades": sum(
            1 for row in skipped_trades if row.get("paper_skip_reason") == "insufficient_cash"
        ),
        "research_trades": len(research_trades),
        "research_seeds": sorted({int(row["seed"]) for row in research_trades}),
        "sessions": len({row["session"] for row in paper_trades}),
        "first_trade": min(row["decision_time"] for row in paper_trades),
        "last_trade": max(row["decision_time"] for row in paper_trades),
        "include_train_validation": include_train_validation,
        "spx_bars": len(spx),
        "stress_per_side": args.stress_per_side,
        "commission_per_side": args.commission_per_side,
        "source_of_truth": "equity.html",
        "outputs": {
            "trades_html": str(args.out_dir / "trades.html"),
            "equity_html": str(args.out_dir / "equity.html"),
            "trades_csv": str(trades_csv),
            "skipped_trades_csv": str(skipped_csv),
            "research_all_seed_trades_csv": str(research_csv),
            "paper_account_summary": str(paper_account_summary_path),
            "report": str(args.out_dir / "report.md"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cleanup_legacy_duplicate_outputs(out_dir: Path) -> None:
    for name in (
        "holdout_only_equity.html",
        "continuous_context_equity.html",
        "continuous_context_trades.csv",
        "continuous_context_skipped_trades.csv",
        "continuous_context_paper_account_summary.json",
    ):
        path = out_dir / name
        if path.exists():
            path.unlink()


def build_replay_bundle(
    *,
    protocol101_dir: Path,
    protocol107_dir: Path,
    protocol112_dir: Path | None,
    spx: list[dict[str, Any]],
    normalized_dirs: tuple[Path, ...],
    starting_equity: float,
    paper_seed: int,
) -> dict[str, Any]:
    research_trades = load_canonical_trades(
        protocol101_dir=protocol101_dir,
        protocol107_dir=protocol107_dir,
        protocol112_dir=protocol112_dir,
    )
    if not research_trades:
        return {
            "research_trades": [],
            "paper_trades": [],
            "skipped_trades": [],
            "paper_account_summary": [],
        }
    research_trades = attach_spx_prices(research_trades, spx)
    research_trades = backfill_option_quote_accounting(research_trades, normalized_dirs)
    research_trades = add_equity_fields(research_trades)
    paper_trades, skipped_trades = build_paper_account_trades(
        research_trades,
        starting_equity=starting_equity,
        paper_seed=paper_seed,
    )
    paper_account_summary = build_paper_account_summary(
        paper_trades,
        starting_equity,
        skipped_trades=skipped_trades,
    )
    return {
        "research_trades": research_trades,
        "paper_trades": paper_trades,
        "skipped_trades": skipped_trades,
        "paper_account_summary": paper_account_summary,
    }


def load_canonical_trades(
    *,
    protocol101_dir: Path,
    protocol107_dir: Path,
    protocol112_dir: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Current frozen model evidence for the registered OOS windows.
    p101_path = protocol101_dir / "serial_policy_trades.json"
    if p101_path.exists():
        for row in json.loads(p101_path.read_text()):
            reported = str(row.get("reported_split", ""))
            if reported == "march_2026":
                continue  # q1_2026 already contains March; avoid duplicate equity.
            if reported not in {"q3_2025", "q4_2025", "q1_2026"}:
                continue
            item = normalize_trade(row)
            item["stage"] = "test"
            item["segment"] = reported
            item["source_protocol"] = "101_event_history_policy"
            rows.append(item)

    # Already-built Q4 2024 temporal stress.
    p107_path = protocol107_dir / "serial_policy_trades.json"
    if p107_path.exists():
        for row in json.loads(p107_path.read_text()):
            item = normalize_trade(row)
            item["stage"] = "external_stress"
            item["segment"] = "q4_2024_external"
            item["source_protocol"] = "107_protocol101_q4_2024_external_stress"
            rows.append(item)

    # Optional train/validation rows from the Protocol 112 money-breakdown replay.
    # These make the chart cover the currently downloaded Q1/Q2 2025 gap, but the
    # stage labels explicitly mark them as train/validation rather than OOS proof.
    if protocol112_dir is not None:
        p112_path = protocol112_dir / "simulated_trades_with_premiums.json"
        if p112_path.exists():
            for row in json.loads(p112_path.read_text()):
                stage = str(row.get("stage", ""))
                split_label = str(row.get("split_label", ""))
                fold = str(row.get("fold", ""))
                include = (
                    stage == "train"
                    and split_label == "q1_2025"
                    and fold == "fold1_train_q1_validate_q2_test_q3"
                ) or (
                    stage == "validation"
                    and split_label == "q2_2025"
                    and fold == "fold1_train_q1_validate_q2_test_q3"
                )
                if not include:
                    continue
                item = normalize_trade(row)
                item["stage"] = stage
                item["segment"] = split_label
                item["source_protocol"] = "112_protocol101_money_breakdown_replay"
                rows.append(item)

    rows.sort(key=lambda row: (int(row["seed"]), row["decision_time"], row["exit_time"], row["candidate_uid"]))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row["seed"], row["segment"], row["candidate_uid"], row["decision_time"], row["exit_time"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def normalize_trade(row: dict[str, Any]) -> dict[str, Any]:
    pnl = float(row.get("pnl", row.get("raw_candidate_pnl", 0.0)) or 0.0)
    right = str(row.get("right", "")).upper()
    return {
        "candidate_uid": str(row.get("candidate_uid", "")),
        "trade_uid": str(row.get("trade_uid", "")),
        "contract_id": str(row.get("contract_id", "")),
        "seed": int(row.get("seed", 0)),
        "entry_seed": int(row.get("entry_seed", row.get("seed", 0)) or 0),
        "session": str(row.get("session", "")),
        "decision_time": iso_utc(row.get("decision_time")),
        "exit_time": iso_utc(row.get("exit_time")),
        "right": right,
        "side": "CALL" if right == "C" else "PUT" if right == "P" else right,
        "offset": float(row.get("offset", 0.0) or 0.0),
        "score": float(row.get("score", 0.0) or 0.0),
        "threshold": float(row.get("threshold", 0.0) or 0.0),
        "pnl": pnl,
        "raw_candidate_pnl": float(row.get("raw_candidate_pnl", pnl) or 0.0),
        "slippage_per_side": float(row.get("slippage_per_side", 0.0) or 0.0),
        "exit_reason": str(row.get("exit_reason", "")),
        "label_source": str(row.get("label_source", "")),
        "entry_ask": none_or_float(row.get("entry_ask")),
        "entry_bid": none_or_float(row.get("entry_bid")),
        "premium_paid": none_or_float(row.get("premium_paid")),
        "fold": str(row.get("fold", "")),
    }


def iso_utc(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def load_spx_bars(spx_dir: Path) -> list[dict[str, Any]]:
    if not spx_dir.exists():
        raise SystemExit(f"missing SPX directory: {spx_dir}")
    frames = []
    for path in sorted(spx_dir.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["event_time", "open", "high", "low", "close"])
            price_columns = ["open", "high", "low", "close"]
        except Exception:
            try:
                frame = pd.read_parquet(path, columns=["event_time", "close"])
                price_columns = ["close"]
            except Exception:
                continue
        if "event_time" not in frame.columns or "close" not in frame.columns:
            continue
        if frame.empty:
            continue
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        for column in price_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in ("open", "high", "low"):
            if column not in frame.columns:
                frame[column] = frame["close"]
        frame = frame.dropna(subset=["event_time", "close"])
        frame["session"] = path.stem
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no SPX parquet files found in {spx_dir}")
    spx = pd.concat(frames, ignore_index=True).drop_duplicates("event_time").sort_values("event_time")
    return [
        {
            "bar": i,
            "t": int(row.event_time.value // 1_000_000),
            "iso": row.event_time.isoformat(),
            "session": str(row.session),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for i, row in enumerate(spx.itertuples(index=False))
    ]


def attach_spx_prices(trades: list[dict[str, Any]], spx: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spx_frame = pd.DataFrame(spx)
    spx_frame["event_time"] = pd.to_datetime(spx_frame["iso"], utc=True)
    spx_frame = spx_frame[["event_time", "close", "bar"]].sort_values("event_time")
    for side in ("decision_time", "exit_time"):
        trades_frame = pd.DataFrame({"_idx": range(len(trades)), "event_time": pd.to_datetime([row[side] for row in trades], utc=True)})
        merged = pd.merge_asof(
            trades_frame.sort_values("event_time"),
            spx_frame,
            on="event_time",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=2),
        )
        for _, row in merged.iterrows():
            value = none_or_float(row["close"])
            target = trades[int(row["_idx"])]
            target["entry_spx" if side == "decision_time" else "exit_spx"] = value
            target["entry_bar" if side == "decision_time" else "exit_bar"] = none_or_float(row["bar"])
    for row in trades:
        row["decision_ms"] = int(pd.Timestamp(row["decision_time"]).value // 1_000_000)
        row["exit_ms"] = int(pd.Timestamp(row["exit_time"]).value // 1_000_000)
    return trades


def find_normalized_path(session: str, normalized_dirs: tuple[Path, ...]) -> Path | None:
    names = (
        f"databento_spxw_0dte_{session}.parquet",
        f"databento_spxw_0dte_{session}_official_context.parquet",
        f"databento_spxw_0dte_{session}_derived_context.parquet",
    )
    for directory in normalized_dirs:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    return None


def backfill_option_quote_accounting(
    trades: list[dict[str, Any]], normalized_dirs: tuple[Path, ...]
) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in trades:
        by_session.setdefault(str(row["session"]), []).append(row)

    columns = ["quote_time", "contract_id", "bid", "ask", "bid_size", "ask_size"]
    for session, session_trades in sorted(by_session.items()):
        path = find_normalized_path(session, normalized_dirs)
        if path is None:
            for trade in session_trades:
                trade["quote_backfill_status"] = "missing_normalized_session"
            continue
        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception:
            for trade in session_trades:
                trade["quote_backfill_status"] = "unreadable_normalized_session"
            continue

        frame["quote_time"] = pd.to_datetime(frame["quote_time"], utc=True)
        frame["contract_id"] = frame["contract_id"].astype(str)
        for column in ["bid", "ask", "bid_size", "ask_size"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        contracts = {str(row["contract_id"]) for row in session_trades}
        frame = frame[frame["contract_id"].isin(contracts)].sort_values(["contract_id", "quote_time"])
        by_contract = {contract: group.reset_index(drop=True) for contract, group in frame.groupby("contract_id", sort=False)}

        for trade in session_trades:
            contract_id = str(trade["contract_id"])
            contract_rows = by_contract.get(contract_id)
            if contract_rows is None or contract_rows.empty:
                trade["quote_backfill_status"] = "missing_contract"
                continue
            decision_time = pd.Timestamp(trade["decision_time"])
            if decision_time.tzinfo is None:
                decision_time = decision_time.tz_localize("UTC")
            else:
                decision_time = decision_time.tz_convert("UTC")
            exit_time = pd.Timestamp(trade["exit_time"])
            if exit_time.tzinfo is None:
                exit_time = exit_time.tz_localize("UTC")
            else:
                exit_time = exit_time.tz_convert("UTC")

            valid_entry = contract_rows[
                (contract_rows["quote_time"] <= decision_time)
                & contract_rows["bid"].notna()
                & contract_rows["ask"].notna()
                & (contract_rows["ask"] > 0)
                & (contract_rows["ask"] >= contract_rows["bid"])
            ]
            if valid_entry.empty:
                trade["quote_backfill_status"] = "missing_entry_quote"
                continue
            entry = valid_entry.iloc[-1]
            entry_ask = float(entry["ask"])
            entry_bid = float(entry["bid"])

            path_rows = contract_rows[
                (contract_rows["quote_time"] >= entry["quote_time"])
                & (contract_rows["quote_time"] <= exit_time)
                & contract_rows["bid"].notna()
            ].copy()
            if path_rows.empty:
                trade["quote_backfill_status"] = "missing_path_quotes"
                continue
            valid_exit = path_rows[
                path_rows["bid"].notna()
                & path_rows["ask"].notna()
                & (path_rows["ask"] >= path_rows["bid"])
            ]
            if valid_exit.empty:
                trade["quote_backfill_status"] = "missing_exit_quote"
                continue
            exit_quote = valid_exit.iloc[-1]
            path_pnl = (path_rows["bid"].astype(float) - entry_ask) * 100.0

            trade["entry_quote_time"] = pd.Timestamp(entry["quote_time"]).isoformat()
            trade["entry_bid"] = entry_bid
            trade["entry_ask"] = entry_ask
            trade["entry_bid_size"] = none_or_float(entry.get("bid_size"))
            trade["entry_ask_size"] = none_or_float(entry.get("ask_size"))
            trade["premium_paid"] = entry_ask * 100.0
            trade["exit_quote_time"] = pd.Timestamp(exit_quote["quote_time"]).isoformat()
            trade["exit_bid"] = float(exit_quote["bid"])
            trade["exit_ask"] = float(exit_quote["ask"])
            trade["path_points"] = int(len(path_rows))
            trade["path_mfe"] = float(path_pnl.max())
            trade["path_mae"] = float(path_pnl.min())
            trade["path_final_pnl"] = float(path_pnl.iloc[-1])
            trade["quote_gap_seconds"] = float((decision_time - pd.Timestamp(entry["quote_time"])).total_seconds())
            trade["quote_backfill_status"] = "ok"
    return trades


def add_equity_fields(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in trades:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    for seed_rows in by_seed.values():
        seed_rows.sort(key=lambda row: (row["exit_ms"], row["decision_ms"], row["candidate_uid"]))
        cumulative = 0.0
        for i, row in enumerate(seed_rows, start=1):
            cumulative += float(row["pnl"])
            row["trade_number"] = i
            row["cumulative_pnl"] = round(cumulative, 6)
    trades.sort(key=lambda row: (int(row["seed"]), row["decision_ms"], row["exit_ms"], row["candidate_uid"]))
    return trades


def premium_dollars(row: dict[str, Any]) -> float | None:
    premium = none_or_float(row.get("premium_paid"))
    if premium is not None:
        return premium
    ask = none_or_float(row.get("entry_ask"))
    if ask is not None:
        return ask * 100.0
    return None


def build_paper_account_trades(
    trades: list[dict[str, Any]],
    *,
    starting_equity: float,
    paper_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows = [dict(row) for row in trades if int(row["seed"]) == int(paper_seed)]
    seed_rows.sort(key=lambda row: (row["decision_ms"], row["exit_ms"], row["candidate_uid"]))
    cash = float(starting_equity)
    last_exit_ms = -math.inf
    cumulative = 0.0
    taken: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in seed_rows:
        row["paper_seed"] = int(paper_seed)
        row["paper_starting_equity"] = float(starting_equity)
        row["paper_cash_before"] = round(cash, 6)
        row["paper_selected"] = False
        row["paper_skip_reason"] = ""
        premium = premium_dollars(row)
        row["paper_premium"] = premium
        if row["decision_ms"] < last_exit_ms:
            row["paper_skip_reason"] = "overlap_open_position"
            skipped.append(row)
            continue
        if premium is None:
            row["paper_skip_reason"] = "missing_entry_premium"
            skipped.append(row)
            continue
        if premium > cash + 1e-9:
            row["paper_skip_reason"] = "insufficient_cash"
            skipped.append(row)
            continue

        pnl = float(row["pnl"])
        cash_before = cash
        cash = cash + pnl
        cumulative = cash - float(starting_equity)
        row["paper_selected"] = True
        row["paper_trade_number"] = len(taken) + 1
        row["trade_number"] = len(taken) + 1
        row["paper_cash_before"] = round(cash_before, 6)
        row["paper_cash_after"] = round(cash, 6)
        row["paper_equity_after"] = round(cash, 6)
        row["paper_cumulative_pnl"] = round(cumulative, 6)
        row["cumulative_pnl"] = round(cumulative, 6)
        row["paper_buying_power_used"] = round(premium, 6)
        row["paper_buying_power_pct_cash"] = round(premium / cash_before, 6) if cash_before else None
        mae = none_or_float(row.get("path_mae"))
        mfe = none_or_float(row.get("path_mfe"))
        row["paper_intratrade_low_equity"] = round(cash_before + min(0.0, mae), 6) if mae is not None else None
        row["paper_intratrade_high_equity"] = round(cash_before + max(0.0, mfe), 6) if mfe is not None else None
        last_exit_ms = max(last_exit_ms, float(row["exit_ms"]))
        taken.append(row)

    return taken, skipped


def build_paper_account_summary(
    trades: list[dict[str, Any]],
    starting_equity: float,
    *,
    skipped_trades: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in trades:
        by_seed.setdefault(int(row["seed"]), []).append(row)

    skipped_trades = skipped_trades or []
    skipped_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in skipped_trades:
        skipped_by_seed.setdefault(int(row["seed"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for seed, rows in sorted(by_seed.items()):
        rows.sort(key=lambda row: (row["exit_ms"], row["decision_ms"], row["candidate_uid"]))
        cash = float(starting_equity)
        peak = cash
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        min_equity = cash
        total_pnl = 0.0
        wins = 0
        daily_pnl: dict[str, float] = {}
        known_premiums: list[float] = []
        known_maes: list[float] = []
        known_mfes: list[float] = []
        intratrade_low_equities: list[float] = []

        for row in rows:
            pnl = float(row["pnl"])
            total_pnl += pnl
            wins += int(pnl >= 0)
            daily_pnl[str(row["session"])] = daily_pnl.get(str(row["session"]), 0.0) + pnl
            premium = premium_dollars(row)
            if premium is not None:
                known_premiums.append(premium)
            mae = none_or_float(row.get("path_mae"))
            if mae is not None:
                known_maes.append(mae)
                cash_before = none_or_float(row.get("paper_cash_before"))
                if cash_before is not None:
                    intratrade_low_equities.append(cash_before + min(0.0, mae))
            mfe = none_or_float(row.get("path_mfe"))
            if mfe is not None:
                known_mfes.append(mfe)

            cash += pnl
            min_equity = min(min_equity, cash)
            peak = max(peak, cash)
            drawdown = cash - peak
            max_drawdown = min(max_drawdown, drawdown)
            if peak > 0:
                max_drawdown_pct = min(max_drawdown_pct, drawdown / peak)

        worst_day_pnl = min(daily_pnl.values()) if daily_pnl else 0.0
        max_known_bp = max(known_premiums) if known_premiums else None
        avg_known_bp = sum(known_premiums) / len(known_premiums) if known_premiums else None
        worst_path_mae = min(known_maes) if known_maes else None
        best_path_mfe = max(known_mfes) if known_mfes else None
        seed_skips = skipped_by_seed.get(seed, [])
        skip_reasons: dict[str, int] = {}
        for row in seed_skips:
            reason = str(row.get("paper_skip_reason", "unknown") or "unknown")
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        min_intratrade_equity = min(intratrade_low_equities) if intratrade_low_equities else None
        summaries.append(
            {
                "seed": seed,
                "starting_equity": round(float(starting_equity), 2),
                "trades": len(rows),
                "skipped_trades": len(seed_skips),
                "skip_reasons": skip_reasons,
                "winning_trades": wins,
                "win_rate": round(wins / len(rows), 6) if rows else 0.0,
                "total_pnl": round(total_pnl, 2),
                "ending_equity": round(float(starting_equity) + total_pnl, 2),
                "return_on_starting_equity": round(total_pnl / float(starting_equity), 6) if starting_equity else None,
                "min_equity": round(min_equity, 2),
                "max_drawdown": round(max_drawdown, 2),
                "max_drawdown_pct": round(max_drawdown_pct, 6),
                "worst_day_pnl": round(worst_day_pnl, 2),
                "max_daily_loss": round(max(0.0, -worst_day_pnl), 2),
                "known_premium_trades": len(known_premiums),
                "premium_coverage": round(len(known_premiums) / len(rows), 6) if rows else 0.0,
                "max_known_buying_power": round(max_known_bp, 2) if max_known_bp is not None else None,
                "avg_known_buying_power": round(avg_known_bp, 2) if avg_known_bp is not None else None,
                "max_known_buying_power_pct_equity": round(max_known_bp / float(starting_equity), 6)
                if max_known_bp is not None and starting_equity
                else None,
                "known_path_trades": len(known_maes),
                "path_coverage": round(len(known_maes) / len(rows), 6) if rows else 0.0,
                "worst_intratrade_mae": round(worst_path_mae, 2) if worst_path_mae is not None else None,
                "max_intratrade_loss": round(max(0.0, -worst_path_mae), 2) if worst_path_mae is not None else None,
                "min_intratrade_equity": round(min_intratrade_equity, 2)
                if min_intratrade_equity is not None
                else None,
                "best_intratrade_mfe": round(best_path_mfe, 2) if best_path_mfe is not None else None,
            }
        )
    return summaries


def write_trades_csv(path: Path, trades: list[dict[str, Any]]) -> None:
    columns = [
        "seed",
        "trade_number",
        "stage",
        "segment",
        "source_protocol",
        "session",
        "decision_time",
        "exit_time",
        "right",
        "side",
        "offset",
        "contract_id",
        "entry_spx",
        "exit_spx",
        "entry_bar",
        "exit_bar",
        "entry_quote_time",
        "exit_quote_time",
        "entry_ask",
        "entry_bid",
        "exit_bid",
        "exit_ask",
        "entry_bid_size",
        "entry_ask_size",
        "premium_paid",
        "quote_gap_seconds",
        "path_mfe",
        "path_mae",
        "path_final_pnl",
        "path_points",
        "quote_backfill_status",
        "paper_seed",
        "paper_selected",
        "paper_skip_reason",
        "paper_cash_before",
        "paper_cash_after",
        "paper_equity_after",
        "paper_cumulative_pnl",
        "paper_premium",
        "paper_buying_power_used",
        "paper_buying_power_pct_cash",
        "paper_intratrade_low_equity",
        "paper_intratrade_high_equity",
        "pnl",
        "cumulative_pnl",
        "score",
        "threshold",
        "exit_reason",
        "candidate_uid",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in trades:
            writer.writerow({column: row.get(column) for column in columns})


def _normalize_trade_chart_bars(
    trades: list[dict[str, Any]],
    spx: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return dense chart coordinates after any session/date filtering.

    ``load_spx_bars`` numbers the full vendor history once.  Callers may then
    retain only a few sessions, leaving large/non-zero bar identifiers.  The
    browser chart uses bar coordinates as array coordinates, so normalize the
    filtered payload and the attached trade markers together before emission.
    """

    ordered = sorted(
        (dict(row) for row in spx),
        key=lambda row: (int(row["t"]), float(row.get("bar", 0.0))),
    )
    original_to_dense: dict[float, int] = {}
    normalized_spx: list[dict[str, Any]] = []
    for dense_bar, row in enumerate(ordered):
        original_bar = float(row.get("bar", dense_bar))
        original_to_dense[original_bar] = dense_bar
        row["bar"] = dense_bar
        normalized_spx.append(row)

    normalized_trades = [dict(row) for row in trades]
    for row in normalized_trades:
        for field in ("entry_bar", "exit_bar"):
            value = row.get(field)
            if value is None:
                continue
            numeric = float(value)
            if numeric not in original_to_dense:
                raise ValueError(f"trade {field} has no matching SPX bar: {numeric}")
            row[field] = float(original_to_dense[numeric])
    return normalized_trades, normalized_spx


def write_trades_html(
    path: Path,
    *,
    trades: list[dict[str, Any]],
    spx: list[dict[str, Any]],
    starting_equity: float,
    skipped_trades: list[dict[str, Any]],
) -> None:
    normalized_trades, normalized_spx = _normalize_trade_chart_bars(trades, spx)
    payload = {
        "title": "Protocol 101 Trade Overlay",
        "startingEquity": float(starting_equity),
        "paperSeed": int(normalized_trades[0]["seed"]) if normalized_trades else None,
        "skippedTrades": len(skipped_trades),
        "unaffordableSkippedTrades": sum(
            1 for row in skipped_trades if row.get("paper_skip_reason") == "insufficient_cash"
        ),
        "spx": normalized_spx,
        "trades": normalized_trades,
    }
    path.write_text(html_shell(title="Protocol 101 Trades", body=TRADE_BODY, payload=payload))


def write_equity_html(
    path: Path,
    *,
    trades: list[dict[str, Any]],
    starting_equity: float,
    skipped_trades: list[dict[str, Any]],
    stress_per_side: float,
    commission_per_side: float,
    chart_title: str,
    subtitle: str,
) -> None:
    payload = {
        "title": chart_title,
        "subtitle": subtitle,
        "startingEquity": float(starting_equity),
        "stressPerSide": float(stress_per_side),
        "commissionPerSide": float(commission_per_side),
        "paperSeed": int(trades[0]["seed"]) if trades else None,
        "skippedTrades": len(skipped_trades),
        "unaffordableSkippedTrades": sum(
            1 for row in skipped_trades if row.get("paper_skip_reason") == "insufficient_cash"
        ),
        "trades": trades,
    }
    path.write_text(html_shell(title=chart_title, body=EQUITY_BODY, payload=payload))


def commission_cost(commission_per_side: float) -> float:
    """Round-trip commission in dollars for one contract.

    UNITS TRAP -- commission is quoted in DOLLARS PER SIDE and must NOT be
    scaled by the 100x contract multiplier, unlike slippage, which is quoted in
    option price points. Scaling it would overstate commission 100x, turning
    $3.08 into $308 per round trip.
    """

    return float(commission_per_side) * 2.0


def slippage_cost(stress_per_side: float) -> float:
    """Round-trip slippage in dollars from a per-side option-price penetration.

    ``stress_per_side`` is in option price points, so it takes the 100x
    multiplier. The $0.10 default is exactly one tick for a contract priced at
    or above $3.00 (see ``option_tick_micros``), i.e. one-tick penetration per
    side, which the fix-research recorded as the honest middle of the passive
    fill ladder.
    """

    return float(stress_per_side) * 2.0 * 100.0


def net_trade_pnl(row: dict[str, Any], commission_per_side: float) -> float:
    """Spread-crossed PnL after commission, before any slippage stress.

    ``row["pnl"]`` is gross: it already crosses the spread (buy at ask, sell at
    bid) but pays no commission at all. This is the honest baseline.
    """

    return float(row["pnl"]) - commission_cost(commission_per_side)


def stressed_trade_pnl(
    row: dict[str, Any],
    stress_per_side: float,
    commission_per_side: float = MEASURED_COMMISSION_PER_SIDE_USD,
) -> float:
    return net_trade_pnl(row, commission_per_side) - slippage_cost(stress_per_side)


def equity_after_commission(
    trades: list[dict[str, Any]],
    starting_equity: float,
    commission_per_side: float = MEASURED_COMMISSION_PER_SIDE_USD,
) -> float:
    return float(starting_equity) + sum(
        net_trade_pnl(row, commission_per_side) for row in trades
    )


def equity_after_stress(
    trades: list[dict[str, Any]],
    starting_equity: float,
    stress_per_side: float,
    commission_per_side: float = MEASURED_COMMISSION_PER_SIDE_USD,
) -> float:
    return float(starting_equity) + sum(
        stressed_trade_pnl(row, stress_per_side, commission_per_side) for row in trades
    )


def block_summary_rows(
    trades: list[dict[str, Any]],
    stress_per_side: float,
    commission_per_side: float = MEASURED_COMMISSION_PER_SIDE_USD,
) -> list[dict[str, Any]]:
    if not trades:
        return []
    frame = pd.DataFrame(trades)
    rows: list[dict[str, Any]] = []
    for (stage, segment), group in frame.groupby(["stage", "segment"], sort=False):
        pnl = float(group["pnl"].sum())
        records = group.to_dict("records")
        net = sum(net_trade_pnl(row, commission_per_side) for row in records)
        stressed = sum(
            stressed_trade_pnl(row, stress_per_side, commission_per_side)
            for row in records
        )
        rows.append(
            {
                "stage": str(stage),
                "segment": str(segment),
                "first": str(group["decision_time"].min())[:10],
                "last": str(group["decision_time"].max())[:10],
                "trades": int(len(group)),
                "pnl": pnl,
                "net_pnl": float(net),
                "stressed_pnl": float(stressed),
                "win_rate": float((group["pnl"] >= 0).mean()) if len(group) else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (row["first"], row["stage"], row["segment"]))


def profit_concentration(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {}
    sorted_by_pnl = sorted(trades, key=lambda row: float(row["pnl"]), reverse=True)
    total_pnl = sum(float(row["pnl"]) for row in trades)
    gross_profit = sum(max(0.0, float(row["pnl"])) for row in trades)
    top_5 = sum(float(row["pnl"]) for row in sorted_by_pnl[:5])
    top_10 = sum(float(row["pnl"]) for row in sorted_by_pnl[:10])
    top_20 = sum(float(row["pnl"]) for row in sorted_by_pnl[:20])
    best = sorted_by_pnl[0]
    worst = min(trades, key=lambda row: float(row["pnl"]))
    daily: dict[str, float] = {}
    for row in trades:
        daily[str(row["session"])] = daily.get(str(row["session"]), 0.0) + float(row["pnl"])
    best_day_session, best_day_pnl = max(daily.items(), key=lambda item: item[1])
    worst_day_session, worst_day_pnl = min(daily.items(), key=lambda item: item[1])
    return {
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "top_5": top_5,
        "top_10": top_10,
        "top_20": top_20,
        "top_5_share_total": top_5 / total_pnl if total_pnl else None,
        "top_10_share_total": top_10 / total_pnl if total_pnl else None,
        "top_20_share_total": top_20 / total_pnl if total_pnl else None,
        "top_20_share_gross": top_20 / gross_profit if gross_profit else None,
        "best_trade": best,
        "worst_trade": worst,
        "best_day_session": best_day_session,
        "best_day_pnl": best_day_pnl,
        "worst_day_session": worst_day_session,
        "worst_day_pnl": worst_day_pnl,
    }


def append_paper_summary_table(lines: list[str], paper_account_summary: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| seed | start_cash | trades | skipped | ending_equity | return_on_start | max_drawdown | max_drawdown_pct | worst_day_pnl | max_daily_loss | worst_intratrade_mae | min_intratrade_equity | max_known_bp | premium_coverage | path_coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in paper_account_summary:
        max_bp = row["max_known_buying_power"]
        max_bp_text = "n/a" if max_bp is None else f"${max_bp:,.0f}"
        worst_mae = row["worst_intratrade_mae"]
        worst_mae_text = "n/a" if worst_mae is None else f"${worst_mae:,.0f}"
        min_intratrade_equity = row["min_intratrade_equity"]
        min_intratrade_equity_text = (
            "n/a" if min_intratrade_equity is None else f"${min_intratrade_equity:,.0f}"
        )
        roc = row["return_on_starting_equity"]
        roc_text = "n/a" if roc is None else f"{roc * 100:.1f}%"
        lines.append(
            "| "
            f"{row['seed']} | "
            f"${row['starting_equity']:,.0f} | "
            f"{row['trades']} | "
            f"{row['skipped_trades']} | "
            f"${row['ending_equity']:,.0f} | "
            f"{roc_text} | "
            f"${row['max_drawdown']:,.0f} | "
            f"{row['max_drawdown_pct'] * 100:.1f}% | "
            f"${row['worst_day_pnl']:,.0f} | "
            f"${row['max_daily_loss']:,.0f} | "
            f"{worst_mae_text} | "
            f"{min_intratrade_equity_text} | "
            f"{max_bp_text} | "
            f"{row['premium_coverage'] * 100:.1f}% | "
            f"{row['path_coverage'] * 100:.1f}% |"
        )


def write_report(
    path: Path,
    *,
    trades: list[dict[str, Any]],
    research_trades: list[dict[str, Any]],
    skipped_trades: list[dict[str, Any]],
    spx: list[dict[str, Any]],
    starting_equity: float,
    paper_account_summary: list[dict[str, Any]],
    include_train_validation: bool,
    stress_per_side: float,
    commission_per_side: float,
) -> None:
    frame = pd.DataFrame(trades)
    research_frame = pd.DataFrame(research_trades)
    by_seed = frame.groupby("seed")["pnl"].agg(["sum", "count", "mean"]).reset_index()
    quote_status = frame.get("quote_backfill_status", pd.Series(dtype=str)).fillna("missing").value_counts().to_dict()
    skipped_reasons = pd.Series([row.get("paper_skip_reason", "unknown") for row in skipped_trades]).value_counts().to_dict()
    unaffordable_skips = int(skipped_reasons.get("insufficient_cash", 0))
    paper_seed = int(frame["seed"].iloc[0]) if not frame.empty else None
    headline_label = "includes_train_validation" if include_train_validation else "holdout_only"
    ending_net = equity_after_commission(trades, starting_equity, commission_per_side)
    ending_stress = equity_after_stress(trades, starting_equity, stress_per_side, commission_per_side)
    lines = [
        "# Protocol 113: Protocol 101 Trade Charts",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Headline mode: `{headline_label}`",
        "- Source of truth: `equity.html`",
        f"- Paper seed: `{paper_seed}`",
        f"- Paper starting cash: `${starting_equity:,.0f}`",
        f"- Paper trades taken: `{len(trades)}`",
        f"- Paper trades skipped: `{len(skipped_trades)}`",
        f"- Skipped for insufficient cash: `{unaffordable_skips}`",
        f"- Research candidate trades available before paper-account filtering: `{len(research_trades)}`",
        f"- Research seeds available: `{sorted(research_frame['seed'].unique().tolist()) if not research_frame.empty else []}`",
        f"- Train/validation rows included: `{include_train_validation}`",
        f"- Commission charged on the base curve: `${commission_per_side:.2f}`/side "
        f"(`${commission_cost(commission_per_side):.2f}` round trip, IBKR-measured 2026-08-04)",
        f"- Net ending equity after commission: `${ending_net:,.0f}`",
        f"- Additional slippage stress line: `${stress_per_side:.2f}`/side in option price points "
        f"(`${slippage_cost(stress_per_side):.2f}` round trip)",
        f"- Stressed ending equity after commission and slippage: `${ending_stress:,.0f}`",
        f"- SPX bars: `{len(spx)}`",
        f"- First trade: `{frame['decision_time'].min()}`",
        f"- Last trade: `{frame['decision_time'].max()}`",
        f"- Quote accounting backfill: `{quote_status}`",
        f"- Paper skip reasons: `{skipped_reasons}`",
        "",
        "## Paper-Account Lens",
        "",
        "This report now shows one frozen-seed paper account, not an all-seed research ensemble. A trade is taken only when one-contract entry premium is known and affordable from current cash. This is still historical replay, not live-trading approval. Equity and daily loss are closed-trade accounting; intratrade MAE is reported separately so the chart does not hide adverse movement before exit.",
        "",
    ]
    append_paper_summary_table(lines, paper_account_summary)
    lines.extend(
        [
            "",
            "## Block PnL",
            "",
            "| stage | segment | first | last | trades | gross_pnl | net_pnl | stressed_pnl | win_rate |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in block_summary_rows(trades, stress_per_side, commission_per_side):
        lines.append(
            "| "
            f"{row['stage']} | {row['segment']} | {row['first']} | {row['last']} | "
            f"{row['trades']} | ${row['pnl']:,.0f} | ${row['net_pnl']:,.0f} | "
            f"${row['stressed_pnl']:,.0f} | {row['win_rate'] * 100:.1f}% |"
        )

    concentration = profit_concentration(trades)
    if concentration:
        best = concentration["best_trade"]
        worst = concentration["worst_trade"]
        def share_text(value: float | None) -> str:
            return "n/a" if value is None else f"{value * 100:.1f}%"

        lines.extend(
            [
                "",
                "## Profit Concentration",
                "",
                "These checks are on the single source-of-truth paper account. High concentration does not disprove the edge, but it tells us whether the curve is being carried by a small number of convex 0DTE wins.",
                "",
                f"- Top 5 trades: `${concentration['top_5']:,.0f}` ({share_text(concentration['top_5_share_total'])} of net PnL)",
                f"- Top 10 trades: `${concentration['top_10']:,.0f}` ({share_text(concentration['top_10_share_total'])} of net PnL)",
                f"- Top 20 trades: `${concentration['top_20']:,.0f}` ({share_text(concentration['top_20_share_total'])} of net PnL, {share_text(concentration['top_20_share_gross'])} of gross profit)",
                f"- Best trade: `{best['session']}` `{best['side']}` `{best['contract_id']}` `${float(best['pnl']):,.0f}`",
                f"- Worst trade: `{worst['session']}` `{worst['side']}` `{worst['contract_id']}` `${float(worst['pnl']):,.0f}`",
                f"- Best day: `{concentration['best_day_session']}` `${concentration['best_day_pnl']:,.0f}`",
                f"- Worst day: `{concentration['worst_day_session']}` `${concentration['worst_day_pnl']:,.0f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Seed Summary",
            "",
            "| seed | trades | total_pnl | avg_pnl |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in by_seed.itertuples(index=False):
        lines.append(f"| {int(row.seed)} | {int(row.count)} | {float(row.sum):.2f} | {float(row.mean):.2f} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `trades.html`: SPX chart with entry/exit markers.",
            "- `equity.html`: the single paper-account equity curve and source of truth.",
            "- `trades.csv`: affordable paper-account trade log used by both charts.",
            "- `skipped_trades.csv`: trades rejected by the paper-account gate.",
            "- `research_all_seed_trades.csv`: full research trade log before single-account filtering.",
            "",
            "Training and validation rows are excluded by default so the chart is closer to a deployment-style paper replay. If `--include-train-validation` is used, the same `equity.html` file is regenerated in that mode rather than creating a second competing equity curve.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def html_shell(*, title: str, body: str, payload: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101418; --panel: #171d23; --text: #e7edf3; --muted: #9aa7b2; --grid: rgba(255,255,255,.12); --line: #6cb6ff; --green: #46d39a; --red: #ff6b6b; --yellow: #ffd166; --blue: #72a7ff; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 18px 22px 10px; border-bottom: 1px solid rgba(255,255,255,.08); }}
    h1 {{ margin: 0 0 6px; font-size: 22px; letter-spacing: 0; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end; padding: 14px 22px; background: var(--panel); border-bottom: 1px solid rgba(255,255,255,.08); }}
    label {{ display: grid; gap: 4px; color: var(--muted); font-size: 12px; }}
    select, input, button {{ background: #0f1419; color: var(--text); border: 1px solid rgba(255,255,255,.16); border-radius: 6px; padding: 7px 9px; font: inherit; }}
    button {{ cursor: pointer; }}
    .statbar {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 12px 22px; color: var(--muted); }}
    .stat {{ background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08); border-radius: 6px; padding: 7px 9px; }}
    .wrap {{ padding: 0 18px 22px; }}
    canvas {{ width: 100%; height: 640px; display: block; background: #0c1014; border: 1px solid rgba(255,255,255,.10); border-radius: 8px; cursor: grab; }}
    canvas.dragging {{ cursor: grabbing; }}
    #tooltip {{ position: fixed; display: none; pointer-events: none; z-index: 10; background: rgba(8,12,16,.96); border: 1px solid rgba(255,255,255,.18); border-radius: 6px; padding: 8px 10px; max-width: 360px; color: var(--text); font-size: 12px; box-shadow: 0 8px 28px rgba(0,0,0,.35); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 14px; }}
    th, td {{ text-align: left; border-bottom: 1px solid rgba(255,255,255,.08); padding: 6px 8px; white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 600; position: sticky; top: 0; background: var(--panel); }}
    tbody tr {{ cursor: pointer; }}
    tbody tr:hover, tbody tr.selected {{ background: rgba(108,182,255,.12); }}
    .tablewrap {{ max-height: 360px; overflow: auto; border: 1px solid rgba(255,255,255,.08); border-radius: 8px; background: var(--panel); }}
    .note {{ color: var(--muted); padding: 0 22px 12px; font-size: 12px; }}
    .replay-warning {{ margin: 10px 22px 0; color: #ffd166; font-size: 12px; font-weight: 650; }}
    h2 {{ margin: 18px 0 8px; font-size: 14px; color: var(--muted); letter-spacing: 0; }}
    .viewbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 0 22px 12px; color: var(--muted); font-size: 12px; }}
    .viewbar input {{ width: 82px; }}
    .viewbar .viewstatus {{ background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08); border-radius: 6px; padding: 7px 9px; }}
  </style>
</head>
<body>
<div id="tooltip"></div>
<script id="payload" type="application/json">{json.dumps(payload, separators=(",", ":"))}</script>
{body}
</body>
</html>
"""


TRADE_BODY = r"""
<header>
  <h1>Protocol 101 Trade Overlay</h1>
  <div class="sub">Single-account replay: one frozen seed, one SPXW 0DTE contract at a time, ask-entry/bid-exit, and a fixed paper cash gate. This is historical replay, not live trading.</div>
  <div class="replay-warning">HISTORICAL REPLAY ONLY. This page does not prove live fills, broker routing, or paper-trading approval.</div>
</header>
<section class="controls">
  <label>Account<select id="seed"></select></label>
  <label>Stage<select id="stage"><option value="all">all</option></select></label>
  <label>Chart<select id="chartMode"><option value="candles">candles</option><option value="line">line</option></select></label>
  <label>Start<input id="startDate" type="date"></label>
  <label>End<input id="endDate" type="date"></label>
  <button id="fullRange">Full range</button>
  <button id="tradedRange">Traded range</button>
  <button id="zoomIn">Zoom in</button>
  <button id="zoomOut">Zoom out</button>
  <button id="resetView">Reset view</button>
</section>
<div id="stats" class="statbar"></div>
<div class="viewbar">
  <span class="viewstatus" id="viewStatus"></span>
  <label>Trade #<input id="tradeJump" type="number" min="1" step="1"></label>
  <button id="jumpTrade">Jump</button>
  <button id="prevTrade">Prev trade</button>
  <button id="nextTrade">Next trade</button>
</div>
<p class="note">Wheel zooms under the cursor. Drag pans across regular-session SPX bars with overnight gaps removed. Double-click a marker or table row to inspect one trade window. Candles are SPX 1-minute OHLC bars.</p>
<main class="wrap">
  <canvas id="chart"></canvas>
  <div class="tablewrap"><table id="tradeTable"></table></div>
</main>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const SPX = DATA.spx;
const TRADES = DATA.trades;
SPX.forEach((p, i) => { if (!Number.isFinite(p.bar)) p.bar = i; });
const tooltip = document.getElementById('tooltip');
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
let hitMarkers = [];
let chartLayout = null;
let dataStart = 0;
let dataEnd = 0;
let tradeStart = 0;
let tradeEnd = 0;
let viewStart = 0;
let viewEnd = 0;
let selectedTradeKey = "";
let dragState = null;
let drawPending = false;
const MIN_VIEW_BARS = 12;
const FOCUS_PAD_BARS = 20;
const WHEEL_ZOOM_IN = 0.90;
const WHEEL_ZOOM_OUT = 1.12;

function yyyyMMdd(ms) { return new Date(ms).toISOString().slice(0,10); }
function money(v) { return (v < 0 ? '-' : '') + '$' + Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0}); }
function dt(ms) { return new Date(ms).toISOString().replace('T',' ').slice(0,16) + 'Z'; }
function minValue(rows, pick) {
  let best = Infinity;
  for (const row of rows) {
    const value = pick(row);
    if (Number.isFinite(value) && value < best) best = value;
  }
  return best;
}
function maxValue(rows, pick) {
  let best = -Infinity;
  for (const row of rows) {
    const value = pick(row);
    if (Number.isFinite(value) && value > best) best = value;
  }
  return best;
}
function tradeKey(t) { return `${t.seed}:${t.trade_number}:${t.candidate_uid}`; }
function clampBar(value) {
  return Math.max(dataStart, Math.min(dataEnd, value));
}
function barAt(value) {
  const idx = Math.max(0, Math.min(SPX.length - 1, Math.round(value)));
  return SPX[idx];
}
function msAtBar(value) {
  return barAt(value)?.t ?? 0;
}
function entryBar(t) {
  return Number.isFinite(t.entry_bar) ? Number(t.entry_bar) : lowerBarForTime(t.decision_ms);
}
function exitBar(t) {
  return Number.isFinite(t.exit_bar) ? Number(t.exit_bar) : lowerBarForTime(t.exit_ms);
}
function lowerBarForTime(ms) {
  let lo = 0, hi = SPX.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (SPX[mid].t < ms) lo = mid + 1; else hi = mid;
  }
  return clampBar(lo);
}
function upperBarForTime(ms) {
  let lo = 0, hi = SPX.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (SPX[mid].t <= ms) lo = mid + 1; else hi = mid;
  }
  return clampBar(lo - 1);
}
function seedValues() { return [...new Set(TRADES.map(t => t.seed))].sort((a,b)=>a-b); }
function stageValues() { return [...new Set(TRADES.map(t => t.stage))].sort(); }
function setupControls() {
  const seed = document.getElementById('seed');
  seed.innerHTML = seedValues().map(s => `<option value="${s}">paper seed ${s}</option>`).join('');
  const stage = document.getElementById('stage');
  stage.innerHTML += stageValues().map(s => `<option value="${s}">${s}</option>`).join('');
  dataStart = minValue(SPX, p => p.bar);
  dataEnd = maxValue(SPX, p => p.bar);
  tradeStart = minValue(TRADES, entryBar);
  tradeEnd = maxValue(TRADES, exitBar);
  viewStart = tradeStart;
  viewEnd = tradeEnd;
  syncDateInputs();
  document.getElementById('fullRange').onclick = () => {
    setView(dataStart, dataEnd, {syncDates: true});
  };
  document.getElementById('tradedRange').onclick = () => {
    setView(tradeStart, tradeEnd, {syncDates: true});
  };
  document.getElementById('resetView').onclick = () => setView(tradeStart, tradeEnd, {syncDates: true});
  document.getElementById('zoomIn').onclick = () => zoomAt(0.72, (viewStart + viewEnd) / 2);
  document.getElementById('zoomOut').onclick = () => zoomAt(1.42, (viewStart + viewEnd) / 2);
  document.getElementById('jumpTrade').onclick = jumpToTradeInput;
  document.getElementById('prevTrade').onclick = () => stepTrade(-1);
  document.getElementById('nextTrade').onclick = () => stepTrade(1);
  document.getElementById('tradeJump').onkeydown = ev => { if (ev.key === 'Enter') jumpToTradeInput(); };
  document.getElementById('startDate').onchange = setDateViewFromInputs;
  document.getElementById('endDate').onchange = setDateViewFromInputs;
  for (const id of ['seed','stage','chartMode']) document.getElementById(id).onchange = () => {
    selectedTradeKey = "";
    scheduleDraw();
  };
  document.getElementById('tradeTable').onclick = ev => {
    const row = ev.target.closest('tr[data-trade-number]');
    if (!row) return;
    const t = findTradeByNumber(Number(row.dataset.tradeNumber));
    if (!t) return;
    selectedTradeKey = tradeKey(t);
    document.getElementById('tradeJump').value = t.trade_number;
    scheduleDraw();
  };
  document.getElementById('tradeTable').ondblclick = ev => {
    const row = ev.target.closest('tr[data-trade-number]');
    if (!row) return;
    const t = findTradeByNumber(Number(row.dataset.tradeNumber));
    if (t) focusTrade(t);
  };
}
function activeRange() {
  return [viewStart, viewEnd];
}
function syncDateInputs() {
  document.getElementById('startDate').value = yyyyMMdd(msAtBar(viewStart));
  document.getElementById('endDate').value = yyyyMMdd(msAtBar(viewEnd));
}
function setDateViewFromInputs() {
  const start = Date.parse(document.getElementById('startDate').value + 'T00:00:00Z');
  const end = Date.parse(document.getElementById('endDate').value + 'T23:59:59Z');
  if (Number.isFinite(start) && Number.isFinite(end)) setView(lowerBarForTime(start), upperBarForTime(end), {syncDates: false});
}
function setView(start, end, opts = {}) {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return;
  if (end < start) [start, end] = [end, start];
  let span = Math.max(MIN_VIEW_BARS, end - start);
  const maxSpan = Math.max(MIN_VIEW_BARS, dataEnd - dataStart);
  span = Math.min(span, maxSpan);
  let nextStart = start;
  let nextEnd = start + span;
  if (nextStart < dataStart) { nextStart = dataStart; nextEnd = dataStart + span; }
  if (nextEnd > dataEnd) { nextEnd = dataEnd; nextStart = dataEnd - span; }
  viewStart = Math.max(dataStart, nextStart);
  viewEnd = Math.min(dataEnd, nextEnd);
  if (opts.syncDates) syncDateInputs();
  scheduleDraw();
}
function zoomAt(factor, focusMs) {
  if (!Number.isFinite(focusMs)) focusMs = (viewStart + viewEnd) / 2;
  const span = viewEnd - viewStart;
  const nextSpan = Math.max(MIN_VIEW_BARS, Math.min(dataEnd - dataStart, span * factor));
  const ratio = Math.max(0, Math.min(1, (focusMs - viewStart) / span));
  setView(focusMs - nextSpan * ratio, focusMs + nextSpan * (1 - ratio), {syncDates: false});
}
function canvasBarFromClientX(clientX) {
  if (!chartLayout) return (viewStart + viewEnd) / 2;
  const r = canvas.getBoundingClientRect();
  const x = Math.max(chartLayout.pad.l, Math.min(chartLayout.w - chartLayout.pad.r, clientX - r.left));
  const plotW = chartLayout.w - chartLayout.pad.l - chartLayout.pad.r;
  return chartLayout.start + (x - chartLayout.pad.l) / plotW * (chartLayout.end - chartLayout.start);
}
function focusTrade(t) {
  selectedTradeKey = tradeKey(t);
  document.getElementById('tradeJump').value = t.trade_number;
  const start = Math.min(entryBar(t), exitBar(t)) - FOCUS_PAD_BARS;
  const end = Math.max(entryBar(t), exitBar(t)) + FOCUS_PAD_BARS;
  setView(start, end, {syncDates: true});
}
function visibleSeedTrades() {
  const seed = Number(document.getElementById('seed').value);
  const stage = document.getElementById('stage').value;
  return TRADES
    .filter(t => t.seed === seed && (stage === 'all' || t.stage === stage))
    .sort((a,b)=>a.trade_number-b.trade_number);
}
function findTradeByNumber(n) {
  return visibleSeedTrades().find(t => t.trade_number === n);
}
function jumpToTradeInput() {
  const n = Number(document.getElementById('tradeJump').value);
  const t = findTradeByNumber(n);
  if (t) focusTrade(t);
}
function stepTrade(direction) {
  const rows = visibleSeedTrades();
  if (!rows.length) return;
  const current = Number(document.getElementById('tradeJump').value);
  let idx = rows.findIndex(t => t.trade_number === current);
  if (idx < 0) {
    idx = direction > 0 ? -1 : rows.length;
  }
  idx = Math.max(0, Math.min(rows.length - 1, idx + direction));
  focusTrade(rows[idx]);
}
function nearestMarker(mx, my, limit = 14) {
  let best = null, bestD = limit;
  for (const m of hitMarkers) {
    const d = Math.hypot(mx - m.x, my - m.y);
    if (d < bestD) { best = m; bestD = d; }
  }
  return best;
}
function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => {
    drawPending = false;
    draw();
  });
}
function activeTrades() {
  const seed = Number(document.getElementById('seed').value);
  const stage = document.getElementById('stage').value;
  const [start, end] = activeRange();
  return TRADES.filter(t => t.seed === seed && (stage === 'all' || t.stage === stage) && entryBar(t) <= end && exitBar(t) >= start);
}
function visibleSpX() {
  const [start, end] = activeRange();
  return SPX.filter(p => p.bar >= Math.floor(start) && p.bar <= Math.ceil(end));
}
function priceLow(p) {
  return Number.isFinite(p.low) ? p.low : p.close;
}
function priceHigh(p) {
  return Number.isFinite(p.high) ? p.high : p.close;
}
function priceOpen(p) {
  return Number.isFinite(p.open) ? p.open : p.close;
}
function priceClose(p) {
  return Number.isFinite(p.close) ? p.close : priceOpen(p);
}
function resize() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(900, Math.floor(rect.width * devicePixelRatio));
  canvas.height = Math.floor(640 * devicePixelRatio);
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}
function draw() {
  resize();
  hitMarkers = [];
  const w = canvas.clientWidth, h = 640;
  const pad = {l:70,r:24,t:24,b:42};
  ctx.clearRect(0,0,w,h);
  const series = visibleSpX();
  const trades = activeTrades();
  if (series.length < 2) return;
  const [start, end] = activeRange();
  const chartMode = document.getElementById('chartMode').value;
  let ymin = chartMode === 'candles' ? minValue(series, priceLow) : minValue(series, priceClose);
  let ymax = chartMode === 'candles' ? maxValue(series, priceHigh) : maxValue(series, priceClose);
  if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) return;
  const span = Math.max(10, ymax-ymin);
  ymin -= span*0.08; ymax += span*0.08;
  const x = bar => pad.l + (bar-start)/(end-start)*(w-pad.l-pad.r);
  const y = v => pad.t + (ymax-v)/(ymax-ymin)*(h-pad.t-pad.b);
  chartLayout = {w, h, pad, start, end};

  ctx.strokeStyle = 'rgba(255,255,255,.10)';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#9aa7b2';
  ctx.font = '11px system-ui';
  for (let i=0;i<=5;i++) {
    const yy = pad.t + i/5*(h-pad.t-pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w-pad.r, yy); ctx.stroke();
    const val = ymax - i/5*(ymax-ymin);
    ctx.fillText(val.toFixed(0), 12, yy+4);
  }
  if (chartMode === 'candles') drawCandles(series, x, y, w, pad);
  else drawCloseLine(series, x, y);

  for (const t of trades) {
    if (t.entry_spx == null || t.exit_spx == null) continue;
    const ex = x(entryBar(t)), ey = y(t.entry_spx), xx = x(exitBar(t)), xy = y(t.exit_spx);
    ctx.strokeStyle = 'rgba(210,218,226,.35)';
    ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(ex,ey); ctx.lineTo(xx,xy); ctx.stroke(); ctx.setLineDash([]);
  }
  for (const t of trades) {
    if (t.entry_spx == null) continue;
    const win = t.pnl >= 0;
    const color = win ? '#46d39a' : '#ff6b6b';
    const ex = x(entryBar(t)), ey = y(t.entry_spx);
    const selected = tradeKey(t) === selectedTradeKey;
    if (selected) drawHalo(ex, ey);
    drawTriangle(ex, ey, t.right === 'C', color);
    hitMarkers.push({x:ex,y:ey,t,kind:'entry'});
    if (t.exit_spx != null) {
      const xx = x(exitBar(t)), xy = y(t.exit_spx);
      if (selected) drawHalo(xx, xy);
      drawX(xx, xy, color);
      hitMarkers.push({x:xx,y:xy,t,kind:'exit'});
    }
  }
  drawAxisLabels(w,h,pad,start,end);
  renderViewStatus(start, end);
  renderStats(trades);
  renderTable(trades);
}
function drawHalo(x,y) {
  ctx.strokeStyle = '#ffd166';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, 13, 0, Math.PI*2);
  ctx.stroke();
}
function drawCloseLine(series, x, y) {
  ctx.strokeStyle = '#6cb6ff';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  series.forEach((p,i)=> {
    const xx=x(p.bar), yy=y(priceClose(p));
    if (i===0) ctx.moveTo(xx,yy); else ctx.lineTo(xx,yy);
  });
  ctx.stroke();
}
function drawCandles(series, x, y, w, pad) {
  const plotW = w - pad.l - pad.r;
  const candleW = Math.max(1, Math.min(12, plotW / Math.max(series.length, 1) * 0.72));
  const bodyMin = candleW < 2 ? 1 : 2;
  for (const p of series) {
    const open = priceOpen(p), high = priceHigh(p), low = priceLow(p), close = priceClose(p);
    if (![open, high, low, close].every(Number.isFinite)) continue;
    const xx = x(p.bar);
    const yo = y(open), yh = y(high), yl = y(low), yc = y(close);
    const up = close >= open;
    const stroke = up ? '#46d39a' : '#ff6b6b';
    const fill = up ? 'rgba(70,211,154,.70)' : 'rgba(255,107,107,.70)';
    ctx.strokeStyle = stroke;
    ctx.lineWidth = candleW < 3 ? 1 : 1.25;
    ctx.beginPath();
    ctx.moveTo(xx, yh);
    ctx.lineTo(xx, yl);
    ctx.stroke();
    const top = Math.min(yo, yc);
    const height = Math.max(bodyMin, Math.abs(yc - yo));
    if (candleW <= 1.5) {
      ctx.strokeStyle = fill;
      ctx.beginPath();
      ctx.moveTo(xx, top);
      ctx.lineTo(xx, top + height);
      ctx.stroke();
    } else {
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.fillRect(xx - candleW/2, top, candleW, height);
      ctx.strokeRect(xx - candleW/2, top, candleW, height);
    }
  }
}
function drawTriangle(x,y,up,color) {
  ctx.fillStyle = color; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
  ctx.beginPath();
  if (up) { ctx.moveTo(x,y-8); ctx.lineTo(x-7,y+6); ctx.lineTo(x+7,y+6); }
  else { ctx.moveTo(x,y+8); ctx.lineTo(x-7,y-6); ctx.lineTo(x+7,y-6); }
  ctx.closePath(); ctx.fill(); ctx.stroke();
}
function drawX(x,y,color) {
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x-5,y-5); ctx.lineTo(x+5,y+5); ctx.moveTo(x+5,y-5); ctx.lineTo(x-5,y+5); ctx.stroke();
}
function drawAxisLabels(w,h,pad,start,end) {
  ctx.fillStyle = '#9aa7b2'; ctx.font = '11px system-ui';
  for (let i=0;i<=6;i++) {
    const bar = start + i/6*(end-start);
    const ms = msAtBar(bar);
    const xx = pad.l + i/6*(w-pad.l-pad.r);
    const label = end - start < 500 ? dt(ms).slice(5,16).replace(' ', ' ') : yyyyMMdd(ms);
    ctx.fillText(label, xx-34, h-16);
  }
}
function renderViewStatus(start, end) {
  const bars = Math.max(1, Math.round(end-start));
  document.getElementById('viewStatus').textContent = `View: ${dt(msAtBar(start))} to ${dt(msAtBar(end))} (${bars.toLocaleString()} market bars)`;
}
function renderStats(trades) {
  const pnl = trades.reduce((a,t)=>a+t.pnl,0);
  const wins = trades.filter(t=>t.pnl>=0).length;
  const calls = trades.filter(t=>t.right==='C').length;
  const puts = trades.filter(t=>t.right==='P').length;
  const maxPremium = trades.reduce((a,t)=>Math.max(a, Number.isFinite(t.paper_premium) ? t.paper_premium : 0), 0);
  let cash = Number(DATA.startingEquity || 0), peak = cash, maxDD = 0;
  const daily = new Map();
  for (const t of trades.slice().sort((a,b)=>a.exit_ms-b.exit_ms || a.decision_ms-b.decision_ms)) {
    cash += t.pnl;
    peak = Math.max(peak, cash);
    maxDD = Math.min(maxDD, cash - peak);
    daily.set(t.session, (daily.get(t.session) || 0) + t.pnl);
  }
  const worstDay = daily.size ? Math.min(...daily.values()) : 0;
  const html = [
    ['Paper start', money(DATA.startingEquity || 0)],
    ['Trades', trades.length],
    ['Replay PnL', money(pnl)],
    ['Ending equity', money((DATA.startingEquity || 0) + pnl)],
    ['Win rate', trades.length ? (wins/trades.length*100).toFixed(1)+'%' : 'n/a'],
    ['Calls / puts', `${calls} / ${puts}`],
    ['Avg PnL', trades.length ? money(pnl/trades.length) : 'n/a'],
    ['Max DD', money(maxDD)],
    ['Worst day', money(worstDay)],
    ['Max known BP', money(maxPremium)],
    ['Unaffordable skips', DATA.unaffordableSkippedTrades ?? 0]
  ].map(([k,v])=>`<div class="stat"><b>${k}</b>: ${v}</div>`).join('');
  document.getElementById('stats').innerHTML = html;
}
function renderTable(trades) {
  const rows = trades.slice().sort((a,b)=>a.decision_ms-b.decision_ms).slice(0,500);
  document.getElementById('tradeTable').innerHTML = `<thead><tr><th>#</th><th>stage</th><th>session</th><th>time</th><th>side</th><th>offset</th><th>SPX</th><th>premium</th><th>cash before</th><th>PnL</th><th>equity after</th><th>exit</th></tr></thead><tbody>` +
    rows.map(t=>`<tr data-trade-number="${t.trade_number}" class="${tradeKey(t)===selectedTradeKey?'selected':''}"><td>${t.trade_number}</td><td>${t.stage}</td><td>${t.session}</td><td>${dt(t.decision_ms).slice(11)}</td><td>${t.side}</td><td>${t.offset}</td><td>${t.entry_spx?.toFixed(1) ?? ''}</td><td>${Number.isFinite(t.paper_premium) ? money(t.paper_premium) : ''}</td><td>${Number.isFinite(t.paper_cash_before) ? money(t.paper_cash_before) : ''}</td><td style="color:${t.pnl>=0?'#46d39a':'#ff6b6b'}">${money(t.pnl)}</td><td>${Number.isFinite(t.paper_equity_after) ? money(t.paper_equity_after) : money((DATA.startingEquity || 0) + t.cumulative_pnl)}</td><td>${t.exit_reason}</td></tr>`).join('') + `</tbody>`;
}
function showMarkerTooltip(best, ev) {
  const t = best.t;
  tooltip.innerHTML = `<b>${best.kind.toUpperCase()} #${t.trade_number}</b><br>${t.stage} / ${t.segment}<br>${t.session} ${t.side} offset ${t.offset}<br>Entry: ${dt(t.decision_ms)} @ SPX ${t.entry_spx?.toFixed(2)}<br>Exit: ${dt(t.exit_ms)} @ SPX ${t.exit_spx?.toFixed(2)}<br>Premium: ${Number.isFinite(t.paper_premium) ? money(t.paper_premium) : 'n/a'}<br>Cash before: ${Number.isFinite(t.paper_cash_before) ? money(t.paper_cash_before) : 'n/a'}<br>PnL: <b style="color:${t.pnl>=0?'#46d39a':'#ff6b6b'}">${money(t.pnl)}</b><br>Equity after: ${Number.isFinite(t.paper_equity_after) ? money(t.paper_equity_after) : 'n/a'}<br>${t.contract_id}<br>${t.exit_reason}`;
  tooltip.style.left = (ev.clientX + 14) + 'px'; tooltip.style.top = (ev.clientY + 14) + 'px'; tooltip.style.display = 'block';
}
canvas.addEventListener('wheel', ev => {
  ev.preventDefault();
  zoomAt(ev.deltaY < 0 ? WHEEL_ZOOM_IN : WHEEL_ZOOM_OUT, canvasBarFromClientX(ev.clientX));
}, {passive: false});
canvas.addEventListener('pointerdown', ev => {
  if (ev.button !== 0) return;
  dragState = {x: ev.clientX, start: viewStart, end: viewEnd, moved: false};
  canvas.classList.add('dragging');
  canvas.setPointerCapture(ev.pointerId);
});
canvas.addEventListener('pointermove', ev => {
  const r = canvas.getBoundingClientRect();
  const mx = ev.clientX-r.left, my = ev.clientY-r.top;
  if (dragState) {
    const dx = ev.clientX - dragState.x;
    if (Math.abs(dx) > 3) dragState.moved = true;
    if (dragState.moved && chartLayout) {
      const plotW = chartLayout.w - chartLayout.pad.l - chartLayout.pad.r;
      const shift = -dx / plotW * (dragState.end - dragState.start);
      setView(dragState.start + shift, dragState.end + shift, {syncDates: false});
    }
    tooltip.style.display='none';
    return;
  }
  const best = nearestMarker(mx, my, 12);
  if (!best) { tooltip.style.display='none'; return; }
  showMarkerTooltip(best, ev);
});
canvas.addEventListener('pointerup', ev => {
  const wasDrag = dragState?.moved;
  dragState = null;
  canvas.classList.remove('dragging');
  try { canvas.releasePointerCapture(ev.pointerId); } catch {}
  if (wasDrag) canvas.dataset.suppressClick = "1";
  setTimeout(() => { canvas.dataset.suppressClick = ""; }, 0);
});
canvas.addEventListener('click', ev => {
  if (canvas.dataset.suppressClick) return;
  const r = canvas.getBoundingClientRect();
  const best = nearestMarker(ev.clientX-r.left, ev.clientY-r.top, 12);
  if (!best) return;
  selectedTradeKey = tradeKey(best.t);
  document.getElementById('tradeJump').value = best.t.trade_number;
  showMarkerTooltip(best, ev);
  scheduleDraw();
});
canvas.addEventListener('dblclick', ev => {
  const r = canvas.getBoundingClientRect();
  const best = nearestMarker(ev.clientX-r.left, ev.clientY-r.top, 18);
  if (best) focusTrade(best.t);
  else zoomAt(0.72, canvasBarFromClientX(ev.clientX));
});
canvas.addEventListener('mouseleave', () => tooltip.style.display='none');
window.addEventListener('resize', scheduleDraw);
setupControls();
draw();
</script>
"""


EQUITY_BODY = r"""
<header>
  <h1 id="chartTitle">Protocol 101 Equity Curve</h1>
  <div class="sub" id="chartSubtitle">Single paper-account replay for one frozen seed. A trade is shown only if one contract was affordable from current cash at entry.</div>
  <div class="replay-warning">HISTORICAL REPLAY ONLY. This is the paper-account inspection curve, not live/paper-trading approval.</div>
</header>
<section class="controls">
  <label>Account<select id="seed"></select></label>
  <label>Starting equity<input id="startingEquity" type="number" min="0" step="1000" disabled></label>
  <button id="reset">Reset</button>
</section>
<div id="stats" class="statbar"></div>
<main class="wrap">
  <canvas id="chart"></canvas>
  <div class="tablewrap"><table id="seedTable"></table></div>
  <h2>Daily PnL</h2>
  <div class="tablewrap"><table id="dailyTable"></table></div>
</main>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const TRADES = DATA.trades;
const tooltip = document.getElementById('tooltip');
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
let hitMarkers = [];
const palette = ['#72a7ff','#46d39a','#ffd166','#ff8fab','#b4a7ff','#7bdff2'];
function money(v) { return (v < 0 ? '-' : '') + '$' + Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0}); }
function maybeMoney(v) { return Number.isFinite(v) ? money(v) : 'n/a'; }
function pct(v) { return Number.isFinite(v) ? (v*100).toFixed(1)+'%' : 'n/a'; }
function dt(ms) { return new Date(ms).toISOString().slice(0,10); }
function seedValues() { return [...new Set(TRADES.map(t => t.seed))].sort((a,b)=>a-b); }
function setupControls() {
  document.getElementById('chartTitle').textContent = DATA.title || 'Protocol 101 Equity Curve';
  document.getElementById('chartSubtitle').textContent = DATA.subtitle || '';
  document.getElementById('seed').innerHTML = seedValues().map(s => `<option value="${s}">paper seed ${s}</option>`).join('');
  document.getElementById('startingEquity').value = DATA.startingEquity;
  document.getElementById('reset').onclick = () => {
    document.getElementById('startingEquity').value = DATA.startingEquity;
    document.getElementById('seed').value = String(seedValues()[0] ?? '');
    draw();
  };
  document.getElementById('seed').onchange = draw;
}
function bySeed() {
  const out = new Map();
  for (const t of TRADES) {
    if (!out.has(t.seed)) out.set(t.seed, []);
    out.get(t.seed).push(t);
  }
  for (const rows of out.values()) rows.sort((a,b)=>a.exit_ms-b.exit_ms || a.decision_ms-b.decision_ms);
  return out;
}
function commissionCost() {
  // dollars per side -- no x100 contract multiplier
  return Number(DATA.commissionPerSide || 0) * 2;
}
function stressCost() {
  // option price points per side -- takes the x100 multiplier
  return Number(DATA.stressPerSide || 0) * 2 * 100;
}
function tradePnl(t, stressed=false) {
  return t.pnl - commissionCost() - (stressed ? stressCost() : 0);
}
function equitySeries(rows, start, stressed=false) {
  let cash = start;
  const points = [{x:0, ms: rows[0]?.decision_ms ?? Date.now(), y: cash, pnl: 0, label: 'start'}];
  rows.forEach((t,i)=> {
    const pnl = tradePnl(t, stressed);
    cash += pnl;
    points.push({x:i+1, ms:t.exit_ms, y:cash, pnl, trade:t, stressed});
  });
  return points;
}
function premiumPaid(t) {
  if (Number.isFinite(t.premium_paid)) return t.premium_paid;
  if (Number.isFinite(t.entry_ask)) return t.entry_ask * 100;
  return NaN;
}
function pathMae(t) {
  return Number.isFinite(t.path_mae) ? t.path_mae : NaN;
}
function accountSummary(rows, startEq, stressed=false) {
  const sorted = rows.slice().sort((a,b)=>a.exit_ms-b.exit_ms || a.decision_ms-b.decision_ms);
  let cash = startEq, peak = startEq, maxDD = 0, maxDDPct = 0, minEquity = startEq, pnl = 0, wins = 0;
  const daily = new Map();
  const premiums = [];
  const maes = [];
  for (const t of sorted) {
    const tradeValue = tradePnl(t, stressed);
    pnl += tradeValue;
    wins += tradeValue >= 0 ? 1 : 0;
    daily.set(t.session, (daily.get(t.session) || 0) + tradeValue);
    const premium = premiumPaid(t);
    if (Number.isFinite(premium)) premiums.push(premium);
    const mae = pathMae(t);
    if (Number.isFinite(mae)) maes.push(mae);
    cash += tradeValue;
    minEquity = Math.min(minEquity, cash);
    peak = Math.max(peak, cash);
    const dd = cash - peak;
    maxDD = Math.min(maxDD, dd);
    if (peak > 0) maxDDPct = Math.min(maxDDPct, dd / peak);
  }
  const worstDay = daily.size ? Math.min(...daily.values()) : 0;
  const maxBp = premiums.length ? Math.max(...premiums) : NaN;
  const avgBp = premiums.length ? premiums.reduce((a,b)=>a+b,0) / premiums.length : NaN;
  const worstMae = maes.length ? Math.min(...maes) : NaN;
  return {
    trades: sorted.length,
    pnl,
    wins,
    ending: startEq + pnl,
    roc: startEq ? pnl / startEq : NaN,
    maxDD,
    maxDDPct,
    minEquity,
    worstDay,
    maxDailyLoss: Math.max(0, -worstDay),
    maxBp,
    avgBp,
    premiumCoverage: sorted.length ? premiums.length / sorted.length : 0,
    worstMae,
    maxIntratradeLoss: Number.isFinite(worstMae) ? Math.max(0, -worstMae) : NaN,
    pathCoverage: sorted.length ? maes.length / sorted.length : 0
  };
}
function dailySummary(rows, startEq, stressed=false) {
  const byDay = new Map();
  for (const t of rows) {
    if (!byDay.has(t.session)) byDay.set(t.session, {session:t.session, trades:0, wins:0, pnl:0});
    const row = byDay.get(t.session);
    const value = tradePnl(t, stressed);
    row.trades += 1;
    row.wins += value >= 0 ? 1 : 0;
    row.pnl += value;
  }
  let cash = startEq;
  return [...byDay.values()]
    .sort((a,b)=>a.session.localeCompare(b.session))
    .map(row => {
      cash += row.pnl;
      return {...row, endingEquity: cash};
    });
}
function resize() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(900, Math.floor(rect.width * devicePixelRatio));
  canvas.height = Math.floor(640 * devicePixelRatio);
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}
function draw() {
  resize(); hitMarkers = [];
  const w = canvas.clientWidth, h = 640, pad = {l:78,r:28,t:24,b:42};
  ctx.clearRect(0,0,w,h);
  const selected = document.getElementById('seed').value;
  const startEq = Number(document.getElementById('startingEquity').value || 0);
  const grouped = bySeed();
  const seeds = [Number(selected)];
  const seed = seeds[0];
  const baseRows = grouped.get(seed) || [];
  const series = [
    {seed, label:'NBBO replay', color:'#72a7ff', stressed:false, points: equitySeries(baseRows, startEq, false)},
    {seed, label:`+$${Number(DATA.stressPerSide || 0).toFixed(2)}/side stress`, color:'#ffd166', stressed:true, points: equitySeries(baseRows, startEq, true)}
  ];
  const maxLen = Math.max(...series.map(s=>s.points.length), 2);
  let ymin = Math.min(...series.flatMap(s=>s.points.map(p=>p.y)));
  let ymax = Math.max(...series.flatMap(s=>s.points.map(p=>p.y)));
  const span = Math.max(1000, ymax-ymin); ymin -= span*.08; ymax += span*.08;
  const x = i => pad.l + i/(maxLen-1)*(w-pad.l-pad.r);
  const y = v => pad.t + (ymax-v)/(ymax-ymin)*(h-pad.t-pad.b);
  ctx.strokeStyle = 'rgba(255,255,255,.10)'; ctx.lineWidth = 1; ctx.fillStyle = '#9aa7b2'; ctx.font = '11px system-ui';
  for (let i=0;i<=5;i++) { const yy=pad.t+i/5*(h-pad.t-pad.b); ctx.beginPath(); ctx.moveTo(pad.l,yy); ctx.lineTo(w-pad.r,yy); ctx.stroke(); const val=ymax-i/5*(ymax-ymin); ctx.fillText(money(val), 10, yy+4); }
  ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.setLineDash([5,4]); ctx.beginPath(); ctx.moveTo(pad.l,y(startEq)); ctx.lineTo(w-pad.r,y(startEq)); ctx.stroke(); ctx.setLineDash([]);
  series.forEach((s,idx)=> {
    const color = s.color || palette[idx % palette.length];
    ctx.strokeStyle = color; ctx.lineWidth = s.stressed ? 1.7 : 2.7;
    if (s.stressed) ctx.setLineDash([6,4]);
    ctx.beginPath();
    s.points.forEach((p,i)=> { const xx=x(i), yy=y(p.y); if(i===0) ctx.moveTo(xx,yy); else ctx.lineTo(xx,yy); });
    ctx.stroke();
    ctx.setLineDash([]);
    if (!s.stressed) {
      s.points.forEach((p,i)=> { if (!p.trade) return; const xx=x(i), yy=y(p.y); ctx.fillStyle=p.pnl>=0?'#46d39a':'#ff6b6b'; ctx.beginPath(); ctx.arc(xx,yy,4, 0, Math.PI*2); ctx.fill(); hitMarkers.push({x:xx,y:yy,seed:s.seed,point:p}); });
    }
    ctx.fillStyle = color; const last = s.points[s.points.length-1]; ctx.fillText(s.label, x(s.points.length-1)+4, y(last.y));
  });
  ctx.fillStyle = '#9aa7b2'; ctx.fillText('trade number', w/2-30, h-12);
  renderStats(grouped, seeds, startEq);
  renderTable(grouped, startEq);
  renderDailyTable(grouped, seeds, startEq);
}
function renderStats(grouped, seeds, startEq) {
  const rows = seeds.flatMap(seed => grouped.get(seed) || []);
  const summaries = seeds.map(seed => accountSummary(grouped.get(seed) || [], startEq, false));
  const stressSummaries = seeds.map(seed => accountSummary(grouped.get(seed) || [], startEq, true));
  const pnl = summaries.reduce((a,s)=>a+s.pnl,0);
  const stressPnl = stressSummaries.reduce((a,s)=>a+s.pnl,0);
  const trades = summaries.reduce((a,s)=>a+s.trades,0);
  const wins = summaries.reduce((a,s)=>a+s.wins,0);
  const worstDD = summaries.length ? Math.min(...summaries.map(s=>s.maxDD)) : 0;
  const worstDay = summaries.length ? Math.min(...summaries.map(s=>s.worstDay)) : 0;
  const finiteBp = summaries.map(s=>s.maxBp).filter(Number.isFinite);
  const maxBp = finiteBp.length ? Math.max(...finiteBp) : NaN;
  const finiteMae = summaries.map(s=>s.worstMae).filter(Number.isFinite);
  const worstMae = finiteMae.length ? Math.min(...finiteMae) : NaN;
  const knownPremiums = rows.filter(t => Number.isFinite(premiumPaid(t))).length;
  const knownPaths = rows.filter(t => Number.isFinite(pathMae(t))).length;
  document.getElementById('stats').innerHTML = [
    ['Paper start', money(startEq)],
    ['Seed', seeds.join(', ')],
    ['Trades', trades],
    ['Replay PnL', money(pnl)],
    ['Ending equity', money(startEq + pnl)],
    ['Stress ending equity', money(startEq + stressPnl)],
    ['ROC on start cash', pct(startEq ? pnl/startEq : NaN)],
    ['Stress ROC', pct(startEq ? stressPnl/startEq : NaN)],
    ['Max DD', money(worstDD)],
    ['Worst day', money(worstDay)],
    ['Worst intratrade MAE', maybeMoney(worstMae)],
    ['Max known BP', maybeMoney(maxBp)],
    ['Unaffordable skips', DATA.unaffordableSkippedTrades ?? 0],
    ['BP coverage', trades ? (knownPremiums/trades*100).toFixed(1)+'%' : 'n/a'],
    ['Path coverage', trades ? (knownPaths/trades*100).toFixed(1)+'%' : 'n/a'],
    ['Win rate', trades ? (wins/trades*100).toFixed(1)+'%' : 'n/a']
  ].map(([k,v])=>`<div class="stat"><b>${k}</b>: ${v}</div>`).join('');
}
function renderTable(grouped, startEq) {
  const rows = seedValues().map(seed => {
    const trades = grouped.get(seed) || [];
    return {seed, ...accountSummary(trades, startEq), stress: accountSummary(trades, startEq, true)};
  });
  document.getElementById('seedTable').innerHTML = '<thead><tr><th>seed</th><th>trades</th><th>win rate</th><th>total pnl</th><th>ending equity</th><th>stress ending</th><th>return on start</th><th>stress return</th><th>max drawdown</th><th>max DD %</th><th>worst day</th><th>max daily loss</th><th>worst MAE</th><th>max known BP</th><th>BP coverage</th><th>path coverage</th></tr></thead><tbody>' +
    rows.map(r=>`<tr><td>${r.seed}</td><td>${r.trades}</td><td>${r.trades?(r.wins/r.trades*100).toFixed(1)+'%':'n/a'}</td><td style="color:${r.pnl>=0?'#46d39a':'#ff6b6b'}">${money(r.pnl)}</td><td>${money(r.ending)}</td><td>${money(r.stress.ending)}</td><td>${pct(r.roc)}</td><td>${pct(r.stress.roc)}</td><td>${money(r.maxDD)}</td><td>${pct(r.maxDDPct)}</td><td>${money(r.worstDay)}</td><td>${money(r.maxDailyLoss)}</td><td>${maybeMoney(r.worstMae)}</td><td>${maybeMoney(r.maxBp)}</td><td>${pct(r.premiumCoverage)}</td><td>${pct(r.pathCoverage)}</td></tr>`).join('') + '</tbody>';
}
function renderDailyTable(grouped, seeds, startEq) {
  const rows = seeds.flatMap(seed => dailySummary(grouped.get(seed) || [], startEq, false));
  const worst = rows.slice().sort((a,b)=>a.pnl-b.pnl).slice(0,20);
  const best = rows.slice().sort((a,b)=>b.pnl-a.pnl).slice(0,20);
  const selected = [...worst, ...best]
    .filter((row, idx, arr)=>arr.findIndex(other=>other.session===row.session)===idx)
    .sort((a,b)=>a.session.localeCompare(b.session));
  document.getElementById('dailyTable').innerHTML = '<thead><tr><th>session</th><th>trades</th><th>win rate</th><th>daily PnL</th><th>ending equity</th></tr></thead><tbody>' +
    selected.map(r=>`<tr><td>${r.session}</td><td>${r.trades}</td><td>${r.trades?(r.wins/r.trades*100).toFixed(1)+'%':'n/a'}</td><td style="color:${r.pnl>=0?'#46d39a':'#ff6b6b'}">${money(r.pnl)}</td><td>${money(r.endingEquity)}</td></tr>`).join('') + '</tbody>';
}
canvas.addEventListener('mousemove', ev => {
  const r = canvas.getBoundingClientRect(); const mx=ev.clientX-r.left, my=ev.clientY-r.top;
  let best=null, bestD=10; for (const m of hitMarkers) { const d=Math.hypot(mx-m.x,my-m.y); if(d<bestD){best=m;bestD=d;} }
  if(!best){tooltip.style.display='none'; return;}
  const t=best.point.trade;
  const displayedStart = Number(document.getElementById('startingEquity').value || 0);
  tooltip.innerHTML = `<b>Seed ${best.seed} trade #${t.trade_number}</b><br>${dt(t.exit_ms)} ${t.side} ${t.offset}<br>${t.stage} / ${t.segment}<br>Trade PnL: <b style="color:${t.pnl>=0?'#46d39a':'#ff6b6b'}">${money(t.pnl)}</b><br>Stress PnL: ${money(tradePnl(t, true))}<br>Cumulative PnL: ${money(t.cumulative_pnl)}<br>Equity: ${money(displayedStart+t.cumulative_pnl)}`;
  tooltip.style.left=(ev.clientX+14)+'px'; tooltip.style.top=(ev.clientY+14)+'px'; tooltip.style.display='block';
});
canvas.addEventListener('mouseleave',()=>tooltip.style.display='none');
window.addEventListener('resize', draw);
setupControls();
draw();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
