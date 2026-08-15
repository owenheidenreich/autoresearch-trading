"""Protocol191: Protocol081 replay of Protocol190 selected entries.

Protocol190 is a fast full-coverage neural screen that used executable
baseline stop/target/25m exits. This runner keeps those selected entry
decisions frozen, recomputes exits with the frozen Protocol081 lifecycle stack,
and replays them through a serial one-account simulator.

This is the bridge test between "full-action entry signal exists" and "it
survives the actual lifecycle exit model." It uses already-collected data only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.protocol066_inference import load_protocol066_artifact
from v4.scripts.build_lifecycle_sequence_dataset import _load_session, _normalized_session_path
from v4.scripts.run_protocol081_live_shadow_router import DEFAULT_PROTOCOL081_MANIFEST
from v4.scripts.run_protocol164_full_action_space_dataset import (
    CONTRACT_MULTIPLIER,
    STARTING_CASH,
    _build_fast_exit_path,
    _contract_path_cache,
    _finite_float,
    _protocol081_exit_batch,
)


LOOP_ID = "v4_aplus_hypothesis_191_protocol081_replay_of_protocol190_entries"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_MODEL_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_190_full_coverage_surface_edge_baseline_exit_screen/protocol183_model_trades.csv"
)
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-trades", type=Path, default=DEFAULT_MODEL_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--protocol081-manifest", type=Path, default=DEFAULT_PROTOCOL081_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = _load_entries(args.model_trades)
    artifact = load_protocol066_artifact(args.protocol081_manifest)
    replayed, path_skips = _build_protocol081_paths(
        entries,
        normalized_dir=args.normalized_dir,
        artifact=artifact,
        forced_flat_time=str(args.forced_flat_time),
    )
    serial, serial_skips = _serial_replay(replayed, starting_cash=float(args.starting_cash), slippage_per_side=0.0)
    serial_10, _ = _serial_replay(replayed, starting_cash=float(args.starting_cash), slippage_per_side=0.10)
    serial_25, _ = _serial_replay(replayed, starting_cash=float(args.starting_cash), slippage_per_side=0.25)
    split_seed = _aggregate_by_split_seed(serial, serial_10, serial_25)
    payload = {
        "protocol": "191_protocol081_replay_of_protocol190_entries",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "source_model_trades": str(args.model_trades),
        "protocol081_manifest": str(args.protocol081_manifest),
        "source_entries": int(len(entries)),
        "path_rows": int(len(replayed)),
        "path_skips": _counts(path_skips, "path_status"),
        "serial_skips": _counts(serial_skips, "skip_reason"),
        "aggregate": split_seed,
        "decision": _decision(split_seed),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(replayed).to_csv(args.out_dir / "protocol191_protocol081_candidate_paths.csv", index=False)
    pd.DataFrame(serial).to_csv(args.out_dir / "protocol191_protocol081_serial_trades.csv", index=False)
    pd.DataFrame(serial_skips).to_csv(args.out_dir / "protocol191_protocol081_serial_skips.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def _load_entries(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["decision_ts"] = pd.to_datetime(frame["decision_time"], utc=True)
    frame["contract_id"] = frame["contract_id"].astype(str)
    frame["session"] = frame["session"].astype(str)
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    if "canonical_entry_uid" not in frame:
        frame["canonical_entry_uid"] = frame["session"].astype(str) + "|" + frame["decision_time"].astype(str) + "|" + frame["contract_id"].astype(str)
    frame["edge"] = pd.to_numeric(frame.get("score", 0.0), errors="coerce").fillna(0.0)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int)
    return frame.sort_values(["fold", "seed", "reported_split", "session", "decision_ts", "contract_id"]).reset_index(drop=True)


def _build_protocol081_paths(
    entries: pd.DataFrame,
    *,
    normalized_dir: Path,
    artifact: Any,
    forced_flat_time: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    if entries.empty:
        return rows, skips
    for session, group in entries.groupby("session", sort=True):
        session_path = _normalized_session_path(normalized_dir, str(session))
        if session_path is None:
            skips.extend({**row._asdict(), "path_status": "missing_normalized_session"} for row in group.itertuples(index=False))
            continue
        contracts = set(group["contract_id"].astype(str))
        normalized = _load_session(session_path, contracts)
        by_contract = {
            contract_id: _contract_path_cache(part)
            for contract_id, part in normalized.groupby("contract_id", sort=False)
        }
        prepared: list[tuple[pd.Series, dict[str, Any], pd.DataFrame]] = []
        for _, entry in group.iterrows():
            cache = by_contract.get(str(entry["contract_id"]))
            if cache is None or len(cache["time_ns"]) == 0:
                skips.append({**entry.to_dict(), "path_status": "missing_contract_path"})
                continue
            trade_row, step_rows = _build_fast_exit_path(entry, cache, forced_flat_time=forced_flat_time)
            if str(trade_row.get("path_status")) != "ok" or not step_rows:
                skips.append({**entry.to_dict(), "path_status": str(trade_row.get("path_status"))})
                continue
            from v4.scripts.run_protocol162_may2026_serial_lifecycle_replay import _prepare_steps

            prepared.append((entry, trade_row, _prepare_steps(pd.DataFrame(step_rows), artifact)))
        exits = _protocol081_exit_batch([item[2] for item in prepared], artifact)
        for (entry, trade_row, _steps), exit_info in zip(prepared, exits):
            rows.append(
                {
                    **entry.to_dict(),
                    "protocol190_exit_time": entry.get("exit_time"),
                    "protocol190_pnl": _finite_float(entry.get("pnl"), 0.0),
                    "path_status": "ok",
                    "entry_quote_time": trade_row.get("entry_quote_time"),
                    "entry_bid": _finite_float(trade_row.get("entry_bid"), np.nan),
                    "entry_ask": _finite_float(trade_row.get("entry_ask"), np.nan),
                    "entry_mid": _finite_float(trade_row.get("entry_mid"), np.nan),
                    "entry_premium": _finite_float(trade_row.get("entry_ask"), 0.0) * CONTRACT_MULTIPLIER,
                    "candidate_exit_time": exit_info["candidate_exit_time"],
                    "candidate_exit_dt": pd.Timestamp(exit_info["candidate_exit_time"]),
                    "candidate_pnl": float(exit_info["candidate_pnl"]),
                    "candidate_exit_reason": str(exit_info["candidate_exit_reason"]),
                    "candidate_action": str(exit_info["candidate_action"]),
                    "predicted_continuation_value": float(exit_info["predicted_continuation_value"]),
                    "predicted_recovery_probability": float(exit_info["predicted_recovery_probability"]),
                    "predicted_decay_probability": float(exit_info["predicted_decay_probability"]),
                    "label_source": "protocol081_replay_of_protocol190_entry",
                }
            )
    return rows, skips


def _serial_replay(rows: list[dict[str, Any]], *, starting_cash: float, slippage_per_side: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    equity_by_key: dict[tuple[str, int, str], float] = {}
    open_until: dict[tuple[str, int, str, str], pd.Timestamp] = {}
    for row in sorted(rows, key=lambda item: (str(item["fold"]), int(item["seed"]), str(item["reported_split"]), str(item["session"]), pd.Timestamp(item["decision_time"]))):
        key = (str(row["fold"]), int(row["seed"]), str(row["reported_split"]))
        session_key = (*key, str(row["session"]))
        equity = equity_by_key.get(key, float(starting_cash))
        decision_time = pd.Timestamp(row["decision_time"])
        if session_key in open_until and decision_time < open_until[session_key]:
            skips.append({**row, "skip_reason": "open_position", "slippage_per_side": float(slippage_per_side)})
            continue
        entry_ask = _finite_float(row.get("entry_ask"), np.nan)
        premium = (entry_ask + float(slippage_per_side)) * CONTRACT_MULTIPLIER
        if not np.isfinite(premium) or premium <= 0.0 or premium > equity:
            skips.append({**row, "skip_reason": "unaffordable_or_invalid", "slippage_per_side": float(slippage_per_side)})
            continue
        round_trip = float(slippage_per_side) * 2.0 * CONTRACT_MULTIPLIER
        pnl = _finite_float(row.get("candidate_pnl"), 0.0) - round_trip
        trade = {
            **row,
            "slippage_per_side": float(slippage_per_side),
            "account_equity_before": float(equity),
            "account_equity_after": float(equity + pnl),
            "pnl": float(pnl),
            "raw_candidate_pnl": _finite_float(row.get("candidate_pnl"), 0.0),
        }
        trades.append(trade)
        equity_by_key[key] = float(equity + pnl)
        open_until[session_key] = pd.Timestamp(row["candidate_exit_dt"])
    return trades, skips


def _aggregate_by_split_seed(
    base: list[dict[str, Any]],
    stress10: list[dict[str, Any]],
    stress25: list[dict[str, Any]],
) -> dict[str, Any]:
    out = {}
    for split in sorted({str(row["reported_split"]) for row in base} | {str(row["reported_split"]) for row in stress10}):
        seed_rows = []
        for seed in sorted({int(row["seed"]) for row in base if str(row["reported_split"]) == split}):
            rows = [row for row in base if str(row["reported_split"]) == split and int(row["seed"]) == seed]
            rows10 = [row for row in stress10 if str(row["reported_split"]) == split and int(row["seed"]) == seed]
            rows25 = [row for row in stress25 if str(row["reported_split"]) == split and int(row["seed"]) == seed]
            seed_rows.append(
                {
                    "seed": int(seed),
                    **_summary(rows),
                    "stress_0_10_total_pnl": float(sum(_finite_float(row.get("pnl"), 0.0) for row in rows10)),
                    "stress_0_25_total_pnl": float(sum(_finite_float(row.get("pnl"), 0.0) for row in rows25)),
                }
            )
        out[split] = {
            "seeds": len(seed_rows),
            "median_total_pnl": _median(seed_rows, "total_pnl"),
            "median_profit_factor": _median(seed_rows, "profit_factor"),
            "median_trades": _median(seed_rows, "trades"),
            "median_stress_0_10_total_pnl": _median(seed_rows, "stress_0_10_total_pnl"),
            "median_stress_0_25_total_pnl": _median(seed_rows, "stress_0_25_total_pnl"),
            "positive_seed_fraction": float(np.mean([row["total_pnl"] > 0.0 for row in seed_rows])) if seed_rows else 0.0,
            "seed_rows": seed_rows,
        }
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_finite_float(row.get("pnl"), 0.0) for row in rows]
    wins = [pnl for pnl in pnls if pnl > 0.0]
    losses = [pnl for pnl in pnls if pnl < 0.0]
    gross_loss = -sum(losses)
    gross_win = sum(wins)
    return {
        "trades": int(len(rows)),
        "total_pnl": float(sum(pnls)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
    }


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(np.median(values)) if values else 0.0


def _counts(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column))
        out[key] = out.get(key, 0) + 1
    return out


def _decision(aggregate: dict[str, Any]) -> str:
    required = ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]
    if all(
        aggregate.get(split, {}).get("median_total_pnl", 0.0) > 0.0
        and aggregate.get(split, {}).get("median_profit_factor", 0.0) >= 1.15
        and aggregate.get(split, {}).get("median_stress_0_10_total_pnl", 0.0) > 0.0
        for split in required
    ):
        return "keep_research_candidate: Protocol190 entries survive Protocol081 serial replay"
    return "research_only: Protocol190 entries do not yet clear Protocol081 serial replay"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol191 Protocol081 Replay Of Protocol190 Entries",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Source entries: `{payload['source_entries']}`",
        f"- Path rows: `{payload['path_rows']}`",
        f"- Path skips: `{payload['path_skips']}`",
        f"- Serial skips: `{payload['serial_skips']}`",
        "",
        "## Aggregate",
        "",
        "| split | seeds | median PnL | PF | stress 0.10 | stress 0.25 | trades | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in payload["aggregate"].items():
        lines.append(
            f"| {split} | {item['seeds']} | {item['median_total_pnl']:.0f} | {item['median_profit_factor']:.2f} | "
            f"{item['median_stress_0_10_total_pnl']:.0f} | {item['median_stress_0_25_total_pnl']:.0f} | "
            f"{item['median_trades']:.0f} | {item['positive_seed_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Protocol081 paths: `{path.parent / 'protocol191_protocol081_candidate_paths.csv'}`",
            f"- Serial trades: `{path.parent / 'protocol191_protocol081_serial_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
