"""Protocol188: full-coverage full-action dataset with baseline exits.

This is a speed-first screening dataset, not a promotion artifact. Protocol187
showed that building frozen Protocol081 exits for every full-ladder candidate
across all collected sessions is too slow for the active research loop. This
runner keeps the same full flat-state candidate universe, account columns, and
executable ask-entry/bid-exit pricing, but uses the simple stop/target/25m
baseline path exit so we can test whether full calendar coverage plus surface
edge is worth the heavier Protocol081 rebuild.

No paid data is downloaded and no broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from v4.scripts.build_lifecycle_sequence_dataset import _deadline, _load_session, _normalized_session_path
from v4.scripts.run_protocol164_full_action_space_dataset import (
    CONTRACT_MULTIPLIER,
    DEFAULT_BLOCKS,
    DEFAULT_NORMALIZED_DIR,
    FULL_ACTION_FEATURE_COLUMNS,
    _contract_path_cache,
    _finite_float,
    _first_exit_index,
    _merge_parts,
    _part_len,
    _part_path,
    _safe_part_name,
    _skip_path,
    parse_blocks,
    processed_rows_to_candidates,
)


LOOP_ID = "v4_aplus_hypothesis_188_full_action_baseline_exit_dataset"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", nargs="*", default=list(DEFAULT_BLOCKS), help="split=processed_dir entries")
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-sessions-per-block", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out_dir / "session_parts"
    skips_dir = args.out_dir / "session_skips"
    if args.resume or args.merge_only:
        parts_dir.mkdir(parents=True, exist_ok=True)
        skips_dir.mkdir(parents=True, exist_ok=True)
    blocks = parse_blocks(args.blocks)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sessions_processed = 0
    stop = False
    if not args.merge_only:
        for split, processed_dir in blocks:
            block_sessions = 0
            for path in sorted(processed_dir.glob("*.pkl")):
                session = path.stem
                part_path = _part_path(parts_dir, split, session)
                skip_path = _skip_path(skips_dir, split, session)
                if args.resume and part_path.exists() and skip_path.exists():
                    sessions_processed += 1
                    block_sessions += 1
                    print(json.dumps({"split": split, "session": session, "status": "cached", "candidates": _part_len(part_path)}), flush=True)
                    if int(args.max_sessions) > 0 and sessions_processed >= int(args.max_sessions):
                        stop = True
                        break
                    if int(args.max_sessions_per_block) > 0 and block_sessions >= int(args.max_sessions_per_block):
                        break
                    continue
                session_rows, session_skips = build_session_baseline_candidates(
                    split=split,
                    session=session,
                    processed_path=path,
                    normalized_dir=args.normalized_dir,
                    starting_cash=float(args.starting_cash),
                    forced_flat_time=str(args.forced_flat_time),
                )
                if args.resume:
                    pd.DataFrame(session_rows).to_parquet(part_path, index=False)
                    pd.DataFrame(session_skips).to_csv(skip_path, index=False)
                else:
                    rows.extend(session_rows)
                    skipped.extend(session_skips)
                sessions_processed += 1
                block_sessions += 1
                print(json.dumps({"split": split, "session": session, "status": "built", "candidates": len(session_rows), "skipped": len(session_skips)}), flush=True)
                if int(args.max_sessions) > 0 and sessions_processed >= int(args.max_sessions):
                    stop = True
                    break
                if int(args.max_sessions_per_block) > 0 and block_sessions >= int(args.max_sessions_per_block):
                    break
            if stop:
                break
    if args.resume or args.merge_only:
        dataset, skipped_frame = _merge_parts(parts_dir, skips_dir)
    else:
        dataset = pd.DataFrame(rows)
        skipped_frame = pd.DataFrame(skipped)
    if not dataset.empty:
        dataset = dataset.sort_values(["split", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)
        dataset.to_parquet(args.out_dir / "protocol188_full_action_baseline_exit_dataset.parquet", index=False)
    skipped_frame.to_csv(args.out_dir / "protocol188_skipped_candidates.csv", index=False)
    payload = {
        "protocol": "188_full_action_baseline_exit_dataset",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "candidate_source": "processed_neural_full_ladder_no_protocol101_gate",
        "exit_source": "baseline_stop50_target100_hold25m_executable_bid_exit",
        "starting_cash": float(args.starting_cash),
        "feature_columns": FULL_ACTION_FEATURE_COLUMNS,
        "rows": int(len(dataset)),
        "skipped_rows": int(len(skipped_frame)),
        "sessions_processed": int(sessions_processed),
        "rows_by_split": _counts(dataset, "split"),
        "sessions_by_split": _sessions_by_split(dataset),
        "path_status": _counts(skipped_frame, "path_status"),
        "decision": "ready_for_full_coverage_screening" if not dataset.empty else "blocked_no_candidates",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "rows": payload["rows"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if payload["decision"] == "ready_for_full_coverage_screening" else 1


def build_session_baseline_candidates(
    *,
    split: str,
    session: str,
    processed_path: Path,
    normalized_dir: Path,
    starting_cash: float,
    forced_flat_time: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    processed_rows = pickle.loads(processed_path.read_bytes())
    candidates = processed_rows_to_candidates(
        split=split,
        session=session,
        rows=processed_rows,
        starting_cash=starting_cash,
    )
    if not candidates:
        return [], []
    session_path = _normalized_session_path(normalized_dir, session)
    if session_path is None:
        return [], [{**row, "path_status": "missing_normalized_session"} for row in candidates]
    contracts = {str(row["contract_id"]) for row in candidates}
    normalized = _load_session(session_path, contracts)
    by_contract = {
        contract_id: _contract_path_cache(group)
        for contract_id, group in normalized.groupby("contract_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source_row, candidate in enumerate(candidates):
        cache = by_contract.get(str(candidate["contract_id"]))
        if cache is None or len(cache["time_ns"]) == 0:
            skipped.append({**candidate, "path_status": "missing_contract_path"})
            continue
        exit_info = _baseline_exit(candidate, cache, forced_flat_time=forced_flat_time)
        if exit_info["path_status"] != "ok":
            skipped.append({**candidate, "path_status": exit_info["path_status"]})
            continue
        rows.append(
            {
                **candidate,
                "source_row": int(source_row),
                "path_status": "ok",
                "candidate_exit_time": exit_info["candidate_exit_time"],
                "candidate_exit_dt": pd.Timestamp(exit_info["candidate_exit_time"]),
                "candidate_exit_step": int(exit_info["candidate_exit_step"]),
                "candidate_pnl": float(exit_info["candidate_pnl"]),
                "candidate_exit_reason": str(exit_info["candidate_exit_reason"]),
                "candidate_action": str(exit_info["candidate_action"]),
                "predicted_continuation_value": 0.0,
                "predicted_recovery_probability": 0.0,
                "predicted_decay_probability": 0.0,
                "path_points": int(exit_info["path_points"]),
                "path_max_pnl": float(exit_info["path_max_pnl"]),
                "path_min_pnl": float(exit_info["path_min_pnl"]),
                "path_final_pnl": float(exit_info["path_final_pnl"]),
                "label_source": "baseline_stop50_target100_hold25m_full_action",
            }
        )
    return rows, skipped


def _baseline_exit(candidate: dict[str, Any], cache: dict[str, Any], *, forced_flat_time: str) -> dict[str, Any]:
    decision_time = pd.Timestamp(candidate["decision_dt"])
    if decision_time.tzinfo is None:
        decision_time = decision_time.tz_localize("UTC")
    else:
        decision_time = decision_time.tz_convert("UTC")
    deadline = _deadline(decision_time, forced_flat_time)
    times = np.asarray(cache["time_ns"], dtype=np.int64)
    decision_ns = int(decision_time.value)
    deadline_ns = int(deadline.value)
    entry_idx = int(np.searchsorted(times, decision_ns, side="right") - 1)
    if entry_idx < 0:
        return {"path_status": "missing_entry_quote"}
    entry_ask = _finite_float(cache["ask"][entry_idx], np.nan)
    if not np.isfinite(entry_ask) or entry_ask <= 0.0:
        return {"path_status": "invalid_entry_ask"}
    start_idx = int(np.searchsorted(times, decision_ns, side="right"))
    end_idx = int(np.searchsorted(times, deadline_ns, side="right"))
    idxs = np.arange(start_idx, end_idx, dtype=np.int64)
    if len(idxs):
        idxs = idxs[np.isfinite(cache["bid"][idxs])]
    if len(idxs) == 0:
        return {"path_status": "missing_future_path"}
    pnls = (cache["bid"][idxs] - entry_ask) * CONTRACT_MULTIPLIER
    exit_idx, reason = _first_exit_index(pnls, entry_ask)
    abs_idx = int(idxs[exit_idx])
    return {
        "path_status": "ok",
        "candidate_exit_time": cache["quote_time"][abs_idx].isoformat(),
        "candidate_exit_step": int(exit_idx),
        "candidate_pnl": float(pnls[exit_idx]),
        "candidate_exit_reason": reason,
        "candidate_action": "stop" if reason == "hard_stop" else ("exit" if reason == "target" else "forced_flat"),
        "path_points": int(len(idxs)),
        "path_max_pnl": float(np.nanmax(pnls)),
        "path_min_pnl": float(np.nanmin(pnls)),
        "path_final_pnl": float(pnls[-1]),
    }


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).sort_index().items()}


def _sessions_by_split(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    return {str(k): int(v) for k, v in frame.groupby("split")["session"].nunique().sort_index().items()}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol188 Full-Action Baseline-Exit Dataset",
        "",
        "Builds a full-calendar, full-ladder candidate screen with executable baseline stop/target/hold exits. This is a fast research screen, not a Protocol081 promotion artifact.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Rows: `{payload['rows']}`",
        f"- Skipped rows: `{payload['skipped_rows']}`",
        f"- Sessions processed: `{payload['sessions_processed']}`",
        f"- Rows by split: `{payload['rows_by_split']}`",
        f"- Sessions by split: `{payload['sessions_by_split']}`",
        f"- Path status skips: `{payload['path_status']}`",
        "",
        "## Outputs",
        "",
        f"- Dataset: `{path.parent / 'protocol188_full_action_baseline_exit_dataset.parquet'}`",
        f"- Skipped candidates: `{path.parent / 'protocol188_skipped_candidates.csv'}`",
        f"- Summary: `{path.parent / 'summary.json'}`",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
