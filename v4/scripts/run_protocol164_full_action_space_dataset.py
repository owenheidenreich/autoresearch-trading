"""Protocol 164: full action-space serial candidate dataset.

This runner builds a research dataset shaped like the live bot's flat-state
choice: wait, or enter one affordable SPXW 0DTE call/put from the full ATM
plus/minus 50 dollar ladder. It intentionally avoids Protocol101's min-edge,
time-bucket, and selected-candidate gates.

No paid data is downloaded and no broker endpoint is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from v4.dataset.spxw_0dte_neural import MARKET_FEATURE_NAMES, OPTION_FEATURE_NAMES
from v4.model.environment_diagnostics import time_bucket
from v4.scripts.build_lifecycle_sequence_dataset import _deadline, _load_session, _minutes_to_forced_flat, _normalized_session_path
from v4.scripts.run_protocol081_live_shadow_router import DEFAULT_PROTOCOL081_MANIFEST
from v4.live.protocol066_inference import prediction_for_step
from v4.scripts.run_protocol162_may2026_serial_lifecycle_replay import _prepare_steps
from v4.live.protocol066_inference import load_protocol066_artifact


LOOP_ID = "v4_aplus_hypothesis_164_full_action_space_dataset"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_BLOCKS = (
    "q1_2025=data/processed/spxw_0dte_neural_q1_2025_official_context",
    "q2_2025=data/processed/spxw_0dte_neural_q2_2025_official_context",
    "q3_2025=data/processed/spxw_0dte_neural_q3_2025_official_context",
    "q4_2025=data/processed/spxw_0dte_neural_q4_2025_official_context",
    "q1_2026=data/processed/spxw_0dte_neural_q1_2026_official_context",
    "recent_2026=data/processed/spxw_0dte_neural_protocol163_recent_official_context",
)
STARTING_CASH = 10_000.0
CONTRACT_MULTIPLIER = 100.0
NO_NEW_ENTRIES_AFTER_MINUTE = 15 * 60 + 30
SESSION_OPEN_MINUTE = 9 * 60 + 30
SESSION_MINUTES = NO_NEW_ENTRIES_AFTER_MINUTE - SESSION_OPEN_MINUTE
FULL_ACTION_FEATURE_COLUMNS = [
    "entry_minutes_since_open",
    "entry_minutes_to_forced_flat",
    "entry_progress",
    "entry_progress_sin",
    "entry_progress_cos",
    "entry_is_first_30m",
    "entry_is_post_open_morning",
    "entry_is_midday",
    "entry_is_late_afternoon",
    "right_is_call",
    "right_is_put",
    "offset",
    "abs_offset",
    "entry_bid",
    "entry_ask",
    "entry_mid",
    "entry_spread",
    "entry_spread_frac",
    "entry_bid_size",
    "entry_ask_size",
    "entry_underlying_price",
    "entry_iv",
    "entry_delta",
    "entry_abs_delta",
    "entry_gamma",
    "entry_theta",
    "entry_abs_theta",
    "entry_gamma_theta_ratio",
    "entry_theta_over_mid",
    "entry_theta_burden",
    "entry_gamma_per_premium",
    "entry_premium_over_underlying",
    "entry_spread_over_mid",
    "entry_size_imbalance",
    "entry_call_delta_signed",
    "entry_put_delta_signed",
    "entry_premium",
    "entry_premium_frac_10k",
    "entry_affordable_10k",
    "market_spx_close",
    "market_vix_close",
    "market_spx_vwap",
    "market_omar",
    "market_session_range",
    "market_momentum_5m",
    "market_momentum_15m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", nargs="*", default=list(DEFAULT_BLOCKS), help="split=processed_dir entries")
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--protocol081-manifest", type=Path, default=DEFAULT_PROTOCOL081_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-sessions-per-block", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--resume", action="store_true", help="Write/read per-session parts and skip completed sessions.")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing per-session parts into the final dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_protocol066_artifact(args.protocol081_manifest)
    blocks = parse_blocks(args.blocks)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sessions_processed = 0
    stop = False
    parts_dir = args.out_dir / "session_parts"
    skips_dir = args.out_dir / "session_skips"
    if args.resume or args.merge_only:
        parts_dir.mkdir(parents=True, exist_ok=True)
        skips_dir.mkdir(parents=True, exist_ok=True)
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
                    part_rows = _part_len(part_path)
                    print(json.dumps({"split": split, "session": session, "status": "cached", "candidates": part_rows}))
                    if int(args.max_sessions) > 0 and sessions_processed >= int(args.max_sessions):
                        stop = True
                        break
                    if int(args.max_sessions_per_block) > 0 and block_sessions >= int(args.max_sessions_per_block):
                        break
                    continue
                session_rows, session_skipped = build_session_candidates(
                    split=split,
                    session=session,
                    processed_path=path,
                    normalized_dir=args.normalized_dir,
                    artifact=artifact,
                    starting_cash=float(args.starting_cash),
                    forced_flat_time=str(args.forced_flat_time),
                    max_candidates=0 if int(args.max_candidates) <= 0 else int(args.max_candidates) - len(rows),
                )
                if args.resume:
                    pd.DataFrame(session_rows).to_parquet(part_path, index=False)
                    pd.DataFrame(session_skipped).to_csv(skip_path, index=False)
                else:
                    rows.extend(session_rows)
                    skipped.extend(session_skipped)
                sessions_processed += 1
                block_sessions += 1
                print(json.dumps({"split": split, "session": session, "status": "built", "candidates": len(session_rows), "skipped": len(session_skipped)}))
                if int(args.max_candidates) > 0 and len(rows) >= int(args.max_candidates):
                    stop = True
                    break
                if int(args.max_sessions) > 0 and sessions_processed >= int(args.max_sessions):
                    stop = True
                    break
                if int(args.max_sessions_per_block) > 0 and block_sessions >= int(args.max_sessions_per_block):
                    break
            if stop:
                break

    if args.resume or args.merge_only:
        dataset, skipped_frame = _merge_parts(parts_dir, skips_dir)
        skipped = skipped_frame.to_dict("records")
    else:
        dataset = pd.DataFrame(rows)
        skipped_frame = pd.DataFrame(skipped)
    if not dataset.empty:
        dataset = dataset.sort_values(["split", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)
        dataset.to_parquet(args.out_dir / "protocol164_full_action_space_dataset.parquet", index=False)
    skipped_frame.to_csv(args.out_dir / "protocol164_skipped_candidates.csv", index=False)
    payload = {
        "protocol": "164_full_action_space_dataset",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "candidate_source": "processed_neural_full_ladder_no_protocol101_gate",
        "exit_source": "frozen_protocol081_lifecycle",
        "protocol081_manifest": str(args.protocol081_manifest),
        "starting_cash": float(args.starting_cash),
        "feature_columns": FULL_ACTION_FEATURE_COLUMNS,
        "rows": int(len(dataset)),
        "skipped_rows": int(len(skipped_frame)),
        "sessions_processed": int(sessions_processed),
        "cached_session_parts": int(len(list(parts_dir.glob("*.parquet"))) if parts_dir.exists() else 0),
        "resume_enabled": bool(args.resume),
        "merge_only": bool(args.merge_only),
        "rows_by_split": _counts(dataset, "split"),
        "path_status": _counts(skipped_frame, "path_status"),
        "decision": "ready_for_protocol165_training" if not dataset.empty else "blocked_no_candidates",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "rows": payload["rows"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0 if payload["decision"] == "ready_for_protocol165_training" else 1


def parse_blocks(raw: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in raw:
        if "=" not in str(item):
            raise ValueError(f"block must be split=path, got {item!r}")
        split, path = str(item).split("=", 1)
        out.append((split, Path(path)))
    return out


def _safe_part_name(split: str, session: str) -> str:
    return f"{split}__{session}".replace("/", "_")


def _part_path(parts_dir: Path, split: str, session: str) -> Path:
    return parts_dir / f"{_safe_part_name(split, session)}.parquet"


def _skip_path(skips_dir: Path, split: str, session: str) -> Path:
    return skips_dir / f"{_safe_part_name(split, session)}.csv"


def _part_len(path: Path) -> int:
    try:
        return int(len(pd.read_parquet(path, columns=["candidate_uid"])))
    except Exception:
        return -1


def _merge_parts(parts_dir: Path, skips_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    part_frames = [pd.read_parquet(path) for path in sorted(parts_dir.glob("*.parquet"))]
    skip_frames = []
    for path in sorted(skips_dir.glob("*.csv")):
        try:
            skip_frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    dataset = pd.concat(part_frames, ignore_index=True, sort=False) if part_frames else pd.DataFrame()
    skipped = pd.concat(skip_frames, ignore_index=True, sort=False) if skip_frames else pd.DataFrame()
    return dataset, skipped


def build_session_candidates(
    *,
    split: str,
    session: str,
    processed_path: Path,
    normalized_dir: Path,
    artifact: Any,
    starting_cash: float,
    forced_flat_time: str,
    max_candidates: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    processed_rows = pickle.loads(processed_path.read_bytes())
    candidates = processed_rows_to_candidates(
        split=split,
        session=session,
        rows=processed_rows,
        starting_cash=starting_cash,
        max_candidates=max_candidates,
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
    prepared_items: list[tuple[int, dict[str, Any], dict[str, Any], pd.DataFrame]] = []
    for source_row, candidate in enumerate(candidates):
        contract_rows = by_contract.get(str(candidate["contract_id"]))
        if contract_rows is None or len(contract_rows["time_ns"]) == 0:
            skipped.append({**candidate, "path_status": "missing_contract_path"})
            continue
        trade = pd.Series(
            {
                **candidate,
                "source_row": source_row,
                "seed": 1,
                "decision_ts": pd.Timestamp(candidate["decision_dt"]),
                "baseline_pnl": 0.0,
                "dynamic_pnl": 0.0,
                "hold_minutes": np.nan,
                "exit_reason": "protocol164_full_action_space",
            }
        )
        trade_row, step_rows = _build_fast_exit_path(trade, contract_rows, forced_flat_time=forced_flat_time)
        if str(trade_row.get("path_status")) != "ok" or not step_rows:
            skipped.append({**candidate, "path_status": str(trade_row.get("path_status"))})
            continue
        steps = _prepare_steps(pd.DataFrame(step_rows), artifact)
        prepared_items.append((source_row, candidate, trade_row, steps))
    exit_infos = _protocol081_exit_batch([item[3] for item in prepared_items], artifact)
    for (source_row, candidate, trade_row, steps), exit_info in zip(prepared_items, exit_infos):
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
                "predicted_continuation_value": float(exit_info["predicted_continuation_value"]),
                "predicted_recovery_probability": float(exit_info["predicted_recovery_probability"]),
                "predicted_decay_probability": float(exit_info["predicted_decay_probability"]),
                "path_points": int(trade_row.get("path_points") or len(steps)),
                "path_max_pnl": _finite_float(trade_row.get("path_max_pnl"), 0.0),
                "path_min_pnl": _finite_float(trade_row.get("path_min_pnl"), 0.0),
                "path_final_pnl": _finite_float(trade_row.get("path_final_pnl"), 0.0),
                "label_source": "protocol081_full_action_space",
            }
        )
    return rows, skipped


def _contract_path_cache(group: pd.DataFrame) -> dict[str, Any]:
    frame = group.sort_values("quote_time").reset_index(drop=True)
    quote_time = pd.to_datetime(frame["quote_time"], utc=True)
    numeric = {}
    for column in [
        "bid",
        "ask",
        "mid",
        "bid_size",
        "ask_size",
        "quote_gap_seconds",
        "option_ohlcv_volume",
        "stat_open_interest",
        "underlying_price",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
    ]:
        numeric[column] = pd.to_numeric(frame.get(column), errors="coerce").to_numpy(dtype=float)
    return {
        "quote_time": list(quote_time),
        "time_ns": quote_time.dt.as_unit("ns").astype("int64").to_numpy(),
        **numeric,
    }


def _protocol081_exit_batch(
    step_frames: list[pd.DataFrame],
    artifact: Any,
    *,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """Batch frozen Protocol081 sequence inference for many candidate paths."""

    if not step_frames:
        return []
    outputs: list[dict[str, Any] | None] = [None] * len(step_frames)
    feature_columns = list(artifact.feature_columns)
    input_dim = len(feature_columns)
    with torch.no_grad():
        for start in range(0, len(step_frames), int(batch_size)):
            batch = step_frames[start : start + int(batch_size)]
            lengths = [len(frame) for frame in batch]
            max_len = max(lengths)
            values = np.zeros((len(batch), max_len, input_dim), dtype=np.float32)
            for idx, frame in enumerate(batch):
                raw = frame[feature_columns].to_numpy(dtype=np.float32)
                values[idx, : len(frame), :] = artifact.scaler.transform(raw)
            value, recovery, decay = artifact.model(torch.from_numpy(values))
            value_np = value.cpu().numpy() * artifact.target_scale
            recovery_np = torch.sigmoid(recovery).cpu().numpy()
            decay_np = torch.sigmoid(decay).cpu().numpy()
            for local_idx, frame in enumerate(batch):
                seq_value = value_np[local_idx, : lengths[local_idx]]
                seq_recovery = recovery_np[local_idx, : lengths[local_idx]]
                seq_decay = decay_np[local_idx, : lengths[local_idx]]
                outputs[start + local_idx] = _protocol081_exit_from_arrays(
                    frame,
                    value=seq_value,
                    recovery=seq_recovery,
                    decay=seq_decay,
                    artifact=artifact,
                )
    return [item for item in outputs if item is not None]


def _protocol081_exit_from_arrays(
    steps: pd.DataFrame,
    *,
    value: np.ndarray,
    recovery: np.ndarray,
    decay: np.ndarray,
    artifact: Any,
) -> dict[str, Any]:
    selected_idx = len(steps) - 1
    selected_prediction = None
    for idx, row in steps.iterrows():
        prediction = prediction_for_step(
            step_index=int(idx),
            value=value,
            recovery=recovery,
            decay=decay,
            override_threshold=artifact.selected_override_threshold,
            step_row=row,
        )
        selected_prediction = prediction
        if prediction.action in {"exit", "stop", "forced_flat"}:
            selected_idx = int(idx)
            break
    row = steps.iloc[selected_idx]
    if selected_prediction is None:
        selected_prediction = prediction_for_step(
            step_index=int(selected_idx),
            value=value,
            recovery=recovery,
            decay=decay,
            override_threshold=artifact.selected_override_threshold,
            step_row=row,
        )
    return {
        "candidate_exit_time": pd.Timestamp(row["quote_time"]).isoformat(),
        "candidate_exit_step": int(row["step_idx"]),
        "candidate_pnl": _finite_float(row["current_pnl"], 0.0),
        "candidate_exit_reason": selected_prediction.reason,
        "candidate_action": selected_prediction.action,
        "predicted_continuation_value": selected_prediction.predicted_continuation_value,
        "predicted_recovery_probability": selected_prediction.predicted_recovery_probability,
        "predicted_decay_probability": selected_prediction.predicted_decay_probability,
        "mfe_to_exit": _finite_float(row.get("mfe_to_now"), 0.0),
        "mae_to_exit": _finite_float(row.get("mae_to_now"), 0.0),
        "future_max_delta_at_exit": _finite_float(row.get("future_max_delta"), 0.0),
        "future_min_delta_at_exit": _finite_float(row.get("future_min_delta"), 0.0),
    }


def _build_fast_exit_path(
    trade: pd.Series,
    contract_rows: pd.DataFrame,
    *,
    forced_flat_time: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build only the causal step features required by frozen Protocol081.

    The older lifecycle attribution builder also computes future labels and
    per-step diagnostics for human analysis. Protocol164 only needs the causal
    sequence inputs plus executable PnL path, so this lighter path is materially
    faster for full-ladder/all-session builds.
    """

    decision_time = pd.Timestamp(trade["decision_ts"])
    if decision_time.tzinfo is None:
        decision_time = decision_time.tz_localize("UTC")
    else:
        decision_time = decision_time.tz_convert("UTC")
    deadline = _deadline(decision_time, forced_flat_time)
    base_trade = {
        "trade_uid": trade["trade_uid"],
        "canonical_entry_uid": trade["canonical_entry_uid"],
        "source_row": int(trade["source_row"]),
        "fold": trade.get("fold"),
        "split": trade.get("split"),
        "seed": int(trade.get("seed")),
        "session": trade.get("session"),
        "decision_time": decision_time.isoformat(),
        "contract_id": trade.get("contract_id"),
        "right": trade.get("right"),
        "offset": _finite_float(trade.get("offset"), np.nan),
        "edge": _finite_float(trade.get("edge"), np.nan),
        "deadline": deadline.isoformat(),
    }
    times = np.asarray(contract_rows["time_ns"], dtype=np.int64)
    decision_ns = int(decision_time.value)
    deadline_ns = int(deadline.value)
    entry_idx = int(np.searchsorted(times, decision_ns, side="right") - 1)
    if entry_idx < 0:
        return {**base_trade, "path_status": "missing_entry_quote"}, []
    entry_ask = _finite_float(contract_rows["ask"][entry_idx], np.nan)
    if not np.isfinite(entry_ask) or entry_ask <= 0.0:
        return {**base_trade, "path_status": "invalid_entry_ask", "entry_ask": entry_ask}, []
    start_idx = int(np.searchsorted(times, decision_ns, side="right"))
    end_idx = int(np.searchsorted(times, deadline_ns, side="right"))
    idxs = np.arange(start_idx, end_idx, dtype=np.int64)
    if len(idxs):
        idxs = idxs[np.isfinite(contract_rows["bid"][idxs])]
    if len(idxs) == 0:
        return {**base_trade, "path_status": "missing_future_path", "entry_ask": entry_ask}, []

    bid = contract_rows["bid"][idxs]
    ask = contract_rows["ask"][idxs]
    mid = contract_rows["mid"][idxs]
    gamma = contract_rows["gamma"][idxs]
    theta = contract_rows["theta"][idxs]
    pnls = (bid - entry_ask) * CONTRACT_MULTIPLIER
    baseline_idx, baseline_reason = _first_exit_index(pnls, entry_ask)
    mfe = np.maximum.accumulate(pnls)
    mae = np.minimum.accumulate(pnls)
    time_since_mfe = _time_since_running_max(pnls)
    giveback = np.maximum(0.0, mfe - pnls)
    giveback_fraction = np.divide(giveback, mfe, out=np.zeros_like(giveback), where=mfe > 0.0)
    spread = ask - bid
    spread_frac = np.divide(spread, mid, out=np.full_like(spread, np.nan), where=np.abs(mid) > 1e-9)
    path_times_ns = times[idxs]
    minutes_since_entry = (path_times_ns - decision_ns).astype(float) / (60.0 * 1_000_000_000.0)
    minutes_since_entry = np.maximum(minutes_since_entry, 1.0)
    minutes_to_deadline = np.maximum((deadline_ns - path_times_ns).astype(float) / (60.0 * 1_000_000_000.0), 0.0)
    theta_over_mid = np.divide(np.abs(theta), np.abs(mid), out=np.zeros_like(theta), where=np.abs(mid) > 1e-9)
    gamma_theta = np.divide(np.abs(gamma), np.abs(theta), out=np.zeros_like(gamma), where=np.abs(theta) > 1e-9)
    entry_mid = _finite_float(contract_rows["mid"][entry_idx], np.nan)
    rows: list[dict[str, Any]] = []
    quote_times = contract_rows["quote_time"]
    for idx, abs_idx in enumerate(idxs):
        timestamp = quote_times[int(abs_idx)]
        rows.append(
            {
                "trade_uid": trade["trade_uid"],
                "canonical_entry_uid": trade["canonical_entry_uid"],
                "source_row": int(trade["source_row"]),
                "fold": trade.get("fold"),
                "split": trade.get("split"),
                "seed": int(trade.get("seed")),
                "session": trade.get("session"),
                "decision_time": decision_time.isoformat(),
                "quote_time": timestamp.isoformat(),
                "contract_id": trade.get("contract_id"),
                "right": trade.get("right"),
                "step_idx": int(idx),
                "path_points": int(len(idxs)),
                "is_baseline_exit_step": bool(idx == baseline_idx),
                "baseline_exit_reason": baseline_reason,
                "minutes_since_entry": float(minutes_since_entry[idx]),
                "minutes_to_deadline": float(minutes_to_deadline[idx]),
                "minutes_to_forced_flat": _minutes_to_forced_flat(timestamp, forced_flat_time),
                "bid": _finite_float(contract_rows["bid"][abs_idx], np.nan),
                "ask": _finite_float(contract_rows["ask"][abs_idx], np.nan),
                "mid": _finite_float(contract_rows["mid"][abs_idx], np.nan),
                "spread": _finite_float(spread[idx], np.nan),
                "spread_frac": _finite_float(spread_frac[idx], np.nan),
                "bid_size": _finite_float(contract_rows["bid_size"][abs_idx], 0.0),
                "ask_size": _finite_float(contract_rows["ask_size"][abs_idx], 0.0),
                "quote_gap_seconds": _finite_float(contract_rows["quote_gap_seconds"][abs_idx], np.nan),
                "option_ohlcv_volume": _finite_float(contract_rows["option_ohlcv_volume"][abs_idx], np.nan),
                "stat_open_interest": _finite_float(contract_rows["stat_open_interest"][abs_idx], np.nan),
                "underlying_price": _finite_float(contract_rows["underlying_price"][abs_idx], np.nan),
                "iv": _finite_float(contract_rows["iv"][abs_idx], np.nan),
                "delta": _finite_float(contract_rows["delta"][abs_idx], np.nan),
                "gamma": _finite_float(contract_rows["gamma"][abs_idx], np.nan),
                "theta": _finite_float(contract_rows["theta"][abs_idx], np.nan),
                "vega": _finite_float(contract_rows["vega"][abs_idx], np.nan),
                "current_pnl": _finite_float(pnls[idx], 0.0),
                "mfe_to_now": _finite_float(mfe[idx], 0.0),
                "mae_to_now": _finite_float(mae[idx], 0.0),
                "giveback_from_mfe": _finite_float(giveback[idx], 0.0),
                "giveback_fraction": _finite_float(giveback_fraction[idx], 0.0),
                "time_since_mfe_minutes": float(time_since_mfe[idx]),
                "pnl_velocity_1": _path_velocity(pnls, idx, 1),
                "pnl_velocity_3": _path_velocity(pnls, idx, 3),
                "pnl_velocity_5": _path_velocity(pnls, idx, 5),
                "realized_pnl_vol_5": _rolling_vol(pnls, idx, 5),
                "realized_pnl_vol_10": _rolling_vol(pnls, idx, 10),
                "bid_over_entry_ask": _safe_ratio(_finite_float(contract_rows["bid"][abs_idx], np.nan), entry_ask, np.nan),
                "mid_over_entry_ask": _safe_ratio(_finite_float(contract_rows["mid"][abs_idx], np.nan), entry_ask, np.nan),
                "theta_over_mid": _finite_float(theta_over_mid[idx], 0.0),
                "gamma_theta_ratio": _finite_float(gamma_theta[idx], 0.0),
                "time_theta_burden": _finite_float(theta_over_mid[idx] * minutes_to_deadline[idx], 0.0),
                "entry_edge": _finite_float(trade.get("edge"), np.nan),
                "entry_offset": _finite_float(trade.get("offset"), np.nan),
                "entry_is_call": float(str(trade.get("right")) == "C"),
                "entry_is_put": float(str(trade.get("right")) == "P"),
            }
        )
    trade_row = {
        **base_trade,
        "path_status": "ok",
        "entry_quote_time": quote_times[int(entry_idx)].isoformat(),
        "entry_bid": _finite_float(contract_rows["bid"][entry_idx], np.nan),
        "entry_ask": entry_ask,
        "entry_mid": entry_mid,
        "entry_spread": _finite_float(contract_rows["ask"][entry_idx], np.nan) - _finite_float(contract_rows["bid"][entry_idx], np.nan),
        "entry_spread_frac": _safe_ratio(
            _finite_float(contract_rows["ask"][entry_idx], np.nan) - _finite_float(contract_rows["bid"][entry_idx], np.nan),
            entry_mid,
            np.nan,
        ),
        "entry_underlying_price": _finite_float(contract_rows["underlying_price"][entry_idx], np.nan),
        "entry_iv": _finite_float(contract_rows["iv"][entry_idx], np.nan),
        "entry_delta": _finite_float(contract_rows["delta"][entry_idx], np.nan),
        "entry_gamma": _finite_float(contract_rows["gamma"][entry_idx], np.nan),
        "entry_theta": _finite_float(contract_rows["theta"][entry_idx], np.nan),
        "path_points": int(len(idxs)),
        "path_max_pnl": _finite_float(np.nanmax(pnls), 0.0),
        "path_min_pnl": _finite_float(np.nanmin(pnls), 0.0),
        "path_final_pnl": _finite_float(pnls[-1], 0.0),
        "baseline_exit_step": int(baseline_idx),
        "baseline_exit_reason": baseline_reason,
        "baseline_path_pnl": _finite_float(pnls[baseline_idx], 0.0),
    }
    return trade_row, rows


def _first_exit_index(path_pnl: np.ndarray, entry_ask: float) -> tuple[int, str]:
    stop_pnl = -0.50 * float(entry_ask) * CONTRACT_MULTIPLIER
    target_pnl = 1.00 * float(entry_ask) * CONTRACT_MULTIPLIER
    for idx, pnl in enumerate(path_pnl):
        if pnl <= stop_pnl:
            return int(idx), "hard_stop"
        if pnl >= target_pnl:
            return int(idx), "target"
    return int(len(path_pnl) - 1), "time_flat"


def _time_since_running_max(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    best = -np.inf
    best_idx = 0
    for idx, value in enumerate(values):
        if value >= best:
            best = float(value)
            best_idx = idx
        out[idx] = float(idx - best_idx)
    return out


def _path_velocity(values: np.ndarray, idx: int, lookback: int) -> float:
    lag = min(int(lookback), int(idx))
    if lag <= 0:
        return 0.0
    return _finite_float((values[idx] - values[idx - lag]) / float(lag), 0.0)


def _rolling_vol(values: np.ndarray, idx: int, lookback: int) -> float:
    start = max(0, int(idx) - int(lookback) + 1)
    window = values[start : int(idx) + 1]
    if len(window) < 2:
        return 0.0
    return _finite_float(float(np.std(window, ddof=0)), 0.0)


def processed_rows_to_candidates(
    *,
    split: str,
    session: str,
    rows: list[dict[str, Any]],
    starting_cash: float,
    max_candidates: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    option_index = {name: idx for idx, name in enumerate(OPTION_FEATURE_NAMES)}
    market_names = list(MARKET_FEATURE_NAMES)
    for row_idx, row in enumerate(rows):
        decision_dt = pd.Timestamp(row["decision_time"])
        if decision_dt.tzinfo is None:
            decision_dt = decision_dt.tz_localize("UTC")
        else:
            decision_dt = decision_dt.tz_convert("UTC")
        local = decision_dt.tz_convert("America/New_York")
        minutes = local.hour * 60.0 + local.minute
        if minutes > NO_NEW_ENTRIES_AFTER_MINUTE:
            continue
        market_last = np.asarray(row["market_window"], dtype=float)[-1]
        market = {f"market_{name}": float(market_last[idx]) if idx < len(market_last) else 0.0 for idx, name in enumerate(market_names)}
        offsets = list(row["strike_offsets"])
        rights = list(row["rights"])
        option_ladder = np.asarray(row["option_ladder"], dtype=float)
        mask = np.asarray(row["candidate_mask"], dtype=bool)
        contract_ids = np.asarray(row["contract_ids"], dtype=object)
        for strike_idx, offset in enumerate(offsets):
            for right_idx, right in enumerate(rights):
                if not bool(mask[strike_idx, right_idx]):
                    continue
                contract_id = str(contract_ids[strike_idx, right_idx])
                if not _is_spxw_five_point_contract(contract_id):
                    continue
                if str(right) not in {"C", "P"}:
                    continue
                features = option_ladder[strike_idx, right_idx]
                bid = _feature(features, option_index, "bid")
                ask = _feature(features, option_index, "ask")
                mid = _feature(features, option_index, "mid")
                spread = _feature(features, option_index, "spread")
                bid_size = _feature(features, option_index, "bid_size")
                ask_size = _feature(features, option_index, "ask_size")
                delta = _feature(features, option_index, "delta")
                gamma = _feature(features, option_index, "gamma")
                theta = _feature(features, option_index, "theta")
                iv = _feature(features, option_index, "iv")
                underlying = _feature(features, option_index, "underlying_price", fallback=market.get("market_spx_close", 0.0))
                if not _valid_entry_quote(bid, ask, mid, spread, bid_size, ask_size):
                    continue
                entry_premium = ask * CONTRACT_MULTIPLIER
                item = {
                    "split": split,
                    "session": session,
                    "decision_time": decision_dt.isoformat(),
                    "decision_dt": decision_dt,
                    "trade_uid": _uid("protocol164", split, session, decision_dt.isoformat(), contract_id),
                    "canonical_entry_uid": _uid(session, decision_dt.isoformat(), contract_id),
                    "candidate_uid": _uid("protocol164_candidate", split, session, decision_dt.isoformat(), contract_id, row_idx),
                    "contract_id": contract_id,
                    "root": "SPXW",
                    "settlement_style": "PM",
                    "right": str(right),
                    "offset": float(offset),
                    "entry_quote_time": decision_dt.isoformat(),
                    "entry_bid": bid,
                    "entry_ask": ask,
                    "entry_mid": mid,
                    "entry_spread": spread,
                    "entry_spread_frac": _safe_ratio(spread, mid, 0.0),
                    "entry_bid_size": bid_size,
                    "entry_ask_size": ask_size,
                    "entry_underlying_price": underlying,
                    "entry_iv": iv,
                    "entry_delta": delta,
                    "entry_gamma": gamma,
                    "entry_theta": theta,
                    "entry_minutes_since_open": _elapsed(minutes),
                    "entry_minutes_to_forced_flat": max(NO_NEW_ENTRIES_AFTER_MINUTE - minutes, 0.0),
                    "entry_progress": _elapsed(minutes) / SESSION_MINUTES,
                    "entry_progress_sin": math.sin(2.0 * math.pi * _elapsed(minutes) / SESSION_MINUTES),
                    "entry_progress_cos": math.cos(2.0 * math.pi * _elapsed(minutes) / SESSION_MINUTES),
                    "entry_is_first_30m": float(minutes < 10 * 60),
                    "entry_is_post_open_morning": float(10 * 60 <= minutes < 11 * 60 + 30),
                    "entry_is_midday": float(11 * 60 + 30 <= minutes < 13 * 60 + 30),
                    "entry_is_late_afternoon": float(minutes >= 13 * 60 + 30),
                    "time_bucket": time_bucket(decision_dt.to_pydatetime()),
                    "right_is_call": float(str(right) == "C"),
                    "right_is_put": float(str(right) == "P"),
                    "abs_offset": abs(float(offset)),
                    "entry_abs_delta": abs(delta),
                    "entry_abs_theta": abs(theta),
                    "entry_gamma_theta_ratio": _safe_ratio(gamma, abs(theta), 0.0),
                    "entry_theta_over_mid": _safe_ratio(abs(theta), abs(mid), 0.0),
                    "entry_theta_burden": _safe_ratio(abs(theta) * max(NO_NEW_ENTRIES_AFTER_MINUTE - minutes, 0.0), abs(mid), 0.0),
                    "entry_gamma_per_premium": _safe_ratio(gamma, ask, 0.0),
                    "entry_premium_over_underlying": _safe_ratio(ask, abs(underlying), 0.0),
                    "entry_spread_over_mid": _safe_ratio(spread, abs(mid), 0.0),
                    "entry_size_imbalance": _safe_ratio(bid_size - ask_size, bid_size + ask_size, 0.0),
                    "entry_call_delta_signed": delta if str(right) == "C" else 0.0,
                    "entry_put_delta_signed": delta if str(right) == "P" else 0.0,
                    "entry_premium": entry_premium,
                    "entry_premium_frac_10k": entry_premium / starting_cash,
                    "entry_affordable_10k": float(entry_premium > 0.0 and entry_premium <= starting_cash),
                    **market,
                }
                out.append(item)
                if max_candidates > 0 and len(out) >= max_candidates:
                    return out
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol164 Full Action-Space Dataset",
        "",
        "Builds flat-state candidate rows from the full processed SPXW 0DTE ladder without Protocol101's min-edge, time-bucket, or selected-candidate gates.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Rows: `{payload['rows']}`",
        f"- Skipped rows: `{payload['skipped_rows']}`",
        f"- Sessions processed: `{payload['sessions_processed']}`",
        f"- Candidate source: `{payload['candidate_source']}`",
        f"- Exit source: `{payload['exit_source']}`",
        f"- Rows by split: `{payload['rows_by_split']}`",
        f"- Path status skips: `{payload['path_status']}`",
        "",
        "## Outputs",
        "",
        f"- Dataset: `{path.parent / 'protocol164_full_action_space_dataset.parquet'}`",
        f"- Skipped candidates: `{path.parent / 'protocol164_skipped_candidates.csv'}`",
        f"- Summary: `{path.parent / 'summary.json'}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).sort_index().items()}


def _feature(values: np.ndarray, index: dict[str, int], name: str, fallback: float = 0.0) -> float:
    idx = index.get(name)
    if idx is None or idx >= len(values):
        return float(fallback)
    return _finite_float(values[idx], fallback)


def _elapsed(minutes: float) -> float:
    return float(min(max(minutes - SESSION_OPEN_MINUTE, 0.0), SESSION_MINUTES))


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)) or abs(float(denominator)) < 1e-9:
        return float(default)
    return float(numerator) / float(denominator)


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _is_spxw_five_point_contract(contract_id: str) -> bool:
    parts = str(contract_id).split("-")
    if len(parts) != 4 or parts[0] != "SPXW" or parts[3] not in {"C", "P"}:
        return False
    try:
        strike = float(parts[2])
    except ValueError:
        return False
    return math.isfinite(strike) and abs(strike % 5.0) < 1e-6


def _valid_entry_quote(bid: float, ask: float, mid: float, spread: float, bid_size: float, ask_size: float) -> bool:
    values = [bid, ask, mid, spread, bid_size, ask_size]
    if any(not math.isfinite(float(value)) for value in values):
        return False
    return bid > 0.0 and ask > 0.0 and mid > 0.0 and ask >= bid and spread >= 0.0


def _uid(*parts: object) -> str:
    return hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]


if __name__ == "__main__":
    raise SystemExit(main())
