"""Protocol200: causal lifecycle continuation policy for Protocol194 entries.

Protocol198/199 found that the current frozen lifecycle can churn same-side
ideas and often exits before material continuation value appears. Protocol200 is
the first training response to that finding.

Scope:
* entries stay frozen to the Protocol194 challenger stream
* the model only learns holding-state exit timing for this protocol
* no fixed profit target, stop percentage, or minimum hold rule is introduced
* evaluation is serial one-account replay, because changed exits change which
  later entries can be taken

The lifecycle model is a causal neural regressor over post-entry state. It sees
only data known at each quote minute: current executable PnL, MFE/MAE to date,
PnL velocity, spread/liquidity, Greeks, time left, and entry context. Labels use
future path information only as supervised training targets.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol198_lifecycle_churn_hold_counterfactual import money, pct


LOOP_ID = "v4_aplus_hypothesis_200_lifecycle_continuation_policy"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
DEFAULT_REPLAY_DIRS = [
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_191_protocol081_replay_of_protocol190_entries"),
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_193_protocol081_replay_of_protocol192_seed45_entries"),
]
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
NY = ZoneInfo("America/New_York")
STARTING_CASH = 10_000.0
CONTRACT_MULTIPLIER = 100.0
TARGET_SCALE = 500.0
TARGET_CLIP = 3_000.0
RISK_PENALTY = 0.35
FEATURE_COLUMNS = [
    "minutes_since_entry",
    "minutes_to_forced_flat",
    "day_progress",
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_frac",
    "bid_size",
    "ask_size",
    "quote_gap_seconds",
    "underlying_price",
    "directional_underlying_move",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "current_pnl",
    "mfe_to_now",
    "mae_to_now",
    "giveback_from_mfe",
    "giveback_fraction",
    "time_since_mfe_minutes",
    "pnl_velocity_1",
    "pnl_velocity_3",
    "pnl_velocity_5",
    "realized_pnl_vol_5",
    "realized_pnl_vol_10",
    "bid_over_entry_ask",
    "mid_over_entry_ask",
    "theta_over_mid",
    "gamma_theta_ratio",
    "time_theta_burden",
    "entry_score",
    "entry_offset",
    "entry_premium",
    "entry_premium_frac_start_cash",
    "entry_is_call",
    "entry_is_put",
]
THRESHOLD_CANDIDATES = (-300, -150, -50, 0, 50, 100, 200, 350, 500, 750, 1_000, 1_500, 2_000)


@dataclass
class PathRecord:
    uid: str
    fold: str
    reported_split: str
    entry_seed: int
    session: str
    decision_ts: pd.Timestamp
    contract_id: str
    right: str
    offset: float
    score: float
    entry_ask: float
    entry_premium: float
    baseline_exit_ts: pd.Timestamp
    baseline_pnl: float
    baseline_exit_reason: str
    features: np.ndarray
    target: np.ndarray
    path_pnl: np.ndarray
    quote_times: list[str]


class ContinuationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.04),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.04),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", action="append", type=Path, default=None)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-steps", type=int, default=650_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay_dirs = args.replay_dir or DEFAULT_REPLAY_DIRS
    candidates = load_candidate_entries(replay_dirs)
    records, path_skips = build_path_records(candidates, normalized_dir=args.normalized_dir, forced_flat_time=args.forced_flat_time)
    if not records:
        raise SystemExit("no executable path records were built")
    baseline_rows = []
    model_rows = []
    threshold_rows = []
    selected_rows = []
    fold_payloads = []
    for spec in fold_specs():
        train_records = [r for r in records if r.reported_split in spec["train_splits"]]
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        train_x_raw, train_y, scaler = fit_training_matrix(train_records, max_train_steps=args.max_train_steps, seed=101)
        validation_predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}
        test_predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}
        for model_seed in args.seeds:
            model, history = train_model(
                train_x_raw,
                train_y,
                scaler=scaler,
                seed=int(model_seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                learning_rate=float(args.learning_rate),
            )
            validation_predictions = predict_records(model, scaler, validation_records)
            validation_predictions_by_seed[int(model_seed)] = validation_predictions
            threshold, sweep = select_threshold(
                validation_records,
                validation_predictions,
                split_name=str(spec["validation_split"]),
                model_seed=int(model_seed),
            )
            threshold_rows.extend(
                {
                    **row,
                    "fold": spec["fold"],
                    "model_seed": int(model_seed),
                }
                for row in sweep
            )
            for split in spec["test_splits"]:
                split_records = [r for r in records if r.reported_split == split]
                predictions = predict_records(model, scaler, split_records)
                test_predictions_by_seed[int(model_seed)] = predictions
                model_trade_rows = simulate_serial(
                    split_records,
                    predictions,
                    threshold=threshold,
                    model_seed=int(model_seed),
                    strategy=f"protocol200:{spec['fold']}:seed{model_seed}",
                )
                model_rows.extend(model_trade_rows)
                selected_rows.extend(model_trade_rows)
                if split == "q1_2026":
                    march_records = [r for r in split_records if r.session >= "2026-03-01"]
                    march_predictions = {r.uid: predictions[r.uid] for r in march_records if r.uid in predictions}
                    model_rows.extend(
                        {
                            **row,
                            "reported_split": "march_2026",
                        }
                        for row in simulate_serial(
                            march_records,
                            march_predictions,
                            threshold=threshold,
                            model_seed=int(model_seed),
                            strategy=f"protocol200:{spec['fold']}:seed{model_seed}:march_subset",
                        )
                    )
            fold_payloads.append(
                {
                    "fold": spec["fold"],
                    "train_splits": list(spec["train_splits"]),
                    "validation_split": spec["validation_split"],
                    "test_splits": list(spec["test_splits"]),
                    "model_seed": int(model_seed),
                    "threshold": float(threshold),
                    "history": history,
                    "train_records": len(train_records),
                    "validation_records": len(validation_records),
                    "train_steps_used": int(len(train_y)),
                }
            )
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [r for r in records if r.reported_split == split]
            baseline_rows.extend(simulate_baseline_serial(split_records, strategy="protocol194_protocol081_baseline"))
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(march_records, strategy="protocol194_protocol081_baseline:march_subset")
                )

    baseline_frame = pd.DataFrame(baseline_rows).drop_duplicates(
        ["reported_split", "entry_seed", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    model_frame = pd.DataFrame(model_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    baseline_summary = summarize_replay(baseline_frame, seed_col="entry_seed")
    model_summary = summarize_replay(model_frame, seed_col="combo_seed")
    comparison = compare_summaries(model_summary, baseline_summary)
    invariants = serial_invariants(model_frame)
    payload = {
        "protocol": "200_lifecycle_continuation_policy",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": True,
        "source_replay_dirs": [str(path) for path in replay_dirs],
        "normalized_dir": str(args.normalized_dir),
        "row_counts": {
            "candidate_entries": int(len(candidates)),
            "path_records": int(len(records)),
            "path_skips": int(len(path_skips)),
            "baseline_trade_rows": int(len(baseline_frame)),
            "model_trade_rows": int(len(model_frame)),
        },
        "pre_registration": {
            "hypothesis": (
                "A causal post-entry continuation model can improve the frozen Protocol194 entry stream by "
                "learning when a position still has positive continuation utility, instead of repeatedly "
                "taking local exits and re-entering the same directional idea."
            ),
            "action_space": "holding state only for this protocol: hold or exit; flat entries stay frozen to Protocol194",
            "no_hardcoded_exit_rules": True,
            "serial_account_replay": True,
            "starting_cash": STARTING_CASH,
            "max_contracts": 1,
            "max_concurrent_positions": 1,
            "target": (
                "risk_adjusted_future_continuation = future_best_pnl_delta - "
                f"{RISK_PENALTY} * future_adverse_pnl_delta, clipped to +/-{TARGET_CLIP}"
            ),
            "threshold_selection": "threshold selected only on chronological validation split",
        },
        "folds": fold_payloads,
        "baseline_summary": baseline_summary,
        "model_summary": model_summary,
        "comparison": comparison,
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(path_skips, "skip_reason"),
        "decision": decide(comparison, invariants),
        "next_gate": next_gate(comparison),
    }
    baseline_frame.to_csv(args.out_dir / "protocol194_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "protocol200_model_serial_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(path_skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_candidate_entries(replay_dirs: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for replay_dir in replay_dirs:
        path = replay_dir / "protocol191_protocol081_candidate_paths.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_replay_dir"] = replay_dir.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["reported_split"].astype(str) != "march_2026"].copy()
    out["decision_ts"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out["candidate_exit_ts"] = pd.to_datetime(out["candidate_exit_time"], utc=True, errors="coerce")
    out = out[out["decision_ts"].notna() & out["candidate_exit_ts"].notna()].copy()
    out["session"] = out["session"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)
    out["right"] = out["right"].astype(str)
    out["fold"] = out["fold"].astype(str)
    out["reported_split"] = out["reported_split"].astype(str)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(0).astype(int)
    for column in ["entry_ask", "entry_premium", "candidate_pnl", "score", "offset", "entry_bid", "entry_mid"]:
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    out = out.drop_duplicates(["reported_split", "seed", "session", "decision_time", "contract_id"], keep="last")
    return out.sort_values(["reported_split", "seed", "session", "decision_ts", "contract_id"]).reset_index(drop=True)


def build_path_records(
    candidates: pd.DataFrame,
    *,
    normalized_dir: Path,
    forced_flat_time: str,
) -> tuple[list[PathRecord], list[dict[str, Any]]]:
    records: list[PathRecord] = []
    skips: list[dict[str, Any]] = []
    for session, group in candidates.groupby("session", sort=True):
        contracts = set(group["contract_id"].astype(str).unique())
        quotes = load_session_quotes(normalized_dir, str(session), contracts)
        if quotes.empty:
            skips.extend(base_skip(row, "missing_session_or_contract_quotes") for _, row in group.iterrows())
            continue
        by_contract = {
            str(contract_id): part.sort_values("quote_time").reset_index(drop=True)
            for contract_id, part in quotes.groupby("contract_id", sort=False)
        }
        forced_flat = forced_flat_timestamp(str(session), forced_flat_time)
        for _, row in group.iterrows():
            record, skip = path_record_for_candidate(row, by_contract.get(str(row["contract_id"])), forced_flat)
            if skip:
                skips.append(skip)
            else:
                records.append(record)
    return records, skips


def load_session_quotes(normalized_dir: Path, session: str, contract_ids: set[str]) -> pd.DataFrame:
    path = find_normalized_path(normalized_dir, session)
    if path is None or not contract_ids:
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
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        available = pd.read_parquet(path)
        for column in columns:
            if column not in available.columns:
                available[column] = np.nan
        frame = available[columns].copy()
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
        & (frame["bid"] >= 0.0)
        & (frame["ask"] > 0.0)
        & (frame["ask"] >= frame["bid"])
    ].copy()


def find_normalized_path(normalized_dir: Path, session: str) -> Path | None:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    if preferred:
        return preferred[0]
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return fallback[0] if fallback else None


def forced_flat_timestamp(session: str, forced_flat_time: str) -> pd.Timestamp:
    hour, minute = [int(part) for part in forced_flat_time.split(":", 1)]
    return pd.Timestamp(session).replace(hour=hour, minute=minute, tzinfo=NY).tz_convert("UTC")


def path_record_for_candidate(
    row: pd.Series,
    quotes: pd.DataFrame | None,
    forced_flat: pd.Timestamp,
) -> tuple[PathRecord | None, dict[str, Any] | None]:
    if quotes is None or quotes.empty:
        return None, base_skip(row, "missing_contract_quotes")
    decision_ts = pd.Timestamp(row["decision_ts"])
    entry_ask = finite_float(row.get("entry_ask"), math.nan)
    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        return None, base_skip(row, "invalid_entry_ask")
    path = quotes[(quotes["quote_time"] >= decision_ts) & (quotes["quote_time"] <= forced_flat)].copy()
    if path.empty:
        return None, base_skip(row, "missing_post_entry_path")
    features, target, path_pnl = build_features_and_target(path, row, entry_ask, decision_ts, forced_flat)
    if len(path_pnl) == 0:
        return None, base_skip(row, "invalid_path_pnl")
    uid = f"{row['reported_split']}|seed{int(row['seed'])}|{row['session']}|{pd.Timestamp(row['decision_ts']).isoformat()}|{row['contract_id']}"
    return (
        PathRecord(
            uid=uid,
            fold=str(row["fold"]),
            reported_split=str(row["reported_split"]),
            entry_seed=int(row["seed"]),
            session=str(row["session"]),
            decision_ts=decision_ts,
            contract_id=str(row["contract_id"]),
            right=str(row["right"]),
            offset=finite_float(row.get("offset"), math.nan),
            score=finite_float(row.get("score"), math.nan),
            entry_ask=entry_ask,
            entry_premium=finite_float(row.get("entry_premium"), entry_ask * CONTRACT_MULTIPLIER),
            baseline_exit_ts=pd.Timestamp(row["candidate_exit_ts"]),
            baseline_pnl=finite_float(row.get("candidate_pnl"), 0.0),
            baseline_exit_reason=str(row.get("candidate_exit_reason", "")),
            features=features,
            target=target,
            path_pnl=path_pnl,
            quote_times=[pd.Timestamp(value).isoformat() for value in path["quote_time"].tolist()],
        ),
        None,
    )


def build_features_and_target(
    path: pd.DataFrame,
    entry: pd.Series,
    entry_ask: float,
    decision_ts: pd.Timestamp,
    forced_flat: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = path.sort_values("quote_time").reset_index(drop=True)
    bid = pd.to_numeric(path["bid"], errors="coerce").to_numpy(dtype=np.float32)
    ask = pd.to_numeric(path["ask"], errors="coerce").to_numpy(dtype=np.float32)
    mid = pd.to_numeric(path["mid"], errors="coerce").to_numpy(dtype=np.float32)
    mid = np.where(np.isfinite(mid), mid, (bid + ask) / 2.0)
    path_pnl = (bid.astype(np.float64) - float(entry_ask)) * CONTRACT_MULTIPLIER
    path_pnl = np.nan_to_num(path_pnl, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    quote_times = pd.to_datetime(path["quote_time"], utc=True)
    minutes_since_entry = np.array([(ts - decision_ts).total_seconds() / 60.0 for ts in quote_times], dtype=np.float32)
    minutes_since_entry = np.maximum(minutes_since_entry, 0.0)
    minutes_to_forced_flat = np.array([(forced_flat - ts).total_seconds() / 60.0 for ts in quote_times], dtype=np.float32)
    minutes_to_forced_flat = np.maximum(minutes_to_forced_flat, 0.0)
    local_minutes = np.array([ts.tz_convert(NY).hour * 60 + ts.tz_convert(NY).minute for ts in quote_times], dtype=np.float32)
    day_progress = (local_minutes - (9 * 60 + 30)) / (6.5 * 60)
    spread = ask - bid
    spread_frac = safe_divide(spread, mid)
    underlying = pd.to_numeric(path["underlying_price"], errors="coerce").to_numpy(dtype=np.float32)
    entry_underlying = first_finite(underlying)
    directional_underlying = (underlying - entry_underlying) * (-1.0 if str(entry.get("right")) == "P" else 1.0)
    bid_size = pd.to_numeric(path["bid_size"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ask_size = pd.to_numeric(path["ask_size"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    quote_gap = pd.to_numeric(path["quote_gap_seconds"], errors="coerce").to_numpy(dtype=np.float32)
    iv = pd.to_numeric(path["iv"], errors="coerce").to_numpy(dtype=np.float32)
    delta = pd.to_numeric(path["delta"], errors="coerce").to_numpy(dtype=np.float32)
    gamma = pd.to_numeric(path["gamma"], errors="coerce").to_numpy(dtype=np.float32)
    theta = pd.to_numeric(path["theta"], errors="coerce").to_numpy(dtype=np.float32)
    vega = pd.to_numeric(path["vega"], errors="coerce").to_numpy(dtype=np.float32)
    mfe = np.maximum.accumulate(path_pnl)
    mae = np.minimum.accumulate(path_pnl)
    mfe_idx = running_argmax(path_pnl)
    step_idx = np.arange(len(path_pnl), dtype=np.float32)
    time_since_mfe = step_idx - mfe_idx.astype(np.float32)
    giveback = np.maximum(0.0, mfe - path_pnl)
    giveback_fraction = np.where(mfe > 0.0, giveback / np.maximum(mfe, 1e-6), 0.0)
    vel1 = velocity(path_pnl, 1)
    vel3 = velocity(path_pnl, 3)
    vel5 = velocity(path_pnl, 5)
    vol5 = rolling_vol(path_pnl, 5)
    vol10 = rolling_vol(path_pnl, 10)
    bid_over_entry = bid / max(float(entry_ask), 1e-6)
    mid_over_entry = mid / max(float(entry_ask), 1e-6)
    theta_over_mid = safe_divide(np.abs(theta), np.abs(mid))
    gamma_theta = safe_divide(np.abs(gamma), np.abs(theta))
    time_theta_burden = theta_over_mid * minutes_to_forced_flat
    entry_score = np.full(len(path_pnl), finite_float(entry.get("score"), 0.0), dtype=np.float32)
    entry_offset = np.full(len(path_pnl), finite_float(entry.get("offset"), 0.0), dtype=np.float32)
    entry_premium = np.full(
        len(path_pnl),
        finite_float(entry.get("entry_premium"), float(entry_ask) * CONTRACT_MULTIPLIER),
        dtype=np.float32,
    )
    features = np.column_stack(
        [
            minutes_since_entry,
            minutes_to_forced_flat,
            day_progress,
            bid,
            ask,
            mid,
            spread,
            spread_frac,
            bid_size,
            ask_size,
            quote_gap,
            underlying,
            directional_underlying,
            iv,
            delta,
            gamma,
            theta,
            vega,
            path_pnl,
            mfe,
            mae,
            giveback,
            giveback_fraction,
            time_since_mfe,
            vel1,
            vel3,
            vel5,
            vol5,
            vol10,
            bid_over_entry,
            mid_over_entry,
            theta_over_mid,
            gamma_theta,
            time_theta_burden,
            entry_score,
            entry_offset,
            entry_premium,
            entry_premium / STARTING_CASH,
            np.full(len(path_pnl), float(str(entry.get("right")) == "C"), dtype=np.float32),
            np.full(len(path_pnl), float(str(entry.get("right")) == "P"), dtype=np.float32),
        ]
    ).astype(np.float32)
    future_best = future_running_max(path_pnl)
    future_worst = future_running_min(path_pnl)
    continuation_upside = future_best - path_pnl
    adverse_excursion = np.maximum(0.0, path_pnl - future_worst)
    target = continuation_upside - RISK_PENALTY * adverse_excursion
    target = np.clip(target, -TARGET_CLIP, TARGET_CLIP).astype(np.float32)
    return features, target, path_pnl


def running_argmax(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.int64)
    best_idx = 0
    best = -np.inf
    for idx, value in enumerate(values):
        if float(value) > best:
            best = float(value)
            best_idx = idx
        out[idx] = best_idx
    return out


def future_running_max(values: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(values[::-1])[::-1]


def future_running_min(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(values[::-1])[::-1]


def velocity(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    if len(values) <= lookback:
        return out
    out[lookback:] = (values[lookback:] - values[:-lookback]) / float(lookback)
    return out


def rolling_vol(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        sample = values[start : idx + 1]
        out[idx] = float(np.std(sample)) if len(sample) >= 2 else 0.0
    return out


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator, dtype=np.float32), dtype=np.float32),
        where=np.isfinite(denominator) & (np.abs(denominator) > 1e-8),
    ).astype(np.float32)


def first_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[0]) if len(finite) else 0.0


def fit_training_matrix(
    records: Sequence[PathRecord],
    *,
    max_train_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, FeatureScaler]:
    x = np.vstack([record.features for record in records]).astype(np.float32)
    y = np.concatenate([record.target for record in records]).astype(np.float32) / TARGET_SCALE
    if max_train_steps > 0 and len(y) > max_train_steps:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(y), size=max_train_steps, replace=False))
        x = x[idx]
        y = y[idx]
    scaler = FeatureScaler.fit(x)
    return scaler.transform(x), y, scaler


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    scaler: FeatureScaler,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
) -> tuple[ContinuationMLP, list[dict[str, float]]]:
    del scaler
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContinuationMLP(input_dim=train_x.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=min(batch_size, len(train_y)),
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.huber_loss(pred, yb, delta=1.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": float(epoch), "train_loss": epoch_loss})
    model.load_state_dict(best_state)
    return model, history


def predict_records(model: ContinuationMLP, scaler: FeatureScaler, records: Sequence[PathRecord]) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for record in records:
            x = scaler.transform(record.features)
            pred = model(torch.from_numpy(x)).cpu().numpy().astype(np.float32) * TARGET_SCALE
            out[record.uid] = pred
    return out


def select_threshold(
    records: Sequence[PathRecord],
    predictions: dict[str, np.ndarray],
    *,
    split_name: str,
    model_seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_threshold = float(THRESHOLD_CANDIDATES[0])
    best_key = (-1e18, -1e18, 0.0)
    for threshold in THRESHOLD_CANDIDATES:
        trades = simulate_serial(records, predictions, threshold=float(threshold), model_seed=model_seed, strategy="threshold_selection")
        metrics = metrics_for_rows(pd.DataFrame(trades))
        key = (float(metrics["total_pnl"]), float(metrics["profit_factor_for_selection"]), -float(metrics["trades"]))
        rows.append({"validation_split": split_name, "threshold": float(threshold), **metrics})
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, rows


def simulate_serial(
    records: Sequence[PathRecord],
    predictions: dict[str, np.ndarray],
    *,
    threshold: float,
    model_seed: int,
    strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity_by_seed: dict[int, float] = {}
    open_until_by_seed_session: dict[tuple[int, str], pd.Timestamp] = {}
    ordered = sorted(records, key=lambda r: (r.entry_seed, r.session, r.decision_ts, r.contract_id))
    for record in ordered:
        equity = equity_by_seed.get(record.entry_seed, STARTING_CASH)
        session_key = (record.entry_seed, record.session)
        if record.decision_ts < open_until_by_seed_session.get(session_key, pd.Timestamp.min.tz_localize("UTC")):
            continue
        if record.entry_premium <= 0.0 or record.entry_premium > equity:
            continue
        pred = predictions.get(record.uid)
        if pred is None or len(pred) != len(record.path_pnl):
            continue
        exit_idx = exit_index_from_prediction(pred, threshold)
        pnl = float(record.path_pnl[exit_idx])
        exit_ts = pd.Timestamp(record.quote_times[exit_idx])
        combo_seed = int(model_seed * 100 + record.entry_seed)
        rows.append(
            {
                "reported_split": record.reported_split,
                "fold": record.fold,
                "model_seed": int(model_seed),
                "entry_seed": int(record.entry_seed),
                "combo_seed": combo_seed,
                "session": record.session,
                "decision_time": record.decision_ts.isoformat(),
                "exit_time": exit_ts.isoformat(),
                "contract_id": record.contract_id,
                "right": record.right,
                "offset": float(record.offset),
                "score": float(record.score),
                "threshold": float(threshold),
                "entry_ask": float(record.entry_ask),
                "entry_premium": float(record.entry_premium),
                "pnl": pnl,
                "account_equity_before": float(equity),
                "account_equity_after": float(equity + pnl),
                "exit_step": int(exit_idx),
                "path_points": int(len(record.path_pnl)),
                "exit_reason": "model_exit" if exit_idx < len(record.path_pnl) - 1 else "mandatory_forced_flat",
                "predicted_continuation_value": float(pred[exit_idx]),
                "baseline_exit_time": record.baseline_exit_ts.isoformat(),
                "baseline_pnl": float(record.baseline_pnl),
                "baseline_exit_reason": record.baseline_exit_reason,
                "strategy": strategy,
            }
        )
        equity_by_seed[record.entry_seed] = float(equity + pnl)
        open_until_by_seed_session[session_key] = exit_ts
    return rows


def simulate_baseline_serial(records: Sequence[PathRecord], *, strategy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity_by_seed: dict[int, float] = {}
    open_until_by_seed_session: dict[tuple[int, str], pd.Timestamp] = {}
    ordered = sorted(records, key=lambda r: (r.entry_seed, r.session, r.decision_ts, r.contract_id))
    for record in ordered:
        equity = equity_by_seed.get(record.entry_seed, STARTING_CASH)
        session_key = (record.entry_seed, record.session)
        if record.decision_ts < open_until_by_seed_session.get(session_key, pd.Timestamp.min.tz_localize("UTC")):
            continue
        if record.entry_premium <= 0.0 or record.entry_premium > equity:
            continue
        pnl = float(record.baseline_pnl)
        rows.append(
            {
                "reported_split": record.reported_split,
                "fold": record.fold,
                "entry_seed": int(record.entry_seed),
                "session": record.session,
                "decision_time": record.decision_ts.isoformat(),
                "exit_time": record.baseline_exit_ts.isoformat(),
                "contract_id": record.contract_id,
                "right": record.right,
                "offset": float(record.offset),
                "score": float(record.score),
                "entry_ask": float(record.entry_ask),
                "entry_premium": float(record.entry_premium),
                "pnl": pnl,
                "account_equity_before": float(equity),
                "account_equity_after": float(equity + pnl),
                "exit_reason": record.baseline_exit_reason,
                "strategy": strategy,
            }
        )
        equity_by_seed[record.entry_seed] = float(equity + pnl)
        open_until_by_seed_session[session_key] = record.baseline_exit_ts
    return rows


def exit_index_from_prediction(prediction: np.ndarray, threshold: float) -> int:
    eligible = np.where(np.asarray(prediction, dtype=float) <= float(threshold))[0]
    return int(eligible[0]) if len(eligible) else int(len(prediction) - 1)


def fold_specs() -> list[dict[str, Any]]:
    return [
        {
            "fold": "train_q3_validate_q4_test_q1",
            "train_splits": ("q3_2025",),
            "validation_split": "q4_2025",
            "test_splits": ("q1_2026",),
        },
        {
            "fold": "train_q3_q4_validate_q1_test_recent",
            "train_splits": ("q3_2025", "q4_2025"),
            "validation_split": "q1_2026",
            "test_splits": ("recent_2026",),
        },
    ]


def summarize_replay(frame: pd.DataFrame, *, seed_col: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for split, split_group in frame.groupby("reported_split", sort=True):
        seed_metrics = []
        for seed, group in split_group.groupby(seed_col, sort=True):
            metrics = metrics_for_rows(group)
            metrics["seed"] = int(seed)
            seed_metrics.append(metrics)
        rows.append(
            {
                "reported_split": str(split),
                "seeds": int(len(seed_metrics)),
                "median_total_pnl": median(seed_metrics, "total_pnl"),
                "median_profit_factor": median(seed_metrics, "profit_factor"),
                "median_trades": median(seed_metrics, "trades"),
                "positive_seed_fraction": float(np.mean([row["total_pnl"] > 0.0 for row in seed_metrics])) if seed_metrics else 0.0,
                "seed_rows": seed_metrics,
            }
        )
    return rows


def metrics_for_rows(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "profit_factor_for_selection": 0.0,
        }
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_loss = -float(losses.sum())
    gross_win = float(wins.sum())
    if gross_loss > 0.0:
        pf = gross_win / gross_loss
    else:
        pf = float("inf") if gross_win > 0.0 else 0.0
    return {
        "trades": int(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float((pnl > 0.0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(pf),
        "profit_factor_for_selection": 999.0 if math.isinf(pf) else float(pf),
    }


def compare_summaries(model_summary: list[dict[str, Any]], baseline_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {row["reported_split"]: row for row in baseline_summary}
    rows = []
    for model in model_summary:
        split = model["reported_split"]
        base = baseline.get(split, {})
        base_pnl = finite_float(base.get("median_total_pnl"), 0.0)
        rows.append(
            {
                "reported_split": split,
                "model_median_total_pnl": finite_float(model.get("median_total_pnl"), 0.0),
                "baseline_median_total_pnl": base_pnl,
                "delta_vs_baseline": finite_float(model.get("median_total_pnl"), 0.0) - base_pnl,
                "model_median_profit_factor": finite_float(model.get("median_profit_factor"), 0.0),
                "baseline_median_profit_factor": finite_float(base.get("median_profit_factor"), 0.0),
                "model_median_trades": finite_float(model.get("median_trades"), 0.0),
                "baseline_median_trades": finite_float(base.get("median_trades"), 0.0),
                "model_positive_seed_fraction": finite_float(model.get("positive_seed_fraction"), 0.0),
                "baseline_positive_seed_fraction": finite_float(base.get("positive_seed_fraction"), 0.0),
                "beats_baseline": bool(finite_float(model.get("median_total_pnl"), 0.0) > base_pnl),
            }
        )
    return rows


def threshold_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for (fold, model_seed), group in frame.groupby(["fold", "model_seed"], sort=True):
        best = group.sort_values(["total_pnl", "profit_factor_for_selection"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "fold": str(fold),
                "model_seed": int(model_seed),
                "selected_threshold": float(best["threshold"]),
                "validation_total_pnl": float(best["total_pnl"]),
                "validation_profit_factor": float(best["profit_factor"]),
                "validation_trades": int(best["trades"]),
            }
        )
    return rows


def serial_invariants(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"overlap_violations": 0, "unaffordable_violations": 0, "nan_time_rows": 0}
    check = frame.copy()
    check["decision_dt"] = pd.to_datetime(check["decision_time"], utc=True, errors="coerce")
    check["exit_dt"] = pd.to_datetime(check["exit_time"], utc=True, errors="coerce")
    overlaps = 0
    for _, group in check.groupby(["reported_split", "combo_seed", "session"], sort=False):
        previous_exit = None
        for row in group.sort_values("decision_dt").itertuples(index=False):
            if previous_exit is not None and row.decision_dt < previous_exit:
                overlaps += 1
            previous_exit = row.exit_dt
    premium = pd.to_numeric(check["entry_premium"], errors="coerce")
    equity = pd.to_numeric(check["account_equity_before"], errors="coerce")
    return {
        "overlap_violations": int(overlaps),
        "unaffordable_violations": int(((premium > equity) | premium.isna() | equity.isna()).sum()),
        "nan_time_rows": int(check["decision_dt"].isna().sum() + check["exit_dt"].isna().sum()),
    }


def decide(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_protocol200_invariant_failure"
    by_split = {row["reported_split"]: row for row in comparison}
    required = ["q1_2026", "march_2026", "recent_2026"]
    if all(by_split.get(split, {}).get("beats_baseline") for split in required):
        return "keep_research_candidate_protocol200_beats_lifecycle_baseline_on_tests"
    if any(by_split.get(split, {}).get("beats_baseline") for split in required):
        return "mixed_protocol200_lifecycle_signal_requires_attribution"
    return "reject_protocol200_no_test_improvement"


def next_gate(comparison: list[dict[str, Any]]) -> str:
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in ["q1_2026", "march_2026", "recent_2026"]):
        return "Run five-seed confirmation and churn attribution against Protocol200 before considering a live-shadow challenger."
    return "Attribute which splits/trades degraded before adding another lifecycle objective; do not promote to paper."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol200 Lifecycle Continuation Policy",
        "",
        "No paid data was downloaded. No broker endpoint was called. No live or paper orders were placed.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Next gate: {payload['next_gate']}",
        "",
        "## Comparison",
        "",
        "| split | model PnL | baseline PnL | delta | model PF | baseline PF | model trades | baseline trades | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
            f"{pct(row['model_positive_seed_fraction'])} |"
        )
    lines.extend(["", "## Thresholds", ""])
    if payload["threshold_summary"]:
        lines.extend(["| fold | seed | threshold | validation PnL | PF | trades |", "|---|---:|---:|---:|---:|---:|"])
        for row in payload["threshold_summary"]:
            lines.append(
                f"| {row['fold']} | {row['model_seed']} | {row['selected_threshold']:.0f} | "
                f"{money(row['validation_total_pnl'])} | {row['validation_profit_factor']:.3f} | {row['validation_trades']} |"
            )
    lines.extend(
        [
            "",
            "## Invariants",
            "",
            f"- Overlap violations: `{payload['invariants']['overlap_violations']}`",
            f"- Unaffordable violations: `{payload['invariants']['unaffordable_violations']}`",
            f"- NaN time rows: `{payload['invariants']['nan_time_rows']}`",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Model trades: `{path.parent / 'protocol200_model_serial_trades.csv'}`",
            f"- Baseline trades: `{path.parent / 'protocol194_baseline_serial_trades.csv'}`",
            f"- Threshold sweep: `{path.parent / 'threshold_sweep.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def median(rows: list[dict[str, Any]], key: str) -> float:
    values = [finite_float(row.get(key), math.nan) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.median(values)) if values else 0.0


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def count_by(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, ""))
        out[key] = out.get(key, 0) + 1
    return out


def base_skip(row: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "reported_split": str(row.get("reported_split", "")),
        "seed": int(row.get("seed", 0)),
        "session": str(row.get("session", "")),
        "decision_time": str(row.get("decision_time", "")),
        "contract_id": str(row.get("contract_id", "")),
        "right": str(row.get("right", "")),
        "skip_reason": reason,
    }


if __name__ == "__main__":
    raise SystemExit(main())
