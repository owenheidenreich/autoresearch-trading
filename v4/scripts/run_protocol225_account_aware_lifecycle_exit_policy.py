"""EXP_2026_05_22_ACCOUNT_AWARE_LIFECYCLE_EXIT_POLICY_V1.

Historically Protocol225. This experiment keeps the Protocol221 return-on-
premium entries and Protocol223 account-aware sizing formula, then trains a
causal neural hold/exit model over the post-entry quote path.

This is the first step after the scale-in/out audit: learn better hold/exit
behavior before allowing add-one-contract actions. The model sees only state
that would be available while holding the trade: executable bid/ask path,
current PnL, MFE/MAE, giveback, PnL velocity, spread/liquidity, time left,
account exposure, and entry Greeks/contract economics.

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
from v4.scripts.run_protocol223_account_aware_confidence_sizing import drawdown_multiplier


ROLE_LABEL = "EXP_2026_05_22_ACCOUNT_AWARE_LIFECYCLE_EXIT_POLICY_V1"
HISTORICAL_ID = "Protocol225"
DEFAULT_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_223_account_aware_confidence_sizing/account_aware_sized_trades.csv"
)
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_225_account_aware_lifecycle_exit_policy")
NY = ZoneInfo("America/New_York")
STARTING_CASH = 10_000.0
CONTRACT_MULTIPLIER = 100.0
TARGET_SCALE = 1_000.0
TARGET_CLIP = 10_000.0
RISK_PENALTY = 0.35
THRESHOLD_CANDIDATES = (-2_000, -1_000, -500, -250, 0, 250, 500, 1_000, 2_000, 4_000, 8_000)
REQUIRED_SPLITS = ("q1_2026", "march_2026", "recent_2026")
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
    "current_unit_pnl",
    "current_position_pnl",
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
    "entry_score",
    "entry_margin",
    "entry_confidence",
    "entry_offset",
    "entry_premium",
    "entry_premium_frac_equity",
    "quantity",
    "premium_frac_realized",
    "account_equity_before",
    "entry_abs_delta",
    "entry_gamma_theta_ratio",
    "entry_theta_burden",
    "entry_gamma_per_premium",
    "entry_spread_over_mid",
    "entry_is_call",
    "entry_is_put",
]


@dataclass
class PathRecord:
    uid: str
    fold: str
    reported_split: str
    session: str
    decision_ts: pd.Timestamp
    baseline_exit_ts: pd.Timestamp
    contract_id: str
    right: str
    offset: float
    score: float
    threshold: float
    entry_ask: float
    entry_premium: float
    entry_ask_size: float
    entry_abs_delta: float
    entry_gamma_theta_ratio: float
    entry_theta_burden: float
    entry_gamma_per_premium: float
    entry_spread_over_mid: float
    baseline_unit_pnl: float
    features: np.ndarray
    target: np.ndarray
    unit_pnl_path: np.ndarray
    quote_times: list[str]


class ContinuationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--forced-flat-time", default="15:30")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-steps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-risk-frac", type=float, default=0.01)
    parser.add_argument("--max-risk-frac", type=float, default=0.10)
    parser.add_argument("--hard-premium-cap-frac", type=float, default=0.12)
    parser.add_argument("--confidence-scale", type=float, default=0.75)
    parser.add_argument("--liquidity-fraction", type=float, default=0.25)
    parser.add_argument("--absolute-max-contracts", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.trades, dataset_path=args.dataset)
    records, skips = build_path_records(entries, normalized_dir=args.normalized_dir, forced_flat_time=str(args.forced_flat_time))
    if not records:
        raise SystemExit("no path records")
    sizing = {
        "min_risk_frac": float(args.min_risk_frac),
        "max_risk_frac": float(args.max_risk_frac),
        "hard_premium_cap_frac": float(args.hard_premium_cap_frac),
        "confidence_scale": float(args.confidence_scale),
        "liquidity_fraction": float(args.liquidity_fraction),
        "absolute_max_contracts": int(args.absolute_max_contracts),
    }
    baseline_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    fold_payloads: list[dict[str, Any]] = []
    for spec in fold_specs():
        train_records = [r for r in records if r.reported_split in spec["train_splits"]]
        validation_records = [r for r in records if r.reported_split == spec["validation_split"]]
        if not train_records or not validation_records:
            continue
        train_x, train_y, scaler = fit_training_matrix(train_records, max_train_steps=int(args.max_train_steps), seed=225)
        for seed in args.seeds:
            model, history = train_model(
                train_x,
                train_y,
                seed=int(seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                learning_rate=float(args.learning_rate),
            )
            validation_predictions = predict_records(model, scaler, validation_records)
            threshold, sweep = select_threshold(
                validation_records,
                validation_predictions,
                sizing=sizing,
                split_name=str(spec["validation_split"]),
                model_seed=int(seed),
            )
            threshold_rows.extend({**row, "fold": spec["fold"], "model_seed": int(seed)} for row in sweep)
            for split in spec["test_splits"]:
                split_records = [r for r in records if r.reported_split == split]
                predictions = predict_records(model, scaler, split_records)
                model_rows.extend(
                    simulate_serial(
                        split_records,
                        predictions,
                        sizing=sizing,
                        threshold=threshold,
                        model_seed=int(seed),
                        strategy=f"protocol225:{spec['fold']}:seed{seed}",
                    )
                )
                if split == "q1_2026":
                    march_records = [r for r in split_records if r.session >= "2026-03-01"]
                    march_predictions = {r.uid: predictions[r.uid] for r in march_records if r.uid in predictions}
                    model_rows.extend(
                        {**row, "reported_split": "march_2026"}
                        for row in simulate_serial(
                            march_records,
                            march_predictions,
                            sizing=sizing,
                            threshold=threshold,
                            model_seed=int(seed),
                            strategy=f"protocol225:{spec['fold']}:seed{seed}:march_subset",
                        )
                    )
            fold_payloads.append(
                {
                    "fold": spec["fold"],
                    "train_splits": list(spec["train_splits"]),
                    "validation_split": spec["validation_split"],
                    "test_splits": list(spec["test_splits"]),
                    "model_seed": int(seed),
                    "threshold": float(threshold),
                    "history": history,
                    "train_records": int(len(train_records)),
                    "validation_records": int(len(validation_records)),
                    "train_steps_used": int(len(train_y)),
                }
            )
        for split in [spec["validation_split"], *spec["test_splits"]]:
            split_records = [r for r in records if r.reported_split == split]
            baseline_rows.extend(simulate_baseline_serial(split_records, sizing=sizing, strategy="protocol223_account_aware_baseline"))
            if split == "q1_2026":
                march_records = [r for r in split_records if r.session >= "2026-03-01"]
                baseline_rows.extend(
                    {**row, "reported_split": "march_2026"}
                    for row in simulate_baseline_serial(
                        march_records,
                        sizing=sizing,
                        strategy="protocol223_account_aware_baseline:march_subset",
                    )
                )
    baseline_frame = pd.DataFrame(baseline_rows).drop_duplicates(
        ["reported_split", "session", "decision_time", "contract_id", "strategy"],
        keep="last",
    )
    model_frame = pd.DataFrame(model_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    baseline_summary = summarize_replay(baseline_frame, seed_col=None)
    model_summary = summarize_replay(model_frame, seed_col="model_seed")
    comparison = compare_summaries(model_summary, baseline_summary)
    invariants = serial_invariants(model_frame)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / causal account-aware lifecycle exit model",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_ACCOUNT_AWARE_LIFECYCLE_EXIT_V1",
        "entry_source": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1",
        "sizing_source": "EXP_2026_05_22_ACCOUNT_AWARE_CONFIDENCE_SIZING_V2",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.trades),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "pre_registration": {
            "hypothesis": (
                "A causal hold/exit model can capture more continuation value from the account-aware sized "
                "return-on-premium stream without enabling scale-in averaging-down."
            ),
            "flat_entries": "frozen Protocol221 trade stream",
            "sizing": sizing,
            "holding_action_space": "hold or exit only; no add/reduce action in this protocol",
            "target": (
                "future_best_position_pnl_delta - "
                f"{RISK_PENALTY} * future_adverse_position_pnl_delta, clipped to +/-{TARGET_CLIP}"
            ),
        },
        "row_counts": {
            "entries": int(len(entries)),
            "path_records": int(len(records)),
            "path_skips": int(len(skips)),
            "baseline_trade_rows": int(len(baseline_frame)),
            "model_trade_rows": int(len(model_frame)),
        },
        "folds": fold_payloads,
        "baseline_summary": baseline_summary,
        "model_summary": model_summary,
        "comparison": comparison,
        "threshold_summary": threshold_summary(threshold_frame),
        "invariants": invariants,
        "path_skip_counts": count_by(skips, "skip_reason"),
        "decision": decide(comparison, invariants),
        "next_experiment": next_experiment(comparison, invariants),
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "model_trades": str(args.out_dir / "protocol225_model_serial_trades.csv"),
            "baseline_trades": str(args.out_dir / "protocol223_baseline_serial_trades.csv"),
            "threshold_sweep": str(args.out_dir / "threshold_sweep.csv"),
            "path_skips": str(args.out_dir / "path_skips.csv"),
        },
    }
    baseline_frame.to_csv(args.out_dir / "protocol223_baseline_serial_trades.csv", index=False)
    model_frame.to_csv(args.out_dir / "protocol225_model_serial_trades.csv", index=False)
    threshold_frame.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    pd.DataFrame(skips).to_csv(args.out_dir / "path_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_entries(path: Path, *, dataset_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["starting_cash"].astype(float).eq(STARTING_CASH)
        & frame["extra_slippage_per_side"].astype(float).eq(0.0)
        & (pd.to_numeric(frame["aa_contracts"], errors="coerce") > 0)
        & ~frame["reported_split"].astype(str).eq("march_2026")
    ].copy()
    if dataset_path.exists():
        columns = [
            "candidate_uid",
            "entry_abs_delta",
            "entry_gamma_theta_ratio",
            "entry_theta_burden",
            "entry_gamma_per_premium",
            "entry_spread_over_mid",
        ]
        features = pd.read_parquet(dataset_path, columns=columns).drop_duplicates("candidate_uid")
        frame = frame.merge(features, on="candidate_uid", how="left", validate="many_to_one")
    for column in ["decision_time", "exit_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    numeric = [
        "score",
        "threshold",
        "entry_ask",
        "entry_premium",
        "entry_ask_size",
        "pnl",
        "offset",
        "entry_abs_delta",
        "entry_gamma_theta_ratio",
        "entry_theta_burden",
        "entry_gamma_per_premium",
        "entry_spread_over_mid",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    required = ["reported_split", "session", "decision_time", "exit_time", "contract_id", "right", "entry_ask", "entry_premium"]
    return frame.dropna(subset=required).sort_values(["reported_split", "session", "decision_time", "contract_id"]).reset_index(drop=True)


def build_path_records(
    entries: pd.DataFrame,
    *,
    normalized_dir: Path,
    forced_flat_time: str,
) -> tuple[list[PathRecord], list[dict[str, Any]]]:
    records: list[PathRecord] = []
    skips: list[dict[str, Any]] = []
    for session, group in entries.groupby("session", sort=True):
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
            record, skip = path_record_for_entry(row, by_contract.get(str(row["contract_id"])), forced_flat)
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


def path_record_for_entry(
    row: pd.Series,
    quotes: pd.DataFrame | None,
    forced_flat: pd.Timestamp,
) -> tuple[PathRecord | None, dict[str, Any] | None]:
    if quotes is None or quotes.empty:
        return None, base_skip(row, "missing_contract_quotes")
    decision_ts = pd.Timestamp(row["decision_time"])
    entry_ask = finite_float(row.get("entry_ask"), math.nan)
    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        return None, base_skip(row, "invalid_entry_ask")
    path = quotes[(quotes["quote_time"] >= decision_ts) & (quotes["quote_time"] <= forced_flat)].copy()
    if path.empty:
        return None, base_skip(row, "missing_post_entry_path")
    features, target, unit_pnl_path = build_features_and_target(path, row, entry_ask, decision_ts, forced_flat)
    if len(unit_pnl_path) == 0:
        return None, base_skip(row, "invalid_path_pnl")
    uid = f"{row['reported_split']}|{row['session']}|{decision_ts.isoformat()}|{row['contract_id']}"
    return (
        PathRecord(
            uid=uid,
            fold=str(row.get("fold", "")),
            reported_split=str(row["reported_split"]),
            session=str(row["session"]),
            decision_ts=decision_ts,
            baseline_exit_ts=pd.Timestamp(row["exit_time"]),
            contract_id=str(row["contract_id"]),
            right=str(row["right"]),
            offset=finite_float(row.get("offset"), math.nan),
            score=finite_float(row.get("score"), 0.0),
            threshold=finite_float(row.get("threshold"), 0.0),
            entry_ask=entry_ask,
            entry_premium=finite_float(row.get("entry_premium"), entry_ask * CONTRACT_MULTIPLIER),
            entry_ask_size=finite_float(row.get("entry_ask_size"), 1.0),
            entry_abs_delta=finite_float(row.get("entry_abs_delta"), 0.0),
            entry_gamma_theta_ratio=finite_float(row.get("entry_gamma_theta_ratio"), 0.0),
            entry_theta_burden=finite_float(row.get("entry_theta_burden"), 0.0),
            entry_gamma_per_premium=finite_float(row.get("entry_gamma_per_premium"), 0.0),
            entry_spread_over_mid=finite_float(row.get("entry_spread_over_mid"), 0.0),
            baseline_unit_pnl=finite_float(row.get("pnl"), 0.0),
            features=features,
            target=target,
            unit_pnl_path=unit_pnl_path,
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
    unit_pnl = ((bid.astype(np.float64) - float(entry_ask)) * CONTRACT_MULTIPLIER).astype(np.float32)
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
    quote_gap = pd.to_numeric(path["quote_gap_seconds"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    quantity = max(1.0, finite_float(entry.get("aa_contracts"), 1.0))
    position_pnl = unit_pnl * quantity
    mfe = np.maximum.accumulate(position_pnl)
    mae = np.minimum.accumulate(position_pnl)
    mfe_idx = running_argmax(position_pnl)
    step_idx = np.arange(len(position_pnl), dtype=np.float32)
    time_since_mfe = step_idx - mfe_idx.astype(np.float32)
    giveback = np.maximum(0.0, mfe - position_pnl)
    giveback_fraction = np.where(mfe > 0.0, giveback / np.maximum(mfe, 1e-6), 0.0)
    vel1 = velocity(position_pnl, 1)
    vel3 = velocity(position_pnl, 3)
    vel5 = velocity(position_pnl, 5)
    vol5 = rolling_vol(position_pnl, 5)
    vol10 = rolling_vol(position_pnl, 10)
    bid_over_entry = bid / max(float(entry_ask), 1e-6)
    mid_over_entry = mid / max(float(entry_ask), 1e-6)
    entry_score = finite_float(entry.get("score"), 0.0)
    entry_threshold = finite_float(entry.get("threshold"), 0.0)
    entry_margin = max(entry_score - entry_threshold, 0.0)
    entry_confidence = finite_float(entry.get("aa_confidence"), 0.0)
    entry_premium = finite_float(entry.get("entry_premium"), float(entry_ask) * CONTRACT_MULTIPLIER)
    account_equity = finite_float(entry.get("aa_equity_before"), STARTING_CASH)
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
            unit_pnl,
            position_pnl,
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
            np.full(len(unit_pnl), entry_score, dtype=np.float32),
            np.full(len(unit_pnl), entry_margin, dtype=np.float32),
            np.full(len(unit_pnl), entry_confidence, dtype=np.float32),
            np.full(len(unit_pnl), finite_float(entry.get("offset"), 0.0), dtype=np.float32),
            np.full(len(unit_pnl), entry_premium, dtype=np.float32),
            np.full(len(unit_pnl), entry_premium / max(account_equity, 1e-6), dtype=np.float32),
            np.full(len(unit_pnl), quantity, dtype=np.float32),
            np.full(len(unit_pnl), finite_float(entry.get("aa_premium_frac_realized"), 0.0), dtype=np.float32),
            np.full(len(unit_pnl), account_equity, dtype=np.float32),
            np.full(len(unit_pnl), finite_float(entry.get("entry_abs_delta"), 0.0), dtype=np.float32),
            np.full(len(unit_pnl), finite_float(entry.get("entry_gamma_theta_ratio"), 0.0), dtype=np.float32),
            np.full(len(unit_pnl), finite_float(entry.get("entry_theta_burden"), 0.0), dtype=np.float32),
            np.full(len(unit_pnl), finite_float(entry.get("entry_gamma_per_premium"), 0.0), dtype=np.float32),
            np.full(len(unit_pnl), finite_float(entry.get("entry_spread_over_mid"), 0.0), dtype=np.float32),
            np.full(len(unit_pnl), float(str(entry.get("right")) == "C"), dtype=np.float32),
            np.full(len(unit_pnl), float(str(entry.get("right")) == "P"), dtype=np.float32),
        ]
    ).astype(np.float32)
    future_best = future_running_max(position_pnl)
    future_worst = future_running_min(position_pnl)
    continuation_upside = future_best - position_pnl
    adverse_excursion = np.maximum(0.0, position_pnl - future_worst)
    target = continuation_upside - RISK_PENALTY * adverse_excursion
    target = np.clip(target, -TARGET_CLIP, TARGET_CLIP).astype(np.float32)
    return features, target, unit_pnl


def fit_training_matrix(records: Sequence[PathRecord], *, max_train_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray, FeatureScaler]:
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
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
) -> tuple[ContinuationMLP, list[dict[str, float]]]:
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
    sizing: dict[str, Any],
    split_name: str,
    model_seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_threshold = float(THRESHOLD_CANDIDATES[0])
    best_key = (-1e18, -1e18, 0.0)
    for threshold in THRESHOLD_CANDIDATES:
        trades = simulate_serial(
            records,
            predictions,
            sizing=sizing,
            threshold=float(threshold),
            model_seed=model_seed,
            strategy="threshold_selection",
        )
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
    sizing: dict[str, Any],
    threshold: float,
    model_seed: int,
    strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity = STARTING_CASH
    peak = equity
    open_until_by_session: dict[str, pd.Timestamp] = {}
    for record in sorted(records, key=lambda r: (r.session, r.decision_ts, r.contract_id)):
        if record.decision_ts < open_until_by_session.get(record.session, pd.Timestamp.min.tz_localize("UTC")):
            continue
        quantity, sizing_info = compute_quantity(record, equity=equity, peak=peak, sizing=sizing)
        if quantity <= 0:
            continue
        pred = predictions.get(record.uid)
        if pred is None or len(pred) != len(record.unit_pnl_path):
            continue
        exit_idx = exit_index_from_prediction(pred, threshold)
        unit_pnl = float(record.unit_pnl_path[exit_idx])
        pnl = unit_pnl * quantity
        exit_ts = pd.Timestamp(record.quote_times[exit_idx])
        before = equity
        equity += pnl
        peak = max(peak, equity)
        rows.append(row_for_trade(record, quantity, before, equity, pnl, unit_pnl, exit_idx, exit_ts, strategy, model_seed, sizing_info, pred[exit_idx], threshold))
        open_until_by_session[record.session] = exit_ts
    return rows


def simulate_baseline_serial(records: Sequence[PathRecord], *, sizing: dict[str, Any], strategy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity = STARTING_CASH
    peak = equity
    open_until_by_session: dict[str, pd.Timestamp] = {}
    for record in sorted(records, key=lambda r: (r.session, r.decision_ts, r.contract_id)):
        if record.decision_ts < open_until_by_session.get(record.session, pd.Timestamp.min.tz_localize("UTC")):
            continue
        quantity, sizing_info = compute_quantity(record, equity=equity, peak=peak, sizing=sizing)
        if quantity <= 0:
            continue
        unit_pnl = float(record.baseline_unit_pnl)
        pnl = unit_pnl * quantity
        before = equity
        equity += pnl
        peak = max(peak, equity)
        rows.append(
            row_for_trade(
                record,
                quantity,
                before,
                equity,
                pnl,
                unit_pnl,
                exit_idx=-1,
                exit_ts=record.baseline_exit_ts,
                strategy=strategy,
                model_seed=0,
                sizing_info=sizing_info,
                predicted_value=math.nan,
                threshold=math.nan,
            )
        )
        open_until_by_session[record.session] = record.baseline_exit_ts
    return rows


def compute_quantity(record: PathRecord, *, equity: float, peak: float, sizing: dict[str, Any]) -> tuple[int, dict[str, float]]:
    margin = max(float(record.score) - float(record.threshold), 0.0)
    confidence_scale = max(float(sizing["confidence_scale"]), 1e-9)
    confidence = margin / (margin + confidence_scale) if margin > 0.0 else 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    risk_frac_raw = float(sizing["min_risk_frac"]) + (
        float(sizing["max_risk_frac"]) - float(sizing["min_risk_frac"])
    ) * confidence * confidence
    drawdown_frac = max(0.0, (peak - equity) / peak) if peak > 0.0 else 1.0
    throttle = drawdown_multiplier(drawdown_frac)
    risk_frac = min(float(sizing["max_risk_frac"]), max(0.0, risk_frac_raw * throttle))
    hard_budget = max(equity, 0.0) * float(sizing["hard_premium_cap_frac"])
    risk_budget = min(max(equity, 0.0) * risk_frac, hard_budget)
    ask_size = max(1, int(math.floor(finite_float(record.entry_ask_size, 1.0))))
    liquidity_cap = max(1, int(math.floor(ask_size * float(sizing["liquidity_fraction"]))))
    max_contracts = max(1, min(int(sizing["absolute_max_contracts"]), liquidity_cap))
    raw_contracts = int(math.floor(risk_budget / record.entry_premium)) if record.entry_premium > 0.0 else 0
    if record.entry_premium > hard_budget + 1e-9:
        quantity = 0
    else:
        quantity = min(max_contracts, raw_contracts)
        if quantity < 1 and record.entry_premium <= equity:
            quantity = 1
        if quantity * record.entry_premium > hard_budget + 1e-9:
            quantity = int(math.floor(hard_budget / record.entry_premium))
        if quantity * record.entry_premium > equity + 1e-9:
            quantity = int(math.floor(equity / record.entry_premium))
    return int(max(quantity, 0)), {
        "confidence": confidence,
        "risk_frac": risk_frac,
        "drawdown_frac": drawdown_frac,
        "hard_budget": hard_budget,
        "risk_budget": risk_budget,
        "liquidity_cap": float(liquidity_cap),
        "raw_contracts": float(raw_contracts),
    }


def row_for_trade(
    record: PathRecord,
    quantity: int,
    before: float,
    after: float,
    pnl: float,
    unit_pnl: float,
    exit_idx: int,
    exit_ts: pd.Timestamp,
    strategy: str,
    model_seed: int,
    sizing_info: dict[str, float],
    predicted_value: float,
    threshold: float,
) -> dict[str, Any]:
    return {
        "reported_split": record.reported_split,
        "fold": record.fold,
        "model_seed": int(model_seed),
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
        "quantity": int(quantity),
        "premium_at_risk": float(quantity * record.entry_premium),
        "premium_frac": float((quantity * record.entry_premium) / before) if before > 0.0 else 0.0,
        "unit_pnl": float(unit_pnl),
        "pnl": float(pnl),
        "account_equity_before": float(before),
        "account_equity_after": float(after),
        "exit_step": int(exit_idx),
        "path_points": int(len(record.unit_pnl_path)),
        "exit_reason": "model_exit" if exit_idx >= 0 and exit_idx < len(record.unit_pnl_path) - 1 else "mandatory_or_baseline_exit",
        "predicted_continuation_value": finite_float(predicted_value, math.nan),
        "baseline_exit_time": record.baseline_exit_ts.isoformat(),
        "baseline_unit_pnl": float(record.baseline_unit_pnl),
        "baseline_position_pnl_at_entry_size": float(record.baseline_unit_pnl * quantity),
        "sizing_confidence": float(sizing_info.get("confidence", math.nan)),
        "sizing_risk_frac": float(sizing_info.get("risk_frac", math.nan)),
        "sizing_drawdown_frac": float(sizing_info.get("drawdown_frac", math.nan)),
        "strategy": strategy,
    }


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


def summarize_replay(frame: pd.DataFrame, *, seed_col: str | None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for split, split_group in frame.groupby("reported_split", sort=True):
        seed_metrics = []
        if seed_col is None:
            metrics = metrics_for_rows(split_group)
            metrics["seed"] = 0
            seed_metrics.append(metrics)
        else:
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
                "median_max_drawdown": median(seed_metrics, "max_drawdown"),
                "median_worst_day": median(seed_metrics, "worst_day"),
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
            "max_drawdown": 0.0,
            "worst_day": 0.0,
        }
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_loss = -float(losses.sum())
    gross_win = float(wins.sum())
    pf = gross_win / gross_loss if gross_loss > 0.0 else (float("inf") if gross_win > 0.0 else 0.0)
    equity = STARTING_CASH + pnl.cumsum()
    dd = equity - equity.cummax()
    session_pnl = frame.assign(_pnl=pnl).groupby("session")["_pnl"].sum()
    return {
        "trades": int(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float((pnl > 0.0).mean()) if len(pnl) else 0.0,
        "profit_factor": float(pf),
        "profit_factor_for_selection": 999.0 if math.isinf(pf) else float(pf),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "worst_day": float(session_pnl.min()) if len(session_pnl) else 0.0,
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
                "model_median_max_drawdown": finite_float(model.get("median_max_drawdown"), 0.0),
                "baseline_median_max_drawdown": finite_float(base.get("median_max_drawdown"), 0.0),
                "model_positive_seed_fraction": finite_float(model.get("positive_seed_fraction"), 0.0),
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
    for _, group in check.groupby(["reported_split", "model_seed", "session"], sort=False):
        previous_exit = None
        for row in group.sort_values("decision_dt").itertuples(index=False):
            if previous_exit is not None and row.decision_dt < previous_exit:
                overlaps += 1
            previous_exit = row.exit_dt
    premium = pd.to_numeric(check["premium_at_risk"], errors="coerce")
    equity = pd.to_numeric(check["account_equity_before"], errors="coerce")
    return {
        "overlap_violations": int(overlaps),
        "unaffordable_violations": int(((premium > equity) | premium.isna() | equity.isna()).sum()),
        "nan_time_rows": int(check["decision_dt"].isna().sum() + check["exit_dt"].isna().sum()),
    }


def decide(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "reject_account_aware_lifecycle_invariant_failure"
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "keep_account_aware_lifecycle_exit_research_candidate"
    if any(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "mixed_account_aware_lifecycle_exit_requires_attribution"
    return "reject_account_aware_lifecycle_exit_no_improvement"


def next_experiment(comparison: list[dict[str, Any]], invariants: dict[str, Any]) -> str:
    if any(int(invariants.get(key, 1)) != 0 for key in ["overlap_violations", "unaffordable_violations", "nan_time_rows"]):
        return "Fix simulator invariants before further lifecycle work."
    by_split = {row["reported_split"]: row for row in comparison}
    if all(by_split.get(split, {}).get("beats_baseline") for split in REQUIRED_SPLITS):
        return "Run stress and attribution, then test reduce/scale-out actions on top of this lifecycle exit model."
    return "Attribute degraded splits. If overholding is the failure, train a slot-aware opportunity-cost target for this account-aware stream."


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Entry source: {payload['entry_source']}",
        f"Sizing source: {payload['sizing_source']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Comparison To Account-Aware Baseline",
        "",
        "| split | model PnL | baseline PnL | delta | model PF | baseline PF | model trades | baseline trades | model DD | baseline DD | positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison"]:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | {money(row['baseline_median_total_pnl'])} | "
            f"{money(row['delta_vs_baseline'])} | {row['model_median_profit_factor']:.3f} | {row['baseline_median_profit_factor']:.3f} | "
            f"{row['model_median_trades']:.0f} | {row['baseline_median_trades']:.0f} | "
            f"{money(row['model_median_max_drawdown'])} | {money(row['baseline_median_max_drawdown'])} | "
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
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Model trades: `{payload['outputs']['model_trades']}`",
            f"- Baseline trades: `{payload['outputs']['baseline_trades']}`",
            f"- Threshold sweep: `{payload['outputs']['threshold_sweep']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


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
        "session": str(row.get("session", "")),
        "decision_time": str(row.get("decision_time", "")),
        "contract_id": str(row.get("contract_id", "")),
        "right": str(row.get("right", "")),
        "skip_reason": reason,
    }


def money(value: Any) -> str:
    number = finite_float(value, 0.0)
    sign = "-" if number < 0.0 else ""
    return f"{sign}${abs(number):,.0f}"


def pct(value: Any) -> str:
    return f"{finite_float(value, 0.0) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
