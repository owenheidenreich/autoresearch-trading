"""Protocol276 integrated entry/lifecycle failure attribution.

This is a foundation-hardening audit, not a new model experiment. It reads the
saved Protocol276 replay artifacts, reconstructs the actual lifecycle paths for
the entry-policy-selected trades, compares those paths to the Protocol274
oracle-entry lifecycle-label distribution, and explains whether failure is
coming from entry selection, lifecycle timing, skip mechanics, or label/policy
alignment.

No paid data is downloaded. No broker endpoint is called. No model is trained.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.scripts.run_protocol274_position_state_action_advantage_dataset import (
    DEFAULT_ACTION_DATASET,
    DEFAULT_NORMALIZED_DIR,
    build_flat_value_maps,
    forced_flat_timestamp,
    labels_for_trade,
    load_session_quotes,
)
from v4.scripts.run_protocol276_integrated_entry_lifecycle_serial_replay import (
    DEFAULT_LIFECYCLE_ARTIFACTS,
    DEFAULT_OUT_DIR as DEFAULT_PROTOCOL276_DIR,
    load_lifecycle_bundle,
)


ROLE_LABEL = "AUDIT_PROTOCOL276_INTEGRATED_LIFECYCLE_FAILURE_ATTRIBUTION_V1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol276_integrated_lifecycle_failure_attribution")
DEFAULT_PROTOCOL274_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet"
)
DEFAULT_PROTOCOL271_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_271_unified_action_advantage_policy/unified_action_advantage_model_trades.csv"
)
CONTRACT_MULTIPLIER = 100.0
PATH_COLUMNS = [
    "split",
    "session",
    "candidate_uid",
    "trade_uid",
    "reported_split",
    "fold",
    "state_time",
    "state_index",
    "oracle_holding_action",
    "model_lifecycle_score",
    "model_lifecycle_action",
    "current_pnl",
    "q_exit",
    "q_hold",
    "a_hold",
    "entry_a_enter",
    "mfe_to_now",
    "mae_to_now",
    "giveback_from_mfe",
    "minutes_since_entry",
    "minutes_to_forced_flat",
]
ACTION_ROW_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol276-dir", type=Path, default=DEFAULT_PROTOCOL276_DIR)
    parser.add_argument("--protocol271-trades", type=Path, default=DEFAULT_PROTOCOL271_TRADES)
    parser.add_argument("--action-dataset", type=Path, default=DEFAULT_ACTION_DATASET)
    parser.add_argument("--protocol274-dataset", type=Path, default=DEFAULT_PROTOCOL274_DATASET)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--lifecycle-artifacts", type=Path, default=DEFAULT_LIFECYCLE_ARTIFACTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trades = load_csv(args.protocol276_dir / "integrated_entry_lifecycle_trades.csv")
    skips = load_csv(args.protocol276_dir / "integrated_entry_lifecycle_skips.csv")
    entry_only = load_csv(args.protocol271_trades)
    protocol276_summary = load_json(args.protocol276_dir / "summary.json")
    action_rows = load_action_rows(args.action_dataset, trades["candidate_uid"].astype(str).unique())
    flat_values = load_flat_values(args.action_dataset)

    attribution_rows, state_rows, reconstruction_skips = reconstruct_trade_paths(
        trades,
        action_rows,
        flat_values,
        normalized_dir=args.normalized_dir,
        lifecycle_artifacts=args.lifecycle_artifacts,
        seed=int(args.seed),
        forced_flat_time=str(args.forced_flat_time),
    )
    trade_attribution = pd.DataFrame(attribution_rows)
    state_paths = pd.DataFrame(state_rows)
    reconstruction_skip_frame = pd.DataFrame(reconstruction_skips)

    state_shift = distribution_shift(state_paths, args.protocol274_dataset)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "foundation audit / Protocol276 failure attribution",
        "source_protocol": "CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1 / Protocol276",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decide(trade_attribution, reconstruction_skip_frame),
        "protocol276_decision": protocol276_summary.get("decision"),
        "reconstructed_trades": int(len(trade_attribution)),
        "reconstructed_state_rows": int(len(state_paths)),
        "reconstruction_skips": int(len(reconstruction_skip_frame)),
        "entry_attribution": entry_attribution(trade_attribution),
        "lifecycle_attribution": lifecycle_attribution(trade_attribution, state_paths),
        "skip_attribution": skip_attribution(skips, reconstruction_skip_frame),
        "entry_only_bridge": entry_only_bridge(trades, entry_only),
        "state_distribution_shift": records(state_shift),
        "root_cause_summary": root_cause_summary(trade_attribution, state_paths, skips),
        "next_direction": (
            "Do not retrain yet. First close the attribution gaps: entry policy is taking many negative-advantage "
            "entries, the lifecycle model holds too often on entry-policy paths, and the actual state distribution "
            "differs from the oracle-entry Protocol274 lifecycle-label distribution."
        ),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "trade_attribution": str(args.out_dir / "trade_attribution.csv"),
            "state_paths": str(args.out_dir / "reconstructed_state_paths.csv"),
            "state_distribution_shift": str(args.out_dir / "state_distribution_shift.csv"),
            "root_cause_by_split": str(args.out_dir / "root_cause_by_split.csv"),
        },
    }

    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    trade_attribution.to_csv(args.out_dir / "trade_attribution.csv", index=False)
    state_paths.reindex(columns=PATH_COLUMNS).to_csv(args.out_dir / "reconstructed_state_paths.csv", index=False)
    state_shift.to_csv(args.out_dir / "state_distribution_shift.csv", index=False)
    summarize_by(trade_attribution, ["reported_split", "root_cause"], "pnl").to_csv(args.out_dir / "root_cause_by_split.csv", index=False)
    summarize_by(trade_attribution, ["reported_split", "lifecycle_timing_mode"], "pnl").to_csv(
        args.out_dir / "lifecycle_timing_by_split.csv", index=False
    )
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_action_rows(path: Path, candidate_uids: Iterable[str]) -> pd.DataFrame:
    ids = [str(value) for value in candidate_uids]
    try:
        frame = pd.read_parquet(path, columns=ACTION_ROW_COLUMNS, filters=[("candidate_uid", "in", ids)])
    except Exception:
        frame = pd.read_parquet(path, columns=ACTION_ROW_COLUMNS)
        frame = frame[frame["candidate_uid"].astype(str).isin(ids)].copy()
    return normalize_action_rows(frame)


def load_flat_values(path: Path) -> dict[tuple[str, str], pd.DataFrame]:
    frame = pd.read_parquet(path, columns=["split", "session", "decision_dt", "session_oracle_value"])
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["session_oracle_value"] = pd.to_numeric(frame["session_oracle_value"], errors="coerce")
    return build_flat_value_maps(frame.dropna(subset=["decision_dt"]))


def normalize_action_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["decision_dt"] = pd.to_datetime(out["decision_dt"], utc=True, errors="coerce")
    if "candidate_exit_dt" in out.columns:
        out["candidate_exit_dt"] = pd.to_datetime(out["candidate_exit_dt"], utc=True, errors="coerce")
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
        "candidate_pnl",
        "q_wait",
        "q_enter",
        "a_enter",
        "session_oracle_value",
        "best_enter_value_at_decision",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["split", "session", "decision_dt", "contract_id", "entry_ask"]).reset_index(drop=True)


def reconstruct_trade_paths(
    trades: pd.DataFrame,
    action_rows: pd.DataFrame,
    flat_values: dict[tuple[str, str], pd.DataFrame],
    *,
    normalized_dir: Path,
    lifecycle_artifacts: Path,
    seed: int,
    forced_flat_time: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if trades.empty:
        return [], [], []
    action_by_candidate = {str(row["candidate_uid"]): row for _, row in action_rows.iterrows()}
    bundles: dict[str, Any] = {}
    quote_cache: dict[tuple[str, str], pd.DataFrame] = {}
    attribution_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        candidate_uid = str(trade.get("candidate_uid", ""))
        action = action_by_candidate.get(candidate_uid)
        if action is None:
            skips.append(base_reconstruction_skip(trade, "missing_action_dataset_row"))
            continue
        fold = str(trade.get("fold", ""))
        if fold not in bundles:
            bundles[fold] = load_lifecycle_bundle(lifecycle_artifacts, fold, seed)
        contract_id = str(action.get("contract_id", ""))
        session = str(action.get("session", ""))
        cache_key = (session, contract_id)
        if cache_key not in quote_cache:
            quote_cache[cache_key] = load_session_quotes(normalized_dir, session, {contract_id})
        forced_flat = forced_flat_timestamp(session, forced_flat_time)
        rows, skip = labels_for_trade(action, quote_cache[cache_key], flat_values.get((str(action["split"]), session)), forced_flat)
        if skip:
            skip["reported_split"] = str(trade.get("reported_split", ""))
            skips.append(skip)
            continue
        path = pd.DataFrame(rows).sort_values("state_index").reset_index(drop=True)
        scored = score_lifecycle_path(path, bundles[fold])
        scored["reported_split"] = str(trade.get("reported_split", ""))
        scored["fold"] = fold
        state_rows.extend(scored[PATH_COLUMNS].to_dict("records"))
        attribution_rows.append(trade_attribution_row(trade, scored))
    return attribution_rows, state_rows, skips


def score_lifecycle_path(path: pd.DataFrame, lifecycle: Any) -> pd.DataFrame:
    frame = path.copy()
    for column in lifecycle.feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
    x = lifecycle.scaler.transform(frame[lifecycle.feature_columns].to_numpy(dtype=np.float32))
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            logits, pred_adv = lifecycle.model(torch.from_numpy(x[start : start + 4096]))
            score = (logits[:, 1] - logits[:, 0]).cpu().numpy() + pred_adv.cpu().numpy() * 0.25
            scores.extend(float(value) for value in score)
    frame["model_lifecycle_score"] = scores
    frame["model_lifecycle_action"] = np.where(frame["model_lifecycle_score"].to_numpy(dtype=float) > lifecycle.threshold, "hold", "exit")
    return frame


def trade_attribution_row(trade: pd.Series, path: pd.DataFrame) -> dict[str, Any]:
    actual_exit_index = int(finite(trade.get("exit_state_index"), -1))
    actual_state = nearest_state(path, actual_exit_index)
    oracle_exit = first_matching(path, path["oracle_holding_action"].astype(str).eq("exit"))
    model_exit = first_matching(path, path["model_lifecycle_action"].astype(str).eq("exit"))
    best_idx = int(pd.to_numeric(path["current_pnl"], errors="coerce").idxmax()) if not path.empty else -1
    best_state = path.loc[best_idx] if best_idx >= 0 else pd.Series(dtype=object)
    model_exit_index = int(model_exit.get("state_index", len(path) - 1)) if model_exit is not None else int(path["state_index"].max())
    first_oracle_exit_index = int(oracle_exit.get("state_index", len(path) - 1)) if oracle_exit is not None else None
    actual_pnl = finite(trade.get("pnl"), 0.0)
    best_pnl = finite(best_state.get("current_pnl"), actual_pnl)
    oracle_exit_pnl = finite(oracle_exit.get("current_pnl"), math.nan) if oracle_exit is not None else math.nan
    false_hold_rows = int((path["oracle_holding_action"].astype(str).eq("exit") & path["model_lifecycle_action"].astype(str).eq("hold")).sum())
    false_exit_rows = int((path["oracle_holding_action"].astype(str).eq("hold") & path["model_lifecycle_action"].astype(str).eq("exit")).sum())
    after_oracle_false_hold = 0
    if first_oracle_exit_index is not None:
        after_oracle = path[pd.to_numeric(path["state_index"], errors="coerce") >= first_oracle_exit_index]
        after_oracle_false_hold = int(
            (after_oracle["oracle_holding_action"].astype(str).eq("exit") & after_oracle["model_lifecycle_action"].astype(str).eq("hold")).sum()
        )
    lifecycle_mode = lifecycle_timing_mode(
        actual_pnl=actual_pnl,
        actual_exit_index=actual_exit_index,
        model_exit_index=model_exit_index,
        first_oracle_exit_index=first_oracle_exit_index,
        best_pnl=best_pnl,
        exit_reason=str(trade.get("exit_reason", "")),
        giveback=finite(trade.get("giveback_from_mfe"), 0.0),
    )
    root = root_cause(
        a_enter=finite(trade.get("a_enter")),
        pnl=actual_pnl,
        lifecycle_mode=lifecycle_mode,
        false_hold_rows=false_hold_rows,
        false_exit_rows=false_exit_rows,
    )
    return {
        "candidate_uid": str(trade.get("candidate_uid", "")),
        "trade_uid": str(trade.get("trade_uid", "")),
        "split": str(trade.get("split", "")),
        "reported_split": str(trade.get("reported_split", "")),
        "session": str(trade.get("session", "")),
        "decision_time": str(trade.get("decision_time", "")),
        "exit_time": str(trade.get("exit_time", "")),
        "fold": str(trade.get("fold", "")),
        "right": str(trade.get("right", "")),
        "offset": finite(trade.get("offset")),
        "entry_premium": finite(trade.get("entry_premium")),
        "pnl": actual_pnl,
        "a_enter": finite(trade.get("a_enter")),
        "q_wait": finite(trade.get("q_wait")),
        "q_enter": finite(trade.get("q_enter")),
        "entry_model_margin": finite(trade.get("entry_model_margin")),
        "entry_model_threshold": finite(trade.get("entry_model_threshold")),
        "exit_reason": str(trade.get("exit_reason", "")),
        "actual_exit_index": actual_exit_index,
        "model_exit_index": model_exit_index,
        "first_oracle_exit_index": first_oracle_exit_index,
        "best_path_index": int(best_state.get("state_index", best_idx)),
        "duration_minutes": finite(trade.get("duration_minutes")),
        "path_rows": int(len(path)),
        "actual_exit_current_pnl": finite(actual_state.get("current_pnl"), actual_pnl),
        "first_oracle_exit_current_pnl": oracle_exit_pnl,
        "best_path_pnl": best_pnl,
        "pnl_vs_first_oracle_exit": none_if_nan(actual_pnl - oracle_exit_pnl),
        "pnl_vs_best_path": float(actual_pnl - best_pnl),
        "mfe_to_exit": finite(trade.get("mfe_to_exit")),
        "mae_to_exit": finite(trade.get("mae_to_exit")),
        "giveback_from_mfe": finite(trade.get("giveback_from_mfe")),
        "false_hold_rows": false_hold_rows,
        "false_exit_rows": false_exit_rows,
        "false_hold_rows_after_first_oracle_exit": after_oracle_false_hold,
        "oracle_hold_fraction_on_path": float(path["oracle_holding_action"].astype(str).eq("hold").mean()),
        "model_hold_fraction_on_path": float(path["model_lifecycle_action"].astype(str).eq("hold").mean()),
        "lifecycle_timing_mode": lifecycle_mode,
        "root_cause": root,
    }


def nearest_state(path: pd.DataFrame, state_index: int) -> pd.Series:
    if path.empty:
        return pd.Series(dtype=object)
    exact = path[pd.to_numeric(path["state_index"], errors="coerce").eq(state_index)]
    if not exact.empty:
        return exact.iloc[0]
    distances = (pd.to_numeric(path["state_index"], errors="coerce") - state_index).abs()
    return path.loc[int(distances.idxmin())]


def first_matching(path: pd.DataFrame, mask: pd.Series) -> pd.Series | None:
    candidates = path[mask]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def lifecycle_timing_mode(
    *,
    actual_pnl: float,
    actual_exit_index: int,
    model_exit_index: int,
    first_oracle_exit_index: int | None,
    best_pnl: float,
    exit_reason: str,
    giveback: float,
) -> str:
    if first_oracle_exit_index is not None and exit_reason == "forced_flat_no_lifecycle_exit_signal":
        return "never_exited_after_oracle_exit"
    if first_oracle_exit_index is not None and model_exit_index - first_oracle_exit_index > 30:
        return "overheld_after_oracle_exit"
    if first_oracle_exit_index is not None and first_oracle_exit_index - actual_exit_index > 5:
        return "early_exit_before_oracle_hold_finished"
    if actual_pnl < 0.0 and giveback > 500.0:
        return "gave_back_mfe_then_lost"
    if best_pnl - actual_pnl > 1000.0:
        return "missed_large_path_profit"
    if actual_pnl < 0.0:
        return "losing_trade_timing_not_primary"
    return "profitable_or_neutral_timing"


def root_cause(*, a_enter: float, pnl: float, lifecycle_mode: str, false_hold_rows: int, false_exit_rows: int) -> str:
    if math.isfinite(a_enter) and a_enter < 0.0 and pnl < 0.0:
        return "entry_policy_took_negative_advantage_loser"
    if lifecycle_mode in {"never_exited_after_oracle_exit", "overheld_after_oracle_exit", "gave_back_mfe_then_lost"}:
        return "lifecycle_overholding_or_late_exit"
    if lifecycle_mode == "early_exit_before_oracle_hold_finished":
        return "lifecycle_early_exit"
    if false_hold_rows > false_exit_rows * 2 and pnl < 0.0:
        return "lifecycle_false_hold_bias"
    if math.isfinite(a_enter) and a_enter >= 0.0 and pnl < 0.0:
        return "positive_entry_label_but_losing_path"
    if pnl >= 0.0:
        return "profitable_trade"
    return "unattributed_loss"


def distribution_shift(state_paths: pd.DataFrame, protocol274_dataset: Path) -> pd.DataFrame:
    actual = summarize_state_distribution(state_paths, "protocol276_entry_policy_paths", split_column="reported_split")
    if not protocol274_dataset.exists():
        return actual
    columns = [
        "split",
        "state_time",
        "candidate_uid",
        "oracle_holding_action",
        "a_hold",
        "current_pnl",
        "mfe_to_now",
        "giveback_from_mfe",
        "minutes_since_entry",
        "entry_a_enter",
    ]
    baseline = pd.read_parquet(protocol274_dataset, columns=columns)
    baseline_rows = [summarize_state_distribution(baseline, "protocol274_oracle_entry_paths", split_column="split")]
    q1 = baseline[baseline["split"].astype(str).eq("q1_2026")].copy()
    if not q1.empty:
        months = pd.to_datetime(q1["state_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
        march = q1[months.eq("2026-03")].copy()
        if not march.empty:
            march["split"] = "march_2026"
            baseline_rows.append(summarize_state_distribution(march, "protocol274_oracle_entry_paths", split_column="split"))
    return pd.concat([*baseline_rows, actual], ignore_index=True, sort=False)


def summarize_state_distribution(frame: pd.DataFrame, source: str, *, split_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for split, group in frame.groupby(split_column, sort=True):
        current = numeric_series(group, "current_pnl")
        a_hold = numeric_series(group, "a_hold")
        giveback = numeric_series(group, "giveback_from_mfe")
        minutes = numeric_series(group, "minutes_since_entry")
        entry_adv = numeric_series(group, "entry_a_enter") if "entry_a_enter" in group.columns else numeric_series(group, "a_enter")
        row = {
            "source": source,
            "split": str(split),
            "state_rows": int(len(group)),
            "trades": int(group["candidate_uid"].astype(str).nunique()) if "candidate_uid" in group.columns else 0,
            "oracle_hold_fraction": float(group["oracle_holding_action"].astype(str).eq("hold").mean()),
            "median_current_pnl": none_if_nan(current.median()),
            "median_a_hold": none_if_nan(a_hold.median()),
            "median_giveback_from_mfe": none_if_nan(giveback.median()),
            "median_minutes_since_entry": none_if_nan(minutes.median()),
            "median_entry_a_enter": none_if_nan(entry_adv.median()),
        }
        if "model_lifecycle_action" in group.columns:
            model_hold = group["model_lifecycle_action"].astype(str).eq("hold")
            oracle_exit = group["oracle_holding_action"].astype(str).eq("exit")
            oracle_hold = group["oracle_holding_action"].astype(str).eq("hold")
            row["model_hold_fraction"] = float(model_hold.mean())
            row["false_hold_rate_on_oracle_exit"] = safe_rate((model_hold & oracle_exit).sum(), oracle_exit.sum())
            row["false_exit_rate_on_oracle_hold"] = safe_rate(((~model_hold) & oracle_hold).sum(), oracle_hold.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def entry_attribution(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {}
    negative = trades[pd.to_numeric(trades["a_enter"], errors="coerce") < 0.0]
    positive_losers = trades[(pd.to_numeric(trades["a_enter"], errors="coerce") >= 0.0) & (pd.to_numeric(trades["pnl"], errors="coerce") < 0.0)]
    return {
        "trades": int(len(trades)),
        "negative_a_enter_trades": int(len(negative)),
        "negative_a_enter_pnl": float(pd.to_numeric(negative["pnl"], errors="coerce").fillna(0.0).sum()),
        "positive_a_enter_losing_trades": int(len(positive_losers)),
        "positive_a_enter_losing_pnl": float(pd.to_numeric(positive_losers["pnl"], errors="coerce").fillna(0.0).sum()),
        "median_a_enter": none_if_nan(pd.to_numeric(trades["a_enter"], errors="coerce").median()),
        "by_entry_advantage_bucket": records(summarize_by(add_entry_advantage_bucket(trades), ["reported_split", "entry_advantage_bucket"], "pnl")),
    }


def lifecycle_attribution(trades: pd.DataFrame, state_paths: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {}
    overhold_modes = {"never_exited_after_oracle_exit", "overheld_after_oracle_exit", "gave_back_mfe_then_lost"}
    overhold = trades[trades["lifecycle_timing_mode"].astype(str).isin(overhold_modes)]
    return {
        "trades": int(len(trades)),
        "overhold_or_late_exit_trades": int(len(overhold)),
        "overhold_or_late_exit_pnl": float(pd.to_numeric(overhold["pnl"], errors="coerce").fillna(0.0).sum()),
        "pnl_vs_first_oracle_exit": none_if_nan(pd.to_numeric(trades["pnl_vs_first_oracle_exit"], errors="coerce").sum()),
        "pnl_vs_best_path": float(pd.to_numeric(trades["pnl_vs_best_path"], errors="coerce").fillna(0.0).sum()),
        "median_model_hold_fraction_on_path": none_if_nan(pd.to_numeric(trades["model_hold_fraction_on_path"], errors="coerce").median()),
        "median_oracle_hold_fraction_on_path": none_if_nan(pd.to_numeric(trades["oracle_hold_fraction_on_path"], errors="coerce").median()),
        "false_hold_rows": int(pd.to_numeric(trades["false_hold_rows"], errors="coerce").fillna(0).sum()),
        "false_exit_rows": int(pd.to_numeric(trades["false_exit_rows"], errors="coerce").fillna(0).sum()),
        "timing_by_split": records(summarize_by(trades, ["reported_split", "lifecycle_timing_mode"], "pnl")),
        "state_rows": int(len(state_paths)),
    }


def skip_attribution(protocol276_skips: pd.DataFrame, reconstruction_skips: pd.DataFrame) -> dict[str, Any]:
    return {
        "protocol276_skip_rows": int(len(protocol276_skips)),
        "protocol276_skip_counts": count_by(protocol276_skips, "skip_reason"),
        "protocol276_skip_by_split": records(summarize_count(protocol276_skips, ["reported_split", "skip_reason"])),
        "reconstruction_skip_rows": int(len(reconstruction_skips)),
        "reconstruction_skip_counts": count_by(reconstruction_skips, "skip_reason"),
    }


def entry_only_bridge(protocol276_trades: pd.DataFrame, entry_only: pd.DataFrame) -> dict[str, Any]:
    if entry_only.empty:
        return {"entry_only_rows": 0}
    protocol276_ids = set(protocol276_trades.get("candidate_uid", pd.Series(dtype=str)).astype(str))
    entry = entry_only.copy()
    entry["integrated_trade_present"] = entry["candidate_uid"].astype(str).isin(protocol276_ids)
    missed = entry[~entry["integrated_trade_present"]]
    return {
        "entry_only_rows": int(len(entry)),
        "entry_only_rows_also_integrated": int(entry["integrated_trade_present"].sum()),
        "entry_only_rows_not_integrated": int(len(missed)),
        "entry_only_pnl_not_integrated": float(pd.to_numeric(missed.get("pnl"), errors="coerce").fillna(0.0).sum()) if not missed.empty else 0.0,
        "missed_by_split": records(summarize_by(missed, ["reported_split"], "pnl")) if not missed.empty else [],
    }


def root_cause_summary(trades: pd.DataFrame, state_paths: pd.DataFrame, skips: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if trades.empty:
        return ["No Protocol276 trades were reconstructed; attribution is blocked."]
    entry = entry_attribution(trades)
    lifecycle = lifecycle_attribution(trades, state_paths)
    skip_counts = count_by(skips, "skip_reason")
    lines.append(
        f"Entry policy selected {entry['negative_a_enter_trades']} negative-A_enter trades for {money(entry['negative_a_enter_pnl'])} PnL."
    )
    lines.append(
        f"Lifecycle timing left {lifecycle['overhold_or_late_exit_trades']} overhold/late-exit trades with {money(lifecycle['overhold_or_late_exit_pnl'])} PnL."
    )
    lines.append(
        f"Actual exits left {money(lifecycle['pnl_vs_best_path'])} versus each trade's best observed path PnL."
    )
    if skip_counts:
        lines.append(f"Replay skipped {sum(skip_counts.values())} rows, dominated by {skip_counts}.")
    if lifecycle.get("false_hold_rows", 0) > lifecycle.get("false_exit_rows", 0):
        lines.append("The lifecycle scorer shows a false-hold bias on actual entry-policy paths.")
    return lines


def decide(trades: pd.DataFrame, reconstruction_skips: pd.DataFrame) -> str:
    if trades.empty:
        return "blocked_protocol276_failure_attribution_no_reconstructed_trades"
    if len(reconstruction_skips):
        return "protocol276_failure_attribution_partial_reconstruction_foundation_work_required"
    return "protocol276_failure_attribution_complete_foundation_work_required"


def summarize_by(frame: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "rows", value, "avg_pnl", "win_rate"])
    working = frame.copy()
    working[value] = pd.to_numeric(working[value], errors="coerce").fillna(0.0)
    rows = []
    for group_key, group in working.groupby(keys, dropna=False, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {key: str(values[idx]) for idx, key in enumerate(keys)}
        row.update(
            {
                "rows": int(len(group)),
                value: float(group[value].sum()),
                "avg_pnl": float(group[value].mean()),
                "win_rate": float((group[value] > 0.0).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_count(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "rows"])
    return frame.groupby(keys, dropna=False, sort=True).size().reset_index(name="rows")


def add_entry_advantage_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["entry_advantage_bucket"] = pd.to_numeric(out["a_enter"], errors="coerce").map(entry_advantage_bucket)
    return out


def entry_advantage_bucket(value: Any) -> str:
    x = finite(value)
    if not math.isfinite(x):
        return "unknown"
    if x <= -500:
        return "strong_negative"
    if x < 0:
        return "negative"
    if x < 500:
        return "positive_lt_500"
    return "positive_gte_500"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Bottom Line",
        "",
        payload["next_direction"],
        "",
        "## Root Cause Summary",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["root_cause_summary"])
    entry = payload["entry_attribution"]
    lifecycle = payload["lifecycle_attribution"]
    lines.extend(
        [
            "",
            "## Entry Attribution",
            "",
            f"- Reconstructed trades: `{payload['reconstructed_trades']}`",
            f"- Negative `A_enter` trades: `{entry.get('negative_a_enter_trades', 0)}` for `{money(entry.get('negative_a_enter_pnl', 0.0))}` PnL.",
            f"- Positive `A_enter` losing trades: `{entry.get('positive_a_enter_losing_trades', 0)}` for `{money(entry.get('positive_a_enter_losing_pnl', 0.0))}` PnL.",
            f"- Median `A_enter`: `{fmt(entry.get('median_a_enter'))}`.",
            "",
            "## Lifecycle Attribution",
            "",
            f"- Reconstructed state rows: `{payload['reconstructed_state_rows']}`",
            f"- Overhold/late-exit trades: `{lifecycle.get('overhold_or_late_exit_trades', 0)}` for `{money(lifecycle.get('overhold_or_late_exit_pnl', 0.0))}` PnL.",
            f"- Aggregate PnL versus first oracle exit: `{money(lifecycle.get('pnl_vs_first_oracle_exit'))}`.",
            f"- Aggregate PnL versus best observed path: `{money(lifecycle.get('pnl_vs_best_path'))}`.",
            f"- False-hold rows: `{lifecycle.get('false_hold_rows', 0)}`; false-exit rows: `{lifecycle.get('false_exit_rows', 0)}`.",
            "",
            "## Timing Modes",
            "",
            table(lifecycle.get("timing_by_split", []), ["reported_split", "lifecycle_timing_mode", "rows", "pnl", "avg_pnl", "win_rate"]),
            "",
            "## State Distribution Shift",
            "",
            table(
                payload["state_distribution_shift"],
                [
                    "source",
                    "split",
                    "state_rows",
                    "trades",
                    "oracle_hold_fraction",
                    "model_hold_fraction",
                    "false_hold_rate_on_oracle_exit",
                    "median_entry_a_enter",
                    "median_current_pnl",
                    "median_a_hold",
                ],
                max_rows=40,
            ),
            "",
            "## Entry-Only Bridge",
            "",
            f"- Protocol271 entry-only rows: `{payload['entry_only_bridge'].get('entry_only_rows', 0)}`",
            f"- Entry-only rows also integrated: `{payload['entry_only_bridge'].get('entry_only_rows_also_integrated', 0)}`",
            f"- Entry-only rows not integrated: `{payload['entry_only_bridge'].get('entry_only_rows_not_integrated', 0)}`",
            f"- Entry-only PnL not integrated: `{money(payload['entry_only_bridge'].get('entry_only_pnl_not_integrated', 0.0))}`",
            "",
            "## Required Next Work",
            "",
            "1. Fix or gate the entry policy so negative flat action-advantage candidates are not treated as valid deployment entries.",
            "2. Build lifecycle labels/diagnostics on the actual Protocol271 entry-policy state distribution before any lifecycle retraining.",
            "3. Keep Protocol101 as paper default and rerun no model experiments until fill evidence, untouched holdout, and live no-order parity gates are closed.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Trade attribution: `{payload['outputs']['trade_attribution']}`",
            f"- Reconstructed state paths: `{payload['outputs']['state_paths']}`",
            f"- State distribution shift: `{payload['outputs']['state_distribution_shift']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
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
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                    "- Result: Protocol276 failure remains foundation work; no new model search is authorized by this audit.",
                ]
            )
            + "\n"
        )


def table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int = 20) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:max_rows]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                values.append(fmt(value))
            elif value is None:
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        overflow = ["..."] + [f"{len(rows) - max_rows} more rows"] + [""] * max(0, len(columns) - 2)
        lines.append("| " + " | ".join(overflow[: len(columns)]) + " |")
    return "\n".join(lines)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [{key: scalar(value) for key, value in row.items()} for row in frame.to_dict("records")]


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return none_if_nan(value)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return value


def count_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


def base_reconstruction_skip(trade: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "candidate_uid": str(trade.get("candidate_uid", "")),
        "trade_uid": str(trade.get("trade_uid", "")),
        "reported_split": str(trade.get("reported_split", "")),
        "session": str(trade.get("session", "")),
        "skip_reason": reason,
    }


def none_if_nan(value: Any) -> float | None:
    x = finite(value)
    return None if not math.isfinite(x) else float(x)


def safe_rate(num: Any, den: Any) -> float | None:
    denom = finite(den, 0.0)
    if denom <= 0:
        return None
    return float(finite(num, 0.0) / denom)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any) -> str:
    out = finite(value)
    return "" if not math.isfinite(out) else f"{out:.3f}"


def money(value: Any) -> str:
    out = finite(value)
    if not math.isfinite(out):
        return ""
    sign = "-" if out < 0 else ""
    return f"{sign}${abs(out):,.0f}"


if __name__ == "__main__":
    raise SystemExit(main())
