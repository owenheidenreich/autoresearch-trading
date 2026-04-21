from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import asdict
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from v2.core.walkforward import FoldSpec, generate_folds
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset, load_day_sidecar
from v3.logger.builder import select_contract
from v3.logger.run import run_day
from v3.logger.schema import BarRecord, DayLog
from v3.oracles.exit_headroom import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
    _time_stop_pnl,
    apply_exit_headroom_oracle,
)
from v3.oracles.opportunity import (
    _build_contract_paths,
    _contract_idx_for_record,
    apply_opportunity_oracle,
)
from v3.teachers.failed_break import FailedBreakTeacher
from v3.teachers.orc import ORCTeacher


ARTIFACT_ROOT = os.path.join("v3", "artifacts")
DEFAULT_DATASET_PATH = os.path.join(ARTIFACT_ROOT, "layer2_dataset.pkl")
DEFAULT_RUN_DIR = os.path.join(ARTIFACT_ROOT, "layer2_entry_side")
DEFAULT_ENTRY_QUANTILE = 0.60
DEFAULT_SIDE_QUANTILE = 0.10

W2A_FEATURE_NAMES = (
    "sigma_pos",
    "omar_retest_dist_norm",
    "omar_range_pct",
    "last10_range_over_omar",
    "inside_first15",
    "late_window_40_120_flag",
    "omar_mid_pos_units",
    "last10_break_state",
)

IDENTITY_COLUMNS = (
    "day",
    "bar_index",
    "fold_id",
)

ANALYSIS_ONLY_COLUMNS = (
    "best_forward_pnl_call",
    "best_forward_pnl_put",
    "worst_forward_pnl_call",
    "worst_forward_pnl_put",
    "entry_value_raw",
    "entry_value_rank",
    "side_margin_raw",
    "time_stop_pnl_call",
    "time_stop_pnl_put",
    "time_stop_value_raw",
    "time_stop_value_rank",
    "time_stop_margin_raw",
    "opportunity_oracle_entry",
    "opportunity_oracle_direction",
    "exit_value_by_selection",
)


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def save_pickle(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_teachers():
    return [ORCTeacher(), FailedBreakTeacher()]


def build_folds_for_dataset(ds: V2Dataset) -> list[FoldSpec]:
    unique_dates = sorted(set(ds.dates))
    return generate_folds(unique_dates)


def fold_test_day_map(folds: Iterable[FoldSpec]) -> dict[str, int]:
    out: dict[str, int] = {}
    for fold in folds:
        for day in fold.test_days:
            out[day] = fold.fold_idx
    return out


def minute_to_abs_index(dataset: V2Dataset, day: str) -> dict[int, int]:
    start, end = dataset.day_bar_range(day)
    out: dict[int, int] = {}
    for abs_idx in range(start, end):
        out[int(dataset.bar_of_day[abs_idx])] = abs_idx
    return out


def w2a_feature_values(dataset: V2Dataset, abs_idx: int) -> dict[str, float]:
    missing = [name for name in W2A_FEATURE_NAMES if name not in dataset.idx]
    if missing:
        raise RuntimeError(
            f"data.pt missing W2a features: {missing}. "
            "Layer-2 export requires FEATURE_CONTRACT_VERSION v2.2 or later."
        )
    row = dataset.X_sim[abs_idx]
    out: dict[str, float] = {}
    for name in W2A_FEATURE_NAMES:
        value = float(row[dataset.idx[name]])
        if name in {"inside_first15", "late_window_40_120_flag", "last10_break_state"}:
            value = float(int(round(value)))
        out[name] = value
    return out


def teacher_feature_values(bar: BarRecord) -> dict[str, float]:
    by_name = {t.teacher_name: t for t in bar.teachers}
    out: dict[str, float] = {
        "teacher_any_triggered": float(any(t.triggered for t in bar.teachers)),
        "teacher_n_triggered": float(sum(1 for t in bar.teachers if t.triggered)),
    }
    for teacher_name in ("orc", "failed_break"):
        t = by_name.get(teacher_name)
        trig = 1.0 if (t is not None and t.triggered) else 0.0
        buy_call = 1.0 if (t is not None and t.action == "buy_call") else 0.0
        buy_put = 1.0 if (t is not None and t.action == "buy_put") else 0.0
        out[f"{teacher_name}_triggered"] = trig
        out[f"{teacher_name}_buy_call"] = buy_call
        out[f"{teacher_name}_buy_put"] = buy_put
    return out


def teacher_direction_hint_from_row(row: pd.Series) -> str:
    if float(row.get("orc_buy_call", 0.0)) > 0.5:
        return "call"
    if float(row.get("orc_buy_put", 0.0)) > 0.5:
        return "put"
    if float(row.get("failed_break_buy_call", 0.0)) > 0.5:
        return "call"
    if float(row.get("failed_break_buy_put", 0.0)) > 0.5:
        return "put"
    return ""


def effective_direction(row: pd.Series, direction_mode: str) -> str:
    model_direction = str(row.get("predicted_direction", ""))
    teacher_direction = teacher_direction_hint_from_row(row)
    if direction_mode == "model":
        return model_direction
    if direction_mode == "teacher_if_triggered_else_model":
        return teacher_direction or model_direction
    if direction_mode == "teacher_if_triggered_else_put":
        return teacher_direction or "put"
    if direction_mode == "always_put":
        return "put"
    raise ValueError(f"Unknown direction_mode={direction_mode!r}")


def selected_value_for_direction_mode(
    row: pd.Series,
    direction_mode: str,
    call_col: str,
    put_col: str,
) -> float:
    direction = effective_direction(row, direction_mode)
    if direction == "call":
        return float(row[call_col]) if pd.notna(row[call_col]) else float("nan")
    if direction == "put":
        return float(row[put_col]) if pd.notna(row[put_col]) else float("nan")
    return float("nan")


def surface_feature_values(bar: BarRecord) -> dict[str, float]:
    n_passing_call = sum(1 for c in bar.contracts if c.passed and c.right == "C")
    n_passing_put = sum(1 for c in bar.contracts if c.passed and c.right == "P")
    s = bar.surface
    return {
        "n_contracts_total": float(s.n_contracts_total),
        "n_contracts_valid": float(s.n_contracts_valid),
        "n_contracts_passing": float(s.n_contracts_passing),
        "n_blocked_solely_by_premium_cap": float(s.n_blocked_solely_by_premium_cap),
        "any_contract_passes": float(int(s.any_contract_passes)),
        "any_passes_without_premium_cap": float(int(s.any_passes_without_premium_cap)),
        "passing_abs_delta_q25": float(s.passing_abs_delta_q25),
        "passing_abs_delta_q50": float(s.passing_abs_delta_q50),
        "passing_abs_delta_q75": float(s.passing_abs_delta_q75),
        "max_abs_delta_blocked_solely_by_cap": float(s.max_abs_delta_blocked_solely_by_cap),
        "n_passing_call": float(n_passing_call),
        "n_passing_put": float(n_passing_put),
        "has_passing_call": float(int(n_passing_call > 0)),
        "has_passing_put": float(int(n_passing_put > 0)),
    }


def build_labeled_day(
    dataset: V2Dataset,
    day: str,
    cfg: GuardrailConfig,
    equity: float,
) -> tuple[DayLog, dict] | tuple[None, None]:
    teachers = build_teachers()
    log = run_day(dataset, day, teachers, cfg, equity=equity)
    if not log.bars:
        return None, None
    sidecar = load_day_sidecar(dataset, day)
    if sidecar is None:
        return None, None
    day_start, _ = dataset.day_bar_range(day)
    apply_opportunity_oracle(log, sidecar, day_start, bars_per_day=390)
    apply_exit_headroom_oracle(log, sidecar, bars_per_day=390)
    return log, sidecar


def _directional_time_stop_pnl(
    bar: BarRecord,
    sidecar: dict,
    paths: dict[int, Any],
    direction: str,
    teacher_name: str = "layer2",
    session_end_bar: int = DEFAULT_SESSION_END_BAR,
    commission: float = DEFAULT_COMMISSION_PER_CONTRACT,
) -> float | None:
    selection = select_contract(bar.contracts, direction, teacher_name)
    if selection is None:
        return None
    match = None
    right = "P" if direction == "put" else "C"
    for c in bar.contracts:
        if c.strike == selection.strike and c.right == right:
            match = c
            break
    if match is None:
        return None
    cid = _contract_idx_for_record(sidecar, bar.bar_index, match)
    if cid is None or cid not in paths:
        return None
    path = paths[cid]
    return _time_stop_pnl(
        path.mids,
        entry_bar=bar.bar_index,
        entry_mid=match.mid,
        entry_spread_frac=match.spread_fraction,
        session_end_bar=session_end_bar,
        commission=commission,
    )


def oracle_slice_outcome(bar: BarRecord) -> str:
    if not bar.labels.opportunity_oracle_entry:
        return ""
    oracle_direction = bar.labels.opportunity_oracle_direction
    if oracle_direction not in ("call", "put"):
        return ""
    if not bar.surface.any_contract_passes:
        return "guardrail_suppression"
    triggered = [t for t in bar.teachers if t.triggered]
    if not triggered:
        return "abstention"
    triggered_dirs = {
        "call" if t.action == "buy_call" else "put"
        for t in triggered
    }
    if oracle_direction not in triggered_dirs:
        return "side_error"
    return "entered_right"


def build_export_rows_for_day(
    dataset: V2Dataset,
    day: str,
    log: DayLog,
    day_fold_id: int,
    sidecar: dict,
) -> list[dict[str, Any]]:
    minute_map = minute_to_abs_index(dataset, day)
    paths = _build_contract_paths(sidecar, 390)
    rows: list[dict[str, Any]] = []
    for bar in log.bars:
        abs_idx = minute_map.get(bar.bar_index)
        if abs_idx is None:
            continue
        oracle_call = bar.labels.best_forward_pnl_call
        oracle_put = bar.labels.best_forward_pnl_put
        if oracle_call is None and oracle_put is None:
            entry_value_raw = np.nan
            side_margin_raw = np.nan
        else:
            bests = [x for x in (oracle_call, oracle_put) if x is not None]
            entry_value_raw = float(max(bests)) if bests else np.nan
            side_margin_raw = (
                float(oracle_call - oracle_put)
                if oracle_call is not None and oracle_put is not None
                else np.nan
            )
        ts_call = _directional_time_stop_pnl(bar, sidecar, paths, "call")
        ts_put = _directional_time_stop_pnl(bar, sidecar, paths, "put")
        ts_values = [x for x in (ts_call, ts_put) if x is not None]
        time_stop_value_raw = float(max(ts_values)) if ts_values else np.nan
        time_stop_margin_raw = (
            float(ts_call - ts_put)
            if ts_call is not None and ts_put is not None
            else np.nan
        )
        row: dict[str, Any] = {
            "day": day,
            "bar_index": int(bar.bar_index),
            "fold_id": int(day_fold_id),
            "underlying_close": float(bar.underlying_close),
            "vwap": float(bar.vwap),
            "vwap_slope": float(bar.vwap_slope),
            "volume_ratio": float(bar.volume_ratio),
            "first15_high": float(bar.first15_high),
            "first15_low": float(bar.first15_low),
            "first15_range_pct": float(bar.first15_range_pct),
            "bars_since_break_above_first15": float(bar.bars_since_break_above_first15),
            "bars_since_break_below_first15": float(bar.bars_since_break_below_first15),
            "vix": float(bar.vix),
            "atm_iv": float(bar.atm_iv),
            "iv_percentile": float(bar.iv_percentile),
            "best_forward_pnl_call": oracle_call,
            "best_forward_pnl_put": oracle_put,
            "worst_forward_pnl_call": bar.labels.worst_forward_pnl_call,
            "worst_forward_pnl_put": bar.labels.worst_forward_pnl_put,
            "entry_value_raw": entry_value_raw,
            "entry_value_rank": np.nan,
            "side_margin_raw": side_margin_raw,
            "time_stop_pnl_call": ts_call,
            "time_stop_pnl_put": ts_put,
            "time_stop_value_raw": time_stop_value_raw,
            "time_stop_value_rank": np.nan,
            "time_stop_margin_raw": time_stop_margin_raw,
            "opportunity_oracle_entry": bool(bar.labels.opportunity_oracle_entry),
            "opportunity_oracle_direction": bar.labels.opportunity_oracle_direction or "",
            "exit_value_by_selection": {
                key: (
                    None
                    if value.get("best_exit_pnl") is None or value.get("time_stop_pnl") is None
                    else float(value["best_exit_pnl"] - value["time_stop_pnl"])
                )
                for key, value in bar.labels.exit_headroom_by_selection.items()
            },
            "oracle_slice_outcome": oracle_slice_outcome(bar),
        }
        row.update(teacher_feature_values(bar))
        row.update(surface_feature_values(bar))
        row.update(w2a_feature_values(dataset, abs_idx))
        rows.append(row)
    return rows


def finalize_export_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Layer-2 export produced zero rows.")
    df = df.sort_values(["day", "bar_index"]).reset_index(drop=True)
    df["entry_value_rank"] = 0.0

    for day, idx in df.groupby("day").groups.items():
        day_idx = list(idx)
        finite = df.loc[day_idx, "entry_value_raw"].notna()
        if finite.any():
            finite_idx = df.loc[day_idx].index[finite]
            ranks = df.loc[finite_idx, "entry_value_raw"].rank(method="average", pct=True)
            df.loc[finite_idx, "entry_value_rank"] = ranks.astype(float)
        ts_finite = df.loc[day_idx, "time_stop_value_raw"].notna()
        if ts_finite.any():
            ts_finite_idx = df.loc[day_idx].index[ts_finite]
            ts_ranks = df.loc[ts_finite_idx, "time_stop_value_raw"].rank(method="average", pct=True)
            df.loc[ts_finite_idx, "time_stop_value_rank"] = ts_ranks.astype(float)
    return df


def feature_names_from_export(df: pd.DataFrame) -> list[str]:
    excluded = set(IDENTITY_COLUMNS) | set(ANALYSIS_ONLY_COLUMNS) | {"oracle_slice_outcome"}
    names = [
        col for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return names


def export_metadata(dataset: V2Dataset, df: pd.DataFrame, folds: list[FoldSpec]) -> dict[str, Any]:
    return {
        "feature_names": feature_names_from_export(df),
        "analysis_columns": list(ANALYSIS_ONLY_COLUMNS) + ["oracle_slice_outcome"],
        "identity_columns": list(IDENTITY_COLUMNS),
        "w2a_feature_names": list(W2A_FEATURE_NAMES),
        "n_rows": int(len(df)),
        "n_days": int(df["day"].nunique()),
        "feature_contract_version": "v2.2",
        "n_v2_features": int(len(dataset.feature_names)),
        "folds": [asdict(fold) for fold in folds],
    }


def load_export_bundle(path: str) -> dict[str, Any]:
    bundle = load_pickle(path)
    if not isinstance(bundle, dict) or "rows" not in bundle or "meta" not in bundle:
        raise RuntimeError(f"Invalid Layer-2 export bundle: {path}")
    return bundle


def save_json(path: str, payload: dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _scored_eligible_rows(
    eligible: pd.DataFrame,
    score_mode: str,
    side_score_weight: float,
) -> pd.DataFrame:
    scored = eligible.copy()
    if score_mode == "product":
        scored["combined_score"] = scored["entry_score"] * scored["side_conf"]
    elif score_mode == "entry_only":
        scored["combined_score"] = scored["entry_score"]
    elif score_mode == "entry_plus_side":
        scored["combined_score"] = scored["entry_score"] + side_score_weight * scored["side_conf"]
    else:
        raise ValueError(f"Unknown score_mode={score_mode!r}")
    return scored


def per_day_choice(
    day_rows: pd.DataFrame,
    entry_threshold: float,
    side_threshold: float,
    *,
    score_mode: str = "product",
    side_score_weight: float = 0.15,
) -> Optional[pd.Series]:
    eligible = day_rows[
        (day_rows["entry_score"] >= entry_threshold)
        & (day_rows["side_conf"] >= side_threshold)
        & (day_rows["predicted_direction"] != "")
    ].copy()
    if eligible.empty:
        return None
    eligible = _scored_eligible_rows(
        eligible,
        score_mode=score_mode,
        side_score_weight=side_score_weight,
    )
    idx = eligible["combined_score"].idxmax()
    return eligible.loc[idx]


def calibrate_thresholds(
    val_pred: pd.DataFrame,
    *,
    direction_mode: str,
    calibration_mode: str,
    score_mode: str,
    side_score_weight: float,
    entry_quantile: float = DEFAULT_ENTRY_QUANTILE,
    side_quantile: float = DEFAULT_SIDE_QUANTILE,
) -> dict[str, float]:
    if calibration_mode == "fixed_quantiles":
        entry_threshold = float(np.quantile(val_pred["entry_score"], entry_quantile))
        side_threshold = float(np.quantile(val_pred["side_conf"], side_quantile))
        picks = []
        for _, day_rows in val_pred.groupby("day"):
            row = per_day_choice(
                day_rows,
                entry_threshold,
                side_threshold,
                score_mode=score_mode,
                side_score_weight=side_score_weight,
            )
            if row is not None:
                picks.append(row)
        coverage = len(picks) / max(val_pred["day"].nunique(), 1)
        chosen_value = float("nan")
        if picks:
            chosen = pd.DataFrame(picks)
            chosen_value = float(np.nanmean(
                chosen.apply(
                    selected_value_for_direction_mode,
                    axis=1,
                    direction_mode=direction_mode,
                    call_col="time_stop_pnl_call",
                    put_col="time_stop_pnl_put",
                )
            ))
        return {
            "calibration_mode": calibration_mode,
            "entry_threshold": entry_threshold,
            "side_threshold": side_threshold,
            "entry_quantile": float(entry_quantile),
            "side_quantile": float(side_quantile),
            "coverage": float(coverage),
            "objective_selected_forward_mean": chosen_value,
        }

    if calibration_mode != "search_mean_time_stop":
        raise ValueError(f"Unknown calibration_mode={calibration_mode!r}")

    entry_grid = np.linspace(0.50, 0.95, 10)
    side_grid = np.linspace(0.00, 0.90, 10)
    best = None
    best_score = -np.inf
    for e_q in entry_grid:
        entry_threshold = float(np.quantile(val_pred["entry_score"], e_q))
        for s_q in side_grid:
            side_threshold = float(np.quantile(val_pred["side_conf"], s_q))
            picks = []
            for _, day_rows in val_pred.groupby("day"):
                row = per_day_choice(
                    day_rows,
                    entry_threshold,
                    side_threshold,
                    score_mode=score_mode,
                    side_score_weight=side_score_weight,
                )
                if row is not None:
                    picks.append(row)
            coverage = len(picks) / max(val_pred["day"].nunique(), 1)
            if coverage < 0.60 or coverage > 1.00:
                continue
            if not picks:
                continue
            chosen = pd.DataFrame(picks)
            chosen_value = np.nanmean(
                chosen.apply(
                    selected_value_for_direction_mode,
                    axis=1,
                    direction_mode=direction_mode,
                    call_col="time_stop_pnl_call",
                    put_col="time_stop_pnl_put",
                )
            )
            if np.isnan(chosen_value):
                continue
            if chosen_value > best_score:
                best_score = float(chosen_value)
                best = {
                    "calibration_mode": calibration_mode,
                    "entry_threshold": entry_threshold,
                    "side_threshold": side_threshold,
                    "entry_quantile": float(e_q),
                    "side_quantile": float(s_q),
                    "coverage": float(coverage),
                    "objective_selected_forward_mean": float(chosen_value),
                }
    if best is None:
        return calibrate_thresholds(
            val_pred,
            direction_mode=direction_mode,
            calibration_mode="fixed_quantiles",
            score_mode=score_mode,
            side_score_weight=side_score_weight,
            entry_quantile=0.75,
            side_quantile=0.10,
        )
    return best


def teacher_baseline_choice(log: DayLog) -> tuple[Optional[BarRecord], Optional[str], Optional[str]]:
    for bar in log.bars:
        if not bar.selections:
            continue
        first = bar.selections[0]
        return bar, first.direction, first.teacher_name
    return None, None, None


def compute_time_stop_pnl_for_direction(
    bar: BarRecord,
    sidecar: dict,
    direction: str,
    teacher_name: str = "layer2",
    session_end_bar: int = DEFAULT_SESSION_END_BAR,
    commission: float = DEFAULT_COMMISSION_PER_CONTRACT,
) -> float | None:
    selection = select_contract(bar.contracts, direction, teacher_name)
    if selection is None:
        return None
    match = None
    for c in bar.contracts:
        if c.strike == selection.strike and c.right == ("P" if direction == "put" else "C"):
            match = c
            break
    if match is None:
        return None
    paths = _build_contract_paths(sidecar, 390)
    cid = _contract_idx_for_record(sidecar, bar.bar_index, match)
    if cid is None or cid not in paths:
        return None
    path = paths[cid]
    return _time_stop_pnl(
        path.mids,
        entry_bar=bar.bar_index,
        entry_mid=match.mid,
        entry_spread_frac=match.spread_fraction,
        session_end_bar=session_end_bar,
        commission=commission,
    )


def replay_metrics_from_pnls(pnls: list[float], starting_equity: float) -> dict[str, float]:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else math.inf
    equity = starting_equity
    peak = starting_equity
    max_dd_pct = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            dd_pct = (peak - equity) / peak * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)
    return {
        "trades": float(len(pnls)),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "pf": float(pf),
        "max_dd_pct": float(max_dd_pct),
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
    }


def side_accuracy_weighted(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    truth = np.sign(df["side_margin_raw"].to_numpy(dtype=float))
    pred = np.where(df["predicted_direction"] == "call", 1.0, -1.0)
    weights = np.abs(df["side_margin_raw"].to_numpy(dtype=float))
    mask = np.isfinite(truth) & (weights > 0)
    if not mask.any():
        return 0.0
    correct = (truth[mask] == pred[mask]).astype(float)
    return float(np.average(correct, weights=weights[mask]))
