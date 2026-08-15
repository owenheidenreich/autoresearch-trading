"""Disposable Protocol101 walking-skeleton model plumbing.

This module is deliberately quarantined from the formal Full Trader campaign.
It implements the smallest historical-only HGB path needed to exercise the
accepted FT2-08 feature adapter, the FT2-10 composer order, and simulator v5.
Nothing here is eligible for promotion, paper registration, or an alpha claim.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


QUARANTINE_LABELS = ("walking_skeleton", "throwaway", "paper-only")
ARTIFACT_PREFIX = "protocol101_walking_skeleton_throwaway_paper_only_"
MODEL_SEED = 101
MODEL_MAX_ITER = 100
MODEL_LEARNING_RATE = 0.05
MODEL_MAX_DEPTH = 3
MODEL_L2 = 1.0
MODEL_MAX_EXAMPLES = 1_000_000
HORIZONS = ("h3", "h5", "h10", "h20", "h45", "h90", "remaining_session")
LABEL_PREFIX = {"remaining_session": "session"}
QUALITY_HEADS = (
    ("early_drawdown_h10_q10_dollars", "h10_early_dd_dollars", 0.10),
    ("early_drawdown_h10_q10_return", "h10_early_dd_return", 0.10),
    ("time_to_first_profit_session_q90_minutes", "session_ttfp_minutes", 0.90),
    ("pre_profit_adverse_session_q10_dollars", "session_ppae_dollars", 0.10),
    ("pre_profit_adverse_session_q10_return", "session_ppae_return", 0.10),
    ("underwater_burden_session_q90_dollars", "session_uwi_dollars", 0.90),
    ("underwater_burden_session_q90_return", "session_uwi_return", 0.90),
    (
        "profitable_window_stability_session_q10_fraction",
        "session_fraction_positive",
        0.10,
    ),
)


def upside_head_specs() -> tuple[tuple[str, str, float], ...]:
    rows: list[tuple[str, str, float]] = []
    for horizon in HORIZONS:
        prefix = LABEL_PREFIX.get(horizon, horizon)
        rows.extend(
            (
                (f"upside_{horizon}_q10_mfe_dollars", f"{prefix}_mfe_dollars", 0.10),
                (f"upside_{horizon}_q10_mfe_return", f"{prefix}_mfe_return", 0.10),
                (
                    f"upside_{horizon}_q10_profit_area_dollars",
                    f"{prefix}_profit_area_dollars",
                    0.10,
                ),
                (
                    f"upside_{horizon}_q10_profit_area_return",
                    f"{prefix}_profit_area_return",
                    0.10,
                ),
            )
        )
    return tuple(rows)


PATH_HEADS = (*QUALITY_HEADS, *upside_head_specs())


def expected_gate_head_specs() -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for horizon in HORIZONS:
        prefix = LABEL_PREFIX.get(horizon, horizon)
        rows.extend(
            (
                (f"expected_upside_{horizon}_mean_mfe_dollars", f"{prefix}_mfe_dollars"),
                (f"expected_upside_{horizon}_mean_mfe_return", f"{prefix}_mfe_return"),
                (
                    f"expected_upside_{horizon}_mean_profit_area_dollars",
                    f"{prefix}_profit_area_dollars",
                ),
                (
                    f"expected_upside_{horizon}_mean_profit_area_return",
                    f"{prefix}_profit_area_return",
                ),
            )
        )
    return tuple(rows)


EXPECTED_GATE_HEADS = expected_gate_head_specs()
ACTION_HEADS = (
    "wait_probability",
    "expected_normalized_regret",
    "q90_normalized_regret",
)


@dataclass(frozen=True)
class HGBConfig:
    seed: int = MODEL_SEED
    max_iter: int = MODEL_MAX_ITER
    learning_rate: float = MODEL_LEARNING_RATE
    max_depth: int = MODEL_MAX_DEPTH
    l2_regularization: float = MODEL_L2
    max_examples: int = MODEL_MAX_EXAMPLES
    min_samples_leaf: int = 30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def finite_sample_quantile(values: Iterable[float], probability: float) -> float:
    finite = np.sort(
        np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    )
    if len(finite) == 0:
        raise ValueError("finite-sample quantile requires finite values")
    rank = int(math.ceil((len(finite) + 1) * float(probability))) - 1
    return float(finite[min(max(rank, 0), len(finite) - 1)])


def phase_from_minute(minute_et: str) -> str:
    hour, minute = (int(value) for value in str(minute_et).split(":")[:2])
    value = hour * 60 + minute
    if value < 9 * 60 + 45:
        return "opening_discovery"
    if value < 11 * 60 + 15:
        return "primary_morning"
    if value < 11 * 60 + 45:
        return "europe_close_transition"
    if value < 13 * 60 + 30:
        return "lunch"
    if value < 15 * 60:
        return "afternoon"
    if value < 15 * 60 + 30:
        return "power_hour_entry"
    return "manage_exit_only"


def premium_band(ask: float) -> str:
    cents = float(ask) * 100.0
    if cents <= 100.0:
        return "cheap_le_1"
    if cents <= 300.0:
        return "small_1_3"
    if cents <= 800.0:
        return "medium_3_8"
    if cents <= 2_000.0:
        return "large_8_20"
    return "very_large_20p"


def causal_horizon_available(decision_time_ns: int, horizon: str) -> bool:
    decision = pd.Timestamp(int(decision_time_ns), unit="ns", tz="UTC").tz_convert(
        "America/New_York"
    )
    fill = decision + pd.Timedelta(minutes=1)
    flat = decision.normalize() + pd.Timedelta(hours=15, minutes=55)
    if horizon == "remaining_session":
        return fill <= decision.normalize() + pd.Timedelta(hours=15, minutes=30)
    minutes = int(str(horizon)[1:])
    return fill + pd.Timedelta(minutes=minutes) <= flat


def action_mask_from_row(
    *,
    candidate_mask: bool,
    contract_id: str,
    current_ask: float,
    complete_ladder: bool,
    context_ready: bool,
    decision_time_ns: int,
    starting_equity: float = 10_000.0,
    realized_session_loss: float = 0.0,
    fee: float = 3.0,
) -> bool:
    if not candidate_mask or not complete_ladder or not context_ready or not contract_id:
        return False
    if not math.isfinite(float(current_ask)) or float(current_ask) < 1.0:
        return False
    decision = pd.Timestamp(int(decision_time_ns), unit="ns", tz="UTC").tz_convert(
        "America/New_York"
    )
    if (decision.hour, decision.minute) >= (15, 30):
        return False
    premium_plus_fee = float(current_ask) * 100.0 + float(fee)
    five_percent = float(starting_equity) * 0.05
    return bool(
        premium_plus_fee <= five_percent
        and float(realized_session_loss) + premium_plus_fee <= five_percent
    )


def _balanced_sample(
    frame: pd.DataFrame,
    *,
    target: str,
    feature_names: tuple[str, ...],
    max_examples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    required = [*feature_names, target, "session"]
    work = frame.loc[:, required].replace([np.inf, -np.inf], np.nan).dropna()
    before = len(work)
    sessions = sorted(work["session"].astype(str).unique())
    if len(work) > int(max_examples):
        per_session = max(1, int(max_examples) // max(1, len(sessions)))
        parts = []
        for index, session in enumerate(sessions):
            block = work[work["session"].astype(str) == session]
            take = min(len(block), per_session)
            parts.append(block.sample(n=take, random_state=int(seed) + index))
        work = pd.concat(parts, ignore_index=True)
        if len(work) > int(max_examples):
            work = work.sample(n=int(max_examples), random_state=int(seed))
    if work.empty:
        raise ValueError(f"no finite training rows for {target}")
    return (
        work.loc[:, list(feature_names)].to_numpy(dtype=float),
        work[target].to_numpy(dtype=float),
        {
            "finite_examples_before_cap": int(before),
            "fit_examples": int(len(work)),
            "fit_sessions": sessions,
            "target_min": float(work[target].min()),
            "target_max": float(work[target].max()),
            "target_mean": float(work[target].mean()),
        },
    )


def fit_regression_head(
    frame: pd.DataFrame,
    *,
    target: str,
    feature_names: tuple[str, ...],
    quantile: float | None,
    config: HGBConfig,
) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    x, y, summary = _balanced_sample(
        frame,
        target=target,
        feature_names=feature_names,
        max_examples=config.max_examples,
        seed=config.seed,
    )
    kwargs: dict[str, Any] = {
        "loss": "quantile" if quantile is not None else "squared_error",
        "learning_rate": config.learning_rate,
        "max_iter": config.max_iter,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "l2_regularization": config.l2_regularization,
        "early_stopping": False,
        "random_state": config.seed,
    }
    if quantile is not None:
        kwargs["quantile"] = float(quantile)
    model = HistGradientBoostingRegressor(**kwargs)
    model.fit(x, y)
    return model, {
        **summary,
        "family": "HistGradientBoostingRegressor",
        "loss": kwargs["loss"],
        "quantile": quantile,
        "n_iter": int(model.n_iter_),
        "config": config.to_dict(),
    }


def fit_wait_head(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    config: HGBConfig,
) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    x, y, summary = _balanced_sample(
        frame,
        target="wait_target",
        feature_names=feature_names,
        max_examples=config.max_examples,
        seed=config.seed,
    )
    classes = sorted(set(int(value) for value in y))
    if classes != [0, 1]:
        raise ValueError(f"WAIT target needs both classes, got {classes}")
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        min_samples_leaf=max(10, config.min_samples_leaf // 2),
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.seed,
    )
    model.fit(x, y.astype(int))
    return model, {
        **summary,
        "family": "HistGradientBoostingClassifier",
        "loss": "log_loss",
        "classes": classes,
        "n_iter": int(model.n_iter_),
        "config": config.to_dict(),
    }


def predict_regression(
    model: HistGradientBoostingRegressor,
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(model.predict(frame.loc[:, list(feature_names)].to_numpy(float)), dtype=float)


def fit_isotonic_state(predicted: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) == 0:
        raise ValueError("isotonic calibration has no finite rows")
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {
            "method": "constant",
            "sample_count": int(len(x)),
            "constant": float(np.mean(y)),
        }
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(x, y)
    return {
        "method": "isotonic",
        "sample_count": int(len(x)),
        "x": [float(value) for value in model.X_thresholds_],
        "y": [float(value) for value in model.y_thresholds_],
    }


def apply_isotonic_state(values: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    if state["method"] == "constant":
        return np.full(len(values), float(state["constant"]), dtype=float)
    return np.interp(
        np.asarray(values, dtype=float),
        np.asarray(state["x"], dtype=float),
        np.asarray(state["y"], dtype=float),
    )


def pooled_wait_frame(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    eligible = frame[frame["action_eligible"].astype(bool)].copy()
    groups: list[dict[str, Any]] = []
    for (session, decision_ns), block in frame.groupby(
        ["session", "decision_time_ns"], sort=True
    ):
        active = eligible[
            (eligible["session"] == session)
            & (eligible["decision_time_ns"] == decision_ns)
        ]
        source = active if not active.empty else block
        item: dict[str, Any] = {
            "session": str(session),
            "decision_time_ns": int(decision_ns),
            "eligible_fraction": float(len(active) / max(1, len(block))),
        }
        for feature in feature_names:
            values = pd.to_numeric(source[feature], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            item[f"{feature}__mean"] = float(np.mean(finite)) if len(finite) else 0.0
            item[f"{feature}__max"] = float(np.max(finite)) if len(finite) else 0.0
            item[f"{feature}__min"] = float(np.min(finite)) if len(finite) else 0.0
        if "wait_target" in block:
            targets = pd.to_numeric(block["wait_target"], errors="coerce").dropna().unique()
            item["wait_target"] = float(targets[0]) if len(targets) == 1 else np.nan
        groups.append(item)
    return pd.DataFrame(groups)


def wait_feature_names(feature_names: tuple[str, ...]) -> tuple[str, ...]:
    return (
        "eligible_fraction",
        *tuple(f"{name}__{stat}" for name in feature_names for stat in ("mean", "max", "min")),
    )


def _midrank_knots(values: np.ndarray, weights: np.ndarray) -> tuple[list[float], list[float]]:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    total = float(np.sum(weights))
    unique: list[float] = []
    knots: list[float] = []
    lower = 0.0
    cursor = 0
    while cursor < len(values):
        value = float(values[cursor])
        end = cursor + 1
        while end < len(values) and float(values[end]) == value:
            end += 1
        tied = float(np.sum(weights[cursor:end]))
        unique.append(value)
        knots.append((lower + 0.5 * tied) / total)
        lower += tied
        cursor = end
    return unique, knots


def build_nested_cdfs(
    frame: pd.DataFrame,
    target_columns: Iterable[str],
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for target in target_columns:
        for band in sorted(frame["premium_band"].dropna().astype(str).unique()):
            band_frame = frame[frame["premium_band"].astype(str) == band]
            exact_phases = sorted(band_frame["market_phase"].dropna().astype(str).unique())
            for phase in [*exact_phases, "__POOLED__"]:
                source = (
                    band_frame
                    if phase == "__POOLED__"
                    else band_frame[band_frame["market_phase"].astype(str) == phase]
                )
                source = source[["session", target]].replace([np.inf, -np.inf], np.nan).dropna()
                if source.empty:
                    continue
                counts = source.groupby("session")[target].transform("count").to_numpy(float)
                session_count = source["session"].nunique()
                weights = 1.0 / (float(session_count) * counts)
                values = source[target].to_numpy(float)
                knots_x, knots_y = _midrank_knots(values, weights)
                key = f"{target}|{band}|{phase}"
                artifacts[key] = {
                    "target": target,
                    "premium_band": band,
                    "market_phase": phase,
                    "distinct_sessions": int(session_count),
                    "finite_rows": int(len(source)),
                    "x": knots_x,
                    "cdf": knots_y,
                }
    return artifacts


def cdf_rank(
    cdfs: dict[str, Any],
    *,
    target: str,
    band: str,
    phase: str,
    value: float,
) -> tuple[float, str]:
    if not math.isfinite(float(value)):
        return float("nan"), "nonfinite"
    exact = f"{target}|{band}|{phase}"
    pooled = f"{target}|{band}|__POOLED__"
    key = next(
        (
            candidate
            for candidate in (exact, pooled)
            if candidate in cdfs
            and int(cdfs[candidate].get("distinct_sessions", 0)) >= 8
            and int(cdfs[candidate].get("finite_rows", 0)) >= 2_000
        ),
        None,
    )
    if key is None:
        return float("nan"), "missing_same_band"
    state = cdfs[key]
    if "_x_np" not in state:
        state["_x_np"] = np.asarray(state["x"], dtype=float)
        state["_y_np"] = np.asarray(state["cdf"], dtype=float)
    x = state["_x_np"]
    y = state["_y_np"]
    if len(x) == 1:
        if value < x[0]:
            return 0.0, key
        if value > x[0]:
            return 1.0, key
        return 0.5, key
    return float(np.interp(float(value), x, y, left=0.0, right=1.0)), key


def realized_u_label(row: pd.Series, cdfs: dict[str, Any]) -> float:
    scores: list[float] = []
    band = str(row["premium_band"])
    phase = str(row["market_phase"])
    for horizon in HORIZONS:
        if not causal_horizon_available(int(row["decision_time_ns"]), horizon):
            continue
        prefix = LABEL_PREFIX.get(horizon, horizon)
        if bool(row.get(f"{prefix}_censored", True)):
            continue
        parts: list[float] = []
        for target in (
            f"{prefix}_mfe_dollars",
            f"{prefix}_mfe_return",
            f"{prefix}_profit_area_dollars",
            f"{prefix}_profit_area_return",
        ):
            rank, _ = cdf_rank(
                cdfs,
                target=target,
                band=band,
                phase=phase,
                value=float(row.get(target, np.nan)),
            )
            parts.append(rank)
        if all(math.isfinite(value) for value in parts):
            scores.append(float(min(parts)))
    return float(np.mean(scores)) if scores else float("nan")


def realized_fee_cleared(row: pd.Series) -> bool:
    axes: list[list[float]] = [[], [], [], []]
    for horizon in HORIZONS:
        if not causal_horizon_available(int(row["decision_time_ns"]), horizon):
            continue
        prefix = LABEL_PREFIX.get(horizon, horizon)
        if bool(row.get(f"{prefix}_censored", True)):
            continue
        values = (
            row.get(f"{prefix}_mfe_dollars", np.nan),
            row.get(f"{prefix}_mfe_return", np.nan),
            row.get(f"{prefix}_profit_area_dollars", np.nan),
            row.get(f"{prefix}_profit_area_return", np.nan),
        )
        if all(math.isfinite(float(value)) for value in values):
            for target, value in zip(axes, values):
                target.append(float(value))
    return bool(axes[0] and all(float(np.mean(values)) > 0.0 for values in axes))


def _vector_cdf_rank(
    frame: pd.DataFrame,
    cdfs: dict[str, Any],
    target: str,
) -> np.ndarray:
    result = np.full(len(frame), np.nan, dtype=float)
    bands = frame["premium_band"].astype(str)
    phases = frame["market_phase"].astype(str)
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    for band in sorted(bands.unique()):
        band_mask = bands.to_numpy() == band
        for phase in sorted(phases[band_mask].unique()):
            mask = band_mask & (phases.to_numpy() == phase) & np.isfinite(values)
            if not mask.any():
                continue
            exact = f"{target}|{band}|{phase}"
            pooled = f"{target}|{band}|__POOLED__"
            key = next(
                (
                    candidate
                    for candidate in (exact, pooled)
                    if candidate in cdfs
                    and int(cdfs[candidate].get("distinct_sessions", 0)) >= 8
                    and int(cdfs[candidate].get("finite_rows", 0)) >= 2_000
                ),
                None,
            )
            if key is None:
                continue
            state = cdfs[key]
            if "_x_np" not in state:
                state["_x_np"] = np.asarray(state["x"], dtype=float)
                state["_y_np"] = np.asarray(state["cdf"], dtype=float)
            x = state["_x_np"]
            y = state["_y_np"]
            if len(x) == 1:
                selected = values[mask]
                ranked = np.where(selected < x[0], 0.0, np.where(selected > x[0], 1.0, 0.5))
            else:
                ranked = np.interp(values[mask], x, y, left=0.0, right=1.0)
            result[mask] = ranked
    return result


def attach_rlac_targets(frame: pd.DataFrame, cdfs: dict[str, Any]) -> pd.DataFrame:
    """Attach the exact one-pass alpha=0 RLAC targets using vectorized CDFs."""

    out = frame.copy()
    count = len(out)
    u_sum = np.zeros(count, dtype=float)
    horizon_count = np.zeros(count, dtype=np.int16)
    axis_sums = np.zeros((count, 4), dtype=float)
    axis_count = np.zeros(count, dtype=np.int16)
    decision = pd.to_datetime(out["decision_time_ns"], unit="ns", utc=True).dt.tz_convert(
        "America/New_York"
    )
    decision_minute = (decision.dt.hour * 60 + decision.dt.minute).to_numpy(int)
    for horizon in HORIZONS:
        prefix = LABEL_PREFIX.get(horizon, horizon)
        if horizon == "remaining_session":
            causal = decision_minute + 1 <= 15 * 60 + 30
        else:
            causal = decision_minute + 1 + int(horizon[1:]) <= 15 * 60 + 55
        uncensored = ~out[f"{prefix}_censored"].fillna(True).astype(bool).to_numpy()
        targets = (
            f"{prefix}_mfe_dollars",
            f"{prefix}_mfe_return",
            f"{prefix}_profit_area_dollars",
            f"{prefix}_profit_area_return",
        )
        values = np.column_stack(
            [pd.to_numeric(out[target], errors="coerce").to_numpy(float) for target in targets]
        )
        ranks = np.column_stack(
            [_vector_cdf_rank(out, cdfs, target) for target in targets]
        )
        valid = (
            causal
            & uncensored
            & np.isfinite(values).all(axis=1)
            & np.isfinite(ranks).all(axis=1)
        )
        u_sum[valid] += np.min(ranks[valid], axis=1)
        horizon_count[valid] += 1
        axis_sums[valid] += values[valid]
        axis_count[valid] += 1
    u_label = np.divide(
        u_sum,
        horizon_count,
        out=np.full(count, np.nan, dtype=float),
        where=horizon_count > 0,
    )
    means = np.divide(
        axis_sums,
        axis_count[:, None],
        out=np.full_like(axis_sums, np.nan),
        where=axis_count[:, None] > 0,
    )
    out["u_label"] = u_label
    out["fee_cleared_label"] = (axis_count > 0) & (means > 0.0).all(axis=1)
    eligible_u = out["action_eligible"].astype(bool) & np.isfinite(u_label)
    group_keys = [out["session"], out["decision_time_ns"]]
    maximum = out["u_label"].where(eligible_u).groupby(group_keys).transform("max")
    group_has_eligible = eligible_u.groupby(group_keys).transform("any")
    passing = eligible_u & out["fee_cleared_label"].astype(bool)
    group_has_pass = passing.groupby(group_keys).transform("any")
    out["wait_target"] = np.where(
        group_has_eligible,
        (~group_has_pass).astype(float),
        np.nan,
    )
    out["normalized_regret"] = np.where(
        eligible_u,
        np.clip(maximum.to_numpy(float) - u_label, 0.0, 1.0),
        np.nan,
    )
    return out


def _cluster_members(block: pd.DataFrame, selected: pd.Series) -> pd.Series:
    return (
        (block["right"].astype(str) == str(selected["right"]))
        & (block["expiry"].astype(str) == str(selected["expiry"]))
        & (
            (block["strike_milli_points"].astype(int) - int(selected["strike_milli_points"]))
            .abs()
            <= 10_000
        )
    )


def compose_decision(
    block: pd.DataFrame,
    *,
    cdfs: dict[str, Any],
    model_gap_error: float,
    mfe_error_margin_dollars: float = 0.0,
    mfe_error_margin_return: float = 0.0,
    source_transfer_error: float = 0.0,
    uncertainty_multiplier: float = 1.0,
    guardrail_alpha: float = 0.0,
    regret_bound_maximum: float = 0.10,
    action_conditioned_gate_available: bool = True,
) -> dict[str, Any]:
    decision_ns = int(block["decision_time_ns"].iloc[0])
    base = {
        "session": str(block["session"].iloc[0]),
        "decision_time_ns": decision_ns,
        "decision_time_utc": pd.Timestamp(decision_ns, unit="ns", tz="UTC").isoformat(),
        "action": "WAIT",
        "selected_contract_id": None,
        "selected_score": None,
        "wait_reason": None,
        "composer_order": [
            "complete_ladder_and_physical_safety_action_mask",
            "required_causal_state_check",
            "causal_horizon_availability",
            f"stage_1_quality_screen_alpha_{float(guardrail_alpha):g}",
            "stage_2_conservative_upside_rank",
            "positive_after_fee_check",
            "directional_substitute_cluster_uncertainty_wait_check",
            "selected_contract_q90_regret_action_check",
            "mandatory_action_conditioned_gate_check",
            "deterministic_tie_break",
        ],
    }
    eligible = block[block["action_eligible"].astype(bool)].copy()
    if eligible.empty:
        return {**base, "wait_reason": "no_physically_eligible_contract"}
    required = [f"{name}__calibrated" for name, _, _ in PATH_HEADS]
    required.extend(
        f"{name}__calibrated_lower" for name, _ in EXPECTED_GATE_HEADS
    )
    required.extend(
        ("expected_normalized_regret__calibrated", "q90_regret_upper_bound")
    )
    eligible = eligible.dropna(subset=required)
    if eligible.empty:
        return {**base, "wait_reason": "required_forecast_state_missing"}
    if float(guardrail_alpha) not in {0.0, 0.1, 0.2, 0.25, 0.3}:
        raise ValueError(f"unregistered guardrail alpha: {guardrail_alpha}")
    if float(guardrail_alpha) > 0.0:
        quality = (
            (
                "early_drawdown_h10_q10_dollars",
                "h10_early_dd_dollars",
                "higher",
            ),
            (
                "early_drawdown_h10_q10_return",
                "h10_early_dd_return",
                "higher",
            ),
            (
                "time_to_first_profit_session_q90_minutes",
                "session_ttfp_minutes",
                "lower",
            ),
            (
                "pre_profit_adverse_session_q10_dollars",
                "session_ppae_dollars",
                "higher",
            ),
            (
                "pre_profit_adverse_session_q10_return",
                "session_ppae_return",
                "higher",
            ),
            (
                "underwater_burden_session_q90_dollars",
                "session_uwi_dollars",
                "lower",
            ),
            (
                "underwater_burden_session_q90_return",
                "session_uwi_return",
                "lower",
            ),
            (
                "profitable_window_stability_session_q10_fraction",
                "session_fraction_positive",
                "higher",
            ),
        )
        retained: list[int] = []
        for index, row in eligible.iterrows():
            passes = True
            for head, target, direction in quality:
                rank, _ = cdf_rank(
                    cdfs,
                    target=target,
                    band=str(row["premium_band"]),
                    phase=str(row["market_phase"]),
                    value=float(row[f"{head}__calibrated"]),
                )
                if not math.isfinite(rank):
                    passes = False
                    break
                if direction == "higher" and rank < float(guardrail_alpha):
                    passes = False
                    break
                if direction == "lower" and rank > 1.0 - float(guardrail_alpha):
                    passes = False
                    break
            if passes:
                retained.append(int(index))
        eligible = eligible.loc[retained]
        if eligible.empty:
            return {**base, "wait_reason": "no_contract_clears_quality_guardrail"}
    available = [
        horizon for horizon in HORIZONS if causal_horizon_available(decision_ns, horizon)
    ]
    if not available:
        return {**base, "wait_reason": "no_causal_horizon_available"}
    rows: list[dict[str, Any]] = []
    for index, row in eligible.iterrows():
        balanced: list[float] = []
        mfe_dollars: list[float] = []
        mfe_returns: list[float] = []
        area_dollars: list[float] = []
        area_returns: list[float] = []
        expected_mfe_dollars: list[float] = []
        expected_mfe_returns: list[float] = []
        expected_area_dollars: list[float] = []
        expected_area_returns: list[float] = []
        cdf_trace: list[str] = []
        failed = False
        for horizon in available:
            prefix = LABEL_PREFIX.get(horizon, horizon)
            predictions = (
                float(row[f"upside_{horizon}_q10_mfe_dollars__calibrated"]),
                float(row[f"upside_{horizon}_q10_mfe_return__calibrated"]),
                float(row[f"upside_{horizon}_q10_profit_area_dollars__calibrated"]),
                float(row[f"upside_{horizon}_q10_profit_area_return__calibrated"]),
            )
            expected_lower = (
                float(row[f"expected_upside_{horizon}_mean_mfe_dollars__calibrated_lower"]),
                float(row[f"expected_upside_{horizon}_mean_mfe_return__calibrated_lower"]),
                float(row[f"expected_upside_{horizon}_mean_profit_area_dollars__calibrated_lower"]),
                float(row[f"expected_upside_{horizon}_mean_profit_area_return__calibrated_lower"]),
            )
            ranks: list[float] = []
            for target, value in zip(
                (
                    f"{prefix}_mfe_dollars",
                    f"{prefix}_mfe_return",
                    f"{prefix}_profit_area_dollars",
                    f"{prefix}_profit_area_return",
                ),
                predictions,
            ):
                rank, trace = cdf_rank(
                    cdfs,
                    target=target,
                    band=str(row["premium_band"]),
                    phase=str(row["market_phase"]),
                    value=value,
                )
                ranks.append(rank)
                cdf_trace.append(trace)
            if not all(
                math.isfinite(value)
                for value in (*predictions, *expected_lower, *ranks)
            ):
                failed = True
                break
            mfe_dollars.append(predictions[0])
            mfe_returns.append(predictions[1])
            area_dollars.append(predictions[2])
            area_returns.append(predictions[3])
            expected_mfe_dollars.append(expected_lower[0])
            expected_mfe_returns.append(expected_lower[1])
            expected_area_dollars.append(expected_lower[2])
            expected_area_returns.append(expected_lower[3])
            balanced.append(min(min(ranks[0], ranks[1]), min(ranks[2], ranks[3])))
        if failed or not balanced:
            continue
        positive = all(
            float(np.mean(values)) > 0.0
            for values in (
                expected_mfe_dollars,
                expected_mfe_returns,
                expected_area_dollars,
                expected_area_returns,
            )
        )
        if not positive:
            continue
        rows.append(
            {
                "index": int(index),
                "primary": float(np.mean(balanced)),
                "secondary": float(np.min(balanced)),
                "tertiary": float(np.mean(mfe_returns)),
                "quaternary": float(np.mean(mfe_dollars)),
                "mean_mfe_dollars": float(np.mean(mfe_dollars)),
                "mean_mfe_return": float(np.mean(mfe_returns)),
                "mean_profit_area_dollars": float(np.mean(area_dollars)),
                "mean_profit_area_return": float(np.mean(area_returns)),
                "gate_mean_expected_mfe_dollars_lower": float(
                    np.mean(expected_mfe_dollars)
                ),
                "gate_mean_expected_mfe_return_lower": float(
                    np.mean(expected_mfe_returns)
                ),
                "gate_mean_expected_profit_area_dollars_lower": float(
                    np.mean(expected_area_dollars)
                ),
                "gate_mean_expected_profit_area_return_lower": float(
                    np.mean(expected_area_returns)
                ),
                "cdf_trace": sorted(set(cdf_trace)),
            }
        )
    if not rows:
        return {**base, "wait_reason": "no_positive_after_fee_expected_upside"}
    ranked = sorted(
        rows,
        key=lambda item: (
            -item["primary"],
            -item["secondary"],
            -item["tertiary"],
            -item["quaternary"],
            str(eligible.loc[item["index"], "expiry"]),
            int(eligible.loc[item["index"], "strike_milli_points"]),
            str(eligible.loc[item["index"], "right"]),
            str(eligible.loc[item["index"], "contract_id"]),
        ),
    )
    chosen = ranked[0]
    selected = eligible.loc[chosen["index"]]
    ranked_frame = eligible.loc[[item["index"] for item in ranked]].copy()
    rank_map = {item["index"]: item["primary"] for item in ranked}
    cluster = _cluster_members(ranked_frame, selected)
    cluster_score = max(rank_map[int(index)] for index in ranked_frame[cluster].index)
    outside = [
        rank_map[int(index)] for index in ranked_frame[~cluster].index
    ]
    outside_score = max(outside) if outside else 0.0
    predicted_gap = float(cluster_score - outside_score)
    wait_margin = float(uncertainty_multiplier) * (
        float(model_gap_error) + float(source_transfer_error)
    )
    common = {
        "proposed_contract_id": str(selected["contract_id"]),
        "proposed_contract_index": int(chosen["index"]),
        "selected_cluster_score": float(cluster_score),
        "outside_cluster_score": float(outside_score),
        "predicted_cluster_gap": predicted_gap,
        "wait_margin": wait_margin,
        "selected_q90_regret_upper_bound": float(selected["q90_regret_upper_bound"]),
        "selected_expected_regret": float(
            selected["expected_normalized_regret__calibrated"]
        ),
        "selected_mean_mfe_dollars": float(chosen["mean_mfe_dollars"]),
        "selected_mean_mfe_return": float(chosen["mean_mfe_return"]),
        "selected_mean_profit_area_dollars": float(
            chosen["mean_profit_area_dollars"]
        ),
        "selected_mean_profit_area_return": float(
            chosen["mean_profit_area_return"]
        ),
        "selected_gate_mean_expected_mfe_dollars_lower": float(
            chosen["gate_mean_expected_mfe_dollars_lower"]
        ),
        "selected_gate_mean_expected_mfe_return_lower": float(
            chosen["gate_mean_expected_mfe_return_lower"]
        ),
        "selected_gate_mean_expected_profit_area_dollars_lower": float(
            chosen["gate_mean_expected_profit_area_dollars_lower"]
        ),
        "selected_gate_mean_expected_profit_area_return_lower": float(
            chosen["gate_mean_expected_profit_area_return_lower"]
        ),
        "mfe_error_margin_dollars": float(mfe_error_margin_dollars),
        "mfe_error_margin_return": float(mfe_error_margin_return),
        "guardrail_alpha": float(guardrail_alpha),
        "available_horizons": available,
        "cdf_trace": chosen["cdf_trace"],
    }
    if not predicted_gap > wait_margin:
        return {**base, **common, "wait_reason": "uncertainty_margin_not_strictly_cleared"}
    if not (
        float(chosen["mean_mfe_dollars"]) > float(mfe_error_margin_dollars)
        and float(chosen["mean_mfe_return"]) > float(mfe_error_margin_return)
    ):
        return {**base, **common, "wait_reason": "mfe_error_margin_not_strictly_cleared"}
    if float(selected["q90_regret_upper_bound"]) > float(regret_bound_maximum):
        return {**base, **common, "wait_reason": "q90_regret_bound_above_0_10"}
    if not action_conditioned_gate_available:
        return {**base, **common, "wait_reason": "action_conditioned_gate_unavailable"}
    return {
        **base,
        **common,
        "action": "BUY",
        "selected_contract_id": str(selected["contract_id"]),
        "selected_score": float(chosen["primary"]),
        "selected_right": str(selected["right"]),
        "selected_strike_idx": int(selected["strike_idx"]),
        "selected_right_idx": int(selected["right_idx"]),
        "selected_current_ask": float(selected["decision_entry_ask"]),
        "wait_reason": None,
    }


def assert_quarantine_payload(payload: Any) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    for label in QUARANTINE_LABELS:
        if label not in encoded:
            raise AssertionError(f"quarantine label absent: {label}")
