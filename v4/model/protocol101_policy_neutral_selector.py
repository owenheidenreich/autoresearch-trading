"""Governed policy-neutral contract-selection primitives for Protocol101.

The module deliberately excludes entry timing and lifecycle decisions.  It
constructs the frozen decision-local target, fits the two preregistered HGB
selectors, and provides the synchronized session-block inference used by the
campaign runner and its independent checker.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel


SCHEMA_VERSION = "Protocol101PolicyNeutralSelectorV1"
FEE_DOLLARS = 3.0
HORIZONS_MINUTES = (1, 2, 5, 10, 15)
BARRIERS = (
    ("f10_a10", 0.10, -0.10),
    ("f25_a20", 0.25, -0.20),
    ("f50_a35", 0.50, -0.35),
)
RETURN_COLUMNS = tuple(f"return_{value}m" for value in HORIZONS_MINUTES)
PATH_COLUMNS = ("mfe_15m", "mae_15m")
BREAKEVEN_COLUMNS = ("time_to_breakeven_quality",)
BARRIER_COLUMNS = tuple(f"barrier_{name}" for name, _, _ in BARRIERS)
RAW_TARGET_COLUMNS = (
    *RETURN_COLUMNS,
    *PATH_COLUMNS,
    *BREAKEVEN_COLUMNS,
    *BARRIER_COLUMNS,
)
UTILITY_COMPONENT_COLUMNS = (
    "return_family",
    "path_family",
    "breakeven_family",
    "barrier_family",
)
M0_FEATURES = ("E.bs.delta", "E.bs.gamma")
M1_FEATURES = tuple(FEATURE_NAMES)
MODEL_SEEDS = (42, 43, 44)
SHUFFLE_SEEDS = tuple(range(8600, 8620))
MODEL_CONFIG = {
    "loss": "squared_error",
    "max_iter": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def numeric_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def latest_causal_bid(
    quote_ns: np.ndarray,
    bids: np.ndarray,
    *,
    decision_ns: int,
    horizon_ns: int,
    max_age_ns: int = 90_000_000_000,
) -> float:
    """Return the last executable bid at or before a horizon."""

    index = int(np.searchsorted(quote_ns, int(horizon_ns), side="right")) - 1
    if index < 0 or int(quote_ns[index]) <= int(decision_ns):
        return float("nan")
    if int(horizon_ns) - int(quote_ns[index]) > int(max_age_ns):
        return float("nan")
    bid = float(bids[index])
    return bid if math.isfinite(bid) and bid >= 0.0 else float("nan")


def path_target_record(
    *,
    quote_ns: np.ndarray,
    bids: np.ndarray,
    decision_ns: int,
    entry_ask: float,
) -> dict[str, float]:
    """Compute the frozen executable 15-minute path target for one contract."""

    if not math.isfinite(entry_ask) or entry_ask <= 0.0:
        raise ValueError("entry ask must be finite and positive")
    quote_ns = np.asarray(quote_ns, dtype=np.int64)
    bids = np.asarray(bids, dtype=np.float64)
    if quote_ns.ndim != 1 or bids.shape != quote_ns.shape:
        raise ValueError("quote path axes are misaligned")
    if len(quote_ns) and np.any(np.diff(quote_ns) <= 0):
        raise ValueError("quote path must be strictly chronological")

    premium = float(entry_ask) * 100.0
    path_end_ns = int(decision_ns) + 15 * 60_000_000_000
    start = int(np.searchsorted(quote_ns, int(decision_ns), side="right"))
    stop = int(np.searchsorted(quote_ns, path_end_ns, side="right"))
    path_times = quote_ns[start:stop]
    path_bids = bids[start:stop]
    finite = np.isfinite(path_bids) & (path_bids >= 0.0)
    path_times = path_times[finite]
    path_bids = path_bids[finite]
    path_returns = (
        ((path_bids - float(entry_ask)) * 100.0 - FEE_DOLLARS) / premium
        if len(path_bids)
        else np.asarray([], dtype=np.float64)
    )

    record: dict[str, float] = {}
    for minutes in HORIZONS_MINUTES:
        horizon_ns = int(decision_ns) + int(minutes) * 60_000_000_000
        bid = latest_causal_bid(
            quote_ns,
            bids,
            decision_ns=int(decision_ns),
            horizon_ns=horizon_ns,
        )
        record[f"return_{minutes}m"] = (
            ((bid - float(entry_ask)) * 100.0 - FEE_DOLLARS) / premium
            if math.isfinite(bid)
            else float("nan")
        )

    record["mfe_15m"] = (
        float(np.max(path_returns)) if len(path_returns) else float("nan")
    )
    record["mae_15m"] = (
        float(np.min(path_returns)) if len(path_returns) else float("nan")
    )
    breakeven = np.flatnonzero(path_returns >= 0.0)
    record["time_to_breakeven_quality"] = (
        -float(
            (int(path_times[int(breakeven[0])]) - int(decision_ns))
            / 60_000_000_000
        )
        if len(breakeven)
        else -16.0
    )

    for name, favorable, adverse in BARRIERS:
        if not len(path_returns):
            record[f"barrier_{name}"] = float("nan")
            continue
        favorable_hits = np.flatnonzero(path_returns >= favorable)
        adverse_hits = np.flatnonzero(
            (path_returns <= adverse) | (path_bids <= 0.0)
        )
        favorable_index = int(favorable_hits[0]) if len(favorable_hits) else None
        adverse_index = int(adverse_hits[0]) if len(adverse_hits) else None
        if favorable_index is not None and (
            adverse_index is None
            or int(path_times[favorable_index]) < int(path_times[adverse_index])
        ):
            outcome = 1.0
        elif adverse_index is not None:
            outcome = -1.0
        else:
            outcome = 0.0
        record[f"barrier_{name}"] = outcome
    return record


def midrank_percentiles(values: Sequence[float]) -> np.ndarray:
    """Return frozen within-decision percentiles; missing values score zero."""

    series = pd.Series(np.asarray(values, dtype=np.float64))
    finite = np.isfinite(series.to_numpy(dtype=float))
    result = np.zeros(len(series), dtype=np.float64)
    count = int(finite.sum())
    if count == 0:
        return result
    if count == 1:
        result[np.flatnonzero(finite)[0]] = 0.5
        return result
    ranks = series[finite].rank(method="average").to_numpy(dtype=float)
    result[finite] = (ranks - 1.0) / float(count - 1)
    return result


def add_primary_utility(frame: pd.DataFrame) -> pd.DataFrame:
    """Add frozen percentile components and primary utility by decision."""

    required = {"session", "decision_time_ns", *RAW_TARGET_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"target frame missing columns: {missing}")
    out = frame.copy()
    keys = ["session", "decision_time_ns"]
    percentile_columns: list[str] = []
    for column in RAW_TARGET_COLUMNS:
        percentile = f"{column}_pct"
        out[percentile] = out.groupby(keys, sort=False)[column].transform(
            lambda values: midrank_percentiles(values.to_numpy(dtype=float))
        )
        percentile_columns.append(percentile)
    out["return_family"] = out[
        [f"{column}_pct" for column in RETURN_COLUMNS]
    ].mean(axis=1)
    out["path_family"] = out[
        [f"{column}_pct" for column in PATH_COLUMNS]
    ].mean(axis=1)
    out["breakeven_family"] = out[
        [f"{column}_pct" for column in BREAKEVEN_COLUMNS]
    ].mean(axis=1)
    out["barrier_family"] = out[
        [f"{column}_pct" for column in BARRIER_COLUMNS]
    ].mean(axis=1)
    out["primary_utility"] = out[list(UTILITY_COMPONENT_COLUMNS)].mean(axis=1)
    if not np.isfinite(out["primary_utility"]).all():
        raise ValueError("primary utility is nonfinite")
    if (
        float(out["primary_utility"].min()) < 0.0
        or float(out["primary_utility"].max()) > 1.0
    ):
        raise ValueError("primary utility escaped [0, 1]")
    return out


def candidate_order(frame: pd.DataFrame) -> np.ndarray:
    """Return positions under the frozen selector tie-break."""

    required = {
        "offset",
        "strike_index",
        "right",
        "contract_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"tie-break frame missing columns: {missing}")
    ordering = pd.DataFrame(
        {
            "position": np.arange(len(frame), dtype=np.int64),
            "abs_offset": np.abs(frame["offset"].to_numpy(dtype=float)),
            "strike_index": frame["strike_index"].to_numpy(dtype=np.int64),
            "right_order": np.where(
                frame["right"].astype(str).to_numpy() == "C", 0, 1
            ),
            "contract_id": frame["contract_id"].astype(str).to_numpy(),
        }
    )
    return ordering.sort_values(
        ["abs_offset", "strike_index", "right_order", "contract_id"],
        kind="mergesort",
    )["position"].to_numpy(dtype=np.int64)


def select_position(frame: pd.DataFrame, scores: Sequence[float]) -> int:
    """Select one candidate by score and the frozen deterministic tie-break."""

    score_values = np.asarray(scores, dtype=np.float64)
    if score_values.shape != (len(frame),):
        raise ValueError("selector score axis does not match risk set")
    finite = np.isfinite(score_values)
    if not finite.any():
        raise ValueError("risk set has no finite selector score")
    maximum = float(np.max(score_values[finite]))
    tied = np.isclose(score_values, maximum, rtol=0.0, atol=1e-12) & finite
    for position in candidate_order(frame):
        if tied[int(position)]:
            return int(position)
    raise AssertionError("finite score maximum was not selectable")


def deterministic_group_sample(
    frame: pd.DataFrame,
    *,
    maximum_candidates: int,
    fold_id: str,
) -> pd.DataFrame:
    """Deterministically cap training data without splitting decisions."""

    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")
    keys = ["session", "decision_time_ns"]
    counts = (
        frame.groupby(keys, sort=False)
        .size()
        .rename("candidate_count")
        .reset_index()
    )
    if int(counts["candidate_count"].sum()) <= int(maximum_candidates):
        return frame.copy()
    counts["priority"] = [
        hashlib.sha256(
            f"{SCHEMA_VERSION}|{fold_id}|{session}|{int(decision)}".encode()
        ).hexdigest()
        for session, decision in zip(
            counts["session"], counts["decision_time_ns"]
        )
    ]
    counts = counts.sort_values("priority", kind="mergesort")
    counts["cumulative"] = counts["candidate_count"].cumsum()
    selected = counts[counts["cumulative"] <= int(maximum_candidates)]
    if selected.empty:
        selected = counts.iloc[:1]
    selected_keys = pd.MultiIndex.from_frame(selected[keys])
    frame_keys = pd.MultiIndex.from_frame(frame[keys])
    result = frame.loc[frame_keys.isin(selected_keys)].copy()
    if len(result) > int(maximum_candidates):
        raise AssertionError("complete-group sample exceeded cap")
    observed_counts = result.groupby(keys, sort=False).size()
    expected_counts = (
        frame.groupby(keys, sort=False).size().loc[observed_counts.index]
    )
    if not observed_counts.equals(expected_counts):
        raise AssertionError("complete decision group was split by sampling")
    return result


def group_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equalize aggregate decision and session training weight."""

    keys = ["session", "decision_time_ns"]
    candidate_counts = frame.groupby(keys, sort=False)["contract_id"].transform(
        "size"
    ).to_numpy(dtype=float)
    decision_counts = (
        frame[keys]
        .drop_duplicates()
        .groupby("session", sort=False)["decision_time_ns"]
        .count()
    )
    per_session = frame["session"].map(decision_counts).to_numpy(dtype=float)
    raw = 1.0 / (candidate_counts * per_session)
    weights = raw / float(np.mean(raw))
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("group-balanced weights are invalid")
    session_totals = pd.Series(weights).groupby(
        frame["session"].reset_index(drop=True)
    ).sum()
    if float(session_totals.max() - session_totals.min()) > 1e-8:
        raise AssertionError("session aggregate training weights are unequal")
    return weights


def permute_targets_within_decision(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Destroy feature/utility alignment while preserving each risk-set target."""

    target = frame["primary_utility"].to_numpy(dtype=np.float64)
    shuffled = target.copy()
    group_receipts: list[dict[str, Any]] = []
    keys = ["session", "decision_time_ns"]
    for identity, indexes in frame.groupby(keys, sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype=np.int64)
        local_seed = int.from_bytes(
            hashlib.sha256(
                f"{SCHEMA_VERSION}|Selector-D1-v1|{seed}|{identity}".encode()
            ).digest()[:8],
            "little",
            signed=False,
        )
        rng = np.random.Generator(np.random.PCG64DXSM(local_seed))
        permutation = rng.permutation(len(positions))
        shuffled[positions] = target[positions[permutation]]
        group_receipts.append(
            {
                "session": str(identity[0]),
                "decision_time_ns": int(identity[1]),
                "candidate_count": int(len(positions)),
                "source_target_hash": numeric_hash(target[positions]),
                "permutation_hash": numeric_hash(permutation.astype(np.int64)),
                "destination_target_hash": numeric_hash(shuffled[positions]),
                "target_multiset_preserved": bool(
                    np.array_equal(
                        np.sort(target[positions]),
                        np.sort(shuffled[positions]),
                    )
                ),
            }
        )
    if not all(item["target_multiset_preserved"] for item in group_receipts):
        raise AssertionError("Selector-D1 target multiset changed")
    return shuffled, {
        "schema_version": SCHEMA_VERSION,
        "selector_d1_version": "Selector-D1-v1",
        "seed": int(seed),
        "decision_groups": int(len(group_receipts)),
        "source_target_hash": numeric_hash(target),
        "destination_target_hash": numeric_hash(shuffled),
        "group_receipt_root_hash": stable_hash(group_receipts),
        "groups": group_receipts,
    }


@dataclass(frozen=True)
class FitReceipt:
    model_name: str
    fold_id: str
    seed: int
    feature_names: tuple[str, ...]
    training_rows: int
    training_decisions: int
    training_sessions: int
    feature_hash: str
    target_hash: str
    weight_hash: str
    model_config: Mapping[str, Any]


def fit_selector(
    frame: pd.DataFrame,
    *,
    model_name: str,
    feature_names: Sequence[str],
    seed: int,
    fold_id: str,
    noise_model: DivergenceNoiseModel,
    target_override: np.ndarray | None = None,
) -> tuple[HistGradientBoostingRegressor, FitReceipt]:
    """Fit one frozen HGB selector on complete decision groups."""

    names = tuple(str(item) for item in feature_names)
    if model_name == "M0" and names != M0_FEATURES:
        raise ValueError("M0 feature contract changed")
    if model_name == "M1" and names != M1_FEATURES:
        raise ValueError("M1 feature contract changed")
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"selector features missing: {missing}")
    x_frame = frame.loc[:, list(names)].copy()
    x_frame["abs_offset"] = np.abs(frame["offset"].to_numpy(dtype=float))
    noisy = noise_model.inject_dataframe(
        x_frame,
        feature_columns=names,
        seed=int(seed),
        scale=1.0,
    )
    x = noisy.loc[:, list(names)].to_numpy(dtype=np.float64)
    target = (
        frame["primary_utility"].to_numpy(dtype=np.float64)
        if target_override is None
        else np.asarray(target_override, dtype=np.float64)
    )
    if target.shape != (len(frame),) or not np.isfinite(target).all():
        raise ValueError("selector training target is invalid")
    weights = group_balanced_weights(frame.reset_index(drop=True))
    model = HistGradientBoostingRegressor(
        **MODEL_CONFIG,
        random_state=int(seed),
    )
    model.fit(x, target, sample_weight=weights)
    receipt = FitReceipt(
        model_name=str(model_name),
        fold_id=str(fold_id),
        seed=int(seed),
        feature_names=names,
        training_rows=int(len(frame)),
        training_decisions=int(
            frame[["session", "decision_time_ns"]].drop_duplicates().shape[0]
        ),
        training_sessions=int(frame["session"].nunique()),
        feature_hash=numeric_hash(x),
        target_hash=numeric_hash(target),
        weight_hash=numeric_hash(weights),
        model_config={**MODEL_CONFIG, "random_state": int(seed)},
    )
    return model, receipt


def score_selector(
    model: HistGradientBoostingRegressor,
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    noise_model: DivergenceNoiseModel | None = None,
    noise_seed: int = 0,
    noise_scale: float = 0.0,
) -> np.ndarray:
    names = tuple(feature_names)
    source = frame.loc[:, list(names)].copy()
    if noise_model is not None and float(noise_scale) != 0.0:
        source["abs_offset"] = np.abs(frame["offset"].to_numpy(dtype=float))
        source = noise_model.inject_dataframe(
            source,
            feature_columns=names,
            seed=int(noise_seed),
            scale=float(noise_scale),
        )
    values = source.loc[:, list(names)].to_numpy(dtype=np.float64)
    return np.asarray(model.predict(values), dtype=np.float64)


def block_schedule(
    sessions: Sequence[str],
    folds: Sequence[str],
    *,
    replicates: int = 20_000,
    block_size: int = 5,
    seed: int = 2_026_072_801,
) -> np.ndarray:
    """Generate one synchronized circular moving-block schedule per fold."""

    sessions = tuple(str(item) for item in sessions)
    folds = tuple(str(item) for item in folds)
    if len(sessions) != len(folds) or len(set(sessions)) != len(sessions):
        raise ValueError("session/fold inference grid is invalid")
    fold_order = tuple(dict.fromkeys(folds))
    fold_arrays = tuple(
        np.asarray(
            [index for index, value in enumerate(folds) if value == fold],
            dtype=np.uint32,
        )
        for fold in fold_order
    )
    children = np.random.SeedSequence(int(seed)).spawn(int(replicates))
    out = np.empty((int(replicates), len(sessions)), dtype=np.uint32)
    for replicate, child in enumerate(children):
        rng = np.random.Generator(np.random.PCG64DXSM(child))
        cursor = 0
        for indexes in fold_arrays:
            count = len(indexes)
            starts = rng.integers(
                0,
                count,
                size=math.ceil(count / int(block_size)),
            )
            local = np.concatenate(
                [
                    (int(start) + np.arange(int(block_size))) % count
                    for start in starts
                ]
            )[:count]
            out[replicate, cursor : cursor + count] = indexes[local]
            cursor += count
    return out


def simultaneous_inference(
    effects: Mapping[str, Sequence[float]],
    *,
    schedule: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute synchronized max-absolute-T intervals and one-sided FWER tests."""

    names = tuple(effects)
    matrix = np.asarray([effects[name] for name in names], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("inference effects must form a finite matrix")
    if schedule.ndim != 2 or schedule.shape[1] != matrix.shape[1]:
        raise ValueError("bootstrap schedule does not match effects")
    observed = matrix.mean(axis=1)
    sampled = np.empty((len(schedule), len(names)), dtype=np.float64)
    chunk = 256
    for start in range(0, len(schedule), chunk):
        stop = min(start + chunk, len(schedule))
        selected = np.take(matrix, schedule[start:stop], axis=1)
        sampled[start:stop] = selected.mean(axis=2).T
    standard_error = sampled.std(axis=0, ddof=1)
    if np.any(~np.isfinite(standard_error)) or np.any(standard_error <= 0.0):
        raise ValueError("nonpositive bootstrap standard error")
    standardized_error = (sampled - observed[None, :]) / standard_error[None, :]
    max_abs = np.max(np.abs(standardized_error), axis=1)
    critical = float(np.quantile(max_abs, 1.0 - float(alpha), method="higher"))
    lower = observed - critical * standard_error
    upper = observed + critical * standard_error

    centered = matrix - observed[:, None]
    null_sampled = np.empty_like(sampled)
    for start in range(0, len(schedule), chunk):
        stop = min(start + chunk, len(schedule))
        selected = np.take(centered, schedule[start:stop], axis=1)
        null_sampled[start:stop] = selected.mean(axis=2).T
    null_t = null_sampled / standard_error[None, :]
    max_null = np.max(null_t, axis=1)
    observed_t = observed / standard_error
    counts = np.asarray(
        [(max_null >= value).sum() for value in observed_t],
        dtype=np.int64,
    )
    p_fwer = (1.0 + counts) / (len(max_null) + 1.0)
    rows = {}
    for index, name in enumerate(names):
        rows[name] = {
            "observed_mean": float(observed[index]),
            "standard_error": float(standard_error[index]),
            "adjusted_ci_95": [float(lower[index]), float(upper[index])],
            "observed_t": float(observed_t[index]),
            "p_fwer_one_sided": float(p_fwer[index]),
            "hard_pass_p_fwer": bool(p_fwer[index] <= float(alpha)),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "synchronized_five_session_moving_block_max_abs_t",
        "alpha": float(alpha),
        "replicates": int(len(schedule)),
        "family": list(names),
        "critical_max_abs_t": critical,
        "schedule_hash": numeric_hash(schedule),
        "max_null_hash": numeric_hash(max_null),
        "rows": rows,
    }


def session_effects(
    frame: pd.DataFrame,
    *,
    selected_column: str,
    baseline_column: str,
) -> pd.Series:
    """Aggregate one selected-minus-baseline utility effect per session."""

    required = {
        "session",
        "decision_time_ns",
        selected_column,
        baseline_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"effect frame missing columns: {missing}")
    opportunity = (
        frame.assign(
            _effect=(
                frame[selected_column].to_numpy(dtype=float)
                - frame[baseline_column].to_numpy(dtype=float)
            )
        )
        .groupby(["session", "decision_time_ns"], sort=False)["_effect"]
        .first()
    )
    return opportunity.groupby(level=0).mean().sort_index()
