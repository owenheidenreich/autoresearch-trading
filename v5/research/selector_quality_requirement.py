"""Outcome-conditioned selector-quality requirements for an existing P&L table.

No score in this module is a predictor.  A later score contains that same
session's realised P&L, so the chronological comparison below is an oracle-law
stability check, not genuine outcome-blind out-of-fold evidence.  The module
fits no model, tests no feature, spends no alpha, and adopts nothing.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


REQUIRED_COLUMNS = ("session", "start", "offset", "pnl", "why")
EXPECTED_ROWS = 10_078
EXPECTED_SESSIONS = 1_011
EXPECTED_SESSION_MIN = "2022-06-01"
EXPECTED_SESSION_MAX = "2026-07-30"
RESERVATION_START = "2026-08-06"
CALIBRATION_END = "2025-07-31"
LATER_START = "2025-08-01"
EXPECTED_PRIMARY_CALIBRATION_SESSIONS = 768
EXPECTED_PRIMARY_LATER_SESSIONS = 243
EXPECTED_REASON_COUNTS = {"horizon": 5_524, "stop": 4_554}
EXPECTED_CELL_COUNTS = {
    ("09:35", 0.0): 1_011,
    ("09:35", 10.0): 1_011,
    ("09:35", 25.0): 502,
    ("11:30", 0.0): 1_011,
    ("11:30", 10.0): 1_011,
    ("11:30", 25.0): 495,
    ("13:30", 0.0): 1_011,
    ("13:30", 10.0): 1_010,
    ("13:30", 25.0): 499,
    ("15:00", 0.0): 1_010,
    ("15:00", 10.0): 1_010,
    ("15:00", 25.0): 497,
}

PRIMARY_CELL = ("09:35", 0.0)
DECLARED_RHOS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
SELECTION_RATES = (0.05, 0.10, 0.20, 0.50, 1.00)


class SelectorRequirementError(RuntimeError):
    """The requirements study cannot preserve its declared measurement law."""


@dataclass(frozen=True)
class StudyConfig:
    """Deterministic analysis constants; tests may request smaller workloads."""

    calibration_end: str | None = CALIBRATION_END
    later_start: str | None = LATER_START
    train_share: float = 0.50  # used only when both date boundaries are None
    selector_draws: int = 2_000
    bootstrap_reps: int = 5_000
    seed: int = 20_260_823
    declared_rhos: tuple[float, ...] = DECLARED_RHOS
    selection_rates: tuple[float, ...] = SELECTION_RATES
    inverse_step: float = 0.005
    inverse_max: float = 0.50
    confidence: float = 0.95

    def validate(self) -> None:
        if (self.calibration_end is None) != (self.later_start is None):
            raise SelectorRequirementError("both chronological date boundaries are required")
        if self.calibration_end is not None:
            end = pd.Timestamp(self.calibration_end)
            start = pd.Timestamp(self.later_start)
            if end >= start:
                raise SelectorRequirementError("chronological boundaries overlap")
        if not 0.1 <= self.train_share <= 0.9:
            raise SelectorRequirementError("train_share must be in [0.1, 0.9]")
        if self.selector_draws < 32:
            raise SelectorRequirementError("selector_draws must be at least 32")
        if self.bootstrap_reps < 100:
            raise SelectorRequirementError("bootstrap_reps must be at least 100")
        if not 0.0 < self.inverse_step <= 0.10:
            raise SelectorRequirementError("inverse_step must be in (0, 0.10]")
        if not 0.0 < self.inverse_max <= 1.0:
            raise SelectorRequirementError("inverse_max must be in (0, 1]")
        if not math.isclose(self.confidence, 0.95):
            raise SelectorRequirementError("this receipt schema fixes confidence at 0.95")
        if tuple(sorted(set(self.declared_rhos))) != self.declared_rhos:
            raise SelectorRequirementError("declared_rhos must be unique and sorted")
        if tuple(sorted(set(self.selection_rates))) != self.selection_rates:
            raise SelectorRequirementError("selection_rates must be unique and sorted")
        if any(not 0.0 <= rho <= 1.0 for rho in self.declared_rhos):
            raise SelectorRequirementError("declared rho outside [0, 1]")
        if any(not 0.0 < rate <= 1.0 for rate in self.selection_rates):
            raise SelectorRequirementError("selection rate outside (0, 1]")


@dataclass
class StudyResult:
    input_manifest: pd.DataFrame
    requirement_curve: pd.DataFrame
    correlation_diagnostics: pd.DataFrame
    inverse_requirement: pd.DataFrame
    inverse_trace: pd.DataFrame
    sanity_floor: dict[str, Any]
    claim_audit: dict[str, Any]
    metadata: dict[str, Any]
    readable_tables: str


@dataclass
class _Selection:
    raw_weights: np.ndarray
    jackknife_weights: np.ndarray
    world_means: np.ndarray
    world_jackknife_means: np.ndarray
    world_counts: np.ndarray
    threshold: np.ndarray
    worlds_with_fewer_than_two: int
    jackknife_valid_worlds: int


@dataclass
class _CellContext:
    start: str
    offset: float
    sessions: np.ndarray
    pnl: np.ndarray
    train_n: int
    train_z: np.ndarray
    later_z: np.ndarray
    train_noise: np.ndarray
    later_noise: np.ndarray
    train_bootstrap: np.ndarray
    later_bootstrap: np.ndarray
    train_mean: float
    train_std: float

    @property
    def later_n(self) -> int:
        return len(self.pnl) - self.train_n


def _seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _percentile_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise SelectorRequirementError("interval input is empty or non-finite")
    alpha = 1.0 - confidence
    return (
        float(np.percentile(values, 100.0 * alpha / 2.0)),
        float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0))),
    )


def validate_pnl_table(frame: pd.DataFrame, *, strict_population: bool = True) -> pd.DataFrame:
    """Validate and chronology-sort the immutable, already-computed P&L table."""

    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise SelectorRequirementError(
            f"P&L schema drift: expected {REQUIRED_COLUMNS}, found {tuple(frame.columns)}"
        )
    result = frame.copy()
    if result.empty:
        raise SelectorRequirementError("P&L table is empty")
    if result[list(REQUIRED_COLUMNS)].isna().any().any():
        raise SelectorRequirementError("P&L table contains missing values")
    numeric = result[["offset", "pnl"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise SelectorRequirementError("P&L table contains non-finite numeric values")
    parsed = pd.to_datetime(result["session"], format="%Y-%m-%d", errors="raise")
    if (parsed >= pd.Timestamp(RESERVATION_START)).any():
        raise SelectorRequirementError("P&L table opens a confirmation-reserved session")
    if result.duplicated(["session", "start", "offset"]).any():
        raise SelectorRequirementError("P&L table has duplicate session/cell keys")
    result["session"] = parsed.dt.strftime("%Y-%m-%d")
    result["start"] = result["start"].astype(str)
    result["offset"] = result["offset"].astype(float)
    result["pnl"] = result["pnl"].astype(float)
    result["why"] = result["why"].astype(str)
    result = result.sort_values(["start", "offset", "session"], kind="stable").reset_index(
        drop=True
    )

    if strict_population:
        if len(result) != EXPECTED_ROWS:
            raise SelectorRequirementError(
                f"P&L row-count drift: expected {EXPECTED_ROWS}, found {len(result)}"
            )
        sessions = sorted(result["session"].unique())
        if len(sessions) != EXPECTED_SESSIONS:
            raise SelectorRequirementError(
                f"session-count drift: expected {EXPECTED_SESSIONS}, found {len(sessions)}"
            )
        if sessions[0] != EXPECTED_SESSION_MIN or sessions[-1] != EXPECTED_SESSION_MAX:
            raise SelectorRequirementError("P&L session range drift")
        reasons = {str(key): int(value) for key, value in result["why"].value_counts().items()}
        if reasons != EXPECTED_REASON_COUNTS:
            raise SelectorRequirementError(
                f"P&L exit-reason drift: expected {EXPECTED_REASON_COUNTS}, found {reasons}"
            )
        cells = {
            (str(start), float(offset)): int(len(block))
            for (start, offset), block in result.groupby(["start", "offset"], sort=True)
        }
        if cells != EXPECTED_CELL_COUNTS:
            raise SelectorRequirementError("P&L cell population drift")
    return result


def load_pnl_table(path: Path, *, strict_population: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise SelectorRequirementError(f"P&L table is missing: {path}")
    return validate_pnl_table(pd.read_csv(path), strict_population=strict_population)


def _split_index(sessions: np.ndarray, config: StudyConfig) -> int:
    sessions = np.asarray(sessions, dtype=str)
    if config.calibration_end is None:
        train_n = int(math.floor(config.train_share * len(sessions)))
    else:
        earlier = sessions <= str(config.calibration_end)
        later = sessions >= str(config.later_start)
        if not np.all(earlier | later):
            raise SelectorRequirementError("a session falls between split boundaries")
        train_n = int(earlier.sum())
        if not np.all(earlier[:train_n]) or not np.all(later[train_n:]):
            raise SelectorRequirementError("chronological split is not a prefix/suffix")
    if train_n < 20 or len(sessions) - train_n < 20:
        raise SelectorRequirementError("chronological split has fewer than 20 sessions")
    return train_n


def _bootstrap_counts(rng: np.random.Generator, reps: int, n: int) -> np.ndarray:
    if n < 2:
        raise SelectorRequirementError("session bootstrap needs at least two sessions")
    return rng.multinomial(n, np.full(n, 1.0 / n), size=reps).astype(float, copy=False)


def _normal_noise(rng: np.random.Generator, draws: int, n: int) -> np.ndarray:
    noise = rng.normal(size=(draws, n))
    if not np.isfinite(noise).all():
        raise SelectorRequirementError("selector noise is non-finite")
    return noise


def construct_scores(z: np.ndarray, noise: np.ndarray, rho: float) -> np.ndarray:
    """Literal nominal-rho Gaussian score; achieved sample correlation is measured."""

    z = np.asarray(z, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if not 0.0 <= rho <= 1.0:
        raise SelectorRequirementError("rho must be in [0, 1]")
    if noise.ndim != 2 or noise.shape[1] != len(z):
        raise SelectorRequirementError("noise and outcome dimensions disagree")
    if not np.isfinite(z).all() or not np.isfinite(noise).all():
        raise SelectorRequirementError("score inputs are non-finite")
    score = rho * z[None, :] + math.sqrt(max(0.0, 1.0 - rho * rho)) * noise
    if not np.isfinite(score).all():
        raise SelectorRequirementError("constructed scores are non-finite")
    return score


def _top_k_mask(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= k <= scores.shape[1]:
        raise SelectorRequirementError("top-k selection is outside its partition")
    if k == scores.shape[1]:
        return np.ones_like(scores, dtype=bool), np.full(scores.shape[0], -np.inf)
    indices = np.argpartition(scores, scores.shape[1] - k, axis=1)[:, -k:]
    mask = np.zeros_like(scores, dtype=bool)
    mask[np.arange(scores.shape[0])[:, None], indices] = True
    threshold = np.min(np.take_along_axis(scores, indices, axis=1), axis=1)
    return mask, threshold


def _drop_best(mask: np.ndarray, pnl: np.ndarray) -> np.ndarray:
    counts = mask.sum(axis=1)
    if np.any(counts < 1):
        raise SelectorRequirementError("a selector world chose no sessions")
    best = np.argmax(np.where(mask, pnl[None, :], -np.inf), axis=1)
    result = mask.copy()
    result[np.arange(len(result)), best] = False
    return result


def _row_dot(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    values = np.einsum("ij,j->i", matrix, vector, optimize=False)
    if not np.isfinite(values).all():
        raise SelectorRequirementError("numeric row reduction became non-finite")
    return values


def _selection(
    train_scores: np.ndarray,
    evaluation_scores: np.ndarray,
    evaluation_pnl: np.ndarray,
    rate: float,
    *,
    in_sample: bool,
) -> _Selection:
    k = min(train_scores.shape[1], max(2, int(round(rate * train_scores.shape[1]))))
    train_mask, threshold = _top_k_mask(train_scores, k)
    if in_sample:
        if evaluation_scores.shape != train_scores.shape:
            raise SelectorRequirementError("in-sample score shape drift")
        mask = train_mask
    elif rate == 1.0:
        mask = np.ones_like(evaluation_scores, dtype=bool)
    else:
        mask = evaluation_scores >= threshold[:, None]
    counts = mask.sum(axis=1)
    if np.any(counts < 1):
        raise SelectorRequirementError(
            f"selection rate {rate:.3f} produced an empty selector world"
        )
    jack = _drop_best(mask, evaluation_pnl)
    raw_sum = _row_dot(mask, evaluation_pnl)
    jack_sum = _row_dot(jack, evaluation_pnl)
    valid_jack = counts >= 2
    return _Selection(
        raw_weights=mask.mean(axis=0),
        jackknife_weights=jack.mean(axis=0),
        world_means=raw_sum / counts,
        world_jackknife_means=jack_sum[valid_jack] / (counts[valid_jack] - 1),
        world_counts=counts.astype(float),
        threshold=threshold,
        worlds_with_fewer_than_two=int((~valid_jack).sum()),
        jackknife_valid_worlds=int(valid_jack.sum()),
    )


def _weighted_bootstrap(
    pnl: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    pnl = np.asarray(pnl, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denominator = float(weights.sum())
    if denominator <= 0.0 or len(weights) != len(pnl):
        raise SelectorRequirementError("weighted estimand has no selected sessions")
    numerator_values = weights * pnl
    profitable_values = weights * (pnl > 0.0)
    boot_denominator = _row_dot(counts, weights)
    if np.any(boot_denominator <= 0.0):
        raise SelectorRequirementError("session bootstrap produced an empty selected sample")
    boot_ticket = _row_dot(counts, numerator_values) / boot_denominator
    boot_calendar = _row_dot(counts, numerator_values) / len(pnl)
    boot_rate = boot_denominator / len(pnl)
    boot_profitable = _row_dot(counts, profitable_values) / boot_denominator
    ticket_lo, ticket_hi = _percentile_interval(boot_ticket, confidence)
    calendar_lo, calendar_hi = _percentile_interval(boot_calendar, confidence)
    rate_lo, rate_hi = _percentile_interval(boot_rate, confidence)
    profitable_lo, profitable_hi = _percentile_interval(boot_profitable, confidence)
    return {
        "mean_pnl_per_selected_ticket": float(numerator_values.sum() / denominator),
        "mean_pnl_ci_95_low": ticket_lo,
        "mean_pnl_ci_95_high": ticket_hi,
        "mean_pnl_per_calendar_session": float(numerator_values.sum() / len(pnl)),
        "calendar_pnl_ci_95_low": calendar_lo,
        "calendar_pnl_ci_95_high": calendar_hi,
        "selected_sessions_expected": denominator,
        "realized_selection_rate": float(denominator / len(pnl)),
        "selection_rate_ci_95_low": rate_lo,
        "selection_rate_ci_95_high": rate_hi,
        "profitable_share": float(profitable_values.sum() / denominator),
        "profitable_share_ci_95_low": profitable_lo,
        "profitable_share_ci_95_high": profitable_hi,
        "bootstrap_ticket_values": boot_ticket,
        "bootstrap_calendar_values": boot_calendar,
    }


def _correlations(scores: np.ndarray, pnl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(pnl, dtype=float)
    xc = x - x.mean()
    sc = scores - scores.mean(axis=1, keepdims=True)
    numerator = _row_dot(sc, xc)
    denominator = np.sqrt(np.sum(sc * sc, axis=1) * np.sum(xc * xc))
    pearson = numerator / denominator
    x_rank = rankdata(x, method="average")
    x_rank -= x_rank.mean()
    score_rank = rankdata(scores, method="average", axis=1)
    score_rank -= score_rank.mean(axis=1, keepdims=True)
    spearman = _row_dot(score_rank, x_rank) / np.sqrt(
        np.sum(score_rank * score_rank, axis=1) * np.sum(x_rank * x_rank)
    )
    if not np.isfinite(pearson).all() or not np.isfinite(spearman).all():
        raise SelectorRequirementError("correlation diagnostic became non-finite")
    return pearson, spearman


def _diagnostic_row(
    context: _CellContext,
    partition: str,
    rho: float,
    scores: np.ndarray,
    pnl: np.ndarray,
) -> dict[str, Any]:
    pearson, spearman = _correlations(scores, pnl)
    p_lo, p_hi = _percentile_interval(pearson)
    s_lo, s_hi = _percentile_interval(spearman)
    return {
        "start": context.start,
        "offset": context.offset,
        "partition": partition,
        "sessions": len(pnl),
        "nominal_gaussian_rho": rho,
        "achieved_pearson_mean": float(pearson.mean()),
        "achieved_pearson_median": float(np.median(pearson)),
        "achieved_pearson_selector_world_95_low": p_lo,
        "achieved_pearson_selector_world_95_high": p_hi,
        "achieved_spearman_mean": float(spearman.mean()),
        "achieved_spearman_median": float(np.median(spearman)),
        "achieved_spearman_selector_world_95_low": s_lo,
        "achieved_spearman_selector_world_95_high": s_hi,
        "interval_type": "fixed-session selector-world construction range; not a confidence interval",
        "constructed_score_uses_same_partition_outcome": True,
    }


def _make_context(block: pd.DataFrame, config: StudyConfig) -> _CellContext:
    block = block.sort_values("session", kind="stable")
    sessions = block["session"].to_numpy(str)
    pnl = block["pnl"].to_numpy(float)
    train_n = _split_index(sessions, config)
    train_pnl = pnl[:train_n]
    train_mean = float(train_pnl.mean())
    train_std = float(train_pnl.std(ddof=0))
    if not math.isfinite(train_std) or train_std <= 0.0:
        raise SelectorRequirementError("calibration P&L cannot be standardized")
    train_z = (train_pnl - train_mean) / train_std
    later_z = (pnl[train_n:] - train_mean) / train_std
    noise_rng = np.random.default_rng(_seed(config.seed, PRIMARY_CELL, "selector"))
    bootstrap_rng = np.random.default_rng(_seed(config.seed, PRIMARY_CELL, "bootstrap"))
    return _CellContext(
        start=PRIMARY_CELL[0],
        offset=PRIMARY_CELL[1],
        sessions=sessions,
        pnl=pnl,
        train_n=train_n,
        train_z=train_z,
        later_z=later_z,
        train_noise=_normal_noise(noise_rng, config.selector_draws, train_n),
        later_noise=_normal_noise(noise_rng, config.selector_draws, len(pnl) - train_n),
        train_bootstrap=_bootstrap_counts(bootstrap_rng, config.bootstrap_reps, train_n),
        later_bootstrap=_bootstrap_counts(
            bootstrap_rng, config.bootstrap_reps, len(pnl) - train_n
        ),
        train_mean=train_mean,
        train_std=train_std,
    )


def _curve_row(
    context: _CellContext,
    rho: float,
    rate: float,
    partition: str,
    selection: _Selection,
    confidence: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if partition == "calibration_in_sample_oracle":
        pnl = context.pnl[: context.train_n]
        bootstrap = context.train_bootstrap
    elif partition == "later_outcome_conditioned_oracle":
        pnl = context.pnl[context.train_n :]
        bootstrap = context.later_bootstrap
    else:
        raise SelectorRequirementError(f"unknown partition {partition}")
    raw = _weighted_bootstrap(pnl, selection.raw_weights, bootstrap, confidence)
    jack = _weighted_bootstrap(pnl, selection.jackknife_weights, bootstrap, confidence)
    raw_world_lo, raw_world_hi = _percentile_interval(selection.world_means)
    jack_world_lo, jack_world_hi = _percentile_interval(selection.world_jackknife_means)
    count_lo, count_hi = _percentile_interval(selection.world_counts)
    row = {
        "start": context.start,
        "offset": context.offset,
        "partition": partition,
        "sessions": len(pnl),
        "nominal_gaussian_rho": rho,
        "target_selection_rate": rate,
        "individually_executable_under_two_ticket_law": True,
        "ticket_dollar_eligibility": "UNKNOWN_ASK_NOT_IN_PNL_TABLE",
        "joint_cell_executability": "UNKNOWN_NOT_EVALUATED",
        "constructed_score_uses_same_partition_outcome": True,
        "predictive_signal_evidence": False,
        **{key: value for key, value in raw.items() if not key.startswith("bootstrap_")},
        "drop_best_mean_pnl_per_selected_ticket": jack["mean_pnl_per_selected_ticket"],
        "drop_best_mean_pnl_ci_95_low": jack["mean_pnl_ci_95_low"],
        "drop_best_mean_pnl_ci_95_high": jack["mean_pnl_ci_95_high"],
        "drop_best_mean_pnl_per_calendar_session": jack[
            "mean_pnl_per_calendar_session"
        ],
        "drop_best_calendar_pnl_ci_95_low": jack["calendar_pnl_ci_95_low"],
        "drop_best_calendar_pnl_ci_95_high": jack["calendar_pnl_ci_95_high"],
        "selector_world_mean_pnl_median": float(np.median(selection.world_means)),
        "selector_world_mean_pnl_95_low": raw_world_lo,
        "selector_world_mean_pnl_95_high": raw_world_hi,
        "selector_world_drop_best_mean_pnl_median": float(
            np.median(selection.world_jackknife_means)
        ),
        "selector_world_drop_best_mean_pnl_95_low": jack_world_lo,
        "selector_world_drop_best_mean_pnl_95_high": jack_world_hi,
        "selector_world_selected_count_median": float(np.median(selection.world_counts)),
        "selector_world_selected_count_95_low": count_lo,
        "selector_world_selected_count_95_high": count_hi,
        "selector_worlds_with_fewer_than_two_selected_sessions": (
            selection.worlds_with_fewer_than_two
        ),
        "jackknife_valid_selector_worlds": selection.jackknife_valid_worlds,
        "jackknife_empty_world_share": float(
            selection.worlds_with_fewer_than_two / len(selection.world_counts)
        ),
        "inferentially_sparse_below_30_selected_sessions": bool(
            np.median(selection.world_counts) < 30
        ),
        "tickets_per_year_expected": raw["realized_selection_rate"] * 252.0,
        "session_interval_conditions_on_fixed_threshold_ensemble": True,
    }
    return row, {
        "raw": raw["bootstrap_ticket_values"],
        "jackknife": jack["bootstrap_ticket_values"],
    }


def _sanity_row(
    context: _CellContext,
    partition: str,
    rate: float,
    selection: _Selection,
    seed: int,
) -> dict[str, Any]:
    if partition == "calibration_in_sample_oracle":
        pnl = context.pnl[: context.train_n]
        counts = context.train_bootstrap
        expected_rate = max(2, int(round(rate * context.train_n))) / context.train_n
    else:
        pnl = context.pnl[context.train_n :]
        counts = context.later_bootstrap
        expected_rate = max(2, int(round(rate * context.train_n))) / context.train_n
    selected = _weighted_bootstrap(pnl, selection.raw_weights, counts, 0.95)
    baseline = _weighted_bootstrap(pnl, np.ones(len(pnl)), counts, 0.95)
    difference = selected["bootstrap_ticket_values"] - baseline["bootstrap_ticket_values"]
    lo, hi = _percentile_interval(difference)
    point = selected["mean_pnl_per_selected_ticket"] - baseline[
        "mean_pnl_per_selected_ticket"
    ]
    world_difference = selection.world_means - float(pnl.mean())
    mc_se = float(world_difference.std(ddof=1) / math.sqrt(len(world_difference)))
    joint_rng = np.random.default_rng(
        _seed(seed, context.start, context.offset, partition, rate, "null-joint")
    )
    joint_difference = difference + joint_rng.normal(0.0, mc_se, size=len(difference))
    joint_lo, joint_hi = _percentile_interval(joint_difference)
    rate_world = selection.world_counts / len(pnl)
    rate_mc_se = float(rate_world.std(ddof=1) / math.sqrt(len(rate_world)))
    rate_gap = float(rate_world.mean() - expected_rate)
    if rate == 1.0:
        economic_mc_pass = bool(abs(point) <= 1e-12)
        rate_mc_pass = bool(abs(rate_gap) <= 1e-12)
    else:
        economic_mc_pass = bool(abs(point) <= 4.0 * mc_se)
        rate_mc_pass = bool(abs(rate_gap) <= 4.0 * rate_mc_se + 1.0 / len(pnl))
    return {
        "start": context.start,
        "offset": context.offset,
        "partition": partition,
        "target_selection_rate": rate,
        "selected_minus_unselected_mean_pnl": point,
        "difference_session_bootstrap_ci_95_low": lo,
        "difference_session_bootstrap_ci_95_high": hi,
        "difference_joint_session_and_selector_mc_ci_95_low": joint_lo,
        "difference_joint_session_and_selector_mc_ci_95_high": joint_hi,
        "selector_world_monte_carlo_se": mc_se,
        "expected_rate_from_calibration_top_k": expected_rate,
        "achieved_rate": float(rate_world.mean()),
        "rate_gap": rate_gap,
        "rate_monte_carlo_se": rate_mc_se,
        "economic_session_interval_contains_zero": bool(lo <= 0.0 <= hi),
        "economic_joint_interval_contains_zero": bool(joint_lo <= 0.0 <= joint_hi),
        "economic_monte_carlo_pass": economic_mc_pass,
        "rate_monte_carlo_pass": rate_mc_pass,
        "passes": bool(joint_lo <= 0.0 <= joint_hi and economic_mc_pass and rate_mc_pass),
    }


def _inverse_rhos(config: StudyConfig) -> tuple[float, ...]:
    grid = np.arange(0.0, config.inverse_max + config.inverse_step / 2.0, config.inverse_step)
    values = sorted(
        {
            *(round(float(value), 10) for value in grid if value <= config.inverse_max),
            *(rho for rho in config.declared_rhos if rho <= config.inverse_max),
        }
    )
    return tuple(values)


def _matrix_bootstrap(
    pnl: np.ndarray, weights: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = counts @ weights
        numerator = counts @ (weights * pnl[:, None])
        values = numerator / denominator
    if np.any(denominator <= 0.0) or not np.isfinite(values).all():
        raise SelectorRequirementError("inverse bootstrap has empty or non-finite cells")
    return values


def _inverse_for_context(
    context: _CellContext, config: StudyConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    later_pnl = context.pnl[context.train_n :]
    descriptors: list[dict[str, Any]] = []
    raw_columns: list[np.ndarray] = []
    jack_columns: list[np.ndarray] = []
    for rho in _inverse_rhos(config):
        train_scores = construct_scores(context.train_z, context.train_noise, rho)
        later_scores = construct_scores(context.later_z, context.later_noise, rho)
        pearson, spearman = _correlations(later_scores, later_pnl)
        for rate in config.selection_rates:
            selection = _selection(
                train_scores, later_scores, later_pnl, rate, in_sample=False
            )
            raw_columns.append(selection.raw_weights)
            jack_columns.append(selection.jackknife_weights)
            raw_denominator = float(selection.raw_weights.sum())
            jack_denominator = float(selection.jackknife_weights.sum())
            descriptors.append(
                {
                    "start": context.start,
                    "offset": context.offset,
                    "target_selection_rate": rate,
                    "nominal_gaussian_rho": rho,
                    "raw_mean_pnl": float(
                        np.dot(selection.raw_weights, later_pnl) / raw_denominator
                    ),
                    "drop_best_mean_pnl": float(
                        np.dot(selection.jackknife_weights, later_pnl) / jack_denominator
                    ),
                    "drop_best_selector_world_5th_percentile": float(
                        np.percentile(selection.world_jackknife_means, 5.0)
                    ),
                    "realized_selection_rate": raw_denominator / len(later_pnl),
                    "achieved_pearson_median": float(np.median(pearson)),
                    "achieved_spearman_median": float(np.median(spearman)),
                }
            )
    raw_weights = np.column_stack(raw_columns)
    jack_weights = np.column_stack(jack_columns)
    raw_boot = _matrix_bootstrap(later_pnl, raw_weights, context.later_bootstrap)
    jack_boot = _matrix_bootstrap(later_pnl, jack_weights, context.later_bootstrap)
    jack_point = np.asarray([row["drop_best_mean_pnl"] for row in descriptors])
    standard_error = jack_boot.std(axis=0, ddof=1)
    if np.any(standard_error <= 0.0) or not np.isfinite(standard_error).all():
        raise SelectorRequirementError("inverse bootstrap standard error is invalid")
    downward_t = (jack_point[None, :] - jack_boot) / standard_error[None, :]
    simultaneous_critical = float(np.percentile(np.max(downward_t, axis=1), 95.0))
    simultaneous_lcb = jack_point - simultaneous_critical * standard_error
    pointwise_low = np.percentile(jack_boot, 5.0, axis=0)
    pointwise_high = np.percentile(jack_boot, 95.0, axis=0)
    raw_low = np.percentile(raw_boot, 5.0, axis=0)
    raw_high = np.percentile(raw_boot, 95.0, axis=0)
    for index, row in enumerate(descriptors):
        row.update(
            {
                "raw_session_bootstrap_5th_percentile": float(raw_low[index]),
                "raw_session_bootstrap_95th_percentile": float(raw_high[index]),
                "drop_best_session_bootstrap_5th_percentile": float(
                    pointwise_low[index]
                ),
                "drop_best_session_bootstrap_95th_percentile": float(
                    pointwise_high[index]
                ),
                "drop_best_simultaneous_one_sided_95_lcb": float(
                    simultaneous_lcb[index]
                ),
                "simultaneous_family": "all inverse rho x selection-rate cells",
                "clears_zero_after_drop_best": bool(simultaneous_lcb[index] > 0.0),
                "selector_world_5th_percentile_clears_zero": bool(
                    row["drop_best_selector_world_5th_percentile"] > 0.0
                ),
                "clears_session_and_selector_world_robustness": bool(
                    simultaneous_lcb[index] > 0.0
                    and row["drop_best_selector_world_5th_percentile"] > 0.0
                ),
                "constructed_score_uses_same_partition_outcome": True,
                "predictive_signal_evidence": False,
            }
        )

    summaries: list[dict[str, Any]] = []
    for rate in config.selection_rates:
        rows = [row for row in descriptors if row["target_selection_rate"] == rate]
        rows.sort(key=lambda row: row["nominal_gaussian_rho"])
        passes = [bool(row["clears_zero_after_drop_best"]) for row in rows]
        sustained = next(
            (index for index, passed in enumerate(passes) if passed and all(passes[index:])),
            None,
        )
        robust_passes = [
            bool(row["clears_session_and_selector_world_robustness"]) for row in rows
        ]
        robust_sustained = next(
            (
                index
                for index, passed in enumerate(robust_passes)
                if passed and all(robust_passes[index:])
            ),
            None,
        )
        base = {
            "start": context.start,
            "offset": context.offset,
            "target_selection_rate": rate,
            "grid_resolution": config.inverse_step,
            "grid_maximum": config.inverse_max,
            "individually_executable_under_two_ticket_law": True,
            "ticket_dollar_eligibility": "UNKNOWN_ASK_NOT_IN_PNL_TABLE",
            "simultaneous_across_rates_and_rhos": True,
            "constructed_score_uses_same_partition_outcome": True,
            "genuine_outcome_blind_requirement": "UNKNOWN",
            "predictive_signal_evidence": False,
        }
        if sustained is None:
            summary = {
                **base,
                "minimum_nominal_rho_grid": None,
                "previous_nominal_rho_grid": config.inverse_max,
                "clears_by_grid_maximum": False,
            }
        else:
            chosen = rows[sustained]
            previous = rows[sustained - 1] if sustained else None
            summary = {
                **base,
                "minimum_nominal_rho_grid": chosen["nominal_gaussian_rho"],
                "previous_nominal_rho_grid": (
                    previous["nominal_gaussian_rho"] if previous else None
                ),
                "clears_by_grid_maximum": True,
                "drop_best_mean_pnl_at_requirement": chosen["drop_best_mean_pnl"],
                "drop_best_simultaneous_95_lcb_at_requirement": chosen[
                    "drop_best_simultaneous_one_sided_95_lcb"
                ],
                "realized_selection_rate_at_requirement": chosen[
                    "realized_selection_rate"
                ],
                "achieved_pearson_median_at_requirement": chosen[
                    "achieved_pearson_median"
                ],
                "achieved_spearman_median_at_requirement": chosen[
                    "achieved_spearman_median"
                ],
            }
        if robust_sustained is None:
            summary.update(
                {
                    "minimum_world_robust_nominal_rho_grid": None,
                    "previous_world_robust_nominal_rho_grid": config.inverse_max,
                    "world_robust_clears_by_grid_maximum": False,
                    "world_robust_requirement_status": "UNKNOWN_ABOVE_GRID_MAXIMUM",
                }
            )
        else:
            robust_chosen = rows[robust_sustained]
            robust_previous = rows[robust_sustained - 1] if robust_sustained else None
            summary.update(
                {
                    "minimum_world_robust_nominal_rho_grid": robust_chosen[
                        "nominal_gaussian_rho"
                    ],
                    "previous_world_robust_nominal_rho_grid": (
                        robust_previous["nominal_gaussian_rho"]
                        if robust_previous
                        else None
                    ),
                    "world_robust_clears_by_grid_maximum": True,
                    "world_robust_requirement_status": "FINITE_WITHIN_GRID",
                    "world_robust_drop_best_mean_pnl_at_requirement": robust_chosen[
                        "drop_best_mean_pnl"
                    ],
                    "world_robust_simultaneous_95_lcb_at_requirement": robust_chosen[
                        "drop_best_simultaneous_one_sided_95_lcb"
                    ],
                    "world_robust_selector_world_5th_percentile_at_requirement": (
                        robust_chosen["drop_best_selector_world_5th_percentile"]
                    ),
                    "world_robust_achieved_pearson_median_at_requirement": robust_chosen[
                        "achieved_pearson_median"
                    ],
                    "world_robust_achieved_spearman_median_at_requirement": robust_chosen[
                        "achieved_spearman_median"
                    ],
                }
            )
        summaries.append(summary)
    return summaries, descriptors, simultaneous_critical


def _claim_audit(primary: pd.DataFrame, config: StudyConfig) -> dict[str, Any]:
    """Reproduce one exact +43.47 construction and show why it is non-unique."""

    pnl = primary.sort_values("session", kind="stable")["pnl"].to_numpy(float)
    z = (pnl - pnl.mean()) / pnl.std(ddof=0)
    rho = 0.05
    rate = 0.20
    k = int(round(rate * len(pnl)))
    rng = np.random.default_rng(25_474)
    score = rho * z + math.sqrt(1.0 - rho * rho) * rng.normal(size=len(pnl))
    selected = np.argpartition(score, len(score) - k)[-k:]
    selected_pnl = pnl[selected]
    ordered = np.sort(selected_pnl)
    drop_best = ordered[:-1]
    counts = _bootstrap_counts(
        np.random.default_rng(_seed(config.seed, "claim", "raw")),
        config.bootstrap_reps,
        len(selected_pnl),
    )
    raw_boot = _row_dot(counts, selected_pnl) / len(selected_pnl)
    jack_counts = _bootstrap_counts(
        np.random.default_rng(_seed(config.seed, "claim", "jack")),
        config.bootstrap_reps,
        len(drop_best),
    )
    jack_boot = _row_dot(jack_counts, drop_best) / len(drop_best)
    raw_lo, raw_hi = _percentile_interval(raw_boot)
    jack_lo, jack_hi = _percentile_interval(jack_boot)
    threshold = NormalDist().inv_cdf(1.0 - rate)
    gaussian_approximation = float(
        pnl.mean()
        + rho
        * pnl.std(ddof=1)
        * math.exp(-0.5 * threshold * threshold)
        / math.sqrt(2.0 * math.pi)
        / rate
    )
    positive = ordered[ordered > 0.0]
    return {
        "asserted_claim": {
            "rho": 0.05,
            "baseline_mean_pnl": float(pnl.mean()),
            "asserted_selected_mean_pnl": 43.47,
            "selection_rate_in_original_claim": "UNKNOWN",
            "rng_and_seed_in_original_claim": "UNKNOWN",
            "correlation_metric_in_original_claim": "UNKNOWN",
            "partition_law_in_original_claim": "UNKNOWN",
        },
        "exact_nonunique_reconstruction": {
            "formula": "rho*z(pnl)+sqrt(1-rho^2)*PCG64(seed=25474) normal noise",
            "selection_law": "top round(20% * 1011) = 202 sessions",
            "selected_sessions": int(k),
            "mean_pnl": float(selected_pnl.mean()),
            "mean_pnl_ci_95_low": raw_lo,
            "mean_pnl_ci_95_high": raw_hi,
            "drop_best_mean_pnl": float(drop_best.mean()),
            "drop_best_mean_pnl_ci_95_low": jack_lo,
            "drop_best_mean_pnl_ci_95_high": jack_hi,
            "drop_top_2_mean_pnl": float(ordered[:-2].mean()),
            "drop_top_5_mean_pnl": float(ordered[:-5].mean()),
            "drop_top_10_mean_pnl": float(ordered[:-10].mean()),
            "achieved_pearson": float(np.corrcoef(score, pnl)[0, 1]),
            "achieved_spearman": float(pd.Series(score).corr(pd.Series(pnl), method="spearman")),
            "median_pnl": float(np.median(selected_pnl)),
            "profitable_share": float((selected_pnl > 0.0).mean()),
            "top_one_share_of_positive_pnl": float(positive[-1] / positive.sum()),
            "top_five_share_of_positive_pnl": float(positive[-5:].sum() / positive.sum()),
            "top_five_selected_wins_sum": float(ordered[-5:].sum()),
            "selected_total_pnl": float(selected_pnl.sum()),
        },
        "gaussian_top_20_percent_analytic_approximation": gaussian_approximation,
        "verdict": "VERIFIED_CONSTRUCTIBLE_BUT_REFUTED_AS_UNIQUE_OR_ROBUST_IMPLICATION",
        "provenance": "UNKNOWN",
        "genuine_outcome_blind_oof_survival": "UNKNOWN_AND_NOT_TESTABLE_FROM_THIS_SCORE_LAW",
        "why": (
            "Correlation alone does not identify selected-tail value. The same headline can be "
            "manufactured under many rates and seeds, and neither its raw nor drop-best session "
            "interval clears zero."
        ),
        "predictive_signal_evidence": False,
    }


def _input_manifest(frame: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (start, offset), block in frame.groupby(["start", "offset"], sort=True):
        block = block.sort_values("session", kind="stable")
        sessions = block["session"].to_numpy(str)
        train_n = _split_index(sessions, config)
        pnl = block["pnl"].to_numpy(float)
        rows.append(
            {
                "start": str(start),
                "offset": float(offset),
                "sessions": len(block),
                "first_session": sessions[0],
                "last_session": sessions[-1],
                "calibration_sessions": train_n,
                "calibration_first_session": sessions[0],
                "calibration_last_session": sessions[train_n - 1],
                "later_sessions": len(block) - train_n,
                "later_first_session": sessions[train_n],
                "later_last_session": sessions[-1],
                "unselected_mean_pnl": float(pnl.mean()),
                "unselected_median_pnl": float(np.median(pnl)),
                "unselected_drop_best_mean_pnl": float(np.sort(pnl)[:-1].mean()),
                "analyzed_for_selector_curve": bool(
                    str(start) == PRIMARY_CELL[0] and float(offset) == PRIMARY_CELL[1]
                ),
            }
        )
    return pd.DataFrame(rows)


def _format_money(value: Any) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "UNKNOWN"
    return f"{float(value):+.2f}"


def _readable_tables(
    curve: pd.DataFrame,
    diagnostics: pd.DataFrame,
    inverse: pd.DataFrame,
    sanity: dict[str, Any],
    claim: dict[str, Any],
) -> str:
    later = curve[curve["partition"] == "later_outcome_conditioned_oracle"].sort_values(
        ["nominal_gaussian_rho", "target_selection_rate"]
    )
    claim_row = claim["exact_nonunique_reconstruction"]
    lines = [
        "# Selector-quality requirement curve — readable tables",
        "",
        "> No signal exists in this study. Every score below contains realised P&L from the",
        "> session it scores. The later column is a chronological outcome-conditioned oracle",
        "> stability check, **not** genuine OOF prediction and not evidence of predictability.",
        "",
        "## Direct audit of the +$43.47 assertion",
        "",
        f"A seed-25474 top-20% construction reproduces **{_format_money(claim_row['mean_pnl'])}** ",
        f"with session CI [{_format_money(claim_row['mean_pnl_ci_95_low'])}, ",
        f"{_format_money(claim_row['mean_pnl_ci_95_high'])}]. Drop-best is ",
        f"**{_format_money(claim_row['drop_best_mean_pnl'])}** with CI ",
        f"[{_format_money(claim_row['drop_best_mean_pnl_ci_95_low'])}, ",
        f"{_format_money(claim_row['drop_best_mean_pnl_ci_95_high'])}].",
        "",
        "Verdict: constructible, but not uniquely implied by rho=0.05 and not robust.",
        "",
        "## Later oracle curve after dropping each world's best selected session",
        "",
        "| nominal rho | target | realised | mean/ticket | 95% CI | $/calendar session |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in later.itertuples(index=False):
        lines.append(
            "| "
            f"{row.nominal_gaussian_rho:.3f} | {row.target_selection_rate:.0%} | "
            f"{row.realized_selection_rate:.1%} | "
            f"{_format_money(row.drop_best_mean_pnl_per_selected_ticket)} | "
            f"[{_format_money(row.drop_best_mean_pnl_ci_95_low)}, "
            f"{_format_money(row.drop_best_mean_pnl_ci_95_high)}] | "
            f"{_format_money(row.drop_best_mean_pnl_per_calendar_session)} |"
        )
    lines.extend(
        [
            "",
            "## Calibration versus later oracle check at nominal rho 0.05",
            "",
            "| target | calibration | later | calibration − later | drop-best gap 95% CI |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    subset = curve[curve["nominal_gaussian_rho"] == 0.05]
    for rate in sorted(subset["target_selection_rate"].unique()):
        rows = subset[subset["target_selection_rate"] == rate]
        train = rows[rows["partition"] == "calibration_in_sample_oracle"].iloc[0]
        test = rows[rows["partition"] == "later_outcome_conditioned_oracle"].iloc[0]
        lines.append(
            f"| {rate:.0%} | {_format_money(train['drop_best_mean_pnl_per_selected_ticket'])} | "
            f"{_format_money(test['drop_best_mean_pnl_per_selected_ticket'])} | "
            f"{_format_money(test['drop_best_calibration_minus_later_oracle_gap'])} | "
            f"[{_format_money(test['drop_best_oracle_gap_ci_95_low'])}, "
            f"{_format_money(test['drop_best_oracle_gap_ci_95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "This era gap is time/source-confounded and cannot be interpreted as a genuine",
            "generalisation or tail-fitting gap; that requested quantity is **UNKNOWN**.",
            "",
            "## Achieved correlation at nominal rho 0.05",
            "",
            "| partition | Pearson median (world 95% range) | Spearman median (world 95% range) |",
            "|---|---:|---:|",
        ]
    )
    for row in diagnostics[diagnostics["nominal_gaussian_rho"] == 0.05].itertuples(
        index=False
    ):
        lines.append(
            f"| {row.partition} | {row.achieved_pearson_median:.4f} "
            f"[{row.achieved_pearson_selector_world_95_low:.4f}, "
            f"{row.achieved_pearson_selector_world_95_high:.4f}] | "
            f"{row.achieved_spearman_median:.4f} "
            f"[{row.achieved_spearman_selector_world_95_low:.4f}, "
            f"{row.achieved_spearman_selector_world_95_high:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Inverse requirement under this Gaussian oracle law",
            "",
            "| target | ensemble-mean rho bracket | world-robust rho bracket | robust achieved Pearson | robust achieved Spearman | world 5th pct | simultaneous 95% LCB |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inverse.sort_values("target_selection_rate").to_dict("records"):
        rho = row.get("minimum_nominal_rho_grid")
        prior = row.get("previous_nominal_rho_grid")
        bracket = (
            f">{row['grid_maximum']:.3f} / UNKNOWN"
            if pd.isna(rho)
            else f"({float(prior):.3f}, {float(rho):.3f}]"
            if not pd.isna(prior)
            else f"[0, {float(rho):.3f}]"
        )
        robust_rho = row.get("minimum_world_robust_nominal_rho_grid")
        robust_prior = row.get("previous_world_robust_nominal_rho_grid")
        robust_bracket = (
            f">{row['grid_maximum']:.3f} / UNKNOWN"
            if pd.isna(robust_rho)
            else f"({float(robust_prior):.3f}, {float(robust_rho):.3f}]"
            if not pd.isna(robust_prior)
            else f"[0, {float(robust_rho):.3f}]"
        )
        robust_pearson = (
            "UNKNOWN"
            if pd.isna(robust_rho)
            else f"{row['world_robust_achieved_pearson_median_at_requirement']:.4f}"
        )
        robust_spearman = (
            "UNKNOWN"
            if pd.isna(robust_rho)
            else f"{row['world_robust_achieved_spearman_median_at_requirement']:.4f}"
        )
        lines.append(
            f"| {row['target_selection_rate']:.0%} | {bracket} | {robust_bracket} | "
            f"{robust_pearson} | {robust_spearman} | "
            f"{_format_money(row.get('world_robust_selector_world_5th_percentile_at_requirement'))} | "
            f"{_format_money(row.get('world_robust_simultaneous_95_lcb_at_requirement'))} |"
        )
    lines.extend(
        [
            "",
            "## Sanity floor",
            "",
            f"rho=0 checks passed: **{sanity['passed_rows']}/{sanity['total_rows']}**.",
            "",
            "Every rate is individually within the signed two-ticket/day count cap because this",
            "single cell offers at most one ticket/session. Dollar eligibility is UNKNOWN because",
            "the P&L CSV does not contain the entry ask. **ADOPT NOTHING.**",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    pnl_csv: Path,
    *,
    config: StudyConfig | None = None,
    strict_population: bool = True,
) -> StudyResult:
    """Run the complete primary-cell requirements study without writing artifacts."""

    cfg = StudyConfig() if config is None else config
    cfg.validate()
    frame = load_pnl_table(pnl_csv, strict_population=strict_population)
    manifest = _input_manifest(frame, cfg)
    primary = frame[(frame["start"] == PRIMARY_CELL[0]) & (frame["offset"] == PRIMARY_CELL[1])]
    if primary.empty:
        raise SelectorRequirementError("primary 09:35 ATM cell is absent")
    context = _make_context(primary, cfg)
    if strict_population and (
        context.train_n != EXPECTED_PRIMARY_CALIBRATION_SESSIONS
        or context.later_n != EXPECTED_PRIMARY_LATER_SESSIONS
    ):
        raise SelectorRequirementError("primary-cell source-seam population drift")

    # The null floor is deliberately complete before any nonzero-rho score is constructed.
    zero_train = construct_scores(context.train_z, context.train_noise, 0.0)
    zero_later = construct_scores(context.later_z, context.later_noise, 0.0)
    sanity_rows: list[dict[str, Any]] = []
    for rate in cfg.selection_rates:
        train_selection = _selection(
            zero_train,
            zero_train,
            context.pnl[: context.train_n],
            rate,
            in_sample=True,
        )
        later_selection = _selection(
            zero_train,
            zero_later,
            context.pnl[context.train_n :],
            rate,
            in_sample=False,
        )
        sanity_rows.extend(
            [
                _sanity_row(
                    context,
                    "calibration_in_sample_oracle",
                    rate,
                    train_selection,
                    cfg.seed,
                ),
                _sanity_row(
                    context,
                    "later_outcome_conditioned_oracle",
                    rate,
                    later_selection,
                    cfg.seed,
                ),
            ]
        )
    sanity = {
        "law": "rho=0 selected mean minus same-partition unselected mean",
        "scope": "09:35 ATM primary cell only; run before every nonzero rho",
        "unit": "whole session; one trade per session",
        "rows": sanity_rows,
        "total_rows": len(sanity_rows),
        "passed_rows": sum(bool(row["passes"]) for row in sanity_rows),
        "all_pass": all(bool(row["passes"]) for row in sanity_rows),
    }
    if not sanity["all_pass"]:
        raise SelectorRequirementError("rho=0 sanity floor failed; all nonzero results are void")

    curve_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for rho in cfg.declared_rhos:
        train_scores = construct_scores(context.train_z, context.train_noise, rho)
        later_scores = construct_scores(context.later_z, context.later_noise, rho)
        diagnostic_rows.extend(
            [
                _diagnostic_row(
                    context,
                    "calibration_prefix",
                    rho,
                    train_scores,
                    context.pnl[: context.train_n],
                ),
                _diagnostic_row(
                    context,
                    "later_partition",
                    rho,
                    later_scores,
                    context.pnl[context.train_n :],
                ),
            ]
        )
        for rate in cfg.selection_rates:
            train_selection = _selection(
                train_scores,
                train_scores,
                context.pnl[: context.train_n],
                rate,
                in_sample=True,
            )
            later_selection = _selection(
                train_scores,
                later_scores,
                context.pnl[context.train_n :],
                rate,
                in_sample=False,
            )
            train_row, train_arrays = _curve_row(
                context,
                rho,
                rate,
                "calibration_in_sample_oracle",
                train_selection,
                cfg.confidence,
            )
            later_row, later_arrays = _curve_row(
                context,
                rho,
                rate,
                "later_outcome_conditioned_oracle",
                later_selection,
                cfg.confidence,
            )
            raw_gap = train_arrays["raw"] - later_arrays["raw"]
            jack_gap = train_arrays["jackknife"] - later_arrays["jackknife"]
            raw_gap_lo, raw_gap_hi = _percentile_interval(raw_gap)
            jack_gap_lo, jack_gap_hi = _percentile_interval(jack_gap)
            gap_values = {
                "calibration_minus_later_oracle_gap": (
                    train_row["mean_pnl_per_selected_ticket"]
                    - later_row["mean_pnl_per_selected_ticket"]
                ),
                "oracle_gap_ci_95_low": raw_gap_lo,
                "oracle_gap_ci_95_high": raw_gap_hi,
                "drop_best_calibration_minus_later_oracle_gap": (
                    train_row["drop_best_mean_pnl_per_selected_ticket"]
                    - later_row["drop_best_mean_pnl_per_selected_ticket"]
                ),
                "drop_best_oracle_gap_ci_95_low": jack_gap_lo,
                "drop_best_oracle_gap_ci_95_high": jack_gap_hi,
                "genuine_oof_generalization_gap": "UNKNOWN",
                "gap_confounded_by_time_and_quote_source": True,
            }
            train_row.update(gap_values)
            later_row.update(gap_values)
            curve_rows.extend([train_row, later_row])

    inverse_rows, inverse_trace, simultaneous_critical = _inverse_for_context(context, cfg)
    curve = pd.DataFrame(curve_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    inverse_frame = pd.DataFrame(inverse_rows)
    inverse_trace_frame = pd.DataFrame(inverse_trace)
    claim = _claim_audit(primary, cfg)
    chronological = curve[
        (curve["nominal_gaussian_rho"] == 0.05)
        & (curve["target_selection_rate"] == 0.20)
    ].sort_values("partition")
    claim["fixed_ensemble_chronological_oracle_check"] = table_records(
        chronological[
            [
                "partition",
                "mean_pnl_per_selected_ticket",
                "mean_pnl_ci_95_low",
                "mean_pnl_ci_95_high",
                "drop_best_mean_pnl_per_selected_ticket",
                "drop_best_mean_pnl_ci_95_low",
                "drop_best_mean_pnl_ci_95_high",
                "calibration_minus_later_oracle_gap",
                "drop_best_calibration_minus_later_oracle_gap",
                "drop_best_oracle_gap_ci_95_low",
                "drop_best_oracle_gap_ci_95_high",
            ]
        ]
    )
    metadata = {
        "schema_version": "v5.selector-quality-requirement-analysis.v1",
        "scope": "09:35 ATM primary cell; all 12 input cells are population-QC only",
        "model_fit": False,
        "alpha_spent": False,
        "constructed_score_uses_same_partition_outcome": True,
        "predictive_signal_evidence": False,
        "genuine_outcome_blind_oof": False,
        "genuine_outcome_blind_oof_status": "UNKNOWN_AND_IMPOSSIBLE_WITHOUT_A_CAUSAL_SCORE",
        "later_partition_role": "outcome-conditioned chronological oracle stability check",
        "score_law": (
            "nominal rho * train-standardized realised pnl + sqrt(1-rho^2) * fixed "
            "standard-normal noise; later outcomes use the frozen train mean and scale"
        ),
        "rho_semantics": (
            "nominal Gaussian coefficient, not forced sample correlation; achieved Pearson and "
            "Spearman are reported over fixed, unshopped selector worlds"
        ),
        "chronology": {
            "calibration_end": cfg.calibration_end,
            "later_start": cfg.later_start,
            "calibration_sessions": context.train_n,
            "later_sessions": context.later_n,
            "threshold_established_on_calibration_prefix_only": True,
            "later_pnl_used_to_set_threshold": False,
            "later_pnl_used_inside_each_later_constructed_score": True,
            "time_and_quote_source_confounded_at_split": True,
        },
        "uncertainty": {
            "economic_interval": "two-sided nonparametric percentile bootstrap",
            "inverse_interval": "one-sided simultaneous studentized lower band",
            "confidence": cfg.confidence,
            "resamples": cfg.bootstrap_reps,
            "unit": "session",
            "selector_worlds": cfg.selector_draws,
            "common_noise_across_rhos": True,
            "shared_session_resamples_across_cells_within_partition": True,
            "fixed_calibration_threshold_ensemble": True,
            "selector_world_range_is_not_a_confidence_interval": True,
        },
        "jackknife": "drop one largest selected P&L per world without refill",
        "inverse_rule": (
            "first nominal rho whose later oracle drop-best simultaneous one-sided 95% LCB "
            "clears zero and remains clear at all higher preregistered grid points; the "
            "world-robust requirement additionally makes the fifth percentile across fixed "
            "selector worlds positive at every higher point"
        ),
        "inverse_grid_step": cfg.inverse_step,
        "inverse_grid_maximum": cfg.inverse_max,
        "inverse_simultaneous_critical": simultaneous_critical,
        "selection_rates": list(cfg.selection_rates),
        "declared_rhos": list(cfg.declared_rhos),
        "two_ticket_law": (
            "all rates are count-executable for this one-trade/session cell; dollar ticket "
            "eligibility is unknown because ask is absent"
        ),
        "alpha_position": "REQUIREMENTS_STUDY_ON_ALREADY_OPEN_PNL_TABLE_NO_ALPHA_CHARGE",
        "adoption": "ADOPT NOTHING",
    }
    readable = _readable_tables(curve, diagnostics, inverse_frame, sanity, claim)
    return StudyResult(
        input_manifest=manifest,
        requirement_curve=curve,
        correlation_diagnostics=diagnostics,
        inverse_requirement=inverse_frame,
        inverse_trace=inverse_trace_frame,
        sanity_floor=sanity,
        claim_audit=claim,
        metadata=metadata,
        readable_tables=readable,
    )


def table_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return canonical-JSON-safe scalar records; non-finite cells fail closed."""

    out: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                converted[key] = None
            elif value is pd.NA:
                converted[key] = None
            else:
                converted[key] = value
        out.append(converted)
    return out
