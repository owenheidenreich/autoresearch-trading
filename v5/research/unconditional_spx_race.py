"""Unconditional SPX barrier races and path distributions.

This module extends the move-terrain census without opening an alpha exposure.
Every path is selected only by an exact clock and horizon.  It never reads a
feature, signal, prior state, option outcome, option P&L, label, fitted value, or
reserved session.

The underlying tape is one SPX parity snapshot per minute.  Consequently a
"touch" means first *observed snapshot* crossing, never an invented intraminute
high/low.  Every cell contributes at most one observation per session.  Wilson
intervals, mean intervals, and order-statistic quantile intervals therefore use
sessions—not overlapping windows or contract rows—as the observation unit.

The implied-volatility companion reads only the causal ``ladder`` table.  Its
primary estimand is the equal-weighted mean of the call and put median IVs in
the project's pre-existing +/-10-point ATM band.  It is an ATM clock curve, not
permission to extrapolate IV to the absent 40- or 60-point OTM wings.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom, t as student_t


SCHEMA_VERSION = "v5.unconditional-spx-race.v1"
CONFIRMATION_START = date(2026, 8, 6)
ERA_CUTOFF = date(2025, 8, 1)

EXPECTED_DISCOVERED_SESSIONS = 1_014
EXPECTED_ANALYZED_SESSIONS = 1_011
EXPECTED_FIRST_SESSION = date(2022, 6, 1)
EXPECTED_LAST_SESSION = date(2026, 7, 30)

TAPE_FILE = re.compile(r"^(20\d{2}-\d{2}-\d{2})\.es_c_0\.ohlcv-1m\.parquet$")
TAPE_SOURCE = "spx_parity_spot"
TAPE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_observation_minute",
    "tape_source",
)

# The padded 2022 early close is absent upstream.  These three files remain in
# both roots and are excluded before any price or IV row is opened.
KNOWN_DEFECTS: Mapping[date, str] = {
    date(2022, 11, 25): "vendor-padded early close; excluded upstream",
    date(2023, 6, 26): "interior whole-book freeze (2 minutes)",
    date(2023, 10, 19): "interior whole-book freeze (3 minutes)",
    date(2023, 10, 25): "interior whole-book freezes (4 and 18 minutes)",
}

HORIZONS_MINUTES = (5, 10, 15, 20, 30, 45, 60, 90)
FAVOURABLE_THRESHOLDS_POINTS = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
# Symmetric with the favourable grid.  This is the declared extension over the
# prior {2,5,10,20} surface and makes the stop trade-off visible without a
# post-outcome threshold choice.
ADVERSE_THRESHOLDS_POINTS = FAVOURABLE_THRESHOLDS_POINTS
OVERSHOOT_THRESHOLDS_POINTS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0)
DISTRIBUTION_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
START_TIMES = (
    "09:31",
    "09:35",
    "09:45",
    "10:00",
    "10:30",
    "11:30",
    "12:30",
    "13:30",
    "14:30",
    "15:00",
    "15:15",
    "15:30",
    "15:40",
    "15:45",
    "15:50",
    "15:55",
)
DIRECTIONS = ("call", "put")
REPORTING_SCOPES = (
    "pooled",
    "era_backfill_2022-06-01_to_2025-07-31",
    "era_owned_2025-08-01_to_2026-07-30",
)
CI_Z = 1.959963984540054

PATH_BASE_COLUMNS = (
    "scope",
    "direction",
    "start_time_et",
    "horizon_minutes",
    "favourable_threshold_points",
    "adverse_threshold_points",
)
PATH_TABLE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "race_surface": (
        *PATH_BASE_COLUMNS,
        "sessions",
        "favourable_first_count",
        "favourable_first_probability",
        "favourable_first_ci_95_low",
        "favourable_first_ci_95_high",
        "adverse_first_count",
        "adverse_first_probability",
        "adverse_first_ci_95_low",
        "adverse_first_ci_95_high",
        "neither_count",
        "neither_probability",
        "neither_ci_95_low",
        "neither_ci_95_high",
    ),
    "overshoot_distribution": (
        *PATH_BASE_COLUMNS,
        "conditioning",
        "total_sessions",
        "conditional_sessions",
        "conditioning_probability",
        "conditioning_ci_95_low",
        "conditioning_ci_95_high",
        "overshoot_threshold_points",
        "at_or_above_count",
        "at_or_above_probability",
        "at_or_above_ci_95_low",
        "at_or_above_ci_95_high",
        "status",
    ),
    "time_to_event_distribution": (
        *PATH_BASE_COLUMNS,
        "conditioning",
        "total_sessions",
        "conditional_sessions",
        "conditioning_probability",
        "conditioning_ci_95_low",
        "conditioning_ci_95_high",
        "elapsed_minutes",
        "reached_by_count",
        "reached_by_probability",
        "reached_by_ci_95_low",
        "reached_by_ci_95_high",
        "status",
    ),
    "stop_compatibility": (
        *PATH_BASE_COLUMNS,
        "total_sessions",
        "favourable_reached_count",
        "favourable_reached_probability",
        "favourable_reached_ci_95_low",
        "favourable_reached_ci_95_high",
        "conditional_sessions",
        "stop_rule",
        "option_percent_stop_mapping",
        "survives_to_favourable_count",
        "survives_to_favourable_probability",
        "survives_to_favourable_ci_95_low",
        "survives_to_favourable_ci_95_high",
        "killed_before_favourable_count",
        "killed_before_favourable_probability",
        "killed_before_favourable_ci_95_low",
        "killed_before_favourable_ci_95_high",
        "status",
    ),
    "conditional_quantiles": (
        *PATH_BASE_COLUMNS,
        "metric",
        "conditioning",
        "unit",
        "total_sessions",
        "conditional_sessions",
        "conditioning_probability",
        "conditioning_ci_95_low",
        "conditioning_ci_95_high",
        "statistic",
        "quantile_probability",
        "estimate",
        "ci_95_low",
        "ci_95_high",
        "interval_method",
        "status",
    ),
}


class UnconditionalRaceError(RuntimeError):
    """The census cannot prove its population, clock, or inference law."""


def minute_number(value: str) -> int:
    """Minutes after midnight for a strict ``HH:MM`` label."""

    try:
        hour, minute = (int(piece) for piece in value.split(":"))
    except Exception as exc:  # noqa: BLE001 - normalize to fail-closed error
        raise UnconditionalRaceError(f"invalid clock label {value!r}") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise UnconditionalRaceError(f"invalid clock label {value!r}")
    return hour * 60 + minute


def minute_label(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


EXPECTED_MINUTES = tuple(
    minute_label(value)
    for value in range(minute_number("09:31"), minute_number("16:00") + 1)
)
EXPECTED_BAR_MINUTES = tuple(
    minute_label(value)
    for value in range(minute_number("09:30"), minute_number("15:59") + 1)
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TapeCorpus:
    root: Path
    sessions: tuple[str, ...]
    dates: tuple[date, ...]
    prices: np.ndarray
    minutes: tuple[str, ...]
    input_manifest: pd.DataFrame
    defect_disposition: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        expected = (len(self.sessions), len(self.minutes))
        if self.prices.shape != expected:
            raise UnconditionalRaceError(
                f"price matrix shape {self.prices.shape} does not match {expected}"
            )
        if len(self.dates) != len(self.sessions):
            raise UnconditionalRaceError("session/date lengths differ")


@dataclass(frozen=True)
class AnalysisTables:
    race_surface: pd.DataFrame
    overshoot_distribution: pd.DataFrame
    time_to_event_distribution: pd.DataFrame
    stop_compatibility: pd.DataFrame
    conditional_quantiles: pd.DataFrame
    iv_by_time: pd.DataFrame
    iv_clock_change: pd.DataFrame
    iv_session_values: pd.DataFrame


@dataclass(frozen=True)
class RaceAnalysisRun:
    """Complete in-memory result consumed by the immutable evidence wrapper."""

    tables: AnalysisTables
    quality_control: Mapping[str, Any]
    input_manifest: pd.DataFrame
    defect_disposition: pd.DataFrame
    readable_tables: str


def _read_tape(path: Path, session: str) -> np.ndarray:
    try:
        frame = pd.read_parquet(path, columns=list(TAPE_COLUMNS))
    except Exception as exc:  # noqa: BLE001
        raise UnconditionalRaceError(f"{session}: tape unreadable: {exc}") from exc
    if len(frame) != len(EXPECTED_MINUTES):
        raise UnconditionalRaceError(
            f"{session}: expected {len(EXPECTED_MINUTES)} snapshots, found {len(frame)}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise UnconditionalRaceError(f"{session}: ts_event is not timezone-aware")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise UnconditionalRaceError(f"{session}: ts_event is duplicated or out of order")
    local = frame.index.tz_convert("America/New_York")
    if set(local.strftime("%Y-%m-%d")) != {session}:
        raise UnconditionalRaceError(f"{session}: filename and local date disagree")
    if tuple(local.strftime("%H:%M")) != EXPECTED_BAR_MINUTES:
        raise UnconditionalRaceError(f"{session}: bar-label clock is not 09:30-15:59 ET")
    minutes = tuple(frame["bar_observation_minute"].astype(str))
    if minutes != EXPECTED_MINUTES:
        raise UnconditionalRaceError(f"{session}: observation clock is not 09:31-16:00 ET")
    observed = tuple((local + pd.Timedelta(minutes=1)).strftime("%H:%M"))
    if observed != minutes:
        raise UnconditionalRaceError(f"{session}: observation minute is not bar t+1")
    if set(frame["tape_source"].astype(str)) != {TAPE_SOURCE}:
        raise UnconditionalRaceError(f"{session}: wrong tape_source")
    close = frame["close"].to_numpy(float)
    if not np.isfinite(close).all() or (close <= 0.0).any():
        raise UnconditionalRaceError(f"{session}: non-finite/non-positive tape price")
    for column in ("open", "high", "low"):
        if not np.array_equal(frame[column].to_numpy(float), close):
            raise UnconditionalRaceError(f"{session}: parity tape is not snapshot-only")
    if not np.array_equal(frame["volume"].to_numpy(float), np.zeros(len(frame))):
        raise UnconditionalRaceError(f"{session}: parity tape volume is not zero")
    return close


def _discover_files(root: Path, pattern: re.Pattern[str]) -> list[tuple[date, Path]]:
    if not root.is_dir():
        raise UnconditionalRaceError(f"missing input directory: {root}")
    discovered: list[tuple[date, Path]] = []
    unexpected: list[str] = []
    for path in root.glob("*.parquet"):
        match = pattern.match(path.name)
        if not match:
            unexpected.append(path.name)
            continue
        discovered.append((date.fromisoformat(match.group(1)), path))
    discovered.sort(key=lambda pair: pair[0])
    if unexpected:
        raise UnconditionalRaceError(f"unexpected parquet names: {sorted(unexpected)[:3]}")
    dates = [value for value, _ in discovered]
    if not discovered:
        raise UnconditionalRaceError(f"input directory contains no sessions: {root}")
    if len(dates) != len(set(dates)):
        raise UnconditionalRaceError(f"duplicate session date in {root}")
    reserved = [value for value in dates if value >= CONFIRMATION_START]
    if reserved:
        raise UnconditionalRaceError(f"confirmation-reserved session present: {reserved[0]}")
    return discovered


def load_tape_corpus(root: Path, *, strict_population: bool = True) -> TapeCorpus:
    """Load the clean tape, excluding defects before their prices are read."""

    root = Path(root)
    discovered = _discover_files(root, TAPE_FILE)
    dates = [value for value, _ in discovered]
    if strict_population:
        facts = (len(discovered), dates[0], dates[-1])
        expected = (
            EXPECTED_DISCOVERED_SESSIONS,
            EXPECTED_FIRST_SESSION,
            EXPECTED_LAST_SESSION,
        )
        if facts != expected:
            raise UnconditionalRaceError(f"tape population drifted: {facts}, expected {expected}")

    rows: list[dict[str, Any]] = []
    clean_sessions: list[str] = []
    clean_dates: list[date] = []
    clean_prices: list[np.ndarray] = []
    disposition: list[dict[str, Any]] = []
    present_dates = set(dates)
    for session_date, path in discovered:
        session = session_date.isoformat()
        row = {
            "input_family": "parity_tape",
            "session": session,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "source_era": "backfill" if session_date < ERA_CUTOFF else "owned",
        }
        reason = KNOWN_DEFECTS.get(session_date)
        if reason:
            row.update(status="EXCLUDED_KNOWN_DEFECT", reason=reason)
            disposition.append(
                {"session": session, "present": True, "disposition": "EXCLUDED", "reason": reason}
            )
        else:
            values = _read_tape(path, session)
            row.update(status="ANALYZED", reason="")
            clean_sessions.append(session)
            clean_dates.append(session_date)
            clean_prices.append(values)
        rows.append(row)
    for defective, reason in KNOWN_DEFECTS.items():
        if defective not in present_dates:
            disposition.append(
                {
                    "session": defective.isoformat(),
                    "present": False,
                    "disposition": "EXCLUDED_UPSTREAM_OR_ABSENT",
                    "reason": reason,
                }
            )
    if strict_population and len(clean_sessions) != EXPECTED_ANALYZED_SESSIONS:
        raise UnconditionalRaceError(
            f"clean population drifted: {len(clean_sessions)}, expected {EXPECTED_ANALYZED_SESSIONS}"
        )
    return TapeCorpus(
        root=root,
        sessions=tuple(clean_sessions),
        dates=tuple(clean_dates),
        prices=np.vstack(clean_prices),
        minutes=EXPECTED_MINUTES,
        input_manifest=pd.DataFrame(rows).sort_values("session").reset_index(drop=True),
        defect_disposition=tuple(sorted(disposition, key=lambda row: row["session"])),
    )


def scope_masks(corpus: TapeCorpus) -> dict[str, np.ndarray]:
    dates = np.asarray(corpus.dates, dtype=object)
    return {
        "pooled": np.ones(len(dates), dtype=bool),
        "era_backfill_2022-06-01_to_2025-07-31": np.asarray(
            [value < ERA_CUTOFF for value in dates], dtype=bool
        ),
        "era_owned_2025-08-01_to_2026-07-30": np.asarray(
            [value >= ERA_CUTOFF for value in dates], dtype=bool
        ),
    }


def valid_configurations(minutes: Sequence[str] = EXPECTED_MINUTES) -> list[tuple[str, int, int]]:
    index = {value: position for position, value in enumerate(minutes)}
    cells: list[tuple[str, int, int]] = []
    for start_time in START_TIMES:
        if start_time not in index:
            raise UnconditionalRaceError(f"declared start absent from tape: {start_time}")
        start = index[start_time]
        for horizon in HORIZONS_MINUTES:
            if start + horizon < len(minutes):
                cells.append((start_time, start, horizon))
    return cells


def wilson_interval(successes: int, trials: int, *, z: float = CI_Z) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise UnconditionalRaceError(
            f"invalid Bernoulli counts: {successes} successes in {trials} trials"
        )
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    low, high = max(0.0, centre - half), min(1.0, centre + half)
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return low, high


def first_crossing(path: np.ndarray, threshold: float, *, above: bool) -> np.ndarray:
    """One-based first crossing minute, or ``horizon + 1`` when absent."""

    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise UnconditionalRaceError("first_crossing needs a finite non-empty matrix")
    hit = values >= threshold if above else values <= threshold
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1) + 1
    return np.where(any_hit, first, values.shape[1] + 1).astype(int)


def path_excursions(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise UnconditionalRaceError("path_excursions needs a finite non-empty matrix")
    return np.maximum(values.max(axis=1), 0.0), np.maximum((-values).max(axis=1), 0.0)


def pre_favourable_adverse(path: np.ndarray, gain_index: np.ndarray) -> np.ndarray:
    """Worst adverse snapshot through the first favourable crossing.

    The favourable crossing snapshot itself cannot add adverse excursion, so
    including it is equivalent to stopping immediately before it and makes a
    first-minute hit mechanically return zero.
    """

    values = np.asarray(path, dtype=float)
    gain = np.asarray(gain_index, dtype=int)
    if values.ndim != 2 or gain.shape != (values.shape[0],):
        raise UnconditionalRaceError("pre-favourable adverse shapes differ")
    horizon = values.shape[1]
    result = np.full(values.shape[0], np.nan, dtype=float)
    cumulative = np.maximum.accumulate(np.maximum(-values, 0.0), axis=1)
    hit = gain <= horizon
    rows = np.flatnonzero(hit)
    result[rows] = cumulative[rows, gain[rows] - 1]
    return result


def _probability_fields(prefix: str, successes: int, trials: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, trials)
    return {
        f"{prefix}_count": int(successes),
        f"{prefix}_probability": successes / trials,
        f"{prefix}_ci_95_low": low,
        f"{prefix}_ci_95_high": high,
    }


def _conditional_probability_fields(
    prefix: str, successes: int, trials: int
) -> dict[str, Any]:
    if trials == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_probability": None,
            f"{prefix}_ci_95_low": None,
            f"{prefix}_ci_95_high": None,
            "status": "NO_CONDITIONING_EVENTS",
        }
    return {**_probability_fields(prefix, successes, trials), "status": "MEASURED"}


def _distribution_rows(
    values: np.ndarray,
    *,
    descriptor: Mapping[str, Any],
    metric: str,
    conditioning: str,
    unit: str,
    total_sessions: int,
) -> list[dict[str, Any]]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    ordered = np.sort(clean)
    condition_low, condition_high = wilson_interval(n, total_sessions)
    common = {
        **descriptor,
        "metric": metric,
        "conditioning": conditioning,
        "unit": unit,
        "total_sessions": total_sessions,
        "conditional_sessions": n,
        "conditioning_probability": n / total_sessions,
        "conditioning_ci_95_low": condition_low,
        "conditioning_ci_95_high": condition_high,
    }
    statistics: list[tuple[str, float | None]] = [("mean", None)] + [
        (f"q{int(round(q * 100)):02d}", q) for q in DISTRIBUTION_QUANTILES
    ]
    rows: list[dict[str, Any]] = []
    for statistic, quantile in statistics:
        row = {**common, "statistic": statistic, "quantile_probability": quantile}
        if n == 0:
            row.update(
                estimate=None,
                ci_95_low=None,
                ci_95_high=None,
                interval_method="none",
                status="NO_CONDITIONING_EVENTS",
            )
        elif statistic == "mean":
            estimate = float(clean.mean())
            if n < 2:
                row.update(
                    estimate=estimate,
                    ci_95_low=None,
                    ci_95_high=None,
                    interval_method="session_t_interval",
                    status="INSUFFICIENT_CONDITIONAL_SESSIONS_FOR_INTERVAL",
                )
            else:
                critical = float(student_t.ppf(0.975, n - 1))
                half = critical * float(clean.std(ddof=1)) / math.sqrt(n)
                row.update(
                    estimate=estimate,
                    ci_95_low=estimate - half,
                    ci_95_high=estimate + half,
                    interval_method="session_t_interval",
                    status="MEASURED",
                )
        else:
            assert quantile is not None
            estimate = float(np.quantile(clean, quantile, method="linear"))
            if n < 2:
                row.update(
                    estimate=estimate,
                    ci_95_low=None,
                    ci_95_high=None,
                    interval_method="distribution_free_order_statistic",
                    status="INSUFFICIENT_CONDITIONAL_SESSIONS_FOR_INTERVAL",
                )
            else:
                # Exact distribution-free order-statistic limits.  Ranks are
                # one-based.  A rank of zero or n+1 means that this sample
                # cannot support a finite 95% bound on that tail; clamping it
                # to the observed min/max would materially under-cover (most
                # notably q99 in sparse favourable-first cells).
                lower_rank = int(binom.ppf(0.025, n, quantile))
                upper_rank = int(binom.ppf(0.975, n, quantile)) + 1
                interval_low = (
                    min(float(ordered[lower_rank - 1]), estimate)
                    if lower_rank >= 1
                    else None
                )
                interval_high = (
                    max(float(ordered[upper_rank - 1]), estimate)
                    if upper_rank <= n
                    else None
                )
                interval_status = (
                    "MEASURED"
                    if interval_low is not None and interval_high is not None
                    else "MEASURED_WITH_OPEN_CONFIDENCE_BOUND"
                )
                row.update(
                    estimate=estimate,
                    ci_95_low=interval_low,
                    ci_95_high=interval_high,
                    interval_method="exact_binomial_order_statistic",
                    status=interval_status,
                )
        rows.append(row)
    return rows


def _race_row(
    gain_index: np.ndarray,
    adverse_index: np.ndarray,
    *,
    mask: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    gain = gain_index[mask]
    adverse = adverse_index[mask]
    favourable_first = gain < adverse
    adverse_first = adverse < gain
    neither = (gain > horizon) & (adverse > horizon)
    impossible_tie = (gain == adverse) & (gain <= horizon)
    if impossible_tie.any():
        raise UnconditionalRaceError("positive and negative scalar barriers tied")
    if not np.all(favourable_first | adverse_first | neither):
        raise UnconditionalRaceError("three race states do not partition sessions")
    trials = int(mask.sum())
    return {
        "sessions": trials,
        **_probability_fields("favourable_first", int(favourable_first.sum()), trials),
        **_probability_fields("adverse_first", int(adverse_first.sum()), trials),
        **_probability_fields("neither", int(neither.sum()), trials),
    }


def analyze_paths(corpus: TapeCorpus) -> tuple[pd.DataFrame, ...]:
    """Compute the full three-state race and all requested path distributions."""

    scopes = scope_masks(corpus)
    race_rows: list[dict[str, Any]] = []
    overshoot_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []

    for start_time, start_index, horizon in valid_configurations(corpus.minutes):
        start_spot = corpus.prices[:, start_index]
        unsigned_path = (
            corpus.prices[:, start_index + 1 : start_index + horizon + 1]
            - start_spot[:, None]
        )
        for direction in DIRECTIONS:
            sign = 1.0 if direction == "call" else -1.0
            signed_path = unsigned_path * sign
            favourable_excursion, _ = path_excursions(signed_path)
            gain_indices = {
                threshold: first_crossing(signed_path, threshold, above=True)
                for threshold in FAVOURABLE_THRESHOLDS_POINTS
            }
            adverse_indices = {
                threshold: first_crossing(signed_path, -threshold, above=False)
                for threshold in ADVERSE_THRESHOLDS_POINTS
            }
            base = {
                "direction": direction,
                "start_time_et": start_time,
                "horizon_minutes": horizon,
            }

            for favourable_threshold, gain_index in gain_indices.items():
                hit = gain_index <= horizon
                pre_adverse = pre_favourable_adverse(signed_path, gain_index)
                for scope, mask in scopes.items():
                    total_sessions = int(mask.sum())
                    hit_mask = mask & hit
                    hit_count = int(hit_mask.sum())
                    hit_low, hit_high = wilson_interval(hit_count, total_sessions)
                    event_descriptor = {
                        "scope": scope,
                        **base,
                        "favourable_threshold_points": favourable_threshold,
                        "adverse_threshold_points": None,
                    }

                    # Time-to-event is conditioned on an eventual +M touch in
                    # any order, exactly as requested.  Every elapsed minute is
                    # emitted, making this the complete empirical CDF.
                    for elapsed in range(1, horizon + 1):
                        reached = int((gain_index[hit_mask] <= elapsed).sum())
                        row = {
                            **event_descriptor,
                            "conditioning": "favourable_reached_any_order",
                            "total_sessions": total_sessions,
                            "conditional_sessions": hit_count,
                            "conditioning_probability": hit_count / total_sessions,
                            "conditioning_ci_95_low": hit_low,
                            "conditioning_ci_95_high": hit_high,
                            "elapsed_minutes": elapsed,
                        }
                        row.update(
                            _conditional_probability_fields(
                                "reached_by", reached, hit_count
                            )
                        )
                        time_rows.append(row)

                    quantile_rows.extend(
                        _distribution_rows(
                            gain_index[hit_mask].astype(float),
                            descriptor=event_descriptor,
                            metric="time_to_favourable_touch",
                            conditioning="favourable_reached_any_order",
                            unit="minutes",
                            total_sessions=total_sessions,
                        )
                    )
                    quantile_rows.extend(
                        _distribution_rows(
                            pre_adverse[hit_mask],
                            descriptor=event_descriptor,
                            metric="adverse_excursion_before_favourable",
                            conditioning="favourable_reached_any_order",
                            unit="spx_points",
                            total_sessions=total_sessions,
                        )
                    )

                    # An underlying stop at -J survives an eventual winner iff
                    # the pre-gain adverse excursion stays strictly inside J.
                    for adverse_threshold in ADVERSE_THRESHOLDS_POINTS:
                        pre_values = pre_adverse[hit_mask]
                        survive = int((pre_values < adverse_threshold).sum())
                        killed = hit_count - survive
                        stop_row = {
                            "scope": scope,
                            **base,
                            "favourable_threshold_points": favourable_threshold,
                            "adverse_threshold_points": adverse_threshold,
                            "total_sessions": total_sessions,
                            "favourable_reached_count": hit_count,
                            "favourable_reached_probability": hit_count / total_sessions,
                            "favourable_reached_ci_95_low": hit_low,
                            "favourable_reached_ci_95_high": hit_high,
                            "conditional_sessions": hit_count,
                            "stop_rule": "touching -J before +M kills; survival requires pre-M adverse < J",
                            "option_percent_stop_mapping": "UNKNOWN_NOT_OPTION_PNL",
                        }
                        survival_fields = _conditional_probability_fields(
                            "survives_to_favourable", survive, hit_count
                        )
                        status = survival_fields.pop("status")
                        killed_fields = _conditional_probability_fields(
                            "killed_before_favourable", killed, hit_count
                        )
                        killed_fields.pop("status")
                        stop_row.update(
                            **survival_fields,
                            **killed_fields,
                            status=status,
                        )
                        stop_rows.append(stop_row)

                for adverse_threshold, adverse_index in adverse_indices.items():
                    favourable_first = gain_index < adverse_index
                    overshoot = favourable_excursion - favourable_threshold
                    for scope, mask in scopes.items():
                        total_sessions = int(mask.sum())
                        condition_mask = mask & favourable_first
                        conditional_sessions = int(condition_mask.sum())
                        condition_low, condition_high = wilson_interval(
                            conditional_sessions, total_sessions
                        )
                        descriptor = {
                            "scope": scope,
                            **base,
                            "favourable_threshold_points": favourable_threshold,
                            "adverse_threshold_points": adverse_threshold,
                        }
                        race_rows.append(
                            {
                                **descriptor,
                                **_race_row(
                                    gain_index,
                                    adverse_index,
                                    mask=mask,
                                    horizon=horizon,
                                ),
                            }
                        )
                        conditional_values = overshoot[condition_mask]
                        for beyond in OVERSHOOT_THRESHOLDS_POINTS:
                            at_or_above = int((conditional_values >= beyond).sum())
                            row = {
                                **descriptor,
                                "conditioning": "favourable_first",
                                "total_sessions": total_sessions,
                                "conditional_sessions": conditional_sessions,
                                "conditioning_probability": conditional_sessions / total_sessions,
                                "conditioning_ci_95_low": condition_low,
                                "conditioning_ci_95_high": condition_high,
                                "overshoot_threshold_points": beyond,
                            }
                            row.update(
                                _conditional_probability_fields(
                                    "at_or_above", at_or_above, conditional_sessions
                                )
                            )
                            overshoot_rows.append(row)
                        quantile_rows.extend(
                            _distribution_rows(
                                conditional_values,
                                descriptor=descriptor,
                                metric="overshoot_after_favourable_first",
                                conditioning="favourable_first",
                                unit="spx_points",
                                total_sessions=total_sessions,
                            )
                        )

    return (
        pd.DataFrame(race_rows),
        pd.DataFrame(overshoot_rows),
        pd.DataFrame(time_rows),
        pd.DataFrame(stop_rows),
        pd.DataFrame(quantile_rows),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnconditionalRaceError(message)


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in frame.columns]
    _require(not missing, f"{name}: missing key columns {missing}")
    duplicate = frame.duplicated(list(keys), keep=False)
    _require(not duplicate.any(), f"{name}: duplicate key rows")


def _require_exact_schema(frame: pd.DataFrame, *, name: str) -> None:
    expected = PATH_TABLE_SCHEMAS[name]
    observed = tuple(frame.columns)
    _require(
        observed == expected,
        f"{name}: schema drifted; observed={observed}, expected={expected}",
    )


def _require_domain(
    frame: pd.DataFrame,
    column: str,
    allowed: Sequence[Any],
    *,
    name: str,
) -> None:
    values = frame[column]
    _require(values.notna().all(), f"{name}: {column} contains null")
    _require(
        values.isin(tuple(allowed)).all(),
        f"{name}: {column} contains a value outside the declared grid",
    )


def _require_literal(frame: pd.DataFrame, column: str, value: str, *, name: str) -> None:
    _require(
        frame[column].eq(value).all(),
        f"{name}: {column} is not literally {value!r}",
    )


def _wilson_arrays(successes: np.ndarray, trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    probability = successes / trials
    z2 = CI_Z * CI_Z
    denominator = 1.0 + z2 / trials
    centre = (probability + z2 / (2.0 * trials)) / denominator
    half = (
        CI_Z
        * np.sqrt((probability * (1.0 - probability) + z2 / (4.0 * trials)) / trials)
        / denominator
    )
    low = np.maximum(0.0, centre - half)
    high = np.minimum(1.0, centre + half)
    low = np.where(successes == 0.0, 0.0, low)
    high = np.where(successes == trials, 1.0, high)
    return low, high


def _require_binomial_fields(
    frame: pd.DataFrame,
    *,
    count_column: str,
    trials_column: str,
    probability_column: str,
    low_column: str,
    high_column: str,
    name: str,
    allow_empty: bool,
) -> None:
    required = (count_column, trials_column, probability_column, low_column, high_column)
    missing = [column for column in required if column not in frame.columns]
    _require(not missing, f"{name}: missing binomial fields {missing}")

    count = pd.to_numeric(frame[count_column], errors="coerce").to_numpy(float)
    trials = pd.to_numeric(frame[trials_column], errors="coerce").to_numpy(float)
    probability = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame[low_column], errors="coerce").to_numpy(float)
    high = pd.to_numeric(frame[high_column], errors="coerce").to_numpy(float)

    _require(np.isfinite(count).all(), f"{name}: count is non-finite")
    _require(np.isfinite(trials).all(), f"{name}: denominator is non-finite")
    _require((count == np.floor(count)).all(), f"{name}: count is not integral")
    _require((trials == np.floor(trials)).all(), f"{name}: denominator is not integral")
    if allow_empty:
        _require((trials >= 0.0).all(), f"{name}: negative denominator")
    else:
        _require((trials > 0.0).all(), f"{name}: non-positive denominator")
    _require((count >= 0.0).all(), f"{name}: negative count")
    _require((count <= trials).all(), f"{name}: count exceeds denominator")

    measured = trials > 0.0
    _require(np.isfinite(probability[measured]).all(), f"{name}: probability is non-finite")
    _require(
        np.allclose(
            probability[measured],
            count[measured] / trials[measured],
            rtol=0.0,
            atol=1e-12,
        ),
        f"{name}: probability does not reproduce counts",
    )
    expected_low, expected_high = _wilson_arrays(count[measured], trials[measured])
    _require(
        np.allclose(low[measured], expected_low, rtol=0.0, atol=1e-12)
        and np.allclose(high[measured], expected_high, rtol=0.0, atol=1e-12),
        f"{name}: Wilson interval does not reproduce counts",
    )

    empty = ~measured
    if empty.any():
        _require(allow_empty, f"{name}: empty denominator is not allowed")
        _require((count[empty] == 0.0).all(), f"{name}: empty cell has a nonzero count")
        _require(
            frame.loc[empty, [probability_column, low_column, high_column]].isna().all().all(),
            f"{name}: empty cell has probability or interval values",
        )


def _require_probability_fields(
    frame: pd.DataFrame,
    *,
    prefix: str,
    trials_column: str,
    name: str,
    allow_empty: bool = False,
) -> None:
    _require_binomial_fields(
        frame,
        count_column=f"{prefix}_count",
        trials_column=trials_column,
        probability_column=f"{prefix}_probability",
        low_column=f"{prefix}_ci_95_low",
        high_column=f"{prefix}_ci_95_high",
        name=name,
        allow_empty=allow_empty,
    )


def _require_conditioning_fields(frame: pd.DataFrame, *, name: str) -> None:
    _require_binomial_fields(
        frame,
        count_column="conditional_sessions",
        trials_column="total_sessions",
        probability_column="conditioning_probability",
        low_column="conditioning_ci_95_low",
        high_column="conditioning_ci_95_high",
        name=name,
        allow_empty=False,
    )


def _require_conditional_probability_status(frame: pd.DataFrame, *, name: str) -> None:
    conditional = pd.to_numeric(frame["conditional_sessions"], errors="coerce")
    expected = np.where(
        conditional.to_numpy(float) > 0.0,
        "MEASURED",
        "NO_CONDITIONING_EVENTS",
    )
    _require(
        np.array_equal(frame["status"].astype(object).to_numpy(), expected),
        f"{name}: status does not literally match the conditioning population",
    )


def _require_distribution_status(frame: pd.DataFrame, *, name: str) -> None:
    conditional = pd.to_numeric(frame["conditional_sessions"], errors="coerce").to_numpy(float)
    is_mean = frame["statistic"].eq("mean").to_numpy()
    is_quantile = ~is_mean
    no_events = conditional == 0.0
    singleton = conditional == 1.0
    measured_mean = (conditional >= 2.0) & is_mean
    measured_quantile = (conditional >= 2.0) & is_quantile

    _require(
        frame.loc[no_events, "status"].eq("NO_CONDITIONING_EVENTS").all(),
        f"{name}: zero-event status drifted",
    )
    _require(
        frame.loc[no_events, "interval_method"].eq("none").all()
        and frame.loc[no_events, ["estimate", "ci_95_low", "ci_95_high"]]
        .isna()
        .all()
        .all(),
        f"{name}: zero-event row contains an estimate or interval",
    )

    _require(
        frame.loc[singleton, "status"]
        .eq("INSUFFICIENT_CONDITIONAL_SESSIONS_FOR_INTERVAL")
        .all(),
        f"{name}: singleton status drifted",
    )
    singleton_estimate = pd.to_numeric(
        frame.loc[singleton, "estimate"], errors="coerce"
    ).to_numpy(float)
    _require(
        np.isfinite(singleton_estimate).all()
        and frame.loc[singleton, ["ci_95_low", "ci_95_high"]].isna().all().all(),
        f"{name}: singleton row lacks a finite estimate or has an interval",
    )
    _require(
        frame.loc[singleton & is_mean, "interval_method"]
        .eq("session_t_interval")
        .all()
        and frame.loc[singleton & is_quantile, "interval_method"]
        .eq("distribution_free_order_statistic")
        .all(),
        f"{name}: singleton interval method drifted",
    )

    _require(
        frame.loc[measured_mean, "status"].eq("MEASURED").all()
        and frame.loc[measured_mean, "interval_method"].eq("session_t_interval").all(),
        f"{name}: measured-mean labels drifted",
    )
    mean_values = frame.loc[
        measured_mean, ["estimate", "ci_95_low", "ci_95_high"]
    ].to_numpy(float)
    _require(
        np.isfinite(mean_values).all(),
        f"{name}: measured mean lacks a finite estimate or interval",
    )

    quantile_part = frame.loc[measured_quantile]
    _require(
        quantile_part["interval_method"].eq("exact_binomial_order_statistic").all(),
        f"{name}: measured-quantile interval method drifted",
    )
    quantile_n = pd.to_numeric(
        quantile_part["conditional_sessions"], errors="coerce"
    ).to_numpy(int)
    quantile_p = pd.to_numeric(
        quantile_part["quantile_probability"], errors="coerce"
    ).to_numpy(float)
    lower_supported = binom.ppf(0.025, quantile_n, quantile_p) >= 1.0
    upper_supported = binom.ppf(0.975, quantile_n, quantile_p) + 1.0 <= quantile_n
    expected_status = np.where(
        lower_supported & upper_supported,
        "MEASURED",
        "MEASURED_WITH_OPEN_CONFIDENCE_BOUND",
    )
    _require(
        np.array_equal(quantile_part["status"].astype(object).to_numpy(), expected_status),
        f"{name}: exact-quantile open-bound status drifted",
    )
    _require(
        np.array_equal(quantile_part["ci_95_low"].notna().to_numpy(), lower_supported)
        and np.array_equal(quantile_part["ci_95_high"].notna().to_numpy(), upper_supported),
        f"{name}: exact-quantile finite/open bounds do not match supported ranks",
    )


def _assert_monotone(
    frame: pd.DataFrame,
    *,
    group: Sequence[str],
    order: str,
    column: str,
    increasing: bool,
    name: str,
) -> None:
    for _, part in frame.groupby(list(group), dropna=False, sort=False):
        values = part.sort_values(order)[column].to_numpy(float)
        values = values[np.isfinite(values)]
        differences = np.diff(values)
        valid = differences >= -1e-12 if increasing else differences <= 1e-12
        _require(bool(valid.all()), f"{name}: monotonicity failed")


def validate_path_tables(
    corpus: TapeCorpus,
    race_surface: pd.DataFrame,
    overshoot_distribution: pd.DataFrame,
    time_to_event_distribution: pd.DataFrame,
    stop_compatibility: pd.DataFrame,
    conditional_quantiles: pd.DataFrame,
) -> dict[str, Any]:
    """Fail closed on every mathematical and population identity in the path census."""

    configurations = valid_configurations(corpus.minutes)
    frames = {
        "race_surface": race_surface,
        "overshoot_distribution": overshoot_distribution,
        "time_to_event_distribution": time_to_event_distribution,
        "stop_compatibility": stop_compatibility,
        "conditional_quantiles": conditional_quantiles,
    }
    for name, frame in frames.items():
        _require_exact_schema(frame, name=name)

    expected_race = (
        len(configurations)
        * len(DIRECTIONS)
        * len(FAVOURABLE_THRESHOLDS_POINTS)
        * len(ADVERSE_THRESHOLDS_POINTS)
        * len(REPORTING_SCOPES)
    )
    expected_overshoot = expected_race * len(OVERSHOOT_THRESHOLDS_POINTS)
    expected_time = (
        sum(horizon for _, _, horizon in configurations)
        * len(DIRECTIONS)
        * len(FAVOURABLE_THRESHOLDS_POINTS)
        * len(REPORTING_SCOPES)
    )
    expected_stop = expected_race
    statistics_per_distribution = 1 + len(DISTRIBUTION_QUANTILES)
    expected_quantiles = (
        len(configurations)
        * len(DIRECTIONS)
        * len(FAVOURABLE_THRESHOLDS_POINTS)
        * len(REPORTING_SCOPES)
        * statistics_per_distribution
        * (2 + len(ADVERSE_THRESHOLDS_POINTS))
    )
    expected_rows = {
        "race_surface": expected_race,
        "overshoot_distribution": expected_overshoot,
        "time_to_event_distribution": expected_time,
        "stop_compatibility": expected_stop,
        "conditional_quantiles": expected_quantiles,
    }
    actual_rows = {
        "race_surface": len(race_surface),
        "overshoot_distribution": len(overshoot_distribution),
        "time_to_event_distribution": len(time_to_event_distribution),
        "stop_compatibility": len(stop_compatibility),
        "conditional_quantiles": len(conditional_quantiles),
    }
    _require(actual_rows == expected_rows, f"path table row-count drift: {actual_rows}")

    race_key = (
        "scope",
        "direction",
        "start_time_et",
        "horizon_minutes",
        "favourable_threshold_points",
        "adverse_threshold_points",
    )
    _require_unique(race_surface, race_key, "race_surface")
    _require_unique(
        overshoot_distribution,
        (*race_key, "overshoot_threshold_points"),
        "overshoot_distribution",
    )
    event_key = race_key[:-1]
    _require_unique(
        time_to_event_distribution,
        (*event_key, "elapsed_minutes"),
        "time_to_event_distribution",
    )
    _require_unique(stop_compatibility, race_key, "stop_compatibility")
    _require_unique(
        conditional_quantiles,
        (*race_key, "metric", "statistic"),
        "conditional_quantiles",
    )

    valid_clock_cells = {(start_time, horizon) for start_time, _, horizon in configurations}
    for name, frame in frames.items():
        _require_domain(frame, "scope", REPORTING_SCOPES, name=name)
        _require_domain(frame, "direction", DIRECTIONS, name=name)
        _require_domain(
            frame,
            "favourable_threshold_points",
            FAVOURABLE_THRESHOLDS_POINTS,
            name=name,
        )
        observed_clock_cells = set(
            frame[["start_time_et", "horizon_minutes"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        _require(
            observed_clock_cells.issubset(valid_clock_cells),
            f"{name}: start/horizon key is outside the declared clock grid",
        )

    for name, frame in (
        ("race_surface", race_surface),
        ("overshoot_distribution", overshoot_distribution),
        ("stop_compatibility", stop_compatibility),
    ):
        _require_domain(
            frame,
            "adverse_threshold_points",
            ADVERSE_THRESHOLDS_POINTS,
            name=name,
        )

    _require(
        time_to_event_distribution["adverse_threshold_points"].isna().all(),
        "time_to_event_distribution: adverse threshold must be null for any-order timing",
    )
    _require_domain(
        overshoot_distribution,
        "overshoot_threshold_points",
        OVERSHOOT_THRESHOLDS_POINTS,
        name="overshoot_distribution",
    )
    elapsed = pd.to_numeric(
        time_to_event_distribution["elapsed_minutes"], errors="coerce"
    ).to_numpy(float)
    elapsed_horizon = pd.to_numeric(
        time_to_event_distribution["horizon_minutes"], errors="coerce"
    ).to_numpy(float)
    _require(np.isfinite(elapsed).all(), "time_to_event_distribution: elapsed minute is non-finite")
    _require(
        (elapsed == np.floor(elapsed)).all(),
        "time_to_event_distribution: elapsed minute is not integral",
    )
    _require(
        ((elapsed >= 1.0) & (elapsed <= elapsed_horizon)).all(),
        "time_to_event_distribution: elapsed minute is outside 1..horizon",
    )

    metric_values = (
        "time_to_favourable_touch",
        "adverse_excursion_before_favourable",
        "overshoot_after_favourable_first",
    )
    _require_domain(
        conditional_quantiles,
        "metric",
        metric_values,
        name="conditional_quantiles",
    )
    event_quantile_mask = conditional_quantiles["metric"].isin(metric_values[:2])
    overshoot_quantile_mask = conditional_quantiles["metric"].eq(metric_values[2])
    _require(
        conditional_quantiles.loc[
            event_quantile_mask, "adverse_threshold_points"
        ].isna().all(),
        "conditional_quantiles: any-order event metric has an adverse threshold",
    )
    _require_domain(
        conditional_quantiles.loc[overshoot_quantile_mask],
        "adverse_threshold_points",
        ADVERSE_THRESHOLDS_POINTS,
        name="conditional_quantiles.overshoot",
    )
    statistic_quantiles = {
        f"q{int(round(quantile * 100)):02d}": quantile
        for quantile in DISTRIBUTION_QUANTILES
    }
    _require_domain(
        conditional_quantiles,
        "statistic",
        ("mean", *statistic_quantiles),
        name="conditional_quantiles",
    )
    mean_rows = conditional_quantiles["statistic"].eq("mean")
    _require(
        conditional_quantiles.loc[mean_rows, "quantile_probability"].isna().all(),
        "conditional_quantiles: mean row has a quantile probability",
    )
    for statistic, quantile in statistic_quantiles.items():
        statistic_rows = conditional_quantiles["statistic"].eq(statistic)
        observed_quantile = pd.to_numeric(
            conditional_quantiles.loc[statistic_rows, "quantile_probability"],
            errors="coerce",
        ).to_numpy(float)
        _require(
            np.isfinite(observed_quantile).all()
            and np.allclose(observed_quantile, quantile, rtol=0.0, atol=1e-12),
            f"conditional_quantiles: {statistic} probability drifted",
        )

    masks = scope_masks(corpus)
    expected_denominators = {scope: int(mask.sum()) for scope, mask in masks.items()}
    for name, frame in (
        ("race_surface", race_surface),
        ("overshoot_distribution", overshoot_distribution),
        ("time_to_event_distribution", time_to_event_distribution),
        ("stop_compatibility", stop_compatibility),
        ("conditional_quantiles", conditional_quantiles),
    ):
        denominator_column = "sessions" if name == "race_surface" else "total_sessions"
        observed = frame.groupby("scope")[denominator_column].unique().to_dict()
        normalized = {key: [int(value) for value in values] for key, values in observed.items()}
        _require(
            normalized == {key: [value] for key, value in expected_denominators.items()},
            f"{name}: reporting denominators drifted: {normalized}",
        )

    for state in ("favourable_first", "adverse_first", "neither"):
        _require_probability_fields(
            race_surface,
            prefix=state,
            trials_column="sessions",
            name=f"race_surface.{state}",
        )
    _require(
        (
            race_surface["favourable_first_count"]
            + race_surface["adverse_first_count"]
            + race_surface["neither_count"]
            == race_surface["sessions"]
        ).all(),
        "race states do not partition sessions",
    )
    _require(
        np.allclose(
            race_surface[
                [
                    "favourable_first_probability",
                    "adverse_first_probability",
                    "neither_probability",
                ]
            ].sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ),
        "race probabilities do not sum to one",
    )

    fixed_j = [*race_key[:4], "favourable_threshold_points"]
    _assert_monotone(
        race_surface,
        group=fixed_j,
        order="adverse_threshold_points",
        column="favourable_first_count",
        increasing=True,
        name="race favourable-first over J",
    )
    _assert_monotone(
        race_surface,
        group=fixed_j,
        order="adverse_threshold_points",
        column="adverse_first_count",
        increasing=False,
        name="race adverse-first over J",
    )
    _assert_monotone(
        race_surface,
        group=fixed_j,
        order="adverse_threshold_points",
        column="neither_count",
        increasing=True,
        name="race neither over J",
    )
    fixed_m = [*race_key[:4], "adverse_threshold_points"]
    _assert_monotone(
        race_surface,
        group=fixed_m,
        order="favourable_threshold_points",
        column="favourable_first_count",
        increasing=False,
        name="race favourable-first over M",
    )
    _assert_monotone(
        race_surface,
        group=fixed_m,
        order="favourable_threshold_points",
        column="adverse_first_count",
        increasing=True,
        name="race adverse-first over M",
    )
    _assert_monotone(
        race_surface,
        group=fixed_m,
        order="favourable_threshold_points",
        column="neither_count",
        increasing=True,
        name="race neither over M",
    )
    fixed_horizon = [
        "scope",
        "direction",
        "start_time_et",
        "favourable_threshold_points",
        "adverse_threshold_points",
    ]
    for state, increasing in (
        ("favourable_first_count", True),
        ("adverse_first_count", True),
        ("neither_count", False),
    ):
        _assert_monotone(
            race_surface,
            group=fixed_horizon,
            order="horizon_minutes",
            column=state,
            increasing=increasing,
            name=f"race {state} over horizon",
        )

    calls = race_surface[race_surface["direction"] == "call"].copy()
    puts = race_surface[race_surface["direction"] == "put"].copy()
    puts = puts.rename(
        columns={
            "favourable_threshold_points": "adverse_threshold_points",
            "adverse_threshold_points": "favourable_threshold_points",
            "favourable_first_count": "put_favourable_first_count",
            "adverse_first_count": "put_adverse_first_count",
            "neither_count": "put_neither_count",
        }
    )
    puts["direction"] = "call"
    symmetry = calls.merge(
        puts[
            [
                *race_key,
                "put_favourable_first_count",
                "put_adverse_first_count",
                "put_neither_count",
            ]
        ],
        on=list(race_key),
        how="left",
        validate="one_to_one",
    )
    _require(len(symmetry) == len(calls), "call/put symmetry join is incomplete")
    _require(
        (
            symmetry["favourable_first_count"]
            == symmetry["put_adverse_first_count"]
        ).all()
        and (
            symmetry["adverse_first_count"]
            == symmetry["put_favourable_first_count"]
        ).all()
        and (symmetry["neither_count"] == symmetry["put_neither_count"]).all(),
        "swapped-threshold call/put identity failed",
    )

    for prefix in ("at_or_above",):
        _require_probability_fields(
            overshoot_distribution,
            prefix=prefix,
            trials_column="conditional_sessions",
            name=f"overshoot_distribution.{prefix}",
            allow_empty=True,
        )
    _require_conditioning_fields(
        overshoot_distribution, name="overshoot_distribution.conditioning"
    )
    _require_literal(
        overshoot_distribution,
        "conditioning",
        "favourable_first",
        name="overshoot_distribution",
    )
    _require_conditional_probability_status(
        overshoot_distribution, name="overshoot_distribution"
    )
    overshoot_join = overshoot_distribution.merge(
        race_surface[[*race_key, "favourable_first_count"]],
        on=list(race_key),
        how="left",
        validate="many_to_one",
    )
    _require(
        (
            overshoot_join["conditional_sessions"]
            == overshoot_join["favourable_first_count"]
        ).all(),
        "overshoot population differs from favourable-first race state",
    )
    zero_tail = overshoot_distribution[
        overshoot_distribution["overshoot_threshold_points"] == 0.0
    ]
    _require(
        (
            zero_tail["at_or_above_count"] == zero_tail["conditional_sessions"]
        ).all(),
        "favourable-first overshoot is negative",
    )
    _assert_monotone(
        overshoot_distribution,
        group=race_key,
        order="overshoot_threshold_points",
        column="at_or_above_count",
        increasing=False,
        name="overshoot survival curve",
    )

    _require_probability_fields(
        time_to_event_distribution,
        prefix="reached_by",
        trials_column="conditional_sessions",
        name="time_to_event_distribution.reached_by",
        allow_empty=True,
    )
    _require_conditioning_fields(
        time_to_event_distribution, name="time_to_event_distribution.conditioning"
    )
    _require_literal(
        time_to_event_distribution,
        "conditioning",
        "favourable_reached_any_order",
        name="time_to_event_distribution",
    )
    _require_conditional_probability_status(
        time_to_event_distribution, name="time_to_event_distribution"
    )
    _assert_monotone(
        time_to_event_distribution,
        group=event_key,
        order="elapsed_minutes",
        column="reached_by_count",
        increasing=True,
        name="time-to-event CDF",
    )
    time_terminal = time_to_event_distribution[
        time_to_event_distribution["elapsed_minutes"]
        == time_to_event_distribution["horizon_minutes"]
    ]
    _require(
        (
            time_terminal["reached_by_count"]
            == time_terminal["conditional_sessions"]
        ).all(),
        "time-to-event CDF does not end at one",
    )

    event_counts = time_terminal[[*event_key, "conditional_sessions"]].rename(
        columns={"conditional_sessions": "event_hit_count"}
    )
    time_binding = time_to_event_distribution.merge(
        event_counts,
        on=list(event_key),
        how="left",
        validate="many_to_one",
    )
    _require(
        (time_binding["conditional_sessions"] == time_binding["event_hit_count"]).all(),
        "time-to-event population differs from eventual favourable hits",
    )

    _require_probability_fields(
        stop_compatibility,
        prefix="favourable_reached",
        trials_column="total_sessions",
        name="stop_compatibility.favourable_reached",
    )
    _require(
        (
            stop_compatibility["conditional_sessions"]
            == stop_compatibility["favourable_reached_count"]
        ).all(),
        "stop conditional sessions differ from favourable-reached count",
    )
    stop_event_binding = stop_compatibility.merge(
        event_counts,
        on=list(event_key),
        how="left",
        validate="many_to_one",
    )
    _require(
        (
            (stop_event_binding["conditional_sessions"] == stop_event_binding["event_hit_count"])
            & (
                stop_event_binding["favourable_reached_count"]
                == stop_event_binding["event_hit_count"]
            )
        ).all(),
        "stop favourable-reached population differs from event hits at some J",
    )
    _require_literal(
        stop_compatibility,
        "stop_rule",
        "touching -J before +M kills; survival requires pre-M adverse < J",
        name="stop_compatibility",
    )
    _require_literal(
        stop_compatibility,
        "option_percent_stop_mapping",
        "UNKNOWN_NOT_OPTION_PNL",
        name="stop_compatibility",
    )
    _require_conditional_probability_status(
        stop_compatibility, name="stop_compatibility"
    )
    for prefix in ("survives_to_favourable", "killed_before_favourable"):
        _require_probability_fields(
            stop_compatibility,
            prefix=prefix,
            trials_column="conditional_sessions",
            name=f"stop_compatibility.{prefix}",
            allow_empty=True,
        )
    _require(
        (
            stop_compatibility["survives_to_favourable_count"]
            + stop_compatibility["killed_before_favourable_count"]
            == stop_compatibility["conditional_sessions"]
        ).all(),
        "stop outcomes do not partition eventual favourable hits",
    )
    stop_join = stop_compatibility.merge(
        race_surface[[*race_key, "favourable_first_count"]],
        on=list(race_key),
        how="left",
        validate="one_to_one",
    )
    _require(
        (
            stop_join["survives_to_favourable_count"]
            == stop_join["favourable_first_count"]
        ).all(),
        "strict stop survival does not equal favourable-first race count",
    )
    _assert_monotone(
        stop_compatibility,
        group=fixed_j,
        order="adverse_threshold_points",
        column="survives_to_favourable_count",
        increasing=True,
        name="stop survival over J",
    )

    _require_conditioning_fields(
        conditional_quantiles, name="conditional_quantiles.conditioning"
    )
    event_distribution_rows = conditional_quantiles["metric"].isin(
        ("time_to_favourable_touch", "adverse_excursion_before_favourable")
    )
    _require_literal(
        conditional_quantiles.loc[event_distribution_rows],
        "conditioning",
        "favourable_reached_any_order",
        name="conditional_quantiles.event",
    )
    _require_literal(
        conditional_quantiles.loc[~event_distribution_rows],
        "conditioning",
        "favourable_first",
        name="conditional_quantiles.overshoot",
    )
    _require_literal(
        conditional_quantiles.loc[
            conditional_quantiles["metric"].eq("time_to_favourable_touch")
        ],
        "unit",
        "minutes",
        name="conditional_quantiles.time",
    )
    _require_literal(
        conditional_quantiles.loc[
            ~conditional_quantiles["metric"].eq("time_to_favourable_touch")
        ],
        "unit",
        "spx_points",
        name="conditional_quantiles.points",
    )
    _require_distribution_status(
        conditional_quantiles, name="conditional_quantiles"
    )
    event_quantiles = conditional_quantiles[
        conditional_quantiles["metric"].isin(
            ("time_to_favourable_touch", "adverse_excursion_before_favourable")
        )
    ].merge(
        event_counts,
        on=list(event_key),
        how="left",
        validate="many_to_one",
    )
    _require(
        (
            event_quantiles["conditional_sessions"]
            == event_quantiles["event_hit_count"]
        ).all(),
        "event-distribution population differs from eventual favourable hits",
    )
    overshoot_quantiles = conditional_quantiles[
        conditional_quantiles["metric"] == "overshoot_after_favourable_first"
    ].merge(
        race_surface[[*race_key, "favourable_first_count"]],
        on=list(race_key),
        how="left",
        validate="many_to_one",
    )
    _require(
        (
            overshoot_quantiles["conditional_sessions"]
            == overshoot_quantiles["favourable_first_count"]
        ).all(),
        "overshoot-quantile population differs from favourable-first hits",
    )

    measured_quantiles = conditional_quantiles[
        conditional_quantiles["status"].astype(str).str.startswith("MEASURED")
    ]
    _require(
        np.isfinite(measured_quantiles["estimate"].to_numpy(float)).all(),
        "conditional distribution contains a non-finite estimate",
    )
    finite_low = measured_quantiles["ci_95_low"].notna()
    finite_high = measured_quantiles["ci_95_high"].notna()
    _require(
        (
            measured_quantiles.loc[finite_low, "ci_95_low"]
            <= measured_quantiles.loc[finite_low, "estimate"]
        ).all()
        and (
            measured_quantiles.loc[finite_high, "estimate"]
            <= measured_quantiles.loc[finite_high, "ci_95_high"]
        ).all(),
        "conditional distribution interval does not contain its estimate",
    )
    closed = measured_quantiles["status"] == "MEASURED"
    open_bound = measured_quantiles["status"] == "MEASURED_WITH_OPEN_CONFIDENCE_BOUND"
    _require(
        (
            measured_quantiles.loc[closed, "ci_95_low"].notna()
            & measured_quantiles.loc[closed, "ci_95_high"].notna()
        ).all(),
        "closed distribution interval is missing a bound",
    )
    _require(
        (
            measured_quantiles.loc[open_bound, ["ci_95_low", "ci_95_high"]]
            .isna()
            .any(axis=1)
        ).all(),
        "open-bound distribution row has two finite bounds",
    )
    quantiles_only = conditional_quantiles[
        conditional_quantiles["quantile_probability"].notna()
    ]
    quantile_group = [*race_key, "metric"]
    _assert_monotone(
        quantiles_only,
        group=quantile_group,
        order="quantile_probability",
        column="estimate",
        increasing=True,
        name="distribution quantiles",
    )
    nonnegative_metrics = conditional_quantiles[
        conditional_quantiles["metric"].isin(
            (
                "overshoot_after_favourable_first",
                "adverse_excursion_before_favourable",
                "time_to_favourable_touch",
            )
        )
        & conditional_quantiles["estimate"].notna()
    ]
    _require(
        (nonnegative_metrics["estimate"] >= 0.0).all(),
        "path distribution contains a negative value",
    )

    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "discovered_tape_sessions": int(len(corpus.input_manifest)),
        "analyzed_sessions": len(corpus.sessions),
        "scope_sessions": expected_denominators,
        "path_table_rows": actual_rows,
        "race_states": ["favourable_first", "adverse_first", "neither"],
        "impossible_same_snapshot_ties": 0,
        "three_state_partition": "PASS",
        "probabilities_reproduced_from_counts": "PASS",
        "intervals_contain_estimates": "PASS",
        "open_exact_quantile_intervals": int(open_bound.sum()),
        "threshold_and_horizon_monotonicity": "PASS",
        "swapped_threshold_call_put_identity": "PASS",
        "stop_survival_equals_favourable_first": "PASS",
        "session_unit_inference": "PASS",
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def clean(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _probability_text(row: pd.Series, prefix: str) -> str:
    count = int(row[f"{prefix}_count"])
    probability = 100.0 * float(row[f"{prefix}_probability"])
    low = 100.0 * float(row[f"{prefix}_ci_95_low"])
    high = 100.0 * float(row[f"{prefix}_ci_95_high"])
    return f"{probability:.1f}% [{low:.1f}, {high:.1f}] (n={count})"


def _estimate_text(row: pd.Series, *, scale: float = 1.0, digits: int = 1) -> str:
    estimate = row.get("estimate")
    if pd.isna(estimate):
        return "UNKNOWN"
    low = row.get("ci_95_low")
    high = row.get("ci_95_high")
    low_text = "open" if pd.isna(low) else f"{scale * float(low):.{digits}f}"
    high_text = "open" if pd.isna(high) else f"{scale * float(high):.{digits}f}"
    return f"{scale * float(estimate):.{digits}f} [{low_text}, {high_text}]"


def _one(frame: pd.DataFrame, **conditions: Any) -> pd.Series:
    selected = frame
    for column, value in conditions.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise UnconditionalRaceError(
            f"readable-table selection expected one row, found {len(selected)}: {conditions}"
        )
    return selected.iloc[0]


def render_readable_tables(tables: AnalysisTables, quality_control: Mapping[str, Any]) -> str:
    """Render an owner-readable subset while preserving the full machine grids."""

    lines = [
        "# Unconditional SPX race and IV clock tables",
        "",
        "**[VERIFIED] Population:** 1,011 pre-reservation sessions; 768 backfill and "
        "243 owned. Each probability contributes one Bernoulli observation per session. "
        "Brackets are pointwise 95% session-unit intervals, never overlapping-window or "
        "contract-row intervals.",
        "",
        "**[VERIFIED] Touch law:** entry is the exact observation at `t`; the path uses "
        "one-minute parity snapshots `t+1..t+H`. A touch is first *observed snapshot* "
        "crossing, not an intraminute high/low. Cells ending after 16:00 are unobservable, "
        "not truncated misses.",
        "",
        "**[UNKNOWN] Option outcome:** `NEITHER` means neither underlying barrier was "
        "observed by the horizon. It is not proof that an option expired, hit a percentage "
        "stop, or lost money. No option P&L, label, exit, signal, or fitted value was read.",
        "",
        "## Selected pooled 20-minute three-state races",
        "",
        "The full 38,400-row surface is in `race_surface.csv`. Every outward cell has "
        "exactly the three states below; impossible same-snapshot double touches remain a "
        "zero-valued QC assertion rather than a fourth state.",
        "",
    ]
    selected_starts = ("09:35", "11:30", "13:30", "15:00", "15:30")
    selected_races = ((5.0, 5.0), (10.0, 5.0), (10.0, 10.0), (20.0, 10.0), (20.0, 20.0))
    race_rows: list[list[str]] = []
    for start in selected_starts:
        for direction in DIRECTIONS:
            for favourable, adverse in selected_races:
                row = _one(
                    tables.race_surface,
                    scope="pooled",
                    direction=direction,
                    start_time_et=start,
                    horizon_minutes=20,
                    favourable_threshold_points=favourable,
                    adverse_threshold_points=adverse,
                )
                race_rows.append(
                    [
                        start,
                        direction,
                        f"+{favourable:g}/-{adverse:g}",
                        _probability_text(row, "favourable_first"),
                        _probability_text(row, "adverse_first"),
                        _probability_text(row, "neither"),
                    ]
                )
    lines.extend(
        [
            _markdown_table(
                ("Start ET", "Side", "Race M/J", "Favourable first", "Adverse first", "Neither"),
                race_rows,
            ),
            "",
            "## Conditional winner path: +10 before -5 over 20 minutes",
            "",
            "Overshoot is the maximum favourable excursion over the *whole remaining "
            "horizon* minus 10 points, conditioned on +10 arriving before -5. Time and "
            "pre-favourable adverse excursion are conditioned on an eventual +10 touch in "
            "either order. They are expectancy inputs, not realized option P&L.",
            "",
        ]
    )
    path_rows: list[list[str]] = []
    for start in selected_starts:
        for direction in DIRECTIONS:
            selected: dict[tuple[str, str], pd.Series] = {}
            for metric in (
                "overshoot_after_favourable_first",
                "time_to_favourable_touch",
                "adverse_excursion_before_favourable",
            ):
                adverse_value = 5.0 if metric == "overshoot_after_favourable_first" else None
                source = tables.conditional_quantiles[
                    tables.conditional_quantiles["scope"].eq("pooled")
                    & tables.conditional_quantiles["direction"].eq(direction)
                    & tables.conditional_quantiles["start_time_et"].eq(start)
                    & tables.conditional_quantiles["horizon_minutes"].eq(20)
                    & tables.conditional_quantiles["favourable_threshold_points"].eq(10.0)
                    & tables.conditional_quantiles["metric"].eq(metric)
                ]
                source = (
                    source[source["adverse_threshold_points"].eq(adverse_value)]
                    if adverse_value is not None
                    else source[source["adverse_threshold_points"].isna()]
                )
                for statistic in ("q50", "q90", "q99"):
                    statistic_rows = source[source["statistic"].eq(statistic)]
                    if len(statistic_rows) == 1:
                        selected[(metric, statistic)] = statistic_rows.iloc[0]
            overshoot_q50 = selected[("overshoot_after_favourable_first", "q50")]
            path_rows.append(
                [
                    start,
                    direction,
                    str(int(overshoot_q50["conditional_sessions"])),
                    _estimate_text(overshoot_q50),
                    _estimate_text(selected[("overshoot_after_favourable_first", "q90")]),
                    _estimate_text(selected[("overshoot_after_favourable_first", "q99")]),
                    _estimate_text(selected[("time_to_favourable_touch", "q50")]),
                    _estimate_text(selected[("time_to_favourable_touch", "q90")]),
                    _estimate_text(selected[("adverse_excursion_before_favourable", "q50")]),
                    _estimate_text(selected[("adverse_excursion_before_favourable", "q90")]),
                ]
            )
    lines.extend(
        [
            _markdown_table(
                (
                    "Start ET",
                    "Side",
                    "+10-first n",
                    "Overshoot q50 pts",
                    "Overshoot q90 pts",
                    "Overshoot q99 pts",
                    "Time q50 min",
                    "Time q90 min",
                    "Pre-M adverse q50 pts",
                    "Pre-M adverse q90 pts",
                ),
                path_rows,
            ),
            "",
            "Exact order-statistic quantile intervals keep an endpoint `open` when the "
            "conditional sample cannot support a finite 95% tail bound. The fixed-threshold "
            "survival/CDF CSVs provide Wilson intervals throughout the tails.",
            "",
            "## Underlying stop compatibility for an eventual +10 touch",
            "",
            "A -J underlying stop survives only when pre-+10 adverse excursion is strictly "
            "less than J; touching -J kills. The survivor count is required by QC to equal "
            "the corresponding +10-before-J race count exactly. Mapping J points to a "
            "-40% option stop is **UNKNOWN** without option path/P&L data.",
            "",
        ]
    )
    stop_rows: list[list[str]] = []
    for start in selected_starts:
        for direction in DIRECTIONS:
            for adverse in (5.0, 10.0, 20.0, 40.0):
                row = _one(
                    tables.stop_compatibility,
                    scope="pooled",
                    direction=direction,
                    start_time_et=start,
                    horizon_minutes=20,
                    favourable_threshold_points=10.0,
                    adverse_threshold_points=adverse,
                )
                stop_rows.append(
                    [
                        start,
                        direction,
                        f"{adverse:g}",
                        str(int(row["favourable_reached_count"])),
                        _probability_text(row, "survives_to_favourable"),
                        _probability_text(row, "killed_before_favourable"),
                    ]
                )
    lines.extend(
        [
            _markdown_table(
                ("Start ET", "Side", "Stop J pts", "Eventual +10 n", "Survives to +10", "Killed first"),
                stop_rows,
            ),
            "",
            "## Implied volatility by exact time",
            "",
            "Values below are IV percentages, although machine CSVs retain volatility "
            "fractions. Primary = equal-weighted call/put median IV within +/-10 SPX "
            "points. Sensitivity = call/put paired at the same nearest physical strike. "
            "Solver-bound values are excluded. This is an ATM level clock curve only: "
            "40- and 60-point OTM IV are outside the stored +/-25 ladder and remain UNKNOWN.",
            "",
        ]
    )
    iv_rows: list[list[str]] = []
    for start in START_TIMES:
        cells: dict[tuple[str, str], pd.Series] = {}
        for region, estimand in (
            ("atm_10pt_primary", "call_median"),
            ("atm_10pt_primary", "put_median"),
            ("atm_10pt_primary", "side_balanced"),
            ("closest_atm_pair_sensitivity", "side_balanced"),
        ):
            cells[(region, estimand)] = _one(
                tables.iv_by_time,
                scope="pooled",
                start_time_et=start,
                region=region,
                estimand=estimand,
                statistic="q50",
            )
        primary = cells[("atm_10pt_primary", "side_balanced")]
        iv_rows.append(
            [
                start,
                str(int(primary["usable_sessions"])),
                _estimate_text(cells[("atm_10pt_primary", "call_median")], scale=100.0, digits=2),
                _estimate_text(cells[("atm_10pt_primary", "put_median")], scale=100.0, digits=2),
                _estimate_text(primary, scale=100.0, digits=2),
                _estimate_text(
                    cells[("closest_atm_pair_sensitivity", "side_balanced")],
                    scale=100.0,
                    digits=2,
                ),
            ]
        )
    lines.extend(
        [
            _markdown_table(
                ("Start ET", "Sessions", "Call ATM10 q50 %", "Put ATM10 q50 %", "Balanced ATM10 q50 %", "Closest-pair q50 %"),
                iv_rows,
            ),
            "",
            "## Era description (not a stability test)",
            "",
            "Acquisition source changes exactly at the era boundary, so source and date "
            "are confounded. These pointwise gaps are descriptive; stability remains UNKNOWN.",
            "",
        ]
    )
    era_rows: list[list[str]] = []
    for start in selected_starts:
        backfill = _one(
            tables.iv_by_time,
            scope="era_backfill_2022-06-01_to_2025-07-31",
            start_time_et=start,
            region="atm_10pt_primary",
            estimand="side_balanced",
            statistic="q50",
        )
        owned = _one(
            tables.iv_by_time,
            scope="era_owned_2025-08-01_to_2026-07-30",
            start_time_et=start,
            region="atm_10pt_primary",
            estimand="side_balanced",
            statistic="q50",
        )
        era_rows.append(
            [
                start,
                _estimate_text(backfill, scale=100.0, digits=2),
                _estimate_text(owned, scale=100.0, digits=2),
            ]
        )
    lines.extend(
        [
            _markdown_table(("Start ET", "Backfill ATM10 q50 %", "Owned ATM10 q50 %"), era_rows),
            "",
            "## Audit summary",
            "",
            f"- Path QC: `{quality_control['path']['status']}`; table rows "
            f"`{quality_control['path']['path_table_rows']}`.",
            f"- IV QC: `{quality_control['iv']['status']}`; session values "
            f"`{quality_control['iv']['iv_session_value_rows']}`.",
            "- No fit, conditioning, option outcome/P&L read, reserved session, vendor/broker "
            "contact, purchase, order, strategy proposal, or adoption occurred.",
            "- **ADOPT NOTHING.**",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(
    tape_root: Path,
    ladder_root: Path,
    *,
    strict_population: bool = True,
) -> RaceAnalysisRun:
    """Run the complete outcome-blind path and causal-ladder census in memory."""

    from v5.research import unconditional_spx_iv as iv_analysis

    _require(
        tuple(START_TIMES) == tuple(iv_analysis.START_TIMES),
        "path and IV clock declarations differ",
    )
    tape = load_tape_corpus(Path(tape_root), strict_population=strict_population)
    iv_census = iv_analysis.load_iv_census(
        Path(ladder_root),
        tape.sessions,
        tape.dates,
        strict_population=strict_population,
    )
    tape_analyzed = set(
        tape.input_manifest.loc[
            tape.input_manifest["status"].eq("ANALYZED"), "session"
        ].astype(str)
    )
    ladder_analyzed = set(
        iv_census.input_manifest.loc[
            iv_census.input_manifest["status"].eq("ANALYZED"), "session"
        ].astype(str)
    )
    _require(
        tape_analyzed == ladder_analyzed == set(tape.sessions),
        "tape and ladder analyzed populations differ",
    )

    path_frames = analyze_paths(tape)
    path_quality = validate_path_tables(tape, *path_frames)
    iv_by_time, iv_clock_change, iv_session_values = iv_analysis.summarize_iv(iv_census)
    iv_quality = iv_analysis.validate_iv_tables(
        iv_census, iv_by_time, iv_clock_change
    )
    tables = AnalysisTables(
        race_surface=path_frames[0],
        overshoot_distribution=path_frames[1],
        time_to_event_distribution=path_frames[2],
        stop_compatibility=path_frames[3],
        conditional_quantiles=path_frames[4],
        iv_by_time=iv_by_time,
        iv_clock_change=iv_clock_change,
        iv_session_values=iv_session_values,
    )
    quality_control: dict[str, Any] = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "path": path_quality,
        "iv": iv_quality,
        "population_alignment": "PASS",
        "analyzed_sessions": len(tape.sessions),
        "backfill_sessions": sum(value < ERA_CUTOFF for value in tape.dates),
        "owned_sessions": sum(value >= ERA_CUTOFF for value in tape.dates),
        "reserved_sessions_used": 0,
        "conditioning": "NONE_EXACT_CLOCK_AND_HORIZON_ONLY",
        "option_outcome_or_pnl_read": False,
        "model_fit": False,
        "adoption": "ADOPT_NOTHING",
    }
    input_manifest = pd.concat(
        [tape.input_manifest, iv_census.input_manifest],
        ignore_index=True,
        sort=False,
    ).sort_values(["input_family", "session"]).reset_index(drop=True)
    _require(
        not input_manifest.duplicated(["input_family", "session"]).any(),
        "combined input manifest contains duplicate family/session keys",
    )
    defect_disposition = pd.DataFrame(tape.defect_disposition)
    readable = render_readable_tables(tables, quality_control)
    return RaceAnalysisRun(
        tables=tables,
        quality_control=quality_control,
        input_manifest=input_manifest,
        defect_disposition=defect_disposition,
        readable_tables=readable,
    )
