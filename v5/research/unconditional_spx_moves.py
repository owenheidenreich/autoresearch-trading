"""Unconditional SPX move terrain on the causal parity-spot tape.

This module measures the market, not a strategy.  Every window is selected only
by its clock time and horizon.  There is no feature, indicator, signal, prior
state, option outcome, fitted quantity, or threshold search anywhere in the
data path.

The tape contains one SPX parity snapshot per minute.  Therefore every touch in
this module means "observed at a one-minute snapshot"; an intra-minute touch can
be missed and is never invented from nonexistent candle highs or lows.

Inference is session-level.  A grid cell has at most one window per session, so
its Bernoulli observations are sessions, not the many overlapping windows in
the whole grid.  Pointwise Wilson score intervals are consequently
session-clustered by construction.  Correlation *between* grid cells is real
and is disclosed rather than turned into a best-cell significance claim.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from v5.research.contract_economics import contract_economics


SCHEMA_VERSION = "v5.unconditional-spx-moves.v1"
CONFIRMATION_START = date(2026, 8, 6)
ERA_CUTOFF = date(2025, 8, 1)

EXPECTED_DISCOVERED_SESSIONS = 1_014
EXPECTED_ANALYZED_SESSIONS = 1_011
EXPECTED_FIRST_SESSION = date(2022, 6, 1)
EXPECTED_LAST_SESSION = date(2026, 7, 30)

SESSION_FILE = re.compile(r"^(20\d{2}-\d{2}-\d{2})\.es_c_0\.ohlcv-1m\.parquet$")
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

# The padded early close is already absent from the finished parity tape.  The
# three interior freezes remain present and must be removed here before a price
# is read from them.
KNOWN_DEFECTS: Mapping[date, str] = {
    date(2022, 11, 25): "vendor-padded early close; excluded upstream",
    date(2023, 6, 26): "interior whole-book freeze (2 minutes)",
    date(2023, 10, 19): "interior whole-book freeze (3 minutes)",
    date(2023, 10, 25): "interior whole-book freezes (4 and 18 minutes)",
}

HORIZONS_MINUTES = (5, 10, 15, 20, 30, 45, 60, 90)
MOVE_THRESHOLDS_POINTS = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
ADVERSE_THRESHOLDS_POINTS = (2.0, 5.0, 10.0, 20.0)

# Exact observation minutes: finer around the open and close, at least hourly
# through the middle.  The first usable snapshot is 09:31, not an invented
# 09:30 value.
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
CI_Z = 1.959963984540054

# Fixed companion scenario from the separately owned contract-economics work.
# The module itself owns the repricing law.  These are the table assumptions,
# not fitted values and not a claim about every historical surface.
COMPANION_SPOT = 6_800.0
COMPANION_SIGMA = 0.13
COMPANION_OTM_POINTS = (0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0)
COMPANION_ATM_SPREAD = 0.10
COMPANION_OTM_SPREAD = 0.20
SIGNED_TICKET_CAP_USD = 2_500.0


class UnconditionalMoveError(RuntimeError):
    """The unconditional census cannot prove its input or statistical law."""


def minute_number(value: str) -> int:
    """Minutes after midnight for a strict ``HH:MM`` clock label."""

    try:
        hour, minute = (int(piece) for piece in value.split(":"))
    except Exception as exc:  # noqa: BLE001 - normalize to one fail-closed error
        raise UnconditionalMoveError(f"invalid clock label {value!r}") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise UnconditionalMoveError(f"invalid clock label {value!r}")
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


def minutes_to_close(start_time: str) -> int:
    """Actual minutes from an observation time to the 16:00 SPXW close."""

    remaining = minute_number("16:00") - minute_number(start_time)
    if remaining < 0:
        raise UnconditionalMoveError(f"start is after the close: {start_time}")
    return remaining


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TapeCorpus:
    """The cleaned parity tape and the manifest that proves its population."""

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
            raise UnconditionalMoveError(
                f"price matrix shape {self.prices.shape} does not match {expected}"
            )
        if len(self.dates) != len(self.sessions):
            raise UnconditionalMoveError("session/date lengths differ")


def _read_one(path: Path, session: str) -> np.ndarray:
    try:
        frame = pd.read_parquet(path, columns=list(TAPE_COLUMNS))
    except Exception as exc:  # noqa: BLE001 - unreadable input is a hard failure
        raise UnconditionalMoveError(f"{session}: tape unreadable: {exc}") from exc
    if len(frame) != len(EXPECTED_MINUTES):
        raise UnconditionalMoveError(
            f"{session}: expected {len(EXPECTED_MINUTES)} snapshots, found {len(frame)}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise UnconditionalMoveError(f"{session}: ts_event is not a timezone-aware index")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise UnconditionalMoveError(f"{session}: ts_event is duplicated or out of order")
    local_index = frame.index.tz_convert("America/New_York")
    if set(local_index.strftime("%Y-%m-%d")) != {session}:
        raise UnconditionalMoveError(f"{session}: filename and local ts_event date disagree")
    bar_minutes = tuple(local_index.strftime("%H:%M"))
    if bar_minutes != EXPECTED_BAR_MINUTES:
        raise UnconditionalMoveError(f"{session}: bar-label clock is not 09:30 through 15:59 ET")
    minutes = tuple(frame["bar_observation_minute"].astype(str))
    if minutes != EXPECTED_MINUTES:
        raise UnconditionalMoveError(
            f"{session}: observation clock is incomplete, duplicated, or out of order"
        )
    observed_from_index = tuple((local_index + pd.Timedelta(minutes=1)).strftime("%H:%M"))
    if observed_from_index != minutes:
        raise UnconditionalMoveError(
            f"{session}: observation minute is not exactly one minute after ts_event"
        )
    sources = set(frame["tape_source"].astype(str))
    if sources != {TAPE_SOURCE}:
        raise UnconditionalMoveError(
            f"{session}: tape_source is {sorted(sources)}, not {TAPE_SOURCE}"
        )
    close = frame["close"].to_numpy(float)
    if not np.isfinite(close).all() or (close <= 0.0).any():
        raise UnconditionalMoveError(f"{session}: close contains non-finite/non-positive values")
    for column in ("open", "high", "low"):
        values = frame[column].to_numpy(float)
        if not np.array_equal(values, close):
            raise UnconditionalMoveError(
                f"{session}: {column} differs from close; the parity tape is snapshot-only"
            )
    if not np.array_equal(frame["volume"].to_numpy(float), np.zeros(len(frame))):
        raise UnconditionalMoveError(f"{session}: parity-index volume is not identically zero")
    return close


def load_tape_corpus(root: Path, *, strict_population: bool = True) -> TapeCorpus:
    """Read the fixed tape, excluding known defective sessions before prices open."""

    root = Path(root)
    if not root.is_dir():
        raise UnconditionalMoveError(f"no parity-tape directory at {root}")
    discovered: list[tuple[date, Path]] = []
    unexpected: list[str] = []
    for path in root.glob("*.parquet"):
        match = SESSION_FILE.match(path.name)
        if not match:
            unexpected.append(path.name)
            continue
        discovered.append((date.fromisoformat(match.group(1)), path))
    discovered.sort(key=lambda pair: pair[0])
    if unexpected:
        raise UnconditionalMoveError(f"unexpected parquet names: {sorted(unexpected)[:3]}")
    if not discovered:
        raise UnconditionalMoveError("parity tape contains no session")
    dates = [session_date for session_date, _ in discovered]
    if len(dates) != len(set(dates)):
        raise UnconditionalMoveError("parity tape contains a duplicate session date")
    if any(session_date >= CONFIRMATION_START for session_date in dates):
        first = min(d for d in dates if d >= CONFIRMATION_START)
        raise UnconditionalMoveError(f"confirmation-reserved session present: {first}")
    if strict_population:
        facts = (len(discovered), dates[0], dates[-1])
        expected = (
            EXPECTED_DISCOVERED_SESSIONS,
            EXPECTED_FIRST_SESSION,
            EXPECTED_LAST_SESSION,
        )
        if facts != expected:
            raise UnconditionalMoveError(
                f"tape population drifted: found {facts}, expected {expected}"
            )

    manifest_rows: list[dict[str, Any]] = []
    clean_sessions: list[str] = []
    clean_dates: list[date] = []
    clean_prices: list[np.ndarray] = []
    disposition: list[dict[str, Any]] = []
    present_dates = set(dates)
    for session_date, path in discovered:
        session = session_date.isoformat()
        row = {
            "session": session,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "calendar_year": session_date.year,
            "source_era": "backfill" if session_date < ERA_CUTOFF else "owned",
        }
        reason = KNOWN_DEFECTS.get(session_date)
        if reason:
            row.update(status="EXCLUDED_KNOWN_DEFECT", reason=reason)
            disposition.append(
                {"session": session, "present": True, "disposition": "EXCLUDED", "reason": reason}
            )
        else:
            values = _read_one(path, session)
            row.update(status="ANALYZED", reason="")
            clean_sessions.append(session)
            clean_dates.append(session_date)
            clean_prices.append(values)
        manifest_rows.append(row)

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
        raise UnconditionalMoveError(
            f"clean population drifted: {len(clean_sessions)}, expected "
            f"{EXPECTED_ANALYZED_SESSIONS}"
        )
    return TapeCorpus(
        root=root,
        sessions=tuple(clean_sessions),
        dates=tuple(clean_dates),
        prices=np.vstack(clean_prices),
        minutes=EXPECTED_MINUTES,
        input_manifest=pd.DataFrame(manifest_rows).sort_values("session").reset_index(drop=True),
        defect_disposition=tuple(sorted(disposition, key=lambda row: row["session"])),
    )


def wilson_interval(successes: int, trials: int, *, z: float = CI_Z) -> tuple[float, float]:
    """Two-sided Wilson score interval on session-level Bernoulli outcomes."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise UnconditionalMoveError(
            f"invalid Bernoulli counts: {successes} successes in {trials} trials"
        )
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    # Avoid machine-epsilon artefacts at the true score boundaries while
    # retaining a non-degenerate opposite endpoint.
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return low, high


def newcombe_difference_interval(
    successes_from: int,
    trials_from: int,
    successes_to: int,
    trials_to: int,
) -> tuple[float, float]:
    """Newcombe score interval for ``p_to - p_from`` over disjoint sessions."""

    p_from = successes_from / trials_from
    p_to = successes_to / trials_to
    from_lo, from_hi = wilson_interval(successes_from, trials_from)
    to_lo, to_hi = wilson_interval(successes_to, trials_to)
    difference = p_to - p_from
    lower = difference - math.sqrt((p_to - to_lo) ** 2 + (from_hi - p_from) ** 2)
    upper = difference + math.sqrt((to_hi - p_to) ** 2 + (p_from - from_lo) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def _probability(successes: int, trials: int) -> dict[str, float | int]:
    low, high = wilson_interval(successes, trials)
    return {
        "successes": int(successes),
        "sessions": int(trials),
        "probability": successes / trials,
        "ci_95_low": low,
        "ci_95_high": high,
    }


def first_crossing(path: np.ndarray, threshold: float, *, above: bool) -> np.ndarray:
    """One-based first crossing minute, or ``horizon + 1`` when never crossed."""

    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise UnconditionalMoveError("first_crossing needs a non-empty session x minute matrix")
    hit = values >= threshold if above else values <= threshold
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1) + 1
    return np.where(any_hit, first, values.shape[1] + 1).astype(int)


def path_excursions(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum favourable and adverse move for a signed future path."""

    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise UnconditionalMoveError("path_excursions needs a finite non-empty matrix")
    # The entry snapshot is the zero origin even though the future path begins
    # at t+1.  Neither favourable nor adverse excursion can therefore be
    # negative when every subsequent snapshot lies on one side of entry.
    return np.maximum(values.max(axis=1), 0.0), np.maximum((-values).max(axis=1), 0.0)


def scope_masks(corpus: TapeCorpus) -> dict[str, np.ndarray]:
    dates = np.asarray(corpus.dates, dtype=object)
    masks: dict[str, np.ndarray] = {
        "pooled": np.ones(len(dates), dtype=bool),
        "era_backfill_2022-06-01_to_2025-07-31": np.asarray(
            [value < ERA_CUTOFF for value in dates], dtype=bool
        ),
        "era_owned_2025-08-01_to_2026-07-30": np.asarray(
            [value >= ERA_CUTOFF for value in dates], dtype=bool
        ),
    }
    for year in range(2022, 2027):
        masks[f"year_{year}"] = np.asarray([value.year == year for value in dates], dtype=bool)
    # The full available endpoint years have different month coverage.  This
    # second contrast fixes common calendar support (June and July) without
    # conditioning on any market state.
    masks["year_2022_june_july"] = np.asarray(
        [value.year == 2022 and value.month in (6, 7) for value in dates], dtype=bool
    )
    masks["year_2026_june_july"] = np.asarray(
        [value.year == 2026 and value.month in (6, 7) for value in dates], dtype=bool
    )
    empty = [name for name, mask in masks.items() if not mask.any()]
    if empty:
        raise UnconditionalMoveError(f"empty reporting scopes: {empty}")
    return masks


def valid_configurations(minutes: Sequence[str] = EXPECTED_MINUTES) -> list[tuple[str, int, int]]:
    index = {value: position for position, value in enumerate(minutes)}
    configurations: list[tuple[str, int, int]] = []
    for start_time in START_TIMES:
        if start_time not in index:
            raise UnconditionalMoveError(f"declared start absent from tape: {start_time}")
        start = index[start_time]
        for horizon in HORIZONS_MINUTES:
            if start + horizon < len(minutes):
                configurations.append((start_time, start, horizon))
    return configurations


def _race_summary(
    gain_index: np.ndarray,
    loss_index: np.ndarray,
    horizon: int,
    mask: np.ndarray,
) -> dict[str, Any]:
    gain = gain_index[mask]
    loss = loss_index[mask]
    gain_first = gain < loss
    loss_first = loss < gain
    tie = (gain == loss) & (gain <= horizon)
    neither = (gain > horizon) & (loss > horizon)
    if not np.all(gain_first | loss_first | tie | neither):
        raise UnconditionalMoveError("race outcomes do not partition the session population")
    trials = int(mask.sum())
    payload: dict[str, Any] = {"sessions": trials}
    for name, values in (
        ("gain_first", gain_first),
        ("loss_first", loss_first),
        ("same_snapshot_tie", tie),
        ("neither", neither),
    ):
        count = int(values.sum())
        low, high = wilson_interval(count, trials)
        payload[f"{name}_count"] = count
        payload[f"{name}_probability"] = count / trials
        payload[f"{name}_ci_95_low"] = low
        payload[f"{name}_ci_95_high"] = high
    return payload


def _contract_probability_columns(
    favourable: np.ndarray,
    start_spot: np.ndarray,
    required_points: float,
    required_pct: float,
    mask: np.ndarray,
) -> dict[str, Any]:
    trials = int(mask.sum())
    fixed = favourable[mask] >= required_points
    scaled = favourable[mask] / start_spot[mask] >= required_pct
    result: dict[str, Any] = {"sessions": trials}
    for name, values in (("fixed_points", fixed), ("scale_adjusted_pct", scaled)):
        count = int(values.sum())
        low, high = wilson_interval(count, trials)
        result[f"{name}_hit_count"] = count
        result[f"{name}_hit_probability"] = count / trials
        result[f"{name}_hit_ci_95_low"] = low
        result[f"{name}_hit_ci_95_high"] = high
    return result


@dataclass(frozen=True)
class AnalysisTables:
    excursion: pd.DataFrame
    race: pd.DataFrame
    contract: pd.DataFrame
    comparisons: pd.DataFrame


def analyze_corpus(corpus: TapeCorpus) -> AnalysisTables:
    """Compute the whole unconditional grid with no selection on prior state."""

    scopes = scope_masks(corpus)
    excursion_rows: list[dict[str, Any]] = []
    race_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for start_time, start_index, horizon in valid_configurations(corpus.minutes):
        start_spot = corpus.prices[:, start_index]
        unsigned_path = (
            corpus.prices[:, start_index + 1 : start_index + horizon + 1]
            - start_spot[:, None]
        )
        for direction in DIRECTIONS:
            sign = 1.0 if direction == "call" else -1.0
            signed_path = unsigned_path * sign
            favourable, adverse = path_excursions(signed_path)
            descriptor = {
                "direction": direction,
                "start_time_et": start_time,
                "horizon_minutes": horizon,
            }
            for scope, mask in scopes.items():
                trials = int(mask.sum())
                for metric, values in (("favourable", favourable), ("adverse", adverse)):
                    for threshold in MOVE_THRESHOLDS_POINTS:
                        count = int((values[mask] >= threshold).sum())
                        excursion_rows.append(
                            {
                                "scope": scope,
                                "metric": metric,
                                **descriptor,
                                "threshold_points": threshold,
                                **_probability(count, trials),
                            }
                        )

            gain_indices = {
                threshold: first_crossing(signed_path, threshold, above=True)
                for threshold in MOVE_THRESHOLDS_POINTS
            }
            loss_indices = {
                threshold: first_crossing(signed_path, -threshold, above=False)
                for threshold in ADVERSE_THRESHOLDS_POINTS
            }
            for scope, mask in scopes.items():
                for gain_threshold, gain_index in gain_indices.items():
                    for loss_threshold, loss_index in loss_indices.items():
                        race_rows.append(
                            {
                                "scope": scope,
                                **descriptor,
                                "gain_threshold_points": gain_threshold,
                                "adverse_threshold_points": loss_threshold,
                                **_race_summary(gain_index, loss_index, horizon, mask),
                            }
                        )

            minutes_left = minutes_to_close(start_time)
            for otm_points in COMPANION_OTM_POINTS:
                spread = (
                    COMPANION_ATM_SPREAD if otm_points == 0.0 else COMPANION_OTM_SPREAD
                )
                strike = (
                    COMPANION_SPOT + otm_points
                    if direction == "call"
                    else COMPANION_SPOT - otm_points
                )
                economics = contract_economics(
                    spot=COMPANION_SPOT,
                    strike=strike,
                    minutes_to_expiry=float(minutes_left),
                    hold_minutes=float(horizon),
                    sigma=COMPANION_SIGMA,
                    spread=spread,
                    is_call=direction == "call",
                )
                base = {
                    **descriptor,
                    "minutes_to_expiry": minutes_left,
                    "otm_points": otm_points,
                    "scenario_spot": COMPANION_SPOT,
                    "scenario_sigma": COMPANION_SIGMA,
                    "scenario_spread_points": spread,
                    "entry_ask_usd": economics.entry_ask_usd,
                    "friction_usd": economics.friction_usd,
                    "decay_only_pnl_usd": economics.decay_only_pnl_usd,
                    "required_move_points": economics.required_move_points,
                    "required_move_pct": economics.required_move_pct,
                    "under_signed_ticket_cap": bool(
                        math.isfinite(economics.entry_ask_usd)
                        and economics.entry_ask_usd <= SIGNED_TICKET_CAP_USD
                    ),
                    "status": "MEASURED_NECESSARY_CONDITION"
                    if economics.reachable
                    else "COMPANION_REQUIREMENT_UNDEFINED_OR_UNREACHABLE",
                }
                for scope, mask in scopes.items():
                    row = {"scope": scope, **base}
                    if economics.reachable:
                        row.update(
                            _contract_probability_columns(
                                favourable,
                                start_spot,
                                economics.required_move_points,
                                economics.required_move_pct,
                                mask,
                            )
                        )
                    else:
                        row.update(
                            {
                                "sessions": int(mask.sum()),
                                "fixed_points_hit_count": None,
                                "fixed_points_hit_probability": None,
                                "fixed_points_hit_ci_95_low": None,
                                "fixed_points_hit_ci_95_high": None,
                                "scale_adjusted_pct_hit_count": None,
                                "scale_adjusted_pct_hit_probability": None,
                                "scale_adjusted_pct_hit_ci_95_low": None,
                                "scale_adjusted_pct_hit_ci_95_high": None,
                            }
                        )
                    contract_rows.append(row)

    excursion = pd.DataFrame(excursion_rows)
    race = pd.DataFrame(race_rows)
    contract = pd.DataFrame(contract_rows)
    comparisons = build_comparisons(excursion, race, contract)
    return AnalysisTables(excursion, race, contract, comparisons)


def validate_tables(corpus: TapeCorpus, tables: AnalysisTables) -> dict[str, Any]:
    """Fail closed on identities that must hold for every completed grid."""

    scopes = scope_masks(corpus)
    scope_sessions = {name: int(mask.sum()) for name, mask in scopes.items()}
    configurations = valid_configurations(corpus.minutes)
    expected_rows = {
        "excursion": (
            len(configurations)
            * len(DIRECTIONS)
            * len(scopes)
            * 2
            * len(MOVE_THRESHOLDS_POINTS)
        ),
        "race": (
            len(configurations)
            * len(DIRECTIONS)
            * len(scopes)
            * len(MOVE_THRESHOLDS_POINTS)
            * len(ADVERSE_THRESHOLDS_POINTS)
        ),
        "contract": (
            len(configurations)
            * len(DIRECTIONS)
            * len(scopes)
            * len(COMPANION_OTM_POINTS)
        ),
    }
    actual_rows = {
        "excursion": len(tables.excursion),
        "race": len(tables.race),
        "contract": len(tables.contract),
    }
    if actual_rows != expected_rows:
        raise UnconditionalMoveError(
            f"analysis row counts differ: {actual_rows}, expected {expected_rows}"
        )

    for family_name, frame in (
        ("excursion", tables.excursion),
        ("race", tables.race),
        ("contract", tables.contract),
    ):
        observed = frame.groupby("scope")["sessions"].unique()
        for scope, values in observed.items():
            if len(values) != 1 or int(values[0]) != scope_sessions[scope]:
                raise UnconditionalMoveError(
                    f"{family_name}: session denominator drift in {scope}: {values}"
                )

    excursion = tables.excursion
    excursion_keys = [
        "scope",
        "metric",
        "direction",
        "start_time_et",
        "horizon_minutes",
        "threshold_points",
    ]
    if excursion.duplicated(excursion_keys).any():
        raise UnconditionalMoveError("excursion grid contains duplicate cells")
    for row in excursion.itertuples(index=False):
        if not 0 <= row.successes <= row.sessions:
            raise UnconditionalMoveError("excursion count is outside its denominator")
        if not math.isclose(row.probability, row.successes / row.sessions):
            raise UnconditionalMoveError("excursion probability does not reproduce its count")
        if not row.ci_95_low <= row.probability <= row.ci_95_high:
            raise UnconditionalMoveError("excursion interval does not contain its estimate")
    threshold_violations = 0
    for _, group in excursion.groupby(excursion_keys[:-1], sort=False):
        counts = group.sort_values("threshold_points")["successes"].to_numpy()
        threshold_violations += int((np.diff(counts) > 0).any())
    horizon_groups = [
        "scope",
        "metric",
        "direction",
        "start_time_et",
        "threshold_points",
    ]
    horizon_violations = 0
    for _, group in excursion.groupby(horizon_groups, sort=False):
        counts = group.sort_values("horizon_minutes")["successes"].to_numpy()
        horizon_violations += int((np.diff(counts) < 0).any())
    if threshold_violations or horizon_violations:
        raise UnconditionalMoveError(
            "excursion monotonicity failed: "
            f"threshold={threshold_violations}, horizon={horizon_violations}"
        )

    symmetry_keys = [
        "scope",
        "start_time_et",
        "horizon_minutes",
        "threshold_points",
    ]
    call_favourable = excursion[
        excursion["direction"].eq("call") & excursion["metric"].eq("favourable")
    ].set_index(symmetry_keys)["successes"]
    put_adverse = excursion[
        excursion["direction"].eq("put") & excursion["metric"].eq("adverse")
    ].set_index(symmetry_keys)["successes"]
    call_adverse = excursion[
        excursion["direction"].eq("call") & excursion["metric"].eq("adverse")
    ].set_index(symmetry_keys)["successes"]
    put_favourable = excursion[
        excursion["direction"].eq("put") & excursion["metric"].eq("favourable")
    ].set_index(symmetry_keys)["successes"]
    if not call_favourable.equals(put_adverse) or not call_adverse.equals(put_favourable):
        raise UnconditionalMoveError("call/put excursion reversal identity failed")

    race = tables.race
    race_keys = [
        "scope",
        "direction",
        "start_time_et",
        "horizon_minutes",
        "gain_threshold_points",
        "adverse_threshold_points",
    ]
    if race.duplicated(race_keys).any():
        raise UnconditionalMoveError("race grid contains duplicate cells")
    partition = race[
        ["gain_first_count", "loss_first_count", "same_snapshot_tie_count", "neither_count"]
    ].sum(axis=1)
    if not np.array_equal(partition.to_numpy(), race["sessions"].to_numpy()):
        raise UnconditionalMoveError("race states do not sum to every session")
    if int(race["same_snapshot_tie_count"].sum()) != 0:
        raise UnconditionalMoveError("positive and negative barriers tied on a scalar snapshot")
    # Build the complete side-specific favourable lookup, not just the call alias.
    favourable_hits = excursion[excursion["metric"].eq("favourable")].set_index(
        [
            "scope",
            "direction",
            "start_time_et",
            "horizon_minutes",
            "threshold_points",
        ]
    )["successes"]
    race_violations = 0
    for row in race.itertuples(index=False):
        key = (
            row.scope,
            row.direction,
            row.start_time_et,
            row.horizon_minutes,
            row.gain_threshold_points,
        )
        race_violations += int(row.gain_first_count > favourable_hits.loc[key])
    if race_violations:
        raise UnconditionalMoveError("gain-first count exceeds favourable-hit count")

    contract = tables.contract
    contract_keys = ["scope", "direction", "start_time_et", "horizon_minutes", "otm_points"]
    if contract.duplicated(contract_keys).any():
        raise UnconditionalMoveError("contract join contains duplicate cells")
    measured = contract[contract["status"].eq("MEASURED_NECESSARY_CONDITION")]
    for prefix in ("fixed_points", "scale_adjusted_pct"):
        counts = measured[f"{prefix}_hit_count"]
        probabilities = measured[f"{prefix}_hit_probability"]
        if ((counts < 0) | (counts > measured["sessions"])).any():
            raise UnconditionalMoveError(f"{prefix} contract hit count is invalid")
        expected = counts / measured["sessions"]
        if not np.allclose(probabilities, expected):
            raise UnconditionalMoveError(f"{prefix} contract hit probability is invalid")
    if not np.isfinite(measured["required_move_points"]).all():
        raise UnconditionalMoveError("measured contract requirement is non-finite")

    comparisons = tables.comparisons
    comparison_keys = [
        column
        for column in comparisons.columns
        if column
        not in {
            "from_successes",
            "from_sessions",
            "from_probability",
            "from_ci_95_low",
            "from_ci_95_high",
            "to_successes",
            "to_sessions",
            "to_probability",
            "to_ci_95_low",
            "to_ci_95_high",
            "difference_to_minus_from",
            "difference_ci_95_low",
            "difference_ci_95_high",
        }
    ]
    if comparisons.duplicated(comparison_keys).any():
        raise UnconditionalMoveError("comparison grid contains duplicate cells")
    if not (
        (comparisons["difference_ci_95_low"] <= comparisons["difference_to_minus_from"])
        & (comparisons["difference_to_minus_from"] <= comparisons["difference_ci_95_high"])
    ).all():
        raise UnconditionalMoveError("comparison interval does not contain its estimate")

    return {
        "status": "PASS",
        "expected_and_actual_rows": expected_rows,
        "scope_session_denominators": scope_sessions,
        "threshold_monotonicity_violations": threshold_violations,
        "horizon_monotonicity_violations": horizon_violations,
        "call_put_reversal_identity": True,
        "race_partition_violations": 0,
        "same_snapshot_ties": 0,
        "race_gain_exceeds_favourable_hit": race_violations,
        "comparison_rows": len(comparisons),
    }


def _append_difference(
    rows: list[dict[str, Any]],
    *,
    family: str,
    metric: str,
    descriptor: Mapping[str, Any],
    from_scope: str,
    to_scope: str,
    successes_from: int,
    trials_from: int,
    successes_to: int,
    trials_to: int,
) -> None:
    p_from = successes_from / trials_from
    p_to = successes_to / trials_to
    from_low, from_high = wilson_interval(successes_from, trials_from)
    to_low, to_high = wilson_interval(successes_to, trials_to)
    difference_low, difference_high = newcombe_difference_interval(
        successes_from, trials_from, successes_to, trials_to
    )
    rows.append(
        {
            "family": family,
            "metric": metric,
            **descriptor,
            "from_scope": from_scope,
            "to_scope": to_scope,
            "from_successes": successes_from,
            "from_sessions": trials_from,
            "from_probability": p_from,
            "from_ci_95_low": from_low,
            "from_ci_95_high": from_high,
            "to_successes": successes_to,
            "to_sessions": trials_to,
            "to_probability": p_to,
            "to_ci_95_low": to_low,
            "to_ci_95_high": to_high,
            "difference_to_minus_from": p_to - p_from,
            "difference_ci_95_low": difference_low,
            "difference_ci_95_high": difference_high,
        }
    )


def build_comparisons(
    excursion: pd.DataFrame, race: pd.DataFrame, contract: pd.DataFrame
) -> pd.DataFrame:
    """Backfill→owned and partial-2022→partial-2026 session-level changes."""

    pairs = (
        (
            "era_backfill_2022-06-01_to_2025-07-31",
            "era_owned_2025-08-01_to_2026-07-30",
        ),
        ("year_2022", "year_2026"),
        ("year_2022_june_july", "year_2026_june_july"),
    )
    rows: list[dict[str, Any]] = []

    excursion_keys = (
        "metric",
        "direction",
        "start_time_et",
        "horizon_minutes",
        "threshold_points",
    )
    for from_scope, to_scope in pairs:
        left = excursion[excursion["scope"] == from_scope].set_index(list(excursion_keys))
        right = excursion[excursion["scope"] == to_scope].set_index(list(excursion_keys))
        if not left.index.equals(right.index):
            raise UnconditionalMoveError("excursion comparison grids differ")
        for key in left.index:
            a, b = left.loc[key], right.loc[key]
            descriptor = dict(zip(excursion_keys[1:], key[1:], strict=True))
            _append_difference(
                rows,
                family="excursion",
                metric=str(key[0]),
                descriptor=descriptor,
                from_scope=from_scope,
                to_scope=to_scope,
                successes_from=int(a["successes"]),
                trials_from=int(a["sessions"]),
                successes_to=int(b["successes"]),
                trials_to=int(b["sessions"]),
            )

    race_keys = (
        "direction",
        "start_time_et",
        "horizon_minutes",
        "gain_threshold_points",
        "adverse_threshold_points",
    )
    for from_scope, to_scope in pairs:
        left = race[race["scope"] == from_scope].set_index(list(race_keys))
        right = race[race["scope"] == to_scope].set_index(list(race_keys))
        if not left.index.equals(right.index):
            raise UnconditionalMoveError("race comparison grids differ")
        for key in left.index:
            a, b = left.loc[key], right.loc[key]
            descriptor = dict(zip(race_keys, key, strict=True))
            _append_difference(
                rows,
                family="race",
                metric="gain_first",
                descriptor=descriptor,
                from_scope=from_scope,
                to_scope=to_scope,
                successes_from=int(a["gain_first_count"]),
                trials_from=int(a["sessions"]),
                successes_to=int(b["gain_first_count"]),
                trials_to=int(b["sessions"]),
            )

    contract_keys = (
        "direction",
        "start_time_et",
        "horizon_minutes",
        "otm_points",
    )
    reachable = contract[contract["status"] == "MEASURED_NECESSARY_CONDITION"]
    for from_scope, to_scope in pairs:
        left = reachable[reachable["scope"] == from_scope].set_index(list(contract_keys))
        right = reachable[reachable["scope"] == to_scope].set_index(list(contract_keys))
        if not left.index.equals(right.index):
            raise UnconditionalMoveError("contract comparison grids differ")
        for key in left.index:
            a, b = left.loc[key], right.loc[key]
            descriptor = dict(zip(contract_keys, key, strict=True))
            for metric, prefix in (
                ("required_move_fixed_points_hit", "fixed_points"),
                ("required_move_scale_adjusted_pct_hit", "scale_adjusted_pct"),
            ):
                _append_difference(
                    rows,
                    family="contract_necessary_condition",
                    metric=metric,
                    descriptor=descriptor,
                    from_scope=from_scope,
                    to_scope=to_scope,
                    successes_from=int(a[f"{prefix}_hit_count"]),
                    trials_from=int(a["sessions"]),
                    successes_to=int(b[f"{prefix}_hit_count"]),
                    trials_to=int(b["sessions"]),
                )
    return pd.DataFrame(rows)


def percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def interval_text(estimate: float, low: float, high: float) -> str:
    return f"{percentage(estimate)} [{percentage(low)}, {percentage(high)}]"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    materialized = [[clean(value) for value in row] for row in rows]
    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in materialized)
    return "\n".join(lines)


def render_readable_tables(corpus: TapeCorpus, tables: AnalysisTables) -> str:
    """Owner-readable selections; the CSV files remain the complete tables."""

    representative_starts = ("09:31", "10:30", "12:30", "14:30", "15:00", "15:30", "15:50")
    representative_horizons = (10, 20, 45, 90)
    lines = [
        "# Unconditional SPX move terrain — readable tables",
        "",
        (
            f"Analyzed {len(corpus.sessions):,} clean sessions. Each probability is the "
            "share of sessions whose one exact clock window met the condition; brackets "
            "are pointwise 95% Wilson intervals over sessions."
        ),
        "",
        "## Pooled favourable and adverse excursions",
        "",
    ]
    pooled = tables.excursion[
        (tables.excursion["scope"] == "pooled")
        & tables.excursion["start_time_et"].isin(representative_starts)
        & tables.excursion["horizon_minutes"].isin(representative_horizons)
        & tables.excursion["threshold_points"].isin((5.0, 10.0, 20.0, 40.0))
    ]
    pivot: list[list[Any]] = []
    for key, group in pooled.groupby(
        ["metric", "direction", "start_time_et", "horizon_minutes"], sort=False
    ):
        metric, direction, start, horizon = key
        values = {}
        for row in group.itertuples(index=False):
            values[float(row.threshold_points)] = interval_text(
                float(row.probability), float(row.ci_95_low), float(row.ci_95_high)
            )
        pivot.append(
            [metric, direction, start, horizon]
            + [values.get(threshold, "—") for threshold in (5.0, 10.0, 20.0, 40.0)]
        )
    lines.append(
        markdown_table(
            ("metric", "side", "start ET", "horizon", "≥5", "≥10", "≥20", "≥40"),
            pivot,
        )
    )
    lines.extend(["", "## Gain-before-adverse race (selected cells)", ""])
    selected_race = tables.race[
        (tables.race["scope"] == "pooled")
        & tables.race["start_time_et"].isin(("09:31", "12:30", "14:30", "15:30"))
        & tables.race["horizon_minutes"].isin((20, 45, 90))
        & tables.race["gain_threshold_points"].isin((5.0, 10.0, 20.0))
        & tables.race["adverse_threshold_points"].isin((5.0, 10.0))
    ]
    race_rows = []
    for row in selected_race.itertuples(index=False):
        race_rows.append(
            (
                row.direction,
                row.start_time_et,
                row.horizon_minutes,
                row.gain_threshold_points,
                row.adverse_threshold_points,
                interval_text(
                    row.gain_first_probability,
                    row.gain_first_ci_95_low,
                    row.gain_first_ci_95_high,
                ),
                percentage(row.neither_probability),
            )
        )
    lines.append(
        markdown_table(
            ("side", "start ET", "horizon", "+M", "−J", "P(+M first)", "neither"),
            race_rows,
        )
    )
    lines.extend(["", "## Companion contract requirement: necessary-condition hits", ""])
    selected_contract = tables.contract[
        (tables.contract["scope"] == "pooled")
        & tables.contract["direction"].isin(DIRECTIONS)
        & tables.contract["start_time_et"].isin(("09:31", "10:30", "13:30", "14:30", "15:00"))
        & tables.contract["horizon_minutes"].isin((10, 20, 45))
        & tables.contract["otm_points"].isin((0.0, 25.0, 60.0))
        & (tables.contract["status"] == "MEASURED_NECESSARY_CONDITION")
    ]
    contract_rows = []
    for row in selected_contract.itertuples(index=False):
        contract_rows.append(
            (
                row.direction,
                row.start_time_et,
                row.horizon_minutes,
                int(row.otm_points),
                f"{row.entry_ask_usd:.0f}",
                "yes" if row.under_signed_ticket_cap else "no",
                f"{row.required_move_points:.2f}",
                f"{100.0 * row.required_move_pct:.3f}%",
                interval_text(
                    row.fixed_points_hit_probability,
                    row.fixed_points_hit_ci_95_low,
                    row.fixed_points_hit_ci_95_high,
                ),
                interval_text(
                    row.scale_adjusted_pct_hit_probability,
                    row.scale_adjusted_pct_hit_ci_95_low,
                    row.scale_adjusted_pct_hit_ci_95_high,
                ),
            )
        )
    lines.append(
        markdown_table(
            (
                "side",
                "start ET",
                "hold",
                "OTM pts",
                "entry $",
                "≤$2k",
                "required pts @6800",
                "required %",
                "P(literal pts)",
                "P(scale-adjusted)",
            ),
            contract_rows,
        )
    )
    lines.extend(["", "## Endpoint drift: 2022 (Jun–Dec) to 2026 (Jan–Jul)", ""])
    selected_comparison = tables.comparisons[
        (tables.comparisons["family"] == "excursion")
        & (tables.comparisons["metric"] == "favourable")
        & (tables.comparisons["from_scope"] == "year_2022")
        & (tables.comparisons["to_scope"] == "year_2026")
        & tables.comparisons["start_time_et"].isin(("09:31", "12:30", "14:30"))
        & tables.comparisons["horizon_minutes"].isin((20, 60))
        & (tables.comparisons["threshold_points"] == 10.0)
    ]
    comparison_rows = []
    for row in selected_comparison.itertuples(index=False):
        comparison_rows.append(
            (
                row.direction,
                row.start_time_et,
                row.horizon_minutes,
                interval_text(row.from_probability, row.from_ci_95_low, row.from_ci_95_high),
                interval_text(row.to_probability, row.to_ci_95_low, row.to_ci_95_high),
                f"{100.0 * row.difference_to_minus_from:+.1f}pp "
                f"[{100.0 * row.difference_ci_95_low:+.1f}, "
                f"{100.0 * row.difference_ci_95_high:+.1f}]",
            )
        )
    lines.append(
        markdown_table(
            ("side", "start ET", "horizon", "2022", "2026", "2026−2022"),
            comparison_rows,
        )
    )
    lines.extend(
        [
            "",
            "The companion table is a necessary-condition join: it reports how often SPX "
            "reached the move that reprices one assumed contract to zero P&L. It is not an "
            "expected-P&L calculation and therefore cannot, by itself, prove profitability. "
            "The scale-adjusted column is the more comparable pooled-history view because "
            "the scenario requirement was solved at SPX 6,800.",
            "",
        ]
    )
    return "\n".join(lines)
