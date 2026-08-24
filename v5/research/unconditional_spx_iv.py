"""Causal-ladder implied-volatility clock census for the SPX race study.

Only quote-time ladder fields are read.  The unit of analysis is one session
at one exact clock; contract rows are reduced inside the session before any
cross-session statistic is formed.  This module never reads an outcome, label,
option exit, P&L, fitted value, feature score, or reserved session.

The primary estimand is the equal-weighted mean of separate call and put
median ``self_iv`` values inside the project's existing +/-10 SPX-point ATM
band.  A paired same-strike closest-ATM estimator is a mandatory sensitivity.
Both are ATM level curves.  The stored ladder ends at +/-25 points, so 40- and
60-point OTM IV are explicitly unsupported rather than extrapolated.
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


SCHEMA_VERSION = "v5.unconditional-spx-iv-clock.v1"
DEFAULT_LADDER_ROOT = Path(
    "/Volumes/AR_TRADING_DATA/"
    "lifecycle_corpus_spx_tape_2022-06-01_2026-07-31/ladder"
)
LADDER_FILE = re.compile(r"^(20\d{2}-\d{2}-\d{2})\.parquet$")
EXPECTED_DISCOVERED_SESSIONS = 1_014
EXPECTED_ANALYZED_SESSIONS = 1_011
EXPECTED_FIRST_SESSION = date(2022, 6, 1)
EXPECTED_LAST_SESSION = date(2026, 7, 30)
CONFIRMATION_START = date(2026, 8, 6)
ERA_CUTOFF = date(2025, 8, 1)
TAPE_SOURCE = "spx_parity_spot"

KNOWN_DEFECTS: Mapping[date, str] = {
    date(2022, 11, 25): "vendor-padded early close; excluded upstream",
    date(2023, 6, 26): "interior whole-book freeze (2 minutes)",
    date(2023, 10, 19): "interior whole-book freeze (3 minutes)",
    date(2023, 10, 25): "interior whole-book freezes (4 and 18 minutes)",
}
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
EXPECTED_MINUTES = tuple(
    f"{minute // 60:02d}:{minute % 60:02d}"
    for minute in range(9 * 60 + 31, 16 * 60 + 1)
)
REPORTING_SCOPES = (
    "pooled",
    "era_backfill_2022-06-01_to_2025-07-31",
    "era_owned_2025-08-01_to_2026-07-30",
)
REGIONS = ("atm_10pt_primary", "closest_atm_pair_sensitivity")
ESTIMANDS = ("call_median", "put_median", "side_balanced")
STATISTICS = ("mean", "q10", "q25", "q50", "q75", "q90", "q95", "q99")
QUANTILES = {
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
    "q95": 0.95,
    "q99": 0.99,
}
BASELINE_START = "09:35"
ATM_BAND_POINTS = 10.0
STORED_LADDER_LIMIT_POINTS = 25.0
IV_LOWER_BOUND = 0.01
IV_UPPER_BOUND = 5.0
IV_BOUND_TOLERANCE = 1e-6
MIN_MINUTES_TO_EXPIRY = 5.0
CI_Z = 1.959963984540054
LADDER_COLUMNS = (
    "session",
    "event_time",
    "expiry",
    "contract_id",
    "strike",
    "underlying_price",
    "minute",
    "moneyness_itm_points",
    "is_call",
    "minutes_to_expiry",
    "self_iv",
    "tape_source",
)


class UnconditionalIVError(RuntimeError):
    """The ladder census cannot prove its causal population or estimand."""


@dataclass(frozen=True)
class IVCensus:
    root: Path
    sessions: tuple[str, ...]
    dates: tuple[date, ...]
    session_values: pd.DataFrame
    input_manifest: pd.DataFrame
    strict_population: bool


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise UnconditionalIVError(
            f"invalid Bernoulli counts: {successes} successes in {trials} trials"
        )
    probability = successes / trials
    z2 = CI_Z * CI_Z
    denominator = 1.0 + z2 / trials
    centre = (probability + z2 / (2.0 * trials)) / denominator
    half = (
        CI_Z
        * math.sqrt(
            (probability * (1.0 - probability) + z2 / (4.0 * trials)) / trials
        )
        / denominator
    )
    low, high = max(0.0, centre - half), min(1.0, centre + half)
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return low, high


def _discover(root: Path) -> list[tuple[date, Path]]:
    root = Path(root)
    if not root.is_dir():
        raise UnconditionalIVError(f"missing ladder directory: {root}")
    discovered: list[tuple[date, Path]] = []
    unexpected: list[str] = []
    for path in root.glob("*.parquet"):
        match = LADDER_FILE.match(path.name)
        if not match:
            unexpected.append(path.name)
            continue
        discovered.append((date.fromisoformat(match.group(1)), path))
    discovered.sort(key=lambda pair: pair[0])
    if unexpected:
        raise UnconditionalIVError(f"unexpected ladder names: {sorted(unexpected)[:3]}")
    if not discovered:
        raise UnconditionalIVError(f"ladder directory contains no sessions: {root}")
    dates = [value for value, _ in discovered]
    if len(dates) != len(set(dates)):
        raise UnconditionalIVError("duplicate ladder session date")
    reserved = [value for value in dates if value >= CONFIRMATION_START]
    if reserved:
        raise UnconditionalIVError(f"confirmation-reserved ladder present: {reserved[0]}")
    return discovered


def _validate_ladder_frame(frame: pd.DataFrame, session: str) -> None:
    if frame.empty:
        raise UnconditionalIVError(f"{session}: empty ladder")
    if set(frame["session"].astype(str)) != {session}:
        raise UnconditionalIVError(f"{session}: filename/session mismatch")
    if set(frame["expiry"].astype(str)) != {session}:
        raise UnconditionalIVError(f"{session}: ladder is not same-day expiry")
    if set(frame["tape_source"].astype(str)) != {TAPE_SOURCE}:
        raise UnconditionalIVError(f"{session}: wrong ladder tape_source")
    if frame.duplicated(["session", "minute", "contract_id"]).any():
        raise UnconditionalIVError(f"{session}: duplicate minute/contract ladder row")
    minutes = frame["minute"].astype(str)
    if not set(minutes).issubset(EXPECTED_MINUTES):
        raise UnconditionalIVError(f"{session}: ladder minute outside 09:31-16:00")
    event_time = pd.to_datetime(frame["event_time"], errors="coerce")
    if event_time.isna().any() or event_time.dt.tz is None:
        raise UnconditionalIVError(f"{session}: event_time is missing or timezone-naive")
    local = event_time.dt.tz_convert("America/New_York")
    if set(local.dt.strftime("%Y-%m-%d")) != {session}:
        raise UnconditionalIVError(f"{session}: event_time local date mismatch")
    if not np.array_equal(local.dt.strftime("%H:%M").to_numpy(), minutes.to_numpy()):
        raise UnconditionalIVError(f"{session}: event_time/minute mismatch")
    for column in ("strike", "underlying_price"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise UnconditionalIVError(f"{session}: invalid {column}")
    moneyness = pd.to_numeric(
        frame["moneyness_itm_points"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(moneyness).all() or (
        np.abs(moneyness) > STORED_LADDER_LIMIT_POINTS + 1e-9
    ).any():
        raise UnconditionalIVError(f"{session}: ladder exceeds persisted +/-25 support")
    minutes_to_expiry = pd.to_numeric(
        frame["minutes_to_expiry"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(minutes_to_expiry).all() or (minutes_to_expiry < 0.0).any():
        raise UnconditionalIVError(f"{session}: invalid minutes_to_expiry")


def _interior_iv(frame: pd.DataFrame) -> pd.Series:
    iv = pd.to_numeric(frame["self_iv"], errors="coerce")
    minutes = pd.to_numeric(frame["minutes_to_expiry"], errors="coerce")
    return (
        np.isfinite(iv)
        & (minutes >= MIN_MINUTES_TO_EXPIRY)
        & (iv > IV_LOWER_BOUND + IV_BOUND_TOLERANCE)
        & (iv < IV_UPPER_BOUND - IV_BOUND_TOLERANCE)
    )


def _bound_hit(frame: pd.DataFrame) -> pd.Series:
    iv = pd.to_numeric(frame["self_iv"], errors="coerce")
    minutes = pd.to_numeric(frame["minutes_to_expiry"], errors="coerce")
    return (
        np.isfinite(iv)
        & (minutes >= MIN_MINUTES_TO_EXPIRY)
        & (
            (iv <= IV_LOWER_BOUND + IV_BOUND_TOLERANCE)
            | (iv >= IV_UPPER_BOUND - IV_BOUND_TOLERANCE)
        )
    )


def _session_rows(frame: pd.DataFrame, session: str, session_date: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    era = "backfill" if session_date < ERA_CUTOFF else "owned"
    for start in START_TIMES:
        minute_frame = frame[frame["minute"].astype(str) == start].copy()
        if minute_frame.empty:
            raise UnconditionalIVError(f"{session}: declared IV clock absent: {start}")
        spot_values = minute_frame["underlying_price"].to_numpy(float)
        if not np.allclose(spot_values, spot_values[0], rtol=0.0, atol=1e-9):
            raise UnconditionalIVError(f"{session} {start}: multiple underlying prices")
        valid = _interior_iv(minute_frame)
        bounds = _bound_hit(minute_frame)
        moneyness = minute_frame["moneyness_itm_points"].to_numpy(float)

        atm = minute_frame[valid & (np.abs(moneyness) <= ATM_BAND_POINTS)].copy()
        call = atm.loc[atm["is_call"].astype(bool), "self_iv"].to_numpy(float)
        put = atm.loc[~atm["is_call"].astype(bool), "self_iv"].to_numpy(float)
        call_value = float(np.median(call)) if len(call) else None
        put_value = float(np.median(put)) if len(put) else None
        balanced = (
            (call_value + put_value) / 2.0
            if call_value is not None and put_value is not None
            else None
        )
        common = {
            "session": session,
            "session_date": session,
            "source_era": era,
            "start_time_et": start,
            "region": "atm_10pt_primary",
            "selected_call_nodes": int(len(call)),
            "selected_put_nodes": int(len(put)),
            "paired_strike_count": 0,
            "solver_bound_excluded_count": int(
                (bounds & (np.abs(moneyness) <= ATM_BAND_POINTS)).sum()
            ),
            "atm_band_points": ATM_BAND_POINTS,
            "stored_ladder_limit_points": STORED_LADDER_LIMIT_POINTS,
            "iv_40_point_otm_support": "UNSUPPORTED_OUTSIDE_STORED_LADDER",
            "iv_60_point_otm_support": "UNSUPPORTED_OUTSIDE_STORED_LADDER",
            "interpretation": "ATM_LEVEL_CLOCK_CURVE_NOT_STRIKE_SPECIFIC",
        }
        for estimand, estimate in (
            ("call_median", call_value),
            ("put_median", put_value),
            ("side_balanced", balanced),
        ):
            rows.append(
                {
                    **common,
                    "estimand": estimand,
                    "iv": estimate,
                    "status": "MEASURED" if estimate is not None else "MISSING_SIDE",
                }
            )

        # Mandatory sensitivity: only strikes with a valid call and put may
        # compete; choose the physical strike nearest parity spot, and average
        # the two nearest strikes when spot is exactly between grid nodes.
        usable = minute_frame[valid].copy()
        by_strike = (
            usable.groupby(["strike", "is_call"], as_index=False)["self_iv"]
            .median()
            .pivot(index="strike", columns="is_call", values="self_iv")
        )
        if True in by_strike.columns and False in by_strike.columns:
            paired = by_strike.dropna(subset=[True, False]).copy()
        else:
            paired = by_strike.iloc[0:0].copy()
        if paired.empty:
            paired_call = paired_put = paired_balanced = None
            nearest_count = 0
        else:
            distances = np.abs(paired.index.to_numpy(float) - float(spot_values[0]))
            minimum = float(distances.min())
            nearest = paired[np.isclose(distances, minimum, rtol=0.0, atol=1e-9)]
            nearest_count = len(nearest)
            paired_call = float(nearest[True].mean())
            paired_put = float(nearest[False].mean())
            paired_balanced = (paired_call + paired_put) / 2.0
        sensitivity_common = {
            **common,
            "region": "closest_atm_pair_sensitivity",
            "selected_call_nodes": nearest_count,
            "selected_put_nodes": nearest_count,
            "paired_strike_count": nearest_count,
            "solver_bound_excluded_count": int(bounds.sum()),
            "interpretation": "PAIRED_CLOSEST_ATM_SENSITIVITY_NOT_STRIKE_SPECIFIC",
        }
        for estimand, estimate in (
            ("call_median", paired_call),
            ("put_median", paired_put),
            ("side_balanced", paired_balanced),
        ):
            rows.append(
                {
                    **sensitivity_common,
                    "estimand": estimand,
                    "iv": estimate,
                    "status": "MEASURED" if estimate is not None else "NO_VALID_PAIRED_STRIKE",
                }
            )
    return rows


def load_iv_census(
    root: Path,
    clean_sessions: Sequence[str],
    clean_dates: Sequence[date],
    *,
    strict_population: bool = True,
) -> IVCensus:
    """Load and reduce every clean ladder file, excluding defects before read."""

    root = Path(root)
    sessions = tuple(str(value) for value in clean_sessions)
    dates = tuple(clean_dates)
    if len(sessions) != len(dates) or tuple(value.isoformat() for value in dates) != sessions:
        raise UnconditionalIVError("clean ladder session/date alignment is invalid")
    if len(sessions) != len(set(sessions)):
        raise UnconditionalIVError("duplicate requested clean session")
    if any(value >= CONFIRMATION_START for value in dates):
        raise UnconditionalIVError("requested clean population includes reserved session")
    if strict_population:
        if len(sessions) != EXPECTED_ANALYZED_SESSIONS:
            raise UnconditionalIVError(
                f"clean ladder population drifted: {len(sessions)}"
            )
        backfill = sum(value < ERA_CUTOFF for value in dates)
        owned = sum(value >= ERA_CUTOFF for value in dates)
        if (backfill, owned) != (768, 243):
            raise UnconditionalIVError(f"clean era counts drifted: {(backfill, owned)}")

    discovered = _discover(root)
    discovered_dates = [value for value, _ in discovered]
    if strict_population:
        facts = (len(discovered), discovered_dates[0], discovered_dates[-1])
        expected = (
            EXPECTED_DISCOVERED_SESSIONS,
            EXPECTED_FIRST_SESSION,
            EXPECTED_LAST_SESSION,
        )
        if facts != expected:
            raise UnconditionalIVError(
                f"ladder population drifted: {facts}, expected {expected}"
            )
    expected_clean = set(dates)
    observed_clean = {value for value in discovered_dates if value not in KNOWN_DEFECTS}
    if observed_clean != expected_clean:
        missing = sorted(expected_clean - observed_clean)
        extra = sorted(observed_clean - expected_clean)
        raise UnconditionalIVError(
            f"ladder/tape clean populations differ: missing={missing[:3]}, extra={extra[:3]}"
        )

    manifest_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for session_date, path in discovered:
        session = session_date.isoformat()
        reason = KNOWN_DEFECTS.get(session_date)
        manifest_row = {
            "input_family": "causal_ladder",
            "session": session,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "source_era": "backfill" if session_date < ERA_CUTOFF else "owned",
        }
        if reason:
            manifest_row.update(status="EXCLUDED_KNOWN_DEFECT", reason=reason)
        else:
            try:
                frame = pd.read_parquet(path, columns=list(LADDER_COLUMNS))
            except Exception as exc:  # noqa: BLE001 - fail closed with session identity
                raise UnconditionalIVError(f"{session}: ladder unreadable: {exc}") from exc
            _validate_ladder_frame(frame, session)
            value_rows.extend(_session_rows(frame, session, session_date))
            manifest_row.update(status="ANALYZED", reason="")
        manifest_rows.append(manifest_row)

    values = pd.DataFrame(value_rows)
    expected_value_rows = len(sessions) * len(START_TIMES) * len(REGIONS) * len(ESTIMANDS)
    if len(values) != expected_value_rows:
        raise UnconditionalIVError(
            f"IV session-row count drifted: {len(values)}, expected {expected_value_rows}"
        )
    key = ["session", "start_time_et", "region", "estimand"]
    if values.duplicated(key).any():
        raise UnconditionalIVError("duplicate IV session estimand")
    return IVCensus(
        root=root,
        sessions=sessions,
        dates=dates,
        session_values=values.sort_values(key).reset_index(drop=True),
        input_manifest=pd.DataFrame(manifest_rows).sort_values("session").reset_index(drop=True),
        strict_population=strict_population,
    )


def _scope_sessions(census: IVCensus) -> dict[str, set[str]]:
    return {
        "pooled": set(census.sessions),
        "era_backfill_2022-06-01_to_2025-07-31": {
            session
            for session, session_date in zip(census.sessions, census.dates, strict=True)
            if session_date < ERA_CUTOFF
        },
        "era_owned_2025-08-01_to_2026-07-30": {
            session
            for session, session_date in zip(census.sessions, census.dates, strict=True)
            if session_date >= ERA_CUTOFF
        },
    }


def _statistic_rows(values: np.ndarray) -> list[dict[str, Any]]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    ordered = np.sort(clean)
    rows: list[dict[str, Any]] = []
    for statistic in STATISTICS:
        row: dict[str, Any] = {"statistic": statistic}
        if n == 0:
            row.update(
                estimate=None,
                ci_95_low=None,
                ci_95_high=None,
                interval_method="none",
                status="NO_USABLE_SESSIONS",
            )
        elif statistic == "mean":
            estimate = float(clean.mean())
            if n < 2:
                row.update(
                    estimate=estimate,
                    ci_95_low=None,
                    ci_95_high=None,
                    interval_method="session_t_interval",
                    status="INSUFFICIENT_SESSIONS_FOR_INTERVAL",
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
            quantile = QUANTILES[statistic]
            estimate = float(np.quantile(ordered, quantile, method="linear"))
            if n < 2:
                row.update(
                    estimate=estimate,
                    ci_95_low=None,
                    ci_95_high=None,
                    interval_method="exact_binomial_order_statistic",
                    status="INSUFFICIENT_SESSIONS_FOR_INTERVAL",
                )
            else:
                lower_rank = int(binom.ppf(0.025, n, quantile))
                upper_rank = int(binom.ppf(0.975, n, quantile)) + 1
                low = (
                    min(float(ordered[lower_rank - 1]), estimate)
                    if lower_rank >= 1
                    else None
                )
                high = (
                    max(float(ordered[upper_rank - 1]), estimate)
                    if upper_rank <= n
                    else None
                )
                row.update(
                    estimate=estimate,
                    ci_95_low=low,
                    ci_95_high=high,
                    interval_method="exact_binomial_order_statistic",
                    status=(
                        "MEASURED"
                        if low is not None and high is not None
                        else "MEASURED_WITH_OPEN_CONFIDENCE_BOUND"
                    ),
                )
        rows.append(row)
    return rows


def _coverage_fields(usable: int, total: int, *, prefix: str) -> dict[str, Any]:
    low, high = wilson_interval(usable, total)
    return {
        f"{prefix}_sessions": usable,
        f"{prefix}_probability": usable / total,
        f"{prefix}_ci_95_low": low,
        f"{prefix}_ci_95_high": high,
    }


def summarize_iv(census: IVCensus) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize IV levels and paired changes with one vote per session."""

    session_values = census.session_values.copy()
    scopes = _scope_sessions(census)
    level_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    for scope, sessions in scopes.items():
        total = len(sessions)
        scoped = session_values[session_values["session"].isin(sessions)]
        for region in REGIONS:
            for estimand in ESTIMANDS:
                selected = scoped[
                    scoped["region"].eq(region) & scoped["estimand"].eq(estimand)
                ]
                baseline = selected[selected["start_time_et"].eq(BASELINE_START)][
                    ["session", "iv"]
                ].rename(columns={"iv": "baseline_iv"})
                for start in START_TIMES:
                    at_time = selected[selected["start_time_et"].eq(start)]
                    usable_values = at_time["iv"].dropna().to_numpy(float)
                    usable = len(usable_values)
                    coverage = _coverage_fields(usable, total, prefix="usable")
                    diagnostic = {
                        "scope": scope,
                        "start_time_et": start,
                        "region": region,
                        "estimand": estimand,
                        "total_sessions": total,
                        **coverage,
                        "median_selected_call_nodes": float(
                            at_time["selected_call_nodes"].median()
                        ),
                        "median_selected_put_nodes": float(
                            at_time["selected_put_nodes"].median()
                        ),
                        "sessions_with_solver_bound_exclusion": int(
                            (at_time["solver_bound_excluded_count"] > 0).sum()
                        ),
                        "solver_bound_excluded_rows": int(
                            at_time["solver_bound_excluded_count"].sum()
                        ),
                        "unit": "volatility_fraction",
                        "iv_40_point_otm_support": "UNSUPPORTED_OUTSIDE_STORED_LADDER",
                        "iv_60_point_otm_support": "UNSUPPORTED_OUTSIDE_STORED_LADDER",
                        "interpretation": "ATM_LEVEL_CLOCK_CURVE_NOT_STRIKE_SPECIFIC",
                    }
                    for statistic in _statistic_rows(usable_values):
                        level_rows.append({**diagnostic, **statistic})

                    paired = at_time[["session", "iv"]].merge(
                        baseline, on="session", how="inner", validate="one_to_one"
                    ).dropna(subset=["iv", "baseline_iv"])
                    differences = (
                        paired["iv"].to_numpy(float)
                        - paired["baseline_iv"].to_numpy(float)
                    )
                    paired_count = len(differences)
                    change_common = {
                        "scope": scope,
                        "from_start_time_et": BASELINE_START,
                        "to_start_time_et": start,
                        "region": region,
                        "estimand": estimand,
                        "total_sessions": total,
                        **_coverage_fields(paired_count, total, prefix="paired"),
                        "unit": "volatility_fraction_change",
                        "interval_scope": "pointwise_session_unit_not_simultaneous",
                        "era_interpretation": "DESCRIPTIVE_SOURCE_DATE_CONFOUNDED",
                    }
                    for statistic in _statistic_rows(differences):
                        change_rows.append({**change_common, **statistic})
    return (
        pd.DataFrame(level_rows),
        pd.DataFrame(change_rows),
        session_values,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnconditionalIVError(message)


def _validate_summary_intervals(frame: pd.DataFrame, coverage_prefix: str, name: str) -> None:
    total = frame["total_sessions"].to_numpy(int)
    usable = frame[f"{coverage_prefix}_sessions"].to_numpy(int)
    probability = frame[f"{coverage_prefix}_probability"].to_numpy(float)
    _require(np.array_equal(usable >= 0, np.ones(len(frame), dtype=bool)), f"{name}: negative coverage")
    _require((usable <= total).all(), f"{name}: coverage exceeds population")
    _require(
        np.allclose(probability, usable / total, rtol=0.0, atol=1e-12),
        f"{name}: coverage probability mismatch",
    )
    for row in frame[
        [
            f"{coverage_prefix}_sessions",
            "total_sessions",
            f"{coverage_prefix}_ci_95_low",
            f"{coverage_prefix}_ci_95_high",
        ]
    ].drop_duplicates().itertuples(index=False, name=None):
        count, trials, low, high = row
        expected_low, expected_high = wilson_interval(int(count), int(trials))
        _require(
            math.isclose(float(low), expected_low, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(high), expected_high, rel_tol=0.0, abs_tol=1e-12),
            f"{name}: coverage interval mismatch",
        )
    measured = frame[frame["status"].astype(str).str.startswith("MEASURED")]
    _require(np.isfinite(measured["estimate"].to_numpy(float)).all(), f"{name}: non-finite estimate")
    finite_low = measured["ci_95_low"].notna()
    finite_high = measured["ci_95_high"].notna()
    _require(
        (measured.loc[finite_low, "ci_95_low"] <= measured.loc[finite_low, "estimate"]).all()
        and (measured.loc[finite_high, "estimate"] <= measured.loc[finite_high, "ci_95_high"]).all(),
        f"{name}: interval does not contain estimate",
    )


def validate_iv_tables(
    census: IVCensus,
    iv_by_time: pd.DataFrame,
    iv_clock_change: pd.DataFrame,
) -> dict[str, Any]:
    """Fail closed on the estimator, population, coverage, and support claims."""

    values = census.session_values
    expected_values = len(census.sessions) * len(START_TIMES) * len(REGIONS) * len(ESTIMANDS)
    _require(len(values) == expected_values, "IV session-value row count drift")
    value_key = ["session", "start_time_et", "region", "estimand"]
    _require(not values.duplicated(value_key).any(), "duplicate IV session-value key")
    expected_summary = len(REPORTING_SCOPES) * len(START_TIMES) * len(REGIONS) * len(ESTIMANDS) * len(STATISTICS)
    _require(len(iv_by_time) == expected_summary, "IV level table row count drift")
    _require(len(iv_clock_change) == expected_summary, "IV change table row count drift")
    _require(
        not iv_by_time.duplicated(["scope", "start_time_et", "region", "estimand", "statistic"]).any(),
        "duplicate IV level key",
    )
    _require(
        not iv_clock_change.duplicated(
            ["scope", "from_start_time_et", "to_start_time_et", "region", "estimand", "statistic"]
        ).any(),
        "duplicate IV change key",
    )
    _validate_summary_intervals(iv_by_time, "usable", "iv_by_time")
    _validate_summary_intervals(iv_clock_change, "paired", "iv_clock_change")
    _require(set(values["region"]) == set(REGIONS), "IV regions drifted")
    _require(set(values["estimand"]) == set(ESTIMANDS), "IV estimands drifted")
    _require(
        set(values["iv_40_point_otm_support"])
        == {"UNSUPPORTED_OUTSIDE_STORED_LADDER"}
        and set(values["iv_60_point_otm_support"])
        == {"UNSUPPORTED_OUTSIDE_STORED_LADDER"},
        "unsupported IV wings were silently populated",
    )

    pivot = values.pivot(
        index=["session", "start_time_et", "region"],
        columns="estimand",
        values="iv",
    )
    complete = pivot.dropna(subset=["call_median", "put_median", "side_balanced"])
    _require(
        np.allclose(
            complete["side_balanced"].to_numpy(float),
            (
                complete["call_median"].to_numpy(float)
                + complete["put_median"].to_numpy(float)
            )
            / 2.0,
            rtol=0.0,
            atol=1e-15,
        ),
        "side-balanced IV does not equal equal-weighted side medians",
    )
    if census.strict_population:
        primary = values[
            values["region"].eq("atm_10pt_primary")
            & values["estimand"].eq("side_balanced")
        ]
        sensitivity = values[
            values["region"].eq("closest_atm_pair_sensitivity")
            & values["estimand"].eq("side_balanced")
        ]
        _require(primary["iv"].notna().all(), "primary ATM IV lacks full session-clock coverage")
        _require(sensitivity["iv"].notna().all(), "closest-ATM sensitivity lacks full coverage")
    baseline = iv_clock_change[
        iv_clock_change["to_start_time_et"].eq(BASELINE_START)
        & iv_clock_change["status"].astype(str).str.startswith("MEASURED")
    ]
    _require(
        np.allclose(baseline["estimate"].to_numpy(float), 0.0, rtol=0.0, atol=1e-15),
        "paired IV baseline change is not zero",
    )
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "discovered_ladder_sessions": len(census.input_manifest),
        "analyzed_ladder_sessions": len(census.sessions),
        "iv_session_value_rows": len(values),
        "iv_by_time_rows": len(iv_by_time),
        "iv_clock_change_rows": len(iv_clock_change),
        "primary_atm_band_points": ATM_BAND_POINTS,
        "primary_side_balanced_complete_sessions": int(
            values[
                values["region"].eq("atm_10pt_primary")
                & values["estimand"].eq("side_balanced")
            ]["iv"].notna().sum()
        ),
        "closest_pair_complete_sessions": int(
            values[
                values["region"].eq("closest_atm_pair_sensitivity")
                & values["estimand"].eq("side_balanced")
            ]["iv"].notna().sum()
        ),
        "solver_bound_rows_excluded": int(
            values[
                values["region"].eq("atm_10pt_primary")
                & values["estimand"].eq("side_balanced")
            ]["solver_bound_excluded_count"].sum()
        ),
        "iv_40_point_otm_support": "UNKNOWN_OUTSIDE_STORED_LADDER",
        "iv_60_point_otm_support": "UNKNOWN_OUTSIDE_STORED_LADDER",
        "session_unit_inference": "PASS",
        "era_interpretation": "DESCRIPTIVE_SOURCE_DATE_CONFOUNDED",
    }
