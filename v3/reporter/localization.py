"""Localization reporter.

Given an oracle cohort (bars labeled by outcome: `entered_right`, `side_error`,
`abstention`, `guardrail_suppression`) and a feature extractor, report the
feature's distribution on the cohort and on a matched control sample.

This reporter explicitly measures a **localization** quantity:
  P(feature passes | bar is in cohort)  vs
  P(feature passes | bar is in matched control)
It CANNOT produce a "capture rate" — that requires a trigger predicate on the
full bar universe, which is the job of `trigger_quality.py`.

When the feature is direction-dependent (e.g., filter B from the legacy
combined_confluence stack: sigma_pos ≤ 0.5 for calls, sigma_pos ≥ −0.5 for
puts), the control cohort MUST match the oracle-direction distribution of the
cohort. The prior `combined_confluence.py` implementation had a bug at
line 190 (random controls all labeled `oracle_direction="call"`), which
biased filter-B acceptance downward on the call side and produced a
meaningless denominator for the "1.92×" enrichment claim. This reporter is
the replacement with matched-direction controls.

Standard use: see `main()` — reproduces the Experiment 5 rerun against the
latest attribution JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from v2.core import market_structure
from v3.harness.v2_adapter import V2Dataset


# --- Types ------------------------------------------------------------------

FeatureFn = Callable[[V2Dataset, str, int, str], Optional[float]]
"""(dataset, day, minute, direction) -> feature value (or None if undefined)."""

SliceFn = Callable[[V2Dataset, str, int], bool]
"""(dataset, day, minute) -> whether this bar is in the slice."""


@dataclass
class LocalizationCohort:
    """A named set of (day, minute, direction) triples drawn from attribution."""

    name: str
    entries: list[tuple[str, int, str]]


@dataclass
class LocalizationMeasurement:
    cohort_name: str
    feature_name: str
    control_strategy: str
    n_cohort: int
    n_control: int
    # Continuous stats (always reported)
    cohort_median: float
    cohort_mean: float
    control_median: float
    control_mean: float
    # Binary stats (present iff a threshold was supplied)
    binary_threshold: Optional[float]
    cohort_positive_rate: Optional[float]
    control_positive_rate: Optional[float]
    enrichment_ratio: Optional[float]


# --- Cohort construction ----------------------------------------------------

def cohorts_from_attribution_json(path: str) -> dict[str, LocalizationCohort]:
    """Parse per-session JSON lines at the bottom of an attribution_full file.

    Returns one cohort per outcome (`entered_right`, `side_error`,
    `abstention`, `guardrail_suppression`).
    """
    buckets: dict[str, list[tuple[str, int, str]]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        outcome = row.get("outcome")
        direction = row.get("oracle_direction")
        if outcome not in ("entered_right", "side_error", "abstention", "guardrail_suppression"):
            continue
        if direction not in ("call", "put"):
            continue
        buckets.setdefault(outcome, []).append(
            (str(row["day"]), int(row["oracle_bar"]), direction)
        )
    return {name: LocalizationCohort(name=name, entries=entries) for name, entries in buckets.items()}


def latest_attribution_file(reference_dir: str = "v3/reference") -> str:
    files = sorted(glob.glob(f"{reference_dir}/attribution_full_*.txt"))
    if not files:
        raise FileNotFoundError(f"no attribution_full_*.txt in {reference_dir}")
    return files[-1]


# --- Matched controls -------------------------------------------------------

def matched_controls(
    ds: V2Dataset,
    cohort: LocalizationCohort,
    multiplier: int = 10,
    rng_seed: int = 101,
) -> list[tuple[str, int, str]]:
    """Sample random (day, minute, direction) triples matched to the cohort.

    Matching dimensions:
    - minute: drawn uniformly from the cohort's own minute distribution
    - direction: drawn uniformly from the cohort's own direction distribution
    - day: any day in the dataset

    This removes the direction-hardcoded bias of combined_confluence.py:190.
    """
    rng = np.random.default_rng(rng_seed)
    cohort_minutes = np.asarray([m for _, m, _ in cohort.entries], dtype=int)
    cohort_dirs = np.asarray([d for _, _, d in cohort.entries])
    all_days = sorted(set(ds.dates))
    n = len(cohort.entries) * max(multiplier, 1)
    days = rng.choice(all_days, size=n)
    minutes = rng.choice(cohort_minutes, size=n)
    dirs = rng.choice(cohort_dirs, size=n)
    return [(str(d), int(m), str(x)) for d, m, x in zip(days, minutes, dirs)]


# --- Measurement ------------------------------------------------------------

def _apply_feature(
    ds: V2Dataset,
    entries: Iterable[tuple[str, int, str]],
    feature_fn: FeatureFn,
) -> np.ndarray:
    values: list[float] = []
    for day, minute, direction in entries:
        v = feature_fn(ds, day, minute, direction)
        if v is None:
            continue
        values.append(float(v))
    return np.asarray(values, dtype=float)


def measure_localization(
    ds: V2Dataset,
    cohort: LocalizationCohort,
    feature_name: str,
    feature_fn: FeatureFn,
    *,
    control_strategy: str = "matched",
    control_multiplier: int = 10,
    binary_threshold: Optional[float] = None,
    binary_direction: str = "<=",
    rng_seed: int = 101,
) -> LocalizationMeasurement:
    """Compute cohort / control distributions and (if `binary_threshold` is set)
    an enrichment ratio.

    `binary_direction` controls the predicate polarity:
      "<="  cohort_positive = (feature_value <= threshold)
      ">="  cohort_positive = (feature_value >= threshold)
    """
    if control_strategy == "matched":
        controls = matched_controls(ds, cohort, multiplier=control_multiplier, rng_seed=rng_seed)
    else:
        # Unmatched baseline (random day + random minute in [0, 389] + cohort direction mix)
        rng = np.random.default_rng(rng_seed)
        all_days = sorted(set(ds.dates))
        cohort_dirs = np.asarray([d for _, _, d in cohort.entries])
        n = len(cohort.entries) * max(control_multiplier, 1)
        days = rng.choice(all_days, size=n)
        minutes = rng.integers(0, 390, size=n)
        dirs = rng.choice(cohort_dirs, size=n) if len(cohort_dirs) else np.array(["call"] * n)
        controls = [(str(d), int(m), str(x)) for d, m, x in zip(days, minutes, dirs)]

    cohort_vals = _apply_feature(ds, cohort.entries, feature_fn)
    control_vals = _apply_feature(ds, controls, feature_fn)

    cohort_median = float(np.median(cohort_vals)) if cohort_vals.size else float("nan")
    cohort_mean = float(cohort_vals.mean()) if cohort_vals.size else float("nan")
    control_median = float(np.median(control_vals)) if control_vals.size else float("nan")
    control_mean = float(control_vals.mean()) if control_vals.size else float("nan")

    cohort_rate: Optional[float] = None
    control_rate: Optional[float] = None
    enrichment: Optional[float] = None
    if binary_threshold is not None and cohort_vals.size and control_vals.size:
        if binary_direction == "<=":
            cohort_rate = float((cohort_vals <= binary_threshold).mean())
            control_rate = float((control_vals <= binary_threshold).mean())
        elif binary_direction == ">=":
            cohort_rate = float((cohort_vals >= binary_threshold).mean())
            control_rate = float((control_vals >= binary_threshold).mean())
        else:
            raise ValueError(f"binary_direction must be '<=' or '>=', got {binary_direction!r}")
        enrichment = (cohort_rate / control_rate) if control_rate > 0 else float("inf")

    return LocalizationMeasurement(
        cohort_name=cohort.name,
        feature_name=feature_name,
        control_strategy=control_strategy,
        n_cohort=int(cohort_vals.size),
        n_control=int(control_vals.size),
        cohort_median=cohort_median,
        cohort_mean=cohort_mean,
        control_median=control_median,
        control_mean=control_mean,
        binary_threshold=binary_threshold,
        cohort_positive_rate=cohort_rate,
        control_positive_rate=control_rate,
        enrichment_ratio=enrichment,
    )


# --- Built-in feature extractors --------------------------------------------

def _bar_close(ds: V2Dataset, day: str, minute: int) -> Optional[float]:
    try:
        day_start, day_end = ds.day_bar_range(day)
    except (ValueError, IndexError):
        return None
    abs_idx = day_start + minute
    if abs_idx >= day_end:
        return None
    c = float(ds.spot_prices[abs_idx])
    return c if c > 0 else None


def feature_sigma_pos(ds: V2Dataset, day: str, minute: int, direction: str) -> Optional[float]:
    """SPY-derived σ-position (SPX-scaled). Direction-agnostic continuous."""
    spy_day = ds.spy_vwap.get(day) if ds.spy_vwap is not None else None
    if spy_day is None:
        return None
    close = _bar_close(ds, day, minute)
    if close is None:
        return None
    return market_structure.sigma_pos(spy_day, minute, close)


def feature_direction_aligned_sigma(
    ds: V2Dataset, day: str, minute: int, direction: str
) -> Optional[float]:
    """Filter-B alignment test (direction-aware).

    For direction='call', returns +1 if sigma_pos ≤ +0.5, else 0.
    For direction='put',  returns +1 if sigma_pos ≥ −0.5, else 0.

    This is the `combined_confluence.py` filter B, re-posed as a per-entry
    feature. Controls must be direction-matched or the B rate is biased
    (that is exactly the `combined_confluence.py:190` bug).
    """
    s = feature_sigma_pos(ds, day, minute, direction)
    if s is None:
        return None
    if direction == "call":
        return 1.0 if s <= 0.5 else 0.0
    return 1.0 if s >= -0.5 else 0.0


def _day_omar(ds: V2Dataset, day: str, _omar_cache: dict[str, dict[str, float]]) -> Optional[dict[str, float]]:
    return _omar_cache.get(day)


def build_omar_cache() -> dict[str, dict[str, float]]:
    """Per-day OMAR constants via the shared helper."""
    spx_bars = market_structure.build_spx_bars()
    return market_structure.build_omar_map(spx_bars)


def make_feature_omar_retest_within(
    omar_cache: dict[str, dict[str, float]], threshold: float = 0.5
) -> FeatureFn:
    """Factory for the OMAR-retest filter-C: close within `threshold`×OMAR.range
    of any of omar.high / omar.low / omar.mid."""

    def _fn(ds: V2Dataset, day: str, minute: int, direction: str) -> Optional[float]:
        omar = omar_cache.get(day)
        if omar is None or omar["range"] <= 0:
            return None
        close = _bar_close(ds, day, minute)
        if close is None:
            return None
        r = omar["range"]
        for k in ("high", "low", "mid"):
            if abs(close - omar[k]) / r <= threshold:
                return 1.0
        return 0.0

    return _fn


def make_feature_b_and_c(
    omar_cache: dict[str, dict[str, float]],
    omar_threshold: float = 0.5,
) -> FeatureFn:
    """Factory for the legacy B∩C predicate, evaluated per entry with proper
    direction handling. Returns 1.0 if both B and C pass, else 0.0."""

    retest = make_feature_omar_retest_within(omar_cache, threshold=omar_threshold)

    def _fn(ds: V2Dataset, day: str, minute: int, direction: str) -> Optional[float]:
        b = feature_direction_aligned_sigma(ds, day, minute, direction)
        c = retest(ds, day, minute, direction)
        if b is None or c is None:
            return None
        return 1.0 if (b > 0.5 and c > 0.5) else 0.0

    return _fn


# --- Default slices ---------------------------------------------------------

def slice_late_session(ds: V2Dataset, day: str, minute: int) -> bool:
    return 40 <= minute <= 120


def slice_inside_first15(ds: V2Dataset, day: str, minute: int) -> bool:
    high, low = ds.first15_by_day.get(day, (0.0, 0.0))
    if high <= 0 or low <= 0:
        return False
    close = _bar_close(ds, day, minute)
    if close is None:
        return False
    return low <= close <= high


def make_slice_near_omar(
    omar_cache: dict[str, dict[str, float]], threshold: float = 0.5
) -> SliceFn:
    def _fn(ds: V2Dataset, day: str, minute: int) -> bool:
        omar = omar_cache.get(day)
        if omar is None or omar["range"] <= 0:
            return False
        close = _bar_close(ds, day, minute)
        if close is None:
            return False
        r = omar["range"]
        return any(abs(close - omar[k]) / r <= threshold for k in ("high", "low", "mid"))
    return _fn


def default_slices(omar_cache: dict[str, dict[str, float]]) -> dict[str, SliceFn]:
    """The 3 research-proven slices. Additional slices must be opt-in via the
    CLI / programmatic config; do NOT canonicalize null / saturated slices."""
    return {
        "late_session": slice_late_session,
        "inside_first15": slice_inside_first15,
        "near_omar": make_slice_near_omar(omar_cache, threshold=0.5),
    }


def filter_cohort_by_slice(cohort: LocalizationCohort, ds: V2Dataset, slice_fn: SliceFn, suffix: str) -> LocalizationCohort:
    entries = [
        (day, minute, direction)
        for (day, minute, direction) in cohort.entries
        if slice_fn(ds, day, minute)
    ]
    return LocalizationCohort(name=f"{cohort.name}__{suffix}", entries=entries)


# --- Formatting -------------------------------------------------------------

def _fmt_rate(r: Optional[float]) -> str:
    if r is None or np.isnan(r):
        return "   n/a"
    return f"{100*r:5.1f}%"


def _fmt_ratio(x: Optional[float]) -> str:
    if x is None or np.isnan(x) or np.isinf(x):
        return "  n/a"
    return f"{x:4.2f}x"


def format_measurement(m: LocalizationMeasurement) -> str:
    head = (
        f"{m.cohort_name:<28} {m.feature_name:<28} "
        f"n_co={m.n_cohort:>5} n_ct={m.n_control:>6}"
    )
    dist = (
        f"  cohort median={m.cohort_median:+7.3f} mean={m.cohort_mean:+7.3f} "
        f" control median={m.control_median:+7.3f} mean={m.control_mean:+7.3f}"
    )
    if m.binary_threshold is None:
        return f"{head}\n{dist}"
    enr = (
        f"  cohort_rate={_fmt_rate(m.cohort_positive_rate)} "
        f"control_rate={_fmt_rate(m.control_positive_rate)} "
        f"enrichment={_fmt_ratio(m.enrichment_ratio)} "
        f"(threshold={m.binary_threshold} direction={'<=' if m.cohort_positive_rate is not None else '??'})"
    )
    return f"{head}\n{dist}\n{enr}"


# --- CLI --------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Localization reporter")
    p.add_argument("--attribution", default=None,
                   help="Path to attribution_full_*.txt (defaults to latest in v3/reference/)")
    p.add_argument("--control-multiplier", type=int, default=10)
    p.add_argument("--rng-seed", type=int, default=101)
    args = p.parse_args(argv)

    attribution_path = args.attribution or latest_attribution_file()
    print(f"Loading attribution from {attribution_path}")
    cohorts = cohorts_from_attribution_json(attribution_path)
    for name, cohort in cohorts.items():
        print(f"  cohort {name}: n={len(cohort.entries)}")

    print("Loading v2 dataset + OMAR cache...")
    ds = V2Dataset.load()
    omar_cache = build_omar_cache()
    slices = default_slices(omar_cache)
    b_and_c = make_feature_b_and_c(omar_cache, omar_threshold=0.5)

    print()
    print("=" * 100)
    print("Feature: sigma_pos (continuous, SPY-derived)")
    print("=" * 100)
    for name in ("entered_right", "side_error", "abstention"):
        cohort = cohorts.get(name)
        if cohort is None:
            continue
        m = measure_localization(
            ds, cohort,
            feature_name="sigma_pos",
            feature_fn=feature_sigma_pos,
            control_strategy="matched",
            control_multiplier=args.control_multiplier,
            rng_seed=args.rng_seed,
        )
        print(format_measurement(m))
        print()

    print("=" * 100)
    print("Experiment 5 rerun: B∩C predicate with MATCHED-direction controls")
    print("  (Replaces the tainted '1.92× enrichment' from combined_confluence.py:190")
    print("   — that script hardcoded oracle_direction='call' for ALL random controls,")
    print("   which biased filter-B acceptance and made the enrichment number meaningless.)")
    print("=" * 100)
    for name in ("entered_right", "side_error", "abstention"):
        cohort = cohorts.get(name)
        if cohort is None:
            continue
        m = measure_localization(
            ds, cohort,
            feature_name="B∩C (sigma dir + OMAR retest)",
            feature_fn=b_and_c,
            control_strategy="matched",
            control_multiplier=args.control_multiplier,
            binary_threshold=0.5,
            binary_direction=">=",
            rng_seed=args.rng_seed,
        )
        print(format_measurement(m))
        print()

    print("=" * 100)
    print("Stratified by default slices (abstention cohort only, B∩C predicate)")
    print("=" * 100)
    abstention = cohorts.get("abstention")
    if abstention is not None:
        for slice_name, slice_fn in slices.items():
            sub = filter_cohort_by_slice(abstention, ds, slice_fn, suffix=slice_name)
            if not sub.entries:
                print(f"{sub.name}: n=0 (slice empty)")
                continue
            m = measure_localization(
                ds, sub,
                feature_name="B∩C",
                feature_fn=b_and_c,
                control_strategy="matched",
                control_multiplier=args.control_multiplier,
                binary_threshold=0.5,
                binary_direction=">=",
                rng_seed=args.rng_seed,
            )
            print(format_measurement(m))
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
