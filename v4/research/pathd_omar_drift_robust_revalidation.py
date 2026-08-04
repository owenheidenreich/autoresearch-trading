"""Does the omar survivor outlive a null that handles drift honestly?

The Option 0 re-screen left one survivor of sixty: ``omar_clipped_neg3_pos3``,
excess IC +0.052/+0.063/+0.078 at 15/30/60m against a drift-preserving wild
bootstrap. The same day, ``pathd_omar_causal_sizing.py`` demonstrated that this
null is mis-specified for this exact feature: it imposes the session's mean
drift at every minute, so a surrogate climbs smoothly and "high omar" implies
"still drifting up" -- visible as a monotone pooled surrogate profile spanning
9.94 SPX points where the real profile is flat.

The defect's direction is anti-conservative for the survivor: spurious drift
variance in the surrogate's omar dilutes the bounded-path reversion artifact in
the surrogate baseline, inflating ``real - surrogate`` on the reversion side.
The omar excess is a ~15% residual on a raw IC that is ~85% artifact, so this
must be resolved before anything is built on it.

Two repaired nulls, both martingale in drift arrival, both preserving the
intraday absolute-return profile:

- **zero-drift wild**: ``r* = s_t * r_t`` -- no mu term.  Exact |r_t| profile,
  no drift.  Directionally conservative for the reversion claim.
- **permuted-time wild**: ``r* = s_t * r_{pi(t)}`` -- signs on time-permuted
  returns.  Marginal |r| distribution preserved, volatility placement
  destroyed: the complementary weakness to the zero-drift wild.

Each null must first pass a KNOWN-ANSWER gate on the statistic where the defect
was demonstrated (pooled fixed-bin 60m profile): surrogate top-minus-bottom bin
spread <= 2.0 points and |Spearman rho| of bin means vs bin index <= 0.6.  The
original wild is re-run unchanged as a reference and is expected to FAIL the
gate -- a built-in positive control that the gate catches the defect.

DIAGNOSTIC-ONLY.  Third analysis of the same 213 development sessions, with the
hypothesis chosen after the previous two results.  Frozen by
``PATHD_OMAR_DRIFT_ROBUST_REVALIDATION_PREREGISTRATION_2026_08_04.md``.  No
training, no broker, no purchase, holdout untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from v4.research.pathd_phase1_entry import (
    DEVELOPMENT_SESSION_COUNT,
    _official_spx,
    development_sessions,
    entry_expanding_folds,
)
from v4.research.pathd_spx_directional_skill_screen import (
    HORIZON_MINUTES,
    SIDE_FREE_FEATURES,
    ScreenError,
    _availability_series,
    _boundaries_ns,
    _corpus_root,
    _session_features,
    _sha256_path,
)
from v4.research.pathd_spx_excess_skill_rescreen import (
    MINUTE_NS,
    _assert_kernel_parity,
    _features_from_path,
    _rank_ic,
    _selection_map,
    _wild_bootstrap,
)
from v4.research.autoresearch_v2.statistics import paired_summary, session_blocked_max_t
from v4.research.phase1_exit_model import stable_hash


SCHEMA_VERSION = "pathd.omar-drift-robust-revalidation.v1"
OMAR_FEATURE = "omar_clipped_neg3_pos3"
OMAR_COLUMN = SIDE_FREE_FEATURES.index(OMAR_FEATURE)

SURROGATES_PER_SESSION = 200
PERMUTATIONS = 20_000
PERMUTATION_SEED = 7203
SURROGATE_SEED = 90211
ALPHA = 0.05
MIN_SIGN_STABLE_FOLDS = 4
RAW_IC_PARITY_TOLERANCE = 1e-9

# Known-answer gate: pooled fixed-bin 60m statistic where the drift defect was
# demonstrated (defective wild: spread 9.94 points, rho ~ +1.0).
KA_HORIZON = 60
KA_BIN_EDGES = (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
KA_MAX_SPREAD_POINTS = 2.0
KA_MAX_ABS_RHO = 0.6

REPAIRED_NULLS = ("zero_drift_wild", "permuted_time_wild")
REFERENCE_NULL = "reference_wild"
NULLS = (*REPAIRED_NULLS, REFERENCE_NULL)

PREREGISTRATION = Path(
    "v4/docs/protocol101/training/research/"
    "PATHD_OMAR_DRIFT_ROBUST_REVALIDATION_PREREGISTRATION_2026_08_04.md"
)
RESCREEN_RECEIPT = Path(
    "v4/audit/autoresearch/pathd_spx_excess_skill_rescreen_2026_08_04/receipt.json"
)
OUTPUT_ROOT = Path(
    "v4/audit/autoresearch/pathd_omar_drift_robust_revalidation_2026_08_04"
)

SMOKE = os.environ.get("PATHD_DRIFT_ROBUST_SMOKE") == "1"
if SMOKE:  # engineering check only; numbers are non-evidentiary by pre-registration
    SURROGATES_PER_SESSION = 20
    PERMUTATIONS = 2_000
    OUTPUT_ROOT = Path(
        os.environ.get("PATHD_DRIFT_ROBUST_SMOKE_DIR", "/tmp")
    ) / "pathd_omar_drift_robust_smoke"


# --------------------------------------------------------------------------
# Surrogates
# --------------------------------------------------------------------------


def _zero_drift_wild(closes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Martingale wild bootstrap: sign-flip the raw log returns, no drift term."""

    log_returns = np.diff(np.log(closes))
    signs = rng.choice((-1.0, 1.0), size=len(log_returns))
    return closes[0] * np.exp(np.concatenate([[0.0], np.cumsum(log_returns * signs)]))


def _permuted_time_wild(closes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Martingale wild with permuted volatility placement.

    ``r*_t = s_t * r_{pi(t)}`` with a uniform within-session permutation ``pi``
    and iid Rademacher signs.  A true martingale that preserves the marginal
    absolute-return distribution while destroying the placement of the intraday
    volatility profile -- the complementary weakness to the zero-drift wild,
    which preserves placement exactly.

    Two drift-burst-preserving drafts of this second null were caught by the
    known-answer gate in the pre-registered smoke check: an uncentered
    block-mean resample (pooled surrogate spread 10.9 points, nearly the
    defective reference wild's 11.5) and a centered one (5.5 points -- a
    30-minute persistent drift burst IS short-horizon momentum at overlapping
    horizons, so "bursty drift but unpredictable" is self-contradictory at
    15-60m).  Both catches are recorded in the pre-registration.
    """

    log_returns = np.diff(np.log(closes))
    shuffled = log_returns[rng.permutation(len(log_returns))]
    signs = rng.choice((-1.0, 1.0), size=len(log_returns))
    return closes[0] * np.exp(np.concatenate([[0.0], np.cumsum(shuffled * signs)]))


SURROGATE_GENERATORS = {
    "zero_drift_wild": _zero_drift_wild,
    "permuted_time_wild": _permuted_time_wild,
    REFERENCE_NULL: _wild_bootstrap,
}


# --------------------------------------------------------------------------
# omar fast path (verified per session against the frozen-kernel mirror)
# --------------------------------------------------------------------------


def _omar_from_path(closes: np.ndarray, selection: np.ndarray) -> np.ndarray:
    session_range = np.maximum.accumulate(closes) - np.minimum.accumulate(closes)
    omar = np.divide(
        closes - closes[0],
        session_range,
        out=np.zeros(len(closes)),
        where=session_range != 0.0,
    )
    valid = selection >= 0
    pick = np.where(valid, selection, 0)
    return np.where(valid, np.clip(omar[pick], -3.0, 3.0), np.nan)


def _targets(
    closes: np.ndarray, selection: np.ndarray, target_selection: Mapping[int, np.ndarray]
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    start = closes[np.where(selection >= 0, selection, 0)]
    for horizon in HORIZON_MINUTES:
        end = target_selection[horizon]
        both = (selection >= 0) & (end >= 0)
        out[horizon] = np.where(both, closes[np.where(end >= 0, end, 0)] - start, np.nan)
    return out


def _omar_ics(omar: np.ndarray, targets: Mapping[int, np.ndarray]) -> np.ndarray:
    column = omar[:, None]
    return np.asarray(
        [_rank_ic(column, targets[horizon])[0] for horizon in HORIZON_MINUTES]
    )


def _member(horizon: int, sign: str) -> str:
    return f"{OMAR_FEATURE}|{horizon}m|{sign}"


MEMBERS = tuple(
    _member(horizon, sign) for horizon in HORIZON_MINUTES for sign in ("+", "-")
)


def _spread(ics: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for position, horizon in enumerate(HORIZON_MINUTES):
        value = ics[position]
        if np.isfinite(value):
            out[_member(horizon, "+")] = float(value)
            out[_member(horizon, "-")] = -float(value)
    return out


def _empty_family() -> dict[str, dict[str, float]]:
    return {member: {} for member in MEMBERS}


# --------------------------------------------------------------------------
# Known-answer pooled bins
# --------------------------------------------------------------------------


def _accumulate_bins(store: list[np.ndarray], omar: np.ndarray, target: np.ndarray) -> None:
    valid = np.isfinite(omar) & np.isfinite(target)
    if not valid.any():
        return
    index = np.clip(
        np.digitize(omar[valid], np.asarray(KA_BIN_EDGES[1:-1]), right=False),
        0,
        len(KA_BIN_EDGES) - 2,
    )
    np.add.at(store[0], index, target[valid])
    np.add.at(store[1], index, 1.0)


def _ka_gate(store: list[np.ndarray]) -> dict[str, Any]:
    counts = store[1]
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, store[0] / np.maximum(counts, 1.0), np.nan)
    filled = np.isfinite(means)
    spread = float(means[filled][-1] - means[filled][0]) if filled.sum() >= 2 else float("nan")
    if filled.sum() >= 3:
        x = rankdata(np.flatnonzero(filled))
        y = rankdata(means[filled])
        x, y = x - x.mean(), y - y.mean()
        denominator = np.sqrt(float(x @ x) * float(y @ y))
        rho = float(x @ y) / denominator if denominator > 0 else float("nan")
    else:
        rho = float("nan")
    passes = (
        np.isfinite(spread)
        and np.isfinite(rho)
        and abs(spread) <= KA_MAX_SPREAD_POINTS
        and abs(rho) <= KA_MAX_ABS_RHO
    )
    return {
        "bin_means": [None if not np.isfinite(v) else float(v) for v in means],
        "bin_counts": [int(v) for v in counts],
        "top_minus_bottom_points": None if not np.isfinite(spread) else spread,
        "spearman_rho_vs_bin_index": None if not np.isfinite(rho) else rho,
        "max_abs_spread_allowed": KA_MAX_SPREAD_POINTS,
        "max_abs_rho_allowed": KA_MAX_ABS_RHO,
        "passes": bool(passes),
    }


# --------------------------------------------------------------------------
# Fold utilities (same shapes as the Option 0 re-screen)
# --------------------------------------------------------------------------


def _fold_means(
    excess: Mapping[str, Mapping[str, float]], folds: Sequence[Mapping[str, Any]]
) -> dict[str, list[float | None]]:
    output: dict[str, list[float | None]] = {}
    for member, values in excess.items():
        means: list[float | None] = []
        for fold in folds:
            sample = [values[session] for session in fold["test"] if session in values]
            means.append(float(np.mean(sample)) if sample else None)
        output[member] = means
    return output


def _sign_stable(mean_value: float | None, fold_means: Sequence[float | None]) -> int:
    if mean_value is None or not np.isfinite(mean_value) or mean_value == 0.0:
        return 0
    want = np.sign(mean_value)
    return sum(
        1 for v in fold_means if v is not None and np.isfinite(v) and np.sign(v) == want
    )


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> dict[str, Any]:
    if OUTPUT_ROOT.exists() and not SMOKE:
        raise ScreenError(f"output already exists, refusing to re-run: {OUTPUT_ROOT}")
    corpus_root = _corpus_root()
    sessions = development_sessions(corpus_root)
    if len(sessions) != DEVELOPMENT_SESSION_COUNT:
        raise ScreenError(f"expected {DEVELOPMENT_SESSION_COUNT} dev sessions")
    if SMOKE:
        sessions = sessions[:8]
    print(
        f"corpus: {corpus_root}\nsessions: {len(sessions)}   "
        f"surrogates/session: {SURROGATES_PER_SESSION}   smoke: {SMOKE}"
    )

    excess = {null: _empty_family() for null in NULLS}
    # One combined control family across both repaired nulls, so the maxT
    # correction holds the overall false-INVALID rate at ALPHA.
    control: dict[str, dict[str, float]] = {
        f"{null}:{member}": {} for null in REPAIRED_NULLS for member in MEMBERS
    }
    measured_raw = _empty_family()
    ka_store = {null: [np.zeros(len(KA_BIN_EDGES) - 1), np.zeros(len(KA_BIN_EDGES) - 1)] for null in NULLS}
    ka_real = [np.zeros(len(KA_BIN_EDGES) - 1), np.zeros(len(KA_BIN_EDGES) - 1)]
    used, dropped = 0, []

    for position, session in enumerate(sessions, start=1):
        spx = _official_spx(
            corpus_root / "raw/index/spx_1m" / f"{session}.official_spx.parquet", session
        )
        boundaries = _boundaries_ns(session)
        try:
            kernel_features, _window, _names = _session_features(spx, session, boundaries)
        except ValueError as error:
            dropped.append({"session": session, "reason": str(error)})
            continue
        used += 1

        available, closes = _availability_series(spx)
        volumes = pd.to_numeric(spx["volume"], errors="raise").to_numpy(dtype=np.float64)
        order = np.argsort(
            (
                pd.to_datetime(spx["event_time"], utc=True)
                .astype("datetime64[ns, UTC]")
                .astype("int64")
                .to_numpy()
                + MINUTE_NS
            ),
            kind="mergesort",
        )
        volumes = volumes[order]

        selection = _selection_map(available, boundaries)
        target_selection = {
            horizon: _selection_map(available, boundaries + horizon * MINUTE_NS)
            for horizon in HORIZON_MINUTES
        }

        # Parity gates: full mirror against the frozen kernel, then the omar
        # fast path against the mirror's omar column.
        mirror = _features_from_path(closes, volumes, selection)
        _assert_kernel_parity(mirror, kernel_features, session)
        fast_omar = _omar_from_path(closes, selection)
        mirror_omar = mirror[:, OMAR_COLUMN]
        both_nan = np.isnan(fast_omar) & np.isnan(mirror_omar)
        if not bool(np.all(both_nan | np.isclose(fast_omar, mirror_omar, rtol=0.0, atol=1e-12))):
            raise ScreenError(f"omar fast path diverges from kernel mirror on {session}")

        real_targets = _targets(closes, selection, target_selection)
        real_ics = _omar_ics(fast_omar, real_targets)
        for member, value in _spread(real_ics).items():
            measured_raw[member][session] = value
        _accumulate_bins(ka_real, fast_omar, real_targets[KA_HORIZON])

        for null in NULLS:
            rng = np.random.default_rng(
                int(
                    hashlib.sha256(f"{SURROGATE_SEED}|{null}|{session}".encode()).hexdigest()[:16],
                    16,
                )
            )
            generator = SURROGATE_GENERATORS[null]
            surrogate_ics = np.empty((SURROGATES_PER_SESSION, len(HORIZON_MINUTES)))
            for draw in range(SURROGATES_PER_SESSION):
                path = generator(closes, rng)
                surrogate_omar = _omar_from_path(path, selection)
                surrogate_targets = _targets(path, selection, target_selection)
                surrogate_ics[draw] = _omar_ics(surrogate_omar, surrogate_targets)
                _accumulate_bins(
                    ka_store[null], surrogate_omar, surrogate_targets[KA_HORIZON]
                )
            surrogate_mean = np.nanmean(surrogate_ics, axis=0)
            for member, value in _spread(real_ics - surrogate_mean).items():
                excess[null][member][session] = value
            if null in REPAIRED_NULLS:
                control_ics = surrogate_ics[0] - np.nanmean(surrogate_ics[1:], axis=0)
                for member, value in _spread(control_ics).items():
                    control[f"{null}:{member}"][session] = value

        if position % 25 == 0 or position == len(sessions):
            print(f"  {position}/{len(sessions)} sessions  ({len(dropped)} dropped)")

    if not used:
        raise ScreenError("no usable sessions")

    # Cross-run raw-IC parity against the Option 0 receipt (full runs only).
    raw_summaries = {member: paired_summary(values) for member, values in measured_raw.items()}
    raw_parity: dict[str, Any] = {"checked": not SMOKE, "max_abs_deviation": None}
    if not SMOKE:
        receipt = json.loads(RESCREEN_RECEIPT.read_text())
        expected = {
            f"{row['feature']}|{row['horizon_minutes']}m|{row['sign']}": row["measured_raw_ic"]
            for row in receipt["results"]
            if row["feature"] == OMAR_FEATURE
        }
        deviations = [
            abs(raw_summaries[member]["mean"] - expected[member])
            for member in MEMBERS
            if member in expected and expected[member] is not None
        ]
        worst = max(deviations) if deviations else float("nan")
        raw_parity["max_abs_deviation"] = worst
        if not deviations or worst > RAW_IC_PARITY_TOLERANCE:
            raise ScreenError(
                f"raw omar ICs do not reproduce the Option 0 receipt (worst {worst})"
            )

    ka_results = {null: _ka_gate(ka_store[null]) for null in NULLS}
    ka_results["real"] = _ka_gate(ka_real)
    valid_nulls = [null for null in REPAIRED_NULLS if ka_results[null]["passes"]]

    folds = entry_expanding_folds(sessions if not SMOKE else development_sessions(corpus_root))
    control_test = session_blocked_max_t(
        control, permutations=PERMUTATIONS, seed=PERMUTATION_SEED
    )
    control_breaches = [
        name
        for name, summary in control_test.items()
        if summary["maxT_p_one_sided"] is not None
        and summary["maxT_p_one_sided"] < ALPHA
    ]
    rows: list[dict[str, Any]] = []
    for null in REPAIRED_NULLS:
        primary = session_blocked_max_t(
            excess[null], permutations=PERMUTATIONS, seed=PERMUTATION_SEED
        )
        fold_means = _fold_means(excess[null], folds)
        for member in MEMBERS:
            summary = primary[member]
            means = fold_means[member]
            stable = _sign_stable(summary["mean"], means)
            rows.append(
                {
                    "null": null,
                    "member": member,
                    "horizon_minutes": int(member.split("|")[1].removesuffix("m")),
                    "sign": member.split("|")[2],
                    "n_sessions": int(summary["n_sessions"]),
                    "measured_raw_ic": raw_summaries[member]["mean"],
                    "excess": summary["mean"],
                    "excess_se": summary["se"],
                    "excess_maxT_p": summary["maxT_p_one_sided"],
                    "sign_stable_folds": stable,
                    "fold_mean_excess": means,
                    "control_excess": control_test[f"{null}:{member}"]["mean"],
                    "control_maxT_p": control_test[f"{null}:{member}"]["maxT_p_one_sided"],
                    "passes": bool(
                        summary["maxT_p_one_sided"] is not None
                        and summary["maxT_p_one_sided"] < ALPHA
                        and stable >= MIN_SIGN_STABLE_FOLDS
                        and summary["mean"] is not None
                        and summary["mean"] > 0.0
                    ),
                }
            )
    reference_summary = {
        member: paired_summary(values) for member, values in excess[REFERENCE_NULL].items()
    }

    def survives(null: str) -> bool:
        by_member = {row["member"]: row for row in rows if row["null"] == null}
        anchor = by_member[_member(60, "-")]
        if not anchor["passes"]:
            return False
        for horizon in (15, 30):
            mean = by_member[_member(horizon, "-")]["excess"]
            if mean is not None and np.isfinite(mean) and mean < 0.0:
                return False
        return True

    survival = {null: survives(null) for null in REPAIRED_NULLS}
    if control_breaches:
        verdict = "INVALID"
    elif not valid_nulls:
        verdict = "NULL_REPAIR_FAILED_BOTH"
    elif len(valid_nulls) < len(REPAIRED_NULLS):
        verdict = (
            "OMAR_NULL_FRAGILE_CLOSED"
            if not all(survival[null] for null in valid_nulls)
            else "PARTIAL_NULL_REPAIR_OMAR_UNRESOLVED"
        )
    elif all(survival.values()):
        verdict = "OMAR_EXCESS_ROBUST_TO_DRIFT_NULLS"
    else:
        verdict = "OMAR_NULL_FRAGILE_CLOSED"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "diagnostic_only": True,
        "smoke": SMOKE,
        "third_look_disclosure": (
            "third analysis of the same development sessions; hypothesis chosen "
            "after the previous two results; nothing here is confirmatory"
        ),
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": _sha256_path(PREREGISTRATION),
        "corpus_root": str(corpus_root),
        "feature": OMAR_FEATURE,
        "horizon_minutes": list(HORIZON_MINUTES),
        "family_size_per_null": len(MEMBERS),
        "surrogates_per_session": SURROGATES_PER_SESSION,
        "repaired_nulls": list(REPAIRED_NULLS),
        "reference_null": REFERENCE_NULL,
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "surrogate_seed": SURROGATE_SEED,
        "alpha": ALPHA,
        "min_sign_stable_folds": MIN_SIGN_STABLE_FOLDS,
        "sessions_used": used,
        "sessions_declared": len(sessions),
        "dropped_sessions": dropped,
        "raw_ic_parity": raw_parity,
        "known_answer_gates": ka_results,
        "valid_nulls": valid_nulls,
        "survival_by_null": survival,
        "control_breaches": control_breaches,
        "results": rows,
        "reference_wild_excess": {
            member: reference_summary[member] for member in MEMBERS
        },
        "source_hashes": {
            "revalidation": _sha256_path(Path(__file__)),
            "rescreen": _sha256_path(Path("v4/research/pathd_spx_excess_skill_rescreen.py")),
            "screen": _sha256_path(Path("v4/research/pathd_spx_directional_skill_screen.py")),
        },
        "protected_holdout_opened": False,
        "paper_order_submitted": False,
        "model_trained": False,
    }
    payload = _clean(payload)
    payload["receipt_sha256"] = stable_hash(payload)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=SMOKE)
    (OUTPUT_ROOT / "receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    frame = pd.DataFrame(rows)
    for index in range(len(folds)):
        frame[f"fold{index}_mean_excess"] = [row["fold_mean_excess"][index] for row in rows]
    frame.drop(columns=["fold_mean_excess"]).to_csv(OUTPUT_ROOT / "results.csv", index=False)

    lines = [
        "# omar drift-robust re-validation -- results",
        "",
        f"verdict: {verdict}   DIAGNOSTIC-ONLY   smoke: {SMOKE}",
        f"sessions: {used}/{len(sessions)}   surrogates/session: {SURROGATES_PER_SESSION}",
        "",
        "known-answer gates (pooled 60m fixed-bin surrogate profile):",
    ]
    for null in (*NULLS, "real"):
        gate = ka_results[null]
        lines.append(
            f"  {null:>18}: spread {gate['top_minus_bottom_points']}   "
            f"rho {gate['spearman_rho_vs_bin_index']}   passes {gate['passes']}"
        )
    lines += [
        "",
        f"{'null':>18}{'member':>34}{'raw IC':>9}{'excess':>9}{'maxT p':>9}{'folds':>7}  passes",
        "-" * 96,
    ]
    for row in rows:
        def cell(value: Any, spec: str) -> str:
            return "nan" if value is None or not np.isfinite(value) else format(value, spec)
        lines.append(
            f"{row['null']:>18}{row['member']:>34}{cell(row['measured_raw_ic'], '+.4f'):>9}"
            f"{cell(row['excess'], '+.4f'):>9}{cell(row['excess_maxT_p'], '.4f'):>9}"
            f"{row['sign_stable_folds']:>5}/5  {row['passes']}"
        )
    lines += [
        "",
        f"survival by null: {survival}",
        f"valid nulls (KA pass): {valid_nulls}",
        f"control breaches: {control_breaches or 'NONE'}",
    ]
    (OUTPUT_ROOT / "results.md").write_text("\n".join(lines) + "\n")

    print(f"\nverdict: {verdict}")
    print(f"known-answer gates: " + ", ".join(f"{n}={ka_results[n]['passes']}" for n in NULLS))
    print(f"survival by null: {survival}")
    print(f"control breaches: {control_breaches or 'NONE'}")
    print(f"written: {OUTPUT_ROOT}")
    return payload


if __name__ == "__main__":
    main()
