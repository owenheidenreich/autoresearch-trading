"""Is anything left after the bounded-path artifact is removed?

Run 1 (``pathd_spx_directional_skill_screen.py``) measured a mean Spearman IC of
+0.522 and ~79% of it was mechanical: the features are functions of the price level
``P(t)``, the target ``P(t+h) - P(t)`` contains ``-P(t)``, and that shared term
forces a negative rank correlation for any bounded path.

This run corrects the *null*.  Three calibrations of the artifact disagreed about
whether the real data mean-reverts more or less than chance, and the disagreement
was surrogate mis-specification -- a constant-sigma walk gets the volatility profile
wrong, and a block bootstrap moves the loud minutes around.

The primary null here is a **drift-preserving Rademacher wild bootstrap**:
``r* = mu + e_t * s_t`` with iid signs.  It preserves the session drift, ``|e_t|``
at every timestamp (hence the intraday U-shape, the position of every volatility
cluster, and total realized variance), the opening level, and the real volume
series.  It destroys only the sign structure.  The surrogate differs from the real
path in exactly one respect: it cannot be predicted.

DIAGNOSTIC-ONLY.  Frozen by
``PATHD_SPX_EXCESS_SKILL_RESCREEN_PREREGISTRATION_2026_08_04.md``: a second look at
the same 213 sessions with the hypothesis chosen after run 1, so no outcome here is
confirmatory.  There is no estimator and no ``fit`` in this module.
"""
from __future__ import annotations

import hashlib
import json
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
    MIN_BOUNDARIES_PER_SESSION,
    SIDE_FREE_FEATURES,
    STALENESS_CAP_NS,
    ScreenError,
    _availability_series,
    _boundaries_ns,
    _corpus_root,
    _session_features,
    _sha256_path,
)
from v4.research.autoresearch_v2.statistics import paired_summary, session_blocked_max_t
from v4.research.phase1_exit_model import stable_hash


SCHEMA_VERSION = "pathd.spx-excess-skill-rescreen.v1"
MINUTE_NS = 60_000_000_000

# minute_of_session is a diagnostic addition, not a feature the entry model had.
MINUTE_FEATURE = "minute_of_session"
FEATURES = (*SIDE_FREE_FEATURES, MINUTE_FEATURE)
SURROGATES_PER_SESSION = 200
BLOCK_MINUTES = 30
PERMUTATIONS = 20_000
PERMUTATION_SEED = 5077
SURROGATE_SEED = 90210
ALPHA = 0.05
MIN_SIGN_STABLE_FOLDS = 4
# The mirror accumulates VWAP by cumsum where the kernel uses a dot product, so
# they differ at floating-point rounding.  Features are O(1)-O(100); any real
# discrepancy is O(0.1) or larger, so this stays far tighter than a real bug.
PARITY_TOLERANCE = 1e-6

PREREGISTRATION = Path(
    "v4/docs/protocol101/training/research/"
    "PATHD_SPX_EXCESS_SKILL_RESCREEN_PREREGISTRATION_2026_08_04.md"
)
OUTPUT_ROOT = Path("v4/audit/autoresearch/pathd_spx_excess_skill_rescreen_2026_08_04")


# --------------------------------------------------------------------------
# Availability: computed once from the real session, reused for every surrogate
# so real and surrogate share an identical NaN structure by construction.
# --------------------------------------------------------------------------


def _selection_map(available: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Bar index selected at each boundary, or -1 when stale/absent."""

    index = np.searchsorted(available, boundaries, side="right") - 1
    safe = np.where(index >= 0, index, 0)
    age = boundaries - available[safe]
    ok = (index >= 0) & (age >= 0) & (age <= STALENESS_CAP_NS)
    return np.where(ok, index, -1)


def _features_from_path(
    closes: np.ndarray, volumes: np.ndarray, selection: np.ndarray
) -> np.ndarray:
    """The nine kernel features, computed from a bare price path.

    Mirrors ``official_spx_market_window_from_rows`` -> ``feature_matrix`` exactly.
    ``_assert_kernel_parity`` proves the mirror on the real path before any
    surrogate is trusted; the run aborts if it does not hold.
    """

    count = len(closes)
    volumes = volumes.astype(np.float64)
    cumulative_pv = np.cumsum(closes * volumes)
    cumulative_v = np.cumsum(volumes)
    cumulative_c = np.cumsum(closes)
    steps = np.arange(1, count + 1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(
            cumulative_v > 0.0, cumulative_pv / np.maximum(cumulative_v, 1e-300),
            cumulative_c / steps,
        )
    session_range = np.maximum.accumulate(closes) - np.minimum.accumulate(closes)
    omar = np.divide(
        closes - closes[0], session_range, out=np.zeros(count), where=session_range != 0.0
    )

    def lagged(minutes: int) -> np.ndarray:
        out = np.full(count, np.nan)
        out[minutes:] = closes[minutes:] - closes[:-minutes]
        return out

    momentum_5m, momentum_15m = lagged(5), lagged(15)

    valid = selection >= 0
    pick = np.where(valid, selection, 0)
    spx = closes[pick]
    spx_denom = np.maximum(np.abs(spx), 1.0)
    picked_range = session_range[pick]
    range_denom = np.maximum(np.abs(picked_range), 1.0)
    gap = spx - vwap[pick]
    m5, m15 = momentum_5m[pick], momentum_15m[pick]

    values = {
        "spx_vwap_gap_points": gap,
        "spx_vwap_gap_bps": gap / spx_denom * 10_000.0,
        "spx_vwap_gap_over_session_range": gap / range_denom,
        "session_range_bps": picked_range / spx_denom * 10_000.0,
        "momentum_5m_bps": m5 / spx_denom * 10_000.0,
        "momentum_15m_bps": m15 / spx_denom * 10_000.0,
        "momentum_5m_over_session_range": m5 / range_denom,
        "momentum_15m_over_session_range": m15 / range_denom,
        "omar_clipped_neg3_pos3": np.clip(omar[pick], -3.0, 3.0),
    }
    out = np.full((len(selection), len(SIDE_FREE_FEATURES)), np.nan)
    for column, name in enumerate(SIDE_FREE_FEATURES):
        out[:, column] = np.where(valid, values[name], np.nan)
    return out


def _assert_kernel_parity(mirror: np.ndarray, kernel: np.ndarray, session: str) -> None:
    both_nan = np.isnan(mirror) & np.isnan(kernel)
    close = np.isclose(mirror, kernel, rtol=0.0, atol=PARITY_TOLERANCE, equal_nan=False)
    if not bool(np.all(both_nan | close)):
        worst = np.nanmax(np.abs(mirror - kernel))
        raise ScreenError(
            f"path-based features do not reproduce the frozen kernel on {session} "
            f"(max abs deviation {worst})"
        )


# --------------------------------------------------------------------------
# Rank IC, vectorized: 200 surrogates x 30 member-cells per session
# --------------------------------------------------------------------------


def _rank_ic(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Spearman IC of every column of ``features`` against ``target``.

    Pearson on ranks, restricted to rows finite in both.  ``rankdata`` supplies the
    same average-tie handling ``scipy.stats.spearmanr`` uses.
    """

    finite_target = np.isfinite(target)
    out = np.full(features.shape[1], np.nan)
    for column in range(features.shape[1]):
        valid = finite_target & np.isfinite(features[:, column])
        if int(valid.sum()) < MIN_BOUNDARIES_PER_SESSION:
            continue
        x, y = features[valid, column], target[valid]
        if np.all(x == x[0]) or np.all(y == y[0]):
            continue
        rx = rankdata(x)
        ry = rankdata(y)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denominator = np.sqrt(float(rx @ rx) * float(ry @ ry))
        if denominator > 0.0:
            out[column] = float(rx @ ry) / denominator
    return out


def _session_ic_block(
    closes: np.ndarray,
    volumes: np.ndarray,
    selection: np.ndarray,
    target_selection: Mapping[int, np.ndarray],
    minutes: np.ndarray,
) -> np.ndarray:
    """IC for every (feature, horizon) cell of one path.  Shape (10, 3)."""

    features = np.column_stack(
        [_features_from_path(closes, volumes, selection), minutes]
    )
    out = np.full((len(FEATURES), len(HORIZON_MINUTES)), np.nan)
    for position, horizon in enumerate(HORIZON_MINUTES):
        end = target_selection[horizon]
        both = (selection >= 0) & (end >= 0)
        target = np.where(
            both, closes[np.where(end >= 0, end, 0)] - closes[np.where(selection >= 0, selection, 0)], np.nan
        )
        out[:, position] = _rank_ic(features, target)
    return out


def _wild_bootstrap(closes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Drift-preserving Rademacher wild bootstrap of the session's log returns."""

    log_returns = np.diff(np.log(closes))
    mu = float(np.mean(log_returns))
    deviation = log_returns - mu
    signs = rng.choice((-1.0, 1.0), size=len(deviation))
    return closes[0] * np.exp(np.concatenate([[0.0], np.cumsum(mu + deviation * signs)]))


def _block_bootstrap(closes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Run 1's circular block bootstrap, retained unchanged as the secondary null."""

    log_returns = np.diff(np.log(closes))
    need = len(closes) - 1
    blocks = int(np.ceil(need / BLOCK_MINUTES))
    starts = rng.integers(0, len(log_returns), size=blocks)
    drawn = np.concatenate(
        [np.take(log_returns, np.arange(s, s + BLOCK_MINUTES), mode="wrap") for s in starts]
    )[:need]
    return closes[0] * np.exp(np.concatenate([[0.0], np.cumsum(drawn)]))


def _member(feature: str, horizon: int, sign: str) -> str:
    return f"{feature}|{horizon}m|{sign}"


def _spread(block: np.ndarray) -> dict[str, float]:
    """Flatten a (10, 3) IC block into signed member values."""

    out: dict[str, float] = {}
    for row, feature in enumerate(FEATURES):
        for column, horizon in enumerate(HORIZON_MINUTES):
            value = block[row, column]
            if np.isfinite(value):
                out[_member(feature, horizon, "+")] = float(value)
                out[_member(feature, horizon, "-")] = -float(value)
    return out


def _empty_family() -> dict[str, dict[str, float]]:
    return {
        _member(feature, horizon, sign): {}
        for feature in FEATURES
        for horizon in HORIZON_MINUTES
        for sign in ("+", "-")
    }


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


def main() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise ScreenError(f"output already exists, refusing to re-run: {OUTPUT_ROOT}")
    corpus_root = _corpus_root()
    sessions = development_sessions(corpus_root)
    if len(sessions) != DEVELOPMENT_SESSION_COUNT:
        raise ScreenError(f"expected {DEVELOPMENT_SESSION_COUNT} dev sessions")
    print(f"corpus: {corpus_root}\nsessions: {len(sessions)}   surrogates/session: {SURROGATES_PER_SESSION}")

    excess_wild = _empty_family()
    excess_block = _empty_family()
    excess_control = _empty_family()
    measured_raw = _empty_family()
    surrogate_noise: list[float] = []
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
        minutes = np.arange(len(boundaries), dtype=np.float64)

        # Parity gate: the path implementation must reproduce the frozen kernel.
        mirror = _features_from_path(closes, volumes, selection)
        _assert_kernel_parity(mirror, kernel_features, session)

        real_block = _session_ic_block(closes, volumes, selection, target_selection, minutes)

        rng = np.random.default_rng(
            int(hashlib.sha256(f"{SURROGATE_SEED}|{session}".encode()).hexdigest()[:16], 16)
        )
        wild = np.stack(
            [
                _session_ic_block(
                    _wild_bootstrap(closes, rng), volumes, selection, target_selection, minutes
                )
                for _ in range(SURROGATES_PER_SESSION)
            ]
        )
        block = np.stack(
            [
                _session_ic_block(
                    _block_bootstrap(closes, rng), volumes, selection, target_selection, minutes
                )
                for _ in range(SURROGATES_PER_SESSION)
            ]
        )

        wild_mean = np.nanmean(wild, axis=0)
        block_mean = np.nanmean(block, axis=0)
        surrogate_noise.append(
            float(np.nanmean(np.nanstd(wild, axis=0)) / np.sqrt(SURROGATES_PER_SESSION))
        )

        # Negative control: one held-out surrogate stands in for the real path.
        control_block = wild[0]
        control_mean = np.nanmean(wild[1:], axis=0)

        for target, values in (
            (excess_wild, real_block - wild_mean),
            (excess_block, real_block - block_mean),
            (excess_control, control_block - control_mean),
            (measured_raw, real_block),
        ):
            for member, value in _spread(values).items():
                target[member][session] = value

        if position % 25 == 0 or position == len(sessions):
            print(f"  {position}/{len(sessions)} sessions  ({len(dropped)} dropped)")

    if not used:
        raise ScreenError("no usable sessions")
    print(f"family: {len(excess_wild)} members; running {PERMUTATIONS} permutations")

    primary = session_blocked_max_t(excess_wild, permutations=PERMUTATIONS, seed=PERMUTATION_SEED)
    control = session_blocked_max_t(excess_control, permutations=PERMUTATIONS, seed=PERMUTATION_SEED)
    secondary = {member: paired_summary(values) for member, values in excess_block.items()}
    raw = {member: paired_summary(values) for member, values in measured_raw.items()}
    folds = entry_expanding_folds(sessions)
    fold_means = _fold_means(excess_wild, folds)

    rows = []
    for member, summary in primary.items():
        feature, horizon, sign = member.split("|")
        means = fold_means[member]
        stable = _sign_stable(summary["mean"], means)
        second = secondary[member]
        agrees = (
            second["mean"] is not None
            and summary["mean"] is not None
            and np.sign(second["mean"]) == np.sign(summary["mean"])
            and second["p_one_sided"] is not None
            and second["p_one_sided"] < ALPHA
        )
        primary_ok = summary["maxT_p_one_sided"] < ALPHA and stable >= MIN_SIGN_STABLE_FOLDS
        rows.append(
            {
                "member": member,
                "feature": feature,
                "horizon_minutes": int(horizon.removesuffix("m")),
                "sign": sign,
                "n_sessions": int(summary["n_sessions"]),
                "measured_raw_ic": raw[member]["mean"],
                "excess_wild": summary["mean"],
                "excess_wild_se": summary["se"],
                "excess_wild_maxT_p": summary["maxT_p_one_sided"],
                "excess_block": second["mean"],
                "excess_block_p": second["p_one_sided"],
                "sign_stable_folds": stable,
                "fold_mean_excess": means,
                "nulls_agree": bool(agrees),
                "status": (
                    "EXCESS_SURVIVES" if primary_ok and agrees
                    else "NULL_DEPENDENT" if primary_ok
                    else "NO_EXCESS"
                ),
                "control_excess": control[member]["mean"],
                "control_maxT_p": control[member]["maxT_p_one_sided"],
            }
        )
    rows.sort(key=lambda row: (row["excess_wild_maxT_p"], -(row["excess_wild"] or 0.0)))

    survivors = [row["member"] for row in rows if row["status"] == "EXCESS_SURVIVES"]
    null_dependent = [row["member"] for row in rows if row["status"] == "NULL_DEPENDENT"]
    control_breaches = [
        member
        for member, summary in control.items()
        if summary["maxT_p_one_sided"] < ALPHA
    ]
    verdict = (
        "INVALID" if control_breaches
        else "EXCESS_SKILL_CANDIDATE" if survivors
        else "NO_EXCESS_SKILL"
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "diagnostic_only": True,
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": _sha256_path(PREREGISTRATION),
        "corpus_root": str(corpus_root),
        "features": list(FEATURES),
        "horizon_minutes": list(HORIZON_MINUTES),
        "family_size": len(excess_wild),
        "surrogates_per_session": SURROGATES_PER_SESSION,
        "primary_null": "drift_preserving_rademacher_wild_bootstrap",
        "secondary_null": f"circular_block_bootstrap_{BLOCK_MINUTES}m",
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "surrogate_seed": SURROGATE_SEED,
        "alpha": ALPHA,
        "sessions_used": used,
        "sessions_declared": len(sessions),
        "dropped_sessions": dropped,
        "mean_surrogate_estimation_noise": float(np.mean(surrogate_noise)),
        "survivors": survivors,
        "null_dependent": null_dependent,
        "control_breaches": control_breaches,
        "results": rows,
        "source_hashes": {
            "rescreen": _sha256_path(Path(__file__)),
            "screen": _sha256_path(Path("v4/research/pathd_spx_directional_skill_screen.py")),
            "pathd_entry_features": _sha256_path(Path("v4/research/pathd_entry_features.py")),
            "protocol101_canonical_stage1_contract": _sha256_path(
                Path("v4/model/protocol101_canonical_stage1_contract.py")
            ),
        },
        "protected_holdout_opened": False,
        "paper_order_submitted": False,
        "model_trained": False,
    }
    payload = _clean(payload)
    payload["receipt_sha256"] = stable_hash(payload)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    frame = pd.DataFrame(rows)
    for index in range(len(folds)):
        frame[f"fold{index}_mean_excess"] = [row["fold_mean_excess"][index] for row in rows]
    frame.drop(columns=["fold_mean_excess"]).to_csv(OUTPUT_ROOT / "results.csv", index=False)

    lines = [
        "# SPX excess-skill re-screen (Option 0) -- results",
        "",
        f"verdict: {verdict}   DIAGNOSTIC-ONLY",
        f"sessions: {used}/{len(sessions)}   family: {len(excess_wild)}"
        f"   surrogates/session: {SURROGATES_PER_SESSION}",
        f"primary null: drift-preserving Rademacher wild bootstrap"
        f"   secondary: circular block bootstrap {BLOCK_MINUTES}m",
        "",
        f"{'member':46}{'raw IC':>9}{'excess':>9}{'maxT p':>9}{'blk exc':>9}"
        f"{'blk p':>8}{'folds':>7}  status",
        "-" * 110,
    ]
    for row in rows[:30]:
        def cell(value: Any, spec: str) -> str:
            return "nan" if value is None or not np.isfinite(value) else format(value, spec)
        lines.append(
            f"{row['member']:46}{cell(row['measured_raw_ic'], '+.4f'):>9}"
            f"{cell(row['excess_wild'], '+.4f'):>9}{row['excess_wild_maxT_p']:>9.4f}"
            f"{cell(row['excess_block'], '+.4f'):>9}{cell(row['excess_block_p'], '.4f'):>8}"
            f"{row['sign_stable_folds']:>5}/5  {row['status']}"
        )
    lines += [
        "",
        f"survivors (both nulls agree): {survivors or 'NONE'}",
        f"null-dependent (primary only): {null_dependent or 'NONE'}",
        f"control breaches: {control_breaches or 'NONE'}",
        f"mean surrogate estimation noise: {np.mean(surrogate_noise):.5f}",
    ]
    (OUTPUT_ROOT / "results.md").write_text("\n".join(lines) + "\n")

    print(f"\nverdict: {verdict}")
    print(f"survivors: {survivors or 'NONE'}")
    print(f"null-dependent: {null_dependent or 'NONE'}")
    print(f"control breaches: {control_breaches or 'NONE'}")
    return payload


if __name__ == "__main__":
    main()
