"""Do the nine side-free SPX context features rank-predict SPX forward returns?

Every Path-D campaign measured a whole strategy at once -- features, an option, an
entry law, an exit law, and friction -- and each returned ``NO_EDGE``.  That design
cannot separate "the features carry no directional information" from "the 0DTE
long-premium wrapper destroys the information they carry".

This screen deletes the option.  It measures the nine features directly against the
index, at the frozen t-60s availability clock, over the 215 development sessions,
with a family-wise maxT correction.  There is no estimator and no ``fit`` anywhere
in this module: the only statistic is a per-session Spearman rank correlation.

Frozen by ``PATHD_SPX_DIRECTIONAL_SKILL_PREREGISTRATION_2026_08_04.md``.
"""
from __future__ import annotations

from datetime import datetime, time
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from v4.model.protocol101_canonical_stage1_contract import (
    CONTEXT_FEATURES,
    FEATURE_NAMES,
    feature_matrix,
)
from v4.research.pathd_entry_features import official_spx_market_window_from_rows
from v4.research.pathd_phase1_entry import (
    DEVELOPMENT_SESSION_COUNT,
    _official_spx,
    development_sessions,
    entry_expanding_folds,
)
from v4.research.autoresearch_v2.statistics import paired_summary, session_blocked_max_t
from v4.research.phase1_exit_model import stable_hash


SCHEMA_VERSION = "pathd.spx-directional-skill-screen.v1"
NY = ZoneInfo("America/New_York")
MINUTE_NS = 60_000_000_000
STALENESS_CAP_NS = 90_000_000_000

# Frozen by the pre-registration.  The three option-side alignment flags are
# excluded: they are sign indicators of features already in the family, and are
# undefined once the option is removed.
SIDE_FREE_FEATURES = tuple(
    name for name in CONTEXT_FEATURES if not name.endswith("_side_alignment_flag")
)
HORIZON_MINUTES = (15, 30, 60)
FIRST_BOUNDARY = time(10, 0)
LAST_BOUNDARY = time(15, 0)
MIN_BOUNDARIES_PER_SESSION = 30
PERMUTATIONS = 20_000
PERMUTATION_SEED = 2033
ALPHA = 0.05
MIN_SIGN_STABLE_FOLDS = 4
MAX_DROPPED_SESSIONS = 10

CORPUS_ROOTS = (
    Path("/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31"),
    Path.home() / ".autoresearch-trading/pathd_2025-08-01_2026-07-31",
)
PREREGISTRATION = Path(
    "v4/docs/protocol101/training/research/"
    "PATHD_SPX_DIRECTIONAL_SKILL_PREREGISTRATION_2026_08_04.md"
)
OUTPUT_ROOT = Path(
    "v4/audit/autoresearch/pathd_spx_directional_skill_screen_2026_08_04"
)


class ScreenError(RuntimeError):
    """Raised when the frozen screen contract is violated."""


def _corpus_root() -> Path:
    for candidate in CORPUS_ROOTS:
        if (candidate / "raw/databento/opra_spxw_cbbo_1m").is_dir():
            return candidate
    raise ScreenError(f"no Path-D corpus found at any of {[str(p) for p in CORPUS_ROOTS]}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boundaries_ns(session: str) -> np.ndarray:
    """Every minute 10:00-15:00 ET inclusive, as exact UTC nanoseconds."""

    day = datetime.fromisoformat(session).date()
    first = int(datetime.combine(day, FIRST_BOUNDARY, tzinfo=NY).timestamp() * 1e9)
    last = int(datetime.combine(day, LAST_BOUNDARY, tzinfo=NY).timestamp() * 1e9)
    if (last - first) % MINUTE_NS:
        raise ScreenError(f"boundary grid is not minute-aligned for {session}")
    return np.arange(first, last + MINUTE_NS, MINUTE_NS, dtype=np.int64)


def _availability_series(spx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The (available_at_ns, close) pair the frozen kernel builds internally.

    ThetaData ``event_time`` is the bar-open stamp, so a bar becomes available one
    minute later.  Reproduced here only so the forward target can use the identical
    rule; ``_assert_close_parity`` proves the reproduction is exact.
    """

    # The corpus stores datetime64[us]; an unqualified astype("int64") would yield
    # MICROseconds and silently shift the clock by 1000x.  Pin the unit first.
    event_ns = (
        pd.to_datetime(spx["event_time"], utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy()
    )
    available = event_ns + MINUTE_NS
    closes = pd.to_numeric(spx["close"], errors="raise").to_numpy(dtype=np.float64)
    order = np.argsort(available, kind="mergesort")
    available, closes = available[order], closes[order]
    if len(np.unique(available)) != len(available):
        raise ScreenError("official SPX availability clocks are duplicated")
    return available, closes


def _close_available_at(
    available: np.ndarray, closes: np.ndarray, boundaries: np.ndarray
) -> np.ndarray:
    """Latest close available at each boundary, subject to the 90s staleness cap."""

    index = np.searchsorted(available, boundaries, side="right") - 1
    valid = index >= 0
    safe = np.where(valid, index, 0)
    age = boundaries - available[safe]
    valid &= (age >= 0) & (age <= STALENESS_CAP_NS)
    return np.where(valid, closes[safe], np.nan)


def _assert_close_parity(kernel_window: np.ndarray, names: Sequence[str], mine: np.ndarray) -> None:
    kernel_close = kernel_window[:, list(names).index("spx_close")]
    both_nan = np.isnan(kernel_close) & np.isnan(mine)
    equal = both_nan | (kernel_close == mine)
    if not bool(np.all(equal)):
        raise ScreenError(
            "forward-target availability rule does not reproduce the frozen kernel close"
        )


def _session_features(
    spx: pd.DataFrame, session: str, boundaries: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Nine side-free context features at every boundary, from the frozen kernel.

    One kernel call per session rather than one per boundary: with
    ``history_minutes = len(boundaries)`` and the decision clock at the final
    boundary, output row ``i`` is computed for boundary ``i`` by exactly the same
    code path a per-boundary call would take.
    """

    window, names, _ = official_spx_market_window_from_rows(
        spx.to_dict("records"),
        session=session,
        decision_time_ns=int(boundaries[-1]),
        history_minutes=int(len(boundaries)),
    )
    if window.shape[0] != len(boundaries):
        raise ScreenError("kernel window does not align with the boundary grid")

    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    columns = [index[name] for name in SIDE_FREE_FEATURES]
    ladder = np.full((1, 2, 1), np.nan, dtype=np.float64)
    offsets = np.asarray([0.0], dtype=np.float64)
    values = np.full((len(boundaries), len(SIDE_FREE_FEATURES)), np.nan, dtype=np.float64)
    spx_close = window[:, list(names).index("spx_close")]
    for position, boundary in enumerate(boundaries):
        close = float(spx_close[position])
        if not np.isfinite(close):
            continue
        matrix = feature_matrix(
            {
                "decision_time": pd.Timestamp(int(boundary), unit="ns", tz="UTC"),
                "atm_strike": float(round(close / 5.0) * 5.0),
                "strike_offsets": offsets,
                "rights": ("C", "P"),
                "option_ladder": ladder,
                "feature_names": ("mid",),
                "market_window": window[position : position + 1],
                "market_feature_names": names,
            }
        )
        values[position] = matrix[0, 0, columns]
    return values, window, tuple(names)


def _session_records(corpus_root: Path, session: str) -> dict[str, Any] | None:
    """Per-session features and forward targets, or None if the session is unusable."""

    spx_path = corpus_root / "raw/index/spx_1m" / f"{session}.official_spx.parquet"
    spx = _official_spx(spx_path, session)
    boundaries = _boundaries_ns(session)
    try:
        features, window, names = _session_features(spx, session, boundaries)
    except ValueError as error:
        # The frozen kernel refuses a session with no fresh bar at the final
        # boundary.  Accounted for explicitly rather than silently skipped.
        return {"session": session, "dropped": str(error)}

    available, closes = _availability_series(spx)
    spot = _close_available_at(available, closes, boundaries)
    _assert_close_parity(window, names, spot)

    targets = {
        horizon: _close_available_at(available, closes, boundaries + horizon * MINUTE_NS) - spot
        for horizon in HORIZON_MINUTES
    }
    return {"session": session, "features": features, "targets": targets, "dropped": None}


def _member(feature: str, horizon: int, sign: str) -> str:
    return f"{feature}|{horizon}m|{sign}"


def _session_ics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-member, per-session Spearman IC over that session's valid boundaries."""

    ics: dict[str, dict[str, float]] = {
        _member(feature, horizon, sign): {}
        for feature in SIDE_FREE_FEATURES
        for horizon in HORIZON_MINUTES
        for sign in ("+", "-")
    }
    for record in records:
        session = str(record["session"])
        features = record["features"]
        for horizon in HORIZON_MINUTES:
            target = record["targets"][horizon]
            for column, feature in enumerate(SIDE_FREE_FEATURES):
                column_values = features[:, column]
                valid = np.isfinite(column_values) & np.isfinite(target)
                if int(valid.sum()) < MIN_BOUNDARIES_PER_SESSION:
                    continue
                x, y = column_values[valid], target[valid]
                if np.all(x == x[0]) or np.all(y == y[0]):
                    continue
                value = float(spearmanr(x, y).statistic)
                if not np.isfinite(value):
                    continue
                ics[_member(feature, horizon, "+")][session] = value
                ics[_member(feature, horizon, "-")][session] = -value
    return ics


def _shuffled_ics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """The null control: within-session permutation of the target."""

    shuffled_records = []
    for record in records:
        rng = np.random.default_rng(
            int(hashlib.sha256(str(record["session"]).encode()).hexdigest()[:16], 16)
        )
        targets = {}
        for horizon, target in record["targets"].items():
            permuted = target.copy()
            rng.shuffle(permuted)
            targets[horizon] = permuted
        shuffled_records.append(
            {"session": record["session"], "features": record["features"], "targets": targets}
        )
    return _session_ics(shuffled_records)


def _assert_sign_identity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """IC(-x, y) must equal -IC(x, y) exactly.  An implementation check, not a control."""

    record = records[0]
    horizon = HORIZON_MINUTES[0]
    target = record["targets"][horizon]
    checked = 0
    for column, _feature in enumerate(SIDE_FREE_FEATURES):
        values = record["features"][:, column]
        valid = np.isfinite(values) & np.isfinite(target)
        if int(valid.sum()) < MIN_BOUNDARIES_PER_SESSION:
            continue
        x, y = values[valid], target[valid]
        if np.all(x == x[0]) or np.all(y == y[0]):
            continue
        forward = float(spearmanr(x, y).statistic)
        reversed_ic = float(spearmanr(-x, y).statistic)
        if not np.isclose(reversed_ic, -forward, rtol=0.0, atol=1e-12):
            raise ScreenError(
                f"sign identity violated: IC(-x,y)={reversed_ic} vs -IC(x,y)={-forward}"
            )
        checked += 1
    if not checked:
        raise ScreenError("sign identity check evaluated no features")
    return {"session": str(record["session"]), "features_checked": checked, "passed": True}


def _fold_means(
    ics: Mapping[str, Mapping[str, float]], folds: Sequence[Mapping[str, Any]]
) -> dict[str, list[float | None]]:
    output: dict[str, list[float | None]] = {}
    for member, values in ics.items():
        means: list[float | None] = []
        for fold in folds:
            sample = [values[session] for session in fold["test"] if session in values]
            means.append(float(np.mean(sample)) if sample else None)
        output[member] = means
    return output


def _sign_stable_folds(mean_ic: float | None, fold_means: Sequence[float | None]) -> int:
    if mean_ic is None or not np.isfinite(mean_ic) or mean_ic == 0.0:
        return 0
    want = np.sign(mean_ic)
    return sum(
        1
        for value in fold_means
        if value is not None and np.isfinite(value) and np.sign(value) == want
    )


def _clean(value: Any) -> Any:
    """Deterministic, NaN-free JSON payload -- ``stable_hash`` sets allow_nan=False."""

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


def _cell(value: Any, spec: str) -> str:
    return "nan" if value is None or not np.isfinite(value) else format(value, spec)


def _render(rows: Sequence[Mapping[str, Any]], verdict: str, header: Mapping[str, Any]) -> str:
    lines = [
        "# SPX directional-skill screen -- results",
        "",
        f"verdict: {verdict}",
        f"sessions: {header['sessions_used']} of {header['sessions_declared']}"
        f"   session-IC observations: {header['ic_observations']}",
        f"family: {header['family_size']} members   permutations: {PERMUTATIONS}"
        f"   seed: {PERMUTATION_SEED}",
        "",
        f"{'member':52} {'n':>4} {'mean IC':>9} {'se':>8} {'maxT p':>8} {'folds':>6}",
        "-" * 92,
    ]
    for row in rows:
        lines.append(
            f"{row['member']:52} {row['n_sessions']:>4d} "
            f"{_cell(row['mean_ic'], '+.5f'):>9} {_cell(row['se'], '.5f'):>8} "
            f"{row['maxT_p_one_sided']:>8.4f} {row['sign_stable_folds']:>4d}/5"
        )
    best = rows[0]
    lines += [
        "",
        f"strongest member: {best['member']}   mean IC {_cell(best['mean_ic'], '+.5f')}"
        f"   maxT p {best['maxT_p_one_sided']:.4f}"
        f"   sign-stable {best['sign_stable_folds']}/5",
    ]
    return "\n".join(lines) + "\n"


def main() -> dict[str, Any]:
    # Refuse up front rather than after the run: a sealed receipt is immutable.
    if OUTPUT_ROOT.exists():
        raise ScreenError(f"output already exists, refusing to re-run: {OUTPUT_ROOT}")
    corpus_root = _corpus_root()
    sessions = development_sessions(corpus_root)
    if len(sessions) != DEVELOPMENT_SESSION_COUNT:
        raise ScreenError(f"expected {DEVELOPMENT_SESSION_COUNT} dev sessions, got {len(sessions)}")
    print(f"corpus: {corpus_root}\nsessions: {len(sessions)}")

    records: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for position, session in enumerate(sessions, start=1):
        record = _session_records(corpus_root, session)
        if record.get("dropped"):
            dropped.append({"session": session, "reason": record["dropped"]})
        else:
            records.append(record)
        if position % 25 == 0 or position == len(sessions):
            print(f"  {position}/{len(sessions)} sessions  ({len(dropped)} dropped)")
    if len(dropped) > MAX_DROPPED_SESSIONS:
        raise ScreenError(f"{len(dropped)} sessions dropped, cap is {MAX_DROPPED_SESSIONS}")
    if not records:
        raise ScreenError("no usable sessions")

    identity = _assert_sign_identity(records)
    ics = _session_ics(records)
    folds = entry_expanding_folds(sessions)
    fold_means = _fold_means(ics, folds)

    print(f"family: {len(ics)} members; running {PERMUTATIONS} permutations")
    observed = session_blocked_max_t(ics, permutations=PERMUTATIONS, seed=PERMUTATION_SEED)
    control_ics = _shuffled_ics(records)
    control = session_blocked_max_t(control_ics, permutations=PERMUTATIONS, seed=PERMUTATION_SEED)

    rows: list[dict[str, Any]] = []
    for member, summary in observed.items():
        feature, horizon, sign = member.split("|")
        means = fold_means[member]
        rows.append(
            {
                "member": member,
                "feature": feature,
                "horizon_minutes": int(horizon.removesuffix("m")),
                "sign": sign,
                "n_sessions": int(summary["n_sessions"]),
                "mean_ic": summary["mean"],
                "se": summary["se"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "p_one_sided": summary["p_one_sided"],
                "maxT_p_one_sided": summary["maxT_p_one_sided"],
                "fold_mean_ic": means,
                "sign_stable_folds": _sign_stable_folds(summary["mean"], means),
                "control_mean_ic": control[member]["mean"],
                "control_maxT_p_one_sided": control[member]["maxT_p_one_sided"],
            }
        )
    rows.sort(key=lambda row: (row["maxT_p_one_sided"], -(row["mean_ic"] or 0.0)))

    candidates = [
        row["member"]
        for row in rows
        if row["maxT_p_one_sided"] < ALPHA
        and row["sign_stable_folds"] >= MIN_SIGN_STABLE_FOLDS
    ]
    control_breaches = [
        member
        for member, summary in control.items()
        if summary["maxT_p_one_sided"] < ALPHA
    ]
    if control_breaches:
        verdict = "INVALID"
    elif candidates:
        verdict = "DIRECTIONAL_SKILL_CANDIDATE"
    else:
        verdict = "NO_DIRECTIONAL_SKILL"

    header = {
        "sessions_declared": len(sessions),
        "sessions_used": len(records),
        "ic_observations": int(sum(row["n_sessions"] for row in rows) // 2),
        "family_size": len(ics),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": _sha256_path(PREREGISTRATION),
        "corpus_root": str(corpus_root),
        "features": list(SIDE_FREE_FEATURES),
        "horizon_minutes": list(HORIZON_MINUTES),
        "family_size": len(ics),
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "alpha": ALPHA,
        "min_boundaries_per_session": MIN_BOUNDARIES_PER_SESSION,
        "boundary_grid_et": [FIRST_BOUNDARY.isoformat(), LAST_BOUNDARY.isoformat()],
        "sessions": header,
        "dropped_sessions": dropped,
        "sign_identity_check": identity,
        "candidates": candidates,
        "control_breaches": control_breaches,
        "results": rows,
        "source_hashes": {
            "screen": _sha256_path(Path(__file__)),
            "pathd_entry_features": _sha256_path(
                Path("v4/research/pathd_entry_features.py")
            ),
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
    (OUTPUT_ROOT / "receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    frame = pd.DataFrame(rows)
    for position in range(len(folds)):
        frame[f"fold{position}_mean_ic"] = [row["fold_mean_ic"][position] for row in rows]
    frame.drop(columns=["fold_mean_ic"]).to_csv(OUTPUT_ROOT / "results.csv", index=False)
    (OUTPUT_ROOT / "results.md").write_text(_render(rows, verdict, header))

    print(f"\nverdict: {verdict}")
    print(f"candidates: {candidates or 'none'}")
    print(f"control breaches: {control_breaches or 'none'}")
    print(f"receipt: {OUTPUT_ROOT / 'receipt.json'}")
    return payload


if __name__ == "__main__":
    main()
