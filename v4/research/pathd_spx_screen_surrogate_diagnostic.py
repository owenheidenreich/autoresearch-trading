"""POST-HOC diagnostic: how much of the screen's IC is a bounded-path artifact?

Not part of the frozen pre-registration.  Run after
``pathd_spx_directional_skill_screen.py`` returned ``DIRECTIONAL_SKILL_CANDIDATE``
with a mean Spearman IC of 0.52 -- a magnitude that is not a plausible edge on the
most heavily arbitraged index in the world.

The suspicion: every top-ranked feature is a function of the *price level*
``P(t)`` (position within the session range, gap to session VWAP), and the target
``y = P(t+h) - P(t)`` contains ``-P(t)``.  For **any** bounded path -- including one
with no predictability whatsoever -- that shared term forces a negative rank
correlation which grows with the horizon.

The screen's pre-registered session-shuffle control cannot detect this: shuffling
``y`` within a session destroys the pairing, but the artifact *lives in* the
pairing structure of the realized path.  The correct null is a **matched
surrogate** -- a path with this session's own realized volatility and drift, and
no predictability.

Surrogate construction: circular block bootstrap of the session's own one-minute
log returns (block length 30 minutes, preserving short-range volatility clustering
while destroying any predictive relationship), replayed from the session's true
opening level.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from v4.research.pathd_phase1_entry import _official_spx, development_sessions
from v4.research.pathd_spx_directional_skill_screen import (
    HORIZON_MINUTES,
    MIN_BOUNDARIES_PER_SESSION,
    SIDE_FREE_FEATURES,
    _availability_series,
    _boundaries_ns,
    _corpus_root,
    _session_records,
)

BLOCK_MINUTES = 30
SURROGATES_PER_SESSION = 20
SEED = 4111
OUTPUT = Path(
    "v4/audit/autoresearch/pathd_spx_directional_skill_screen_2026_08_04/"
    "surrogate_diagnostic.json"
)
# The three features that dominate the screen, plus a momentum reference.
FOCUS = (
    "omar_clipped_neg3_pos3",
    "spx_vwap_gap_points",
    "spx_vwap_gap_over_session_range",
    "momentum_15m_bps",
    "momentum_5m_bps",
    "session_range_bps",
)


def _features_from_path(prices: np.ndarray) -> dict[str, np.ndarray]:
    """The screen's nine context features, recomputed from a bare price path.

    Mirrors ``feature_matrix`` for the level-derived features only; the surrogate
    has no option ladder and no vendor bars, so the frozen kernel cannot be used.
    Volume is unavailable for a surrogate, so VWAP degrades to the prefix mean --
    the same fallback the kernel uses when total volume is zero.
    """

    count = len(prices)
    index = np.arange(1, count + 1, dtype=np.float64)
    vwap = np.cumsum(prices) / index
    run_max = np.maximum.accumulate(prices)
    run_min = np.minimum.accumulate(prices)
    session_range = run_max - run_min
    gap = prices - vwap
    spx_denom = np.maximum(np.abs(prices), 1.0)
    range_denom = np.maximum(np.abs(session_range), 1.0)
    omar = np.divide(
        prices - prices[0], session_range, out=np.zeros(count), where=session_range != 0.0
    )

    def lagged(minutes: int) -> np.ndarray:
        out = np.full(count, np.nan)
        out[minutes:] = prices[minutes:] - prices[:-minutes]
        return out

    momentum_5m, momentum_15m = lagged(5), lagged(15)
    return {
        "omar_clipped_neg3_pos3": np.clip(omar, -3.0, 3.0),
        "spx_vwap_gap_points": gap,
        "spx_vwap_gap_over_session_range": gap / range_denom,
        "session_range_bps": session_range / spx_denom * 10_000.0,
        "momentum_5m_bps": momentum_5m / spx_denom * 10_000.0,
        "momentum_15m_bps": momentum_15m / spx_denom * 10_000.0,
    }


def _ic(feature: np.ndarray, target: np.ndarray) -> float | None:
    valid = np.isfinite(feature) & np.isfinite(target)
    if int(valid.sum()) < MIN_BOUNDARIES_PER_SESSION:
        return None
    x, y = feature[valid], target[valid]
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def main() -> dict:
    corpus_root = _corpus_root()
    sessions = development_sessions(corpus_root)
    rng = np.random.default_rng(SEED)

    measured: dict[str, dict[int, list[float]]] = {
        name: {h: [] for h in HORIZON_MINUTES} for name in FOCUS
    }
    surrogate: dict[str, dict[int, list[float]]] = {
        name: {h: [] for h in HORIZON_MINUTES} for name in FOCUS
    }
    used = 0
    for position, session in enumerate(sessions, start=1):
        record = _session_records(corpus_root, session)
        if record.get("dropped"):
            continue
        used += 1
        columns = {name: index for index, name in enumerate(SIDE_FREE_FEATURES)}
        for name in FOCUS:
            for horizon in HORIZON_MINUTES:
                value = _ic(
                    record["features"][:, columns[name]], record["targets"][horizon]
                )
                if value is not None:
                    measured[name][horizon].append(value)

        # Matched surrogates from this session's own realized one-minute returns.
        spx = _official_spx(
            corpus_root / "raw/index/spx_1m" / f"{session}.official_spx.parquet", session
        )
        _, closes = _availability_series(spx)
        closes = closes[np.isfinite(closes) & (closes > 0.0)]
        log_returns = np.diff(np.log(closes))
        if len(log_returns) < BLOCK_MINUTES * 2:
            continue
        need = len(_boundaries_ns(session)) + max(HORIZON_MINUTES)
        blocks = int(np.ceil(need / BLOCK_MINUTES))
        for _ in range(SURROGATES_PER_SESSION):
            starts = rng.integers(0, len(log_returns), size=blocks)
            drawn = np.concatenate(
                [np.take(log_returns, np.arange(s, s + BLOCK_MINUTES), mode="wrap") for s in starts]
            )[:need]
            path = float(closes[0]) * np.exp(np.cumsum(drawn))
            grid = len(_boundaries_ns(session))
            features = _features_from_path(path[:grid])
            for horizon in HORIZON_MINUTES:
                target = path[horizon : horizon + grid] - path[:grid]
                for name in FOCUS:
                    value = _ic(features[name], target)
                    if value is not None:
                        surrogate[name][horizon].append(value)
        if position % 50 == 0:
            print(f"  {position}/{len(sessions)} sessions")

    rows = []
    for name in FOCUS:
        for horizon in HORIZON_MINUTES:
            real = float(np.mean(measured[name][horizon]))
            null = np.asarray(surrogate[name][horizon], dtype=float)
            null_mean = float(np.mean(null))
            null_se = float(np.std(null, ddof=1) / np.sqrt(len(null)))
            rows.append(
                {
                    "feature": name,
                    "horizon_minutes": horizon,
                    "measured_mean_ic": real,
                    "surrogate_mean_ic": null_mean,
                    "surrogate_se": null_se,
                    "excess_ic": real - null_mean,
                    "explained_fraction": (
                        None if real == 0.0 else float(null_mean / real)
                    ),
                    "n_sessions": len(measured[name][horizon]),
                    "n_surrogates": int(len(null)),
                }
            )

    payload = {
        "schema_version": "pathd.spx-screen-surrogate-diagnostic.v1",
        "note": "POST-HOC. Not part of the frozen pre-registration.",
        "block_minutes": BLOCK_MINUTES,
        "surrogates_per_session": SURROGATES_PER_SESSION,
        "seed": SEED,
        "sessions_used": used,
        "rows": rows,
        "protected_holdout_opened": False,
        "model_trained": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    frame = pd.DataFrame(rows).sort_values(
        ["feature", "horizon_minutes"]
    )
    print(
        f"\n{'feature':34} {'h':>4} {'measured':>10} {'surrogate':>10} "
        f"{'excess':>9} {'explained':>10}"
    )
    print("-" * 82)
    for row in frame.to_dict("records"):
        share = row["explained_fraction"]
        print(
            f"{row['feature']:34} {row['horizon_minutes']:>3}m "
            f"{row['measured_mean_ic']:>+10.4f} {row['surrogate_mean_ic']:>+10.4f} "
            f"{row['excess_ic']:>+9.4f} "
            f"{'n/a' if share is None else f'{share:>9.1%}'}"
        )
    print(f"\nwritten: {OUTPUT}")
    return payload


if __name__ == "__main__":
    main()
