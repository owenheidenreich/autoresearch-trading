"""What is the omar effect worth to a trader who only knows the past?

CORRECTS ``pathd_omar_economic_sizing.py``, which is not causal.

That script bucketed boundaries into **within-session deciles** of omar. The
decile cut is computed from the whole session, so it depends on boundaries that
have not happened yet: across 40 sessions the 10th-percentile cut ranges from
-0.941 to +0.852. An omar of -0.20 is "bottom decile" in one session and
mid-pack in another, and which one is not knowable until the session is over.
The same objection applies to the within-session Spearman IC that the re-screen
reports. Both are legitimate **association** statistics; neither is tradable.

This version conditions on **fixed absolute omar thresholds**, declared here and
not fitted. ``omar = (close - session open) / session range`` is bounded in
[-1, +1] by construction -- the numerator cannot exceed the denominator -- so a
fixed grid is well defined and is fully available at decision time.

The bounded-path artifact is removed exactly as before, by differencing against
drift-preserving wild-bootstrap surrogates.

RESULT: the effect size is NOT IDENTIFIED.  Three defensible estimators of the
same quantity -- within-session deciles (look-ahead), per-session bin-matched,
and pooled bin-matched -- give +0.82, +4.54 and +3.60 SPX points per trade, and
the session-level version of the last gives +9.35.  A quantity that moves by 10x
under a change of aggregation is not a number to spend money against.

Two causes, both diagnosed here and both recorded in the output:

1.  Pooling across sessions with fixed bins mixes within-session and
    between-session variation, and the between-session component dominates
    (Simpson's paradox): trending sessions spend most of their time at one
    extreme of omar *and* have a forward drift.

2.  The wild bootstrap imposes the session's drift at EVERY minute, so a
    surrogate path climbs steadily and "high omar" implies "still drifting up".
    That manufactures a positive omar->forward relation in the surrogate --
    visible below as a monotone surrogate column running -5.30 to +4.69 -- which
    inflates the corrected fade signal.  The real path's drift arrives in bursts,
    so it carries no such implication.  Drift handling is exactly what the owner
    flagged as load-bearing.

POST-HOC and DESCRIPTIVE. No hypothesis test, no gate, no p-values: this repairs
a causality defect in a descriptive number, it does not introduce a new claim.
The re-screen's pre-registered IC result is an ASSOCIATION finding and is
unaffected; what is unidentified is the conversion of that association to money.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v4.research.pathd_phase1_entry import (
    _official_spx,
    development_sessions,
    entry_expanding_folds,
)
from v4.research.pathd_spx_directional_skill_screen import (
    _availability_series,
    _boundaries_ns,
    _corpus_root,
)
from v4.research.pathd_spx_excess_skill_rescreen import (
    FEATURES,
    MINUTE_NS,
    SURROGATE_SEED,
    _features_from_path,
    _selection_map,
    _wild_bootstrap,
)

HORIZON = 60
SURROGATES = 100
# Fixed and declared, not fitted.  omar is bounded in [-1, +1] by construction.
BIN_EDGES = (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# The causal trading rule the sizing describes: fade the session extremes.
LONG_BELOW = -0.6
SHORT_ABOVE = 0.6
ES_DOLLARS_PER_POINT = 50.0
ES_ASSUMED_ROUND_TRIP_DOLLARS = 17.0  # tick-structure assumption; NOT measured
OUTPUT = Path(
    "v4/audit/autoresearch/pathd_spx_excess_skill_rescreen_2026_08_04/"
    "omar_causal_sizing.json"
)


def _bin_means(omar: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean forward move and count per FIXED omar bin."""

    valid = np.isfinite(omar) & np.isfinite(target)
    means = np.full(len(BIN_EDGES) - 1, np.nan)
    counts = np.zeros(len(BIN_EDGES) - 1, dtype=int)
    if not valid.any():
        return means, counts
    x, y = omar[valid], target[valid]
    index = np.clip(np.digitize(x, np.asarray(BIN_EDGES[1:-1]), right=False), 0, len(means) - 1)
    for position in range(len(means)):
        selected = y[index == position]
        counts[position] = len(selected)
        if len(selected):
            means[position] = float(np.mean(selected))
    return means, counts


def _rule_from_bins(
    corrected: np.ndarray, counts: np.ndarray
) -> tuple[float, int]:
    """Fade-rule value, corrected BIN BY BIN.

    Differencing the rule value on the real path against the rule value on a
    surrogate is NOT a bias correction: the surrogate has a different omar
    distribution, so the two sides select different trade populations (it
    inflates this number ~16x).  The correction must be applied at matched omar,
    then aggregated with the REAL trade counts.
    """

    signs = np.where(
        np.asarray(BIN_EDGES[1:]) <= LONG_BELOW + 1e-12,
        1.0,
        np.where(np.asarray(BIN_EDGES[:-1]) >= SHORT_ABOVE - 1e-12, -1.0, 0.0),
    )
    usable = (signs != 0.0) & np.isfinite(corrected) & (counts > 0)
    weight = counts[usable].astype(float)
    if not weight.sum():
        return float("nan"), 0
    value = float(np.sum(weight * signs[usable] * corrected[usable]) / weight.sum())
    return value, int(weight.sum())


def _accumulate(store: list[np.ndarray], omar: np.ndarray, target: np.ndarray) -> None:
    """Pooled sum-of-y and count per bin, across every session and surrogate."""

    valid = np.isfinite(omar) & np.isfinite(target)
    if not valid.any():
        return
    index = np.clip(
        np.digitize(omar[valid], np.asarray(BIN_EDGES[1:-1]), right=False),
        0,
        len(BIN_EDGES) - 2,
    )
    np.add.at(store[0], index, target[valid])
    np.add.at(store[1], index, 1.0)


def _rule_signs() -> np.ndarray:
    return np.where(
        np.asarray(BIN_EDGES[1:]) <= LONG_BELOW + 1e-12,
        1.0,
        np.where(np.asarray(BIN_EDGES[:-1]) >= SHORT_ABOVE - 1e-12, -1.0, 0.0),
    )


def main() -> dict:
    corpus_root = _corpus_root()
    sessions = development_sessions(corpus_root)
    column = FEATURES.index("omar_clipped_neg3_pos3")
    bins = len(BIN_EDGES) - 1
    pooled_real = [np.zeros(bins), np.zeros(bins)]
    pooled_surrogate = [np.zeros(bins), np.zeros(bins)]

    real_bins, surrogate_bins, counts_total = [], [], np.zeros(len(BIN_EDGES) - 1, dtype=int)
    rule_by_session: dict[str, float] = {}
    trades_total = 0
    used = []

    for session in sessions:
        spx = _official_spx(
            corpus_root / "raw/index/spx_1m" / f"{session}.official_spx.parquet", session
        )
        boundaries = _boundaries_ns(session)
        available, closes = _availability_series(spx)
        event_ns = (
            pd.to_datetime(spx["event_time"], utc=True)
            .astype("datetime64[ns, UTC]")
            .astype("int64")
            .to_numpy()
        )
        volumes = pd.to_numeric(spx["volume"], errors="raise").to_numpy(dtype=np.float64)[
            np.argsort(event_ns + MINUTE_NS, kind="mergesort")
        ]
        selection = _selection_map(available, boundaries)
        end = _selection_map(available, boundaries + HORIZON * MINUTE_NS)
        both = (selection >= 0) & (end >= 0)
        if int(both.sum()) < 30:
            continue

        def evaluate(path: np.ndarray):
            omar = _features_from_path(path, volumes, selection)[:, column]
            target = np.where(
                both,
                path[np.where(end >= 0, end, 0)] - path[np.where(selection >= 0, selection, 0)],
                np.nan,
            )
            return omar, target

        omar, target = evaluate(closes)
        means, counts = _bin_means(omar, target)
        _accumulate(pooled_real, omar, target)

        rng = np.random.default_rng(
            int(hashlib.sha256(f"{SURROGATE_SEED}|causal|{session}".encode()).hexdigest()[:16], 16)
        )
        draws = []
        for _ in range(SURROGATES):
            s_omar, s_target = evaluate(_wild_bootstrap(closes, rng))
            draws.append(_bin_means(s_omar, s_target)[0])
            _accumulate(pooled_surrogate, s_omar, s_target)
        surrogate_mean = np.nanmean(np.stack(draws), axis=0)

        used.append(session)
        real_bins.append(means)
        surrogate_bins.append(surrogate_mean)
        counts_total += counts
        value, trades = _rule_from_bins(means - surrogate_mean, counts)
        trades_total += trades
        if np.isfinite(value):
            rule_by_session[session] = value

    real = np.nanmean(np.stack(real_bins), axis=0)
    surrogate = np.nanmean(np.stack(surrogate_bins), axis=0)
    corrected = real - surrogate


    per_session = np.asarray([rule_by_session[s] for s in sorted(rule_by_session)], dtype=float)
    mean_points = float(np.mean(per_session))
    se_points = float(np.std(per_session, ddof=1) / np.sqrt(len(per_session)))
    folds = entry_expanding_folds(sessions)
    fold_means = [
        float(np.mean([rule_by_session[s] for s in fold["test"] if s in rule_by_session]))
        for fold in folds
    ]
    positive_folds = sum(1 for value in fold_means if value > 0.0)

    # Pooled estimator: count-weighted, no per-session equal weighting.
    pooled_real_mean = pooled_real[0] / np.maximum(pooled_real[1], 1.0)
    pooled_surrogate_mean = pooled_surrogate[0] / np.maximum(pooled_surrogate[1], 1.0)
    pooled_corrected = pooled_real_mean - pooled_surrogate_mean
    signs = _rule_signs()
    weight = pooled_real[1] * (signs != 0.0)
    pooled_rule = float(np.sum(weight * signs * pooled_corrected) / weight.sum())

    payload = {
        "schema_version": "pathd.omar-causal-sizing.v1",
        "note": (
            "POST-HOC and DESCRIPTIVE. Corrects the look-ahead in "
            "pathd_omar_economic_sizing.py, which used within-session deciles."
        ),
        "horizon_minutes": HORIZON,
        "bin_edges": list(BIN_EDGES),
        "long_below": LONG_BELOW,
        "short_above": SHORT_ABOVE,
        "sessions": len(used),
        "surrogates_per_session": SURROGATES,
        "bin_counts": [int(v) for v in counts_total],
        "bin_mean_forward_points_real": [float(v) for v in real],
        "bin_mean_forward_points_surrogate": [float(v) for v in surrogate],
        "bin_mean_forward_points_bias_corrected": [float(v) for v in corrected],
        "estimator_disagreement": {
            "within_session_deciles_LOOKAHEAD": 0.817,
            "per_session_bin_matched": mean_points,
            "pooled_bin_matched": pooled_rule,
            "verdict": "EFFECT_SIZE_NOT_IDENTIFIED",
            "why": (
                "pooling with fixed bins mixes within- and between-session variation "
                "(trending sessions sit at one omar extreme and carry a forward drift); "
                "and the wild bootstrap imposes the session drift at every minute, so "
                "high omar implies still-drifting-up in the surrogate but not in the "
                "real path, manufacturing a positive surrogate slope"
            ),
        },
        "pooled_bin_mean_forward_points_real": [float(v) for v in pooled_real_mean],
        "pooled_bin_mean_forward_points_surrogate": [
            float(v) for v in pooled_surrogate_mean
        ],
        "pooled_bin_mean_forward_points_bias_corrected": [
            float(v) for v in pooled_corrected
        ],
        "pooled_rule_points_per_trade": pooled_rule,
        "rule_mean_points_per_trade_bias_corrected": mean_points,
        "rule_se_points": se_points,
        "rule_t": mean_points / se_points if se_points > 0 else None,
        "rule_fold_means": fold_means,
        "rule_positive_folds": positive_folds,
        "rule_trades_total": trades_total,
        "rule_trades_per_session": trades_total / max(len(used), 1),
        "es_dollars_per_trade": mean_points * ES_DOLLARS_PER_POINT,
        "es_assumed_round_trip_dollars": ES_ASSUMED_ROUND_TRIP_DOLLARS,
        "es_assumed_friction_is_measured": False,
        "protected_holdout_opened": False,
        "model_trained": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"CAUSAL omar bins -> mean forward {HORIZON}m SPX move, {len(used)} sessions\n")
    print(f"{'omar bin':>16}{'n':>9}{'real':>10}{'surrogate':>11}{'corrected':>11}")
    print("-" * 57)
    for position in range(len(real)):
        label = f"[{BIN_EDGES[position]:+.1f},{BIN_EDGES[position + 1]:+.1f})"
        print(
            f"{label:>16}{counts_total[position]:>9d}{real[position]:>+10.3f}"
            f"{surrogate[position]:>+11.3f}{corrected[position]:>+11.3f}"
        )
    print(
        f"\ncausal fade rule (long <= {LONG_BELOW}, short >= {SHORT_ABOVE}), bias-corrected:"
    )
    print(f"  {mean_points:+.4f} SPX pts/trade   se {se_points:.4f}   t {mean_points / se_points:+.2f}")
    print(f"  positive folds {positive_folds}/5   {fold_means}")
    print(f"  {trades_total} candidate trades over {len(used)} sessions"
          f" ({trades_total / len(used):.1f}/session, before serial occupancy)")
    print("\nPOOLED (count-weighted, no per-session equal weighting):")
    print(f"{'omar bin':>16}{'n':>9}{'real':>10}{'surrogate':>11}{'corrected':>11}")
    print("-" * 57)
    for position in range(bins):
        label = f"[{BIN_EDGES[position]:+.1f},{BIN_EDGES[position + 1]:+.1f})"
        print(
            f"{label:>16}{pooled_real[1][position]:>9.0f}"
            f"{pooled_real_mean[position]:>+10.3f}{pooled_surrogate_mean[position]:>+11.3f}"
            f"{pooled_corrected[position]:>+11.3f}"
        )
    print("\n" + "=" * 68)
    print("THE SAME QUANTITY, THREE DEFENSIBLE ESTIMATORS:")
    print(f"  within-session deciles (LOOK-AHEAD)      +0.817 pts  = ${0.817 * 50:>7.2f}")
    print(f"  per-session bin-matched                  {mean_points:+.3f} pts  "
          f"= ${mean_points * 50:>7.2f}")
    print(f"  pooled bin-matched                       {pooled_rule:+.3f} pts  "
          f"= ${pooled_rule * 50:>7.2f}")
    print(f"\n  vs assumed ES round trip ${ES_ASSUMED_ROUND_TRIP_DOLLARS:.2f} (NOT measured)")
    print("  => EFFECT SIZE NOT IDENTIFIED. A number that moves 10x under a change")
    print("     of aggregation is not a number to spend money against.")
    print("=" * 68)
    print(f"\nwritten: {OUTPUT}")
    return payload


if __name__ == "__main__":
    main()
