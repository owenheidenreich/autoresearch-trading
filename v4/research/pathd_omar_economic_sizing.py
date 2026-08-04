"""How big is the surviving omar effect, in SPX points?

POST-HOC and DESCRIPTIVE.  Not part of the frozen pre-registration, not a
hypothesis test, and it reports no p-values.

The Option 0 re-screen left exactly one survivor -- ``omar_clipped_neg3_pos3``,
the position of price within the session's realized range -- with a
bias-corrected rank IC of ~0.078 at 60 minutes.  An information coefficient is
not money, and the owner's A/B/C decision turns on whether the effect is large
enough to clear friction on any instrument.  This converts it to SPX points.

The raw decile spread is NOT the answer: most of it is the same bounded-path
artifact the re-screen removed.  So the spread is measured on the real path and
on matched wild-bootstrap surrogates, and the difference is reported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v4.research.pathd_phase1_entry import _official_spx, development_sessions
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
DECILES = 10
SURROGATES = 100
MIN_BOUNDARIES = 30
OUTPUT = Path(
    "v4/audit/autoresearch/pathd_spx_excess_skill_rescreen_2026_08_04/"
    "omar_economic_sizing.json"
)
# ES contract arithmetic, for converting points to dollars.
ES_DOLLARS_PER_POINT = 50.0
ES_ASSUMED_ROUND_TRIP_DOLLARS = 17.0  # tick-structure assumption; NOT measured


def _decile_means(feature: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean forward move per within-session decile of the feature."""

    valid = np.isfinite(feature) & np.isfinite(target)
    out = np.full(DECILES, np.nan)
    if int(valid.sum()) < MIN_BOUNDARIES:
        return out
    x, y = feature[valid], target[valid]
    edges = np.quantile(x, np.linspace(0.0, 1.0, DECILES + 1)[1:-1])
    bucket = np.searchsorted(edges, x, side="right")
    for index in range(DECILES):
        selected = y[bucket == index]
        if len(selected):
            out[index] = float(np.mean(selected))
    return out


def main() -> dict:
    corpus_root = _corpus_root()
    sessions = development_sessions(corpus_root)
    column = FEATURES.index("omar_clipped_neg3_pos3")
    real_rows, surrogate_rows = [], []

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
        if int(both.sum()) < MIN_BOUNDARIES:
            continue

        def evaluate(path: np.ndarray) -> np.ndarray:
            features = _features_from_path(path, volumes, selection)
            target = np.where(
                both,
                path[np.where(end >= 0, end, 0)] - path[np.where(selection >= 0, selection, 0)],
                np.nan,
            )
            return _decile_means(features[:, column], target)

        real_rows.append(evaluate(closes))
        rng = np.random.default_rng(
            int(hashlib.sha256(f"{SURROGATE_SEED}|sizing|{session}".encode()).hexdigest()[:16], 16)
        )
        surrogate_rows.append(
            np.nanmean(
                np.stack([evaluate(_wild_bootstrap(closes, rng)) for _ in range(SURROGATES)]),
                axis=0,
            )
        )

    real = np.nanmean(np.stack(real_rows), axis=0)
    surrogate = np.nanmean(np.stack(surrogate_rows), axis=0)
    corrected = real - surrogate
    spread_raw = float(real[0] - real[-1])
    spread_corrected = float(corrected[0] - corrected[-1])
    per_trade_points = spread_corrected / 2.0

    payload = {
        "schema_version": "pathd.omar-economic-sizing.v1",
        "note": "POST-HOC and DESCRIPTIVE. No hypothesis test, no p-values.",
        "horizon_minutes": HORIZON,
        "sessions": len(real_rows),
        "surrogates_per_session": SURROGATES,
        "decile_mean_forward_move_points_real": [float(v) for v in real],
        "decile_mean_forward_move_points_surrogate": [float(v) for v in surrogate],
        "decile_mean_forward_move_points_bias_corrected": [float(v) for v in corrected],
        "bottom_minus_top_decile_points_raw": spread_raw,
        "bottom_minus_top_decile_points_bias_corrected": spread_corrected,
        "per_trade_expected_points": per_trade_points,
        "es_per_trade_expected_dollars": per_trade_points * ES_DOLLARS_PER_POINT,
        "es_assumed_round_trip_dollars": ES_ASSUMED_ROUND_TRIP_DOLLARS,
        "es_assumed_friction_is_measured": False,
        "protected_holdout_opened": False,
        "model_trained": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"omar decile -> mean forward {HORIZON}m SPX move, {len(real_rows)} sessions\n")
    print(f"{'decile':>7}{'real':>10}{'surrogate':>11}{'corrected':>11}")
    print("-" * 39)
    for index in range(DECILES):
        print(
            f"{index + 1:>7}{real[index]:>+10.3f}{surrogate[index]:>+11.3f}"
            f"{corrected[index]:>+11.3f}"
        )
    print(f"\nbottom-minus-top decile   raw {spread_raw:+.3f} pts"
          f"   bias-corrected {spread_corrected:+.3f} pts")
    print(f"per-trade expected move   {per_trade_points:+.3f} SPX pts"
          f"  = ${per_trade_points * ES_DOLLARS_PER_POINT:+.2f} on one ES contract")
    print(f"vs ES assumed round-trip friction ${ES_ASSUMED_ROUND_TRIP_DOLLARS:.2f} (NOT measured)")
    print(f"\nwritten: {OUTPUT}")
    return payload


if __name__ == "__main__":
    main()
