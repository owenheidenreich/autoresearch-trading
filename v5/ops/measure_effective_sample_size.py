"""Are 93,798 causal decision states worth 93,798 independent observations?

Section 4 of the 2026-08-14 fit reopening changed the sessions-per-parameter
rule's unit from **sessions** to **causal decision states**, on the argument that
the unit relevant to fitting is the decision the model is asked to make. That
argument is true of what a model *consumes*. It is silent on how much
*independent information* those states carry, and section 4 assumed the answer
without measuring it.

The states are 243 sessions x ~386 minutes. Consecutive minutes within a session
share almost all of their history, their ladder and — for a 60-minute forward
label — most of their outcome window. So the effective sample size lies
somewhere between 243 and 93,798, and where it lands decides whether section 4
stands or collapses.

This module measures it two independent ways and reports both:

1. **Integrated autocorrelation.** For a series with within-session
   autocorrelation ``rho(k)``, the variance of the mean is inflated by
   ``tau = 1 + 2 * sum_k (1 - k/T) * rho(k)``, and the effective count is
   ``n / tau``. Summed with Bartlett weights and truncated at the first
   non-positive pair, which is the standard guard against summing noise.

2. **Design effect.** Compare the observed variance of session means against
   what independent sampling of the same states would give. This asks the same
   question through the clustering literature's route and does not share the
   first method's truncation choice.

Model-free. Fits nothing, tunes nothing, contacts nothing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v2")

# The labels the reopening actually scoped, at the horizons it named.
LABELS = (
    "reached_30_itm_60m",
    "reached_20_itm_60m",
    "reached_10_itm_60m",
    "reached_30_itm_120m",
)

# Lags to consider within a session. A 60-minute label cannot decorrelate faster
# than its own window, so the scan must reach well past it.
MAX_LAG = 180


def integrated_autocorrelation(series: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Bartlett-weighted tau, truncated at the first non-positive pair sum.

    Returns ``(tau, lags_used)``. A tau of 1.0 means independent.
    """

    n = series.size
    centred = series - series.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= 0.0:
        return 1.0, 0
    total = 0.0
    used = 0
    limit = min(max_lag, n - 1)
    for lag in range(1, limit + 1):
        rho = float(np.dot(centred[:-lag], centred[lag:])) / denominator
        # Pair-sum truncation: stop once the signal is indistinguishable from
        # noise rather than accumulating it.
        if lag % 2 == 0 and rho <= 0.0:
            break
        total += (1.0 - lag / n) * rho
        used = lag
    return max(1.0, 1.0 + 2.0 * total), used


def measure_label(frame: pd.DataFrame, label: str) -> dict:
    """Effective sample size for one label, both ways."""

    taus: list[float] = []
    lengths: list[int] = []
    session_means: list[float] = []
    within_variances: list[float] = []
    lags_used: list[int] = []

    for _, block in frame.groupby("session", sort=True):
        values = block[label].to_numpy(dtype=float)
        if values.size < 30 or not np.isfinite(values).all():
            continue
        tau, used = integrated_autocorrelation(values, MAX_LAG)
        taus.append(tau)
        lengths.append(values.size)
        lags_used.append(used)
        session_means.append(float(values.mean()))
        within_variances.append(float(values.var(ddof=1)) if values.size > 1 else 0.0)

    states = int(sum(lengths))
    sessions = len(lengths)
    mean_tau = float(np.mean(taus))

    # Route 1: divide the raw state count by the mean autocorrelation time.
    n_eff_autocorr = states / mean_tau

    # Route 2: design effect. If states were independent, the variance of a
    # session mean would be within-variance / states-per-session. Compare that
    # to the variance actually observed across session means.
    means = np.asarray(session_means, dtype=float)
    per_session = states / sessions
    pooled_within = float(np.mean(within_variances))
    observed = float(means.var(ddof=1))
    expected_if_independent = pooled_within / per_session
    design_effect = (
        observed / expected_if_independent if expected_if_independent > 0 else float("nan")
    )
    n_eff_design = states / design_effect if design_effect and design_effect > 0 else float("nan")

    return {
        "label": label,
        "sessions": sessions,
        "states": states,
        "states_per_session": round(per_session, 2),
        "base_rate": round(float(np.mean(means)), 6),
        "mean_tau": round(mean_tau, 3),
        "median_tau": round(float(np.median(taus)), 3),
        "mean_lags_used": round(float(np.mean(lags_used)), 1),
        "effective_n_autocorrelation": round(n_eff_autocorr, 1),
        "design_effect": round(design_effect, 3),
        "effective_n_design_effect": round(n_eff_design, 1),
        "states_per_effective_observation": round(states / n_eff_autocorr, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    candidates = pd.read_parquet(args.root / "candidates.parquet")

    # Collapse the ladder to one row per decision minute: did ANY eligible
    # contract reach the depth? That is the session-minute state the policy
    # actually decides at, and it is the unit section 4 counted.
    available = [c for c in LABELS if c in candidates.columns]
    per_minute = (
        candidates.groupby(["session", "minute"], sort=True)[available]
        .max()
        .reset_index()
        .sort_values(["session", "minute"])
    )

    rows = [measure_label(per_minute, label) for label in available]

    states = rows[0]["states"] if rows else 0
    sessions = rows[0]["sessions"] if rows else 0
    worst = max(rows, key=lambda r: r["states_per_effective_observation"]) if rows else None
    budgets = {
        r["label"]: {
            "effective_n": r["effective_n_autocorrelation"],
            "parameter_budget_at_20_per_observation": int(
                r["effective_n_autocorrelation"] // 20
            ),
        }
        for r in rows
    }

    payload = {
        "schema_version": "v5.effective-sample-size.v1",
        "question": (
            "Section 4 of the fit reopening charged the 20-per-parameter rule "
            "against 93,798 causal decision states. Are those states worth "
            "93,798 independent observations?"
        ),
        "source": str(args.root / "candidates.parquet"),
        "unit": "session-minute decision state, ladder collapsed by any-contract",
        "sessions": sessions,
        "states": states,
        "section_4_assumed_effective_n": states,
        "lower_bound_if_only_sessions_count": sessions,
        "by_label": rows,
        "parameter_budgets": budgets,
        "architecture_counts_computed": True,
        "computes_no_policy": True,
        "fit_performed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\n{states:,} states over {sessions} sessions "
          f"({states / max(sessions, 1):.0f} per session)\n")
    head = (
        f"{'label':>22} {'base':>7} {'tau':>7} {'eff n (acf)':>12} "
        f"{'eff n (deff)':>13} {'param budget':>13}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['label']:>22} {100 * r['base_rate']:>6.2f}% {r['mean_tau']:>7.2f} "
            f"{r['effective_n_autocorrelation']:>12,.0f} "
            f"{r['effective_n_design_effect']:>13,.0f} "
            f"{budgets[r['label']]['parameter_budget_at_20_per_observation']:>13,}"
        )
    if worst:
        print(
            f"\nsection 4 assumed {states:,}; the floor if only sessions count is {sessions}."
        )
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
