"""Executable acceptance gate and rejection tests for Path-D model iterations.

This module turns hard-won April/May findings into tests a loop can run every
iteration, so a closed failure mode cannot be re-run a third time.

Each rejection test exists because it would have caught a real, expensive failure:

- ``exit_first_step_rate``   Protocols 029/030/031 -- loss-only and giveback early
  exits, closed by April with an explicit "exit-loop stop". Path-D re-ran them:
  95.1% of learned exits fire at index 0.
- ``tail_preservation``      "0DTE long options are convex... optimizing win rate
  can destroy the actual payoff profile." The Path-D exit cut p99 from $3,422 to
  $86 while looking fine on mean PnL.
- ``decile_monotonicity``    April's "post-hoc V2 score coverage thresholds"
  (every CI crossed zero). Path-D's entry deciles: top -$15.37, middle -$10.79.
- ``label_balance``          The 98%-exit-label disaster: "the label is part of
  the model architecture." Path-D's exit target is 17.46% positive -- the mirror
  failure, nearly all loser-defense and no upside capture.

Read-only. No training, no broker, no paid data, no firewall access.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Thresholds are policy, fixed here so an iteration cannot quietly relax them.
MAX_FIRST_STEP_EXIT_RATE = 0.50      # >50% is the P029-031 signature
MIN_TAIL_RATIO = 0.80                # p99 and top-decile vs a hold baseline
MIN_LABEL_POSITIVE_RATE = 0.25       # below this the head learns "always defend"
MAX_LABEL_POSITIVE_RATE = 0.90       # above this the head learns "always act"


class GateError(RuntimeError):
    """A gate input was malformed. Fail closed rather than score a bad run."""


@dataclass(frozen=True)
class TestResult:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def exit_first_step_rate(exit_indices: Sequence[int]) -> TestResult:
    """FAIL when the policy is effectively 'exit immediately'."""

    idx = np.asarray(list(exit_indices), dtype=float)
    if idx.size == 0:
        raise GateError("exit_first_step_rate received no exits")
    rate = float((idx <= 0).mean())
    return TestResult(
        name="exit_first_step_rate",
        passed=bool(rate <= MAX_FIRST_STEP_EXIT_RATE),
        detail=(
            f"{100*rate:.1f}% of exits fire at the first step "
            f"(limit {100*MAX_FIRST_STEP_EXIT_RATE:.0f}%). "
            "Above the limit this is the Protocol 029/030/031 signature."
        ),
        metrics={"first_step_rate": rate, "limit": MAX_FIRST_STEP_EXIT_RATE},
    )


def tail_preservation(
    policy_values: Sequence[float], baseline_values: Sequence[float]
) -> TestResult:
    """FAIL when a policy improves the mean by amputating the convex tail."""

    policy = _finite(policy_values)
    baseline = _finite(baseline_values)
    if policy.size == 0 or baseline.size == 0:
        raise GateError("tail_preservation received an empty series")

    p99_policy = float(np.percentile(policy, 99))
    p99_base = float(np.percentile(baseline, 99))

    def top_decile_sum(values: np.ndarray) -> float:
        k = max(1, int(len(values) * 0.10))
        return float(np.sort(values)[-k:].sum())

    top_policy = top_decile_sum(policy)
    top_base = top_decile_sum(baseline)

    # A non-positive baseline tail means the comparison is undefined, not passed.
    if p99_base <= 0 or top_base <= 0:
        raise GateError(
            "tail_preservation baseline has no positive tail; choose a hold-to-horizon baseline"
        )

    p99_ratio = p99_policy / p99_base
    top_ratio = top_policy / top_base
    worst = min(p99_ratio, top_ratio)
    return TestResult(
        name="tail_preservation",
        passed=bool(worst >= MIN_TAIL_RATIO),
        detail=(
            f"p99 {p99_policy:,.0f} vs baseline {p99_base:,.0f} (ratio {p99_ratio:.2f}); "
            f"top-decile {top_policy:,.0f} vs {top_base:,.0f} (ratio {top_ratio:.2f}); "
            f"limit {MIN_TAIL_RATIO:.2f}. The convex tail is the reason to hold a long option."
        ),
        metrics={
            "p99_policy": p99_policy,
            "p99_baseline": p99_base,
            "p99_ratio": p99_ratio,
            "top_decile_policy": top_policy,
            "top_decile_baseline": top_base,
            "top_decile_ratio": top_ratio,
            "limit": MIN_TAIL_RATIO,
        },
    )


def decile_monotonicity(
    scores: Sequence[float], outcomes: Sequence[float], folds: Sequence[int] | None = None
) -> TestResult:
    """FAIL when the ranker's best decile is not better than its middle decile.

    April closed post-hoc score-coverage selection; a flat or non-monotonic decile
    curve means the score carries no usable ranking information.
    """

    frame = pd.DataFrame({"score": np.asarray(scores, float), "y": np.asarray(outcomes, float)})
    frame["fold"] = np.asarray(folds, int) if folds is not None else 0
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 100:
        raise GateError("decile_monotonicity needs at least 100 rows")

    def assign(series: pd.Series) -> pd.Series:
        if series.nunique() < 10:
            return pd.Series(np.nan, index=series.index)
        return pd.qcut(series, 10, labels=False, duplicates="drop").astype(float)

    # transform keeps a Series shape whether there is one fold or many; apply would
    # return a DataFrame in the single-group case.
    frame["decile"] = frame.groupby("fold")["score"].transform(assign)
    usable = frame.dropna(subset=["decile"])
    if usable.empty:
        raise GateError("decile_monotonicity: score is constant within every fold")

    means = usable.groupby("decile")["y"].mean()
    top = float(means.loc[means.index.max()])
    bottom = float(means.loc[means.index.min()])
    best = float(means.max())
    best_decile = float(means.idxmax())
    is_top_best = bool(np.isclose(best, top))

    return TestResult(
        name="decile_monotonicity",
        passed=is_top_best and bool(top > bottom),
        detail=(
            f"top decile {top:,.2f}, bottom {bottom:,.2f}, best {best:,.2f} at decile "
            f"{best_decile:.0f}. Top must be the best and beat the bottom; otherwise the "
            "score has no ranking power (April: coverage thresholds, all CIs crossed zero)."
        ),
        metrics={
            "top_decile_mean": top,
            "bottom_decile_mean": bottom,
            "best_decile_mean": best,
            "best_decile_index": best_decile,
        },
    )


def label_balance(labels: Sequence[float]) -> TestResult:
    """FAIL when the target is so imbalanced the head learns a constant.

    Run this BEFORE training. It is the cheapest test here and would have caught
    both the 98%-exit-label disaster and Path-D's 17.46% mirror failure.
    """

    y = _finite(labels)
    if y.size == 0:
        raise GateError("label_balance received no labels")
    rate = float((y > 0).mean())
    ok = MIN_LABEL_POSITIVE_RATE <= rate <= MAX_LABEL_POSITIVE_RATE
    return TestResult(
        name="label_balance",
        passed=bool(ok),
        detail=(
            f"positive-label rate {100*rate:.2f}% (band "
            f"{100*MIN_LABEL_POSITIVE_RATE:.0f}-{100*MAX_LABEL_POSITIVE_RATE:.0f}%). "
            "The label is part of the model architecture."
        ),
        metrics={
            "positive_rate": rate,
            "min": MIN_LABEL_POSITIVE_RATE,
            "max": MAX_LABEL_POSITIVE_RATE,
        },
    )


# --------------------------------------------------------------------------
# Charter-mandated diagnostics
#
# PROTOCOL101_TRADER_CHARTER.md (SIGNED 2026-07-25) lists these under
# "New (to implement)", and PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md
# repeats the harvest ratio and concentration under Mandatory Diagnostics.
# Every packet must report them.
#
# Charter discipline: win rate and concentration are REPORT-ONLY, never
# objectives. "A system rewarded for win rate itself learns to scratch
# everything and pay fees for nothing." Only the big-loss share is a gate.
# --------------------------------------------------------------------------

SCRATCH_BAND = 0.05          # charter: any exit between -5% and +5% is a scratch (fee-aware)
BIG_LOSS_RETURN = -0.25      # charter big losses average -30%; boundary named so it is visible
MAX_BIG_LOSS_SHARE = 0.02    # charter: "the killer discipline is the bottom row staying under 2%"


def four_bucket_distribution(returns_fraction: Sequence[float]) -> TestResult:
    """Charter outcome profile, fee-aware. Gates ONLY on the big-loss share.

    ``returns_fraction`` is per-trade return on premium paid, as a fraction
    (0.40 == +40%). Target shape, directional not literal: many scratches and
    small wins, a real minority of big wins, big losses rare enough to count on
    fingers.
    """

    r = _finite(returns_fraction)
    if r.size == 0:
        raise GateError("four_bucket_distribution received no returns")

    big_win = float((r > SCRATCH_BAND).mean())
    scratch = float(((r >= -SCRATCH_BAND) & (r <= SCRATCH_BAND)).mean())
    small_loss = float(((r < -SCRATCH_BAND) & (r >= BIG_LOSS_RETURN)).mean())
    big_loss = float((r < BIG_LOSS_RETURN).mean())

    return TestResult(
        name="four_bucket_distribution",
        passed=bool(big_loss <= MAX_BIG_LOSS_SHARE),
        detail=(
            f"big win {100*big_win:.1f}% (Pickles 18%) | scratch {100*scratch:.1f}% (73%) | "
            f"small loss {100*small_loss:.1f}% (8%) | BIG LOSS {100*big_loss:.2f}% "
            f"(limit {100*MAX_BIG_LOSS_SHARE:.0f}%). Only the big-loss share gates; "
            "the rest is the shape report."
        ),
        metrics={
            "big_win_share": big_win,
            "scratch_share": scratch,
            "small_loss_share": small_loss,
            "big_loss_share": big_loss,
            "big_loss_limit": MAX_BIG_LOSS_SHARE,
        },
    )


def harvest_ratio(
    realized: Sequence[float], peak_available: Sequence[float]
) -> dict[str, float]:
    """Realized PnL / peak available PnL per trade. Stage-2's report card.

    Report-only. It measures how much of the payoff the exit actually captured,
    which is the metric the charter uses to judge tail harvesting.
    """

    real = np.asarray(realized, dtype=float)
    peak = np.asarray(peak_available, dtype=float)
    if real.shape != peak.shape:
        raise GateError("harvest_ratio requires matching realized/peak lengths")
    usable = np.isfinite(real) & np.isfinite(peak) & (peak > 0)
    if not usable.any():
        raise GateError("harvest_ratio found no trades with positive peak available PnL")

    ratios = real[usable] / peak[usable]
    return {
        "n_trades_with_upside": int(usable.sum()),
        "mean_harvest_ratio": float(ratios.mean()),
        "median_harvest_ratio": float(np.median(ratios)),
        "aggregate_harvest_ratio": float(real[usable].sum() / peak[usable].sum()),
    }


def underwater_duration(equity_curve: Sequence[float]) -> dict[str, float]:
    """Longest stretch below the running high-water mark. Report-only.

    Charter commitment 1: droughts should be FLAT, never deep. This makes
    "flat vs clawback" visible per candidate.
    """

    equity = _finite(equity_curve)
    if equity.size == 0:
        raise GateError("underwater_duration received an empty equity curve")

    high_water = np.maximum.accumulate(equity)
    underwater = equity < high_water
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    drawdown = high_water - equity
    return {
        "longest_underwater_steps": int(longest),
        "underwater_fraction": float(underwater.mean()),
        "max_drawdown": float(drawdown.max()),
    }


def pnl_concentration(values: Sequence[float]) -> dict[str, float]:
    """Share of positive PnL from the top trades. REPORT-ONLY per the charter.

    "Not allergic to lumpy profits: the big-win column carrying most of the
    profit is the design, not a red flag."
    """

    v = _finite(values)
    if v.size == 0:
        raise GateError("pnl_concentration received no values")
    positive = v[v > 0]
    total_positive = float(positive.sum())
    if total_positive <= 0:
        return {"top_5_share": float("nan"), "top_20_share": float("nan"), "positive_pnl": 0.0}
    ordered = np.sort(positive)[::-1]
    return {
        "positive_pnl": total_positive,
        "top_5_share": float(ordered[:5].sum() / total_positive),
        "top_20_share": float(ordered[:20].sum() / total_positive),
    }


# --------------------------------------------------------------------------
# Acceptance tiers (from the standing work order)
# --------------------------------------------------------------------------


def acceptance_tier(
    *,
    pooled_policy: float,
    pooled_comparator: float,
    fold_deltas: Mapping[str, float],
    bootstrap_lcb: float,
    negative_controls_accepted: Mapping[str, bool],
    rejection_tests: Iterable[TestResult],
    maxt_survived: bool,
    concentrated: bool,
) -> dict[str, Any]:
    """Return TIER_A / TIER_B / NO_EDGE / INVALID with every component shown.

    INVALID takes precedence: an accepted negative control discards the run, exactly
    as in the four-box replay and the 2026-08-03 feasibility gate.
    """

    tests = list(rejection_tests)
    beats_pooled = pooled_policy > pooled_comparator
    positive_folds = sum(1 for value in fold_deltas.values() if value > 0.0)
    enough_folds = positive_folds >= 4 and len(fold_deltas) >= 5
    controls_clean = not any(negative_controls_accepted.values())
    tests_pass = all(test.passed for test in tests)

    if not controls_clean:
        tier = "INVALID"
    elif (
        beats_pooled
        and enough_folds
        and bootstrap_lcb > 0.0
        and tests_pass
        and maxt_survived
        and not concentrated
    ):
        tier = "TIER_A"
    elif beats_pooled:
        tier = "TIER_B"
    else:
        tier = "NO_EDGE"

    return {
        "tier": tier,
        "components": {
            "beats_comparator_pooled": beats_pooled,
            "positive_fold_deltas": positive_folds,
            "fold_count": len(fold_deltas),
            "bootstrap_lcb": bootstrap_lcb,
            "negative_controls_clean": controls_clean,
            "rejection_tests_pass": tests_pass,
            "maxt_survived": bool(maxt_survived),
            "not_concentrated": not concentrated,
        },
        "rejection_tests": [test.as_dict() for test in tests],
        "note": (
            "TIER_A means 'worth a forward live-paper test', never 'deployable'. "
            "The protected holdout is SPENT."
        ),
    }
