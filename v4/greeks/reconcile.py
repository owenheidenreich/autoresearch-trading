"""Reconcile our computed Black-Scholes Greeks against OptionsDX vendor Greeks.

Phase-0 sanity check: if our BS implementation disagrees with OptionsDX
beyond a tolerance, either our IV inversion is wrong, our Greek formulas
are wrong, or our conventions are wrong. Catching those at Phase 0 prevents
silent bugs in feature engineering downstream.

Why this matters: Databento (Phase 1) does NOT provide Greeks; we compute
them ourselves. If our implementation hasn't been validated against a
trusted source first, every Greek-derived feature is suspect. OptionsDX
is that trusted source for Phase 0 validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pyarrow as pa

from v4.checks.sanity import CheckResult
from v4.greeks.black_scholes import greeks, implied_vol, to_optionsdx_conventions


@dataclass(frozen=True)
class ReconciliationStat:
    metric: str
    n: int
    mean_abs_error: float
    max_abs_error: float
    tolerance: float
    passed: bool


# Tolerances chosen for OptionsDX 4-decimal published precision.
DEFAULT_TOLERANCES = {
    "delta": 0.01,           # OptionsDX delta to 3 decimals; allow 0.01 for 0DTE convexity
    "gamma": 0.005,          # gamma small for SPX index; tight tolerance
    "vega_per_1pct": 0.05,   # OptionsDX vega is per 1% IV; tolerate 0.05 absolute
    "theta_per_day": 1.5,    # 0DTE theta is huge — tolerate 1.5/day
    "rho_per_1pct": 0.10,
}


def _years_to_expiry(event_time, expiry: date) -> float:
    """Calendar years from event_time to 4pm ET on expiry date.

    SPXW is PM-settled at 4pm ET. Phase 0 uses calendar year fraction
    (365.25-day year). Phase-1+ may switch to trading-day fraction; doc
    the choice in DATA_CONTRACT.md before changing.
    """
    # event_time is timezone-aware datetime in UTC
    from datetime import datetime, time, timezone
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    settle_local = datetime.combine(expiry, time(16, 0), tzinfo=et)
    settle_utc = settle_local.astimezone(timezone.utc)
    seconds = (settle_utc - event_time).total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / (365.25 * 24 * 3600)


def reconcile_optionsdx_table(
    table: pa.Table,
    *,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    tolerances: dict[str, float] | None = None,
) -> tuple[CheckResult, list[ReconciliationStat]]:
    """For each Normalized row from OptionsDX with vendor-supplied IV+Greeks,
    recompute Greeks via BS using the same IV; compare against vendor values.

    Returns:
        - aggregate CheckResult (pass = all metrics within tolerance)
        - per-metric stats for the report

    A failure here is a Phase-0 blocker: do not proceed until our BS Greeks
    match the vendor's within tolerance for the rows we can compare.
    """
    tolerances = tolerances or DEFAULT_TOLERANCES

    rows_used = 0
    abs_errors: dict[str, list[float]] = {k: [] for k in DEFAULT_TOLERANCES}

    iv_list = table["iv"].to_pylist()
    delta_list = table["delta"].to_pylist()
    gamma_list = table["gamma"].to_pylist()
    vega_list = table["vega"].to_pylist()
    theta_list = table["theta"].to_pylist()
    rho_list = table["rho"].to_pylist()
    underlying_list = table["underlying_price"].to_pylist()
    strike_list = table["strike"].to_pylist()
    expiry_list = table["expiry"].to_pylist()
    event_list = table["event_time"].to_pylist()
    right_list = table["right"].to_pylist()

    for i in range(table.num_rows):
        if iv_list[i] is None or underlying_list[i] is None:
            continue
        S = float(underlying_list[i])
        K = float(strike_list[i])
        sigma = float(iv_list[i])
        T = _years_to_expiry(event_list[i], expiry_list[i])
        if T <= 0 or sigma <= 0:
            continue

        is_call = right_list[i] == "C"
        try:
            g = greeks(
                S=S, K=K, T=T, sigma=sigma,
                r=risk_free_rate, q=dividend_yield, is_call=is_call,
            )
        except (ValueError, ArithmeticError):
            continue

        gx = to_optionsdx_conventions(g)
        rows_used += 1

        if delta_list[i] is not None:
            abs_errors["delta"].append(abs(gx["delta"] - float(delta_list[i])))
        if gamma_list[i] is not None:
            abs_errors["gamma"].append(abs(gx["gamma"] - float(gamma_list[i])))
        if vega_list[i] is not None:
            abs_errors["vega_per_1pct"].append(
                abs(gx["vega_per_1pct"] - float(vega_list[i]))
            )
        if theta_list[i] is not None:
            abs_errors["theta_per_day"].append(
                abs(gx["theta_per_day"] - float(theta_list[i]))
            )
        if rho_list[i] is not None:
            abs_errors["rho_per_1pct"].append(
                abs(gx["rho_per_1pct"] - float(rho_list[i]))
            )

    stats: list[ReconciliationStat] = []
    for metric, errs in abs_errors.items():
        if not errs:
            stats.append(
                ReconciliationStat(
                    metric=metric, n=0, mean_abs_error=0.0, max_abs_error=0.0,
                    tolerance=tolerances[metric], passed=True,
                )
            )
            continue
        mae = sum(errs) / len(errs)
        max_e = max(errs)
        stats.append(
            ReconciliationStat(
                metric=metric, n=len(errs),
                mean_abs_error=mae, max_abs_error=max_e,
                tolerance=tolerances[metric],
                passed=max_e <= tolerances[metric],
            )
        )

    overall_passed = all(s.passed for s in stats)
    if rows_used == 0:
        result = CheckResult(
            name="greeks_reconciliation_optionsdx",
            passed=False,
            details="no rows had vendor IV + underlying — cannot reconcile",
        )
    elif overall_passed:
        result = CheckResult(
            name="greeks_reconciliation_optionsdx",
            passed=True,
            details=f"{rows_used} rows reconciled within tolerance: " + ", ".join(
                f"{s.metric} max |err|={s.max_abs_error:.4f} <= {s.tolerance}"
                for s in stats if s.n > 0
            ),
        )
    else:
        worst = max(
            (s for s in stats if s.n > 0),
            key=lambda s: s.max_abs_error / max(s.tolerance, 1e-12),
        )
        result = CheckResult(
            name="greeks_reconciliation_optionsdx",
            passed=False,
            details=(
                f"reconciliation failed on {worst.metric}: "
                f"max |err|={worst.max_abs_error:.4f} > tol {worst.tolerance}"
            ),
            bad_rows=worst.n,
        )

    return result, stats
