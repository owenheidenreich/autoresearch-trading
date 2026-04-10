"""Compare QC computed Greeks vs our Black-Scholes estimates.

Usage:
    python -m v2.research.compare_greeks v2/research/qc_08_results_raw.txt

Reads the raw output from qc_08_greeks_comparison.py (pasted from QC Algorithm Lab),
computes our BS Greeks using the same inputs (SPX, strike, IV, minutes_to_close),
and prints a side-by-side comparison with error metrics.

This tells us whether our BS estimates are close enough or if we need real Greeks.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from statistics import mean, median

# ---------------------------------------------------------------------------
# Black-Scholes (copied from archive/v1/training/prepare.py lines 333-353)
# ---------------------------------------------------------------------------

try:
    from scipy.stats import norm as _norm_dist
except ImportError:
    # Minimal fallback for environments without scipy
    import os
    print("scipy not found. Install with: pip install scipy", file=sys.stderr)
    sys.exit(1)

BARS_PER_DAY = 390
RISK_FREE_RATE = 0.05  # same as prepare.py


def _bs_d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_greeks(S, K, T_years, r, sigma):
    """Black-Scholes Greeks for a call. Returns (delta, gamma, theta_annual, vega).
    Matches our prepare.py implementation."""
    if T_years <= 1e-10 or sigma <= 0 or S <= 0:
        return None
    sqrt_T = math.sqrt(T_years)
    d1 = _bs_d1(S, K, T_years, r, sigma)
    d2 = d1 - sigma * sqrt_T
    nd1 = _norm_dist.cdf(d1)
    npdf_d1 = _norm_dist.pdf(d1)

    delta = nd1
    gamma = npdf_d1 / (S * sigma * sqrt_T)
    theta_annual = (-(S * npdf_d1 * sigma) / (2.0 * sqrt_T)
                    - r * K * math.exp(-r * T_years) * _norm_dist.cdf(d2))
    vega = S * npdf_d1 * sqrt_T / 100.0  # per 1% IV move

    return delta, gamma, theta_annual, vega


def bs_greeks_put(S, K, T_years, r, sigma):
    """Black-Scholes Greeks for a put."""
    if T_years <= 1e-10 or sigma <= 0 or S <= 0:
        return None
    sqrt_T = math.sqrt(T_years)
    d1 = _bs_d1(S, K, T_years, r, sigma)
    d2 = d1 - sigma * sqrt_T
    npdf_d1 = _norm_dist.pdf(d1)

    delta = _norm_dist.cdf(d1) - 1.0  # put delta is negative
    gamma = npdf_d1 / (S * sigma * sqrt_T)  # same as call
    theta_annual = (-(S * npdf_d1 * sigma) / (2.0 * sqrt_T)
                    + r * K * math.exp(-r * T_years) * _norm_dist.cdf(-d2))
    vega = S * npdf_d1 * sqrt_T / 100.0  # same as call

    return delta, gamma, theta_annual, vega


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Reading:
    date: str
    bar: int
    right: str  # C or P
    spx: float
    strike: float
    mid: float
    bid: float
    ask: float
    iv: float
    qc_delta: float
    qc_gamma: float
    qc_theta: float  # QC theta (annualized by convention)
    qc_vega: float
    mtc: int  # minutes to close
    is_0dte: bool


@dataclass
class Comparison:
    reading: Reading
    bs_delta: float
    bs_gamma: float
    bs_theta_annual: float
    bs_vega: float


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_qc_output(filepath: str) -> list[Reading]:
    """Parse the raw self.Error() output from qc_08_greeks_comparison.py."""
    with open(filepath, "r") as f:
        raw = f.read().strip()

    # The output format is: header || reading1 || reading2 || ...
    parts = [p.strip() for p in raw.split("||")]

    # First part is the header, skip it
    readings = []
    for part in parts[1:]:
        if not part:
            continue
        fields = part.split("|")
        if len(fields) < 15:
            print(f"Skipping malformed record ({len(fields)} fields): {part[:60]}",
                  file=sys.stderr)
            continue
        try:
            readings.append(Reading(
                date=fields[0].strip(),
                bar=int(fields[1]),
                right=fields[2].strip(),
                spx=float(fields[3]),
                strike=float(fields[4]),
                mid=float(fields[5]),
                bid=float(fields[6]),
                ask=float(fields[7]),
                iv=float(fields[8]),
                qc_delta=float(fields[9]),
                qc_gamma=float(fields[10]),
                qc_theta=float(fields[11]),
                qc_vega=float(fields[12]),
                mtc=int(fields[13]),
                is_0dte=fields[14].strip() == "1",
            ))
        except (ValueError, IndexError) as e:
            print(f"Parse error: {e} in: {part[:80]}", file=sys.stderr)

    return readings


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compute_comparisons(readings: list[Reading]) -> list[Comparison]:
    """For each QC reading, compute our BS Greeks and pair them."""
    results = []
    for r in readings:
        # Convert minutes-to-close to years (same as prepare.py)
        T_years = r.mtc / (252.0 * BARS_PER_DAY)

        if r.iv <= 0 or T_years <= 1e-10:
            continue

        if r.right == "C":
            bs = bs_greeks(r.spx, r.strike, T_years, RISK_FREE_RATE, r.iv)
        else:
            bs = bs_greeks_put(r.spx, r.strike, T_years, RISK_FREE_RATE, r.iv)

        if bs is None:
            continue

        bs_delta, bs_gamma, bs_theta_annual, bs_vega = bs
        results.append(Comparison(
            reading=r,
            bs_delta=bs_delta,
            bs_gamma=bs_gamma,
            bs_theta_annual=bs_theta_annual,
            bs_vega=bs_vega,
        ))

    return results


def pct_error(actual, estimated):
    """Percentage error. Returns None if actual is near zero."""
    if abs(actual) < 1e-10:
        return None
    return (estimated - actual) / abs(actual) * 100.0


def abs_error(actual, estimated):
    return estimated - actual


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(comparisons: list[Comparison]):
    if not comparisons:
        print("No valid comparisons. Check input file.")
        return

    n_0dte = sum(1 for c in comparisons if c.reading.is_0dte)
    n_other = len(comparisons) - n_0dte
    print(f"\n{'='*80}")
    print(f"GREEKS COMPARISON: QC vs Black-Scholes")
    print(f"{'='*80}")
    print(f"Total readings: {len(comparisons)} ({n_0dte} true 0DTE, {n_other} other DTE)")
    print()

    # Filter to 0DTE only for the main analysis
    data_0dte = [c for c in comparisons if c.reading.is_0dte]
    if not data_0dte:
        print("WARNING: No true 0DTE readings found. Showing all DTE data.")
        data_0dte = comparisons

    # --- Per-reading detail table ---
    print(f"{'date':<12} {'bar':>4} {'R':>1} {'mtc':>4} "
          f"{'QC_d':>8} {'BS_d':>8} {'d_err':>7} "
          f"{'QC_g':>10} {'BS_g':>10} {'g_err%':>7} "
          f"{'QC_t':>9} {'BS_t':>9} {'t_err%':>7} "
          f"{'spread':>7}")
    print("-" * 120)

    for c in sorted(data_0dte, key=lambda x: (x.reading.date, x.reading.bar, x.reading.right)):
        r = c.reading
        d_err = abs_error(c.reading.qc_delta, c.bs_delta)
        g_pct = pct_error(c.reading.qc_gamma, c.bs_gamma)
        t_pct = pct_error(c.reading.qc_theta, c.bs_theta_annual)
        spread = r.ask - r.bid if r.bid > 0 and r.ask > 0 else 0

        g_str = f"{g_pct:>6.1f}%" if g_pct is not None else "   N/A"
        t_str = f"{t_pct:>6.1f}%" if t_pct is not None else "   N/A"

        print(f"{r.date:<12} {r.bar:>4} {r.right:>1} {r.mtc:>4} "
              f"{r.qc_delta:>8.4f} {c.bs_delta:>8.4f} {d_err:>+7.4f} "
              f"{r.qc_gamma:>10.6f} {c.bs_gamma:>10.6f} {g_str:>7} "
              f"{r.qc_theta:>9.2f} {c.bs_theta_annual:>9.2f} {t_str:>7} "
              f"${spread:>6.2f}")

    # --- Summary by time bucket ---
    print(f"\n{'='*80}")
    print("SUMMARY BY TIME OF DAY (0DTE only)")
    print(f"{'='*80}")

    buckets = {
        "Open (bar 10-30)": lambda c: c.reading.bar <= 30,
        "Morning (bar 31-120)": lambda c: 31 <= c.reading.bar <= 120,
        "Midday (bar 121-240)": lambda c: 121 <= c.reading.bar <= 240,
        "Afternoon (bar 241-360)": lambda c: 241 <= c.reading.bar <= 360,
        "Close (bar 361+)": lambda c: c.reading.bar > 360,
    }

    print(f"\n{'Bucket':<25} {'N':>4} "
          f"{'delta_err':>10} {'gamma_err%':>11} {'theta_err%':>11} "
          f"{'med_spread':>11}")
    print("-" * 80)

    for name, filt in buckets.items():
        subset = [c for c in data_0dte if filt(c)]
        if not subset:
            print(f"{name:<25} {'--':>4}")
            continue

        d_errs = [abs_error(c.reading.qc_delta, c.bs_delta) for c in subset]
        g_errs = [e for e in (pct_error(c.reading.qc_gamma, c.bs_gamma) for c in subset) if e is not None]
        t_errs = [e for e in (pct_error(c.reading.qc_theta, c.bs_theta_annual) for c in subset) if e is not None]
        spreads = [c.reading.ask - c.reading.bid for c in subset
                   if c.reading.bid > 0 and c.reading.ask > 0]

        med_d = median(d_errs) if d_errs else 0
        med_g = median(g_errs) if g_errs else 0
        med_t = median(t_errs) if t_errs else 0
        med_s = median(spreads) if spreads else 0

        print(f"{name:<25} {len(subset):>4} "
              f"{med_d:>+10.4f} {med_g:>+10.1f}% {med_t:>+10.1f}% "
              f"${med_s:>10.2f}")

    # --- Bid-ask spread summary ---
    print(f"\n{'='*80}")
    print("BID-ASK SPREAD ANALYSIS")
    print(f"{'='*80}")

    spreads_by_bucket = {}
    for c in data_0dte:
        r = c.reading
        if r.bid <= 0 or r.ask <= 0:
            continue
        spread = r.ask - r.bid
        bucket = "Open" if r.bar <= 60 else "Mid" if r.bar <= 240 else "Close"
        key = (bucket, r.right)
        if key not in spreads_by_bucket:
            spreads_by_bucket[key] = []
        spreads_by_bucket[key].append(spread)

    print(f"\n{'Period':<10} {'Right':>5} {'N':>4} {'Median':>8} {'Mean':>8} {'Our est':>8} {'Diff':>8}")
    print("-" * 55)

    # Our fixed estimate for comparison
    our_spread = 0.30

    for (bucket, right), vals in sorted(spreads_by_bucket.items()):
        med_val = median(vals)
        mean_val = mean(vals)
        diff = med_val - our_spread
        print(f"{bucket:<10} {right:>5} {len(vals):>4} "
              f"${med_val:>7.2f} ${mean_val:>7.2f} ${our_spread:>7.2f} ${diff:>+7.2f}")

    # --- Verdict ---
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    all_g_errs = [abs(e) for e in (pct_error(c.reading.qc_gamma, c.bs_gamma) for c in data_0dte) if e is not None]
    all_t_errs = [abs(e) for e in (pct_error(c.reading.qc_theta, c.bs_theta_annual) for c in data_0dte) if e is not None]
    all_spreads = [c.reading.ask - c.reading.bid for c in data_0dte
                   if c.reading.bid > 0 and c.reading.ask > 0]

    if all_g_errs:
        med_g_err = median(all_g_errs)
        print(f"\nGamma: median absolute error = {med_g_err:.1f}%")
        if med_g_err < 5:
            print("  -> BS gamma is GOOD. No need for real Greeks.")
        elif med_g_err < 20:
            print("  -> BS gamma has MODERATE error. Could improve with real Greeks.")
        else:
            print("  -> BS gamma is POOR. Real Greeks would likely help training.")

    if all_t_errs:
        med_t_err = median(all_t_errs)
        print(f"\nTheta: median absolute error = {med_t_err:.1f}%")
        if med_t_err < 5:
            print("  -> BS theta is GOOD. No need for real Greeks.")
        elif med_t_err < 20:
            print("  -> BS theta has MODERATE error. Could improve with real Greeks.")
        else:
            print("  -> BS theta is POOR. Real Greeks would likely help training.")

    if all_spreads:
        med_spread = median(all_spreads)
        print(f"\nBid-ask spread: median = ${med_spread:.2f} (our estimate: ${our_spread:.2f})")
        ratio = med_spread / our_spread if our_spread > 0 else 999
        if 0.5 < ratio < 2.0:
            print("  -> Our $0.30 estimate is REASONABLE.")
        else:
            print(f"  -> Our $0.30 estimate is OFF by {ratio:.1f}x. Real spreads would help labels.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m v2.research.compare_greeks <qc_output_file>")
        print("  e.g. python -m v2.research.compare_greeks v2/research/qc_08_results_raw.txt")
        sys.exit(1)

    filepath = sys.argv[1]
    readings = parse_qc_output(filepath)
    print(f"Parsed {len(readings)} readings from {filepath}")

    comparisons = compute_comparisons(readings)
    print(f"Computed {len(comparisons)} valid BS comparisons")

    print_report(comparisons)
