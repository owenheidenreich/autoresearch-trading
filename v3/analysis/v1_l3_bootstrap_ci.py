"""Phase 1A — Bootstrap confidence interval on V1+L3 OOS PF.

Loads the 20 V1+L3 composed OOS trades, resamples with replacement
N times, computes profit factor (PF) on each resample, and reports
the bootstrap distribution. The 95% CI lower bound is the gate
value: lower-bound >= 1.0 = Phase 1A PASS; < 1.0 = TERMINAL FAIL.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRADES = (
    REPO_ROOT
    / "v3/artifacts/layer2_directional_composed_oos/composed_oos_trades_V1.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "v3/artifacts/v1_l3_bootstrap_ci"


def load_pnls(path: Path) -> np.ndarray:
    pnls: list[float] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pnls.append(float(row["pnl"]))
    return np.asarray(pnls, dtype=np.float64)


def profit_factor(pnls: np.ndarray) -> float:
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    if neg == 0.0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def bootstrap_pfs(pnls: np.ndarray, n_iter: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = pnls.shape[0]
    out = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        out[i] = profit_factor(pnls[idx])
    return out


def percentile_summary(pfs: np.ndarray) -> dict[str, float]:
    finite = pfs[np.isfinite(pfs)]
    return {
        "n_iter": int(pfs.size),
        "n_finite": int(finite.size),
        "n_inf": int((~np.isfinite(pfs)).sum()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p2_5": float(np.percentile(finite, 2.5)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p97_5": float(np.percentile(finite, 97.5)),
    }


def fraction_above(pfs: np.ndarray, threshold: float) -> float:
    finite = pfs[np.isfinite(pfs)]
    if finite.size == 0:
        return 0.0
    return float((finite >= threshold).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--n-iter", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    pnls = load_pnls(args.trades)
    if pnls.size == 0:
        raise SystemExit(f"no trades loaded from {args.trades}")

    observed_pf = profit_factor(pnls)

    pfs = bootstrap_pfs(pnls, n_iter=args.n_iter, seed=args.seed)
    summary = percentile_summary(pfs)
    frac_pf_ge_1 = fraction_above(pfs, 1.0)
    frac_pf_ge_1_5 = fraction_above(pfs, 1.5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "trades_file": str(args.trades),
        "n_trades": int(pnls.size),
        "observed_pf": observed_pf,
        "n_iter": args.n_iter,
        "seed": args.seed,
        "bootstrap": summary,
        "fraction_pf_ge_1": frac_pf_ge_1,
        "fraction_pf_ge_1_5": frac_pf_ge_1_5,
    }
    out_path = args.out_dir / "v1_l3_bootstrap_ci.json"
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"# Phase 1A bootstrap CI on V1+L3 OOS")
    print(f"trades file:   {args.trades}")
    print(f"n trades:      {pnls.size}")
    print(f"observed PF:   {observed_pf:.4f}")
    print(f"n_iter:        {args.n_iter}")
    print(f"seed:          {args.seed}")
    print()
    print(f"Bootstrap PF distribution:")
    print(f"  mean   = {summary['mean']:.4f}")
    print(f"  std    = {summary['std']:.4f}")
    print(f"  min    = {summary['min']:.4f}")
    print(f"  p2.5   = {summary['p2_5']:.4f}   <-- 95% CI lower")
    print(f"  p25    = {summary['p25']:.4f}")
    print(f"  p50    = {summary['p50']:.4f}")
    print(f"  p75    = {summary['p75']:.4f}")
    print(f"  p97.5  = {summary['p97_5']:.4f}   <-- 95% CI upper")
    print(f"  max    = {summary['max']:.4f}")
    print(f"  n_inf  = {summary['n_inf']}  (resamples with no losers)")
    print()
    print(f"Fraction PF >= 1.0:  {frac_pf_ge_1:.4f}")
    print(f"Fraction PF >= 1.5:  {frac_pf_ge_1_5:.4f}")
    print()

    ci_lower = summary["p2_5"]
    if ci_lower >= 1.5:
        verdict = "STRONG-PASS (95% CI lower >= 1.5)"
    elif ci_lower >= 1.0:
        verdict = "PASS (95% CI lower >= 1.0)"
    else:
        verdict = "TERMINAL-FAIL (95% CI lower < 1.0)"
    print(f"Phase 1A verdict: {verdict}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
