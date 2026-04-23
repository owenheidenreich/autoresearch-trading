"""0DTE contract decay-pattern analysis.

For each contract observable at a candidate entry bar (9:45 / 10:00 / 10:30 /
11:00 ET), trace its mid forward to end-of-session and record:
- entry_premium, entry_abs_delta, entry_minute
- max_drawdown_pct (most negative pnl% seen over the forward path)
- max_gain_pct (most positive pnl% seen)
- final_pnl_pct (session-close mid vs entry)
- recovery flag: did max_drawdown <= -60% and final_pnl_pct > 0?

Output: stratified tables by premium tier and delta tier.
Scratch-only; decisions are driven by the output, code is throwaway.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

from v2.core.chain_data import CONTRACT_FEATURE_FIELDS
from v3.harness.v2_adapter import V2Dataset, load_day_sidecar

# CONTRACT_FEATURE_FIELDS column indices
CIDX_VALID = 0
CIDX_STRIKE = 1
CIDX_RIGHT_IS_PUT = 2
CIDX_MID = 3
CIDX_DELTA = 8

# Entry bars we'll sample (minutes from 09:30 ET)
ENTRY_BARS = [15, 30, 60, 90]  # 09:45, 10:00, 10:30, 11:00
SESSION_END_BAR = 375  # ~15:45 ET, avoids the final-15-min chaos

# Premium tiers (dollars)
PREMIUM_TIERS = [
    ("< $100",      0,    100),
    ("$100-300",    100,  300),
    ("$300-700",    300,  700),
    ("$700-1500",   700,  1500),
    ("$1500-3000",  1500, 3000),
    (">= $3000",    3000, 1e9),
]

DELTA_TIERS = [
    ("0.05-0.15",   0.05, 0.15),
    ("0.15-0.25",   0.15, 0.25),
    ("0.25-0.35",   0.25, 0.35),
    ("0.35-0.50",   0.35, 0.50),
    ("0.50-0.75",   0.50, 0.75),
    ("0.75-0.99",   0.75, 0.99),
]


def tier_of(value, tiers):
    for name, lo, hi in tiers:
        if lo <= value < hi:
            return name
    return None


def contract_path(sidecar, contract_idx, start_bar, end_bar):
    """Trace a contract's (bar, mid, valid) path from start to end_bar.

    Returns np arrays of valid-masked (bars, mids). A bar where the contract
    is not observed is skipped.
    """
    ptrs = sidecar["bar_ptrs"]
    row_idx = sidecar["row_contract_idx"]
    feats = sidecar["row_features"]
    bars_out = []
    mids_out = []
    for b in range(start_bar, min(end_bar + 1, len(ptrs) - 1)):
        s = int(ptrs[b])
        e = int(ptrs[b + 1])
        if e <= s:
            continue
        mask = row_idx[s:e] == contract_idx
        hits = np.where(mask)[0]
        if len(hits) == 0:
            continue
        r = feats[s + hits[0]]
        if r[CIDX_VALID] < 0.5:
            continue
        mid = float(r[CIDX_MID])
        if not (mid > 0):
            continue
        bars_out.append(b)
        mids_out.append(mid)
    return np.asarray(bars_out), np.asarray(mids_out)


def analyze_day(sidecar, entry_bar):
    """For one day and one entry bar, yield (entry_premium, entry_abs_delta,
    entry_minute, max_dd_pct, max_gain_pct, final_pnl_pct) per observable contract.
    """
    ptrs = sidecar["bar_ptrs"]
    row_idx = sidecar["row_contract_idx"]
    feats = sidecar["row_features"]
    if entry_bar + 1 >= len(ptrs):
        return
    s = int(ptrs[entry_bar])
    e = int(ptrs[entry_bar + 1])
    if e <= s:
        return
    entry_rows = feats[s:e]
    entry_contract_ids = row_idx[s:e]
    for i in range(entry_rows.shape[0]):
        r = entry_rows[i]
        if r[CIDX_VALID] < 0.5:
            continue
        mid_entry = float(r[CIDX_MID])
        if not (mid_entry > 0):
            continue
        abs_d = abs(float(r[CIDX_DELTA]))
        cid = int(entry_contract_ids[i])

        bars, mids = contract_path(sidecar, cid, entry_bar + 1, SESSION_END_BAR)
        if len(mids) < 3:
            continue  # not enough forward observations

        mids_arr = np.asarray(mids)
        pnl_pct = (mids_arr - mid_entry) / mid_entry * 100.0
        max_dd = float(pnl_pct.min())
        max_gain = float(pnl_pct.max())
        final = float(pnl_pct[-1])
        yield {
            "entry_premium": mid_entry * 100.0,
            "entry_abs_delta": abs_d,
            "entry_minute": entry_bar,
            "max_dd_pct": max_dd,
            "max_gain_pct": max_gain,
            "final_pnl_pct": final,
            "recovered": max_dd <= -60.0 and final > 0.0,
            "went_to_zero": final <= -90.0,
        }


def main(n_days: int = 40):
    ds = V2Dataset.load()
    all_days = sorted(set(ds.dates))
    # Evenly sample n_days across the full cache
    step = max(1, len(all_days) // n_days)
    sample_days = all_days[::step][:n_days]
    print(f"Analyzing {len(sample_days)} days (sampled from {len(all_days)}): "
          f"{sample_days[0]} ... {sample_days[-1]}")

    records = []
    for day in sample_days:
        sc = load_day_sidecar(ds, day)
        if sc is None:
            continue
        for entry_bar in ENTRY_BARS:
            for r in analyze_day(sc, entry_bar):
                records.append(r)

    print(f"Total contract-entries analyzed: {len(records):,}")
    print()

    # --- Aggregate by premium tier ---
    def summarize(group_fn, tiers, label):
        print(f"=== Stratified by {label} ===")
        print(f"{'tier':<16}{'n':>8}{'mean_dd':>10}{'med_dd':>10}"
              f"{'p05_dd':>10}{'mean_fin':>10}{'med_fin':>10}"
              f"{'%zero':>8}{'%recov':>8}")
        buckets = defaultdict(list)
        for rec in records:
            tier = tier_of(group_fn(rec), tiers)
            if tier is None:
                continue
            buckets[tier].append(rec)
        for name, _, _ in tiers:
            rs = buckets.get(name, [])
            if not rs:
                print(f"{name:<16}{0:>8}")
                continue
            dd = np.asarray([r["max_dd_pct"] for r in rs])
            fin = np.asarray([r["final_pnl_pct"] for r in rs])
            pct_zero = 100.0 * np.mean([r["went_to_zero"] for r in rs])
            pct_recov = 100.0 * np.mean([r["recovered"] for r in rs])
            print(f"{name:<16}{len(rs):>8}"
                  f"{float(dd.mean()):>10.1f}{float(np.median(dd)):>10.1f}"
                  f"{float(np.percentile(dd, 5)):>10.1f}"
                  f"{float(fin.mean()):>10.1f}{float(np.median(fin)):>10.1f}"
                  f"{pct_zero:>7.1f}%{pct_recov:>7.1f}%")
        print()

    summarize(lambda r: r["entry_premium"], PREMIUM_TIERS, "entry premium")
    summarize(lambda r: r["entry_abs_delta"], DELTA_TIERS, "entry abs(delta)")

    # --- Cross: premium tier × entry time ---
    print("=== %final<=-90% (wiped) by premium tier and entry minute ===")
    print(f"{'premium':<16}" + "".join(f"{b:>8}" for b in ENTRY_BARS))
    for p_name, p_lo, p_hi in PREMIUM_TIERS:
        row = f"{p_name:<16}"
        for b in ENTRY_BARS:
            subset = [r for r in records
                      if r["entry_minute"] == b and p_lo <= r["entry_premium"] < p_hi]
            if not subset:
                row += f"{'---':>8}"
            else:
                pct = 100.0 * np.mean([r["went_to_zero"] for r in subset])
                row += f"{pct:>7.1f}%"
        print(row)
    print()

    # --- Recovery-after-drawdown fraction ---
    print("=== Of contracts hitting max_dd <= -60%, what % ended positive? ===")
    print(f"{'premium':<16}{'n_hit_-60%':>14}{'% recovered':>14}")
    for p_name, p_lo, p_hi in PREMIUM_TIERS:
        subset = [r for r in records
                  if p_lo <= r["entry_premium"] < p_hi and r["max_dd_pct"] <= -60.0]
        if not subset:
            print(f"{p_name:<16}{0:>14}{'---':>14}")
            continue
        pct_rec = 100.0 * np.mean([r["final_pnl_pct"] > 0 for r in subset])
        print(f"{p_name:<16}{len(subset):>14}{pct_rec:>13.1f}%")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(n)
