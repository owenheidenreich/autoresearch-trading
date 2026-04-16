"""Audit: What does the candidate set look like if restricted to near-ATM?

Answers:
1. How many contracts per bar are within ATM ± N strikes?
2. How often is the oracle's best contract near-ATM vs far-OTM?
3. What is the ranking difficulty (margin, PnL spread) within ATM-only vs all?
4. How does oracle PnL change if restricted to ATM ± 3 strikes?
5. What fraction of oracle value is captured by the top-K near-ATM contracts?

Usage:
    python3 -m v2.analysis.candidate_set_audit
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.core.chain_data import padded_snapshot


def main():
    data_path = "v2/data.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    meta = data.get("metadata", {})
    sidecar_dir = meta["chain_sidecar_dir"]
    max_contracts = int(meta["max_contracts_per_bar"])
    features = data["X"]
    dates = data["dates"]
    bar_of_day = data["bar_of_day"]
    spot_prices = data["spot_prices"]
    promote_mask = data["promote_mask"].numpy()

    # Collect eligible bars
    eligible = []
    for i in range(30, len(features)):
        if not promote_mask[i]:
            continue
        bod = int(bar_of_day[i])
        if bod < 30 or bod >= 270:
            continue
        eligible.append(i)

    # Sample for speed
    rng = np.random.RandomState(42)
    if len(eligible) > 8000:
        eligible = list(rng.choice(eligible, 8000, replace=False))
    eligible.sort()

    print(f"Analyzing {len(eligible):,} promote bars...")

    sidecar_cache = {}
    t0 = time.time()

    # Per-bar stats
    all_n_valid = []
    all_n_call = []
    all_n_put = []
    oracle_moneyness_pct = []  # |strike - spot| / spot * 100 for oracle contract
    oracle_is_atm3 = []  # oracle within ATM ± 3 strikes
    oracle_is_atm5 = []  # oracle within ATM ± 5 strikes
    oracle_pnl_full = []
    oracle_pnl_atm3 = []  # best PnL if restricted to ATM ± 3
    oracle_pnl_atm5 = []  # best PnL if restricted to ATM ± 5
    oracle_pnl_atm_side = []  # best PnL if restricted to ATM ± 3 on oracle side only
    n_atm3 = []
    n_atm5 = []
    margin_full = []  # best - 2nd best (full set)
    margin_atm3 = []  # best - 2nd best (ATM ± 3 only)
    oracle_side = []  # 'C' or 'P'
    bar_of_days = []

    for count, i in enumerate(eligible):
        if count % 2000 == 0 and count > 0:
            print(f"  {count}/{len(eligible)}...")
        day = dates[i]
        local_bar = int(bar_of_day[i])
        spot = float(spot_prices[i])

        if day not in sidecar_cache:
            sc_path = os.path.join(sidecar_dir, f"{day}.pt")
            if not os.path.exists(sc_path):
                continue
            sidecar_cache[day] = torch.load(sc_path, map_location="cpu", weights_only=False)
        sc = sidecar_cache[day]

        c, l, _ = padded_snapshot(sc, local_bar, max_contracts)

        # c shape: (max_contracts, 22)
        # l shape: (max_contracts,)
        # Contract features: 0=valid, 1=strike, 2=right_is_put, 11=moneyness_pct
        valid = c[:, 0] > 0.5
        n_v = int(valid.sum())
        if n_v == 0:
            continue

        strikes = c[:, 1]
        is_put = c[:, 2] > 0.5
        moneyness = np.abs(c[:, 11])  # |moneyness_pct| — after z-scoring in model, but raw here

        # Compute moneyness from raw strike vs spot (more reliable than z-scored feature)
        raw_moneyness_pct = np.abs((strikes - spot) / spot * 100)

        # Valid contract labels
        valid_entries = []
        for k in range(max_contracts):
            if valid[k] and np.isfinite(l[k]):
                side = "P" if is_put[k] else "C"
                valid_entries.append({
                    "idx": k,
                    "strike": float(strikes[k]),
                    "side": side,
                    "pnl": float(l[k]),
                    "moneyness_pct": float(raw_moneyness_pct[k]),
                })

        if not valid_entries:
            continue

        n_calls = sum(1 for e in valid_entries if e["side"] == "C")
        n_puts = sum(1 for e in valid_entries if e["side"] == "P")

        # Sort by PnL to find oracle
        sorted_by_pnl = sorted(valid_entries, key=lambda x: -x["pnl"])
        oracle = sorted_by_pnl[0]
        oracle_m = oracle["moneyness_pct"]

        # ATM ± N: contracts within N% of spot
        atm3_entries = [e for e in valid_entries if e["moneyness_pct"] <= 0.5]  # within 0.5% of spot ≈ ATM ± 3 strikes
        atm5_entries = [e for e in valid_entries if e["moneyness_pct"] <= 1.0]  # within 1.0% of spot ≈ ATM ± 5 strikes

        # ATM ± 3 on oracle side only
        atm3_side = [e for e in valid_entries if e["moneyness_pct"] <= 0.5 and e["side"] == oracle["side"]]

        best_atm3_pnl = max((e["pnl"] for e in atm3_entries), default=float("nan"))
        best_atm5_pnl = max((e["pnl"] for e in atm5_entries), default=float("nan"))
        best_atm3_side_pnl = max((e["pnl"] for e in atm3_side), default=float("nan"))

        # Margin (best - 2nd best)
        full_margin = sorted_by_pnl[0]["pnl"] - sorted_by_pnl[1]["pnl"] if len(sorted_by_pnl) >= 2 else 0.0
        atm3_sorted = sorted(atm3_entries, key=lambda x: -x["pnl"])
        atm3_margin_val = atm3_sorted[0]["pnl"] - atm3_sorted[1]["pnl"] if len(atm3_sorted) >= 2 else 0.0

        all_n_valid.append(len(valid_entries))
        all_n_call.append(n_calls)
        all_n_put.append(n_puts)
        oracle_moneyness_pct.append(oracle_m)
        oracle_is_atm3.append(oracle_m <= 0.5)
        oracle_is_atm5.append(oracle_m <= 1.0)
        oracle_pnl_full.append(oracle["pnl"])
        oracle_pnl_atm3.append(best_atm3_pnl)
        oracle_pnl_atm5.append(best_atm5_pnl)
        oracle_pnl_atm_side.append(best_atm3_side_pnl)
        n_atm3.append(len(atm3_entries))
        n_atm5.append(len(atm5_entries))
        margin_full.append(full_margin)
        margin_atm3.append(atm3_margin_val)
        oracle_side.append(oracle["side"])
        bar_of_days.append(local_bar)

    elapsed = time.time() - t0
    print(f"Analysis: {elapsed:.1f}s")

    # Convert
    n_valid = np.array(all_n_valid)
    n_call = np.array(all_n_call)
    n_put = np.array(all_n_put)
    o_money = np.array(oracle_moneyness_pct)
    o_atm3 = np.array(oracle_is_atm3)
    o_atm5 = np.array(oracle_is_atm5)
    o_pnl = np.array(oracle_pnl_full)
    o_pnl_a3 = np.array(oracle_pnl_atm3)
    o_pnl_a5 = np.array(oracle_pnl_atm5)
    o_pnl_as = np.array(oracle_pnl_atm_side)
    n_a3 = np.array(n_atm3)
    n_a5 = np.array(n_atm5)
    m_full = np.array(margin_full)
    m_atm3 = np.array(margin_atm3)
    o_side = np.array(oracle_side)
    bod = np.array(bar_of_days)
    n = len(n_valid)

    print(f"\n{'='*75}")
    print(f"  CANDIDATE SET AUDIT ({n:,} bars)")
    print(f"{'='*75}")

    # --- 1. Contract counts ---
    print(f"\n--- 1. Contract Counts ---")
    print(f"  All valid:   mean={n_valid.mean():.1f}  median={np.median(n_valid):.0f}  "
          f"min={n_valid.min()}  max={n_valid.max()}")
    print(f"  Calls:       mean={n_call.mean():.1f}  median={np.median(n_call):.0f}")
    print(f"  Puts:        mean={n_put.mean():.1f}  median={np.median(n_put):.0f}")
    print(f"  ATM ± 0.5%:  mean={n_a3.mean():.1f}  median={np.median(n_a3):.0f}")
    print(f"  ATM ± 1.0%:  mean={n_a5.mean():.1f}  median={np.median(n_a5):.0f}")

    # --- 2. Oracle moneyness ---
    print(f"\n--- 2. Oracle Contract Moneyness ---")
    print(f"  Oracle |moneyness|:  mean={o_money.mean():.3f}%  median={np.median(o_money):.3f}%")
    print(f"  Oracle within ATM ± 0.5%: {o_atm3.mean():.1%}")
    print(f"  Oracle within ATM ± 1.0%: {o_atm5.mean():.1%}")
    print(f"  Oracle within ATM ± 2.0%: {(o_money <= 2.0).mean():.1%}")
    print(f"  Oracle > 3% OTM:          {(o_money > 3.0).mean():.1%}")

    print(f"\n  Oracle moneyness distribution:")
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 100)]:
        mask = (o_money >= lo) & (o_money < hi)
        frac = mask.mean()
        mean_pnl = o_pnl[mask].mean() if mask.any() else 0
        print(f"    {lo:.2f}-{hi:.2f}%: {frac:.1%} of bars, oracle PnL={mean_pnl:.3f}")

    # --- 3. Oracle side ---
    print(f"\n--- 3. Oracle Side ---")
    call_frac = (o_side == "C").mean()
    put_frac = (o_side == "P").mean()
    print(f"  Call oracle: {call_frac:.1%}")
    print(f"  Put oracle:  {put_frac:.1%}")

    # --- 4. PnL capture with reduced candidate sets ---
    print(f"\n--- 4. PnL Capture with Reduced Candidate Sets ---")
    both_valid_a3 = np.isfinite(o_pnl) & np.isfinite(o_pnl_a3)
    both_valid_a5 = np.isfinite(o_pnl) & np.isfinite(o_pnl_a5)
    both_valid_as = np.isfinite(o_pnl) & np.isfinite(o_pnl_as)

    print(f"  Full set oracle PnL:         mean={o_pnl.mean():.4f} ({o_pnl.mean()*100:.1f}%)")
    if both_valid_a3.any():
        ratio = o_pnl_a3[both_valid_a3].mean() / o_pnl[both_valid_a3].mean()
        print(f"  ATM ± 0.5% best PnL:        mean={o_pnl_a3[both_valid_a3].mean():.4f} "
              f"({ratio:.1%} of full oracle)")
    if both_valid_a5.any():
        ratio = o_pnl_a5[both_valid_a5].mean() / o_pnl[both_valid_a5].mean()
        print(f"  ATM ± 1.0% best PnL:        mean={o_pnl_a5[both_valid_a5].mean():.4f} "
              f"({ratio:.1%} of full oracle)")
    if both_valid_as.any():
        ratio = o_pnl_as[both_valid_as].mean() / o_pnl[both_valid_as].mean()
        print(f"  ATM ± 0.5% (oracle side):   mean={o_pnl_as[both_valid_as].mean():.4f} "
              f"({ratio:.1%} of full oracle)")

    # Bars where ATM-restricted oracle is still > 4% profitable
    if both_valid_a3.any():
        a3_profitable = (o_pnl_a3[both_valid_a3] > 0.04).mean()
        print(f"\n  ATM ± 0.5% oracle > 4%:     {a3_profitable:.1%} of bars")
    if both_valid_a5.any():
        a5_profitable = (o_pnl_a5[both_valid_a5] > 0.04).mean()
        print(f"  ATM ± 1.0% oracle > 4%:     {a5_profitable:.1%} of bars")

    # --- 5. Margin analysis ---
    print(f"\n--- 5. Margin (best - 2nd best PnL) ---")
    print(f"  Full set:    mean={m_full.mean():.4f}  median={np.median(m_full):.4f}")
    a3_valid_margin = m_atm3[n_a3 >= 2]
    if len(a3_valid_margin) > 0:
        print(f"  ATM ± 0.5%:  mean={a3_valid_margin.mean():.4f}  median={np.median(a3_valid_margin):.4f}")
    print(f"  Full margin < 1%:    {(m_full < 0.01).mean():.1%}")
    if len(a3_valid_margin) > 0:
        print(f"  ATM margin < 1%:     {(a3_valid_margin < 0.01).mean():.1%}")
    print(f"  Full margin < 5%:    {(m_full < 0.05).mean():.1%}")

    # --- 6. Candidate set size vs ranking difficulty ---
    print(f"\n--- 6. ATM-Restricted Set Size Distribution ---")
    for threshold, name in [(0.5, "ATM±0.5%"), (1.0, "ATM±1.0%")]:
        n_arr = n_a3 if threshold == 0.5 else n_a5
        print(f"\n  {name}:")
        for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 15), (15, 999)]:
            mask = (n_arr >= lo) & (n_arr < hi)
            if mask.any():
                print(f"    {lo}-{hi} contracts: {mask.mean():.1%} of bars")

    # --- 7. Time-of-day interaction ---
    print(f"\n--- 7. Oracle Moneyness by Time of Day ---")
    for lo, hi, label in [(30, 90, "open"), (90, 150, "mid-morn"),
                           (150, 210, "midday"), (210, 270, "afternoon")]:
        mask = (bod >= lo) & (bod < hi)
        if mask.any():
            print(f"  {label:>12s}: oracle |m|={o_money[mask].mean():.3f}%  "
                  f"ATM±0.5%={o_atm3[mask].mean():.1%}  "
                  f"n_valid={n_valid[mask].mean():.1f}  "
                  f"n_atm3={n_a3[mask].mean():.1f}")

    print(f"\n{'='*75}")
    print(f"  AUDIT COMPLETE")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
