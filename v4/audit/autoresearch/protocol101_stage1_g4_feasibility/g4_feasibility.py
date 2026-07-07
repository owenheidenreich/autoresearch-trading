"""G4 feasibility: what per-fold drawdown is achievable in this game at all?
Measures drawdown for random selection, an ORACLE (best actual candidate per
minute), and a cheap-only oracle, through the same serial simulator/folds.
Diagnostics-tier, read-only, no model."""
import numpy as np
from v4.scripts.run_protocol101_stage1_experiment import (
    FEE_PER_TRADE, extract_table, fold_boundaries, governed_sessions, replay, subset,
)

sessions = governed_sessions("train")
table = extract_table(sessions)
folds = fold_boundaries(table.session_name)
session_arr = np.asarray(table.session_name)
rng = np.random.default_rng(7)

def per_minute_pick(tbl, values, floor_dollars):
    order = {}
    for i, (s, dt) in enumerate(zip(tbl.session_name, tbl.decision_time)):
        dollar = values[i]
        if dollar > floor_dollars:
            key = (s, dt)
            if key not in order or values[i] > values[order[key]]:
                order[key] = i
    return np.asarray(sorted(order.values()), dtype=int)

print("fold | random_DD  randPnL | oracle_DD  oraclePnL | cheapOracle_DD  cheapPnL")
agg = {"random": [], "oracle": [], "cheap": []}
for fold in folds:
    test = subset(table, np.isin(session_arr, fold["test_sessions"]))
    # random: one random candidate per minute
    minutes = {}
    for i, (s, dt) in enumerate(zip(test.session_name, test.decision_time)):
        minutes.setdefault((s, dt), []).append(i)
    rand_sel = np.array(sorted(rng.choice(v) for v in minutes.values()))
    # oracle: best actual pnl per minute, if > fee
    oracle_sel = per_minute_pick(test, test.pnl, FEE_PER_TRADE)
    # cheap oracle: best actual RETURN per minute among premium<=$3 contracts
    ret = test.pnl / np.maximum(test.entry_ask * 100, 1e-6)
    cheap_mask = test.entry_ask <= 3.0
    ret_masked = np.where(cheap_mask, test.pnl, -1e9)  # only cheap eligible, rank by pnl
    cheap_sel = per_minute_pick(test, ret_masked, FEE_PER_TRADE)
    r = replay(test, rand_sel, FEE_PER_TRADE, "rand")
    o = replay(test, oracle_sel, FEE_PER_TRADE, "oracle")
    c = replay(test, cheap_sel, FEE_PER_TRADE, "cheap")
    agg["random"].append(r["max_drawdown"]); agg["oracle"].append(o["max_drawdown"]); agg["cheap"].append(c["max_drawdown"])
    print(f"{fold['fold']} | ${r['max_drawdown']:>7.0f} ${r['net_pnl']:>7.0f} | ${o['max_drawdown']:>7.0f} ${o['net_pnl']:>8.0f} | ${c['max_drawdown']:>7.0f} ${c['net_pnl']:>7.0f}")
print()
for k, v in agg.items():
    print(f"{k}: mean per-fold DD ${np.mean(v):.0f}, max ${np.max(v):.0f}")
