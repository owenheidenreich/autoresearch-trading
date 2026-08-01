"""Path-D exit backtest — SMOKE TEST (Tier-S research, quarantined/throwaway).

Trains a GBT to predict the non-circular a_hold target (advantage vs frozen hold-to-flat)
with leave-one-DAY-out OOF, then serial-replays the learned 1-second exit vs transparent
baselines under a conservative fill model. Machinery validation + directional hint ONLY.
NOT a feasibility result: 6 days, deterministic entry fixture, no frozen-entry OOF.

Labels: walking_skeleton | throwaway | tier-s | research. No promotion/paper/real-money.
"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd

OUT = "v4/audit/autoresearch/protocol101_pathd_exit_research"
LAB = f"{OUT}/pathd_exit_labels_smoke.parquet"
TICK = 0.10
ROUNDTRIP_FEE = 3.0          # $/contract, per the frozen $3 reserve
MULT = 100                   # $/share -> $/contract
FEATURES = ["bid","ask","mid","spread","pnl","mfe_to_now","giveback_from_peak","mins_held","mins_to_close"]
POS_KEYS = ["day","entry_et","symbol"]

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_SK = True
except Exception as e:
    HAVE_SK = False; SK_ERR = str(e)

def trade_pnl(entry_fill, exit_bid):
    return (exit_bid - TICK - entry_fill) * MULT - ROUNDTRIP_FEE   # sell at bid-tick, buy at entry_fill

def replay_policy(pos_df, exit_mask):
    """exit at first second where exit_mask True; else hold-to-flat (last row)."""
    g = pos_df.sort_values("ts")
    idx = np.where(exit_mask.values)[0]
    row = g.iloc[idx[0]] if len(idx) else g.iloc[-1]
    return trade_pnl(g["entry_fill"].iloc[0], row["bid"])

def main():
    if not HAVE_SK:
        print("sklearn unavailable:", SK_ERR); return
    X = pd.read_parquet(LAB)
    X = X.dropna(subset=FEATURES+["a_hold","bid"]).copy()
    days = sorted(X["day"].unique())

    # leave-one-day-out OOF prediction of a_hold
    X["pred_a_hold"] = np.nan
    for d in days:
        tr = X[X["day"]!=d]; te = X[X["day"]==d]
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                          max_depth=4, random_state=0)
        m.fit(tr[FEATURES], tr["a_hold"])
        X.loc[X["day"]==d, "pred_a_hold"] = m.predict(te[FEATURES])
    # OOF fit quality
    ss_res = float(((X["a_hold"]-X["pred_a_hold"])**2).sum())
    ss_tot = float(((X["a_hold"]-X["a_hold"].mean())**2).sum())
    oof_r2 = 1 - ss_res/ss_tot

    # per-position serial replay: learned exit vs baselines
    res = {p:[] for p in ["learned","hold_to_flat","exit_immediate","trail_stop_50"]}
    for _, g in X.groupby(POS_KEYS):
        g = g.sort_values("ts")
        res["learned"].append(replay_policy(g, g["pred_a_hold"] < 0.0))
        res["hold_to_flat"].append(replay_policy(g, pd.Series(False, index=g.index)))
        res["exit_immediate"].append(replay_policy(g, pd.Series([True]+[False]*(len(g)-1), index=g.index)))
        res["trail_stop_50"].append(replay_policy(g, g["giveback_from_peak"] >= 0.50))  # $50/contract giveback

    def stats(v):
        v = np.array(v)
        return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)),
                    total=float(v.sum()), win_rate=float((v>0).mean()),
                    p10=float(np.percentile(v,10)), p90=float(np.percentile(v,90)))
    summary = {k: stats(v) for k,v in res.items()}

    # threshold sensitivity for the learned policy
    thr = {}
    for t in (-0.50,-0.25,0.0,0.25,0.50):
        v=[]
        for _, g in X.groupby(POS_KEYS):
            g=g.sort_values("ts"); v.append(replay_policy(g, g["pred_a_hold"] < t))
        thr[f"theta={t}"] = round(float(np.mean(v)),2)

    out = dict(
        labels="walking_skeleton|throwaway|tier-s|research",
        days=days, positions=int(X[POS_KEYS].drop_duplicates().shape[0]),
        oof_a_hold_r2=round(oof_r2,4),
        per_policy_mean_pnl_per_contract={k:round(s["mean"],2) for k,s in summary.items()},
        summary=summary,
        learned_threshold_sensitivity=thr,
        note=("Machinery validation + directional hint ONLY. 6 days, deterministic entry "
              "fixture, leave-one-day-out OOF (not frozen-entry OOF), hold-to-flat reference. "
              "NOT a feasibility result and cannot justify promotion/paper/real-money."),
    )
    json.dump(out, open(f"{OUT}/exit_backtest_smoke.json","w"), indent=1)
    print("=== OOF a_hold R^2 (leave-one-day-out):", round(oof_r2,4))
    print("\n=== mean realized PnL per contract by policy ===")
    for k,s in summary.items():
        print(f"  {k:16} mean=${s['mean']:8.2f}  median=${s['median']:8.2f}  win%={s['win_rate']:.2f}  total=${s['total']:9.2f}")
    print("\n=== learned-policy threshold sensitivity (mean $/contract) ===")
    for k,v in thr.items(): print(f"  {k:12} {v}")
    print(f"\nwrote {OUT}/exit_backtest_smoke.json")

if __name__ == "__main__":
    main()
