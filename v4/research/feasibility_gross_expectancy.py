"""GROSS-EXPECTANCY FEASIBILITY STUDY — which instrument/horizon is even reachable?

The question is NOT "does this instrument have positive drift". Nothing reliably
does; if it did, no model would be needed. The question is:

    HOW BIG DOES A DIRECTIONAL EDGE HAVE TO BE TO CLEAR THE FRICTION HURDLE?

For a symmetric move of size M with round-trip friction F, a trader right with
probability p earns  EV = (2p-1)*M - F.  Breakeven is therefore

    required_win_rate = 0.50 + F / (2*M)          [the "hurdle"]

That converts friction into a single comparable number across instruments: the
accuracy you must achieve before you make a cent. A required win rate near 50% is
reachable; one near 75% is not.

Instruments measured from the EXISTING corpus (no paid download):
  - SPXW 0DTE options  (measured quotes: CBBO 1s trajectories)
  - ES futures         (GLBX 1-minute OHLCV, 261 sessions)
  - VX futures         (XCBF 1-minute OHLCV, 88 sessions)
  - SPX index          (reference only -- not directly tradable)

Read-only. No training, no broker, no paid data.
"""
from __future__ import annotations

import glob
import json
import numpy as np
import pandas as pd

HORIZONS_MIN = [1, 5, 15, 30, 60]
OUT = "/private/tmp/claude-501/-Users-gduby-Documents-autoresearch-trading/1a05beac-bbff-4be6-a1b4-b972c630c2a1/scratchpad/feasibility_result"

# Friction, $ per contract, round trip.
#   Options: MEASURED on 156,950 candidates (aggressive $26.48; passive-realistic $18.56).
#   Futures: ASSUMED from published tick structure -- OHLCV carries no quotes.
#            ES = $50/pt, tick 0.25 ($12.50), ~always 1 tick wide -> $12.50 crossing + ~$4.50 comms.
#            VX = $1000/pt, tick 0.05 ($50.00), typically 1 tick wide -> $50 crossing + ~$5 comms.
FRICTION = {
    "SPXW_0DTE_option (aggressive)": 26.48,
    "SPXW_0DTE_option (passive)": 18.56,
    "ES_future": 17.00,
    "VX_future": 55.00,
}
MULT = {"ES_future": 50.0, "VX_future": 1000.0}
ASSUMED = {"ES_future", "VX_future"}

rows = []


def futures_moves(name: str, folder: str, mult: float) -> None:
    files = sorted(
        glob.glob(
            f"/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31/raw/databento/{folder}/**/*.parquet",
            recursive=True,
        )
    )
    per_h: dict[int, list[float]] = {h: [] for h in HORIZONS_MIN}
    drift: dict[int, list[float]] = {h: [] for h in HORIZONS_MIN}
    for path in files:
        d = pd.read_parquet(path, columns=["close"])
        c = d["close"].to_numpy(float)
        for h in HORIZONS_MIN:
            if len(c) <= h:
                continue
            delta = (c[h:] - c[:-h]) * mult
            delta = delta[np.isfinite(delta)]
            per_h[h].append(np.abs(delta))
            drift[h].append(delta)
    for h in HORIZONS_MIN:
        if not per_h[h]:
            continue
        a = np.concatenate(per_h[h])
        s = np.concatenate(drift[h])
        rows.append(
            {
                "instrument": name,
                "horizon_min": h,
                "n": int(a.size),
                "median_abs_move": float(np.median(a)),
                "mean_abs_move": float(a.mean()),
                "gross_drift": float(s.mean()),
                "friction": FRICTION[name],
                "assumed_friction": name in ASSUMED,
            }
        )


futures_moves("ES_future", "glbx_es_ohlcv_1m", MULT["ES_future"])
futures_moves("VX_future", "xcbf_vx_ohlcv_1m", MULT["VX_future"])

# ---- SPXW 0DTE options: measured from the 1-second trajectories -------------
outcomes = pd.read_parquet(
    "/Volumes/AR_TRADING_DATA/reports/phase1_four_box/trajectory_outcomes.parquet"
)
opt: dict[int, list[float]] = {h: [] for h in HORIZONS_MIN}
for r in outcomes.to_dict("records"):
    p = (
        f"/Volumes/AR_TRADING_DATA/exit_features/session={r['session']}"
        f"/{r['trajectory_id']}.parquet"
    )
    try:
        f = pd.read_parquet(p, columns=["option_bid"])
    except Exception:
        continue
    b = f["option_bid"].to_numpy(float) * 100.0  # dollars per contract
    for h in HORIZONS_MIN:
        step = h * 60  # trajectory rows are 1 second
        if len(b) <= step:
            continue
        delta = b[step:] - b[:-step]
        opt[h].append(delta[np.isfinite(delta)])

for h in HORIZONS_MIN:
    if not opt[h]:
        continue
    s = np.concatenate(opt[h])
    for label in ("SPXW_0DTE_option (aggressive)", "SPXW_0DTE_option (passive)"):
        rows.append(
            {
                "instrument": label,
                "horizon_min": h,
                "n": int(s.size),
                "median_abs_move": float(np.median(np.abs(s))),
                "mean_abs_move": float(np.abs(s).mean()),
                "gross_drift": float(s.mean()),
                "friction": FRICTION[label],
                "assumed_friction": False,
            }
        )

df = pd.DataFrame(rows)
df["hurdle_ratio"] = df["friction"] / df["median_abs_move"]
df["required_win_rate_pct"] = 100.0 * (0.5 + df["friction"] / (2.0 * df["median_abs_move"]))

lines = []
lines.append("=" * 104)
lines.append("GROSS-EXPECTANCY FEASIBILITY — required win rate to clear friction")
lines.append("=" * 104)
lines.append("required_win% = 50 + friction / (2 x median |move| over the horizon).")
lines.append("<=55% reachable   55-60% hard   60-70% implausible   >70% dead.  (*) friction ASSUMED.")
lines.append("")
lines.append(
    f"{'instrument':32s} {'horiz':>6s} {'med|move|$':>11s} {'drift$':>9s} "
    f"{'friction$':>10s} {'hurdle':>8s} {'req.win%':>9s}"
)
lines.append("-" * 104)
for inst in [
    "ES_future",
    "VX_future",
    "SPXW_0DTE_option (passive)",
    "SPXW_0DTE_option (aggressive)",
]:
    sub = df[df["instrument"] == inst].sort_values("horizon_min")
    for _, r in sub.iterrows():
        star = "*" if r["assumed_friction"] else " "
        lines.append(
            f"{inst:32s} {int(r['horizon_min']):5d}m {r['median_abs_move']:11,.2f} "
            f"{r['gross_drift']:9,.2f} {r['friction']:9,.2f}{star} "
            f"{r['hurdle_ratio']:8.3f} {r['required_win_rate_pct']:8.1f}%"
        )
    lines.append("")

text = "\n".join(lines)
print(text, flush=True)
df.to_csv(OUT + ".csv", index=False)
with open(OUT + ".txt", "w") as fh:
    fh.write(text)
with open(OUT + ".json", "w") as fh:
    json.dump(df.to_dict("records"), fh, indent=2)
print(f"wrote {OUT}.{{txt,csv,json}}", flush=True)
