"""Horizon vs the Charter's big-loss discipline.

The one conclusion that survived every attack was "hold longer": friction is
roughly fixed per round trip while the move being chased grows with horizon. But
holding longer also lets each loser run further, and the SIGNED charter caps the
big-loss bucket at 2% -- "the killer discipline is the bottom row staying under
2%".

So there are two opposing constraints:
  short holds -> friction dominates
  long holds  -> the loss distribution blows out

This sweeps holds from 5 to 240 minutes and asks whether ANY horizon satisfies
both. If none does, the class closes on charter grounds, independently of every
edge argument -- and with no new data.

Returns are on premium paid, per the charter. Read-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v4.research.pathd_model_gate import SCRATCH_BAND, BIG_LOSS_RETURN, MAX_BIG_LOSS_SHARE

HOLDS_MIN = [5, 15, 30, 60, 90, 120, 180, 240]
FEE_ROUND_TRIP = 3.00
MULT = 100.0

outcomes = pd.read_parquet(
    "/Volumes/AR_TRADING_DATA/reports/phase1_four_box/trajectory_outcomes.parquet"
)

paths: list[tuple[np.ndarray, float]] = []
for row in outcomes.to_dict("records"):
    p = (
        f"/Volumes/AR_TRADING_DATA/exit_features/session={row['session']}"
        f"/{row['trajectory_id']}.parquet"
    )
    try:
        f = pd.read_parquet(p, columns=["option_bid", "entry_fill_option_price"])
    except Exception:
        continue
    bid = f["option_bid"].to_numpy(float)
    fill = float(f["entry_fill_option_price"].iloc[0])
    if bid.size < 2 or not np.isfinite(fill) or fill <= 0:
        continue
    paths.append((bid, fill))

print(f"trajectories loaded: {len(paths)}\n")

rows = []
for hold in HOLDS_MIN:
    step = hold * 60
    rets, nets = [], []
    for bid, fill in paths:
        if bid.size <= step:
            continue  # require a genuine full-length hold, no forced-flat truncation
        exit_bid = bid[step]
        if not np.isfinite(exit_bid):
            continue
        premium = fill * MULT
        net = (exit_bid - fill) * MULT - FEE_ROUND_TRIP
        rets.append(net / premium)
        nets.append(net)
    if len(rets) < 50:
        rows.append({"hold_min": hold, "n": len(rets)})
        continue
    r = np.asarray(rets)
    n = np.asarray(nets)
    rows.append({
        "hold_min": hold,
        "n": len(r),
        "mean_net_$": float(n.mean()),
        "big_win_%": float(100 * (r > SCRATCH_BAND).mean()),
        "scratch_%": float(100 * ((r >= -SCRATCH_BAND) & (r <= SCRATCH_BAND)).mean()),
        "big_loss_%": float(100 * (r < BIG_LOSS_RETURN).mean()),
        "charter_ok": bool((r < BIG_LOSS_RETURN).mean() <= MAX_BIG_LOSS_SHARE),
    })

frame = pd.DataFrame(rows)
print("=" * 92)
print("HORIZON vs CHARTER BIG-LOSS DISCIPLINE   (charter limit: big_loss <= 2.0%)")
print("=" * 92)
print(frame.to_string(index=False))
print()

ok = frame[frame.get("charter_ok", False) == True]  # noqa: E712
if ok.empty:
    print("RESULT: NO horizon from 5 to 240 minutes satisfies the charter's 2% big-loss limit.")
    print("        The class fails on charter grounds independently of any edge argument.")
else:
    print("RESULT: charter-satisfying horizons exist:")
    print(ok.to_string(index=False))
