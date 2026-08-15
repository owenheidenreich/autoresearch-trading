# Supervised Pilot Policy Summary

Do not buy more data yet. The first supervised neural pilot does not clear the next-data-purchase gate.

| Policy | Neural trades | Neural PnL | Neural PF | Max DD | Random mean | ATM call | ATM put | VWAP/OMAR | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| policy0_stop35_target60_hold10m | 35 | -60 | 0.937 | -464 | -18490 | -36264 | -11149 | -26308 | False |
| policy1_stop50_target100_hold25m | 38 | -401 | 0.773 | -1146 | -13248 | -25455 | -3220 | -5752 | False |
| policy2_stop65_target150_hold45m | 13 | -541 | 0.041 | -564 | -11557 | -24682 | 4968 | -1465 | False |

## Next Recommended Work
- Diagnose why policy2 ATM puts work in March while the neural scorer misses them.
- Add side-conditional and regime-conditional baselines before training a larger model.
- Move from candidate-level independent scoring to a decision-level action model with no-trade/call/put outputs.
- Add stricter drawdown and per-day concentration gates before any more paid data.
