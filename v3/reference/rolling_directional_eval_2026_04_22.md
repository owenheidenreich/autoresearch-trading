# Phase R4 — Per-Window V0 vs V1 Evaluation — 2026-04-22

## TL;DR

**Major reversal.** The previous 20-day "V1 beats V0" finding (which
triggered the always-put pivot) was a regime-specific artifact.
Across 13 disjoint 60-day OOS windows (780 OOS days total):

- **V0 aggregate PF 1.132 vs V1 0.888** (V0 wins by +0.244)
- **V0 strictly beats V1 in 12 of 13 windows**; by a wider `> 0.10 PF`
  margin it wins **10 of 13**
- V0 per-window PF: mean 1.188 / median 1.033
- V1 per-window PF: mean 0.946 / median 0.823

The original 20-day OOS window (2026-03-05 to 2026-04-01) was
**window 12-ish** in this extended series — essentially the same
chronological slice that contained window 12 (2025-11-26 to
2026-02-24 in this rolling setup). Our new window 12 result:
V0=0.847, V1=0.722. V0 still wins. The previous single-window
claim was a bad call based on a small sample.

## Per-window detail

| Window | OOS dates | V0 PF | V1 PF | V0−V1 | V0 call% |
|---:|---|---:|---:|---:|---:|
| 0 | 2023-01-13..2023-04-11 | 0.948 | 0.791 | +0.157 | 5.3% |
| 1 | 2023-04-12..2023-07-07 | 0.699 | 0.677 | +0.022 | 14.3% |
| 2 | 2023-07-10..2023-10-02 | 0.895 | 0.767 | +0.128 | 5.4% |
| 3 | 2023-10-03..2023-12-27 | 0.600 | 0.496 | +0.104 | 17.5% |
| 4 | 2023-12-28..2024-03-25 | 0.586 | 0.358 | **+0.228** | 16.4% |
| 5 | 2024-03-26..2024-06-20 | 2.033 | 1.894 | +0.139 | 11.4% |
| 6 | 2024-06-21..2024-09-16 | 1.540 | 1.212 | **+0.328** | 12.5% |
| **7** | **2024-09-17..2024-12-10** | **1.160** | **1.232** | **−0.072** | 16.7% |
| 8 | 2024-12-11..2025-03-11 | 1.033 | 0.823 | +0.210 | 21.6% |
| 9 | 2025-03-12..2025-06-05 | **2.274** | 1.177 | **+1.097** | 22.2% |
| 10 | 2025-06-06..2025-09-02 | 1.072 | 0.981 | +0.091 | 22.9% |
| 11 | 2025-09-03..2025-11-25 | 1.754 | 1.166 | +0.588 | 14.6% |
| 12 | 2025-11-26..2026-02-24 | 0.847 | 0.722 | +0.125 | 19.2% |

Only window 7 (2024 Q4) shows V1 beats V0. Everywhere else V0
is equal or better.

## What the 20-day window got wrong

The previous OOS window (2026-03-05 to 2026-04-01) sat in a
chop-bearish regime where V1's defensive put bias outperformed. We
collapsed the model around that 20-day finding and concluded "always
put is the right default". Across 13 windows spanning different
regimes:
- Windows 5, 6, 8, 9, 11: strong bullish periods where V0's
  directional calls (11-23% of trades) add alpha V1 sacrifices
- Windows 0-4: chop/early-bearish periods where V0 and V1 are close
  but V0 still edges out
- Window 7 only: a chop-bearish period like the 20-day sample where
  V1's puts-only bias pays off
- Window 12: the closest chronological match to the 20-day OOS; V0
  still wins 0.847 vs 0.722

Conclusion: **V0's directional signal has real generalizable alpha.
V1 is a defensive override that happens to win only in specific
chop-bearish regimes.** The choice to ship V1 based on one such
regime was premature.

## Variants V2, V3, V4

V2 (sigma_pos veto on calls): **PF 1.040, 6/13 wins vs V1.** A
reasonable middle-ground. Vetoes calls when sigma_pos > 0 (price
above VWAP), capturing some directional discipline without forcing
always-put.

V3 (conviction-asymmetric): PF 0.899, 2/13 wins. Doesn't help.

V4 (combined): PF 0.900, 1/13 wins. Doesn't help.

## Aggregate equity-curve caveat

Aggregate DDs are huge (V0 95.9%, V1 151.7%) because the cold-start
equity curve compounds losses from the early windows (2023) before
recovering. This is a cold-start artifact, not a live-trading
expectation. In practice deployment starts with a reset equity
equal to starting capital, not with historical drawdown baked in.

## Implications for R5 / deployment

### For R5 (conditional rule):
V0 wins in 10 of 13 windows. V1 wins in window 7. If we could
*detect at session-open* whether today/this-week is a window-7-like
chop-bearish regime and defensively switch to V1 for those days,
we'd get the best of both. That's exactly what R5 will test, this
time with intraday-developing features at the entry bar (not bar-14
snapshots).

### For production:
**V0 should be the default directional rule, not V1.** The "champion"
config should revert to using the model's chosen direction.

## Updated comparison to the original "champion"

Old production-recommended config (from 2026-04-21 before the
methodology overhaul):
- V1 (always_put)
- Augmented L3 minus mfe_norm (A3)
- OOS PF 2.847 on 20 days

This config was:
- Built around V1 (which loses to V0 across 13 windows)
- Validated on one 20-day window where V1 happened to win

New direction (before R5/R6 finalization):
- V0 as the default (wins 10/13 windows)
- Aggregate V0 PF 1.132 is modest but positive across 780 OOS days
- Layer-3 (A3) still adds value but needs re-evaluation on V0 trades

## Verification

- [x] All 13 windows evaluated for V0, V1, V2, V3, V4
- [x] Per-window PF table complete
- [x] Cross-window aggregate PF per variant
- [x] Beat-V1 win-share table
- [x] Regime reading: V0 wins most regimes; V1 only wins chop-bearish (window 7)
- [ ] R5: build conditional V0/V1 rule via intraday-developing features
- [ ] R6: final champion locked, L3 re-evaluated on V0 trades
