---
date: 2026-04-26
parent: h3f_regret_falsified_2026_04_26.md
status: HYPOTHESIS-GENERATING — qualitative analysis of 1664 OOS trades reveals L2 systematically picks wrong side in specific (sigma_pos × iv) cells; L3 has been fighting upstream errors; next experiment should be at L2 with cell-conditional side features
---

# Phase C: Qualitative Analysis of Failure Modes

## Setup

After H3a / H3e / sideblind / C-Pre / L2 Phase 1 / TCN / H3f all
showed L3 oracle interventions can't break the regime spread, user
proposed Phase C: pull the worst-loss trades and look at them bar by bar
to find patterns the human eye sees that the model misses.

Method: from the existing 1664-trade 5-seed OOS sample, stratify by
sigma_pos × iv_percentile cells, identify worst-N and best-N trades per
cell, compare entry features, side bias, time-of-day, predicted scores.

Output: `v3/artifacts/research/phase_c_trade_review.csv` (full 1664-trade
table for offline review).

## Key finding: L2 systematically picks the WRONG SIDE in specific cells

Per-cell side × PF table (5 seeds, n_total per cell ranges 153-225):

```
cell    call_n  call_PF   put_n   put_PF  | preferred side
------  ------  -------   -----   ------  | --------------
s0_iv0    68     1.46      58     3.12    | PUTS
s0_iv1   103     1.67     104     0.85    | calls
s0_iv2    69     0.68     153     1.75    | PUTS (L2 enters 69 calls anyway → losers)
s1_iv0   115     1.50      90     2.02    | puts
s1_iv1    97     1.37      72     2.52    | PUTS
s1_iv2    52     1.59     128     4.12    | PUTS
s2_iv0   115     3.11     109     0.94    | CALLS (L2 enters 109 puts anyway → losers)
s2_iv1    62     2.78     116     0.97    | CALLS (L2 enters 116 puts anyway → losers)
s2_iv2    43     4.19     110     2.15    | calls
```

**Side preference flips with sigma_pos in a trader-coherent way:**
- s2 (price extreme-high vs VWAP): calls win (momentum continues)
- s1 (price near VWAP): puts always win (mean reversion to/from above?)
- s0 (price extreme-low vs VWAP): mixed — puts in extreme IV cells,
  but calls in the mid-IV cell (s0_iv1) — possibly bounce dynamic

L2 does NOT cleanly track this pattern. It enters:
- 69 calls in s0_iv2 where calls average PF 0.68 (loser)
- 109 puts in s2_iv0 where puts average PF 0.94 (loser)
- 116 puts in s2_iv1 where puts average PF 0.97 (loser)
- 104 puts in s0_iv1 where puts average PF 0.85 (loser)

**Hundreds of trades on the systematically wrong side.** L3 exit policy
can't rescue trades that should not have been entered as that side.

## Specific patterns in chop × low (s0_iv1, the persistent floor cell)

Worst 10 vs Best 10 in s0_iv1, mean comparison:

```
                          best     worst    diff
pred_dollar_score        -1.01    -0.16    -0.85   ← model is more pessimistic on best!
pred_return_score        -0.22    +0.11    -0.34
pred_win_prob            +0.47    +0.48    -0.01
decision_margin          +0.79    +0.46    +0.34
first15_range_pct        +0.0041  +0.0022  +0.0019  ← BIG morning range = winners
last10_range_over_omar   +0.91    +2.01    -1.11    ← extended last10 = losers
```

**The model is anti-calibrated in this cell.** When pred_dollar_score is
most negative (model expects losses), trades realize $3000+ winners.
When pred_dollar_score is mildly negative (model is neutral), trades
crash to -$900. This confirms the L2 Phase 1 finding (anti-calibration
in floor cell).

**Morning range matters:** big first15_range_pct (~0.4%, "active morning")
correlates with winners. Small first15_range_pct (~0.2%, "quiet morning")
correlates with losers. Same cell. The model has the feature but isn't
using it to differentiate well.

**Last-10-range over OMAR matters inversely:** when last10 range is
extended (>2x OMAR range), trades fail. When last10 is contained (<1x
OMAR), trades win. Suggests "trades entered during late-session
expansion" fail more than "trades entered during late-session
contraction."

## Time-of-day pattern

```
All trades:  bar_index median 94 (≈10:33 ET), q25-q75 [60, 110]
Worst 100:   bar_index median 74 (≈10:13 ET), q25-q75 [53, 102]
```

**Worst trades cluster earlier in the session by ~20 bars.** Early-session
entries are more failure-prone. Possible mechanisms:
- Catching opening fakes that don't follow through
- More time for theta+choppy-action to drag the position
- L2's signal is weaker before the morning structure establishes

## Reframing Phase B (domain features)

The user's original plan was A → C → B with B as "GEX, charm flows,
gamma flip." Phase C reveals the missing signal isn't GEX/charm — it's
**cell-conditional side preference and time/range interactions L2 isn't
extracting.**

Specifically, the next experiment should be at **L2, not L3**:

1. **Side-decision features that encode (sigma_pos × iv_percentile)
   cell membership.** L2 needs to learn the cell-conditional side
   preferences shown in the table above. Approach: add explicit
   interaction features (one-hot cell labels, or computed "expected
   side preference for this cell" from historical data — strictly
   train-time, not future) to L2's input.

2. **Morning-range × cell interaction.** `first15_range_pct` has signal
   in floor cells. Either explicit interaction feature or a non-linear
   model that picks it up.

3. **Time-of-day × cell awareness.** Early-session entries in floor
   cells are risky.

These are NOT regime gating (user-rejected). They're **regime-aware
features** that the SAME single L2 model uses. The model takes regime
as input; it doesn't switch between models.

## Cumulative picture

Six L3 experiments hit the representation ceiling. L2 audit + Phase C
now show the binding constraint: **L2 picks the wrong side in specific
cells.** L3 has been trying to improve the wrong trades.

Path forward: an L2 retraining with cell-conditional features. Cost:
~6-8 GPU-hours per 3-seed promotion run on H100 (~$200-300 Akash). Big
investment, but the diagnostic is now sharp enough to justify it.

## Cost

- ~30 min dev (Phase C analysis script)
- ~5 sec compute (analysis only, no training)
- $0 GPU
- Total: ~30 min. Cheap and high-value qualitative work.

## Files

```
scripts/research_phase_c_failure_modes.py
v3/artifacts/research/phase_c_trade_review.csv  (full 1664 trades for offline review)
v3/reference/phase_c_failure_modes_2026_04_26.md  (this doc)
```

## What's next (re-prioritized)

Original plan: A → C → B (GEX/charm features).

Revised: **A done, C done, B (the original generic features) DEFERRED.
The newly-identified next experiment is L2 retraining with cell-
conditional features.** Call it L2-Redesign-Phase-2.

Awaiting user buy-in on:
1. The path forward (L2 retraining vs deeper Phase C analysis vs accept ceiling)
2. GPU spend authorization (~$200-300 for the 3-seed promotion run)
