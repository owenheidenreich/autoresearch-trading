# OMAR Findings — 2026-04-20

> Research trail: [decay_analysis_2026_04_20.md](decay_analysis_2026_04_20.md) → [attribution_full_2026-04-20.txt](attribution_full_2026-04-20.txt) → [stage1_research_findings_2026_04_20.md](stage1_research_findings_2026_04_20.md) → **[this doc, #4]**. See [INDEX.md](INDEX.md) for the full reading order.

## What OMAR is

**OMAR (Opening Minute Algo Run)** = the high and low of the first 1-minute
bar of the session (9:30-9:31 ET).

- `OMAR_high` = first-minute high
- `OMAR_low`  = first-minute low
- `OMAR_mid`  = (high + low) / 2
- `OMAR_range` = high − low

OMAR is **not** in Pickles' written journal by name, but it is a metric he is
known to use in practice: as a reference level that the market tends to
retest when overextended, and as a daily-calibrated unit of "a small move."

## Scripts that produced these findings

- `v3/analysis/omar_and_magic_time.py` — scale-metric investigation and MAGIC TIME concentration check
- `v3/analysis/omar_retest.py` — OMAR as a retest location, discriminative test vs random controls

## Finding 1 — OMAR as a daily-vol scale metric

Tested correlation with VIX regime on 1,016 days:

| Scale metric | Correlation with VIX | Day-to-day variation (p90/p10) |
|---|---|---|
| OMAR range | +0.385 | 4.4× |
| first15 range | +0.520 | 3.6× |
| ATR-15 | **+0.581** | **2.8×** |

**Verdict:** OMAR is the weakest of the three statistically. ATR-15 is the
best calibration metric because it averages 15 bars and is the least noisy.

**Policy:** use ATR-15 internally for statistical scaling wherever we need
a volatility-aware threshold; use OMAR externally as the trader-readable
display unit ("today's OMAR is 8 points — small day").

Typical OMAR magnitudes across the 1,016-day cache:
- median: 5.83 points
- p10: 3.07
- p90: 13.50

## Finding 2 — OMAR as a retest location (Pickles' actual use)

Abstention oracle bars (the 467 days where teachers missed the session-best
trade) cluster meaningfully near OMAR levels:

| Cohort | Median |distance| to nearest OMAR level (OMAR-range units) | % within 0.5× OMAR of any level |
|---|---|---|
| abstention | 0.74 | **43.7%** |
| entered_right | 1.54 | 24.8% |
| random control bars | 1.22 | 28.5% |

**Enrichment:** 43.7% (abstention) / 28.5% (random) = **1.53× more likely**
to be near OMAR on a missed oracle bar than on a random bar.

**Caveat:** abstention bars are where teachers missed opportunities, but
entered_right bars (teachers caught) are NOT near OMAR — they're clean
breakouts of the first15 range, which by construction moves price away from
OMAR. So "near OMAR" correlates with *the kind of setup teachers miss*, not
*every profitable setup*.

## Finding 3 — OMAR is a location signal, not a trigger

Forward-move test on random bars (control for causation):

| Bar type | n | Median next-20-bar SPX excursion |
|---|---|---|
| near OMAR (\|dist\| ≤ 0.5) | 118 | 0.152% |
| far from OMAR (\|dist\| ≥ 1.5) | 180 | 0.179% |

**Near-OMAR random bars have SMALLER forward moves, not bigger.** This
matters: it means OMAR proximity by itself is not a directional trigger.
OMAR is a *place* where big-move setups tend to occur *when other signals
align* — a confluence filter, not a standalone signal.

This matches Pickles' framework: OMAR is one input in a confluence, never
the sole reason to trade.

## The three distinct uses of OMAR in the system

| # | Use | How it's applied |
|---|---|---|
| 1 | **Scale anchor** (display unit) | "Tight squeeze today" = pre-entry 10-bar range ≤ 1× OMAR |
| 2 | **Retest zone** (confluence filter) | Price within 0.5-1× OMAR of OMAR H/L/M increases probability of a setup |
| 3 | **Daily-character read** (gut calibration) | OMAR ≤ 4 pts = quiet day; OMAR ≥ 10 pts = wild day |

## Proposed late-session teacher rule (v1 spec)

A new teacher playbook, stateless over `BarContext` augmented with OMAR:

**Name:** `NarrowRangeBreakout` (NR10 / post-MAGIC-TIME teacher)

**Hard preconditions (all must be true):**
- bar time in [40, 120] (post-MAGIC-TIME through pre-lunch, i.e., 10:10-11:30 ET)
- close is INSIDE first15 range
- pre-entry 10-bar SPX range ≤ 1.0× OMAR
- |distance from close to nearest OMAR level| ≤ 1.0× OMAR range

**Trigger:**
- current bar's close breaks the high of the last 10 bars → `BUY_CALL`
- current bar's close breaks the low of the last 10 bars → `BUY_PUT`

**Expected capture (from the abstention cohort):**
- Recall: ~44% of abstention oracle bars qualify for this rule
- Precision: ~29% of random non-oracle bars would also fire (false alarm baseline)
- Ratio: 1.5× — modest but meaningful edge

**CORRECTION (added 2026-04-20 during time-of-day analysis):** the 44%/29%
numbers are for abstention oracle bars specifically. On the FULL population
of NR10 triggers (13,104 across 986 days, ~13 per day), adding the
OMAR-retest filter does NOT improve forward-move quality. See
`v3/analysis/time_of_day_cap.py` output: NR10 near-OMAR triggers have
equal-or-slightly-worse mfe_median than NR10 triggers far from OMAR.

The NR10 trigger raw edge is weak: ~5-8 bps median MFE, 66% fake-out rate.
OMAR enrichment on abstention oracle bars is a real property of that cohort,
but it does NOT translate to filter-level edge on the overall trigger
population. This is consistent with OMAR being "a confluence filter, not a
standalone trigger" — confluence needs MORE signals, not just OMAR.

**Therefore: do NOT implement the NarrowRangeBreakout teacher with just
NR10 + OMAR. The teacher needs additional filters (VWAP bands, S/R
confluence, tighter squeeze threshold) before it has real edge.** Research
on those is in progress.

**What this rule does NOT have:**
- Volume confirmation (SPX has no direct volume; VWAP/volume work is deferred)
- S/R confluence (Pickles' multi-input level module is a separate workstream)
- Direction disambiguation beyond "follow the break" (same side-error risk as ORC)

**What still needs deciding before implementation:**
- **Aggressiveness dial:** the 1.0× OMAR squeeze and 1.0× OMAR retest zone
  are the "capture more" settings. Tightening to 0.65× would reduce false
  alarms but miss more abstentions.
- **Breakout confirmation:** does the current bar need to move ≥ N× the
  prior 10-bar range to count as a "real" breakout, or is any new high/low
  enough?

## BarContext fields this teacher will need

The existing `BarContext` does NOT carry OMAR levels. If we implement this
teacher, the v2 adapter (or a dedicated preprocessing step) must attach:

- `omar_high`, `omar_low`, `omar_mid`, `omar_range` — constants per day
- `last10_high`, `last10_low` — rolling 10-bar close max/min (per bar)
- `pre_entry_10bar_range` — derived (high − low over last 10 bars)

These are trivially computable from the SPX 1-min series already in cache.

## Open questions for future research

1. **Does OMAR-retest-enrichment hold in different regimes?** The 43.7%
   number is full-cache aggregate. Does it degrade on high-VIX days?
   High-chop days?

2. **Does the near-OMAR filter help ORC too?** ORC's entered_right cohort
   is NOT typically near OMAR (24.8%). But what about ORC's side-error
   cohort? If side-error ORC fires happen more often away from OMAR, then
   "OMAR proximity" could help ORC's directional inference too. This is
   a thread to follow after the side-error deep-dive.

3. **How does OMAR behave around macro events?** FOMC days, CPI prints.
   Pickles warns MAGIC TIME may not happen when big 4 indexes diverge.
   Is OMAR similarly disrupted, or does it still retest?
