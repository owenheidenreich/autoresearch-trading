# v3 Research Reference Index

Everything in `v3/reference/` is an **artifact of evidence-based decisions**
made during the v3 Stage 1 research phase. Read in the order below to
follow the full research trail.

The master plan these findings serve is
`~/.claude/plans/new-clean-context-for-spicy-kay.md`.

## Findings docs (quick index)

1. [decay_analysis_2026_04_20.md](decay_analysis_2026_04_20.md) — Layer 0 rails (LOCKED)
2. [attribution_full_2026-04-20.txt](attribution_full_2026-04-20.txt) — 986-day pipeline baseline
3. [stage1_research_findings_2026_04_20.md](stage1_research_findings_2026_04_20.md) — four sharp findings
4. [omar_findings_2026_04_20.md](omar_findings_2026_04_20.md) — OMAR retest + MAGIC TIME
5. [time_of_day_cap_2026_04_20.md](time_of_day_cap_2026_04_20.md) — `[40, 120]` window LOCKED
6. [vwap_bands_2026_04_20.md](vwap_bands_2026_04_20.md) — VWAP direction rule (validated)
7. [sr_confluence_2026_04_20.md](sr_confluence_2026_04_20.md) — basic S/R (NULL)
8. [volume_profile_2026_04_20.md](volume_profile_2026_04_20.md) — intraday VP (cohort-only)
9. [side_error_deep_dive_2026_04_20.md](side_error_deep_dive_2026_04_20.md) — VWAP σ-position (**A1 source**)
10. [combined_confluence_2026_04_20.md](combined_confluence_2026_04_20.md) — B+C stack (cohort-only)
11. [hvn_lvn_2026_04_20.md](hvn_lvn_2026_04_20.md) — HVN/LVN (NULL)

## Reading order (logical, not just chronological)

### 1. [decay_analysis_2026_04_20.md](decay_analysis_2026_04_20.md)

**Question:** How wide should the Layer 0 hard rails be for SPX 0DTE at a
$25k account framing?

**Answer:** Moderate tier — premium cap $1200 (4.8%), daily brake $1500 (6%),
2 full-premium losers. Calibrated from 5,194 0DTE contract-price paths
stratified by entry premium tier and delta tier. Rejects fixed stop-loss
fractions in Layer 0 because 5-9% of -60%-drawdown contracts recover.

**Script:** `v3/analysis/decay_analysis.py`

**Downstream impact:** these rail numbers are now locked in `v3/config.py`
and used by every subsequent run.

---

### 2. [attribution_full_2026-04-20.txt](attribution_full_2026-04-20.txt)

**Question:** How does the current Stage 1 policy (ORC + FailedBreak teachers)
actually perform against the opportunity oracle across the full 986-day
SPX cache?

**Answer:** raw dump. Contains:
- Feasibility diagnostic table (all four metrics + severity)
- Attribution outcome counts (guardrail_suppression / abstention / side_error / entered_right)
- Selection-quality stats across 44,877 teacher-entered bars
- Per-session JSON (one line per session with outcome + gap fields)

**Script:** `v3/analysis/run_full_attribution.py`

**Downstream impact:** the per-session JSON is read by the deep-dive
analyses (#3 and #4 below).

---

### 3. [stage1_research_findings_2026_04_20.md](stage1_research_findings_2026_04_20.md)

**Question:** What does #2's data mean, and what should we change?

**Answer:** four sharp findings:
1. Fixed -35/+60 stops COST $120k vs naive hold — confirms #1's prediction
2. Contract selection is usually correct, but the attribution's $0 was a
   biased artifact — honest stats from `selection_quality.py` show median $0,
   mean $6.69 gap, 14% of entries positive. Second-order priority.
3. Abstention bars are late-session, inside the opening range, with tight
   pre-entry consolidation — a distinct pattern teachers don't address
4. Side-error bars DO trigger ORC but in the wrong direction — ORC's
   "break + VWAP agrees" is insufficient for directional inference

**Scripts:** `v3/analysis/miss_analysis.py`, `v3/analysis/abstention_deep_dive.py`

**Downstream impact:** sets the teacher-rework priorities. #4 below addresses
the abstention priority; side-error deep-dive is still pending.

---

### 4. [omar_findings_2026_04_20.md](omar_findings_2026_04_20.md)

**Question:** Can OMAR (opening-minute range) and MAGIC TIME (09:55-10:10)
be used to build a late-session teacher that captures the abstention bars?

**Answer:** yes, with an important correction (added after docs #5/#6):
1. Display scale — trader-readable daily volatility unit
2. Retest confluence — 1.53× enrichment on abstention ORACLE bars
   (43.7% within 0.5× OMAR of a level, vs 28.5% random)
3. NOT a directional trigger — near-OMAR random bars have SMALLER forward moves
4. **Correction:** OMAR enrichment is a property of the abstention oracle
   cohort, but filtering the full NR10 trigger population by OMAR proximity
   does NOT improve forward-move edge (see doc #5).

MAGIC TIME confirmed: 61% of `entered_right` bars in [09:55, 10:10].

**Scripts:** `v3/analysis/omar_and_magic_time.py`, `v3/analysis/omar_retest.py`

**Downstream impact:** OMAR stays in the teacher's spec as a non-gating
input and display unit; it does not provide trigger-level edge.

---

### 5. [time_of_day_cap_2026_04_20.md](time_of_day_cap_2026_04_20.md)

**Question:** Is [40, 120] the right time window for the late-session teacher,
and does the OMAR filter actually improve the NR10 trigger's forward-move
quality?

**Answer:**
- Time cap `[40, 120]` (10:10-11:30 ET) LOCKED. Signal quality degrades
  gradually after 12:00 ET (p75 MFE drops ~37% from morning).
- NR10 raw trigger fires ~13×/day with 5-8 bps median MFE and 66% fake-out.
  WEAK edge alone — needs additional filters.
- OMAR-proximity filter applied to the full NR10 population does NOT improve
  forward moves. The OMAR enrichment on abstention oracle bars is a
  cohort-level property, not a trigger-level filter.

**Script:** `v3/analysis/time_of_day_cap.py`

**Downstream impact:** time cap locked at 11:30 ET; OMAR demoted from
"trigger filter" to "display / context." Teacher NOT shippable without
stronger additional filters.

---

### 6. [vwap_bands_2026_04_20.md](vwap_bands_2026_04_20.md)

**Question:** Does Pickles' VWAP direction rule ("long at/below VWAP, put
at/above VWAP, never long at +2σ") hold empirically on SPX 0DTE?

**Answer:** YES (direction validated) but modestly.
- Oracle CALL bars: median SPX position −0.21σ (below VWAP).
- Oracle PUT bars: median SPX position +0.17σ (above VWAP).
- Control random: median −0.04σ (at VWAP).
- 96.6% of entered_right calls are below +1σ. 96.6% of entered_right puts
  are above −1σ. ±2σ extremes are rare for all cohorts (<5%).

**Proposed teacher filter:**
- BUY_CALL requires: SPX close ≤ VWAP + 0.5σ
- BUY_PUT  requires: SPX close ≥ VWAP − 0.5σ

**Script:** `v3/analysis/vwap_bands.py`

**Downstream impact:** VWAP direction filter ready to add to the teacher.
Modest edge, not transformative. Pickles' "first-touch vs second-test" rule
not yet tested; deferred.

---

### 7. [sr_confluence_2026_04_20.md](sr_confluence_2026_04_20.md)

**Question:** Do abstention oracle bars cluster near Pickles-style
computable S/R levels (prior-day H/L/C/M, weekly pivots, Fibonacci, IB H/L,
round numbers)?

**Answer:** NO for abstention. 66.2% of abstention oracle bars are within
0.5× OMAR of any such level, vs 67.1% of random controls — 0.99×
enrichment (none). Per-level-type breakdown shows no individual family
provides edge either.

**Secondary finding:** entered_right bars DO show modest enrichment (1.13× /
1.50× confluence ratio). S/R confluence may help ORC's direction inference
— tested in the pending side-error deep-dive.

**Script:** `v3/analysis/sr_confluence.py`

**Downstream impact:** do NOT add computable-S/R confluence to the late-
session teacher. Not relevant for abstention capture.

---

### 8. [volume_profile_2026_04_20.md](volume_profile_2026_04_20.md)

**Question:** Pickles' strongest S/R inputs are Volume Profile levels
(POC, VAH, VAL). Do oracle bars cluster near them?

**Answer:** PRIOR-DAY VP no. INTRADAY DEVELOPING VP YES — 1.32× enrichment
on abstention oracle bars (67.6% vs 51.3% control). Positioned as "strongest
filter" at the time of this doc — later revised in doc #10 to "real but
marginal when stacked on VWAP+OMAR."

- Abstention cluster NEAR intraday VP (late breakouts start inside the
  developing value area).
- Entered_right bars are AWAY from intraday VP (0.80× — clean ORC
  breakouts happen outside the congestion zone).
- Prior-day VP is saturated (78% of all bars near it) — no discrimination.

**Script:** `v3/analysis/volume_profile.py`

**Downstream impact:** VP is a real cohort-level enrichment signal but
largely overlaps with OMAR in the stacked-filter test. Deprioritized in
the final teacher spec.

---

### 9. [side_error_deep_dive_2026_04_20.md](side_error_deep_dive_2026_04_20.md)

**Question:** 402 bars where ORC fired the wrong direction. What
distinguishes ORC's true breakouts from its false breakouts?

**Answer:** **VWAP σ-position is the direction discriminator.**
- ORC_call correct: median SPX at −0.26σ (below VWAP)
- ORC_call wrong: median SPX at +0.14σ (above VWAP)
- ORC_put correct: median SPX at +0.09σ (above VWAP)
- ORC_put wrong: median SPX at −0.31σ (below VWAP)

Pickles' "long below VWAP, put above VWAP" rule is confirmed as a direction
filter. Sign flips cleanly between correct and wrong fires, symmetric on
both directions.

**Secondary finding:** `breakout_confirmation` (v2 feature: confirmed break
of prev-session-high/IB-high with volume+momentum) correlates with WRONG
ORC fires — a "chased breakout" overextension signal. Do NOT use as a
positive filter for ORC.

**Proposed ORC direction filter:**
- BUY_CALL requires sigma_pos ≤ 0.0 (price at-or-below VWAP)
- BUY_PUT requires sigma_pos ≥ 0.0 (price at-or-above VWAP)

Sample-size caveat: 52 correct fires per direction. Ship as hypothesis,
measure live side-error rate, iterate.

**Script:** `v3/analysis/side_error_dive.py`

**Downstream impact:** ORC gets a new hard gate on VWAP direction. Expected
reduction in side-error count; verify after implementation.

---

### 10. [combined_confluence_2026_04_20.md](combined_confluence_2026_04_20.md)

**Question:** Are the three candidate late-session filters (A=VP, B=VWAP,
C=OMAR) independent (compound edge) or redundant (overlapping signal)?

**Answer:**
- **OMAR (C) alone is the strongest single filter** at 1.66× enrichment
  (I was wrong earlier to position VP as strongest — OMAR wins).
- **Best 2-filter stack is B+C** (VWAP + OMAR): 1.92× enrichment, 39.6%
  abstention capture.
- **Adding A (VP) produces marginal gain** (1.97× at 29.1% capture) because
  A and C are 1.35× correlated (both measure "price in congestion zone").
- **Filters A+B+C correctly exclude ORC territory:** entered_right at
  0.87× under control, side_error at 0.67×. Late-session filter doesn't
  interfere with ORC's domain.

**Revised teacher spec:** drop A, keep B+C. Same 1.92× precision with 37%
more captures than the 3-filter stack.

**Correction to earlier docs:** OMAR's role upgraded from "display-only"
back to "filter duty." The time_of_day_cap doc's finding that OMAR doesn't
help NR10 forward moves is about trigger-level profit, not cohort-level
enrichment — different measurement.

**Script:** `v3/analysis/combined_confluence.py`

**Downstream impact:** teacher spec tightened from 3 filters to 2 filters
without losing edge.

---

### 11. [hvn_lvn_2026_04_20.md](hvn_lvn_2026_04_20.md)

**Question:** Does finding HVN/LVN peaks and troughs in the developing
volume-by-price histogram add signal beyond POC/VAH/VAL?

**Answer:** NO. Near-HVN enrichment on abstention is only 1.08× (noise
level). Near-LVN is 1.24× — but side_error is also 1.22× near LVN, so the
signal is direction-agnostic (identifies volatile bars, not oracle bars).
"Inside LVN" is weakly anti-correlated with entered_right (0.82×): clean ORC
breakouts happen AFTER the LVN, not inside it.

HVN/LVN structural decomposition cannot replace or strengthen the existing
B+C stack (1.92×). Do NOT add as a filter.

**Script:** `v3/analysis/hvn_lvn.py`

**Downstream impact:** removes HVN/LVN from the pending research list. B+C
teacher spec unchanged.

---

## Late-session teacher spec — FINAL (ready to implement)

Integrated from all research (docs #4-#10). Revised after combined-confluence
test found OMAR is stronger than VP, and A+VP adds marginal value over B+C:

```
NarrowRangeBreakout teacher (NR10):
  Window:          minutes [40, 120] (10:10-11:30 ET, post-MAGIC TIME)
  Eligibility:     SPX close inside first15 range
  Trigger:         current close breaks last-10-bar high (BUY_CALL)
                   or last-10-bar low (BUY_PUT)
  Squeeze gate:    last-10-bar SPX range ≤ 1.0× OMAR range
  Direction gate:  BUY_CALL requires SPX ≤ VWAP + 0.5σ (SPY-derived)
                   BUY_PUT  requires SPX ≥ VWAP − 0.5σ
  Retest gate:     SPX close within 0.5× OMAR of OMAR high, low, or mid
```

Expected behavior on abstention cohort: 39.6% capture at 1.92× enrichment
vs random-bar rate.

## ORC teacher direction filter (new, from side-error deep-dive)

Separate from the late-session teacher, ORC gets a new direction gate:

```
ORC teacher additions:
  BUY_CALL additional gate: sigma_pos ≤ 0.0 (price at-or-below VWAP)
  BUY_PUT  additional gate: sigma_pos ≥ 0.0 (price at-or-above VWAP)
```

Expected behavior: reduce the 402 side-error count by excluding ORC fires
on the wrong side of VWAP. Measure after implementation.

## What's still pending (research, not blocking implementation)

- **Real-time (per-bar) developing VP vs checkpoints.** Implementation
  should recompute at every bar regardless; worth confirming the 1.32×
  enrichment holds with fine-grained recomputation.
- **Forward-move quality on the filtered NR10 triggers.** We've measured
  enrichment vs control; worth also measuring whether the filtered triggers
  have bigger forward moves than unfiltered NR10 triggers.
- **ORC + retest gate combined.** Adding C (OMAR retest) to ORC might
  further improve ORC beyond the VWAP direction fix. Untested.

## What's still pending (implementation, not research)

After the above research is locked:

- Implement `NarrowRangeBreakout` teacher in `v3/teachers/`
- Attach OMAR + last-10-bar fields to `BarContext` (via v2 adapter or a
  small preprocessing module)
- Re-run full attribution with the new teacher in the mix; measure whether
  abstention + side-error counts drop vs the Stage 1 baseline

## Conventions

- Every reference doc here should cite its analysis script so the numbers
  are reproducible.
- Every memory entry in `~/.claude/projects/.../memory/` tagged `v3_*`
  points to one of these reference docs.
- Dates in filenames are ISO-8601 (YYYY_MM_DD). New research goes into a
  new file; existing files are updated in place with a clear edit note
  when findings are corrected.
