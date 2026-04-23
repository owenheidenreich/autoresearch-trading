# Time-of-Day Cap for Late-Session Teacher — 2026-04-20

> Research trail: [decay_analysis](decay_analysis_2026_04_20.md) → [attribution_full](attribution_full_2026-04-20.txt) → [stage1_research_findings](stage1_research_findings_2026_04_20.md) → [omar_findings](omar_findings_2026_04_20.md) → **[this doc, #5]**. See [INDEX.md](INDEX.md) for the full reading order.

## Question

Is `[40, 120]` (10:10-11:30 ET) actually the right time window for the late-session
`NarrowRangeBreakout` (NR10) teacher, or should we extend later, start earlier, or
tighten? And — as a bonus — does the OMAR retest filter actually improve the NR10
trigger's edge, as implied by the earlier OMAR enrichment finding?

## Method

Simulated the NR10 trigger across every eligible bar in minutes `[30, 240]`
(10:00-13:30 ET) on the full 986-day SPX cache. Trigger fires when:

- close is inside the first-15 range
- last 10-bar SPX range ≤ 1.0× OMAR range
- current close breaks last-10-bar high (call) or low (put)

For each trigger, measured the 20-bar forward SPX move (direction-adjusted) and
a fake-out flag (close retraces inside the pre-range within 10 bars).

Also partitioned triggers into "near OMAR" (|dist to nearest OMAR level| ≤ 0.5×
OMAR range) and "far from OMAR" to test whether OMAR filtering adds forward
edge on the overall trigger population.

Script: `v3/analysis/time_of_day_cap.py`.

## Results — NR10 quality by 30-minute bucket

13,104 total triggers across 986 days (~13/day):

| Bucket       | n    | trig/day | mfe_med_bps | mfe_p75_bps | final_med_bps | fake_rate |
|--------------|------|----------|-------------|-------------|---------------|-----------|
| 10:00-10:30  | 2019 | 2.05     | **8.0**     | **16.9**    | 0.2           | 67%       |
| 10:30-11:00  | 1785 | 1.81     | 6.6         | 14.7        | 0.3           | 66%       |
| 11:00-11:30  | 1936 | 1.96     | 6.8         | 14.1        | 1.6           | 66%       |
| 11:30-12:00  | 1943 | 1.97     | 6.0         | 11.6        | 0.9           | 65%       |
| 12:00-12:30  | 1798 | 1.82     | 5.3         | 10.5        | 0.5           | 68%       |
| 12:30-13:00  | 1849 | 1.88     | 5.4         | 10.2        | 0.6           | 66%       |
| 13:00-13:30  | 1774 | 1.80     | 5.2         | 10.5        | 0.1           | 66%       |

### Reading

- **Trigger frequency is roughly constant** (~1.8-2.0/day) across all buckets.
  NR10 fires throughout the day at similar density; it's not a morning-only setup.
- **Signal quality degrades gradually.** Median MFE peaks at 8.0 bps in 10:00-10:30
  and floors at 5.2-5.4 bps in 12:00+. p75 MFE drops ~37% from morning (16.9) to
  midday (10.2).
- **Fake-out rate is flat at 65-68%** across the entire window. ~2/3 of NR10
  triggers retrace inside the pre-range within 10 bars regardless of time.
- **Median final (close-to-close at bar+20) is essentially zero** (≤1.6 bps)
  in every bucket. NR10 triggers have no net directional edge at the 20-minute
  mark — moves peak then retrace.

## Results — NR10 + near-OMAR filter

9,550 of 13,104 triggers (73%) are near OMAR. Comparison of near-OMAR vs
NOT-near-OMAR triggers across buckets:

| Bucket       | near-OMAR mfe_med | NOT-near-OMAR mfe_med |
|--------------|-------------------|------------------------|
| 10:00-10:30  | 8.1               | **7.8**                |
| 10:30-11:00  | 6.5               | **7.0**                |
| 11:00-11:30  | 6.6               | **7.0**                |
| 11:30-12:00  | 5.8               | **6.4**                |
| 12:00-12:30  | 5.2               | **5.6**                |
| 12:30-13:00  | 5.1               | **6.2**                |
| 13:00-13:30  | 5.1               | **5.7**                |

**In every bucket, NOT-near-OMAR triggers have equal-or-slightly-better forward
moves than near-OMAR triggers.** The OMAR proximity filter does not concentrate
edge in the overall NR10 trigger population.

## Reconciling with the earlier OMAR enrichment finding

The OMAR analysis showed abstention oracle bars cluster near OMAR levels
(43.7% vs 28.5% random — a 1.5× enrichment). That finding is still true.
But it is narrower than implied by a naive read:

- Of the ~467 abstention oracle bars, 44% are near OMAR.
- Of the ~13,000 NR10 triggers overall, 73% are near OMAR.
- Both oracle and non-oracle triggers share the same OMAR-proximity pattern.
- Filtering by OMAR proximity removes ~27% of triggers but does NOT concentrate
  edge on the oracle-bar subset.

**OMAR is a property of the abstention oracle cohort, not a discriminating
filter on the trigger population.** Using it as a teacher filter doesn't improve
the teacher's forward-move edge.

## Conclusions

### Time-cap recommendation (locked)

`[40, 120]` (10:10-11:30 ET). Rationale:

- 10:10 start respects MAGIC TIME (ORC already owns 09:55-10:10).
- 11:30 cap catches the best-quality bucket range (6.0-6.8 bps mfe_med).
- Extending to 12:00 is tolerable but picks up softer signal.
- 12:00+ is meaningfully weaker (p75 MFE drops ~30%); cap excludes it.

### NR10 trigger raw edge (important — not locked, just documented)

The NR10 trigger ALONE has weak edge:
- ~5-8 bps median favorable move
- 66% fake-out rate
- Zero net edge at the 20-bar mark

This means NR10 is NOT a viable standalone teacher. It needs additional filters
that genuinely discriminate. The OMAR-retest filter does not — it's a
conditional property of oracle bars, not a signal that separates high-quality
triggers from noise.

### What needs to come from other research

VWAP direction (validated, modest) and S/R confluence (in progress) are the
remaining candidate filters. If they fail to add material edge, the
`NarrowRangeBreakout` teacher should be re-scoped or abandoned, not shipped.

## Key correction that flows back to OMAR findings

The `omar_findings_2026_04_20.md` doc has been amended with a correction note
about the ~44% recall / 29% false-alarm claim being specific to the abstention
cohort rather than to the overall trigger population.
