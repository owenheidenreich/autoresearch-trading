# S/R Confluence (Basic Levels) — Negative Result — 2026-04-20

> Research trail: [decay_analysis](decay_analysis_2026_04_20.md) → [attribution_full](attribution_full_2026-04-20.txt) → [stage1_research_findings](stage1_research_findings_2026_04_20.md) → [omar_findings](omar_findings_2026_04_20.md) → [time_of_day_cap](time_of_day_cap_2026_04_20.md) → [vwap_bands](vwap_bands_2026_04_20.md) → **[this doc, #7]** → [volume_profile](volume_profile_2026_04_20.md). See [INDEX.md](INDEX.md) for the full reading order.

## Question

Does basic S/R confluence — using the subset of Pickles' level types that are
directly computable from the SPX minute cache — identify abstention oracle
bars more often than random bars?

## Levels tested (computable from current cache)

- Prior-day SPX high, low, close, midpoint
- Weekly pivots via the traditional formula: P = (H+L+C)/3, R1 = 2P−L,
  S1 = 2P−H, R2 = P+(H−L), S2 = P−(H−L)
- Fibonacci 38.2% / 50% / 61.8% retracement of prior session H/L range
- Initial Balance high/low (first 30 min of current session)
- Round numbers: ±2× (25, 50, 100) around the session's OMAR mid

## Levels NOT tested here (deferred or unavailable)

- Overnight H/L — no pre-market data in cache
- Volume profile POC/VAH/VAL — tested separately in [volume_profile_2026_04_20.md](volume_profile_2026_04_20.md)
- Pickles' own Sunday-published R1/R2/R3 — proprietary
- Multi-year macro static levels — would require decade+ of daily data

## Method

For each oracle bar across cohorts, measure:
- Minimum distance to any level (in OMAR-range units)
- Count of levels within 0.5× OMAR of current close (confluence count)

Random control = one random bar per day in minutes [40, 120].

Script: `v3/analysis/sr_confluence.py`.

## Results

| Cohort       | n   | median dist | % within 0.5× OMAR | p75 confluence | enrichment vs ctrl |
|--------------|----:|------------:|-------------------:|---------------:|-------------------:|
| entered_right| 117 | 0.26        | 76.1%              | 3              | **1.13× / 1.50×**  |
| abstention   | 467 | 0.33        | **66.2%**          | 2              | **0.99× / 1.00×**  |
| side_error   | 402 | 0.33        | 64.9%              | 2              | 0.97× / 1.00×      |
| control      | 986 | 0.32        | 67.1%              | 2              | 1.00× baseline      |

Per-level-type breakdown (abstention vs control):

| Level family   | abstention | control | enrichment |
|----------------|-----------:|--------:|-----------:|
| prior_day      | 23.6%      | 22.4%   | ~1.00×     |
| Fibonacci      | 16.1%      | 14.2%   | 1.13×      |
| weekly pivots  | 7.7%       | 9.6%    | 0.80×      |
| initial balance| 28.7%      | 29.6%   | 0.97×      |
| round numbers  | 28.7%      | 28.8%   | 1.00×      |

## Findings

### Finding 1 — Abstention bars show ZERO enrichment near basic S/R levels

Every level type tested sits at control-rate (within 10%) for the abstention
cohort. Computable S/R confluence does NOT help identify late-session oracle
opportunities. The filter candidate fails.

### Finding 2 — Entered_right bars DO show modest S/R enrichment

1.13× pct enrichment and 1.50× confluence-count enrichment on entered_right.
Clean ORC breakouts happen near pre-defined levels more than random bars do.
This suggests S/R confluence could be a useful *directional-inference* filter
for ORC (potentially reducing the 41% side-error rate), even though it
doesn't help abstention capture.

### Finding 3 — Implications by level type

- No single level type dominates — abstention vs control are within 2% on all
  families. If any of these level types mattered for late-session breakouts,
  we'd expect to see at least one 1.2× enrichment. We don't.

## Conclusions

### For the late-session teacher

**Do NOT add computable S/R confluence as a filter.** The signal isn't there
in the level set we can construct from the current cache. Moving on.

### For ORC

**Potentially useful**: the 1.13× / 1.50× enrichment for entered_right
suggests S/R confluence could help ORC's direction inference. Defer to the
side-error deep-dive for a focused test.

### What's NOT ruled out

This investigation does NOT rule out volume profile levels (POC/VAH/VAL) as
an S/R filter. Those were tested separately in the next step of the research
trail — see `volume_profile_2026_04_20.md` for the result.
