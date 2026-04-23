# Combined Confluence — Independence Check + Stacked Enrichment — 2026-04-20

> Research trail: [decay_analysis](decay_analysis_2026_04_20.md) → [attribution_full](attribution_full_2026-04-20.txt) → [stage1_research_findings](stage1_research_findings_2026_04_20.md) → [omar_findings](omar_findings_2026_04_20.md) → [time_of_day_cap](time_of_day_cap_2026_04_20.md) → [vwap_bands](vwap_bands_2026_04_20.md) → [sr_confluence](sr_confluence_2026_04_20.md) → [volume_profile](volume_profile_2026_04_20.md) → [side_error_deep_dive](side_error_deep_dive_2026_04_20.md) → **[this doc, #10]**. See [INDEX.md](INDEX.md).

## Question

The three candidate filters for the late-session teacher — intraday VP
proximity (A), VWAP direction tilt (B), OMAR retest (C) — each showed
some oracle-bar enrichment individually. Do they stack independently
(compound edge), or are they measuring the same underlying phenomenon
(redundant signals)?

If independent: stacking improves precision materially, and we ship all three.
If redundant: one or two filters capture the same signal, and we simplify.

## Method

For each bar in each cohort, compute:
- A: SPX close within 0.5× OMAR of intraday developing POC, VAH, or VAL
- B: VWAP direction filter satisfied
  (sigma_pos ≤ +0.5 for candidate-call bars; ≥ −0.5 for candidate-put bars)
- C: SPX close within 0.5× OMAR of OMAR high, low, or mid

Measure per-cohort satisfaction rates for each filter singly, all pairs, and
the full stack. Compute enrichment vs control.

Check independence on the control cohort via P(A∩B) vs P(A)×P(B).

Script: `v3/analysis/combined_confluence.py`.

## Filter satisfaction rates per cohort

| Cohort | n | A (VP) | B (VWAP) | C (OMAR) | A∩B | A∩C | B∩C | A∩B∩C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| entered_right | 117 | 35.0% | 87.2% | 24.8% | 34.2% | 13.7% | 20.5% | 12.8% |
| **abstention** | 467 | **64.9%** | **91.6%** | **43.7%** | **59.7%** | **31.9%** | **39.6%** | **29.1%** |
| side_error | 402 | 55.0% | 89.3% | 16.2% | 49.5% | 11.4% | 13.7% | 10.0% |
| control | 986 | 52.5% | 78.9% | 26.4% | 42.1% | 18.7% | 20.6% | 14.8% |

## Independence check (control cohort)

| Filter pair | Observed | Expected if independent | Ratio |
|---|---:|---:|---:|
| A∩B | 42.1% | 41.5% | 1.02× (independent) |
| **A∩C** | **18.7%** | **13.9%** | **1.35× (NOT independent)** |
| B∩C | 20.6% | 20.8% | 0.99× (independent) |
| A∩B∩C | 14.8% | 10.9% | 1.35× |

**A (VP) and C (OMAR) are correlated** — when price is near the intraday
value area, it's ~35% more likely than chance to also be near an OMAR level.
Both measure "price is in congestion-zone territory," so they partially
overlap. A+B and B+C are effectively independent.

## Enrichment vs control (key table)

| Stack | abstention enrichment |
|---|---:|
| A alone | 1.24× |
| B alone | 1.16× |
| **C alone** | **1.66×** |
| A∩B | 1.42× |
| A∩C | 1.71× |
| **B∩C** | **1.92×** |
| A∩B∩C | **1.97×** |

| Stack | entered_right enrichment | side_error enrichment |
|---|---:|---:|
| A∩B∩C | 0.87× | 0.67× |

## Key readings

### Finding 1 — OMAR (C) is the strongest single filter, not VP (A)

I was wrong in the volume_profile doc to position A as "the strongest
filter." When measured on the same control cohort as A and B, filter C
alone has 1.66× enrichment vs A's 1.24× and B's 1.16×. OMAR retest is
the strongest single discriminator for abstention oracle bars.

### Finding 2 — Best 2-filter combo is B+C (VWAP + OMAR)

B∩C gives 1.92× enrichment, nearly as strong as the 3-filter stack (1.97×).
Adding A (VP) to B+C produces only a marginal 0.05× improvement because A
and C are already correlated (1.35× co-occurrence baseline).

### Finding 3 — The stacked filter CORRECTLY excludes ORC territory

- entered_right bars (ORC's existing wins): 0.87× under control on A∩B∩C
  — the late-session filter correctly declines to fire in ORC's domain.
- side_error bars (ORC-gone-wrong): 0.67× — similarly excluded.

This confirms the late-session teacher naturally stays out of ORC's way.
No teacher-interference adjustment needed.

### Finding 4 — Capture-rate vs precision tradeoff

| Stack | Absolute capture of abstention | Enrichment |
|---|---:|---:|
| B+C | 39.6% | 1.92× |
| A+B+C | 29.1% | 1.97× |

B+C captures ~185 of 467 abstention bars at 1.92× precision.
A+B+C captures ~136 at 1.97× precision.

**B+C is the better practical choice:** 37% more abstention bars caught
at essentially the same precision. The incremental A filter trades capture
for marginal precision.

## Revised late-session teacher spec (B+C, drop A)

```
NarrowRangeBreakout teacher (revised after combined-confluence test):
  Window:          minutes [40, 120] (10:10-11:30 ET)
  Eligibility:     SPX close inside first15 range
  Trigger:         current close breaks last-10-bar high (BUY_CALL)
                   or last-10-bar low  (BUY_PUT)
  Squeeze gate:    last-10-bar SPX range ≤ 1.0× OMAR
  Filter B:        BUY_CALL requires SPX ≤ VWAP + 0.5σ
                   BUY_PUT  requires SPX ≥ VWAP − 0.5σ
  Filter C:        SPX close within 0.5× OMAR of OMAR high, low, or mid
  OMAR:            Both a filter (C) AND a display unit (range scale)
```

**OMAR is reinstated to filter duty.** Earlier findings positioned OMAR as
"display-only"; the combined-confluence test shows OMAR is actually the
strongest individual filter for the abstention cohort. The prior claim was
based on NR10 forward-move quality, which measured something different.

**Intraday VP (A) is deprioritized.** Still a real signal (1.24× alone), but
its additional discrimination on top of B+C is marginal (0.05× enrichment
gain). Not worth the implementation complexity for v1. Can be added later if
live teacher performance suggests more precision is needed.

## What this does NOT answer

- **Forward-move quality of the B+C stack on NR10 triggers.** We've measured
  enrichment vs control — the filter-retained bars are 1.92× more likely to
  be abstention oracles than random. But we haven't directly measured
  whether the B+C-filtered NR10 triggers have BIGGER forward moves (mfe_bps)
  than the raw NR10 triggers. Expected yes given the enrichment, but worth
  confirming after implementation.
- **Independence check on abstention cohort specifically.** The independence
  test ran on control. On abstention bars the filter satisfaction rates are
  higher; co-occurrence structure might differ.
- **Does the filter generalize to ORC?** The side-error deep-dive proposes a
  VWAP σ-threshold for ORC direction. Adding the OMAR retest check to ORC
  might further improve ORC's side-error rate. Untested.

## Reproducibility

- Script: `v3/analysis/combined_confluence.py`
- Uses intraday developing VP at checkpoints (30/60/90/120 minutes)
- SPY-derived VWAP with ±1σ/±2σ bands
- OMAR levels from first-minute SPX H/L
