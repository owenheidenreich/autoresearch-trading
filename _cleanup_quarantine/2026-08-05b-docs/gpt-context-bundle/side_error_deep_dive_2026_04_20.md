# Side-Error Deep-Dive — Why ORC Fires the Wrong Direction — 2026-04-20

> Research trail: [decay_analysis](decay_analysis_2026_04_20.md) → [attribution_full](attribution_full_2026-04-20.txt) → [stage1_research_findings](stage1_research_findings_2026_04_20.md) → [omar_findings](omar_findings_2026_04_20.md) → [time_of_day_cap](time_of_day_cap_2026_04_20.md) → [vwap_bands](vwap_bands_2026_04_20.md) → [sr_confluence](sr_confluence_2026_04_20.md) → [volume_profile](volume_profile_2026_04_20.md) → **[this doc, #9]** → [combined_confluence](combined_confluence_2026_04_20.md). See [INDEX.md](INDEX.md).

## Question

402 sessions (41% of all oracle bars) had ORC fire at the oracle bar in the
**wrong direction**. 82% of those had ORC's gate condition met (break of
first15 range + VWAP agrees) — just in the opposite direction from what was
actually profitable.

What features at the oracle bar distinguish ORC's true breakouts
(entered_right) from its false breakouts (side_error)?

## Method

From the full-dataset attribution output, pull all bars where ORC fired
(either direction). Classify:

- `ORC_call_correct` (n=52): ORC said BUY_CALL, oracle direction was call
- `ORC_call_wrong`   (n=171): ORC said BUY_CALL, oracle said put was best
- `ORC_put_correct`  (n=52): ORC said BUY_PUT, oracle direction was put
- `ORC_put_wrong`    (n=159): ORC said BUY_PUT, oracle said call was best

At each bar, measure candidate distinguishing features:

- `breakout_confirmation` (feature 47 in data.pt): confirmed break of
  prev-session high/IB high with volume + momentum support
- `volume_ratio`
- `vwap_dist` magnitude (how far price is from VWAP)
- `vwap_slope` magnitude
- `first15_range_pct`
- `atm_iv`
- SPY-derived VWAP σ-position
- Overnight gap (prev close → today open)

Script: `v3/analysis/side_error_dive.py`.

## Key result — VWAP σ-position is the direction discriminator

| Feature (median) | ORC_call correct | ORC_call wrong | ORC_put correct | ORC_put wrong |
|---|---:|---:|---:|---:|
| **VWAP σ-position** | **−0.26σ** | **+0.14σ** | **+0.09σ** | **−0.31σ** |
| `|vwap_dist|` | 0.13% | 0.22% | 0.13% | 0.24% |
| `volume_ratio` | 0.62 | 0.69 | 0.62 | 0.78 |
| `first15_range_pct` | 0.25% | 0.28% | 0.25% | 0.27% |
| `atm_iv` | 0.104 | 0.092 | 0.093 | 0.103 |

**The VWAP σ-position median flips sign between correct and wrong ORC fires,
symmetrically for both directions.** This is the cleanest directional split
in the data.

Concrete reading:
- When ORC says BUY_CALL at a bar where price is **below VWAP**, it's right
  more often (median position −0.26σ).
- When ORC says BUY_CALL at a bar where price is **above VWAP**, it's wrong
  more often (median position +0.14σ).
- Symmetric mirror for BUY_PUT.

This is Pickles' "long below VWAP, put above VWAP" rule showing up exactly
where the breakout-trading problem lives: the side-error cohort.

## Secondary finding — `breakout_confirmation` has a paradoxical signal

The v2-pipeline `breakout_confirmation` feature fires only when price breaks
the MAX of (prev-session high, first-30-min high) with volume + momentum
support. Per-cohort sign-alignment with ORC's direction:

| | % with breakout_confirm sign agreeing with ORC direction |
|---|---:|
| ORC_call_correct | 3.8% |
| **ORC_call_wrong** | **14.0%** |
| ORC_put_correct | 1.9% |
| **ORC_put_wrong** | **16.4%** |

**Confirmed breakouts (by the v2-pipeline definition) are MORE likely to be
wrong than simple first-15 breaks.** Plausible interpretation: "chased
breakout" overextension — by the time price has confirmed through multiple
resistance levels with volume, it's at an exhaustion point more often than
at the start of a trend.

Sample sizes make this a tentative finding, not load-bearing. But it argues
against naively adding `breakout_confirmation` as a positive filter to ORC.

## Proposed ORC direction filter

Add to the ORC teacher:

```
ORC BUY_CALL gate additions:
- sigma_pos (SPX position vs VWAP in σ-units, SPY-derived) ≤ 0.0
  (price must be at-or-below VWAP)

ORC BUY_PUT gate additions:
- sigma_pos ≥ 0.0
  (price must be at-or-above VWAP)
```

Expected behavior at this threshold:
- Keeps most of the correct ORC fires (median sigma_pos is −0.26σ for calls,
  +0.09σ for puts — both within the allowed zone after adding the rule).
- Excludes majority of wrong fires (median +0.14σ for wrong calls and −0.31σ
  for wrong puts — both in the excluded zone).
- Sample sizes are small enough that we should treat this as a hypothesis to
  validate, not a locked truth. The 0.0 threshold may want to be tightened
  to ±0.1σ or ±0.2σ based on live results.

## Sample-size caveat

- 52 correct call fires vs 171 wrong call fires is enough to establish
  directional tendency. It's NOT enough to calibrate the exact σ-threshold
  with precision.
- Recommended implementation: ship with σ-threshold at 0.0 (at-VWAP boundary),
  measure live attribution side-error rate after the filter is added, and
  iterate.

## What this does NOT address

- ORC's abstention gap (47% of sessions where ORC didn't trigger at the
  oracle bar). That's the domain of the separate late-session teacher.
- Forward-move quality AFTER the direction filter is added. We've
  established direction inference, not that the retained ORC fires have
  strong forward edge. Re-run the full attribution after the filter is
  implemented to confirm.

## Reproducibility

- Script: `v3/analysis/side_error_dive.py`
- Reads per-session JSON from `v3/reference/attribution_full_2026-04-20.txt`
- Uses SPX data.pt + SPY-derived VWAP bands
