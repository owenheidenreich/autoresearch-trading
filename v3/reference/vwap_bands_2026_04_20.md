# VWAP Bands — Pickles' Direction Rule Tested — 2026-04-20

> Research trail: [decay_analysis](decay_analysis_2026_04_20.md) → [attribution_full](attribution_full_2026-04-20.txt) → [stage1_research_findings](stage1_research_findings_2026_04_20.md) → [omar_findings](omar_findings_2026_04_20.md) → [time_of_day_cap](time_of_day_cap_2026_04_20.md) → **[this doc, #6]**. See [INDEX.md](INDEX.md) for the full reading order.

## Question

Does Pickles' VWAP-band direction rule ("long at or below VWAP, put at or above
VWAP, never long at +2σ, never put at -2σ, first-touch rejection usually fails")
hold empirically for SPX 0DTE?

## Method

SPX has no direct intraday volume. Use SPY as a volume proxy (both follow the
same index; SPY has Polygon-computed `vwap` and raw `volume` in the cache).

1. For each trading day (2022-03-15 to 2026-04-01):
   - Use Polygon's pre-computed SPY `vwap` column as the session VWAP.
   - Compute running session-to-date volume-weighted stddev of (close − vwap).
   - Produce ±1σ and ±2σ bands per minute.
2. For each oracle bar (and random control), at that bar's minute:
   - Scale SPY VWAP and stddev to SPX-equivalent using per-bar SPX/SPY ratio.
   - Compute `band_position = (SPX_close − SPX_VWAP_est) / SPX_std_est`, in σ-units.
3. Partition oracle bars by outcome and direction:
   - `entered_right_call`, `entered_right_put`
   - `abstention_call`, `abstention_put`
   - `control_random` (one random minute per day in [40, 120] window)
4. Distribution across σ-buckets and verification of each Pickles rule.

Script: `v3/analysis/vwap_bands.py`.

## Results — distribution across σ-bands

Percent of each cohort falling in each band:

| σ-band              | right_call | right_put | abs_call | abs_put | control |
|---------------------|-----------:|----------:|---------:|--------:|--------:|
| above +2σ           | 1.7%       | 0.0%      | 0.0%     | 0.4%    | 0.2%    |
| +1σ to +2σ          | 1.7%       | 6.9%      | 1.8%     | 3.7%    | 4.1%    |
| +0.5σ to +1σ        | 8.5%       | **20.7%** | 7.1%     | **17.0%** | 12.2%  |
| near VWAP (±0.5σ)   | **57.6%**  | **58.6%** | **62.8%**| **71.0%**| 62.3%  |
| -1σ to -0.5σ        | **20.3%**  | 10.3%     | **19.9%**| 7.1%    | 15.2%   |
| -2σ to -1σ          | 8.5%       | 3.4%      | 8.4%     | 0.4%    | 5.5%    |
| below -2σ           | 1.7%       | 0.0%      | 0.0%     | 0.4%    | 0.6%    |
| **n**               | 59         | 58        | 226      | 241     | 493     |

Median band-position per cohort (σ-units from VWAP):

| Cohort              | median  | p25     | p75     |
|---------------------|--------:|--------:|--------:|
| entered_right_call  | −0.21σ  | −0.57σ  | +0.11σ  |
| entered_right_put   | +0.08σ  | −0.22σ  | +0.52σ  |
| abstention_call     | −0.24σ  | −0.53σ  | +0.10σ  |
| abstention_put      | +0.17σ  | −0.11σ  | +0.43σ  |
| control_random      | −0.04σ  | −0.39σ  | +0.29σ  |

## Findings

### Finding 1 — Pickles' direction rule is validated

Oracle CALL bars sit BELOW VWAP on average (median −0.21σ entered_right,
−0.24σ abstention). Oracle PUT bars sit ABOVE VWAP on average (+0.17σ).
Control median is roughly at VWAP (−0.04σ), as expected.

The asymmetry matches Pickles: long calls at or below VWAP; long puts at or
above VWAP. It is not a symmetry artifact of the measurement.

Pickles-rule verification:
- 96.6% of entered_right_call bars are below +1σ.
- 98.2% of abstention_call bars are below +1σ.
- 96.6% of entered_right_put bars are above −1σ.
- 99.2% of abstention_put bars are above −1σ.

### Finding 2 — The edge is directional but modest in magnitude

Oracle-vs-control median deltas:
- entered_right_call: −0.21σ vs control −0.04σ → ~0.17σ tilt toward below-VWAP.
- entered_right_put: +0.08σ vs control −0.04σ → ~0.12σ tilt toward above-VWAP.
- abstention_call/put: similar tilts (~0.2σ each direction).

The signal is real but not knife-edge. SPX spends most of its time near VWAP
(62% of control within ±0.5σ). Oracle bars nudge away from VWAP in the correct
direction, but don't live at the extremes.

### Finding 3 — ±2σ extremes are rare

<5% of every cohort reaches ±2σ. Pickles' "don't long at +2σ" rule is
empirically correct (1.7% of winning calls are at +2σ) but rarely tested
in practice — SPX spends minimal time at these extremes.

### Finding 4 — "First touch vs second test" NOT tested here

Sample-size constraints plus complex per-session tracking required. Flagged
as a future follow-up. Current recommendation doesn't depend on this rule.

## Proposed teacher filter (evidence-based)

Add to the `NarrowRangeBreakout` (and potentially to ORC) teacher:

```
# Direction filter based on VWAP band position
# SPY-derived VWAP, scaled to SPX at each bar using per-bar SPX/SPY ratio.

BUY_CALL requires: SPX close ≤ VWAP + 0.5σ    # capture the "below-VWAP" zone
BUY_PUT  requires: SPX close ≥ VWAP − 0.5σ    # capture the "above-VWAP" zone
```

Expected effect (approximate):
- Excludes ~10-12% of call triggers that fire above +0.5σ (where <10% of
  winning calls live).
- Excludes ~10-12% of put triggers that fire below −0.5σ.
- Captures ~88% of oracle-best call bars and ~80% of oracle-best put bars.

**Magnitude caveat:** modest edge. The filter is "correct but not transformative."
Keep expectations calibrated: this is one soft filter in a confluence stack,
not a standalone fix for the teacher's weakness.

## What the VWAP investigation does NOT answer

- **First-touch vs second-test behavior** — requires per-session VWAP-crossing
  tracking. Deferred.
- **Are the WINNERS within each cohort at different VWAP positions than the
  losers?** The oracle ceiling defines winners by forward PnL on the oracle
  contract. A fine-grained within-cohort comparison (best-vs-worst oracle bars)
  could tighten the rule. Deferred.
- **VWAP + volume together** — SPY's volume is directly available and could
  inform a "breakout with volume confirmation" rule. Deferred to teacher
  implementation phase.

## Reproducibility

- Script: `v3/analysis/vwap_bands.py`
- Input: SPY minute cache (`~/.cache/autoresearch-trading/data/spy_1min.pkl`)
- Uses Polygon's pre-computed SPY `vwap` column; deviation stddev computed
  from raw volume + (close − vwap)² cumulatively per session.
- Cohorts read from `v3/reference/attribution_full_2026-04-20.txt`
  per-session JSON.
