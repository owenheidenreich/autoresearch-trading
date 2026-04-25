---
date: 2026-04-24
day: 2024-04-01
status: golden-day trace implemented
---

# Golden Day Trace — 2024-04-01

The one-day trace validates the proposed "single market day microscope" path.
The day was rebuilt through the production action-surface builder and matched
the saved full SPX live artifact row-for-row.

Key finding:

- Future labels are strongly put-dominant: 77 put-best bars vs 14 call-best bars.
- The quarantined seed-42 model scores the day almost entirely call-dominant:
  90 call-best score bars vs 1 put-best score bar.
- The daily policy selects bar 43, call strike 5250.0. Its target utility is
  -1288.27, while the best put on the same completed bar has target utility
  1699.47. The put-call label edge is 1985.63.
- This means the correct put was present in the 24-token action space; the
  failure is not missing contracts or next-bar label construction.

The overfit canary also passed:

- action-match rate: 0.736
- strong-put bars: 72
- strong-put score put-above-call rate: 1.0
- loss drop: 1.404

Interpretation:

The architecture and loss can learn the obvious one-day put-dominant structure
when isolated. The W5 failure is therefore more likely caused by cross-day
training distribution, side-prior pressure, calibration, or seed/window
interaction, not by an inability to represent the action space for this day.

Generated local artifacts:

- `v3/artifacts/golden_day_trace/2024-04-01/report.md`
- `v3/artifacts/golden_day_trace/2024-04-01/dashboard.html`
- `v3/artifacts/golden_day_trace/2024-04-01/day_summary.csv`
- `v3/artifacts/golden_day_trace/2024-04-01/contract_tokens.csv`
- `v3/artifacts/golden_day_trace/2024-04-01-overfit/overfit_summary.json`

