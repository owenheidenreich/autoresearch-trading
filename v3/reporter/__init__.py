"""Stage 1 diagnostic reporters.

Two reporters live here:

- `feasibility.py` — the four SPX/$25k feasibility diagnostics (coverage,
  premium_cap_bind, guardrail_suppression, low_delta_forcing). Consumes
  `DayLog` output from the logger. No oracle dependency.

- `attribution.py` (future) — the four-fault error attribution taxonomy
  (guardrail suppression, abstention gap, side error gap, exit gap).
  Requires oracle labels to be populated.
"""
