"""Oracles: hindsight-optimal labels populated over the already-logged
opportunity surface.

Two oracles, both populate `BarRecord.labels` in-place on an existing `DayLog`:

- `opportunity.py` — the PRIMARY oracle. Over all eligible bars and all
  guardrail-passing contracts, pick the hindsight-best (entry_bar,
  direction, contract, exit_bar) tuple subject to realistic fills. Also
  computes per-bar per-direction best/worst forward PnL for attribution.

- `exit_headroom.py` — the SECONDARY oracle. Conditional on an actual
  entry (teacher selection), measure how much a better hindsight exit
  would have recovered. Written later after opportunity oracle is solid.
"""
