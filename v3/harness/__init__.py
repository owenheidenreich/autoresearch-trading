"""Harness glue code: adapters from v2 data artifacts to v3 inputs.

This package is the ONLY place that should import from v2. Everything under
`v3/` (guardrails, teachers, logger, oracles) stays data-source-agnostic so
a future XSP or IBKR path can plug in here without touching the core.
"""
