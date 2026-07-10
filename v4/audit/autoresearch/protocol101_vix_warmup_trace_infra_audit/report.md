# Protocol101 VIX Warm-Up Trace Infrastructure Audit

- Status: `blocked`
- Decision: `vix_warmup_infra_added_but_current_sources_do_not_close_initial_15m_gap`
- Blockers: `6`

Infrastructure was added so replay inputs can request a `decision_start_et` earlier than 09:31 while preserving the default 09:31 behavior.

The current local burned-day sources still do not close the initial VIX-change gap: historical official SPX/VIX files start at 09:30 ET, and the IBKR replay input builder still first produces usable checkpoints at 09:31 ET on these captures.

No VIX feature was admitted and no L0/L1/L3 audit was rerun.
