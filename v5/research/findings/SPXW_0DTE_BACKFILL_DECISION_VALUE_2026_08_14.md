# The quote backfill is valuable only with a smaller shared lifecycle model

**What changed for the bot:** the same-day SPXW quote request now has a concrete, evidence-sized model and
a frozen training law. It is no longer a generic request to buy more data.

## Decision value

The 794 older non-empty 0DTE sessions project to **768 complete episodes** at the current 243/251
completion rate, for **1,011 total**; the best possible ceiling is 1,037. Scaling the already-measured
within-session dependence gives:

- observed-completeness design-effect effective observations: **2,455–4,228**;
- corresponding 20:1 parameter budget: **122–211**;
- all-complete budget: **125–216**; and
- evidence multiplier: roughly **4.2x**.

The smallest existing open shared architecture is 226 parameters and needs at least **1,081** complete
sessions under the optimistic design-effect rate. Therefore the backfill alone would still leave every
existing shared architecture outside the complete conservative range. The already-closed 48-parameter
entry model is not a reason to buy data and may not be retried.

## The missing prerequisite now exists

`compact_shared_lifecycle` is one 120-parameter neural model, counted from the built module. It observes
completed candle geometry, all five clock fields, a permutation-invariant summary of the whole live
ladder, the account and the held position. The shared state changes contract ordering through explicit
side and moneyness interactions, learns WAIT, and supplies HOLD/SELL values. Origin regime stays in the
position state, so a morning trade remains morning-owned after 12:46.

It clears the worst projected budget by two parameters. This is deliberately one shared model, not four
independent specialists.

Nine real-data interface cells passed on first/middle/last sessions at 10:00, 13:30 and 15:00. The audit
opened feature arrays only, not cached targets or economics. The model preserved whole-chain context,
masked entry actions separately, emitted finite values for all four actions, and was invariant to masked
future inputs.

## Frozen path after acquisition

The candidate protocol fixes one architecture, deterministic chronology, two trades/day maximum,
nested out-of-fold entries for exit training, frozen entry parameters before exit fitting, a shuffled
path, a composition-matched control, $500 risk kills and midpoint-first evaluation. Any failed kill closes
the member without a nearby retry.

## Evidence

- value declaration: `v5/work/entry-exit-attribution/SPXW_0DTE_BACKFILL_VALUE_DECLARATION_V1.json`
- value receipt: `v5/work/entry-exit-attribution/SPXW_0DTE_BACKFILL_VALUE_V1.json`
- built design: `v5/work/entry-exit-attribution/COMPACT_SHARED_LIFECYCLE_DESIGN_V1.md`
- design receipt: `v5/work/entry-exit-attribution/COMPACT_SHARED_LIFECYCLE_DESIGN_V1.json`
- real-interface receipt: `v5/work/entry-exit-attribution/COMPACT_SHARED_LIFECYCLE_REAL_INTERFACE_V1.json`
- frozen candidate protocol: `v5/work/entry-exit-attribution/COMPACT_SHARED_LIFECYCLE_PROTOCOL_V1.md`

No vendor was contacted, no data was downloaded, no money was spent and no model was fit.
