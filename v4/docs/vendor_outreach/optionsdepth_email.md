# Draft: OptionsDepth pre-purchase data-rights email

**Send to**: their pricing/sales contact (per https://www.optionsdepth.com/pricing) or their support form.

**Goal**: confirm in writing the seven items the v4 protocol requires before subscribing. A "no" on any of them does not end the relationship — it just means OptionsDepth becomes a dashboard input rather than a model-feature source, and the v4 dealer-flow architecture has to be re-thought.

**Tone**: direct, professional, no apologies. They sell to quants; they expect these questions.

---

## Suggested email

> **Subject**: Pre-purchase data-rights questions for Pro Max
>
> Hi OptionsDepth team,
>
> I'm an independent researcher evaluating Pro Max for a SPX 0DTE quantitative
> research project. Before I subscribe, I'd like to confirm a few specifics
> about the historical and intraday data product. Quick answers in any form
> would be appreciated.
>
> **1. Historical export.** Does Pro Max allow bulk export or API download
> of historical intraday dealer-exposure snapshots in a machine-readable
> format (CSV, JSON, Parquet, or REST)? I would like to be able to retain
> snapshots locally for offline analysis.
>
> **2. Retention after cancellation.** If I export and locally archive
> snapshots during a subscription month and then cancel, am I permitted
> to continue using those archived files for personal research?
>
> **3. Model training rights.** Are subscribers permitted to use the
> exported data as input features to train private quantitative models,
> as long as the data is not redistributed or republished?
>
> **4. Point-in-time integrity.** When I download a snapshot at, say,
> 14:30 ET on 2026-01-15, am I getting the same numbers a subscriber
> downloaded in real time at 14:30 ET on 2026-01-15? Or are historical
> snapshots subject to revisions/restatements?
>
> **5. Timestamp + cadence.** Each snapshot's documented refresh cadence
> is 10 minutes. Is each snapshot tagged with a precise timestamp
> (millisecond / second)? If a 10-minute snapshot is delivered late, does
> the timestamp reflect the publish time or the nominal slot?
>
> **6. Field definitions.** Is there a published data dictionary for the
> Pro Max dealer-exposure fields (GEX bucketing, charm exposure, vanna,
> intraday OI deltas, exchange-tagged customer-vs-MM flow, etc.)? I want
> to confirm field names and definitions are stable across releases.
>
> **7. Index coverage.** Does Pro Max coverage at $249/mo include both
> SPX and VIX positioning data?
>
> The architectural design of my project hinges on your answers to these
> seven questions, so written confirmation is what I need before
> committing. I'm happy to sign an NDA if that's useful for items 1–3.
>
> Thanks in advance — looking forward to working with what you've built
> if it lines up.
>
> Best,
> [your name]

---

## Filing the response

When the response arrives:

1. Save the raw email + any attachments to `v4/docs/vendor_outreach/optionsdepth_response_YYYY-MM-DD.md`.
2. Update [../PHASE_0_5_VENDOR_VERIFICATION.md](../PHASE_0_5_VENDOR_VERIFICATION.md) status table.
3. Add a research-ledger entry with the verdict (proceed / disqualified / partial).

If they confirm all 7 → Phase 1 OptionsDepth one-cycle is authorized ($250 one-time, archive every snapshot, cancel).

If any item is disqualifying → see PHASE_0_5_VENDOR_VERIFICATION.md "What if a vendor disqualifies?" section.

---

## What you are *not* asking

Deliberately omitted from the email:
- Discount / promo codes — not relevant at this stage; you want correctness, not pricing.
- Promises about future fields — you're evaluating the current product.
- Service-level guarantees (uptime, latency) — research use, not live trading.
- Refund policy — not material; one-cycle commitment is small and the data is owned regardless.

If the response volunteers any of these, file them but don't let them displace the seven core questions.
